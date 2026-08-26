import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from rich import print
from torch.utils.data import DataLoader, Dataset

if torch.cuda.is_initialized():
    print(
        "[red]CUDA already initialised in this process, PYTORCH_CUDA_ALLOC_CONF "
        "set above has NO EFFECT, it only applies at context creation. Restart "
        "the kernel if you're chasing a fragmentation/OOM issue.[/red]"
    )

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CONFIG_124m = {
    "vocab_size": 50257,
    "context_len": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "train_ctx": 256,
}

TRAIN_CFG = {
    "data_dir": "/kaggle/input/datasets/thbdev02/newone/results/data",
    "out_dir": "/kaggle/working/out",
    "resume_from_input": None,  # e.g. "/kaggle/input/vader-checkpoints-v3/out/checkpoint.pt"
    "batch_size": 8,  # dropped from 32 after an OOM at that setting, see chat notes;
    "grad_accum_steps": 16,  # effective batch = batch_size * grad_accum_steps, kept at 128
    "max_lr": 6e-4,
    "min_lr": 6e-5,  # ~10% of max, standard cosine floor
    "warmup_steps": 400,
    "max_steps": 61_035,  # 2e9 train tokens / (batch_size=8 * grad_accum_steps=16 * train_ctx=256)
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_every": 250,
    "eval_iters": 50,
    "ckpt_every": 500,
    "sample_prompt": "Once upon a time",
    "log_every": 20,
    "sample_temp": 0.8,
    "sample_top_k": 40,
    "log_every": 20,
}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.w_key(x).view(b, num_tokens, self.num_heads, self.head_dim)
        queries = self.w_query(x).view(b, num_tokens, self.num_heads, self.head_dim)
        values = self.w_value(x).view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_w = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_w = self.dropout(attn_w)

        context_vec = (attn_w @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_len=cfg["context_len"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        self.out_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("layers.2.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layers"])
                )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


class MemmapDataset(Dataset):
    def __init__(self, bin_path, ctx_len):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.ctx_len = ctx_len

    def __len__(self):
        return (len(self.data) - 1) // self.ctx_len

    def __getitem__(self, idx):
        start = idx * self.ctx_len
        chunk = self.data[start : start + self.ctx_len + 1].astype(np.int64)
        chunk = torch.from_numpy(chunk)
        return chunk[:-1], chunk[1:]


def create_dataloader(bin_path, ctx_len, batch_size, shuffle):
    dataset = MemmapDataset(bin_path, ctx_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )


def generate(model, idx, max_new_tokens, context_size, temp=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cnd = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cnd)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )
        if temp > 0.0:
            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            idx_nxt = torch.multinomial(probs, num_samples=1)
        else:
            idx_nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_nxt == eos_id:
            break
        idx = torch.cat((idx, idx_nxt), dim=1)
    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_and_print_sample(model, tokenizer, device, prompt, ctx_len, temp=0.0, top_k=None):
    model.eval()
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model, encoded, max_new_tokens=60, context_size=ctx_len, temp=temp, top_k=top_k)
    print("[bold cyan]sample:[/bold cyan]", token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()


# --------------------------------------------------------------------------
# LR schedule: linear warmup, cosine decay
# --------------------------------------------------------------------------
def get_lr(step, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["max_lr"] * (step + 1) / cfg["warmup_steps"]
    if step > cfg["max_steps"]:
        return cfg["min_lr"]
    decay_ratio = (step - cfg["warmup_steps"]) / (cfg["max_steps"] - cfg["warmup_steps"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg["min_lr"] + coeff * (cfg["max_lr"] - cfg["min_lr"])


# --------------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    it = iter(loader)
    for i in range(eval_iters):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), yb.flatten()
            )
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scaler, step):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["step"]


def main():
    cfg = TRAIN_CFG
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device} (single-GPU, no DDP: see chat notes on Kaggle quota cost)")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"GPU memory before model init: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        if allocated > 0.5:
            print(
                "[red]non-trivial memory already in use before the model exists, "
                "this kernel likely has a previous run's tensors still alive, "
                "restart the kernel and rerun[/red]"
            )

    os.makedirs(cfg["out_dir"], exist_ok=True)
    tokenizer = tiktoken.get_encoding("gpt2")
    ctx_len = CONFIG_124m["train_ctx"]

    train_loader = create_dataloader(
        os.path.join(cfg["data_dir"], "train.bin"), ctx_len, cfg["batch_size"], shuffle=True
    )
    val_loader = create_dataloader(
        os.path.join(cfg["data_dir"], "validation.bin"), ctx_len, cfg["batch_size"], shuffle=False
    )

    torch.manual_seed(123)
    model = Model(CONFIG_124m).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["max_lr"], weight_decay=cfg["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda")

    ckpt_path = os.path.join(cfg["out_dir"], "checkpoint.pt")

    if cfg.get("resume_from_input") and not os.path.exists(ckpt_path):
        if os.path.exists(cfg["resume_from_input"]):
            import shutil

            shutil.copy(cfg["resume_from_input"], ckpt_path)
            print(f"[yellow]copied checkpoint forward from {cfg['resume_from_input']}[/yellow]")
        else:
            print(
                f"[red]resume_from_input set but not found at {cfg['resume_from_input']}, "
                f"check the attached dataset slug, starting from step 0[/red]"
            )

    start_step = 0
    if os.path.exists(ckpt_path):
        start_step = load_checkpoint(ckpt_path, model, optimizer, scaler, device)
        print(f"[yellow]resumed from checkpoint at step {start_step}[/yellow]")
    else:
        print("[dim]no checkpoint found, starting from step 0[/dim]")

    train_iter = iter(train_loader)
    t0 = time.time()

    for step in range(start_step, cfg["max_steps"]):
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg["grad_accum_steps"]):
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb = next(train_iter)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), yb.flatten()
                )
                loss = loss / cfg["grad_accum_steps"]

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % cfg["log_every"] == 0:
            dt = time.time() - t0
            toks_per_step = cfg["batch_size"] * cfg["grad_accum_steps"] * ctx_len
            print(
                f"step {step:06d} | loss {accum_loss:.4f} | lr {lr:.2e} "
                f"| {dt / cfg['log_every']:.2f}s/step | ~{toks_per_step / max(dt / cfg['log_every'], 1e-9):.0f} tok/s"
            )
            t0 = time.time()

        if step % cfg["eval_every"] == 0 and step > 0:
            val_loss = estimate_loss(model, val_loader, device, cfg["eval_iters"])
            print(f"[bold green]step {step:06d} val_loss {val_loss:.4f}[/bold green]")
            generate_and_print_sample(model, tokenizer, device, cfg["sample_prompt"], ctx_len, temp=cfg["sample_temp"], top_k=cfg["sample_top_k"])

        if step % cfg["ckpt_every"] == 0 and step > 0:
            save_checkpoint(ckpt_path, model, optimizer, scaler, step + 1)
            print(f"[dim]checkpoint saved at step {step}[/dim]")

    save_checkpoint(ckpt_path, model, optimizer, scaler, cfg["max_steps"])
    torch.save(model.state_dict(), os.path.join(cfg["out_dir"], "vader_model_final.pth"))
    print("[bold green]training complete, final weights saved[/bold green]")


if __name__ == "__main__":
    main()
