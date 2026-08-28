import json
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import tiktoken
import torch
import torch.nn as nn
from rich import print
from torch.utils.data import DataLoader, Dataset


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
}

SFT_CFG = {
    "stage2_weights": "/kaggle/input/models/berserkarch/vader2/pytorch/default/1/vader_stage2_final.pth",
    "train_path": "/kaggle/input/datasets/berserkarch/newone2/out/sft_train.jsonl",
    "val_path": "/kaggle/input/datasets/berserkarch/newone2/out/sft_val.jsonl",
    "out_dir": "/kaggle/working/out_sft",
    "resume_from_input": None,

    "batch_size": 8,
    "grad_accum_steps": 16,

    "max_lr": 5e-5,
    "min_lr": 5e-6,
    "warmup_steps": 50,

    "weight_decay": 0.1,
    "grad_clip": 1.0,

    "eval_every": 100,
    "eval_iters": 30,

    "ckpt_every": 200,

    "sample_temp": 0.7,
    "sample_top_k": 40,
    "sample_max_new_tokens": 60,

    "log_every": 20,
    "num_epochs": 3,
}

# GPT-2 <|endoftext|>
# Used as both EOS and padding token.
PAD_ID = 50256

SYSTEM_TEXT = (
    "You are Vader, a language model built by Gaurav Raj (thehackersbrain). "
    "You were named after Darth Vader from Star Wars. Speak with a bit of that flavour, "
    "but you're an AI, not a Sith Lord."
)


# --------------------------------------------------------------------------
# Multi-Head Attention
# --------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout_p = dropout

    def forward(self, x, attn_mask):
        b, num_tokens, _ = x.shape

        keys = self.w_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.w_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.w_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        context_vec = torch.nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            dropout_p=(self.dropout_p if self.training else 0.0),
        )

        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        return self.out_proj(context_vec)


# --------------------------------------------------------------------------
# LayerNorm
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Feed Forward
# --------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# --------------------------------------------------------------------------
# Transformer Block
# --------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, attn_mask):
        # Attention
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, attn_mask)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying
        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx, attn_mask):
        _, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_ids = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_ids)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(x, attn_mask)

        x = self.final_norm(x)

        return self.out_head(x)


# --------------------------------------------------------------------------
# Attention mask
# --------------------------------------------------------------------------
def build_attn_mask(pad_mask, device):
    """
    pad_mask:
        (batch, seq_len)
        True  = real token
        False = padding

    Returns:
        (batch, 1, seq_len, seq_len)

    Bool SDPA semantics:
        True  = attend
        False = block

    Combines causal + padding masking.
    """
    _, seq_len = pad_mask.shape

    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )

    mask = causal.unsqueeze(0) & pad_mask.unsqueeze(1)

    return mask.unsqueeze(1)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len):
        self.examples = []
        skipped = 0

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                prompt_ids = tokenizer.encode_ordinary(row["prompt"])
                response_ids = tokenizer.encode_ordinary(row["response"]) + [PAD_ID]

                input_ids = prompt_ids + response_ids

                if len(input_ids) > max_len:
                    skipped += 1
                    continue

                # Response-only supervision.
                #
                # Prompt tokens are ignored.
                # Response tokens, including EOS, are supervised.
                labels = [-100] * len(prompt_ids) + response_ids

                self.examples.append((input_ids, labels))

        print(
            f"[dim]loaded {len(self.examples):,} examples "
            f"from {jsonl_path}, skipped {skipped:,} over max_len[/dim]"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# --------------------------------------------------------------------------
# Collate
# --------------------------------------------------------------------------
def collate_fn(batch):
    max_len = max(len(ids) for ids, _ in batch)

    input_batch = []
    label_batch = []
    pad_mask = []

    for ids, labels in batch:
        pad_len = max_len - len(ids)

        input_batch.append(ids + [PAD_ID] * pad_len)
        label_batch.append(labels + [-100] * pad_len)
        pad_mask.append([True] * len(ids) + [False] * pad_len)

    input_ids = torch.tensor(input_batch, dtype=torch.long)
    labels = torch.tensor(label_batch, dtype=torch.long)
    pad_mask = torch.tensor(pad_mask, dtype=torch.bool)

    return input_ids, labels, pad_mask


# --------------------------------------------------------------------------
# Causal LM loss
# --------------------------------------------------------------------------
def compute_sft_loss(logits, labels):
    """
    Standard autoregressive next-token loss.

    Given:

        input:   A B C D EOS
        target:  B C D EOS

    the model learns to predict the NEXT token rather than
    reconstructing the token it is currently seeing.

    labels contain -100 over the prompt, so supervision remains
    response-only.
    """

    shift_logits = logits[:, :-1, :].contiguous()

    shift_labels = labels[:, 1:].contiguous()

    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def generate(model, idx, max_new_tokens, context_size, device, temp=0.0, top_k=None):
    assert idx.size(0) == 1, "generate() only supports batch size 1"

    for _ in range(max_new_tokens):
        idx_cnd = idx[:, -context_size:]

        pad_mask = torch.ones_like(idx_cnd, dtype=torch.bool)
        attn_mask = build_attn_mask(pad_mask, device)

        with torch.no_grad():
            logits = model(idx_cnd, attn_mask)

        # Predict next token from final position.
        logits = logits[:, -1, :]

        # Top-k sampling
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)

            logits = torch.where(
                logits < min_val,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Sampling
        if temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            idx_nxt = torch.multinomial(probs, num_samples=1)

        # Greedy decoding
        else:
            idx_nxt = torch.argmax(logits, dim=-1, keepdim=True)

        # EOS
        if idx_nxt.item() == PAD_ID:
            break

        idx = torch.cat((idx, idx_nxt), dim=1)

    return idx


# --------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------
def sample_identity_checks(model, tokenizer, device, ctx_len, temp, top_k, max_new_tokens):
    prompts = [
        (
            f"### System:\n"
            f"{SYSTEM_TEXT}\n\n"
            f"### Instruction:\n"
            f"Who are you?\n\n"
            f"### Response:\n"
        ),
        (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "Explain what a black hole is.\n\n"
            "### Response:\n"
        ),
    ]

    model.eval()

    for prompt_num, prompt in enumerate(prompts, start=1):
        encoded_ids = tokenizer.encode_ordinary(prompt)
        encoded = torch.tensor(encoded_ids, dtype=torch.long).unsqueeze(0).to(device)

        prompt_len = encoded.size(1)

        out = generate(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=ctx_len,
            device=device,
            temp=temp,
            top_k=top_k,
        )

        # ONLY decode newly generated tokens.
        generated_ids = out[0, prompt_len:].tolist()
        generated_text = tokenizer.decode(generated_ids)

        print(f"\n[bold cyan]sample {prompt_num} prompt:[/bold cyan] {prompt.replace(chr(10), ' ')}")

        if generated_text.strip():
            print(f"[bold green]sample {prompt_num} response:[/bold green] {generated_text.replace(chr(10), ' ')}")
        else:
            print(f"[bold red]sample {prompt_num} response:[/bold red] <empty / EOS immediately>")

    model.train()


# --------------------------------------------------------------------------
# Learning-rate schedule
# --------------------------------------------------------------------------
def get_lr(step, max_steps, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["max_lr"] * (step + 1) / cfg["warmup_steps"]

    if step > max_steps:
        return cfg["min_lr"]

    decay_ratio = (step - cfg["warmup_steps"]) / (max_steps - cfg["warmup_steps"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return cfg["min_lr"] + coeff * (cfg["max_lr"] - cfg["min_lr"])


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters, use_amp):
    model.eval()

    losses = []
    it = iter(loader)

    for _ in range(eval_iters):
        try:
            input_ids, labels, pad_mask = next(it)
        except StopIteration:
            it = iter(loader)
            input_ids, labels, pad_mask = next(it)

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        pad_mask = pad_mask.to(device)

        attn_mask = build_attn_mask(pad_mask, device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids, attn_mask)
            loss = compute_sft_loss(logits, labels)

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


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


def load_stage2_weights(model, path, device):
    raw_sd = torch.load(path, map_location=device)

    filtered_sd = {k: v for k, v in raw_sd.items() if not k.endswith(".att.mask")}

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            f"mismatch loading stage-2 weights: missing={missing}, unexpected={unexpected}"
        )

    print(f"[green]loaded stage-2 weights from {path}[/green]")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    cfg = SFT_CFG

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Device: {device}")

    if not use_amp:
        print(
            "[yellow]"
            "CUDA not available — running in fp32 on CPU. "
            "This will be very slow for a 124M model; only expect this "
            "path to be hit during local smoke-testing."
            "[/yellow]"
        )

    if torch.cuda.is_available() and (torch.cuda.memory_allocated(device) / 1e9 > 0.5):
        print(
            "[red]stale CUDA context detected, restart the kernel before continuing[/red]"
        )

    os.makedirs(cfg["out_dir"], exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")
    ctx_len = CONFIG_124m["context_len"]

    # ----------------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------------
    train_ds = SFTDataset(cfg["train_path"], tokenizer, ctx_len)
    val_ds = SFTDataset(cfg["val_path"], tokenizer, ctx_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ----------------------------------------------------------------------
    # Training steps
    # ----------------------------------------------------------------------
    steps_per_epoch = len(train_ds) // (cfg["batch_size"] * cfg["grad_accum_steps"])

    if steps_per_epoch == 0:
        raise ValueError(
            f"train_ds has {len(train_ds)} examples, which is smaller than "
            f"batch_size * grad_accum_steps = {cfg['batch_size'] * cfg['grad_accum_steps']}. "
            "steps_per_epoch would be 0 and training would silently no-op. "
            "Lower batch_size/grad_accum_steps or grow the dataset."
        )

    max_steps = steps_per_epoch * cfg["num_epochs"]

    print(
        f"[yellow]{steps_per_epoch} steps/epoch x {cfg['num_epochs']} epochs "
        f"= {max_steps} total steps[/yellow]"
    )

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------
    model = Model(CONFIG_124m).to(device)

    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg["max_lr"],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = os.path.join(cfg["out_dir"], "checkpoint.pt")

    # ----------------------------------------------------------------------
    # Resume / Stage-2
    # ----------------------------------------------------------------------
    if cfg.get("resume_from_input") and not os.path.exists(ckpt_path):
        import shutil

        shutil.copy(cfg["resume_from_input"], ckpt_path)
        print(f"[yellow]copied SFT checkpoint forward from {cfg['resume_from_input']}[/yellow]")

    if os.path.exists(ckpt_path):
        start_step = load_checkpoint(ckpt_path, model, optimizer, scaler, device)
        print(f"[yellow]resumed SFT at step {start_step}[/yellow]")
    else:
        load_stage2_weights(model, cfg["stage2_weights"], device)
        start_step = 0
        print("[green]starting SFT at step 0[/green]")

    # ----------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------
    train_iter = iter(train_loader)

    val_log_path = os.path.join(cfg["out_dir"], "val_loss_log.csv")

    if not os.path.exists(val_log_path):
        with open(val_log_path, "w", encoding="utf-8") as f:
            f.write("step,val_loss\n")

    t0 = time.time()

    for step in range(start_step, max_steps):
        lr = get_lr(step, max_steps, cfg)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        for _ in range(cfg["grad_accum_steps"]):
            try:
                input_ids, labels, pad_mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, labels, pad_mask = next(train_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            pad_mask = pad_mask.to(device)

            attn_mask = build_attn_mask(pad_mask, device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids, attn_mask)
                loss = compute_sft_loss(logits, labels)
                loss = loss / cfg["grad_accum_steps"]

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % cfg["log_every"] == 0:
            dt = time.time() - t0

            print(
                f"step {step:05d} | loss {accum_loss:.4f} | lr {lr:.2e} | "
                f"{dt / cfg['log_every']:.2f}s/step"
            )

            t0 = time.time()

        if step % cfg["eval_every"] == 0 and step > 0:
            val_loss = estimate_loss(model, val_loader, device, cfg["eval_iters"], use_amp)

            print(f"[bold green]step {step:05d} val_loss {val_loss:.4f}[/bold green]")

            with open(val_log_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{val_loss}\n")

            sample_identity_checks(
                model=model,
                tokenizer=tokenizer,
                device=device,
                ctx_len=ctx_len,
                temp=cfg["sample_temp"],
                top_k=cfg["sample_top_k"],
                max_new_tokens=cfg["sample_max_new_tokens"],
            )

        if step % cfg["ckpt_every"] == 0 and step > 0:
            save_checkpoint(ckpt_path, model, optimizer, scaler, step + 1)
            print(f"[dim]checkpoint saved at step {step}[/dim]")

    save_checkpoint(ckpt_path, model, optimizer, scaler, max_steps)

    final_weights_path = os.path.join(cfg["out_dir"], "vader.pth")
    torch.save(model.state_dict(), final_weights_path)

    print("[bold green]SFT complete, final weights saved[/bold green]")
    print(f"[dim]final weights: {final_weights_path}[/dim]")
    print(f"[dim]checkpoint: {ckpt_path}[/dim]")


if __name__ == "__main__":
    main()
