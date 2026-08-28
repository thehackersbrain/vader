import argparse

import tiktoken
import torch
import torch.nn as nn
from rich import print
from rich.console import Console


# ==========================================================================
# Configuration
# ==========================================================================
CONFIG_124M = {
    "vocab_size": 50257,
    "context_len": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

MODEL_PATH = "vader.pth"

EOS_ID = 50256

SYSTEM_TEXT = (
    "You are Vader, a language model built by Gaurav Raj (thehackersbrain). "
    "You were named after Darth Vader from Star Wars. Speak with a bit of that flavour, "
    "but you're an AI, not a Sith Lord."
)


# ==========================================================================
# Model
# ==========================================================================
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

    def forward(self, x):
        b, num_tokens, _ = x.shape
        keys = (
            self.w_key(x)
            .view(b, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        queries = (
            self.w_query(x)
            .view(b, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        values = (
            self.w_value(x)
            .view(b, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        context_vec = torch.nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            is_causal=True,
            dropout_p=0.0,
        )
        context_vec = (
            context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        )
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

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_ids = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_ids)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


# ==========================================================================
# Checkpoint loading
# ==========================================================================
def load_weights(model, path, device):
    raw_sd = torch.load(path, map_location=device)
    filtered_sd = {k: v for k, v in raw_sd.items() if not k.endswith(".att.mask")}
    dropped = len(raw_sd) - len(filtered_sd)
    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)

    if dropped:
        print(f"[dim]dropped {dropped} legacy attention-mask key(s)[/dim]")
    if missing:
        raise RuntimeError(
            "Missing model keys after loading:\n"
            + "\n".join(f"  - {k}" for k in missing)
        )
    if unexpected:
        raise RuntimeError(
            "Unexpected model keys after loading:\n"
            + "\n".join(f"  - {k}" for k in unexpected)
        )

    print(f"[green]loaded weights from {path}[/green]")


# ==========================================================================
# Tokenisation
# ==========================================================================
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


# ==========================================================================
# Generation
# ==========================================================================
@torch.inference_mode()
def generate(
    model, idx, max_new_tokens, context_size, temp=0.3, top_k=10, eos_id=EOS_ID
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            top_logits, _ = torch.topk(logits, k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val, torch.full_like(logits, float("-inf")), logits
            )

        if temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ==========================================================================
# Prompt formatting
# ==========================================================================
def build_prompt(instruction, use_system=True):
    if use_system:
        return (
            f"### System:\n{SYSTEM_TEXT}\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n"
    )


# ==========================================================================
# Single generation
# ==========================================================================
def run_prompt(
    model, tokenizer, device, prompt, max_new_tokens, context_size, temp, top_k
):
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    output = generate(
        model=model,
        idx=encoded,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temp=temp,
        top_k=top_k,
        eos_id=EOS_ID,
    )
    prompt_len = encoded.size(1)
    generated_ids = output[0, prompt_len:]
    return tokenizer.decode(generated_ids.tolist())


# ==========================================================================
# Interactive mode
# ==========================================================================
def interactive_loop(
    model,
    tokenizer,
    device,
    max_new_tokens,
    context_size,
    temp,
    top_k,
    use_system=True,
):
    console = Console()

    print()
    print("[bold cyan]VADER interactive mode[/bold cyan]")
    print("[dim]Type 'exit' or 'quit' to stop.[/dim]")
    print()

    while True:
        try:
            instruction = console.input("[bold yellow]You:[/bold yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not instruction:
            continue
        if instruction.lower() in {"exit", "quit"}:
            break

        prompt = build_prompt(instruction, use_system=use_system)

        response = run_prompt(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temp=temp,
            top_k=top_k,
        )

        print(f"[bold cyan]Vader:[/bold cyan] {response.strip()}")
        print()


# ==========================================================================
# CLI
# ==========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run Vader language model inference.")
    parser.add_argument(
        "--model", default=MODEL_PATH, help="Path to Vader .pth weights."
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Run a single instruction instead of interactive mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=CONFIG_124M["context_len"],
        help="Maximum context window.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. 0 = greedy decoding.",
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="Top-k sampling. Use 0 to disable."
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Use the generic instruction template instead of Vader's system prompt.",
    )
    return parser.parse_args()


# ==========================================================================
# Main
# ==========================================================================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bold]Device in use:[/bold] {device}")
    if device.type == "cuda":
        print(f"[dim]GPU: {torch.cuda.get_device_name(0)}[/dim]")

    if not (1 <= args.context_size <= CONFIG_124M["context_len"]):
        raise ValueError(
            f"context_size must be between 1 and {CONFIG_124M['context_len']}"
        )
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if args.temperature < 0:
        raise ValueError("temperature cannot be negative")

    top_k = None if args.top_k <= 0 else args.top_k

    tokenizer = tiktoken.get_encoding("gpt2")

    model = Model(CONFIG_124M)
    load_weights(model, args.model, device)
    model.to(device)
    model.eval()

    print(f"[dim]context window: {args.context_size} tokens[/dim]")
    print(
        f"[dim]temperature: {args.temperature} | top_k: {top_k} | max_new_tokens: {args.max_new_tokens}[/dim]"
    )
    print()

    if args.prompt is not None:
        prompt = build_prompt(args.prompt, use_system=not args.no_system)
        response = run_prompt(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            context_size=args.context_size,
            temp=args.temperature,
            top_k=top_k,
        )
        print("[bold cyan]Vader:[/bold cyan]")
        print(response.strip())
        return

    interactive_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        context_size=args.context_size,
        temp=args.temperature,
        top_k=top_k,
        use_system=not args.no_system,
    )


if __name__ == "__main__":
    main()
