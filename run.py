import torch
import torch.nn as nn
import tiktoken
from rich import print

CONFIG_124m = {
    "vocab_size": 50257,
    "context_len": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

TRAINED_CTX = 256

MODEL = "vader.pth"


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
            nn.GELU(
                approximate="tanh"
            ),  # parameter-free, identical formula to the hand-rolled version, safe
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
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def generate(
    model, idx, max_new_tokens, context_size, temp=0.0, top_k=None, eos_id=None
):
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


def load_weights(model, path, device):
    raw_sd = torch.load(path, map_location=device)
    filtered_sd = {k: v for k, v in raw_sd.items() if not k.endswith(".att.mask")}

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    dropped = len(raw_sd) - len(filtered_sd)
    if dropped:
        print(
            f"[dim]dropped {dropped} legacy 'mask' buffer keys from the old training script[/dim]"
        )
    if missing:
        raise RuntimeError(f"missing keys after filtering, real mismatch: {missing}")
    if unexpected:
        raise RuntimeError(
            f"unexpected keys after filtering, real mismatch: {unexpected}"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")

    model = Model(CONFIG_124m)
    load_weights(model, MODEL, device)
    model.to(device)
    model.eval()

    context_size = TRAINED_CTX

    prompt = "Every effort moves you"
    encoded = text_to_token_ids(prompt, tokenizer).to(device)

    token_ids = generate(
        model=model,
        idx=encoded,
        max_new_tokens=50,
        context_size=context_size,
        temp=0.8,
        top_k=40,
        eos_id=None,
    )
    print(token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
