# Vader

A GPT-2-scale language model built from scratch, trained on a single free-tier GPU. No frameworks doing the heavy lifting, no pretrained checkpoints, the transformer, tokenisation pipeline, and training loop are all hand-written.

## Architecture

Standard decoder-only transformer, GPT-2 recipe:

| | |
|---|---|
| Vocab size | 50,257 (GPT-2 BPE via `tiktoken`) |
| Context length | 1024 |
| Embedding dim | 768 |
| Layers | 12 |
| Attention heads | 12 |
| Dropout | 0.1 |
| Position encoding | Learned absolute (`nn.Embedding`) |
| Weight tying | Token embedding ↔ output head |

## Training

- Single-GPU, no distributed training (Kaggle free-tier T4/P100, weekly quota-constrained)
- Effective batch size 128 (`batch_size=8 × grad_accum_steps=16`)
- AdamW, cosine LR schedule with linear warmup
- Mixed precision (fp16 autocast + `GradScaler`)
- Checkpointing supports resuming across sessions, and across separate accounts when a single session's quota runs out mid-training
- Data pipeline is memmap-backed (`np.memmap` over pre-tokenised `uint16` `.bin` files), avoids loading the full corpus into RAM

### Repo structure

```
.
├── train.py                      # main training loop, model definition
├── base_dataset.py               # tokenises raw text corpus into train/validation .bin
└── README.md
```

### Running it

Requires `torch`, `tiktoken`, `numpy`, `rich`.

```bash
pip install torch tiktoken numpy rich
python base_dataset.py       # produces data/train.bin, data/validation.bin
python train.py              # resumes from out/checkpoint.pt if present, else starts fresh
```

Config lives in `TRAIN_CFG` and `GPT_CONFIG_124m` at the top of `train.py`, no CLI flags currently.

## Roadmap

1. **Base pretrain** — in progress
2. **Context extension + domain mix-in** — extend `train_ctx` toward the full 1024-token `context_len`, combined with mixing in a Star Wars/Wookieepedia corpus (CC BY-SA 3.0) for domain flavour
3. **Instruction fine-tuning** — SFT on an open instruction dataset once context length supports realistic instruction/response pairs
4. **Packaging** — convert to sharded `.safetensors` + `config.json` + tokenizer files for standard `transformers`-compatible distribution

## Data & licensing

- Base training corpus: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` config, streamed and tokenised up to a 2B-token cap (~0.5% held out for validation). Licensed under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) (Open Data Commons Attribution), use is also subject to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use) since FineWeb-Edu is Common Crawl-derived.
- Star Wars domain data: scraped from [Wookieepedia](https://starwars.fandom.com), licensed CC BY-SA 3.0. Only tokenised `.bin` output is intended for public release.

## Author

Built by [thehackersbrain](https://github.com/thehackersbrain).
