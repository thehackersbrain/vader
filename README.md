# Vader

A GPT-2-scale language model built from scratch, trained on a single free-tier GPU. No frameworks doing the heavy lifting, no pretrained checkpoints, the transformer, tokenisation pipeline, and training loop are all hand-written.

Base pretrain, context-extension + Star Wars domain mix-in, and instruction fine-tuning with a hybrid factual/Star Wars-flavoured identity are all complete. Packaging for standard `transformers`-style distribution is in progress.

## Architecture

Standard decoder-only transformer, GPT-2 recipe:

|                   |                                   |
| ----------------- | --------------------------------- |
| Vocab size        | 50,257 (GPT-2 BPE via `tiktoken`) |
| Context length    | 1024                              |
| Embedding dim     | 768                               |
| Layers            | 12                                |
| Attention heads   | 12                                |
| Dropout           | 0.1                               |
| Position encoding | Learned absolute (`nn.Embedding`) |
| Weight tying      | Token embedding ↔ output head    |

Attention is implemented via `torch.nn.functional.scaled_dot_product_attention` throughout (no manually materialised attention matrix or causal mask buffer). Pretraining and the context-extension stage use `is_causal=True`; SFT builds an explicit combined causal + padding boolean mask, since instruction/response pairs are variable-length and right-padded.

## Training

Three-stage pipeline, each stage picking up from the previous stage's final weights:

**1. Base pretrain** — `train.py`

- Trained at `train_ctx=256` (positions 256–1023 of the 1024-token embedding table start untrained)
- Corpus: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT`, capped at 2B tokens
- AdamW, cosine LR schedule with linear warmup, mixed precision (fp16 autocast + `GradScaler`)
- Effective batch size 128 (`batch_size=8 × grad_accum_steps=16`)

**2. Context extension + Star Wars mix-in** — `train_stage2.py`

- Continues from the base checkpoint at full `train_ctx=1024`, populating the previously-untrained position range
- Mixes in the Star Wars corpus (see Data & Licensing) at a small fraction of batches (`sw_mix_ratio`) alongside continued fineweb-edu training, so general capability isn't overwritten by narrow domain content
- Lower LR than base pretrain (continued pretraining on an already-trained model, not a fresh run)

**3. Instruction fine-tuning (SFT)** — `train_sft.py`

- Continues from the stage-2 checkpoint
- Alpaca-style instruction format, with and without a system-prompt block (`### System: ... ### Instruction: ... ### Response:`), so the model learns to condition on a system prompt rather than ignoring it as out-of-distribution input
- Response-only loss masking (`ignore_index=-100` over the prompt span) — the model is trained to predict the response, not reconstruct the prompt
- A small hand-written identity block (name, creator, Star Wars namesake) is duplicated into the training mix at a set repetition rate, trained in both system-prompt and no-system-prompt form
- Lowest LR of the three stages

All three training scripts share the same checkpoint/resume pattern: resuming across sessions, and across separate Kaggle accounts when a single session's weekly quota runs out mid-training. Data pipelines for stages 1–2 are memmap-backed (`np.memmap` over pre-tokenised `uint16` `.bin` files); SFT uses a per-example, dynamically-padded dataset instead, since example boundaries matter for loss masking.

### Repo structure

```
.
├── train.py                    # stage 1: base pretrain, model definition
├── train_stage2.py              # stage 2: context extension + Star Wars mix-in
├── train_sft.py                 # stage 3: instruction fine-tuning with identity
├── run.py                       # inference — single-prompt or interactive chat
├── base-dataset.py              # tokenises fineweb-edu into train/validation .bin
├── star-wars-dataset.py         # scrapes + converts the Star Wars domain corpus (sanity/scrape/convert)
├── tokenize_star_wars.py        # tokenises the published Star Wars dataset into stage-2 train/validation .bin
├── build_sft_dataset.py         # builds the instruction + identity SFT dataset (Alpaca + system-prompt format)
├── convert_to_safetensors.py    # packaging: .pth -> .safetensors + config.json
└── README.md
```

### Running it

Requires `torch`, `tiktoken`, `numpy`, `rich`, `datasets`, `tqdm`, `requests`, `pandas`, `pyarrow`, `mwparserfromhell`, `safetensors`.

```bash
pip install torch tiktoken numpy rich datasets tqdm requests pandas pyarrow mwparserfromhell safetensors

# stage 1: base pretrain
python base-dataset.py                     # produces data/train.bin, data/validation.bin
python train.py                            # resumes from out/checkpoint.pt if present, else starts fresh

# stage 2: context extension + Star Wars mix-in
python star-wars-dataset.py sanity         # confirms the API returns real text before committing to a full scrape
python star-wars-dataset.py scrape         # scrapes Wookieepedia -> star_wars_articles.jsonl
python star-wars-dataset.py convert        # converts JSONL -> star_wars_corpus.parquet, publish to HF from here
python tokenize_star_wars.py               # tokenises the published dataset -> data/star_wars/{train,validation}.bin
python train_stage2.py                     # continues from stage 1's final weights

# stage 3: instruction fine-tuning
python build_sft_dataset.py                # -> sft_train.jsonl, sft_val.jsonl
python train_sft.py                        # continues from stage 2's final weights

# inference
python run.py                              # interactive chat, Vader's system prompt + identity active
python run.py --prompt "who are you?"      # single-shot
python run.py --no-system --prompt "..."   # generic instruction template, no identity framing

# packaging
python convert_to_safetensors.py --input vader.pth --output model.safetensors --config-output config.json
```

Config for each training stage lives in a `*_CFG` dict at the top of its script, no CLI flags. `star-wars-dataset.py` takes `--out-dir`/`--resume-from` for `scrape` and `--input`/`--output` for `convert`; `run.py` takes `--prompt`, `--temperature`, `--top-k`, `--max-new-tokens`, `--context-size`, `--no-system`. Run any script with `-h` for details. Always run `sanity` before `scrape`, it's a two-second check that catches API response issues before they silently burn hours of runtime.

## Roadmap

1. ~~**Base pretrain**~~ — complete
2. ~~**Context extension + domain mix-in**~~ — complete
3. ~~**Instruction fine-tuning**~~ — complete
4. **Packaging** — in progress. `.pth` → `.safetensors` conversion done (weight-tying handled: `out_head.weight` is dropped before saving since it's aliased to `tok_emb.weight`, and re-tied on load). Still open: whether to ship a custom `modeling_vader.py` for `trust_remote_code=True` loading, or remap weights into genuine `GPT2LMHeadModel` format for zero-custom-code `transformers` compatibility.

## Data & licensing

- Base training corpus: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` config, streamed and tokenised up to a 2B-token cap (~0.5% held out for validation). Licensed under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) (Open Data Commons Attribution), use is also subject to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use) since FineWeb-Edu is Common Crawl-derived.
- Star Wars domain data: scraped from [Wookieepedia](https://starwars.fandom.com), licensed CC BY-SA 3.0. Full dataset published on HuggingFace: [thehackersbrain/star-wars-dataset](https://huggingface.co/datasets/thehackersbrain/star-wars-dataset) (211,065 articles, ~82.4M tokens).
- Instruction data: [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned), plus a small hand-written identity set.

## License

Code: [add your chosen license — MIT/Apache-2.0 are the common picks].

Model weights are a derivative work of ODC-By-licensed (FineWeb-Edu) and CC BY-SA-licensed (Wookieepedia/Star Wars corpus) data. ODC-By requires attribution on redistribution, provided above; CC BY-SA's ShareAlike clause is worth reading properly before publishing checkpoints, it may require derivative works — arguably including trained weights — to carry a compatible share-alike license. Code and weights don't need to share one license.

## Author

Built by [Gaurav Raj (@thehackersbrain)](https://thehackersbrain.dev).
