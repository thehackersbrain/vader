# Vader — Next Objectives

Status as of last check: base pretrain in progress, step ~49,220 / 61,035, val_loss 3.4090 (new low). GPT-2-124M architecture (85.4M unique params, weight-tied), single Kaggle GPU, `train_ctx=256`.

## 1. Finish base pretrain (current)

- Let the cosine schedule run to `max_steps=61,035`.
- Keep checkpoint/dataset hopping across Kaggle accounts as needed (`train.bin`/`validation.bin` already public, only `checkpoint.pt` needs manual transfer).
- Optional, no urgency: restart kernel clean before next resume (stale CUDA context flagged 2.26GB allocated pre-init last session).

## 2. Combined context-extension + Star Wars mix-in

Merged into a single continued-pretraining pass rather than two sequential ones, saves a full extra pass over the data.

- [ ] Finish Wookieepedia scrape (`scrape_wookieepedia.py`) — CC BY-SA 3.0, plaintext via Fandom API
- [ ] Tokenise via `prepare_star_wars_data.py`, note total token count
- [ ] Recompute `sw_mix_ratio` from actual token count: `target_exposure_tokens / remaining_tokens_in_pass`
- [ ] Bump `train_ctx` from 256 → 512 or 1024 for this pass, so positions currently untrained (256–1023) get gradient signal
- [ ] Use a lower LR than base `max_lr` (continued pretrain, not a fresh run) — cosine re-warmup over a short schedule, not the full 61k-step curve
- [ ] Consider structuring some Wookieepedia articles as longer documents during this pass — doubles as context-extension data and domain data in the same batch
- [ ] Wire `sw_available` mixing logic (already built, currently inert with no `data_dir_sw` populated)
- [ ] Log val_loss to CSV this time (`val_loss_log.csv` in `out_dir`) — 2-line addition, survives checkpoint restarts, makes the eventual trend plot trivial instead of scrollback archaeology

## 3. Instruction fine-tuning (SFT)

Comes last, deliberately — format/behaviour layer on top of finished content + context work, so it's only done once.

- [ ] Pick an existing open instruction dataset (Alpaca-style / Dolly / OASST) rather than building one from scratch
- [ ] Confirm dataset fits within the now-extended context length (512/1024) — this was the original blocker at `train_ctx=256`
- [ ] Add chat-formatting logic (role tokens, EOS handling) — Vader has none of this currently
- [ ] Lower LR again relative to stage 2, SFT typically wants a smaller, shorter run than pretraining
- [ ] Sanity-check for catastrophic forgetting of general + Star Wars knowledge post-SFT (quick qualitative sample check, not a formal eval)

## 4. Package for distribution

- [ ] Convert final `.pth` → sharded `.safetensors` (strip `out_head.weight` before saving — tied to `tok_emb.weight`, re-tie on load instead of duplicating on disk)
- [ ] Write `config.json` from `CONFIG_124m` (architecture spec, mirrors what training script currently hardcodes)
- [ ] Export tokenizer: `vocab.json` + `merges.txt` (extractable from tiktoken's GPT-2 BPE data) or wrap in HF's fast-tokenizer `tokenizer.json`
- [ ] `generation_config.json` — default sample_temp/top_k, same values already used during training eval
- [ ] `chat_template.jinja` — only relevant post-SFT, defines how role-tagged messages serialise to raw token strings
- [ ] `model.safetensors.index.json` — shard map, needed if weights split across multiple files
- [ ] Dataset card / README noting Wookieepedia (CC BY-SA 3.0) as a data source if the tokenised Star Wars corpus or any raw scrape output goes public

## Open decisions (not blocking, revisit when relevant)

- Exact `train_ctx` target for stage 2: 512 (cheaper, still a meaningful jump) vs 1024 (uses full `context_len`, costs more compute per step — quadratic attention scaling)
- Exact `sw_mix_ratio` — depends on final scraped token count, not yet known
- Whether raw Wookieepedia scrape output should ever go public (attribution/share-alike obligation under CC BY-SA if so; tokenised `.bin` files carry no such obligation)
