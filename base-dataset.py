import json
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
OUT_DIR = "/kaggle/working/data"
MAX_TOKENS = 2_000_000_000
VAL_EVERY_N = 200                # 1 in 200 examples (~0.5%) goes to validation
FLUSH_EVERY_TOKENS = 2_000_000   # disk write cadence, tune down if you distrust the connection


def load_progress(progress_path):
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            return json.load(f)
    return {"examples_seen": 0, "train_tokens": 0, "val_tokens": 0}


def save_progress(progress_path, state):
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, progress_path)


def append_tokens(filename, token_ids):
    if not token_ids:
        return
    with open(filename, "ab") as f:
        np.array(token_ids, dtype=np.uint16).tofile(f)


def main(out_dir=OUT_DIR, max_tokens=MAX_TOKENS):
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "validation.bin")
    progress_path = os.path.join(out_dir, "progress.json")

    state = load_progress(progress_path)
    if state["examples_seen"] > 0:
        print(
            f"resuming: {state['examples_seen']:,} examples already consumed, "
            f"{state['train_tokens']:,} train tokens already on disk"
        )

    if state["train_tokens"] >= max_tokens:
        print("already at or past max_tokens, nothing to do")
        return

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)
    if state["examples_seen"] > 0:
        ds = ds.skip(state["examples_seen"])

    train_buffer, val_buffer = [], []
    buffered_tokens = 0

    pbar = tqdm(total=max_tokens, initial=state["train_tokens"], unit="tok", desc="train tokens")

    for example in ds:
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)

        if (state["examples_seen"] % VAL_EVERY_N) == 0:
            val_buffer.extend(ids)
        else:
            train_buffer.extend(ids)
            pbar.update(len(ids))

        state["examples_seen"] += 1
        buffered_tokens += len(ids)

        if buffered_tokens >= FLUSH_EVERY_TOKENS:
            append_tokens(train_path, train_buffer)
            append_tokens(val_path, val_buffer)
            state["train_tokens"] += len(train_buffer)
            state["val_tokens"] += len(val_buffer)
            save_progress(progress_path, state)
            train_buffer, val_buffer = [], []
            buffered_tokens = 0

        if state["train_tokens"] + len(train_buffer) >= max_tokens:
            break

    append_tokens(train_path, train_buffer)
    append_tokens(val_path, val_buffer)
    state["train_tokens"] += len(train_buffer)
    state["val_tokens"] += len(val_buffer)
    save_progress(progress_path, state)
    pbar.close()

    print(f"done: {state['train_tokens']:,} train tokens, {state['val_tokens']:,} val tokens in {out_dir}")


if __name__ == "__main__":
    main()
