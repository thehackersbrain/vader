import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "/kaggle/working/data/star_wars"
VAL_EVERY_N = 200  # same 0.5% validation split convention as base_dataset.py


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    ds = load_dataset("thehackersbrain/star-wars-dataset", split="train")
    print(f"{len(ds):,} articles")

    train_ids, val_ids = [], []
    for i, row in enumerate(tqdm(ds, desc="tokenising")):
        ids = enc.encode_ordinary(row["text"])
        ids.append(enc.eot_token)
        if i % VAL_EVERY_N == 0:
            val_ids.extend(ids)
        else:
            train_ids.extend(ids)

    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    train_arr.tofile(os.path.join(OUT_DIR, "train.bin"))
    val_arr.tofile(os.path.join(OUT_DIR, "validation.bin"))

    print(f"train: {len(train_arr):,} tokens")
    print(f"val:   {len(val_arr):,} tokens")
    print(f"total: {len(train_arr) + len(val_arr):,} tokens  <-- use this for sw_mix_ratio")


if __name__ == "__main__":
    main()
