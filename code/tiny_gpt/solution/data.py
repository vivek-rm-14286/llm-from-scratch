"""Data pipeline: download text, train the tokenizer, tokenize ONCE to uint16 files,
then sample random windows from a memory map.

    python data.py                                  # tiny Shakespeare, about 1 MB
    python data.py --dataset tinystories --stories 20000 --vocab-size 4096 --out data/tinystories
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np
import torch

from bpe import ByteBPE

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
EOT = "<|endoftext|>"


def load_text(dataset: str, out_dir: str, stories: int = 20000) -> str:
    """Return the corpus as one string. Documents are separated by EOT."""
    if dataset == "shakespeare":
        path = os.path.join(out_dir, "input.txt")
        if not os.path.exists(path):
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        with open(path, encoding="utf-8") as f:
            return f.read()
    if dataset == "tinystories":
        from datasets import load_dataset                     # optional dependency
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        return EOT.join(row["text"].strip() for row in ds.take(stories)) + EOT
    with open(dataset, encoding="utf-8") as f:                # any local text file
        return f.read()


def prepare(dataset: str = "shakespeare", out_dir: str = "data/shakespeare", vocab_size: int = 2048,
            val_frac: float = 0.1, stories: int = 20000, tokenizer_chars: int = 2_000_000) -> dict:
    """Write tokenizer.json, train.bin, val.bin and meta.json into `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    text = load_text(dataset, out_dir, stories)

    t0 = time.time()
    tok = ByteBPE()
    tok.train(text[:tokenizer_chars].replace(EOT, "\n"), vocab_size - 1)   # leave one id for EOT
    tok.add_special_tokens([EOT])
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    t1 = time.time()

    ids = np.array(tok.encode(text), dtype=np.uint16)         # vocab <= 65,536 fits in 2 bytes
    t2 = time.time()
    n_val = int(len(ids) * val_frac)
    ids[:-n_val].tofile(os.path.join(out_dir, "train.bin"))   # the validation split is the tail
    ids[-n_val:].tofile(os.path.join(out_dir, "val.bin"))

    meta = {"dataset": dataset, "vocab_size": len(tok.vocab), "chars": len(text),
            "train_tokens": int(len(ids) - n_val), "val_tokens": n_val,
            "bytes_per_token": round(len(text.encode("utf-8")) / len(ids), 3),
            "tokenizer_train_s": round(t1 - t0, 1), "encode_s": round(t2 - t1, 1)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_tokens(path: str) -> np.memmap:
    """Memory-map a .bin file. Nothing is read until a window is sliced."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size: int, block_size: int, device="cpu", generator=None):
    """Random windows. x: (B, T) int64, y: x shifted left by one token."""
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator).tolist()
    x = np.stack([data[s:s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + 1 + block_size] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shakespeare", help="shakespeare | tinystories | path to a .txt file")
    ap.add_argument("--out", default="data/shakespeare")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--stories", type=int, default=20000, help="tinystories only: number of stories to stream")
    args = ap.parse_args()
    meta = prepare(args.dataset, args.out, args.vocab_size, stories=args.stories)
    print(json.dumps(meta, indent=2))
    tok = ByteBPE.load(os.path.join(args.out, "tokenizer.json"))
    window = load_tokens(os.path.join(args.out, "val.bin"))[:40]
    print("first validation tokens:", [tok.decode([i]) for i in window[:12]])
    print(repr(tok.decode(window)))
