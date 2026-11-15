"""Step 4. Data pipeline: download text, train the tokenizer, tokenize ONCE to uint16
files, then sample random windows from a memory map.

    python data.py                                  # tiny Shakespeare, about 1 MB
    python data.py --dataset tinystories --stories 20000 --vocab-size 4096 --out data/tinystories

Grade with:  pytest tests/test_train.py -k "prepare or batch"        (from code/tiny_gpt)
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
    """Return the corpus as one string. Documents are separated by EOT. (Given.)"""
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
    """Write four files into `out_dir` and return the meta dict.

    tokenizer.json  a ByteBPE trained on the first `tokenizer_chars` characters (with EOT replaced
                    by a newline) to vocab_size - 1 ids, plus the special token EOT as the last id,
                    so that len(tok.vocab) == vocab_size
    train.bin       the first (1 - val_frac) of the token ids, as raw uint16
    val.bin         the last val_frac of the token ids, as raw uint16
    meta.json       at least {"vocab_size", "train_tokens", "val_tokens"}
    """
    os.makedirs(out_dir, exist_ok=True)
    text = load_text(dataset, out_dir, stories)
    # TODO: train and save the tokenizer; encode the WHOLE text once into a np.uint16 array;
    #       split off the tail as validation; write both with ndarray.tofile; write meta.json.
    raise NotImplementedError("data.prepare")


def load_tokens(path: str) -> np.memmap:
    """Memory-map a .bin file written by `prepare` (dtype uint16, read-only)."""
    # TODO
    raise NotImplementedError("data.load_tokens")


def get_batch(data, batch_size: int, block_size: int, device="cpu", generator=None):
    """Random windows from a 1-D token array.

    Returns x, y: int64 tensors of shape (batch_size, block_size) on `device`, where
    y is x shifted left by one token (y[b, t] is the token that follows x[b, t]).
    Draw the start offsets with torch.randint(..., generator=generator) so a run can be
    reproduced and resumed. Never index past the end of `data`.
    """
    # TODO
    raise NotImplementedError("data.get_batch")


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
