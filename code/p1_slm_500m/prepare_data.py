"""Tokenize a corpus into flat binary shards that train.py memory-maps.

Each shard is a raw array of token ids (uint16 if vocab < 65536, else uint32),
with <|endoftext|> between documents. No headers: dtype and counts live in
<out>/meta.json.

    # 10B tokens of FineWeb-Edu (the full sample-10BT subset)
    python prepare_data.py --tokenizer tokenizer/tokenizer.json --out data/fwe \
        --source HuggingFaceFW/fineweb-edu:sample-10BT:text:1.0 --total_tokens 10e9

    # A mix: 85% FineWeb-Edu, 10% FineMath, 5% Python code
    python prepare_data.py --tokenizer tokenizer/tokenizer.json --out data/mix \
        --total_tokens 50e9 \
        --source HuggingFaceFW/fineweb-edu:sample-100BT:text:0.85 \
        --source HuggingFaceTB/finemath:finemath-4plus:text:0.10 \
        --source codeparrot/codeparrot-clean::content:0.05

    # Offline: a local text file, or the built-in synthetic corpus used by --tiny
    python prepare_data.py --local_text corpus.txt --tokenizer tok.json --out data/local
    python prepare_data.py --tiny --out data/tiny
"""
import argparse
import json
import os
import random
from multiprocessing import Pool

import numpy as np

_TOK = None  # per-worker tokenizer


def _init_worker(tok_path):
    global _TOK
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(tok_path)


def _encode_batch(texts):
    eot = _TOK.token_to_id("<|endoftext|>")
    out = []
    for enc in _TOK.encode_batch(texts, add_special_tokens=False):
        out.extend(enc.ids)
        out.append(eot)
    return out


def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


class ShardWriter:
    """Accumulates tokens and writes fixed-size shards; the first `val_tokens` go to val."""

    def __init__(self, out_dir, prefix, dtype, shard_tokens, val_tokens):
        self.out_dir, self.prefix, self.dtype = out_dir, prefix, dtype
        self.shard_tokens, self.val_tokens = shard_tokens, val_tokens
        # A preallocated numpy buffer: a Python list of 100M ints would need ~3 GB.
        self.buf = np.empty(shard_tokens, dtype=dtype)
        self.fill, self.n_shards, self.total = 0, 0, 0
        self.val = np.empty(val_tokens, dtype=dtype)
        self.val_fill, self.val_done = 0, val_tokens == 0

    def add(self, ids):
        ids = np.asarray(ids, dtype=self.dtype)
        if not self.val_done:
            take = min(len(ids), self.val_tokens - self.val_fill)
            self.val[self.val_fill:self.val_fill + take] = ids[:take]
            self.val_fill += take
            ids = ids[take:]
            if self.val_fill == self.val_tokens:
                self._flush_val()
        while len(ids):
            take = min(len(ids), self.shard_tokens - self.fill)
            self.buf[self.fill:self.fill + take] = ids[:take]
            self.fill += take
            self.total += take
            ids = ids[take:]
            if self.fill == self.shard_tokens:
                self._flush_train()

    def _flush_val(self):
        self.val[: self.val_fill].tofile(os.path.join(self.out_dir, f"{self.prefix}_val.bin"))
        self.val_done = True

    def _flush_train(self):
        name = f"{self.prefix}_train_{self.n_shards:05d}.bin"
        self.buf[: self.fill].tofile(os.path.join(self.out_dir, name))
        self.n_shards += 1
        self.fill = 0

    def close(self):
        if not self.val_done and self.val_fill:
            self._flush_val()
        if self.fill:
            self._flush_train()


def tokenize_stream(texts, tok_path, writer, budget, workers, batch_docs=512):
    """Tokenize `texts` in parallel until `budget` train tokens are written.

    Pool.imap would eagerly pull the *entire* streaming dataset into memory, so we
    keep a bounded window of in-flight batches instead (simple backpressure).
    Results are consumed in submission order, so output is deterministic.
    """
    from collections import deque
    max_inflight = workers * 4
    with Pool(workers, initializer=_init_worker, initargs=(tok_path,)) as pool:
        inflight, n_done = deque(), 0
        batches = batched(texts, batch_docs)
        exhausted = False
        while True:
            while not exhausted and len(inflight) < max_inflight:
                try:
                    inflight.append(pool.apply_async(_encode_batch, (next(batches),)))
                except StopIteration:
                    exhausted = True
            if not inflight:
                break
            writer.add(inflight.popleft().get())
            n_done += 1
            if writer.total >= budget:
                break
            if n_done % 2000 == 0:
                print(f"  {writer.prefix}: {writer.total / 1e6:,.0f}M train tokens")
    writer.close()


def hf_texts(dataset, name, field):
    from datasets import load_dataset
    ds = load_dataset(dataset, name=name or None, split="train", streaming=True)
    for ex in ds:
        t = ex[field]
        if t:
            yield t


# ---------------------------------------------------------------------------
# Synthetic corpus for the --tiny smoke test (no downloads needed)
# ---------------------------------------------------------------------------
def synthetic_corpus(n_docs=3000, seed=0):
    rng = random.Random(seed)
    names = ["Ada", "Alan", "Grace", "Linus", "Barbara", "Ken", "Dennis", "Margaret"]
    things = ["the compiler", "a small model", "the river", "an old map", "the garden",
              "a red kite", "the library", "a quiet robot"]
    verbs = ["found", "built", "painted", "repaired", "studied", "followed", "described"]
    places = ["in the morning", "near the sea", "after lunch", "on a hill", "at the lab"]
    docs = []
    for _ in range(n_docs):
        sents = []
        for _ in range(rng.randint(3, 8)):
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            sents.append(rng.choice([
                f"{rng.choice(names)} {rng.choice(verbs)} {rng.choice(things)} {rng.choice(places)}.",
                f"{a} plus {b} equals {a + b}.",
                f"Then {rng.choice(names)} said that {rng.choice(things)} was {rng.choice(['big', 'small', 'new', 'old'])}.",
            ]))
        docs.append(" ".join(sents))
    return docs


def build_tiny(out_dir, vocab_size=512):
    """Synthetic text -> tiny BPE tokenizer -> train/val shards. Takes seconds."""
    from train_tokenizer import train_tokenizer
    os.makedirs(out_dir, exist_ok=True)
    docs = synthetic_corpus()
    tok, tok_path = train_tokenizer(iter(docs), vocab_size, out_dir)
    eot = tok.token_to_id("<|endoftext|>")
    ids = []
    for enc in tok.encode_batch(docs, add_special_tokens=False):
        ids.extend(enc.ids + [eot])
    n_val = len(ids) // 10
    np.asarray(ids[:n_val], dtype=np.uint16).tofile(os.path.join(out_dir, "tiny_val.bin"))
    np.asarray(ids[n_val:], dtype=np.uint16).tofile(os.path.join(out_dir, "tiny_train_00000.bin"))
    meta = dict(dtype="uint16", vocab_size=tok.get_vocab_size(), eot_id=eot,
                tokenizer=os.path.abspath(tok_path), train_tokens=len(ids) - n_val, val_tokens=n_val)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"tiny dataset: {len(ids):,} tokens in {out_dir}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", help="path to tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", default=[],
                    help="dataset:config:text_field:fraction (config may be empty)")
    ap.add_argument("--local_text", help="a local UTF-8 file; blank lines separate documents")
    ap.add_argument("--total_tokens", type=float, default=10e9)
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--val_tokens", type=int, default=10_000_000, help="per source")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    if args.tiny:
        build_tiny(args.out)
        return

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    vocab = tok.get_vocab_size()
    dtype = np.uint16 if vocab < 2**16 else np.uint32
    os.makedirs(args.out, exist_ok=True)

    totals = {}
    if args.local_text:
        from train_tokenizer import file_iterator
        w = ShardWriter(args.out, "local", dtype, args.shard_tokens, args.val_tokens)
        tokenize_stream(file_iterator([args.local_text]), args.tokenizer, w,
                        args.total_tokens, args.workers)
        totals["local"] = w.total
    for spec in args.source:
        dataset, name, field, frac = spec.split(":")
        budget = int(float(frac) * args.total_tokens)
        prefix = dataset.split("/")[-1] + (f"-{name}" if name else "")
        print(f"{prefix}: target {budget / 1e9:.2f}B train tokens")
        w = ShardWriter(args.out, prefix, dtype, args.shard_tokens, args.val_tokens)
        tokenize_stream(hf_texts(dataset, name, field), args.tokenizer, w, budget, args.workers)
        totals[prefix] = w.total

    meta = dict(dtype=np.dtype(dtype).name, vocab_size=vocab,
                eot_id=tok.token_to_id("<|endoftext|>"),
                tokenizer=os.path.abspath(args.tokenizer), train_tokens=totals)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
