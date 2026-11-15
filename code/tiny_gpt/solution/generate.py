"""Sampling (temperature, top-k, top-p) and generation with a KV cache.

    python generate.py --ckpt out/gpt2/ckpt_best.pt --prompt "ROMEO:" --tokens 200
    python generate.py --ckpt out/gpt2/ckpt_best.pt --benchmark
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F


def sample_next(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
    """logits: (B, V) for the last position. Returns the next token ids, shape (B, 1)."""
    logits = logits.float()
    if temperature == 0.0:                                        # greedy
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        before = sorted_p.cumsum(dim=-1) - sorted_p               # probability mass ranked above each token
        sorted_p = sorted_p.masked_fill(before >= top_p, 0.0)     # keeps the token that crosses top_p
        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_p)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0,
             top_p: float = 1.0, use_cache: bool = True):
    """idx: (B, T) prompt ids. Returns (B, T + max_new_tokens)."""
    was_training = model.training
    model.eval()
    block = model.cfg.block_size
    cache = None
    for _ in range(max_new_tokens):
        if not use_cache:
            logits = model(idx[:, -block:])                       # re-run the whole window
        elif cache is None or cache[0]["k"].size(2) >= block:
            # First step, or the cache is full: rebuild it from the most recent tokens.
            # After an overflow keep half a window, so the next rebuild is block/2 tokens away.
            keep = block if cache is None else block // 2
            cache = model.new_cache()
            logits = model(idx[:, -keep:], cache=cache)           # prefill
        else:
            logits = model(idx[:, -1:], cache=cache)              # decode: one new token
        next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p)
        idx = torch.cat([idx, next_id], dim=1)
    model.train(was_training)
    return idx


if __name__ == "__main__":
    from bpe import ByteBPE
    from train import load_checkpoint, pick_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/gpt2/ckpt_best.pt")
    ap.add_argument("--data", default="data/shakespeare", help="directory holding tokenizer.json")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="time cached vs uncached decoding")
    ap.add_argument("--device", default=pick_device())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteBPE.load(os.path.join(args.data, "tokenizer.json"))
    model, ckpt = load_checkpoint(args.ckpt, args.device)
    idx = torch.tensor([tok.encode(args.prompt)], device=args.device)
    if args.benchmark:
        n = model.cfg.block_size - idx.size(1)                    # stay inside one window
        for use_cache in (False, True):
            generate(model, idx, 8, 0.0, use_cache=use_cache)     # warm up
            t0 = time.time()
            generate(model, idx, n, 0.0, use_cache=use_cache)
            dt = time.time() - t0
            print(f"use_cache={use_cache}: {n} tokens in {dt:.2f}s = {n / dt:.0f} tok/s")
    else:
        out = generate(model, idx, args.tokens, args.temperature, args.top_k, args.top_p, not args.no_cache)
        print(tok.decode(out[0].tolist()))
