"""Step 7. Sampling (temperature, top-k, top-p) and generation with a KV cache.

    python generate.py --ckpt out/gpt2/ckpt_best.pt --prompt "ROMEO:" --tokens 200
    python generate.py --ckpt out/gpt2/ckpt_best.pt --benchmark

Grade with:  pytest tests/test_generate.py        (from code/tiny_gpt)
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F


def sample_next(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
    """logits: (B, V) for the last position. Returns the next token ids, shape (B, 1)."""
    # Do not modify `logits` in place: the caller may still need them.
    # TODO, in this order:
    #   1. temperature == 0.0 -> return the argmax (greedy)
    #   2. divide the logits by the temperature
    #   3. top_k > 0  -> set every logit below the k-th largest to -inf (k may exceed the vocabulary)
    #   4. softmax
    #   5. top_p < 1  -> sort probabilities in descending order; drop each token whose preceding
    #                    cumulative mass is already >= top_p (this keeps the token that crosses
    #                    top_p); scatter back to the original order and renormalize
    #   6. torch.multinomial(probs, num_samples=1)
    raise NotImplementedError("generate.sample_next")


@torch.no_grad()
def generate(model, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0,
             top_p: float = 1.0, use_cache: bool = True):
    """idx: (B, T) prompt ids. Returns (B, T + max_new_tokens)."""
    # TODO: remember model.training, switch to eval mode, and restore the mode before returning.
    # Each step produces logits for the last position, samples with sample_next, and appends.
    #   use_cache=False: run model(idx[:, -block_size:]) every step.
    #   use_cache=True:  first step: cache = model.new_cache(); prefill with the (cropped) prompt.
    #                    later steps: feed only the newest token, model(idx[:, -1:], cache=cache).
    #                    when the cache already holds block_size positions: start a new cache and
    #                    prefill it with the last block_size // 2 tokens.
    raise NotImplementedError("generate.generate")


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
