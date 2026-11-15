"""Step 3. The sanity checks from the Debugging lesson. Run them before every real training run.

    python checks.py                 # GPT-2 style
    python checks.py --modern
"""
import argparse
import json
import math
import os

import torch

from data import get_batch, load_tokens
from model import GPT, GPTConfig
from train import configure_optimizer


def check_shift(x, y):
    """Assert that y is x shifted left by one token."""
    # TODO
    raise NotImplementedError("checks.check_shift")


@torch.no_grad()
def check_step0_loss(model, x, y, tol: float = 0.15) -> float:
    """An untrained model should predict close to uniform: loss about ln(vocab)."""
    # TODO: eval mode; loss = model(x, y)[1]; assert |loss - ln(vocab_size)| < tol; return the loss.
    raise NotImplementedError("checks.check_step0_loss")


@torch.no_grad()
def check_causal(model, x) -> float:
    """Change the LAST input token. Logits at every earlier position must not move."""
    # TODO: copy x, change the last token of every row, run both through the model, assert the
    #       logits at positions 0..T-2 are (almost exactly) equal, and that the last position
    #       DID change. Return the largest difference over the earlier positions.
    raise NotImplementedError("checks.check_causal")


@torch.no_grad()
def check_cache(model, x, prefill: int = 8) -> float:
    """Logits from prefill + one-token-at-a-time decoding must equal one full forward pass."""
    # TODO: full = model(x). Then with cache = model.new_cache(): feed x[:, :prefill], then one
    #       token at a time; concatenate the logits along time; assert max |difference| < 1e-3
    #       and return it.
    raise NotImplementedError("checks.check_cache")


def check_overfit(cfg: GPTConfig, x, y, steps: int = 150, lr: float = 1e-3, target: float = 0.1) -> float:
    """A fresh model (no dropout) trained on ONE batch must memorize it."""
    # TODO: build a fresh GPT from `cfg` with dropout 0, an optimizer from configure_optimizer
    #       (weight_decay=0.0), and run `steps` updates on the SAME (x, y) with gradient clipping
    #       at 1.0. Assert the final loss is below `target` and return it.
    raise NotImplementedError("checks.check_overfit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/shakespeare")
    ap.add_argument("--modern", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    data = load_tokens(os.path.join(args.data, "train.bin"))
    with open(os.path.join(args.data, "meta.json")) as f:
        vocab = json.load(f)["vocab_size"]
    kw = dict(vocab_size=vocab, block_size=64, n_layer=4, n_head=4, d_model=128)
    cfg = GPTConfig.modern(**kw) if args.modern else GPTConfig(**kw)
    model = GPT(cfg).to(args.device)
    x, y = get_batch(data, 8, cfg.block_size, args.device)

    check_shift(x, y)
    print("shift:    PASS  targets are inputs shifted by one")
    print(f"step 0:   PASS  loss {check_step0_loss(model, x, y):.3f} vs ln({vocab}) = {math.log(vocab):.3f}")
    print(f"causal:   PASS  max change in past logits {check_causal(model, x):.1e}")
    print(f"kv cache: PASS  max |cached - uncached| {check_cache(model, x):.1e}")
    print(f"overfit:  PASS  loss on one batch after 150 steps {check_overfit(cfg, x, y):.3f}")
