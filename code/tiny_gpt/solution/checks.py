"""The sanity checks from the Debugging lesson. Run them before every real training run.

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
    assert (x[:, 1:] == y[:, :-1]).all(), "targets are not the inputs shifted by one"


@torch.no_grad()
def check_step0_loss(model, x, y, tol: float = 0.15) -> float:
    """An untrained model should predict close to uniform: loss about ln(vocab)."""
    model.eval()
    loss = model(x, y)[1].item()
    expected = math.log(model.cfg.vocab_size)
    assert abs(loss - expected) < tol, f"step-0 loss {loss:.3f}, expected about {expected:.3f}"
    return loss


@torch.no_grad()
def check_causal(model, x) -> float:
    """Change the LAST input token. Logits at every earlier position must not move."""
    model.eval()
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % model.cfg.vocab_size
    a, b = model(x), model(x2)
    diff = (a[:, :-1] - b[:, :-1]).abs().max().item()
    assert diff < 1e-5, f"a future token changed past logits by {diff:.2e}: the causal mask leaks"
    assert (a[:, -1] - b[:, -1]).abs().max().item() > 0, "the last position ignores its own input"
    return diff


@torch.no_grad()
def check_cache(model, x, prefill: int = 8) -> float:
    """Logits from prefill + one-token-at-a-time decoding must equal one full forward pass."""
    model.eval()
    full = model(x)
    cache = model.new_cache()
    parts = [model(x[:, :prefill], cache=cache)]
    for t in range(prefill, x.size(1)):
        parts.append(model(x[:, t:t + 1], cache=cache))
    diff = (full - torch.cat(parts, dim=1)).abs().max().item()
    assert diff < 1e-3, f"cached and uncached logits differ by {diff:.2e}"
    return diff


def check_overfit(cfg: GPTConfig, x, y, steps: int = 150, lr: float = 1e-3, target: float = 0.1) -> float:
    """A fresh model (no dropout) trained on ONE batch must memorize it."""
    cfg = GPTConfig(**{**cfg.__dict__, "dropout": 0.0})
    model = GPT(cfg).to(x.device)
    opt = configure_optimizer(model, lr, weight_decay=0.0)
    for _ in range(steps):
        loss = model(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    loss = loss.item()
    assert loss < target, f"loss after {steps} steps on one batch is {loss:.3f}, expected below {target}"
    return loss


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
