"""Step 7: the same MLP, same initial weights, same batches, in PyTorch.

    python -m starter.compare_torch                  # Adam
    python -m starter.compare_torch --optimizer sgd  # SGD with momentum and weight decay

Needs `torch` (optional dependency). Both sides run in float64.
`run_both` is done; `to_torch` is yours.

Tests: pytest tests/test_torch.py
"""
from __future__ import annotations

import argparse

import numpy as np

from .data import batches, load_digits_data
from .nn import Linear, make_mlp
from .optim import SGD, Adam
from .tensor import cross_entropy


def to_torch(model):
    """Build a torch.nn.Sequential with the same architecture and the same weights.

    Our Linear stores W as (n_in, n_out) and computes x @ W. torch.nn.Linear
    stores weight as (n_out, n_in) and computes x @ weight.T, so W is transposed.
    """
    import torch

    # For each layer in model.layers: a Linear becomes
    # torch.nn.Linear(n_in, n_out, dtype=torch.float64) with weight and bias
    # copied under torch.no_grad(); ReLU/Tanh become torch.nn.ReLU()/Tanh().
    raise NotImplementedError("TODO: to_torch")


def run_both(optimizer: str = "adam", steps: int = 200, batch_size: int = 32, lr: float | None = None,
             momentum: float = 0.9, weight_decay: float = 0.0, hidden=(64,), seed: int = 0):
    """Train both models on identical batches. Returns (our_losses, torch_losses, weight_gap).

    weight_gap is the largest absolute difference between any pair of
    corresponding weights after the last step.
    """
    import torch

    data = load_digits_data(seed)
    x, y = data["x_train"], data["y_train"]
    ours = make_mlp([x.shape[1], *hidden, 10], np.random.default_rng(seed))
    theirs = to_torch(ours)

    if optimizer == "sgd":
        lr = 0.05 if lr is None else lr
        opt = SGD(ours.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        topt = torch.optim.SGD(theirs.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "adam":
        lr = 1e-3 if lr is None else lr
        opt = Adam(ours.parameters(), lr=lr, weight_decay=weight_decay)
        topt = torch.optim.Adam(theirs.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        lr = 1e-3 if lr is None else lr
        opt = Adam(ours.parameters(), lr=lr, weight_decay=weight_decay, decoupled=True)
        topt = torch.optim.AdamW(theirs.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    our_losses, torch_losses = [], []
    rng = np.random.default_rng(seed + 1)
    while len(our_losses) < steps:
        for xb, yb in batches(x, y, batch_size, rng):
            loss = cross_entropy(ours(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            our_losses.append(float(loss.data))

            tloss = torch.nn.functional.cross_entropy(theirs(torch.from_numpy(xb)), torch.from_numpy(yb))
            topt.zero_grad()
            tloss.backward()
            topt.step()
            torch_losses.append(tloss.item())
            if len(our_losses) == steps:
                break

    gap = 0.0
    torch_params = list(theirs.parameters())
    for p, tp in zip(ours.parameters(), torch_params):
        t = tp.detach().numpy()
        gap = max(gap, float(np.abs(p.data - (t.T if t.ndim == 2 else t)).max()))
    return np.array(our_losses), np.array(torch_losses), gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adam")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()
    wd = args.weight_decay if args.weight_decay is not None else (1e-4 if args.optimizer == "sgd" else 0.0)

    a, b, gap = run_both(args.optimizer, steps=args.steps, weight_decay=wd)
    print("step    ours        torch       |diff|")
    for i in sorted({0, 1, 2, 10, 50, 100, args.steps - 1}):
        if i < args.steps:
            print(f"{i:4d}  {a[i]:.8f}  {b[i]:.8f}  {abs(a[i] - b[i]):.1e}")
    diff = float(np.abs(a - b).max())
    print(f"{args.optimizer}, weight decay {wd}: max |loss difference| over {args.steps} steps = {diff:.2e}")
    print(f"max |weight difference| after {args.steps} steps = {gap:.2e}")
    print("MATCH" if diff < args.tol else "MISMATCH")


if __name__ == "__main__":
    main()
