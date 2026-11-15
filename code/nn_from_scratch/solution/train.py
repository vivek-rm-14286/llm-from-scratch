"""Step 6: train an MLP on handwritten digits.

    python -m solution.train                       # 8x8 digits, no download
    python -m solution.train --dataset mnist       # downloads MNIST (about 12 MB)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .data import batches, load_digits_data, load_mnist_data
from .nn import accuracy, make_mlp
from .optim import SGD, Adam
from .tensor import cross_entropy


def evaluate(model, x, y, batch_size: int = 1000):
    """Return (mean loss, accuracy) over a whole split, without updating anything."""
    total_loss, correct = 0.0, 0.0
    for xb, yb in batches(x, y, batch_size):
        logits = model(xb)
        total_loss += float(cross_entropy(logits, yb).data) * len(xb)
        correct += accuracy(logits, yb) * len(xb)
    return total_loss / len(x), correct / len(x)


def train_epoch(model, optimizer, x, y, batch_size: int, rng) -> float:
    """One pass over the training set. Returns the mean training loss."""
    total_loss = 0.0
    for xb, yb in batches(x, y, batch_size, rng):
        loss = cross_entropy(model(xb), yb)       # forward: builds the graph
        optimizer.zero_grad()                     # clear gradients from the last step
        loss.backward()                           # backward: fills p.grad
        optimizer.step()                          # update the weights
        total_loss += float(loss.data) * len(xb)
    return total_loss / len(x)


def make_optimizer(name: str, params, lr: float, momentum: float, weight_decay: float):
    if name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return Adam(params, lr=lr, weight_decay=weight_decay, decoupled=True)
    raise ValueError(f"unknown optimizer {name!r}")


def fit(data, hidden=(64,), epochs: int = 30, batch_size: int = 32, optimizer: str = "adam",
        lr: float | None = None, momentum: float = 0.9, weight_decay: float = 0.0,
        activation: str = "relu", seed: int = 0, verbose: bool = True):
    """Train an MLP and return (model, history).

    history is a list with one dict per epoch:
    {"epoch", "train_loss", "val_loss", "val_acc"}.
    lr=None picks 0.05 for sgd and 0.003 for adam/adamw.
    """
    if lr is None:
        lr = 0.05 if optimizer == "sgd" else 3e-3
    rng = np.random.default_rng(seed)
    n_in, n_out = data["x_train"].shape[1], int(data["y_train"].max()) + 1
    model = make_mlp([n_in, *hidden, n_out], rng, activation=activation)
    opt = make_optimizer(optimizer, model.parameters(), lr, momentum, weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, opt, data["x_train"], data["y_train"], batch_size, rng)
        val_loss, val_acc = evaluate(model, data["x_val"], data["y_val"])
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        if verbose:
            print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  "
                  f"val loss {val_loss:.4f}  val acc {val_acc:.4f}")
    return model, history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["digits", "mnist"], default="digits")
    ap.add_argument("--data-dir", default="data/mnist", help="where MNIST is cached")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adam")
    ap.add_argument("--lr", type=float, default=None, help="default: 0.05 for sgd, 0.003 for adam/adamw")
    ap.add_argument("--momentum", type=float, default=0.9, help="sgd only")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_digits_data(args.seed) if args.dataset == "digits" else load_mnist_data(args.data_dir, args.seed)
    print(f"{args.dataset}: {len(data['x_train'])} train / {len(data['x_val'])} val / "
          f"{len(data['x_test'])} test, {data['x_train'].shape[1]} features")
    t0 = time.perf_counter()
    model, _ = fit(data, hidden=args.hidden, epochs=args.epochs, batch_size=args.batch_size,
                   optimizer=args.optimizer, lr=args.lr, momentum=args.momentum,
                   weight_decay=args.weight_decay, activation=args.activation, seed=args.seed)
    elapsed = time.perf_counter() - t0
    n_params = sum(p.data.size for p in model.parameters())
    test_loss, test_acc = evaluate(model, data["x_test"], data["y_test"])
    print(f"parameters {n_params}  training time {elapsed:.1f} s")
    print(f"test loss {test_loss:.4f}  test accuracy {test_acc:.4f}")


if __name__ == "__main__":
    main()
