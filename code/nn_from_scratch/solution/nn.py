"""Step 3: layers built on `Tensor`."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Module:
    """Anything with a forward pass and (possibly) parameters."""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x if isinstance(x, Tensor) else Tensor(x))

    def parameters(self) -> list:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    """y = x @ W + b, with W of shape (n_in, n_out) and b of shape (n_out,).

    init="he":     W ~ Normal(0, sqrt(2 / n_in))                 (for ReLU)
    init="xavier": W ~ Uniform(-a, a), a = sqrt(6 / (n_in + n_out))  (for tanh)
    The bias starts at zero.
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, init: str = "he"):
        if init == "he":
            w = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        elif init == "xavier":
            a = np.sqrt(6.0 / (n_in + n_out))
            w = rng.uniform(-a, a, size=(n_in, n_out))
        else:
            raise ValueError(f"unknown init {init!r}")
        self.W = Tensor(w)
        self.b = Tensor(np.zeros(n_out))

    def forward(self, x):
        return x @ self.W + self.b                    # b broadcasts over the batch

    def parameters(self):
        return [self.W, self.b]


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def make_mlp(sizes, rng, activation: str = "relu", init: str | None = None) -> Sequential:
    """sizes=[64, 64, 10] builds Linear(64, 64) -> ReLU -> Linear(64, 10).

    There is an activation after every layer except the last one, which
    outputs raw logits. `init` defaults to "he" for relu and "xavier" for tanh.
    """
    if init is None:
        init = "he" if activation == "relu" else "xavier"
    act = {"relu": ReLU, "tanh": Tanh}[activation]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], rng, init=init))
        if i < len(sizes) - 2:
            layers.append(act())
    return Sequential(*layers)


def l2_penalty(params, lam: float) -> Tensor:
    """(lam / 2) * sum of squared entries over all tensors in `params`."""
    total = Tensor(0.0)
    for p in params:
        total = total + (p * p).sum()
    return total * (0.5 * lam)


def accuracy(logits, targets) -> float:
    data = logits.data if isinstance(logits, Tensor) else np.asarray(logits)
    return float((data.argmax(axis=1) == np.asarray(targets)).mean())
