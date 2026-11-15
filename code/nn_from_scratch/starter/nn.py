"""Step 3: layers built on `Tensor`.

Tests: pytest tests/test_nn.py
"""
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
        # Draw the weights from `rng` (rng.normal / rng.uniform) with
        # size=(n_in, n_out), then set self.W and self.b as Tensors.
        # Raise ValueError for an unknown `init`.
        raise NotImplementedError("TODO: Linear.__init__")

    def forward(self, x):
        raise NotImplementedError("TODO: Linear.forward")

    def parameters(self):
        """Return [W, b], in that order."""
        raise NotImplementedError("TODO: Linear.parameters")


class ReLU(Module):
    def forward(self, x):
        raise NotImplementedError("TODO: ReLU.forward")


class Tanh(Module):
    def forward(self, x):
        raise NotImplementedError("TODO: Tanh.forward")


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x):
        """Feed x through every layer in order."""
        raise NotImplementedError("TODO: Sequential.forward")

    def parameters(self):
        """All parameters of all layers, in layer order."""
        raise NotImplementedError("TODO: Sequential.parameters")


def make_mlp(sizes, rng, activation: str = "relu", init: str | None = None) -> Sequential:
    """sizes=[64, 64, 10] builds Linear(64, 64) -> ReLU -> Linear(64, 10).

    There is an activation after every layer except the last one, which
    outputs raw logits. `init` defaults to "he" for relu and "xavier" for tanh.
    """
    raise NotImplementedError("TODO: make_mlp")


def l2_penalty(params, lam: float) -> Tensor:
    """(lam / 2) * sum of squared entries over all tensors in `params`."""
    raise NotImplementedError("TODO: l2_penalty")


def accuracy(logits, targets) -> float:
    """Fraction of rows whose largest logit is at the target class.

    `logits` is a Tensor or an array of shape (N, C). Returns a Python float.
    """
    raise NotImplementedError("TODO: accuracy")
