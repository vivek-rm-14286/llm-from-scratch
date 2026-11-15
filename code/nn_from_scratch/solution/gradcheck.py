"""Step 5: compare backprop gradients with finite differences.

Run `python -m solution.gradcheck` for a report on a small MLP.
"""
from __future__ import annotations

import numpy as np


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of the scalar function `f()` with respect to `x`.

    `f` takes no arguments and returns a float. It must read `x` (the same
    array object) each time it is called. Each entry of `x` is nudged in place
    by +eps and -eps and restored afterward.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / (||a|| + ||b||), and 0.0 when both are exactly zero."""
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else 0.0


def gradcheck(loss_fn, params, eps: float = 1e-6) -> float:
    """Return the worst relative error between backprop and finite differences.

    loss_fn: no arguments, rebuilds the graph and returns a scalar Tensor.
    params:  the Tensors to check.
    """
    for p in params:
        p.grad = np.zeros_like(p.data)
    loss_fn().backward()
    analytic = [p.grad.copy() for p in params]
    worst = 0.0
    for p, a in zip(params, analytic):
        numeric = numerical_grad(lambda: float(loss_fn().data), p.data, eps)
        worst = max(worst, rel_error(a, numeric))
    return worst


def main():
    from .nn import make_mlp, l2_penalty
    from .tensor import cross_entropy

    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 64))
    y = rng.integers(0, 10, size=16)
    for activation in ("relu", "tanh"):
        model = make_mlp([64, 32, 10], rng, activation=activation)
        params = model.parameters()

        def loss_fn():
            return cross_entropy(model(x), y) + l2_penalty(params, 1e-2)

        err = gradcheck(loss_fn, params)
        n = sum(p.data.size for p in params)
        print(f"MLP 64-32-10 ({activation}), {n} parameters: max relative error {err:.2e}")


if __name__ == "__main__":
    main()
