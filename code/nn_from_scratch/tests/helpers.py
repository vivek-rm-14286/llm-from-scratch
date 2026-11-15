"""Finite differences for the tests, independent of the code under test."""
import numpy as np


def numeric_grad(f, x, eps=1e-6):
    """Central differences of the float-valued f() with respect to array x (in place)."""
    g = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(*x.shape):
        old = x[idx]
        x[idx] = old + eps
        fp = f()
        x[idx] = old - eps
        fm = f()
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g


def check_grads(build, tensors, tol=1e-6):
    """build() returns a scalar Tensor made from `tensors`. Compare backprop with numeric_grad."""
    for t in tensors:
        t.grad = np.zeros_like(t.data)
    build().backward()
    for i, t in enumerate(tensors):
        assert t.grad.shape == t.data.shape, f"input {i}: grad shape {t.grad.shape} != data shape {t.data.shape}"
        num = numeric_grad(lambda: float(build().data), t.data)
        err = np.abs(t.grad - num).max() / max(1.0, np.abs(num).max())
        assert err < tol, f"input {i}: backprop and finite differences differ by {err:.2e}"
