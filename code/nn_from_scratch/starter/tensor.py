"""Step 2: the same engine with numpy arrays instead of floats.

One `Tensor` node holds a whole array, so a layer of 64 units over a batch of
32 examples is three graph nodes instead of several thousand. The two ideas
that are new compared with `Value`:

* broadcasting: when numpy stretched an input in the forward pass, the
  backward pass must sum the gradient back to the input's shape (`unbroadcast`);
* matrix multiplication has matrix-shaped local derivatives.

Tests: pytest tests/test_tensor.py
"""
from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing numpy broadcasting.

    Broadcasting copies a value to several places. Each copy contributes to
    the loss, so the gradient of the original is the sum over the copies.

    Two cases to undo:
      * leading axes that broadcasting added: (N, D) + (D,)  -> sum over axis 0
        until `grad.ndim == len(shape)`;
      * axes stretched from size 1: (N, 1) * (N, D) -> sum over that axis with
        keepdims=True.
    Examples: unbroadcast(ones((4, 3)), (3,)) == [4, 4, 4];
              unbroadcast(ones((4, 3)), (4, 1)).shape == (4, 1).
    """
    raise NotImplementedError("TODO: unbroadcast")


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)  # same shape as data, always
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    # ---- elementwise operations (with broadcasting) ------------------------
    def __add__(self, other):
        """self + other with numpy broadcasting. `other`: Tensor, array or number.

        Same pattern as Value.__add__, but pass each gradient through
        `unbroadcast(..., self.shape)` before adding it to `.grad`.
        """
        raise NotImplementedError("TODO: Tensor.__add__")

    def __mul__(self, other):
        """Elementwise self * other with broadcasting."""
        raise NotImplementedError("TODO: Tensor.__mul__")

    def __pow__(self, k):
        """Elementwise self ** k for a constant int or float k."""
        raise NotImplementedError("TODO: Tensor.__pow__")

    def exp(self):
        raise NotImplementedError("TODO: Tensor.exp")

    def log(self):
        raise NotImplementedError("TODO: Tensor.log")

    def tanh(self):
        raise NotImplementedError("TODO: Tensor.tanh")

    def relu(self):
        """Elementwise max(0, self). Gradient passes where self.data > 0."""
        raise NotImplementedError("TODO: Tensor.relu")

    # ---- matrix multiplication ---------------------------------------------
    def __matmul__(self, other):
        """C = A @ B for 2-D A (n, k) and B (k, m).

        dL/dA = dL/dC @ B.T      shape (n, k)
        dL/dB = A.T @ dL/dC      shape (k, m)
        The shapes leave no other way to combine the three matrices.
        """
        raise NotImplementedError("TODO: Tensor.__matmul__")

    # ---- reductions --------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        """Sum over all elements (axis=None) or over one integer axis.

        Every input element contributes with derivative 1, so the backward
        pass copies `out.grad` to every element that was summed. If the axis
        was dropped (keepdims=False), put it back with `np.expand_dims` and
        then `np.broadcast_to(..., self.shape)`.
        """
        raise NotImplementedError("TODO: Tensor.sum")

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    # ---- backpropagation ---------------------------------------------------
    def backward(self):
        """Identical to Value.backward, except the seed is `np.ones_like(self.data)`."""
        raise NotImplementedError("TODO: Tensor.backward")

    # ---- conveniences (already done) ---------------------------------------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rmatmul__(self, other):                     # ndarray @ Tensor
        return Tensor(other) @ self

    __array_ufunc__ = None    # makes `ndarray + Tensor` call Tensor.__radd__, not numpy

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r})"


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean softmax cross-entropy as one fused operation.

    logits: Tensor of shape (N, C). targets: N integer class ids.
    Returns a scalar Tensor (shape ()) whose only child is `logits`.

    Forward: log_probs = z - log(sum(exp(z), axis=1)) with
             z = logits - max(logits, axis=1), so exp never overflows;
             loss = -mean(log_probs[i, targets[i]]).
    Backward: dL/dlogits = (softmax(logits) - onehot(targets)) / N, times out.grad.
    """
    raise NotImplementedError("TODO: cross_entropy")
