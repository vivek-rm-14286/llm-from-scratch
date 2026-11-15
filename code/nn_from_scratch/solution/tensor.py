"""Step 2: the same engine with numpy arrays instead of floats.

One `Tensor` node holds a whole array, so a layer of 64 units over a batch of
32 examples is three graph nodes instead of several thousand. The two ideas
that are new compared with `Value`:

* broadcasting: when numpy stretched an input in the forward pass, the
  backward pass must sum the gradient back to the input's shape (`unbroadcast`);
* matrix multiplication has matrix-shaped local derivatives.
"""
from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing numpy broadcasting.

    Broadcasting copies a value to several places. Each copy contributes to
    the loss, so the gradient of the original is the sum over the copies.
    """
    while grad.ndim > len(shape):                 # leading axes that were added
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):           # axes that were stretched from 1
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    # ---- elementwise operations (with broadcasting) ------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Tensor(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    # ---- matrix multiplication ---------------------------------------------
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.ndim == 2 and other.data.ndim == 2, "2-D @ 2-D only"
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T      # dL/dA = dL/dC . B^T
            other.grad += self.data.T @ out.grad      # dL/dB = A^T . dL/dC
        out._backward = _backward
        return out

    # ---- reductions --------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)           # put the summed axis back
            self.grad += np.broadcast_to(g, self.shape)   # every element got 1 * g
        out._backward = _backward
        return out

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    # ---- backpropagation: identical to Value.backward ----------------------
    def backward(self):
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                topo.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                for child in node._prev:
                    if id(child) not in seen:
                        stack.append((child, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    # ---- conveniences ------------------------------------------------------
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
    Forward: log-softmax with the max subtracted first, so exp never overflows.
    Backward: dL/dlogits = (softmax(logits) - onehot(targets)) / N.
    """
    targets = np.asarray(targets, dtype=np.int64)
    n = logits.shape[0]
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    out = Tensor(-log_probs[np.arange(n), targets].mean(), (logits,), "cross_entropy")

    def _backward():
        g = np.exp(log_probs)                         # softmax probabilities
        g[np.arange(n), targets] -= 1.0
        logits.grad += g / n * out.grad
    out._backward = _backward
    return out
