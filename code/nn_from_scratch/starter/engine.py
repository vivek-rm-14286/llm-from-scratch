"""Step 1: a scalar autograd engine.

A `Value` wraps one Python float. Every operation returns a new `Value` that
remembers its inputs (`_prev`) and a closure (`_backward`) that pushes the
output's gradient back to those inputs with the chain rule. `backward()`
orders the graph topologically and runs the closures from the output back.

Tests: pytest tests/test_engine.py
"""
from __future__ import annotations

import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                       # dL/d(this value), filled by backward()
        self._backward = lambda: None         # leaf nodes have nothing to propagate
        self._prev = tuple(_children)         # the Values this one was computed from
        self._op = _op                        # for debugging only

    # ---- binary operations -------------------------------------------------
    def __add__(self, other):
        """self + other. `other` may be a Value or a plain number.

        Pattern for every operation:
          1. wrap `other` in a Value if it is a number;
          2. create `out = Value(result, (self, other), "+")`;
          3. define a closure `_backward()` that ADDS (+=) to `self.grad` and
             `other.grad` the local derivative times `out.grad`;
          4. set `out._backward = _backward` and return `out`.
        """
        raise NotImplementedError("TODO: Value.__add__")

    def __mul__(self, other):
        """self * other. d(ab)/da = b, d(ab)/db = a."""
        raise NotImplementedError("TODO: Value.__mul__")

    def __pow__(self, k):
        """self ** k for a constant int or float k. d(a^k)/da = k * a^(k-1)."""
        raise NotImplementedError("TODO: Value.__pow__")

    # ---- unary operations --------------------------------------------------
    def exp(self):
        """e ** self. The derivative is the output itself."""
        raise NotImplementedError("TODO: Value.exp")

    def log(self):
        """Natural logarithm. d(ln a)/da = 1 / a."""
        raise NotImplementedError("TODO: Value.log")

    def tanh(self):
        """d(tanh a)/da = 1 - tanh(a)^2."""
        raise NotImplementedError("TODO: Value.tanh")

    def relu(self):
        """max(0, self). The derivative is 1 where self > 0, else 0."""
        raise NotImplementedError("TODO: Value.relu")

    # ---- backpropagation ---------------------------------------------------
    def backward(self):
        """Fill `.grad` of every node this value depends on.

        1. Build a topological order of the graph: a list in which every node
           comes after all the nodes in its `_prev`.
        2. Set `self.grad = 1.0` (dL/dL = 1).
        3. Call `node._backward()` for each node in REVERSE topological order.

        A recursive depth-first search is the shortest way to do step 1, but
        `sum()` over n Values builds a chain n nodes deep and Python's default
        recursion limit is 1000. One test builds a chain of 5,000 nodes, so use
        an explicit stack.
        """
        raise NotImplementedError("TODO: Value.backward")

    # ---- conveniences built from the operations above (already done) -------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return self ** -1 * other

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
