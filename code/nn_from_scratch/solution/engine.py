"""Step 1: a scalar autograd engine.

A `Value` wraps one Python float. Every operation returns a new `Value` that
remembers its inputs (`_prev`) and a closure (`_backward`) that pushes the
output's gradient back to those inputs with the chain rule. `backward()`
orders the graph topologically and runs the closures from the output back.
"""
from __future__ import annotations

import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    # ---- binary operations -------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad                 # d(a+b)/da = 1
            other.grad += out.grad                # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad    # d(ab)/da = b
            other.grad += self.data * out.grad    # d(ab)/db = a
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    # ---- unary operations --------------------------------------------------
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad      # d(e^a)/da = e^a
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data     # d(ln a)/da = 1/a
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # ---- backpropagation ---------------------------------------------------
    def backward(self):
        """Fill `.grad` of every node this value depends on.

        The topological sort is iterative. A recursive depth-first search is
        shorter, but `sum()` over n values builds a chain n nodes deep and
        Python's default recursion limit is 1000.
        """
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                topo.append(node)                 # after everything it depends on
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                for child in node._prev:
                    if id(child) not in seen:
                        stack.append((child, False))
        self.grad = 1.0                           # dL/dL = 1
        for node in reversed(topo):
            node._backward()

    # ---- conveniences built from the operations above ----------------------
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
