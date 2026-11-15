"""An MLP made of `Value` scalars, trained on the digits. Slow on purpose.

    python -m solution.scalar_demo

It works, and it shows why Step 2 exists: every multiplication is a Python
object, so one small batch already builds tens of thousands of graph nodes.
"""
from __future__ import annotations

import math
import random
import time

from .data import load_digits_data
from .engine import Value


class Neuron:
    def __init__(self, n_in, rnd, act=True):
        self.w = [Value(rnd.gauss(0.0, math.sqrt(2.0 / n_in))) for _ in range(n_in)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x):
        s = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return s.relu() if self.act else s

    def parameters(self):
        return self.w + [self.b]


class ScalarMLP:
    def __init__(self, sizes, seed=0):
        rnd = random.Random(seed)
        self.layers = [[Neuron(sizes[i], rnd, act=i < len(sizes) - 2) for _ in range(sizes[i + 1])]
                       for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = [neuron(x) for neuron in layer]
        return x

    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]


def cross_entropy(logits, target):
    """Softmax cross-entropy for one example, from exp, log, + and *."""
    m = max(v.data for v in logits)                 # constant shift for stability
    total = sum(((v - m).exp() for v in logits), Value(0.0))
    return total.log() - (logits[target] - m)


def main(steps=20, batch_size=16, lr=0.1):
    data = load_digits_data()
    x, y = data["x_train"].tolist(), data["y_train"].tolist()
    model = ScalarMLP([64, 16, 10])
    params = model.parameters()
    print(f"{len(params)} parameters, batch size {batch_size}")
    t0 = time.perf_counter()
    for step in range(steps):
        lo = (step * batch_size) % (len(x) - batch_size)
        losses = [cross_entropy(model(x[i]), y[i]) for i in range(lo, lo + batch_size)]
        loss = sum(losses, Value(0.0)) * (1.0 / batch_size)
        for p in params:
            p.grad = 0.0
        loss.backward()
        for p in params:
            p.data -= lr * p.grad
        if step % 5 == 0 or step == steps - 1:
            print(f"step {step:3d}  loss {loss.data:.4f}")
    dt = (time.perf_counter() - t0) / steps
    print(f"{dt * 1000:.0f} ms per step, {dt * 1000 / batch_size:.1f} ms per example")


if __name__ == "__main__":
    main()
