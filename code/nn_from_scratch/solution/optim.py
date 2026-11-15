"""Step 4: optimizers. Each one turns `p.grad` into an in-place update of `p.data`."""
from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with optional momentum and L2 weight decay.

        g = grad + weight_decay * p
        v = momentum * v + g
        p = p - lr * v
    """

    def __init__(self, params, lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.v):
            g = p.grad + self.weight_decay * p.data
            v *= self.momentum
            v += g
            p.data -= self.lr * v


class Adam(Optimizer):
    """Adam with bias correction.

    weight_decay with decoupled=False adds weight_decay * p to the gradient
    (L2 regularization, what torch.optim.Adam does). With decoupled=True the
    weights are shrunk directly by lr * weight_decay * p before the Adam step
    (AdamW, what torch.optim.AdamW does).
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, decoupled: bool = False):
        super().__init__(params)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.weight_decay, self.decoupled = weight_decay, decoupled
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            if self.weight_decay and self.decoupled:
                p.data *= 1.0 - self.lr * self.weight_decay
            elif self.weight_decay:
                g = g + self.weight_decay * p.data
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            m_hat = m / (1.0 - self.b1 ** self.t)
            v_hat = v / (1.0 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
