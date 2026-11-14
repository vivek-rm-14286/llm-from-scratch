"""Exercise: this character-level GPT training script has 5 planted bugs.

    python buggy_train.py

It runs on a laptop CPU in about 15 seconds and prints a loss that looks fine at a glance and is not.
Work through the checklist in "Debugging a Training Run" (docs/pretraining/debugging.md), fix one bug
at a time, and re-run after each fix. Done when the step-0 loss is about ln(vocab) = 4.50 and the
validation loss ends near 1.5. fixed_train.py is the reference solution.
"""
import argparse  # only used as training text
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

STEPS = 300

torch.manual_seed(0)
torch.set_num_threads(4)

# ---- data: the source code of a standard-library module, as characters (no download) ----
# The text differs slightly between Python versions (90 distinct characters on 3.12, 89 on 3.14).
text = inspect.getsource(argparse)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, T, B = len(chars), 64, 32
BASE_LR, WARMUP = 3e-3, 20
gen = torch.Generator().manual_seed(0)


def get_batch(split):
    d = train_data if split == "train" else val_data
    gen.manual_seed(0)
    ix = torch.randint(len(d) - T - 1, (B,), generator=gen)
    x = torch.stack([d[i:i + T] for i in ix])
    y = torch.stack([d[i:i + T] for i in ix])
    return x, y


# ---- model ----
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.register_buffer("mask", torch.tril(torch.ones(T, T), diagonal=1).bool())

    def forward(self, x):
        Bx, Tx, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(Bx, Tx, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.h)
        att = att.masked_fill(~self.mask[:Tx, :Tx], float("-inf")).softmax(dim=-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(Bx, Tx, C))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, d=64, h=4, n_layer=2):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(n_layer))
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
        self.apply(self._init)
        nn.init.normal_(self.head.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss


def lr_at(step):
    """Linear warmup, then constant."""
    return BASE_LR * (step + 1)


@torch.no_grad()
def estimate(model, split, batches=10):
    model.eval()
    out = sum(model(*get_batch(split))[1].item() for _ in range(batches)) / batches
    model.train()
    return out


model = GPT()
opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

for step in range(STEPS + 1):
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        print(f"step {step:4d}  train {loss.item():.3f}  val {estimate(model, 'val'):.3f}")
