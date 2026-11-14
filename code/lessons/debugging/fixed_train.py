"""A small character-level GPT training script that works, with the debugging checks built in.

Runs on a laptop CPU in about 15 seconds:

    python fixed_train.py                 # all checks pass, val loss falls to about 1.5
    python fixed_train.py --bug 3         # re-introduce one planted bug (1-5) to see its symptom
    python fixed_train.py --overfit       # check 2: memorize a single batch
    python fixed_train.py --lr 1e-2 --steps 100   # LR-range check: sweep --lr over a few decades

The five bugs (see buggy_train.py, which has all of them at once):
    1  output head re-initialized with std=1.0   -> step-0 loss far above ln(vocab)
    2  targets are not shifted (y == x)          -> loss falls to ~0, train and val
    3  causal mask lets each position see t+1    -> loss falls to ~0, train and val
    4  get_batch re-seeds its generator          -> every step is the same batch: train falls, val rises
    5  schedule returns BASE_LR * (step + 1)     -> loss stalls near 2.7, then explodes
"""
import argparse
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--bug", type=int, nargs="*", default=[], help="bug numbers (1-5) to re-introduce")
p.add_argument("--overfit", action="store_true", help="train on one fixed batch")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--lr", type=float, default=3e-3, help="peak learning rate")
args = p.parse_args()
BUGS = set(args.bug)

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
BASE_LR, WARMUP = args.lr, 20
gen = torch.Generator().manual_seed(0)


def get_batch(split):
    d = train_data if split == "train" else val_data
    if 4 in BUGS:
        gen.manual_seed(0)                                   # BUG 4: same offsets on every call
    ix = torch.randint(len(d) - T - 1, (B,), generator=gen)
    x = torch.stack([d[i:i + T] for i in ix])
    if 2 in BUGS:
        y = torch.stack([d[i:i + T] for i in ix])            # BUG 2: target == input
    else:
        y = torch.stack([d[i + 1:i + T + 1] for i in ix])    # next-character targets
    return x, y


# ---- model ----
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        diagonal = 1 if 3 in BUGS else 0                     # BUG 3: diagonal=1 exposes position t+1
        self.register_buffer("mask", torch.tril(torch.ones(T, T), diagonal=diagonal).bool())

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
        if 1 in BUGS:
            nn.init.normal_(self.head.weight)                # BUG 1: default std is 1.0, logits start huge

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
    if 5 in BUGS:
        return BASE_LR * (step + 1)                          # BUG 5: no /WARMUP, no cap, no decay
    if step < WARMUP:
        return BASE_LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, args.steps - WARMUP)
    return BASE_LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate(model, split, batches=10):
    model.eval()
    out = sum(model(*get_batch(split))[1].item() for _ in range(batches)) / batches
    model.train()
    return out


@torch.no_grad()
def causal_leak(model):
    """Change the last input token; logits at earlier positions must not move."""
    model.eval()
    x, _ = get_batch("val")
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % V
    diff = (model(x)[0][:, :-1] - model(x2)[0][:, :-1]).abs().max().item()
    model.train()
    return diff


model = GPT()
opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

# ---- check 1: step-0 loss ----
x, y = get_batch("train")
loss0 = model(x, y)[1].item()
print(f"check 1  step-0 loss {loss0:.3f}   expected ln({V}) = {math.log(V):.3f}")
# ---- check 3: targets shifted, no future leak ----
print(f"check 3  targets shifted: {bool((x[:, 1:] == y[:, :-1]).all())}   "
      f"causal leak: {causal_leak(model):.2e} (must be 0)")

fixed = get_batch("train") if args.overfit else None
for step in range(args.steps + 1):
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = fixed if fixed is not None else get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    before = {k: v.detach().clone() for k, v in model.named_parameters()} if step % 50 == 0 else None
    opt.step()
    if step % 50 == 0:
        ratio = max(((v.detach() - before[k]).norm() / (v.detach().norm() + 1e-12)).item()
                    for k, v in model.named_parameters() if v.ndim == 2)
        val = float("nan") if args.overfit else estimate(model, "val")
        print(f"step {step:4d}  lr {lr:.1e}  train {loss.item():.3f}  val {val:.3f}  "
              f"grad-norm {gnorm:5.2f}  max update/weight {ratio:.1e}")
