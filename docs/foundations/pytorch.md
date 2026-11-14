# Python & PyTorch

PyTorch is the tool used for almost every model on this site. It gives you three things: a numpy-like array type (the tensor) that runs on GPUs, automatic differentiation so you never write a backward pass by hand, and a small set of building blocks (`nn.Module`, optimizers, data loaders) for organizing models and training. This page covers what you need to read and write model code fluently, and the bugs you will hit along the way.

!!! abstract "What you'll learn"
    - Tensors: creation, shape, dtype, device, views, and indexing
    - Broadcasting rules and `einsum`, the two tools that make attention code short
    - Autograd: how `requires_grad`, `backward()`, and `.grad` work, and how to turn them off
    - `nn.Module`, parameters, `state_dict`, and `train()`/`eval()` modes
    - `Dataset` and `DataLoader`
    - The canonical training loop, with mixed precision
    - Devices (CPU, CUDA, MPS), dtypes (fp32, bf16, fp16), and the most common shape bugs

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
```

## Tensors

A **tensor** is an n-dimensional array, like a numpy `ndarray`, with three extra properties: it has a `device` (where it lives), a `dtype` (what number format it uses), and it can track gradients.

```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape, x.dtype, x.device)       # torch.Size([2, 2]) torch.float32 cpu

torch.zeros(2, 3); torch.ones(2, 3); torch.randn(2, 3)   # common constructors
torch.arange(6).reshape(2, 3)          # [[0,1,2],[3,4,5]]
torch.empty(2, 3)                      # uninitialized memory: do not read before writing

import numpy as np
a = np.arange(3.0)
t = torch.from_numpy(a)                # shares memory with a
back = t.numpy()                       # also shares memory (CPU only)
```

### Shapes, views, and copies

The shape conventions you will see throughout the site:

| Tensor | Shape | Letters |
|--------|-------|---------|
| Token IDs | `(B, T)` | batch, time (sequence length) |
| Hidden states | `(B, T, C)` | C = channels = `d_model` |
| Attention heads | `(B, H, T, D)` | H heads, D = head dim, `C = H * D` |
| Logits | `(B, T, V)` | V = vocab size |

Reshaping operations:

```python
x = torch.randn(2, 8, 12)              # (B, T, C)
B, T, C = x.shape
H = 3; D = C // H

h = x.view(B, T, H, D)                 # split channels into heads: (2, 8, 3, 4)
h = h.transpose(1, 2)                  # (B, H, T, D): (2, 3, 8, 4)
print(h.is_contiguous())               # False: transpose changes strides, not memory

back = h.transpose(1, 2).contiguous().view(B, T, C)   # merge heads back
# or: back = h.transpose(1, 2).reshape(B, T, C)       # reshape copies if needed
print(torch.equal(back, x))            # True
```

- `view` never copies; it fails if the memory layout is not compatible (after a `transpose` or `permute`).
- `reshape` returns a view when it can and copies when it must.
- `unsqueeze(dim)` adds a size-1 dimension; `squeeze(dim)` removes one.
- `permute` reorders any number of dimensions; `transpose` swaps two.

### Indexing

```python
logits = torch.randn(4, 10)            # (N, V)
targets = torch.tensor([3, 1, 0, 9])
picked = logits[torch.arange(4), targets]   # (4,): logit of the target per row
last = torch.randn(2, 8, 12)[:, -1, :]      # (B, C): the last time step
mask = logits > 0                           # boolean mask
logits.masked_fill(~mask, float("-inf"))    # used for causal masks and top-k
```

## Broadcasting

**Broadcasting** lets you operate on tensors of different shapes without copying. Shapes are compared from the right; two dimensions are compatible if they are equal or one of them is 1 (or missing). The size-1 dimension is stretched.

```python
x = torch.randn(4, 3)          # (N, C)
mean = x.mean(dim=0)           # (C,)
centered = x - mean            # (4, 3) - (3,) -> (4, 3)

row_mean = x.mean(dim=1, keepdim=True)   # (4, 1): keepdim is the key
x - row_mean                              # (4, 3) - (4, 1) -> (4, 3)

# Pairwise differences: (N, 1, C) - (1, M, C) -> (N, M, C)
a = torch.randn(5, 1, 3); b = torch.randn(1, 7, 3)
print((a - b).shape)           # torch.Size([5, 7, 3])
```

!!! danger "The silent broadcasting bug"
    If `pred` has shape `(N, 1)` and `target` has shape `(N,)`, then `pred - target` broadcasts to `(N, N)`. `F.mse_loss` warns about this, but a hand-written `((pred - target) ** 2).mean()` silently computes the wrong number and the model still "trains". Always check shapes of anything you subtract or compare.

## einsum

`torch.einsum` writes tensor contractions with index letters. Indices that appear in the inputs but not the output are summed over. It is the clearest way to express attention.

```python
A = torch.randn(3, 4); B = torch.randn(4, 5)
torch.einsum("ik,kj->ij", A, B)           # matrix multiply, same as A @ B

x = torch.randn(2, 8, 16); W = torch.randn(16, 32)
torch.einsum("btc,cd->btd", x, W)          # apply a linear map to every position

q = torch.randn(2, 4, 8, 16)               # (B, H, T, D)
k = torch.randn(2, 4, 8, 16)
scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / 16 ** 0.5   # (B, H, T, T)
print(torch.allclose(scores, q @ k.transpose(-2, -1) / 4.0, atol=1e-5))  # True

torch.einsum("ii->", torch.eye(3))         # trace = 3
torch.einsum("i,j->ij", torch.ones(2), torch.arange(3.0))   # outer product
```

`@` / `torch.matmul` is usually just as fast and is what most production code uses. Use `einsum` when it makes the intent clearer.

## Autograd

PyTorch records operations on tensors with `requires_grad=True` and builds a computation graph. Calling `.backward()` on a scalar walks the graph in reverse, applying the chain rule, and stores gradients in each leaf tensor's `.grad`.

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x, y = torch.tensor(3.0), torch.tensor(5.0)

loss = (w * x + b - y) ** 2
loss.backward()
print(w.grad, b.grad)          # tensor(12.) tensor(4.)  (matches the math page)
```

Things to know:

- **Gradients accumulate.** Calling `backward()` twice adds to `.grad`. That is why the training loop zeroes gradients each step, and it is also what makes gradient accumulation work.
- **Only scalars** can call `.backward()` without arguments. Reduce the loss with `.mean()` or `.sum()` first.
- **The graph is freed** after `backward()` unless you pass `retain_graph=True`.
- **`.detach()`** returns a tensor that shares data but is cut off from the graph. Use it for logging (`loss.item()` also works for scalars) and for targets that should not receive gradients, such as the target network in [DQN](../decision/policy-methods.md).

### Turning autograd off

```python
with torch.no_grad():          # no graph is recorded: less memory, faster
    preds = w * x + b

@torch.inference_mode()        # stricter and slightly faster, for pure inference
def generate(model, x):
    return model(x)
```

Always evaluate and generate under `no_grad` or `inference_mode`. Forgetting this keeps every activation alive and is a common cause of out-of-memory errors during evaluation.

## nn.Module

`nn.Module` is the base class for models and layers. Assign sub-modules and `nn.Parameter`s as attributes in `__init__` and they are registered automatically; define the computation in `forward`.

```python title="mlp.py"
class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))

model = MLP(10, 64, 3)
print(sum(p.numel() for p in model.parameters()))   # 10*64+64 + 64*3+3 = 899
for name, p in model.named_parameters():
    print(name, tuple(p.shape))
```

Key methods:

- `model.parameters()` / `named_parameters()`: what the optimizer updates.
- `model.train()` / `model.eval()`: switch dropout and batch norm behavior. Dropout is active only in train mode.
- `model.to(device)` / `model.to(dtype)`: move or cast all parameters and buffers.
- `model.state_dict()` / `load_state_dict()`: save and load weights.
- `register_buffer(name, tensor)`: state that moves with the model but is not trained, such as a causal mask or RoPE frequencies.

!!! warning "Use `nn.ModuleList`, not a Python list"
    `self.layers = [nn.Linear(8, 8) for _ in range(4)]` does **not** register the layers. They will not appear in `parameters()`, will not move with `.to(device)`, and will not be trained. Use `nn.ModuleList([...])`.

## Dataset and DataLoader

A `Dataset` answers two questions: how many examples (`__len__`) and what is example `i` (`__getitem__`). A `DataLoader` batches, shuffles, and optionally loads in parallel worker processes.

```python title="data.py"
from torch.utils.data import Dataset, DataLoader

class TokenWindows(Dataset):
    """Fixed-length next-token prediction windows over one long token stream."""
    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, i):
        chunk = self.tokens[i : i + self.block_size + 1]
        return chunk[:-1], chunk[1:]       # input, target shifted by one

tokens = torch.randint(0, 100, (10_000,))
ds = TokenWindows(tokens, block_size=32)
dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
xb, yb = next(iter(dl))
print(xb.shape, yb.shape)                  # (8, 32) (8, 32)
```

This input/target shift by one position is exactly how language models are trained, and you will see it again in [Data Pipelines](../pretraining/data.md). On CUDA, set `pin_memory=True` and `num_workers` > 0 to overlap data loading with compute.

## Devices and dtypes

### Devices

```python
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
xb, yb = xb.to(device), yb.to(device)
```

Every tensor in an operation must be on the same device. Move the model once, then move each batch inside the loop.

### Dtypes

| dtype | Bits | Range | Notes |
|-------|------|-------|-------|
| `float32` | 32 | huge | Default. Safe, slow-ish, memory-hungry. |
| `bfloat16` | 16 | same as fp32 | Fewer mantissa bits. Preferred for training on recent NVIDIA GPUs (A100, H100, RTX 30xx/40xx). No loss scaling needed. |
| `float16` | 16 | small (max ~65504) | Needs a gradient scaler in training to avoid underflow. Used on older GPUs. |
| `int64` (`long`) | 64 | - | Token IDs and class labels. `F.cross_entropy` targets must be `long`. |

**Mixed precision** runs most of the forward and backward pass in 16-bit, while keeping master weights and sensitive reductions in fp32. In PyTorch this is `torch.autocast`:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
loss.backward()   # outside autocast
```

On CUDA, also enable TF32 matmuls for fp32 operations, a free speedup on Ampere and newer GPUs:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# or: torch.set_float32_matmul_precision("high")
```

## The training loop

Nearly every training script on this site is a variation of this loop. Learn it until you can write it without looking.

```python title="train_loop.py" hl_lines="23 24 25 26 27 28"
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# Toy data: 3 Gaussian clusters in 2D
N = 3000
y = torch.randint(0, 3, (N,))
centers = torch.tensor([[2.0, 0.0], [-1.0, 2.0], [-1.0, -2.0]])
X = centers[y] + 0.7 * torch.randn(N, 2)
train_dl = DataLoader(TensorDataset(X[:2500], y[:2500]), batch_size=64, shuffle=True)
val_dl = DataLoader(TensorDataset(X[2500:], y[2500:]), batch_size=256)

model = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 3)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

for epoch in range(5):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                       # 1. forward
        loss = F.cross_entropy(logits, yb)       # 2. loss
        opt.zero_grad(set_to_none=True)          # 3. clear old gradients
        loss.backward()                          # 4. backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 5. clip
        opt.step()                               # 6. update

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(-1) == yb).sum().item()
            total += len(yb)
    print(f"epoch {epoch} train_loss {loss.item():.4f} val_acc {correct/total:.3f}")

torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, "ckpt.pt")
```

The order of `zero_grad`, `backward`, `step` matters; everything else around it (schedulers, accumulation, logging, checkpointing) is added in [The Training Loop](../pretraining/training.md).

### Gradient accumulation

To simulate a batch `k` times larger than fits in memory, divide the loss by `k`, call `backward()` `k` times, then step once:

```python
accum = 4
opt.zero_grad(set_to_none=True)
for i, (xb, yb) in enumerate(train_dl):
    loss = F.cross_entropy(model(xb.to(device)), yb.to(device)) / accum
    loss.backward()                    # gradients add up across micro-batches
    if (i + 1) % accum == 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
```

## Common shape bugs

Most of the errors you will see in model code are one of these. Learn to recognize the messages.

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `mat1 and mat2 shapes cannot be multiplied (32x128 and 64x10)` | Layer input size does not match the tensor's last dim | Check `in_features`; print shapes before the layer |
| `Expected input batch_size (256) to match target batch_size (8)` | Flattened logits to `(B*T, V)` but left targets as `(B, T)` | Flatten both: `logits.view(-1, V)`, `targets.view(-1)` |
| `view size is not compatible with input tensor's size and stride` | `view` after `transpose`/`permute` | `.contiguous().view(...)` or `.reshape(...)` |
| `Expected all tensors to be on the same device` | Model on GPU, data (or a new tensor created in `forward`) on CPU | `.to(device)`; create tensors with `device=x.device` |
| `expected scalar type Long but found Float` | Float class labels in `cross_entropy` | `targets.long()` |
| Loss is fine but accuracy is wrong | `argmax` over the wrong dim | `argmax(dim=-1)` for class scores |
| Loss decreases suspiciously slowly, no error | Silent broadcasting, e.g. `(N,1) - (N,)` | Assert shapes; use `squeeze(-1)` deliberately |
| Model "cheats" and loss goes near 0 | Causal mask missing or wrong, model sees future tokens | Test that output at position `t` does not change when you edit token `t+1` |

The cross-entropy case is so common in language models that it is worth seeing:

```python
B, T, V = 2, 8, 100
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))
# F.cross_entropy(logits, targets)  # wrong: treats dim 1 (T) as classes
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
print(loss)   # near or a bit above ln(100) = 4.6 for random logits
```

!!! tip "Shape assertions are cheap"
    Add `assert x.shape == (B, T, C), x.shape` at the boundaries of your modules while developing. They cost nothing and catch the silent bugs that exceptions do not.

!!! warning "Common pitfalls"
    - **Forgetting `opt.zero_grad()`.** Gradients accumulate across steps and training behaves erratically.
    - **Forgetting `model.eval()` and `torch.no_grad()` during evaluation.** Dropout stays on, results are noisy, and memory grows.
    - **Calling `loss.item()` (or `print(loss)`) every step on GPU.** It forces a CPU-GPU sync. Fine for small scripts; log every N steps in real training.
    - **Creating tensors inside `forward` without a device.** `torch.zeros(T, T)` lands on CPU. Use `device=x.device` or a registered buffer.
    - **Using `float16` without a `GradScaler`.** Small gradients underflow to zero. Prefer `bfloat16` when your hardware supports it.
    - **In-place ops on tensors autograd needs** (`x += 1`, `relu_()`) can raise "modified by an inplace operation" errors. Use out-of-place versions.

## Check your understanding

??? question "What is the result shape of `torch.randn(8, 1, 5) + torch.randn(3, 1)`?"
    Align from the right: `(8, 1, 5)` and `(3, 1)`. Last dim: 5 vs 1 gives 5. Middle: 1 vs 3 gives 3. First: 8. Result `(8, 3, 5)`.

??? question "Write an einsum that computes the attention-weighted sum of values, given `att` (B, H, T, T) and `v` (B, H, T, D)."
    `torch.einsum("bhqk,bhkd->bhqd", att, v)`, which is the same as `att @ v`.

??? question "You call `loss.backward()` twice without zeroing. What is in `w.grad`?"
    The sum of both gradients, twice the single-step gradient if the inputs were the same (you would also need `retain_graph=True` for the second call to succeed on the same graph).

??? question "Why does `F.cross_entropy` fail with logits of shape (B, T, V) and targets (B, T)?"
    It expects classes in dimension 1, so it reads (B, T, V) as batch B with T classes and V extra dims. Flatten to (B·T, V) and (B·T,), or permute to (B, V, T).

??? question "Your model trains on GPU but crashes in `forward` with a device error only during generation. What is a likely cause?"
    A tensor created inside `forward` or the generation loop, such as a mask or position indices from `torch.arange(T)`, without `device=...`. It lands on CPU while everything else is on GPU.

## Further reading

- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) official tutorial
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) in the PyTorch docs
- [torch.einsum documentation](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html)
- Andrej Karpathy, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), the "makemore" lectures for PyTorch fluency
