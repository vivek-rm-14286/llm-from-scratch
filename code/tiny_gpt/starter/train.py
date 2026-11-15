"""Step 5. Training loop: AdamW, warmup + cosine, gradient clipping, mixed precision where it
helps, validation loss, checkpoint and resume.

    python train.py                               # GPT-2 style, about 11M parameters
    python train.py --modern --out out/modern     # RMSNorm + RoPE + SwiGLU
    python train.py --resume out/gpt2/ckpt_last.pt     # continue an interrupted run

Grade with:  pytest tests/test_train.py        (from code/tiny_gpt)
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import torch

from data import get_batch, load_tokens
from model import GPT, GPTConfig


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, max_lr: float, warmup_steps: int, max_steps: int, min_lr: float = 0.0) -> float:
    """Linear warmup to max_lr over `warmup_steps`, then cosine decay to min_lr at `max_steps`."""
    # TODO: step < warmup_steps  -> max_lr * (step + 1) / warmup_steps
    #       step >= max_steps    -> min_lr
    #       otherwise            -> min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress)),
    #                               progress = (step - warmup_steps) / (max_steps - warmup_steps)
    raise NotImplementedError("train.lr_at")


def configure_optimizer(model, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95)):
    """AdamW with weight decay on matrices only (not biases or norm gains)."""
    # TODO: two parameter groups: p.dim() >= 2 with `weight_decay`, the rest with 0.0.
    #       Every trainable parameter goes in exactly one group. Return torch.optim.AdamW(groups, lr=lr, betas=betas).
    raise NotImplementedError("train.configure_optimizer")


@torch.no_grad()
def estimate_loss(model, data, batch_size: int, block_size: int, device, iters: int = 20, generator=None) -> float:
    """Mean loss over `iters` random batches, in eval mode (dropout off)."""
    # TODO: model.eval(); average model(x, y)[1].item() over `iters` batches drawn with
    #       get_batch(data, batch_size, block_size, device, generator); model.train(); return a float.
    raise NotImplementedError("train.estimate_loss")


def save_checkpoint(path: str, model, optimizer, step: int, best_val: float, generator=None):
    """Everything needed to continue exactly: weights, optimizer moments, step, config, data RNG."""
    # TODO: torch.save a dict with the keys
    #         "model"      model.state_dict()
    #         "optimizer"  optimizer.state_dict()
    #         "step", "best_val"
    #         "config"     dataclasses.asdict(model.cfg)
    #         "rng"        generator.get_state(), or None
    #       to path + ".tmp", then os.replace it onto `path` (atomic: never a half-written file).
    raise NotImplementedError("train.save_checkpoint")


def load_checkpoint(path: str, device="cpu"):
    """Returns (model, ckpt). The caller restores the optimizer and the RNG from `ckpt`."""
    # TODO: ckpt = torch.load(path, map_location=device, weights_only=True); rebuild the model from
    #       ckpt["config"], load the weights, move it to `device`, return (model, ckpt).
    raise NotImplementedError("train.load_checkpoint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/shakespeare")
    ap.add_argument("--out", default="out/gpt2")
    ap.add_argument("--modern", action="store_true", help="RMSNorm + RoPE + SwiGLU, no biases")
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no-tie", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default=pick_device())
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)             # so `python train.py | tee log.txt` shows progress
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)          # drives batch sampling; saved in checkpoints
    eval_gen = torch.Generator()
    device = args.device
    os.makedirs(args.out, exist_ok=True)
    train_data = load_tokens(os.path.join(args.data, "train.bin"))
    val_data = load_tokens(os.path.join(args.data, "val.bin"))

    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device)
        cfg, step, best_val = model.cfg, ckpt["step"], ckpt["best_val"]
    else:
        with open(os.path.join(args.data, "meta.json")) as f:
            vocab_size = json.load(f)["vocab_size"]
        kw = dict(vocab_size=vocab_size, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
                  d_model=args.d_model, dropout=args.dropout, tie_weights=not args.no_tie)
        cfg = GPTConfig.modern(**kw) if args.modern else GPTConfig(**kw)
        model, step, best_val = GPT(cfg).to(device), 0, float("inf")
    optimizer = configure_optimizer(model, args.lr, args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer"])
        gen.set_state(ckpt["rng"])
    print(f"device {device} | {model.num_params():,} parameters | ln(vocab) = {math.log(cfg.vocab_size):.3f}")

    # bf16 autocast on CUDA. On MPS and CPU fp32 is the default (see the project page for the measurement).
    use_amp = args.amp == "on" or (args.amp == "auto" and device == "cuda")
    amp_dtype = torch.bfloat16 if device != "mps" else torch.float16
    autocast = torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp)

    def evaluate():
        eval_gen.manual_seed(0)                              # same validation batches every time
        tr = estimate_loss(model, train_data, args.batch_size, cfg.block_size, device, args.eval_iters, eval_gen)
        va = estimate_loss(model, val_data, args.batch_size, cfg.block_size, device, args.eval_iters, eval_gen)
        return tr, va

    # TODO: the loop. While step < args.max_steps:
    #   1. every args.eval_every steps: evaluate(), print train and val loss; if val improved
    #      (and step > 0), save ckpt_best.pt
    #   2. set the learning rate of every param group to lr_at(step, ...)
    #   3. x, y = get_batch(train_data, args.batch_size, cfg.block_size, device, gen)
    #   4. forward under `autocast`; optimizer.zero_grad(set_to_none=True); loss.backward()
    #   5. grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    #   6. optimizer.step(); step += 1
    #   7. every args.log_every steps print loss, lr, grad norm and tokens per second
    #      (call loss.item() BEFORE reading the clock: it waits for the device to finish)
    #   8. every args.eval_every steps, and at the end, save ckpt_last.pt
    # After the loop: one last evaluate() and, if it is the best, ckpt_best.pt.
    raise NotImplementedError("train.main: the training loop")


if __name__ == "__main__":
    main()
