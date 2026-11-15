"""Training loop: AdamW, warmup + cosine, gradient clipping, mixed precision where it
helps, validation loss, checkpoint and resume.

    python train.py                               # GPT-2 style, about 11M parameters
    python train.py --modern --out out/modern     # RMSNorm + RoPE + SwiGLU
    python train.py --resume out/gpt2/ckpt_last.pt     # continue an interrupted run
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
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def configure_optimizer(model, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95)):
    """AdamW with weight decay on matrices only (not biases or norm gains)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


@torch.no_grad()
def estimate_loss(model, data, batch_size: int, block_size: int, device, iters: int = 20, generator=None) -> float:
    """Mean loss over `iters` random batches, in eval mode (dropout off)."""
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, batch_size, block_size, device, generator)
        total += model(x, y)[1].item()
    model.train()
    return total / iters


def save_checkpoint(path: str, model, optimizer, step: int, best_val: float, generator=None):
    """Everything needed to continue exactly: weights, optimizer moments, step, config, data RNG."""
    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
            "best_val": best_val, "config": asdict(model.cfg),
            "rng": None if generator is None else generator.get_state()}
    torch.save(ckpt, path + ".tmp")
    os.replace(path + ".tmp", path)                 # atomic: never leave a half-written file


def load_checkpoint(path: str, device="cpu"):
    """Returns (model, ckpt). The caller restores the optimizer and the RNG from `ckpt`."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


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

    model.train()
    t_start = t_log = time.time()
    tokens_per_step = args.batch_size * cfg.block_size
    while step < args.max_steps:
        if step % args.eval_every == 0:
            tr, va = evaluate()
            print(f"step {step:5d} | eval train {tr:.4f} val {va:.4f} | elapsed {time.time() - t_start:.0f}s")
            if va < best_val and step > 0:
                best_val = va
                save_checkpoint(os.path.join(args.out, "ckpt_best.pt"), model, optimizer, step, best_val, gen)
            t_log = time.time()                              # do not count eval time as training time

        lr = lr_at(step, args.lr, args.warmup_steps, args.max_steps, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, args.batch_size, cfg.block_size, device, gen)
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        step += 1

        if step % args.log_every == 0:
            loss_val = loss.item()                           # .item() waits for the device to finish
            dt = time.time() - t_log
            t_log = time.time()
            print(f"step {step:5d} | loss {loss_val:.4f} | lr {lr:.2e} | grad norm {grad_norm:.2f} | "
                  f"{args.log_every * tokens_per_step / dt:,.0f} tok/s")
        if step % args.eval_every == 0 or step == args.max_steps:
            save_checkpoint(os.path.join(args.out, "ckpt_last.pt"), model, optimizer, step, best_val, gen)

    tr, va = evaluate()
    if va < best_val:
        best_val = va
        save_checkpoint(os.path.join(args.out, "ckpt_best.pt"), model, optimizer, step, best_val, gen)
    print(f"step {step:5d} | eval train {tr:.4f} val {va:.4f} | best val {best_val:.4f} | "
          f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
