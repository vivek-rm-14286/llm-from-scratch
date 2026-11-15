"""Pretrain the SLM. Single GPU, multi-GPU (DDP via torchrun), or CPU/MPS for debugging.

    # 1-minute smoke test on a laptop (builds a synthetic dataset if needed)
    python train.py --tiny
    # laptop warm-up: the ~31M model on TinyStories (see the project page, Part 0)
    python train.py --config configs/p30m_laptop.json

    # one GPU
    python train.py --config configs/p500m_10b.json
    # 8 GPUs on one node
    torchrun --standalone --nproc_per_node=8 train.py --config configs/p500m_10b.json
    # resume (picks up model, optimizer, step and schedule)
    torchrun --standalone --nproc_per_node=8 train.py --config configs/p500m_10b.json --resume out/p500m/ckpt_latest.pt
    # only compute validation loss / perplexity of a checkpoint
    python train.py --config configs/p500m_10b.json --resume out/p500m/ckpt_latest.pt --eval_only

Any key in the JSON config can be overridden on the command line.
"""
import argparse
import contextlib
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import PRESETS, get_config
from count_params import flops_per_token
from model import Transformer


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON file whose keys set defaults for the flags below")
    ap.add_argument("--tiny", action="store_true", help="CPU/MPS debug preset")
    # model
    ap.add_argument("--model", default="p500m", choices=list(PRESETS))
    ap.add_argument("--seq_len", type=int, default=2048)
    # data
    ap.add_argument("--data_dir", default="data/fwe")
    ap.add_argument("--out_dir", default="out/p500m")
    # batch
    ap.add_argument("--batch_size", type=int, default=16, help="micro-batch per GPU (sequences)")
    ap.add_argument("--total_batch_tokens", type=int, default=2**19, help="tokens per optimizer step")
    # optimization
    ap.add_argument("--lr", type=float, default=8e-4, help="peak learning rate")
    ap.add_argument("--min_lr_frac", type=float, default=0.0, help="final lr as fraction of lr")
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--schedule", default="wsd", choices=["wsd", "cosine"])
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--decay_frac", type=float, default=0.2, help="WSD: fraction of steps spent decaying")
    ap.add_argument("--max_steps", type=int, default=19073, help="10B tokens at 2^19 tokens/step")
    # system
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--compile", type=int, default=-1, help="1/0; default on for CUDA")
    ap.add_argument("--peak_tflops", type=float, default=989.0, help="per-GPU peak for MFU (H100 bf16)")
    ap.add_argument("--seed", type=int, default=1337)
    # logging / checkpoints
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--sample_every", type=int, default=1000)
    ap.add_argument("--sample_prompt", default="The most important thing about")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--keep_every", type=int, default=5000, help="also keep ckpt_<step>.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="slm-500m")

    pre, _ = ap.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            ap.set_defaults(**json.load(f))
    if pre.tiny:
        ap.set_defaults(model="tiny", seq_len=128, data_dir="data/tiny", out_dir="out/tiny",
                        batch_size=8, total_batch_tokens=8 * 128 * 2, lr=3e-3,
                        warmup_steps=10, max_steps=60, log_every=5, eval_every=20,
                        eval_batches=5, sample_every=30, save_every=30, keep_every=0,
                        dtype="fp32", compile=0, sample_prompt="Ada")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Data: random windows from memory-mapped token shards
# ---------------------------------------------------------------------------
class ShardSampler:
    def __init__(self, data_dir, split, seq_len, dtype):
        self.files = sorted(glob.glob(os.path.join(data_dir, f"*_{split}*.bin")))
        assert self.files, f"no {split} shards in {data_dir}"
        self.dtype, self.seq_len = np.dtype(dtype), seq_len
        self.sizes = np.array([os.path.getsize(f) // self.dtype.itemsize for f in self.files])
        # Sampling a shard proportionally to its length == sampling tokens uniformly,
        # so a data mix keeps the proportions it was written with.
        self.probs = self.sizes / self.sizes.sum()

    def get_batch(self, batch_size, rng, device):
        shard_ids = rng.choice(len(self.files), size=batch_size, p=self.probs)
        rows = []
        for s in shard_ids:
            # Re-open the memmap each time: long-lived memmaps can creep in RSS.
            data = np.memmap(self.files[s], dtype=self.dtype, mode="r")
            start = rng.integers(0, len(data) - self.seq_len - 1)
            rows.append(np.array(data[start:start + self.seq_len + 1], dtype=np.int64))
        buf = torch.from_numpy(np.stack(rows))
        x, y = buf[:, :-1].contiguous(), buf[:, 1:].contiguous()
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Learning-rate schedules
# ---------------------------------------------------------------------------
def lr_at(step, args):
    peak, floor = args.lr, args.lr * args.min_lr_frac
    if step < args.warmup_steps:
        return peak * (step + 1) / args.warmup_steps
    if args.schedule == "cosine":
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(t, 1.0)))
    # WSD: warmup -> constant -> linear decay over the last `decay_frac` of training.
    decay_start = int(args.max_steps * (1 - args.decay_frac))
    if step < decay_start:
        return peak
    t = (step - decay_start) / max(1, args.max_steps - decay_start)
    return floor + (peak - floor) * (1 - t)


def make_optimizer(model, args, device):
    # Decay matrices (linear layers, embedding); never decay norm gains.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8,
                             fused=device.type == "cuda")


# ---------------------------------------------------------------------------
def main():
    args = get_args()

    # ---- distributed setup ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, local_rank, world = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world = 0, 0, 1
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device(args.device)
    master = rank == 0
    torch.manual_seed(args.seed + rank)

    if args.dtype == "auto":
        args.dtype = "bf16" if device.type == "cuda" else "fp32"
    use_compile = (device.type == "cuda") if args.compile == -1 else bool(args.compile)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # bf16 autocast: matmuls run in bf16, master weights + optimizer state stay fp32.
    autocast = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if args.dtype == "bf16" else contextlib.nullcontext())

    # ---- data ----
    if args.tiny and not os.path.exists(os.path.join(args.data_dir, "meta.json")) and master:
        from prepare_data import build_tiny
        build_tiny(args.data_dir)
    if ddp:
        dist.barrier()
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    train_data = ShardSampler(args.data_dir, "train", args.seq_len, meta["dtype"])
    val_data = ShardSampler(args.data_dir, "val", args.seq_len, meta["dtype"])

    # ---- model ----
    overrides = {"max_seq_len": args.seq_len}
    if args.model == "tiny":
        overrides["vocab_size"] = meta["vocab_size"]
    cfg = get_config(args.model, **overrides)
    assert meta["vocab_size"] <= cfg.vocab_size, "tokenizer vocab larger than model vocab"
    model = Transformer(cfg).to(device)
    model.init_weights()
    raw_model = model
    n_params = raw_model.num_params()

    optimizer = make_optimizer(raw_model, args, device)
    start_step, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        if not args.eval_only:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt.get("best_val", best_val)
        if master:
            print(f"resumed from {args.resume} at step {start_step}")

    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    tokens_per_micro = args.batch_size * args.seq_len * world
    assert args.total_batch_tokens % tokens_per_micro == 0, \
        "total_batch_tokens must be divisible by batch_size * seq_len * world_size"
    grad_accum = args.total_batch_tokens // tokens_per_micro
    fpt = flops_per_token(cfg, n_params)
    if master:
        print(f"params: {n_params:,} | device {device} x{world} | dtype {args.dtype} | compile {use_compile}")
        print(f"tokens/step {args.total_batch_tokens:,} = {args.batch_size} seq x {args.seq_len} "
              f"x {world} gpu x {grad_accum} accum | total {args.total_batch_tokens * args.max_steps / 1e9:.2f}B tokens")
        os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = None
    if master:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(meta["tokenizer"])

    wb = None
    if args.wandb and master:
        import wandb
        wb = wandb.init(project=args.wandb_project, config={**vars(args), **cfg.to_dict()})

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for i in range(args.eval_batches):
            rng = np.random.default_rng((999, rank, i))  # same batches every eval, independent of --seed
            x, y = val_data.get_batch(args.batch_size, rng, device)
            with autocast:
                _, loss = model(x, y)
            losses.append(loss.item())
        val = torch.tensor(sum(losses) / len(losses), device=device)
        if ddp:
            dist.all_reduce(val, op=dist.ReduceOp.AVG)
        model.train()
        return val.item()

    def save(step, name):
        ckpt = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "best_val": best_val, "model_config": cfg.to_dict(),
                "args": vars(args), "tokenizer": meta["tokenizer"]}
        path = os.path.join(args.out_dir, name)
        torch.save(ckpt, path + ".tmp")
        os.replace(path + ".tmp", path)  # atomic: a crash mid-save never corrupts the old file

    def sample(n=48):
        prompt = args.sample_prompt
        raw_model.eval()
        ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
        with autocast:
            out = list(raw_model.generate(ids, n, temperature=0.8, top_p=0.95,
                                          stop_ids={meta["eot_id"]}))
        raw_model.train()
        return prompt + tokenizer.decode(out)

    if args.eval_only:
        v = evaluate()
        if master:
            print(f"val loss {v:.4f} | perplexity {math.exp(v):.2f}")
        if ddp:
            dist.destroy_process_group()
        return

    # ---- training loop ----
    model.train()
    tokens_seen = start_step * args.total_batch_tokens
    for step in range(start_step, args.max_steps):
        last = step == args.max_steps - 1

        if step % args.eval_every == 0 or last:
            v = evaluate()
            if master:
                print(f"step {step:6d} | val loss {v:.4f} | ppl {math.exp(v):.1f}")
                if wb:
                    wb.log({"val/loss": v, "val/ppl": math.exp(v)}, step=step)
                if v < best_val:
                    best_val = v
        if master and args.sample_every and step > 0 and (step % args.sample_every == 0 or last):
            print(f"  sample: {sample()!r}")

        lr = lr_at(step, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        log_now = step % args.log_every == 0 or last
        if log_now and device.type == "cuda":
            torch.cuda.synchronize()  # time exactly this step's work
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for micro in range(grad_accum):
            rng = np.random.default_rng((args.seed, rank, step, micro))  # resumable, per-rank
            x, y = train_data.get_batch(args.batch_size, rng, device)
            # Only all-reduce gradients on the last micro-step.
            sync = contextlib.nullcontext() if (not ddp or micro == grad_accum - 1) else model.no_sync()
            with sync:
                with autocast:
                    _, loss = model(x, y)
                loss = loss / grad_accum
                loss.backward()
            loss_accum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()
        tokens_seen += args.total_batch_tokens

        if log_now:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            tok_s = args.total_batch_tokens / dt
            mfu = fpt * tok_s / (args.peak_tflops * 1e12 * world) if device.type == "cuda" else float("nan")
            mfu_s = f"{mfu:.1%}" if device.type == "cuda" else "n/a"
            if master:
                print(f"step {step:6d} | loss {loss_accum.item():.4f} | lr {lr:.2e} | "
                      f"gnorm {grad_norm.item():.2f} | {dt * 1000:.0f} ms | {tok_s:,.0f} tok/s | "
                      f"MFU {mfu_s} | {tokens_seen / 1e9:.3f}B tok")
                if wb:
                    wb.log({"train/loss": loss_accum.item(), "lr": lr, "grad_norm": grad_norm.item(),
                            "tok_per_s": tok_s, "mfu": mfu, "tokens": tokens_seen}, step=step)

        if master and ((step + 1) % args.save_every == 0 or last):
            save(step + 1, "ckpt_latest.pt")
            if args.keep_every and (step + 1) % args.keep_every == 0:
                save(step + 1, f"ckpt_{step + 1:07d}.pt")

    if master:
        print(f"done. best val loss {best_val:.4f}. checkpoint: {args.out_dir}/ckpt_latest.pt")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
