"""SFT from scratch: masked next-token loss, cosine schedule, explicit training loop."""
import math
import time

import torch
import torch.nn.functional as F

from common import release_cache

from .data import collate, length_bucketed_batches


def sft_loss(model, x, y):
    """Mean cross-entropy over positions whose label is not -100.

    x, y: (B, T). Position t predicts token t+1, so logits[:, :-1] is compared
    with y[:, 1:]. `model(input_ids=x).logits` has shape (B, T, V).
    """
    logits = model(input_ids=x).logits[:, :-1, :]
    targets = y[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                           targets.reshape(-1), ignore_index=-100)


def lr_at(step, *, base_lr, warmup_steps, total_steps, min_ratio=0.1):
    """Linear warmup for warmup_steps, then cosine decay to min_ratio * base_lr.

    step is 0-based: step 0 gets base_lr / warmup_steps, step warmup_steps - 1 gets
    base_lr, and step total_steps - 1 or later gets about min_ratio * base_lr.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps + 1) / max(1, total_steps - warmup_steps))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_loss(model, data, *, pad_id, max_len, batch_size=16, device="cpu"):
    """Token-weighted mean masked loss over `data`, a list of (ids, labels).
    Equal to sft_loss on the whole set as one batch. Leaves the model in its
    previous train/eval mode."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    for s in range(0, len(order), batch_size):
        x, y = collate([data[i] for i in order[s : s + batch_size]], pad_id, max_len)
        n = int((y[:, 1:] != -100).sum())
        if n == 0:
            continue
        total += sft_loss(model, x.to(device), y.to(device)).item() * n
        count += n
        release_cache(device)                  # MPS: see common.release_cache
    model.train(was_training)
    return total / max(1, count)


def train_sft(model, train_data, eval_data, *, pad_id, max_len=256, batch_size=16,
              epochs=1, lr=1e-4, warmup_steps=30, weight_decay=0.0, grad_clip=1.0,
              eval_every=100, device="cpu", seed=0, log=print):
    """Full fine-tuning with AdamW. train_data / eval_data: lists of (ids, labels).

    Returns a history: list of dicts with keys step, train_loss, eval_loss, lr, seconds.
    The first entry is step 0 (before any update) and the last is the final step.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.95))
    lengths = [min(len(ids), max_len) for ids, _ in train_data]
    steps_per_epoch = math.ceil(len(train_data) / batch_size)
    total_steps = steps_per_epoch * epochs
    use_amp = str(device).startswith("cuda")

    model.to(device).train()
    t0 = time.time()
    ev = eval_loss(model, eval_data, pad_id=pad_id, max_len=max_len, device=device)
    history = [{"step": 0, "train_loss": None, "eval_loss": ev, "lr": 0.0, "seconds": 0.0}]
    log(f"step 0/{total_steps} eval_loss {ev:.4f}")

    step, running = 0, []
    for epoch in range(epochs):
        for idx in length_bucketed_batches(lengths, batch_size, seed=seed + epoch):
            x, y = collate([train_data[i] for i in idx], pad_id, max_len)
            x, y = x.to(device), y.to(device)
            cur_lr = lr_at(step, base_lr=lr, warmup_steps=warmup_steps, total_steps=total_steps)
            for group in opt.param_groups:
                group["lr"] = cur_lr
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = sft_loss(model, x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            step += 1
            running.append(loss.item())
            release_cache(device)
            if step % eval_every == 0 or step == total_steps:
                ev = eval_loss(model, eval_data, pad_id=pad_id, max_len=max_len, device=device)
                tr = sum(running) / len(running)
                running = []
                history.append({"step": step, "train_loss": tr, "eval_loss": ev,
                                "lr": cur_lr, "seconds": time.time() - t0})
                log(f"step {step}/{total_steps} train_loss {tr:.4f} eval_loss {ev:.4f} "
                    f"lr {cur_lr:.2e} {time.time() - t0:.0f}s")
    return history
