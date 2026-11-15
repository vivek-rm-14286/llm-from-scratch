"""STARTER: fill in every function marked TODO, then run `pytest`.

SFT from scratch: masked next-token loss, cosine schedule, explicit training loop."""
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
    # TODO: logits = model(input_ids=x).logits; drop the last position, drop the first label,
    #       then F.cross_entropy(..., ignore_index=-100) on the flattened tensors (cast logits to float).
    raise NotImplementedError("TODO: implement sft_loss")


def lr_at(step, *, base_lr, warmup_steps, total_steps, min_ratio=0.1):
    """Linear warmup for warmup_steps, then cosine decay to min_ratio * base_lr.

    step is 0-based: step 0 gets base_lr / warmup_steps, step warmup_steps - 1 gets
    base_lr, and step total_steps - 1 or later gets about min_ratio * base_lr.
    """
    # TODO: Warmup: base_lr * (step + 1) / warmup_steps.
    #       After: progress = min(1, (step - warmup_steps + 1) / max(1, total_steps - warmup_steps))
    #       and lr = base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + cos(pi * progress))).
    raise NotImplementedError("TODO: implement lr_at")


@torch.no_grad()
def eval_loss(model, data, *, pad_id, max_len, batch_size=16, device="cpu"):
    """Token-weighted mean masked loss over `data`, a list of (ids, labels).
    Equal to sft_loss on the whole set as one batch. Leaves the model in its
    previous train/eval mode."""
    # TODO: Remember model.training, switch to eval, loop over batches from collate,
    #       weight each batch loss by its number of labeled targets (y[:, 1:] != -100), restore the mode.
    #       Call release_cache(device) after each batch (it matters on Apple Silicon, see common.py).
    raise NotImplementedError("TODO: implement eval_loss")


def train_sft(model, train_data, eval_data, *, pad_id, max_len=256, batch_size=16,
              epochs=1, lr=1e-4, warmup_steps=30, weight_decay=0.0, grad_clip=1.0,
              eval_every=100, device="cpu", seed=0, log=print):
    """Full fine-tuning with AdamW. train_data / eval_data: lists of (ids, labels).

    Returns a history: list of dicts with keys step, train_loss, eval_loss, lr, seconds.
    The first entry is step 0 (before any update) and the last is the final step.
    """
    # TODO: 1. AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)).
    #       2. total_steps = ceil(len(train_data) / batch_size) * epochs.
    #       3. Record eval_loss at step 0.
    #       4. Each epoch: batches = length_bucketed_batches(lengths, batch_size, seed=seed + epoch);
    #          for each: collate, set the lr with lr_at, forward (bf16 autocast on CUDA only),
    #          zero_grad, backward, clip_grad_norm_, opt.step(), release_cache(device).
    #       5. Every eval_every steps and at the last step: eval_loss, append a history row, log it.
    raise NotImplementedError("TODO: implement train_sft")
