"""STARTER: fill in every function marked TODO, then run `pytest`.

DPO from scratch: masked sequence log-probs, the loss, and the training loop."""
import copy
import random
import time

import torch
import torch.nn.functional as F

from common import release_cache

from .data import IGNORE, collate, render_chatml


def sequence_logprobs(model, input_ids, labels):
    """Sum of log p(token | prefix) over positions where labels != -100.

    input_ids, labels: (B, T); labels is input_ids with prompt and padding set to -100.
    Returns a (B,) tensor. The log-softmax runs in float32.
    """
    # TODO: Shift exactly as in sft_loss. mask = targets != IGNORE; replace masked targets with 0
    #       so gather() has valid indices; log_softmax in float32; gather; multiply by mask; sum over time.
    raise NotImplementedError("TODO: implement sequence_logprobs")


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """DPO loss from summed log-probs, each of shape (B,).

    pi_* come from the trainable policy, ref_* from the frozen reference; w = chosen,
    l = rejected. Returns (loss, stats) where stats has float entries
    reward_acc, margin, chosen_reward, rejected_reward.
    """
    # TODO: chosen_reward = beta * (pi_w - ref_w); rejected_reward = beta * (pi_l - ref_l);
    #       margin = chosen_reward - rejected_reward; loss = -F.logsigmoid(margin).mean().
    #       reward_acc is the share of margins above 0. Return python floats in stats (.item()).
    raise NotImplementedError("TODO: implement dpo_loss")


def build_dpo_example(pair, encode, im_start_id, im_end_id, nl_ids):
    """pair: {"messages": [...prompt turns...], "chosen": str, "rejected": str}.

    Returns ((chosen_ids, chosen_labels), (rejected_ids, rejected_labels)). Both share
    the same prompt tokens; only the final assistant turn is labeled.
    """
    # TODO: Render the prompt turns with render_chatml (all labels IGNORE), render the response as
    #       a single assistant turn, and concatenate ids and labels. Do it for chosen and for rejected.
    raise NotImplementedError("TODO: implement build_dpo_example")


def dpo_batch(examples, pad_id, max_len):
    """examples: list of build_dpo_example outputs. Returns (ids, labels) of shape
    (2B, T): the B chosen rows first, then the B rejected rows, padded together."""
    # TODO: One collate call over [all chosen rows] + [all rejected rows].
    raise NotImplementedError("TODO: implement dpo_batch")


def freeze_copy(model):
    """A frozen deep copy in eval mode: the DPO reference model."""
    # TODO: copy.deepcopy, .eval(), and requires_grad_(False) on every parameter.
    raise NotImplementedError("TODO: implement freeze_copy")


def dpo_forward(policy, ref_model, ids, labels, beta):
    """Loss and stats for one (2B, T) batch laid out as in dpo_batch."""
    # TODO: B = ids.size(0) // 2. Reference log-probs under torch.no_grad(), policy log-probs with
    #       gradients, then dpo_loss(pi[:B], pi[B:], ref[:B], ref[B:], beta).
    raise NotImplementedError("TODO: implement dpo_forward")


@torch.no_grad()
def eval_dpo(policy, ref_model, examples, *, pad_id, max_len, beta, batch_size=8, device="cpu"):
    """Pair-weighted mean of loss, reward_acc and margin over held-out pairs."""
    was_training = policy.training
    policy.eval()
    tot = {"loss": 0.0, "reward_acc": 0.0, "margin": 0.0}
    for s in range(0, len(examples), batch_size):
        chunk = examples[s : s + batch_size]
        ids, labels = dpo_batch(chunk, pad_id, max_len)
        loss, st = dpo_forward(policy, ref_model, ids.to(device), labels.to(device), beta)
        tot["loss"] += loss.item() * len(chunk)
        tot["reward_acc"] += st["reward_acc"] * len(chunk)
        tot["margin"] += st["margin"] * len(chunk)
        release_cache(device)
    policy.train(was_training)
    return {k: v / max(1, len(examples)) for k, v in tot.items()}


def train_dpo(policy, ref_model, train_examples, eval_examples, *, pad_id, max_len=256,
              beta=0.1, lr=2e-6, batch_size=8, epochs=2, warmup_steps=5, grad_clip=1.0,
              log_every=10, device="cpu", seed=0, log=print):
    """DPO with AdamW and a constant learning rate after linear warmup.

    The reference model is never updated. Returns a history: list of dicts with keys
    step, loss, reward_acc, margin, chosen_reward, rejected_reward (training averages
    since the last log) plus eval_loss, eval_reward_acc, eval_margin when eval pairs exist.
    """
    # TODO: 1. AdamW(policy.parameters(), lr=lr, weight_decay=0.0); random.Random(seed) for shuffling.
    #       2. Append a history row at step 0 (use eval_dpo when eval_examples is non-empty; store its
    #          values under eval_loss, eval_reward_acc, eval_margin).
    #       3. Each epoch: shuffle the indices, for each batch: dpo_batch, lr = lr * min(1, (step + 1) / warmup_steps),
    #          dpo_forward, zero_grad, backward, clip_grad_norm_, opt.step(),
    #          release_cache(device). Never touch ref_model.
    #       4. Collect the per-step stats (plus "loss") and every log_every steps, and at the last step,
    #          append their averages and the eval metrics as a history row.
    raise NotImplementedError("TODO: implement train_dpo")
