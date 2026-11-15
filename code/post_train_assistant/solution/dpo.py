"""DPO from scratch: masked sequence log-probs, the loss, and the training loop."""
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
    logits = model(input_ids=input_ids).logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != IGNORE
    safe = targets.masked_fill(~mask, 0)
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(-1)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """DPO loss from summed log-probs, each of shape (B,).

    pi_* come from the trainable policy, ref_* from the frozen reference; w = chosen,
    l = rejected. Returns (loss, stats) where stats has float entries
    reward_acc, margin, chosen_reward, rejected_reward.
    """
    chosen_reward = beta * (pi_w - ref_w)
    rejected_reward = beta * (pi_l - ref_l)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()
    return loss, {
        "reward_acc": (margin > 0).float().mean().item(),
        "margin": margin.mean().item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }


def build_dpo_example(pair, encode, im_start_id, im_end_id, nl_ids):
    """pair: {"messages": [...prompt turns...], "chosen": str, "rejected": str}.

    Returns ((chosen_ids, chosen_labels), (rejected_ids, rejected_labels)). Both share
    the same prompt tokens; only the final assistant turn is labeled.
    """
    out = []
    for key in ("chosen", "rejected"):
        prompt_ids, _ = render_chatml(pair["messages"], encode, im_start_id, im_end_id, nl_ids)
        resp_ids, resp_labels = render_chatml([{"role": "assistant", "content": pair[key]}],
                                              encode, im_start_id, im_end_id, nl_ids)
        out.append((prompt_ids + resp_ids, [IGNORE] * len(prompt_ids) + resp_labels))
    return tuple(out)


def dpo_batch(examples, pad_id, max_len):
    """examples: list of build_dpo_example outputs. Returns (ids, labels) of shape
    (2B, T): the B chosen rows first, then the B rejected rows, padded together."""
    rows = [ex[0] for ex in examples] + [ex[1] for ex in examples]
    return collate(rows, pad_id, max_len)


def freeze_copy(model):
    """A frozen deep copy in eval mode: the DPO reference model."""
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def dpo_forward(policy, ref_model, ids, labels, beta):
    """Loss and stats for one (2B, T) batch laid out as in dpo_batch."""
    B = ids.size(0) // 2
    with torch.no_grad():
        ref_lp = sequence_logprobs(ref_model, ids, labels)
    pi_lp = sequence_logprobs(policy, ids, labels)
    return dpo_loss(pi_lp[:B], pi_lp[B:], ref_lp[:B], ref_lp[B:], beta)


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
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    policy.to(device).train()
    ref_model.to(device).eval()
    total_steps = epochs * ((len(train_examples) + batch_size - 1) // batch_size)
    history, acc, step, t0 = [], [], 0, time.time()

    def record():
        row = {"step": step, "seconds": time.time() - t0}
        if acc:
            for k in acc[0]:
                row[k] = sum(a[k] for a in acc) / len(acc)
        if eval_examples:
            ev = eval_dpo(policy, ref_model, eval_examples, pad_id=pad_id, max_len=max_len,
                          beta=beta, batch_size=batch_size, device=device)
            row.update({"eval_" + k: v for k, v in ev.items()})
        history.append(row)
        log(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        acc.clear()

    record()                                   # step 0: loss is ln 2, margin is 0
    for epoch in range(epochs):
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        for s in range(0, len(order), batch_size):
            ids, labels = dpo_batch([train_examples[i] for i in order[s : s + batch_size]],
                                    pad_id, max_len)
            for group in opt.param_groups:
                group["lr"] = lr * min(1.0, (step + 1) / max(1, warmup_steps))
            loss, stats = dpo_forward(policy, ref_model, ids.to(device), labels.to(device), beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            opt.step()
            step += 1
            stats["loss"] = loss.item()
            acc.append(stats)
            release_cache(device)
            if step % log_every == 0 or step == total_steps:
                record()
    return history
