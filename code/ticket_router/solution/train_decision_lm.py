"""Train the decision LM with cross-entropy over the queue options.

    python -m solution.train_decision_lm
"""
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .common import DEFAULT_DATA, DEFAULT_OUT, load_xy, save_eval_logits, seed_everything
    from .data import QUEUES
    from .decision_lm import BASE_LM, DecisionLM, QueueScorer, make_context, queue_options
except ImportError:
    from common import DEFAULT_DATA, DEFAULT_OUT, load_xy, save_eval_logits, seed_everything
    from data import QUEUES
    from decision_lm import BASE_LM, DecisionLM, QueueScorer, make_context, queue_options


def make_batch(texts, labels, idx, rng):
    """Shuffle the option order per example and return (contexts, options, target index)."""
    options = queue_options()
    contexts, opts, targets = [], [], []
    for j in idx:
        order = list(range(len(QUEUES)))
        rng.shuffle(order)                               # harmless here, required for letter-style scoring
        contexts.append(make_context(texts[j]))
        opts.append([options[k] for k in order])
        targets.append(order.index(int(labels[j])))
    return contexts, opts, torch.tensor(targets)


def train(texts, labels, name=BASE_LM, epochs=2, batch_size=16, top_layers=6, lr=5e-5, head_lr=1e-3,
          device="cpu", seed=0, log_every=10):
    seed_everything(seed)
    rng = random.Random(seed)
    model = DecisionLM(name).to(device)
    model.set_trainable(top_layers)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": lr},
                             {"params": model.scorer.parameters(), "lr": head_lr}], weight_decay=0.01)
    steps = epochs * ((len(texts) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    step = 0
    model.train()
    for epoch in range(epochs):
        order = list(range(len(texts)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            contexts, opts, targets = make_batch(texts, labels, order[i:i + batch_size], rng)
            loss = F.cross_entropy(model(contexts, opts), targets.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % log_every == 0 or step == steps:
                print(f"  epoch {epoch + 1} step {step}/{steps} loss {loss.item():.3f}", flush=True)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=BASE_LM)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--top-layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-size", type=int, default=None)
    args = ap.parse_args()

    texts, y = load_xy(args.data, "train", args.train_size)
    t0 = time.perf_counter()
    model = train(texts, y, args.model, args.epochs, args.batch_size, args.top_layers, args.lr,
                  device=args.device, seed=args.seed)
    train_s = time.perf_counter() - t0
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"decision LM: trained {n_trainable / 1e6:.1f}M of "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {len(texts)} messages "
          f"in {train_s:.0f} s ({args.device})")
    model_dir = Path(args.out) / "decision_lm"
    model.save(model_dir)
    model.to("cpu")
    save_eval_logits(model_dir, QueueScorer(model).logits, args.data,
                     {"model": "decision_lm", "checkpoint": args.model, "epochs": args.epochs,
                      "top_layers": args.top_layers, "lr": args.lr, "n_train": len(texts),
                      "train_seconds": round(train_s, 1), "device": args.device})


if __name__ == "__main__":
    main()
