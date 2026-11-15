"""Train the decision LM with cross-entropy over the queue options.

    python -m starter.train_decision_lm
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
    # TODO: For each example index: shuffle the option order, build the context with make_context,
    # and record where the true queue landed. Return (contexts, options, LongTensor of targets).
    raise NotImplementedError("TODO: implement make_batch in starter/train_decision_lm.py")


def train(texts, labels, name=BASE_LM, epochs=2, batch_size=16, top_layers=6, lr=5e-5, head_lr=1e-3,
          device="cpu", seed=0, log_every=10):
    # TODO: model = DecisionLM(name); model.set_trainable(top_layers).
    # AdamW with two groups: trainable backbone parameters at `lr`, the scorer at `head_lr`.
    # Each step: loss = F.cross_entropy(model(contexts, opts), targets). Decay the LR linearly to 0.
    # Return model.eval().
    raise NotImplementedError("TODO: implement train in starter/train_decision_lm.py")


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
