"""Model (b): fine-tune a small encoder end to end with a classification head.

    python -m solution.train_encoder

The default checkpoint is BAAI/bge-small-en-v1.5 (33M parameters, 12 layers, hidden size 384),
the same network the baseline keeps frozen, so the comparison isolates the effect of fine-tuning.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .common import DEFAULT_DATA, DEFAULT_OUT, load_xy, save_eval_logits, seed_everything
    from .data import QUEUES
except ImportError:
    from common import DEFAULT_DATA, DEFAULT_OUT, load_xy, save_eval_logits, seed_everything
    from data import QUEUES

ENCODER = "BAAI/bge-small-en-v1.5"
MAX_LEN = 96


class EncoderClassifier:
    def __init__(self, model, tok, device="cpu"):
        self.model, self.tok, self.device = model.to(device), tok, device

    @torch.inference_mode()
    def logits(self, texts, batch_size=64):
        self.model.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(list(texts[i:i + batch_size]), truncation=True, max_length=MAX_LEN,
                           padding=True, return_tensors="pt").to(self.device)
            out.append(self.model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out)

    def save(self, model_dir):
        self.model.save_pretrained(model_dir)
        self.tok.save_pretrained(model_dir)

    @classmethod
    def load(cls, model_dir, device="cpu"):
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32)
        return cls(model.eval(), tok, device)


def train(texts, labels, name=ENCODER, epochs=4, batch_size=32, lr=5e-5, warmup_frac=0.1,
          device="cpu", seed=0, log_every=20):
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(QUEUES), dtype=torch.float32).to(device)   # new, randomly initialized head
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(texts) + batch_size - 1) // batch_size)
    warmup = max(1, int(warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(                     # linear warmup, then linear decay
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup)))
    y = torch.as_tensor(labels)
    rng = np.random.default_rng(seed)
    step = 0
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(texts))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to(device)
            loss = F.cross_entropy(model(**enc).logits, y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % log_every == 0 or step == steps:
                print(f"  epoch {epoch + 1} step {step}/{steps} loss {loss.item():.3f}")
    return EncoderClassifier(model.eval(), tok, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=ENCODER)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-size", type=int, default=None)
    args = ap.parse_args()

    texts, y = load_xy(args.data, "train", args.train_size)
    t0 = time.perf_counter()
    clf = train(texts, y, args.model, args.epochs, args.batch_size, args.lr, device=args.device, seed=args.seed)
    train_s = time.perf_counter() - t0
    print(f"encoder: fine-tuned {args.model} on {len(texts)} messages in {train_s:.0f} s ({args.device})")
    model_dir = Path(args.out) / "encoder"
    clf.save(model_dir)
    save_eval_logits(model_dir, clf.logits, args.data,
                     {"model": "encoder", "checkpoint": args.model, "epochs": args.epochs, "lr": args.lr,
                      "n_train": len(texts), "train_seconds": round(train_s, 1), "device": args.device})


if __name__ == "__main__":
    main()
