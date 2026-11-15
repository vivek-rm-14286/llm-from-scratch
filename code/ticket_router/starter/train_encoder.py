"""Model (b): fine-tune a small encoder end to end with a classification head.

    python -m starter.train_encoder

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
        # TODO: Tokenize in batches (truncation, max_length=MAX_LEN, padding), run the model under
        # inference mode, and return the concatenated logits as a numpy array.
        raise NotImplementedError("TODO: implement logits in starter/train_encoder.py")

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
    # TODO: Load AutoTokenizer and AutoModelForSequenceClassification(name, num_labels=len(QUEUES), dtype=torch.float32).
    # AdamW(lr, weight_decay=0.01), linear warmup over the first warmup_frac of the steps, then linear decay.
    # Each step: tokenize a shuffled batch, cross-entropy on model(**enc).logits, clip gradients to 1.0.
    # Return EncoderClassifier(model.eval(), tok, device).
    raise NotImplementedError("TODO: implement train in starter/train_encoder.py")


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
