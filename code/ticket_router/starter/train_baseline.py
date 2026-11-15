"""Model (a): frozen sentence embeddings + logistic regression.

    python -m starter.train_baseline
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from .common import DEFAULT_DATA, DEFAULT_OUT, load_xy, save_eval_logits
    from .data import QUEUES
except ImportError:
    from common import DEFAULT_DATA, DEFAULT_OUT, load_xy, save_eval_logits
    from data import QUEUES

EMBEDDER = "BAAI/bge-small-en-v1.5"


class EmbeddingLR:
    """A frozen embedder plus a linear layer. Inference needs numpy and the embedder, not scikit-learn."""

    def __init__(self, embedder_name, weight, bias, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.embedder_name = embedder_name
        self.encoder = SentenceTransformer(embedder_name, device=device)
        self.weight = np.asarray(weight, dtype=np.float32)      # (C, d)
        self.bias = np.asarray(bias, dtype=np.float32)          # (C,)

    def embed(self, texts, batch_size=64):
        return self.encoder.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                   show_progress_bar=False)

    def logits(self, texts):
        # TODO: Embed the texts, then apply the linear layer: X @ weight.T + bias.
        raise NotImplementedError("TODO: implement logits in starter/train_baseline.py")

    @classmethod
    def fit(cls, texts, labels, embedder_name=EMBEDDER, C=10.0):
        # TODO: Embed the training texts (normalized embeddings), fit sklearn LogisticRegression(C=C, max_iter=2000),
        # and copy clf.coef_ / clf.intercept_ into a (n_queues, d) weight and (n_queues,) bias,
        # indexed by clf.classes_. Return an EmbeddingLR.
        raise NotImplementedError("TODO: implement fit in starter/train_baseline.py")

    def save(self, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez(model_dir / "head.npz", weight=self.weight, bias=self.bias)
        (model_dir / "config.json").write_text(json.dumps({"embedder": self.embedder_name, "queues": QUEUES}))

    @classmethod
    def load(cls, model_dir):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        head = np.load(model_dir / "head.npz")
        return cls(cfg["embedder"], head["weight"], head["bias"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--embedder", default=EMBEDDER)
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--train-size", type=int, default=None, help="use only the first N training rows")
    args = ap.parse_args()

    texts, y = load_xy(args.data, "train", args.train_size)
    t0 = time.perf_counter()
    model = EmbeddingLR.fit(texts, y, args.embedder, args.C)
    train_s = time.perf_counter() - t0
    print(f"baseline: embedded {len(texts)} messages and fit logistic regression in {train_s:.1f} s")
    model_dir = Path(args.out) / "baseline"
    model.save(model_dir)
    save_eval_logits(model_dir, model.logits, args.data,
                     {"model": "baseline", "embedder": args.embedder, "C": args.C,
                      "n_train": len(texts), "train_seconds": round(train_s, 1)})


if __name__ == "__main__":
    main()
