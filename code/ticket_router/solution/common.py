"""Small helpers shared by the training scripts."""
import json
import random
import time
from pathlib import Path

import numpy as np

try:
    from .data import QUEUES, load_split
except ImportError:
    from data import QUEUES, load_split

ROOT = Path(__file__).resolve().parent.parent          # code/ticket_router
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "out"
SPLITS = ("cal", "test", "ood")


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_xy(data_dir, split, limit=None, label_key="label"):
    """Texts and integer labels for a split. OOD rows get label -1."""
    rows = load_split(data_dir, split)
    if limit:
        rows = rows[:limit]
    texts = [r["text"] for r in rows]
    labels = np.array([QUEUES.index(r[label_key]) if r[label_key] in QUEUES else -1 for r in rows])
    return texts, labels


def save_eval_logits(model_dir, logits_fn, data_dir, meta):
    """Score the calibration, test and OOD splits once and store the raw logits.

    calibrate.py works from these files, so it never has to load a model.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split in SPLITS:
        texts, _ = load_xy(data_dir, split)
        t0 = time.perf_counter()
        arrays[split] = np.asarray(logits_fn(texts), dtype=np.float32)
        print(f"  scored {split:4s} ({len(texts)} messages) in {time.perf_counter() - t0:.1f} s")
    np.savez(model_dir / "logits.npz", **arrays)
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    _, y = load_xy(data_dir, "test", label_key="clean_label")
    acc = float((arrays["test"].argmax(1) == y).mean())
    print(f"  test accuracy (uncalibrated argmax): {acc:.3f}")
    return acc
