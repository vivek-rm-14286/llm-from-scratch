"""Datasets and mini-batching."""
from __future__ import annotations

import gzip
import os
import urllib.request

import numpy as np

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def load_digits_data(seed: int = 0):
    """scikit-learn's bundled 8x8 digits: 1,797 images, 64 features, 10 classes.

    Returns a dict with x_train/y_train (1,257), x_val/y_val (180) and
    x_test/y_test (360). Pixels are scaled from 0..16 to 0..1. The split is
    stratified and fixed by `seed`. Nothing is downloaded.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)
    x_rest, x_test, y_rest, y_test = train_test_split(
        x, y, test_size=360, stratify=y, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest, y_rest, test_size=180, stratify=y_rest, random_state=seed)
    return {"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val,
            "x_test": x_test, "y_test": y_test}


def load_mnist_data(data_dir: str = "data/mnist", seed: int = 0):
    """MNIST: 28x28 images, 784 features. Downloads about 12 MB on first use.

    Returns 55,000 train / 5,000 validation / 10,000 test examples.
    """
    os.makedirs(data_dir, exist_ok=True)
    arrays = {}
    for key, name in MNIST_FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"downloading {MNIST_URL + name}")
            urllib.request.urlretrieve(MNIST_URL + name, path)
        with gzip.open(path, "rb") as f:
            raw = f.read()
        if key.endswith("_x"):
            arrays[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784) / 255.0
        else:
            arrays[key] = np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    perm = np.random.default_rng(seed).permutation(len(arrays["train_x"]))
    val, train = perm[:5000], perm[5000:]
    return {"x_train": arrays["train_x"][train], "y_train": arrays["train_y"][train],
            "x_val": arrays["train_x"][val], "y_val": arrays["train_y"][val],
            "x_test": arrays["test_x"], "y_test": arrays["test_y"]}


def batches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator | None = None):
    """Yield (x_batch, y_batch) pairs that cover the data exactly once.

    With `rng`, the order is `rng.permutation(len(x))`, a fresh shuffle per
    call. Without it, the original order is kept. The last batch may be smaller.
    """
    n = len(x)
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]
