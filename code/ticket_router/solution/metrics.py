"""Metrics, calibration and the decision rule. numpy only, so every function is easy to test by hand."""
import math

import numpy as np


def softmax(logits, temperature=1.0):
    """Row-wise softmax of logits / temperature. logits: (N, C) or (C,)."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)              # stability: largest logit becomes 0
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(logits, labels, temperature=1.0):
    """Mean negative log-likelihood of the true labels."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    labels = np.asarray(labels)
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def accuracy(pred, labels):
    return float((np.asarray(pred) == np.asarray(labels)).mean())


def confusion_matrix(labels, pred, n_classes):
    """cm[t, p] = number of examples with true class t predicted as p."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    return cm


def macro_f1(pred, labels, n_classes):
    """Unweighted mean of the per-class F1 scores. A class with no predictions and no examples scores 0."""
    cm = confusion_matrix(labels, pred, n_classes)
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE with equal-width confidence bins.

    Returns (ece, rows); each row is (lo, hi, mean confidence, accuracy, count) for a non-empty bin.
    A confidence c falls in the bin with lo < c <= hi.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            bin_conf, bin_acc = conf[in_bin].mean(), correct[in_bin].mean()
            ece += in_bin.mean() * abs(bin_acc - bin_conf)
            rows.append((float(lo), float(hi), float(bin_conf), float(bin_acc), int(in_bin.sum())))
    return float(ece), rows


def fit_temperature(logits, labels, lo=0.05, hi=20.0, iters=60):
    """The scalar T > 0 that minimizes NLL of softmax(logits / T).

    NLL has a single minimum in log T, so a golden-section search on log T is enough.
    """
    a, b = math.log(lo), math.log(hi)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = nll(logits, labels, math.exp(c)), nll(logits, labels, math.exp(d))
    for _ in range(iters):
        if fc < fd:                                    # minimum is in [a, d]
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = nll(logits, labels, math.exp(c))
        else:                                          # minimum is in [c, b]
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = nll(logits, labels, math.exp(d))
    return float(math.exp((a + b) / 2))


def bayes_threshold(cost_misroute, cost_escalate):
    """Confidence below which escalating is cheaper than routing.

    Routing costs (1 - p) * cost_misroute in expectation, escalating costs cost_escalate.
    Route when (1 - p) * cost_misroute <= cost_escalate, i.e. p >= 1 - cost_escalate / cost_misroute.
    """
    return 1.0 - cost_escalate / cost_misroute


def selective_stats(conf, correct, threshold, cost_misroute=5.0, cost_escalate=1.0):
    """What happens if every prediction with confidence < threshold is escalated to a human."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    routed = conf >= threshold
    n_routed = int(routed.sum())
    errors = int((routed & ~correct).sum())
    cost = (errors * cost_misroute + (len(conf) - n_routed) * cost_escalate) / len(conf)
    return {
        "threshold": float(threshold),
        "abstain_rate": float(1.0 - routed.mean()),
        "coverage": float(routed.mean()),
        "error_rate_routed": float(errors / n_routed) if n_routed else 0.0,
        "misroute_rate_overall": float(errors / len(conf)),
        "mean_cost": float(cost),
    }


def sweep_threshold(conf, correct, cost_misroute=5.0, cost_escalate=1.0, grid=None):
    """Try every threshold on the grid; return (best threshold, list of selective_stats rows).

    Ties go to the lowest threshold (the one that escalates least).
    """
    grid = np.round(np.linspace(0.0, 1.0, 101), 2) if grid is None else grid
    rows = [selective_stats(conf, correct, t, cost_misroute, cost_escalate) for t in grid]
    best = min(rows, key=lambda r: (r["mean_cost"], r["threshold"]))
    return best["threshold"], rows


def latency_percentiles(samples_ms):
    """p50 and p95 of a list of per-request latencies."""
    s = np.asarray(samples_ms, dtype=np.float64)
    return {"p50_ms": float(np.percentile(s, 50)), "p95_ms": float(np.percentile(s, 95)),
            "mean_ms": float(s.mean())}
