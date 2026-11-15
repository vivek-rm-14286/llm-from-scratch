"""Metrics, calibration and the decision rule. numpy only, so every function is easy to test by hand."""
import math

import numpy as np


def softmax(logits, temperature=1.0):
    """Row-wise softmax of logits / temperature. logits: (N, C) or (C,)."""
    # TODO: Divide by the temperature, subtract the row maximum for stability, exponentiate, normalize.
    raise NotImplementedError("TODO: implement softmax in starter/metrics.py")


def nll(logits, labels, temperature=1.0):
    """Mean negative log-likelihood of the true labels."""
    # TODO: log-softmax of logits / temperature, pick the true-label column, negate, average.
    raise NotImplementedError("TODO: implement nll in starter/metrics.py")


def accuracy(pred, labels):
    # TODO: Fraction of positions where pred == labels.
    raise NotImplementedError("TODO: implement accuracy in starter/metrics.py")


def confusion_matrix(labels, pred, n_classes):
    """cm[t, p] = number of examples with true class t predicted as p."""
    # TODO: cm[t, p] += 1 for every (true, predicted) pair.
    raise NotImplementedError("TODO: implement confusion_matrix in starter/metrics.py")


def macro_f1(pred, labels, n_classes):
    """Unweighted mean of the per-class F1 scores. A class with no predictions and no examples scores 0."""
    # TODO: Per class: F1 = 2 TP / (2 TP + FP + FN), read from the confusion matrix. Return the plain mean.
    raise NotImplementedError("TODO: implement macro_f1 in starter/metrics.py")


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE with equal-width confidence bins.

    Returns (ece, rows); each row is (lo, hi, mean confidence, accuracy, count) for a non-empty bin.
    A confidence c falls in the bin with lo < c <= hi.
    """
    # TODO: confidence = max probability, correct = (argmax == label).
    # For each of n_bins equal-width bins (lo < conf <= hi): gap = |accuracy - mean confidence|.
    # ECE = sum over bins of (fraction of samples in the bin) * gap.
    # Return (ece, rows) with rows = [(lo, hi, mean_conf, accuracy, count), ...] for non-empty bins.
    raise NotImplementedError("TODO: implement expected_calibration_error in starter/metrics.py")


def fit_temperature(logits, labels, lo=0.05, hi=20.0, iters=60):
    """The scalar T > 0 that minimizes NLL of softmax(logits / T).

    NLL has a single minimum in log T, so a golden-section search on log T is enough.
    """
    # TODO: Minimize nll(logits, labels, T) over T in [lo, hi].
    # A golden-section search on log T works; so does a fine grid followed by refinement.
    raise NotImplementedError("TODO: implement fit_temperature in starter/metrics.py")


def bayes_threshold(cost_misroute, cost_escalate):
    """Confidence below which escalating is cheaper than routing.

    Routing costs (1 - p) * cost_misroute in expectation, escalating costs cost_escalate.
    Route when (1 - p) * cost_misroute <= cost_escalate, i.e. p >= 1 - cost_escalate / cost_misroute.
    """
    # TODO: Route when (1 - p) * cost_misroute <= cost_escalate. Solve for p.
    raise NotImplementedError("TODO: implement bayes_threshold in starter/metrics.py")


def selective_stats(conf, correct, threshold, cost_misroute=5.0, cost_escalate=1.0):
    """What happens if every prediction with confidence < threshold is escalated to a human."""
    # TODO: routed = conf >= threshold. Count routed-and-wrong predictions.
    # mean_cost = (errors * cost_misroute + n_abstained * cost_escalate) / n.
    # Return a dict with keys threshold, abstain_rate, coverage, error_rate_routed,
    # misroute_rate_overall, mean_cost.
    raise NotImplementedError("TODO: implement selective_stats in starter/metrics.py")


def sweep_threshold(conf, correct, cost_misroute=5.0, cost_escalate=1.0, grid=None):
    """Try every threshold on the grid; return (best threshold, list of selective_stats rows).

    Ties go to the lowest threshold (the one that escalates least).
    """
    # TODO: Evaluate selective_stats on a grid of thresholds (default 0.00, 0.01, ..., 1.00).
    # Return (threshold with the lowest mean_cost, all rows). Break ties toward the lower threshold.
    raise NotImplementedError("TODO: implement sweep_threshold in starter/metrics.py")


def latency_percentiles(samples_ms):
    """p50 and p95 of a list of per-request latencies."""
    s = np.asarray(samples_ms, dtype=np.float64)
    return {"p50_ms": float(np.percentile(s, 50)), "p95_ms": float(np.percentile(s, 95)),
            "mean_ms": float(s.mean())}
