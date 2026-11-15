"""Metric functions against values computed by hand."""
import math

import numpy as np
import pytest


@pytest.fixture(scope="module")
def M(impl):
    return impl("metrics")


def test_softmax_rows_sum_to_one_and_temperature_flattens(M):
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    p = M.softmax(logits)
    assert np.allclose(p.sum(1), 1.0)
    assert np.allclose(p[1], 1 / 3)
    assert np.allclose(p[0], np.exp(logits[0]) / np.exp(logits[0]).sum())
    assert M.softmax(logits, 4.0)[0].max() < p[0].max()
    assert np.isfinite(M.softmax(np.array([[1000.0, 0.0]]))).all()


def test_nll_by_hand(M):
    logits = np.log(np.array([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]]))
    expected = -(math.log(0.5) + math.log(0.1)) / 2
    assert M.nll(logits, np.array([0, 2])) == pytest.approx(expected)


def test_accuracy_confusion_and_macro_f1_by_hand(M):
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    assert M.accuracy(pred, y) == pytest.approx(4 / 6)
    assert M.confusion_matrix(y, pred, 3).tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # F1: class 0 = 2*1/(2+1+1) = 0.5, class 1 = 4/5 = 0.8, class 2 = 2/3
    assert M.macro_f1(pred, y, 3) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_ece_by_hand(M):
    # confidences 0.95, 0.95, 0.65, 0.65 with outcomes right, wrong, right, right
    probs = np.array([[0.95, 0.05], [0.95, 0.05], [0.65, 0.35], [0.35, 0.65]])
    labels = np.array([0, 1, 0, 1])
    ece, rows = M.expected_calibration_error(probs, labels, n_bins=10)
    # bin (0.9, 1.0]: conf 0.95, acc 0.5, weight 0.5 -> 0.225 ; bin (0.6, 0.7]: conf 0.65, acc 1.0 -> 0.175
    assert ece == pytest.approx(0.5 * 0.45 + 0.5 * 0.35)
    assert len(rows) == 2
    assert rows[0][2:] == pytest.approx((0.65, 1.0, 2))
    assert rows[1][2:] == pytest.approx((0.95, 0.5, 2))


def test_ece_is_zero_for_a_perfectly_calibrated_bin(M):
    probs = np.array([[0.75, 0.25]] * 4)
    assert M.expected_calibration_error(probs, np.array([0, 0, 0, 1]))[0] == pytest.approx(0.0)


def _overconfident(n=4000, c=5, scale=3.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    z = rng.normal(0, 1, (n, c))
    z[np.arange(n), y] += 1.5
    return scale * z, y


def test_temperature_scaling_lowers_nll_and_ece_and_keeps_predictions(M):
    logits, y = _overconfident()
    fit, held = slice(0, 2000), slice(2000, None)
    T = M.fit_temperature(logits[fit], y[fit])
    assert 1.5 < T < 3.0                                   # the logits were multiplied by 3
    assert M.nll(logits[held], y[held], T) < M.nll(logits[held], y[held]) - 0.05
    raw, cal = M.softmax(logits[held]), M.softmax(logits[held], T)
    assert M.expected_calibration_error(cal, y[held])[0] < M.expected_calibration_error(raw, y[held])[0] / 2
    assert (raw.argmax(1) == cal.argmax(1)).all()


def test_temperature_below_one_for_underconfident_logits(M):
    logits, y = _overconfident(scale=0.3)
    assert M.fit_temperature(logits, y) < 0.5


def test_bayes_threshold(M):
    assert M.bayes_threshold(5.0, 1.0) == pytest.approx(0.8)
    assert M.bayes_threshold(2.0, 1.0) == pytest.approx(0.5)


def test_selective_stats_by_hand(M):
    conf = np.array([0.95, 0.9, 0.7, 0.6, 0.3])
    correct = np.array([True, False, True, False, False])
    s = M.selective_stats(conf, correct, 0.65, cost_misroute=5.0, cost_escalate=1.0)
    assert s["abstain_rate"] == pytest.approx(2 / 5)
    assert s["coverage"] == pytest.approx(3 / 5)
    assert s["error_rate_routed"] == pytest.approx(1 / 3)
    assert s["mean_cost"] == pytest.approx((5.0 + 2 * 1.0) / 5)
    everything = M.selective_stats(conf, correct, 0.0)
    assert everything["abstain_rate"] == 0.0 and everything["mean_cost"] == pytest.approx(3.0)


def test_sweep_finds_the_cheapest_threshold(M):
    conf = np.array([0.95, 0.9, 0.7, 0.6, 0.3])
    correct = np.array([True, True, True, False, False])
    best, rows = M.sweep_threshold(conf, correct, 5.0, 1.0)
    assert 0.6 < best <= 0.7                               # escalate exactly the two wrong ones
    assert min(r["mean_cost"] for r in rows) == pytest.approx(2 / 5)


def test_latency_percentiles(M):
    r = M.latency_percentiles(list(range(1, 101)))
    assert r["p50_ms"] == pytest.approx(50.5)
    assert r["p95_ms"] == pytest.approx(95.05)
