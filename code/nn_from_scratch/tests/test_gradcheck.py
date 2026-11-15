"""Step 5: the gradient-check harness (gradcheck.py)."""
import numpy as np
import pytest


def test_numerical_grad_of_a_quadratic(impl):
    x = np.array([[1.0, -2.0, 3.0], [0.5, 0.0, -1.5]])
    before = x.copy()
    g = impl.gradcheck.numerical_grad(lambda: float((x ** 2).sum() + x[0, 0] * x[1, 2]), x)
    expected = 2 * before
    expected[0, 0] += before[1, 2]
    expected[1, 2] += before[0, 0]
    np.testing.assert_allclose(g, expected, atol=1e-7)
    np.testing.assert_array_equal(x, before)                    # x is restored


def test_rel_error(impl):
    re = impl.gradcheck.rel_error
    a = np.array([3.0, 4.0])
    assert re(a, a) == 0.0
    assert re(a, -a) == pytest.approx(1.0)
    assert re(a, a * (1 + 1e-6)) == pytest.approx(5e-7, rel=1e-3)
    assert re(np.zeros(3), np.zeros(3)) == 0.0                  # dead units: 0 vs 0 is a match


def test_gradcheck_passes_on_a_correct_mlp(impl):
    rng = np.random.default_rng(0)
    model = impl.nn.make_mlp([10, 8, 4], rng)
    x, y = rng.normal(size=(6, 10)), rng.integers(0, 4, size=6)
    params = model.parameters()
    err = impl.gradcheck.gradcheck(
        lambda: impl.tensor.cross_entropy(model(x), y) + impl.nn.l2_penalty(params, 0.01), params)
    assert 0.0 <= err < 1e-6


def test_gradcheck_catches_a_wrong_backward(impl):
    T = impl.tensor.Tensor
    rng = np.random.default_rng(0)
    w, x = T(rng.normal(size=(4, 3))), rng.normal(size=(5, 4))

    def buggy_square(t):
        out = T(t.data ** 2, (t,), "buggy_square")

        def _backward():
            t.grad += t.data * out.grad                         # the factor 2 is missing
        out._backward = _backward
        return out

    assert impl.gradcheck.gradcheck(lambda: buggy_square(T(x) @ w).sum(), [w]) > 0.1
    assert impl.gradcheck.gradcheck(lambda: ((T(x) @ w) ** 2).sum(), [w]) < 1e-6


def test_gradcheck_leaves_parameters_unchanged(impl):
    rng = np.random.default_rng(0)
    model = impl.nn.make_mlp([5, 4, 3], rng)
    before = [p.data.copy() for p in model.parameters()]
    x, y = rng.normal(size=(4, 5)), rng.integers(0, 3, size=4)
    impl.gradcheck.gradcheck(lambda: impl.tensor.cross_entropy(model(x), y), model.parameters())
    for b, p in zip(before, model.parameters()):
        np.testing.assert_array_equal(b, p.data)
