"""Step 2: the numpy tensor engine (tensor.py)."""
import numpy as np
import pytest

from helpers import check_grads


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_unbroadcast(impl):
    ub = impl.tensor.unbroadcast
    g = np.arange(12.0).reshape(4, 3)
    np.testing.assert_allclose(ub(g, (4, 3)), g)
    np.testing.assert_allclose(ub(g, (3,)), g.sum(axis=0))
    np.testing.assert_allclose(ub(g, (4, 1)), g.sum(axis=1, keepdims=True))
    np.testing.assert_allclose(ub(g, (1, 3)), g.sum(axis=0, keepdims=True))
    np.testing.assert_allclose(ub(g, ()), g.sum())
    assert ub(g, ()).shape == ()


def test_add_broadcasts_a_bias(impl, rng):
    T = impl.tensor.Tensor
    x, b = T(rng.normal(size=(4, 3))), T(rng.normal(size=3))
    out = x + b
    np.testing.assert_allclose(out.data, x.data + b.data)
    out.sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((4, 3)))
    np.testing.assert_allclose(b.grad, np.full(3, 4.0))     # one copy per row


def test_grad_add_mul_broadcast(impl, rng):
    T = impl.tensor.Tensor
    a, b, c = T(rng.normal(size=(4, 3))), T(rng.normal(size=(4, 1))), T(rng.normal(size=3))
    check_grads(lambda: ((a + b) * c + a * b - c * 2.0 + 1.0).sum(), [a, b, c])


def test_scalar_operands_and_reflected_ops(impl, rng):
    T = impl.tensor.Tensor
    a = T(rng.normal(size=(2, 3)))
    np.testing.assert_allclose((2.0 * a + 1.0).data, 2 * a.data + 1)
    np.testing.assert_allclose((1.0 - a).data, 1 - a.data)
    np.testing.assert_allclose((-a).data, -a.data)
    check_grads(lambda: (3.0 - a * a * 0.5).sum(), [a])


@pytest.mark.parametrize("op", ["exp", "log", "tanh", "relu", "pow3", "pow_half"])
def test_grad_elementwise(impl, rng, op):
    T = impl.tensor.Tensor
    x = T(rng.uniform(0.5, 2.0, size=(3, 4)) * (rng.choice([-1, 1], size=(3, 4)) if op in ("tanh", "relu", "pow3") else 1))
    w = T(rng.normal(size=(3, 4)))                  # so that out.grad is not all ones
    f = {"exp": lambda: x.exp(), "log": lambda: x.log(), "tanh": lambda: x.tanh(),
         "relu": lambda: x.relu(), "pow3": lambda: x ** 3, "pow_half": lambda: x ** 0.5}[op]
    check_grads(lambda: (f() * w).sum(), [x])


def test_relu_forward(impl):
    T = impl.tensor.Tensor
    np.testing.assert_allclose(T([-1.0, 0.0, 2.5]).relu().data, [0.0, 0.0, 2.5])


def test_matmul(impl, rng):
    T = impl.tensor.Tensor
    a, b = T(rng.normal(size=(5, 4))), T(rng.normal(size=(4, 3)))
    w = T(rng.normal(size=(5, 3)))
    np.testing.assert_allclose((a @ b).data, a.data @ b.data)
    check_grads(lambda: ((a @ b) * w).sum(), [a, b])


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, False), (0, True), (1, True)])
def test_sum(impl, rng, axis, keepdims):
    T = impl.tensor.Tensor
    x = T(rng.normal(size=(4, 3)))
    out = x.sum(axis=axis, keepdims=keepdims)
    np.testing.assert_allclose(out.data, x.data.sum(axis=axis, keepdims=keepdims))
    assert out.data.shape == x.data.sum(axis=axis, keepdims=keepdims).shape
    check_grads(lambda: (x.sum(axis=axis, keepdims=keepdims) ** 2).sum(), [x])


def test_mean(impl, rng):
    T = impl.tensor.Tensor
    x = T(rng.normal(size=(4, 3)))
    assert float(x.mean().data) == pytest.approx(x.data.mean())
    check_grads(lambda: (x * x).mean(), [x])


def test_reuse_accumulates(impl, rng):
    T = impl.tensor.Tensor
    x = T(rng.normal(size=(3, 3)))
    (x @ x + x * x + x).sum().backward()
    ones = np.ones((3, 3))
    np.testing.assert_allclose(x.grad, ones @ x.data.T + x.data.T @ ones + 2 * x.data + 1)


def test_tensor_agrees_with_value(impl):
    V, T = impl.engine.Value, impl.tensor.Tensor
    a, b = V(0.3), V(-1.2)
    ((a * b + a).tanh() * b + (a * a).exp()).backward()
    ta, tb = T(0.3), T(-1.2)
    ((ta * tb + ta).tanh() * tb + (ta * ta).exp()).backward()
    assert float(ta.grad) == pytest.approx(a.grad)
    assert float(tb.grad) == pytest.approx(b.grad)


def test_cross_entropy_value(impl, rng):
    T, ce = impl.tensor.Tensor, impl.tensor.cross_entropy
    logits, y = rng.normal(size=(6, 5)), rng.integers(0, 5, size=6)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    loss = ce(T(logits), y)
    assert loss.data.shape == ()
    assert float(loss.data) == pytest.approx(-np.log(p[np.arange(6), y]).mean())
    assert float(ce(T(np.zeros((4, 10))), [0, 1, 2, 3]).data) == pytest.approx(np.log(10))


def test_cross_entropy_gradient(impl, rng):
    T, ce = impl.tensor.Tensor, impl.tensor.cross_entropy
    logits, y = T(rng.normal(size=(6, 5))), rng.integers(0, 5, size=6)
    ce(logits, y).backward()
    p = np.exp(logits.data) / np.exp(logits.data).sum(axis=1, keepdims=True)
    p[np.arange(6), y] -= 1
    np.testing.assert_allclose(logits.grad, p / 6, atol=1e-12)
    check_grads(lambda: ce(logits, y) * 2.5, [logits])        # out.grad must be used


def test_cross_entropy_is_stable_for_large_logits(impl):
    T, ce = impl.tensor.Tensor, impl.tensor.cross_entropy
    logits = T([[1000.0, 0.0, -1000.0], [0.0, 1000.0, 0.0]])
    loss = ce(logits, [0, 0])
    loss.backward()
    assert float(loss.data) == pytest.approx(500.0)
    assert np.isfinite(logits.grad).all()


def test_cross_entropy_matches_composition_of_primitives(impl, rng):
    T, ce = impl.tensor.Tensor, impl.tensor.cross_entropy
    data, y = rng.normal(size=(6, 5)), rng.integers(0, 5, size=6)
    onehot = np.eye(5)[y]
    a = T(data)
    fused = ce(a, y)
    fused.backward()
    b = T(data)
    log_z = b.exp().sum(axis=1, keepdims=True).log()
    composed = ((log_z - b) * onehot).sum() * (1.0 / 6)
    composed.backward()
    assert float(fused.data) == pytest.approx(float(composed.data))
    np.testing.assert_allclose(a.grad, b.grad, atol=1e-12)
