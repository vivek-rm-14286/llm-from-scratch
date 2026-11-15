"""Step 3: layers, initialization and the L2 penalty (nn.py)."""
import numpy as np
import pytest

from helpers import check_grads


def test_linear_forward_and_shapes(impl):
    lin = impl.nn.Linear(5, 3, np.random.default_rng(0))
    assert lin.W.shape == (5, 3) and lin.b.shape == (3,)
    assert [p.shape for p in lin.parameters()] == [(5, 3), (3,)]
    lin.b.data[:] = [1.0, 2.0, 3.0]
    x = np.random.default_rng(1).normal(size=(4, 5))
    np.testing.assert_allclose(lin(x).data, x @ lin.W.data + lin.b.data)


def test_he_init(impl):
    lin = impl.nn.Linear(400, 300, np.random.default_rng(0), init="he")
    assert lin.W.data.std() == pytest.approx(np.sqrt(2 / 400), rel=0.03)
    assert abs(lin.W.data.mean()) < 0.01
    assert np.abs(lin.W.data).max() > 3 * np.sqrt(2 / 400)      # normal, not uniform
    np.testing.assert_array_equal(lin.b.data, np.zeros(300))


def test_xavier_init(impl):
    lin = impl.nn.Linear(400, 300, np.random.default_rng(0), init="xavier")
    a = np.sqrt(6 / 700)
    assert np.abs(lin.W.data).max() <= a
    assert lin.W.data.std() == pytest.approx(a / np.sqrt(3), rel=0.03)   # uniform(-a, a)
    np.testing.assert_array_equal(lin.b.data, np.zeros(300))


def test_unknown_init_raises(impl):
    with pytest.raises(ValueError):
        impl.nn.Linear(4, 3, np.random.default_rng(0), init="zeros")


def test_init_is_reproducible(impl):
    a = impl.nn.make_mlp([8, 6, 4], np.random.default_rng(7))
    b = impl.nn.make_mlp([8, 6, 4], np.random.default_rng(7))
    c = impl.nn.make_mlp([8, 6, 4], np.random.default_rng(8))
    assert all(np.array_equal(p.data, q.data) for p, q in zip(a.parameters(), b.parameters()))
    assert not np.array_equal(a.parameters()[0].data, c.parameters()[0].data)


def test_make_mlp_structure(impl):
    nn = impl.nn
    model = nn.make_mlp([64, 64, 10], np.random.default_rng(0))
    kinds = [type(layer).__name__ for layer in model.layers]
    assert kinds == ["Linear", "ReLU", "Linear"]               # no activation on the logits
    assert sum(p.data.size for p in model.parameters()) == 64 * 64 + 64 + 64 * 10 + 10
    deep = nn.make_mlp([64, 32, 16, 10], np.random.default_rng(0), activation="tanh")
    assert [type(layer).__name__ for layer in deep.layers] == ["Linear", "Tanh", "Linear", "Tanh", "Linear"]
    assert deep(np.zeros((7, 64))).shape == (7, 10)


def test_mlp_forward_matches_numpy(impl):
    model = impl.nn.make_mlp([6, 5, 3], np.random.default_rng(0))
    w1, b1, w2, b2 = [p.data for p in model.parameters()]
    b1[:] = 0.1
    x = np.random.default_rng(1).normal(size=(4, 6))
    np.testing.assert_allclose(model(x).data, np.maximum(0, x @ w1 + b1) @ w2 + b2)


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_mlp_gradients(impl, activation):
    rng = np.random.default_rng(0)
    model = impl.nn.make_mlp([6, 5, 4, 3], rng, activation=activation)
    for p in model.parameters():
        p.data += rng.normal(scale=0.1, size=p.shape)           # non-zero biases
    x, y = rng.normal(size=(8, 6)), rng.integers(0, 3, size=8)
    check_grads(lambda: impl.tensor.cross_entropy(model(x), y), model.parameters())


def test_zero_grad(impl):
    model = impl.nn.make_mlp([6, 5, 3], np.random.default_rng(0))
    impl.tensor.cross_entropy(model(np.ones((2, 6))), [0, 1]).backward()
    assert any(np.abs(p.grad).sum() > 0 for p in model.parameters())
    model.zero_grad()
    assert all(np.abs(p.grad).sum() == 0 for p in model.parameters())
    assert all(p.grad.shape == p.data.shape for p in model.parameters())


def test_l2_penalty(impl):
    T = impl.tensor.Tensor
    a, b = T([[1.0, -2.0], [3.0, 0.5]]), T([2.0, -1.0])
    pen = impl.nn.l2_penalty([a, b], lam=0.1)
    assert float(pen.data) == pytest.approx(0.05 * (1 + 4 + 9 + 0.25 + 4 + 1))
    pen.backward()
    np.testing.assert_allclose(a.grad, 0.1 * a.data)            # d/dw (lam/2) w^2 = lam * w
    np.testing.assert_allclose(b.grad, 0.1 * b.data)


def test_accuracy(impl):
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    assert impl.nn.accuracy(logits, [1, 0, 0, 0]) == pytest.approx(0.75)
    assert impl.nn.accuracy(impl.tensor.Tensor(logits), np.array([1, 0, 1, 0])) == pytest.approx(1.0)
