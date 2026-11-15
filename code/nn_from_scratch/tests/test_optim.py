"""Step 4: SGD with momentum, Adam, weight decay (optim.py)."""
import numpy as np
import pytest


def param(impl, values, grad):
    p = impl.tensor.Tensor(np.array(values, dtype=float))
    p.grad = np.array(grad, dtype=float)
    return p


def test_sgd_plain_step(impl):
    p = param(impl, [1.0, -2.0], [0.5, -1.0])
    impl.optim.SGD([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, [0.95, -1.9])


def test_sgd_updates_the_models_own_tensors(impl):
    model = impl.nn.make_mlp([4, 3, 2], np.random.default_rng(0))
    before = [p.data.copy() for p in model.parameters()]
    opt = impl.optim.SGD(model.parameters(), lr=0.1)
    impl.tensor.cross_entropy(model(np.ones((3, 4))), [0, 1, 1]).backward()
    opt.step()
    assert any(not np.array_equal(b, p.data) for b, p in zip(before, model.parameters()))


def test_sgd_momentum(impl):
    p = param(impl, [1.0], [1.0])
    opt = impl.optim.SGD([p], lr=0.1, momentum=0.9)
    opt.step()                       # v = 1          -> p = 0.9
    np.testing.assert_allclose(p.data, [0.9])
    p.grad = np.array([1.0])
    opt.step()                       # v = 0.9 + 1    -> p = 0.9 - 0.19
    np.testing.assert_allclose(p.data, [0.71])
    p.grad = np.array([-2.0])
    opt.step()                       # v = 1.71 - 2   -> p = 0.71 + 0.029
    np.testing.assert_allclose(p.data, [0.739])


def test_sgd_weight_decay_equals_l2_penalty_in_the_loss(impl):
    T = impl.tensor.Tensor
    rng = np.random.default_rng(0)
    w0, x = rng.normal(size=(4, 3)), rng.normal(size=(5, 4))
    lam = 0.1
    a, b = T(w0.copy()), T(w0.copy())
    opt_a = impl.optim.SGD([a], lr=0.05, momentum=0.9, weight_decay=lam)
    opt_b = impl.optim.SGD([b], lr=0.05, momentum=0.9)
    for _ in range(5):
        opt_a.zero_grad()
        ((x @ a) ** 2).mean().backward()
        opt_a.step()
        opt_b.zero_grad()
        (((x @ b) ** 2).mean() + impl.nn.l2_penalty([b], lam)).backward()
        opt_b.step()
    np.testing.assert_allclose(a.data, b.data, atol=1e-12)


def test_zero_grad(impl):
    p = param(impl, [[1.0, 2.0]], [[3.0, 4.0]])
    opt = impl.optim.SGD([p], lr=0.1, momentum=0.9)
    opt.zero_grad()
    np.testing.assert_array_equal(p.grad, np.zeros((1, 2)))
    opt.step()                                   # zero gradient, empty velocity: no movement
    np.testing.assert_array_equal(p.data, [[1.0, 2.0]])


def test_adam_first_step_has_size_lr(impl):
    # After bias correction, m_hat = g and v_hat = g^2, so the step is lr * sign(g).
    p = param(impl, [1.0, 1.0, 1.0], [1e-3, -5.0, 200.0])
    impl.optim.Adam([p], lr=0.01).step()
    np.testing.assert_allclose(p.data, [0.99, 1.01, 0.99], atol=1e-6)


def adam_reference(w, grads, lr, b1, b2, eps, wd=0.0, decoupled=False):
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t, g in enumerate(grads, start=1):
        if wd and decoupled:
            w = w * (1 - lr * wd)
        elif wd:
            g = g + wd * w
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w = w - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return w


@pytest.mark.parametrize("wd,decoupled", [(0.0, False), (0.1, False), (0.1, True)])
def test_adam_matches_the_update_equations(impl, wd, decoupled):
    rng = np.random.default_rng(0)
    w0 = rng.normal(size=(3, 2))
    grads = [rng.normal(size=(3, 2)) for _ in range(6)]
    p = impl.tensor.Tensor(w0.copy())
    opt = impl.optim.Adam([p], lr=0.01, betas=(0.8, 0.9), eps=1e-8, weight_decay=wd, decoupled=decoupled)
    for g in grads:
        p.grad = g.copy()
        opt.step()
    np.testing.assert_allclose(p.data, adam_reference(w0, grads, 0.01, 0.8, 0.9, 1e-8, wd, decoupled), atol=1e-12)


def test_l2_and_decoupled_weight_decay_differ_in_adam(impl):
    a, b = param(impl, [1.0, -1.0], [0.3, 0.3]), param(impl, [1.0, -1.0], [0.3, 0.3])
    impl.optim.Adam([a], lr=0.01, weight_decay=0.5).step()
    impl.optim.Adam([b], lr=0.01, weight_decay=0.5, decoupled=True).step()
    assert not np.allclose(a.data, b.data)


@pytest.mark.parametrize("name,kwargs", [("SGD", dict(lr=0.1, momentum=0.9)), ("Adam", dict(lr=0.1))])
def test_optimizers_minimize_a_quadratic(impl, name, kwargs):
    T = impl.tensor.Tensor
    w, target = T(np.array([5.0, -3.0])), np.array([1.0, 2.0])
    opt = getattr(impl.optim, name)([w], **kwargs)
    for _ in range(300):
        opt.zero_grad()
        ((w - target) ** 2).sum().backward()
        opt.step()
    np.testing.assert_allclose(w.data, target, atol=1e-3)
