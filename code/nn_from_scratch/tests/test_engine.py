"""Step 1: the scalar autograd engine (engine.py)."""
import math

import pytest


def test_add_and_mul_forward(impl):
    V = impl.engine.Value
    a, b = V(2.0), V(-3.0)
    assert (a + b).data == -1.0
    assert (a * b).data == -6.0
    assert (a + 1).data == 3.0 and (1 + a).data == 3.0
    assert (a * 4).data == 8.0 and (4 * a).data == 8.0


def test_lesson_example(impl):
    V = impl.engine.Value
    w, x, b, y = V(2.0), V(3.0), V(1.0), V(5.0)
    loss = (w * x + b - y) ** 2
    loss.backward()
    assert loss.data == pytest.approx(4.0)
    assert w.grad == pytest.approx(12.0)
    assert b.grad == pytest.approx(4.0)
    assert x.grad == pytest.approx(8.0)


def test_gradients_accumulate_when_a_value_is_reused(impl):
    V = impl.engine.Value
    a = V(3.0)
    out = a * a + a            # d/da = 2a + 1
    out.backward()
    assert a.grad == pytest.approx(7.0)
    b = V(2.0)
    out = b + b + b
    out.backward()
    assert b.grad == pytest.approx(3.0)


def test_pow_and_division(impl):
    V = impl.engine.Value
    a, b = V(4.0), V(2.0)
    out = a / b + a ** 0.5 + 1 / b - (3 - a)
    out.backward()
    assert out.data == pytest.approx(4 / 2 + 2 + 0.5 - (3 - 4))
    assert a.grad == pytest.approx(1 / 2 + 0.5 * 4 ** -0.5 + 1)
    assert b.grad == pytest.approx(-4 / 4 - 1 / 4)


@pytest.mark.parametrize("name,x,expected,dexpected", [
    ("exp", 0.7, math.exp(0.7), math.exp(0.7)),
    ("log", 0.7, math.log(0.7), 1 / 0.7),
    ("tanh", 0.7, math.tanh(0.7), 1 - math.tanh(0.7) ** 2),
    ("relu", 0.7, 0.7, 1.0),
    ("relu", -0.7, 0.0, 0.0),
])
def test_unary_ops(impl, name, x, expected, dexpected):
    a = impl.engine.Value(x)
    out = getattr(a, name)() * 3.0          # times 3: checks that out.grad is used
    out.backward()
    assert out.data == pytest.approx(3 * expected)
    assert a.grad == pytest.approx(3 * dexpected)


def test_compound_expression_matches_finite_differences(impl):
    V = impl.engine.Value

    def f(x, w):
        h = (x * w + 0.3).tanh()
        return (h * x).exp() + (h * h + 1.5).log() - (w * 0.5).relu() ** 2

    x0, w0, eps = 0.5, -2.0, 1e-6
    x, w = V(x0), V(w0)
    f(x, w).backward()
    dx = (f(V(x0 + eps), V(w0)).data - f(V(x0 - eps), V(w0)).data) / (2 * eps)
    dw = (f(V(x0), V(w0 + eps)).data - f(V(x0), V(w0 - eps)).data) / (2 * eps)
    assert x.grad == pytest.approx(dx, rel=1e-6)
    assert w.grad == pytest.approx(dw, rel=1e-6)


def test_diamond_graph_visits_each_node_once(impl):
    # a feeds b and c, which both feed d. A node's _backward must run once,
    # after both of its consumers have delivered their gradient.
    V = impl.engine.Value
    a = V(2.0)
    b, c = a * 3.0, a + 1.0
    d = b * c                  # 3a(a+1) -> derivative 6a + 3 = 15
    e = d * d                  # derivative 2 * d * 15 = 2 * 18 * 15
    e.backward()
    assert a.grad == pytest.approx(540.0)


def test_softmax_cross_entropy_from_primitives(impl):
    V = impl.engine.Value
    logits = [V(1.0), V(2.0), V(0.5)]
    total = sum((v.exp() for v in logits), V(0.0))
    loss = total.log() - logits[1]
    loss.backward()
    z = [math.exp(v.data) for v in logits]
    p = [zi / sum(z) for zi in z]
    assert loss.data == pytest.approx(-math.log(p[1]))
    assert [v.grad for v in logits] == pytest.approx([p[0], p[1] - 1, p[2]])


def test_deep_chain_does_not_hit_the_recursion_limit(impl):
    V = impl.engine.Value
    x = V(1.0)
    total = sum((x * 1.0 for _ in range(5000)), V(0.0))   # a chain 5,000 nodes deep
    total.backward()
    assert x.grad == pytest.approx(5000.0)


def test_scalar_mlp_learns(impl):
    V = impl.engine.Value
    demo = impl.scalar_demo
    model = demo.ScalarMLP([2, 8, 2], seed=0)
    data = [([0.0, 0.0], 0), ([0.0, 1.0], 1), ([1.0, 0.0], 1), ([1.0, 1.0], 0)]   # XOR
    losses = []
    for _ in range(200):
        loss = sum((demo.cross_entropy(model(x), t) for x, t in data), V(0.0)) * 0.25
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        for p in model.parameters():
            p.data -= 0.5 * p.grad
        losses.append(loss.data)
    assert losses[-1] < 0.1 < losses[0]
