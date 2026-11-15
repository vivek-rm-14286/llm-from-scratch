"""Step 6: mini-batches and the training loop (data.py, train.py)."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def digits(impl):
    return impl.data.load_digits_data(seed=0)


def test_batches_cover_the_data_once(impl):
    x, y = np.arange(20).reshape(10, 2), np.arange(10)
    out = list(impl.data.batches(x, y, 4))
    assert [len(xb) for xb, _ in out] == [4, 4, 2]              # the last batch is smaller
    np.testing.assert_array_equal(np.concatenate([yb for _, yb in out]), y)   # no rng: original order
    np.testing.assert_array_equal(np.concatenate([xb for xb, _ in out]), x)


def test_batches_shuffle_with_rng(impl):
    x, y = np.arange(200).reshape(100, 2), np.arange(100)
    rng = np.random.default_rng(0)
    first = np.concatenate([yb for _, yb in impl.data.batches(x, y, 16, rng)])
    second = np.concatenate([yb for _, yb in impl.data.batches(x, y, 16, rng)])
    assert sorted(first) == list(range(100)) and sorted(second) == list(range(100))
    assert list(first) != list(range(100))
    assert list(first) != list(second)                          # a new order every epoch
    again = np.concatenate([yb for _, yb in impl.data.batches(x, y, 16, np.random.default_rng(0))])
    np.testing.assert_array_equal(first, again)                 # same seed, same order
    for xb, yb in impl.data.batches(x, y, 16, np.random.default_rng(1)):
        np.testing.assert_array_equal(xb[:, 0], 2 * yb)         # x and y stay aligned


def test_initial_loss_is_near_ln10(impl, digits):
    model = impl.nn.make_mlp([64, 64, 10], np.random.default_rng(0))
    loss, acc = impl.train.evaluate(model, digits["x_val"], digits["y_val"])
    assert 1.8 < loss < 3.2          # ln(10) = 2.30 for a model that knows nothing
    assert acc < 0.4


def test_evaluate_weights_batches_by_size_and_changes_nothing(impl, digits):
    model = impl.nn.make_mlp([64, 16, 10], np.random.default_rng(0))
    before = [p.data.copy() for p in model.parameters()]
    x, y = digits["x_val"][:50], digits["y_val"][:50]
    loss, acc = impl.train.evaluate(model, x, y, batch_size=16)          # batches of 16, 16, 16, 2
    logits = model(x)
    assert loss == pytest.approx(float(impl.tensor.cross_entropy(logits, y).data))
    assert acc == pytest.approx(impl.nn.accuracy(logits, y))
    for b, p in zip(before, model.parameters()):
        np.testing.assert_array_equal(b, p.data)


def test_train_epoch_reduces_the_loss(impl, digits):
    rng = np.random.default_rng(0)
    model = impl.nn.make_mlp([64, 32, 10], rng)
    opt = impl.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    x, y = digits["x_train"], digits["y_train"]
    first = impl.train.train_epoch(model, opt, x, y, 32, rng)
    second = impl.train.train_epoch(model, opt, x, y, 32, rng)
    assert isinstance(first, float)
    assert second < first < 2.5
    assert second < 0.5


def test_overfits_a_single_batch(impl, digits):
    rng = np.random.default_rng(0)
    model = impl.nn.make_mlp([64, 64, 10], rng)
    opt = impl.optim.Adam(model.parameters(), lr=0.01)
    x, y = digits["x_train"][:32], digits["y_train"][:32]
    for _ in range(100):
        loss = impl.train.train_epoch(model, opt, x, y, 32, rng)
    assert loss < 0.01


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_fit_reaches_95_percent_on_digits(impl, digits, optimizer):
    model, history = impl.train.fit(digits, hidden=(64,), epochs=30, optimizer=optimizer, seed=0, verbose=False)
    assert len(history) == 30
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    _, test_acc = impl.train.evaluate(model, digits["x_test"], digits["y_test"])
    assert test_acc >= 0.95


def test_weight_decay_shrinks_the_weights(impl, digits):
    norms = []
    for wd in (0.0, 0.01):
        model, _ = impl.train.fit(digits, epochs=10, optimizer="sgd", weight_decay=wd, seed=0, verbose=False)
        norms.append(sum(float((p.data ** 2).sum()) for p in model.parameters()))
    assert norms[1] < 0.8 * norms[0]
