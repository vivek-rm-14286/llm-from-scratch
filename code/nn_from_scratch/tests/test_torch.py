"""Step 7: parity with PyTorch (compare_torch.py). Skipped when torch is not installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_to_torch_gives_the_same_logits(impl):
    rng = np.random.default_rng(0)
    model = impl.nn.make_mlp([64, 32, 16, 10], rng)
    for p in model.parameters():
        p.data += rng.normal(scale=0.1, size=p.shape)
    twin = impl.compare_torch.to_torch(model)
    x = rng.normal(size=(5, 64))
    np.testing.assert_allclose(model(x).data, twin(torch.from_numpy(x)).detach().numpy(), atol=1e-12)


def test_gradients_match_autograd(impl):
    rng = np.random.default_rng(0)
    model = impl.nn.make_mlp([64, 32, 10], rng)
    twin = impl.compare_torch.to_torch(model)
    x, y = rng.normal(size=(8, 64)), rng.integers(0, 10, size=8)
    impl.tensor.cross_entropy(model(x), y).backward()
    torch.nn.functional.cross_entropy(twin(torch.from_numpy(x)), torch.from_numpy(y)).backward()
    for p, tp in zip(model.parameters(), twin.parameters()):
        t = tp.grad.numpy()
        np.testing.assert_allclose(p.grad, t.T if t.ndim == 2 else t, atol=1e-12)


@pytest.mark.parametrize("optimizer,wd", [("sgd", 1e-4), ("adam", 0.0), ("adam", 1e-4), ("adamw", 1e-2)])
def test_loss_curves_match(impl, optimizer, wd):
    ours, theirs, gap = impl.compare_torch.run_both(optimizer, steps=100, weight_decay=wd)
    assert len(ours) == len(theirs) == 100
    assert ours[-1] < ours[0]
    assert np.abs(ours - theirs).max() < 1e-8
    assert gap < 1e-8
