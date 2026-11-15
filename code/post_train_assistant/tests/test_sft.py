import math

import pytest
import torch


def hand_masked_loss(model, x, y):
    """Mean of -log p over labeled positions, with explicit loops."""
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=x).logits.double(), dim=-1)
    total, count = 0.0, 0
    for b in range(x.size(0)):
        for t in range(1, x.size(1)):
            if y[b, t] != -100:
                total -= logp[b, t - 1, y[b, t]].item()      # position t-1 predicts token t
                count += 1
    return total / count


def make_batch(impl, fns, conv):
    encode, im_start, im_end, nl, pad = fns
    short = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there."}]
    data = [impl.data.render_chatml(c, encode, im_start, im_end, nl) for c in (conv, short)]
    return data, impl.data.collate(data, pad, 128)


def test_sft_loss_equals_hand_computation(impl, tiny_model, fns, conv):
    batch = make_batch(impl, fns, conv)
    _, (x, y) = batch
    with torch.no_grad():
        loss = impl.sft.sft_loss(tiny_model, x, y).item()
    assert loss == pytest.approx(hand_masked_loss(tiny_model, x, y), abs=1e-4)
    assert 3.0 < loss < 6.0                                   # near ln(vocab) for a random model


def test_sft_loss_counts_only_labeled_tokens(impl, tiny_model, fns, conv):
    batch = make_batch(impl, fns, conv)
    data, (x, y) = batch
    n_targets = int((y != -100).sum())
    assert n_targets == 5 + 13                                # 4,<end>,1,7,<end> + "Hello there."<end>
    with torch.no_grad():
        per_example = [impl.sft.sft_loss(tiny_model, *impl.data.collate([d], 0, 128)).item() for d in data]
        both = impl.sft.sft_loss(tiny_model, x, y).item()
    assert both == pytest.approx((5 * per_example[0] + 13 * per_example[1]) / 18, abs=1e-4)


def test_sft_loss_has_gradient(impl, tiny_model, fns, conv):
    batch = make_batch(impl, fns, conv)
    _, (x, y) = batch
    loss = impl.sft.sft_loss(tiny_model, x, y)
    loss.backward()
    assert tiny_model.get_input_embeddings().weight.grad is not None


def test_lr_schedule(impl):
    kw = dict(base_lr=1e-3, warmup_steps=10, total_steps=110)
    lr = lambda s: impl.sft.lr_at(s, **kw)
    assert lr(0) == pytest.approx(1e-4)
    assert lr(4) == pytest.approx(5e-4)
    assert lr(9) == pytest.approx(1e-3)
    assert lr(59) == pytest.approx(1e-3 * (0.1 + 0.9 * 0.5))    # halfway through the cosine
    assert lr(109) == pytest.approx(1e-4)                       # min_ratio 0.1
    assert lr(500) == pytest.approx(1e-4)
    assert all(lr(s) >= lr(s + 1) for s in range(9, 109))


def test_eval_loss_is_token_weighted_and_restores_mode(impl, tiny_model, fns, conv):
    batch = make_batch(impl, fns, conv)
    data, (x, y) = batch
    tiny_model.train()
    got = impl.sft.eval_loss(tiny_model, data, pad_id=0, max_len=128, batch_size=1)
    assert tiny_model.training
    tiny_model.eval()
    assert got == pytest.approx(hand_masked_loss(tiny_model, x, y), abs=1e-4)


def test_train_sft_learns_addition(impl, tiny_model, fns):
    encode, im_start, im_end, nl, pad = fns
    convs = [[{"role": "user", "content": f"{a}+{b}=?"}, {"role": "assistant", "content": str(a + b)}]
             for a in range(6) for b in range(6)]
    data = [impl.data.render_chatml(c, encode, im_start, im_end, nl) for c in convs]
    logs = []
    hist = impl.sft.train_sft(tiny_model, data[:30], data[30:], pad_id=pad, max_len=64, batch_size=6,
                              epochs=12, lr=3e-3, warmup_steps=5, eval_every=20, device="cpu",
                              log=logs.append)
    assert hist[0]["step"] == 0 and hist[-1]["step"] == 60 and len(hist) == 4
    assert {"step", "train_loss", "eval_loss", "lr", "seconds"} <= set(hist[-1])
    assert hist[0]["eval_loss"] > 3.0
    assert hist[-1]["eval_loss"] < 0.6 * hist[0]["eval_loss"]
    assert math.isfinite(hist[-1]["train_loss"]) and logs
