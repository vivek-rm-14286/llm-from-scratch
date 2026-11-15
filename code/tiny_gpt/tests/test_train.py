"""Steps 4 and 5: the data pipeline and the training loop."""
import json
import math
import os

import numpy as np
import pytest
import torch

TEXT = ("ROMEO:\nBut, soft! what light through yonder window breaks?\n"
        "It is the east, and Juliet is the sun.\n\nJULIET:\nO Romeo, Romeo! wherefore art thou Romeo?\n\n") * 40


# ------------------------------------------------------------------ data
def test_prepare_writes_token_files(tmp_path):
    import bpe
    import data
    src = tmp_path / "corpus.txt"
    src.write_text(TEXT, encoding="utf-8")
    out = str(tmp_path / "prepared")
    meta = data.prepare(str(src), out, vocab_size=300)
    for name in ("tokenizer.json", "train.bin", "val.bin", "meta.json"):
        assert os.path.exists(os.path.join(out, name)), f"{name} was not written"
    assert json.load(open(os.path.join(out, "meta.json")))["vocab_size"] == 300 == meta["vocab_size"]

    train, val = data.load_tokens(os.path.join(out, "train.bin")), data.load_tokens(os.path.join(out, "val.bin"))
    assert isinstance(train, np.memmap) and train.dtype == np.uint16
    assert len(train) == meta["train_tokens"] and len(val) == meta["val_tokens"]
    assert 0.05 < len(val) / (len(train) + len(val)) < 0.15          # about 10% held out
    assert int(train.max()) < 300

    tok = bpe.ByteBPE.load(os.path.join(out, "tokenizer.json"))
    ids = np.concatenate([train, val]).tolist()
    assert tok.decode(ids) == TEXT                                   # the files hold the whole corpus, in order
    assert tok.special["<|endoftext|>"] == 299                       # the last id is reserved for EOT


def test_get_batch_shapes_and_shift():
    import data
    tokens = np.arange(1000, dtype=np.uint16)
    g = torch.Generator().manual_seed(0)
    x, y = data.get_batch(tokens, batch_size=8, block_size=16, generator=g)
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == y.dtype == torch.int64
    assert (x[:, 1:] == y[:, :-1]).all(), "targets are not the inputs shifted by one"
    assert (y[:, -1] == x[:, -1] + 1).all()
    assert len(set(x[:, 0].tolist())) > 1, "every window starts at the same place"
    g2 = torch.Generator().manual_seed(0)
    x2, _ = data.get_batch(tokens, batch_size=8, block_size=16, generator=g2)
    assert (x == x2).all(), "the same generator state must give the same batch"


def test_get_batch_can_reach_the_end():
    import data
    tokens = np.arange(20, dtype=np.uint16)
    g = torch.Generator().manual_seed(0)
    last = max(int(data.get_batch(tokens, 4, 16, generator=g)[1].max()) for _ in range(50))
    assert last <= 19


# ------------------------------------------------------------------ schedule and optimizer
def test_lr_schedule():
    import train
    lr = lambda s: train.lr_at(s, max_lr=1e-3, warmup_steps=10, max_steps=110, min_lr=1e-4)
    assert lr(0) == pytest.approx(1e-4)                 # (0 + 1) / 10 of the peak: never exactly zero
    assert lr(4) == pytest.approx(5e-4)
    assert lr(9) == pytest.approx(1e-3)                 # end of warmup = peak
    assert lr(10) == pytest.approx(1e-3)
    assert lr(60) == pytest.approx(5.5e-4)              # halfway through the cosine
    assert lr(110) == pytest.approx(1e-4)
    assert lr(500) == pytest.approx(1e-4)               # stays at min_lr afterwards
    values = [lr(s) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(values, values[1:])), "the cosine part must not increase"


def test_optimizer_groups():
    import model as M
    import train
    net = M.GPT(M.GPTConfig(vocab_size=50, block_size=8, n_layer=1, n_head=2, d_model=16))
    opt = train.configure_optimizer(net, lr=1e-3, weight_decay=0.1)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["betas"] == (0.9, 0.95)
    seen = 0
    for group in opt.param_groups:
        for p in group["params"]:
            seen += p.numel()
            expected = 0.1 if p.dim() >= 2 else 0.0     # matrices decay; biases and norm gains do not
            assert group["weight_decay"] == expected
    assert seen == net.num_params(), "every parameter must be in exactly one group"


def test_estimate_loss_is_eval_mode_and_restores_train_mode():
    import model as M
    import train
    torch.manual_seed(0)
    net = M.GPT(M.GPTConfig(vocab_size=50, block_size=8, n_layer=1, n_head=2, d_model=16, dropout=0.5)).train()
    tokens = (np.arange(500) % 50).astype(np.uint16)
    a = train.estimate_loss(net, tokens, 4, 8, "cpu", iters=3, generator=torch.Generator().manual_seed(0))
    b = train.estimate_loss(net, tokens, 4, 8, "cpu", iters=3, generator=torch.Generator().manual_seed(0))
    assert isinstance(a, float) and a == pytest.approx(b), "dropout must be off while estimating the loss"
    assert abs(a - math.log(50)) < 0.3
    assert net.training, "estimate_loss must put the model back in train mode"


# ------------------------------------------------------------------ checkpoint / resume
def _steps(net, opt, tokens, gen, n):
    import data
    losses = []
    for _ in range(n):
        x, y = data.get_batch(tokens, 4, 8, generator=gen)
        loss = net(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_checkpoint_resume_is_exact(tmp_path):
    import model as M
    import train
    tokens = (np.arange(2000) * 7 % 50).astype(np.uint16)
    cfg = M.GPTConfig.modern(vocab_size=50, block_size=8, n_layer=2, n_head=2, d_model=16)
    torch.manual_seed(0)
    net = M.GPT(cfg)
    opt = train.configure_optimizer(net, lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    _steps(net, opt, tokens, gen, 5)
    path = str(tmp_path / "ckpt.pt")
    train.save_checkpoint(path, net, opt, step=5, best_val=1.25, generator=gen)
    assert not os.path.exists(path + ".tmp")
    uninterrupted = _steps(net, opt, tokens, gen, 5)

    net2, ckpt = train.load_checkpoint(path)
    assert ckpt["step"] == 5 and ckpt["best_val"] == 1.25
    assert net2.cfg == cfg
    opt2 = train.configure_optimizer(net2, lr=1e-2)
    opt2.load_state_dict(ckpt["optimizer"])
    gen2 = torch.Generator()
    gen2.set_state(ckpt["rng"])
    resumed = _steps(net2.train(), opt2, tokens, gen2, 5)
    assert resumed == pytest.approx(uninterrupted, abs=1e-5), "the resumed run diverged from the uninterrupted one"
