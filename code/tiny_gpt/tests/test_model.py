"""Steps 2, 3 and 6: the transformer, the sanity checks, and the modern upgrade.

Every test is parametrized over two architectures. The "gpt2" cases pass after
step 2; the "modern" cases pass after step 6.
"""
import math

import pytest
import torch

import model as M

ARCHS = ["gpt2", "modern"]


def small_cfg(arch, **kw):
    base = dict(vocab_size=97, block_size=32, n_layer=2, n_head=4, d_model=64)
    base.update(kw)
    return M.GPTConfig.modern(**base) if arch == "modern" else M.GPTConfig(**base)


def build(arch, seed=0, **kw):
    torch.manual_seed(seed)
    return M.GPT(small_cfg(arch, **kw)).eval()


def batch(cfg, B=4, T=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(cfg.vocab_size, (B, T or cfg.block_size), generator=g)


# ------------------------------------------------------------------ parameter count
def formula(cfg):
    """Parameter count derived by hand (see step 2 of the project page)."""
    d, L, V, T = cfg.d_model, cfg.n_layer, cfg.vocab_size, cfg.block_size
    if cfg.rmsnorm and cfg.rope and cfg.swiglu and not cfg.bias:
        h = 8 * math.ceil(8 * d / 3 / 8)
        n = V * d + L * (4 * d * d + 3 * d * h + 2 * d) + d
    else:
        n = V * d + T * d + L * (12 * d * d + 13 * d) + 2 * d
    return n if cfg.tie_weights else n + V * d


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("dims", [dict(), dict(d_model=96, n_layer=3, n_head=6, vocab_size=300, block_size=48)])
def test_param_count_matches_formula(arch, dims):
    cfg = small_cfg(arch, **dims)
    assert M.GPT(cfg).num_params() == formula(cfg)
    assert M.count_params(cfg) == formula(cfg)


def test_param_count_default_configs():
    assert M.count_params(M.GPTConfig()) == 11_532_288
    assert M.count_params(M.GPTConfig.modern()) == 11_408_256


@pytest.mark.parametrize("arch", ARCHS)
def test_weight_tying(arch):
    tied = build(arch)
    assert tied.lm_head.weight is tied.tok_emb.weight
    untied = build(arch, tie_weights=False)
    assert untied.lm_head.weight is not untied.tok_emb.weight
    cfg = untied.cfg
    assert untied.num_params() - tied.num_params() == cfg.vocab_size * cfg.d_model


def test_modern_has_no_position_table_or_biases():
    model = build("modern")
    names = [n for n, _ in model.named_parameters()]
    assert not any("pos_emb" in n for n in names)
    assert not any(n.endswith("bias") for n in names)


# ------------------------------------------------------------------ forward interface
@pytest.mark.parametrize("arch", ARCHS)
def test_forward_shapes(arch):
    model = build(arch)
    x = batch(model.cfg)
    logits = model(x)
    assert logits.shape == (4, model.cfg.block_size, model.cfg.vocab_size)
    logits2, loss = model(x, x)
    assert logits2.shape == logits.shape and loss.dim() == 0
    assert model(x[:, :5]).shape == (4, 5, model.cfg.vocab_size)      # shorter than block_size is fine


# ------------------------------------------------------------------ sanity checks
@pytest.mark.parametrize("arch", ARCHS)
def test_step0_loss_is_ln_vocab(arch):
    model = build(arch, vocab_size=2048, d_model=128)
    x = batch(model.cfg, B=8)
    y = batch(model.cfg, B=8, seed=2)
    with torch.no_grad():
        loss = model(x, y)[1].item()
    assert abs(loss - math.log(2048)) < 0.15, f"step-0 loss {loss:.3f}, expected about {math.log(2048):.3f}"


@pytest.mark.parametrize("arch", ARCHS)
def test_causality(arch):
    model = build(arch)
    x = batch(model.cfg)
    t = 10
    x2 = x.clone()
    x2[:, t] = (x2[:, t] + 1) % model.cfg.vocab_size
    with torch.no_grad():
        a, b = model(x), model(x2)
    leak = (a[:, :t] - b[:, :t]).abs().max().item()
    assert leak < 1e-5, f"changing token {t} moved earlier logits by {leak:.2e}: the causal mask leaks"
    assert (a[:, t:] - b[:, t:]).abs().max().item() > 1e-4, "later positions ignore the changed token"


@pytest.mark.parametrize("arch", ARCHS)
def test_position_matters(arch):
    """Without position information a causal transformer cannot tell 'a b' from 'b a' at the last position
    in general, and the last-position logits for [a, b, c] and [b, a, c] would be identical."""
    model = build(arch)
    x = torch.tensor([[5, 9, 7]])
    x2 = torch.tensor([[9, 5, 7]])
    with torch.no_grad():
        diff = (model(x)[:, -1] - model(x2)[:, -1]).abs().max().item()
    assert diff > 1e-5, "the model ignores token order"


@pytest.mark.parametrize("arch", ARCHS)
def test_overfit_one_batch(arch):
    import train as T
    torch.manual_seed(0)
    model = M.GPT(small_cfg(arch)).train()
    x = batch(model.cfg)
    y = torch.roll(x, -1, dims=1)
    opt = T.configure_optimizer(model, lr=3e-3, weight_decay=0.0)
    for _ in range(200):
        loss = model(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    assert loss.item() < 0.1, f"loss on one batch after 200 steps is {loss.item():.3f}, expected below 0.1"


# ------------------------------------------------------------------ KV cache
@pytest.mark.parametrize("arch", ARCHS)
def test_cached_logits_equal_uncached(arch):
    model = build(arch)
    x = batch(model.cfg, B=2)
    with torch.no_grad():
        full = model(x)
        cache = model.new_cache()
        parts = [model(x[:, :6], cache=cache)]                   # prefill 6 tokens
        parts.append(model(x[:, 6:9], cache=cache))              # a 3-token chunk
        for t in range(9, x.size(1)):                            # then one token at a time
            parts.append(model(x[:, t:t + 1], cache=cache))
    cached = torch.cat(parts, dim=1)
    assert cached.shape == full.shape
    diff = (cached - full).abs().max().item()
    assert diff < 1e-4, f"cached and uncached logits differ by {diff:.2e}"


@pytest.mark.parametrize("arch", ARCHS)
def test_cache_holds_keys_and_values_per_layer(arch):
    model = build(arch)
    cache = model.new_cache()
    assert len(cache) == model.cfg.n_layer
    with torch.no_grad():
        model(batch(model.cfg, B=2, T=7), cache=cache)
    head_dim = model.cfg.d_model // model.cfg.n_head
    for layer in cache:
        assert layer["k"].shape == (2, model.cfg.n_head, 7, head_dim)
        assert layer["v"].shape == (2, model.cfg.n_head, 7, head_dim)


# ------------------------------------------------------------------ modern parts
def test_rmsnorm():
    norm = M.RMSNorm(16)
    x = torch.randn(3, 5, 16) * 7 + 2
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)           # unit RMS
    assert torch.allclose(y, x / x.pow(2).mean(-1, keepdim=True).sqrt(), atol=1e-4)   # no re-centering
    assert [tuple(p.shape) for p in norm.parameters()] == [(16,)]         # a gain, no bias


def test_rope_relative_position_property():
    torch.manual_seed(0)
    Dh = 32
    cos, sin = M.rope_cache(100, Dh)
    assert cos.shape == sin.shape == (100, Dh)
    q, k = torch.randn(1, 1, 1, Dh), torch.randn(1, 1, 1, Dh)
    Q = M.apply_rope(q.expand(1, 1, 100, Dh), cos, sin)[0, 0]     # q placed at every position
    K = M.apply_rope(k.expand(1, 1, 100, Dh), cos, sin)[0, 0]
    scores = Q @ K.T                                              # scores[m, n] = <q at m, k at n>
    for offset in (0, 1, 7, 40):
        diag = torch.diagonal(scores, offset=-offset)             # every pair with m - n = offset
        assert (diag - diag[0]).abs().max().item() < 1e-3, f"score depends on absolute position (offset {offset})"
    assert (scores[10, 3] - scores[10, 4]).abs().item() > 1e-3    # but it does depend on the offset
    assert torch.allclose(Q[0], q[0, 0, 0], atol=1e-6)            # position 0 is not rotated
    assert torch.allclose(Q.norm(dim=-1), q.norm().expand(100), atol=1e-4)   # rotation preserves length


def test_swiglu_hidden_keeps_parameter_budget():
    cfg = M.GPTConfig.modern(d_model=384)
    assert cfg.mlp_hidden == 1024                                 # 8d/3
    assert M.GPTConfig(d_model=384).mlp_hidden == 4 * 384
    assert M.GPTConfig.modern(d_model=256).mlp_hidden == 688      # 682.67 rounded up to a multiple of 8
