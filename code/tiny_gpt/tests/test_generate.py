"""Step 7: sampling and cached generation."""
import pytest
import torch

LOGITS = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.07, 0.03]]))


def draws(n=400, **kw):
    import generate
    torch.manual_seed(0)
    return {int(generate.sample_next(LOGITS, **kw)) for _ in range(n)}


def test_sample_shape_and_greedy():
    import generate
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]])
    out = generate.sample_next(logits, temperature=0.0)
    assert out.shape == (2, 1)
    assert out.tolist() == [[1], [0]]
    assert generate.sample_next(logits, temperature=1.0).shape == (2, 1)


def test_sampling_does_not_modify_logits():
    import generate
    logits = LOGITS.clone()
    generate.sample_next(logits, temperature=0.7, top_k=2, top_p=0.9)
    assert torch.equal(logits, LOGITS)


def test_full_distribution_is_reachable():
    assert draws(n=2000) == {0, 1, 2, 3, 4}


def test_top_k():
    assert draws(top_k=2) == {0, 1}
    assert draws(top_k=1) == {0}
    assert draws(n=2000, top_k=50) == {0, 1, 2, 3, 4}      # k larger than the vocabulary is not an error


def test_top_p():
    assert draws(top_p=0.6) == {0, 1}          # 0.5 is short of 0.6, so the token that crosses it is kept
    assert draws(top_p=0.8) == {0, 1, 2}
    assert draws(top_p=0.3) == {0}             # the top token alone already exceeds 0.3


def test_low_temperature_sharpens():
    import generate
    torch.manual_seed(0)
    cold = sum(int(generate.sample_next(LOGITS, temperature=0.25)) == 0 for _ in range(300))
    hot = sum(int(generate.sample_next(LOGITS, temperature=2.0)) == 0 for _ in range(300))
    assert cold > 250 > hot


@pytest.mark.parametrize("arch", ["gpt2", "modern"])
def test_generate_cached_equals_uncached(arch):
    import generate
    import model as M
    kw = dict(vocab_size=61, block_size=32, n_layer=2, n_head=2, d_model=32)
    torch.manual_seed(0)
    net = M.GPT(M.GPTConfig.modern(**kw) if arch == "modern" else M.GPTConfig(**kw)).train()
    prompt = torch.randint(61, (2, 5))
    a = generate.generate(net, prompt, 20, temperature=0.0, use_cache=True)
    b = generate.generate(net, prompt, 20, temperature=0.0, use_cache=False)
    assert a.shape == (2, 25)
    assert torch.equal(a[:, :5], prompt)
    assert torch.equal(a, b), "greedy decoding with and without the cache must give the same tokens"
    assert net.training, "generate must restore the mode the model was in"


@pytest.mark.parametrize("use_cache", [True, False])
def test_generate_past_the_context_window(use_cache):
    import generate
    import model as M
    torch.manual_seed(0)
    net = M.GPT(M.GPTConfig(vocab_size=61, block_size=16, n_layer=1, n_head=2, d_model=32))
    out = generate.generate(net, torch.randint(61, (1, 4)), 40, temperature=1.0, top_k=10, use_cache=use_cache)
    assert out.shape == (1, 44)
    assert int(out.max()) < 61
