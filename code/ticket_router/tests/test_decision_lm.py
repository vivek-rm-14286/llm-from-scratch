"""Order invariance of the decision LM, checked on a tiny random backbone (no download)."""
import zlib

import pytest
import torch


class ToyTokenizer:
    """Whitespace tokenizer with stable hashed ids. Enough to exercise packing and masking."""
    pad_token_id = 0

    class _Enc:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        return self._Enc([1 + zlib.crc32(w.encode()) % 498 for w in text.split()])


@pytest.fixture(scope="module")
def dlm(impl):
    return impl("decision_lm")


@pytest.fixture()
def model(dlm):
    from transformers import LlamaConfig, LlamaModel
    torch.manual_seed(0)
    backbone = LlamaModel(LlamaConfig(vocab_size=500, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=2)).eval()
    m = dlm.DecisionLM("toy", backbone=backbone, tokenizer=ToyTokenizer()).eval()
    torch.nn.init.normal_(m.scorer.weight)                  # the zero-initialized head would hide every bug
    return m


CONTEXT = "Support message: we were charged twice this month"
OPTIONS = ["billing: invoices and refunds", "api: keys and rate limits", "access: login",
           "other: anything else that does not fit"]


def test_mask_blocks_other_options(dlm):
    seg = torch.tensor([[0, 0, 1, 1, 2, 2, -1]])
    allowed = dlm.build_option_mask(seg)[0]
    assert allowed.shape == (7, 7)
    assert allowed[1].tolist() == [True, True, False, False, False, False, False]    # context: causal
    assert allowed[3].tolist() == [True, True, True, True, False, False, False]      # option 1 sees context + itself
    assert allowed[5].tolist() == [True, True, False, False, True, True, False]      # option 2 never sees option 1
    assert not allowed[:6, 6].any() and allowed[6, 6]                                 # nothing attends to padding
    assert not allowed[2, 3]                                                          # still causal inside an option


def test_pack_restarts_positions_after_the_context(model):
    ids, seg, pos, last = model.pack(CONTEXT, OPTIONS)
    n_ctx = seg.count(0)
    assert len(ids) == len(seg) == len(pos)
    assert pos[:n_ctx] == list(range(n_ctx))
    for k in range(1, len(OPTIONS) + 1):
        idx = [i for i, s in enumerate(seg) if s == k]
        assert [pos[i] for i in idx] == list(range(n_ctx, n_ctx + len(idx)))
        assert last[k - 1] == idx[-1]


def test_scores_do_not_depend_on_option_order(model):
    with torch.no_grad():
        base = model([CONTEXT], [OPTIONS])[0]
        assert base.std() > 1e-3                                                      # the scores are not all equal
        for perm in ([3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]):
            shuffled = model([CONTEXT], [[OPTIONS[i] for i in perm]])[0]
            assert torch.allclose(shuffled, base[perm], atol=1e-4)


def test_each_score_equals_scoring_that_option_alone(model):
    with torch.no_grad():
        together = model([CONTEXT], [OPTIONS])[0]
        alone = torch.stack([model([CONTEXT], [[o]])[0, 0] for o in OPTIONS])
    assert torch.allclose(together, alone, atol=1e-4)


def test_batching_with_padding_and_fewer_options(model):
    with torch.no_grad():
        single = model([CONTEXT], [OPTIONS])[0]
        batch = model([CONTEXT, "Support message: hi"], [OPTIONS, OPTIONS[:2]])
    assert batch.shape == (2, 4)
    assert torch.allclose(batch[0], single, atol=1e-4)
    assert torch.isinf(batch[1, 2:]).all() and torch.isfinite(batch[1, :2]).all()
