"""pytest setup: `pytest` grades starter/ (the default), `pytest --impl solution` grades
solution/. `PTA_IMPL=solution pytest` does the same through an environment variable.

The fixtures build a character-level Hugging Face tokenizer and a tiny randomly
initialized GPT-2, so the tests need no download and run in seconds on a CPU.
"""
import os
import string
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common import load_impl  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--impl", default=os.environ.get("PTA_IMPL", "starter"),
                     choices=["solution", "starter"],
                     help="package to test: starter (default) or solution")


def pytest_report_header(config):
    return f"implementation under test: {config.getoption('--impl')}/"


@pytest.fixture(scope="session")
def impl(request):
    return load_impl(request.config.getoption("--impl"))


def make_char_tokenizer(with_chatml=True):
    """One token per character. With with_chatml, <|im_start|> and <|im_end|> are added."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    chars = ["[PAD]", "[UNK]"] + list(string.ascii_letters + string.digits + " \n.,:;!?+-=*/()'\"")
    t = Tokenizer(models.WordLevel({c: i for i, c in enumerate(chars)}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), "isolated")
    t.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]")
    if with_chatml:
        tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    return tok


def make_tiny_model(vocab_size, seed=0):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=vocab_size, n_positions=128, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=None, eos_token_id=None,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2LMHeadModel(cfg).eval()


@pytest.fixture
def tok():
    return make_char_tokenizer()


@pytest.fixture
def fns(tok):
    """(encode, im_start_id, im_end_id, nl_ids, pad_id) as the scripts pass them around."""
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    return (encode, tok.convert_tokens_to_ids("<|im_start|>"),
            tok.convert_tokens_to_ids("<|im_end|>"), encode("\n"), tok.pad_token_id)


@pytest.fixture
def tiny_model(tok):
    return make_tiny_model(len(tok))


CONV = [{"role": "user", "content": "2+2=?"}, {"role": "assistant", "content": "4"},
        {"role": "user", "content": "10+7=?"}, {"role": "assistant", "content": "17"}]


@pytest.fixture
def conv():
    return [dict(m) for m in CONV]
