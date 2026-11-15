# Project 2: Tiny GPT

Write a complete language-model stack yourself, at a size a laptop trains in minutes:
a byte-level BPE tokenizer, a decoder-only transformer (GPT-2 style first, then RMSNorm + RoPE + SwiGLU),
a memory-mapped data pipeline, a training loop with checkpoint/resume, and sampling with a KV cache.
The default model has 11.5M parameters and trains on tiny Shakespeare (about 1 MB of public-domain text).

Full walkthrough: https://vivek-rm-14286.github.io/llm-from-scratch/#/projects/tiny-gpt

## Layout

| Path | What it is |
|---|---|
| `starter/` | Signatures, docstrings and TODOs. This is where you work. |
| `solution/` | The complete reference implementation. Read it after you pass a step, not before. |
| `tests/` | pytest tests that grade either directory. |
| `conftest.py`, `pytest.ini` | Add the `--impl` option that selects the directory under test. |

Both directories hold the same six files:

| File | Step | Contents |
|---|---|---|
| `bpe.py` | 1 | `ByteBPE`: `train`, `encode`, `decode`, `save`, `load` |
| `model.py` | 2, 3, 6 | `GPTConfig`, `GPT`, `count_params`, `RMSNorm`, `rope_cache`, `apply_rope`, `SwiGLU`, the KV cache |
| `checks.py` | 3 | Step-0 loss, overfit one batch, causality, cached == uncached |
| `data.py` | 4 | `prepare` (tokenize once to uint16 files), `load_tokens`, `get_batch` |
| `train.py` | 5 | `lr_at`, `configure_optimizer`, `estimate_loss`, `save_checkpoint`, `load_checkpoint`, the loop |
| `generate.py` | 7 | `sample_next` (temperature, top-k, top-p), `generate` (with and without the cache) |

## Setup

```bash
cd code/tiny_gpt
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # torch, numpy, pytest (+ datasets for the optional TinyStories slice)
```

## Grading

```bash
pytest                          # grades starter/ : everything fails until you fill in the TODOs
pytest --impl solution          # grades the reference: 50 tests pass in under 10 seconds on a laptop CPU
TINY_GPT_IMPL=solution pytest   # same, through an environment variable
pytest tests/test_tokenizer.py  # one step at a time
pytest -k gpt2                  # only the GPT-2-style cases (before the step 6 upgrade)
```

## Running it

Run from inside `starter/` (or `solution/`); `data/` and `out/` are created there and are git-ignored.

```bash
cd starter
python data.py                                   # downloads tiny Shakespeare, trains the tokenizer, writes train.bin / val.bin
python checks.py && python checks.py --modern    # sanity checks; all five lines must say PASS
python train.py --out out/gpt2                   # GPT-2 style, 11.5M parameters
python train.py --modern --out out/modern        # RMSNorm + RoPE + SwiGLU, 11.4M parameters
python generate.py --ckpt out/gpt2/ckpt_best.pt --prompt "ROMEO:" --tokens 200
python generate.py --ckpt out/gpt2/ckpt_best.pt --benchmark      # cached vs uncached decoding speed
python train.py --resume out/gpt2/ckpt_last.pt                  # continue an interrupted run
```

Optional TinyStories slice (needs `datasets` and a network connection):

```bash
python data.py --dataset tinystories --stories 20000 --vocab-size 4096 --out data/tinystories
python train.py --data data/tinystories --out out/ts --dropout 0.0 --max-steps 3000
python generate.py --ckpt out/ts/ckpt_best.pt --data data/tinystories --prompt "Once upon a time"
```

## Data and licences

- Tiny Shakespeare: 1,115,394 characters of Shakespeare's plays (public domain), fetched from the `karpathy/char-rnn` repository.
- TinyStories (`roneneldan/TinyStories` on the Hugging Face Hub): CDLA-Sharing-1.0.
