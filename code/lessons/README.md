# Lesson scripts

Small runnable scripts that belong to concept pages (not to one of the four projects). Each runs on a laptop CPU.

| Folder | Lesson | What is in it |
|---|---|---|
| `debugging/` | [Debugging a Training Run](https://vivek-rm-14286.github.io/llm-from-scratch/#/pretraining/debugging) | `buggy_train.py` (five planted bugs), `fixed_train.py` (reference; `--bug N` re-plants one bug), `diagnostics.py` (activation, update-ratio and first-NaN helpers) |
| `serving/` | [Quantization & Serving](https://vivek-rm-14286.github.io/llm-from-scratch/#/inference/serving) | `loadtest.py` (TTFT, time per output token, throughput), `fake_server.py` (a local OpenAI-compatible streaming server to test against), `quant_compare.py`, `batched_generate.py` |

Requirements: Python 3.12+, `torch`; `transformers` for the serving scripts that load a model.
