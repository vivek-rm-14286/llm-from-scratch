# Project 7: Post-train a Small Model, SFT Then DPO

Turn an open base model (SmolLM2-135M by default) into a small assistant with post-training code written by hand: ChatML rendering and loss masking, an explicit SFT loop, a preference dataset labeled by a transparent programmatic judge, and DPO against a frozen reference. Every stage is evaluated with the same judge, so base, SFT and DPO can be compared number for number.

Full walkthrough: https://vivek-rm-14286.github.io/llm-from-scratch/#/projects/post-train-assistant

## Layout

| Path | What it is |
|---|---|
| `starter/` | Signatures, docstrings and `TODO`s. This is where the work happens. |
| `solution/` | The complete reference implementation. |
| `tests/` | pytest tests that grade either package. They use a tiny random model: no download. |
| `conftest.py` | Adds `--impl` and builds the tiny tokenizer and model for the tests. |

Both packages have the same four files:

| File | Step | Contents |
|---|---|---|
| `data.py` | 2 | `ensure_chatml_tokens`, `render_chatml`, `render_prompt`, `collate`, `length_bucketed_batches` |
| `sft.py` | 2 | `sft_loss`, `lr_at`, `eval_loss`, `train_sft` |
| `judge.py` | 3 | `has_repeated_ngram`, `check_response`, `judge`, `pick_pair` |
| `dpo.py` | 4 | `sequence_logprobs`, `dpo_loss`, `build_dpo_example`, `dpo_batch`, `freeze_copy`, `dpo_forward`, `train_dpo` |

The scripts at the top level are complete and call into whichever package `--impl` names (default `solution`):

| Script | Step | What it does |
|---|---|---|
| `prepare_data.py` | 0 | Streams a slice of `HuggingFaceTB/smoltalk` (Apache-2.0) and writes the SFT set and two held-out prompt sets to `data/` |
| `evaluate.py` | 1, 5 | Eval loss, rubric pass-rate, win-rate against an earlier evaluation, sample outputs |
| `run_sft.py` | 2 | SFT of the base model, saved to `out/sft` |
| `build_prefs.py` | 3 | Samples completions from the SFT model, judges them, writes `data/prefs.jsonl` |
| `run_dpo.py` | 4 | DPO from `out/sft` against a frozen copy of itself, saved to `out/dpo` |
| `crosscheck_dpo.py` | 6 | The DPO loss on a fixed batch against a hand computation and TRL's `DPOTrainer` |
| `constraints.py` | | The checkable answer constraints, shared by data preparation and the judge |
| `llm_judge.py` | | Optional LLM judge through any OpenAI-compatible endpoint |
| `common.py` | | Model loading, device selection, batched generation, JSON I/O |

## Setup

```bash
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install -r requirements.txt        # trl is optional: delete that line to skip it
```

The default run fits a laptop: Apple Silicon (MPS) or a CPU with 16 GB of RAM. A CUDA GPU is used automatically when present.

## Tests

```bash
pytest                          # grades starter/: 37 failures until the TODOs are filled in
pytest --impl solution          # grades solution/: 38 passed
PTA_IMPL=solution pytest        # the same, through an environment variable
pytest tests/test_data.py       # one file at a time: test_data, test_sft, test_judge, test_dpo
pytest -x                       # stop at the first failure
```

## The run

Replace nothing to use the reference; add `--impl starter` to every command once your tests pass.

```bash
python prepare_data.py                                           # 6,000 train, 300 eval, 400 + 200 held-out prompts
python evaluate.py --name base                                   # Step 1: the base model
python run_sft.py                                                # Step 2: SFT, 375 steps
python evaluate.py --model out/sft --name sft --against base
python build_prefs.py                                            # Step 3: 4 samples per prompt, judged
python run_dpo.py                                                # Step 4: DPO
python evaluate.py --model out/dpo --name dpo --against sft      # Step 5: win-rate against SFT
python crosscheck_dpo.py                                         # Step 6: loss cross-check, no download
```

Everything is written to `data/` and `out/` (both git-ignored): checkpoints in `out/sft` and `out/dpo`, generations in `out/gen_<name>.jsonl`, metrics in `out/eval_<name>.json`, training curves in `out/sft/sft_history.json` and `out/dpo/dpo_history.json`.

Useful flags: `--device cpu|mps|cuda`, `--seed`, `run_sft.py --limit 1000` for a quick pass, `--model HuggingFaceTB/SmolLM2-360M` on `evaluate.py` and `run_sft.py` for the larger model, `build_prefs.py --llm-judge` with `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `JUDGE_MODEL` set.
