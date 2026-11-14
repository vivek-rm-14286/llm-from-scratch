# Project 1: Build a ~500M-parameter SLM from scratch

A Llama-style decoder (RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings) written in
about 200 lines of PyTorch, plus everything needed to train and release it: tokenizer training,
data sharding, pretraining (single GPU or DDP), SFT into a chat model, evaluation, conversion to
Hugging Face format and upload to the Hub.

Full walkthrough: https://vivek-rm-14286.github.io/llm-from-scratch/#/projects/p1-500m-model

## Files

| File | What it does | Needs GPU |
|---|---|---|
| `config.py` | `ModelConfig` dataclass and the presets `p500m` (486.6M), `p125m`, `p30m` (31.5M), `tiny` | no |
| `model.py` | The transformer, KV cache and sampling | no |
| `count_params.py` | Parameter count by formula and by instantiation, plus the FLOPs / GPU-hours estimate | no |
| `train_tokenizer.py` | Byte-level BPE with HF `tokenizers` | no |
| `prepare_data.py` | Streams a dataset, tokenizes it in parallel, writes `uint16` `.bin` shards and `meta.json` | no |
| `train.py` | Pretraining: bf16, `torch.compile`, gradient accumulation, WSD/cosine schedule, DDP, resume, MFU | for real runs |
| `sft.py` | ChatML SFT with loss masking on SmolTalk | for real runs |
| `chat.py` | Interactive chat or plain completion | no |
| `eval_hellaswag.py` | HellaSwag scoring by log-probability | recommended |
| `convert_to_hf.py` | Export to `LlamaForCausalLM`; `--check` asserts logit equality | no |
| `push_to_hub.py` | Writes a model card and uploads the folder | no |
| `configs/*.json` | `p30m_laptop`, `p125m_24gb`, `p500m_10b`, `p500m_50b` | |

## Smoke test (about a minute, CPU or Apple Silicon)

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

python count_params.py                 # 486,605,824 parameters, and the compute estimate
python train.py --tiny                 # synthetic data, 60 steps; val loss falls from ~6.1 to ~1.3
python sft.py --tiny                   # SFT on the tiny checkpoint
python chat.py --ckpt out/tiny_sft/ckpt_latest.pt --prompt "What is 3 plus 4?"
python convert_to_hf.py --check        # prints PASS when the logits match HF LlamaForCausalLM
```

Run every script from this directory.

## Laptop warm-up: the 31M model on TinyStories

```bash
python count_params.py --model p30m    # 31,465,984 parameters
python train_tokenizer.py --dataset roneneldan/TinyStories --name "" \
    --num_docs 200000 --vocab_size 8192 --out tokenizer_ts
python prepare_data.py --tokenizer tokenizer_ts/tokenizer.json --out data/tinystories \
    --source roneneldan/TinyStories::text:1.0 --total_tokens 60e6 \
    --val_tokens 1000000 --shard_tokens 20000000
python train.py --config configs/p30m_laptop.json     # 50M tokens; hours on a laptop, see the page
```

## The real runs

```bash
python train_tokenizer.py --num_docs 1000000 --vocab_size 49152 --out tokenizer
python prepare_data.py --tokenizer tokenizer/tokenizer.json --out data/fwe \
    --source HuggingFaceFW/fineweb-edu:sample-10BT:text:1.0 --total_tokens 10e9

torchrun --standalone --nproc_per_node=8 train.py --config configs/p500m_10b.json
# after an interruption: the same command plus --resume out/p500m/ckpt_latest.pt

python sft.py --init out/p500m/ckpt_latest.pt --out_dir out/sft
python eval_hellaswag.py --ckpt out/p500m/ckpt_latest.pt
python convert_to_hf.py --ckpt out/sft/ckpt_latest.pt --out hf/slm-500m-instruct --chat
python push_to_hub.py --folder hf/slm-500m-instruct --repo <user>/slm-500m-instruct --dry_run
```

The 10B-token run is about 26 H100-hours at 40% MFU (about 3.5 hours on 8 GPUs); the 50B-token run
is about 131 H100-hours. Both figures are estimates from `count_params.py`, not measured runs.
