# Project 6: Incident Triage Copilot (QLoRA fine-tune)

Fine-tune an open-weight model (Qwen2.5-7B-Instruct, or Qwen2.5-1.5B-Instruct for small GPUs) to turn
raw monitoring alerts, log snippets and metrics into strict triage JSON:

```json
{"severity": "P2", "category": "database", "affected_service": "orders-db",
 "probable_root_cause": "...", "recommended_next_steps": ["...", "..."],
 "escalate_to": "database-oncall", "confidence": 0.9}
```

Full walkthrough: https://vivek-rm-14286.github.io/llm-from-scratch/#/projects/incident-triage-finetune

## Files

| File | What it does | Needs GPU |
|---|---|---|
| `prompts.py` | System prompt, label vocabularies, input formatting (stdlib only) | no |
| `generate_data.py` | Synthetic scenario generator -> `data/{train,val,test}.jsonl` (stdlib only) | no |
| `teacher_label.py` | Label real incidents with a teacher LLM, export human-approved rows | no |
| `schema.py` | Pydantic schema + JSON extraction (guardrails) | no |
| `inference.py` | Triage with validation + retry, OpenAI-compatible or local HF backend | HF backend only |
| `train_qlora.py` | QLoRA SFT with transformers + peft + trl + bitsandbytes | yes |
| `evaluate.py` | Validity, field accuracy, severity confusion matrix, baseline comparison; `--self-check` needs no model | depends on backend |
| `merge_adapter.py` | Merge LoRA into base weights for serving | no (CPU ok) |

## Quickstart

```bash
uv venv && source .venv/bin/activate      # or python -m venv .venv

# 1. Data (no dependencies)
python generate_data.py --out data --n-per-template 120

uv pip install -r requirements.txt        # bitsandbytes is installed on Linux only

# 1b. Smoke test, no model: gold answers replayed through the pipeline; prints SELF-CHECK PASS
python evaluate.py --self-check

# 2. Baseline: prompt the base model (serve it first, e.g. with vLLM)
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 &
LLM_BASE_URL=http://localhost:8000/v1 python evaluate.py --model Qwen/Qwen2.5-7B-Instruct \
    --few-shot 3 --name baseline --limit 200

#    No 24 GB GPU? Use Ollama on a Mac, or the local HF backend with the 1.5B model:
# ollama pull qwen2.5:7b
# LLM_BASE_URL=http://localhost:11434/v1 python evaluate.py --model qwen2.5:7b --few-shot 3 --name baseline --limit 200
# python evaluate.py --backend hf --no-4bit --model Qwen/Qwen2.5-1.5B-Instruct --few-shot 3 --name baseline --limit 200

# 3. Train (single 24 GB GPU)
python train_qlora.py --model Qwen/Qwen2.5-7B-Instruct --data data --out out/qwen7b-triage
#    16 GB GPU / Colab T4:
# python train_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct --out out/qwen1.5b-triage --fp16 --batch-size 2

# 4. Evaluate the adapter and compare with the baseline
python evaluate.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --adapter out/qwen7b-triage \
    --name finetuned --limit 200 --compare reports/baseline.json
```

## Serving

**vLLM with the adapter (no merge needed):**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules triage=out/qwen7b-triage --max-model-len 8192
LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=triage python inference.py incident.txt
```

**vLLM with merged weights:**

```bash
python merge_adapter.py --base Qwen/Qwen2.5-7B-Instruct --adapter out/qwen7b-triage --out out/qwen7b-triage-merged
vllm serve out/qwen7b-triage-merged --served-model-name triage --max-model-len 8192
```

**GGUF via llama.cpp, then Ollama:**

```bash
git clone https://github.com/ggml-org/llama.cpp && uv pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py out/qwen7b-triage-merged --outfile triage-f16.gguf --outtype f16
# build llama.cpp (cmake -B build && cmake --build build --config Release), then:
./llama.cpp/build/bin/llama-quantize triage-f16.gguf triage-q4_k_m.gguf Q4_K_M
printf 'FROM ./triage-q4_k_m.gguf\nPARAMETER temperature 0\n' > Modelfile
ollama create triage -f Modelfile
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=triage python inference.py incident.txt
```

Re-run `evaluate.py` against every serving format: quantization and merging can change outputs.
