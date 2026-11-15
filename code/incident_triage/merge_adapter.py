#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model weights for serving (vLLM, GGUF export).

    python merge_adapter.py --base Qwen/Qwen2.5-7B-Instruct --adapter out/qwen7b-triage --out out/qwen7b-triage-merged

The base model is loaded in bf16 (not 4-bit): you merge into full-precision weights.
Needs roughly 2 bytes per parameter of CPU RAM or GPU memory (about 15 GB for 7B).
Re-run evaluate.py on the merged model; merging into bf16 weights that were trained against
NF4-quantized weights can shift outputs slightly.
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu", help="'cpu' (safe) or 'auto' to use GPU")
args = ap.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map=args.device)
model = PeftModel.from_pretrained(model, args.adapter)
model = model.merge_and_unload()  # W <- W + (alpha/r) * B @ A, then drop the LoRA modules
model.save_pretrained(args.out, safe_serialization=True)
AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
print(f"merged model written to {args.out}")
