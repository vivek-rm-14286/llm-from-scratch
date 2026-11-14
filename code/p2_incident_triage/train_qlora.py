#!/usr/bin/env python3
"""QLoRA fine-tuning for the Incident Triage Copilot with transformers + peft + trl.

Uses the SFTConfig/SFTTrainer API of TRL 1.x (stable since 0.20: `max_length`, `processing_class`,
conversational prompt/completion datasets with completion-only loss). See requirements.txt.

    # 7B on a single 24 GB GPU (RTX 3090/4090, A10G, L4):
    python train_qlora.py --model Qwen/Qwen2.5-7B-Instruct --data data --out out/qwen7b-triage

    # Smaller GPU / free Colab T4 (16 GB): 1.5B model, fp16 compute
    python train_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct --out out/qwen1.5b-triage --fp16 --batch-size 2
"""
from __future__ import annotations

import argparse
import dataclasses
import json

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def to_prompt_completion(row: dict) -> dict:
    """Split chat messages so loss is computed only on the assistant's JSON."""
    msgs = row["messages"]
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out/qwen7b-triage")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--fp16", action="store_true", help="use fp16 instead of bf16 (e.g. T4 / V100)")
    args = ap.parse_args()

    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16

    # ---- data -------------------------------------------------------------------------
    ds = load_dataset("json", data_files={"train": f"{args.data}/train.jsonl",
                                          "validation": f"{args.data}/val.jsonl"})
    ds = ds.map(to_prompt_completion, remove_columns=ds["train"].column_names)

    # ---- model in 4-bit (QLoRA) ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4, from the QLoRA paper
        bnb_4bit_use_double_quant=True,      # quantize the quantization constants too
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing during training

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # ---- trainer --------------------------------------------------------------------------
    # transformers 5 removed `warmup_ratio`; a float `warmup_steps` below 1 is now the ratio.
    has_ratio = "warmup_ratio" in {f.name for f in dataclasses.fields(SFTConfig)}
    warmup = {"warmup_ratio": 0.03} if has_ratio else {"warmup_steps": 0.03}
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        **warmup,                            # 3% linear warmup
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,           # mask system+user tokens out of the loss
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,  # TRL wraps the 4-bit model with LoRA (exact k-bit prep varies by TRL version)
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    trainer.save_model(args.out)        # saves only the LoRA adapter (roughly 160 MB at rank 16 on the 7B model)
    tokenizer.save_pretrained(args.out)
    with open(f"{args.out}/train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
