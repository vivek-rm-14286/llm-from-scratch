#!/usr/bin/env python3
"""Run triage against a served model (vLLM / Ollama / any OpenAI-compatible endpoint)
or a local Hugging Face model (+ optional LoRA adapter), with schema guardrails and retries.

Examples:
    # Served model (vLLM or Ollama), incident text from a file:
    LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=triage python inference.py incident.txt

    # Local HF model + adapter:
    python inference.py incident.txt --backend hf --model Qwen/Qwen2.5-7B-Instruct --adapter out/qwen7b-triage
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from prompts import SYSTEM_PROMPT
from schema import Triage, json_schema, parse_triage


class OpenAIBackend:
    """Any OpenAI-compatible chat endpoint: vLLM, Ollama (/v1), llama.cpp server, hosted APIs."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", structured: bool = False):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.structured = structured

    def generate(self, messages: list[dict], max_tokens: int = 512) -> str:
        kwargs = {}
        if self.structured:  # constrained decoding; supported by recent vLLM and Ollama releases
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "triage", "schema": json_schema()},
            }
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0, max_tokens=max_tokens, **kwargs
        )
        return resp.choices[0].message.content or ""


class HFBackend:
    """Local transformers model, optionally 4-bit, optionally with a PEFT adapter."""

    def __init__(self, model_name: str, adapter: str | None = None, load_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter or model_name)
        quant = None
        if load_4bit:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant, dtype=torch.bfloat16, device_map="auto"
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, messages: list[dict], max_tokens: int = 512) -> str:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def business_rule_errors(t: Triage, incident_text: str) -> list[str]:
    """Checks the schema can't express. Each error triggers a retry with feedback."""
    errors = []
    if t.affected_service not in incident_text:
        errors.append(f"affected_service '{t.affected_service}' does not appear in the input; "
                      "use a service name exactly as written in the alerts/logs")
    if t.severity == "P4" and t.escalate_to != "none":
        errors.append("severity P4 requires escalate_to 'none'")
    if t.severity != "P4" and t.escalate_to == "none":
        errors.append("severity P1-P3 requires an escalation target")
    return errors


def build_messages(incident_text: str, few_shot: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot or []:  # each ex is a training row with chat messages
        msgs += [ex["messages"][1], ex["messages"][2]]
    msgs.append({"role": "user", "content": incident_text})
    return msgs


def triage(backend, incident_text: str, max_retries: int = 2, few_shot: list[dict] | None = None) -> dict:
    """Generate -> validate -> (retry with error feedback). Never returns unvalidated output."""
    messages = build_messages(incident_text, few_shot)
    t0 = time.perf_counter()
    raw, error = "", None
    for attempt in range(1, max_retries + 2):
        raw = backend.generate(messages)
        result, error = parse_triage(raw)
        if result is not None:
            rule_errors = business_rule_errors(result, incident_text)
            if not rule_errors:
                return {"ok": True, "triage": result.model_dump(), "raw": raw, "attempts": attempt,
                        "first_try_valid": attempt == 1, "latency_s": time.perf_counter() - t0}
            error = "; ".join(rule_errors)
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Your previous answer was rejected: {error}. "
                                        "Return ONLY the corrected JSON object."},
        ]
    return {"ok": False, "triage": None, "raw": raw, "attempts": max_retries + 1, "error": error,
            "first_try_valid": False, "latency_s": time.perf_counter() - t0}


def make_backend(args):
    if args.backend == "hf":
        return HFBackend(args.model, args.adapter, load_4bit=not args.no_4bit)
    return OpenAIBackend(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        model=args.model or os.getenv("LLM_MODEL", "triage"),
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        structured=args.structured,
    )


def add_backend_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--backend", choices=["openai", "hf"], default="openai")
    ap.add_argument("--model", default=None, help="served model name (openai) or HF model id (hf)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (hf backend)")
    ap.add_argument("--no-4bit", action="store_true", help="load HF model in bf16 instead of 4-bit")
    ap.add_argument("--structured", action="store_true", help="request JSON-schema constrained decoding")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("incident", nargs="?", help="file with alerts/logs/metrics text (default: stdin)")
    ap.add_argument("--retries", type=int, default=2)
    add_backend_args(ap)
    args = ap.parse_args()
    if args.backend == "hf" and not args.model:
        ap.error("--model is required with --backend hf")
    text = open(args.incident, encoding="utf-8").read() if args.incident else sys.stdin.read()
    result = triage(make_backend(args), text, max_retries=args.retries)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
