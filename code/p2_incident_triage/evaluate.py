#!/usr/bin/env python3
"""Evaluate a triage model on the held-out test split.

Metrics: JSON/schema validity (first try and after retries), field-level accuracy, severity
confusion matrix, severity off-by-one rate, root-cause token F1, next-step recall, latency.

    # Baseline: prompt-only base model served by vLLM/Ollama, with 3 few-shot examples
    LLM_BASE_URL=http://localhost:8000/v1 python evaluate.py --model Qwen/Qwen2.5-7B-Instruct \
        --few-shot 3 --name baseline

    # Fine-tuned: local HF model + adapter, then compare against the baseline report
    python evaluate.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --adapter out/qwen7b-triage \
        --name finetuned --compare reports/baseline.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from inference import add_backend_args, make_backend, triage
from prompts import SEVERITIES

EXACT_FIELDS = ("severity", "category", "affected_service", "escalate_to")
_TOKEN = re.compile(r"[a-z0-9_.\-]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_f1(pred: str, gold: str) -> float:
    p, g = Counter(tokens(pred)), Counter(tokens(gold))
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def step_recall(pred_steps: list[str], gold_steps: list[str], threshold: float = 0.5) -> float:
    """Fraction of gold steps matched (token F1 >= threshold) by some predicted step."""
    if not gold_steps:
        return 1.0
    hit = sum(any(token_f1(p, g) >= threshold for p in pred_steps) for g in gold_steps)
    return hit / len(gold_steps)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def confusion_matrix(pairs: list[tuple[str, str]]) -> str:
    cm = Counter(pairs)
    header = "gold\\pred " + " ".join(f"{s:>5}" for s in (*SEVERITIES, "inv"))
    rows = [header]
    for g in SEVERITIES:
        rows.append(f"{g:>9} " + " ".join(f"{cm[(g, p)]:>5}" for p in (*SEVERITIES, "inv")))
    return "\n".join(rows)


def score(records: list[dict]) -> dict:
    n = len(records)
    valid = [r for r in records if r["ok"]]
    m: dict = {
        "n": n,
        "valid_first_try": sum(r["first_try_valid"] for r in records) / n,
        "valid_after_retries": len(valid) / n,
        "mean_attempts": statistics.mean(r["attempts"] for r in records),
        "latency_p50_s": statistics.median(r["latency_s"] for r in records),
    }
    # Invalid outputs count as wrong for every field (strict accuracy over all n).
    for field in EXACT_FIELDS:
        m[f"acc_{field}"] = sum(r["ok"] and r["triage"][field] == r["gold"][field] for r in records) / n
    m["acc_all_exact_fields"] = sum(
        r["ok"] and all(r["triage"][f] == r["gold"][f] for f in EXACT_FIELDS) for r in records) / n
    sev_dist = [abs(int(r["triage"]["severity"][1]) - int(r["gold"]["severity"][1])) for r in valid]
    m["severity_off_by_more_than_one"] = sum(d > 1 for d in sev_dist) / max(len(valid), 1)
    m["severity_under_triage_rate"] = sum(
        int(r["triage"]["severity"][1]) > int(r["gold"]["severity"][1]) for r in valid) / max(len(valid), 1)
    m["root_cause_token_f1"] = statistics.mean(
        token_f1(r["triage"]["probable_root_cause"], r["gold"]["probable_root_cause"]) if r["ok"] else 0.0
        for r in records)
    m["next_step_recall"] = statistics.mean(
        step_recall(r["triage"]["recommended_next_steps"], r["gold"]["recommended_next_steps"]) if r["ok"] else 0.0
        for r in records)
    m["confidence_mae"] = statistics.mean(
        abs(r["triage"]["confidence"] - r["gold"]["confidence"]) for r in valid) if valid else None
    # Per-template accuracy reveals which incident types the model struggles with.
    per_t: dict[str, list[bool]] = {}
    for r in records:
        per_t.setdefault(r["template"], []).append(
            r["ok"] and all(r["triage"][f] == r["gold"][f] for f in EXACT_FIELDS))
    m["per_template_exact"] = {t: sum(v) / len(v) for t, v in sorted(per_t.items())}
    m["severity_confusion"] = confusion_matrix(
        [(r["gold"]["severity"], r["triage"]["severity"] if r["ok"] else "inv") for r in records])
    return m


def print_report(name: str, m: dict, other: dict | None = None) -> None:
    print(f"\n=== {name} (n={m['n']}) ===")
    for k, v in m.items():
        if k in ("n", "severity_confusion", "per_template_exact") or v is None:
            continue
        line = f"{k:32s} {v:8.3f}"
        if other and isinstance(other.get(k), (int, float)):
            line += f"   (vs {other[k]:.3f}, delta {v - other[k]:+.3f})"
        print(line)
    print("\nper-template exact-field accuracy:")
    for t, v in m["per_template_exact"].items():
        print(f"  {t:28s} {v:.3f}")
    print("\nseverity confusion matrix (rows = gold, cols = predicted, inv = invalid output):")
    print(m["severity_confusion"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N examples")
    ap.add_argument("--few-shot", type=int, default=0, help="k training examples in the prompt (baseline)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--name", default="run")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--compare", default=None, help="path to another report JSON to diff against")
    add_backend_args(ap)
    args = ap.parse_args()
    if args.backend == "hf" and not args.model:
        ap.error("--model is required with --backend hf")

    rows = load_jsonl(f"{args.data}/{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    few_shot = random.Random(0).sample(load_jsonl(f"{args.data}/train.jsonl"), args.few_shot) if args.few_shot else None
    backend = make_backend(args)

    records = []
    for i, row in enumerate(rows, start=1):
        res = triage(backend, row["messages"][1]["content"], max_retries=args.retries, few_shot=few_shot)
        res["gold"] = json.loads(row["messages"][2]["content"])
        res["template"] = row["meta"]["template"]
        records.append(res)
        if i % 25 == 0:
            print(f"{i}/{len(rows)} done")

    metrics = score(records)
    Path(args.reports).mkdir(exist_ok=True)
    with open(f"{args.reports}/{args.name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{args.reports}/{args.name}.predictions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    other = json.load(open(args.compare)) if args.compare else None
    print_report(args.name, metrics, other)


if __name__ == "__main__":
    main()
