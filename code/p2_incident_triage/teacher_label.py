#!/usr/bin/env python3
"""Label real (anonymized) incidents with a strong teacher LLM, then route them to human review.

Step 1 - label:   python teacher_label.py label raw_incidents.jsonl review_queue.jsonl
    raw_incidents.jsonl rows: {"id": "...", "text": "<alerts/logs/metrics as in prompts.format_user_prompt>",
                               "template": "real"}   # template is used for leakage-aware splitting
    Uses TEACHER_BASE_URL / TEACHER_MODEL / TEACHER_API_KEY (any OpenAI-compatible endpoint).

Step 2 - review:  open review_queue.jsonl, fix labels, set "status": "approved" or "rejected".

Step 3 - export:  python teacher_label.py export review_queue.jsonl data/real_train.jsonl
    Writes approved rows in the same chat format as generate_data.py.
"""
from __future__ import annotations

import argparse
import json
import os

from inference import OpenAIBackend, triage
from prompts import SYSTEM_PROMPT


def label(src: str, dst: str) -> None:
    backend = OpenAIBackend(
        base_url=os.environ["TEACHER_BASE_URL"],
        model=os.environ["TEACHER_MODEL"],
        api_key=os.getenv("TEACHER_API_KEY", "not-needed"),
    )
    n_ok = n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            res = triage(backend, row["text"], max_retries=2)
            n += 1
            n_ok += res["ok"]
            fout.write(json.dumps({
                "id": row["id"], "text": row["text"], "template": row.get("template", "real"),
                "label": res["triage"], "teacher_raw": res["raw"],
                "status": "pending_review" if res["ok"] else "teacher_failed",
            }) + "\n")
    print(f"labeled {n} incidents, {n_ok} schema-valid; review {dst} before exporting")


def export(src: str, dst: str) -> None:
    kept = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            if row.get("status") != "approved":
                continue
            fout.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["text"]},
                    {"role": "assistant", "content": json.dumps(row["label"])},
                ],
                "meta": {"template": row["template"], "source_id": row["id"]},
            }) + "\n")
            kept += 1
    print(f"exported {kept} approved examples to {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["label", "export"])
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    (label if a.command == "label" else export)(a.src, a.dst)
