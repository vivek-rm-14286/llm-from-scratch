#!/usr/bin/env python3
"""Evaluation harness: retrieval metrics per mode, ACL leak check, and (optionally) generation
quality with an LLM judge.

    python eval.py                       # retrieval only: recall@1/3/5, MRR, ACL leaks per mode
    python eval.py --generate            # + answers, citation validity, abstention, judge faithfulness

Stop the FastAPI app first when using Qdrant local mode: only one process can open the index.
Judge endpoint: JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY (default: same as LLM_*). Use a model
at least as strong as the generator, and spot-check its verdicts by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import generate
import settings
from ingest import ingest
from retrieve import MODES, Retriever, unique_docs

JUDGE_PROMPT = """You are grading a support assistant's answer.

Question: {question}

Sources given to the assistant:
{sources}

Reference answer (written by a human): {reference}

Assistant answer:
{answer}

Tasks:
1. List each factual claim in the assistant answer and whether the SOURCES support it.
2. Decide whether the assistant answer is correct and complete compared with the reference.
Return ONLY JSON: {{"claims": [{{"claim": "...", "supported": true}}], "correct": true}}"""


def load_golden(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    return next((1.0 / i for i, d in enumerate(ranked, start=1) if d in relevant), 0.0)


def eval_retrieval(retriever: Retriever, golden: list[dict], k: int) -> dict:
    report = {}
    for mode in MODES:
        r1, r3, rk, mrr, leaks = [], [], [], [], 0
        for g in golden:
            results = retriever.search(g["question"], g["user_access"], k=k, mode=mode)
            leaks += sum(c["access_level"] > settings.ACCESS_LEVELS[g["user_access"]] for c in results)
            if not g["relevant"]:
                continue
            docs, rel = unique_docs(results), set(g["relevant"])
            r1.append(recall_at_k(docs, rel, 1))
            r3.append(recall_at_k(docs, rel, 3))
            rk.append(recall_at_k(docs, rel, k))
            mrr.append(reciprocal_rank(docs, rel))
        report[mode] = {"recall@1": statistics.mean(r1), "recall@3": statistics.mean(r3),
                        f"recall@{k}": statistics.mean(rk), "mrr": statistics.mean(mrr), "acl_leaks": leaks}
    return report


def judge(llm, model: str, g: dict, chunks: list[dict], answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=g["question"], sources=generate.format_sources(chunks),
                                 reference=g["reference_answer"], answer=answer)
    resp = llm.chat.completions.create(model=model, temperature=0.0, max_tokens=800,
                                       messages=[{"role": "user", "content": prompt}])
    text = resp.choices[0].message.content or ""
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
        claims = data.get("claims", [])
        faith = sum(bool(c.get("supported")) for c in claims) / len(claims) if claims else 1.0
        return {"faithfulness": faith, "correct": bool(data.get("correct")), "judge_ok": True}
    except (ValueError, AttributeError, TypeError):
        return {"faithfulness": None, "correct": None, "judge_ok": False}


def eval_generation(retriever: Retriever, golden: list[dict], k: int, mode: str) -> tuple[dict, list[dict]]:
    from openai import OpenAI

    gen_llm = generate.get_llm()
    judge_llm = OpenAI(base_url=os.getenv("JUDGE_BASE_URL", settings.LLM_BASE_URL),
                       api_key=os.getenv("JUDGE_API_KEY", settings.LLM_API_KEY))
    judge_model = os.getenv("JUDGE_MODEL", settings.LLM_MODEL)
    rows = []
    for g in golden:
        chunks = retriever.search(g["question"], g["user_access"], k=k, mode=mode)
        out = generate.answer(g["question"], chunks, llm=gen_llm)
        row = {"id": g["id"], "question": g["question"], "access": g["user_access"],
               "answerable": g["answerable"], **out}
        if g["answerable"] and not out["abstained"]:
            row.update(judge(judge_llm, judge_model, g, chunks, out["answer"]))
        rows.append(row)
        print(f"{g['id']} abstained={out['abstained']} cited={[c['n'] for c in out['citations']]} "
              f"faith={row.get('faithfulness')} correct={row.get('correct')}")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    judged = [r for r in answerable if r.get("judge_ok")]
    summary = {
        "n": len(rows),
        "answered_with_citation": sum(bool(r["citations"]) for r in answerable if not r["abstained"])
                                  / max(sum(not r["abstained"] for r in answerable), 1),
        "invalid_citation_rate": sum(bool(r["invalid_citations"]) for r in rows) / len(rows),
        "false_abstention_rate": sum(r["abstained"] for r in answerable) / max(len(answerable), 1),
        "correct_abstention_rate": sum(r["abstained"] for r in unanswerable) / max(len(unanswerable), 1),
        "faithfulness": statistics.mean(r["faithfulness"] for r in judged) if judged else None,
        "correctness": statistics.mean(r["correct"] for r in judged) if judged else None,
        "judge_parse_failures": sum(1 for r in answerable if r.get("judge_ok") is False),
    }
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default=str(settings.DATA_DIR / "golden.jsonl"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--mode", default="hybrid_rerank", choices=MODES, help="mode used for --generate")
    ap.add_argument("--reingest", action="store_true")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    client = settings.get_qdrant_client()
    if args.reingest or not client.collection_exists(settings.COLLECTION) or not settings.CHUNKS_PATH.exists():
        print(ingest(client=client))
    retriever = Retriever(client=client)
    golden = load_golden(args.golden)

    report = {"retrieval": eval_retrieval(retriever, golden, args.k)}
    print(f"\n{'mode':15s} " + " ".join(f"{m:>9s}" for m in next(iter(report['retrieval'].values()))))
    for mode, m in report["retrieval"].items():
        print(f"{mode:15s} " + " ".join(f"{v:9.3f}" if isinstance(v, float) else f"{v:9d}" for v in m.values()))

    if args.generate:
        summary, rows = eval_generation(retriever, golden, args.k, args.mode)
        report["generation"] = summary
        report["generation_rows"] = rows
        print("\ngeneration:", json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
