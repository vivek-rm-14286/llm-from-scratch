#!/usr/bin/env python3
"""Evaluation harness: retrieval metrics per mode, ACL leak check, and (optionally) generation
quality with an LLM judge.

    python eval.py                       # retrieval only: recall@1/3/5, MRR, ACL leaks per mode and type
    python eval.py --generate            # + answers, citation validity, abstention, judge faithfulness
    python eval.py --baseline ci/baseline.json   # CI gate: exit 1 on any ACL leak or a recall/MRR drop

Stop the FastAPI app first when using Qdrant local mode: only one process can open the index.
Judge endpoint: JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY (default: same as LLM_*). Use a model
at least as strong as the generator, and spot-check its verdicts by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import generate
import settings
from ingest import ingest
from retrieve import MODES, Retriever

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


def ranked_keys(results: list[dict], relevant: set[str]) -> list[str]:
    """Collapse chunk results to the granularity of the labels, in rank order. A label is either one
    section ("doc:alerting#maintenance-windows") or a whole document ("ticket:T-1006"). A chunk whose
    section is labeled counts as that section; any other chunk counts as its document."""
    seen, out = set(), []
    for r in results:
        key = r["section_id"] if r.get("section_id") in relevant else r["doc_id"]
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    return next((1.0 / i for i, d in enumerate(ranked, start=1) if d in relevant), 0.0)


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def eval_retrieval(retriever: Retriever, golden: list[dict], k: int, modes=MODES) -> dict:
    """Per mode: recall@1/3/k and MRR over questions that have relevant sources, the same split by
    question type, and the number of retrieved chunks above the asker's access level (must be 0)."""
    report = {}
    for mode in modes:
        rows, leaks = [], 0
        for g in golden:
            results = retriever.search(g["question"], g["user_access"], k=k, mode=mode)
            leaks += sum(c["access_level"] > settings.ACCESS_LEVELS[g["user_access"]] for c in results)
            if not g["relevant"]:
                continue  # nothing to find: these questions only feed the leak count and --generate
            rel = set(g["relevant"])
            keys = ranked_keys(results, rel)
            rows.append({"type": g.get("type", "all"), "r1": recall_at_k(keys, rel, 1),
                         "r3": recall_at_k(keys, rel, 3), "rk": recall_at_k(keys, rel, k),
                         "rr": reciprocal_rank(keys, rel)})
        by_type = {}
        for t in sorted({r["type"] for r in rows}):
            sub = [r for r in rows if r["type"] == t]
            by_type[t] = {"n": len(sub), f"recall@{k}": _mean([r["rk"] for r in sub]),
                          "mrr": _mean([r["rr"] for r in sub])}
        report[mode] = {"recall@1": _mean([r["r1"] for r in rows]), "recall@3": _mean([r["r3"] for r in rows]),
                        f"recall@{k}": _mean([r["rk"] for r in rows]), "mrr": _mean([r["rr"] for r in rows]),
                        "acl_leaks": leaks, "n_scored": len(rows), "by_type": by_type}
    return report


def print_retrieval(report: dict, k: int) -> None:
    cols = ["recall@1", "recall@3", f"recall@{k}", "mrr", "acl_leaks"]
    print(f"\n{'mode':15s} " + " ".join(f"{c:>9s}" for c in cols))
    for mode, m in report.items():
        print(f"{mode:15s} " + " ".join(f"{m[c]:9.3f}" if isinstance(m[c], float) else f"{m[c]:9d}" for c in cols))
    types = list(next(iter(report.values()))["by_type"])
    for metric in (f"recall@{k}", "mrr"):
        print(f"\n{metric} by question type")
        print(f"{'mode':15s} " + " ".join(f"{t:>11s}" for t in types))
        for mode, m in report.items():
            print(f"{mode:15s} " + " ".join(f"{m['by_type'][t][metric]:11.3f}" for t in types))
        if metric == "mrr":
            print(f"{'n':15s} " + " ".join(f"{next(iter(report.values()))['by_type'][t]['n']:11d}" for t in types))


def gate(report: dict, baseline: dict, k: int, max_drop: float) -> list[str]:
    """CI gate: any ACL leak fails; recall@k or MRR dropping more than max_drop below the committed
    baseline fails. Retrieval here is deterministic, so no noise allowance is needed beyond max_drop."""
    failures = []
    for mode, m in report.items():
        if m["acl_leaks"]:
            failures.append(f"{mode}: {m['acl_leaks']} retrieved chunk(s) above the asker's access level")
        for metric in (f"recall@{k}", "mrr"):
            base = baseline.get(mode, {}).get(metric)
            if base is not None and m[metric] < base - max_drop:
                failures.append(f"{mode}: {metric} {base:.3f} -> {m[metric]:.3f} (allowed drop {max_drop})")
    return failures


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
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES, help="retrieval modes to score")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--mode", default="hybrid_rerank", choices=MODES, help="mode used for --generate")
    ap.add_argument("--reingest", action="store_true")
    ap.add_argument("--out", default="eval_report.json")
    ap.add_argument("--baseline", help="JSON written by --write-baseline; exit 1 on a leak or a regression")
    ap.add_argument("--max-drop", type=float, default=0.02, help="allowed absolute drop vs --baseline")
    ap.add_argument("--write-baseline", help="write the retrieval metrics of this run to this path")
    args = ap.parse_args()

    client = settings.get_qdrant_client()
    if args.reingest or not client.collection_exists(settings.COLLECTION) or not settings.CHUNKS_PATH.exists():
        print(ingest(client=client))
    retriever = Retriever(client=client)
    golden = load_golden(args.golden)

    report = {"embed_model": settings.EMBED_MODEL, "rerank_model": settings.RERANK_MODEL,
              "chunks": len(retriever.chunks), "questions": len(golden),
              "retrieval": eval_retrieval(retriever, golden, args.k, args.modes)}
    print(f"{report['chunks']} chunks, {report['questions']} questions, "
          f"embedder {settings.EMBED_MODEL}, reranker {settings.RERANK_MODEL}")
    print_retrieval(report["retrieval"], args.k)

    if args.generate:
        summary, rows = eval_generation(retriever, golden, args.k, args.mode)
        report["generation"] = summary
        report["generation_rows"] = rows
        print("\ngeneration:", json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport written to {args.out}")
    client.close()

    if args.write_baseline:
        slim = {mode: {key: v for key, v in m.items() if key != "by_type"} for mode, m in report["retrieval"].items()}
        with open(args.write_baseline, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"baseline written to {args.write_baseline}")
    if args.baseline:
        with open(args.baseline) as f:
            failures = gate(report["retrieval"], json.load(f), args.k, args.max_drop)
        print("\nGATE " + ("FAILED:\n  " + "\n  ".join(failures) if failures else "PASSED"))
        sys.exit(1 if failures else 0)
    if any(m["acl_leaks"] for m in report["retrieval"].values()):
        sys.exit("ACL leak detected")  # a leak is never an acceptable result, with or without a baseline


if __name__ == "__main__":
    main()
