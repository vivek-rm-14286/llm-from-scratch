"""Step 3: build an on-policy preference dataset.

For each held-out prompt, sample several completions from the SFT model, score each with
the programmatic judge, and keep (chosen, rejected) = (best, worst) when they differ.
"""
import argparse
from collections import Counter

from common import (add_common_args, generate, load_impl, load_model, pick_device, read_jsonl,
                    tokenizer_fns, write_json, write_jsonl)
from constraints import instruction_for, words


def main():
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument("--model", default="out/sft")
    p.add_argument("--prompts", default="data/pref_prompts.jsonl")
    p.add_argument("--out", default="data/prefs.jsonl")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--min-gap", type=float, default=1)
    p.add_argument("--llm-judge", action="store_true",
                   help="add an LLM score in [0, 1] as a tie-breaker (see llm_judge.py)")
    args = p.parse_args()

    impl, device = load_impl(args.impl), pick_device(args.device)
    model, tok = load_model(args.model, device)
    encode, im_start, im_end, nl, pad_id = tokenizer_fns(tok, model, impl)
    rows = read_jsonl(args.prompts)
    prompts = [impl.data.render_prompt(r["messages"], encode, im_start, im_end, nl) for r in rows]

    candidates = [[] for _ in rows]
    for k in range(args.samples):                      # one pass per sample, different seed
        outs = generate(model, tok, prompts, im_end_id=im_end, pad_id=pad_id,
                        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                        seed=args.seed + k, device=device)
        for cands, (text, finished) in zip(candidates, outs):
            cands.append((text, finished))

    if args.llm_judge:
        from llm_judge import llm_score

    pairs, decided_by, n_cand, n_pass = [], Counter(), 0, 0
    for r, cands in zip(rows, candidates):
        judged = [impl.judge.judge(r["question"], t, r["constraint"], f) for t, f in cands]
        scores = [j["score"] for j in judged]
        n_cand += len(judged)
        n_pass += sum(j["passed"] for j in judged)
        if args.llm_judge:
            extra = [llm_score(r["question"], instruction_for(r["constraint"]), t) or 0.0
                     for t, _ in cands]
            scores = [s + e for s, e in zip(scores, extra)]   # e < 1 check: tie-breaker only
        picked = impl.judge.pick_pair(scores, min_gap=args.min_gap)
        if picked is None or cands[picked[0]][0] == cands[picked[1]][0]:
            continue
        w, l = picked
        for name, ok in judged[w]["checks"].items():
            if ok and not judged[l]["checks"][name]:
                decided_by[name] += 1
        pairs.append({"question": r["question"], "constraint": r["constraint"],
                      "messages": r["messages"],
                      "chosen": cands[w][0], "rejected": cands[l][0],
                      "chosen_score": scores[w], "rejected_score": scores[l]})

    write_jsonl(args.out, pairs)
    mw = lambda key: sum(len(words(x[key])) for x in pairs) / max(1, len(pairs))
    stats = {"prompts": len(rows), "samples_per_prompt": args.samples, "pairs": len(pairs),
             "candidate_pass_rate": n_pass / n_cand,
             "chosen_mean_words": mw("chosen"), "rejected_mean_words": mw("rejected"),
             "chosen_shorter": sum(len(words(x["chosen"])) < len(words(x["rejected"])) for x in pairs),
             "checks_separating_pairs": dict(decided_by)}
    write_json(args.out.replace(".jsonl", "_stats.json"), stats)
    print(f"{len(pairs)} pairs from {len(rows)} prompts x {args.samples} samples "
          f"(candidate pass-rate {stats['candidate_pass_rate']:.3f})")
    print(f"mean words: chosen {stats['chosen_mean_words']:.1f}, rejected {stats['rejected_mean_words']:.1f}; "
          f"chosen is shorter in {stats['chosen_shorter']}/{len(pairs)} pairs")
    print("checks that separate chosen from rejected:", dict(decided_by))


if __name__ == "__main__":
    main()
