"""Steps 1 and 5: evaluate one model on the held-out test prompts.

Reports the masked eval loss on the SFT eval set, the rubric pass-rate for greedy and
sampled decoding, per-check and per-constraint pass-rates, and mean answer length. With
--against it also reports the preference win-rate versus another model's saved outputs.
"""
import argparse
from collections import defaultdict

from common import (DEFAULT_MODEL, add_common_args, generate, load_impl, load_model,
                    pick_device, read_jsonl, tokenizer_fns, write_json, write_jsonl)
from constraints import instruction_for, words


def summarize(rows, key):
    n = len(rows)
    checks = defaultdict(int)
    by_kind = defaultdict(lambda: [0, 0])
    for r in rows:
        j = r[key]["judge"]
        for name, ok in j["checks"].items():
            checks[name] += ok
        tag = instruction_for(r["constraint"])
        by_kind[tag][0] += j["passed"]
        by_kind[tag][1] += 1
    return {
        "pass_rate": sum(r[key]["judge"]["passed"] for r in rows) / n,
        "mean_score": sum(r[key]["judge"]["score"] for r in rows) / n,
        "mean_words": sum(len(words(r[key]["text"])) for r in rows) / n,
        "checks": {k: v / n for k, v in checks.items()},
        "by_constraint": {k: f"{a}/{b}" for k, (a, b) in by_kind.items()},
    }


def win_rate(rows, other, key):
    """Share of prompts where this model's judge score beats the other's; ties count half."""
    other = {r["messages"][0]["content"]: r for r in other}
    win = tie = loss = 0
    for r in rows:
        a = r[key]["judge"]["score"]
        b = other[r["messages"][0]["content"]][key]["judge"]["score"]
        win, tie, loss = win + (a > b), tie + (a == b), loss + (a < b)
    n = win + tie + loss
    return {"win": win, "tie": tie, "loss": loss, "win_rate": (win + 0.5 * tie) / n}


def main():
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--name", required=True, help="label for the output files, e.g. base, sft, dpo")
    p.add_argument("--prompts", default="data/test_prompts.jsonl")
    p.add_argument("--sft-eval", default="data/sft_eval.jsonl")
    p.add_argument("--against", default=None, help="name of an earlier evaluation to compare with")
    p.add_argument("--out", default="out")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--show", type=int, default=3)
    args = p.parse_args()

    impl, device = load_impl(args.impl), pick_device(args.device)
    model, tok = load_model(args.model, device)
    encode, im_start, im_end, nl, pad_id = tokenizer_fns(tok, model, impl)

    evalset = [impl.data.render_chatml(r["messages"], encode, im_start, im_end, nl)
               for r in read_jsonl(args.sft_eval)]
    loss = impl.sft.eval_loss(model, evalset, pad_id=pad_id, max_len=args.max_len, device=device)

    rows = read_jsonl(args.prompts)
    prompts = [impl.data.render_prompt(r["messages"], encode, im_start, im_end, nl) for r in rows]
    for key, temp in (("greedy", 0.0), ("sampled", args.temperature)):
        outs = generate(model, tok, prompts, im_end_id=im_end, pad_id=pad_id, temperature=temp,
                        max_new_tokens=args.max_new_tokens, seed=args.seed, device=device)
        for r, (text, finished) in zip(rows, outs):
            r[key] = {"text": text, "finished": finished,
                      "judge": impl.judge.judge(r["question"], text, r["constraint"], finished)}

    report = {"name": args.name, "model": args.model, "n_prompts": len(rows),
              "sft_eval_loss": loss,
              "greedy": summarize(rows, "greedy"), "sampled": summarize(rows, "sampled")}
    if args.against:
        other = read_jsonl(f"{args.out}/gen_{args.against}.jsonl")
        report["vs_" + args.against] = {k: win_rate(rows, other, k) for k in ("greedy", "sampled")}
    write_jsonl(f"{args.out}/gen_{args.name}.jsonl", rows)
    write_json(f"{args.out}/eval_{args.name}.json", report)

    print(f"[{args.name}] SFT eval loss {loss:.4f}")
    for key in ("greedy", "sampled"):
        s = report[key]
        print(f"[{args.name}] {key:8s} pass-rate {s['pass_rate']:.3f}  mean score {s['mean_score']:.2f}/6  "
              f"mean words {s['mean_words']:.1f}")
        print("   checks: " + "  ".join(f"{k} {v:.2f}" for k, v in s["checks"].items()))
        print("   by constraint: " + "  ".join(f"{k} {v}" for k, v in s["by_constraint"].items()))
        if args.against:
            w = report["vs_" + args.against][key]
            print(f"   vs {args.against}: win {w['win']} tie {w['tie']} loss {w['loss']} "
                  f"-> win-rate {w['win_rate']:.3f}")
    for r in rows[: args.show]:
        print("-" * 70)
        print("USER:", r["messages"][0]["content"])
        print(f"{args.name.upper()} (greedy):", r["greedy"]["text"])


if __name__ == "__main__":
    main()
