"""Build the SFT set and the held-out prompt sets from a slice of HuggingFaceTB/smoltalk
(Apache-2.0). Streams the subset, so only the rows it reads are downloaded.

Writes to data/:
  sft_train.jsonl, sft_eval.jsonl   {"messages": [user, assistant]}
  pref_prompts.jsonl, test_prompts.jsonl   {"question", "constraint", "messages": [user]}
"""
import argparse
import random

from common import DEFAULT_MODEL, write_jsonl
from constraints import constraints_met_by, instruction_for

MENU = [{"kind": "max_words", "n": 20}, {"kind": "max_words", "n": 40},
        {"kind": "one_sentence"}, {"kind": "numbered_list", "n": 3}]


def with_instruction(question, constraint):
    return question.rstrip() + "\n" + instruction_for(constraint)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="openhermes-100k")
    p.add_argument("--tokenizer", default=DEFAULT_MODEL)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-eval", type=int, default=300)
    p.add_argument("--n-pref", type=int, default=400)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=200, help="question + answer token budget")
    p.add_argument("--max-question-tokens", type=int, default=80)
    p.add_argument("--relabel-prob", type=float, default=0.5)
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))

    need = args.n_train + args.n_eval + args.n_pref + args.n_test
    stream = load_dataset("HuggingFaceTB/smoltalk", args.subset, split="train", streaming=True)
    kept, seen, read = [], set(), 0
    for ex in stream:
        read += 1
        m = ex["messages"]
        if len(m) != 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant":
            continue                                     # single-turn, no system prompt
        q, a = m[0]["content"].strip(), m[1]["content"].strip()
        if not q or not a or q in seen:
            continue                                     # exact-duplicate prompts
        nq = ntok(q)
        if nq > args.max_question_tokens or nq + ntok(a) > args.max_tokens:
            continue
        seen.add(q)
        kept.append((q, a))
        if len(kept) >= need:
            break
    if len(kept) < need:
        raise SystemExit(f"only {len(kept)} usable rows in {read}; lower the --n-* values")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    cut1 = args.n_train
    cut2 = cut1 + args.n_eval
    cut3 = cut2 + args.n_pref

    def sft_rows(pairs):
        """Hindsight relabeling: when an answer already satisfies a constraint, sometimes
        append that instruction to the question, so the data teaches the instruction."""
        rows, n_relabeled = [], 0
        for q, a in pairs:
            options = constraints_met_by(a)
            if options and rng.random() < args.relabel_prob:
                q = with_instruction(q, rng.choice(options))
                n_relabeled += 1
            rows.append({"messages": [{"role": "user", "content": q},
                                      {"role": "assistant", "content": a}]})
        return rows, n_relabeled

    def prompt_rows(pairs):
        rows = []
        for i, (q, _) in enumerate(pairs):
            c = MENU[i % len(MENU)]
            rows.append({"question": q, "constraint": c,
                         "messages": [{"role": "user", "content": with_instruction(q, c)}]})
        return rows

    train, n_rel = sft_rows(kept[:cut1])
    evalset, _ = sft_rows(kept[cut1:cut2])
    write_jsonl(f"{args.out}/sft_train.jsonl", train)
    write_jsonl(f"{args.out}/sft_eval.jsonl", evalset)
    write_jsonl(f"{args.out}/pref_prompts.jsonl", prompt_rows(kept[cut2:cut3]))
    write_jsonl(f"{args.out}/test_prompts.jsonl", prompt_rows(kept[cut3:need]))
    print(f"read {read} rows of smoltalk/{args.subset}, kept {len(kept)}")
    print(f"sft_train {len(train)} ({n_rel} with a relabeled constraint), sft_eval {len(evalset)}, "
          f"pref_prompts {args.n_pref}, test_prompts {args.n_test}")


if __name__ == "__main__":
    main()
