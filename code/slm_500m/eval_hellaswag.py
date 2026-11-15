"""HellaSwag (validation, 10,042 items) scored the way lm-evaluation-harness does it.

For each item the model sees a context and 4 candidate endings. We compute the
log-probability of each ending given the context and pick the highest:
  acc       : argmax of the summed log-prob of the ending tokens
  acc_norm  : argmax of summed log-prob divided by the ending's length in characters
              (removes the bias toward short endings; this is the number usually reported)
Random guessing gives 25%.

    python eval_hellaswag.py --ckpt out/p500m/ckpt_latest.pt
    python eval_hellaswag.py --ckpt out/p500m/ckpt_latest.pt --limit 1000   # quick estimate
    python eval_hellaswag.py --ckpt ... --jsonl hellaswag_val.jsonl          # offline copy
"""
import argparse
import json
import re

import torch
import torch.nn.functional as F

from chat import load


def preprocess(text):
    # Same cleanup as lm-eval-harness's hellaswag task (WikiHow artefacts).
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def items(args):
    if args.jsonl:
        rows = [json.loads(l) for l in open(args.jsonl)]
    else:
        from datasets import load_dataset
        rows = load_dataset("Rowan/hellaswag", split="validation")
    for i, r in enumerate(rows):
        if args.limit and i >= args.limit:
            break
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        yield (preprocess(r["activity_label"] + ": " + ctx),
               [preprocess(e) for e in r["endings"]], int(r["label"]))


@torch.no_grad()
def score(model, tok, ctx, endings, device):
    """Return (sum_logprob, sum_logprob / n_chars) for each ending."""
    ctx_ids = tok.encode(ctx).ids
    seqs, spans = [], []
    for e in endings:
        cont = tok.encode(" " + e).ids
        full = (ctx_ids + cont)[-model.cfg.max_seq_len:]
        seqs.append(full)
        spans.append(len(cont))
    T = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    logits, _ = model(x.to(device))
    logp = F.log_softmax(logits.float(), dim=-1)
    out = []
    for i, (s, n) in enumerate(zip(seqs, spans)):
        L = len(s)
        # Tokens at positions L-n .. L-1 are predicted by positions L-n-1 .. L-2.
        tgt = torch.tensor(s[L - n:], device=device)
        lp = logp[i, L - n - 1:L - 1].gather(-1, tgt[:, None]).sum().item()
        out.append((lp, lp / len(endings[i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jsonl", help="local hellaswag_val.jsonl instead of the HF dataset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    model, tok = load(args.ckpt, device)

    n = acc = acc_norm = 0
    for ctx, endings, label in items(args):
        s = score(model, tok, ctx, endings, device)
        acc += max(range(4), key=lambda j: s[j][0]) == label
        acc_norm += max(range(4), key=lambda j: s[j][1]) == label
        n += 1
        if n % 500 == 0:
            print(f"{n}: acc {acc / n:.4f}  acc_norm {acc_norm / n:.4f}")
    print(f"HellaSwag ({n} items): acc {acc / n:.4f} | acc_norm {acc_norm / n:.4f} | random 0.25")


if __name__ == "__main__":
    main()
