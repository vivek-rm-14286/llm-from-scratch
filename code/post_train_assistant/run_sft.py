"""Step 2: supervised fine-tuning of a base model on ChatML-rendered conversations."""
import argparse
import random
import time

import torch

from common import (DEFAULT_MODEL, add_common_args, load_impl, load_model, pick_device,
                    read_jsonl, tokenizer_fns, write_json)


def main():
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train", default="data/sft_train.jsonl")
    p.add_argument("--eval", default="data/sft_eval.jsonl")
    p.add_argument("--out", default="out/sft")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--limit", type=int, default=0, help="use only the first N training rows")
    args = p.parse_args()

    impl, device = load_impl(args.impl), pick_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tok = load_model(args.model, device)
    encode, im_start, im_end, nl, pad_id = tokenizer_fns(tok, model, impl)
    print(f"device {device}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
          f"<|im_start|>={im_start} <|im_end|>={im_end}, vocab {len(tok)}")

    def render(path, limit=0):
        rows = read_jsonl(path)
        rows = rows[:limit] if limit else rows
        data = [impl.data.render_chatml(r["messages"], encode, im_start, im_end, nl) for r in rows]
        # drop examples whose targets were all cut off by truncation
        return [d for d in data if any(t != -100 for t in d[1][: args.max_len])]

    train, evalset = render(args.train, args.limit), render(args.eval)
    n_tok = sum(len(i) for i, _ in train)
    n_tgt = sum(sum(t != -100 for t in l) for _, l in train)
    print(f"train {len(train)} conversations, {n_tok} tokens, {n_tgt} trained tokens "
          f"({100 * n_tgt / n_tok:.0f}%); eval {len(evalset)}")

    t0 = time.time()
    history = impl.sft.train_sft(
        model, train, evalset, pad_id=pad_id, max_len=args.max_len, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, warmup_steps=args.warmup_steps,
        eval_every=args.eval_every, device=device, seed=args.seed)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    write_json(f"{args.out}/sft_history.json", {"args": vars(args), "history": history,
                                                "seconds": time.time() - t0})
    print(f"eval loss {history[0]['eval_loss']:.4f} -> {history[-1]['eval_loss']:.4f} "
          f"in {time.time() - t0:.0f}s; saved to {args.out}")


if __name__ == "__main__":
    main()
