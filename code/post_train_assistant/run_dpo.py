"""Step 4: DPO from the SFT checkpoint against a frozen copy of itself."""
import argparse
import random
import time

import torch

from common import (add_common_args, load_impl, load_model, pick_device, read_jsonl,
                    tokenizer_fns, write_json)


def main():
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument("--model", default="out/sft")
    p.add_argument("--prefs", default="data/prefs.jsonl")
    p.add_argument("--out", default="out/dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    impl, device = load_impl(args.impl), pick_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy, tok = load_model(args.model, device)
    encode, im_start, im_end, nl, pad_id = tokenizer_fns(tok, policy, impl)
    ref_model = impl.dpo.freeze_copy(policy)           # the reference is the starting policy

    pairs = read_jsonl(args.prefs)
    random.Random(args.seed).shuffle(pairs)
    examples = [impl.dpo.build_dpo_example(x, encode, im_start, im_end, nl) for x in pairs]
    n_eval = max(1, int(len(examples) * args.eval_frac))
    eval_ex, train_ex = examples[:n_eval], examples[n_eval:]
    print(f"device {device}; {len(train_ex)} training pairs, {len(eval_ex)} held-out pairs, "
          f"beta {args.beta}, lr {args.lr}")

    t0 = time.time()
    history = impl.dpo.train_dpo(
        policy, ref_model, train_ex, eval_ex, pad_id=pad_id, max_len=args.max_len,
        beta=args.beta, lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
        log_every=args.log_every, device=device, seed=args.seed)
    policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    write_json(f"{args.out}/dpo_history.json", {"args": vars(args), "history": history,
                                                "seconds": time.time() - t0})
    print(f"done in {time.time() - t0:.0f}s; saved to {args.out}")


if __name__ == "__main__":
    main()
