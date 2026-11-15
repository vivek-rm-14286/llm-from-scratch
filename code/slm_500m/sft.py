"""Supervised fine-tuning (instruction tuning) of the pretrained base model.

    python sft.py --init out/p500m/ckpt_latest.pt --out_dir out/sft
    python sft.py --tiny      # after `python train.py --tiny`; synthetic chats, CPU/MPS

Chat format (ChatML, the same template convert_to_hf.py writes for HF):

    <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    <|im_start|>user\nWhat is 2+2?<|im_end|>\n
    <|im_start|>assistant\n2+2 is 4.<|im_end|>\n

Loss masking: only the assistant's content and its closing <|im_end|> are trained
on (label != -100). The model learns to *answer*, not to imitate users, and
learns that <|im_end|> is where an answer stops.
"""
import argparse
import contextlib
import math
import os
import random
import time

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from model import Transformer

IGNORE = -100


def encode_chat(messages, tok, max_len):
    """Return (input_ids, labels) for one conversation."""
    im_start, im_end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    nl = tok.encode("\n", add_special_tokens=False).ids
    ids, labels = [], []
    for m in messages:
        header = [im_start] + tok.encode(m["role"] + "\n", add_special_tokens=False).ids
        body = tok.encode(m["content"], add_special_tokens=False).ids + [im_end]
        ids += header + body + nl
        train = m["role"] == "assistant"
        labels += [IGNORE] * len(header) + (body if train else [IGNORE] * len(body)) + [IGNORE] * len(nl)
    # Next-token prediction: position t predicts token t+1.
    ids, labels = ids[: max_len + 1], labels[: max_len + 1]
    return ids[:-1], labels[1:]


def load_conversations(args):
    if args.tiny:
        rng = random.Random(0)
        convs = []
        for _ in range(2000):
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            convs.append([{"role": "user", "content": f"What is {a} plus {b}?"},
                          {"role": "assistant", "content": f"{a} plus {b} equals {a + b}."}])
        return convs
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.dataset_config, split="train")
    if args.max_examples:
        ds = ds.shuffle(seed=0).select(range(min(args.max_examples, len(ds))))
    return [ex["messages"] for ex in ds]


def batches(examples, batch_size, pad_id, device, shuffle=True):
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order) - batch_size + 1, batch_size):
        chunk = [examples[j] for j in order[i:i + batch_size]]
        T = max(len(x) for x, _ in chunk)
        # Right-padding is safe with a causal mask: real tokens never attend to pads after them.
        x = torch.full((batch_size, T), pad_id, dtype=torch.long)
        y = torch.full((batch_size, T), IGNORE, dtype=torch.long)
        for r, (ids, lab) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids)
            y[r, :len(lab)] = torch.tensor(lab)
        yield x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="out/p500m/ckpt_latest.pt", help="pretrained checkpoint")
    ap.add_argument("--out_dir", default="out/sft")
    ap.add_argument("--dataset", default="HuggingFaceTB/smoltalk")
    ap.add_argument("--dataset_config", default="all")
    ap.add_argument("--max_examples", type=int, default=0, help="0 = all")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.tiny:
        args.init, args.out_dir = "out/tiny/ckpt_latest.pt", "out/tiny_sft"
        args.max_len, args.batch_size, args.grad_accum, args.epochs, args.lr = 64, 16, 1, 1, 1e-3

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device)
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
                else contextlib.nullcontext())
    torch.manual_seed(0)
    random.seed(0)

    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).train()
    tok = Tokenizer.from_file(ckpt["tokenizer"])
    pad_id = tok.token_to_id("<|pad|>")

    convs = load_conversations(args)
    examples = [encode_chat(c, tok, min(args.max_len, cfg.max_seq_len)) for c in convs]
    examples = [e for e in examples if any(l != IGNORE for l in e[1])]  # drop if answer truncated away
    random.shuffle(examples)
    n_val = max(args.batch_size, len(examples) // 100)
    val, train = examples[:n_val], examples[n_val:]
    print(f"{len(train):,} train / {len(val):,} val conversations; "
          f"{sum(len(x) for x, _ in train) / 1e6:.1f}M train tokens")

    steps_per_epoch = len(train) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay, fused=device.type == "cuda")

    def lr_at(s):  # warmup + cosine to 0
        if s < warmup:
            return args.lr * (s + 1) / warmup
        return 0.5 * args.lr * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        for x, y in batches(val, args.batch_size, pad_id, device, shuffle=False):
            with autocast:
                _, loss = model(x, y)
            tot, n = tot + loss.item(), n + 1
        model.train()
        return tot / max(n, 1)

    print(f"val loss before SFT: {evaluate():.4f}")
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        it = batches(train, args.batch_size, pad_id, device)
        for _ in range(steps_per_epoch):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            # Each micro-batch's loss is a mean over its own answer tokens; averaging
            # micro-batches is a close, simple approximation to a per-token mean.
            for _ in range(args.grad_accum):
                x, y = next(it)
                with autocast:
                    _, loss = model(x, y)
                (loss / args.grad_accum).backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} | loss {loss.item():.4f} | "
                      f"gnorm {gn.item():.2f} | lr {lr_at(step):.2e} | {time.time() - t0:.0f}s")
            step += 1
        print(f"epoch {epoch} done | val loss {evaluate():.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "ckpt_latest.pt")
    torch.save({"model": model.state_dict(), "model_config": cfg.to_dict(), "step": step,
                "tokenizer": ckpt["tokenizer"], "sft": True}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
