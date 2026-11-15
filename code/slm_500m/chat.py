"""Talk to your model, with a KV cache, temperature and top-p sampling.

    python chat.py --ckpt out/sft/ckpt_latest.pt                 # interactive ChatML chat
    python chat.py --ckpt out/p500m/ckpt_latest.pt --base        # raw text completion
    python chat.py --ckpt out/tiny_sft/ckpt_latest.pt --prompt "What is 3 plus 4?"   # one-shot
"""
import argparse
import sys

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from model import Transformer


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return model.to(device, dtype).eval(), Tokenizer.from_file(ckpt["tokenizer"])


def chat_prompt_ids(history, tok, system=None):
    im_start, im_end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    nl = tok.encode("\n").ids
    ids = []
    msgs = ([{"role": "system", "content": system}] if system else []) + history
    for m in msgs:
        ids += [im_start] + tok.encode(m["role"] + "\n").ids + tok.encode(m["content"]).ids + [im_end] + nl
    return ids + [im_start] + tok.encode("assistant\n").ids  # generation prompt


def stream(model, tok, ids, args, stop_ids):
    """Generate and print incrementally. Decoding the growing list (not single tokens)
    keeps multi-byte UTF-8 characters that span tokens intact."""
    keep = model.cfg.max_seq_len - args.max_new_tokens  # leave room to generate
    x = torch.tensor([ids[-keep:]], device=args.device)
    out, printed = [], ""
    for t in model.generate(x, args.max_new_tokens, args.temperature, args.top_p, stop_ids):
        if t in stop_ids:
            break
        out.append(t)
        text = tok.decode(out)
        if not text.endswith("�"):  # wait until a partial character completes
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text
    print()
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", action="store_true", help="plain completion, no chat template")
    ap.add_argument("--prompt", help="answer one prompt and exit")
    ap.add_argument("--system", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.device = torch.device(args.device)

    model, tok = load(args.ckpt, args.device)
    eot, im_end = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|im_end|>")
    args.max_new_tokens = min(args.max_new_tokens, model.cfg.max_seq_len // 2)

    if args.base:
        prompts = [args.prompt] if args.prompt else iter(lambda: input("\nprompt> "), None)
        for p in prompts:
            sys.stdout.write(p)
            stream(model, tok, tok.encode(p).ids, args, {eot})
        return

    history = []
    while True:
        try:
            user = args.prompt or input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user.strip() in {"/reset", "/clear"}:
            history = []
            continue
        history.append({"role": "user", "content": user})
        sys.stdout.write("bot> ")
        reply = stream(model, tok, chat_prompt_ids(history, tok, args.system), args, {im_end, eot})
        history.append({"role": "assistant", "content": reply})
        if args.prompt:
            break


if __name__ == "__main__":
    main()
