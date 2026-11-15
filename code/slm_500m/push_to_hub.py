"""Write a model card and upload a converted HF folder to the Hugging Face Hub.

    hf auth login            # once; needs a token with write access
    python push_to_hub.py --folder hf/slm-500m-instruct --repo your-name/slm-500m-instruct \
        --tokens_trained 10e9 --hellaswag_acc_norm 0.xx --val_loss x.xx

Only numbers you pass in are written into the card; nothing is made up.
"""
import argparse
import json
import os

CARD = """---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
datasets:
{datasets}
tags:
- from-scratch
- llama
- small-language-model
---

# {name}

A {params_m:.0f}M-parameter Llama-architecture language model trained from scratch
as part of the [LLM From Scratch](https://vivek-rm-14286.github.io/llm-from-scratch/) course
(code: https://github.com/vivek-rm-14286/llm-from-scratch/tree/main/code/slm_500m).

{kind}

## Architecture

| | |
|---|---|
| Parameters | {params:,} (embeddings tied) |
| Layers / hidden / FFN | {layers} / {hidden} / {ffn} (SwiGLU) |
| Attention | {heads} query heads, {kv_heads} KV heads (GQA), head dim {head_dim}, RoPE theta {theta:g} |
| Context length | {ctx} tokens |
| Vocabulary | {vocab:,} (byte-level BPE) |
| Pretraining tokens | {tokens} |

## Evaluation

{evals}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}")
msgs = [{{"role": "user", "content": "Explain photosynthesis in two sentences."}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
out = model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

## Limitations

This is a small educational model. It makes factual errors, has weak reasoning and
math, may produce biased or unsafe text, and has had no safety tuning beyond SFT
on public instruction data. Do not use it for anything that matters.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo", required=True, help="user-or-org/name")
    ap.add_argument("--tokens_trained", type=float)
    ap.add_argument("--datasets", nargs="+",
                    default=["HuggingFaceFW/fineweb-edu", "HuggingFaceTB/smoltalk"])
    ap.add_argument("--val_loss", type=float)
    ap.add_argument("--hellaswag_acc_norm", type=float)
    ap.add_argument("--base", action="store_true", help="card for a base (non-chat) model")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="write README.md, don't upload")
    args = ap.parse_args()

    with open(os.path.join(args.folder, "config.json")) as f:
        c = json.load(f)
    from safetensors import safe_open
    params = 0
    for fn in os.listdir(args.folder):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(args.folder, fn), "pt") as f:
                # lm_head is tied to the embedding (and usually not even saved): skip it.
                params += sum(_prod(f.get_slice(k).get_shape()) for k in f.keys() if k != "lm_head.weight")

    evals = []
    if args.val_loss is not None:
        evals.append(f"- Validation loss (held-out FineWeb-Edu): {args.val_loss:.3f}")
    if args.hellaswag_acc_norm is not None:
        evals.append(f"- HellaSwag acc_norm (0-shot, validation): {args.hellaswag_acc_norm:.3f} (random = 0.25)")
    theta = c.get("rope_theta") or (c.get("rope_parameters") or {}).get("rope_theta", 10000.0)
    card = CARD.format(
        name=args.repo.split("/")[-1], repo=args.repo, params=params, params_m=params / 1e6,
        kind=("Base model: pretrained only, it continues text and does not follow instructions."
              if args.base else
              "Instruction-tuned with SFT using the ChatML template (`<|im_start|>` / `<|im_end|>`)."),
        layers=c["num_hidden_layers"], hidden=c["hidden_size"], ffn=c["intermediate_size"],
        heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"],
        head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"], theta=theta,
        ctx=c["max_position_embeddings"], vocab=c["vocab_size"],
        tokens=f"{args.tokens_trained / 1e9:.0f}B" if args.tokens_trained else "not stated",
        evals="\n".join(evals) or "Not evaluated yet.",
        datasets="\n".join(f"- {d}" for d in args.datasets),
    )
    with open(os.path.join(args.folder, "README.md"), "w") as f:
        f.write(card)
    print(f"wrote {args.folder}/README.md")
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.folder, repo_id=args.repo, commit_message="Upload model")
    print(f"https://huggingface.co/{args.repo}")


def _prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n


if __name__ == "__main__":
    main()
