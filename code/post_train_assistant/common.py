"""Shared plumbing for the scripts: implementation loading, device, model, generation, I/O.
Nothing here is part of the exercise."""
import importlib
import json
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def load_impl(name):
    """Import data / sft / dpo / judge from solution/ or starter/."""
    if name not in ("solution", "starter"):
        raise SystemExit("--impl must be 'solution' or 'starter'")
    return SimpleNamespace(**{m: importlib.import_module(f"{name}.{m}")
                              for m in ("data", "sft", "dpo", "judge")})


def add_common_args(parser):
    parser.add_argument("--impl", default="solution", choices=["solution", "starter"])
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def pick_device(name=None):
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def release_cache(device):
    """Return cached GPU buffers to the system. On Apple Silicon the MPS allocator keeps
    a buffer for every tensor shape it has seen, and every batch here has a different
    length: without this call a 135M-parameter model grows past 6 GB and starts swapping."""
    if str(device).startswith("mps"):
        torch.mps.empty_cache()


def load_model(path, device):
    """Load a causal LM and its tokenizer in float32 (training uses autocast on CUDA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    return model.to(device), tok


def tokenizer_fns(tok, model, impl):
    """Returns (encode, im_start_id, im_end_id, nl_ids, pad_id), resizing embeddings if needed."""
    im_start, im_end = impl.data.ensure_chatml_tokens(tok, model)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return encode, im_start, im_end, encode("\n"), pad_id


@torch.no_grad()
def generate(model, tok, prompts, *, im_end_id, pad_id, max_new_tokens=96, temperature=0.0,
             top_p=0.95, batch_size=32, seed=0, device="cpu"):
    """Batched generation with left padding. prompts: list of token-id lists.

    temperature 0 means greedy. Returns a list of (text, finished): finished is True
    only if the model emitted <|im_end|> before max_new_tokens.
    """
    torch.manual_seed(seed)
    was_training = model.training
    model.eval()
    stops = {im_end_id, tok.eos_token_id}
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        T = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, T - len(prompts[i]):] = torch.tensor(prompts[i])
            att[r, T - len(prompts[i]):] = 1
        kwargs = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 \
            else dict(do_sample=False)
        gen = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                             max_new_tokens=max_new_tokens, eos_token_id=sorted(stops),
                             pad_token_id=pad_id, **kwargs)
        for r, i in enumerate(idx):
            new = gen[r, T:].tolist()
            cut = next((k for k, t in enumerate(new) if t in stops), None)
            finished = cut is not None and new[cut] == im_end_id
            text = tok.decode(new if cut is None else new[:cut], skip_special_tokens=False)
            out[i] = (text.strip(), finished)
        release_cache(device)
    model.train(was_training)
    return out


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
