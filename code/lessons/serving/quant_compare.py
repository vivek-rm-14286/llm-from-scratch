# pip install torch transformers   (CPU, about a minute including the 270 MB download)
import argparse, contextlib, copy, io, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(name)
base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

with contextlib.redirect_stdout(io.StringIO()) as buf:
    import this                                                # the Zen of Python, as eval text
text = buf.getvalue() + "\n" + argparse.__doc__
ids = tok(text, return_tensors="pt")["input_ids"][:, :1024]

def rtn_(model, bits, group=64):
    """Round-to-nearest, symmetric, one scale per `group` weights, on every 2-D Linear weight."""
    qmax = 2 ** (bits - 1) - 1
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.weight.shape[1] % group == 0:
            w = m.weight.data
            wg = w.reshape(w.size(0), -1, group)
            scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
            m.weight.data = (torch.clamp(torch.round(wg / scale), -qmax - 1, qmax) * scale).reshape(w.shape)

@torch.no_grad()
def evaluate(model):
    out = model(ids, labels=ids)
    return math.exp(out.loss.item()), out.logits[0, :-1].argmax(-1)

ppl0, top0 = evaluate(base)
print(f"fp32   perplexity {ppl0:7.2f}   top-1 agreement with fp32 100.0%")
for bits in (8, 6, 4, 3):
    m = copy.deepcopy(base)
    rtn_(m, bits)
    ppl, top = evaluate(m)
    agree = (top == top0).float().mean().item() * 100
    print(f"int{bits}   perplexity {ppl:7.2f}   top-1 agreement with fp32 {agree:5.1f}%")
