# pip install torch transformers   (CPU, a few seconds after the 270 MB download)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

questions = ["What is the capital of France?",
             "Name a primary color.",
             "Give one word that rhymes with 'cat', then explain in a sentence why rhymes are easy to remember."]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True) for q in questions]

def generate(side):
    tok.padding_side = side
    enc = tok(prompts, return_tensors="pt", padding=True)        # pads to the longest prompt
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=12, do_sample=False)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

single = []
for p in prompts:                                                # reference: one prompt at a time
    enc = tok(p, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=12, do_sample=False)
    single.append(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))

for side in ("left", "right"):
    batch = generate(side)
    print(f"--- padding_side={side}")
    for s, b in zip(single, batch):
        print(f"{'same     ' if s == b else 'DIFFERENT'} | {b!r}")
