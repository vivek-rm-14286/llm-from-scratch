# Continued Pretraining & Model Merging

SFT and LoRA teach a model *how to behave*. They're poor at teaching it a large body of new *knowledge*: a new language, a specialized field's vocabulary, an internal codebase. For that, you keep pretraining: more next-token prediction on raw domain text. And once you have several fine-tuned variants of the same base (one good at math, one at code, one at your domain), you can often combine them into one model without any training at all, by doing arithmetic on their weights. This page covers both: **continued pretraining** (when it's worth it, data mixing and replay, learning-rate re-warming, extending the tokenizer) and **model merging** (model soups, SLERP, task arithmetic, TIES, DARE), with from-scratch implementations over state dicts and the equivalent `mergekit` configs.

!!! abstract "What you'll learn"
    - When continued (domain-adaptive) pretraining beats SFT, and when it doesn't
    - Data mixing with replay to avoid forgetting, and how to re-warm the learning rate
    - Extending a tokenizer for a new language or domain, with embedding initialization code
    - The math of linear merging, SLERP, task vectors, TIES and DARE, implemented from scratch
    - A worked merge experiment you can run on a laptop CPU
    - Merging with `mergekit`, merging LoRA adapters correctly, and evaluating merges

## Continued pretraining

**Continued pretraining** (CPT), also called **domain-adaptive pretraining** (DAPT), takes a pretrained base model and keeps training it with the same next-token objective, on a new corpus. It's [pretraining](../pretraining/index.md) restarted from a checkpoint instead of from random weights.

### When it beats SFT

Fine-tuning on instruction pairs mostly changes *format and style*: it surfaces knowledge the model already has. It can't cheaply inject a lot of new knowledge, and trying to (with thousands of Q&A pairs about your domain) tends to teach the model to hallucinate confidently in the domain's style. CPT is the right tool when:

- **The domain's text is a large share of what the model doesn't know.** Legal, medical, financial filings, a specific programming language, or your company's documentation and code, at the scale of hundreds of millions to billions of tokens.
- **A new language.** A model trained mostly on English that you want fluent in, say, Tamil or Swahili. It needs to learn the language's statistics, which takes raw text, not instructions.
- **The vocabulary is unusual** enough that the tokenizer splits domain terms into many pieces (more on that below).

Famous examples: Code Llama (Llama 2 continued on hundreds of billions of code tokens), and many medical and multilingual variants of open models. The original DAPT paper showed that a second phase of pretraining on in-domain text improved downstream tasks across biomedical, CS, news and review domains, even for a model that was already strong.

When it's *not* worth it: if you have under ~10M tokens of domain text, if the task is mostly about format or behavior, or if retrieval would answer the questions (see [RAG](../rag/index.md)). Then SFT, LoRA ([LoRA & QLoRA](lora-qlora.md)) or RAG is cheaper. The [Fine-tuning overview](index.md) has a decision flowchart.

The usual pipeline: **base model → CPT on domain text → SFT (and maybe preference tuning) on domain-flavored instructions**. If you start from an instruct model instead, CPT will erode its chat behavior; you'll need to redo post-training or merge the chat ability back in (merging, below, is a popular way to do that).

### Data mixing with replay

Training only on domain text causes **catastrophic forgetting**: the model gets better at your domain and worse at everything else, including general reasoning and instruction following that you'll need later. The standard fix is **replay**: mix general-purpose data, similar to the original pretraining mix, into the CPT data.

- **How much?** Commonly 5-30% general data. Studies of continual pretraining found that even a small replay fraction removes most of the forgetting. More replay when the domain is very different from the original data (a new language), less when it's close.
- **What?** Ideally data resembling the original pretraining mix: high-quality web text, some code, some math. Open corpora like FineWeb-Edu, DCLM or Dolma (see [Data Pipelines](../pretraining/data.md)) are reasonable stand-ins when you don't have the original.
- **Annealing data.** Many recipes end CPT with a short phase on the highest-quality domain and general data while the learning rate decays.

A simple weighted sampler over sources:

```python title="mixture.py"
import math, random

def mixture_stream(sources, weights, seed=0):
    """Yield documents from several iterators, picking a source per document by weight.
    sources: dict name -> iterator of documents; weights: dict name -> float."""
    rng = random.Random(seed)
    names = list(sources)
    w = [weights[n] for n in names]
    while True:
        name = rng.choices(names, weights=w)[0]
        try:
            yield name, next(sources[name])
        except StopIteration:                    # source exhausted: drop it, renormalize
            i = names.index(name); names.pop(i); w.pop(i)
            if not names:
                return

def cpt_lr(step, max_lr=1e-4, min_lr=1e-5, warmup=1000, total=50_000):
    """Re-warm from ~0 to max_lr, then cosine to min_lr. max_lr is usually below the
    original pretraining peak (often 1/3 to 1/10 of it)."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))

import itertools
domain = (f"domain doc {i}" for i in range(10_000))
general = itertools.cycle(["general web doc", "general code doc", "general math doc"])  # replay
stream = mixture_stream({"domain": domain, "replay": general}, {"domain": 0.75, "replay": 0.25})
counts = {}
for name, doc in itertools.islice(stream, 10_000):
    counts[name] = counts.get(name, 0) + 1
print(counts, [round(cpt_lr(s), 7) for s in (0, 999, 1000, 25_000, 50_000)])
```

```text
{'replay': 2496, 'domain': 7504} [1e-07, 0.0001, 0.0001, 5.64e-05, 1e-05]
```

In a real pipeline each source yields token chunks from a pre-tokenized shard, and the weights are chosen per source (domain, web, code, math).

### Learning-rate re-warming

A released base model has usually finished its schedule with a decayed, tiny learning rate. If you continue at that tiny LR, it learns the new domain very slowly. If you jump straight to the original peak, you get a loss spike and lose more of what it knew. The standard recipe (studied in detail in work on continual pretraining):

1. **Re-warm**: linearly increase the LR from near zero over roughly the first 0.5-1% of steps.
2. **Peak below the original**: often around 1/3 to 1/10 of the original pretraining peak. Higher learns the domain faster and forgets more.
3. **Decay** with cosine (shown in `cpt_lr` above) or a WSD schedule (see [The Training Loop](../pretraining/training.md#warmup-stable-decay-wsd)), ending near the original final LR.

Expect general-domain loss to rise at the start of CPT even with replay and recover partially as the LR decays. Track both domain and general validation loss throughout.

If you don't have the original optimizer state (you usually don't), start fresh AdamW moments; warmup is what keeps the first steps stable.

### Tokenizer extension

A tokenizer trained mostly on English splits text in a new script or a specialized domain into many short pieces. More tokens per word means slower training and inference, shorter effective context, and harder learning. **Tokenizer extension** adds new tokens, then trains their embeddings during CPT.

Two ways to get the new tokens:

- **Add whole words or terms** with `tokenizer.add_tokens(...)`. Simple, good for a few hundred to a few thousand domain terms. Added tokens are matched before the BPE step, so they behave a little differently from learned merges (for example around leading spaces).
- **Train a new BPE tokenizer** on domain text, take the tokens not already in the original vocabulary, and merge them into the original vocab and merge list. This is what language-extension projects (for example Chinese LLaMA) did, and is the right choice for a new language. See [Tokenization](../llm/tokenization.md) for training BPE.

**Initializing the new embeddings** matters. Random init makes the new rows wildly out of distribution: early training can produce huge losses and the model may sample new tokens erratically. Better: initialize each new token's embedding as the **mean of the embeddings of the pieces the old tokenizer split it into**. Then the new token starts out "meaning" roughly what its pieces meant together:

```python title="extend_tokenizer.py"
import torch
from transformers import AutoTokenizer

def add_tokens_with_init(model, tokenizer, new_tokens):
    """Add tokens and initialize each new embedding as the mean of the embeddings of the
    pieces the OLD tokenizer split it into. Works for tied and untied output heads."""
    pieces = {t: tokenizer(t, add_special_tokens=False).input_ids for t in new_tokens}  # before adding
    n_added = tokenizer.add_tokens(new_tokens)
    if n_added == 0:
        return 0
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    emb_in = model.get_input_embeddings().weight
    emb_out = model.get_output_embeddings().weight
    tied = emb_out.data_ptr() == emb_in.data_ptr()
    with torch.no_grad():
        for tok, ids in pieces.items():
            new_id = tokenizer.convert_tokens_to_ids(tok)
            emb_in[new_id] = emb_in[ids].mean(0)
            if not tied:
                emb_out[new_id] = emb_out[ids].mean(0)
    return n_added

if __name__ == "__main__":
    from transformers import LlamaConfig, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained("gpt2")
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg)
    new = ["electroencephalogram", "pharmacokinetics", "tachycardia"]
    before = {t: tok(t).input_ids for t in new}
    n = add_tokens_with_init(model, tok, new)
    print("added", n, "vocab", len(tok), "emb", tuple(model.get_input_embeddings().weight.shape))
    for t in new:
        print(t, len(before[t]), "pieces ->", tok(t).input_ids)
    e = model.get_input_embeddings().weight
    nid = tok.convert_tokens_to_ids(new[0])
    print("init ok:", torch.allclose(e[nid], e[before[new[0]]].mean(0)))
    out = model(torch.tensor([tok("tachycardia is").input_ids]))
    print("logits", tuple(out.logits.shape))
```

```text
added 3 vocab 50260 emb (50260, 64)
electroencephalogram 6 pieces -> [50257]
pharmacokinetics 4 pieces -> [50258]
tachycardia 4 pieces -> [50259]
init ok: True
logits (1, 2, 50260)
```

(This uses a tiny randomly initialized Llama-architecture model with GPT-2's tokenizer so it runs on a CPU; with a real checkpoint, replace the config with `AutoModelForCausalLM.from_pretrained(...)`. Tested with `transformers` 5.x.)

Details that bite:

- **Untied output embeddings** (most models above ~1B) need the same initialization for the LM head row; the function handles both.
- `mean_resizing=False` disables transformers' default of initializing new rows from the mean and covariance of *all* old embeddings, since we're setting them ourselves. Mean-of-all is a reasonable fallback when there's no good decomposition.
- **Pad the vocab** to a multiple of 64 (`resize_token_embeddings(..., pad_to_multiple_of=64)`) for faster matmuls ([GPU Performance](../pretraining/gpu-performance.md)).
- **Train embeddings first**: some recipes run a short phase where only the embedding and LM-head rows train (everything else frozen) before unfreezing the whole model.
- If you use LoRA for CPT, the embeddings and LM head must be trainable too (`modules_to_save=["embed_tokens", "lm_head"]` in PEFT), or the new tokens never learn anything.

## Model merging

Take two models fine-tuned from the **same base**. Average their weights. Surprisingly often, the result is good at both tasks. This works because fine-tuning moves the weights only a little from a shared starting point, and in that neighbourhood the loss landscape is close to linear: different fine-tunes land in a common low-loss basin. Merging two models trained from *different* random initializations doesn't work (their neurons aren't aligned), so the rule is: **same base, same architecture.**

Merging costs no training compute, runs on a CPU, and is how many top open models on leaderboards were made. It's also a practical tool: combine a domain CPT model with the original instruct model, or a math fine-tune with a code fine-tune.

The notation: $\theta_{\text{base}}$ is the base model, $\theta_i$ the fine-tuned models, and the **task vector** of model $i$ is

$$
\tau_i = \theta_i - \theta_{\text{base}}
$$

Every method below operates tensor by tensor on state dicts.

### Linear merge (model soup)

$$
\theta_{\text{merged}} = \sum_i w_i\, \theta_i, \qquad \sum_i w_i = 1
$$

Model soups showed that averaging several fine-tunes of the same model with different hyperparameters often beats the best single one. "Greedy soup": sort models by validation score, add each to the soup only if it improves the validation score.

### SLERP

Linear interpolation between two weight vectors passes through the inside of the "sphere": the midpoint of two vectors of the same norm has a *smaller* norm, which can shrink the weights. **Spherical linear interpolation** moves along the arc instead, preserving the geometry:

$$
\mathrm{slerp}(a, b; t) = \frac{\sin((1-t)\,\Omega)}{\sin\Omega}\, a + \frac{\sin(t\,\Omega)}{\sin\Omega}\, b, \qquad \cos\Omega = \frac{a\cdot b}{\|a\|\,\|b\|}
$$

Only defined for two models. In practice $t$ is often varied by layer and by module type (attention vs. MLP), which is what the mergekit config below does.

### Task arithmetic

Treat each task vector as a direction that adds a skill:

$$
\theta_{\text{merged}} = \theta_{\text{base}} + \lambda \sum_i \tau_i
$$

$\lambda$ is a scaling factor, tuned on validation data (often 0.3-1.0). Negating a task vector *removes* a behavior: $\theta_{\text{base}} - \lambda \tau_{\text{toxic}}$ reduces toxicity learned by a model fine-tuned on toxic text. Task arithmetic is the conceptual basis for TIES and DARE.

### TIES

Adding task vectors causes **interference**. Two problems: (1) most entries of each $\tau_i$ are tiny, noisy changes that add up across models; (2) where two models changed the same weight in *opposite* directions, summing cancels both. TIES-merging (Trim, Elect sign, Merge):

1. **Trim**: in each $\tau_i$, keep only the top-$k$% entries by magnitude (the *density*, e.g. 20%); zero the rest.
2. **Elect sign**: for each parameter, pick the sign with the larger total magnitude across models: $\gamma = \mathrm{sign}\left(\sum_i \hat\tau_i\right)$.
3. **Disjoint merge**: average only the entries that agree with the elected sign (ignoring zeros), then add $\lambda$ times the result to the base.

### DARE

DARE (Drop And REscale) is a different way to sparsify task vectors: drop each entry at random with probability $p$, then rescale the survivors by $1/(1-p)$:

$$
\hat\tau = \frac{m \odot \tau}{1-p}, \qquad m_j \sim \mathrm{Bernoulli}(1-p)
$$

The rescaling keeps $\mathbb{E}[\hat\tau] = \tau$. The finding was that for large LLMs, you can drop 90% or more of the fine-tuning delta this way with little loss, because SFT deltas are highly redundant. Sparser deltas interfere less when merged. DARE is usually combined with TIES sign election ("DARE-TIES") or with task arithmetic ("DARE linear").

### From scratch, over state dicts

```python title="merging.py"
import torch

# All functions take state dicts {name: tensor} with identical keys and shapes.
# Compute in fp32 even if the checkpoints are bf16; cast back when saving.

def linear_merge(sds, weights=None):
    """Model soup: weighted average of parameters."""
    weights = weights or [1.0 / len(sds)] * len(sds)
    return {k: sum(w * sd[k].float() for w, sd in zip(weights, sds)) for k in sds[0]}

def slerp(a, b, t, eps=1e-8):
    """Spherical interpolation between two tensors, treated as flat vectors."""
    a_f, b_f = a.flatten().float(), b.flatten().float()
    a_n, b_n = a_f / (a_f.norm() + eps), b_f / (b_f.norm() + eps)
    dot = (a_n * b_n).sum().clamp(-1.0, 1.0)
    omega = torch.arccos(dot)
    if omega.abs() < 1e-4:                         # nearly parallel: SLERP ~ LERP
        out = (1 - t) * a_f + t * b_f
    else:
        so = torch.sin(omega)
        out = (torch.sin((1 - t) * omega) / so) * a_f + (torch.sin(t * omega) / so) * b_f
    return out.view_as(a)

def slerp_merge(sd_a, sd_b, t=0.5):
    return {k: slerp(sd_a[k], sd_b[k], t) for k in sd_a}

def task_vectors(base, finetuned):
    """tau_i = theta_i - theta_base, per tensor."""
    return [{k: ft[k].float() - base[k].float() for k in base} for ft in finetuned]

def task_arithmetic(base, finetuned, lam=1.0):
    """theta = theta_base + lam * sum_i tau_i. Negative lam on one tau 'forgets' that task."""
    taus = task_vectors(base, finetuned)
    return {k: base[k].float() + lam * sum(t[k] for t in taus) for k in base}

def trim(tau, density):
    """Keep the top-`density` fraction of entries by magnitude, zero the rest."""
    if density >= 1.0:
        return tau
    k = max(1, int(density * tau.numel()))
    thresh = tau.abs().flatten().kthvalue(tau.numel() - k + 1).values
    return tau * (tau.abs() >= thresh)

def ties_merge(base, finetuned, density=0.2, lam=1.0):
    """TIES: trim small updates, elect a sign per parameter, average only agreeing updates."""
    taus = task_vectors(base, finetuned)
    out = {}
    for k in base:
        T = torch.stack([trim(t[k], density) for t in taus])        # (n_models, *shape)
        sign = torch.sign(T.sum(0))                                  # elected sign: total mass
        agree = (torch.sign(T) == sign) & (T != 0)                   # entries that agree with it
        merged = (T * agree).sum(0) / agree.sum(0).clamp(min=1)      # disjoint mean
        out[k] = base[k].float() + lam * merged
    return out

def dare(tau, p, generator=None):
    """DARE: drop each delta with prob p, rescale survivors by 1/(1-p). Unbiased in expectation."""
    mask = torch.rand(tau.shape, generator=generator) >= p
    return tau * mask / (1 - p)

def dare_merge(base, finetuned, p=0.9, lam=1.0, use_ties=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    taus = task_vectors(base, finetuned)
    taus = [{k: dare(t[k], p, g) for k in t} for t in taus]
    if use_ties:                                   # "DARE-TIES": sign election on sparsified deltas
        fake_ft = [{k: base[k].float() + t[k] for k in t} for t in taus]
        return ties_merge(base, fake_ft, density=1.0, lam=lam)
    return {k: base[k].float() + lam * sum(t[k] for t in taus) for k in base}
```

A note on `trim`: `kthvalue` on a flattened tensor works but is slow for very large tensors; production tools use `topk` or quantile estimates per tensor and process one tensor at a time to keep memory low.

### A merge experiment you can run

To see these methods behave, take a small network with a shared trunk and one classification head per task. "Pretrain" it on task 0, fine-tune one copy on task A and another on task B, and merge the trunks. Each merged model gets task A's head from the A fine-tune and task B's head from the B fine-tune, the way a multi-task merge works in practice:

```python title="merge_demo.py"
import copy, torch, torch.nn as nn, torch.nn.functional as F
from merging import *

torch.manual_seed(0)
D, C = 32, 4
W_true = torch.randn(3, D, C)                       # tasks 0 (pretraining), A, B

def task_data(task, n=4096):
    x = torch.randn(n, D)
    return x, (x @ W_true[task]).argmax(-1)

def train(model, task, steps, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x, y = task_data(task)
    for _ in range(steps):
        i = torch.randint(0, len(x), (256,))
        loss = F.cross_entropy(model(x[i], task), y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return model

class Net(nn.Module):                                # shared trunk, one head per task
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(256, C) for _ in range(3)])
    def forward(self, x, task):
        return self.heads[task](self.trunk(x))

@torch.no_grad()
def acc(model, task):
    x, y = task_data(task, 8192)
    return (model(x, task).argmax(-1) == y).float().mean().item()

base = train(Net(), 0, 600, 1e-3)                    # "pretrained" model
ft_a = train(copy.deepcopy(base), 1, 600, 1e-3)      # fine-tuned on task A
ft_b = train(copy.deepcopy(base), 2, 600, 1e-3)      # fine-tuned on task B
def with_heads(sd):
    """Load merged weights, then give the model task A's head from ft_a and task B's from ft_b."""
    m = copy.deepcopy(base)
    m.load_state_dict(sd)
    m.heads[1].load_state_dict(ft_a.heads[1].state_dict())
    m.heads[2].load_state_dict(ft_b.heads[2].state_dict())
    return m

sd0, sda, sdb = base.state_dict(), ft_a.state_dict(), ft_b.state_dict()
print(f"{'model':24s} task A  task B")
for name, m in [("base trunk + new heads", with_heads(sd0)), ("ft on A", ft_a), ("ft on B", ft_b),
                ("linear 50/50", with_heads(linear_merge([sda, sdb]))),
                ("slerp t=0.5", with_heads(slerp_merge(sda, sdb, 0.5))),
                ("task arith lam=1", with_heads(task_arithmetic(sd0, [sda, sdb], 1.0))),
                ("ties d=0.2", with_heads(ties_merge(sd0, [sda, sdb], 0.2, 1.0))),
                ("dare p=0.9", with_heads(dare_merge(sd0, [sda, sdb], 0.9, 1.0))),
                ("dare-ties p=0.9", with_heads(dare_merge(sd0, [sda, sdb], 0.9, 1.0, use_ties=True)))]:
    print(f"{name:24s} {acc(m, 1):.3f}   {acc(m, 2):.3f}")

# sanity checks
a, b = torch.randn(100), torch.randn(100)
assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-5) and torch.allclose(slerp(a, b, 1.0), b, atol=1e-5)
tau = torch.randn(200_000)
print("DARE mean preserved:", round(dare(tau, 0.9).mean().item(), 3), "vs", round(tau.mean().item(), 3),
      "| kept fraction:", round((dare(tau, 0.9) != 0).float().mean().item(), 3))
print("trim keeps:", (trim(tau, 0.2) != 0).float().mean().item())
```

```text
model                    task A  task B
base trunk + new heads   0.335   0.411
ft on A                  0.955   0.231
ft on B                  0.209   0.947
linear 50/50             0.881   0.847
slerp t=0.5              0.868   0.843
task arith lam=1         0.906   0.882
ties d=0.2               0.813   0.803
dare p=0.9               0.732   0.695
dare-ties p=0.9          0.745   0.700
DARE mean preserved: 0.003 vs -0.002 | kept fraction: 0.1
trim keeps: 0.20000000298023224
```

What to take from it:

- Each fine-tuned model is great on its own task and near chance on the other (its trunk drifted away from what the other head expects). Every merge recovers most of the performance on **both** tasks from one set of trunk weights, with no extra training.
- Hyperparameters matter. In a quick sweep on this setup, TIES at density 0.5 and $\lambda = 1$ scored about 0.88 on both tasks, and DARE at $p = 0.5$ about 0.88/0.85, much better than the aggressive settings above. Task arithmetic did best at $\lambda = 1$ and worse at 0.5 or 1.5. Always sweep on validation data.
- **DARE at $p = 0.9$ hurts here** because a tiny MLP's fine-tuning delta isn't very redundant. The high drop rates reported for LLMs rely on the redundancy of billion-parameter deltas; don't assume they transfer to small models.

### Merging with mergekit

For real checkpoints, use **mergekit**: it implements all of these (and more), processes tensors lazily so a 70B merge runs in modest RAM, and handles sharded safetensors. Install with `pip install mergekit` and write a YAML config.

A TIES merge of two fine-tunes of the same base:

```yaml title="ties.yml"
models:
  - model: your-org/llama-3.1-8b-math        # fine-tune 1
    parameters:
      density: 0.5                           # keep top 50% of each task vector
      weight: 0.5
  - model: your-org/llama-3.1-8b-code        # fine-tune 2
    parameters:
      density: 0.5
      weight: 0.5
merge_method: ties                           # or dare_ties, task_arithmetic, linear
base_model: meta-llama/Llama-3.1-8B          # the shared base
parameters:
  normalize: true                            # divide by the sum of weights
dtype: bfloat16
```

A SLERP merge with interpolation factors varying across layers, separately for attention and MLP:

```yaml title="slerp.yml"
slices:
  - sources:
      - model: your-org/model-a
        layer_range: [0, 32]
      - model: your-org/model-b
        layer_range: [0, 32]
merge_method: slerp
base_model: your-org/model-a
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]           # interpolated across the layer range
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5                             # everything else
dtype: bfloat16
```

Run:

```bash
mergekit-yaml ties.yml ./merged-model --cuda    # --cuda optional; CPU works, just slower
```

The output directory is a normal Hugging Face model folder. Copy the tokenizer and chat template from the model whose chat format you want (mergekit can do this for you with its tokenizer options), and check `config.json` is what you expect.

### Merging LoRA adapters

Two different things are called "merging LoRA":

1. **Merging an adapter into its base** (`merge_and_unload`), covered in [LoRA & QLoRA](lora-qlora.md#merging-adapters).
2. **Combining several adapters** trained on the same base into one.

For (2), the trap: LoRA's update is a product, $\Delta W = BA$. Averaging the factors is **not** averaging the updates:

$$
\tfrac12(B_1 + B_2)\,\tfrac12(A_1 + A_2) \neq \tfrac12 B_1A_1 + \tfrac12 B_2A_2
$$

```python title="lora_factor_averaging.py"
import torch
torch.manual_seed(0)
d, r = 64, 8
B1, A1 = torch.randn(d, r), torch.randn(r, d)
B2, A2 = torch.randn(d, r), torch.randn(r, d)
right = 0.5 * (B1 @ A1) + 0.5 * (B2 @ A2)                  # average the weight updates
wrong = (0.5 * (B1 + B2)) @ (0.5 * (A1 + A2))              # average the factors
print("relative error of factor-averaging:", ((wrong - right).norm() / right.norm()).item())
cat = torch.cat([0.5 * B1, 0.5 * B2], 1) @ torch.cat([A1, A2], 0)   # rank 2r, exact
print("concat exact:", torch.allclose(cat, right, atol=1e-4))
```

```text
relative error of factor-averaging: 0.7173267602920532
concat exact: True
```

Correct options: compute each adapter's full $\Delta W_i = \frac{\alpha}{r} B_i A_i$, merge those (linear, TIES, DARE, treating $\Delta W_i$ as the task vector); or **concatenate** factors into a rank $r_1 + r_2$ adapter, which is exact for a linear combination. PEFT implements both:

```python title="peft_combine.py"
# model is a PeftModel with adapters "math" and "code" loaded on the same base
model.add_weighted_adapter(adapters=["math", "code"], weights=[0.5, 0.5],
                           adapter_name="math_code", combination_type="ties_svd", density=0.5)  # TIES on full deltas, then SVD
model.add_weighted_adapter(adapters=["math", "code"], weights=[1.0, 1.0],
                           adapter_name="math_plus_code", combination_type="cat")  # exact, rank r1+r2
model.set_adapter("math_code")
```

The `_svd` suffix matters: `"ties_svd"`, `"dare_ties_svd"`, `"dare_linear_svd"` and plain `"svd"` merge the full deltas $\Delta W_i$ and then compress back to low rank with SVD. Without the suffix, `"linear"`, `"ties"`, `"dare_ties"` and `"dare_linear"` apply the method to the $A$ and $B$ factors separately, which is the factor-averaging shortcut above (cheap, but only approximate). If adapters will be served side by side rather than merged, see multi-adapter serving in [LoRA & QLoRA](lora-qlora.md#multi-adapter-serving).

### Evaluating merges

A merge is a new model; it needs the same evaluation as any other.

- **Evaluate every constituent task** and a general-capability suite. A merge that's good at math and code but has lost instruction following is a regression. Use the harness and practices in [Evaluation](../pretraining/evaluation.md).
- **Compare against the constituents** and against the simplest baseline (linear average). Fancy methods don't always win.
- **Sweep the knobs** ($\lambda$, density, $p$, per-layer $t$) on a validation split, and report on a separate test split, since you're doing hyperparameter search.
- **Check generation, not just benchmarks**: merges can produce repetition, broken chat formatting or wrong stop-token behavior that multiple-choice evals don't catch.
- **Beware leaderboard merges.** Many merged models were selected by leaderboard score, which risks overfitting to (or contamination from) those benchmarks. Trust your own held-out evals.

!!! warning "Common pitfalls"
    - **Different bases.** Merging models fine-tuned from different base checkpoints (or different random inits) produces garbage. Check that the base model and architecture match exactly.
    - **Tokenizer and vocab mismatch.** If one model added tokens or resized embeddings, the embedding matrices don't line up. Mergekit has options to build a union tokenizer; otherwise, don't merge.
    - **Chat template mismatch.** Two fine-tunes with different chat templates merge into a model that follows neither reliably.
    - **CPT without replay**, or at the original peak LR. Both cause heavy forgetting. Start with ~10-20% replay and a lower peak, and track general validation loss.
    - **Random init for new tokens.** Causes early loss spikes and stray new tokens in generations. Initialize from sub-token means.
    - **Averaging LoRA factors** instead of the products $BA$.
    - **Merging in low precision.** Do the arithmetic in fp32 and cast the result to bf16, especially for SLERP and sign election.

## Check your understanding

??? question "You have 2 GB of internal engineering docs and want a model that knows your systems. CPT, SFT, or RAG?"
    2 GB is roughly half a billion tokens: enough for continued pretraining to teach domain vocabulary and facts, followed by SFT on domain-flavored instructions so it stays a usable assistant. But if the knowledge changes often or answers must cite sources, RAG is cheaper and easier to keep current, and many teams combine RAG with a light fine-tune instead. SFT alone on generated Q&A pairs is the weakest option for injecting this much knowledge.

??? question "Why re-warm the learning rate at the start of continued pretraining, and why to a lower peak than the original?"
    The released checkpoint ended with a tiny LR, which would make learning the new domain very slow. Jumping straight to a high LR from a converged point causes instability and forgetting. A short warmup lets the optimizer (with fresh Adam moments) stabilize, and a lower peak trades a little adaptation speed for much less forgetting of general ability.

??? question "Why initialize a new token's embedding as the mean of its old sub-token embeddings rather than randomly?"
    The model already has a representation for that sequence of pieces. The mean of their embeddings starts the new token near the region of embedding space associated with that meaning, so the model's existing layers can make sense of it immediately. A random vector is out of distribution, produces large early losses, and wastes training steps.

??? question "Two fine-tunes changed the same weight by +0.02 and -0.018. What does task arithmetic do, and what does TIES do?"
    Task arithmetic sums them to +0.002: both changes nearly cancel, and neither model's adaptation survives. TIES elects a sign by total magnitude (positive here), keeps only the agreeing entry, and averages agreeing entries only, so the merged update is +0.02. This preserves the dominant model's change instead of destroying both.

??? question "Why can't you merge two LoRA adapters by averaging their A and B matrices?"
    The update is the product $BA$. The product of averages contains cross terms like $B_1A_2$ that neither adapter learned, so it isn't the average of the updates. Merge the full deltas $B_iA_i$ (or concatenate the factors into a higher-rank adapter) instead.

## Further reading

- Gururangan et al., *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks* - [arXiv:2004.10964](https://arxiv.org/abs/2004.10964)
- Gupta et al., *Continual Pre-Training of Large Language Models: How to (re)warm your model?* - [arXiv:2308.04014](https://arxiv.org/abs/2308.04014)
- Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models* - [arXiv:2403.08763](https://arxiv.org/abs/2403.08763)
- Rozière et al., *Code Llama: Open Foundation Models for Code* - [arXiv:2308.12950](https://arxiv.org/abs/2308.12950)
- Cui et al., *Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca* - [arXiv:2304.08177](https://arxiv.org/abs/2304.08177)
- Wortsman et al., *Model soups* - [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)
- Ilharco et al., *Editing Models with Task Arithmetic* - [arXiv:2212.04089](https://arxiv.org/abs/2212.04089)
- Yadav et al., *TIES-Merging: Resolving Interference When Merging Models* - [arXiv:2306.01708](https://arxiv.org/abs/2306.01708)
- Yu et al., *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch* (DARE) - [arXiv:2311.03099](https://arxiv.org/abs/2311.03099)
- Goddard et al., *Arcee's MergeKit: A Toolkit for Merging Large Language Models* - [arXiv:2403.13257](https://arxiv.org/abs/2403.13257); [github.com/arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)
- Biderman et al., *LoRA Learns Less and Forgets Less* - [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- PEFT docs: [model merging](https://huggingface.co/docs/peft/developer_guides/model_merging)
