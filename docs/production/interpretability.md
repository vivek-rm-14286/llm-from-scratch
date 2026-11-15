# Interpretability

You have trained transformers, so you know exactly what computation they perform: matrix multiplies, softmaxes, nonlinearities. What you do not know is *what the numbers mean*. Which direction in layer 6 encodes "this sentence is negative"? Which attention head copies a name from earlier in the prompt? Why did the model answer "Rome"? Mechanistic interpretability tries to answer such questions by treating a trained network as an artifact to reverse-engineer. The techniques on this page all run on GPT-2 small on a laptop CPU, and they give you real tools for debugging models, not only for research.

!!! abstract "What you'll learn"
    - Why interpretability matters for debugging and for safety
    - Capturing activations with PyTorch forward hooks
    - Linear probes on hidden states, and the control experiments that keep them honest
    - The logit lens: decoding intermediate layers into vocabulary
    - Attention pattern analysis and finding induction heads
    - Activation patching (causal tracing) to locate where a behavior is computed
    - Superposition, and training a sparse autoencoder from scratch on cached activations
    - Steering vectors, the main tools (TransformerLens, nnsight), and the limits of all of this

## Why interpretability?

- **Debugging.** When a fine-tune breaks something, behavioral evals tell you *that* it broke. Internal analysis can tell you *where*: which layer's representation of a concept changed, or whether a capability was lost or just suppressed.
- **Safety and monitoring.** If you can read a concept from activations (deception, a harmful-intent signal, "the model knows it doesn't know"), you can monitor it more cheaply and robustly than by reading outputs. Arditi et al. (2024) found that refusal in several chat models is largely mediated by a single direction in the residual stream, which shows both how fragile safety training can be and how interpretable it can be.
- **Science.** Understanding how in-context learning, factual recall or arithmetic is implemented tells you what training actually produces.

Throughout, keep the residual stream picture from [The Transformer, Line by Line](../llm/transformer.md) in mind: each block reads from and adds to a running vector per position, $x_{\ell+1} = x_\ell + \text{Attn}_\ell(x_\ell) + \text{MLP}_\ell(\cdot)$. Most techniques here read or edit that vector.

## Capturing activations with hooks

PyTorch lets you register a function that runs whenever a module's forward pass finishes. It sees the module's inputs and output, and if it returns a value, that value **replaces** the output. Reading activations and editing them use the same mechanism.

```python
cache = {}
def hook(module, inputs, output):
    cache["block6"] = output[0] if isinstance(output, tuple) else output  # (batch, seq, d_model)
handle = model.transformer.h[6].register_forward_hook(hook)
model(input_ids)
handle.remove()          # always remove hooks, or they keep firing on every later call
```

In Hugging Face GPT-2, `model.transformer.h[i]` is block *i*, and depending on the library version a block returns either a tensor or a tuple whose first element is the hidden state. The `isinstance` check handles both. The code on this page was tested with `transformers` 5.x and `torch` 2.x on CPU. GPT-2 small is about 500 MB to download.

## Linear probes

A **probe** is a simple classifier trained to predict a property from a model's hidden states. If a *linear* probe succeeds, the information is linearly decodable at that layer, which is a strong hint (not proof) that the model represents and uses it.

```python title="probe.py"
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression

torch.manual_seed(0)
name = "gpt2"          # or "HuggingFaceTB/SmolLM2-135M" (then use model.model.layers below)
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

# A toy concept: is the sentence about a positive or negative experience?
pos_adj = ["wonderful", "great", "delightful", "excellent", "lovely", "fantastic", "superb", "pleasant",
           "amazing", "brilliant", "charming", "enjoyable"]
neg_adj = ["terrible", "awful", "horrible", "dreadful", "miserable", "disappointing", "bad", "painful",
           "boring", "dull", "unpleasant", "tedious"]
subjects = ["The movie", "Our dinner", "The trip", "That concert", "The hotel", "My day", "The book",
            "The meeting", "Her speech", "The game"]
texts, labels, is_test = [], [], []
for s in subjects:
    for adjs, lab in [(pos_adj, 1), (neg_adj, 0)]:
        for j, a in enumerate(adjs):
            texts.append(f"{s} was {a}."); labels.append(lab)
            is_test.append(j >= 8)          # hold out 4 adjectives per class: test generalization, not memory

# Capture the residual stream after every block with forward hooks
cache = {}
def make_hook(i):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        cache[i] = hidden.detach()           # (batch, seq, d_model)
    return hook
handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(model.transformer.h)]

feats = {i: [] for i in range(len(model.transformer.h))}
with torch.no_grad():
    for t in texts:
        ids = tok(t, return_tensors="pt")
        model(**ids)
        for i in feats:
            feats[i].append(cache[i][0, -1].numpy())   # last token's activation
for h in handles: h.remove()

y, test = np.array(labels), np.array(is_test)
rng = np.random.default_rng(0)
for layer in [0, 2, 4, 6, 8, 10, 11]:
    X = np.stack(feats[layer])
    Xtr, Xte, ytr, yte = X[~test], X[test], y[~test], y[test]
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)
    # control: same probe on shuffled labels; should be ~chance
    ctrl = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, rng.permutation(ytr))
    print(f"layer {layer:2d}: probe acc {probe.score(Xte, yte):.2f}   shuffled-label control {ctrl.score(Xte, yte):.2f}")
```

```text
layer  0: probe acc 0.89   shuffled-label control 0.41
layer  2: probe acc 1.00   shuffled-label control 0.47
layer  4: probe acc 1.00   shuffled-label control 0.44
...
layer 11: probe acc 1.00   shuffled-label control 0.44
```

We probe the final "." token, which never differs between classes, so whatever the probe reads there was moved in by attention. And the test set uses **adjectives the probe never saw**, so it cannot succeed by memorizing token identities. Sentiment is already decodable after block 0 and perfectly so from block 2 onward.

That is the easy part. The hard part is interpreting the result honestly:

- **Probes measure decodability, not use.** A linear probe can find information the model never uses. To show the model *uses* a direction, intervene on it (patching or steering, below) and check that behavior changes.
- **Use controls.** The shuffled-label control should sit near chance (it does). Hewitt & Liang (2019) also suggest *control tasks*: if a probe is expressive enough to learn arbitrary labels, high accuracy says more about the probe than the model. Keep probes linear and regularized.
- **Split by the thing you want to generalize over.** A random train/test split here would mix the same adjectives into both sets and overstate the result.
- **Toy datasets are easy.** Templated sentences differ only in the adjective. Real probing work uses varied, naturally occurring text.

Probes are practical too. A probe on a model's own hidden states can predict "this answer is likely wrong" or "this prompt is a jailbreak attempt" at almost no extra cost, since you already ran the forward pass. That makes probes a cheap kind of [guard classifier](../decision/routing-models.md).

## The logit lens

The final layer's residual stream is turned into next-token logits by the final layer norm followed by the unembedding matrix. The **logit lens** applies that same readout to *intermediate* layers, to see what the model would predict if it stopped early.

```python title="logit_lens.py"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

prompt = "The Eiffel Tower is located in the city of"
ids = tok(prompt, return_tensors="pt")
with torch.no_grad():
    out = model(**ids, output_hidden_states=True)

# hidden_states: tuple of (n_layers + 1) tensors; [0] = embeddings, [-1] = after final ln_f
ln_f, unembed = model.transformer.ln_f, model.lm_head
for layer, h in enumerate(out.hidden_states):
    resid = h[0, -1]                                   # last position
    if layer < len(out.hidden_states) - 1:
        resid = ln_f(resid)                            # final entry already has ln_f applied
    probs = unembed(resid).softmax(-1)
    top = probs.topk(3)
    words = [repr(tok.decode(int(i))) for i in top.indices]
    target_p = probs[tok.encode(" Paris")[0]].item()
    print(f"layer {layer:2d}  top-3 {', '.join(words):40s}  P(' Paris')={target_p:.3f}")
```

```text
layer  0  top-3 ' destro', ' mathemat', ' challeng'       P(' Paris')=0.000
layer  1  top-3 ' the', ' course', ' a'                   P(' Paris')=0.000
...
layer  6  top-3 ' England', ' the', ' La'                 P(' Paris')=0.000
layer  7  top-3 ' East', ' La', ' Georgia'                P(' Paris')=0.001
layer  8  top-3 ' Ing', ' San', ' Rome'                   P(' Paris')=0.001
layer  9  top-3 ' Rome', ' San', ' La'                    P(' Paris')=0.015
layer 10  top-3 ' London', ' Paris', ' Amsterdam'         P(' Paris')=0.194
layer 11  top-3 ' Paris', ' London', ' Amsterdam'         P(' Paris')=0.165
layer 12  top-3 ' Paris', ' London', ' Amsterdam'         P(' Paris')=0.064
```

The story is readable. Early layers predict generic continuations ("the", "course"). The middle layers work out "a place name" (England, Georgia, San). The answer type narrows to "a city" (Rome, London) and finally to Paris around layers 9-11. The last block then *lowers* the probability of Paris while keeping it top-1, which is a common pattern: final layers often calibrate or hedge.

Caveat: intermediate layers were never trained to be read by the final unembedding, so the lens can be misleading, especially in early layers and in some model families. The **tuned lens** (Belrose et al., 2023) trains a small affine map per layer to fix this.

## Attention patterns and induction heads

Attention weights are the easiest internals to look at: for each head, a matrix showing how much each position reads from each earlier position. Looking at them directly is tempting but limited, since a head's effect also depends on what it writes (its value and output matrices). Still, some heads have patterns so distinctive that you can find them by measurement.

The best-understood example is the **induction head** (Elhage et al., 2021; Olsson et al., 2022). Given a sequence `... A B ... A`, an induction head at the second `A` attends to the token after the first `A` (that is, `B`) and copies it, predicting `B`. It needs two heads working together: a "previous-token head" in an earlier layer writes "the token before me was A" into each position, and the induction head searches for that. Induction heads are thought to be a major mechanism behind in-context learning, and they appear fairly suddenly during training, coinciding with a visible bump in the loss curve.

You can find them with repeated random tokens, where the only way to predict the second copy is to look back:

```python title="induction.py"
import torch
from transformers import AutoModelForCausalLM

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()  # eager returns attn weights

T = 50
rand = torch.randint(1000, 20000, (8, T))            # 8 sequences of random tokens
ids = torch.cat([rand, rand], dim=1)                 # ...repeated once: [A B C ... A B C ...]
with torch.no_grad():
    out = model(ids, output_attentions=True)

# An induction head at position T+i (second copy of token i) attends to position i+1:
# the token that came *after* the previous occurrence of the current token.
scores = []
for layer, att in enumerate(out.attentions):        # att: (batch, heads, seq, seq)
    q = torch.arange(T, 2 * T - 1)
    k = q - T + 1
    s = att[:, :, q, k].mean(dim=(0, 2))            # mean attention to the "induction" offset, per head
    scores += [(s[h].item(), layer, h) for h in range(att.shape[1])]
for s, layer, h in sorted(scores, reverse=True)[:6]:
    print(f"L{layer}H{h}: induction score {s:.2f}")

# Behavioral check: loss on the second copy should be far lower than on the first
logp = out.logits.log_softmax(-1)[:, :-1].gather(-1, ids[:, 1:, None])[..., 0]
print(f"mean loss, first copy {-logp[:, :T-1].mean():.2f}  second copy {-logp[:, T:].mean():.2f}")
```

```text
L5H5: induction score 0.94
L7H10: induction score 0.92
L6H9: induction score 0.92
L5H1: induction score 0.91
L7H2: induction score 0.84
L10H1: induction score 0.52
mean loss, first copy 12.30  second copy 0.20
```

Several heads in layers 5-7 put over 90% of their attention exactly on the induction offset, and the loss on the repeated half drops from about 12 nats (random guessing over a large vocabulary) to 0.2. This is the same experiment you can run on your own small models from [Project 9](../projects/slm-500m.md) at different checkpoints to watch induction heads form.

## Activation patching (causal tracing)

Probes and lenses are correlational. **Activation patching** is causal: run the model on two inputs that differ in one controlled way, then copy one activation from the "clean" run into the "corrupted" run and measure how much of the clean behavior comes back. Components where patching restores the behavior are the ones carrying the relevant information. Meng et al. (2022) used this to localize factual recall; Wang et al. (2022) used it to map the "indirect object identification" (IOI) circuit in GPT-2.

We use the IOI task. In "When Mary and John went to the store, John gave a drink to", the answer is " Mary". Swap the second name and the answer flips to " John". The metric is the logit difference between the two names.

```python title="patching.py"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
blocks = model.transformer.h

clean   = "When Mary and John went to the store, John gave a drink to"
corrupt = "When Mary and John went to the store, Mary gave a drink to"
c_ids, x_ids = tok(clean, return_tensors="pt").input_ids, tok(corrupt, return_tensors="pt").input_ids
assert c_ids.shape == x_ids.shape, "patching needs aligned positions"
mary, john = tok.encode(" Mary")[0], tok.encode(" John")[0]

def logit_diff(logits):                      # > 0 means the model prefers " Mary"
    return (logits[0, -1, mary] - logits[0, -1, john]).item()

# 1) cache clean activations (output of every block)
clean_cache = {}
def save(i):
    def hook(m, inp, out):
        clean_cache[i] = (out[0] if isinstance(out, tuple) else out).detach().clone()
    return hook
hs = [b.register_forward_hook(save(i)) for i, b in enumerate(blocks)]
with torch.no_grad(): clean_ld = logit_diff(model(c_ids).logits)
for h in hs: h.remove()
with torch.no_grad(): corrupt_ld = logit_diff(model(x_ids).logits)
print(f"clean logit diff {clean_ld:.2f}, corrupted {corrupt_ld:.2f}")

# 2) run corrupted input, but overwrite one (layer, position) with the clean activation
def patch(layer, pos):
    def hook(m, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hidden = hidden.clone()
        hidden[:, pos] = clean_cache[layer][:, pos]
        return (hidden, *out[1:]) if isinstance(out, tuple) else hidden   # returned value replaces output
    return hook

T = c_ids.shape[1]
effect = torch.zeros(len(blocks), T)
for layer in range(len(blocks)):
    for pos in range(T):
        h = blocks[layer].register_forward_hook(patch(layer, pos))
        with torch.no_grad(): ld = logit_diff(model(x_ids).logits)
        h.remove()
        effect[layer, pos] = (ld - corrupt_ld) / (clean_ld - corrupt_ld)   # 0 = no effect, 1 = full recovery

toks = [tok.decode(int(t)) for t in c_ids[0]]
print("pos  token        " + " ".join(f"L{l:<3d}" for l in range(len(blocks))))
for pos in range(T):
    print(f"{pos:3d}  {toks[pos]!r:12s} " + " ".join(f"{effect[l, pos]:4.1f}" for l in range(len(blocks))))
```

```text
clean logit diff 2.01, corrupted -1.79
pos  token        L0   L1   L2   L3   L4   L5   L6   L7   L8   L9   L10  L11
  0  'When'        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
  ...
  8  ','           0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
  9  ' John'       1.0  1.0  1.0  1.0  0.9  0.9  0.8  0.5  0.1  0.0  0.0  0.0
 10  ' gave'      -0.0 -0.0 -0.0  0.0  0.1  0.0  0.1  0.0  0.0  0.0  0.0  0.0
 ...
 13  ' to'         0.0  0.0  0.0  0.0 -0.0  0.0  0.0  0.4  1.0  1.0  1.0  1.0
```

Read the table as a flow of information. Up to layer 6, everything the model needs is at position 9 (the second name): patching it there restores the answer. Between layers 7 and 8 the information *moves* to the final position, and from layer 8 on, patching only the last position is enough. Attention heads in layers 7-8 must be copying it across, which matches the IOI circuit's "name mover" heads found by Wang et al. To go further, patch individual heads' outputs (hook the attention module, or `c_proj`'s input) instead of whole blocks.

Practical notes: choose corruptions that change only the property you care about and keep tokenization aligned; normalize the effect as above so results are comparable; and be aware that patching whole residual streams early on trivially restores everything, since the residual stream at a position contains everything about that position.

## Superposition and sparse autoencoders

If you look at individual neurons, most are **polysemantic**: one neuron fires for academic citations, Korean text and HTTP requests. Elhage et al. (2022) proposed an explanation called **superposition**: a model has many more features it wants to represent than it has dimensions, and because each feature is rarely active, it can pack them into nearly-orthogonal directions and tolerate a little interference. Features are then *directions* in activation space, not individual neurons.

A **sparse autoencoder (SAE)** tries to recover those directions. It maps a $d$-dimensional activation $x$ to $m \gg d$ non-negative feature activations $f$, and reconstructs $x$ as a sparse sum of learned feature directions:

$$
f = \text{ReLU}(W_{\text{enc}}(x - b_{\text{dec}}) + b_{\text{enc}}), \qquad \hat{x} = W_{\text{dec}} f + b_{\text{dec}}
$$

$$
\mathcal{L} = \lVert x - \hat{x} \rVert_2^2 + \lambda \sum_j f_j \, \lVert W_{\text{dec}, :, j} \rVert_2
$$

The reconstruction term keeps the features faithful; the L1 term makes only a few of them active per token. Each column of $W_{\text{dec}}$ is a candidate feature direction. Weighting the L1 term by the decoder column norms stops the model from cheating by shrinking $f$ and growing $W_{\text{dec}}$.

Step 1: cache activations once (about a minute on CPU for 100K tokens).

```python title="sae_cache.py"
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
LAYER, N_TOKENS = 6, 100_000

texts = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if len(t) > 200]
acts, tok_ids, store = [], [], {}
h = model.transformer.h[LAYER].register_forward_hook(
    lambda m, i, o: store.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
n = 0
with torch.no_grad():
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=128).input_ids
        model(ids)
        acts.append(store["x"][0, 1:])           # drop position 0: its norm is huge and atypical
        tok_ids.append(ids[0, 1:])
        n += ids.shape[1] - 1
        if n >= N_TOKENS: break
h.remove()
torch.save({"X": torch.cat(acts), "tok_ids": torch.cat(tok_ids)}, "acts_l6.pt")
```

Step 2: train the SAE from scratch (about two minutes on a laptop CPU).

```python title="sae_train.py"
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("gpt2")
data = torch.load("acts_l6.pt"); X, tok_ids = data["X"], data["tok_ids"]
X = (X - X.mean(0)) / X.norm(dim=1).mean() * X.shape[1] ** 0.5   # center; scale mean norm to sqrt(d)

class SAE(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d)
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.T)                 # tied init helps early training
            self.dec.weight /= self.dec.weight.norm(dim=0, keepdim=True)
    def forward(self, x):
        f = F.relu(self.enc(x - self.dec.bias))                      # feature activations (sparse, >= 0)
        return self.dec(f), f

d, m = X.shape[1], 4 * X.shape[1]                                    # 4x expansion: 3072 features
sae = SAE(d, m)
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
l1_coef, steps, bs = 2.0, 4000, 1024

for step in range(steps):
    x = X[torch.randint(0, len(X), (bs,))]
    x_hat, f = sae(x)
    mse = (x_hat - x).pow(2).sum(-1).mean()
    l1 = (f * sae.dec.weight.norm(dim=0)).sum(-1).mean()            # decoder-norm-weighted L1
    loss = mse + l1_coef * l1
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == steps - 1:
        with torch.no_grad():
            l0 = (f > 0).float().sum(-1).mean()                      # avg active features per token
            fvu = (x_hat - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
        print(f"step {step:5d}  mse {mse.item():7.2f}  L0 {l0.item():6.1f}  frac var unexplained {fvu.item():.3f}")

# Inspect: which tokens most strongly activate a few features?
with torch.no_grad():
    F_all = torch.cat([sae(X[i:i + 8192])[1] for i in range(0, len(X), 8192)])
freq = (F_all > 0).float().mean(0)
print(f"alive features: {(freq > 0).sum().item()} / {m}")
cands = torch.nonzero((freq > 1e-4) & (freq < 1e-2)).flatten()[:5]   # sparse-but-alive features
for j in cands.tolist():
    top = F_all[:, j].topk(8).indices
    print(f"feature {j:5d} (fires on {freq[j]:.2%} of tokens):", [tok.decode(int(tok_ids[i])) for i in top])
```

```text
step     0  mse  277.47  L0 1535.7  frac var unexplained 0.682
step   500  mse  121.34  L0   82.4  frac var unexplained 0.298
...
step  3999  mse   86.03  L0   20.9  frac var unexplained 0.212
alive features: 3072 / 3072
feature     0 (fires on 0.87% of tokens): ['ne', 'r', 'c', 'ph', 'anti', 'at', 'ne', 'ph']
feature     1 (fires on 0.31% of tokens): [' Super', ' Super', ' Super', ' Super', ' Super', ' Super', ' Super', ' super']
feature     2 (fires on 0.28% of tokens): [' District', ' District', ' District', ' District', ' District', ' District', ' District', ' district']
feature     3 (fires on 0.30% of tokens): [' considered', ' considered', ' considered', ' considered', ' considered', ' considered', ' considered', ' considered']
feature     4 (fires on 0.13% of tokens): [' Along', ' Along', ' Along', ' Along', ' along', ' along', ' along', ' along']
```

About 21 of 3,072 features are active per token, and together they reconstruct roughly 79% of the variance. Many features are clean single-concept detectors (features 1-4 each fire on one word), while others are fuzzier (feature 0 fires on word-start fragments of multi-token words). At this tiny scale, most features are token-level, as you would expect. With more data, more features and more training, SAEs on larger models produce abstract features too: Templeton et al. (2024) reported features for concepts such as specific landmarks, code bugs and deception in a production model.

The two numbers to watch are **L0** (sparsity) and **fraction of variance unexplained** (fidelity). Raise `l1_coef` and you get sparser, often more interpretable features with worse reconstruction; lower it and the reverse. Also watch for **dead features** that never fire. Common fixes are resampling dead features during training, or replacing the L1 penalty with a TopK activation that keeps exactly *k* features per token (Gao et al., 2024). A stronger test of fidelity is to splice $\hat{x}$ back into the model with a hook and measure how much the language-modeling loss goes up.

Pretrained SAEs exist for several open models, for example Gemma Scope for Gemma 2 (Lieberum et al., 2024), so you do not always have to train your own.

## Steering vectors

If concepts are directions, you can *add* a direction to the residual stream during generation and push the model's behavior. The simplest recipe (Turner et al., 2023; Zou et al., 2023) is the difference of mean activations between contrasting prompts:

```python title="steering.py"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
block = model.transformer.h[6]

def mean_act(prompts):
    store = []
    h = block.register_forward_hook(lambda m, i, o: store.append((o[0] if isinstance(o, tuple) else o)[0, -1]))
    with torch.no_grad():
        for p in prompts: model(tok(p, return_tensors="pt").input_ids)
    h.remove()
    return torch.stack(store).mean(0)

pos = ["I love this. It is wonderful", "What a great, happy day", "This is fantastic and I adore it"]
neg = ["I hate this. It is awful", "What a terrible, sad day", "This is horrible and I despise it"]
steer = mean_act(pos) - mean_act(neg)                     # a direction in the residual stream

def generate(prompt, alpha):
    def add(m, i, o):                                     # add alpha * steer at every position
        return (o[0] + alpha * steer, *o[1:]) if isinstance(o, tuple) else o + alpha * steer
    h = block.register_forward_hook(add)
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=25, do_sample=True, top_p=0.9, temperature=0.8,
                             pad_token_id=tok.eos_token_id)
    h.remove()
    return tok.decode(out[0, ids.shape[1]:])

for alpha in [-1, 0, 1]:
    torch.manual_seed(1)                                  # same sampling noise for each alpha
    print(f"alpha={alpha:+d}:", repr(generate("I think the new restaurant downtown is", alpha)))
```

One run gave:

```text
alpha=-1: ' going to be the worst in the country. People want to live here. They want to be in here," he said.'
alpha=+0: ' going to be the place to look for a new restaurant," he said.\n\n"I think it will be a new'
alpha=+1: ' a great place to have your weekend and get a great beer, a great meal, and a great watch.\n\nI'
```

The effect is visible but noisy. Treat a single sample as an anecdote, and measure over many samples with a sentiment classifier. At larger values of `alpha` (2 or more, since this vector's norm is about 25) the text becomes incoherent. Picking the layer and scale is empirical. The same technique is used in safety research: adding or removing a "refusal direction" changes whether chat models refuse (Arditi et al., 2024). That is why open-weight safety cannot rely on refusal training alone.

## Tools

The from-scratch hooks above are worth writing once. For real work, use a library:

- **TransformerLens** reimplements many open models with named hook points on every intermediate activation (`blocks.6.hook_resid_post`, `blocks.5.attn.hook_pattern`), plus caching and patching utilities. It is the standard tool for mechanistic interpretability on small and medium models.
- **nnsight** wraps ordinary Hugging Face/PyTorch models and lets you write interventions as code inside a tracing context. It can also run them remotely on large models.
- **SAELens** for training and loading sparse autoencoders, and **Neuronpedia** for browsing published SAE features.

## Limits of interpretability

Be skeptical, including of your own results:

- **Illusions are easy.** Probes find information the model does not use. Logit-lens readings in early layers are unreliable. Patching results depend on the choice of corruption. Attention weights are not explanations by themselves.
- **Toy tasks may not generalize.** The IOI circuit explains one templated task in one model. Real behavior on diverse text involves many overlapping mechanisms.
- **SAE features are not ground truth.** They depend on the dictionary size, sparsity penalty and training data. Features split and merge as you scale, and reconstructions still miss a meaningful fraction of the model's computation.
- **Interpretation is human labeling.** "This is the deception feature" is a hypothesis based on top activations. Test it causally and with held-out examples.
- **Scale.** Exhaustive analysis of a frontier model remains far out of reach. Today, interpretability is a useful tool for debugging and monitoring and an active research field, not a safety guarantee.

!!! warning "Common pitfalls"
    - **Forgetting to remove hooks.** They keep firing (and modifying outputs) on every subsequent forward pass. Use `try/finally` or a context manager.
    - **Hooking the wrong thing.** Know whether you are capturing a block's output, the attention output, or the MLP output, and whether layer norm has been applied.
    - **Random train/test splits for probes** that leak the same tokens into both sets.
    - **Misaligned positions** in patching because the clean and corrupted prompts tokenize to different lengths.
    - **Including the first position** in SAE training data. Its activation norm is often far larger than other positions and dominates the loss.
    - **Over-reading a single generation** from a steering experiment.

## Check your understanding

??? question "A linear probe for 'is this statement true' gets 90% accuracy at layer 12. What can and can't you conclude?"
    You can conclude that truth-related information is linearly decodable from layer 12 on this dataset. You cannot conclude that the model uses that direction to produce its outputs, or that the probe found "truth" rather than a correlated surface feature of your dataset. Check with controls, a held-out distribution (different topics or phrasing), and a causal test: does adding or removing the probe direction change the model's answers?

??? question "Why does the logit lens apply the final layer norm before the unembedding?"
    The unembedding was trained to read residual vectors that have passed through the final layer norm. Without it, intermediate residuals have the wrong scale and offset for the unembedding, so the decoded distribution would be meaningless. Even with it, early layers were not trained to be read this way, which is the motivation for the tuned lens.

??? question "In the patching table, why does patching position 9 stop working after layer 8?"
    By then, attention heads have already copied the relevant information (which name is repeated) from position 9 to the final position. In the corrupted run, the final position has already read the corrupted information from position 9 at earlier layers, so patching position 9 later has no downstream reader left. The flow of information is visible directly in where patching works.

??? question "What does the L1 coefficient trade off in an SAE, and what are dead features?"
    Higher L1 gives sparser codes (lower L0), which tend to be more interpretable, but worse reconstruction (higher variance unexplained), and the reverse for lower L1. Dead features are dictionary entries that never activate on any input. They waste capacity and are usually handled by resampling them during training or by using TopK activations.

??? question "Why do induction heads need a previous-token head in an earlier layer?"
    At the second `A`, the induction head must find the position whose *previous* token was `A`. A single attention head can match only on what is stored at each position. A previous-token head first writes "my predecessor was A" into each position's residual stream, and the induction head then matches its query (current token A) against that stored information and copies the token found there.

## Further reading

- Alain & Bengio, [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644) (2016).
- Hewitt & Liang, [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368) (2019).
- Belrose et al., [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112) (2023).
- Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (2021).
- Olsson et al., [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) (2022).
- Meng et al., [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (2022).
- Wang et al., [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593) (2022).
- Heimersheim & Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255) (2024).
- Elhage et al., [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) (2022).
- Bricken et al., [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (2023).
- Cunningham et al., [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600) (2023).
- Templeton et al., [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (2024).
- Gao et al., [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093) (2024).
- Lieberum et al., [Gemma Scope](https://arxiv.org/abs/2408.05147) (2024).
- Turner et al., [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248) (2023).
- Zou et al., [Representation Engineering](https://arxiv.org/abs/2310.01405) (2023).
- Arditi et al., [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (2024).
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [nnsight](https://nnsight.net/), [SAELens](https://github.com/jbloomAus/SAELens).
