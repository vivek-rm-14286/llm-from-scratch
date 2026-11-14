# Supervised Fine-Tuning (SFT)

SFT is the least glamorous and most important step of post-training. Mechanically it's just pretraining on a different dataset: same model, same next-token cross-entropy loss. Three details make it work: a **chat template** that marks who is speaking, a **loss mask** so the model learns to write answers rather than questions, and **data** good enough to be worth imitating. Get those right and a base model turns into a usable assistant in a few GPU-hours.

!!! abstract "What you'll learn"
    - How instruction data is structured (single-turn, multi-turn, system prompts)
    - Chat templates: ChatML, the Llama 3 format, and why template mismatches are silent killers
    - Loss masking on prompt tokens, implemented from scratch on top of your own GPT
    - Packing multiple conversations into one sequence, and the cross-contamination gotcha
    - Which datasets to start from, and why quality beats quantity
    - A complete from-scratch SFT loop, and the same job with TRL's `SFTTrainer`
    - How to evaluate a chat model without fooling yourself

## Instruction data formats

Almost every modern SFT dataset uses the same "messages" format: a list of turns, each with a `role` and `content`.

```json
{
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What's 17 * 23?"},
    {"role": "assistant", "content": "17 * 23 = 391."},
    {"role": "user", "content": "And divided by 3?"},
    {"role": "assistant", "content": "391 / 3 ≈ 130.33."}
  ]
}
```

Older datasets use Alpaca-style `instruction` / `input` / `output` columns or ShareGPT-style `conversations` with `from: human / gpt`. Convert everything to `messages` up front; one format for your whole pipeline removes a class of bugs.

Roles you'll see:

- **system**: persistent instructions (persona, rules, tools available). Optional.
- **user**: the human turn.
- **assistant**: the model turn. These are the only tokens you usually train on.
- **tool** (or `ipython`): results returned from a tool call, in agentic datasets.

## Chat templates

The model only sees a flat token sequence, so the messages must be serialized with special tokens that mark role boundaries. This serialization is the **chat template**. The model learns it during SFT, so you must use *exactly the same template* at inference.

=== "ChatML (Qwen, SmolLM, many others)"

    ```text
    <|im_start|>system
    You are a concise assistant.<|im_end|>
    <|im_start|>user
    What's 17 * 23?<|im_end|>
    <|im_start|>assistant
    17 * 23 = 391.<|im_end|>
    ```

=== "Llama 3 style"

    ```text
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a concise assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

    What's 17 * 23?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    17 * 23 = 391.<|eot_id|>
    ```

The pieces that matter:

1. **Role markers** (`<|im_start|>user`, `<|start_header_id|>user<|end_header_id|>`) tell the model whose turn it is.
2. **End-of-turn token** (`<|im_end|>`, `<|eot_id|>`) is what the model emits to say "I'm done". If the model never learns to produce it, it rambles into fake user turns. Your inference stop condition must check for it.
3. **Generation prompt**: at inference you render the conversation and append the opening of an assistant turn (`<|im_start|>assistant\n`), then let the model continue.

In Hugging Face, the template is a Jinja string stored in `tokenizer_config.json`, applied with:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
messages = [{"role": "user", "content": "What's 17 * 23?"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
# <|im_start|>system\nYou are a helpful AI assistant named SmolLM...<|im_end|>\n
# <|im_start|>user\nWhat's 17 * 23?<|im_end|>\n<|im_start|>assistant\n
```

Note that this template injects a default system prompt. Templates often have such quirks, which is one more reason to always print the rendered string once before training.

!!! tip "Special tokens must be real tokens"
    `<|im_end|>` should be a single token ID, not the seven characters `<`, `|`, `im`, ... If you're adding a template to your own tokenizer (see [Tokenization](../llm/tokenization.md)), register the markers as special tokens and resize the embedding matrix. Fresh embedding rows start random, so give the model enough SFT steps to learn them.

## Loss masking: train on answers, not questions

During pretraining every token is a target. In SFT, if you also train on user turns, the model spends capacity learning to *write user questions*, and with short answers the prompt tokens can dominate the loss. The standard fix: compute the loss only on assistant tokens (including their end-of-turn token).

We mark non-target positions with label `-100`, which PyTorch's `cross_entropy` skips via `ignore_index`.

```text
tokens:  <|im_start|> user \n What 's 2+2 ? <|im_end|> \n <|im_start|> assistant \n 4 <|im_end|> \n
labels:   -100      -100 ... -100 ...         -100       ...       -100           4  <|im_end|> -100
```

### Implementing it from scratch

We'll build directly on the GPT from [The Transformer, Line by Line](../llm/transformer.md). All we need is a model with `forward(idx) -> logits` of shape `(B, T, vocab)` and a tokenizer with `encode(str) -> list[int]` that recognizes the special tokens.

The trick for a correct mask is to **tokenize segment by segment**: render each turn's "header" (never trained) and "body" (trained only if it's an assistant turn) separately, then concatenate the IDs. That way you never have to search for marker positions in a token stream.

```python title="sft_data.py"
import torch

IGNORE = -100

def render_chatml(messages, encode, im_start_id, im_end_id, nl_ids):
    """Tokenize a conversation in ChatML, returning (input_ids, labels).

    encode: str -> list[int] for ordinary text (no special tokens)
    im_start_id / im_end_id: ids of <|im_start|> and <|im_end|>
    nl_ids: token ids for "\n"
    """
    ids, labels = [], []
    for msg in messages:
        header = [im_start_id] + encode(msg["role"]) + nl_ids
        body = encode(msg["content"]) + [im_end_id]  # (1)!
        trailer = nl_ids

        ids += header
        labels += [IGNORE] * len(header)

        ids += body
        if msg["role"] == "assistant":
            labels += body  # learn the answer AND when to stop
        else:
            labels += [IGNORE] * len(body)

        ids += trailer
        labels += [IGNORE] * len(trailer)
    return ids, labels


def collate(batch, pad_id, max_len):
    """Right-pad a list of (ids, labels) into tensors, truncating to max_len."""
    batch = [(i[:max_len], l[:max_len]) for i, l in batch]
    T = max(len(i) for i, _ in batch)
    x = torch.full((len(batch), T), pad_id, dtype=torch.long)
    y = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    for b, (i, l) in enumerate(batch):
        x[b, : len(i)] = torch.tensor(i)
        y[b, : len(l)] = torch.tensor(l)
    return x, y
```

1. The `<|im_end|>` token is part of the trained body. This is how the model learns to stop.

The loss is the usual shifted next-token loss. Position $t$ predicts token $t+1$, so we compare `logits[:, :-1]` with `labels[:, 1:]`:

```python title="sft_loss.py"
import torch.nn.functional as F

def sft_loss(model, x, y):
    logits = model(x)                       # (B, T, V)
    logits = logits[:, :-1, :]              # predictions for positions 1..T-1
    targets = y[:, 1:]                      # what should have been predicted
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,                  # masked prompt + padding positions
    )
```

Mathematically, with mask $m_t = 1$ for assistant tokens and $0$ otherwise:

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_t m_t} \sum_{t} m_t \log \pi_\theta(x_t \mid x_{<t})
$$

!!! note "Per-token vs. per-sequence averaging"
    `cross_entropy` with `ignore_index` averages over *all unmasked tokens in the batch*, so long answers contribute more gradient than short ones. That's usually what you want. Some recipes average per sequence first; it rarely matters much, but be aware that changing it changes the effective learning rate when you use gradient accumulation (the token count differs between micro-batches). For exactness with accumulation, sum the losses and divide by the total target tokens across all micro-batches.

Because padding is on the right and the model is causal, padded positions can't affect real positions. You don't need an attention mask for right-padding in a decoder-only model; the `-100` labels handle the loss.

## Packing

Conversations vary wildly in length. If you pad every batch to the longest example, most of your compute is spent on pad tokens. **Packing** concatenates several examples into one fixed-length sequence:

```text
[conv A .................][conv B .....][conv C ..........][pad]
```

Simple and effective, with one gotcha: with a plain causal mask, tokens of conversation B can attend to conversation A. This "cross-contamination" is mostly harmless in pretraining but noticeably hurts SFT when examples are short and unrelated. Two fixes:

1. **Block-diagonal attention** so each example only attends to itself, plus **position IDs that reset** at each example boundary. Flash-attention's variable-length kernels do this efficiently; in HF this is what padding-free packing does.
2. Accept contamination but keep labels masked properly. Cheap and common for small experiments.

Here is greedy packing that also returns per-token document IDs, which you can turn into a block-diagonal mask for your own GPT:

```python title="packing.py"
import torch

def pack(examples, seq_len, pad_id):
    """examples: list of (ids, labels). Returns tensors x, y, doc of shape (N, seq_len)."""
    rows, cur_ids, cur_lab, cur_doc = [], [], [], []
    doc_id = 0
    for ids, labels in examples:
        ids, labels = ids[:seq_len], labels[:seq_len]
        if len(cur_ids) + len(ids) > seq_len:  # flush current row
            rows.append((cur_ids, cur_lab, cur_doc))
            cur_ids, cur_lab, cur_doc = [], [], []
        cur_ids += ids
        cur_lab += labels
        cur_doc += [doc_id] * len(ids)
        doc_id += 1
    if cur_ids:
        rows.append((cur_ids, cur_lab, cur_doc))

    N = len(rows)
    x = torch.full((N, seq_len), pad_id, dtype=torch.long)
    y = torch.full((N, seq_len), -100, dtype=torch.long)
    doc = torch.full((N, seq_len), -1, dtype=torch.long)
    for r, (i, l, d) in enumerate(rows):
        x[r, : len(i)] = torch.tensor(i)
        y[r, : len(l)] = torch.tensor(l)
        doc[r, : len(d)] = torch.tensor(d)
    return x, y, doc


def block_causal_mask(doc):
    """(B, T) doc ids -> (B, 1, T, T) boolean mask, True = may attend."""
    T = doc.size(1)
    causal = torch.ones(T, T, dtype=torch.bool, device=doc.device).tril()
    same_doc = doc[:, :, None] == doc[:, None, :]
    return (causal & same_doc)[:, None, :, :]
```

You can pass this mask as `attn_mask` to `F.scaled_dot_product_attention` inside your attention block. If you also reset position IDs per document (compute them with a cumulative count within each `doc` run), each packed example sees exactly what it would see alone.

## Datasets

Good starting points on the Hugging Face Hub (all in `messages` format or trivially convertible):

| Dataset | Size (roughly) | Notes |
|---|---|---|
| `HuggingFaceTB/smoltalk` | ~1M examples | Used for SmolLM2. Mix of synthetic subsets (Magpie-style, rewriting, summarization, math, constraints). Has smaller subsets, good for small models. |
| `allenai/tulu-3-sft-mixture` | ~940k examples | Tülu 3's SFT mix: math, code, precise instruction following, safety, multilingual, with decontamination. |
| `teknium/OpenHermes-2.5` | ~1M examples | Widely used synthetic GPT-4-generated mix. ShareGPT format (`conversations` with `from`/`value`). |
| `HuggingFaceH4/ultrachat_200k` | ~200k | Filtered multi-turn synthetic chats, used for Zephyr. |

For a first run on a small model, take 10k to 50k examples from smoltalk and train for 1 to 3 epochs. That's enough to get clean chat behavior.

### Quality > quantity

The SFT model imitates its data, warts included. Concrete filters that pay off:

- **Deduplicate** (exact and near-duplicate) prompts. Repeated prompts get memorized.
- **Decontaminate** against your eval sets (n-gram overlap with benchmark questions).
- **Drop broken examples**: truncated responses, refusals to benign requests ("As an AI language model..."), responses in the wrong language, code that doesn't parse.
- **Balance the mix.** If 70% of your data is math, the model's chat skills suffer. Tülu 3 tuned its mix with ablations per skill.
- **Match response style to what you want.** If every response is 800 words with headers, your model will answer "hi" with a report.

A few thousand hand-checked, diverse, high-quality examples routinely beat 100k sloppy ones (the LIMA result). In practice you want both: a decent volume of good synthetic data, plus aggressive filtering.

## Hyperparameters

SFT is fine-tuning, not pretraining, so everything is gentler. Reasonable defaults for full fine-tuning:

| Hyperparameter | Typical value | Notes |
|---|---|---|
| Learning rate | 1e-5 to 2e-5 (7B-8B); up to ~5e-5 to 1e-4 for sub-1B models | Tülu 3 8B used 5e-6. With LoRA, 1e-4 to 2e-4. |
| Schedule | Linear or cosine decay, ~3% warmup | |
| Epochs | 1 to 3 | More epochs overfit fast; watch held-out loss. |
| Effective batch | 64 to 256 sequences | Use gradient accumulation. |
| Max length | 2048 to 4096 | Whatever covers ~95%+ of your data. |
| Weight decay | 0 to 0.1 | |
| Precision | bf16 | |

If held-out loss rises while training loss drops after epoch 1, you're memorizing. Stop earlier or add data.

## A from-scratch SFT loop

Putting it together with your own GPT (`model(idx) -> logits`) and tokenizer:

```python title="train_sft.py"
import math, random, torch
from sft_data import render_chatml, collate
from sft_loss import sft_loss

def train_sft(model, tokenizer_fns, conversations, *, pad_id, max_len=1024,
              batch_size=8, grad_accum=4, epochs=2, lr=2e-5, warmup_frac=0.03,
              device="cuda"):
    encode, im_start, im_end, nl = tokenizer_fns
    data = [render_chatml(c, encode, im_start, im_end, nl) for c in conversations]
    # Drop examples where truncation removed every target token.
    data = [d for d in data if any(t != -100 for t in d[1][:max_len])]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(data) / (batch_size * grad_accum))
    total = steps_per_epoch * epochs
    warmup = max(1, int(warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    model.to(device).train()
    step = 0
    for epoch in range(epochs):
        random.shuffle(data)
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        for i, batch in enumerate(batches):
            x, y = collate(batch, pad_id, max_len)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                loss = sft_loss(model, x, y) / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0 or i == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{total} "
                          f"loss {loss.item() * grad_accum:.3f} "
                          f"lr {sched.get_last_lr()[0]:.2e}")
    return model
```

And chat inference using the same template (greedy for clarity; see [Sampling](../inference/inference.md) for better decoding):

```python title="chat.py"
import torch

@torch.no_grad()
def chat(model, messages, encode, decode, im_start, im_end, nl,
         max_new_tokens=256, device="cuda"):
    ids = []
    for m in messages:
        ids += [im_start] + encode(m["role"]) + nl + encode(m["content"]) + [im_end] + nl
    ids += [im_start] + encode("assistant") + nl  # generation prompt
    x = torch.tensor([ids], device=device)
    out = []
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]
        nxt = logits.argmax(-1, keepdim=True)
        if nxt.item() == im_end:   # the stop token we trained on
            break
        out.append(nxt.item())
        x = torch.cat([x, nxt], dim=1)
    return decode(out)
```

## The same job with TRL

For real models, use Hugging Face TRL. Its `SFTTrainer` handles chat templates, masking, packing, and LoRA. The API has moved a lot between versions; the code below targets **TRL ≥ 0.20** (`pip install "trl>=0.20" transformers datasets`), where `max_seq_length` was renamed `max_length` and `assistant_only_loss` exists.

```python title="trl_sft.py"
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

model_id = "HuggingFaceTB/SmolLM2-360M"   # a base model
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")  # (1)!

ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")
# ds has "train" and "test" splits with a "messages" column

config = SFTConfig(
    output_dir="smollm2-360m-sft",
    max_length=2048,
    packing=False,                  # (2)!
    assistant_only_loss=True,       # (3)!
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model_id,
    args=config,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model()
```

1. Borrow the instruct tokenizer for its chat template, since the base tokenizer may not have one. The ChatML markers are already in SmolLM2's vocabulary.
2. `packing=True` packs examples; recent TRL versions default to padding-free "BFD" packing with position-ID resets when the attention implementation supports it.
3. `assistant_only_loss` needs a chat template that marks assistant spans with Jinja `{% generation %}` tags. If yours doesn't, TRL will error; fall back to the prompt/completion dataset format with `completion_only_loss=True`, or patch the template.

!!! info "If your TRL version complains"
    Older tutorials use `DataCollatorForCompletionOnlyLM`, `dataset_text_field`, `max_seq_length`, or `tokenizer=` in the trainer. Those were removed or renamed across the 0.1x → 0.2x releases. Check `pip show trl` and the [SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer) for your version.

## Evaluating chat models

Training loss tells you the model fits the data, not that it's a good assistant. Layer your evals:

1. **Eyeball 20 fixed prompts after every run.** Include multi-turn, a refusal-worthy request, a formatting instruction, and something the base model knew. Catches template bugs instantly.
2. **Held-out SFT loss** for overfitting.
3. **Capability benchmarks** with a harness (lm-evaluation-harness, lighteval): MMLU, GSM8K, HumanEval-style code. These should stay roughly flat versus the base model; SFT doesn't add knowledge but can destroy some.
4. **Instruction following**: IFEval checks verifiable constraints ("use no commas", "answer in 3 bullets") programmatically.
5. **LLM-as-judge**: MT-Bench, AlpacaEval 2 (length-controlled), Arena-Hard. Useful for chat quality, but judges favor long answers and their own style. Prefer length-controlled metrics.

More on harnesses in [Evaluation](../pretraining/evaluation.md).

!!! warning "Common pitfalls"
    - **Template mismatch** between training and inference (or between `apply_chat_template` and your own renderer). Print the rendered string; diff the token IDs.
    - **Forgetting to train on the end-of-turn token.** The model never learns to stop and invents the next user turn.
    - **Tokens mapped to `-100` everywhere** after truncation. Long system prompts can push the whole answer past `max_len`; filter those examples.
    - **Pad token = EOS token with no masking.** If you set `pad_token = eos_token` and then train on pads, the model learns to emit EOS constantly, or never (if you mask all EOS). Mask pads via labels, not via token identity.
    - **Learning rate from pretraining.** 3e-4 on a 7B model during SFT will wreck it.
    - **Too many epochs** on small data: lovely training loss, parroted answers.

## Check your understanding

??? question "Why tokenize the conversation segment-by-segment instead of rendering the whole string and tokenizing once?"
    BPE merges can cross the boundary between a marker and the following text, so the token boundaries in a fully rendered string may not line up with where you think each turn starts. Tokenizing segments separately guarantees you know exactly which IDs belong to which turn, so the mask is correct by construction. The cost is tiny differences from "natural" tokenization at boundaries, which is fine as long as inference uses the same rendering.

??? question "Your SFT model answers correctly but then writes 'User: thanks!' and keeps going. What's wrong?"
    It never learned to emit the end-of-turn token: either `<|im_end|>` was masked out of the labels, it was tokenized as plain characters instead of a special token, or your inference loop stops on a different token than the one you trained on.

??? question "What does packing without block-diagonal attention cost you?"
    Tokens in later examples can attend to unrelated earlier examples in the same row, so the model learns from contexts it will never see at inference. Labels are still correct, so it mostly costs some quality on short examples. Block-diagonal masks with position-ID resets remove the problem at no extra compute.

??? question "Why should MMLU stay roughly flat after SFT, and what does it mean if it drops sharply?"
    MMLU measures knowledge, which comes from pretraining. SFT changes behavior, not knowledge. A sharp drop means you're damaging the base model: learning rate too high, too many epochs, or a narrow data mix causing catastrophic forgetting. See [Fine-tuning](../finetuning/index.md).

## Further reading

- Zhou et al., *LIMA: Less Is More for Alignment* - [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- Wang et al., *Self-Instruct* - [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
- Lambert et al., *Tülu 3* (Section on SFT data curation) - [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)
- Allal et al., *SmolLM2* (smoltalk construction) - [arXiv:2502.02737](https://arxiv.org/abs/2502.02737)
- Zhou et al., *Instruction-Following Evaluation for LLMs* (IFEval) - [arXiv:2311.07911](https://arxiv.org/abs/2311.07911)
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* - [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- Hugging Face, *Chat templates* - [huggingface.co/docs/transformers/chat_templating](https://huggingface.co/docs/transformers/chat_templating)
- TRL `SFTTrainer` docs - [huggingface.co/docs/trl/sft_trainer](https://huggingface.co/docs/trl/sft_trainer)
