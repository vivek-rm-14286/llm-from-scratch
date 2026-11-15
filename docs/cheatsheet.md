# Cheat Sheet

The formulas, memory rules and default settings used across the site, on one page. Each row links to the lesson that derives it; treat the defaults as starting points and read the linked section before changing them.

Symbols: $N$ parameters, $D$ training tokens, $L$ layers, $d$ model width, $T$ sequence length, $V$ vocabulary size, $H$ attention heads, $B$ batch size.

## Compute

| Quantity | Formula | Derived in |
|---|---|---|
| Training FLOPs | $C \approx 6ND$ (2 forward + 4 backward per parameter per token) | [Scaling Laws & Compute](llm/scaling-laws.md) |
| With the attention term | $\text{FLOPs/token} \approx 6N + 12\,L\,d\,T$ | [MFU](pretraining/training.md#mfu-are-you-using-the-gpu) |
| MFU | $\dfrac{\text{tokens/s} \times \text{FLOPs/token}}{\text{peak FLOP/s}}$; roughly 35-50% when well tuned, below ~25% means something is wrong | [MFU](pretraining/training.md#mfu-are-you-using-the-gpu) |
| Training time | $\dfrac{\text{FLOPs/token} \times D}{n_{\text{GPU}} \times \text{peak FLOP/s} \times \text{MFU}}$ | [Estimating training time and cost](llm/scaling-laws.md#estimating-training-time-and-cost) |
| Compute-optimal data | about 20 tokens per parameter (Chinchilla); small models train far past it | [Scaling Laws & Compute](llm/scaling-laws.md) |
| Tokens per optimizer step | micro-batch × $T$ × grad-accum steps × number of GPUs; 0.25M-1M for 100M-500M models | [Batch size in tokens](pretraining/training.md#batch-size-in-tokens) |
| Step-0 loss | $\ln V$ (10.80 for $V$ = 49,152) | [Debugging a Training Run](pretraining/debugging.md#1-the-step-0-loss-should-be-lnvocab) |

Peak dense bf16 throughput: H100 989 TFLOP/s, A100 312 TFLOP/s. H100 memory bandwidth: about 3.35 TB/s.

**Project 9 as the worked example** ([Build a 500M SLM From Scratch](projects/slm-500m.md)): 486,605,824 parameters, 32 layers, $d$ = 1024, $T$ = 2048.

| Run | Total FLOPs | H100-hours at 40% MFU | On 8×H100 |
|---|---|---|---|
| 10B tokens | 3.7 × 10¹⁹ | about 26 | about 3.5 h |
| 50B tokens | 1.9 × 10²⁰ | about 131 | about 16 h |

Rental price: roughly USD 2-4 per H100-hour.

## Parameter counts

| Model | Formula | Derived in |
|---|---|---|
| GPT-2-style block | $12d^2$ per block ($4d^2$ attention + $8d^2$ MLP), so $N \approx 12Ld^2$ plus embeddings $V d$ (tied) | [The Transformer, Line by Line](llm/transformer.md) |
| SwiGLU MLP | $3\,d\,d_{\text{ff}}$; $d_{\text{ff}} = \tfrac{8}{3}d$ matches a $4d$ GELU MLP | [Modern Architectures & SLM Design](llm/modern-architectures.md) |
| GQA + SwiGLU block (Project 9) | $2d \cdot H d_h + 2d \cdot H_{kv} d_h + 3\,d\,d_{\text{ff}} + 2d$, with $d_h = d/H$ | [Project 9](projects/slm-500m.md) |
| Whole model (tied embeddings) | $L \times \text{block} + Vd + d$ | [Project 9](projects/slm-500m.md) |
| LoRA adapter on one matrix | $r\,(d_{\text{in}} + d_{\text{out}})$ | [LoRA & QLoRA](finetuning/lora-qlora.md) |

Check for Project 9 ($H$ = 16, $H_{kv}$ = 4, $d_{\text{ff}}$ = 3584, $V$ = 49,152): block = 13,633,536; 32 blocks + 50,331,648 embedding + 1,024 = 486,605,824.

## Memory

| What | Rule | Derived in |
|---|---|---|
| Training, mixed precision + AdamW | about 16 bytes per parameter (weights, gradients, fp32 master copy, two Adam moments), before activations. 500M parameters: about 8 GB | [Training memory](start/setup.md#training-memory) |
| Inference weights | bf16: 2 bytes per parameter; int8: 1; 4-bit: about 0.56 (4.5 bits effective) | [Memory math](inference/serving.md#memory-math) |
| Inference total | weights + KV cache + roughly 10-20% runtime overhead | [Inference memory](start/setup.md#inference-memory) |
| KV cache | $2 \times L \times H_{kv} \times d_h \times \text{bytes per element} \times \text{tokens} \times B$ | [KV cache memory](inference/inference.md#kv-cache-memory) |
| Decode time per token (batch 1) | (weight bytes + KV bytes read) / memory bandwidth | [Why decode is memory-bound](inference/inference.md#why-decode-is-memory-bound) |

| Example | Value |
|---|---|
| 7B model weights, bf16 / 4-bit | 14 GB / about 3.5 GB plus quantization scales |
| 8B model weights, bf16 / int8 / int4 | 16 GB / 8 GB / about 4.5 GB |
| KV cache, Llama 3 8B (32 layers, 8 KV heads, head dim 128, bf16) | 128 KiB per token; 1 GiB per sequence at 8K context |
| KV cache, Project 9 (32 layers, 4 KV heads, head dim 64, bf16) | 32 KiB per token; 64 MiB per sequence at 2,048 tokens |
| 8B full fine-tune, bf16 + AdamW | about 128 GB before activations |
| 8B LoRA r=16, bf16 base | about 17 GB before activations |
| 8B QLoRA r=16, NF4 base | about 6.5 GB before activations (base about 5.5-6 GB, trainable state about 0.7 GB) |

The three fine-tuning rows are from [Memory math: 7B/8B on a 24 GB GPU](finetuning/lora-qlora.md).

## Default hyperparameters

| Setting | Pretraining (Project 9, 10B run) | SFT, full fine-tune | LoRA / QLoRA | DPO |
|---|---|---|---|---|
| Learning rate | 8e-4 peak (1e-3 for the 50B run) | 1e-5 to 2e-5 at 7B-8B; up to ~5e-5 to 1e-4 below 1B | 2e-4 (1e-4 to 3e-4) | 5e-7 to 5e-6 full; 5e-6 to 5e-5 with LoRA |
| Schedule | WSD: 1,000 warmup steps, decay over the last 20% | linear or cosine, ~3% warmup | cosine or linear, 3% warmup | cosine, 10% warmup in the example |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 | AdamW | AdamW | AdamW |
| Batch | 524,288 tokens per step ($T$ = 2048) | 64 to 256 sequences (effective) | as SFT | 4 × 8 accumulation in the example |
| Length | one pass over the data | 1 to 3 epochs | 2 epochs in the example | 1 epoch in the example |
| Method-specific | gradient clip 1.0, bf16 autocast | max length 2048 to 4096; loss on assistant tokens only | `r` = 16, `lora_alpha` = 2 × r, dropout 0.05, `target_modules="all-linear"` | $\beta$ = 0.1 (0.01 to 0.5) |
| Source | [Project 9](projects/slm-500m.md), [The Training Loop](pretraining/training.md) | [SFT: Hyperparameters](post-training/sft.md#hyperparameters) | [Hyperparameter defaults](finetuning/lora-qlora.md#hyperparameter-defaults) | [DPO from scratch](post-training/preference.md#dpo-from-scratch) |

PPO on small discrete-control tasks ([Hyperparameter tips](decision/policy-methods.md#hyperparameter-tips)):

| Hyperparameter | Typical |
|---|---|
| Learning rate | 2.5e-4 to 3e-4 (Adam, `eps=1e-5`) |
| Rollout size (T × N) | 1k to 8k steps |
| Epochs per rollout / minibatches | 3 to 10 / 4 to 32 |
| Clip ε | 0.1 to 0.2 |
| γ, λ | 0.99, 0.95 |
| Entropy coef / value coef | 0.0 to 0.01 / 0.5 |
| Grad-norm clip | 0.5 |
| Target KL (optional early stop) | 0.01 to 0.03 |

## Tensor shapes

For batch $B$, sequence length $T$, width $d$, $H$ heads, vocabulary $V$ ([Shapes table](llm/transformer.md#shapes-table)):

| Step | Tensor | Shape |
|---|---|---|
| Input | `idx` | `(B, T)` |
| Token embedding | `tok_emb(idx)` | `(B, T, d)` |
| Block input/output | `x` | `(B, T, d)` |
| Fused QKV | `qkv(x)` | `(B, T, 3d)` |
| Per-head Q, K, V | `q, k, v` | `(B, H, T, d/H)` |
| Attention scores | `q @ k.T` | `(B, H, T, T)` |
| MLP hidden | `fc(x)` | `(B, T, 4d)` |
| Logits | `lm_head(x)` | `(B, T, V)` |
| Next-token logits in `generate()` | `logits[:, -1, :]` | `(B, V)` |
| Targets | `idx` shifted left by one | `(B, T)` |

## Sampling defaults

| Use case | Reasonable settings |
|---|---|
| Code, extraction, math final answers | greedy, or `temperature=0.2` |
| Chat assistant | `temperature=0.7`, `top_p=0.9` (or the model card's recommendation) |
| Creative writing | `temperature=0.9-1.1`, `min_p=0.05` |
| Reasoning models | follow the model card; several recommend roughly `temperature=0.6` and warn against greedy |
| Repetition penalty, if used | 1.05 to 1.2 |

Order of operations: penalties on raw logits, then temperature, then truncation (top-k, min-p, top-p). From [Sampling strategies](inference/inference.md#sampling-strategies).

## RAG metrics

| Metric | Definition | Use |
|---|---|---|
| Recall@k | fraction of the relevant items that appear in the top $k$ | The first number to look at: the model can only use what was retrieved |
| MRR | mean over queries of $1/\text{rank}$ of the first relevant result | Rewards putting the right result near the top |
| nDCG@k | $\text{DCG@k} / \text{IDCG@k}$, with $\text{DCG@k} = \sum_{i=1}^{k} \dfrac{\text{rel}_i}{\log_2(i+1)}$ | Graded relevance with a position discount |
| RRF | $\text{RRF}(d) = \sum_{r} \dfrac{1}{k + \text{rank}_r(d)}$, $k = 60$ | Fusing BM25 and dense rankings without tuning |

Definitions from [Retrieval metrics](rag/advanced.md#retrieval-metrics) and [Reciprocal rank fusion](rag/advanced.md#reciprocal-rank-fusion).

## Evaluation noise

| Quantity | Formula |
|---|---|
| Standard error of an accuracy | $\sqrt{p(1-p)/n}$; 95% interval $\approx \pm 1.96\,\text{SE}$ |
| Example | 200 items at 85%: about ±5 points |

From [How big must an eval set be?](pretraining/evaluation.md#how-big-must-an-eval-set-be).
