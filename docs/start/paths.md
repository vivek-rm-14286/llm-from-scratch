# Choose Your Path

The [Learning Roadmap](roadmap.md) covers the whole site in 21 weeks. If the goal is narrower, most of it can be skipped. This page lists five reading paths, each with its pages in order, the reading time, and the project that goes with it.

!!! abstract "What you'll learn"
    - Which pages to read, and in which order, for each of five goals
    - How many hours of reading each path is, and how long its project takes
    - What each path leaves out, so you know what to come back for

## How to read the numbers

**Reading hours** are the sum of the reading times shown at the top of each page. They cover reading only. Running the code and doing the Build it tasks usually takes at least as long again, so plan for two to three times the reading figure.

**Project hours** are separate estimates of hands-on time, not measurements. They exclude time spent waiting for training runs. The [projects overview](../projects/index.md#the-projects-side-by-side) has the hardware and cost for each project.

| Path | Reading hours | Project hours (estimate) | Project |
|---|---|---|---|
| [RAG only](#rag-only) | 4 | 12-16 | Project 3 |
| [Fine-tune an open model only](#fine-tune-an-open-model-only) | 5.5 | 12-16 | Project 2 |
| [Build an SLM from scratch](#build-an-slm-from-scratch) | 6.5 | 34-47 | Project 1 |
| [Decision models only](#decision-models-only) | 3.5 | 10-14 | Project 4 |
| [Everything](#everything) | 16 | 68-93 | All four |

## RAG only

For building retrieval systems on top of existing models. No GPU is needed, and no model is trained.

1. [Start Here overview](index.md)
2. [Setup, Hardware & Budget](setup.md): the environment sections; skip GPU renting
3. [Math You Actually Need](../foundations/math.md): vectors, dot products and cosine similarity are what retrieval runs on
4. [Tokenization](../llm/tokenization.md): chunk sizes and context budgets are counted in tokens
5. [Embeddings & Attention](../llm/attention.md): what an embedding is
6. [Inference overview](../inference/index.md)
7. [Decoding: Sampling, KV Cache & Batching](../inference/inference.md): sampling settings and what limits the generator's speed
8. [RAG overview](../rag/index.md)
9. [RAG Fundamentals](../rag/fundamentals.md)
10. [Advanced RAG & Evaluation](../rag/advanced.md)
11. [Prompt & Context Engineering](../applications/prompting.md): grounded prompts, citations, structured output
12. [Safety, Red-Teaming & Guardrails](../production/safety.md): prompt injection through retrieved documents
13. [LLMOps: Shipping & Operating Models](../production/llmops.md): evals in CI, observability and cost
14. [Build a Production RAG System](../projects/p3-rag.md): Project 3

**Reading:** about 4 hours. **Project:** an estimated 12-16 hours of hands-on work.

Left out: how models are trained. A RAG system often fails because of the generator, and [Fine-tuning](../finetuning/index.md) is the next thing to read when prompting is not enough. [Agents & Tool Use](../applications/agents.md) is the natural follow-up if retrieval becomes one tool among several.

## Fine-tune an open model only

For adapting an existing open-weight model to one task with LoRA or QLoRA. It needs one 24 GB GPU, or a free Colab GPU with a smaller model.

1. [Start Here overview](index.md)
2. [Setup, Hardware & Budget](setup.md): including the GPU memory math
3. [Math You Actually Need](../foundations/math.md)
4. [Python & PyTorch](../foundations/pytorch.md)
5. [Neural Networks & Backprop](../foundations/neural-networks.md)
6. [Tokenization](../llm/tokenization.md)
7. [Embeddings & Attention](../llm/attention.md)
8. [The Transformer, Line by Line](../llm/transformer.md): the modules LoRA attaches to
9. [The Training Loop](../pretraining/training.md): loss curves, learning-rate schedules and mixed precision
10. [Supervised Fine-Tuning (SFT)](../post-training/sft.md): chat templates and loss masking
11. [Distillation & Synthetic Data](../post-training/distillation-synthetic-data.md): building a training set with a generator and a teacher model
12. [Fine-tuning overview](../finetuning/index.md)
13. [LoRA & QLoRA](../finetuning/lora-qlora.md)
14. [Continued Pretraining & Model Merging](../finetuning/merging-continued-pretraining.md)
15. [Inference overview](../inference/index.md)
16. [Decoding: Sampling, KV Cache & Batching](../inference/inference.md)
17. [Quantization & Serving](../inference/serving.md)
18. [Prompt & Context Engineering](../applications/prompting.md): the prompted baseline and structured output
19. [Fine-tune an Open-Weight Model](../projects/p2-finetune.md): Project 2

**Reading:** about 5.5 hours. **Project:** an estimated 12-16 hours of hands-on work.

Left out: pretraining at scale, preference optimization and RL-based post-training. Read [Preference Optimization](../post-training/preference.md) next if you have comparison data (one answer preferred over another) instead of gold answers.

## Build an SLM from scratch

For understanding and training a language model end to end. Part 0 of the project runs on a laptop; the full run needs rented data-center GPUs.

1. [Start Here overview](index.md)
2. [Setup, Hardware & Budget](setup.md): all of it, including the section on rented GPU nodes
3. [Foundations overview](../foundations/index.md)
4. [Math You Actually Need](../foundations/math.md)
5. [Python & PyTorch](../foundations/pytorch.md)
6. [Neural Networks & Backprop](../foundations/neural-networks.md)
7. [Language Models overview](../llm/index.md)
8. [Tokenization](../llm/tokenization.md)
9. [Embeddings & Attention](../llm/attention.md)
10. [The Transformer, Line by Line](../llm/transformer.md)
11. [Modern Architectures & SLM Design](../llm/modern-architectures.md)
12. [Scaling Laws & Compute](../llm/scaling-laws.md)
13. [Pretraining overview](../pretraining/index.md)
14. [Data Pipelines](../pretraining/data.md)
15. [The Training Loop](../pretraining/training.md)
16. [Debugging a Training Run](../pretraining/debugging.md)
17. [Distributed Training](../pretraining/distributed.md)
18. [GPU Performance Engineering](../pretraining/gpu-performance.md): on a first pass, only the throughput checklist and the profiling section
19. [Evaluation](../pretraining/evaluation.md)
20. [Build a 500M SLM From Scratch](../projects/p1-500m-model.md): Project 1, Part 0 and Part A
21. [Post-training overview](../post-training/index.md)
22. [Supervised Fine-Tuning (SFT)](../post-training/sft.md)
23. [Decoding: Sampling, KV Cache & Batching](../inference/inference.md): sampling and the KV cache used by `chat.py`

**Reading:** about 6.5 hours. **Project:** an estimated 34-47 hours of hands-on work. Project 1 Part B (SFT, chat and release) comes after the last three pages in the list.

Left out: fine-tuning existing models, RAG and applications. The electives [Architecture Families & Beyond Transformers](../llm/architecture-families.md) and [Multimodal Models](../llm/multimodal.md) are a good follow-up once the decoder-only model is clear.

## Decision models only

For models that choose actions: reinforcement learning, imitation learning and small fast policies. Everything runs on a laptop CPU.

1. [Start Here overview](index.md)
2. [Setup, Hardware & Budget](setup.md): the environment sections only; no GPU is needed
3. [Foundations overview](../foundations/index.md)
4. [Math You Actually Need](../foundations/math.md)
5. [Python & PyTorch](../foundations/pytorch.md)
6. [Neural Networks & Backprop](../foundations/neural-networks.md)
7. [Decision Models overview](../decision/index.md)
8. [Reinforcement Learning Basics](../decision/rl-basics.md)
9. [Imitation Learning](../decision/imitation.md)
10. [DQN & PPO](../decision/policy-methods.md)
11. [System 1 & System 2 Models](../decision/system-one.md)
12. [A Decision Model That Plays](../projects/p4-decision-model.md): Project 4
13. [Decision Models for Routing & Classification](../decision/routing-models.md): uses a pretrained encoder; skim [Tokenization](../llm/tokenization.md) first if tokens are new to you

**Reading:** about 3.5 hours. **Project:** an estimated 10-14 hours of hands-on work.

Left out: everything about language models. [Reasoning & RL with Verifiable Rewards](../post-training/reasoning-rl.md) shows the same policy-gradient ideas applied to language models, but it assumes the Language Models and Post-training pages.

## Everything

Follow the [Learning Roadmap](roadmap.md): every core page in a 21-week order, with all four projects. The sidebar order differs from it in one place: the roadmap moves Decision Models and Project 4 forward to weeks 11-12.

**Reading:** about 16 hours for all 55 pages, of which about 1 hours are the two electives ([Architecture Families & Beyond Transformers](../llm/architecture-families.md), [Multimodal Models](../llm/multimodal.md)) that no project depends on. **Projects:** an estimated 68-93 hours of hands-on work across all four.

The roadmap budgets more than reading plus projects (21 weeks at 8-12 hours a week), because it also includes running the code on each page, the Build it tasks, the weekly exercises, and slack for runs that have to be repeated.

## Combining paths

The paths share their first pages, so adding a second one costs less than its total:

- **RAG, then fine-tuning.** After the RAG path, the fine-tuning path adds the PyTorch, backprop, transformer, training-loop, SFT, synthetic-data and fine-tuning pages, and Project 2.
- **Fine-tuning, then building an SLM.** After the fine-tuning path, the SLM path adds the rest of Language Models, the rest of Pretraining, and Project 1.
- **Decision models, then anything else.** The Foundations pages are shared with every other path.

## Check your understanding

??? question "You want to ship a support chatbot over your company's documents in a month. Which path, and what do you skip?"
    The RAG path with Project 3. Skip Foundations beyond the math page, all of Pretraining and Post-training, and Decision Models. Add the fine-tuning path later only if evaluation shows that prompting plus retrieval is not good enough.

??? question "Why does the fine-tuning path include The Training Loop, a Pretraining page?"
    Fine-tuning is the same optimization loop on a smaller scale. Reading a loss curve, choosing a learning rate and warmup, and recognizing overfitting or divergence are all taught there, and they are what you need when a QLoRA run misbehaves.

??? question "The reading time for a path is 5 hours. How much calendar time should you plan?"
    Roughly 10-15 hours for the pages once you run the code and do the Build it tasks, plus the project's hands-on estimate, plus waiting time for any training runs. At 8-12 hours a week that is usually three to five weeks for a single path.
