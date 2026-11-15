# Choose Your Path

The [Learning Roadmap](roadmap.md) covers the whole site in 29 weeks. If the goal is narrower, most of it can be skipped. This page lists six reading paths, each with its pages in order, the reading time, and the projects that go with it.

!!! abstract "What you'll learn"
    - Which pages to read, and in which order, for each of six goals
    - How many hours of reading each path is, and how long its projects take
    - What each path leaves out, so you know what to come back for

## How to read the numbers

**Reading hours** are the sum of the reading times shown at the top of each page, project pages included. They cover reading only. Running the code and doing the Build it tasks usually takes at least as long again, so plan for two to three times the reading figure.

**Project hours** are separate estimates of hands-on time, not measurements. They exclude time spent waiting for training runs. The [projects overview](../projects/index.md#the-projects-side-by-side) has the level, hardware and cost for each project.

| Path | Reading hours | Project hours (estimate) | Projects |
|---|---|---|---|
| [RAG only](#rag-only) | 4 | 12-16 | Project 5 |
| [Fine-tune an open model only](#fine-tune-an-open-model-only) | 5.5 | 12-16 | Project 6 (7 optional) |
| [Build an SLM from scratch](#build-an-slm-from-scratch) | 7 | 48-67 | Projects 1, 2, 9 (7 optional) |
| [Decision models only](#decision-models-only) | 4 | 18-24 | Projects 3, 4 |
| [Agents and production](#agents-and-production) | 5.5 | 26-36 | Projects 5, 8 |
| [Everything](#everything) | 16.5 | 114-157 | All nine, in order |

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
14. [Build a Production RAG System](../projects/support-rag.md): Project 5

**Reading:** about 4 hours. **Project:** an estimated 12-16 hours of hands-on work.

Left out: how models are trained. A RAG system often fails because of the generator, and [Fine-tuning](../finetuning/index.md) is the next thing to read when prompting is not enough. The [Agents and production](#agents-and-production) path is the natural follow-up if retrieval becomes one tool among several: it continues from this one to Project 8.

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
19. [Fine-tune an Open-Weight Model](../projects/incident-triage-finetune.md): Project 6

**Reading:** about 5.5 hours. **Project:** an estimated 12-16 hours of hands-on work.

Optional extension: [Preference Optimization (RLHF, DPO)](../post-training/preference.md), then [Post-train a Small Model: SFT, Then DPO](../projects/post-train-assistant.md) (about 17 more minutes of reading and an estimated 10-14 hours of hands-on work). Project 7 is the sibling of Project 6: SFT and then DPO with training loops written by hand, on a model small enough for a laptop. Do it if you have comparison data (one answer preferred over another) instead of gold answers, or to see what the libraries in Project 6 do for you.

Left out: pretraining at scale and RL-based post-training. If the task is a choice among fixed options, read [Decision Models for Routing & Classification](../decision/routing-models.md) and look at Project 3 before fine-tuning a generative model.

## Build an SLM from scratch

For understanding and training a language model end to end. The path climbs from a neural network in numpy (Project 1) to a 10M-parameter GPT on a laptop (Project 2) to the 500M model (Project 9). Only the full run of Project 9 needs rented data-center GPUs.

1. [Start Here overview](index.md)
2. [Setup, Hardware & Budget](setup.md): all of it, including the section on rented GPU nodes
3. [Foundations overview](../foundations/index.md)
4. [Math You Actually Need](../foundations/math.md)
5. [Python & PyTorch](../foundations/pytorch.md)
6. [Neural Networks & Backprop](../foundations/neural-networks.md)
7. [Neural Network From Scratch](../projects/nn-from-scratch.md): Project 1
8. [Language Models overview](../llm/index.md)
9. [Tokenization](../llm/tokenization.md)
10. [Embeddings & Attention](../llm/attention.md)
11. [The Transformer, Line by Line](../llm/transformer.md)
12. [Modern Architectures & SLM Design](../llm/modern-architectures.md)
13. [Scaling Laws & Compute](../llm/scaling-laws.md)
14. [Tiny GPT](../projects/tiny-gpt.md): Project 2
15. [Pretraining overview](../pretraining/index.md)
16. [Data Pipelines](../pretraining/data.md)
17. [The Training Loop](../pretraining/training.md)
18. [Debugging a Training Run](../pretraining/debugging.md)
19. [Distributed Training](../pretraining/distributed.md)
20. [GPU Performance Engineering](../pretraining/gpu-performance.md): on a first pass, only the throughput checklist and the profiling section
21. [Evaluation](../pretraining/evaluation.md)
22. [Build a 500M SLM From Scratch](../projects/slm-500m.md): Project 9, Part 0 and Part A
23. [Post-training overview](../post-training/index.md)
24. [Supervised Fine-Tuning (SFT)](../post-training/sft.md)
25. [Decoding: Sampling, KV Cache & Batching](../inference/inference.md): sampling and the KV cache used by `chat.py`

**Reading:** about 7 hours. **Projects:** an estimated 48-67 hours of hands-on work. Project 9 Part B (SFT, chat and release) comes after the last three pages in the list.

Optional extension, before Part B: [Preference Optimization (RLHF, DPO)](../post-training/preference.md), then [Post-train a Small Model: SFT, Then DPO](../projects/post-train-assistant.md) (about 17 more minutes of reading and an estimated 10-14 hours of hands-on work). Project 7 writes by hand the SFT loop that Part B runs as a script, and adds DPO.

Left out: fine-tuning existing models, RAG and applications. The electives [Architecture Families & Beyond Transformers](../llm/architecture-families.md) and [Multimodal Models](../llm/multimodal.md) are a good follow-up once the decoder-only model is clear.

## Decision models only

For models that choose instead of write: routers and classifiers with calibrated probabilities (Project 3), then reinforcement learning, imitation learning and small fast policies (Project 4). Everything runs on a laptop.

1. [Start Here overview](index.md)
2. [Setup, Hardware & Budget](setup.md): the environment sections only; no GPU is needed
3. [Foundations overview](../foundations/index.md)
4. [Math You Actually Need](../foundations/math.md)
5. [Python & PyTorch](../foundations/pytorch.md)
6. [Neural Networks & Backprop](../foundations/neural-networks.md)
7. [Tokenization](../llm/tokenization.md): Project 3 feeds text to a pretrained encoder, and its input is tokens
8. [Embeddings & Attention](../llm/attention.md): what an embedding is, and what the encoder computes
9. [Decision Models overview](../decision/index.md)
10. [Decision Models for Routing & Classification](../decision/routing-models.md)
11. [Ticket Router: a Calibrated Decision Model](../projects/ticket-router.md): Project 3
12. [Reinforcement Learning Basics](../decision/rl-basics.md)
13. [Imitation Learning](../decision/imitation.md)
14. [DQN & PPO](../decision/policy-methods.md)
15. [System 1 & System 2 Models](../decision/system-one.md)
16. [A Decision Model That Plays](../projects/game-agent.md): Project 4

**Reading:** about 4 hours. **Projects:** an estimated 18-24 hours of hands-on work.

Left out: almost everything about language models. For Project 4 alone, skip the two Language Models pages and the routing page. [Reasoning & RL with Verifiable Rewards](../post-training/reasoning-rl.md) shows the same policy-gradient ideas applied to language models, but it assumes the Language Models and Post-training pages.

## Agents and production

For building applications on top of existing models and operating them: retrieval, tools, MCP, guardrails and evaluation gates. No GPU is needed; both projects call an LLM through Ollama or any OpenAI-compatible API.

1. [Start Here overview](index.md)
2. [Setup, Hardware & Budget](setup.md): the environment sections; skip GPU renting
3. [Math You Actually Need](../foundations/math.md): vectors, dot products and cosine similarity are what retrieval runs on
4. [Tokenization](../llm/tokenization.md): context budgets are counted in tokens
5. [Embeddings & Attention](../llm/attention.md): what an embedding is
6. [Inference overview](../inference/index.md)
7. [Decoding: Sampling, KV Cache & Batching](../inference/inference.md)
8. [Quantization & Serving](../inference/serving.md): running the local endpoint that both projects call
9. [RAG overview](../rag/index.md)
10. [RAG Fundamentals](../rag/fundamentals.md)
11. [Advanced RAG & Evaluation](../rag/advanced.md)
12. [Build a Production RAG System](../projects/support-rag.md): Project 5
13. [Applications overview](../applications/index.md)
14. [Prompt & Context Engineering](../applications/prompting.md)
15. [Agents & Tool Use](../applications/agents.md)
16. [Safety & Production overview](../production/index.md)
17. [Safety, Red-Teaming & Guardrails](../production/safety.md)
18. [LLMOps: Shipping & Operating Models](../production/llmops.md)
19. [Decision Models for Routing & Classification](../decision/routing-models.md): the router that the agent uses to pick a tool or a team
20. [On-Call Agent: Tools, MCP, Guardrails](../projects/oncall-agent.md): Project 8

**Reading:** about 5.5 hours. **Projects:** an estimated 26-36 hours of hands-on work.

Project 8 integrates three earlier builds: the Project 5 service, the Project 3 router and the Project 6 fine-tuned model. This path builds only the first. Add the [Decision models only](#decision-models-only) path up to Project 3 and the [fine-tuning path](#fine-tune-an-open-model-only) if the agent should use your own router and your own model.

Left out: how models are trained, and [Interpretability](../production/interpretability.md).

## Everything

Follow the [Learning Roadmap](roadmap.md): every core page in a 29-week order, with all nine projects in order of difficulty. The roadmap does not follow the sidebar order. It interleaves the tracks so that each project comes right after the pages it needs: Decision Models moves forward to weeks 8-11, and most of the Pretraining module moves back to weeks 25-27, next to Project 9.

**Reading:** about 16.5 hours for all 60 pages, of which about 1 hour are the two electives ([Architecture Families & Beyond Transformers](../llm/architecture-families.md), [Multimodal Models](../llm/multimodal.md)) that no project depends on. **Projects:** an estimated 114-157 hours of hands-on work across all nine.

The roadmap budgets more than reading plus projects (29 weeks at 8-12 hours a week), because it also includes running the code on each page, the Build it tasks, the weekly exercises, and slack for runs that have to be repeated.

## Combining paths

The paths share their first pages, so adding a second one costs less than its total:

- **RAG, then agents.** The agents path starts with the RAG path. After Project 5 it adds the serving, applications, production and routing pages, and Project 8.
- **RAG, then fine-tuning.** After the RAG path, the fine-tuning path adds the PyTorch, backprop, transformer, training-loop, SFT, synthetic-data and fine-tuning pages, and Project 6.
- **Fine-tuning, then building an SLM.** After the fine-tuning path, the SLM path adds Projects 1 and 2, the rest of Language Models, the rest of Pretraining, and Project 9.
- **Decision models, then anything else.** The Foundations pages are shared with every other path.

## Check your understanding

??? question "You want to ship a support chatbot over your company's documents in a month. Which path, and what do you skip?"
    The RAG path with Project 5. Skip Foundations beyond the math page, all of Pretraining and Post-training, and Decision Models. Add the fine-tuning path later only if evaluation shows that prompting plus retrieval is not good enough.

??? question "Why does the fine-tuning path include The Training Loop, a Pretraining page?"
    Fine-tuning is the same optimization loop on a smaller scale. Reading a loss curve, choosing a learning rate and warmup, and recognizing overfitting or divergence are all taught there, and they are what you need when a QLoRA run misbehaves.

??? question "Why does the SLM path include Projects 1 and 2 instead of going straight to Project 9?"
    Project 9 supplies complete code and its full run costs money. Projects 1 and 2 are where backpropagation, the tokenizer and the GPT are written by hand, for free, on a laptop. With those done, Project 9's code reads as a scaled-up version of something already built, and its bugs are cheaper to find.

??? question "The reading time for a path is 5 hours. How much calendar time should you plan?"
    Roughly 10-15 hours for the pages once you run the code and do the Build it tasks, plus the projects' hands-on estimate, plus waiting time for any training runs. At 8-12 hours a week that is usually three to five weeks for a path with one project.
