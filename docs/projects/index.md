# Projects

Reading only gets you so far. These four projects make you build each part of the stack yourself, end to end. Each one has a design walkthrough, complete runnable code in the [repository](https://github.com/vivek-rm-14286/llm-from-scratch/tree/main/code), a list of milestones, and stretch goals. You can run and modify the supplied code, or write each file yourself and compare it with the supplied one; Projects 1 and 2 list the checks that tell you when your version is right.

<div class="grid cards" markdown>

-   **1. Build a ~500M SLM from scratch**

    ---

    A laptop warm-up first: a 31M model on TinyStories. **Part A:** train a byte-level BPE tokenizer, stream and shard FineWeb-Edu, write a Llama-style model (RMSNorm, RoPE, SwiGLU, GQA), pretrain it with DDP and bf16, and evaluate it. **Part B:** run SFT to make it a chat model, convert it to Hugging Face format and publish it.

    *Prerequisites:* Part A: [Transformer](../llm/transformer.md), [Modern architectures](../llm/modern-architectures.md), [Pretraining](../pretraining/index.md). Part B: [SFT](../post-training/sft.md)

    [Start project 1](p1-500m-model.md)

-   **2. Fine-tune an open-weight model: Incident Triage Copilot**

    ---

    An on-call engineer pastes raw alerts, log lines and metrics. The model returns strict JSON with severity, category, probable root cause, next steps and escalation target. You will measure a prompted baseline, generate and review training data, run QLoRA, evaluate field by field, then serve the model.

    *Prerequisites:* [SFT](../post-training/sft.md), [Fine-tuning](../finetuning/index.md), [LoRA & QLoRA](../finetuning/lora-qlora.md). Used as reference: [Inference](../inference/inference.md), [Structured output](../applications/prompting.md#structured-output)

    [Start project 2](p2-finetune.md)

-   **3. Production RAG: Support Engineer Copilot**

    ---

    Answer support questions from product docs and past resolved tickets, with citations, hybrid search, reranking and per-document access levels, served by a FastAPI service and checked by an evaluation harness.

    *Prerequisites:* [RAG fundamentals](../rag/fundamentals.md), [Advanced RAG](../rag/advanced.md)

    [Start project 3](p3-rag.md)

-   **4. A decision model that plays**

    ---

    Design an original 2D game, record demonstrations, train a policy by behavior cloning, improve it with your own PPO, then export it to ONNX as a real-time System 1 controller. There is also an in-browser demo you can train live.

    *Prerequisites:* [RL basics](../decision/rl-basics.md), [DQN & PPO](../decision/policy-methods.md), [Imitation learning](../decision/imitation.md)

    [Start project 4](p4-decision-model.md)

</div>

## The projects side by side

Costs and times are estimates, and no GPU run was measured for this table. "Your time" is hands-on work at the roadmap's pace, and excludes the reading.

| | Hardware | Rough compute cost | Compute time | Your time | Deliverable | Prerequisite modules |
|---|---|---|---|---|---|---|
| **1, Part 0** | Laptop (Apple Silicon) or free Colab GPU | USD 0 | About 2.5 hours on an M1 Pro for 50M tokens | 3-4 hours | A 31.5M model that writes simple stories | Language Models |
| **1, Part A** | CPU machine for data; 8x H100 node (or 1 H100) rented for the run | Roughly USD 80-155 for the 10B-token run, including a 1.5x safety factor | About 3.5 hours on 8x H100, about 26 hours on one | 25-35 hours over 4 weeks | A 486.6M base model with validation loss and HellaSwag | Language Models, Pretraining |
| **1, Part B** | One rented data-center GPU | Roughly USD 6-20 | 3-5 hours | 6-8 hours | A chat model on the Hub in Hugging Face format | Post-training (the SFT page) |
| **2** | One 24 GB GPU for the 7B model, or 16 GB / Colab T4 with the 1.5B model. Baseline also runs on a Mac | A few USD of rented 24 GB GPU time, or USD 0 on Colab | Training: an hour or less | 12-16 hours | A QLoRA adapter that beats a prompted baseline, served and re-evaluated | Post-training (SFT), Fine-tuning; Inference as reference |
| **3** | Laptop. No GPU. An LLM through Ollama or any OpenAI-compatible API for generation | USD 0 locally | Minutes | 12-16 hours | A FastAPI RAG service with citations, access control and an evaluation report | RAG |
| **4** | Laptop CPU | USD 0 | Minutes | 10-14 hours | A PPO-tuned policy exported to ONNX, with a held-out evaluation | Foundations, Decision Models |

The cheap path for Project 1 Part A is the `p125m` preset on one 24 GB GPU: about 12 hours on an RTX 4090 for 2.5B tokens.

!!! note "Order"
    Projects 3 and 4 need no GPU, so you can do them early. Project 2 needs one 24 GB GPU for the 7B model, or 16 GB / Colab T4 with the 1.5B model. Project 1 is the most expensive; the [roadmap](../start/roadmap.md) starts it in week 7 and launches the full run in week 10. If you mainly want applied skills, do Projects 2 and 3 first. [Choose Your Path](../start/paths.md) lists the pages each project needs.
