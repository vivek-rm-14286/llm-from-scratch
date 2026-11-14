# Projects

Reading only gets you so far. These four projects make you build each part of the stack yourself, end to end. Each one has a design walkthrough, complete runnable code in the [repository](https://github.com/vivek-rm-14286/llm-from-scratch/tree/main/code), a list of milestones, and stretch goals.

<div class="grid cards" markdown>

-   **1. Build a ~500M SLM from scratch**

    ---

    Train a byte-level BPE tokenizer, stream and shard FineWeb-Edu, write a Llama-style model (RMSNorm, RoPE, SwiGLU, GQA), pretrain it with DDP and bf16, run SFT to make it a chat model, evaluate it, convert it to Hugging Face format and publish it.

    *Prerequisites:* [Transformer](../llm/transformer.md), [Modern architectures](../llm/modern-architectures.md), [Pretraining](../pretraining/index.md), [SFT](../post-training/sft.md)

    [:octicons-arrow-right-24: Start project 1](p1-500m-model.md)

-   **2. Fine-tune an open-weight model: Incident Triage Copilot**

    ---

    An on-call engineer pastes raw alerts, log lines and metrics. The model returns strict JSON with severity, category, probable root cause, next steps and escalation target. You will measure a prompted baseline, generate and review training data, run QLoRA, evaluate field by field, then serve the model.

    *Prerequisites:* [Fine-tuning](../finetuning/index.md), [LoRA & QLoRA](../finetuning/lora-qlora.md), [Inference](../inference/inference.md)

    [:octicons-arrow-right-24: Start project 2](p2-finetune.md)

-   **3. Production RAG: Support Engineer Copilot**

    ---

    Answer support questions from product docs and past resolved tickets, with citations, hybrid search, reranking and per-document access levels, served by a FastAPI service and checked by an evaluation harness.

    *Prerequisites:* [RAG fundamentals](../rag/fundamentals.md), [Advanced RAG](../rag/advanced.md)

    [:octicons-arrow-right-24: Start project 3](p3-rag.md)

-   **4. A decision model that plays**

    ---

    Design an original 2D game, record demonstrations, train a policy by behavior cloning, improve it with your own PPO, then export it to ONNX as a real-time System 1 controller. There is also an in-browser demo you can train live.

    *Prerequisites:* [RL basics](../decision/rl-basics.md), [DQN & PPO](../decision/policy-methods.md), [Imitation learning](../decision/imitation.md)

    [:octicons-arrow-right-24: Start project 4](p4-decision-model.md)

</div>

!!! note "Order"
    Projects 3 and 4 need no GPU, so you can do them early. Project 2 needs one 24 GB GPU, or a smaller model on Colab. Do project 1 last, because it uses nearly everything on this site.
