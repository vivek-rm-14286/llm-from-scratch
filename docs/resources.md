# Resources

A short, curated list of the courses, books, papers, and code that are most worth your time alongside this site. Each item is grouped with the section of the site it supports best.

## Foundations

- **Andrej Karpathy, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)**: a video series that starts by building micrograd (a tiny autograd engine) and ends with a GPT. The best companion to [Foundations](foundations/index.md) and [Language Models](llm/index.md).
- **Karpathy, [micrograd](https://github.com/karpathy/micrograd)**: the roughly 100-line autograd engine the [Neural Networks](foundations/neural-networks.md#a-micrograd-style-autograd-engine) page is modeled on.
- **3Blue1Brown, [Neural Networks](https://www.3blue1brown.com/topics/neural-networks)** and **[Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)**: visual intuition for the math.
- **Goodfellow, Bengio, Courville, [Deep Learning](https://www.deeplearningbook.org/)**: the standard textbook, free to read online.
- **[Dive into Deep Learning](https://d2l.ai/)**: an interactive book with runnable code.

## Language models and transformers

- **Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (2017): the paper that introduced the transformer.
- **Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**: the most widely used visual walkthrough of the architecture.
- **Sebastian Raschka, *Build a Large Language Model (From Scratch)*** ([book](https://www.manning.com/books/build-a-large-language-model-from-scratch), [code](https://github.com/rasbt/LLMs-from-scratch)): a book-length, code-first build of a GPT-style model, through pretraining and fine-tuning.
- **Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** (GPT-2) and **Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)** (GPT-3).

## Pretraining and scaling

- **Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)**: a small, readable codebase for training GPT-2-class models. A useful reference for [Project 1](projects/p1-500m-model.md).
- **[Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)**: a course that builds every part of the LM stack, from tokenizer to training to alignment, with demanding assignments.
- **Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** and **Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)** (Chinchilla).
- **Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054)** and **Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740)**.

## Post-training and fine-tuning

- **Hugging Face, [smol-course](https://github.com/huggingface/smol-course)**: a hands-on course on aligning and fine-tuning small language models with TRL, runnable on modest hardware.
- **Hu et al., [LoRA](https://arxiv.org/abs/2106.09685)** and **Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314)**.
- **Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)**.

## RAG

- **Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)** (2020): the paper that named RAG.

## Decision models and reinforcement learning

- **Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)** (2nd edition): the standard RL textbook, free online. Read chapters 1-6 and 13 alongside [Decision Models](decision/index.md).
- **OpenAI, [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)**: a clear introduction to deep RL with key algorithms (including PPO) explained and implemented.
- **Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)**: the PPO paper, used in [Project 4](projects/p4-decision-model.md).
- **Hugging Face, [Deep RL Course](https://huggingface.co/learn/deep-rl-course)**: a free, practical course with hands-on environments.

## Tools

- [PyTorch tutorials](https://pytorch.org/tutorials/), [uv](https://docs.astral.sh/uv/), [Weights & Biases docs](https://docs.wandb.ai/), [TensorBoard](https://www.tensorflow.org/tensorboard), [Hugging Face Hub docs](https://huggingface.co/docs/hub/index). See [Setup](start/setup.md).
