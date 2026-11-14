# Glossary

Short definitions of the terms used across the site, in alphabetical order. Each entry links to the page where the idea is explained in depth.


## A

Activation
:   The output of a layer, or the nonlinear function (ReLU, GELU, SiLU) applied after a linear layer. Saved activations dominate training memory at long sequence lengths. See [Neural Networks](foundations/neural-networks.md#activation-functions).

Activation checkpointing
:   Saving only some activations in the forward pass and recomputing the rest during backward, trading extra compute for less memory. See [The Training Loop](pretraining/training.md).

Adam / AdamW
:   Adaptive optimizers that keep moving averages of the gradient and squared gradient for each parameter. AdamW applies weight decay directly to the weights and is the standard LLM optimizer. See [Neural Networks](foundations/neural-networks.md#optimizers).

Advantage
:   In RL, how much better an action was than the policy's average action from that state: $A = Q(s,a) - V(s)$. Used to weight policy-gradient updates. See [DQN & PPO](decision/policy-methods.md).

Agent
:   A system where a model repeatedly observes, decides on an action (often a tool call), and sees the result, until the task is done or a step limit is hit. See [Agents & Tool Use](applications/agents.md).

Alignment
:   Making a model's behavior match human intent: helpful, honest, and safe. In practice done with SFT and preference optimization. See [Post-training](post-training/index.md).

Arithmetic intensity
:   FLOPs performed per byte moved from memory. Low-intensity operations (normalization, decode-time attention) are memory-bound, and high-intensity ones (large matmuls) are compute-bound. See [GPU Performance Engineering](pretraining/gpu-performance.md).

Attention
:   An operation where each position computes a weighted average of other positions' value vectors, with weights from query-key dot products. See [Embeddings & Attention](llm/attention.md).

Autograd
:   Automatic differentiation: recording operations during the forward pass and applying the chain rule backward to compute gradients. See [Python & PyTorch](foundations/pytorch.md#autograd).

Autoregressive
:   Generating a sequence one element at a time, each conditioned on all previous ones. Decoder-only LLMs are autoregressive.

## B

Backpropagation
:   The algorithm for computing gradients of a loss with respect to every parameter by applying the chain rule from the output backward. See [Neural Networks](foundations/neural-networks.md#backpropagation-derived).

Bandit (multi-armed bandit)
:   The simplest RL problem: repeated choices among options with unknown payoffs and no state, which isolates the explore/exploit trade-off. A contextual bandit also sees a feature vector each round. See [RL Basics](decision/rl-basics.md#multi-armed-bandits-exploration-in-its-purest-form).

Base model
:   A model after pretraining only, before any post-training such as SFT. It continues text rather than following instructions. See [Pretraining](pretraining/index.md).

Batch size
:   The number of examples (or, for LLMs, tokens) used to compute one gradient update. See [The Training Loop](pretraining/training.md).

Behavior cloning
:   Imitation learning by supervised learning on (state, expert action) pairs. See [Imitation Learning](decision/imitation.md).

Bellman equation
:   The recursion that defines a value function: the value of a state or action equals the expected immediate reward plus the discounted value of what follows. The optimality version takes a max over next actions and is what Q-learning and DQN solve. See [RL Basics](decision/rl-basics.md#bellman-equations).

bfloat16 (bf16)
:   A 16-bit floating-point format with the same exponent range as fp32 but fewer mantissa bits. The default for training on modern GPUs. See [Python & PyTorch](foundations/pytorch.md#dtypes).

BM25
:   A classic keyword-based ranking function used for sparse retrieval and in hybrid search. See [RAG Fundamentals](rag/fundamentals.md).

BPE (byte-pair encoding)
:   A tokenization algorithm that starts from bytes or characters and repeatedly merges the most frequent adjacent pair into a new token. See [Tokenization](llm/tokenization.md).

Broadcasting
:   Automatic expansion of size-1 dimensions so tensors of different shapes can be combined elementwise. See [Python & PyTorch](foundations/pytorch.md#broadcasting).

## C

Calibration
:   The property that predicted probabilities match observed frequencies: of the predictions made with confidence 0.8, about 80% are correct. Measured with ECE and usually fixed with temperature scaling, which divides the logits by one scalar fitted on held-out data. See [Routing & Classification](decision/routing-models.md#calibration).

Canary deployment
:   Sending a small share of live traffic to a new model version and comparing its metrics with the current version before a full rollout. See [LLMOps](production/llmops.md).

Causal mask
:   A mask that prevents each position in attention from seeing later positions, so the model cannot peek at the token it is predicting. See [Embeddings & Attention](llm/attention.md).

Chain rule
:   The calculus rule that the derivative of a composition is the product of the derivatives of its parts. The basis of backpropagation. See [Math](foundations/math.md#the-chain-rule).

Chat template
:   The exact format (special tokens, role markers) a chat model expects for multi-turn conversations. Getting it wrong silently degrades fine-tuning. See [SFT](post-training/sft.md).

Checkpoint
:   A saved snapshot of model weights, and usually optimizer state, from which training or inference can resume.

Chinchilla scaling
:   The finding (Hoffmann et al., 2022) that compute-optimal training uses roughly 20 tokens per parameter. See [Scaling Laws & Compute](llm/scaling-laws.md).

Chunking
:   Splitting documents into passages for embedding and retrieval in RAG. See [RAG Fundamentals](rag/fundamentals.md).

CLIP
:   Contrastive Language-Image Pretraining: an image encoder and a text encoder trained so matching image-caption pairs have similar embeddings. CLIP-style encoders are the vision tower in many VLMs. See [Multimodal Models](llm/multimodal.md).

Constrained decoding
:   Masking the logits at each step so only tokens allowed by a grammar or JSON schema can be sampled. The output is valid by construction. See [Prompt & Context Engineering](applications/prompting.md).

Context window
:   The maximum number of tokens a model can attend to at once.

Continued pretraining
:   Further next-token training of an existing base model on domain data, usually mixed with general data (replay) to limit forgetting. See [Continued Pretraining & Model Merging](finetuning/merging-continued-pretraining.md).

Cosine similarity
:   The dot product of two vectors divided by their lengths. The standard way to compare embeddings. See [Math](foundations/math.md#vectors-and-the-dot-product).

Cross-entropy
:   The loss $-\log q(y)$ for the correct class $y$; the training objective of language models. See [Math](foundations/math.md#cross-entropy).

## D

DAgger (dataset aggregation)
:   An imitation-learning algorithm that runs the learner, asks the expert to label the states the learner visited, adds them to the dataset, and retrains. It fixes the distribution shift of plain behavior cloning. See [Imitation Learning](decision/imitation.md#dagger-dataset-aggregation).

DARE
:   Drop And REscale: randomly zeroing a fraction p of a task vector's entries and scaling the rest by 1/(1-p) before merging, to reduce interference. See [Continued Pretraining & Model Merging](finetuning/merging-continued-pretraining.md).

Data parallelism (DDP)
:   Training with a copy of the model on each GPU, each processing different data, and averaging gradients. See [Distributed Training](pretraining/distributed.md).

Decision Transformer
:   A GPT-style model trained on logged trajectories of (return-to-go, state, action) tokens to predict the next action. At test time you condition it on the return you want. See [Imitation Learning](decision/imitation.md#the-decision-transformer-rl-as-sequence-modeling).

Decoder-only
:   A transformer with only causally masked self-attention blocks, as in GPT, Llama, and Qwen. See [The Transformer](llm/transformer.md).

Deduplication
:   Removing exact or near-duplicate documents from training data. See [Data Pipelines](pretraining/data.md).

Distillation (knowledge distillation)
:   Training a smaller student model to match a larger teacher's output distribution (soft targets) or generations. This is how many strong SLMs are built, and how a slow decision-maker is turned into a fast policy. See [Distillation & Synthetic Data](post-training/distillation-synthetic-data.md) and [System 1 & System 2 Models](decision/system-one.md#1-imitation-of-a-slow-expert-distillation).

DPO (direct preference optimization)
:   A method that trains directly on preference pairs with a classification-style loss, without a separate reward model or RL loop. See [Preference Optimization](post-training/preference.md).

DQN (deep Q-network)
:   An RL algorithm that learns action values with a neural network, using a replay buffer and target network. See [DQN & PPO](decision/policy-methods.md).

Dropout
:   Randomly zeroing activations during training to reduce overfitting. See [Neural Networks](foundations/neural-networks.md#regularization).

## E

ECE (expected calibration error)
:   The sample-weighted average gap between confidence and accuracy across confidence bins. Zero means perfectly calibrated. See [Routing & Classification](decision/routing-models.md#reliability-diagrams-and-ece).

Einsum
:   Einstein summation notation for tensor contractions, e.g. `"bhqd,bhkd->bhqk"`. See [Python & PyTorch](foundations/pytorch.md#einsum).

Embedding
:   A learned vector representing a discrete item (a token) or a dense vector representing a whole text for retrieval. See [Embeddings & Attention](llm/attention.md) and [RAG Fundamentals](rag/fundamentals.md).

Encoder-decoder
:   A transformer with a bidirectional encoder over the input and an autoregressive decoder that attends to it through cross-attention, as in T5 and Whisper. See [Architecture Families](llm/architecture-families.md).

Entropy
:   The expected surprise of a distribution, $-\sum p \log p$. Used as an exploration bonus in RL. See [Math](foundations/math.md#entropy).

Episode
:   One run of an RL agent in an environment, from reset until it terminates (a real end state) or is truncated (for example by a time limit). See [RL Basics](decision/rl-basics.md).

Epoch
:   One full pass over the training dataset.

## F

Few-shot prompting
:   Including a few input-output examples in the prompt so the model infers the task.

Fine-tuning
:   Continuing training of a pretrained model on a smaller, task-specific dataset. See [Fine-tuning](finetuning/index.md).

FlashAttention
:   A fused, memory-efficient attention kernel that avoids materializing the full attention matrix. See [Inference](inference/inference.md).

FLOPs
:   Floating-point operations. Training a dense transformer costs roughly $6ND$ FLOPs. See [Scaling Laws & Compute](llm/scaling-laws.md).

FP8
:   8-bit floating-point formats: E4M3 (more precision, used for weights and activations) and E5M2 (more range, used for gradients). Supported by tensor cores on recent GPUs for faster training and inference. See [GPU Performance Engineering](pretraining/gpu-performance.md).

FSDP
:   Fully Sharded Data Parallel: data parallelism that also shards parameters, gradients, and optimizer state across GPUs. See [Distributed Training](pretraining/distributed.md).

Function calling
:   A model emitting a structured tool call (a tool name plus JSON arguments) instead of text. The application runs the tool and returns the result to the model. Also called tool calling. See [Agents & Tool Use](applications/agents.md).

## G

GAE (generalized advantage estimation)
:   An advantage estimate that sums discounted one-step TD errors with weight $(\gamma\lambda)^l$. The parameter $\lambda$ trades the bias of a one-step estimate against the variance of a full Monte Carlo return. See [DQN & PPO](decision/policy-methods.md#generalized-advantage-estimation-gae).

GELU
:   Gaussian Error Linear Unit, a smooth ReLU-like activation used in GPT-2 and BERT. See [Neural Networks](foundations/neural-networks.md#activation-functions).

GQA (grouped-query attention)
:   Attention where several query heads share one key/value head, shrinking the KV cache. See [Modern Architectures](llm/modern-architectures.md).

Gradient
:   The vector of partial derivatives of a function; it points in the direction of steepest increase. See [Math](foundations/math.md#partial-derivatives-and-gradients).

Gradient accumulation
:   Summing gradients over several micro-batches before one optimizer step, to simulate a larger batch. See [Python & PyTorch](foundations/pytorch.md#gradient-accumulation).

Gradient clipping
:   Rescaling gradients when their norm exceeds a threshold, to prevent destabilizing updates.

GRPO
:   Group Relative Policy Optimization: a PPO-style method that estimates advantages by comparing several sampled answers to the same prompt, without a value network. See [Reasoning & RL](post-training/reasoning-rl.md).

Guardrail
:   A check outside the model that filters or validates inputs and outputs, such as PII redaction, a safety classifier, or schema validation. See [Safety, Red-Teaming & Guardrails](production/safety.md).

## H

Hallucination
:   A model generating fluent but false or unsupported content. RAG with citations is a common mitigation. See [RAG](rag/index.md).

Hybrid search
:   Combining sparse (keyword, BM25) and dense (embedding) retrieval. See [Advanced RAG](rag/advanced.md).

Hyperparameter
:   A setting chosen before training rather than learned, such as learning rate, batch size, or number of layers.

## I

Imitation learning
:   Learning a policy from expert demonstrations rather than from reward. See [Imitation Learning](decision/imitation.md).

Inference
:   Running a trained model to produce outputs. See [Inference](inference/index.md).

Initialization
:   How weights are set before training (Xavier, He, small normal). Bad initialization can make training explode or stall. See [Neural Networks](foundations/neural-networks.md#initialization).

Instruction tuning
:   Supervised fine-tuning on instruction-response pairs so a base model follows instructions. See [SFT](post-training/sft.md).

## J

Jailbreak
:   An input crafted to make a model ignore its safety training and produce content it would normally refuse. See [Safety, Red-Teaming & Guardrails](production/safety.md).

## K

Kernel fusion
:   Combining several GPU operations into one kernel so intermediate results stay in fast on-chip memory instead of making round trips to HBM. See [GPU Performance Engineering](pretraining/gpu-performance.md).

KL divergence
:   A measure of how one probability distribution differs from another; asymmetric and non-negative. Used as a penalty in RLHF and PPO. See [Math](foundations/math.md#kl-divergence).

Knowledge distillation
:   See Distillation.

KV cache
:   Stored keys and values for past tokens during generation, so each new token does not recompute them. See [Decoding: Sampling, KV Cache & Batching](inference/inference.md).

## L

LayerNorm
:   Normalization of each example's features to zero mean and unit variance, with learned scale and shift. See [Neural Networks](foundations/neural-networks.md#normalization-and-residual-connections).

Learning rate
:   The step size of gradient descent; the most important hyperparameter. See [Neural Networks](foundations/neural-networks.md#learning-rate-schedules).

Linear probe
:   A linear classifier trained on frozen hidden states to test whether a concept is linearly decodable from them. See [Interpretability](production/interpretability.md).

LLMOps
:   The practices for shipping and operating LLM systems: versioning, eval gates in CI, staged rollout, observability, cost control, and incident response. See [LLMOps](production/llmops.md).

Logit lens
:   Applying the final norm and unembedding to intermediate hidden states, to see which tokens each layer is already predicting. See [Interpretability](production/interpretability.md).

Logits
:   Raw, unnormalized scores output by a model before softmax. See [Math](foundations/math.md#softmax).

LoRA (low-rank adaptation)
:   Fine-tuning by freezing the base weights and training small low-rank matrices added to them. See [LoRA & QLoRA](finetuning/lora-qlora.md).

Loss function
:   The scalar a model is trained to minimize, such as cross-entropy.

## M

Mamba
:   A selective state space model: a recurrent layer whose dynamics depend on the input, with linear-time training and constant-memory inference per token. See [Architecture Families](llm/architecture-families.md).

Markov decision process (MDP)
:   The formal RL setting: states, actions, transition probabilities, rewards, and a discount factor. See [RL Basics](decision/rl-basics.md).

Masked language model
:   A model trained to predict randomly masked tokens using context from both sides, as in BERT. Good for embeddings and classification, not for open-ended generation. See [Architecture Families](llm/architecture-families.md).

MCP
:   Model Context Protocol: an open protocol through which servers expose tools, resources and prompts that any compatible client or agent can use. See [Agents & Tool Use](applications/agents.md).

MFU (model FLOPs utilization)
:   Achieved training FLOP/s divided by the hardware's peak. A measure of training efficiency. See [Setup](start/setup.md#compute-and-time).

Mixed precision
:   Training with 16-bit computation while keeping master weights and some reductions in fp32. See [Python & PyTorch](foundations/pytorch.md#dtypes).

Mixture of experts (MoE)
:   An architecture where each token is routed to a few of many feed-forward "expert" networks, increasing parameters without proportional compute. See [Modern Architectures](llm/modern-architectures.md).

MLP (multi-layer perceptron)
:   A stack of linear layers with nonlinearities between them. Each transformer block contains one (the FFN). See [Neural Networks](foundations/neural-networks.md#from-perceptron-to-mlp).

Model merging
:   Combining the weights of several fine-tuned models into one without further training, using methods such as linear averaging, SLERP, task arithmetic, TIES or DARE. See [Continued Pretraining & Model Merging](finetuning/merging-continued-pretraining.md).

Momentum
:   An optimizer technique that accumulates an exponential moving average of gradients to smooth and accelerate updates.

Multi-head attention
:   Running several attention operations in parallel on different learned projections, then concatenating. See [Embeddings & Attention](llm/attention.md).

## N

Next-token prediction
:   The pretraining objective of language models: predict token $t+1$ from tokens $1..t$. See [Pretraining](pretraining/index.md).

Normalization
:   Layers (BatchNorm, LayerNorm, RMSNorm) that re-standardize activations to stabilize training.

## O

Online softmax
:   Computing softmax in one streaming pass by keeping a running maximum and a rescaled running sum. This is the trick that lets FlashAttention process attention in tiles. See [GPU Performance Engineering](pretraining/gpu-performance.md).

Optimizer state
:   Per-parameter values an optimizer stores between steps, such as Adam's moments. Often larger than the weights themselves. See [Setup](start/setup.md#training-memory).

Overfitting
:   Fitting training data so closely that performance on new data gets worse.

## P

Parameter
:   A learned weight or bias in a model.

PEFT (parameter-efficient fine-tuning)
:   Methods such as LoRA that train only a small fraction of parameters. See [Fine-tuning](finetuning/index.md).

Perplexity
:   The exponential of average per-token cross-entropy. Lower is better; only comparable with the same tokenizer. See [Math](foundations/math.md#perplexity).

Policy
:   In RL, the mapping from states to actions (or action probabilities). See [RL Basics](decision/rl-basics.md).

Positional encoding
:   Information about token position added to a transformer, since attention alone is order-agnostic. See [The Transformer](llm/transformer.md).

PPO (proximal policy optimization)
:   A policy-gradient RL algorithm that limits each update with a clipped objective. Used for game agents and for RLHF. See [DQN & PPO](decision/policy-methods.md).

Pre-norm
:   Applying normalization to the input of each residual branch rather than after it; standard in modern transformers.

Pretraining
:   Training a model from random initialization on a large unlabeled corpus with next-token prediction. See [Pretraining](pretraining/index.md).

Prompt injection
:   Instructions planted in input the model reads, such as a user message, a retrieved document or a tool output, that try to override the developer's instructions. Indirect injection through RAG and tools is the hardest kind to defend. See [Safety, Red-Teaming & Guardrails](production/safety.md).

## Q

Q-value
:   The expected return from taking action $a$ in state $s$ and following the policy afterward. See [RL Basics](decision/rl-basics.md).

QLoRA
:   LoRA on top of a base model quantized to 4-bit, allowing fine-tuning of large models on a single GPU. See [LoRA & QLoRA](finetuning/lora-qlora.md).

Quantization
:   Storing weights (and sometimes activations) in fewer bits, e.g. int8 or 4-bit, to reduce memory and speed up inference. See [Quantization & Serving](inference/serving.md#quantization).

## R

RAG (retrieval-augmented generation)
:   Retrieving relevant documents and adding them to the prompt so the model answers from them. See [RAG](rag/index.md).

ReAct
:   A prompting and agent pattern that interleaves reasoning steps with actions (tool calls) and their observations. See [Agents & Tool Use](applications/agents.md).

Recall@k
:   The fraction of relevant documents that appear in the top $k$ retrieved results. See [Advanced RAG & Evaluation](rag/advanced.md).

Red-teaming
:   Deliberately attacking your own system, manually and automatically, to find failures before users do. Results are often reported as attack success rate. See [Safety, Red-Teaming & Guardrails](production/safety.md).

Reinforcement learning (RL)
:   Learning to act by trial and error to maximize cumulative reward. See [RL Basics](decision/rl-basics.md).

ReLU
:   Rectified linear unit, $\max(0, x)$. See [Neural Networks](foundations/neural-networks.md#activation-functions).

Reranker
:   A model, often a cross-encoder, that re-scores a small set of retrieved candidates for higher precision. See [Advanced RAG](rag/advanced.md).

Residual connection
:   Adding a layer's input to its output, $x + f(x)$, so gradients flow easily through deep networks. See [Neural Networks](foundations/neural-networks.md#normalization-and-residual-connections).

Return
:   The (discounted) sum of rewards from a time step onward. See [RL Basics](decision/rl-basics.md).

Reward model
:   A model trained to score responses by human preference, used in RLHF. See [Preference Optimization](post-training/preference.md).

RLHF
:   Reinforcement learning from human feedback: train a reward model on preferences, then optimize the policy against it, typically with PPO and a KL penalty. See [Preference Optimization](post-training/preference.md).

RLVR
:   Reinforcement learning with verifiable rewards: RL where the reward comes from an automatic check (a math answer, unit tests) rather than a learned reward model. See [Reasoning & RL](post-training/reasoning-rl.md).

RMSNorm
:   Normalization by the root mean square of features, without mean subtraction. Used by Llama, Qwen, Gemma. See [Neural Networks](foundations/neural-networks.md#normalization-and-residual-connections).

Roofline model
:   A plot of attainable FLOP/s against arithmetic intensity, bounded by memory bandwidth on the left and peak compute on the right. It tells you which limit a kernel hits. See [GPU Performance Engineering](pretraining/gpu-performance.md).

RoPE (rotary position embedding)
:   Encoding position by rotating query and key vectors by position-dependent angles. See [Modern Architectures](llm/modern-architectures.md).

Routing model
:   A small, fast model that decides which tool, model, or queue should handle an input. See [Routing & Classification](decision/routing-models.md).

RWKV
:   A recurrent architecture with transformer-like parallel training and RNN-like constant-memory inference, in the linear-attention family. See [Architecture Families](llm/architecture-families.md).

## S

Sampling
:   Choosing the next token from the model's distribution, controlled by temperature, top-k, and top-p. See [Decoding: Sampling, KV Cache & Batching](inference/inference.md).

Scaling laws
:   Empirical power-law relationships between loss and model size, data, and compute. See [Scaling Laws & Compute](llm/scaling-laws.md).

Self-attention
:   Attention where queries, keys, and values all come from the same sequence. See [Embeddings & Attention](llm/attention.md).

SFT (supervised fine-tuning)
:   Fine-tuning on demonstration input-output pairs, usually with loss only on the output tokens. See [SFT](post-training/sft.md).

SGD
:   Stochastic gradient descent: gradient descent using mini-batch gradient estimates. See [Neural Networks](foundations/neural-networks.md#optimizers).

SLERP
:   Spherical linear interpolation between two weight vectors, which preserves their norm better than plain averaging. A common way to merge two models. See [Continued Pretraining & Model Merging](finetuning/merging-continued-pretraining.md).

SLM (small language model)
:   A language model small enough to run cheaply or on-device, roughly from tens of millions to a few billion parameters. See [Modern Architectures & SLM Design](llm/modern-architectures.md).

Softmax
:   The function that turns logits into a probability distribution, $e^{z_i}/\sum_j e^{z_j}$. See [Math](foundations/math.md#softmax).

Sparse autoencoder
:   An autoencoder with a wide, sparse hidden layer trained on a model's activations, to break superposed features into more interpretable directions. See [Interpretability](production/interpretability.md).

Special tokens
:   Reserved tokens such as beginning-of-sequence, end-of-sequence, and chat role markers. See [Tokenization](llm/tokenization.md).

State space model
:   A sequence layer based on a linear recurrence over a hidden state, computable as a recurrence or a convolution. S4 and Mamba are examples. See [Architecture Families](llm/architecture-families.md).

Steering vector
:   A direction added to a model's hidden states at inference time to push its behavior, for example toward a style or concept, without changing weights. See [Interpretability](production/interpretability.md).

SwiGLU
:   A gated feed-forward variant, $(\text{SiLU}(xW_1) \odot xW_3)W_2$, used in most modern LLMs. See [Neural Networks](foundations/neural-networks.md#activation-functions).

Synthetic data
:   Training data generated by models, for example instructions, answers or textbook-style text, then filtered by verification or judges. Central to modern SLM training. See [Distillation & Synthetic Data](post-training/distillation-synthetic-data.md).

System 1 / System 2
:   Fast, reactive decision-making versus slow, deliberate reasoning; used here to contrast small reactive policies with reasoning models. See [System 1 & System 2 Models](decision/system-one.md).

## T

Task vector
:   The difference between fine-tuned and base weights. Adding, scaling or combining task vectors is the basis of task arithmetic merging. See [Continued Pretraining & Model Merging](finetuning/merging-continued-pretraining.md).

Temperature
:   A divisor applied to logits before softmax; lower is more deterministic, higher more random. See [Math](foundations/math.md#temperature).

Tensor
:   An n-dimensional array with a device and dtype; PyTorch's core data type. See [Python & PyTorch](foundations/pytorch.md#tensors).

Tensor parallelism
:   Splitting individual weight matrices across GPUs so each computes part of every layer. See [Distributed Training](pretraining/distributed.md).

Throughput
:   Work done per unit time, e.g. training tokens per second or generated tokens per second.

TIES
:   A merging method that trims small task-vector entries, elects a sign per parameter, and averages only the entries that agree with it. See [Continued Pretraining & Model Merging](finetuning/merging-continued-pretraining.md).

Token
:   The unit of text a model reads and writes: a word, word piece, or byte sequence from the tokenizer's vocabulary. See [Tokenization](llm/tokenization.md).

Tokenizer
:   The component that converts text to token IDs and back. See [Tokenization](llm/tokenization.md).

Top-k / top-p sampling
:   Restricting sampling to the $k$ most likely tokens, or to the smallest set whose probabilities sum to $p$ (nucleus sampling). See [Decoding: Sampling, KV Cache & Batching](inference/inference.md).

Transformer
:   The architecture of attention and MLP blocks with residual connections underlying modern language models. See [The Transformer](llm/transformer.md).

Triton
:   A Python-embedded language for writing GPU kernels at the block level. It is easier than CUDA and is used by torch.compile to generate fused kernels. See [GPU Performance Engineering](pretraining/gpu-performance.md).

## V

Validation set
:   Held-out data used to measure generalization and choose checkpoints or hyperparameters, never trained on.

Value function
:   In RL, the expected return from a state under a policy, $V(s)$. See [RL Basics](decision/rl-basics.md).

Vanishing gradients
:   Gradients shrinking toward zero as they propagate back through many layers, stalling learning in early layers.

Vector database
:   A store optimized for nearest-neighbor search over embeddings, such as Qdrant. See [RAG Fundamentals](rag/fundamentals.md).

Vision Transformer (ViT)
:   A transformer applied to an image split into fixed-size patches, each linearly embedded as a token. See [Multimodal Models](llm/multimodal.md).

VLM
:   Vision-language model: a language model that also takes images, usually through a vision encoder plus a projector that maps image features into the LLM's embedding space. See [Multimodal Models](llm/multimodal.md).

Vocabulary
:   The set of all tokens a tokenizer can produce; its size $V$ sets the output layer width.

## W

Warmup
:   Gradually increasing the learning rate from near zero over the first steps of training. See [Neural Networks](foundations/neural-networks.md#learning-rate-schedules).

Weight decay
:   Shrinking weights toward zero each step as a regularizer. See [Neural Networks](foundations/neural-networks.md#regularization).

Weight tying
:   Sharing the token embedding matrix with the output projection, saving parameters in small models. See [Modern Architectures](llm/modern-architectures.md).

## Z

ZeRO
:   A family of techniques that shard optimizer state, gradients, and parameters across data-parallel GPUs to save memory. See [Distributed Training](pretraining/distributed.md).

Zero-shot
:   Performing a task from instructions alone, with no examples in the prompt.
