# Setup, Hardware & Budget

Before you train anything, you need three things: a reproducible Python environment, a realistic idea of what your hardware can do, and a way to estimate what a run will cost before you pay for it. This page covers all three, plus the accounts (Hugging Face, an experiment tracker) that the projects assume you have.

!!! abstract "What you'll learn"
    - How to set up a Python project with `uv` and a PyTorch build that matches your accelerator (CUDA or Apple MPS)
    - The four hardware tiers this site targets and what each can realistically do
    - How to calculate GPU memory for training and for inference: weights, gradients, optimizer state, activations, KV cache
    - How to estimate the compute time and cost of a run before launching it
    - How to set up experiment tracking and the Hugging Face Hub

## The Python environment

### Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager. It replaces `pip`, `venv`, and `pyenv` for most purposes and produces a lockfile, so your environment is reproducible on a rented GPU box.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a project with a pinned Python version
uv init llm-lab --python 3.12
cd llm-lab
```

### Install PyTorch for your accelerator

PyTorch ships different wheels for different hardware. Get the exact command for your platform from the [PyTorch install selector](https://pytorch.org/get-started/locally/); the patterns look like this:

=== "Apple Silicon (MPS)"

    ```bash
    # Default macOS wheels include MPS support
    uv add torch numpy
    ```

=== "NVIDIA GPU (CUDA)"

    ```bash
    # Pick the CUDA version that your driver supports (check with nvidia-smi).
    # The index URL encodes the CUDA version, e.g. cu124, cu126, cu128.
    uv add torch --index https://download.pytorch.org/whl/cu126
    uv add numpy
    ```

=== "CPU only"

    ```bash
    uv add torch --index https://download.pytorch.org/whl/cpu
    uv add numpy
    ```

Then add the libraries the projects use:

```bash
uv add transformers datasets tokenizers accelerate peft trl \
       sentence-transformers qdrant-client gymnasium \
       wandb tensorboard matplotlib tqdm
# bitsandbytes (for QLoRA) is CUDA-focused; install it only on NVIDIA machines
uv add bitsandbytes   # NVIDIA only
```

!!! tip "Pin versions"
    APIs in `transformers`, `trl`, and `peft` shift between releases (TRL's `SFTConfig`/`SFTTrainer` arguments in particular). Commit `uv.lock` to git, and when a project page states a version, use it.

### Check your device

```python title="check_device.py"
import torch

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print("torch", torch.__version__, "device", device)

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    print("bf16 supported:", torch.cuda.is_bf16_supported())

# A quick matmul to confirm the device actually works
x = torch.randn(4096, 4096, device=device)
y = x @ x
print("ok", y.shape)
```

Run it with `uv run python check_device.py`.

!!! note "MPS caveats"
    Apple's MPS backend runs most PyTorch operations, but some are missing or fall back to CPU, `torch.compile` support is limited, and there is no `bitsandbytes`. MPS is excellent for learning and small models. For anything that needs CUDA-only kernels (FlashAttention, 4-bit QLoRA), rent an NVIDIA GPU. See the [MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html).

## Hardware tiers

The site is written for four tiers. Each project says which tier it needs.

| Tier | Example | Memory | Good for |
|------|---------|--------|----------|
| **1. Laptop** | Apple Silicon Mac (M-series), or any laptop CPU | 16-128 GB unified (Mac) | All Foundations and Language Models pages, tiny transformers (up to a few tens of millions of params), Project 4, Project 3 retrieval and small generators, inference of quantized small models |
| **2. Consumer GPU** | RTX 3090 / 4090 (24 GB), or a 5090 (32 GB) | 24-32 GB VRAM | Training models up to roughly 100-300M params, QLoRA fine-tuning of models up to roughly 7-14B params, Project 2 and Project 3 fully |
| **3. Single data-center GPU** | One A100 80 GB or H100 80 GB, rented | 80 GB HBM | Debugging and short runs of Project 1, full fine-tuning of small models, LoRA on larger ones |
| **4. 8-GPU node** | 8x H100 80 GB with NVLink, rented | 640 GB total | Project 1 full pretraining run in hours rather than days |

Apple Silicon is unusual because the GPU shares system memory. A 64 GB Mac can *load* models that do not fit on a 24 GB GPU, but its compute throughput and memory bandwidth are well below an H100, so training is much slower.

## GPU memory math

Out-of-memory errors are the most common failure when you scale up. You can predict them with arithmetic.

### Bytes per number

| dtype | Bytes | Used for |
|-------|-------|----------|
| fp32 | 4 | Master weights, optimizer state, small models |
| bf16 / fp16 | 2 | Mixed-precision training weights and activations, inference |
| int8 / fp8 | 1 | Quantized inference, some training on recent GPUs |
| 4-bit (NF4, GPTQ, AWQ) | 0.5 | Quantized inference, QLoRA base weights |

### Inference memory

For inference you need the weights plus the **KV cache** (the stored keys and values for every past token, so attention does not recompute them):

$$
\text{weights} = N_\text{params} \times \text{bytes per param}
$$

$$
\text{KV cache} = 2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \times \text{seq len} \times \text{batch} \times \text{bytes}
$$

The factor of 2 is for keys and values. Add roughly 10-20% overhead for the framework, temporary buffers, and fragmentation.

!!! example "A 7B model in bf16"
    Weights: $7 \times 10^9 \times 2$ bytes = 14 GB. The same model in 4-bit is about 3.5 GB plus some overhead for quantization scales. That is why a 4-bit 7B model runs on a laptop and the bf16 version needs a 24 GB GPU.

### Training memory

Training needs much more than inference. With AdamW and standard mixed precision, each parameter costs:

| Item | Bytes per param |
|------|-----------------|
| bf16 weights (used in forward/backward) | 2 |
| bf16 gradients | 2 |
| fp32 master copy of weights | 4 |
| AdamW first moment $m$ (fp32) | 4 |
| AdamW second moment $v$ (fp32) | 4 |
| **Total** | **about 16** |

Implementations differ (PyTorch AMP with `autocast` keeps fp32 weights and fp32 gradients, which is also about 16 bytes per parameter), but **16 bytes per parameter** is the number to remember. A 500M model therefore needs about 8 GB just for weights, gradients, and optimizer state, before any activations.

**Activations** are the intermediate values saved during the forward pass so the backward pass can compute gradients. They scale with batch size and sequence length, not with parameter count. For a transformer, a rough estimate per layer, in bf16 with a fused attention kernel (FlashAttention / SDPA) that does not store the full attention matrix, is:

$$
\text{activations per layer} \approx 34 \times \text{seq len} \times \text{batch} \times d_\text{model} \text{ bytes}
$$

Without a fused kernel, add a term that grows with the square of sequence length for the attention scores. Treat the constant as approximate: it depends on the exact architecture. Two tools reduce activation memory when you run out:

- **Gradient accumulation**: run several small micro-batches and sum their gradients before one optimizer step. Same effective batch, less memory.
- **Activation checkpointing**: store only some activations and recompute the rest during backward. Trades roughly 30% more compute for a large memory cut.

Here is a small estimator you can adapt:

```python title="mem_estimate.py"
def training_memory_gb(n_params, n_layers, d_model, seq_len, micro_batch,
                       bytes_per_param_state=16, act_const=34):
    states = n_params * bytes_per_param_state
    acts = n_layers * act_const * seq_len * micro_batch * d_model
    return states / 1e9, acts / 1e9

def inference_memory_gb(n_params, bytes_per_param, n_layers, n_kv_heads,
                        d_head, seq_len, batch, kv_bytes=2):
    weights = n_params * bytes_per_param
    kv = 2 * n_layers * n_kv_heads * d_head * seq_len * batch * kv_bytes
    return weights / 1e9, kv / 1e9

# A ~500M model: 24 layers, d_model 1280 (illustrative config)
s, a = training_memory_gb(500e6, n_layers=24, d_model=1280,
                          seq_len=2048, micro_batch=8)
print(f"train: states {s:.1f} GB + activations {a:.1f} GB")

w, kv = inference_memory_gb(500e6, 2, n_layers=24, n_kv_heads=4,
                            d_head=64, seq_len=4096, batch=16)
print(f"infer: weights {w:.1f} GB + KV cache {kv:.1f} GB")
```

Real usage will be somewhat higher than the estimate. Measure with `torch.cuda.max_memory_allocated()` after a few steps, and use the estimate only to pick a starting micro-batch size.

### Fine-tuning memory (LoRA and QLoRA)

With LoRA, the base model is frozen, so it needs no gradients or optimizer state: only the small adapter matrices do. With QLoRA, the frozen base is also stored in 4-bit. A 7-8B base in 4-bit is roughly 4-6 GB (the embeddings and output head usually stay in 16-bit), the adapters and their optimizer state are small, and activations dominate the rest. That is why QLoRA on a 7-8B model fits on a 24 GB card. Details are on the [LoRA & QLoRA](../finetuning/lora-qlora.md) page.

## Compute and time

Memory tells you whether a run fits. Compute tells you how long it takes. For a dense transformer, training costs roughly:

$$
\text{FLOPs} \approx 6 \times N \times D
$$

where $N$ is the parameter count and $D$ is the number of training tokens (2 FLOPs per parameter per token for the forward pass, 4 for the backward). The derivation is on [Scaling Laws & Compute](../llm/scaling-laws.md).

To convert to time, divide by what your hardware *actually* achieves, not its peak. The ratio of achieved to peak throughput is **MFU** (model FLOPs utilization). A well-tuned single-node run often lands somewhere around 30-50% MFU; a first attempt is often lower.

$$
\text{time} \approx \frac{6 N D}{\text{num GPUs} \times \text{peak FLOP/s} \times \text{MFU}}
$$

!!! example "Project 1 back-of-envelope"
    $N = 5 \times 10^8$, $D = 10^{10}$ tokens (about 20 tokens per parameter, the Chinchilla rule of thumb). FLOPs $\approx 6 \times 5\times10^8 \times 10^{10} = 3 \times 10^{19}$.

    An H100's dense bf16 peak is on the order of $10^{15}$ FLOP/s. At 40% MFU that is $4 \times 10^{14}$ FLOP/s per GPU, so the run takes about $7.5 \times 10^4$ seconds, around **21 GPU-hours**. On an 8-GPU node, about 2.5-3 hours of wall-clock time. Training on more tokens (which small models usually benefit from) scales the time linearly.

## Renting GPUs

You do not need to own data-center GPUs. Common options:

- **GPU clouds and marketplaces**: Lambda, RunPod, Vast.ai, CoreWeave, Together, Modal, and others. Usually the cheapest per hour, with simple per-second or per-hour billing.
- **Hyperscalers**: AWS, Google Cloud, Azure. More expensive on-demand and often need quota requests for large GPUs, but useful if you already have credits or need their surrounding services.
- **Notebooks**: Google Colab and Kaggle offer limited free or cheap GPU time, fine for Foundations and small experiments.

### Rough price ranges

!!! warning "Prices change constantly"
    The ranges below are rough, approximate on-demand figures for late 2026 and vary widely by provider, region, commitment, and availability. Spot/interruptible instances can be much cheaper. Always check current pricing before you plan a budget.

| GPU | Approximate on-demand price per GPU-hour |
|-----|------------------------------------------|
| RTX 4090 (24 GB) | roughly USD 0.30-0.80 |
| A100 80 GB | roughly USD 1-2.50 |
| H100 80 GB | roughly USD 2-4 on GPU clouds, often higher on hyperscalers |
| 8x H100 node | roughly 8x the single-GPU price, so very roughly USD 16-35 per hour |

### Budgeting a run

Budget in three parts:

1. **Debugging**, on the cheapest hardware that can run a small config. This is where most of the bugs are found and it should be cheap. Do it on your laptop or a single consumer GPU.
2. **The main run**: GPU-hours from the formula above, times the hourly price, times a safety factor of about 1.5 for restarts, slower-than-expected throughput, and evaluation.
3. **Storage and egress**: datasets and checkpoints. A 500M checkpoint with optimizer state is several GB; keep only the last few plus milestone checkpoints.

Using the example above: about 21 GPU-hours at roughly USD 2-4 per hour is roughly USD 40-85, times 1.5 gives a budget of roughly USD 60-130 for a Chinchilla-sized Project 1 run. Training on several times more tokens scales that proportionally.

!!! danger "Stop the meter"
    Idle GPUs bill the same as busy ones. Before launching: set a billing alert, write checkpoints to persistent storage, and schedule a shutdown (for example `sudo shutdown -h +600` for 10 hours). Many people's most expensive mistake is a forgotten instance.

## Experiment tracking

You will run many experiments, and you need to compare loss curves across them. Two common choices:

=== "Weights & Biases"

    Hosted, with a free tier for personal use. Logs metrics, configs, system stats (GPU utilization, memory), and lets you compare runs in a browser. See the [W&B docs](https://docs.wandb.ai/).

    ```python
    import wandb
    run = wandb.init(project="llm-from-scratch", config={"lr": 3e-4, "batch": 64})
    for step in range(100):
        loss = 1.0 / (step + 1)
        wandb.log({"train/loss": loss}, step=step)
    run.finish()
    ```

    Run `uv run wandb login` once and paste your API key.

=== "TensorBoard"

    Local and offline. Good if you do not want an account or are on a machine without internet access.

    ```python
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter("runs/exp1")
    for step in range(100):
        writer.add_scalar("train/loss", 1.0 / (step + 1), step)
    writer.close()
    ```

    View with `uv run tensorboard --logdir runs`.

Whichever you use, always log: training loss, validation loss, learning rate, gradient norm, tokens/second, and the full config. Gradient norm and learning rate are what you will need when diagnosing a loss spike.

## Hugging Face account and Hub

The [Hugging Face Hub](https://huggingface.co/docs/hub/index) hosts open-weight models, datasets, and tokenizers. You will download base models for Projects 2 and 3 and can upload your own checkpoints and adapters.

1. Create an account at huggingface.co.
2. Create an access token (Settings, Access Tokens). A read token is enough for downloading; use a write token to push models.
3. Log in from the command line (see the [hub CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)):

    ```bash
    uv run hf auth login        # older releases: huggingface-cli login
    ```

4. Some models (for example Llama and Gemma families) are **gated**: you must accept the license on the model page before downloads work.

```python
from huggingface_hub import snapshot_download
path = snapshot_download("HuggingFaceTB/SmolLM2-135M")
print(path)
```

!!! tip "Cache location"
    Models download to `~/.cache/huggingface` by default. On rented machines, set `HF_HOME` to a large persistent volume so you do not re-download on every restart and do not fill the root disk.

!!! warning "Common pitfalls"
    - **CUDA wheel does not match the driver.** If `torch.cuda.is_available()` is `False` on a GPU box, check `nvidia-smi` and reinstall from the matching index URL.
    - **Forgetting the optimizer in memory estimates.** A model that fits for inference may need around 8x more memory to train with AdamW.
    - **Using peak FLOPs for time estimates.** Use 30-50% MFU at best for planning, lower for a first attempt.
    - **Checkpoints only on ephemeral disk.** When a spot instance is reclaimed, its local disk goes with it.
    - **Tokens in git.** Never commit your Hugging Face or W&B tokens. Use environment variables or the CLI login.

## Check your understanding

??? question "How much memory do weights, gradients, and AdamW state need for a 1B-parameter model in mixed precision?"
    About 16 bytes per parameter, so about 16 GB, before activations.

??? question "Your run fits with micro-batch 4 but not 8. How do you train with an effective batch of 32 sequences?"
    Keep micro-batch 4 and use 8 gradient accumulation steps. Optionally add activation checkpointing to fit a larger micro-batch, which usually improves throughput.

??? question "Estimate the KV cache for 32 layers, 8 KV heads, head dim 128, 8192 tokens, batch 1, in bf16."
    $2 \times 32 \times 8 \times 128 \times 8192 \times 1 \times 2$ bytes, which is about 1.07 GB.

??? question "A run's estimate is 100 GPU-hours at roughly USD 3 per hour. What budget do you set?"
    Roughly USD 300 for the run, times about 1.5 for restarts and slack, so roughly USD 450, plus storage. Then add a billing alert and an automatic shutdown.

## Further reading

- [uv documentation](https://docs.astral.sh/uv/)
- [PyTorch: Get Started](https://pytorch.org/get-started/locally/)
- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (where the 16-bytes-per-parameter accounting comes from)
- Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla)
