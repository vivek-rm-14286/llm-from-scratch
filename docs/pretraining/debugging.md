# Debugging a Training Run

A training script with a bug rarely crashes. It prints a loss, the loss goes down, and the model is quietly useless: it learned to copy its input, or saw one batch 10,000 times, or peeked at the answer through a mask that was off by one. This page is an ordered checklist for a run that misbehaves, cheapest checks first, followed by a small script with five planted bugs to practice on. Every check runs in seconds on a tiny model, and that is the point: debug at the smallest scale that still shows the problem, not on the 8-GPU run.

!!! abstract "What you'll learn"
    - The number the loss must show at step 0, and what it means when it doesn't
    - The single-batch overfit test, and the two bugs that make a loss fall *too* well
    - How to read gradient norms, per-layer activation statistics and update-to-weight ratios
    - A quick learning-rate range check, and how to read the train/validation gap
    - How to find where a NaN first appears, and the bf16 mistakes that cause them
    - The data-loader bugs that no model-side check will catch

The loop itself (loss, AdamW, schedules, clipping, bf16) is built in [The Training Loop](training.md). This page assumes you have one and it is not behaving.

## The checklist

Work top to bottom. Each step assumes the ones above it passed, so do not skip ahead to the interesting ones.

| # | Check | Passes when | Typical cause of failure |
|---|---|---|---|
| 1 | Step-0 loss | About $\ln(\text{vocab size})$ | Bad init, wrong loss reduction, labels misaligned with logits |
| 2 | Overfit one batch | Loss falls to near 0 | Model or optimizer wiring: frozen parameters, missing `zero_grad`, wrong LR |
| 3 | Shift and leak | Targets are inputs shifted by one; future tokens do not change past logits | Unshifted targets, causal mask off by one |
| 4 | Gradients, activations, updates | Finite, similar scale across layers, update/weight around 1e-3 | Init scale, missing normalization, LR far off |
| 5 | LR range | Loss after ~100 steps is best in the middle of a sweep | LR or schedule wrong by 10x or more |
| 6 | Train/val gap | Validation tracks training, slightly above | Loader repeats data, split leakage, real overfitting |
| 7 | NaN hunt | You can name the first module that produces a non-finite value | Division by zero, `log(0)`, fully masked rows, loss computed in bf16 |
| 8 | Data loader | Decoded batches read as text; shards and workers are not duplicated | Dtype, shuffling, per-worker seeds, wrong tokenizer |

### 1. The step-0 loss should be ln(vocab)

A freshly initialized model knows nothing, so its best prediction is a uniform distribution over the vocabulary. The cross-entropy of a uniform guess is

$$
\mathcal{L}_0 = -\ln\frac{1}{V} = \ln V
$$

For a 50,257-token vocabulary that is 10.82; for Project 1's 49,152 tokens, 10.80; for a 90-character vocabulary, 4.50. Print the loss before the first optimizer step and compare.

- **Much higher** (say 18 instead of 4.5): the logits start large, so the model is confidently wrong. Look at the initialization of the output head and the embeddings (std 0.02, not PyTorch's default of 1.0 for `nn.Embedding`), or at a missing final normalization layer.
- **Much lower**: the model can see the answer before training. Go straight to check 3.
- **Off by a factor** such as the sequence length: the loss is summed where it should be averaged.

Starting high is not harmless. The first few hundred steps are spent shrinking logits, and the large early gradients are what [warmup](training.md#learning-rate-schedules) exists to survive.

### 2. Overfit a single batch

Fetch one batch, then train on it alone for a few hundred steps with no dropout or augmentation. A model with enough capacity must memorize it: the loss should fall to near zero (0.02 for the small model below, after 150 steps). If it cannot, the bug is in the model or the optimizer wiring, and no amount of data or tuning will help. Usual suspects: parameters that never reach the optimizer, `requires_grad=False` left on, a `detach()` in the forward pass, `optimizer.zero_grad()` missing so gradients accumulate, or a learning rate that is orders of magnitude off.

This test says nothing about generalization. It only proves that gradients flow from the loss to every parameter and that the optimizer applies them.

### 3. Shifted targets and causal-mask leaks

The opposite symptom is a loss that falls *too* well. On real text, a small model's next-token loss does not approach zero: there is irreducible uncertainty about what comes next. If training **and validation** loss both head toward 0, the model is being handed the answer. Two bugs do this:

- **Targets not shifted.** `y` must be `x` moved one position left (`y[t] = x[t+1]`). With `y = x` the task is "output the token you were just given", which any model learns in a few dozen steps.
- **Causal mask leak.** If position $t$ can attend to position $t+1$, it can read its own target. An off-by-one in `tril`/`triu`, a mask that is built but never applied, or a padding mask that overwrites the causal mask all do it.

Both have direct tests that do not depend on reading a loss curve:

```python title="leak_checks.py"
import torch

def check_shift(x, y):
    assert (x[:, 1:] == y[:, :-1]).all(), "targets are not the inputs shifted by one"

@torch.no_grad()
def check_causal(model, x, vocab_size):
    """Change the LAST input token. Logits at every earlier position must be identical."""
    model.eval()
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % vocab_size
    diff = (model(x)[0][:, :-1] - model(x2)[0][:, :-1]).abs().max().item()   # model returns (logits, loss)
    model.train()
    assert diff == 0.0, f"future token changed past logits by {diff:.2e}: causal mask leaks"
```

Run both once at startup. They cost one forward pass, and the causal check also catches leaks through anything else that mixes positions, such as a normalization layer computed over the time dimension.

### 4. Gradient norms, activations and update ratios

When the basic wiring is right and training is still slow or unstable, look inside. Three numbers per layer are enough:

- **Gradient norm.** Log the total norm that [gradient clipping](training.md#gradient-clipping) returns (it is the norm *before* clipping). A healthy run shows a noisy but stable value; a steady climb, or spikes of 10-100x, come before divergence. A norm of exactly 0 for some parameter means it is disconnected from the loss.
- **Activation statistics.** The mean, standard deviation and maximum of each layer's output for one batch. Standard deviations that grow or shrink by a constant factor per layer mean the initialization or normalization is wrong; a huge maximum next to an ordinary standard deviation is an outlier that will overflow first.
- **Update-to-weight ratio.** $\lVert \Delta W \rVert / \lVert W \rVert$ for one optimizer step. Around $10^{-3}$ is a good sign. Near $10^{-5}$ the layer is barely training (LR too low, or weight far too large); near $10^{-1}$ every step rewrites the layer (LR too high).

```python title="diagnostics.py"
import torch
import torch.nn as nn

def activation_stats(model, x, types=(nn.Linear, nn.LayerNorm)):
    """One forward pass; returns {module name: (mean, std, max |value|)} of each module's output."""
    stats, hooks = {}, []
    for name, m in model.named_modules():
        if isinstance(m, types):
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, name=name: stats.__setitem__(
                    name, (out.mean().item(), out.std().item(), out.abs().max().item()))))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return stats

def update_ratios(model, optimizer):
    """Call INSTEAD of optimizer.step(). Returns {param name: ||update|| / ||weight||} for 2-D weights."""
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.ndim == 2}
    optimizer.step()
    return {n: ((p.detach() - before[n]).norm() / (before[n].norm() + 1e-12)).item()
            for n, p in model.named_parameters() if n in before}
```

Call these every few hundred steps, not every step: the copies in `update_ratios` double the memory of the weights for a moment. On the working script below the largest update ratio starts at 7.5e-3 during warmup, peaks at 4.4e-2 at the top of the schedule and decays to 3e-3; a tiny model at a high learning rate sits above the $10^{-3}$ rule of thumb, and a 500M model sits close to it.

### 5. Learning-rate range sanity

Before trusting a learning rate, run a short sweep: the same 100 steps at learning rates spaced by about 3x across three or four decades, and compare the loss at the end. For the small model below:

| Peak LR | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 | 3e-1 |
|---|---|---|---|---|---|---|---|---|
| Train loss at step 100 | 3.75 | 3.18 | 2.35 | 2.06 | 2.02 | 2.48 | 2.57 | 2.86 |

The curve has a basin. Too low and the loss has barely moved from 4.5; too high and it stalls or blows up. Pick a value in the basin's lower half (here 3e-3), because the best LR for 100 steps is usually a little too aggressive for a full run. If your chosen LR sits at an edge of the basin, or the sweep has no basin at all, something upstream is wrong. Also **print the learning rate with every log line**: schedule bugs (warmup that never ends, decay in the wrong units, a scheduler stepped twice per iteration) are invisible in the loss until it is too late and obvious in the LR column. Typical values and schedules are in [AdamW for language models](training.md#adamw-for-language-models).

### 6. The train/validation gap

Evaluate on held-out data every few hundred steps and read the two curves together:

| Pattern | Meaning |
|---|---|
| Val slightly above train, both falling | Healthy |
| Train falls, val flat or rising | The model memorizes: too little data, or the loader keeps serving the same examples |
| Val equal to or below train, by a suspicious margin | Validation data leaked into training, or check 3 failed |
| Both flat near $\ln V$ | Not learning: LR, frozen parameters, or broken inputs |
| Both fall, then jump up together | Instability: see [Loss spikes and divergence](training.md#loss-spikes-and-divergence) |

In single-epoch pretraining on a large corpus the gap should be small, because the model never sees a token twice. A large gap there almost always means the loader is repeating itself (check 8), not that the model is too big. [What a healthy loss curve looks like](training.md#what-a-healthy-loss-curve-looks-like) shows the expected shape.

### 7. NaN hunting

A NaN loss is the end of a chain, and the loss is the last place it shows up. Find the first place.

1. **Stop at the first bad step.** Add `if not torch.isfinite(loss): raise ...` and save the offending batch and the checkpoint from just before it, so the failure is reproducible.
2. **Find the first non-finite activation** with forward hooks:

    ```python
    def first_nonfinite(model, x):
        """Name of the first leaf module whose output contains NaN or inf, or None."""
        found, hooks = [], []
        for name, m in model.named_modules():
            if not list(m.children()):
                hooks.append(m.register_forward_hook(
                    lambda mod, inp, out, name=name: found.append(name)
                    if not found and not torch.isfinite(out).all() else None))
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        return found[0] if found else None
    ```

3. **If the forward pass is clean, the NaN is born in backward.** Wrap one step in `torch.autograd.set_detect_anomaly(True)`. It is several times slower, so use it only on the failing step. It raises at the operation whose gradient went bad and prints the forward traceback that created it, for example `RuntimeError: Function 'LogBackward0' returned nan values in its 0th output.` for a `w * torch.log(w)` evaluated at `w = 0`.

The usual sources: `log(0)` or division by a norm that can be zero (add an epsilon); `sqrt` at 0, whose gradient is infinite; an attention row where **every** position is masked, because the softmax of all `-inf` is NaN; and a learning rate high enough that the weights overflow.

**bf16 pitfalls.** bfloat16 keeps float32's range and only about three significant decimal digits: `1 + 0.003` is exactly `1` in bf16, and `256 + 1` is `256`. Under [mixed precision](training.md#mixed-precision-with-bf16) that is fine for matmuls and wrong for anything that accumulates small numbers. Compute the loss in float32 (`logits.float()` before `cross_entropy`), keep running sums and normalization statistics in float32, and keep the master weights and optimizer state in float32, which autocast does for you as long as the model itself is not cast to bf16. An epsilon such as `1e-8` added to a value near 1 vanishes in bf16; add it where the value is float32. With fp16 (not bf16) the range is the problem as well: values above 65,504 overflow to inf, which is why fp16 needs a gradient scaler and bf16 does not. If a run is stable in float32 and NaNs under autocast, bisect by excluding one module at a time from autocast.

### 8. Data-loader bugs

The model-side checks pass on garbage data. Look at the data itself:

- **Decode a batch and read it.** `print(tokenizer.decode(x[0]))` should be readable text from your corpus, and `decode(y[0])` the same text one token later. This catches the wrong tokenizer, the wrong dtype for the shards (`uint16` read as `int32` gives noise), and byte-order mistakes.
- **Check that batches differ.** Hash the first few batches. Identical hashes mean a generator that is re-seeded on every call or an iterator that is re-created each step.
- **Check workers and ranks.** With `num_workers > 0` or several GPUs, each worker must get a different slice. A dataset that seeds itself the same way in every worker serves each example `num_workers` times, which looks like a smaller dataset and a growing train/val gap. See [Distributed Training](distributed.md) for per-rank sharding.
- **Check the split.** Validation shards must not be in the training glob, and near-duplicate documents on both sides flatter the validation loss ([deduplication](data.md)).
- **Check token id range.** `x.max() < vocab_size`. On a GPU an out-of-range id is a device-side assert with an unhelpful traceback; run one batch on CPU to get the real error.

## The exercise: five planted bugs

The repo has a small character-level GPT (2 layers, 64 dimensions, about 0.1M parameters) trained on the source code of a standard-library module, so it needs no download and runs on a laptop CPU in about 15 seconds: [`code/lessons/debugging`](https://github.com/vivek-rm-14286/llm-from-scratch/tree/main/code/lessons/debugging).

- `buggy_train.py` has five planted bugs. It runs without error.
- `fixed_train.py` is the reference, with checks 1 and 3 built in and the learning rate, gradient norm and update ratio on every log line. `--bug N` re-introduces one bug so you can see its symptom alone.
- `diagnostics.py` holds the helpers from checks 4 and 7.

```bash
cd code/lessons/debugging
python buggy_train.py
```

```text
step    0  train 18.021  val 20.345
step   50  train 1.592  val 2.035
step  100  train 1.625  val 2.000
step  150  train 2.048  val 2.311
step  200  train 5.160  val 5.680
step  250  train 14.898  val 20.333
step  300  train 21.461  val 23.975
```

The outputs on this page are from Python 3.12 and PyTorch 2.x on a laptop CPU. Other Python versions ship a slightly different `argparse` source (89 characters on 3.14), so the loss values shift a little; the symptoms do not.

The vocabulary has 90 characters, so the first line is already wrong: 18.0 where $\ln 90 = 4.50$ is expected. The working script prints this:

```text
check 1  step-0 loss 4.547   expected ln(90) = 4.500
check 3  targets shifted: True   causal leak: 0.00e+00 (must be 0)
step    0  lr 1.5e-04  train 4.544  val 4.477  grad-norm  3.72  max update/weight 7.5e-03
step   50  lr 2.9e-03  train 2.144  val 2.317  grad-norm  0.89  max update/weight 4.4e-02
step  100  lr 2.5e-03  train 1.977  val 2.050  grad-norm  0.82  max update/weight 2.8e-02
step  150  lr 1.8e-03  train 1.644  val 1.790  grad-norm  0.88  max update/weight 1.8e-02
step  200  lr 1.1e-03  train 1.528  val 1.659  grad-norm  1.28  max update/weight 9.9e-03
step  250  lr 5.1e-04  train 1.510  val 1.556  grad-norm  1.34  max update/weight 4.2e-03
step  300  lr 3.0e-04  train 1.332  val 1.511  grad-norm  1.18  max update/weight 3.1e-03
```

## Build it

**1. Fix `buggy_train.py`.** Start from the buggy script and follow the checklist. Fix one bug at a time and re-run after each fix; several bugs hide behind others, so the symptom changes as you go. You are done when the step-0 loss is within 0.1 of 4.50, the checks from `leak_checks.py` pass, and the validation loss ends near 1.5 (1.45 to 1.55 depending on how you write the schedule). Do not diff against `fixed_train.py` until you have found all five.

**2. Run the single-batch test.** `python fixed_train.py --overfit` trains on one fixed batch. Expected result: training loss below 0.1 by step 100 and about 0.02 by step 300.

**3. See each symptom alone.** Run `python fixed_train.py --bug N` for N from 1 to 5 and match each output to a row of the checklist table before reading the solution.

??? success "Solution sketch"
    The five bugs, in the order the checklist finds them:

    | # | Bug | Found by | Symptom alone (`--bug N`) | Fix |
    |---|---|---|---|---|
    | 1 | `nn.init.normal_(self.head.weight)` re-initializes the output head with the default std of 1.0 | Check 1 | Step-0 loss 18.3 instead of 4.5; first gradient norm 218 | Delete the line (the `_init` function already uses std 0.02) |
    | 5 | `lr_at` returns `BASE_LR * (step + 1)`: no division by the warmup length, no cap | Printing the LR (check 5) | LR reaches 0.9; loss stalls near 2.7, then climbs past 40 | `BASE_LR * min(1.0, (step + 1) / WARMUP)`, plus a decay if you want one |
    | 2 | `y` is built from the same slice as `x` | `check_shift` (check 3) | "targets shifted: False"; train and val loss both 0.005 | `y = d[i + 1:i + T + 1]` |
    | 3 | `torch.tril(..., diagonal=1)` lets each position attend one step ahead | `check_causal` (check 3) | Causal leak 1.3e-2; train and val both fall to 0.05 | `diagonal=0` |
    | 4 | `gen.manual_seed(0)` inside `get_batch` | Train/val gap (check 6), or hashing batches (check 8) | Train loss 0.02 while val rises from 4.5 to 5.7 | Seed the generator once, outside the function |

    With all five present, the bad initialization and the runaway learning rate dominate what you see, which is why the output above explodes. Fix those two and the run looks excellent, with loss near zero on both splits, because bugs 2 and 3 hand the model its targets. Fix bug 2 and the training loss still reaches 0.003, through the mask leak and the repeated batch. Fix bug 3 and it still reaches 0.02 while validation climbs to 5.7, because every training step sees the same 32 sequences. Only then does the fifth fix give a normal curve. (Bug 4 also makes `estimate` average ten copies of one validation batch, so the validation number was noisier than it looked all along.)

    Note which bugs a loss curve alone could have told you about: one of five. The step-0 number, the two assertions, the LR column and the train/val comparison found the rest.

!!! warning "Common pitfalls"
    - **Debugging at full scale.** Every check here works on a model that trains in seconds. Reproduce the problem small first.
    - **Changing several things between runs.** Fix one thing, re-run, compare. Keep the seed fixed while debugging.
    - **Treating a falling loss as proof of correctness.** Leaks and repeated batches make the loss fall faster than a correct run.
    - **Logging only the loss.** Without the learning rate, gradient norm and validation loss on the same line, most of this checklist cannot be applied after the fact.
    - **Leaving `set_detect_anomaly` on.** It slows training several-fold. Use it for one failing step.
    - **Raising clipping or lowering the LR to hide a NaN.** That postpones it. Find the operation that produced it.

## Check your understanding

??? question "A model with a 32,000-token vocabulary reports a step-0 loss of 10.37. Is that right? What if it reported 3.2?"
    $\ln 32000 = 10.37$, so yes: the model starts at a uniform guess. A loss of 3.2 before any training means the model already has information about the targets: unshifted targets, a mask leak, or (if weights were loaded) it is not a fresh model.

??? question "Training and validation loss on English text both reach 0.01 after 500 steps of a 2-layer model. What do you check?"
    Check 3. Real text has irreducible next-token uncertainty, so near-zero loss on held-out data means the model sees its targets. Assert that `y` is `x` shifted by one, then run the causal test: change the last input token and confirm earlier logits do not move.

??? question "Why does the single-batch overfit test not tell you the data pipeline is correct?"
    It uses one batch, so it only exercises the model, the loss and the optimizer. A loader that repeats batches, mixes splits or decodes shards with the wrong dtype passes it. Those need the checks that look at the data: decoding a batch, hashing batches, and the train/val gap.

??? question "The update-to-weight ratio of your embedding matrix is 1e-6 while every other layer is near 1e-3. What might be wrong?"
    The embedding is barely training. Either its gradient is tiny (it is disconnected, frozen, or scaled down by a missing or misplaced normalization) or its weights are far larger than the others, for example initialized with std 1.0 instead of 0.02, so the same absolute update is relatively small.

??? question "A run is stable in float32 and produces NaN within 100 steps under bf16 autocast. Name two likely causes."
    A loss or normalization computed in bf16, where small terms and epsilons vanish next to values near 1 (cast to float32 first); and an attention row that is fully masked, which is NaN in any precision but may only occur with the batch shapes used in the bf16 run. Bisect by excluding modules from autocast one at a time.

## Further reading

- Andrej Karpathy, *A Recipe for Training Neural Networks* - [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/)
- Smith, *Cyclical Learning Rates for Training Neural Networks* (the LR range test) - [arXiv:1506.01186](https://arxiv.org/abs/1506.01186)
- PyTorch docs: [Automatic mixed precision](https://pytorch.org/docs/stable/amp.html) and [anomaly detection](https://pytorch.org/docs/stable/autograd.html#anomaly-detection)
- Google Research, *Deep Learning Tuning Playbook* - [github.com/google-research/tuning_playbook](https://github.com/google-research/tuning_playbook)
