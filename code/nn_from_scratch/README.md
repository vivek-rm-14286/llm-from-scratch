# Project 1: Neural Network From Scratch

A scalar autograd engine, a numpy tensor engine built on the same idea, layers, optimizers, a gradient checker and a training loop that classifies handwritten digits. Then the same model in PyTorch, with matching loss curves, as proof that the engine computes what a real framework computes.

Full walkthrough: https://vivek-rm-14286.github.io/llm-from-scratch/#/projects/nn-from-scratch

## Layout

| Path | What it is |
|---|---|
| `starter/` | Signatures, docstrings and `TODO`s. This is where the work happens. |
| `solution/` | The complete reference implementation. |
| `tests/` | pytest tests that grade either package. |
| `conftest.py` | Adds `--impl` to choose the package under test. |

Both packages have the same files:

| File | Step | Contents |
|---|---|---|
| `engine.py` | 1 | `Value`: scalar autograd (+, *, pow, exp, log, tanh, relu, topological-sort `backward`) |
| `scalar_demo.py` | 1 | an MLP made of `Value`s on the digits (complete in both packages; slow on purpose) |
| `tensor.py` | 2 | `Tensor`: the same engine on numpy arrays, `unbroadcast`, `@`, `sum`, fused `cross_entropy` |
| `nn.py` | 3 | `Linear` (He and Xavier init), `ReLU`, `Tanh`, `Sequential`, `make_mlp`, `l2_penalty`, `accuracy` |
| `optim.py` | 4 | `SGD` (momentum, weight decay), `Adam` (L2 or decoupled weight decay) |
| `gradcheck.py` | 5 | finite-difference gradient check |
| `data.py` | 6 | digits and MNIST loaders (complete), `batches` |
| `train.py` | 6 | `train_epoch`, `evaluate`, `fit` and the command line |
| `compare_torch.py` | 7 | `to_torch` and a step-by-step loss comparison with PyTorch |

## Setup

```bash
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install -r requirements.txt        # torch is optional: delete that line to skip it
```

No GPU. Everything runs in seconds on a laptop CPU.

## Tests

```bash
pytest                          # grades starter/: 82 failures until the TODOs are filled in
pytest --impl solution          # grades solution/: 82 passed
NN_IMPL=solution pytest         # the same, through an environment variable
pytest tests/test_engine.py     # one step at a time
pytest -x                       # stop at the first failure
```

Without torch installed, the 6 tests in `tests/test_torch.py` are skipped as one module.

## Training

Replace `solution` with `starter` once your tests pass.

```bash
python -m solution.scalar_demo                      # Step 1: the scalar engine on digits (slow)
python -m solution.gradcheck                        # Step 5: max relative error, about 2e-9
python -m solution.train                            # 8x8 digits, no download, under a second
python -m solution.train --optimizer sgd --weight-decay 1e-3
python -m solution.train --dataset mnist --hidden 256 --epochs 10 --batch-size 64   # downloads about 12 MB
python -m solution.compare_torch                    # Step 7: must print MATCH
python -m solution.compare_torch --optimizer sgd
```

The default dataset is scikit-learn's bundled `load_digits` (1,797 images of 8×8 pixels), split 1,257 / 180 / 360 into train, validation and test. MNIST is only downloaded with `--dataset mnist`, into `data/mnist/`.
