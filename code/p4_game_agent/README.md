# Project 4: A Decision Model That Plays — Orbital Courier

A tiny original 2D game plus a complete decision-model pipeline:
scripted expert → (optional) human demos → behavior cloning (+ DAgger) → PPO fine-tuning → evaluation → ONNX export.

Full walkthrough: https://vivek-rm-14286.github.io/llm-from-scratch/projects/p4-decision-model/

## The game

A courier drone with momentum flies around a 10×10 arena. It has to collect 5 packages, one at a time, while avoiding 6 hazards that drift and bounce around.
It observes 16 hazard-distance rays, 16 wall-distance rays, its velocity, and the direction and distance to the package (37 floats).
It has 9 discrete actions: coast, or thrust in one of 8 compass directions.

| Event | Reward |
|---|---|
| each step | -0.01 |
| progress toward package | +0.5 × (distance decrease) |
| package collected | +10 |
| crash into hazard | -10 (episode ends) |
| 5th package | episode ends with success |
| 600 steps | truncated |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pygame/onnx are only needed for rendering/export
```

## Pipeline

```bash
python env.py                                            # API check + one random episode
python expert.py --episodes 200 --out demos/expert.npz   # scripted demonstrations
python play_record.py --out demos/human.npz --episodes 5 # optional: your own demos (arrow keys)
python train_bc.py --demos "demos/*.npz" --epochs 30 --dagger-iters 3 --out checkpoints/bc.pt
python train_ppo.py --init checkpoints/bc.pt --iters 150 --out checkpoints/ppo.pt
python train_ppo.py --iters 150 --out checkpoints/ppo_scratch.pt   # baseline: no BC init
python evaluate.py --policy expert --episodes 100
python evaluate.py --policy checkpoints/bc.pt --episodes 100
python evaluate.py --policy checkpoints/ppo.pt --episodes 100
python evaluate.py --policy checkpoints/ppo.pt --episodes 5 --render   # watch it play
python export_onnx.py --policy checkpoints/ppo.pt --out checkpoints/ppo.onnx
python evaluate.py --policy checkpoints/ppo.onnx --episodes 100
```

Quick smoke test (about a minute on a laptop CPU):

```bash
python expert.py --episodes 20 --out demos/smoke.npz
python train_bc.py --demos demos/smoke.npz --epochs 3 --dagger-iters 1 --dagger-episodes 3 --eval-episodes 5 --out checkpoints/smoke_bc.pt
python train_ppo.py --init checkpoints/smoke_bc.pt --iters 3 --n-envs 4 --n-steps 128 --critic-warmup 1 --eval-every 3 --out checkpoints/smoke_ppo.pt
python evaluate.py --policy checkpoints/smoke_ppo.pt --episodes 5
python export_onnx.py --policy checkpoints/smoke_ppo.pt --out checkpoints/smoke.onnx
```

## Files

| File | What it does |
|---|---|
| `env.py` | `OrbitalCourierEnv` (Gymnasium API, numpy physics, ray-cast sensors, optional pygame renderer) |
| `expert.py` | potential-field heuristic that reads only the observation; demo generator |
| `play_record.py` | keyboard play with pygame and demonstration recording |
| `train_bc.py` | `PolicyNet` MLP, behavior cloning, optional DAgger with the scripted expert |
| `train_ppo.py` | single-file PPO (GAE, clipped objective, critic warmup, KL early stop), BC init |
| `evaluate.py` | mean return, success/crash rate on held-out seeds; `--render` for live play |
| `export_onnx.py` | ONNX export, numerical check, and per-step latency (torch vs onnxruntime) |

## Notes

- Rendering on Python 3.14 with some pygame wheels: the font module may fail to import. The renderer then skips the HUD text. `pygame-ce` or Python 3.12 avoids this.
- Headless machines: `SDL_VIDEODRIVER=dummy python evaluate.py ... --render` runs the renderer without a window.
- All evaluation uses seeds ≥ 100000, which training never uses.
