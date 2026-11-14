"""Evaluate a policy (checkpoint, ONNX file, or the scripted expert) and optionally watch it play.

Usage:
    python evaluate.py --policy checkpoints/ppo.pt --episodes 100
    python evaluate.py --policy expert --episodes 100
    python evaluate.py --policy checkpoints/ppo.onnx --render          # needs pygame + a display
    python evaluate.py --policy checkpoints/bc.pt --stochastic         # sample instead of argmax
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from env import OrbitalCourierEnv
from expert import expert_action
from train_bc import load_policy


def make_policy_fn(spec: str, stochastic: bool = False, seed: int = 0):
    if spec == "expert":
        return expert_action
    if spec.endswith(".onnx"):
        import onnxruntime as ort
        sess = ort.InferenceSession(spec, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return lambda obs: int(np.argmax(sess.run(None, {name: obs[None, :]})[0][0]))
    policy = load_policy(spec).eval()
    gen = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def act(obs):
        logits = policy(torch.as_tensor(obs).unsqueeze(0))[0]
        if stochastic:
            return int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).item())
        return int(logits.argmax().item())
    return act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="path to .pt / .onnx, or 'expert'")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=100_000, help="held-out seeds, disjoint from training")
    p.add_argument("--render", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--n-hazards", type=int, default=6)
    args = p.parse_args()

    env = OrbitalCourierEnv(render_mode="human" if args.render else None, n_hazards=args.n_hazards)
    act = make_policy_fn(args.policy, args.stochastic)
    rets, lens, pkgs, wins, crashes = [], [], [], 0, 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ret, steps = 0.0, 0
        while True:
            obs, r, term, trunc, info = env.step(act(obs))
            ret += r
            steps += 1
            if term or trunc:
                break
        rets.append(ret)
        lens.append(steps)
        pkgs.append(info["collected"])
        wins += int(info["success"])
        crashes += int(info["event"] == "crash")
        if args.render:
            print(f"episode {ep}: return {ret:.1f}, packages {info['collected']}, event {info['event']}")
    env.close()
    n = args.episodes
    print(f"policy={args.policy}  episodes={n}")
    print(f"  mean return   {np.mean(rets):.2f} +- {np.std(rets) / np.sqrt(n):.2f} (s.e.)")
    print(f"  success rate  {wins / n:.1%}   crash rate {crashes / n:.1%}")
    print(f"  packages/ep   {np.mean(pkgs):.2f}   mean length {np.mean(lens):.0f} steps")


if __name__ == "__main__":
    main()
