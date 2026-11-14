"""A scripted heuristic expert for Orbital Courier, plus a demo generator.

The expert reads ONLY the observation vector (no privileged simulator state),
so a neural policy that sees the same observation can, in principle, copy it
exactly. That makes behavior cloning well-posed.

Idea: a potential-field controller.
  desired direction = attraction to the package
                      - repulsion from every ray that sees a hazard
                      - repulsion from walls that are very close
  desired velocity  = desired direction * a speed that slows near the package
  action            = the thrust direction that best reduces (desired_v - v)

Usage:
    python expert.py --episodes 200 --out demos/expert.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from env import ACTION_DIRS, N_RAYS, RAY_DIRS, V_MAX, ARENA, OrbitalCourierEnv


def expert_action(obs: np.ndarray, hazard_gain: float = 2.5, wall_gain: float = 1.5) -> int:
    haz = obs[:N_RAYS]
    wall = obs[N_RAYS:2 * N_RAYS]
    vel = obs[2 * N_RAYS:2 * N_RAYS + 2] * V_MAX
    tdir = obs[2 * N_RAYS + 2:2 * N_RAYS + 4]
    tdist = obs[2 * N_RAYS + 4] * ARENA * np.sqrt(2)

    desired = tdir.copy()
    # hazards: strong push away from close hits, quadratic falloff
    closeness = np.clip(1.0 - haz / 0.6, 0.0, None)  # only rays shorter than 60% of range
    desired -= hazard_gain * (closeness ** 2) @ RAY_DIRS
    # walls: gentle push when nearly touching
    wclose = np.clip(1.0 - wall / 0.15, 0.0, None)
    desired -= wall_gain * (wclose ** 2) @ RAY_DIRS

    n = np.linalg.norm(desired)
    if n < 1e-6:
        return 0
    speed = V_MAX * min(1.0, 0.3 + tdist / 3.0)  # brake when close to the package
    desired_v = desired / n * speed
    need = desired_v - vel
    if np.linalg.norm(need) < 0.02:
        return 0  # close enough: coast
    scores = ACTION_DIRS[1:] @ need
    return int(np.argmax(scores)) + 1


def rollout(env: OrbitalCourierEnv, policy, seed: int, record: bool = True):
    obs, info = env.reset(seed=seed)
    obs_buf, act_buf, ret = [], [], 0.0
    while True:
        a = policy(obs)
        if record:
            obs_buf.append(obs)
            act_buf.append(a)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        if term or trunc:
            break
    return obs_buf, act_buf, ret, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="demos/expert.npz")
    p.add_argument("--only-success", action="store_true", help="keep only winning episodes")
    args = p.parse_args()

    env = OrbitalCourierEnv()
    all_obs, all_act, rets, wins = [], [], [], 0
    for ep in range(args.episodes):
        o, a, ret, info = rollout(env, expert_action, seed=args.seed + ep)
        rets.append(ret)
        wins += int(info["success"])
        if args.only_success and not info["success"]:
            continue
        all_obs += o
        all_act += a
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, obs=np.array(all_obs, np.float32), actions=np.array(all_act, np.int64))
    print(f"expert: mean return {np.mean(rets):.2f} +- {np.std(rets):.2f}, "
          f"success {wins / args.episodes:.1%}, saved {len(all_act)} transitions to {args.out}")


if __name__ == "__main__":
    main()
