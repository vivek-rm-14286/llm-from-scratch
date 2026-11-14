"""Behavior cloning (+ optional DAgger) for Orbital Courier.

Supervised learning on (observation, action) pairs: minimize cross-entropy
between the policy's action distribution and the demonstrator's action.

Usage:
    python train_bc.py --demos demos/*.npz --epochs 30 --out checkpoints/bc.pt
    python train_bc.py --expert-episodes 200 --dagger-iters 3   # no demo files needed
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import N_ACTIONS, OBS_DIM, OrbitalCourierEnv
from expert import expert_action, rollout


class PolicyNet(nn.Module):
    """Small MLP: observation -> action logits. Shared by BC, PPO, eval and ONNX export."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def save_policy(policy: PolicyNet, path: str, **meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hidden = policy.net[0].out_features
    torch.save({"state_dict": policy.state_dict(), "hidden": hidden, **meta}, path)


def load_policy(path: str) -> PolicyNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy = PolicyNet(hidden=ckpt["hidden"])
    policy.load_state_dict(ckpt["state_dict"])
    return policy


def torch_policy_fn(policy: PolicyNet):
    """Wrap a PolicyNet as obs -> greedy action."""
    @torch.no_grad()
    def act(obs):
        return int(policy(torch.as_tensor(obs).unsqueeze(0)).argmax(-1).item())
    return act


def fit(policy, obs, acts, epochs, lr, batch_size, val_frac=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(acts)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X = torch.as_tensor(obs)
    Y = torch.as_tensor(acts)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        policy.train()
        idx = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        tot = 0.0
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            loss = F.cross_entropy(policy(X[b]), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        policy.eval()
        with torch.no_grad():
            val_acc = (policy(X[val_idx]).argmax(-1) == Y[val_idx]).float().mean().item()
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"  epoch {ep + 1:3d}  train loss {tot / len(tr_idx):.4f}  val acc {val_acc:.3f}")
    return policy


def evaluate_quick(policy_fn, episodes=20, seed=10_000):
    env = OrbitalCourierEnv()
    rets, wins = [], 0
    for e in range(episodes):
        _, _, r, info = rollout(env, policy_fn, seed=seed + e, record=False)
        rets.append(r)
        wins += int(info["success"])
    return float(np.mean(rets)), wins / episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", nargs="*", default=[], help="npz files from play_record.py / expert.py")
    p.add_argument("--expert-episodes", type=int, default=0, help="generate expert demos on the fly")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dagger-iters", type=int, default=0, help="DAgger rounds (needs the expert)")
    p.add_argument("--dagger-episodes", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--out", default="checkpoints/bc.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    obs_list, act_list = [], []
    files = [f for pat in args.demos for f in glob.glob(pat)]
    for f in files:
        d = np.load(f)
        obs_list.append(d["obs"])
        act_list.append(d["actions"])
        print(f"loaded {len(d['actions'])} transitions from {f}")
    if args.expert_episodes > 0:
        env = OrbitalCourierEnv()
        for e in range(args.expert_episodes):
            o, a, _, _ = rollout(env, expert_action, seed=args.seed + e)
            obs_list.append(np.array(o, np.float32))
            act_list.append(np.array(a, np.int64))
        print(f"generated {args.expert_episodes} expert episodes")
    if not obs_list:
        raise SystemExit("no data: pass --demos or --expert-episodes")
    obs = np.concatenate(obs_list)
    acts = np.concatenate(act_list)
    print(f"dataset: {len(acts)} transitions, action histogram {np.bincount(acts, minlength=N_ACTIONS)}")

    policy = PolicyNet(hidden=args.hidden)
    print("behavior cloning")
    fit(policy, obs, acts, args.epochs, args.lr, args.batch_size, seed=args.seed)
    ret, sr = evaluate_quick(torch_policy_fn(policy), args.eval_episodes)
    print(f"BC policy: mean return {ret:.2f}, success {sr:.1%}")

    # DAgger: run the *learner*, ask the expert what it would have done, aggregate, refit.
    env = OrbitalCourierEnv()
    for it in range(args.dagger_iters):
        new_obs = []
        learner = torch_policy_fn(policy)
        for e in range(args.dagger_episodes):
            o, _, _, _ = rollout(env, learner, seed=50_000 + it * 1000 + e)
            new_obs += o
        new_obs = np.array(new_obs, np.float32)
        new_acts = np.array([expert_action(x) for x in new_obs], np.int64)
        obs = np.concatenate([obs, new_obs])
        acts = np.concatenate([acts, new_acts])
        print(f"DAgger iter {it + 1}: +{len(new_acts)} relabeled states (total {len(acts)})")
        fit(policy, obs, acts, max(1, args.epochs // 2), args.lr, args.batch_size, seed=args.seed + it + 1)
        ret, sr = evaluate_quick(torch_policy_fn(policy), args.eval_episodes)
        print(f"  after DAgger iter {it + 1}: mean return {ret:.2f}, success {sr:.1%}")

    save_policy(policy, args.out, source="bc")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
