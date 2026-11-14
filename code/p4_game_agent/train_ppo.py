"""PPO for Orbital Courier, optionally initialized from a behavior-cloned policy.

Single-file PPO (clipped objective + GAE + value loss + entropy bonus).
No vector-env wrapper: we step a plain list of envs and reset them by hand,
so the episode-boundary logic is explicit and easy to read.

Usage:
    python train_ppo.py --init checkpoints/bc.pt --iters 200 --out checkpoints/ppo.pt
    python train_ppo.py --iters 3 --n-envs 4 --n-steps 128          # smoke test
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import OBS_DIM, OrbitalCourierEnv
from train_bc import PolicyNet, load_policy, save_policy, torch_policy_fn, evaluate_quick


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.pi = PolicyNet(hidden=hidden)            # same architecture as BC -> weights load directly
        self.v = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.pi(obs))

    def value(self, obs):
        return self.v(obs).squeeze(-1)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards/values/dones: (T, N). dones[t]=1 if the episode ended at step t
    (terminated or truncated; a truncation is already bootstrapped into rewards[t]).

    A_t = sum_k (gamma*lam)^k delta_{t+k},  delta_t = r_t + gamma*V(s_{t+1})*(1-done_t) - V(s_t)
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=None, help="BC checkpoint to start the actor from")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=256, help="steps per env per iteration")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--target-kl", type=float, default=0.02, help="early-stop epochs past this KL")
    p.add_argument("--critic-warmup", type=int, default=5,
                   help="iterations that train only the critic (protects a BC-initialized actor)")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--out", default="checkpoints/ppo.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ac = ActorCritic(hidden=args.hidden)
    if args.init:
        bc = load_policy(args.init)
        ac.pi.load_state_dict(bc.state_dict())
        print(f"actor initialized from {args.init}")
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)

    envs = [OrbitalCourierEnv() for _ in range(args.n_envs)]
    obs = np.stack([e.reset(seed=args.seed * 1000 + i)[0] for i, e in enumerate(envs)])
    ep_ret = np.zeros(args.n_envs)
    recent_returns, recent_success = [], []

    T, N = args.n_steps, args.n_envs
    for it in range(1, args.iters + 1):
        t0 = time.time()
        buf_obs = torch.zeros(T, N, OBS_DIM)
        buf_act = torch.zeros(T, N, dtype=torch.long)
        buf_logp = torch.zeros(T, N)
        buf_rew = torch.zeros(T, N)
        buf_done = torch.zeros(T, N)
        buf_val = torch.zeros(T, N)

        # ---------------- rollout
        for t in range(T):
            o = torch.as_tensor(obs)
            with torch.no_grad():
                d = ac.dist(o)
                a = d.sample()
                buf_logp[t] = d.log_prob(a)
                buf_val[t] = ac.value(o)
            buf_obs[t] = o
            buf_act[t] = a
            for i, env in enumerate(envs):
                nobs, r, term, trunc, info = env.step(a[i].item())
                ep_ret[i] += r
                if trunc and not term:
                    # time-limit truncation is not a real terminal: bootstrap from V(s_final)
                    with torch.no_grad():
                        r += args.gamma * ac.value(torch.as_tensor(nobs).unsqueeze(0)).item()
                buf_rew[t, i] = r
                buf_done[t, i] = float(term or trunc)
                if term or trunc:
                    recent_returns.append(ep_ret[i])
                    recent_success.append(float(info["success"]))
                    ep_ret[i] = 0.0
                    nobs, _ = env.reset()
                obs[i] = nobs

        with torch.no_grad():
            last_v = ac.value(torch.as_tensor(obs))
        adv, ret = compute_gae(buf_rew, buf_val, buf_done, last_v, args.gamma, args.lam)

        # ---------------- update
        b_obs = buf_obs.reshape(T * N, OBS_DIM)
        b_act, b_logp = buf_act.reshape(-1), buf_logp.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
        warmup = it <= args.critic_warmup
        approx_kl = 0.0
        clipfrac = 0.0
        for epoch in range(args.epochs):
            perm = torch.randperm(T * N)
            for s in range(0, T * N, args.minibatch):
                mb = perm[s:s + args.minibatch]
                v = ac.value(b_obs[mb])
                v_loss = F.mse_loss(v, b_ret[mb])
                if warmup:
                    loss = args.vf_coef * v_loss
                else:
                    d = ac.dist(b_obs[mb])
                    logp = d.log_prob(b_act[mb])
                    ratio = torch.exp(logp - b_logp[mb])
                    a_mb = b_adv[mb]
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                    pg1 = ratio * a_mb
                    pg2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_mb
                    pi_loss = -torch.min(pg1, pg2).mean()
                    entropy = d.entropy().mean()
                    loss = pi_loss + args.vf_coef * v_loss - args.ent_coef * entropy
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                        clipfrac = ((ratio - 1).abs() > args.clip).float().mean().item()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
            if not warmup and approx_kl > args.target_kl:
                break

        mean_ret = np.mean(recent_returns[-50:]) if recent_returns else float("nan")
        succ = np.mean(recent_success[-50:]) if recent_success else float("nan")
        print(f"iter {it:4d} | ep_ret(50) {mean_ret:7.2f} | success(50) {succ:5.1%} | "
              f"v_loss {v_loss.item():7.3f} | kl {approx_kl:.4f} | clipfrac {clipfrac:.3f} | "
              f"{'critic-warmup | ' if warmup else ''}{time.time() - t0:.1f}s")

        if it % args.eval_every == 0 or it == args.iters:
            r, sr = evaluate_quick(torch_policy_fn(ac.pi), episodes=20)
            print(f"  greedy eval: mean return {r:.2f}, success {sr:.1%}")
            save_policy(ac.pi, args.out, source="ppo", iteration=it)

    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
