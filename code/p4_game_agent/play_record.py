"""Play Orbital Courier with the keyboard and record demonstrations.

Arrow keys (or WASD) thrust; combine two keys for diagonals. No key = coast.
++esc++ quits, R restarts the current episode without saving it.
Each finished episode is appended; the file is written when you quit.

Usage:
    pip install pygame
    python play_record.py --out demos/human.npz --episodes 5
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from env import ACTION_DIRS, OrbitalCourierEnv


def keys_to_action(pressed, pygame) -> int:
    dx = (pressed[pygame.K_RIGHT] or pressed[pygame.K_d]) - (pressed[pygame.K_LEFT] or pressed[pygame.K_a])
    dy = (pressed[pygame.K_UP] or pressed[pygame.K_w]) - (pressed[pygame.K_DOWN] or pressed[pygame.K_s])
    if dx == 0 and dy == 0:
        return 0
    v = np.array([dx, dy], dtype=float)
    v /= np.linalg.norm(v)
    return int(np.argmax(ACTION_DIRS[1:] @ v)) + 1  # nearest of the 8 compass directions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="demos/human.npz")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--keep-crashes", action="store_true",
                   help="also save episodes that ended in a crash (default: drop them)")
    args = p.parse_args()

    import pygame

    env = OrbitalCourierEnv(render_mode="human")
    env.metadata["render_fps"] = args.fps
    all_obs, all_act = [], []
    ep, quit_ = 0, False
    while ep < args.episodes and not quit_:
        obs, _ = env.reset(seed=args.seed + ep)
        env.render()
        ep_obs, ep_act, ret = [], [], 0.0
        restart = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    quit_ = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    restart = True
            if quit_ or restart:
                break
            a = keys_to_action(pygame.key.get_pressed(), pygame)
            ep_obs.append(obs)
            ep_act.append(a)
            obs, r, term, trunc, info = env.step(a)  # env.step renders in human mode
            ret += r
            if term or trunc:
                keep = args.keep_crashes or info["event"] != "crash"
                print(f"episode {ep}: return {ret:.1f}, packages {info['collected']}, "
                      f"{info['event'] or 'timeout'} -> {'saved' if keep else 'dropped'}")
                if keep:
                    all_obs += ep_obs
                    all_act += ep_act
                ep += 1
                break
    env.close()
    if all_act:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez_compressed(args.out, obs=np.array(all_obs, np.float32), actions=np.array(all_act, np.int64))
        print(f"saved {len(all_act)} transitions to {args.out}")
    else:
        print("nothing recorded")


if __name__ == "__main__":
    main()
