"""Orbital Courier: a tiny 2D arena game as a Gymnasium environment.

A drone with momentum collects packages while avoiding drifting hazards.
The simulation is pure numpy and headless; pygame is imported only when you
ask for rendering.

Observation (float32, length OBS_DIM = 2 * N_RAYS + 5):
    [0:N_RAYS]            hazard ray distances, normalized to [0, 1] (1 = nothing seen)
    [N_RAYS:2*N_RAYS]     wall ray distances, normalized to [0, 1]
    [2N:2N+2]             drone velocity / V_MAX
    [2N+2:2N+4]           unit vector pointing at the package
    [2N+4]                distance to the package / arena diagonal

Actions (Discrete(9)): 0 = coast, 1..8 = thrust in one of 8 compass directions
(E, NE, N, NW, W, SW, S, SE). +y is "up".
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ARENA = 10.0  # square arena side length
DRONE_R = 0.25
PKG_R = 0.35
HAZ_R = 0.6
ACCEL = 0.05
DAMPING = 0.97
V_MAX = 0.35
N_RAYS = 16
RAY_RANGE = 4.0
OBS_DIM = 2 * N_RAYS + 5
N_ACTIONS = 9

RAY_ANGLES = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
RAY_DIRS = np.stack([np.cos(RAY_ANGLES), np.sin(RAY_ANGLES)], axis=1)  # (N_RAYS, 2)

_ang = np.arange(8) * (np.pi / 4)
ACTION_DIRS = np.concatenate([np.zeros((1, 2)), np.stack([np.cos(_ang), np.sin(_ang)], 1)])  # (9, 2)


class OrbitalCourierEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, n_hazards: int = 6,
                 packages_to_win: int = 5, max_steps: int = 600,
                 shaping: float = 0.5):
        self.render_mode = render_mode
        self.n_hazards = n_hazards
        self.packages_to_win = packages_to_win
        self.max_steps = max_steps
        self.shaping = shaping
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._screen = None
        self._clock = None

    # ------------------------------------------------------------------ core
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.pos = np.array([ARENA / 2, ARENA / 2])
        self.vel = np.zeros(2)
        self.haz_pos = np.zeros((self.n_hazards, 2))
        for i in range(self.n_hazards):
            while True:  # spawn hazards away from the drone
                p = rng.uniform(HAZ_R, ARENA - HAZ_R, size=2)
                if np.linalg.norm(p - self.pos) > 3.0:
                    break
            self.haz_pos[i] = p
        ang = rng.uniform(0, 2 * np.pi, size=self.n_hazards)
        speed = rng.uniform(0.02, 0.06, size=self.n_hazards)
        self.haz_vel = np.stack([np.cos(ang), np.sin(ang)], 1) * speed[:, None]
        self._spawn_package()
        self.steps = 0
        self.collected = 0
        return self._obs(), self._info()

    def _spawn_package(self):
        rng = self.np_random
        for _ in range(100):
            p = rng.uniform(1.0, ARENA - 1.0, size=2)
            far_from_drone = np.linalg.norm(p - self.pos) > 2.0
            far_from_haz = np.all(np.linalg.norm(self.haz_pos - p, axis=1) > HAZ_R + 1.0)
            if far_from_drone and far_from_haz:
                break
        self.pkg = p

    def step(self, action):
        action = int(action)
        d_prev = np.linalg.norm(self.pkg - self.pos)

        # drone physics: thrust, damping, speed cap, soft wall bounce
        self.vel = self.vel * DAMPING + ACTION_DIRS[action] * ACCEL
        speed = np.linalg.norm(self.vel)
        if speed > V_MAX:
            self.vel *= V_MAX / speed
        self.pos = self.pos + self.vel
        for k in range(2):
            if self.pos[k] < DRONE_R:
                self.pos[k], self.vel[k] = DRONE_R, abs(self.vel[k]) * 0.5
            elif self.pos[k] > ARENA - DRONE_R:
                self.pos[k], self.vel[k] = ARENA - DRONE_R, -abs(self.vel[k]) * 0.5

        # hazards drift and bounce off walls
        self.haz_pos += self.haz_vel
        lo, hi = HAZ_R, ARENA - HAZ_R
        for k in range(2):
            out = (self.haz_pos[:, k] < lo) | (self.haz_pos[:, k] > hi)
            self.haz_vel[out, k] *= -1
            self.haz_pos[:, k] = np.clip(self.haz_pos[:, k], lo, hi)

        self.steps += 1
        d_new = np.linalg.norm(self.pkg - self.pos)
        reward = -0.01 + self.shaping * (d_prev - d_new)  # time cost + progress shaping
        terminated = False
        event = None

        if d_new < DRONE_R + PKG_R:
            reward += 10.0
            self.collected += 1
            event = "pickup"
            if self.collected >= self.packages_to_win:
                terminated = True
                event = "win"
            else:
                self._spawn_package()

        if np.any(np.linalg.norm(self.haz_pos - self.pos, axis=1) < DRONE_R + HAZ_R):
            reward -= 10.0
            terminated = True
            event = "crash"

        truncated = (not terminated) and self.steps >= self.max_steps
        info = self._info()
        info["event"] = event
        if event == "crash":  # picking up the last package and crashing on the same step is a crash
            info["success"] = False
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    def _info(self):
        return {"collected": self.collected, "success": self.collected >= self.packages_to_win}

    # ----------------------------------------------------------- sensors
    def _hazard_rays(self) -> np.ndarray:
        """Distance along each ray to the first hazard circle, capped at RAY_RANGE."""
        rel = self.haz_pos - self.pos                      # (H, 2)
        t = RAY_DIRS @ rel.T                               # (R, H) projection on ray
        perp2 = (rel ** 2).sum(1)[None, :] - t ** 2        # squared perpendicular distance
        r = HAZ_R + DRONE_R                                # inflate by drone size
        hit = (t > 0) & (perp2 < r ** 2)
        dist = t - np.sqrt(np.clip(r ** 2 - perp2, 0, None))
        dist = np.where(hit, np.clip(dist, 0, None), RAY_RANGE)
        return np.minimum(dist.min(1), RAY_RANGE)

    def _wall_rays(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            tx = np.where(RAY_DIRS[:, 0] > 0, (ARENA - self.pos[0]) / RAY_DIRS[:, 0],
                          np.where(RAY_DIRS[:, 0] < 0, -self.pos[0] / RAY_DIRS[:, 0], np.inf))
            ty = np.where(RAY_DIRS[:, 1] > 0, (ARENA - self.pos[1]) / RAY_DIRS[:, 1],
                          np.where(RAY_DIRS[:, 1] < 0, -self.pos[1] / RAY_DIRS[:, 1], np.inf))
        return np.minimum(np.minimum(tx, ty), RAY_RANGE)

    def _obs(self) -> np.ndarray:
        to_pkg = self.pkg - self.pos
        d = np.linalg.norm(to_pkg)
        obs = np.concatenate([
            self._hazard_rays() / RAY_RANGE,
            self._wall_rays() / RAY_RANGE,
            self.vel / V_MAX,
            to_pkg / (d + 1e-8),
            [d / (ARENA * math.sqrt(2))],
        ])
        return obs.astype(np.float32)

    # ----------------------------------------------------------- rendering
    def render(self):
        if self.render_mode is None:
            return None
        import pygame  # lazy import: training never needs pygame

        size = 600
        scale = size / ARENA
        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((size, size))
                pygame.display.set_caption("Orbital Courier")
            else:
                self._screen = pygame.Surface((size, size))
            self._clock = pygame.time.Clock()
            try:
                self._font = pygame.font.SysFont(None, 24)
            except Exception:  # some pygame builds ship without a working font module
                self._font = None

        def px(p):  # world -> screen (flip y)
            return int(p[0] * scale), int(size - p[1] * scale)

        s = self._screen
        s.fill((16, 18, 32))
        hz = self._hazard_rays()
        for i in range(N_RAYS):
            end = self.pos + RAY_DIRS[i] * hz[i]
            col = (200, 70, 70) if hz[i] < RAY_RANGE else (50, 55, 80)
            pygame.draw.line(s, col, px(self.pos), px(end), 1)
        for p in self.haz_pos:
            pygame.draw.circle(s, (220, 60, 60), px(p), int(HAZ_R * scale))
        pygame.draw.circle(s, (250, 200, 60), px(self.pkg), int(PKG_R * scale))
        pygame.draw.circle(s, (90, 200, 255), px(self.pos), int(DRONE_R * scale))
        if self._font is not None:
            txt = self._font.render(f"packages {self.collected}/{self.packages_to_win}  step {self.steps}",
                                    True, (230, 230, 230))
            s.blit(txt, (8, 8))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(s), (1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None


def make_env(**kwargs) -> OrbitalCourierEnv:
    return OrbitalCourierEnv(**kwargs)


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = OrbitalCourierEnv()
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    total, steps = 0.0, 0
    while True:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        steps += 1
        if term or trunc:
            break
    print(f"obs_dim={obs.shape[0]} random episode: steps={steps} return={total:.2f} info={info}")
