"""Export the policy MLP to ONNX and measure single-observation latency.

A System 1 controller has to answer every frame. At 60 Hz you have ~16.7 ms per
frame for *everything* (physics, rendering, input); the policy should take a
tiny fraction of that. This script measures it for PyTorch eager vs ONNX Runtime.

Usage:
    python export_onnx.py --policy checkpoints/ppo.pt --out checkpoints/ppo.onnx
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from env import OBS_DIM, OrbitalCourierEnv
from train_bc import load_policy


def bench(fn, x, n=2000, warmup=200):
    for _ in range(warmup):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(x)
    return (time.perf_counter() - t0) / n * 1e6  # microseconds per call


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--out", default="checkpoints/policy.onnx")
    p.add_argument("--iters", type=int, default=2000)
    args = p.parse_args()

    torch.set_num_threads(1)  # realistic for an embedded game loop; also lowers jitter
    policy = load_policy(args.policy).eval()
    dummy = torch.zeros(1, OBS_DIM)
    torch.onnx.export(
        policy, (dummy,), args.out,
        input_names=["obs"], output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"exported {args.out}")

    obs, _ = OrbitalCourierEnv().reset(seed=0)
    x_t = torch.as_tensor(obs).unsqueeze(0)
    with torch.no_grad():
        ref = policy(x_t).numpy()
        us_torch = bench(lambda x: policy(x), x_t, args.iters)
    print(f"torch eager : {us_torch:8.1f} us / step")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping ONNX check (pip install onnxruntime)")
        return
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(args.out, so, providers=["CPUExecutionProvider"])
    x_np = obs[None, :].astype(np.float32)
    out = sess.run(None, {"obs": x_np})[0]
    print(f"max |torch - onnx| = {np.abs(out - ref).max():.2e}  (same argmax: {out.argmax() == ref.argmax()})")
    us_ort = bench(lambda x: sess.run(None, {"obs": x}), x_np, args.iters)
    print(f"onnxruntime : {us_ort:8.1f} us / step")
    budget_us = 1e6 / 60
    print(f"60 Hz frame budget is {budget_us:.0f} us -> policy uses {us_ort / budget_us:.2%} of it")


if __name__ == "__main__":
    main()
