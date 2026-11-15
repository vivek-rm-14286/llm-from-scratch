"""Per-request latency of each router on CPU: one message at a time, as a live inbox would send them.

    python -m starter.benchmark
"""
import argparse
import json
import time
from pathlib import Path

import torch

try:
    from .common import DEFAULT_DATA, DEFAULT_OUT, load_xy
    from .metrics import latency_percentiles
    from .router import keyword_router, load_router
except ImportError:
    from common import DEFAULT_DATA, DEFAULT_OUT, load_xy
    from metrics import latency_percentiles
    from router import keyword_router, load_router


def time_router(router, texts, warmup=10):
    # TODO: Call router.route on the first `warmup` texts and discard the timings.
    # Then time router.route(t) for every text with time.perf_counter (milliseconds)
    # and return latency_percentiles(samples).
    raise NotImplementedError("TODO: implement time_router in starter/benchmark.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    texts, _ = load_xy(args.data, "test", args.n)
    manifest = json.loads((Path(args.out) / "router.json").read_text())

    results = {"keyword": time_router(keyword_router(), texts)}
    for name in manifest["models"]:
        results[name] = time_router(load_router(args.out, model=name), texts)
    print(f"{args.n} single requests each, CPU, {args.threads} threads, fp32")
    print(f"{'model':12s} {'p50 ms':>8s} {'p95 ms':>8s} {'mean ms':>8s}")
    for name, r in results.items():
        print(f"{name:12s} {r['p50_ms']:8.2f} {r['p95_ms']:8.2f} {r['mean_ms']:8.2f}")
    (Path(args.out) / "latency.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
