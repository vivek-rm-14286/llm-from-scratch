"""Calibrate, evaluate and compare every trained model, then write the router manifest.

    python -m starter.calibrate

Reads   out/<model>/logits.npz            (written by the training scripts)
Writes  out/report.json, out/reliability.png, out/confusion.png, out/router.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .common import DEFAULT_DATA, DEFAULT_OUT, load_xy
    from .data import QUEUES, load_split
    from . import metrics as M
except ImportError:
    from common import DEFAULT_DATA, DEFAULT_OUT, load_xy
    from data import QUEUES, load_split
    import metrics as M

MODELS = ("baseline", "encoder", "decision_lm")


def evaluate_model(logits, y_cal, y_test, kinds, cost_misroute, cost_escalate):
    """All numbers for one model. The temperature is fit on the calibration split only."""
    # TODO: 1. T = fit_temperature on logits['cal'] / y_cal (never on test).
    # 2. On test: accuracy and macro-F1 (unchanged by T), NLL and ECE before and after scaling.
    # 3. threshold = bayes_threshold(costs); selective_stats on the calibrated test confidences.
    # 4. OOD: share of logits['ood'] whose calibrated confidence falls below the threshold.
    # Return a dict with the keys main() prints: accuracy, macro_f1, temperature, nll_raw, nll_cal,
    # ece_raw, ece_cal, threshold, at_threshold, cost_no_abstain, ood_abstain_rate, cost_with_ood_traffic,
    # accuracy_by_kind, abstain_by_kind, reliability_raw, reliability_cal, confusion.
    raise NotImplementedError("TODO: implement evaluate_model in starter/calibrate.py")


def plot_reliability(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(report)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        r = report[name]
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect")
        for key, label, color in (("reliability_raw", f"raw (ECE {r['ece_raw']:.3f})", "tab:red"),
                                  ("reliability_cal", f"T={r['temperature']:.2f} (ECE {r['ece_cal']:.3f})", "tab:blue")):
            ax.plot([row[2] for row in r[key]], [row[3] for row in r[key]], "o-", color=color, label=label)
        ax.set_title(name)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(QUEUES)), QUEUES, rotation=45, ha="right")
    ax.set_yticks(range(len(QUEUES)), QUEUES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(QUEUES)):
        for j in range(len(QUEUES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_confusion(cm):
    width = max(len(q) for q in QUEUES)
    print(" " * (width + 2) + " ".join(f"{q[:6]:>6s}" for q in QUEUES) + "   (rows = true, columns = predicted)")
    for q, row in zip(QUEUES, cm):
        print(f"  {q:<{width}s}" + " ".join(f"{v:6d}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--cost-misroute", type=float, default=5.0)
    ap.add_argument("--cost-escalate", type=float, default=1.0)
    ap.add_argument("--default", choices=MODELS, default=None, help="force the default model of load_router()")
    args = ap.parse_args()
    out = Path(args.out)

    _, y_cal = load_xy(args.data, "cal", label_key="clean_label")
    _, y_test = load_xy(args.data, "test", label_key="clean_label")
    kinds = np.array([r["kind"] for r in load_split(args.data, "test")])

    report = {}
    for name in MODELS:
        f = out / name / "logits.npz"
        if f.exists():
            report[name] = evaluate_model(np.load(f), y_cal, y_test, kinds, args.cost_misroute, args.cost_escalate)
    if not report:
        raise SystemExit(f"no logits found under {out}; run a training script first")

    tau = M.bayes_threshold(args.cost_misroute, args.cost_escalate)
    print(f"costs: misroute {args.cost_misroute:g}, escalate {args.cost_escalate:g}  ->  abstain below p = {tau:.2f}\n")
    print(f"{'model':12s} {'acc':>6s} {'F1':>6s} {'T':>5s} {'ECE raw':>8s} {'ECE cal':>8s} "
          f"{'abstain':>8s} {'err|routed':>10s} {'cost':>6s} {'no-abst':>8s} {'OOD abst':>9s}")
    for name, r in report.items():
        a = r["at_threshold"]
        print(f"{name:12s} {r['accuracy']:6.3f} {r['macro_f1']:6.3f} {r['temperature']:5.2f} {r['ece_raw']:8.3f} "
              f"{r['ece_cal']:8.3f} {a['abstain_rate']:8.3f} {a['error_rate_routed']:10.3f} {a['mean_cost']:6.3f} "
              f"{r['cost_no_abstain']:8.3f} {r['ood_abstain_rate']:9.3f}")
    print("\naccuracy by kind of message (and share abstained):")
    for name, r in report.items():
        print(f"  {name:12s} " + "  ".join(
            f"{k} {v:.3f} ({r['abstain_by_kind'][k]:.0%})" for k, v in r["accuracy_by_kind"].items()))

    default = args.default or min(report, key=lambda m: report[m]["cost_with_ood_traffic"])
    print(f"\ndefault model for load_router(): {default} "
          f"(lowest cost on test + OOD traffic: {report[default]['cost_with_ood_traffic']:.3f})")
    print(f"\nconfusion matrix, {default}:")
    print_confusion(report[default]["confusion"])

    plot_reliability(report, out / "reliability.png")
    plot_confusion(report[default]["confusion"], f"{default}: confusion matrix (test)", out / "confusion.png")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "router.json").write_text(json.dumps({
        "default": default, "threshold": tau, "queues": QUEUES,
        "costs": {"misroute": args.cost_misroute, "escalate": args.cost_escalate},
        "models": {m: {"temperature": r["temperature"]} for m, r in report.items()}}, indent=2))
    print(f"\nwrote {out / 'report.json'}, {out / 'reliability.png'}, {out / 'confusion.png'}, {out / 'router.json'}")


if __name__ == "__main__":
    main()
