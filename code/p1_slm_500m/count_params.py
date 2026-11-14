"""Count parameters two ways (by formula and by instantiating on the meta device)
and print the training-compute estimate for a few token budgets.

    python count_params.py                 # the ~500M config
    python count_params.py --model p125m
    python count_params.py --tokens 10e9 50e9 --mfu 0.4 --peak_tflops 989
"""
import argparse

import torch

from config import PRESETS, get_config
from model import Transformer


def analytic_count(cfg):
    d, L, hd = cfg.d_model, cfg.n_layers, cfg.head_dim
    attn = d * (cfg.n_heads * hd) * 2 + d * (cfg.n_kv_heads * hd) * 2  # Wq, Wo, Wk, Wv
    ffn = 3 * d * cfg.ffn_hidden                                         # W1, W2, W3
    norms = 2 * d                                                        # two RMSNorm gains
    per_layer = attn + ffn + norms
    emb = cfg.vocab_size * d                                             # shared with lm_head
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * d
    total = L * per_layer + emb + head + d                               # + final norm
    return dict(attn=attn, ffn=ffn, norms=norms, per_layer=per_layer,
                emb=emb, head=head, total=total)


def flops_per_token(cfg, n_params):
    # 6N covers forward+backward through every weight matmul (the tied lm_head
    # matmul is real compute, so N includes the embedding matrix here).
    # The attention score/value matmuls add 12 * L * d * T per token (PaLM paper, App. B).
    return 6 * n_params + 12 * cfg.n_layers * cfg.d_model * cfg.max_seq_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="p500m", choices=list(PRESETS))
    ap.add_argument("--tokens", type=float, nargs="+", default=[10e9, 50e9])
    ap.add_argument("--peak_tflops", type=float, default=989.0, help="H100 SXM dense bf16")
    ap.add_argument("--mfu", type=float, default=0.40)
    args = ap.parse_args()

    cfg = get_config(args.model)
    with torch.device("meta"):          # no memory allocated, no init run
        model = Transformer(cfg)
    actual = model.num_params()
    a = analytic_count(cfg)

    print(f"config: {cfg}")
    print(f"head_dim={cfg.head_dim}  GQA groups={cfg.n_heads // cfg.n_kv_heads}")
    print(f"per layer: attn={a['attn']:,}  ffn={a['ffn']:,}  norms={a['norms']:,}  "
          f"-> {a['per_layer']:,}")
    print(f"{cfg.n_layers} layers           = {cfg.n_layers * a['per_layer']:,}")
    print(f"embedding (tied)    = {a['emb']:,}")
    print(f"analytic total      = {a['total']:,}")
    print(f"instantiated total  = {actual:,}  ({actual / 1e6:.1f}M)")
    print(f"non-embedding       = {actual - a['emb']:,}")
    assert actual == a["total"], "formula and module disagree!"

    fpt = flops_per_token(cfg, actual)
    eff = args.peak_tflops * 1e12 * args.mfu
    print(f"\ntraining FLOPs/token ~ {fpt / 1e9:.2f} GFLOP "
          f"(6N = {6 * actual / 1e9:.2f}, attention = {(fpt - 6 * actual) / 1e9:.2f})")
    print(f"assuming {args.peak_tflops:.0f} TFLOP/s peak at {args.mfu:.0%} MFU "
          f"-> {eff / 1e12:.0f} TFLOP/s effective per GPU")
    for D in args.tokens:
        total = fpt * D
        gpu_h = total / eff / 3600
        d_str = f"{D / 1e9:5.0f}B" if D >= 1e9 else f"{D / 1e6:5.0f}M"
        h_str = f"{gpu_h:,.0f}" if gpu_h >= 10 else f"{gpu_h:.2f}"
        print(f"  D = {d_str} tokens ({D / actual:5.0f} tokens/param): "
              f"{total:.2e} FLOPs  ~ {h_str} GPU-hours")


if __name__ == "__main__":
    main()
