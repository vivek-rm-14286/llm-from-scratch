"""A decoder-only transformer with switchable parts.

Default config = GPT-2 style: learned positions, LayerNorm, GELU MLP, biases.
`GPTConfig.modern(...)` = RMSNorm + RoPE + SwiGLU, no biases.

Interface (same as the GPT in the Transformer lesson):
    model(idx)                  -> logits (B, T, V)
    model(idx, targets)         -> (logits, loss)
    model(idx, cache=cache)     -> logits for the new tokens only; `cache` is updated in place
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256        # maximum context length
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0
    bias: bool = True            # biases in Linear and LayerNorm (GPT-2 style)
    tie_weights: bool = True     # lm_head shares the token embedding matrix
    rmsnorm: bool = False        # RMSNorm instead of LayerNorm
    rope: bool = False           # rotary positions instead of a learned position table
    swiglu: bool = False         # SwiGLU MLP instead of GELU MLP
    rope_base: float = 10000.0

    @classmethod
    def modern(cls, **kw) -> "GPTConfig":
        return cls(**{"bias": False, "rmsnorm": True, "rope": True, "swiglu": True, **kw})

    @property
    def mlp_hidden(self) -> int:
        """4d for GELU. For SwiGLU, 8d/3 rounded up to a multiple of 8 (same parameter budget)."""
        if not self.swiglu:
            return 4 * self.d_model
        return 8 * math.ceil(8 * self.d_model / 3 / 8)


def count_params(cfg: GPTConfig) -> int:
    """Closed-form parameter count. Must equal GPT(cfg).num_params()."""
    d, L, V, h = cfg.d_model, cfg.n_layer, cfg.vocab_size, cfg.mlp_hidden
    b = 1 if cfg.bias else 0
    norm = d if cfg.rmsnorm else d + b * d                 # gain (+ bias for LayerNorm)
    attn = 3 * d * d + b * 3 * d + d * d + b * d           # qkv + output projection
    if cfg.swiglu:
        mlp = 3 * d * h                                    # gate, up, down; never biased
    else:
        mlp = d * h + b * h + h * d + b * d
    block = attn + mlp + 2 * norm
    total = V * d + L * block + norm                       # token embedding, blocks, final norm
    if not cfg.rope:
        total += cfg.block_size * d                        # learned position table
    if not cfg.tie_weights:
        total += V * d                                     # separate lm_head
    return total


# ----------------------------------------------------------------------------- parts
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None):
    """cos and sin tables of shape (seq_len, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)   # (T, Dh/2)
    angles = torch.cat([angles, angles], dim=-1)                                   # (T, Dh)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, Dh); cos, sin: (T, Dh). Rotates each pair (x_i, x_{i + Dh/2})."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.rope = cfg.rope
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.proj.RESIDUAL_SCALE = True
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos=None, sin=None, cache=None):
        """cos, sin: RoPE tables already sliced to this call's absolute positions.
        cache: None, or a dict holding this layer's past "k" and "v"."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if self.rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # rotate BEFORE caching k
        if cache is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        else:
            past = cache["k"].size(2) if "k" in cache else 0
            if past:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v                             # (B, H, past + T, Dh)
            # new position i sits at absolute position past + i and may see keys 0..past+i
            mask = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=cfg.bias)
        self.proj = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=cfg.bias)
        self.proj.RESIDUAL_SCALE = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.proj = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)
        self.proj.RESIDUAL_SCALE = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.gate(x)) * self.up(x)))


def make_norm(cfg: GPTConfig) -> nn.Module:
    return RMSNorm(cfg.d_model) if cfg.rmsnorm else nn.LayerNorm(cfg.d_model, bias=cfg.bias)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg)
        self.mlp = SwiGLU(cfg) if cfg.swiglu else MLP(cfg)

    def forward(self, x, cos=None, sin=None, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))


# ----------------------------------------------------------------------------- model
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = None if cfg.rope else nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        if cfg.rope:
            cos, sin = rope_cache(cfg.block_size, cfg.d_model // cfg.n_head, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)   # buffers: move with .to(), not trained
            self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_SCALE", False):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())   # tied weights are counted once

    def new_cache(self) -> list[dict]:
        """An empty KV cache: one dict per layer."""
        return [{} for _ in self.blocks]

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape
        past = cache[0]["k"].size(2) if cache is not None and "k" in cache[0] else 0
        assert past + T <= self.cfg.block_size, f"sequence length {past + T} > block_size"
        x = self.tok_emb(idx)
        cos = sin = None
        if self.cfg.rope:
            cos, sin = self.rope_cos[past:past + T], self.rope_sin[past:past + T]
        else:
            x = x + self.pos_emb(torch.arange(past, past + T, device=idx.device))
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if cache is None else cache[i])
        logits = self.lm_head(self.norm_f(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


if __name__ == "__main__":
    for name, cfg in [("gpt2-style", GPTConfig()), ("modern", GPTConfig.modern())]:
        model = GPT(cfg)
        print(f"{name:10s} instantiated {model.num_params():,}  formula {count_params(cfg):,}")
