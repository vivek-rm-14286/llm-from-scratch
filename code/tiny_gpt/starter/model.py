"""Steps 2, 3 and 6. A decoder-only transformer with switchable parts.

Default config = GPT-2 style: learned positions, LayerNorm, GELU MLP, biases.
`GPTConfig.modern(...)` = RMSNorm + RoPE + SwiGLU, no biases.

Interface (same as the GPT in the Transformer lesson):
    model(idx)                  -> logits (B, T, V)
    model(idx, targets)         -> (logits, loss)
    model(idx, cache=cache)     -> logits for the new tokens only; `cache` is updated in place

Suggested order:
    step 2: count_params (gpt2 branch), CausalSelfAttention, MLP, Block, GPT   -> pytest -k gpt2
    step 3: the cache branch of CausalSelfAttention.forward and GPT.forward     (can wait until step 7)
    step 6: RMSNorm, rope_cache, apply_rope, SwiGLU, the modern branches        -> pytest -k "modern or rope or rmsnorm"

Grade with:  pytest tests/test_model.py        (from code/tiny_gpt)
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:                 # given: the tests construct configs with these field names
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
        # TODO: 4 * d_model without SwiGLU; with SwiGLU the smallest multiple of 8 that is >= 8 * d_model / 3
        #       (d_model=384 -> 1024, d_model=256 -> 688).
        raise NotImplementedError("GPTConfig.mlp_hidden")


def count_params(cfg: GPTConfig) -> int:
    """Closed-form parameter count. Must equal GPT(cfg).num_params().

    Derive it on paper first: token embedding, optional position table, per block
    (attention qkv + output projection, MLP, two norms), final norm, and a separate
    lm_head only when the weights are not tied. Handle cfg.bias, cfg.rmsnorm,
    cfg.rope and cfg.swiglu.
    """
    # TODO
    raise NotImplementedError("count_params")


# ----------------------------------------------------------------------------- parts
class RMSNorm(nn.Module):
    """x / rms(x) * gain, over the last dimension. One parameter vector `weight` (init 1), no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # TODO (step 6)
        raise NotImplementedError("RMSNorm.__init__")

    def forward(self, x):
        # TODO (step 6): rms = sqrt(mean(x^2, last dim) + eps)
        raise NotImplementedError("RMSNorm.forward")


def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None):
    """Return (cos, sin), each of shape (seq_len, head_dim).

    Frequencies theta_i = base^(-2i/head_dim) for i in 0..head_dim/2-1; angle[pos, i] = pos * theta_i.
    Rotate-half layout: the angles are repeated twice along the last dimension, so that
    dimension i is paired with dimension i + head_dim/2.
    """
    # TODO (step 6)
    raise NotImplementedError("rope_cache")


def apply_rope(x, cos, sin):
    """x: (B, H, T, Dh); cos, sin: (T, Dh). Rotate each pair (x_i, x_{i + Dh/2}) by its angle.

    result = x * cos + rotate_half(x) * sin, where rotate_half([x1, x2]) = [-x2, x1].
    """
    # TODO (step 6)
    raise NotImplementedError("apply_rope")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        # TODO: self.qkv  = Linear(d, 3d, bias=cfg.bias)   one fused projection for q, k, v
        #       self.proj = Linear(d, d, bias=cfg.bias)    output projection; mark it with
        #                   self.proj.RESIDUAL_SCALE = True for the scaled init in GPT._init_weights
        #       a residual dropout, and whatever else forward needs (n_head, head_dim, cfg.rope, cfg.dropout)
        raise NotImplementedError("CausalSelfAttention.__init__")

    def forward(self, x, cos=None, sin=None, cache=None):
        """x: (B, T, d) -> (B, T, d).

        cos, sin: RoPE tables already sliced to this call's absolute positions (None without RoPE).
        cache:    None for a normal forward pass. Otherwise a dict that holds this layer's past
                  keys and values under "k" and "v", each (B, H, past, Dh); empty on the first call.
        """
        # TODO (step 2): project to q, k, v; reshape each to (B, H, T, Dh).
        # TODO (step 6): if RoPE is on, rotate q and k (not v). Do it BEFORE k goes into the cache.
        # TODO (step 2): cache is None -> F.scaled_dot_product_attention(q, k, v, is_causal=True,
        #                dropout_p=... only in training mode)
        # TODO (step 3): cache given   -> concatenate past k, v in front (dim=2), store the result
        #                back in cache["k"], cache["v"], and attend with a boolean mask of shape
        #                (T, past + T) in which new position i may see keys 0..past+i (True = attend).
        # TODO (step 2): merge heads back to (B, T, d), apply proj and the residual dropout.
        raise NotImplementedError("CausalSelfAttention.forward")


class MLP(nn.Module):
    """GPT-2 MLP: Linear(d, mlp_hidden) -> GELU (tanh approximation) -> Linear(mlp_hidden, d) -> dropout."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # TODO: two Linear layers with bias=cfg.bias; mark the second with RESIDUAL_SCALE = True
        raise NotImplementedError("MLP.__init__")

    def forward(self, x):
        # TODO
        raise NotImplementedError("MLP.forward")


class SwiGLU(nn.Module):
    """proj(silu(gate(x)) * up(x)) -> dropout. Three bias-free Linear layers; hidden size cfg.mlp_hidden."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # TODO (step 6): gate, up: d -> hidden; proj: hidden -> d with RESIDUAL_SCALE = True
        raise NotImplementedError("SwiGLU.__init__")

    def forward(self, x):
        # TODO (step 6)
        raise NotImplementedError("SwiGLU.forward")


def make_norm(cfg: GPTConfig) -> nn.Module:
    """RMSNorm(d) when cfg.rmsnorm, else nn.LayerNorm(d, bias=cfg.bias)."""
    # TODO (step 2: LayerNorm only is enough; step 6: add the RMSNorm branch)
    raise NotImplementedError("make_norm")


class Block(nn.Module):
    """Pre-norm block: x = x + attn(norm1(x)); x = x + mlp(norm2(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # TODO: norm1, attn, norm2, mlp (SwiGLU when cfg.swiglu, else MLP)
        raise NotImplementedError("Block.__init__")

    def forward(self, x, cos=None, sin=None, cache=None):
        # TODO: pass cos, sin and this layer's cache dict through to the attention
        raise NotImplementedError("Block.forward")


# ----------------------------------------------------------------------------- model
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # TODO: self.tok_emb  = Embedding(vocab_size, d)
        #       self.pos_emb  = Embedding(block_size, d), or None when cfg.rope
        #       self.drop, self.blocks (a ModuleList of n_layer Blocks), self.norm_f
        #       self.lm_head  = Linear(d, vocab_size, bias=False)
        #       if cfg.tie_weights: self.lm_head.weight = self.tok_emb.weight
        #       if cfg.rope: precompute rope_cache(block_size, head_dim, cfg.rope_base) and store both
        #                    tables with self.register_buffer(..., persistent=False)
        #       finally: self.apply(self._init_weights)
        # The tests read: model.cfg, model.tok_emb, model.lm_head, and parameter names
        # (no parameter name may contain "pos_emb" when cfg.rope is on).
        raise NotImplementedError("GPT.__init__")

    def _init_weights(self, module):
        # TODO: Linear weights ~ N(0, 0.02), scaled by (2 * n_layer) ** -0.5 when the module is
        #       marked RESIDUAL_SCALE; biases zero; Embedding weights ~ N(0, 0.02).
        raise NotImplementedError("GPT._init_weights")

    def num_params(self) -> int:
        """Number of parameters, with tied weights counted once."""
        # TODO
        raise NotImplementedError("GPT.num_params")

    def new_cache(self) -> list[dict]:
        """An empty KV cache: one empty dict per layer."""
        # TODO (step 3)
        raise NotImplementedError("GPT.new_cache")

    def forward(self, idx, targets=None, cache=None):
        """idx: (B, T) int64.

        targets=None -> logits (B, T, V).  targets given -> (logits, loss), mean cross-entropy.
        With a cache, `idx` holds only the NEW tokens. Their absolute positions start at
        `past`, the number of positions already in the cache (0 if it is empty).
        """
        # TODO: assert past + T <= block_size
        #       token embeddings; add learned positions past..past+T-1, or slice the RoPE tables
        #       dropout, the blocks (block i gets cache[i]), final norm, lm_head
        #       loss = F.cross_entropy(logits.view(-1, V), targets.reshape(-1))
        raise NotImplementedError("GPT.forward")


if __name__ == "__main__":
    for name, cfg in [("gpt2-style", GPTConfig()), ("modern", GPTConfig.modern())]:
        model = GPT(cfg)
        print(f"{name:10s} instantiated {model.num_params():,}  formula {count_params(cfg):,}")
