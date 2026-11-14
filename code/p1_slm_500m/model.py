"""A Llama-style decoder-only transformer, written to be read.

Design choices (each explained on the project page):
  * RMSNorm, pre-norm residual blocks
  * Rotary position embeddings (RoPE) using the *rotate_half* convention,
    exactly like Hugging Face's LlamaForCausalLM, so converted weights load
    with no permutation
  * Grouped-query attention (GQA) through F.scaled_dot_product_attention
  * SwiGLU feed-forward
  * Tied input embedding / output head, no bias terms anywhere
  * An optional KV cache for fast generation
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute the statistic in fp32 (bf16 is too coarse for mean of squares),
        # then cast back. This matches HF LlamaRMSNorm bit-for-bit in fp32.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    """Return cos/sin tables of shape (max_seq_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)              # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (T, head_dim)  <- HF layout
    return emb.cos(), emb.sin()


def rotate_half(x):
    """(x1, x2) -> (-x2, x1) where x1/x2 are the first/second half of the last dim."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:
            # kv_cache: dict with preallocated "k"/"v" of shape (B, n_kv, max_T, hd)
            kv_cache["k"][:B, :, start_pos:start_pos + T] = k
            kv_cache["v"][:B, :, start_pos:start_pos + T] = v
            k = kv_cache["k"][:B, :, :start_pos + T]
            v = kv_cache["v"][:B, :, :start_pos + T]

        # GQA: query head h reads K/V head h // n_rep (same as HF repeat_kv).
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        if start_pos == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v)  # one new token sees everything
        else:
            # Chunk of T new tokens after start_pos cached ones: explicit causal mask.
            S = start_pos + T
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=start_pos)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)  # up
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache, start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # one matrix, two uses

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        # persistent=False: derived from config, so not stored in checkpoints.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ---- initialisation ----------------------------------------------------
    @torch.no_grad()
    def init_weights(self):
        std = 0.02
        for name, p in self.named_parameters():
            if p.dim() == 2:
                # Residual-output projections get scaled down by 1/sqrt(2L) so the
                # residual stream variance doesn't grow with depth (GPT-2 trick).
                if name.endswith("wo.weight") or name.endswith("w2.weight"):
                    nn.init.normal_(p, 0.0, std / math.sqrt(2 * self.cfg.n_layers))
                else:
                    nn.init.normal_(p, 0.0, std)
            else:
                nn.init.ones_(p)  # RMSNorm gains
        # The rope buffers are not parameters; recompute in case we were built on 'meta'.
        cos, sin = precompute_rope(self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta)
        self.rope_cos = cos.to(self.rope_cos.device)
        self.rope_sin = sin.to(self.rope_sin.device)

    def num_params(self) -> int:
        # .parameters() de-duplicates shared tensors, so tied weights count once.
        return sum(p.numel() for p in self.parameters())

    # ---- forward -----------------------------------------------------------
    def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos:start_pos + T].to(x.dtype)
        sin = self.rope_sin[start_pos:start_pos + T].to(x.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, None if kv_caches is None else kv_caches[i], start_pos)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # -100 marks positions we don't train on (padding, user turns in SFT).
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    # ---- generation --------------------------------------------------------
    def new_kv_caches(self, batch_size, device, dtype):
        shape = (batch_size, self.cfg.n_kv_heads, self.cfg.max_seq_len, self.cfg.head_dim)
        return [{"k": torch.zeros(shape, device=device, dtype=dtype),
                 "v": torch.zeros(shape, device=device, dtype=dtype)}
                for _ in self.layers]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.95,
                 stop_ids=(), use_cache=True):
        """Sample tokens. idx: (1, T) prompt. Yields one token id at a time."""
        dtype = next(self.parameters()).dtype
        caches = self.new_kv_caches(idx.size(0), idx.device, dtype) if use_cache else None
        pos = 0
        x = idx
        for _ in range(max_new_tokens):
            if idx.size(1) >= self.cfg.max_seq_len:
                break
            if use_cache:
                logits, _ = self(x, kv_caches=caches, start_pos=pos)
                pos += x.size(1)
            else:
                logits, _ = self(idx)
            next_id = sample_next(logits[:, -1, :], temperature, top_p)
            idx = torch.cat([idx, next_id], dim=1)
            x = next_id
            tok = next_id.item()
            yield tok
            if tok in stop_ids:
                break


def sample_next(logits, temperature=0.8, top_p=0.95):
    """Temperature + nucleus (top-p) sampling over a (B, V) logits tensor."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cum = sorted_probs.cumsum(dim=-1)
        # Drop tokens once the mass *before* them already exceeds top_p.
        sorted_probs[(cum - sorted_probs) > top_p] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, choice)
    return torch.multinomial(probs, 1)
