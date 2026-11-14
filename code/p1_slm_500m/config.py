"""Model configs for Project 1.

All sizes are chosen so the parameter count can be checked by hand (see
count_params.py and the project page). Every preset uses the same
architecture: a Llama-style decoder with RMSNorm, RoPE, SwiGLU, GQA,
tied input/output embeddings and no bias terms.
"""
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 49152      # 3 * 2^14: multiple of 64/128 for fast matmuls
    d_model: int = 1024          # hidden size
    n_layers: int = 32           # deep-and-thin
    n_heads: int = 16            # query heads -> head_dim = 64
    n_kv_heads: int = 4          # GQA: 4 query heads share each K/V head
    ffn_hidden: int = 3584       # SwiGLU hidden size (3.5 x d_model)
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self):
        return asdict(self)


PRESETS = {
    # ~487M params. The main target of this project.
    "p500m": ModelConfig(),
    # ~126M params. The "cheap path" that fits a single 24 GB GPU comfortably.
    "p125m": ModelConfig(d_model=768, n_layers=14, n_heads=12, n_kv_heads=4,
                         ffn_hidden=2048),
    # A few hundred thousand params: runs on a laptop CPU/MPS in a minute.
    "tiny": ModelConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=4,
                        n_kv_heads=2, ffn_hidden=172, max_seq_len=128),
}


def get_config(name: str, **overrides) -> ModelConfig:
    base = PRESETS[name]
    cfg = ModelConfig(**{**base.to_dict(), **overrides})
    assert cfg.d_model % cfg.n_heads == 0, "d_model must divide by n_heads"
    assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads must divide by n_kv_heads"
    assert cfg.head_dim % 2 == 0, "RoPE needs an even head_dim"
    return cfg
