"""Convert a checkpoint from train.py / sft.py into a Hugging Face LlamaForCausalLM folder.

    python convert_to_hf.py --ckpt out/p500m/ckpt_latest.pt --out hf/slm-500m-base
    python convert_to_hf.py --ckpt out/sft/ckpt_latest.pt   --out hf/slm-500m-instruct --chat
    python convert_to_hf.py --check     # self-test: random tiny model, logits must match HF

Because model.py implements RoPE with the same rotate_half layout as HF Llama,
the mapping is a pure rename. (Checkpoints from Meta's original Llama code use
the interleaved-pairs layout and need a permutation of Wq/Wk rows; ours do not.)
"""
import argparse
import os
import tempfile

import torch

from config import ModelConfig, get_config
from model import Transformer

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def hf_config(cfg: ModelConfig, bos_id=0, eos_id=0, pad_id=None):
    from transformers import LlamaConfig
    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.ffn_hidden,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        hidden_act="silu",
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope_theta,       # transformers v5 folds this into rope_parameters
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=cfg.tie_embeddings,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        dtype="bfloat16",
    )


def to_hf_state_dict(sd, cfg: ModelConfig):
    """Rename our keys to HF Llama keys. No tensor is reshaped or permuted."""
    out = {"model.embed_tokens.weight": sd["tok_emb.weight"],
           "model.norm.weight": sd["norm.weight"],
           "lm_head.weight": sd["tok_emb.weight"] if cfg.tie_embeddings else sd["lm_head.weight"]}
    names = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn.wq.weight": "self_attn.q_proj.weight",
        "attn.wk.weight": "self_attn.k_proj.weight",
        "attn.wv.weight": "self_attn.v_proj.weight",
        "attn.wo.weight": "self_attn.o_proj.weight",
        "ffn.w1.weight": "mlp.gate_proj.weight",
        "ffn.w3.weight": "mlp.up_proj.weight",
        "ffn.w2.weight": "mlp.down_proj.weight",
    }
    for i in range(cfg.n_layers):
        for ours, theirs in names.items():
            out[f"model.layers.{i}.{theirs}"] = sd[f"layers.{i}.{ours}"]
    return out


def build_hf_model(model: Transformer, dtype=torch.float32, **ids):
    from transformers import LlamaForCausalLM
    cfg = model.cfg
    hf = LlamaForCausalLM(hf_config(cfg, **ids)).to(dtype)
    sd = to_hf_state_dict({k: v.to(dtype) for k, v in model.state_dict().items()}, cfg)
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    # The only acceptable "missing" keys are non-persistent buffers like rotary inv_freq.
    assert not unexpected, unexpected
    assert all("rotary" in k or "inv_freq" in k for k in missing), missing
    if cfg.tie_embeddings:
        hf.tie_weights()
    return hf


def save_tokenizer(tokenizer_path, out_dir, chat):
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<|endoftext|>",
        eos_token="<|im_end|>" if chat else "<|endoftext|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tok.chat_template = CHATML_TEMPLATE
    tok.save_pretrained(out_dir)
    return tok


def convert(ckpt_path, out_dir, chat, dtype=torch.bfloat16):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    tok = save_tokenizer(ckpt["tokenizer"], out_dir, chat)
    hf = build_hf_model(model, dtype, bos_id=tok.bos_token_id, eos_id=tok.eos_token_id,
                        pad_id=tok.pad_token_id)
    hf.save_pretrained(out_dir)
    print(f"wrote {out_dir}: {sum(p.numel() for p in hf.parameters()):,} params, eos={tok.eos_token!r}")


@torch.no_grad()
def self_check():
    """Random tiny model -> HF -> compare logits, a save/load round trip, and greedy decoding."""
    from transformers import LlamaForCausalLM
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=300, n_layers=3, n_heads=4, n_kv_heads=2, rope_theta=500.0)
    model = Transformer(cfg)
    model.init_weights()
    for n, p in model.named_parameters():   # make norms non-trivial so a swapped norm would show
        if p.dim() == 1:
            p.uniform_(0.5, 1.5)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 40))
    ours, _ = model(x)

    hf = build_hf_model(model).eval()
    theirs = hf(x).logits
    d1 = (ours - theirs).abs().max().item()
    print(f"max |logit diff| ours vs HF          : {d1:.2e}")

    with tempfile.TemporaryDirectory() as d:
        hf.save_pretrained(d)
        hf2 = LlamaForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    d2 = (ours - hf2(x).logits).abs().max().item()
    print(f"max |logit diff| after save/load     : {d2:.2e}")

    prompt = x[:1, :8]
    mine = list(model.generate(prompt, 20, temperature=0.0))
    ref = hf.generate(prompt, max_new_tokens=20, do_sample=False, eos_token_id=None,
                      pad_token_id=0)[0, 8:].tolist()
    print(f"greedy decode (our KV cache) == HF   : {mine == ref}")
    nocache = list(model.generate(prompt, 20, temperature=0.0, use_cache=False))
    print(f"KV-cache decode == no-cache decode   : {mine == nocache}")
    ok = d1 < 1e-4 and d2 < 1e-4 and mine == ref and mine == nocache
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--chat", action="store_true", help="SFT model: eos = <|im_end|>")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        self_check()
    else:
        os.makedirs(args.out, exist_ok=True)
        convert(args.ckpt, args.out, args.chat)


if __name__ == "__main__":
    main()
