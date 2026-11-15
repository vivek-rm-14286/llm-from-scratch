"""Model (c): a decision language model.

A small decoder LM reads the message once, then scores every queue option in the same forward
pass. Each option attends to the message and to itself, never to another option, and every
option starts at the same position id, so the scores cannot depend on the option order.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from .data import QUEUE_DESCRIPTIONS, QUEUES
except ImportError:
    from data import QUEUE_DESCRIPTIONS, QUEUES

BASE_LM = "HuggingFaceTB/SmolLM2-135M"
MAX_CONTEXT_TOKENS = 96


def make_context(text):
    return f"Support message: {text}\nRoute this message to the right queue."


def queue_options(queues=QUEUES):
    return [f"{q}: {QUEUE_DESCRIPTIONS[q]}" for q in queues]


def build_option_mask(seg):
    """Boolean attention mask for packed sequences.

    seg: (B, L) integer tensor. 0 = context token, k >= 1 = token of option k, -1 = padding.
    Returns allowed: (B, L, L) bool, allowed[b, i, j] is True when token i may attend to token j:
    j is not after i, and j is a context token or belongs to the same option as i.
    Every token may attend to itself, so no row is fully masked.
    """
    # TODO: allowed[b, i, j] = (j <= i) and (seg[b, j] == 0 or seg[b, j] == seg[b, i]) and seg[b, j] >= 0.
    # Then OR with the identity so that padding rows are not fully masked.
    raise NotImplementedError("TODO: implement build_option_mask in starter/decision_lm.py")


class DecisionLM(nn.Module):
    """context + N options -> N logits in one forward pass, order-invariant by construction."""

    def __init__(self, name=BASE_LM, backbone=None, tokenizer=None, max_context_tokens=MAX_CONTEXT_TOKENS):
        super().__init__()
        if backbone is None:                             # tests pass a tiny random backbone instead
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(name)
            backbone = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.name, self.tok, self.backbone = name, tokenizer, backbone
        self.max_context_tokens = max_context_tokens
        self.scorer = nn.Linear(backbone.config.hidden_size, 1)
        nn.init.zeros_(self.scorer.weight)               # start from uniform probabilities
        nn.init.zeros_(self.scorer.bias)
        self.sep = self._ids("\nOption:")
        self._option_cache = {}

    def _ids(self, text):
        return list(self.tok(text, add_special_tokens=False).input_ids)

    def _option_ids(self, option):
        if option not in self._option_cache:
            self._option_cache[option] = self.sep + self._ids(" " + option)
        return self._option_cache[option]

    def pack(self, context, options):
        """One sequence holding the context and all options.

        Returns (ids, seg, pos, last): token ids, segment ids (0 = context, k = option k),
        position ids (every option restarts right after the context) and, per option,
        the index of its last token.
        """
        # TODO: Tokenize the context (truncate to self.max_context_tokens): segment 0, positions 0..n_ctx-1.
        # For option k (1-based): ids from self._option_ids(option), segment k,
        # positions n_ctx, n_ctx+1, ... (the SAME start for every option), and record the index of its last token.
        raise NotImplementedError("TODO: implement pack in starter/decision_lm.py")

    def forward(self, contexts, options_list):
        """contexts: list of B strings. options_list: list of B lists of option strings.
        Returns (B, max_options) logits; missing options are -inf."""
        # TODO: pack every example, pad ids (pad id), seg (-1) and pos (0) to the longest sequence.
        # mask = build_option_mask(seg)[:, None]  -> (B, 1, L, L) boolean, True = may attend.
        # h = self.backbone(input_ids=..., attention_mask=mask, position_ids=...).last_hidden_state
        # logit of option k = self.scorer(h at that option's last token). Fill missing options with -inf.
        raise NotImplementedError("TODO: implement forward in starter/decision_lm.py")

    # ---- trainable subset, saving and loading -------------------------------------------------

    def set_trainable(self, top_layers):
        """Freeze everything except the top `top_layers` blocks, the final norm and the scorer."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for layer in self.backbone.layers[-top_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        for p in self.scorer.parameters():
            p.requires_grad = True
        self.top_layers = top_layers

    def save(self, model_dir):
        """Store only the tensors that training changed; the rest is reloaded from the base checkpoint."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        trained = {k: v.detach().cpu() for k, v in self.state_dict().items()
                   if k.startswith("scorer.") or self.get_parameter(k).requires_grad}
        torch.save(trained, model_dir / "trained.pt")
        (model_dir / "config.json").write_text(json.dumps(
            {"base": self.name, "queues": QUEUES, "max_context_tokens": self.max_context_tokens}))

    @classmethod
    def load(cls, model_dir):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        model = cls(cfg["base"], max_context_tokens=cfg["max_context_tokens"])
        model.load_state_dict(torch.load(model_dir / "trained.pt", map_location="cpu"), strict=False)
        return model.eval()


class QueueScorer:
    """Wraps a DecisionLM with the fixed queue options so it looks like the other two models."""

    def __init__(self, model, queues=QUEUES):
        self.model, self.options = model, queue_options(queues)

    @torch.inference_mode()
    def logits(self, texts, batch_size=16):
        self.model.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [make_context(t) for t in texts[i:i + batch_size]]
            out.append(self.model(batch, [self.options] * len(batch)).float().cpu().numpy())
        return np.concatenate(out)

    @classmethod
    def load(cls, model_dir):
        return cls(DecisionLM.load(model_dir))
