"""The stable entry point of the project.

    from starter.router import load_router
    router = load_router()                       # reads code/ticket_router/out/router.json
    router.route("we were charged twice")        # {"queue": ..., "probs": {...}, "confidence": ..., "abstain": ...}

If no trained artifact exists, load_router returns a keyword baseline (router.name says so),
so callers always get an object with the same interface.
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np

try:
    from .data import QUEUES
    from .metrics import softmax
except ImportError:                                # imported with solution/ itself on sys.path
    from data import QUEUES
    from metrics import softmax

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "out"
KEYWORD_NAME = "keyword-baseline (no trained artifact found)"

KEYWORDS = {
    "billing": ["invoice", "charge", "charged", "refund", "billing", "billed", "payment", "subscription",
                "plan", "card", "price", "pricing", "overage", "receipt"],
    "ingestion": ["ingestion", "data points", "metrics", "logs", "gaps", "missing", "dropped", "rejected",
                  "timestamp", "backfill", "series", "no data", "nm-4", "nm-5"],
    "alerting": ["alert", "alerts", "monitor", "paged", "page", "notification", "notifications", "on-call",
                 "escalation", "maintenance", "flapping", "nm-7"],
    "agent-install": ["agent", "nimbus-agent", "install", "installed", "installer", "upgrade", "agent.yaml",
                      "proxy", "helm", "daemonset", "collector", "nm-1", "nm-2"],
    "api": ["api", "endpoint", "429", "sdk", "terraform", "webhook", "token", "rate limit", "/v1", "nm-6"],
    "access": ["login", "log in", "password", "sso", "saml", "2fa", "two-factor", "role", "roles", "invite",
               "invitation", "admin", "permission", "permissions", "locked"],
    "other": ["feature request", "roadmap", "docs", "soc 2", "trial", "training", "status page", "feedback"],
}


class Router:
    """A scoring function plus the decision rule: temperature, then threshold."""

    def __init__(self, name, logits_fn, temperature=1.0, threshold=0.8, queues=QUEUES):
        self.name = name
        self._logits_fn = logits_fn                # list[str] -> (N, n_queues) array of logits
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.queues = list(queues)

    def route_batch(self, texts):
        # TODO: probs = softmax(self._logits_fn(texts), self.temperature).
        # For each row return {'queue': top queue, 'probs': {queue: float}, 'confidence': float(max prob),
        # 'abstain': bool(confidence < self.threshold)}. Use plain Python floats and bools.
        raise NotImplementedError("TODO: implement route_batch in starter/router.py")

    def route(self, text: str) -> dict:
        return self.route_batch([text])[0]


def keyword_logits(texts, queues=QUEUES, scale=1.5):
    """Count keyword hits per queue. No hits at all gives uniform probabilities, hence an abstention."""
    # TODO: For each text and queue, count how many KEYWORDS[queue] occur in the lower-cased text
    # (match whole words) and multiply by `scale`. Zero hits everywhere must give equal logits.
    raise NotImplementedError("TODO: implement keyword_logits in starter/router.py")


def keyword_router(threshold=0.8):
    return Router(KEYWORD_NAME, keyword_logits, temperature=1.0, threshold=threshold)


def _load_model(kind, model_dir):
    """Import lazily: the keyword baseline must work without torch installed."""
    if kind == "baseline":
        try:
            from .train_baseline import EmbeddingLR
        except ImportError:
            from train_baseline import EmbeddingLR
        return EmbeddingLR.load(model_dir).logits
    if kind == "encoder":
        try:
            from .train_encoder import EncoderClassifier
        except ImportError:
            from train_encoder import EncoderClassifier
        return EncoderClassifier.load(model_dir).logits
    if kind == "decision_lm":
        try:
            from .decision_lm import QueueScorer
        except ImportError:
            from decision_lm import QueueScorer
        return QueueScorer.load(model_dir).logits
    raise ValueError(f"unknown model kind {kind!r}")


def load_router(path=None, model=None):
    """Load the trained router.

    path:  directory holding router.json (default: code/ticket_router/out), or the json file itself.
    model: "baseline", "encoder" or "decision_lm" to override the default chosen by calibrate.py.
    Falls back to the keyword baseline when router.json does not exist.
    """
    # TODO: Resolve the manifest: `path` (default DEFAULT_OUT) is a directory with router.json, or the file.
    # If it does not exist: warnings.warn(...) mentioning the keyword baseline, return keyword_router().
    # Otherwise read it, pick `model or cfg['default']`, load that model's logits function with
    # _load_model(kind, manifest.parent / kind) and return Router(kind, logits_fn, temperature, threshold, queues).
    raise NotImplementedError("TODO: implement load_router in starter/router.py")


if __name__ == "__main__":
    import sys
    r = load_router()
    for message in sys.argv[1:] or ["We were charged twice for the Pro plan.", "getting 403", "banana bread recipe?"]:
        res = r.route(message)
        print(f"[{r.name}] {message!r} -> {res['queue']} ({res['confidence']:.2f})"
              f"{'  ABSTAIN: escalate to a human' if res['abstain'] else ''}")
