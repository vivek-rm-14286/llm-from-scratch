"""The stable entry point of the project.

    from solution.router import load_router
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
        probs = softmax(np.asarray(self._logits_fn(list(texts)), dtype=np.float64), self.temperature)
        results = []
        for p in probs:
            top = int(p.argmax())
            confidence = float(p[top])
            results.append({
                "queue": self.queues[top],
                "probs": {q: float(v) for q, v in zip(self.queues, p)},
                "confidence": confidence,
                "abstain": bool(confidence < self.threshold),   # True = escalate to a human
            })
        return results

    def route(self, text: str) -> dict:
        return self.route_batch([text])[0]


def keyword_logits(texts, queues=QUEUES, scale=1.5):
    """Count keyword hits per queue. No hits at all gives uniform probabilities, hence an abstention."""
    out = np.zeros((len(texts), len(queues)))
    for i, text in enumerate(texts):
        low = text.lower()
        for j, q in enumerate(queues):
            hits = sum(1 for kw in KEYWORDS[q]
                       if re.search(r"(?<![a-z0-9])" + re.escape(kw) + (r"(?![a-z])" if kw[-1].isalpha() else ""), low))
            out[i, j] = scale * hits
    return out


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
    path = Path(path) if path is not None else DEFAULT_OUT
    manifest = path if path.suffix == ".json" else path / "router.json"
    if not manifest.exists():
        warnings.warn(f"{manifest} not found: using the {KEYWORD_NAME}. "
                      "Run the training steps and `python -m solution.calibrate` to create it.", stacklevel=2)
        return keyword_router()
    cfg = json.loads(manifest.read_text())
    kind = model or cfg["default"]
    entry = cfg["models"][kind]
    return Router(kind, _load_model(kind, manifest.parent / kind), entry["temperature"],
                  cfg["threshold"], cfg["queues"])


if __name__ == "__main__":
    import sys
    r = load_router()
    for message in sys.argv[1:] or ["We were charged twice for the Pro plan.", "getting 403", "banana bread recipe?"]:
        res = r.route(message)
        print(f"[{r.name}] {message!r} -> {res['queue']} ({res['confidence']:.2f})"
              f"{'  ABSTAIN: escalate to a human' if res['abstain'] else ''}")
