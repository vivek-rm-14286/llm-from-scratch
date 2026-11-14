"""Configuration for the Support Engineer Copilot. Every value can be overridden by an env var."""
from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("NIMBUS_DATA_DIR", BASE_DIR / "sample_data"))
INDEX_DIR = Path(os.getenv("NIMBUS_INDEX_DIR", BASE_DIR / "index"))
QDRANT_PATH = INDEX_DIR / "qdrant"          # Qdrant local mode (embedded, no server)
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"    # chunk texts + metadata, used for BM25
QDRANT_URL = os.getenv("NIMBUS_QDRANT_URL")  # e.g. http://localhost:6333 to use a Qdrant server
COLLECTION = os.getenv("NIMBUS_COLLECTION", "nimbus_support")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# BGE v1.5 recommends this instruction on short retrieval queries (not on passages).
QUERY_INSTRUCTION = os.getenv("QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: ")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # or BAAI/bge-reranker-base

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")  # Ollama default
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed-for-local")

# Guardrails (guardrails.py): on by default; NIMBUS_GUARDRAILS=0 turns them off for comparisons.
GUARDRAILS = os.getenv("NIMBUS_GUARDRAILS", "1") != "0"
# Links and email addresses in an answer survive only if their host is one of these (or a subdomain).
ALLOWED_LINK_DOMAINS = os.getenv(
    "NIMBUS_ALLOWED_LINK_DOMAINS", "nimbus-monitor.example,nimbus-internal.example,proxy.internal").split(",")

ACCESS_LEVELS = {"public": 0, "internal": 1, "restricted": 2}

# Demo identity store: API key -> access level name. In production this comes from your
# identity provider (OIDC/SAML groups), never from the request body.
API_KEYS: dict[str, str] = json.loads(os.getenv(
    "NIMBUS_API_KEYS",
    '{"public-demo-key": "public", "support-demo-key": "internal", "security-demo-key": "restricted"}',
))


def get_qdrant_client():
    from qdrant_client import QdrantClient

    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=os.getenv("NIMBUS_QDRANT_API_KEY"))
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))  # only one process may open a local path at a time
