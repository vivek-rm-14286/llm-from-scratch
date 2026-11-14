"""FastAPI service: POST /ingest (restricted keys only) and POST /ask.

    uvicorn app:app --port 8080
    curl -s localhost:8080/ask -H 'X-API-Key: support-demo-key' -H 'Content-Type: application/json' \
         -d '{"question": "Agents in our EU workspace log NM-1001"}'
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import generate
import settings
from ingest import ingest, load_embedder
from retrieve import MODES, Retriever

STATE: dict = {}
LOCK = threading.Lock()  # Qdrant local mode + BM25 rebuild are not meant for concurrent mutation


@asynccontextmanager
async def lifespan(_: FastAPI):
    client = settings.get_qdrant_client()
    embedder = load_embedder()
    if not client.collection_exists(settings.COLLECTION) or not settings.CHUNKS_PATH.exists():
        print("no index found, ingesting", settings.DATA_DIR)
        ingest(settings.DATA_DIR, client=client, embedder=embedder)
    STATE["retriever"] = Retriever(client=client, embedder=embedder)
    STATE["llm"] = generate.get_llm()
    yield
    client.close()


app = FastAPI(title="Nimbus Support Copilot", lifespan=lifespan)


def caller_access(x_api_key: str | None = Header(default=None)) -> str:
    """Identity comes from the credential, never from the request body."""
    if not x_api_key or x_api_key not in settings.API_KEYS:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
    return settings.API_KEYS[x_api_key]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    mode: str = Field(default="hybrid_rerank", pattern="^(" + "|".join(MODES) + ")$")


@app.get("/health")
def health():
    return {"ok": True, "chunks": len(STATE["retriever"].chunks) if "retriever" in STATE else 0}


@app.post("/ingest")
def ingest_endpoint(access: str = Depends(caller_access)):
    if access != "restricted":
        raise HTTPException(status_code=403, detail="ingest requires a restricted-level key")
    with LOCK:
        r: Retriever = STATE["retriever"]
        stats = ingest(settings.DATA_DIR, client=r.client, embedder=r.embedder)
        r.reload()
    return stats


@app.post("/ask")
def ask(req: AskRequest, access: str = Depends(caller_access)):
    r: Retriever = STATE["retriever"]
    t0 = time.perf_counter()
    with LOCK:
        chunks = r.search(req.question, access=access, k=req.top_k, mode=req.mode)
    t1 = time.perf_counter()
    result = generate.answer(req.question, chunks, llm=STATE["llm"])
    t2 = time.perf_counter()
    return {
        **result,
        "access": access,
        "retrieved": [{"chunk_id": c["chunk_id"], "score": c.get("rerank_score", c.get("score"))}
                      for c in chunks],
        "latency_ms": {"retrieve": round((t1 - t0) * 1000), "generate": round((t2 - t1) * 1000)},
    }
