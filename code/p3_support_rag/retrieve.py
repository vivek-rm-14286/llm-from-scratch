#!/usr/bin/env python3
"""Hybrid retrieval with access control: dense (Qdrant) + BM25 -> RRF -> cross-encoder rerank.

    python retrieve.py "What does NM-4021 mean?" --access internal --mode hybrid_rerank
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import settings

MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
STOPWORDS = set("a an and are as at be but by can do does for from how i if in is it its my not of "
                "on or our so that the their then there this to was we what when where which why "
                "will with you your".split())


def bm25_tokenize(text: str) -> list[str]:
    """Keeps identifiers like 'nm-4021', 'proxy_url', 'agent.yaml' whole AND adds their parts."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        out.append(tok)
        parts = re.split(r"[-_.]", tok)
        if len(parts) > 1:
            out.extend(p for p in parts if p and p not in STOPWORDS)
    return out


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    def __init__(self, client=None, embedder=None, reranker=None):
        from ingest import load_embedder

        self.client = client or settings.get_qdrant_client()
        self.embedder = embedder or load_embedder()
        self._reranker = reranker
        self.reload()

    # Called after every ingest so BM25 sees the same chunks as Qdrant.
    def reload(self) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks: dict[str, dict] = {}
        if settings.CHUNKS_PATH.exists():
            with open(settings.CHUNKS_PATH, encoding="utf-8") as f:
                for line in f:
                    c = json.loads(line)
                    self.chunks[c["chunk_id"]] = c
        self.chunk_ids = list(self.chunks)
        corpus = [bm25_tokenize(self.chunks[cid]["text"]) for cid in self.chunk_ids]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(settings.RERANK_MODEL)
        return self._reranker

    # ---- first-stage retrievers (both enforce the access filter) ------------------------
    def dense(self, query: str, max_level: int, k: int) -> list[str]:
        from qdrant_client import models

        vec = self.embedder.encode(settings.QUERY_INSTRUCTION + query, normalize_embeddings=True)
        hits = self.client.query_points(
            settings.COLLECTION,
            query=vec.tolist(),
            query_filter=models.Filter(must=[
                models.FieldCondition(key="access_level", range=models.Range(lte=max_level)),
            ]),
            limit=k,
            with_payload=["chunk_id"],
        ).points
        return [h.payload["chunk_id"] for h in hits]

    def sparse(self, query: str, max_level: int, k: int) -> list[str]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(bm25_tokenize(query))
        allowed = [(s, cid) for s, cid in zip(scores, self.chunk_ids)
                   if self.chunks[cid]["access_level"] <= max_level and s > 0]
        allowed.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in allowed[:k]]

    # ---- full pipeline --------------------------------------------------------------------
    def search(self, query: str, access: str, k: int = 5, mode: str = "hybrid_rerank",
               candidates: int = 20) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        max_level = settings.ACCESS_LEVELS[access]
        if mode == "dense":
            ranked = [(cid, None) for cid in self.dense(query, max_level, k)]
        elif mode == "sparse":
            ranked = [(cid, None) for cid in self.sparse(query, max_level, k)]
        else:
            fused = reciprocal_rank_fusion([
                self.dense(query, max_level, candidates),
                self.sparse(query, max_level, candidates),
            ])
            ranked = fused[: candidates if mode == "hybrid_rerank" else k]

        results = []
        for cid, score in ranked:
            c = dict(self.chunks[cid])
            if c["access_level"] > max_level:  # defense in depth: should never trigger
                raise RuntimeError(f"ACL violation: {cid} returned for access={access}")
            c["score"] = score
            results.append(c)

        if mode == "hybrid_rerank" and results:
            scores = self.reranker.predict([(query, c["text"]) for c in results])
            for c, s in zip(results, scores):
                c["rerank_score"] = float(s)
            results.sort(key=lambda c: c["rerank_score"], reverse=True)
            results = results[:k]
        return results


def unique_docs(results: list[dict]) -> list[str]:
    """Collapse chunk results to document ids in rank order (for doc-level metrics)."""
    seen, out = set(), []
    for r in results:
        if r["doc_id"] not in seen:
            seen.add(r["doc_id"])
            out.append(r["doc_id"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--access", default="public", choices=list(settings.ACCESS_LEVELS))
    ap.add_argument("--mode", default="hybrid_rerank", choices=MODES)
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    for r in Retriever().search(a.query, a.access, k=a.k, mode=a.mode):
        print(f"{r['chunk_id']:45s} {r['access']:10s} rrf={r['score']} rerank={r.get('rerank_score')}")
