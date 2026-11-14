#!/usr/bin/env python3
"""Ingest markdown docs + resolved tickets: parse -> chunk -> embed -> index in Qdrant.

    python ingest.py                      # indexes sample_data/ into ./index
    NIMBUS_DATA_DIR=/path/to/data python ingest.py

Data layout:
    <data>/docs/*.md        markdown with a small front matter block (title, access, url, updated)
    <data>/tickets.jsonl    {"id", "title", "problem", "resolution", "access", "resolved_at", ...}
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import settings

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------
def parse_front_matter(text: str) -> tuple[dict, str]:
    """Minimal 'key: value' front matter between --- lines (avoids a YAML dependency)."""
    meta: dict[str, str] = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            text = text[end + 4 :]
    return meta, text.lstrip("\n")


def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for path in sorted((data_dir / "docs").glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        access = meta.get("access", "internal")  # fail closed: unlabeled docs are not public
        if access not in settings.ACCESS_LEVELS:
            raise ValueError(f"{path}: unknown access level {access!r}")
        docs.append({
            "doc_id": f"doc:{path.stem}", "doc_type": "doc", "title": meta.get("title", path.stem),
            "url": meta.get("url", ""), "access": access, "updated": meta.get("updated", ""), "body": body,
        })
    tickets_path = data_dir / "tickets.jsonl"
    if tickets_path.exists():
        for line in tickets_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            access = t.get("access", "internal")
            if access not in settings.ACCESS_LEVELS:
                raise ValueError(f"ticket {t['id']}: unknown access level {access!r}")
            docs.append({
                "doc_id": f"ticket:{t['id']}", "doc_type": "ticket", "title": f"{t['id']}: {t['title']}",
                "url": "", "access": access, "updated": t.get("resolved_at", ""),
                "body": f"Problem: {t['problem']}\nResolution: {t['resolution']}",
            })
    return docs


# ---------------------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------------------
def split_sections(markdown: str, title: str) -> list[tuple[str, str]]:
    """Return (heading_path, section_text) pairs; ignores '#' lines inside code fences."""
    sections, path, buf, in_code = [], [title], [], False

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append((" > ".join(path), text))
        buf.clear()

    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            # The H1 usually repeats the title; keep the path as [title, H2, H3, ...].
            path = [title] if level == 1 else path[: level - 1] + [heading]
            continue
        buf.append(line)
    flush()
    return sections


def pack_paragraphs(text: str, max_chars: int, overlap: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > max_chars:  # hard-split giant paragraphs
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:max_chars])
            p = p[max_chars - overlap :]
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p if overlap else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(doc: dict, max_chars: int = 1000, overlap: int = 150) -> list[dict]:
    if doc["doc_type"] == "ticket":  # tickets are small and atomic: one chunk each
        pieces = [(doc["title"], doc["body"])]
    else:
        pieces = [(h, c) for h, s in split_sections(doc["body"], doc["title"])
                  for c in pack_paragraphs(s, max_chars, overlap)]
    chunks = []
    for i, (heading, text) in enumerate(pieces):
        chunk_id = f"{doc['doc_id']}#{i}"
        chunks.append({
            "point_id": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),  # deterministic -> idempotent upserts
            "chunk_id": chunk_id,
            "doc_id": doc["doc_id"],
            "doc_type": doc["doc_type"],
            "title": doc["title"],
            "section": heading,
            "url": doc["url"],
            "updated": doc["updated"],
            "access": doc["access"],
            "access_level": settings.ACCESS_LEVELS[doc["access"]],
            # The heading path is embedded with the text so the chunk carries its own context.
            "text": f"{heading}\n\n{text}",
        })
    return chunks


# ---------------------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------------------
def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.EMBED_MODEL)


def ingest(data_dir: Path | None = None, client=None, embedder=None) -> dict:
    from qdrant_client import models

    data_dir = Path(data_dir or settings.DATA_DIR)
    client = client or settings.get_qdrant_client()
    embedder = embedder or load_embedder()

    docs = load_documents(data_dir)
    chunks = [c for d in docs for c in chunk_document(d)]
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                              batch_size=32, show_progress_bar=False)

    # Full rebuild keeps the example simple and guarantees deleted docs disappear.
    # For large corpora: upsert changed docs and delete by doc_id filter instead.
    if client.collection_exists(settings.COLLECTION):
        client.delete_collection(settings.COLLECTION)
    client.create_collection(
        settings.COLLECTION,
        vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
    )
    if settings.QDRANT_URL:  # payload indexes speed up filtered search on a Qdrant server
        client.create_payload_index(settings.COLLECTION, "access_level", models.PayloadSchemaType.INTEGER)
    points = [
        models.PointStruct(id=c["point_id"], vector=v.tolist(), payload=c)
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(points), 128):
        client.upsert(settings.COLLECTION, points=points[i : i + 128])

    settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    stats = {"documents": len(docs), "chunks": len(chunks), "embed_model": settings.EMBED_MODEL,
             "collection": settings.COLLECTION}
    with open(settings.INDEX_DIR / "index_meta.json", "w") as f:
        json.dump(stats, f, indent=2)  # store the model name with the index
    return stats


if __name__ == "__main__":
    print(json.dumps(ingest(), indent=2))
