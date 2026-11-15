#!/usr/bin/env python3
"""Fine-tune the embedding model on (question, chunk) pairs with hard negatives, and measure dense
recall@k before and after on held-out questions.

    python finetune_embedder.py                        # 3 epochs, a minute or two on a laptop CPU
    python finetune_embedder.py --max-steps 5 --cpu    # smoke test
    EMBED_MODEL=models/bge-small-nimbus python eval.py --reingest   # full four-mode table with the new model

Needs sentence-transformers >= 3.0 (SentenceTransformerTrainer), datasets and accelerate.

Data: the answerable golden questions are split by a fixed seed into a training half and a held-out
half; questions that share a relevant source stay on the same side. The golden set is tiny, so
treat the result as a demonstration of the method. With real data, train on thousands of pairs mined from resolved tickets or search
logs and never on the questions you report results on.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import settings
from eval import load_golden, ranked_keys, recall_at_k, reciprocal_rank
from ingest import chunk_document, load_documents


def split_golden(golden: list[dict], test_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split answerable questions into train and held-out. Questions that share their text or any
    relevant source are kept on the same side; otherwise the model is trained on the exact passage
    it is later tested on and the held-out score is inflated."""
    items = [g for g in golden if g["relevant"]]
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, g in enumerate(items):
        for key in [g["question"], *g["relevant"]]:
            if key in owner:
                parent[find(i)] = find(owner[key])
            owner[key] = i
    groups = defaultdict(list)
    for i, g in enumerate(items):
        groups[find(i)].append(g)
    ordered = [groups[key] for key in sorted(groups)]
    random.Random(seed).shuffle(ordered)
    train, test = [], []
    for group in ordered:  # fill the held-out side first, group by group
        (test if len(test) < test_fraction * len(items) else train).extend(group)
    return train, test


def is_relevant(chunk: dict, relevant: set[str]) -> bool:
    return chunk["section_id"] in relevant or chunk["doc_id"] in relevant


class DenseIndex:
    """Exact cosine search over all chunks with the same access filter as retrieve.py."""

    def __init__(self, model, chunks: list[dict]):
        self.model, self.chunks = model, chunks
        self.vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                                    batch_size=32, show_progress_bar=False)
        self.levels = np.array([c["access_level"] for c in chunks])

    def search(self, question: str, access: str, k: int) -> list[dict]:
        q = self.model.encode(settings.QUERY_INSTRUCTION + question, normalize_embeddings=True)
        scores = self.vectors @ q
        scores[self.levels > settings.ACCESS_LEVELS[access]] = -np.inf
        return [self.chunks[i] for i in np.argsort(-scores)[:k]]

    def evaluate(self, questions: list[dict], k: int) -> dict:
        rk, rr = [], []
        for g in questions:
            rel = set(g["relevant"])
            keys = ranked_keys(self.search(g["question"], g["user_access"], k), rel)
            rk.append(recall_at_k(keys, rel, k))
            rr.append(reciprocal_rank(keys, rel))
        return {"n": len(questions), f"recall@{k}": round(float(np.mean(rk)), 3), "mrr": round(float(np.mean(rr)), 3)}


def build_triplets(index: DenseIndex, train: list[dict], n_negatives: int, pool: int) -> list[dict]:
    """One row per (question, positive chunk, hard negative). Hard negatives are the chunks the
    CURRENT model ranks highest that are not labeled relevant and that the asker may see."""
    rows = []
    for g in train:
        rel = set(g["relevant"])
        candidates = index.search(g["question"], g["user_access"], pool)
        positives = [c for c in index.chunks if is_relevant(c, rel)]
        negatives = [c for c in candidates if not is_relevant(c, rel)][:n_negatives]
        for pos in positives:
            for neg in negatives:
                rows.append({"anchor": settings.QUERY_INSTRUCTION + g["question"],  # same prefix as at query time
                             "positive": pos["text"], "negative": neg["text"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=settings.EMBED_MODEL)
    ap.add_argument("--out", default="models/bge-small-nimbus")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps (smoke test)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--negatives", type=int, default=3, help="hard negatives per question")
    ap.add_argument("--test-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU or MPS is available")
    args = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    try:  # module layout of sentence-transformers 6.x
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:  # 3.x to 5.x
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers

    chunks = [c for d in load_documents(settings.DATA_DIR) for c in chunk_document(d)]
    train, test = split_golden(load_golden(settings.DATA_DIR / "golden.jsonl"), args.test_fraction, args.seed)

    model = SentenceTransformer(args.base, device="cpu" if args.cpu else None)
    index = DenseIndex(model, chunks)
    before = {"train": index.evaluate(train, args.k), "held_out": index.evaluate(test, args.k)}
    print("before:", json.dumps(before))

    rows = build_triplets(index, train, args.negatives, pool=20)
    dataset = Dataset.from_list(rows)  # column order matters: anchor, positive, negative
    print(f"{len(train)} training questions -> {len(rows)} (anchor, positive, negative) rows; "
          f"{len(test)} held-out questions")

    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(len(rows) / args.batch_size * args.epochs)
    warmup_steps = int(0.1 * total_steps)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(args.out).with_name(Path(args.out).name + "-checkpoints")),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,  # 10% of training; an integer works on every transformers version
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # a repeated positive in a batch would act as a false negative
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        use_cpu=args.cpu,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=dataset,
        loss=MultipleNegativesRankingLoss(model),  # InfoNCE: in-batch negatives plus the mined hard negative
    )
    trainer.train()
    model.save_pretrained(args.out)

    index = DenseIndex(model, chunks)  # re-embed every chunk with the tuned model
    after = {"train": index.evaluate(train, args.k), "held_out": index.evaluate(test, args.k)}
    print("after: ", json.dumps(after))
    print(f"saved to {args.out}. Re-index before using it: EMBED_MODEL={args.out} python eval.py --reingest")


if __name__ == "__main__":
    main()
