"""
Ranking evaluation metrics: nDCG@k, MRR@k, Recall@k.

All functions operate on a run dict and a qrels dict using the same
BEIR-style schema so they can be reused across phases 2, 3, and 8.

run   : dict[query_id, dict[doc_id, score]]   (higher score = higher rank)
qrels : dict[query_id, dict[doc_id, label]]   (label >= 1 = relevant)
"""

import math
from typing import Any


def ndcg_at_k(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
    relevance_threshold: int = 1,
) -> float:
    """Mean nDCG@k over all queries that have at least one relevant document."""
    scores = []
    for qid, doc_scores in run.items():
        relevant = {
            did: label
            for did, label in qrels.get(qid, {}).items()
            if label >= relevance_threshold
        }
        if not relevant:
            continue

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # DCG
        dcg = sum(
            _gain(relevant.get(did, 0)) / math.log2(rank + 2)
            for rank, (did, _) in enumerate(ranked)
        )

        # Ideal DCG
        ideal_gains = sorted(
            [_gain(label) for label in relevant.values()], reverse=True
        )[:k]
        idcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal_gains))

        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(sum(scores) / len(scores)) if scores else 0.0


def mrr_at_k(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
    relevance_threshold: int = 1,
) -> float:
    """Mean Reciprocal Rank @k."""
    scores = []
    for qid, doc_scores in run.items():
        relevant = {
            did for did, label in qrels.get(qid, {}).items()
            if label >= relevance_threshold
        }
        if not relevant:
            continue

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        rr = 0.0
        for rank, (did, _) in enumerate(ranked):
            if did in relevant:
                rr = 1.0 / (rank + 1)
                break

        scores.append(rr)

    return float(sum(scores) / len(scores)) if scores else 0.0


def recall_at_k(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k: int = 20,
    relevance_threshold: int = 1,
) -> float:
    """Mean Recall@k — fraction of relevant docs retrieved in top-k."""
    scores = []
    for qid, doc_scores in run.items():
        relevant = {
            did for did, label in qrels.get(qid, {}).items()
            if label >= relevance_threshold
        }
        if not relevant:
            continue

        ranked_ids = {
            did for did, _ in
            sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        }
        scores.append(len(relevant & ranked_ids) / len(relevant))

    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_all_metrics(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    relevance_threshold: int = 1,
) -> dict[str, float]:
    """Compute the full standard metric set used across all phases."""
    return {
        "ndcg@10":   ndcg_at_k(run, qrels, k=10, relevance_threshold=relevance_threshold),
        "mrr@10":    mrr_at_k(run, qrels, k=10, relevance_threshold=relevance_threshold),
        "recall@20": recall_at_k(run, qrels, k=20, relevance_threshold=relevance_threshold),
    }


def _gain(label: int) -> float:
    """Binary gain: any label >= 1 gives gain 1.0 (standard BEIR nDCG)."""
    return 1.0 if label >= 1 else 0.0
