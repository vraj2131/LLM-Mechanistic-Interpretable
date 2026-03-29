"""
Run BM25 queries and produce retrieval runs.

Output formats:
  - run dict  : {query_id: {doc_id: score}}  (for metric computation)
  - DataFrame : query_id, doc_id, bm25_score, bm25_rank  (for parquet storage)

Usage:
    from src.retrieval.bm25_retriever import retrieve
    run, run_df = retrieve(index, doc_ids, queries, top_k=20)
"""

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.retrieval.bm25_index import _tokenize
from src.utils.logging import get_logger

log = get_logger(__name__)


def retrieve(
    index: BM25Okapi,
    doc_ids: list[str],
    queries: dict[str, str],
    top_k: int = 20,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Retrieve top-k documents for each query using BM25.

    Args:
        index: Built BM25Okapi index.
        doc_ids: Ordered list of doc IDs (must match index build order).
        queries: {query_id: query_text}
        top_k: Number of candidates to retrieve per query.

    Returns:
        run: {query_id: {doc_id: bm25_score}}
        run_df: DataFrame with columns query_id, doc_id, bm25_score, bm25_rank
    """
    run: dict[str, dict[str, float]] = {}
    rows = []

    doc_ids_arr = np.array(doc_ids)

    for qid, qtext in tqdm(queries.items(), desc="BM25 retrieval", unit="query"):
        tokens = _tokenize(qtext)
        scores = index.get_scores(tokens)          # shape: (n_docs,)

        # Partial sort — only need top_k
        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        top_doc_ids = doc_ids_arr[top_indices]
        top_scores = scores[top_indices]

        run[qid] = {did: float(score) for did, score in zip(top_doc_ids, top_scores)}

        for rank, (did, score) in enumerate(zip(top_doc_ids, top_scores), start=1):
            rows.append({
                "query_id": qid,
                "doc_id": did,
                "bm25_score": float(score),
                "bm25_rank": rank,
            })

    run_df = pd.DataFrame(rows, columns=["query_id", "doc_id", "bm25_score", "bm25_rank"])
    log.info(
        f"Retrieval complete: {len(run)} queries, "
        f"{len(run_df)} total candidates (top-{top_k})"
    )
    return run, run_df
