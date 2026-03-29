"""
Build query-document pair tables from a loaded BEIR dataset.

Two modes:
  - "qrels_only"     : pairs from annotated qrels only (Phase 1).
  - "top_k_from_run" : pairs from a BM25 retrieval run DataFrame (Phase 2+).
                       Adds bm25_score / bm25_rank columns; qrel labels joined in.

The output parquet schema is the canonical join key for activations, features,
and probe targets throughout the entire pipeline.

Usage:
    from src.data.pair_builder import build_pairs_from_qrels, build_pairs_from_run
    from src.data.loader import load_beir_dataset

    ds = load_beir_dataset("scifact")
    df = build_pairs_from_qrels(ds, relevance_threshold=1)
    df.to_parquet("data/interim/scifact/query_doc_pairs.parquet", index=False)
"""

from pathlib import Path
from typing import Literal

import pandas as pd

from src.data.schema import BeirDataset, DatasetName
from src.utils.io import save_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)

PairMode = Literal["qrels_only", "top_k_from_run"]

# Canonical column order — all downstream modules rely on this
_COLUMNS_QRELS = [
    "query_id",
    "doc_id",
    "query_text",
    "doc_title",
    "doc_text",
    "relevance_label",
    "is_relevant",
    "dataset",
]

_COLUMNS_RUN = _COLUMNS_QRELS + ["bm25_score", "bm25_rank"]


def build_pairs_from_qrels(
    dataset: BeirDataset,
    relevance_threshold: int = 1,
) -> pd.DataFrame:
    """Build a pair table from annotated qrels (Phase 1).

    Only (query, doc) pairs that appear in the qrels are included.
    Docs not in the corpus are silently skipped with a warning.

    Args:
        dataset: Loaded BeirDataset.
        relevance_threshold: Labels >= this value are marked is_relevant=True.

    Returns:
        DataFrame with columns defined in _COLUMNS_QRELS.
    """
    rows = []
    skipped = 0

    for query_id, doc_labels in dataset.qrels.items():
        query_text = dataset.queries.get(query_id, "")
        if not query_text:
            continue

        for doc_id, label in doc_labels.items():
            doc = dataset.corpus.get(doc_id)
            if doc is None:
                skipped += 1
                continue

            rows.append({
                "query_id": str(query_id),
                "doc_id": str(doc_id),
                "query_text": query_text,
                "doc_title": doc.get("title", ""),
                "doc_text": doc.get("text", ""),
                "relevance_label": int(label),
                "is_relevant": int(label) >= relevance_threshold,
                "dataset": dataset.name,
            })

    if skipped:
        log.warning(f"Skipped {skipped} qrel entries: doc_id not found in corpus.")

    df = pd.DataFrame(rows, columns=_COLUMNS_QRELS)
    log.info(
        f"Built qrels-only pair table: {len(df)} pairs "
        f"({df['is_relevant'].sum()} relevant, {(~df['is_relevant']).sum()} non-relevant)"
    )
    return df


def build_pairs_from_run(
    dataset: BeirDataset,
    run_df: pd.DataFrame,
    relevance_threshold: int = 1,
) -> pd.DataFrame:
    """Build a pair table from a BM25 retrieval run (Phase 2+).

    Includes every (query_id, doc_id) in run_df. Relevance labels are joined
    from qrels; pairs absent from qrels get relevance_label=0.

    run_df must have columns: query_id, doc_id, bm25_score, bm25_rank.

    Args:
        dataset: Loaded BeirDataset.
        run_df: BM25 retrieval results DataFrame.
        relevance_threshold: Labels >= this value are marked is_relevant=True.

    Returns:
        DataFrame with columns defined in _COLUMNS_RUN.
    """
    # Flatten qrels into a lookup: (query_id, doc_id) -> label
    qrel_lookup: dict[tuple[str, str], int] = {
        (str(qid), str(did)): label
        for qid, docs in dataset.qrels.items()
        for did, label in docs.items()
    }

    rows = []
    skipped = 0

    for _, row in run_df.iterrows():
        query_id = str(row["query_id"])
        doc_id = str(row["doc_id"])

        query_text = dataset.queries.get(query_id, "")
        doc = dataset.corpus.get(doc_id)

        if not query_text or doc is None:
            skipped += 1
            continue

        label = qrel_lookup.get((query_id, doc_id), 0)

        rows.append({
            "query_id": query_id,
            "doc_id": doc_id,
            "query_text": query_text,
            "doc_title": doc.get("title", ""),
            "doc_text": doc.get("text", ""),
            "relevance_label": label,
            "is_relevant": label >= relevance_threshold,
            "dataset": dataset.name,
            "bm25_score": float(row["bm25_score"]),
            "bm25_rank": int(row["bm25_rank"]),
        })

    if skipped:
        log.warning(f"Skipped {skipped} run entries: query or doc not found.")

    df = pd.DataFrame(rows, columns=_COLUMNS_RUN)
    log.info(
        f"Built run-based pair table: {len(df)} pairs "
        f"({df['is_relevant'].sum()} relevant, {(~df['is_relevant']).sum()} non-relevant)"
    )
    return df


def save_pairs(df: pd.DataFrame, dataset_name: DatasetName, interim_dir: str | Path = "data/interim") -> Path:
    """Save pair table to data/interim/{dataset_name}/query_doc_pairs.parquet."""
    out_path = Path(interim_dir) / dataset_name / "query_doc_pairs.parquet"
    save_parquet(df, out_path)
    log.info(f"Saved pair table to {out_path}")
    return out_path
