"""
End-to-end pointwise LLM reranking pipeline.

Steps:
  1. Load BM25 candidate pairs from data/interim/{dataset}/query_doc_pairs.parquet
  2. Build prompts for every (query, doc) pair
  3. Run single-forward-pass logit-based scoring (no generation)
  4. Re-rank candidates per query by expected LLM score
  5. Evaluate nDCG@10 / MRR@10 / Recall@20 vs BM25 baseline
  6. Save reranker_scores.parquet to data/processed/{dataset}/

Usage (CLI):
    python -m src.reranking.evaluate_reranker --dataset scifact

Usage (Python):
    from src.reranking.evaluate_reranker import run_reranker_pipeline
    metrics, scores_df = run_reranker_pipeline("scifact")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_beir_dataset
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.results_table import print_metrics
from src.reranking.prompt_builder import build_prompts_for_pairs
from src.reranking.qwen_inference import load_model, score_pairs
from src.utils.config import load_config
from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger
from src.utils.reproducibility import set_all_seeds

log = get_logger(__name__)


def _build_run_from_scores_df(
    scores_df: pd.DataFrame, score_col: str = "reranker_score",
) -> dict[str, dict[str, float]]:
    """Convert scores DataFrame to run dict {qid: {doc_id: score}}."""
    run: dict[str, dict[str, float]] = {}
    for row in scores_df.itertuples(index=False):
        run.setdefault(row.query_id, {})[row.doc_id] = float(getattr(row, score_col))
    return run


def run_reranker_pipeline(
    dataset_name: str,
    data_root: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
    processed_dir: str | Path = "data/processed",
    batch_size: int | None = None,
    variant: str | None = None,
    model=None,
    tokenizer=None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run the full pointwise LLM reranking pipeline for one dataset.

    Args:
        dataset_name: "scifact" or "nfcorpus".
        data_root: Path to raw BEIR data.
        interim_dir: Path to interim parquets (BM25 pair table).
        processed_dir: Path to write reranker_scores.parquet.
        batch_size: Inference batch size. None uses configs/reranker.yaml default.
        variant: Optional prompt variant name for Phase 9 experiments.
        model: Pre-loaded model (avoids re-loading when called multiple times).
        tokenizer: Pre-loaded tokenizer.

    Returns:
        metrics: dict with ndcg@10, mrr@10, recall@20
        scores_df: DataFrame with query_id, doc_id, reranker_score,
                   reranker_expected_score, reranker_rank, prob_0..prob_3
    """
    set_all_seeds(42)
    cfg = load_config("configs/reranker.yaml")
    batch_size = batch_size or cfg.batch_size

    # 1. Load BM25 candidate pairs
    pairs_path = Path(interim_dir) / dataset_name / "query_doc_pairs.parquet"
    pairs_df = load_parquet(pairs_path)
    log.info(f"Loaded {len(pairs_df)} candidate pairs from {pairs_path}")

    # 2. Load dataset for qrels
    dataset = load_beir_dataset(dataset_name, data_root=data_root)

    # 3. Load model if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model(cfg=cfg)

    # 4. Build prompts
    prompts = build_prompts_for_pairs(pairs_df, tokenizer, variant=variant, cfg=cfg)

    # 5. Run logit-based scoring (single forward pass, no generation)
    result = score_pairs(prompts, model, tokenizer, batch_size=batch_size)

    # 6. Build scores DataFrame
    scores_df = pairs_df[["query_id", "doc_id"]].copy()
    scores_df["reranker_score"] = result["scores"]              # discrete 0-3
    scores_df["reranker_expected_score"] = result["expected_scores"]  # continuous

    # Store full probability distribution for analysis
    probs_arr = np.array(result["probs"])
    for k in range(4):
        scores_df[f"prob_{k}"] = probs_arr[:, k]

    # Rank by expected score (continuous — better tie-breaking than discrete)
    scores_df["reranker_rank"] = (
        scores_df.groupby("query_id")["reranker_expected_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Log score distribution
    score_counts = scores_df["reranker_score"].value_counts().sort_index()
    log.info(f"Score distribution: {dict(score_counts)}")

    # 7. Evaluate using expected score (continuous) for ranking
    run = _build_run_from_scores_df(scores_df, score_col="reranker_expected_score")
    metrics = compute_all_metrics(run, dataset.qrels, relevance_threshold=1)

    # Also compute BM25 baseline for comparison
    bm25_run: dict[str, dict[str, float]] = {}
    for row in pairs_df.itertuples(index=False):
        bm25_run.setdefault(row.query_id, {})[row.doc_id] = float(row.bm25_score)
    bm25_metrics = compute_all_metrics(bm25_run, dataset.qrels, relevance_threshold=1)

    print_metrics(
        {
            f"BM25 baseline": bm25_metrics,
            f"Qwen reranker (variant={variant!r})": metrics,
        },
        title=f"{dataset_name} — Reranker vs BM25",
    )

    # 8. Save
    out_path = Path(processed_dir) / dataset_name / "reranker_scores.parquet"
    save_parquet(scores_df, out_path)
    log.info(f"Saved reranker scores -> {out_path}")

    return metrics, scores_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pointwise LLM reranker.")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus"])
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--variant", default=None,
                        choices=["no_rubric", "scale_0_10", "flipped_order"])
    args = parser.parse_args()

    run_reranker_pipeline(
        dataset_name=args.dataset,
        data_root=args.data_root,
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        variant=args.variant,
    )
