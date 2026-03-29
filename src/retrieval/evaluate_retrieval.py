"""
Full BM25 retrieval pipeline: index → retrieve → evaluate → save outputs.

This is the main entry point for Phase 2. It can be run as a script or
called from the notebook.

Usage (CLI):
    python -m src.retrieval.evaluate_retrieval --dataset scifact --top_k 20

Usage (Python):
    from src.retrieval.evaluate_retrieval import run_retrieval_pipeline
    metrics, run_df = run_retrieval_pipeline("scifact")
"""

import argparse
from pathlib import Path

from src.data.loader import load_beir_dataset
from src.data.pair_builder import build_pairs_from_run, save_pairs
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.results_table import print_metrics
from src.retrieval.bm25_index import build_index
from src.retrieval.bm25_retriever import retrieve
from src.utils.config import load_config
from src.utils.io import save_parquet
from src.utils.logging import get_logger
from src.utils.reproducibility import set_all_seeds

log = get_logger(__name__)


def run_retrieval_pipeline(
    dataset_name: str,
    data_root: str | Path = "data/raw",
    cache_root: str | Path = "data/caches",
    interim_dir: str | Path = "data/interim",
    top_k: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
    force_rebuild: bool = False,
) -> tuple[dict[str, float], object]:
    """End-to-end BM25 retrieval pipeline for one dataset.

    Steps:
      1. Load BEIR dataset
      2. Build (or load) BM25 index
      3. Retrieve top-k candidates for all queries
      4. Evaluate against qrels
      5. Save BM25 run parquet and rebuild working pair table

    Args:
        dataset_name: "scifact" or "nfcorpus".
        data_root: Where raw BEIR data lives.
        cache_root: Where to store BM25 index files.
        interim_dir: Where to write parquet outputs.
        top_k: Candidates to retrieve per query.
        k1: BM25 k1 parameter.
        b: BM25 b parameter.
        force_rebuild: If True, rebuild index even if cached.

    Returns:
        metrics: dict with ndcg@10, mrr@10, recall@20
        run_df: DataFrame of retrieval results
    """
    set_all_seeds(42)

    # 1. Load dataset
    dataset = load_beir_dataset(dataset_name, data_root=data_root)

    # 2. Build / load BM25 index
    index_cache = Path(cache_root) / "bm25_indexes" / dataset_name
    index, doc_ids = build_index(
        dataset.corpus,
        cache_dir=index_cache,
        k1=k1,
        b=b,
        force=force_rebuild,
    )

    # 3. Retrieve
    run, run_df = retrieve(index, doc_ids, dataset.queries, top_k=top_k)

    # 4. Evaluate
    metrics = compute_all_metrics(run, dataset.qrels, relevance_threshold=1)
    print_metrics({f"BM25 (top-{top_k})": metrics}, title=f"{dataset_name} — BM25 Baseline")

    # 5a. Save BM25 run parquet (raw retrieval results)
    run_path = Path(interim_dir) / dataset_name / "bm25_top20.parquet"
    save_parquet(run_df, run_path)
    log.info(f"Saved BM25 run -> {run_path}")

    # 5b. Rebuild working pair table with BM25 top-k candidates
    #     This replaces the qrels-only table from Phase 1.
    #     Non-relevant docs (missing from qrels) get label=0 automatically.
    pair_df = build_pairs_from_run(dataset, run_df, relevance_threshold=1)
    save_pairs(pair_df, dataset_name, interim_dir=interim_dir)
    log.info(
        f"Working pair table updated: {len(pair_df)} pairs "
        f"({pair_df['is_relevant'].sum()} relevant / "
        f"{(~pair_df['is_relevant']).sum()} non-relevant)"
    )

    return metrics, run_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BM25 retrieval baseline.")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus"])
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--cache_root", default="data/caches")
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    run_retrieval_pipeline(
        dataset_name=args.dataset,
        data_root=args.data_root,
        cache_root=args.cache_root,
        interim_dir=args.interim_dir,
        top_k=args.top_k,
        force_rebuild=args.force_rebuild,
    )
