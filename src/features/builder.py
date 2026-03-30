"""
Feature pipeline — computes all 7 probe targets and saves features.parquet.

Probe targets (from configs/features.yaml):
  lexical_overlap     — continuous, computed here
  query_term_freq     — continuous, computed here
  doc_length_bucket   — categorical 0-3, computed here
  doc_length          — continuous, computed here
  bm25_score          — continuous, carried from pairs parquet
  bm25_rank           — continuous, carried from pairs parquet
  is_relevant         — binary, carried from pairs parquet
  relevance_label     — continuous, carried from pairs parquet

Row order in features.parquet matches query_doc_pairs.parquet exactly,
which in turn matches manifest.json — so Phase 6 can zip without merging.

Usage (CLI):
    python -m src.features.builder --datasets scifact nfcorpus

Usage (Python):
    from src.features.builder import build_features
    features_df = build_features("scifact")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.features.document import compute_document_features, fit_boundaries
from src.features.lexical import compute_lexical_features
from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)

# Columns carried directly from pairs parquet (no recomputation needed)
_CARRY_COLS = ["query_id", "doc_id", "bm25_score", "bm25_rank", "is_relevant", "relevance_label"]

# Path to store fitted boundaries so NFCorpus reuses SciFact's quartiles
_BOUNDARIES_FILE = "data/interim/scifact/doc_length_boundaries.json"


def _save_boundaries(boundaries: list[float]) -> None:
    path = Path(_BOUNDARIES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"q1": boundaries[0], "q2": boundaries[1], "q3": boundaries[2]}, f)


def _load_boundaries() -> list[float] | None:
    path = Path(_BOUNDARIES_FILE)
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return [d["q1"], d["q2"], d["q3"]]


def build_features(
    dataset_name: str,
    interim_dir: str | Path = "data/interim",
    primary_dataset: str = "scifact",
) -> pd.DataFrame:
    """Compute all features for a dataset and save to features.parquet.

    Args:
        dataset_name: "scifact" or "nfcorpus".
        interim_dir: Directory containing query_doc_pairs.parquet.
        primary_dataset: The primary dataset whose boundaries are used for
                         doc_length_bucket (ensures consistent thresholds).

    Returns:
        features_df: DataFrame with all probe targets, same row order as pairs.
    """
    pairs_path = Path(interim_dir) / dataset_name / "query_doc_pairs.parquet"
    pairs_df = load_parquet(pairs_path)
    log.info(f"Building features for {dataset_name}: {len(pairs_df)} pairs")

    # --- Lexical features ---
    lex = compute_lexical_features(pairs_df)

    # --- Document features ---
    # Fit boundaries on primary dataset; reuse for OOD dataset
    if dataset_name == primary_dataset:
        boundaries = fit_boundaries(pairs_df)
        _save_boundaries(boundaries)
        log.info(f"Fitted doc_length boundaries: Q1={boundaries[0]:.0f}, Q2={boundaries[1]:.0f}, Q3={boundaries[2]:.0f}")
    else:
        boundaries = _load_boundaries()
        if boundaries is None:
            log.warning(
                f"No saved boundaries found — fitting on {dataset_name} itself. "
                f"Run {primary_dataset} first for consistent bucketing."
            )
        else:
            log.info(f"Loaded doc_length boundaries from {primary_dataset}: {boundaries}")

    doc = compute_document_features(pairs_df, boundaries=boundaries)

    # --- Assemble DataFrame (preserving row order) ---
    features_df = pairs_df[_CARRY_COLS].copy()
    features_df["lexical_overlap"]   = lex["lexical_overlap"]
    features_df["query_term_freq"]   = lex["query_term_freq"]
    features_df["doc_length"]        = doc["doc_length"]
    features_df["doc_length_bucket"] = doc["doc_length_bucket"]

    # Log basic stats
    log.info(
        f"Features built — "
        f"lexical_overlap: mean={features_df['lexical_overlap'].mean():.3f}, "
        f"query_term_freq: mean={features_df['query_term_freq'].mean():.4f}, "
        f"doc_length: mean={features_df['doc_length'].mean():.0f}"
    )

    # Save
    out_path = Path(interim_dir) / dataset_name / "features.parquet"
    save_parquet(features_df, out_path)
    log.info(f"Saved features -> {out_path}")

    return features_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build probe target features.")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    parser.add_argument("--interim_dir", default="data/interim")
    args = parser.parse_args()

    # Always build primary dataset first so boundaries are saved
    datasets = args.datasets
    if "scifact" in datasets:
        datasets = ["scifact"] + [d for d in datasets if d != "scifact"]

    for name in datasets:
        build_features(name, interim_dir=args.interim_dir)
