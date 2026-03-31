"""
SAE feature analysis.

  get_feature_activations()  — run SAE on all activations, return sparse codes
  top_activating_examples()  — for each feature, find top-N (query, doc) pairs
  correlate_with_ir_features() — Pearson r between SAE feature activations and
                                  IR probe targets (from features.parquet)
  analyze_features()         — full pipeline, saves results to JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.sae.model import TopKSAE
from src.utils.logging import get_logger

log = get_logger(__name__)

IR_FEATURE_COLS = [
    "lexical_overlap",
    "query_term_freq",
    "bm25_score",
    "bm25_rank",
    "is_relevant",
    "relevance_label",
    "doc_length_bucket",
]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_feature_activations(
    sae: TopKSAE,
    acts: np.ndarray,
    batch_size: int = 2048,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run SAE encoder on all activations, return sparse codes (N, hidden_dim)."""
    if device is None:
        device = _get_device()
    sae = sae.to(device)
    sae.eval()

    X = torch.from_numpy(acts.astype(np.float32))
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

    all_sparse = []
    with torch.no_grad():
        for (batch_x,) in loader:
            _, sparse = sae(batch_x.to(device))
            all_sparse.append(sparse.cpu().numpy())

    return np.concatenate(all_sparse, axis=0)  # (N, hidden_dim)


def top_activating_examples(
    sparse_codes: np.ndarray,
    pairs_df: pd.DataFrame,
    n: int = 20,
) -> dict[int, list[dict]]:
    """For each SAE feature, find top-N pairs by activation strength.

    Args:
        sparse_codes: (N, hidden_dim) sparse feature activations
        pairs_df:     DataFrame with columns query_id, doc_id (N rows, aligned)
        n:            Number of top examples to keep per feature

    Returns:
        {feature_idx: [{"rank": 1, "query_id": ..., "doc_id": ..., "activation": ...}, ...]}
    """
    N, hidden_dim = sparse_codes.shape
    assert len(pairs_df) == N, f"pairs_df length {len(pairs_df)} != sparse_codes {N}"

    # Only analyse features that ever activate (skip dead features)
    active_mask = (sparse_codes > 0).any(axis=0)
    active_indices = np.where(active_mask)[0]

    log.info(f"Analysing top-{n} examples for {len(active_indices):,} active features "
             f"(of {hidden_dim:,} total)")

    results = {}
    query_ids = pairs_df["query_id"].values
    doc_ids   = pairs_df["doc_id"].values

    for feat_idx in active_indices:
        col = sparse_codes[:, feat_idx]
        top_n_idx = np.argpartition(col, -min(n, N))[-min(n, N):]
        top_n_idx = top_n_idx[np.argsort(col[top_n_idx])[::-1]]

        results[int(feat_idx)] = [
            {
                "rank": int(rank + 1),
                "query_id": str(query_ids[i]),
                "doc_id":   str(doc_ids[i]),
                "activation": float(col[i]),
            }
            for rank, i in enumerate(top_n_idx)
            if col[i] > 0
        ]

    return results


def correlate_with_ir_features(
    sparse_codes: np.ndarray,
    features_df: pd.DataFrame,
    top_k_features: int = 100,
) -> pd.DataFrame:
    """Compute Pearson correlation between each SAE feature activation and IR targets.

    Only computed for the top_k_features most active SAE features (by mean activation)
    to keep runtime manageable.

    Returns:
        DataFrame: rows=SAE features, cols=IR targets, values=Pearson r
    """
    # Select top-k most active features by mean activation
    mean_acts = sparse_codes.mean(axis=0)
    top_feat_idx = np.argsort(mean_acts)[::-1][:top_k_features]

    corr_rows = []
    for feat_idx in top_feat_idx:
        feat_acts = sparse_codes[:, feat_idx].astype(np.float64)
        row = {"feature_idx": int(feat_idx), "mean_activation": float(mean_acts[feat_idx])}
        for col in IR_FEATURE_COLS:
            if col not in features_df.columns:
                continue
            ir_vals = features_df[col].values.astype(np.float64)
            # Pearson r — skip if feature has zero variance
            if feat_acts.std() < 1e-8 or ir_vals.std() < 1e-8:
                row[f"r_{col}"] = 0.0
            else:
                r = np.corrcoef(feat_acts, ir_vals)[0, 1]
                row[f"r_{col}"] = float(r) if np.isfinite(r) else 0.0
        corr_rows.append(row)

    return pd.DataFrame(corr_rows).set_index("feature_idx")


def analyze_features(
    sae: TopKSAE,
    acts: np.ndarray,
    pairs_df: pd.DataFrame,
    features_df: pd.DataFrame,
    layer: int,
    out_dir: str | Path,
    top_n: int = 20,
    top_k_corr: int = 100,
    batch_size: int = 2048,
) -> dict:
    """Full feature analysis pipeline for one SAE.

    Saves:
      {out_dir}/top_examples_layer{layer}.json
      {out_dir}/ir_correlations_layer{layer}.parquet

    Returns:
        dict with summary stats
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Computing sparse codes for layer {layer}...")
    sparse_codes = get_feature_activations(sae, acts, batch_size=batch_size)

    # Top activating examples
    log.info("Finding top activating examples...")
    top_examples = top_activating_examples(sparse_codes, pairs_df, n=top_n)
    with open(out_dir / f"top_examples_layer{layer}.json", "w") as f:
        json.dump(top_examples, f)
    log.info(f"  Saved top examples: {len(top_examples)} active features")

    # IR feature correlations
    log.info("Computing IR feature correlations...")
    corr_df = correlate_with_ir_features(sparse_codes, features_df, top_k_features=top_k_corr)
    corr_df.to_parquet(out_dir / f"ir_correlations_layer{layer}.parquet")
    log.info(f"  Saved correlations for {len(corr_df)} features")

    # Summary: features with highest |r| per IR target
    r_cols = [c for c in corr_df.columns if c.startswith("r_")]
    best_per_target = {}
    for col in r_cols:
        target = col[2:]  # strip "r_"
        best_idx = corr_df[col].abs().idxmax()
        best_per_target[target] = {
            "feature_idx": int(best_idx),
            "pearson_r": round(float(corr_df.loc[best_idx, col]), 4),
        }

    active_count = int((sparse_codes > 0).any(axis=0).sum())
    dead_count   = sae.hidden_dim - active_count

    return {
        "layer": layer,
        "n_active_features": active_count,
        "n_dead_features":   dead_count,
        "dead_feature_pct":  round(100.0 * dead_count / sae.hidden_dim, 2),
        "best_corr_per_target": best_per_target,
    }
