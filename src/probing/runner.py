"""
Probe runner — outer loop over all layers × all targets.

For each (layer, target) pair:
  1. Load activations from cache (mmap — no full RAM load)
  2. Load target column from features.parquet
  3. Stratified train/test split
  4. Train probe with CV hyperparameter selection
  5. Compute bootstrap 95% CI on test scores
  6. Save probe weights as steering direction (.npy) for Phase 8

Results saved to:
  data/processed/{dataset}/probe_results.json   — scores + CIs
  data/processed/{dataset}/probe_weights/        — coef per (layer, target)

Usage (CLI):
    python -m src.probing.runner --dataset scifact
    python -m src.probing.runner --dataset scifact --n_jobs 4

Usage (Python):
    from src.probing.runner import run_probing
    results = run_probing("scifact")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.activations.cache_manager import load_activation_cache, assert_pairs_aligned
from src.probing.bootstrap import bootstrap_ci
from src.probing.probe import train_ridge_probe, train_logistic_probe, ProbeResult
from src.probing.targets import PROBE_TARGETS, ProbeTarget
from src.utils.config import load_config
from src.utils.io import load_parquet, save_npy
from src.utils.logging import get_logger
from src.utils.reproducibility import set_all_seeds

log = get_logger(__name__)


def _result_to_dict(r: ProbeResult, ci_lower: float, ci_upper: float) -> dict:
    return {
        "layer": r.layer,
        "target": r.target,
        "probe_type": r.probe_type,
        "score": round(r.score, 6),
        "cv_score": round(r.cv_score, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "train_size": r.train_size,
        "test_size": r.test_size,
    }


def _probe_one_layer_target(
    layer_idx: int,
    target: ProbeTarget,
    acts: np.ndarray,
    y: np.ndarray,
    cfg,
    rng: np.random.Generator,
    weights_dir: Path,
) -> dict:
    """Train one probe and return its result dict."""
    X = acts.astype(np.float32)

    # Stratified split for classification; random split for regression
    is_clf = target.probe_type == "logistic"
    stratify = y.astype(int) if is_clf else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1 - cfg.train_test_split,
        random_state=42,
        stratify=stratify,
    )

    alpha_grid = list(cfg.ridge.alpha_grid)
    C_grid = list(cfg.logistic.C_grid)
    cv_folds = cfg.cv_folds

    if target.probe_type == "ridge":
        result = train_ridge_probe(
            X_train, y_train, X_test, y_test,
            alpha_grid=alpha_grid, cv_folds=cv_folds,
            layer=layer_idx, target=target.name, rng=rng,
        )
        metric_fn = lambda yt, yp: float(1 - ((yt - yp)**2).sum() / ((yt - yt.mean())**2).sum()
                                        if ((yt - yt.mean())**2).sum() > 0 else 0.0)
        # coef/intercept are in original (unscaled) space — apply directly to X_test
        y_score_test = result.coef @ X_test.T + result.intercept
        ci_lower, ci_upper = bootstrap_ci(y_test, y_score_test, metric_fn,
                                          n=cfg.bootstrap_n, rng=rng)
    else:
        result = train_logistic_probe(
            X_train, y_train, X_test, y_test,
            C_grid=C_grid, cv_folds=cv_folds,
            layer=layer_idx, target=target.name, rng=rng,
        )
        # coef/intercept are in original space; use sigmoid for probabilities
        from scipy.special import expit
        y_proba = expit(X_test @ result.coef + result.intercept)
        ci_lower, ci_upper = bootstrap_ci(
            y_test.astype(int), y_proba, roc_auc_score,
            n=cfg.bootstrap_n, rng=rng,
        )

    # Save probe weights (steering direction for Phase 8)
    weights_path = weights_dir / f"layer_{layer_idx}_{target.name}.npy"
    np.save(weights_path, result.coef)

    return _result_to_dict(result, ci_lower, ci_upper)


def run_probing(
    dataset_name: str,
    interim_dir: str | Path = "data/interim",
    cache_root: str | Path = "data/caches",
    processed_dir: str | Path = "data/processed",
    n_jobs: int = 1,
) -> list[dict]:
    """Run all (layer × target) probes for a dataset.

    Args:
        dataset_name: "scifact" or "nfcorpus".
        interim_dir: Path to features.parquet.
        cache_root: Path to activation caches.
        processed_dir: Where to write probe_results.json + probe weights.
        n_jobs: Parallel jobs for joblib. -1 = all cores.

    Returns:
        List of result dicts (one per layer × target combination).
    """
    set_all_seeds(42)
    cfg = load_config("configs/probing.yaml", "configs/base.yaml")
    rng = np.random.default_rng(42)

    # Load features
    features_df = load_parquet(Path(interim_dir) / dataset_name / "features.parquet")
    log.info(f"Loaded features: {len(features_df)} pairs, columns: {list(features_df.columns)}")

    # Load activation cache manifest + verify alignment
    _, manifest = load_activation_cache(dataset_name, cache_root=cache_root, layers=[0])
    assert_pairs_aligned(manifest, features_df)
    n_layers = manifest["n_layers"]

    # Output dirs
    out_dir = Path(processed_dir) / dataset_name
    weights_dir = out_dir / "probe_weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    n_targets = len(PROBE_TARGETS)
    total = n_layers * n_targets
    done = 0

    for layer_idx in range(n_layers):
        # Load one layer at a time to keep RAM low
        acts_dict, _ = load_activation_cache(
            dataset_name, cache_root=cache_root,
            layers=[layer_idx], mmap=False,
        )
        acts = acts_dict[layer_idx]  # (n_pairs, hidden_dim) float16

        for target in PROBE_TARGETS:
            y_raw = features_df[target.column].values

            # Binarise for logistic targets
            if target.probe_type == "logistic":
                if target.name == "doc_length_bucket":
                    # Multiclass → binary: bucket >= 2 (Long/VeryLong) vs Short/Medium
                    y = (y_raw >= 2).astype(int)
                else:
                    y = y_raw.astype(int)
            else:
                y = y_raw.astype(np.float32)

            result_dict = _probe_one_layer_target(
                layer_idx=layer_idx,
                target=target,
                acts=acts,
                y=y,
                cfg=cfg,
                rng=rng,
                weights_dir=weights_dir,
            )
            all_results.append(result_dict)
            done += 1

            if done % 10 == 0 or done == total:
                log.info(f"  [{done}/{total}] layer={layer_idx} target={target.name} "
                         f"score={result_dict['score']:.4f} "
                         f"CI=[{result_dict['ci_lower']:.4f}, {result_dict['ci_upper']:.4f}]")

    # Save results
    results_path = out_dir / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved {len(all_results)} probe results -> {results_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear probing on activation caches.")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus"])
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--cache_root", default="data/caches")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    run_probing(
        dataset_name=args.dataset,
        interim_dir=args.interim_dir,
        cache_root=args.cache_root,
        processed_dir=args.processed_dir,
        n_jobs=args.n_jobs,
    )
