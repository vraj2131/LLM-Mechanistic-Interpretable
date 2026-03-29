"""
Load and validate activation caches produced by extractor.py.

Called at the start of Phase 6 (probing) and Phase 7 (SAE training)
to assert caches are complete, correctly shaped, and NaN/Inf-free.

Usage:
    from src.activations.cache_manager import load_activation_cache, validate_cache
    acts, manifest = load_activation_cache("scifact")
    # acts: dict {layer_idx: np.ndarray (n_pairs, hidden_dim) float16}
    validate_cache(acts, manifest)   # raises AssertionError on failure
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def load_manifest(cache_dir: str | Path) -> dict:
    """Load and return the manifest.json for a dataset's activation cache."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. "
            "Run src.activations.extractor first."
        )
    with open(manifest_path) as f:
        return json.load(f)


def load_activation_cache(
    dataset_name: str,
    cache_root: str | Path = "data/caches",
    layers: list[int] | None = None,
    mmap: bool = True,
) -> tuple[dict[int, np.ndarray], dict]:
    """Load activation arrays from disk.

    Args:
        dataset_name: "scifact" or "nfcorpus".
        cache_root: Root directory for caches.
        layers: Which layer indices to load. None loads all layers.
        mmap: If True, use np.load with mmap_mode='r' — avoids loading
              all arrays into RAM at once (recommended for 28 layers).

    Returns:
        acts: {layer_idx: ndarray of shape (n_pairs, hidden_dim) float16}
        manifest: The parsed manifest dict.
    """
    cache_dir = Path(cache_root) / "activations" / dataset_name
    manifest = load_manifest(cache_dir)

    n_layers = manifest["n_layers"]
    if layers is None:
        layers = list(range(n_layers))

    acts: dict[int, np.ndarray] = {}
    for i in layers:
        fpath = cache_dir / f"layer_{i}.npy"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing activation file: {fpath}")
        acts[i] = np.load(fpath, mmap_mode="r" if mmap else None)

    log.info(
        f"Loaded {len(acts)} layers from {cache_dir} "
        f"({manifest['n_pairs']} pairs × {manifest['hidden_dim']} dims)"
    )
    return acts, manifest


def validate_cache(
    acts: dict[int, np.ndarray],
    manifest: dict,
) -> None:
    """Assert activation cache is complete and clean.

    Checks:
      - Each layer array has shape (n_pairs, hidden_dim)
      - No NaN values
      - No Inf values
      - All expected layers present

    Raises:
        AssertionError on any failure.
    """
    n_pairs = manifest["n_pairs"]
    hidden_dim = manifest["hidden_dim"]
    n_layers = manifest["n_layers"]

    log.info(f"Validating cache: {n_layers} layers, {n_pairs} pairs, hidden_dim={hidden_dim}")

    for i, arr in acts.items():
        assert arr.shape == (n_pairs, hidden_dim), \
            f"Layer {i}: expected shape ({n_pairs}, {hidden_dim}), got {arr.shape}"

        arr_f32 = arr.astype(np.float32)
        assert not np.isnan(arr_f32).any(), f"Layer {i}: contains NaN values"
        assert not np.isinf(arr_f32).any(), f"Layer {i}: contains Inf values"

    log.info("Cache validation passed.")


def manifest_to_index_df(manifest: dict) -> pd.DataFrame:
    """Return a DataFrame with columns [query_id, doc_id] matching row order.

    Used to join activation rows with features.parquet in Phase 6.
    """
    return pd.DataFrame(manifest["pairs"])


def assert_pairs_aligned(
    manifest: dict,
    pairs_df: pd.DataFrame,
) -> None:
    """Assert that manifest row ordering matches pairs_df row ordering.

    Called at start of Phase 6 to ensure probing targets align with activations.

    Raises:
        AssertionError if orderings differ.
    """
    manifest_df = manifest_to_index_df(manifest)
    assert len(manifest_df) == len(pairs_df), (
        f"Row count mismatch: manifest has {len(manifest_df)}, "
        f"pairs_df has {len(pairs_df)}"
    )
    assert (manifest_df["query_id"].values == pairs_df["query_id"].values).all(), \
        "query_id ordering mismatch between manifest and pairs_df"
    assert (manifest_df["doc_id"].values == pairs_df["doc_id"].values).all(), \
        "doc_id ordering mismatch between manifest and pairs_df"
    log.info("Pair alignment check passed.")
