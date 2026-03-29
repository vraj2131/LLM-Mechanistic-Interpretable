"""
Extract residual stream activations at the decision token from all 28 layers.

Strategy:
  - Call model.model(...) (backbone only — no lm_head, no logits computed)
  - All 28 hooks fire in one forward pass per batch
  - Each hook immediately slices [:, -1, :].half().cpu() — minimal memory
  - Accumulate per-layer lists, then stack and save as .npy (float16)
  - Write manifest.json with (query_id, doc_id) ordering + metadata

This runs ~3-4x faster than Phase 3 because:
  1. No logits (152K-dim vocab head skipped entirely)
  2. Larger batch size possible (no logits OOM)

Usage (CLI):
    python -m src.activations.extractor --dataset scifact
    python -m src.activations.extractor --dataset nfcorpus

Usage (Python):
    from src.activations.extractor import run_extraction
    run_extraction("scifact")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.activations.hooks import register_hooks, remove_hooks
from src.activations.token_position import get_decision_token_pos
from src.reranking.prompt_builder import build_prompts_for_pairs
from src.reranking.qwen_inference import load_model
from src.utils.config import load_config
from src.utils.io import load_parquet
from src.utils.logging import get_logger
from src.utils.reproducibility import set_all_seeds

log = get_logger(__name__)

_MANIFEST_FILE = "manifest.json"


def run_extraction(
    dataset_name: str,
    data_root: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
    cache_root: str | Path = "data/caches",
    batch_size: int = 32,
    model=None,
    tokenizer=None,
    force: bool = False,
) -> Path:
    """Extract activations for all pairs in a dataset.

    Args:
        dataset_name: "scifact" or "nfcorpus".
        data_root: Path to raw BEIR data (for prompt building).
        interim_dir: Path to query_doc_pairs.parquet.
        cache_root: Root dir for activation caches.
        batch_size: Pairs per forward pass. 32 works on M3 Pro (no logits OOM).
        model: Pre-loaded model. If None, loads Qwen automatically.
        tokenizer: Pre-loaded tokenizer.
        force: Re-extract even if cache already exists.

    Returns:
        cache_dir: Path to the directory containing layer_*.npy + manifest.json.
    """
    set_all_seeds(42)

    cache_dir = Path(cache_root) / "activations" / dataset_name
    manifest_path = cache_dir / _MANIFEST_FILE

    # Check cache
    if not force and manifest_path.exists():
        log.info(f"Activation cache already exists at {cache_dir}. Use force=True to re-extract.")
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs
    pairs_path = Path(interim_dir) / dataset_name / "query_doc_pairs.parquet"
    pairs_df = load_parquet(pairs_path)
    log.info(f"Loaded {len(pairs_df)} pairs from {pairs_path}")

    # Load model if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    log.info(f"Model: {n_layers} layers, hidden_dim={hidden_dim}")

    # Build prompts
    cfg = load_config("configs/reranker.yaml")
    prompts = build_prompts_for_pairs(pairs_df, tokenizer, cfg=cfg)
    n_pairs = len(prompts)

    # Accumulate activations per layer: list of (batch, hidden_dim) tensors
    layer_buffers: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    pair_order: list[tuple[str, str]] = []

    device = next(model.parameters()).device
    is_mps = device.type == "mps"

    # Register all hooks once before the loop
    storage, handles = register_hooks(model, layers=range(n_layers), token_pos=-1)

    try:
        for i in tqdm(range(0, n_pairs, batch_size), desc="Extracting activations", unit="batch"):
            batch_prompts = prompts[i : i + batch_size]
            batch_rows = pairs_df.iloc[i : i + batch_size]

            # Track (query_id, doc_id) ordering
            pair_order.extend(zip(batch_rows["query_id"], batch_rows["doc_id"]))

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=900,
            ).to(device)

            # Forward through backbone only — no logits, no lm_head
            with torch.no_grad():
                model.model(**enc)

            # Collect from hooks (already sliced to decision token, float16, CPU)
            for layer_idx in range(n_layers):
                layer_buffers[layer_idx].append(storage[layer_idx].clone())
                del storage[layer_idx]  # free immediately

            if is_mps:
                torch.mps.empty_cache()

    finally:
        remove_hooks(handles)

    # Stack and save each layer
    log.info(f"Saving {n_layers} activation matrices to {cache_dir} ...")
    for layer_idx in range(n_layers):
        arr = torch.cat(layer_buffers[layer_idx], dim=0).numpy()  # (n_pairs, hidden_dim)
        assert arr.shape == (n_pairs, hidden_dim), \
            f"Layer {layer_idx} shape mismatch: {arr.shape}"
        np.save(cache_dir / f"layer_{layer_idx}.npy", arr)
        del layer_buffers[layer_idx]  # free as we go

    # Write manifest
    manifest = {
        "model_id": model.config._name_or_path,
        "dataset": dataset_name,
        "n_pairs": n_pairs,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "dtype": "float16",
        "decision_token_strategy": "last_input",
        "pairs": [{"query_id": qid, "doc_id": did} for qid, did in pair_order],
        "layer_files": {str(i): f"layer_{i}.npy" for i in range(n_layers)},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_mb = sum(
        (cache_dir / f"layer_{i}.npy").stat().st_size for i in range(n_layers)
    ) / 1e6
    log.info(f"Extraction complete: {n_pairs} pairs × {n_layers} layers — {total_mb:.0f} MB total")
    return cache_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract activations for a BEIR dataset.")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache_root", default="data/caches")
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_extraction(
        dataset_name=args.dataset,
        cache_root=args.cache_root,
        interim_dir=args.interim_dir,
        batch_size=args.batch_size,
        force=args.force,
    )
