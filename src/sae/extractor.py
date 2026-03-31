"""
Extract ALL token-position activations for SAE training.

Unlike Phase 4 (decision token only), SAE training needs every token position
across all prompts to get sufficient training data (~900K vectors per layer
for SciFact: 6000 pairs × ~150 tokens avg).

Only extracts the 3 SAE target layers (7, 17, 21) — not all 28 — to keep
storage manageable (~4 GB total in float16).

Output:
  data/caches/sae_activations/{dataset}/layer_{i}.npy  — (N_tokens, 1536) float16
  data/caches/sae_activations/{dataset}/manifest.json  — metadata

Usage (CLI):
    python -m src.sae.extractor --dataset scifact
    python -m src.sae.extractor --dataset scifact --layers 7 17 21

Usage (Python):
    from src.sae.extractor import run_sae_extraction
    run_sae_extraction("scifact")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.reranking.prompt_builder import build_prompts_for_pairs
from src.reranking.qwen_inference import load_model
from src.utils.config import load_config
from src.utils.io import load_parquet
from src.utils.logging import get_logger
from src.utils.reproducibility import set_all_seeds

log = get_logger(__name__)


def _make_allpos_hook(layer_idx: int, storage: dict):
    """Hook that captures ALL token positions (not just the last one)."""
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden: (batch, seq_len, hidden_dim)
        storage[layer_idx].append(hidden.half().detach().cpu())
    return hook


def run_sae_extraction(
    dataset_name: str,
    data_root: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
    cache_root: str | Path = "data/caches/sae_activations",
    layers: list[int] | None = None,
    batch_size: int = 4,
) -> None:
    """Extract all-position activations at SAE target layers.

    Args:
        dataset_name: "scifact" or "nfcorpus"
        layers:       Which layers to extract. Defaults to sae.yaml target_layers.
        batch_size:   Lower than Phase 4 because we store all positions (longer sequences).
    """
    set_all_seeds(42)
    cfg = load_config("configs/sae.yaml", "configs/base.yaml",
                      "configs/reranker.yaml", "configs/data.yaml")

    if layers is None:
        layers = list(cfg.target_layers)
    layers = sorted(layers)
    log.info(f"SAE extraction: dataset={dataset_name}  layers={layers}")

    # ── Load pairs (with text columns) ─────────────────────────────────────
    # query_doc_pairs.parquet has query_text, doc_title, doc_text needed for prompts
    pairs_df = load_parquet(Path(interim_dir) / dataset_name / "query_doc_pairs.parquet")
    log.info(f"Loaded {len(pairs_df):,} pairs")

    # ── Load model ──────────────────────────────────────────────────────────
    model, tokenizer = load_model()

    # ── Build prompts ───────────────────────────────────────────────────────
    prompts = build_prompts_for_pairs(pairs_df, tokenizer, cfg=cfg)
    log.info(f"Built {len(prompts):,} prompts")

    # ── Output dirs ─────────────────────────────────────────────────────────
    out_dir = Path(cache_root) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract ─────────────────────────────────────────────────────────────
    # Storage: list of (batch, seq_len, hidden_dim) tensors per layer
    layer_buffers: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    # Register all-positions hooks
    handles = []
    for l in layers:
        hook_fn = _make_allpos_hook(l, layer_buffers)
        handle = model.model.layers[l].register_forward_hook(hook_fn)
        handles.append(handle)

    device = next(model.parameters()).device
    model.eval()

    token_counts = []  # track seq_len per prompt for manifest

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting SAE acts"):
            batch_prompts = prompts[i: i + batch_size]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # Track actual sequence lengths (non-padding tokens)
            attention_mask = enc["attention_mask"]
            seq_lens = attention_mask.sum(dim=1).cpu().tolist()
            token_counts.extend([int(s) for s in seq_lens])

            with torch.no_grad():
                model.model(**enc)

            if device.type == "mps":
                torch.mps.empty_cache()

    finally:
        for h in handles:
            h.remove()

    # ── Concatenate and save ─────────────────────────────────────────────────
    # Each buffer entry is (batch, seq_len, hidden_dim) — we want (N_tokens, hidden_dim)
    # We flatten batch × seq_len but only keep non-padding positions

    log.info("Concatenating and saving...")
    total_tokens = 0

    for l in layers:
        tensors = layer_buffers[l]
        # Flatten: concatenate along batch dim first → (total_padded_tokens, hidden_dim)
        # But we stored (batch, seq_len, hidden) so stack and reshape
        all_hidden = []
        idx = 0
        for t in tensors:
            # t: (batch_in_this_call, seq_len, hidden_dim)
            b = t.shape[0]
            for bi in range(b):
                sl = token_counts[idx]  # actual non-padding length
                all_hidden.append(t[bi, :sl, :])  # (sl, hidden_dim)
                idx += 1
        stacked = torch.cat(all_hidden, dim=0).numpy()  # (N_tokens, hidden_dim) float16
        total_tokens = stacked.shape[0]

        out_path = out_dir / f"layer_{l}.npy"
        np.save(out_path, stacked)
        log.info(f"  layer {l}: {stacked.shape}  → {out_path}")
        del stacked, all_hidden

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest = {
        "dataset": dataset_name,
        "layers": layers,
        "n_pairs": len(pairs_df),
        "total_tokens": total_tokens,
        "hidden_dim": cfg.architecture.input_dim,
        "dtype": "float16",
        "activation_source": "all_positions",
        "token_counts_per_pair": token_counts,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest saved → {out_dir / 'manifest.json'}")
    log.info(f"Total tokens extracted: {total_tokens:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all-position activations for SAE training.")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus"])
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to extract (default: from configs/sae.yaml)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cache_root", default="data/caches/sae_activations")
    args = parser.parse_args()

    run_sae_extraction(
        dataset_name=args.dataset,
        layers=args.layers,
        batch_size=args.batch_size,
        cache_root=args.cache_root,
    )
