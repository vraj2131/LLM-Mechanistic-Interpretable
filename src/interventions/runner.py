"""
Phase 8 — Causal Interventions CLI runner.

Runs both probe-direction steering and SAE feature steering experiments
for a dataset and saves results to outputs/final/interventions/{dataset}/.

Usage:
    # Full run (both probe + SAE), layer 17, primary targets
    python -m src.interventions.runner --dataset scifact

    # Probe only, fast (layer 17, is_relevant, alphas ±1 ±3 ±5)
    python -m src.interventions.runner --dataset scifact --probe_only

    # SAE only
    python -m src.interventions.runner --dataset scifact --sae_only

    # Custom layers / targets
    python -m src.interventions.runner --dataset scifact \\
        --probe_layers 7 17 21 \\
        --probe_targets is_relevant lexical_overlap bm25_score \\
        --alpha_multipliers -5 -3 -1 1 3 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.interventions.evaluator import run_all_interventions
from src.utils.logging import get_logger

log = get_logger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(description="Phase 8: Causal intervention experiments")
    p.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus"])
    p.add_argument("--data_root", default="data/raw")
    p.add_argument("--interim_dir", default="data/interim")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--checkpoint_dir", default="outputs/final/sae_checkpoints")
    p.add_argument("--output_dir", default="outputs/final/interventions")
    p.add_argument("--batch_size", type=int, default=8)

    # Experiment selection
    p.add_argument("--probe_only", action="store_true")
    p.add_argument("--sae_only", action="store_true")

    # Probe config
    p.add_argument("--probe_layers", type=int, nargs="+", default=None,
                   help="Layers to steer (default: 7 17 21)")
    p.add_argument("--probe_targets", type=str, nargs="+", default=None,
                   help="Probe targets (default: is_relevant lexical_overlap bm25_score)")
    p.add_argument("--alpha_multipliers", type=float, nargs="+", default=None,
                   help="Signed alpha multipliers (default: -5 -3 -1 1 3 5)")

    # SAE config
    p.add_argument("--sae_layer", type=int, default=17)
    p.add_argument("--sae_features", type=int, nargs="+", default=None,
                   help="SAE feature indices (default: top-3 by r_is_relevant)")

    return p.parse_args()


def main():
    args = _parse_args()

    run_probe = not args.sae_only
    run_sae = not args.probe_only

    log.info(f"Phase 8 — Causal Interventions | dataset={args.dataset}")
    log.info(f"  run_probe={run_probe}, run_sae={run_sae}")

    results = run_all_interventions(
        dataset_name=args.dataset,
        data_root=args.data_root,
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        probe_targets=args.probe_targets,
        probe_layers=args.probe_layers,
        alpha_multipliers=args.alpha_multipliers,
        sae_layer=args.sae_layer,
        sae_feature_indices=args.sae_features,
        run_probe=run_probe,
        run_sae=run_sae,
    )

    # Print compact summary to stdout
    probe_results = results.get("probe", [])
    sae_results = results.get("sae", [])

    print("\n" + "=" * 70)
    print(f"PHASE 8 RESULTS — {args.dataset.upper()}")
    print("=" * 70)

    if probe_results:
        print("\n--- Probe Steering (top ΔnDCG conditions) ---")
        sorted_probe = sorted(probe_results, key=lambda r: abs(r["delta_ndcg"]), reverse=True)
        for r in sorted_probe[:10]:
            sig = "*" if r["significant"] else " "
            print(
                f"  {sig} {r['target']:<20} layer={r['layer']}  α={r['alpha_multiplier']:+.0f}"
                f"  ΔnDCG={r['delta_ndcg']:+.4f}  ΔMRR={r['delta_mrr']:+.4f}"
                f"  p={r['p_value']:.3f}"
                f"{'  [COLLAPSE]' if r['collapsed'] else ''}"
            )

    if sae_results:
        print("\n--- SAE Feature Steering (top ΔnDCG conditions) ---")
        sorted_sae = sorted(sae_results, key=lambda r: abs(r["delta_ndcg"]), reverse=True)
        for r in sorted_sae[:10]:
            sig = "*" if r["significant"] else " "
            alpha_str = f"α={r['alpha']:.0f}" if r["alpha"] is not None else "ablate"
            print(
                f"  {sig} feat={r['feature_idx']}  {r['mode']:<8}  {alpha_str}"
                f"  ΔnDCG={r['delta_ndcg']:+.4f}  ΔMRR={r['delta_mrr']:+.4f}"
                f"  p={r['p_value']:.3f}"
                f"{'  [COLLAPSE]' if r['collapsed'] else ''}"
            )

    n_sig_probe = sum(1 for r in probe_results if r.get("significant"))
    n_sig_sae = sum(1 for r in sae_results if r.get("significant"))
    print(f"\nSignificant results: {n_sig_probe}/{len(probe_results)} probe, "
          f"{n_sig_sae}/{len(sae_results)} SAE")
    print("=" * 70)


if __name__ == "__main__":
    main()
