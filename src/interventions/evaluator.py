"""
Phase 8 — Causal Intervention Evaluator.

Runs two experiment families and saves results:

  1. Probe-direction steering:
       For each (target, layer, alpha_multiplier, direction):
         - Register a ProbeSteeringHook
         - Re-score all query-doc pairs
         - Compute ΔnDCG@10 and ΔMRR@10 vs. baseline
         - Paired t-test on per-query nDCG deltas
         - Detect output collapse

  2. SAE feature steering:
       For each (feature_idx, mode, alpha):
         - Register a SAEFeatureHook at layer 17
         - Re-score all pairs
         - Compute same metrics

Results are returned as list-of-dicts, one row per experiment condition.

Usage:
    from src.interventions.evaluator import run_probe_interventions, run_sae_interventions
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

from src.data.loader import load_beir_dataset
from src.evaluation.metrics import ndcg_at_k, mrr_at_k, compute_all_metrics
from src.reranking.prompt_builder import build_prompts_for_pairs
from src.reranking.qwen_inference import score_pairs
from src.interventions.steering import ProbeSteeringHook, SAEFeatureHook
from src.utils.config import load_config
from src.utils.io import load_parquet
from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-query metric helpers
# ---------------------------------------------------------------------------

def _per_query_ndcg(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
    relevance_threshold: int = 1,
) -> dict[str, float]:
    """Return {qid: nDCG@k} for all queries with at least one relevant doc."""
    import math
    per_query = {}
    for qid, doc_scores in run.items():
        relevant = {
            did: label
            for did, label in qrels.get(qid, {}).items()
            if label >= relevance_threshold
        }
        if not relevant:
            continue
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        dcg = sum(
            (1.0 if relevant.get(did, 0) >= 1 else 0.0) / math.log2(rank + 2)
            for rank, (did, _) in enumerate(ranked)
        )
        ideal_gains = sorted([1.0 for _ in relevant], reverse=True)[:k]
        idcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal_gains))
        per_query[qid] = dcg / idcg if idcg > 0 else 0.0
    return per_query


def _per_query_mrr(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
    relevance_threshold: int = 1,
) -> dict[str, float]:
    per_query = {}
    for qid, doc_scores in run.items():
        relevant = {
            did for did, label in qrels.get(qid, {}).items()
            if label >= relevance_threshold
        }
        if not relevant:
            continue
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        rr = 0.0
        for rank, (did, _) in enumerate(ranked):
            if did in relevant:
                rr = 1.0 / (rank + 1)
                break
        per_query[qid] = rr
    return per_query


def _scores_df_to_run(scores_df: pd.DataFrame, score_col: str = "expected_score") -> dict:
    run: dict[str, dict[str, float]] = {}
    for row in scores_df.itertuples(index=False):
        run.setdefault(row.query_id, {})[row.doc_id] = float(getattr(row, score_col))
    return run


def _detect_collapse(
    scores: list[float],
    threshold: float = 0.05,
) -> bool:
    """Return True if score std is below threshold (all outputs collapsed)."""
    return float(np.std(scores)) < threshold


# ---------------------------------------------------------------------------
# Core inference with optional hook
# ---------------------------------------------------------------------------

def _score_pairs_with_hook(
    prompts: list[str],
    pairs_df: pd.DataFrame,
    model,
    tokenizer,
    hook,
    batch_size: int,
) -> pd.DataFrame:
    """Score all pairs with a registered hook, return scores DataFrame."""
    if hook is not None:
        hook.register(model)
    try:
        result = score_pairs(prompts, model, tokenizer, batch_size=batch_size)
    finally:
        if hook is not None:
            hook.remove()

    out = pairs_df[["query_id", "doc_id"]].copy()
    out["expected_score"] = result["expected_scores"]
    out["score"] = result["scores"]
    return out


# ---------------------------------------------------------------------------
# Metric computation + stats
# ---------------------------------------------------------------------------

def _compute_intervention_stats(
    intervened_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    qrels: dict,
    significance_level: float = 0.05,
    collapse_threshold: float = 0.05,
) -> dict:
    """Compute ΔnDCG@10, ΔMRR@10 and paired t-test vs baseline."""
    run_int = _scores_df_to_run(intervened_df)
    run_base = _scores_df_to_run(baseline_df)

    ndcg_int = ndcg_at_k(run_int, qrels)
    mrr_int = mrr_at_k(run_int, qrels)
    ndcg_base = ndcg_at_k(run_base, qrels)
    mrr_base = mrr_at_k(run_base, qrels)

    # Per-query for paired t-test
    pq_int = _per_query_ndcg(run_int, qrels)
    pq_base = _per_query_ndcg(run_base, qrels)
    shared_qids = sorted(set(pq_int) & set(pq_base))

    t_stat, p_value = float("nan"), float("nan")
    if len(shared_qids) >= 5:
        deltas = [pq_int[q] - pq_base[q] for q in shared_qids]
        t_res = ttest_rel([pq_int[q] for q in shared_qids],
                          [pq_base[q] for q in shared_qids])
        t_stat = float(t_res.statistic)
        p_value = float(t_res.pvalue)

    collapsed = _detect_collapse(
        intervened_df["expected_score"].tolist(),
        threshold=collapse_threshold,
    )

    return {
        "ndcg_at10": round(ndcg_int, 6),
        "mrr_at10": round(mrr_int, 6),
        "delta_ndcg": round(ndcg_int - ndcg_base, 6),
        "delta_mrr": round(mrr_int - mrr_base, 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < significance_level if not np.isnan(p_value) else False,
        "collapsed": collapsed,
        "n_queries": len(shared_qids),
    }


# ---------------------------------------------------------------------------
# Probe steering experiments
# ---------------------------------------------------------------------------

def run_probe_interventions(
    dataset_name: str,
    model,
    tokenizer,
    baseline_df: pd.DataFrame,
    qrels: dict,
    prompts: list[str],
    pairs_df: pd.DataFrame,
    targets: list[str] | None = None,
    layers: list[int] | None = None,
    alpha_multipliers: list[float] | None = None,
    processed_dir: str | Path = "data/processed",
    batch_size: int = 8,
    significance_level: float = 0.05,
    collapse_threshold: float = 0.05,
) -> list[dict]:
    """Run probe-direction steering for all (target × layer × alpha) combinations.

    Args:
        dataset_name:       "scifact" or "nfcorpus"
        model / tokenizer:  Pre-loaded Qwen model
        baseline_df:        Scores without intervention (expected_score column)
        qrels:              Ground-truth relevance judgments
        prompts:            Pre-built prompt list aligned with pairs_df
        pairs_df:           Query-doc pair DataFrame
        targets:            Probe target names. None = ["is_relevant","lexical_overlap","bm25_score"]
        layers:             Layers to intervene on. None = [7, 17, 21]
        alpha_multipliers:  Signed multipliers. None = [-5,-3,-1,1,3,5]
        processed_dir:      Root dir for probe weights
        batch_size:         Inference batch size
        significance_level: p-value threshold for significance flag
        collapse_threshold: std threshold to flag score collapse

    Returns:
        List of result dicts, one per (target, layer, alpha) condition.
    """
    if targets is None:
        targets = ["is_relevant", "lexical_overlap", "bm25_score"]
    if layers is None:
        layers = [7, 17, 21]
    if alpha_multipliers is None:
        alpha_multipliers = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]

    weights_dir = Path(processed_dir) / dataset_name / "probe_weights"
    results = []
    total = len(targets) * len(layers) * len(alpha_multipliers)
    done = 0

    for target in targets:
        for layer in layers:
            w_path = weights_dir / f"layer_{layer}_{target}.npy"
            if not w_path.exists():
                log.warning(f"Probe weight not found: {w_path} — skipping")
                continue
            w = np.load(w_path)
            w_norm = float(np.linalg.norm(w))

            for alpha_mult in alpha_multipliers:
                t0 = time.time()
                hook = ProbeSteeringHook(layer=layer, probe_weight=w, alpha=alpha_mult)
                int_df = _score_pairs_with_hook(
                    prompts, pairs_df, model, tokenizer, hook, batch_size
                )
                stats = _compute_intervention_stats(
                    int_df, baseline_df, qrels,
                    significance_level=significance_level,
                    collapse_threshold=collapse_threshold,
                )
                elapsed = time.time() - t0
                done += 1

                row = {
                    "experiment": "probe_steering",
                    "dataset": dataset_name,
                    "target": target,
                    "layer": layer,
                    "alpha_multiplier": alpha_mult,
                    "probe_norm": round(w_norm, 4),
                    "elapsed_s": round(elapsed, 1),
                    **stats,
                }
                results.append(row)
                log.info(
                    f"[{done}/{total}] probe {target} layer={layer} α={alpha_mult:+.0f} "
                    f"→ ΔnDCG={stats['delta_ndcg']:+.4f} p={stats['p_value']:.3f}"
                    f"{' COLLAPSED' if stats['collapsed'] else ''}"
                )

    return results


# ---------------------------------------------------------------------------
# SAE feature steering experiments
# ---------------------------------------------------------------------------

def _build_sae_feature_plan(
    dataset_name: str,
    layer: int,
    feature_indices: list[int] | None,
) -> list[dict]:
    """Return list of {feature_idx, ir_target, r_value} dicts to steer.

    Default: top feature per IR target (by |r|), one per target column.
    This covers both positively and negatively correlated features.
    Fallback (no parquet): hardcoded top-3 is_relevant features.
    """
    corr_path = (
        Path("outputs/final/sae_analysis") / dataset_name
        / f"ir_correlations_layer{layer}.parquet"
    )
    if feature_indices is not None:
        # Caller supplied explicit indices — label them as "custom", r_value=None
        return [{"feature_idx": fi, "ir_target": "custom", "r_value": None}
                for fi in feature_indices]

    if not corr_path.exists():
        log.warning(f"SAE correlation parquet not found: {corr_path} — using fallback")
        return [
            {"feature_idx": 30,   "ir_target": "is_relevant",   "r_value": +0.382},
            {"feature_idx": 2468, "ir_target": "is_relevant",   "r_value": +0.365},
            {"feature_idx": 1327, "ir_target": "is_relevant",   "r_value": +0.361},
        ]

    corr_df = pd.read_parquet(corr_path)
    # IR target columns in the parquet are named r_<target>
    ir_target_cols = [c for c in corr_df.columns if c.startswith("r_")]
    plan = []
    seen_features: set[int] = set()
    for col in ir_target_cols:
        ir_target = col[2:]  # strip "r_"
        best_idx = int(corr_df[col].abs().idxmax())
        r_val = float(corr_df.loc[best_idx, col])
        if best_idx not in seen_features:
            plan.append({"feature_idx": best_idx, "ir_target": ir_target, "r_value": r_val})
            seen_features.add(best_idx)
        else:
            # Feature already scheduled under another target — find next best
            ranked = corr_df[col].abs().sort_values(ascending=False)
            for alt_idx in ranked.index:
                alt_idx = int(alt_idx)
                if alt_idx not in seen_features:
                    plan.append({
                        "feature_idx": alt_idx,
                        "ir_target": ir_target,
                        "r_value": float(corr_df.loc[alt_idx, col]),
                    })
                    seen_features.add(alt_idx)
                    break

    log.info(f"SAE feature plan ({dataset_name} layer {layer}):")
    for item in plan:
        log.info(f"  feat={item['feature_idx']:>4}  target={item['ir_target']:<25}  r={item['r_value']:+.3f}")
    return plan


def run_sae_interventions(
    dataset_name: str,
    model,
    tokenizer,
    baseline_df: pd.DataFrame,
    qrels: dict,
    prompts: list[str],
    pairs_df: pd.DataFrame,
    layer: int = 17,
    feature_indices: list[int] | None = None,
    modes: list[str] | None = None,
    amplify_alphas: list[float] | None = None,
    checkpoint_dir: str | Path = "outputs/final/sae_checkpoints",
    batch_size: int = 8,
    significance_level: float = 0.05,
    collapse_threshold: float = 0.05,
) -> list[dict]:
    """Run SAE feature ablation and amplification experiments.

    By default, selects the top feature per IR target (by |Pearson r|) from the
    SAE correlation parquet — one feature per target, covering positive and
    negative correlations alike.  The ``r_value`` field in each result row records
    the correlation sign so results can be correctly interpreted:
      - r > 0: feature fires for high-target-value docs → ablation should decrease
               that signal, amplification should increase it.
      - r < 0: feature fires for low-target-value docs → ablation removes a
               suppressive signal, amplification pushes toward low-target-value.

    Args:
        feature_indices: Override auto-selection with explicit feature indices.
                         None = auto-select top feature per IR target.
        modes:           ["ablate", "amplify"]. None = both.
        amplify_alphas:  Alpha multipliers for amplify mode. None = [1.0, 3.0, 5.0].

    Returns:
        List of result dicts, one per (feature_idx, mode, alpha) condition.
        Each row includes ``ir_target`` and ``r_value`` for interpretation.
    """
    if modes is None:
        modes = ["ablate", "amplify"]
    if amplify_alphas is None:
        amplify_alphas = [1.0, 3.0, 5.0]

    feature_plan = _build_sae_feature_plan(dataset_name, layer, feature_indices)

    # Load SAE once
    from src.sae.model import TopKSAE
    ckpt_dir = Path(checkpoint_dir) / dataset_name / f"layer{layer}"
    with open(ckpt_dir / "metadata.json") as f:
        meta = json.load(f)
    expansion_factor = meta["hidden_dim"] // meta["input_dim"]
    sae = TopKSAE(
        input_dim=meta["input_dim"],
        expansion_factor=expansion_factor,
        k=meta["k"],
    )
    sae.load_state_dict(torch.load(ckpt_dir / "sae.pt", map_location="cpu", weights_only=True))
    sae.eval()

    conditions = []
    for feat_info in feature_plan:
        for mode in modes:
            alphas = amplify_alphas if mode == "amplify" else [1.0]
            for alpha in alphas:
                conditions.append((feat_info, mode, alpha))

    results = []
    total = len(conditions)
    for done, (feat_info, mode, alpha) in enumerate(conditions, 1):
        feat_idx = feat_info["feature_idx"]
        t0 = time.time()
        hook = SAEFeatureHook(
            layer=layer,
            sae=sae,
            feature_indices=[feat_idx],
            mode=mode,
            alpha=alpha,
        )
        int_df = _score_pairs_with_hook(
            prompts, pairs_df, model, tokenizer, hook, batch_size
        )
        stats = _compute_intervention_stats(
            int_df, baseline_df, qrels,
            significance_level=significance_level,
            collapse_threshold=collapse_threshold,
        )
        elapsed = time.time() - t0

        row = {
            "experiment": "sae_steering",
            "dataset": dataset_name,
            "layer": layer,
            "feature_idx": feat_idx,
            "ir_target": feat_info["ir_target"],
            "r_value": feat_info["r_value"],
            "mode": mode,
            "alpha": alpha if mode == "amplify" else None,
            "elapsed_s": round(elapsed, 1),
            **stats,
        }
        results.append(row)
        r_str = f"r={feat_info['r_value']:+.3f}" if feat_info["r_value"] is not None else ""
        log.info(
            f"[{done}/{total}] SAE feat={feat_idx} ({feat_info['ir_target']} {r_str})"
            f" mode={mode} α={alpha}"
            f" → ΔnDCG={stats['delta_ndcg']:+.4f} p={stats['p_value']:.3f}"
            f"{' COLLAPSED' if stats['collapsed'] else ''}"
        )

    return results


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_all_interventions(
    dataset_name: str = "scifact",
    data_root: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
    processed_dir: str | Path = "data/processed",
    checkpoint_dir: str | Path = "outputs/final/sae_checkpoints",
    output_dir: str | Path = "outputs/final/interventions",
    batch_size: int = 8,
    probe_targets: list[str] | None = None,
    probe_layers: list[int] | None = None,
    alpha_multipliers: list[float] | None = None,
    sae_layer: int = 17,
    sae_feature_indices: list[int] | None = None,
    run_probe: bool = True,
    run_sae: bool = True,
) -> dict[str, list[dict]]:
    """Orchestrate both experiment families and save results.

    Returns:
        {"probe": [...], "sae": [...]}
    """
    from src.reranking.qwen_inference import load_model
    from src.utils.reproducibility import set_all_seeds

    set_all_seeds(42)
    out_dir = Path(output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs and dataset
    pairs_path = Path(interim_dir) / dataset_name / "query_doc_pairs.parquet"
    pairs_df = load_parquet(pairs_path)
    dataset = load_beir_dataset(dataset_name, data_root=data_root)
    qrels = dataset.qrels

    # Load model
    cfg = load_config("configs/reranker.yaml")
    batch_size = batch_size or cfg.batch_size
    model, tokenizer = load_model(cfg=cfg)

    # Build prompts once (reused across all conditions)
    prompts = build_prompts_for_pairs(pairs_df, tokenizer, cfg=cfg)

    # Baseline scores — load from Phase 3 cache if available, otherwise re-score
    cached_scores_path = Path(processed_dir) / dataset_name / "reranker_scores.parquet"
    baseline_path = out_dir / "baseline_scores.parquet"

    if cached_scores_path.exists():
        log.info(f"Loading baseline from Phase 3 cache: {cached_scores_path}")
        raw = pd.read_parquet(cached_scores_path)
        baseline_df = pairs_df[["query_id", "doc_id"]].copy()
        baseline_df = baseline_df.merge(
            raw[["query_id", "doc_id", "reranker_expected_score"]],
            on=["query_id", "doc_id"],
        )
        baseline_df = baseline_df.rename(columns={"reranker_expected_score": "expected_score"})
    else:
        log.info("No cached baseline found — scoring baseline (no intervention)...")
        baseline_df = _score_pairs_with_hook(
            prompts, pairs_df, model, tokenizer, hook=None, batch_size=batch_size
        )

    baseline_run = _scores_df_to_run(baseline_df)
    baseline_metrics = compute_all_metrics(baseline_run, qrels)
    log.info(f"Baseline: nDCG@10={baseline_metrics['ndcg@10']:.4f}  MRR@10={baseline_metrics['mrr@10']:.4f}")

    # Save baseline for this run
    baseline_df.to_parquet(baseline_path, index=False)
    with open(out_dir / "baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    all_results: dict[str, list[dict]] = {"probe": [], "sae": []}

    if run_probe:
        log.info("=== Running probe-direction steering experiments ===")
        probe_results = run_probe_interventions(
            dataset_name=dataset_name,
            model=model,
            tokenizer=tokenizer,
            baseline_df=baseline_df,
            qrels=qrels,
            prompts=prompts,
            pairs_df=pairs_df,
            targets=probe_targets,
            layers=probe_layers,
            alpha_multipliers=alpha_multipliers,
            processed_dir=processed_dir,
            batch_size=batch_size,
        )
        all_results["probe"] = probe_results
        probe_path = out_dir / "probe_intervention_results.json"
        with open(probe_path, "w") as f:
            json.dump(probe_results, f, indent=2)
        log.info(f"Saved {len(probe_results)} probe results → {probe_path}")

    if run_sae:
        log.info("=== Running SAE feature steering experiments ===")
        sae_results = run_sae_interventions(
            dataset_name=dataset_name,
            model=model,
            tokenizer=tokenizer,
            baseline_df=baseline_df,
            qrels=qrels,
            prompts=prompts,
            pairs_df=pairs_df,
            layer=sae_layer,
            feature_indices=sae_feature_indices,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
        )
        all_results["sae"] = sae_results
        sae_path = out_dir / "sae_intervention_results.json"
        with open(sae_path, "w") as f:
            json.dump(sae_results, f, indent=2)
        log.info(f"Saved {len(sae_results)} SAE results → {sae_path}")

    # Combined summary
    summary_path = out_dir / "intervention_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "baseline_metrics": baseline_metrics,
                "n_probe_conditions": len(all_results["probe"]),
                "n_sae_conditions": len(all_results["sae"]),
                "significant_probe": [
                    r for r in all_results["probe"] if r.get("significant")
                ],
                "significant_sae": [
                    r for r in all_results["sae"] if r.get("significant")
                ],
            },
            f,
            indent=2,
        )
    log.info(f"Summary saved → {summary_path}")
    return all_results
