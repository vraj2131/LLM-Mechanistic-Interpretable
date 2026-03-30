"""
Probe result visualisations.

  plot_heatmap()         — 28 × 7 heatmap (layers × targets), colour = R²/AUROC
  plot_layerwise_curves()— per-target line plots with 95% CI bands
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.probing.targets import PROBE_TARGETS, TARGET_NAMES

matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 11})

# Colour palette: distinct hues per target
_TARGET_COLORS = [
    "#4878d0", "#ee854a", "#6acc65", "#d65f5f",
    "#956cb4", "#8c613c", "#dc7ec0",
]


def _load_results(results_path: str | Path) -> pd.DataFrame:
    import json
    with open(results_path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def _make_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[int]]:
    """Return (matrix, target_names, layer_indices) sorted consistently."""
    layers = sorted(df["layer"].unique())
    targets = TARGET_NAMES  # canonical order
    mat = np.full((len(layers), len(targets)), np.nan)
    for i, layer in enumerate(layers):
        for j, tname in enumerate(targets):
            row = df[(df["layer"] == layer) & (df["target"] == tname)]
            if not row.empty:
                mat[i, j] = row["score"].values[0]
    return mat, targets, layers


def plot_heatmap(
    results_path: str | Path,
    out_path: str | Path | None = None,
    title: str = "Probe Score Heatmap (layers × targets)",
) -> plt.Figure:
    """28 × 7 heatmap. Rows = layers (0 at top), cols = targets.

    Colour scale is per-column (each target normalised 0→1) so features with
    different absolute scales are visually comparable.
    """
    df = _load_results(results_path)
    mat, targets, layers = _make_matrix(df)

    # Per-column normalisation for visual comparability
    mat_norm = mat.copy()
    for j in range(mat.shape[1]):
        col = mat_norm[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        lo, hi = valid.min(), valid.max()
        if hi > lo:
            mat_norm[:, j] = (col - lo) / (hi - lo)
        else:
            mat_norm[:, j] = 0.5

    target_labels = [t.description for t in PROBE_TARGETS]
    metric_labels = [t.metric.upper() for t in PROBE_TARGETS]

    fig, ax = plt.subplots(figsize=(len(targets) * 1.5 + 1.5, len(layers) * 0.38 + 1.5))
    im = ax.imshow(mat_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    # Annotate with actual score values
    for i in range(len(layers)):
        for j in range(len(targets)):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "white" if mat_norm[i, j] > 0.65 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(
        [f"{lbl}\n({m})" for lbl, m in zip(target_labels, metric_labels)],
        fontsize=9,
    )
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_xlabel("Probe Target", labelpad=8)
    ax.set_ylabel("Layer", labelpad=8)
    ax.set_title(title, fontsize=13, pad=12)

    plt.colorbar(im, ax=ax, label="Normalised score (per target)", fraction=0.02, pad=0.02)
    plt.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    return fig


def plot_layerwise_curves(
    results_path: str | Path,
    out_path: str | Path | None = None,
    title: str = "Probe Score by Layer",
) -> plt.Figure:
    """One subplot per probe target showing score ± 95% CI band across layers."""
    df = _load_results(results_path)
    targets = PROBE_TARGETS
    n_targets = len(targets)
    ncols = 4
    nrows = (n_targets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), sharey=False)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, target in enumerate(targets):
        ax = axes_flat[idx]
        sub = df[df["target"] == target.name].sort_values("layer")
        layers = sub["layer"].values
        scores = sub["score"].values
        ci_lo = sub["ci_lower"].values
        ci_hi = sub["ci_upper"].values

        color = _TARGET_COLORS[idx % len(_TARGET_COLORS)]
        ax.plot(layers, scores, color=color, linewidth=2, marker="o", markersize=3)
        ax.fill_between(layers, ci_lo, ci_hi, alpha=0.25, color=color)

        ax.set_title(f"{target.description}\n({target.metric.upper()})", fontsize=10)
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel(target.metric.upper(), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(layers.min() - 0.5, layers.max() + 0.5)

        # Mark best layer
        if len(scores) > 0:
            best_idx = int(np.argmax(scores))
            ax.axvline(layers[best_idx], color=color, linestyle="--", alpha=0.5, linewidth=1)

    # Hide unused subplots
    for idx in range(n_targets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    return fig


def plot_best_layer_summary(
    results_path: str | Path,
    out_path: str | Path | None = None,
    title: str = "Peak Probe Score per Target",
) -> plt.Figure:
    """Bar chart: best score per target + layer of peak."""
    df = _load_results(results_path)
    targets = PROBE_TARGETS

    best_scores, best_layers, metrics = [], [], []
    for t in targets:
        sub = df[df["target"] == t.name]
        if sub.empty:
            best_scores.append(0.0)
            best_layers.append(-1)
        else:
            best_row = sub.loc[sub["score"].idxmax()]
            best_scores.append(best_row["score"])
            best_layers.append(int(best_row["layer"]))
        metrics.append(t.metric.upper())

    x = np.arange(len(targets))
    target_labels = [t.description for t in targets]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(x, best_scores, color=_TARGET_COLORS[:len(targets)], alpha=0.85, edgecolor="white")

    for bar, layer, metric, score in zip(bars, best_layers, metrics, best_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"L{layer}\n{score:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Peak score (R² or AUROC)")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, min(1.1, max(best_scores) * 1.25 + 0.05))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    return fig
