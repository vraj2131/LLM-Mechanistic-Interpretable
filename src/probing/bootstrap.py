"""
Bootstrap confidence intervals for probe scores.

For each (layer, target) probe score, draw n_bootstrap resamples of the
test set predictions and recompute the metric to get a 95% CI.

This is lighter than re-training probes: we resample (y_true, y_pred) pairs,
not activations + labels, so it runs in milliseconds per probe.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Compute bootstrap percentile CI for a scalar metric.

    Args:
        y_true: Ground-truth labels/values.
        y_score: Model predictions/scores.
        metric_fn: Callable(y_true, y_score) -> float.
        n: Number of bootstrap resamples.
        alpha: Significance level (0.05 → 95% CI).
        rng: Random generator for reproducibility.

    Returns:
        (lower, upper) confidence interval bounds.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    scores = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        try:
            s = metric_fn(y_true[sample], y_score[sample])
            scores.append(s)
        except Exception:
            # Skip degenerate samples (e.g. only one class)
            continue

    if not scores:
        return (0.0, 0.0)

    scores_arr = np.array(scores)
    lower = float(np.percentile(scores_arr, 100 * alpha / 2))
    upper = float(np.percentile(scores_arr, 100 * (1 - alpha / 2)))
    return lower, upper
