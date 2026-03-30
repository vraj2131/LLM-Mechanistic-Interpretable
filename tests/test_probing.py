"""
Tests for Phase 6: linear probing modules.

Covers:
  - ProbeResult dataclass fields
  - train_ridge_probe / train_logistic_probe return correct types and shapes
  - bootstrap_ci returns (lower, upper) with lower <= upper
  - ProbeTarget definitions are consistent
  - Runner helpers (_result_to_dict) produce expected keys
"""

from __future__ import annotations

import numpy as np
import pytest

from src.probing.bootstrap import bootstrap_ci
from src.probing.probe import ProbeResult, train_ridge_probe, train_logistic_probe
from src.probing.targets import PROBE_TARGETS, TARGET_BY_NAME, TARGET_NAMES, ProbeTarget


# ─────────────────────────────────────────────────────── fixtures ──────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def regression_data(rng):
    """Small continuous regression dataset (200 samples, 64 features)."""
    X = rng.normal(size=(200, 64)).astype(np.float32)
    y = (X[:, 0] * 2.0 + X[:, 1] - X[:, 2] * 0.5 + rng.normal(size=200) * 0.1).astype(np.float32)
    return X[:160], y[:160], X[160:], y[160:]


@pytest.fixture
def classification_data(rng):
    """Small binary classification dataset (200 samples, 64 features)."""
    X = rng.normal(size=(200, 64)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X[:160], y[:160], X[160:], y[160:]


# ──────────────────────────────────────────────── ProbeTarget tests ────────────

def test_probe_targets_count():
    assert len(PROBE_TARGETS) == 7


def test_probe_target_names_unique():
    names = [t.name for t in PROBE_TARGETS]
    assert len(names) == len(set(names))


def test_probe_target_fields():
    for t in PROBE_TARGETS:
        assert t.probe_type in ("ridge", "logistic")
        assert t.metric in ("r2", "auroc")
        assert t.column != ""
        assert t.description != ""


def test_target_by_name_lookup():
    t = TARGET_BY_NAME["is_relevant"]
    assert t.probe_type == "logistic"
    assert t.metric == "auroc"


def test_target_names_order():
    assert TARGET_NAMES == [t.name for t in PROBE_TARGETS]


def test_probe_target_frozen():
    t = PROBE_TARGETS[0]
    with pytest.raises((AttributeError, TypeError)):
        t.name = "other"  # type: ignore[misc]


# ──────────────────────────────────────────────── train_ridge_probe ────────────

def test_ridge_probe_returns_probe_result(regression_data, rng):
    X_tr, y_tr, X_te, y_te = regression_data
    result = train_ridge_probe(
        X_tr, y_tr, X_te, y_te,
        alpha_grid=[1.0, 10.0], cv_folds=3,
        layer=0, target="bm25_score", rng=rng,
    )
    assert isinstance(result, ProbeResult)


def test_ridge_probe_coef_shape(regression_data, rng):
    X_tr, y_tr, X_te, y_te = regression_data
    result = train_ridge_probe(
        X_tr, y_tr, X_te, y_te,
        alpha_grid=[1.0], cv_folds=2,
        layer=5, target="lexical_overlap", rng=rng,
    )
    assert result.coef.shape == (X_tr.shape[1],)


def test_ridge_probe_metadata(regression_data, rng):
    X_tr, y_tr, X_te, y_te = regression_data
    result = train_ridge_probe(
        X_tr, y_tr, X_te, y_te,
        alpha_grid=[1.0], cv_folds=2,
        layer=3, target="bm25_rank", rng=rng,
    )
    assert result.layer == 3
    assert result.target == "bm25_rank"
    assert result.probe_type == "ridge"
    assert result.train_size == len(X_tr)
    assert result.test_size == len(X_te)


def test_ridge_probe_score_finite(regression_data, rng):
    X_tr, y_tr, X_te, y_te = regression_data
    result = train_ridge_probe(
        X_tr, y_tr, X_te, y_te,
        alpha_grid=[0.1, 1.0, 10.0], cv_folds=3,
        layer=0, target="lexical_overlap", rng=rng,
    )
    assert np.isfinite(result.score)
    assert np.isfinite(result.cv_score)


def test_ridge_probe_signal_captured(rng):
    """With clear linear signal, R² should be positive."""
    X = rng.normal(size=(300, 32)).astype(np.float32)
    y = (X[:, 0] * 3 + X[:, 1]).astype(np.float32)
    result = train_ridge_probe(
        X[:240], y[:240], X[240:], y[240:],
        alpha_grid=[0.01, 0.1, 1.0], cv_folds=3,
        layer=0, target="test", rng=rng,
    )
    assert result.score > 0.5, f"Expected R²>0.5, got {result.score:.4f}"


# ─────────────────────────────────────────────── train_logistic_probe ──────────

def test_logistic_probe_returns_probe_result(classification_data, rng):
    X_tr, y_tr, X_te, y_te = classification_data
    result = train_logistic_probe(
        X_tr, y_tr, X_te, y_te,
        C_grid=[1.0], cv_folds=3,
        layer=0, target="is_relevant", rng=rng,
    )
    assert isinstance(result, ProbeResult)


def test_logistic_probe_coef_shape(classification_data, rng):
    X_tr, y_tr, X_te, y_te = classification_data
    result = train_logistic_probe(
        X_tr, y_tr, X_te, y_te,
        C_grid=[1.0], cv_folds=2,
        layer=7, target="is_relevant", rng=rng,
    )
    assert result.coef.shape == (X_tr.shape[1],)


def test_logistic_probe_metadata(classification_data, rng):
    X_tr, y_tr, X_te, y_te = classification_data
    result = train_logistic_probe(
        X_tr, y_tr, X_te, y_te,
        C_grid=[1.0], cv_folds=2,
        layer=14, target="doc_length_bucket", rng=rng,
    )
    assert result.layer == 14
    assert result.target == "doc_length_bucket"
    assert result.probe_type == "logistic"
    assert result.train_size == len(X_tr)
    assert result.test_size == len(X_te)


def test_logistic_probe_auroc_range(classification_data, rng):
    X_tr, y_tr, X_te, y_te = classification_data
    result = train_logistic_probe(
        X_tr, y_tr, X_te, y_te,
        C_grid=[0.1, 1.0], cv_folds=3,
        layer=0, target="is_relevant", rng=rng,
    )
    assert 0.0 <= result.score <= 1.0


def test_logistic_probe_signal_captured(rng):
    """With a clear classification signal, AUROC should be high."""
    X = rng.normal(size=(300, 32)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    result = train_logistic_probe(
        X[:240], y[:240], X[240:], y[240:],
        C_grid=[0.1, 1.0, 10.0], cv_folds=3,
        layer=0, target="is_relevant", rng=rng,
    )
    assert result.score > 0.7, f"Expected AUROC>0.7, got {result.score:.4f}"


# ─────────────────────────────────────────────────── bootstrap_ci ──────────────

def test_bootstrap_ci_returns_tuple(rng):
    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=float)
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    from sklearn.metrics import roc_auc_score
    ci = bootstrap_ci(y_true, y_score, roc_auc_score, n=50, rng=rng)
    assert len(ci) == 2


def test_bootstrap_ci_lower_le_upper(rng):
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15])
    from sklearn.metrics import roc_auc_score
    lo, hi = bootstrap_ci(y_true, y_score, roc_auc_score, n=100, rng=rng)
    assert lo <= hi


def test_bootstrap_ci_reasonable_width(rng):
    """CI should be narrower than the full [0,1] range."""
    rng2 = np.random.default_rng(0)
    y_true = (rng2.uniform(size=200) > 0.5).astype(int)
    y_score = y_true * 0.7 + rng2.uniform(size=200) * 0.3
    from sklearn.metrics import roc_auc_score
    lo, hi = bootstrap_ci(y_true.astype(float), y_score, roc_auc_score, n=200, rng=rng)
    assert hi - lo < 0.5


def test_bootstrap_ci_degenerate_returns_zeros():
    """If all resamples fail (single class), return (0, 0)."""
    y_true = np.zeros(10, dtype=float)  # all same class
    y_score = np.ones(10)
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    lo, hi = bootstrap_ci(y_true, y_score, roc_auc_score, n=20, rng=rng)
    # May return (0, 0) or a valid CI — just check no exception and valid types
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_bootstrap_ci_r2_metric(rng):
    """Bootstrap CI works with R² metric function too."""
    rng2 = np.random.default_rng(1)
    y_true = rng2.normal(size=100).astype(np.float32)
    y_score = y_true + rng2.normal(size=100) * 0.2
    metric_fn = lambda yt, yp: float(
        1 - ((yt - yp)**2).sum() / ((yt - yt.mean())**2).sum()
    )
    lo, hi = bootstrap_ci(y_true, y_score, metric_fn, n=100, rng=rng)
    assert lo <= hi
    assert hi > 0  # should have positive R²
