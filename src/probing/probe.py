"""
LinearProbe — wraps Ridge / LogisticRegression with cross-validated
hyperparameter selection and returns R² (regression) or AUROC (classification).

Design:
  - Stratified train/test split (80/20) preserves class balance for binary targets
  - 5-fold stratified CV on train set to pick best alpha/C
  - Final model trained on full train set, evaluated on held-out test set
  - Probe weights (.coef_) saved as steering directions for Phase 8
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    layer: int
    target: str
    probe_type: str          # "ridge" or "logistic"
    score: float             # R² or AUROC on test set
    cv_score: float          # mean CV score on train set
    coef: np.ndarray         # shape (hidden_dim,) — steering direction
    intercept: float
    train_size: int
    test_size: int


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def train_ridge_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: list[float],
    cv_folds: int,
    layer: int,
    target: str,
    rng: np.random.Generator,
) -> ProbeResult:
    """Train Ridge regression probe with CV alpha selection."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    best_alpha, best_cv = alpha_grid[0], -np.inf
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                             random_state=int(rng.integers(1000)))
        # Use regression CV — StratifiedKFold needs integer labels, fall back to KFold
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv_folds, shuffle=True,
                   random_state=int(rng.integers(1000)))
        scores = cross_val_score(model, X_tr, y_train, cv=kf, scoring="r2")
        mean_cv = scores.mean()
        if mean_cv > best_cv:
            best_cv = mean_cv
            best_alpha = alpha

    final = Ridge(alpha=best_alpha)
    final.fit(X_tr, y_train)
    y_pred = final.predict(X_te)
    test_score = _r2(y_test, y_pred)

    # Transform coef/intercept back to original (unscaled) input space so they
    # can be applied directly to raw activations (needed for Phase 8 steering).
    # Scaled model: y = coef_s @ ((x - mean) / scale) + intercept_s
    # Rearranged:   y = (coef_s / scale) @ x  +  (intercept_s - (coef_s/scale) @ mean)
    coef_orig = final.coef_ / scaler.scale_
    intercept_orig = float(final.intercept_) - float(coef_orig @ scaler.mean_)

    return ProbeResult(
        layer=layer,
        target=target,
        probe_type="ridge",
        score=test_score,
        cv_score=best_cv,
        coef=coef_orig,
        intercept=intercept_orig,
        train_size=len(X_train),
        test_size=len(X_test),
    )


def train_logistic_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C_grid: list[float],
    cv_folds: int,
    layer: int,
    target: str,
    rng: np.random.Generator,
) -> ProbeResult:
    """Train Logistic Regression probe with CV C selection. Reports AUROC."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    best_C, best_cv = C_grid[0], -np.inf
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=int(rng.integers(1000)))

    for C in C_grid:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        scores = cross_val_score(model, X_tr, y_train, cv=kf, scoring="roc_auc")
        mean_cv = scores.mean()
        if mean_cv > best_cv:
            best_cv = mean_cv
            best_C = C

    final = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    final.fit(X_tr, y_train)
    y_proba = final.predict_proba(X_te)[:, 1]
    test_score = float(roc_auc_score(y_test, y_proba))

    # Transform to original (unscaled) input space — same logic as ridge above.
    coef_orig = final.coef_[0] / scaler.scale_
    intercept_orig = float(final.intercept_[0]) - float(coef_orig @ scaler.mean_)

    return ProbeResult(
        layer=layer,
        target=target,
        probe_type="logistic",
        score=test_score,
        cv_score=best_cv,
        coef=coef_orig,
        intercept=intercept_orig,
        train_size=len(X_train),
        test_size=len(X_test),
    )
