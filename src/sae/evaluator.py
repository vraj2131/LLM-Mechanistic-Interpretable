"""
SAE evaluation metrics.

  reconstruction_mse()  — mean squared error between input and reconstruction
  mean_l0()             — average number of active features per sample
  dead_feature_pct()    — fraction of features never active on a dataset
  evaluate_sae()        — compute all three on a numpy array of activations
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.sae.model import TopKSAE


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_sae(
    sae: TopKSAE,
    acts: np.ndarray,
    batch_size: int = 2048,
    device: torch.device | None = None,
) -> dict:
    """Compute reconstruction MSE, mean L0, and dead feature % on a numpy array.

    Args:
        sae:        Trained TopKSAE (will be moved to device).
        acts:       (N, input_dim) float16 or float32 activations.
        batch_size: Batch size for inference.
        device:     Defaults to best available.

    Returns:
        dict with keys: mse, mean_l0, dead_feature_pct, n_samples, hidden_dim
    """
    if device is None:
        device = _get_device()

    sae = sae.to(device)
    sae.eval()

    X = torch.from_numpy(acts.astype(np.float32))
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

    total_mse = 0.0
    total_l0  = 0.0
    n_batches = 0
    ever_active = torch.zeros(sae.hidden_dim, dtype=torch.bool, device=device)

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_hat, sparse = sae(batch_x)

            mse = torch.nn.functional.mse_loss(x_hat, batch_x).item()
            l0  = (sparse > 0).float().sum(dim=-1).mean().item()

            total_mse += mse
            total_l0  += l0
            n_batches += 1
            ever_active |= (sparse > 0).any(dim=0)

    dead_count = (~ever_active).sum().item()

    return {
        "mse":              total_mse / n_batches,
        "mean_l0":          total_l0  / n_batches,
        "dead_feature_pct": 100.0 * dead_count / sae.hidden_dim,
        "n_samples":        len(acts),
        "hidden_dim":       sae.hidden_dim,
    }


def reconstruction_mse(
    sae: TopKSAE,
    acts: np.ndarray,
    batch_size: int = 2048,
    device: torch.device | None = None,
) -> float:
    return evaluate_sae(sae, acts, batch_size, device)["mse"]


def mean_l0(
    sae: TopKSAE,
    acts: np.ndarray,
    batch_size: int = 2048,
    device: torch.device | None = None,
) -> float:
    return evaluate_sae(sae, acts, batch_size, device)["mean_l0"]


def dead_feature_pct(
    sae: TopKSAE,
    acts: np.ndarray,
    batch_size: int = 2048,
    device: torch.device | None = None,
) -> float:
    return evaluate_sae(sae, acts, batch_size, device)["dead_feature_pct"]
