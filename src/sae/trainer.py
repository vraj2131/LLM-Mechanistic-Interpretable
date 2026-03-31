"""
SAE training loop.

Design:
  - Loss = MSE(x, x_hat) only — TopK enforces sparsity structurally
  - Adam optimiser, lr=1e-4
  - Decoder columns normalised to unit norm after every gradient step
  - 10% validation split for early stopping / monitoring
  - Checkpoints saved to outputs/final/sae_checkpoints/layer{i}/

Usage (CLI):
    python -m src.sae.trainer --layer 17 --dataset scifact
    python -m src.sae.trainer --layer 17 --dataset scifact --epochs 50

Usage (Python):
    from src.sae.trainer import train_sae
    result = train_sae(layer=17, dataset_name="scifact")
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.sae.model import TopKSAE
from src.sae.sae_store import save_sae, load_sae
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.reproducibility import set_all_seeds

log = get_logger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_sae_activations(
    layer: int,
    dataset_name: str,
    sae_cache_root: str | Path = "data/caches/sae_activations",
) -> np.ndarray:
    """Load all-positions activation cache for one layer."""
    path = Path(sae_cache_root) / dataset_name / f"layer_{layer}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"SAE activation cache not found: {path}\n"
            f"Run: python -m src.sae.extractor --dataset {dataset_name}"
        )
    log.info(f"Loading SAE activations: {path}")
    acts = np.load(path, mmap_mode="r")
    log.info(f"  shape={acts.shape}  dtype={acts.dtype}")
    return acts


def train_sae(
    layer: int,
    dataset_name: str = "scifact",
    sae_cache_root: str | Path = "data/caches/sae_activations",
    checkpoint_dir: str | Path = "outputs/final/sae_checkpoints",
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
) -> dict:
    """Train one TopK SAE on all-positions activations at a given layer.

    Returns:
        dict with keys: layer, final_train_loss, final_val_loss, epochs_trained,
                        dead_feature_pct, checkpoint_path
    """
    set_all_seeds(42)
    cfg = load_config("configs/sae.yaml", "configs/base.yaml")

    epochs     = epochs     or cfg.training.epochs
    batch_size = batch_size or cfg.training.batch_size
    lr         = lr         or cfg.training.lr

    input_dim        = cfg.architecture.input_dim
    expansion_factor = cfg.architecture.expansion_factor
    k                = cfg.architecture.k

    device = _get_device()
    log.info(f"Training SAE: layer={layer}  device={device}  epochs={epochs}")

    # ── Load activations ────────────────────────────────────────────────────
    acts_np = _load_sae_activations(layer, dataset_name, sae_cache_root)

    # Subsample if dataset is larger than max_samples (keeps runtime predictable)
    max_samples = getattr(cfg.training, "max_samples", None)
    if max_samples and len(acts_np) > max_samples:
        rng_np = np.random.default_rng(42)
        idx = rng_np.choice(len(acts_np), size=max_samples, replace=False)
        idx.sort()
        acts_np = acts_np[idx]
        log.info(f"  Subsampled to {max_samples:,} vectors")

    X = torch.from_numpy(acts_np.astype(np.float32))  # (N, input_dim)

    # ── Train / val split ───────────────────────────────────────────────────
    val_size   = int(len(X) * cfg.training.val_split)
    train_size = len(X) - val_size
    train_ds, val_ds = random_split(
        TensorDataset(X), [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    log.info(f"  train={train_size:,}  val={val_size:,}  batches/epoch={len(train_loader)}")

    # ── Model + optimiser ───────────────────────────────────────────────────
    sae = TopKSAE(input_dim=input_dim, expansion_factor=expansion_factor, k=k).to(device)
    optimiser = torch.optim.Adam(sae.parameters(), lr=lr)

    # ── Training loop ───────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        t0 = time.time()
        sae.train()
        train_losses = []

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            x_hat, _ = sae(batch_x)
            loss = nn.functional.mse_loss(x_hat, batch_x)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            sae.normalise_decoder()

            train_losses.append(loss.item())

        # ── Validation ──────────────────────────────────────────────────────
        sae.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                x_hat, _ = sae(batch_x)
                val_losses.append(nn.functional.mse_loss(x_hat, batch_x).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(
                f"  epoch {epoch+1:>3}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"({elapsed:.1f}s)"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            # Save best checkpoint
            ckpt_dir = Path(checkpoint_dir) / f"layer{layer}"
            save_sae(sae, ckpt_dir, metadata={
                "layer": layer,
                "dataset": dataset_name,
                "epoch": best_epoch,
                "val_loss": best_val_loss,
                "train_loss": train_loss,
                "input_dim": input_dim,
                "hidden_dim": sae.hidden_dim,
                "k": k,
            })

    log.info(f"  Best val loss {best_val_loss:.4f} at epoch {best_epoch}")

    # ── Dead feature analysis ────────────────────────────────────────────────
    dead_pct = _compute_dead_feature_pct(sae, val_loader, device, sae.hidden_dim)
    log.info(f"  Dead features: {dead_pct:.1f}%  (threshold: {cfg.dead_feature_pct_threshold}%)")

    return {
        "layer": layer,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_trained": epochs,
        "dead_feature_pct": dead_pct,
        "checkpoint_path": str(Path(checkpoint_dir) / f"layer{layer}"),
        "history": history,
    }


def _compute_dead_feature_pct(
    sae: TopKSAE,
    val_loader: DataLoader,
    device: torch.device,
    hidden_dim: int,
) -> float:
    """Fraction of features that never activate on the validation set."""
    ever_active = torch.zeros(hidden_dim, dtype=torch.bool, device=device)
    sae.eval()
    with torch.no_grad():
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(device)
            _, sparse = sae(batch_x)
            ever_active |= (sparse > 0).any(dim=0)
    dead = (~ever_active).sum().item()
    return 100.0 * dead / hidden_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE on one layer.")
    parser.add_argument("--layer",   type=int, required=True)
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "nfcorpus"])
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--sae_cache_root",  default="data/caches/sae_activations")
    parser.add_argument("--checkpoint_dir",  default="outputs/final/sae_checkpoints")
    args = parser.parse_args()

    result = train_sae(
        layer=args.layer,
        dataset_name=args.dataset,
        sae_cache_root=args.sae_cache_root,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
    )
    print(f"\nDone. Best val loss={result['best_val_loss']:.4f}  "
          f"dead features={result['dead_feature_pct']:.1f}%  "
          f"checkpoint={result['checkpoint_path']}")
