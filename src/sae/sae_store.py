"""
SAE checkpoint save / load.

Saves:
  {ckpt_dir}/sae.pt        — model state_dict
  {ckpt_dir}/metadata.json — hyperparams + training stats
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.sae.model import TopKSAE


def save_sae(
    sae: TopKSAE,
    ckpt_dir: str | Path,
    metadata: dict | None = None,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(sae.state_dict(), ckpt_dir / "sae.pt")

    if metadata is not None:
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def load_sae(
    ckpt_dir: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[TopKSAE, dict]:
    """Load SAE from checkpoint directory.

    Returns:
        (sae, metadata)
    """
    ckpt_dir = Path(ckpt_dir)

    meta_path = ckpt_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {ckpt_dir}")
    with open(meta_path) as f:
        metadata = json.load(f)

    input_dim  = metadata["input_dim"]
    hidden_dim = metadata["hidden_dim"]
    k          = metadata["k"]
    expansion_factor = hidden_dim // input_dim

    sae = TopKSAE(input_dim=input_dim, expansion_factor=expansion_factor, k=k)
    state = torch.load(ckpt_dir / "sae.pt", map_location=device, weights_only=True)
    sae.load_state_dict(state)
    sae.to(device)
    sae.eval()

    return sae, metadata
