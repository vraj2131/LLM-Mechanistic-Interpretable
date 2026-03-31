"""
TopK Sparse Autoencoder (SAE) for mechanistic interpretability.

Architecture:
  encoder: Linear(input_dim → hidden_dim) → TopK selection → ReLU
  decoder: Linear(hidden_dim → input_dim)  [columns kept unit-normed]

Forward pass:
  x_centered = x - decoder.bias          # centre before encoding
  pre_act    = encoder(x_centered)        # (batch, hidden_dim)
  sparse     = topk_relu(pre_act, k)      # keep top-k activations, zero rest
  x_hat      = decoder(sparse)            # reconstruct
  loss       = MSE(x, x_hat)             # no L1 — TopK enforces sparsity

Decoder column normalisation (call after every optimiser step):
  sae.normalise_decoder()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 1536,
        expansion_factor: int = 8,
        k: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion_factor
        self.k = k

        # Encoder: bias absorbs the pre-activation mean shift
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)

        # Decoder: bias doubles as the "pre-encoder centering" term
        # (we subtract decoder.bias before encoding, as per Anthropic convention)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder.bias)
        # Decoder columns initialised as unit vectors
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        self.normalise_decoder()

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        """Project decoder columns to unit norm. Call after every optimiser step."""
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse feature activations (batch, hidden_dim)."""
        x_centered = x - self.decoder.bias  # centre in decoder output space
        pre_act = self.encoder(x_centered)   # (batch, hidden_dim)
        return self._topk_relu(pre_act)

    def _topk_relu(self, pre_act: torch.Tensor) -> torch.Tensor:
        """Keep top-k values per sample, zero the rest, then ReLU."""
        topk_vals, topk_idx = pre_act.topk(self.k, dim=-1)
        sparse = torch.zeros_like(pre_act)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return F.relu(sparse)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) float32 activations

        Returns:
            x_hat:  (batch, input_dim) reconstruction
            sparse: (batch, hidden_dim) sparse feature activations
        """
        sparse = self.encode(x)
        x_hat = self.decoder(sparse)
        return x_hat, sparse

    def get_feature_directions(self) -> torch.Tensor:
        """Return decoder columns (input_dim, hidden_dim) — each column is a feature direction."""
        return self.decoder.weight.detach()  # shape: (input_dim, hidden_dim)
