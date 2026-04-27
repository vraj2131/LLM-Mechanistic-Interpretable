"""
Forward-modifying hooks for causal interventions in Phase 8.

Two intervention types:
  ProbeSteeringHook  — adds ±α·w to the decision token residual stream at a given
                       layer, where w is a saved probe weight vector.
  SAEFeatureHook     — zero-ablates or amplifies specific SAE feature directions in
                       the residual stream at a given layer.

Usage:
    hook = ProbeSteeringHook(layer=17, probe_weight=w, alpha=3.0)
    hook.register(model)
    # ... run forward pass ...
    hook.remove()
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from src.utils.logging import get_logger

log = get_logger(__name__)


class ProbeSteeringHook:
    """Injects a probe-direction steering vector at the decision token.

    The perturbation added is:  alpha * (w / ||w||) * ||w||  =  alpha * w
    where alpha is a signed scalar (positive = steer toward feature; negative = away).

    Mathematically this is equivalent to: hidden[:, -1, :] += alpha * w
    """

    def __init__(
        self,
        layer: int,
        probe_weight: np.ndarray,
        alpha: float,
        token_pos: int = -1,
    ) -> None:
        self.layer = layer
        self.alpha = alpha
        self.token_pos = token_pos
        self._handle = None

        # Pre-compute the steering vector once (alpha * w as a CPU float32 tensor).
        # It is moved to the model's device at registration time.
        w = probe_weight.astype(np.float32)
        self._vec_cpu = torch.from_numpy(alpha * w)  # (hidden_dim,)

    def register(self, model) -> "ProbeSteeringHook":
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        vec = self._vec_cpu.to(device=device, dtype=dtype)

        pos = self.token_pos

        def _hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:, pos, :] = hidden[:, pos, :] + vec
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        self._handle = model.model.layers[self.layer].register_forward_hook(_hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @classmethod
    def from_file(
        cls,
        layer: int,
        target: str,
        alpha: float,
        processed_dir: str | Path = "data/processed",
        dataset: str = "scifact",
        token_pos: int = -1,
    ) -> "ProbeSteeringHook":
        """Load probe weight from disk and construct the hook."""
        path = Path(processed_dir) / dataset / "probe_weights" / f"layer_{layer}_{target}.npy"
        w = np.load(path)
        return cls(layer=layer, probe_weight=w, alpha=alpha, token_pos=token_pos)


@contextmanager
def probe_steering_context(model, layer: int, probe_weight: np.ndarray, alpha: float):
    """Context manager that registers and auto-removes a probe steering hook."""
    hook = ProbeSteeringHook(layer=layer, probe_weight=probe_weight, alpha=alpha)
    hook.register(model)
    try:
        yield hook
    finally:
        hook.remove()


class SAEFeatureHook:
    """Modifies specific SAE feature activations at the decision token.

    Two modes:
      "ablate"    — subtracts the feature's contribution:
                    h_new = h - f_val * decoder_col
      "amplify"   — adds extra copies of the feature direction:
                    h_new = h + alpha * decoder_col

    where f_val is the current activation of the feature (from SAE encode),
    and decoder_col is the unit-normed decoder column for that feature.
    """

    def __init__(
        self,
        layer: int,
        sae,
        feature_indices: list[int],
        mode: str,          # "ablate" or "amplify"
        alpha: float = 3.0,
        token_pos: int = -1,
    ) -> None:
        if mode not in ("ablate", "amplify"):
            raise ValueError(f"mode must be 'ablate' or 'amplify', got {mode!r}")
        self.layer = layer
        self.sae = sae
        self.feature_indices = feature_indices
        self.mode = mode
        self.alpha = alpha
        self.token_pos = token_pos
        self._handle = None

    def register(self, model) -> "SAEFeatureHook":
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        sae = self.sae.to(device=device, dtype=dtype)
        feature_indices = self.feature_indices
        mode = self.mode
        alpha = self.alpha
        pos = self.token_pos

        def _hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h = hidden[:, pos, :].float()  # (batch, hidden_dim)

            with torch.no_grad():
                sparse = sae.encode(h)  # (batch, sae_hidden_dim)
                # decoder weight: (input_dim, sae_hidden_dim)
                dec_cols = sae.decoder.weight  # (input_dim, sae_hidden_dim)

                for feat_idx in feature_indices:
                    col = dec_cols[:, feat_idx]  # (input_dim,) unit-normed
                    if mode == "ablate":
                        # Remove each sample's current activation of this feature
                        f_vals = sparse[:, feat_idx]  # (batch,)
                        h = h - f_vals.unsqueeze(1) * col.unsqueeze(0)
                    else:  # amplify
                        h = h + alpha * col.unsqueeze(0)

            hidden[:, pos, :] = h.to(dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        self._handle = model.model.layers[self.layer].register_forward_hook(_hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @classmethod
    def from_checkpoint(
        cls,
        layer: int,
        feature_indices: list[int],
        mode: str,
        alpha: float = 3.0,
        checkpoint_dir: str | Path = "outputs/final/sae_checkpoints",
        dataset: str = "scifact",
        token_pos: int = -1,
    ) -> "SAEFeatureHook":
        from src.sae.model import TopKSAE
        import json

        ckpt_dir = Path(checkpoint_dir) / dataset / f"layer{layer}"
        meta_path = ckpt_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        expansion_factor = meta["hidden_dim"] // meta["input_dim"]
        sae = TopKSAE(
            input_dim=meta["input_dim"],
            expansion_factor=expansion_factor,
            k=meta["k"],
        )
        sae.load_state_dict(torch.load(ckpt_dir / "sae.pt", map_location="cpu", weights_only=True))
        sae.eval()
        return cls(
            layer=layer,
            sae=sae,
            feature_indices=feature_indices,
            mode=mode,
            alpha=alpha,
            token_pos=token_pos,
        )


@contextmanager
def sae_feature_context(model, layer, sae, feature_indices, mode, alpha=3.0):
    """Context manager that registers and auto-removes an SAE feature hook."""
    hook = SAEFeatureHook(layer=layer, sae=sae, feature_indices=feature_indices,
                          mode=mode, alpha=alpha)
    hook.register(model)
    try:
        yield hook
    finally:
        hook.remove()
