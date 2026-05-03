"""
Fast intervention scorer using split forward pass.

Instead of running all 28 layers per condition:
  Old: N_conditions × N_batches × 28 layers
  New: N_batches × (split_layer+1 layers  +  N_conditions × (28-split_layer) layers)

For split_layer=17, N_conditions=82:
  Old: 82 × 750 × 28  = 1,722,000 layer-passes
  New: 750 × (18 + 82×10) = 750 × 838 = 628,500 layer-passes  (~2.7x faster)

Usage:
    scorer = FastInterventionScorer(model, tokenizer, split_layer=17)
    scorer.prepare(prompts, pairs_df)          # tokenize once
    baseline = scorer.score_baseline()
    steered  = scorer.score_with_probe(w, alpha=3.0)
    steered  = scorer.score_with_sae_feature(sae, feat_idx=30, mode="ablate")
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.logging import get_logger

log = get_logger(__name__)


class FastInterventionScorer:
    """Split-forward-pass scorer for causal intervention experiments.

    The model forward pass is split at ``split_layer``:
      - Phase 1 (run once per batch): layers 0 … split_layer  → h_split
      - Phase 2 (run per condition):  perturb h_split → layers split_layer+1 … N-1 → score
    """

    def __init__(
        self,
        model,
        tokenizer,
        split_layer: int = 17,
        batch_size: int = 8,
        max_length: int = 700,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.split_layer = split_layer
        self.batch_size = batch_size
        self.max_length = max_length

        self.device = next(model.parameters()).device
        self.dtype  = next(model.parameters()).dtype

        self._model_internals = model.model
        self._n_layers = len(model.model.layers)
        self._score_token_ids = self._get_score_token_ids()

        # Populated by prepare()
        self._batches: list[dict] = []
        self._n_pairs: int = 0
        self._pair_index: list[tuple] = []  # (query_id, doc_id) per pair

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _get_score_token_ids(self) -> list[int]:
        ids = []
        for digit in ["0", "1", "2", "3"]:
            tok = self.tokenizer.encode(digit, add_special_tokens=False)
            ids.append(tok[0])
        return ids

    def prepare(self, prompts: list[str], pairs_df) -> None:
        """Tokenize all prompts once and store as batched tensors."""
        log.info(f"Pre-tokenising {len(prompts)} prompts (batch_size={self.batch_size}) ...")
        self._n_pairs = len(prompts)
        self._pair_index = list(zip(pairs_df["query_id"], pairs_df["doc_id"]))
        self._batches = []

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            # Store input_ids on CPU — moved to device per batch
            self._batches.append({"input_ids": enc["input_ids"]})
        log.info(f"Prepared {len(self._batches)} batches.")

    # ------------------------------------------------------------------
    # Internal forward-pass helpers
    # ------------------------------------------------------------------

    def _embed_and_rope(self, input_ids: torch.Tensor):
        """Embed tokens and compute RoPE. Returns (hidden, position_ids, position_embeddings)."""
        mi = self._model_internals
        hidden = mi.embed_tokens(input_ids)

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(bs, -1)

        # RoPE — transformers 5.x computes cos/sin at model level
        position_embeddings = mi.rotary_emb(hidden, position_ids)

        return hidden, position_ids, position_embeddings

    def _run_layers(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Run decoder layers [start, end) and return hidden state.

        attention_mask=None: SDPA handles causal masking internally via is_causal=True.
        Safe here because we use left-padding and only read the last token position.
        """
        for i in range(start, end):
            out = self._model_internals.layers[i](
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden

    def _logits_to_expected_score(self, hidden: torch.Tensor) -> list[float]:
        """Run norm + lm_head on last token, return expected score per sample."""
        h = self._model_internals.norm(hidden)
        # Only compute logits for the last token
        logits = self.model.lm_head(h[:, -1:, :])          # (batch, 1, vocab)
        score_logits = logits[:, 0, self._score_token_ids]  # (batch, 4)
        probs = F.softmax(score_logits.float(), dim=-1)
        vals  = torch.arange(4, dtype=torch.float32, device=probs.device)
        return (probs * vals).sum(dim=-1).cpu().tolist()

    # ------------------------------------------------------------------
    # Core: run all conditions in one pass over the data
    # ------------------------------------------------------------------

    def score_conditions(
        self,
        conditions: list[tuple[str, Callable | None]],
        desc: str = "Scoring",
    ) -> dict[str, list[float]]:
        """Score all (label, perturbation_fn) conditions in a single data pass.

        perturbation_fn(h_split) modifies h_split[:, -1, :] in-place.
        Pass None for the baseline (no modification).

        Returns:
            {label: [expected_score_per_pair]}
        """
        all_scores: dict[str, list[float]] = {label: [] for label, _ in conditions}

        for batch in tqdm(self._batches, desc=desc, unit="batch"):
            input_ids = batch["input_ids"].to(self.device)

            with torch.no_grad():
                # Phase 1: layers 0 … split_layer (run once per batch)
                hidden, position_ids, position_embeddings = self._embed_and_rope(input_ids)
                h_split = self._run_layers(
                    hidden, position_ids, position_embeddings,
                    start=0, end=self.split_layer + 1,
                )

                # Phase 2: per-condition perturbation + upper layers
                for label, perturb_fn in conditions:
                    h = h_split.clone()
                    if perturb_fn is not None:
                        perturb_fn(h)
                    h = self._run_layers(
                        h, position_ids, position_embeddings,
                        start=self.split_layer + 1, end=self._n_layers,
                    )
                    scores = self._logits_to_expected_score(h)
                    all_scores[label].extend(scores)

            # Free MPS memory between batches
            if self.device.type == "mps":
                torch.mps.empty_cache()

        return all_scores

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def score_baseline(self) -> list[float]:
        result = self.score_conditions([("baseline", None)], desc="Baseline")
        return result["baseline"]

    def score_with_probe(
        self,
        probe_weight: np.ndarray,
        alpha: float,
        label: str | None = None,
    ) -> list[float]:
        w = torch.from_numpy(alpha * probe_weight.astype(np.float32)).to(
            device=self.device, dtype=self.dtype
        )
        def perturb(h): h[:, -1, :] = h[:, -1, :] + w
        lbl = label or f"probe_α={alpha:+.1f}"
        result = self.score_conditions([(lbl, perturb)], desc=lbl)
        return result[lbl]

    def score_probe_sweep(
        self,
        probe_weight: np.ndarray,
        alphas: list[float],
        target: str,
        layer: int,
    ) -> dict[str, list[float]]:
        """Score all alphas for one (target, layer) probe in a single data pass."""
        conditions = []
        for alpha in alphas:
            w = torch.from_numpy(alpha * probe_weight.astype(np.float32)).to(
                device=self.device, dtype=self.dtype
            )
            w_captured = w  # capture in closure

            def make_perturb(vec):
                def perturb(h): h[:, -1, :] = h[:, -1, :] + vec
                return perturb

            conditions.append((f"α={alpha:+.1f}", make_perturb(w_captured)))

        desc = f"Probe {target} layer={layer}"
        return self.score_conditions(conditions, desc=desc)

    def score_sae_feature(
        self,
        sae,
        feature_idx: int,
        mode: str,
        alpha: float = 3.0,
        label: str | None = None,
    ) -> list[float]:
        sae_dev = sae.to(device=self.device, dtype=self.dtype)
        dec_col = sae_dev.decoder.weight[:, feature_idx].detach()  # (input_dim,)

        if mode == "ablate":
            def perturb(h):
                x = h[:, -1, :].to(self.dtype)
                with torch.no_grad():
                    sparse = sae_dev.encode(x)
                f_vals = sparse[:, feature_idx]
                h[:, -1, :] = x - f_vals.unsqueeze(1) * dec_col.unsqueeze(0)
        else:  # amplify
            def perturb(h):
                h[:, -1, :] = h[:, -1, :] + (alpha * dec_col).to(self.dtype)

        lbl = label or f"sae_feat{feature_idx}_{mode}_α{alpha}"
        result = self.score_conditions([(lbl, perturb)], desc=lbl)
        return result[lbl]

    def score_sae_sweep(
        self,
        sae,
        feature_idx: int,
        modes: list[str],
        amplify_alphas: list[float],
        ir_target: str,
    ) -> dict[str, list[float]]:
        """Score all (mode, alpha) combos for one SAE feature in a single data pass."""
        sae_dev = sae.to(device=self.device, dtype=self.dtype)
        dec_col = sae_dev.decoder.weight[:, feature_idx].detach()

        conditions = []
        for mode in modes:
            alphas_for_mode = amplify_alphas if mode == "amplify" else [1.0]
            for alpha in alphas_for_mode:
                if mode == "ablate":
                    def make_ablate():
                        def perturb(h):
                            x = h[:, -1, :].float()
                            with torch.no_grad():
                                sparse = sae_dev.encode(x)
                            f_vals = sparse[:, feature_idx]
                            h[:, -1, :] = (x - f_vals.unsqueeze(1) * dec_col.unsqueeze(0)).to(self.dtype)
                        return perturb
                    conditions.append((f"{mode}", make_ablate()))
                else:
                    def make_amplify(a):
                        def perturb(h):
                            h[:, -1, :] = h[:, -1, :] + (a * dec_col).to(self.dtype)
                        return perturb
                    conditions.append((f"{mode}_α{alpha}", make_amplify(alpha)))

        desc = f"SAE feat={feature_idx} ({ir_target})"
        return self.score_conditions(conditions, desc=desc)
