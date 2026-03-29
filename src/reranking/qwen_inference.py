"""
Batched Qwen2.5-1.5B-Instruct inference for pointwise reranking.

Uses a single forward pass per batch (no autoregressive generation).
Extracts logits at the decision token position and computes scores from
the probability distribution over score tokens ("0", "1", "2", "3").

This approach is:
  - ~8-10x faster than model.generate() (1 forward pass vs ~8)
  - Scientifically stronger (full P(0), P(1), P(2), P(3) distribution)
  - Compatible with Phase 4 hooks (activations extracted in same pass)

Usage:
    from src.reranking.qwen_inference import load_model, score_pairs
    model, tokenizer = load_model()
    result = score_pairs(prompts, model, tokenizer, batch_size=8)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger(__name__)


def load_model(
    model_id: str | None = None,
    dtype: str | None = None,
    device: str | None = None,
    cfg=None,
) -> tuple:
    """Load Qwen2.5-1.5B-Instruct and its tokenizer.

    Args:
        model_id: HuggingFace model ID. Defaults to configs/reranker.yaml value.
        dtype: "float16" or "float32". Defaults to configs/reranker.yaml value.
        device: "cuda", "mps", or "cpu". Auto-detects if None.
        cfg: Pre-loaded OmegaConf config (avoids re-reading yaml).

    Returns:
        (model, tokenizer) tuple ready for inference.
    """
    if cfg is None:
        cfg = load_config("configs/reranker.yaml")

    model_id = model_id or cfg.model_id
    dtype_str = dtype or cfg.dtype

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # MPS (Apple Silicon) works best with float32; float16 can cause NaN on some ops
    if device == "mps" and dtype_str == "float16":
        log.info("MPS detected — using float32 for numerical stability")
        dtype_str = "float32"

    torch_dtype = torch.float16 if dtype_str == "float16" else torch.float32
    log.info(f"Loading {model_id} on {device} ({dtype_str}) ...")

    # HuggingFace caches downloads in ~/.cache/huggingface/hub/ automatically.
    # Models are never re-downloaded once cached.
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map only works reliably with CUDA/accelerate.
    # For MPS and CPU, load to CPU first then move to device.
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
        ).to(device)

    model.eval()
    log.info(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")
    return model, tokenizer


def _get_score_token_ids(tokenizer) -> list[int]:
    """Return token IDs for "0", "1", "2", "3" in order."""
    ids = []
    for digit in ["0", "1", "2", "3"]:
        token_id = tokenizer.encode(digit, add_special_tokens=False)
        # Some tokenizers may produce multiple tokens; take the first
        ids.append(token_id[0])
    return ids


def score_pairs(
    prompts: list[str],
    model,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 900,
) -> dict:
    """Score prompts via single forward pass + logit extraction.

    For each prompt, extracts the logits at the decision token position
    (last token of the input) and computes a softmax over tokens "0"-"3".
    The argmax is the predicted score; the expected value (weighted sum)
    provides a continuous score for ranking.

    Args:
        prompts: List of fully formatted prompt strings.
        model: Loaded HuggingFace CausalLM model.
        tokenizer: Corresponding tokenizer (padding_side="left").
        batch_size: Number of prompts per forward pass.
        max_length: Max tokenized sequence length.

    Returns:
        dict with keys:
          - "scores": list[int]        — argmax score per prompt (0-3)
          - "expected_scores": list[float] — weighted expected score per prompt
          - "probs": list[list[float]]  — P(0), P(1), P(2), P(3) per prompt
          - "input_ids_list": list[Tensor] — input_ids per batch (for Phase 4)
    """
    score_token_ids = _get_score_token_ids(tokenizer)
    score_values = torch.arange(4, dtype=torch.float32)  # [0, 1, 2, 3]

    all_scores: list[int] = []
    all_expected: list[float] = []
    all_probs: list[list[float]] = []
    input_ids_list: list[torch.Tensor] = []

    device = next(model.parameters()).device
    is_mps = device.type == "mps"

    for i in tqdm(range(0, len(prompts), batch_size), desc="Reranker inference", unit="batch"):
        batch_prompts = prompts[i : i + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        input_ids_list.append(enc["input_ids"].cpu())

        with torch.no_grad():
            # logits_to_keep=1 → only compute logits for the last token.
            # Avoids materializing the full (batch × seq_len × 152K) tensor
            # which causes OOM on MPS at batch_size >= 4.
            outputs = model(**enc, logits_to_keep=1)
            logits = outputs.logits  # (batch, 1, vocab_size)

        # logits_to_keep=1 returns shape (batch, 1, vocab_size)
        decision_logits = logits[:, -1, :]  # (batch, vocab_size)

        # Extract logits for score tokens only and compute softmax
        score_logits = decision_logits[:, score_token_ids]  # (batch, 4)
        probs = F.softmax(score_logits.float(), dim=-1)  # (batch, 4)

        # Argmax score (discrete 0-3)
        scores = probs.argmax(dim=-1)  # (batch,)

        # Expected score (continuous, better for ranking)
        expected = (probs * score_values.to(probs.device)).sum(dim=-1)  # (batch,)

        all_scores.extend(scores.cpu().tolist())
        all_expected.extend(expected.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        # Free MPS memory between batches to prevent progressive slowdown
        if is_mps:
            torch.mps.empty_cache()

    log.info(f"Inference complete: {len(all_scores)} pairs scored")
    return {
        "scores": all_scores,
        "expected_scores": all_expected,
        "probs": all_probs,
        "input_ids_list": input_ids_list,
    }
