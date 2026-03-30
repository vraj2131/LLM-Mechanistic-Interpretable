"""
Document-level features.

Features:
  doc_length         — total token count (title + text, whitespace tokenized)
  doc_length_bucket  — ordinal bucket 0-3 (Short/Medium/Long/VeryLong)
                       boundaries fit on the primary dataset (SciFact) so
                       NFCorpus uses the same fixed thresholds for comparability.

Fixed SciFact quartile boundaries (fit once, reused everywhere):
  bucket 0: length < Q1
  bucket 1: Q1 <= length < Q2  (median)
  bucket 2: Q2 <= length < Q3
  bucket 3: length >= Q3
"""

from __future__ import annotations

import re

import numpy as np

# Fixed quartile boundaries computed on SciFact corpus (title+text, whitespace tokens).
# Recomputed here at import time from the data — but cached so it only runs once.
_BOUNDARIES: list[float] | None = None
_BUCKET_LABELS = ["Short", "Medium", "Long", "VeryLong"]


def _tokenize(text: str) -> list[str]:
    return re.split(r"\s+", text.lower().strip())


def doc_length(title: str, text: str) -> int:
    """Token count of title + text concatenated."""
    combined = f"{title} {text}".strip()
    tokens = [t for t in _tokenize(combined) if t]
    return len(tokens)


def fit_boundaries(pairs_df) -> list[float]:
    """Compute Q1/Q2/Q3 doc length boundaries from a pairs DataFrame.

    Should be called once on the SciFact pair table and the result reused
    for NFCorpus to ensure consistent bucketing.

    Returns:
        [q1, q2, q3] as floats.
    """
    lengths = [
        doc_length(row.doc_title, row.doc_text)
        for row in pairs_df.itertuples(index=False)
    ]
    q1, q2, q3 = np.percentile(lengths, [25, 50, 75])
    return [float(q1), float(q2), float(q3)]


def length_to_bucket(length: int, boundaries: list[float]) -> int:
    """Map a doc length to bucket 0-3 using precomputed boundaries."""
    q1, q2, q3 = boundaries
    if length < q1:
        return 0
    elif length < q2:
        return 1
    elif length < q3:
        return 2
    else:
        return 3


def compute_document_features(
    pairs_df,
    boundaries: list[float] | None = None,
) -> dict[str, list]:
    """Compute doc_length and doc_length_bucket for all rows.

    Args:
        pairs_df: DataFrame with columns doc_title, doc_text.
        boundaries: [q1, q2, q3] from fit_boundaries(). If None, fits on
                    pairs_df itself (use only for the primary dataset).

    Returns:
        Dict with keys "doc_length" and "doc_length_bucket".
    """
    lengths = [
        doc_length(row.doc_title, row.doc_text)
        for row in pairs_df.itertuples(index=False)
    ]

    if boundaries is None:
        q1, q2, q3 = np.percentile(lengths, [25, 50, 75])
        boundaries = [float(q1), float(q2), float(q3)]

    buckets = [length_to_bucket(l, boundaries) for l in lengths]

    return {"doc_length": lengths, "doc_length_bucket": buckets}
