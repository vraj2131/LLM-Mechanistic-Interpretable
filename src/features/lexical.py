"""
Lexical overlap features between query and document.

Uses whitespace tokenization — identical to the BM25 index tokenizer —
so features are directly comparable to BM25 scores.

Features:
  lexical_overlap  — |query_terms ∩ doc_terms| / |query_terms|
                     (recall-oriented: what fraction of query terms appear in doc)
  query_term_freq  — mean TF of query terms in document
                     (frequency-oriented: how often query terms appear on average)
"""

from __future__ import annotations

import re

import numpy as np


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization — matches BM25 index tokenizer."""
    return re.split(r"\s+", text.lower().strip())


def lexical_overlap(query: str, doc_text: str) -> float:
    """Fraction of unique query terms that appear at least once in the document.

    Returns value in [0, 1]. Returns 0.0 if query is empty.
    """
    q_terms = set(_tokenize(query))
    q_terms.discard("")
    if not q_terms:
        return 0.0

    d_terms = set(_tokenize(doc_text))
    return len(q_terms & d_terms) / len(q_terms)


def query_term_freq(query: str, doc_text: str) -> float:
    """Mean term frequency of query terms in the document.

    TF = count of term in doc / total doc tokens.
    Returns the mean TF across all unique query terms.
    Returns 0.0 if query or doc is empty.
    """
    q_terms = list(set(_tokenize(query)))
    q_terms = [t for t in q_terms if t]
    if not q_terms:
        return 0.0

    d_tokens = _tokenize(doc_text)
    d_len = len(d_tokens)
    if d_len == 0:
        return 0.0

    from collections import Counter
    d_counts = Counter(d_tokens)
    tfs = [d_counts.get(t, 0) / d_len for t in q_terms]
    return float(np.mean(tfs))


def compute_lexical_features(pairs_df) -> dict[str, list]:
    """Compute lexical_overlap and query_term_freq for all rows in pairs_df.

    Args:
        pairs_df: DataFrame with columns query_text, doc_title, doc_text.

    Returns:
        Dict with keys "lexical_overlap" and "query_term_freq", each a list
        aligned with pairs_df rows.
    """
    overlaps = []
    qtfs = []

    for row in pairs_df.itertuples(index=False):
        # Concatenate title + text to match BM25 index construction
        doc = f"{row.doc_title} {row.doc_text}".strip()
        overlaps.append(lexical_overlap(row.query_text, doc))
        qtfs.append(query_term_freq(row.query_text, doc))

    return {"lexical_overlap": overlaps, "query_term_freq": qtfs}
