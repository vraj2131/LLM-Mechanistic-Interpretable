"""
Probe target definitions — maps feature names to probe configuration.

Each target specifies:
  - column: column name in features.parquet
  - probe_type: "ridge" (continuous) or "logistic" (binary/categorical)
  - metric: "r2" or "auroc"
  - description: human-readable label for plots
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    name: str
    column: str
    probe_type: str   # "ridge" or "logistic"
    metric: str       # "r2" or "auroc"
    description: str


# Canonical probe targets — order determines heatmap row order
PROBE_TARGETS: list[ProbeTarget] = [
    ProbeTarget(
        name="lexical_overlap",
        column="lexical_overlap",
        probe_type="ridge",
        metric="r2",
        description="Lexical Overlap",
    ),
    ProbeTarget(
        name="query_term_freq",
        column="query_term_freq",
        probe_type="ridge",
        metric="r2",
        description="Query Term Freq",
    ),
    ProbeTarget(
        name="bm25_score",
        column="bm25_score",
        probe_type="ridge",
        metric="r2",
        description="BM25 Score",
    ),
    ProbeTarget(
        name="bm25_rank",
        column="bm25_rank",
        probe_type="ridge",
        metric="r2",
        description="BM25 Rank",
    ),
    ProbeTarget(
        name="doc_length_bucket",
        column="doc_length_bucket",
        probe_type="logistic",
        metric="auroc",
        description="Doc Length Bucket",
    ),
    ProbeTarget(
        name="is_relevant",
        column="is_relevant",
        probe_type="logistic",
        metric="auroc",
        description="Is Relevant",
    ),
    ProbeTarget(
        name="relevance_label",
        column="relevance_label",
        probe_type="ridge",
        metric="r2",
        description="Relevance Label",
    ),
]

TARGET_NAMES = [t.name for t in PROBE_TARGETS]
TARGET_BY_NAME = {t.name: t for t in PROBE_TARGETS}
