"""
Shared dataclasses and type aliases for the project.

Every downstream module imports types from here to stay consistent.
"""

from dataclasses import dataclass
from typing import Literal

DatasetName = Literal["scifact", "nfcorpus"]

# Raw BEIR types
BeirCorpus = dict[str, dict[str, str]]   # doc_id -> {"title": ..., "text": ...}
BeirQueries = dict[str, str]              # query_id -> query_text
BeirQrels = dict[str, dict[str, int]]    # query_id -> {doc_id: relevance_label}


@dataclass
class BeirDataset:
    """Container for a loaded BEIR dataset."""
    name: DatasetName
    corpus: BeirCorpus
    queries: BeirQueries
    qrels: BeirQrels


@dataclass
class QueryDocPair:
    """
    A single (query, document) pair with labels and metadata.

    This is the atomic unit flowing through the entire pipeline.
    """
    query_id: str
    doc_id: str
    query_text: str
    doc_title: str
    doc_text: str
    relevance_label: int    # raw BEIR label: SciFact 0/1, NFCorpus 0/1/2
    is_relevant: bool       # label >= relevance_threshold (default 1)
    dataset: DatasetName
