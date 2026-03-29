"""Unit tests for src/data/pair_builder.py — no BEIR download required."""

import pandas as pd
import pytest

from src.data.schema import BeirDataset
from src.data.pair_builder import build_pairs_from_qrels, build_pairs_from_run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_dataset() -> BeirDataset:
    corpus = {
        "d1": {"title": "Doc One", "text": "cats and dogs"},
        "d2": {"title": "Doc Two", "text": "quantum mechanics"},
        "d3": {"title": "Doc Three", "text": "deep learning"},
    }
    queries = {
        "q1": "animal companions",
        "q2": "physics fundamentals",
    }
    qrels = {
        "q1": {"d1": 1, "d2": 0},
        "q2": {"d2": 1, "d3": 0},
    }
    return BeirDataset(name="scifact", corpus=corpus, queries=queries, qrels=qrels)


@pytest.fixture
def tiny_run_df() -> pd.DataFrame:
    """Simulates a BM25 top-2 retrieval run."""
    return pd.DataFrame([
        {"query_id": "q1", "doc_id": "d1", "bm25_score": 3.5, "bm25_rank": 1},
        {"query_id": "q1", "doc_id": "d3", "bm25_score": 1.2, "bm25_rank": 2},
        {"query_id": "q2", "doc_id": "d2", "bm25_score": 4.0, "bm25_rank": 1},
        {"query_id": "q2", "doc_id": "d1", "bm25_score": 0.5, "bm25_rank": 2},
    ])


# ---------------------------------------------------------------------------
# build_pairs_from_qrels tests
# ---------------------------------------------------------------------------

class TestBuildPairsFromQrels:
    def test_row_count(self, tiny_dataset):
        df = build_pairs_from_qrels(tiny_dataset)
        # q1->d1, q1->d2, q2->d2, q2->d3 = 4 pairs
        assert len(df) == 4

    def test_required_columns(self, tiny_dataset):
        df = build_pairs_from_qrels(tiny_dataset)
        expected = {"query_id", "doc_id", "query_text", "doc_title",
                    "doc_text", "relevance_label", "is_relevant", "dataset"}
        assert expected.issubset(set(df.columns))

    def test_is_relevant_threshold(self, tiny_dataset):
        df = build_pairs_from_qrels(tiny_dataset, relevance_threshold=1)
        # d1 (label=1) relevant for q1; d2 (label=1) relevant for q2
        assert df.loc[df["doc_id"] == "d1", "is_relevant"].values[0] == True
        assert df.loc[(df["query_id"] == "q1") & (df["doc_id"] == "d2"), "is_relevant"].values[0] == False

    def test_dataset_field(self, tiny_dataset):
        df = build_pairs_from_qrels(tiny_dataset)
        assert (df["dataset"] == "scifact").all()

    def test_text_content(self, tiny_dataset):
        df = build_pairs_from_qrels(tiny_dataset)
        row = df.loc[(df["query_id"] == "q1") & (df["doc_id"] == "d1")].iloc[0]
        assert row["query_text"] == "animal companions"
        assert row["doc_title"] == "Doc One"
        assert row["doc_text"] == "cats and dogs"

    def test_missing_doc_skipped(self, tiny_dataset):
        # Add a qrel entry pointing to a non-existent doc
        tiny_dataset.qrels["q1"]["d_missing"] = 1
        df = build_pairs_from_qrels(tiny_dataset)
        assert "d_missing" not in df["doc_id"].values

    def test_no_bm25_columns(self, tiny_dataset):
        df = build_pairs_from_qrels(tiny_dataset)
        assert "bm25_score" not in df.columns
        assert "bm25_rank" not in df.columns


# ---------------------------------------------------------------------------
# build_pairs_from_run tests
# ---------------------------------------------------------------------------

class TestBuildPairsFromRun:
    def test_row_count(self, tiny_dataset, tiny_run_df):
        df = build_pairs_from_run(tiny_dataset, tiny_run_df)
        assert len(df) == 4

    def test_bm25_columns_present(self, tiny_dataset, tiny_run_df):
        df = build_pairs_from_run(tiny_dataset, tiny_run_df)
        assert "bm25_score" in df.columns
        assert "bm25_rank" in df.columns

    def test_bm25_values_preserved(self, tiny_dataset, tiny_run_df):
        df = build_pairs_from_run(tiny_dataset, tiny_run_df)
        row = df.loc[(df["query_id"] == "q2") & (df["doc_id"] == "d2")].iloc[0]
        assert row["bm25_score"] == pytest.approx(4.0)
        assert row["bm25_rank"] == 1

    def test_labels_joined_from_qrels(self, tiny_dataset, tiny_run_df):
        df = build_pairs_from_run(tiny_dataset, tiny_run_df)
        # q1->d1 is in qrels with label=1
        row = df.loc[(df["query_id"] == "q1") & (df["doc_id"] == "d1")].iloc[0]
        assert row["relevance_label"] == 1
        assert row["is_relevant"] == True

    def test_unlabeled_doc_gets_zero(self, tiny_dataset, tiny_run_df):
        # q1->d3 is NOT in qrels, should get label=0
        df = build_pairs_from_run(tiny_dataset, tiny_run_df)
        row = df.loc[(df["query_id"] == "q1") & (df["doc_id"] == "d3")].iloc[0]
        assert row["relevance_label"] == 0
        assert row["is_relevant"] == False
