"""Unit tests for BM25 index, retrieval, and evaluation metrics."""

import json
import tempfile
from pathlib import Path

import pytest

from src.retrieval.bm25_index import build_index, load_index
from src.retrieval.bm25_retriever import retrieve
from src.evaluation.metrics import ndcg_at_k, mrr_at_k, recall_at_k, compute_all_metrics


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_corpus():
    return {
        "d1": {"title": "cats", "text": "cats and dogs are pets"},
        "d2": {"title": "physics", "text": "quantum mechanics and relativity"},
        "d3": {"title": "ml",     "text": "deep learning neural networks"},
        "d4": {"title": "animals","text": "dogs cats birds are animals"},
    }


@pytest.fixture
def tiny_queries():
    return {
        "q1": "cats dogs pets",
        "q2": "quantum physics",
        "q3": "neural networks learning",
    }


@pytest.fixture
def tiny_qrels():
    return {
        "q1": {"d1": 1, "d4": 1},
        "q2": {"d2": 1},
        "q3": {"d3": 1},
    }


@pytest.fixture
def index_and_ids(tiny_corpus, tmp_path):
    index, ids = build_index(tiny_corpus, cache_dir=tmp_path / "idx")
    return index, ids


# ---------------------------------------------------------------------------
# BM25 index tests
# ---------------------------------------------------------------------------

class TestBM25Index:
    def test_build_creates_files(self, tiny_corpus, tmp_path):
        cache = tmp_path / "idx"
        build_index(tiny_corpus, cache_dir=cache)
        assert (cache / "bm25_index.pkl").exists()
        assert (cache / "corpus_ids.json").exists()

    def test_corpus_ids_ordering(self, tiny_corpus, tmp_path):
        cache = tmp_path / "idx"
        _, ids = build_index(tiny_corpus, cache_dir=cache)
        assert set(ids) == set(tiny_corpus.keys())
        assert len(ids) == len(tiny_corpus)

    def test_load_roundtrip(self, tiny_corpus, tmp_path):
        cache = tmp_path / "idx"
        build_index(tiny_corpus, cache_dir=cache)
        index2, ids2 = load_index(cache)
        assert len(ids2) == len(tiny_corpus)

    def test_load_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_index(tmp_path / "nonexistent")

    def test_skip_rebuild_if_cached(self, tiny_corpus, tmp_path):
        cache = tmp_path / "idx"
        _, ids1 = build_index(tiny_corpus, cache_dir=cache)
        _, ids2 = build_index(tiny_corpus, cache_dir=cache, force=False)
        assert ids1 == ids2  # same object reloaded, not rebuilt


# ---------------------------------------------------------------------------
# BM25 retrieval tests
# ---------------------------------------------------------------------------

class TestBM25Retriever:
    def test_run_keys(self, index_and_ids, tiny_queries):
        index, ids = index_and_ids
        run, _ = retrieve(index, ids, tiny_queries, top_k=2)
        assert set(run.keys()) == set(tiny_queries.keys())

    def test_top_k_respected(self, index_and_ids, tiny_queries):
        index, ids = index_and_ids
        run, run_df = retrieve(index, ids, tiny_queries, top_k=2)
        for qid, docs in run.items():
            assert len(docs) <= 2
        assert run_df["bm25_rank"].max() <= 2

    def test_scores_descending(self, index_and_ids, tiny_queries):
        index, ids = index_and_ids
        _, run_df = retrieve(index, ids, tiny_queries, top_k=4)
        for qid, group in run_df.groupby("query_id"):
            scores = group.sort_values("bm25_rank")["bm25_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_retrieved(self, index_and_ids, tiny_queries):
        """d2 should score highest for the physics query."""
        index, ids = index_and_ids
        run, _ = retrieve(index, ids, tiny_queries, top_k=4)
        q2_ranked = sorted(run["q2"].items(), key=lambda x: x[1], reverse=True)
        assert q2_ranked[0][0] == "d2"

    def test_dataframe_columns(self, index_and_ids, tiny_queries):
        index, ids = index_and_ids
        _, run_df = retrieve(index, ids, tiny_queries, top_k=2)
        assert set(run_df.columns) == {"query_id", "doc_id", "bm25_score", "bm25_rank"}

    def test_rank_starts_at_1(self, index_and_ids, tiny_queries):
        index, ids = index_and_ids
        _, run_df = retrieve(index, ids, tiny_queries, top_k=3)
        assert run_df.groupby("query_id")["bm25_rank"].min().eq(1).all()


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.fixture
    def perfect_run(self, tiny_qrels):
        """Run where the first result is always the relevant doc."""
        return {
            "q1": {"d1": 2.0, "d4": 1.0, "d2": 0.5},
            "q2": {"d2": 3.0, "d1": 0.1},
            "q3": {"d3": 2.5, "d1": 0.1},
        }

    @pytest.fixture
    def worst_run(self, tiny_qrels):
        """Run where relevant docs are never in the top results."""
        return {
            "q1": {"d2": 2.0, "d3": 1.0},   # d1/d4 missing
            "q2": {"d1": 3.0, "d3": 0.5},   # d2 missing
            "q3": {"d1": 2.0, "d2": 0.5},   # d3 missing
        }

    def test_perfect_ndcg(self, perfect_run, tiny_qrels):
        score = ndcg_at_k(perfect_run, tiny_qrels, k=10)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_perfect_mrr(self, perfect_run, tiny_qrels):
        score = mrr_at_k(perfect_run, tiny_qrels, k=10)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_perfect_recall(self, perfect_run, tiny_qrels):
        score = recall_at_k(perfect_run, tiny_qrels, k=10)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_zero_mrr_when_no_relevant_retrieved(self, worst_run, tiny_qrels):
        score = mrr_at_k(worst_run, tiny_qrels, k=10)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_zero_recall_when_no_relevant_retrieved(self, worst_run, tiny_qrels):
        score = recall_at_k(worst_run, tiny_qrels, k=10)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_mrr_second_position(self, tiny_qrels):
        run = {"q2": {"d1": 2.0, "d2": 1.5}}   # d2 is relevant, ranked 2nd
        score = mrr_at_k(run, tiny_qrels, k=10)
        assert score == pytest.approx(0.5, abs=1e-6)

    def test_compute_all_metrics_keys(self, perfect_run, tiny_qrels):
        m = compute_all_metrics(perfect_run, tiny_qrels)
        assert set(m.keys()) == {"ndcg@10", "mrr@10", "recall@20"}

    def test_empty_run_returns_zero(self, tiny_qrels):
        m = compute_all_metrics({}, tiny_qrels)
        assert all(v == 0.0 for v in m.values())

    def test_cutoff_k_enforced(self, tiny_qrels):
        """Relevant doc at rank 2 should not contribute to MRR@1."""
        run = {"q2": {"d1": 2.0, "d2": 1.5}}
        score_k1 = mrr_at_k(run, tiny_qrels, k=1)
        score_k2 = mrr_at_k(run, tiny_qrels, k=2)
        assert score_k1 == pytest.approx(0.0)
        assert score_k2 == pytest.approx(0.5)
