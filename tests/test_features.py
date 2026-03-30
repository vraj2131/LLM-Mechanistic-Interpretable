"""Unit tests for feature engineering — no GPU, no model loading."""

import pandas as pd
import pytest

from src.features.lexical import lexical_overlap, query_term_freq, compute_lexical_features
from src.features.document import doc_length, fit_boundaries, length_to_bucket, compute_document_features


# ---------------------------------------------------------------------------
# lexical_overlap
# ---------------------------------------------------------------------------

class TestLexicalOverlap:
    def test_perfect_overlap(self):
        assert lexical_overlap("cat dog", "cat dog bird") == 1.0

    def test_zero_overlap(self):
        assert lexical_overlap("cat dog", "fish bird") == 0.0

    def test_partial_overlap(self):
        result = lexical_overlap("cat dog fish", "cat bird tree")
        assert abs(result - 1/3) < 1e-6

    def test_empty_query(self):
        assert lexical_overlap("", "some document text") == 0.0

    def test_case_insensitive(self):
        assert lexical_overlap("CAT DOG", "cat dog") == 1.0

    def test_duplicate_query_terms(self):
        # "cat cat" → unique terms = {"cat"}, found in doc → 1.0
        assert lexical_overlap("cat cat", "cat bird") == 1.0

    def test_single_term_match(self):
        assert lexical_overlap("neural", "neural networks are powerful") == 1.0

    def test_single_term_no_match(self):
        assert lexical_overlap("quantum", "neural networks are powerful") == 0.0


# ---------------------------------------------------------------------------
# query_term_freq
# ---------------------------------------------------------------------------

class TestQueryTermFreq:
    def test_empty_query(self):
        assert query_term_freq("", "some document") == 0.0

    def test_empty_doc(self):
        assert query_term_freq("cat", "") == 0.0

    def test_term_appears_once(self):
        # "cat" appears 1/3 tokens
        result = query_term_freq("cat", "cat dog bird")
        assert abs(result - 1/3) < 1e-6

    def test_term_appears_multiple(self):
        # "cat" appears 2/4 tokens
        result = query_term_freq("cat", "cat cat dog bird")
        assert abs(result - 2/4) < 1e-6

    def test_multiple_query_terms_mean(self):
        # "cat": 1/3, "dog": 1/3 → mean = 1/3
        result = query_term_freq("cat dog", "cat dog bird")
        assert abs(result - 1/3) < 1e-6

    def test_term_not_in_doc(self):
        result = query_term_freq("fish", "cat dog bird")
        assert result == 0.0


# ---------------------------------------------------------------------------
# doc_length
# ---------------------------------------------------------------------------

class TestDocLength:
    def test_basic_length(self):
        assert doc_length("title", "one two three") == 4

    def test_empty_title(self):
        assert doc_length("", "one two three") == 3

    def test_empty_both(self):
        assert doc_length("", "") == 0

    def test_title_added_to_text(self):
        assert doc_length("hello world", "foo bar") == 4


# ---------------------------------------------------------------------------
# fit_boundaries / length_to_bucket
# ---------------------------------------------------------------------------

class TestDocLengthBucket:
    def test_bucket_boundaries(self):
        boundaries = [100.0, 200.0, 300.0]
        assert length_to_bucket(50, boundaries) == 0
        assert length_to_bucket(100, boundaries) == 1
        assert length_to_bucket(200, boundaries) == 2
        assert length_to_bucket(300, boundaries) == 3
        assert length_to_bucket(500, boundaries) == 3

    def test_fit_boundaries_returns_three(self):
        df = pd.DataFrame([
            {"doc_title": "", "doc_text": " ".join(["w"] * l)}
            for l in [10, 20, 30, 40, 50, 60, 70, 80, 100, 200]
        ])
        b = fit_boundaries(df)
        assert len(b) == 3
        assert b[0] < b[1] < b[2]

    def test_four_distinct_buckets(self):
        df = pd.DataFrame([
            {"doc_title": "", "doc_text": " ".join(["w"] * l)}
            for l in range(1, 101)
        ])
        b = fit_boundaries(df)
        result = compute_document_features(df, boundaries=b)
        buckets = set(result["doc_length_bucket"])
        assert buckets == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# compute_lexical_features (DataFrame interface)
# ---------------------------------------------------------------------------

class TestComputeLexicalFeatures:
    def test_output_keys(self):
        df = pd.DataFrame([
            {"query_text": "cat dog", "doc_title": "Animals", "doc_text": "cat and dog"},
            {"query_text": "quantum physics", "doc_title": "Science", "doc_text": "relativity"},
        ])
        result = compute_lexical_features(df)
        assert "lexical_overlap" in result
        assert "query_term_freq" in result
        assert len(result["lexical_overlap"]) == 2
        assert len(result["query_term_freq"]) == 2

    def test_values_in_range(self):
        df = pd.DataFrame([
            {"query_text": "neural network", "doc_title": "", "doc_text": "neural network model"},
        ])
        result = compute_lexical_features(df)
        assert 0.0 <= result["lexical_overlap"][0] <= 1.0
        assert result["query_term_freq"][0] >= 0.0
