"""Unit tests for ScoreParser."""

import pytest

from src.reranking.score_parser import ScoreParser


@pytest.fixture
def parser():
    return ScoreParser()


class TestScoreParserParse:
    def test_bare_digit_0(self, parser):
        assert parser.parse("0") == 0

    def test_bare_digit_3(self, parser):
        assert parser.parse("3") == 3

    def test_digit_with_newline(self, parser):
        assert parser.parse("2\n") == 2

    def test_digit_with_leading_space(self, parser):
        assert parser.parse("  1 ") == 1

    def test_digit_after_label(self, parser):
        assert parser.parse("Score: 3.") == 3

    def test_first_digit_wins(self, parser):
        # "2 or 3" — first match wins
        assert parser.parse("2 or 3") == 2

    def test_digit_4_not_valid(self, parser):
        # 4 is outside [0-3], should fallback
        assert parser.parse("4") == 0

    def test_fallback_on_no_digit(self, parser):
        assert parser.parse("not a number") == 0

    def test_fallback_on_empty_string(self, parser):
        assert parser.parse("") == 0

    def test_fallback_on_whitespace(self, parser):
        assert parser.parse("   ") == 0

    def test_digit_beyond_scan_window_ignored(self, parser):
        # Put a valid digit beyond the 20-char scan window
        long_prefix = "x" * 25 + "2"
        assert parser.parse(long_prefix) == 0  # fallback — digit not seen

    def test_custom_fallback_score(self):
        p = ScoreParser(fallback_score=1)
        assert p.parse("invalid") == 1


class TestScoreParserStats:
    def test_fallback_rate_zero_when_all_valid(self, parser):
        parser.parse("1")
        parser.parse("2")
        assert parser.fallback_rate() == 0.0

    def test_fallback_rate_one_when_all_invalid(self, parser):
        parser.parse("invalid")
        parser.parse("also invalid")
        assert parser.fallback_rate() == 1.0

    def test_fallback_rate_partial(self, parser):
        parser.parse("1")   # valid
        parser.parse("bad") # fallback
        assert parser.fallback_rate() == pytest.approx(0.5)

    def test_total_parsed_count(self, parser):
        for _ in range(5):
            parser.parse("1")
        assert parser.total_parsed == 5

    def test_fallback_rate_zero_before_any_call(self):
        p = ScoreParser()
        assert p.fallback_rate() == 0.0

    def test_reset_stats(self, parser):
        parser.parse("bad")
        parser.reset_stats()
        assert parser.total_parsed == 0
        assert parser.total_fallbacks == 0
        assert parser.fallback_rate() == 0.0


class TestScoreParserBatch:
    def test_batch_length_matches(self, parser):
        outputs = ["1", "2", "bad", "3", "0"]
        scores = parser.parse_batch(outputs)
        assert len(scores) == len(outputs)

    def test_batch_values_correct(self, parser):
        outputs = ["1", "bad", "3"]
        scores = parser.parse_batch(outputs)
        assert scores == [1, 0, 3]

    def test_batch_updates_stats(self, parser):
        parser.parse_batch(["1", "invalid", "2", "nope"])
        assert parser.total_parsed == 4
        assert parser.total_fallbacks == 2
