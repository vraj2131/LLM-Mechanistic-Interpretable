"""
Parse relevance scores (0-3) from raw Qwen output strings.

The model is instructed to respond with a single integer 0-3.  In practice
it sometimes adds punctuation, spaces, or extra tokens.  We extract the
first digit in {0,1,2,3} from the first 20 characters of the decoded output.

Fallback: if no valid digit is found, return 0 and increment a counter so
callers can compute the fallback rate.  The plan requires fallback rate < 5%.

Usage:
    from src.reranking.score_parser import ScoreParser
    parser = ScoreParser()
    score = parser.parse("2\n")           # → 2
    score = parser.parse("Score: 3.")     # → 3
    score = parser.parse("invalid")       # → 0  (fallback)
    print(parser.fallback_rate())         # → 0.333...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SCORE_RE = re.compile(r"\b([0-3])\b")
_SCAN_CHARS = 20  # only look in first N chars to avoid hallucinated long outputs


@dataclass
class ScoreParser:
    """Stateful parser that tracks total calls and fallback count."""

    fallback_score: int = 0
    _total: int = field(default=0, init=False, repr=False)
    _fallbacks: int = field(default=0, init=False, repr=False)

    def parse(self, raw_output: str) -> int:
        """Extract an integer 0-3 from *raw_output*.

        Searches only the first ``_SCAN_CHARS`` characters of the string.

        Returns:
            Parsed integer in [0, 3], or ``self.fallback_score`` on failure.
        """
        self._total += 1
        snippet = raw_output.strip()[:_SCAN_CHARS]
        match = _SCORE_RE.search(snippet)
        if match:
            return int(match.group(1))
        self._fallbacks += 1
        return self.fallback_score

    def parse_batch(self, raw_outputs: list[str]) -> list[int]:
        """Parse a list of raw model outputs, returning aligned scores."""
        return [self.parse(o) for o in raw_outputs]

    def fallback_rate(self) -> float:
        """Fraction of calls that fell back to the default score."""
        if self._total == 0:
            return 0.0
        return self._fallbacks / self._total

    def reset_stats(self) -> None:
        """Reset counters (useful between datasets / variants)."""
        self._total = 0
        self._fallbacks = 0

    @property
    def total_parsed(self) -> int:
        return self._total

    @property
    def total_fallbacks(self) -> int:
        return self._fallbacks
