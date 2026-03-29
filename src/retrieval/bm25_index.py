"""
Build and persist a BM25Okapi index from a BEIR corpus.

The index is stored as two files:
  - bm25_index.pkl   : serialized BM25Okapi object
  - corpus_ids.json  : ordered list of doc_ids so BM25 array positions map back to BEIR IDs

Usage:
    from src.retrieval.bm25_index import build_index, load_index
    index, ids = build_index(corpus, cache_dir="data/caches/bm25_indexes/scifact")
    index, ids = load_index(cache_dir="data/caches/bm25_indexes/scifact")
"""

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.data.schema import BeirCorpus
from src.utils.logging import get_logger

log = get_logger(__name__)

_INDEX_FILE = "bm25_index.pkl"
_IDS_FILE = "corpus_ids.json"


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenization with lowercasing.

    Deliberately simple — matches the tokenizer used for lexical features
    in Phase 5 so BM25 scores and overlap features are computed consistently.
    """
    return text.lower().split()


def build_index(
    corpus: BeirCorpus,
    cache_dir: str | Path,
    k1: float = 1.5,
    b: float = 0.75,
    force: bool = False,
) -> tuple[BM25Okapi, list[str]]:
    """Build a BM25Okapi index from a BEIR corpus and save it to disk.

    Args:
        corpus: BEIR corpus dict {doc_id: {"title": ..., "text": ...}}.
        cache_dir: Directory to save index and ID map.
        k1: BM25 k1 parameter (term frequency saturation).
        b: BM25 b parameter (length normalization).
        force: If True, rebuild even if a cached index exists.

    Returns:
        (BM25Okapi index, ordered list of doc_ids)
    """
    cache_dir = Path(cache_dir)
    index_path = cache_dir / _INDEX_FILE
    ids_path = cache_dir / _IDS_FILE

    if not force and index_path.exists() and ids_path.exists():
        log.info(f"Loading cached BM25 index from {cache_dir}")
        return load_index(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Building BM25 index over {len(corpus):,} documents (k1={k1}, b={b}) ...")

    # Fix ordering — BM25Okapi scores are returned as a positional array
    doc_ids = list(corpus.keys())

    tokenized_corpus = [
        _tokenize(corpus[did].get("title", "") + " " + corpus[did].get("text", ""))
        for did in doc_ids
    ]

    index = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    # Persist
    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(ids_path, "w") as f:
        json.dump(doc_ids, f)

    log.info(f"Index saved to {cache_dir} ({index_path.stat().st_size / 1e6:.1f} MB)")
    return index, doc_ids


def load_index(cache_dir: str | Path) -> tuple[BM25Okapi, list[str]]:
    """Load a previously built BM25 index from disk.

    Returns:
        (BM25Okapi index, ordered list of doc_ids)
    """
    cache_dir = Path(cache_dir)
    index_path = cache_dir / _INDEX_FILE
    ids_path = cache_dir / _IDS_FILE

    if not index_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"No BM25 index found at {cache_dir}. "
            "Run build_index() first."
        )

    with open(index_path, "rb") as f:
        index = pickle.load(f)

    with open(ids_path) as f:
        doc_ids = json.load(f)

    log.info(f"Loaded BM25 index: {len(doc_ids):,} documents from {cache_dir}")
    return index, doc_ids
