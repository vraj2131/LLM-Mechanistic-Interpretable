"""
Download BEIR datasets to data/raw/ via direct HTTPS requests.

Each dataset is assembled from two HuggingFace repos:
  - BeIR/{name}        — corpus and queries files
  - BeIR/{name}-qrels  — qrels TSV files

Writes everything in standard BEIR format (corpus.jsonl, queries.jsonl,
qrels/test.tsv) so GenericDataLoader can load them.

Usage (CLI):
    python -m src.data.download --datasets scifact nfcorpus

Usage (Python):
    from src.data.download import download_dataset
    data_path = download_dataset("scifact", out_dir="data/raw")
"""

import argparse
import gzip
import io
import json
import tempfile
from pathlib import Path

import requests

from src.utils.logging import get_logger

log = get_logger(__name__)

_HF_BASE = "https://huggingface.co/datasets"

AVAILABLE_DATASETS = {"scifact", "nfcorpus"}

# Direct HTTPS download URLs for each dataset file.
# format: list of (url, local_filename_hint)
_SPECS: dict[str, dict] = {
    "scifact": {
        "corpus_url":  f"{_HF_BASE}/BeIR/scifact/resolve/main/corpus.jsonl.gz",
        "queries_url": f"{_HF_BASE}/BeIR/scifact/resolve/main/queries.jsonl.gz",
        "qrels_url":   f"{_HF_BASE}/BeIR/scifact-qrels/resolve/main/test.tsv",
        "corpus_format": "jsonl_gz",
        "queries_format": "jsonl_gz",
    },
    "nfcorpus": {
        "corpus_url":  f"{_HF_BASE}/BeIR/nfcorpus/resolve/main/corpus/corpus-00000-of-00001.parquet",
        "queries_url": f"{_HF_BASE}/BeIR/nfcorpus/resolve/main/queries/queries-00000-of-00001.parquet",
        "qrels_url":   f"{_HF_BASE}/BeIR/nfcorpus-qrels/resolve/main/test.tsv",
        "corpus_format": "parquet",
        "queries_format": "parquet",
    },
}


def download_dataset(name: str, out_dir: str | Path = "data/raw") -> Path:
    """Download a BEIR dataset and write it in BEIR-compatible format.

    Skips download if the target directory is already complete.

    Args:
        name: Dataset name ("scifact" or "nfcorpus").
        out_dir: Parent directory. Dataset is written to out_dir/name/.

    Returns:
        Path to the dataset directory.
    """
    if name not in _SPECS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {AVAILABLE_DATASETS}.")

    out_dir = Path(out_dir)
    dataset_dir = out_dir / name

    if _is_complete(dataset_dir):
        log.info(f"Dataset '{name}' already exists at {dataset_dir} — skipping.")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "qrels").mkdir(exist_ok=True)

    spec = _SPECS[name]
    log.info(f"Downloading '{name}' from HuggingFace ...")

    # corpus
    raw = _fetch(spec["corpus_url"], "corpus")
    _write_jsonl(raw, spec["corpus_format"], dataset_dir / "corpus.jsonl", "corpus")

    # queries
    raw = _fetch(spec["queries_url"], "queries")
    _write_jsonl(raw, spec["queries_format"], dataset_dir / "queries.jsonl", "queries")

    # qrels
    log.info("  Downloading qrels/test.tsv ...")
    raw = _fetch(spec["qrels_url"], "qrels")
    qrels_path = dataset_dir / "qrels" / "test.tsv"
    qrels_path.write_bytes(raw)
    n_qrels = qrels_path.read_text().count("\n") - 1
    log.info(f"  Wrote {n_qrels:,} qrel entries -> {qrels_path}")

    log.info(f"Dataset '{name}' ready at {dataset_dir}")
    return dataset_dir


def _fetch(url: str, label: str, timeout: int = 120) -> bytes:
    """Download url and return raw bytes, with progress logging."""
    log.info(f"  Fetching {label} from {url.split('/resolve/')[0].split('/')[-1]} ...")
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    chunks = []
    downloaded = 0
    last_pct = -1

    for chunk in resp.iter_content(chunk_size=256 * 1024):  # 256 KB
        chunks.append(chunk)
        downloaded += len(chunk)
        if total > 0:
            pct = int(downloaded * 100 / total)
            if pct // 10 != last_pct // 10:
                log.info(f"    {pct}% ({downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB)")
                last_pct = pct

    log.info(f"  Downloaded {downloaded / 1e6:.2f} MB")
    return b"".join(chunks)


def _write_jsonl(raw: bytes, fmt: str, out_path: Path, kind: str) -> None:
    """Decode raw bytes (jsonl_gz or parquet) and write to out_path as JSONL."""
    n = 0
    with open(out_path, "w") as out_f:
        if fmt == "jsonl_gz":
            with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as gz_f:
                for line in gz_f:
                    row = json.loads(line)
                    row.setdefault("title", "")
                    row.setdefault("text", "")
                    out_f.write(json.dumps({
                        "_id": row["_id"],
                        "title": row["title"],
                        "text": row["text"],
                    }) + "\n")
                    n += 1
        elif fmt == "parquet":
            import pandas as pd
            df = pd.read_parquet(io.BytesIO(raw))
            for _, row in df.iterrows():
                out_f.write(json.dumps({
                    "_id": str(row["_id"]),
                    "title": str(row.get("title", "")),
                    "text": str(row.get("text", "")),
                }) + "\n")
                n += 1
        else:
            raise ValueError(f"Unknown format: {fmt}")

    log.info(f"  Wrote {n:,} {kind} entries -> {out_path}")


def _is_complete(dataset_dir: Path) -> bool:
    return (
        dataset_dir.exists()
        and (dataset_dir / "corpus.jsonl").exists()
        and (dataset_dir / "corpus.jsonl").stat().st_size > 0
        and (dataset_dir / "queries.jsonl").exists()
        and (dataset_dir / "qrels" / "test.tsv").exists()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BEIR datasets via HTTPS.")
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(AVAILABLE_DATASETS),
        choices=list(AVAILABLE_DATASETS),
    )
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()

    for name in args.datasets:
        download_dataset(name, out_dir=args.out_dir)
