"""
Load a BEIR dataset from disk into typed Python objects.

Usage:
    from src.data.loader import load_beir_dataset
    ds = load_beir_dataset("scifact", data_root="data/raw")
    ds.corpus, ds.queries, ds.qrels
"""

from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader

from src.data.schema import BeirDataset, DatasetName
from src.utils.logging import get_logger

log = get_logger(__name__)


def load_beir_dataset(
    name: DatasetName,
    data_root: str | Path = "data/raw",
    split: str = "test",
) -> BeirDataset:
    """Load corpus, queries, and qrels for a BEIR dataset.

    Args:
        name: Dataset name ("scifact" or "nfcorpus").
        data_root: Parent directory where the dataset folder lives.
        split: Which qrel split to load (typically "test").

    Returns:
        BeirDataset with .corpus, .queries, .qrels populated.
    """
    data_path = Path(data_root) / name

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_path}. "
            f"Run `python -m src.data.download --datasets {name}` first."
        )

    log.info(f"Loading {name} from {data_path} (split={split}) ...")
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split=split)

    log.info(
        f"Loaded {name}: {len(corpus)} docs, {len(queries)} queries, "
        f"{sum(len(v) for v in qrels.values())} qrel entries"
    )
    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)
