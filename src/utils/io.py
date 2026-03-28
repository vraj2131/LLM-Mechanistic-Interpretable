"""
Parquet and NumPy I/O helpers.

Usage:
    from src.utils.io import save_parquet, load_parquet, save_npy, load_npy
"""

from pathlib import Path
import numpy as np
import pandas as pd


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_npy(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_npy(path: str | Path, dtype: np.dtype | None = None) -> np.ndarray:
    arr = np.load(path)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr
