"""
Format metric dicts into clean printable tables.

Usage:
    from src.evaluation.results_table import print_metrics, metrics_to_df
    print_metrics({"BM25": {"ndcg@10": 0.68, "mrr@10": 0.71, "recall@20": 0.88}})
"""

import pandas as pd


def print_metrics(
    results: dict[str, dict[str, float]],
    title: str = "Retrieval / Reranking Metrics",
) -> None:
    """Pretty-print a table of system -> metric -> value.

    Args:
        results: {system_name: {metric_name: value}}
        title: Header line printed above the table.
    """
    df = metrics_to_df(results)
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"{'=' * 50}\n")


def metrics_to_df(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert results dict to a DataFrame with systems as rows, metrics as columns."""
    return pd.DataFrame(results).T
