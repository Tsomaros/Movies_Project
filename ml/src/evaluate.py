"""Metrics shared by the baselines and the models.

Revenue and budget are predicted on the log scale (log1p of 2024 dollars), the
score on its 0-10 scale. Point metrics are computed on that scale; revenue and
budget also get the median absolute % error in dollars, which is easier to read.
Ranges are scored by how often they contain the true value and how wide they are.
"""

import numpy as np
import pandas as pd

DOLLAR_TARGETS = {"y_log_revenue", "y_log_budget"}
RANGE_QUANTILES = (0.1, 0.9)  # a 10-90% range should contain the truth 80% of the time


def point_metrics(y, pred, dollars: bool = False) -> dict[str, float]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    spread = np.sum((y - y.mean()) ** 2)
    out = {
        "mae": np.mean(np.abs(err)),
        "rmse": np.sqrt(np.mean(err ** 2)),
        "r2": 1 - np.sum(err ** 2) / spread if spread > 0 else np.nan,
    }
    if dollars:
        actual = np.expm1(y)
        out["mdape"] = np.median(np.abs(np.expm1(pred) - actual) / actual)
    return out


def range_metrics(y, lo, hi) -> dict[str, float]:
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return {"coverage": np.mean((y >= lo) & (y <= hi)), "width": np.median(hi - lo)}


def score(target: str, y, pred, lo=None, hi=None) -> dict[str, float]:
    out = {"n": len(y), **point_metrics(y, pred, dollars=target in DOLLAR_TARGETS)}
    if lo is not None:
        out.update(range_metrics(y, lo, hi))
    return out


def metrics_table(results: list[dict]) -> pd.DataFrame:
    """Rows of {"target", "model", "split", **metrics} -> one tidy DataFrame."""
    return pd.DataFrame(results).set_index(["target", "model", "split"])
