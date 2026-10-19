"""Simple baselines every model has to beat, on the time split.

Each baseline is fit on the training years only and scored on train, valid and
test. Ranges come from the 10th/90th percentile of the train residuals, so they
show the coverage a model's ranges have to beat.

Usage:
    python ml/src/baselines.py      # prints the table and writes ml/reports/baselines.csv
"""

import argparse
import logging

import numpy as np
import pandas as pd

from clean_data import ROOT
from evaluate import RANGE_QUANTILES, metrics_table, score
from features import TARGETS, build_features, load_history, split_frames, target_rows

REPORT_PATH = ROOT / "ml" / "reports" / "baselines.csv"
MIN_GENRE_FILMS = 20  # genres with fewer train films fall back to the global median

log = logging.getLogger("baselines")


class GlobalMedian:
    def fit(self, train: pd.DataFrame, target: str) -> "GlobalMedian":
        self.median = train[target].median()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.median)


class GenreMedian:
    """Mean of the train medians of a film's genres; the global median when none is known."""

    def fit(self, train: pd.DataFrame, target: str) -> "GenreMedian":
        self.median = train[target].median()
        self.genres = [c for c in train.columns if c.startswith("genre_")]
        flags = train[self.genres].to_numpy(bool)
        y = train[target].to_numpy(float)
        self.medians = np.array([np.median(y[flags[:, j]]) if flags[:, j].sum() >= MIN_GENRE_FILMS else np.nan
                                 for j in range(len(self.genres))])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        flags = df[self.genres].to_numpy(bool) & ~np.isnan(self.medians)
        known = flags.sum(axis=1)
        total = (flags * np.nan_to_num(self.medians)).sum(axis=1)
        return np.where(known > 0, total / np.maximum(known, 1), self.median)


class BudgetLinear:
    """log-revenue as a straight line in log-budget; the genre median for films without a budget."""

    def fit(self, train: pd.DataFrame, target: str) -> "BudgetLinear":
        has = train["log_budget_adj"].notna()
        self.slope, self.intercept = np.polyfit(train.loc[has, "log_budget_adj"], train.loc[has, target], 1)
        self.fallback = GenreMedian().fit(train, target)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        line = self.intercept + self.slope * df["log_budget_adj"].to_numpy(float)
        return np.where(np.isnan(line), self.fallback.predict(df), line)


BASELINES = {
    "y_log_revenue": {"global_median": GlobalMedian, "genre_median": GenreMedian, "budget_linear": BudgetLinear},
    "y_score": {"global_median": GlobalMedian, "genre_median": GenreMedian},
    "y_log_budget": {"global_median": GlobalMedian, "genre_median": GenreMedian},
}


class WithRange:
    """A baseline plus a range from the train residual quantiles."""

    def __init__(self, model):
        self.model = model

    def fit(self, train: pd.DataFrame, target: str) -> "WithRange":
        self.model.fit(train, target)
        resid = train[target].to_numpy(float) - self.model.predict(train)
        self.lo, self.hi = np.quantile(resid, RANGE_QUANTILES)
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred = self.model.predict(df)
        return pred, pred + self.lo, pred + self.hi


def evaluate_baselines(df: pd.DataFrame) -> pd.DataFrame:
    splits = split_frames(df)
    results = []
    for target in TARGETS:
        rows = {name: target_rows(frame, target) for name, frame in splits.items()}
        for model_name, cls in BASELINES[target].items():
            model = WithRange(cls()).fit(rows["train"], target)
            for split, frame in rows.items():
                pred, lo, hi = model.predict(frame)
                results.append({"target": target, "model": model_name, "split": split,
                                **score(target, frame[target], pred, lo, hi)})
    return metrics_table(results)


def print_table(table: pd.DataFrame) -> None:
    for target, t in table.groupby(level="target", sort=False):
        cols = [c for c in ["n", "mae", "rmse", "r2", "mdape", "coverage", "width"] if t[c].notna().any()]
        print(f"\n{target}")
        print(t.droplevel("target")[cols].to_string(float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df, _ = build_features(load_history())
    table = evaluate_baselines(df)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_PATH, float_format="%.4f")
    log.info("Wrote %s", REPORT_PATH)
    print_table(table)


if __name__ == "__main__":
    main()
