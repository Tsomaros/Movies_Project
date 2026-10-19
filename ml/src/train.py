"""LightGBM models for log-revenue and score, each with a 10-90% range.

Each model is three LightGBM regressors: an L2 one for the point prediction and
two quantile ones (alpha 0.1 and 0.9) for the range. All are fit on the training
years. The validation years (2016-2018) are used for early stopping, which
picks the number of trees, and to calibrate the ranges: each range is widened
by one margin per model so that 80% of valid films fall inside (conformalized
quantile regression). The test years (2019+) are never seen in training or
calibration, so they give the honest numbers for the README.

Revenue gets two models because the creator's budget is optional: `lgbm` uses
the budget, `lgbm_no_budget` does not. The score model never uses the budget.

Usage:
    python ml/src/train.py      # prints the table, writes ml/reports/models.csv, coverage_by_year.csv and metrics.md
"""

import argparse
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import REPORT_PATH as BASELINES_PATH, evaluate_baselines, print_table
from evaluate import RANGE_QUANTILES, metrics_table, score
from features import build_features, feature_columns, load_history, split_frames, target_rows

REPORTS = BASELINES_PATH.parent
MODELS_PATH = REPORTS / "models.csv"
METRICS_MD_PATH = REPORTS / "metrics.md"
COVERAGE_PATH = REPORTS / "coverage_by_year.csv"
SHOCK_YEARS = (2020, 2021)  # cinema closures; reported separately, never used to tune anything

# Fixed, modest settings; only the number of trees is chosen (by early stopping on valid).
PARAMS = dict(learning_rate=0.03, n_estimators=2000, num_leaves=15, min_child_samples=30, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
EARLY_STOPPING = 100

# (target, model name) -> feature columns to leave out
MODELS = {
    ("y_log_revenue", "lgbm"): set(),
    ("y_log_revenue", "lgbm_no_budget"): {"log_budget_adj"},
    ("y_score", "lgbm"): {"log_budget_adj"},
}

log = logging.getLogger("train")


def features_for(df: pd.DataFrame, drop=()) -> list[str]:
    return [c for c in feature_columns(df) if c not in drop]


class RangeModel:
    """A point model plus 10% and 90% quantile models on the same features."""

    def __init__(self, cols: list[str], **params):
        self.cols = cols
        self.params = {**PARAMS, **params}

    def _fit_one(self, train: pd.DataFrame, valid: pd.DataFrame, target: str, **objective) -> lgb.LGBMRegressor:
        model = lgb.LGBMRegressor(**self.params, **objective)
        model.fit(train[self.cols], train[target], eval_X=valid[self.cols], eval_y=valid[target],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        return model

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, target: str, calibrate: bool = True) -> "RangeModel":
        self.point = self._fit_one(train, valid, target, objective="regression")
        self.lo, self.hi = (self._fit_one(train, valid, target, objective="quantile", alpha=a)
                            for a in RANGE_QUANTILES)
        self.margin = 0.0
        if calibrate:
            self.margin = self._conformal_margin(valid, target)
        return self

    def _conformal_margin(self, valid: pd.DataFrame, target: str) -> float:
        """Conformalized quantile regression: the widening that makes the valid ranges cover 80%.

        A film's score is how far its true value falls outside the raw range (negative when inside).
        The margin is the finite-sample-corrected 80% quantile of these scores.
        """
        lo, hi = self._raw_range(valid[self.cols])
        y = valid[target].to_numpy(float)
        scores = np.maximum(lo - y, y - hi)
        level = RANGE_QUANTILES[1] - RANGE_QUANTILES[0]
        n = len(scores)
        return float(np.quantile(scores, min(1.0, level * (n + 1) / n), method="higher"))

    def _raw_range(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.lo.predict(X), self.hi.predict(X)
        return np.minimum(a, b), np.maximum(a, b)  # quantile models can cross

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = df[self.cols]
        lo, hi = self._raw_range(X)
        lo, hi = lo - self.margin, hi + self.margin
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)  # a negative margin (over-wide ranges) can cross them
        return np.clip(self.point.predict(X), lo, hi), lo, hi

    def importance(self) -> pd.Series:
        gain = self.point.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain, index=self.cols).sort_values(ascending=False)


def evaluate_models(df: pd.DataFrame, **params) -> tuple[pd.DataFrame, dict]:
    splits = split_frames(df)
    results, fitted = [], {}
    for (target, name), drop in MODELS.items():
        rows = {split: target_rows(frame, target) for split, frame in splits.items()}
        model = RangeModel(features_for(df, drop), **params).fit(rows["train"], rows["valid"], target)
        fitted[target, name] = model
        for split, frame in rows.items():
            pred, lo, hi = model.predict(frame)
            results.append({"target": target, "model": name, "split": split,
                            **score(target, frame[target], pred, lo, hi)})
    return metrics_table(results), fitted


def coverage_by_year(fitted: dict, df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Range coverage per release year, with the share of films falling below and above the range."""
    rows = []
    for (target, name), model in fitted.items():
        frame = target_rows(split_frames(df)[split], target)
        _, lo, hi = model.predict(frame)
        y = frame[target].to_numpy(float)
        years = pd.to_datetime(frame["release_date"]).dt.year.to_numpy()
        for year in np.unique(years):
            at = years == year
            rows.append({"target": target, "model": name, "year": int(year), "n": int(at.sum()),
                         "coverage": np.mean((y[at] >= lo[at]) & (y[at] <= hi[at])),
                         "below": np.mean(y[at] < lo[at]), "above": np.mean(y[at] > hi[at])})
    return pd.DataFrame(rows).set_index(["target", "model", "year"])


def coverage_excluding(by_year: pd.DataFrame, years=SHOCK_YEARS) -> pd.Series:
    """Overall coverage per model, leaving out `years`."""
    kept = by_year[~by_year.index.get_level_values("year").isin(years)]
    hits = (kept["coverage"] * kept["n"]).groupby(level=["target", "model"], sort=False).sum()
    return hits / kept["n"].groupby(level=["target", "model"], sort=False).sum()


def markdown_table(table: pd.DataFrame, splits=("valid", "test")) -> str:
    """Valid and test rows per target as markdown, ready for the README."""
    fmt = {"n": "{:.0f}", "mae": "{:.3f}", "r2": "{:.3f}", "mdape": "{:.0%}", "coverage": "{:.0%}", "width": "{:.2f}"}
    out = []
    for target, t in table.groupby(level="target", sort=False):
        cols = [c for c in fmt if c in t and t[c].notna().any()]
        out += [f"**{target}**", "", "| model | split | " + " | ".join(cols) + " |",
                "|---|---|" + "---|" * len(cols)]
        for (model, split), row in t.droplevel("target").iterrows():
            if split in splits:
                out.append(f"| {model} | {split} | " + " | ".join(fmt[c].format(row[c]) for c in cols) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df, _ = build_features(load_history())
    models, fitted = evaluate_models(df)
    baselines = evaluate_baselines(df)
    targets = models.index.get_level_values("target").unique()
    table = pd.concat([part.xs(t, level="target", drop_level=False)  # baselines, then models, per target
                       for t in targets for part in (baselines, models)])
    by_year = coverage_by_year(fitted, df)
    excluded = coverage_excluding(by_year)
    shock = ", ".join(map(str, SHOCK_YEARS))
    note = [f"**Test range coverage without {shock}**", ""] + [
        f"- {target} / {name}: {cov:.0%}" for (target, name), cov in excluded.items()] + [""]
    table.to_csv(MODELS_PATH, float_format="%.4f")
    by_year.to_csv(COVERAGE_PATH, float_format="%.4f")
    METRICS_MD_PATH.write_text(markdown_table(table) + "\n" + "\n".join(note), encoding="utf-8")
    log.info("Wrote %s, %s and %s", MODELS_PATH, COVERAGE_PATH, METRICS_MD_PATH)

    print_table(table)
    print("\nTest range coverage by release year:")
    print(by_year.round(2).to_string())
    print("\n" + "\n".join(note))
    for (target, name), model in fitted.items():
        trees = [m.best_iteration_ for m in (model.point, model.lo, model.hi)]
        print(f"\n{target} / {name}: trees (point, q10, q90) = {trees}, range margin = {model.margin:+.3f}; "
              "top features by gain:")
        print(model.importance().head(10).round(0).to_string())


if __name__ == "__main__":
    main()
