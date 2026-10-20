"""LightGBM models for log-revenue and score, each with a 10-90% range.

Each model is three LightGBM regressors: an L2 one for the point prediction and
two quantile ones (alpha 0.1 and 0.9) for the range. All are fit on the training
years. The validation years (2016-2018) are used for early stopping, which
picks the number of trees, and to calibrate the ranges: each range is widened
by one margin per model so that 80% of valid films fall inside (conformalized
quantile regression). The test years (2019+) are never seen in training or
calibration, so they give the honest numbers for the README.

Revenue gets two models because the creator's budget is optional: `lgbm` uses
the budget, `lgbm_no_budget` does not. The score model never uses the budget,
and the budget model estimates it from everything else. If the budget model
isn't clearly better than the genre median on valid, it is flagged as a rough
estimate (see `budget_quality`).

The models saved for the app (`fit_live`) use the same settings but are fit on
films up to 2023, with early stopping and calibration on 2024-25, so their
ranges reflect recent years. The evaluation above is what the README reports.

Usage:
    python ml/src/train.py      # prints the table, writes the reports and ml/models/predictor.joblib
"""

import argparse
import logging
from datetime import date

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import REPORT_PATH as BASELINES_PATH, evaluate_baselines, print_table
from clean_data import ROOT
from evaluate import RANGE_QUANTILES, metrics_table, score
from features import build_features, feature_columns, fit_spec, load_history, split_frames, target_rows

REPORTS = BASELINES_PATH.parent
MODELS_PATH = REPORTS / "models.csv"
METRICS_MD_PATH = REPORTS / "metrics.md"
COVERAGE_PATH = REPORTS / "coverage_by_year.csv"
IMPORTANCE_PATH = REPORTS / "importance.csv"
MODEL_PATH = ROOT / "ml" / "models" / "predictor.joblib"
SHOCK_YEARS = (2020, 2021)  # cinema closures; reported separately, never used to tune anything
LIVE_TRAIN_END = 2023  # saved models: train <= 2023, early stopping and calibration on 2024+

# Fixed, modest settings; only the number of trees is chosen (by early stopping on valid).
PARAMS = dict(learning_rate=0.03, n_estimators=2000, num_leaves=15, min_child_samples=30, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
EARLY_STOPPING = 100

# (target, model name) -> feature columns to leave out
MODELS = {
    ("y_log_revenue", "lgbm"): set(),
    ("y_log_revenue", "lgbm_no_budget"): {"log_budget_adj"},
    ("y_score", "lgbm"): {"log_budget_adj"},
    ("y_log_budget", "lgbm"): {"log_budget_adj"},  # the film's own budget is the target
}

# The budget model is a "rough estimate" unless, on valid, it reaches this R² and beats the
# genre-median MAE by this share. Fixed before seeing the results; test never decides it.
ROUGH_MIN_R2 = 0.30
ROUGH_MIN_GAIN = 0.15

# Feature columns summed into readable factors for the explanations: (column prefix, factor), first match wins.
FACTOR_GROUPS = [
    ("genre_", "Genres"), ("n_genres", "Genres"),
    ("ov_", "Overview"), ("overview_words", "Overview"),
    ("director_", "Director's track record"),
    ("actor_", "Lead actors' track record"),
    ("company_", "Production company"),
    ("log_budget_adj", "Budget"), ("runtime", "Runtime"), ("release_year", "Release year"),
    ("release_month", "Release month"), ("original_language", "Language"),
]

log = logging.getLogger("train")


def features_for(df: pd.DataFrame, drop=()) -> list[str]:
    return [c for c in feature_columns(df) if c not in drop]


def factor_of(col: str) -> str:
    return next((factor for prefix, factor in FACTOR_GROUPS if col.startswith(prefix)), col)


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

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """SHAP values of the point model (LightGBM's TreeSHAP) summed per factor, plus a "base" column.

        Per row, the factors plus base add up to the point model's raw prediction (before clipping into the range).
        """
        values = self.point.predict(df[self.cols], pred_contrib=True)
        per_col = pd.DataFrame(values[:, :-1], index=df.index, columns=self.cols)
        out = per_col.T.groupby([factor_of(c) for c in self.cols], sort=False).sum().T
        out["base"] = values[:, -1]
        return out


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


def budget_quality(table: pd.DataFrame) -> dict:
    """Valid R² of the budget model, its MAE gain over the genre median, and whether it's a rough estimate."""
    model = table.loc[("y_log_budget", "lgbm", "valid")]
    baseline = table.loc[("y_log_budget", "genre_median", "valid")]
    gain = 1 - model["mae"] / baseline["mae"]
    return {"r2": float(model["r2"]), "gain": float(gain),
            "rough": bool(model["r2"] < ROUGH_MIN_R2 or gain < ROUGH_MIN_GAIN)}


def factor_importance(fitted: dict, df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Mean |SHAP| per factor and model on `split`, with each factor's share of the model's total."""
    rows = []
    for (target, name), model in fitted.items():
        frame = target_rows(split_frames(df)[split], target)
        effect = model.contributions(frame).drop(columns="base").abs().mean()
        for factor, value in effect.sort_values(ascending=False).items():
            rows.append({"target": target, "model": name, "factor": factor, "mean_abs_effect": value,
                         "share": value / effect.sum()})
    return pd.DataFrame(rows).set_index(["target", "model", "factor"])


def fit_live(history: dict[str, pd.DataFrame], train_end: int = LIVE_TRAIN_END, spec=None, **params) -> dict:
    """The models the app uses: fit on films up to `train_end`, early stopping and calibration on later films.

    Returns the bundle predict.py loads (without the budget verdict, which comes from the evaluation split).
    """
    spec = spec or fit_spec(history, train_end=train_end)
    df, _ = build_features(history, spec)
    year = pd.to_datetime(df["release_date"]).dt.year
    train, valid = df[year <= train_end], df[year > train_end]
    models, coverage = {}, {}
    for (target, name), drop in MODELS.items():
        model = RangeModel(features_for(df, drop), **params).fit(
            target_rows(train, target), target_rows(valid, target), target)
        rows = target_rows(valid, target)
        _, lo, hi = model.predict(rows)
        models[target, name] = model
        coverage[target, name] = float(np.mean((rows[target] >= lo) & (rows[target] <= hi)))
    return {"spec": spec, "models": models, "train_end": train_end,
            "valid_years": (int(year[year > train_end].min()), int(year.max())), "valid_coverage": coverage}


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

    history = load_history()
    df, _ = build_features(history)
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
    quality = budget_quality(table)
    note += [f"**Budget model: {'rough estimate' if quality['rough'] else 'reliable'}**", "",
             f"Valid R² {quality['r2']:.3f} (needs {ROUGH_MIN_R2:.2f}), MAE {quality['gain']:.0%} below the "
             f"genre median (needs {ROUGH_MIN_GAIN:.0%}).", ""]
    importance = factor_importance(fitted, df)
    table.to_csv(MODELS_PATH, float_format="%.4f")
    by_year.to_csv(COVERAGE_PATH, float_format="%.4f")
    importance.to_csv(IMPORTANCE_PATH, float_format="%.4f")
    METRICS_MD_PATH.write_text(markdown_table(table) + "\n" + "\n".join(note), encoding="utf-8")
    log.info("Wrote %s, %s, %s and %s", MODELS_PATH, COVERAGE_PATH, IMPORTANCE_PATH, METRICS_MD_PATH)

    log.info("Fitting the app models on films up to %d", LIVE_TRAIN_END)
    bundle = fit_live(history)
    bundle.update(budget_rough=quality["rough"], trained_on=date.today().isoformat())
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    log.info("Wrote %s", MODEL_PATH)

    print_table(table)
    print("\nTest range coverage by release year:")
    print(by_year.round(2).to_string())
    print("\n" + "\n".join(note))
    for (target, name), model in fitted.items():
        trees = [m.best_iteration_ for m in (model.point, model.lo, model.hi)]
        print(f"\n{target} / {name}: trees (point, q10, q90) = {trees}, range margin = {model.margin:+.3f}; "
              "top features by gain:")
        print(model.importance().head(10).round(0).to_string())

    print("\nFactor importance on test (share of mean |SHAP|):")
    print(importance["share"].unstack(["target", "model"]).round(2).to_string())
    print(f"\nApp models ({bundle['valid_years'][0]}-{bundle['valid_years'][1]} valid coverage):")
    for (target, name), cov in bundle["valid_coverage"].items():
        print(f"- {target} / {name}: {cov:.0%}")


if __name__ == "__main__":
    # Run main() from the imported module so the saved RangeModels pickle as train.RangeModel, not __main__.
    import train

    train.main()
