import numpy as np
import pandas as pd

from features import FORBIDDEN
from train import (MODELS, RangeModel, coverage_by_year, coverage_excluding, evaluate_models, features_for,
                   markdown_table)

FAST = dict(n_estimators=60, learning_rate=0.1, min_child_samples=5, n_jobs=1)


def frame(n_train=300, n_valid=100, n_test=100, seed=0):
    """Revenue rises with x and budget, score with x; noise is the rest."""
    rng = np.random.default_rng(seed)
    n = n_train + n_valid + n_test
    x, budget = rng.normal(0, 1, n), rng.uniform(14, 18, n)
    return pd.DataFrame({
        "id": np.arange(n), "title": "t", "release_date": "2000-01-01",
        "split": ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test,
        "genre_drama": rng.integers(0, 2, n), "x": x, "noise": rng.normal(0, 1, n), "log_budget_adj": budget,
        "original_language": pd.Categorical(rng.choice(["en", "other"], n), categories=["en", "other"]),
        "y_log_revenue": 2 * x + budget + rng.normal(0, 0.3, n),
        "y_score": 6 + 0.8 * x + rng.normal(0, 0.2, n),
        "y_log_budget": budget,
    })


def test_test_targets_never_reach_training():
    df = frame()
    changed = df.copy()
    test = changed["split"] == "test"
    changed.loc[test, ["y_log_revenue", "y_score"]] += 100
    _, a = evaluate_models(df, **FAST)
    _, b = evaluate_models(changed, **FAST)
    for key in MODELS:
        for pa, pb in zip(a[key].predict(df), b[key].predict(df)):
            np.testing.assert_array_equal(pa, pb)


def test_inputs_have_no_budget_or_leaky_columns():
    _, fitted = evaluate_models(frame(), **FAST)
    for (target, name), model in fitted.items():
        assert not [c for c in model.cols if c in FORBIDDEN or c.startswith("y_")]
        assert ("log_budget_adj" in model.cols) == (name == "lgbm" and target == "y_log_revenue")


def test_ranges_are_ordered():
    df = frame()
    train, valid = df[df.split == "train"], df[df.split == "valid"]
    model = RangeModel(features_for(df), **FAST).fit(train, valid, "y_score")
    pred, lo, hi = model.predict(df)
    assert (lo <= pred).all() and (pred <= hi).all()


def test_calibration_brings_valid_coverage_to_80_percent():
    df = frame(n_valid=400)
    train, valid = df[df.split == "train"], df[df.split == "valid"]
    cols = features_for(df)
    for calibrate in (False, True):
        model = RangeModel(cols, **FAST).fit(train, valid, "y_score", calibrate=calibrate)
        _, lo, hi = model.predict(valid)
        coverage = np.mean((valid["y_score"] >= lo) & (valid["y_score"] <= hi))
        if calibrate:
            assert 0.8 <= coverage <= 0.82
        else:
            assert model.margin == 0.0


def test_models_beat_the_median_on_a_planted_signal():
    df = frame()
    table, _ = evaluate_models(df, **FAST)
    for target in ["y_log_revenue", "y_score"]:
        median_mae = np.mean(np.abs(df.query("split == 'valid'")[target] - df.query("split == 'train'")[target].median()))
        assert (table.xs((target, "valid"), level=("target", "split"))["mae"] < 0.8 * median_mae).all()
    # with the budget, revenue is almost fully explained
    assert table.loc[("y_log_revenue", "lgbm", "valid"), "r2"] > 0.9


def test_missing_scores_are_skipped():
    df = frame()
    df.loc[df.index[:50], "y_score"] = np.nan
    table, _ = evaluate_models(df, **FAST)
    assert table.loc[("y_score", "lgbm", "train"), "n"] == 250


def test_coverage_by_year_adds_up_to_the_test_coverage():
    df = frame()
    test = df["split"] == "test"
    df.loc[test, "release_date"] = [f"{2019 + i % 4}-06-01" for i in range(test.sum())]
    table, fitted = evaluate_models(df, **FAST)
    by_year = coverage_by_year(fitted, df)
    for key in MODELS:
        part = by_year.loc[key]
        assert list(part.index) == [2019, 2020, 2021, 2022]
        np.testing.assert_allclose(part["coverage"] + part["below"] + part["above"], 1)
        overall = np.average(part["coverage"], weights=part["n"])
        assert np.isclose(overall, table.loc[(*key, "test"), "coverage"])
        kept = part.drop([2020, 2021])
        assert np.isclose(coverage_excluding(by_year)[key], np.average(kept["coverage"], weights=kept["n"]))


def test_markdown_table_has_valid_and_test_only():
    table, _ = evaluate_models(frame(), **FAST)
    md = markdown_table(table)
    assert "| lgbm_no_budget | test |" in md and "| train |" not in md
