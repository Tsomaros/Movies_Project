import numpy as np
import pandas as pd
import pytest

from baselines import BudgetLinear, GenreMedian, evaluate_baselines
from evaluate import point_metrics, range_metrics


def frame(n_train=60, n_valid=10, n_test=10, seed=0):
    """Train films are drama (y ~ 10) or comedy (y ~ 20); valid/test are the same mix."""
    rng = np.random.default_rng(seed)
    n = n_train + n_valid + n_test
    drama = np.arange(n) % 2 == 0
    budget = rng.uniform(14, 18, n)
    y = np.where(drama, 10.0, 20.0)
    return pd.DataFrame({
        "split": ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test,
        "genre_drama": drama.astype(int), "genre_comedy": (~drama).astype(int), "genre_western": 0,
        "log_budget_adj": budget,
        "y_log_revenue": y + rng.normal(0, 0.1, n), "y_score": np.where(drama, 7.0, 6.0), "y_log_budget": budget,
    })


def test_genre_median_per_genre_and_fallback():
    train = frame().query("split == 'train'")
    model = GenreMedian().fit(train, "y_score")
    test = pd.DataFrame({"genre_drama": [1, 0, 1, 0], "genre_comedy": [0, 1, 1, 0], "genre_western": [0, 0, 0, 1]})
    # drama, comedy, both (mean of the two), western only (too few train films -> global median)
    assert model.predict(test).tolist() == [7.0, 6.0, 6.5, 6.5]


def test_budget_linear_recovers_line():
    df = frame()
    df["y_log_revenue"] = 2.0 + 0.9 * df["log_budget_adj"]
    model = BudgetLinear().fit(df, "y_log_revenue")
    assert model.slope == pytest.approx(0.9) and model.intercept == pytest.approx(2.0)
    no_budget = df.head(2).assign(log_budget_adj=np.nan)
    assert np.isfinite(model.predict(no_budget)).all()


def test_baselines_fit_on_train_only():
    df = frame()
    changed = df.copy()
    held_out = changed["split"] != "train"
    changed.loc[held_out, ["y_log_revenue", "y_score", "y_log_budget"]] += 100  # only the truth moves
    a, b = evaluate_baselines(df), evaluate_baselines(changed)
    pd.testing.assert_frame_equal(a.xs("train", level="split"), b.xs("train", level="split"))
    # predictions didn't move, so the error on held-out rows grows by exactly 100
    assert (b.xs("valid", level="split")["mae"] > 90).all()


def test_point_metrics():
    m = point_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["r2"] == pytest.approx(1 - 4 / 2)
    assert point_metrics([np.log1p(100.0)], [np.log1p(150.0)], dollars=True)["mdape"] == pytest.approx(0.5)


def test_range_metrics():
    m = range_metrics([1, 5, 10, 20], lo=[0, 0, 0, 0], hi=[2, 4, 10, 30])
    assert m["coverage"] == 0.75 and m["width"] == pytest.approx(7.0)  # widths 2, 4, 10, 30


def test_missing_scores_are_left_out():
    df = frame()
    df.loc[df.index[:20], "y_score"] = np.nan
    table = evaluate_baselines(df)
    assert table.loc[("y_score", "genre_median", "train"), "n"] == 40
    assert table.loc[("y_log_revenue", "genre_median", "train"), "n"] == 60
