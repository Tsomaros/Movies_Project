"""The no-leakage rule checked over the whole feature table: a film's features may only depend on earlier films."""

import numpy as np
import pandas as pd
import pytest

from features import build_features, feature_columns, fit_spec
from test_predict import synthetic_history

OUTCOMES = ["budget_adj", "revenue_adj", "has_financials", "vote_average", "vote_count"]
CUTOFFS = ["1995-01-15", "2005-06-15", "2016-01-01"]


@pytest.fixture(scope="module")
def base():
    h = synthetic_history()
    spec = fit_spec(h, n_topics=2, min_df=1)
    df, _ = build_features(h, spec)
    return h, spec, df.set_index("id")


def with_movies(h, movies):
    return {**h, "movies": movies}


def scramble(movies, mask, seed=1):
    """Random outcomes for the films in `mask`, wildly unlike the real ones."""
    rng = np.random.default_rng(seed)
    m, n = movies.copy(), int(mask.sum())
    m.loc[mask, "budget_adj"] = rng.uniform(1e4, 5e9, n)
    m.loc[mask, "revenue_adj"] = rng.uniform(1e4, 5e9, n)
    m.loc[mask, "has_financials"] = rng.integers(0, 2, n)
    m.loc[mask, "vote_average"] = rng.uniform(0, 10, n)
    m.loc[mask, "vote_count"] = rng.integers(0, 5000, n)
    return m


def earlier_rows(df, cutoff):
    return df[pd.to_datetime(df["release_date"]) < pd.Timestamp(cutoff)]


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_future_outcomes_dont_reach_earlier_features(base, cutoff):
    h, spec, df = base
    m = h["movies"]
    future = pd.to_datetime(m["release_date"]) >= pd.Timestamp(cutoff)
    after, _ = build_features(with_movies(h, scramble(m, future)), spec)
    before = earlier_rows(df, cutoff)
    cols = feature_columns(df.reset_index())
    assert len(before) > 50
    pd.testing.assert_frame_equal(before[cols], after.set_index("id").loc[before.index, cols])


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_dropping_future_films_changes_nothing_earlier(base, cutoff):
    h, spec, df = base
    keep_ids = set(h["movies"].loc[pd.to_datetime(h["movies"]["release_date"]) < pd.Timestamp(cutoff), "id"])
    cut = {name: t[t["id" if name == "movies" else "movie_id"].isin(keep_ids)] for name, t in h.items()}
    after, _ = build_features(cut, spec)
    before = earlier_rows(df, cutoff)
    cols = feature_columns(df.reset_index())
    pd.testing.assert_frame_equal(before[cols], after.set_index("id").loc[before.index, cols])


def test_own_outcomes_reach_only_the_targets(base):
    h, spec, df = base
    m = h["movies"]
    for movie_id in df.index[::23]:  # 20 films spread over the years
        own = m["id"] == movie_id
        changed = m.copy()
        changed.loc[own, ["revenue_adj", "vote_average", "vote_count"]] = [7e9, 0.5, 9000]
        row = build_features(with_movies(h, changed), spec)[0].set_index("id").loc[movie_id]
        diff = [c for c in df.columns if not (row[c] == df.loc[movie_id, c] or
                                              (pd.isna(row[c]) and pd.isna(df.loc[movie_id, c])))]
        assert set(diff) <= {"y_log_revenue", "y_score"}, (movie_id, diff)

        changed = m.copy()
        changed.loc[own, "budget_adj"] = 3e9
        row = build_features(with_movies(h, changed), spec)[0].set_index("id").loc[movie_id]
        diff = [c for c in df.columns if row[c] != df.loc[movie_id, c] and not pd.isna(df.loc[movie_id, c])]
        assert diff == ["log_budget_adj", "y_log_budget"], (movie_id, diff)


def test_spec_learns_from_training_years_only(base):
    h, spec, _ = base
    m = h["movies"].copy()
    later = m["release_year"] > 2015
    m.loc[later, "overview"] = "zombie apocalypse spaceship kungfu"
    m.loc[later, "original_language"] = "xx"
    m.loc[later, "has_financials"] = 1
    other = fit_spec(with_movies(h, m), n_topics=2, min_df=1)
    assert other.languages == spec.languages
    texts = pd.Series(["family drama about a farm", "zombie apocalypse spaceship kungfu", ""])
    pd.testing.assert_frame_equal(other.overview.transform(texts), spec.overview.transform(texts))
