import numpy as np
import pandas as pd
import pytest

from features import FORBIDDEN, build_features, feature_columns, fit_spec

WORDS = ["heist crew plans a robbery", "space crew explores a planet", "family drama about a farm",
         "detective hunts a killer", "robot learns about love"]


def film(movie_id, date="2010-06-01", director=None, actors=(), company=None, revenue=50_000_000,
         budget=10_000_000, genres=("Drama",), language="en", overview=None, vote_average=7.0, vote_count=100):
    has_fin = int(revenue is not None and budget is not None)
    return {
        "movie": {"id": movie_id, "title": f"Movie {movie_id}", "release_date": date,
                  "release_year": int(date[:4]), "release_month": int(date[5:7]), "runtime": 100,
                  "original_language": language, "overview": overview if overview is not None else WORDS[movie_id % 5],
                  "budget_adj": budget, "revenue_adj": revenue, "vote_average": vote_average,
                  "vote_count": vote_count, "has_financials": has_fin},
        "genres": [{"movie_id": movie_id, "genre_id": i, "name": g} for i, g in enumerate(genres)],
        "companies": [{"movie_id": movie_id, "company_id": company, "name": f"C{company}", "company_order": 0}]
        if company else [],
        "cast_members": [{"movie_id": movie_id, "person_id": p, "name": f"A{p}", "character": "", "cast_order": i}
                         for i, p in enumerate(actors)],
        "crew": [{"movie_id": movie_id, "person_id": director, "name": f"D{director}", "job": "Director"}]
        if director else [],
    }


COLUMNS = {
    "movies": ["id", "title", "release_date", "release_year", "release_month", "runtime", "original_language",
               "overview", "budget_adj", "revenue_adj", "vote_average", "vote_count", "has_financials"],
    "genres": ["movie_id", "genre_id", "name"],
    "companies": ["movie_id", "company_id", "name", "company_order"],
    "cast_members": ["movie_id", "person_id", "name", "character", "cast_order"],
    "crew": ["movie_id", "person_id", "name", "job"],
}


def history(*films):
    tables = {"movies": pd.DataFrame([f["movie"] for f in films], columns=COLUMNS["movies"])}
    for name in ["genres", "companies", "cast_members", "crew"]:
        tables[name] = pd.DataFrame([row for f in films for row in f[name]], columns=COLUMNS[name])
    return tables


def features(*films):
    h = history(*films)
    df, _ = build_features(h, fit_spec(h, n_topics=2, min_df=1))
    return df.set_index("id")


# Five background films so the overview encoder always has some text to fit on.
BACKGROUND = [film(100 + i, date=f"200{i}-01-01") for i in range(5)]


def test_later_films_dont_change_track_record():
    base = [*BACKGROUND, film(1, "2005-01-01", director=7, actors=[8]), film(2, "2010-01-01", director=7, actors=[8])]
    later = film(3, "2012-01-01", director=7, actors=[8], revenue=900_000_000, vote_average=9.9)
    before = features(*base).loc[2]
    after = features(*base, later).loc[2]
    cols = [c for c in before.index if c.startswith(("director_", "actor_", "company_"))]
    pd.testing.assert_series_equal(before[cols], after[cols])
    assert before["director_n_prior"] == 1


def test_same_day_film_is_not_prior():
    df = features(*BACKGROUND, film(1, "2010-05-01", director=7), film(2, "2010-05-01", director=7))
    assert df.loc[1, "director_n_prior"] == 0 and df.loc[2, "director_n_prior"] == 0


def test_debut_has_no_track_record():
    df = features(*BACKGROUND, film(1, director=7, actors=[8, 9], company=5))
    row = df.loc[1]
    assert row["director_n_prior"] == 0 and row["actor_n_prior_max"] == 0 and row["company_n_prior"] == 0
    assert np.isnan(row["director_med_log_rev"]) and np.isnan(row["actor_score_mean"])


def test_revenue_stats_skip_prior_films_without_financials():
    df = features(*BACKGROUND,
                  film(1, "2001-01-01", director=7, revenue=None),  # counted, but no revenue to use
                  film(2, "2002-01-01", director=7, revenue=100_000_000),
                  film(3, "2010-01-01", director=7))
    assert df.loc[3, "director_n_prior"] == 2
    assert df.loc[3, "director_med_log_rev"] == pytest.approx(np.log1p(100_000_000))
    assert 1 not in df.index  # films without financials are history only, not training rows


def test_prior_score_needs_enough_votes():
    df = features(*BACKGROUND, film(1, "2001-01-01", director=7, vote_average=2.0, vote_count=3),
                  film(2, "2002-01-01", director=7, vote_average=8.0), film(3, "2010-01-01", director=7))
    assert df.loc[3, "director_score"] == pytest.approx(8.0)


def test_top_three_actors_only_and_years_active():
    df = features(*BACKGROUND, film(1, "2001-01-01", actors=[8, 50]),
                  film(2, "2010-01-01", actors=[1, 2, 3, 8]))  # actor 8 is 4th billed in film 2
    assert df.loc[2, "actor_n_prior_max"] == 0
    df = features(*BACKGROUND, film(1, "2001-01-01", director=7), film(2, "2010-01-01", director=7))
    assert df.loc[2, "director_years_active"] == 9


def test_no_forbidden_columns():
    cols = feature_columns(features(*BACKGROUND, film(1)).reset_index())
    assert not FORBIDDEN & set(cols)
    assert not [c for c in cols if c.startswith("y_")]


def test_genres_and_language_bucket():
    df = features(*BACKGROUND, film(1, genres=("Drama", "Science Fiction")), film(2, "2019-01-01", language="xx"))
    assert df.loc[1, ["genre_drama", "genre_science_fiction", "n_genres"]].tolist() == [1, 1, 2]
    assert df.loc[1, "original_language"] == "en"
    assert df.loc[2, "original_language"] == "other"  # languages come from the train years only
    assert df.loc[2, "genre_science_fiction"] == 0


def test_overview_encoder_fits_on_train_years_only():
    h1 = history(*BACKGROUND, film(1, "2019-01-01", overview="zombie apocalypse outbreak"))
    h2 = history(*BACKGROUND, film(1, "2019-01-01", overview="pirate treasure island"))
    vocab = lambda h: set(fit_spec(h, n_topics=2, min_df=1).overview.tfidf.vocabulary_)
    assert vocab(h1) == vocab(h2)
    assert "zombie" not in vocab(h1)


def test_score_target_needs_enough_votes():
    df = features(*BACKGROUND, film(1, vote_average=0.0, vote_count=0), film(2, vote_average=7.5, vote_count=50))
    assert np.isnan(df.loc[1, "y_score"]) and df.loc[2, "y_score"] == 7.5
    assert not np.isnan(df.loc[1, "y_log_revenue"])  # still a revenue/budget training row


def test_pre_1980_films_are_history_not_targets():
    df = features(*BACKGROUND, film(1, "1975-01-01", director=7), film(2, "2010-01-01", director=7))
    assert 1 not in df.index
    assert df.loc[2, "director_n_prior"] == 1


def test_empty_overview_gives_zero_topics():
    df = features(*BACKGROUND, film(1, overview=""))
    assert df.loc[1, ["ov_0", "ov_1", "overview_words"]].tolist() == [0, 0, 0]
