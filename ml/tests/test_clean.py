import sqlite3
from contextlib import closing
from datetime import date

import pandas as pd
import pytest

from clean_data import clean, clean_amount, inflation_factor, load_cpi, merge_sources, tmdb_to_tables, write_sqlite

TODAY = date(2026, 10, 19)


def record(movie_id, title=None, release_date="2010-06-01", budget=10_000_000, revenue=50_000_000,
           status="Released", **extra):
    return {
        "id": movie_id, "title": title or f"Movie {movie_id}", "release_date": release_date, "budget": budget, "revenue": revenue,
        "status": status, "runtime": 100, "genres": [{"id": 18, "name": "Drama"}],
        "credits": {"cast": [{"id": 1, "name": "Actor", "order": 0}],
                    "crew": [{"id": 2, "name": "Director Person", "job": "Director"},
                             {"id": 3, "name": "Editor Person", "job": "Editor"}]},
        **extra,
    }


def run(records, kaggle_records=()):
    tables = tmdb_to_tables(records)
    if kaggle_records:
        kaggle = tmdb_to_tables(list(kaggle_records))
        kaggle["movies"]["source"] = "kaggle"
        tables = merge_sources(tables, kaggle)
    return clean(tables, load_cpi(), today=TODAY)


def test_clean_amount_nulls_zero_tiny_and_missing():
    out = clean_amount(pd.Series([0, None, 1, 4_999, 5_000, 200_000_000]))
    assert out.isna().tolist() == [True, True, True, True, False, False]


def test_has_financials_needs_both_valid_amounts():
    m = run([record(1), record(2, budget=0), record(3, revenue=1), record(4, budget=None),
             record(5, budget=20_000_000, revenue=20_000_000)])["movies"].set_index("id")
    assert m["has_financials"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 0, 5: 0}
    assert m.loc[5, "budget"] == 20_000_000 and pd.isna(m.loc[5, "revenue"])  # revenue == budget is a placeholder


def test_movies_without_financials_are_kept_for_browsing():
    assert len(run([record(1, budget=0, revenue=0)])["movies"]) == 1


def test_duplicates_removed():
    m = run([record(1), record(1), record(2, title="Same", release_date="2000-01-01"),
             record(3, title="Same", release_date="2000-01-01")])["movies"]
    assert sorted(m["id"]) == [1, 2]


def test_unreleased_and_future_movies_dropped():
    m = run([record(1), record(2, status="Rumored"), record(3, release_date="2027-01-01"),
             record(4, release_date="")])["movies"]
    assert m["id"].tolist() == [1]


def test_tmdb_wins_over_kaggle():
    out = run([record(1, budget=10_000_000)], [record(1, budget=99_000_000), record(2)])
    m = out["movies"].set_index("id")
    assert m.loc[1, "budget"] == 10_000_000 and m.loc[1, "source"] == "tmdb"
    assert m.loc[2, "source"] == "kaggle"
    assert sorted(out["genres"]["movie_id"]) == [1, 2]  # no duplicate child rows for id 1


def test_crew_trimmed_to_key_jobs():
    assert run([record(1)])["crew"]["job"].tolist() == ["Director"]


def test_inflation_factor():
    cpi = load_cpi()
    f = inflation_factor(pd.Series([2024, 1980, 2025]), cpi)
    assert f[0] == 1.0
    assert f[1] == pytest.approx(3.81, abs=0.01)
    assert f[2] == 1.0


def test_adjusted_amounts():
    m = run([record(1, release_date="1980-05-01", budget=1_000_000)])["movies"]
    assert m.loc[0, "budget_adj"] == pytest.approx(3_806_905, rel=1e-4)


def test_write_sqlite_round_trip_and_rebuild(tmp_path):
    db = tmp_path / "movies.db"
    tables = run([record(1), record(2, budget=0)])
    write_sqlite(tables, db)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("insert into movies (id, title) values (99, 'stale row')")
    write_sqlite(tables, db)  # rebuild replaces the file, it doesn't append

    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("select id, has_financials from movies order by id").fetchall() == [(1, 1), (2, 0)]
        assert conn.execute("select count(*) from crew").fetchone()[0] == 2
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("insert into movies (id, title) values (1, 'duplicate')")
    db.unlink()  # all connections are closed, so Windows can delete the file
    assert not (tmp_path / "movies.tmp.db").exists()
