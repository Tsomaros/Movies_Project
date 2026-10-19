"""Clean the raw TMDB + Kaggle data and store it in SQLite.

Reads ml/data/raw/tmdb_movies.jsonl (primary) and the Kaggle TMDB 5000 CSVs
(fallback, only for movies the API pull doesn't have), cleans them and writes
ml/data/movies.db. Every movie is kept for browsing; `has_financials` marks the
rows with a valid budget and revenue, which are the ones used for training.

Usage:
    python ml/src/clean_data.py
"""

import argparse
import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import pandas as pd

from collect_tmdb import KEY_CREW_JOBS, MOVIES_PATH, RAW_DIR, TOP_CAST

ROOT = Path(__file__).resolve().parents[2]
KAGGLE_DIR = RAW_DIR / "kaggle"
CPI_PATH = ROOT / "ml" / "data" / "cpi_us.csv"
DB_PATH = ROOT / "ml" / "data" / "movies.db"

MIN_AMOUNT = 5_000  # budgets/revenues below this are unit errors (e.g. "1" meaning $1M)
CPI_BASE_YEAR = 2024

MOVIE_COLS = [
    "id", "title", "original_title", "overview", "release_date", "runtime", "original_language",
    "budget", "revenue", "vote_average", "vote_count", "popularity", "poster_path",
    "certification", "status", "source",
]
CHILD_TABLES = {
    "genres": ["movie_id", "genre_id", "name"],
    "companies": ["movie_id", "company_id", "name", "company_order"],
    "cast_members": ["movie_id", "person_id", "name", "character", "cast_order"],
    "crew": ["movie_id", "person_id", "name", "job"],
}

log = logging.getLogger("clean_data")


def us_certification(release_dates: dict | None) -> str | None:
    """US rating (PG-13, R, ...), preferring the theatrical release (type 3)."""
    for country in (release_dates or {}).get("results", []):
        if country.get("iso_3166_1") != "US":
            continue
        entries = sorted(country.get("release_dates", []), key=lambda d: (d.get("type") != 3, d.get("type", 99)))
        for entry in entries:
            if entry.get("certification"):
                return entry["certification"]
    return None


def child_rows(movie_id: int, genres: list, companies: list, cast: list, crew: list) -> dict[str, list[dict]]:
    """Normalize the list fields shared by both sources into child-table rows."""
    cast = sorted(cast, key=lambda c: c.get("order", 999))[:TOP_CAST]
    return {
        "genres": [{"movie_id": movie_id, "genre_id": g["id"], "name": g["name"]} for g in genres],
        "companies": [
            {"movie_id": movie_id, "company_id": c["id"], "name": c["name"], "company_order": i}
            for i, c in enumerate(companies)
        ],
        "cast_members": [
            {"movie_id": movie_id, "person_id": c["id"], "name": c["name"],
             "character": c.get("character"), "cast_order": c.get("order")}
            for c in cast
        ],
        "crew": [
            {"movie_id": movie_id, "person_id": c["id"], "name": c["name"], "job": c["job"]}
            for c in crew if c.get("job") in KEY_CREW_JOBS
        ],
    }


def to_tables(movies: list[dict], children: list[dict[str, list[dict]]]) -> dict[str, pd.DataFrame]:
    tables = {"movies": pd.DataFrame(movies, columns=MOVIE_COLS)}
    for name, cols in CHILD_TABLES.items():
        tables[name] = pd.DataFrame([row for c in children for row in c[name]], columns=cols)
    return tables


def tmdb_to_tables(records: list[dict]) -> dict[str, pd.DataFrame]:
    movies, children = [], []
    for r in records:
        movie = {col: r.get(col) for col in MOVIE_COLS}
        movie["certification"] = us_certification(r.get("release_dates"))
        movie["source"] = "tmdb"
        movies.append(movie)
        credits = r.get("credits") or {}
        children.append(child_rows(r["id"], r.get("genres") or [], r.get("production_companies") or [],
                                   credits.get("cast", []), credits.get("crew", [])))
    return to_tables(movies, children)


def kaggle_to_tables(movies_df: pd.DataFrame, credits_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    credits = credits_df.set_index("movie_id")
    movies, children = [], []
    for r in movies_df.to_dict("records"):
        r = {k: (None if pd.isna(v) else v) for k, v in r.items()}
        movie = {col: r.get(col) for col in MOVIE_COLS}
        movie["source"] = "kaggle"
        movies.append(movie)
        cast = json.loads(credits.at[r["id"], "cast"]) if r["id"] in credits.index else []
        crew = json.loads(credits.at[r["id"], "crew"]) if r["id"] in credits.index else []
        children.append(child_rows(r["id"], json.loads(r["genres"] or "[]"),
                                   json.loads(r["production_companies"] or "[]"), cast, crew))
    return to_tables(movies, children)


def load_tmdb() -> dict[str, pd.DataFrame]:
    with MOVIES_PATH.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    log.info("TMDB API: %d records", len(records))
    return tmdb_to_tables(records)


def load_kaggle() -> dict[str, pd.DataFrame]:
    movies = pd.read_csv(KAGGLE_DIR / "tmdb_5000_movies.csv")
    credits = pd.read_csv(KAGGLE_DIR / "tmdb_5000_credits.csv")
    log.info("Kaggle: %d movies", len(movies))
    return kaggle_to_tables(movies, credits)


def merge_sources(tmdb: dict[str, pd.DataFrame], kaggle: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """TMDB API rows win; Kaggle only adds movies the API pull doesn't have."""
    new_ids = set(kaggle["movies"]["id"]) - set(tmdb["movies"]["id"])
    log.info("Kaggle adds %d movies not in the API pull", len(new_ids))
    merged = {}
    for name, df in tmdb.items():
        extra = kaggle[name]
        extra = extra[extra["id" if name == "movies" else "movie_id"].isin(new_ids)]
        merged[name] = pd.concat([df, extra], ignore_index=True)
    return merged


def clean_amount(s: pd.Series) -> pd.Series:
    """Zero, missing or implausibly small money values become NA."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= MIN_AMOUNT).astype("Int64")


def load_cpi(path: Path = CPI_PATH) -> pd.Series:
    cpi = pd.read_csv(path, comment="#")
    return cpi.set_index("year")["cpi"]


def inflation_factor(years: pd.Series, cpi: pd.Series, base_year: int = CPI_BASE_YEAR) -> pd.Series:
    """Multiplier to convert money from `years` into base-year dollars. Years after the table: 1.0."""
    factor = years.map(cpi[base_year] / cpi)
    return factor.mask(years > cpi.index.max(), 1.0)


def clean(tables: dict[str, pd.DataFrame], cpi: pd.Series, today: date | None = None) -> dict[str, pd.DataFrame]:
    today = today or date.today()
    m = tables["movies"].copy()

    def drop(mask: pd.Series, reason: str) -> pd.DataFrame:
        log.info("Dropped %d rows: %s", int(mask.sum()), reason)
        return m[~mask]

    m = drop(m["id"].duplicated(), "duplicate id")
    m["release_date"] = pd.to_datetime(m["release_date"], errors="coerce")
    m = drop(m.duplicated(subset=["title", "release_date"]), "duplicate title + release date")
    m = drop(m["status"] != "Released", "not released")
    m = drop(m["release_date"].isna() | (m["release_date"] > pd.Timestamp(today)), "missing or future release date")

    for col in ["budget", "revenue"]:
        raw = pd.to_numeric(m[col], errors="coerce").fillna(0)
        m[col] = clean_amount(m[col])
        tiny = int(((raw > 0) & m[col].isna()).sum())
        log.info("%s: %d set to NULL (under $%s), %d missing or zero", col, tiny, f"{MIN_AMOUNT:,}", int((raw == 0).sum()))
    placeholder = (m["budget"] == m["revenue"]).fillna(False).astype(bool)  # revenue copied from budget
    m.loc[placeholder, "revenue"] = pd.NA
    log.info("revenue: %d set to NULL (equal to budget, likely a placeholder)", int(placeholder.sum()))
    m["runtime"] = pd.to_numeric(m["runtime"], errors="coerce").where(lambda r: r > 0)

    m["release_year"] = m["release_date"].dt.year
    m["release_month"] = m["release_date"].dt.month
    m["release_date"] = m["release_date"].dt.strftime("%Y-%m-%d")
    factor = inflation_factor(m["release_year"], cpi)
    for col in ["budget", "revenue"]:
        m[f"{col}_adj"] = (m[col].astype("Float64") * factor).round().astype("Int64")
    m["has_financials"] = (m["budget"].notna() & m["revenue"].notna()).astype(int)
    log.info("%d movies kept, %d with financials", len(m), int(m["has_financials"].sum()))

    out = {"movies": m.reset_index(drop=True)}
    keep = set(m["id"])
    for name in CHILD_TABLES:
        df = tables[name]
        out[name] = df[df["movie_id"].isin(keep)].drop_duplicates().reset_index(drop=True)
    return out


def write_sqlite(tables: dict[str, pd.DataFrame], path: Path = DB_PATH) -> None:
    """Build the DB in a temp file, then swap it in, so a failed run never leaves a broken DB."""
    tmp = path.with_suffix(".tmp.db")
    tmp.unlink(missing_ok=True)
    with closing(sqlite3.connect(tmp)) as conn, conn:  # closing() releases the file lock on Windows
        for name, df in tables.items():
            df.to_sql(name, conn, index=False)
        conn.executescript("""
            CREATE UNIQUE INDEX ix_movies_id ON movies(id);
            CREATE INDEX ix_movies_year ON movies(release_year);
            CREATE INDEX ix_movies_title ON movies(title);
            CREATE INDEX ix_genres_movie ON genres(movie_id);
            CREATE INDEX ix_companies_movie ON companies(movie_id);
            CREATE INDEX ix_cast_movie ON cast_members(movie_id);
            CREATE INDEX ix_cast_person ON cast_members(person_id);
            CREATE INDEX ix_crew_movie ON crew(movie_id);
            CREATE INDEX ix_crew_person ON crew(person_id);
        """)
    try:
        os.replace(tmp, path)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{path} is in use by another program (DB browser, backend?). Close it and re-run.")
    log.info("Wrote %s", path)


def summarize(tables: dict[str, pd.DataFrame]) -> None:
    m = tables["movies"]
    print(f"Movies: {len(m):,} ({int(m['has_financials'].sum()):,} with budget + revenue)")
    print(f"Release years: {int(m['release_year'].min())}-{int(m['release_year'].max())}")
    print("By source: " + ", ".join(f"{k}={v:,}" for k, v in m["source"].value_counts().items()))
    for name in CHILD_TABLES:
        print(f"{name}: {len(tables[name]):,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tables = clean(merge_sources(load_tmdb(), load_kaggle()), load_cpi())
    write_sqlite(tables)
    summarize(tables)


if __name__ == "__main__":
    main()
