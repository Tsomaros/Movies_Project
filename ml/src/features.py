"""Build model features from ml/data/movies.db.

Base features (genres, runtime, release month/year, language, budget, overview
topics) plus director, actor and lead-company track records. Track
records only use films released strictly before the target film, so nothing a
model sees could only be known after release. popularity, vote_count and
vote_average of the target film are never features.

Usage:
    python ml/src/features.py      # writes ml/data/features.csv and prints a summary
"""

import argparse
import logging
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from clean_data import CHILD_TABLES, DB_PATH, ROOT

FEATURES_PATH = ROOT / "ml" / "data" / "features.csv"

TRAIN_END_YEAR = 2015  # train <= 2015, validate 2016-2018, test >= 2019
VALID_END_YEAR = 2018
MIN_TARGET_YEAR = 1980  # older films are history for track records, not training rows
TOP_LANGUAGES = 10
TOP_BILLED = 3
MIN_SCORE_VOTES = 50  # films with fewer votes don't have a usable score (neither as prior nor as target)
N_TOPICS = 20

FORBIDDEN = {"popularity", "vote_count", "vote_average"}  # known only after release
TARGETS = ["y_log_revenue", "y_score", "y_log_budget"]
META = ["id", "title", "release_date", "split"]

log = logging.getLogger("features")


def split_of(years: pd.Series) -> pd.Series:
    return pd.Series(np.select([years <= TRAIN_END_YEAR, years <= VALID_END_YEAR], ["train", "valid"], "test"),
                     index=years.index)


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The time split: {"train": ..., "valid": ..., "test": ...}."""
    return {name: df[df["split"] == name] for name in ["train", "valid", "test"]}


def target_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rows that have this target (y_score is missing for films with too few votes)."""
    return df[df[target].notna()]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_history(path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    """All movies (with or without financials) and their child tables."""
    with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as conn:
        return {name: pd.read_sql(f"select * from {name}", conn) for name in ["movies", *CHILD_TABLES]}


class OverviewEncoder:
    """TF-IDF of the overview reduced to a few SVD topic dimensions, plus the word count.

    A blank overview gets the average values of the training films that have one, so a creator who leaves it
    out gets a neutral overview rather than that of the few (mostly small) films listed without one.
    """

    def __init__(self, n_topics: int = N_TOPICS, min_df: int = 5):
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
        self.svd = TruncatedSVD(n_topics, random_state=0)
        self.columns = [f"ov_{i}" for i in range(n_topics)]

    def fit(self, texts: pd.Series) -> "OverviewEncoder":
        texts = texts.fillna("")
        self.svd.fit(self.tfidf.fit_transform(texts))
        self.blank = self._encode(texts[texts.str.strip() != ""]).mean()
        return self

    def _encode(self, texts: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame(self.svd.transform(self.tfidf.transform(texts)), index=texts.index, columns=self.columns)
        out["overview_words"] = texts.str.split().str.len().astype(float)
        return out

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        texts = texts.fillna("")
        out = self._encode(texts)
        out.loc[texts.str.strip() == "", self.blank.index] = self.blank.to_numpy()
        return out


@dataclass
class FeatureSpec:
    """Everything learned from the training years, reused unchanged for validation, test and predict()."""

    genres: list[str]
    languages: list[str]
    overview: OverviewEncoder


def fit_spec(history: dict[str, pd.DataFrame], n_topics: int = N_TOPICS, min_df: int = 5,
             train_end: int = TRAIN_END_YEAR) -> FeatureSpec:
    """Fit on films up to `train_end` (the saved app models use a later year than the evaluation)."""
    movies = history["movies"]
    train = movies[movies["release_year"] <= train_end]
    trainable = train[train["has_financials"] == 1]
    languages = trainable["original_language"].value_counts().index[:TOP_LANGUAGES].tolist()
    genres = sorted(history["genres"]["name"].dropna().unique())
    return FeatureSpec(genres, languages, OverviewEncoder(n_topics, min_df).fit(train["overview"]))


def base_features(movies: pd.DataFrame, history: dict[str, pd.DataFrame], spec: FeatureSpec) -> pd.DataFrame:
    m = movies.set_index("id")
    X = pd.DataFrame(index=m.index)

    genres = history["genres"]
    onehot = pd.crosstab(genres["movie_id"], genres["name"]).clip(upper=1)
    onehot = onehot.reindex(index=m.index, columns=spec.genres, fill_value=0)
    X[[f"genre_{slug(g)}" for g in spec.genres]] = onehot.to_numpy()
    X["n_genres"] = onehot.sum(axis=1)

    X["runtime"] = m["runtime"]
    X["release_year"] = m["release_year"]
    X["release_month"] = m["release_month"]
    lang = m["original_language"].where(m["original_language"].isin(spec.languages), "other")
    X["original_language"] = pd.Categorical(lang, categories=[*spec.languages, "other"])
    X["log_budget_adj"] = np.log1p(m["budget_adj"].astype(float))
    return X.join(spec.overview.transform(m["overview"]))


def films_table(movies: pd.DataFrame) -> pd.DataFrame:
    """Per film: date plus the outcomes it contributes to later films' track records."""
    f = movies[["id", "release_date", "release_year"]].copy()
    f["release_date"] = pd.to_datetime(f["release_date"])
    f["log_rev"] = np.log1p(movies["revenue_adj"].where(movies["has_financials"] == 1).astype(float))
    f["log_budget"] = np.log1p(movies["budget_adj"].where(movies["has_financials"] == 1).astype(float))
    f["score"] = movies["vote_average"].where(movies["vote_count"] >= MIN_SCORE_VOTES).astype(float)
    return f


def track_record(links: pd.DataFrame, films: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Track record of each (target, entity) pair, from films released strictly before the target.

    links:   (movie_id, entity_id) for the history films, e.g. every director credit.
    targets: (key, entity_id, release_date, release_year), one row per entity of a target film.
             A target needn't exist in `films`, which is how predict() handles a new film.
    Returns one row per target row: n_prior (0 if unknown), med_log_rev, med_log_budget, mean_score, years_active.
    """
    hist = links.drop_duplicates().merge(films, left_on="movie_id", right_on="id")
    hist = hist.rename(columns={"release_date": "prior_date", "release_year": "prior_year"})
    pairs = targets.merge(hist[["entity_id", "prior_date", "prior_year", "log_rev", "log_budget", "score"]],
                          on="entity_id")
    prior = pairs[pairs["prior_date"] < pairs["release_date"]]
    stats = prior.groupby(["key", "entity_id"]).agg(
        n_prior=("prior_date", "size"),
        med_log_rev=("log_rev", "median"),
        med_log_budget=("log_budget", "median"),
        mean_score=("score", "mean"),
        first_year=("prior_year", "min"),
    ).reset_index()
    out = targets.merge(stats, on=["key", "entity_id"], how="left")
    out["n_prior"] = out["n_prior"].fillna(0).astype(int)
    out["years_active"] = out["release_year"] - out["first_year"]
    return out[["key", "entity_id", "n_prior", "med_log_rev", "med_log_budget", "mean_score", "years_active"]]


def target_links(links: pd.DataFrame, films: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    t = links[links["movie_id"].isin(ids)].drop_duplicates()
    t = t.merge(films[["id", "release_date", "release_year"]], left_on="movie_id", right_on="id")
    return t.rename(columns={"movie_id": "key"})[["key", "entity_id", "release_date", "release_year"]]


def track_record_features(history: dict[str, pd.DataFrame], ids: pd.Index) -> pd.DataFrame:
    films = films_table(history["movies"])
    crew, cast, companies = history["crew"], history["cast_members"], history["companies"]

    directors = crew.loc[crew["job"] == "Director", ["movie_id", "person_id"]].rename(columns={"person_id": "entity_id"})
    d = track_record(directors, films, target_links(directors, films, ids)).groupby("key").agg(
        director_n_prior=("n_prior", "max"),  # co-directed films: most experienced director, mean of the stats
        director_med_log_rev=("med_log_rev", "mean"),
        director_med_log_budget=("med_log_budget", "mean"),
        director_score=("mean_score", "mean"),
        director_years_active=("years_active", "max"),
    )

    actors = cast[["movie_id", "person_id"]].rename(columns={"person_id": "entity_id"})  # history: any top-10 billing
    top_billed = cast.loc[cast["cast_order"] < TOP_BILLED, ["movie_id", "person_id"]].rename(columns={"person_id": "entity_id"})
    a = track_record(actors, films, target_links(top_billed, films, ids)).groupby("key").agg(
        actor_n_prior_mean=("n_prior", "mean"),
        actor_n_prior_max=("n_prior", "max"),
        actor_med_log_rev_mean=("med_log_rev", "mean"),
        actor_med_log_rev_max=("med_log_rev", "max"),
        actor_med_log_budget_mean=("med_log_budget", "mean"),
        actor_med_log_budget_max=("med_log_budget", "max"),
        actor_score_mean=("mean_score", "mean"),
    )

    lead = companies.loc[companies["company_order"] == 0, ["movie_id", "company_id"]].rename(columns={"company_id": "entity_id"})
    c = track_record(lead, films, target_links(lead, films, ids)).groupby("key").agg(
        company_n_prior=("n_prior", "max"),
        company_med_log_rev=("med_log_rev", "max"),
        company_med_log_budget=("med_log_budget", "max"),
    )

    out = pd.DataFrame(index=ids).join(d).join(a).join(c)
    counts = [col for col in out.columns if "_n_prior" in col]
    out[counts] = out[counts].fillna(0)  # no director/cast/company listed = unknown
    return out


def check_no_leakage(columns) -> None:
    bad = [c for c in columns if c in FORBIDDEN or c in TARGETS or c.startswith("y_")]
    if bad:
        raise AssertionError(f"Leaky feature columns: {bad}")


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in META and c not in TARGETS]
    check_no_leakage(cols)
    return cols


def build_features(history: dict[str, pd.DataFrame], spec: FeatureSpec | None = None,
                   ids=None) -> tuple[pd.DataFrame, FeatureSpec]:
    """Features + targets for `ids` (default: every movie with financials from MIN_TARGET_YEAR on)."""
    movies = history["movies"]
    spec = spec or fit_spec(history)
    if ids is not None:
        targets = movies[movies["id"].isin(ids)]
    else:
        targets = movies[(movies["has_financials"] == 1) & (movies["release_year"] >= MIN_TARGET_YEAR)]

    X = base_features(targets, history, spec)
    X = X.join(track_record_features(history, X.index))
    check_no_leakage(X.columns)

    t = targets.set_index("id")
    meta = pd.DataFrame({"title": t["title"], "release_date": t["release_date"], "split": split_of(t["release_year"])})
    y = pd.DataFrame({
        "y_log_revenue": np.log1p(t["revenue_adj"].astype(float)),
        "y_score": t["vote_average"].where(t["vote_count"] >= MIN_SCORE_VOTES).astype(float),
        "y_log_budget": np.log1p(t["budget_adj"].astype(float)),
    })
    df = meta.join(X).join(y).rename_axis("id").reset_index()
    return df, spec


def summarize(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df):,}, feature columns: {len(feature_columns(df))}")
    by_split = df.groupby("split", sort=False).agg(
        rows=("id", "size"),
        score_rows=("y_score", "count"),
        unknown_director=("director_n_prior", lambda s: f"{(s == 0).mean():.0%}"),
        unknown_all_actors=("actor_n_prior_max", lambda s: f"{(s == 0).mean():.0%}"),
        unknown_company=("company_n_prior", lambda s: f"{(s == 0).mean():.0%}"),
    )
    print(by_split.reindex(["train", "valid", "test"]).to_string())
    nan = df[feature_columns(df)].isna().mean().sort_values(ascending=False)
    print("Highest NaN rates: " + ", ".join(f"{c}={v:.0%}" for c, v in nan[nan > 0].head(8).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df, _ = build_features(load_history())
    df.to_csv(FEATURES_PATH, index=False)
    log.info("Wrote %s", FEATURES_PATH)
    summarize(df)


if __name__ == "__main__":
    main()
