"""A small synthetic DB and model bundle, so the API tests need neither the real data nor the trained models."""

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "ml" / "src"), str(ROOT / "ml" / "tests")]

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from clean_data import write_sqlite  # noqa: E402
from features import fit_spec  # noqa: E402
from test_features import film, history  # noqa: E402
from test_predict import FAST, synthetic_history  # noqa: E402
from train import fit_live  # noqa: E402

# Browse-only films with titles and names the tests look for.
EXTRAS = {
    5000: ("Batman Begins", film(5000, "2005-06-15", director=900, actors=[901], company=902, genres=("Action",),
                                 vote_average=9.9, vote_count=3)),
    5001: ("100% Wolf", film(5001, "2020-03-01", genres=("Animation",))),
    5002: ("O'Brien_s Day", film(5002, "2021-04-01")),
}
NAMES = {("crew", 1): "Christopher Nolan", ("crew", 900): "Nolan Grey", ("cast_members", 901): "Nolan North"}


def browse_history() -> dict[str, pd.DataFrame]:
    h = synthetic_history()
    extra = history(*[f for _, f in EXTRAS.values()])
    h = {name: pd.concat([h[name], extra[name]], ignore_index=True) for name in h}
    m = h["movies"]
    for movie_id, (title, _) in EXTRAS.items():
        m.loc[m["id"] == movie_id, "title"] = title
    for (table, person), name in NAMES.items():
        h[table].loc[h[table]["person_id"] == person, "name"] = name
    m["original_title"] = m["title"]
    m["popularity"] = m["vote_count"] / 10 + m["id"] % 7
    m["poster_path"] = [f"/p{i}.jpg" if i % 2 == 0 else None for i in m["id"]]
    m["certification"], m["status"], m["source"] = None, "Released", "tmdb"
    m["budget"], m["revenue"] = m["budget_adj"], m["revenue_adj"]
    return h


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    h = browse_history()
    bundle = fit_live(h, train_end=2023, spec=fit_spec(h, n_topics=2, min_df=1, train_end=2023), **FAST)
    bundle.update(budget_rough=False, trained_on="test")
    folder = tmp_path_factory.mktemp("api")
    write_sqlite(h, folder / "movies.db")
    joblib.dump(bundle, folder / "predictor.joblib")
    return h, bundle, Settings(folder / "movies.db", folder / "predictor.joblib", ["http://localhost:5173"])


@pytest.fixture(scope="session")
def client(data):
    with TestClient(create_app(data[2])) as c:
        yield c
