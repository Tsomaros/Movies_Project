"""Budget, revenue and score ranges for a new film, with the top factors behind each.

Loads the models saved by train.py (ml/models/predictor.joblib) and the film
history from ml/data/movies.db. The new film is added to the history as a film
without outcomes and goes through the same feature code as training, so its
track records only use films released before its release date. Unknown or
missing director, actor and company ids simply have no track record.

Usage:
    python ml/src/predict.py      # prints the prediction for an example film
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import TOP_BILLED, build_features, load_history
from train import MODEL_PATH

NEW_ID = -1  # id of the film being predicted inside the history copy
ROUGH_LABEL = "rough estimate"
SCORE_RANGE = (0.0, 10.0)


@dataclass
class FilmInput:
    """What a creator enters. People and the company are TMDB ids; the budget is optional, in today's dollars."""

    genres: list[str]
    runtime: int
    release_date: str  # YYYY-MM-DD
    original_language: str = "en"
    company_id: int | None = None
    director_id: int | None = None
    actor_ids: list[int] = field(default_factory=list)  # top billed first; only the first 3 are used
    budget: float | None = None
    overview: str = ""


class Predictor:
    def __init__(self, bundle: dict | str | Path = MODEL_PATH, history: dict[str, pd.DataFrame] | None = None):
        self.bundle = bundle if isinstance(bundle, dict) else joblib.load(bundle)
        self.spec, self.models = self.bundle["spec"], self.bundle["models"]
        self.history = history if history is not None else load_history()

    def _check(self, film: FilmInput) -> pd.Timestamp:
        unknown = sorted(set(film.genres) - set(self.spec.genres))
        if unknown:
            raise ValueError(f"Unknown genres: {unknown}")
        if film.budget is not None and film.budget <= 0:
            raise ValueError("Budget must be positive")
        if not film.runtime or film.runtime <= 0:
            raise ValueError("Runtime must be positive")
        return pd.Timestamp(film.release_date)

    def _with_film(self, film: FilmInput) -> dict[str, pd.DataFrame]:
        """A copy of the history with the new film added as a film without outcomes."""
        released = self._check(film)
        movie = {"id": NEW_ID, "title": "New film", "release_date": released.strftime("%Y-%m-%d"),
                 "release_year": released.year, "release_month": released.month, "runtime": film.runtime,
                 "original_language": film.original_language, "overview": film.overview or "",
                 "budget_adj": np.nan if film.budget is None else float(film.budget), "revenue_adj": np.nan,
                 "vote_average": np.nan, "vote_count": 0, "has_financials": 0}
        rows = {
            "movies": [movie],
            "genres": [{"movie_id": NEW_ID, "genre_id": -1, "name": g} for g in dict.fromkeys(film.genres)],
            "companies": [{"movie_id": NEW_ID, "company_id": film.company_id, "company_order": 0}]
            if film.company_id is not None else [],
            "cast_members": [{"movie_id": NEW_ID, "person_id": p, "cast_order": i}
                             for i, p in enumerate(film.actor_ids[:TOP_BILLED])],
            "crew": [{"movie_id": NEW_ID, "person_id": film.director_id, "job": "Director"}]
            if film.director_id is not None else [],
        }
        return {name: pd.concat([table, pd.DataFrame(rows[name], columns=table.columns)], ignore_index=True)
                if rows[name] else table for name, table in self.history.items()}

    def features(self, film: FilmInput) -> pd.DataFrame:
        """The film's one-row feature frame, built exactly as for training."""
        df, _ = build_features(self._with_film(film), self.spec, ids=[NEW_ID])
        return df

    def predict(self, film: FilmInput, top_k: int = 5) -> dict:
        row = self.features(film)
        has_budget = film.budget is not None
        out = {}
        for kind, target, name in [("budget", "y_log_budget", "lgbm"),
                                   ("revenue", "y_log_revenue", "lgbm" if has_budget else "lgbm_no_budget"),
                                   ("score", "y_score", "lgbm")]:
            model = self.models[target, name]
            point, low, high = (float(v[0]) for v in model.predict(row))
            effects = model.contributions(row).iloc[0].drop("base")
            top = effects[effects.abs().sort_values(ascending=False).index[:top_k]]
            if kind == "score":
                low, point, high = np.clip([low, point, high], *SCORE_RANGE)
                drivers = [{"factor": f, "effect": float(e), "text": f"{f}: {e:+.1f} points"} for f, e in top.items()]
            else:
                low, point, high = np.expm1([low, point, high])
                drivers = [{"factor": f, "effect": float(e), "text": f"{f}: {np.expm1(e):+.0%}"} for f, e in top.items()]
            out[kind] = {"low": float(low), "point": float(point), "high": float(high), "model": name,
                         "label": None, "drivers": drivers}
        if self.bundle.get("budget_rough"):
            out["budget"]["label"] = ROUGH_LABEL
        if has_budget:
            out["budget"]["given"] = float(film.budget)
        r = row.iloc[0]
        out["prior_films"] = {"director": int(r["director_n_prior"]), "actors_max": int(r["actor_n_prior_max"]),
                              "company": int(r["company_n_prior"])}
        return out


def main() -> None:
    film = FilmInput(genres=["Action", "Science Fiction"], runtime=120, release_date="2027-07-15",
                     overview="A crew of astronauts races to stop a rogue AI.")
    print(json.dumps(Predictor().predict(film), indent=2))


if __name__ == "__main__":
    main()
