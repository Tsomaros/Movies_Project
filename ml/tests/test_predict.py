import joblib
import numpy as np
import pandas as pd
import pytest

from features import fit_spec
from predict import ROUGH_LABEL, FilmInput, Predictor
from test_features import film, history
from train import fit_live

FAST = dict(n_estimators=60, learning_rate=0.1, min_child_samples=5, n_jobs=1)


def synthetic_history(n=460, seed=0):
    """Films 1980-2025; each director has a level that drives budget, revenue and score."""
    rng = np.random.default_rng(seed)
    level = rng.normal(0, 1, 20)
    films = []
    for i in range(n):
        d = int(rng.integers(20))
        budget = 1e7 * np.exp(level[d] + rng.normal(0, 0.3))
        films.append(film(i, f"{1980 + i % 46}-{1 + i % 12:02d}-15", director=d + 1, actors=[100 + d, 200 + i % 7],
                          company=1 + d % 4, budget=budget, revenue=budget * np.exp(1 + level[d] + rng.normal(0, 0.3)),
                          genres=("Drama",) if i % 2 else ("Action", "Comedy"),
                          vote_average=float(np.clip(6 + level[d] + rng.normal(0, 0.2), 0, 10))))
    return history(*films)


@pytest.fixture(scope="module")
def setup():
    h = synthetic_history()
    bundle = fit_live(h, train_end=2023, spec=fit_spec(h, n_topics=2, min_df=1, train_end=2023), **FAST)
    bundle["budget_rough"] = False
    return h, bundle, Predictor(bundle, h)


def new_film(**changes):
    base = dict(genres=["Drama"], runtime=110, release_date="2026-05-01", director_id=1, actor_ids=[100, 201],
                company_id=2, overview="family drama about a farm")
    return FilmInput(**{**base, **changes})


def test_ranges_are_ordered_and_in_units(setup):
    _, _, predictor = setup
    out = predictor.predict(new_film(budget=20_000_000))
    for kind in ["budget", "revenue", "score"]:
        assert out[kind]["low"] <= out[kind]["point"] <= out[kind]["high"]
    assert out["budget"]["low"] > 0 and out["revenue"]["low"] > 0
    assert 0 <= out["score"]["low"] and out["score"]["high"] <= 10
    assert out["budget"]["given"] == 20_000_000


def test_revenue_model_follows_the_budget(setup):
    _, _, predictor = setup
    assert predictor.predict(new_film(budget=20_000_000))["revenue"]["model"] == "lgbm"
    assert predictor.predict(new_film())["revenue"]["model"] == "lgbm_no_budget"


def test_unknown_people_have_no_track_record(setup):
    _, _, predictor = setup
    out = predictor.predict(new_film(director_id=999_999, actor_ids=[888_888], company_id=777_777))
    assert out["prior_films"] == {"director": 0, "actors_max": 0, "company": 0}
    assert np.isfinite(out["revenue"]["point"])
    assert predictor.predict(new_film(director_id=None, actor_ids=[], company_id=None))["prior_films"]["director"] == 0


def test_later_films_dont_reach_the_features(setup):
    h, bundle, predictor = setup
    later = history(film(10_000, "2026-09-01", director=1, actors=[100], company=2, revenue=9e9, vote_average=10.0))
    h2 = {name: pd.concat([h[name], later[name]], ignore_index=True) for name in h}
    a = predictor.features(new_film())
    b = Predictor(bundle, h2).features(new_film())
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[0, "director_n_prior"] == sum(h["crew"]["person_id"] == 1)


def test_contributions_add_up_to_the_point_model(setup):
    _, bundle, predictor = setup
    row = predictor.features(new_film(budget=5_000_000))
    for model in bundle["models"].values():
        np.testing.assert_allclose(model.contributions(row).sum(axis=1), model.point.predict(row[model.cols]))


def test_drivers_are_the_largest_factors(setup):
    _, _, predictor = setup
    drivers = predictor.predict(new_film(), top_k=3)["revenue"]["drivers"]
    assert len(drivers) == 3
    effects = [abs(d["effect"]) for d in drivers]
    assert effects == sorted(effects, reverse=True)
    assert all(d["text"].startswith(d["factor"] + ": ") for d in drivers)


def test_saved_bundle_gives_the_same_predictions(setup, tmp_path):
    h, bundle, predictor = setup
    path = tmp_path / "predictor.joblib"
    joblib.dump(bundle, path)
    assert Predictor(path, h).predict(new_film()) == predictor.predict(new_film())


def test_rough_label_follows_the_bundle(setup):
    h, bundle, predictor = setup
    assert predictor.predict(new_film())["budget"]["label"] is None
    assert Predictor({**bundle, "budget_rough": True}, h).predict(new_film())["budget"]["label"] == ROUGH_LABEL


def test_bad_inputs_are_rejected(setup):
    _, _, predictor = setup
    with pytest.raises(ValueError):
        predictor.predict(new_film(genres=["Space Opera"]))
    with pytest.raises(ValueError):
        predictor.predict(new_film(budget=0))


def test_live_models_use_the_later_years_for_calibration(setup):
    _, bundle, _ = setup
    assert bundle["valid_years"] == (2024, 2025)
    assert set(bundle["valid_coverage"]) == set(bundle["models"])
