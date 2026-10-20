from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

FILM = dict(genres=["Drama"], runtime=110, release_date="2025-05-01", director_id=1, actor_ids=[100, 201],
            company_id=2, overview="family drama about a farm")


def post(client, **changes):
    return client.post("/predict", json={**FILM, **changes})


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "trained_on": "test", "valid_years": [2024, 2025]}


def test_predict_returns_ordered_ranges_and_drivers(client):
    r = post(client, budget=20_000_000)
    assert r.status_code == 200
    body = r.json()
    for kind in ["budget", "revenue", "score"]:
        assert body[kind]["low"] <= body[kind]["point"] <= body[kind]["high"]
        assert body[kind]["drivers"] and all(d["text"].startswith(d["factor"]) for d in body[kind]["drivers"])
    assert body["budget"]["given"] == 20_000_000
    assert body["revenue"]["model"] == "lgbm"
    assert post(client).json()["revenue"]["model"] == "lgbm_no_budget"
    assert body["warnings"] == []


def test_predict_matches_the_predictor(client, data):
    from predict import FilmInput, Predictor
    h, bundle, _ = data
    direct = Predictor(bundle, h).predict(FilmInput(**FILM))
    body = post(client).json()
    assert body["revenue"]["point"] == pytest.approx(direct["revenue"]["point"])
    assert body["prior_films"] == direct["prior_films"]


def test_unknown_people_get_a_warning_not_an_error(client):
    r = post(client, director_id=999_999, actor_ids=[888_888], company_id=777_777)
    assert r.status_code == 200
    body = r.json()
    assert body["prior_films"] == {"director": 0, "actors_max": 0, "company": 0}
    assert "tends toward small films" in body["warnings"][0]
    partial = post(client, director_id=999_999).json()["warnings"]
    assert partial == ["No earlier films in our data for the director, so they count as newcomers."]
    assert post(client, director_id=None, actor_ids=[], company_id=None).status_code == 200


def test_predict_sees_only_earlier_films(client, data):
    from predict import FilmInput, Predictor
    h, bundle, _ = data
    film = {**FILM, "release_date": "2010-05-01"}
    earlier = set(h["movies"].loc[pd.to_datetime(h["movies"]["release_date"]) < "2010-05-01", "id"])
    cut = {name: t[t["id" if name == "movies" else "movie_id"].isin(earlier)] for name, t in h.items()}
    assert len(earlier) < len(h["movies"])
    direct = Predictor(bundle, cut).predict(FilmInput(**film))
    body = client.post("/predict", json=film).json()
    for kind in ["budget", "revenue", "score"]:
        for key in ["low", "point", "high"]:
            assert body[kind][key] == pytest.approx(direct[kind][key], rel=1e-9)
    assert body["prior_films"] == direct["prior_films"]


def test_concurrent_predictions_match_sequential_ones(client):
    inputs = [{}, {"budget": 20_000_000}, {"director_id": 999_999}, {"actor_ids": []},
              {"budget": 150_000_000, "genres": ["Action", "Comedy"]}, {"company_id": None}, {"runtime": 180},
              {"release_date": "2001-02-03"}]
    sequential = [post(client, **changes).json() for changes in inputs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(lambda changes: post(client, **changes).json(), inputs))
    assert concurrent == sequential
    assert post(client).json() == sequential[0]  # the shared history wasn't changed


def test_budget_limits_and_blank_overview(client):
    assert post(client, budget=2_000_000_000).status_code == 200
    assert post(client, budget=2_000_000_001).status_code == 422
    assert post(client, overview="").status_code == 200


def test_late_year_and_rare_language_warnings(client):
    notes = post(client, release_date="2028-01-01", original_language="xx").json()["warnings"]
    assert any("up to 2025" in n for n in notes)
    assert any("'xx'" in n for n in notes)


@pytest.mark.parametrize("changes", [
    {"genres": ["Space Opera"]}, {"genres": []}, {"runtime": 0}, {"actor_ids": [1, 2, 3, 4]},
    {"actor_ids": [5, 5]}, {"budget": -1}, {"release_date": "2025-13-01"}, {"original_language": "English"},
    {"overview": "x" * 1001}, {"unexpected": 1},
])
def test_bad_input_is_rejected(client, changes):
    assert post(client, **changes).status_code == 422


def test_movies_search_filter_and_paginate(client, data):
    m = data[0]["movies"]
    body = client.get("/movies", params={"q": "batman"}).json()
    assert [i["title"] for i in body["items"]] == ["Batman Begins"]
    assert body["items"][0]["genres"] == ["Action"] and body["items"][0]["poster_url"].endswith("/p5000.jpg")

    params = {"genre": "Drama", "year_from": 2000, "year_to": 2009, "page_size": 7}
    first = client.get("/movies", params=params).json()
    drama = set(data[0]["genres"].query("name == 'Drama'")["movie_id"])
    expected = m[m["id"].isin(drama) & m["release_year"].between(2000, 2009)]
    assert first["total"] == len(expected)
    pages = [first, *(client.get("/movies", params={**params, "page": p}).json()
                      for p in range(2, -(-first["total"] // 7) + 1))]
    ids = [i["id"] for p in pages for i in p["items"]]
    assert sorted(ids) == sorted(expected["id"]) and len(first["items"]) == 7


def test_sorting(client):
    years = [i["release_year"] for i in client.get("/movies", params={"sort": "release_date", "order": "asc"})
             .json()["items"]]
    assert years == sorted(years) and years[0] == 1980
    counts = [i["vote_count"] for i in client.get("/movies").json()["items"]]
    assert counts == sorted(counts, reverse=True)  # the default order
    top = client.get("/movies", params={"sort": "vote_average", "page_size": 100}).json()["items"]
    assert "Batman Begins" not in [i["title"] for i in top]  # 3 votes is too few to rank
    assert all(i["vote_count"] >= 50 for i in top)
    assert client.get("/movies", params={"sort": "budget"}).status_code == 422


def test_search_treats_wildcards_and_quotes_literally(client):
    titles = lambda q: [i["title"] for i in client.get("/movies", params={"q": q}).json()["items"]]
    assert titles("%") == ["100% Wolf"]
    assert titles("_") == ["O'Brien_s Day"]
    assert titles("'") == ["O'Brien_s Day"]
    assert titles("x' OR '1'='1") == []


def test_movie_detail(client):
    body = client.get("/movies/5000").json()
    assert body["title"] == "Batman Begins" and body["genres"] == ["Action"]
    assert body["directors"] == [{"id": 900, "name": "Nolan Grey"}]
    assert body["crew"] == [{"id": 903, "name": "Wren Writer", "job": "Screenplay"}]  # directors aren't repeated
    assert body["cast"] == [{"id": 901, "name": "Nolan North", "character": ""}]
    assert body["companies"] == [{"id": 902, "name": "C902"}]
    assert client.get("/movies/123456789").status_code == 404


def test_meta(client, data):
    assert client.get("/meta/genres").json() == data[1]["spec"].genres
    langs = client.get("/meta/languages").json()["languages"]
    assert {"code": "en", "name": "English"} in langs


def test_people_search(client, data):
    directors = client.get("/people", params={"q": "nolan", "role": "director"}).json()
    assert [d["name"] for d in directors] == ["Christopher Nolan", "Nolan Grey"]
    assert directors[0]["n_films"] == (data[0]["crew"]["person_id"] == 1).sum()
    assert directors[1]["known_for"] == "Batman Begins"
    actors = client.get("/people", params={"q": "nolan", "role": "actor"}).json()
    assert [a["name"] for a in actors] == ["Nolan North"]
    assert client.get("/people", params={"q": "n", "role": "actor"}).status_code == 422
    assert client.get("/people", params={"q": "nolan"}).status_code == 422
    assert client.get("/companies", params={"q": "c902"}).json()[0]["id"] == 902
