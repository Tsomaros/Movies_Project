# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The project follows `Movie Prediction App 3-Week Project Plan.pdf` (a 15-working-day plan dated 2026-10-19), one day at a time. Day 1 (setup and data collection) is in place. `backend/` and `frontend/` are placeholders until Days 8 and 10.

## Commands (Windows, run from the repo root)

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r ml/requirements.txt
.venv/Scripts/python ml/src/collect_tmdb.py --limit 20   # smoke test
.venv/Scripts/python ml/src/collect_tmdb.py              # full pull (~10k movies, resumable)
.venv/Scripts/python ml/src/check_kaggle.py              # download (if missing) + summarize the Kaggle fallback CSVs
```

Credentials go in `.env` at the repo root (see `.env.example`): `TMDB_READ_TOKEN` or `TMDB_API_KEY` for TMDB. `KAGGLE_USERNAME` + `KAGGLE_KEY` are optional; the public Kaggle dataset downloads anonymously.

## Data files (git-ignored, under `ml/data/raw/`)

- `movie_ids.json`: IDs found by discovery (most-voted per release year, 1980–2025). It's reused on later runs unless you pass `--refresh-ids`.
- `tmdb_movies.jsonl`: one raw `/movie/{id}?append_to_response=credits,release_dates` response per line. Credits are trimmed to the top 10 cast and key crew. Resuming works by skipping IDs already in this file.
- `missing_ids.json`: IDs that returned 404.
- `kaggle/tmdb_5000_movies.csv`, `kaggle/tmdb_5000_credits.csv`: downloaded by `check_kaggle.py` via kagglehub, or placed there by hand.

## What the app does

A full-stack app with two audiences:
- **Creators** enter a movie's parameters (genres, runtime, release month/year, language, production company, director, top 3 billed actors, optional budget) and get **ranges** for budget, revenue and score, each with the top factors behind it.
- **The public** browses, searches and filters movies and views detail pages.

Stack: Python, scikit-learn/LightGBM, FastAPI, Pydantic, SQLite, React (Vite) with a router, Docker / docker-compose. Data comes from the TMDB API, with the Kaggle TMDB 5000 dataset as a fallback for older films.

## Planned layout

```
/ml        data collection (TMDB API pull), cleaning, feature engineering, training, evaluation
/backend   FastAPI service that loads the joblib models and serves the API
/frontend  React (Vite) app: creator form, results page, public browse/detail pages
```

Data flow: TMDB/Kaggle → cleaning → SQLite → feature pipeline → LightGBM models saved with joblib → a single `predict()` function → FastAPI `POST /predict` → React results page.

Planned API: `POST /predict`, `GET /movies` (search, filter, paginate), `GET /movies/{id}`, `GET /meta/genres`.

## ML rules that must hold

- **No target leakage.** Never use `popularity`, `vote_count` or `vote_average` as features; they exist only after release. Director and actor track-record features must be computed **only from films released before the target film**. The plan calls for tests that enforce this rule.
- **Time-based split, not random:** train ≤ 2015, validate 2016–2018, test 2019 onward.
- Always compare against simple baselines (for example, the median per genre) and report the metrics honestly in the README.
- Revenue is modelled as log-revenue. Ranges come from LightGBM quantile models.
- Cleaning: drop rows with zero or missing budget or revenue, remove duplicates, and either adjust for inflation or include release year as a feature.
- `predict()` returns the budget, revenue and score ranges plus the top drivers (SHAP or feature importance). If the budget model performs poorly, return a wide range labelled "rough estimate".
- The API must handle unknown actors and directors gracefully.

## Scope constraints

- Out of scope: accounts, reviews, trailer analysis, chatbots.
- If behind schedule, cut in this order: the compare ("try changing one thing") feature, then the public filters, then the budget model, then the SHAP charts (keep a text version).
- Time-box data cleaning to 2 days.
- The UI must show the TMDB attribution notice and logo, plus a clear note on the model's limitations.

## Definition of done

1. A creator gets ranges for cost, revenue and score, with explanations.
2. A visitor can search and view movies.
3. The model is evaluated on the time-based split against a baseline, and the README reports the numbers.
4. The app is deployed (Render, Fly.io or Railway free tier) with the TMDB attribution and limitations shown.
