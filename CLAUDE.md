# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The project follows `Movie Prediction App 3-Week Project Plan.pdf` (a 15-working-day plan dated 2026-10-19), one day at a time. Day 1 (setup and data collection), Day 2 (cleaning → SQLite) and Day 3 (feature engineering) are in place. `backend/` and `frontend/` are placeholders until Days 8 and 10.

## Commands (Windows, run from the repo root)

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r ml/requirements.txt
.venv/Scripts/python ml/src/collect_tmdb.py --limit 20   # smoke test
.venv/Scripts/python ml/src/collect_tmdb.py              # full pull (~10k movies, resumable)
.venv/Scripts/python ml/src/collect_tmdb.py --add-ids --sort revenue.desc --per-year 500     # append more IDs, fetch only new ones
.venv/Scripts/python ml/src/collect_tmdb.py --add-ids --sort vote_count.desc --per-year 500
.venv/Scripts/python ml/src/check_kaggle.py              # download (if missing) + summarize the Kaggle fallback CSVs
.venv/Scripts/python ml/src/clean_data.py                # clean both sources -> ml/data/movies.db (rebuilt each run)
.venv/Scripts/python ml/src/features.py                  # features for trainable rows -> ml/data/features.csv (git-ignored)
.venv/Scripts/python -m pytest ml/tests -q               # tests (one test: add ::test_name to ml/tests/test_clean.py)
```

Credentials go in `.env` at the repo root (see `.env.example`): `TMDB_READ_TOKEN` or `TMDB_API_KEY` for TMDB. `KAGGLE_USERNAME` + `KAGGLE_KEY` are optional; the public Kaggle dataset downloads anonymously.

## Data files (git-ignored, under `ml/data/raw/`)

- `movie_ids.json`: IDs found by discovery, for release years 1980–2025: the top 220 per year by votes, then the top 500 per year by revenue, then the top 500 by votes, appended with `--add-ids`. Later runs reuse it as is. `--add-ids` appends new IDs, and `--refresh-ids` overwrites the file.
- `tmdb_movies.jsonl`: one raw `/movie/{id}?append_to_response=credits,release_dates` response per line. Credits are trimmed to the top 10 cast and key crew. Resuming works by skipping IDs already in this file.
- `missing_ids.json`: IDs that returned 404.
- `kaggle/tmdb_5000_movies.csv`, `kaggle/tmdb_5000_credits.csv`: downloaded by `check_kaggle.py` via kagglehub, or placed there by hand.

## Clean data (`ml/data/`)

- `movies.db` (git-ignored), built by `clean_data.py`. Tables: `movies`, `genres`, `companies`, `cast_members` (top 10), `crew` (key jobs). The TMDB API wins over Kaggle; Kaggle only adds ids missing from the pull. Every movie is kept for browsing, and **training must filter on `has_financials = 1`**: budget and revenue both ≥ $5,000, with smaller values stored as NULL. Revenue is also set to NULL when it equals the budget, which is likely a placeholder. `budget_adj`/`revenue_adj` are in 2024 dollars.
- `cpi_us.csv` (committed): annual US CPI-U, 1916–2024, used for the inflation adjustment. Years after 2024 use a factor of 1.0.

## Features (`ml/src/features.py`)

- `build_features(load_history())` returns one row per trainable movie: `META` columns, the features, and the `TARGETS` (`y_log_revenue`, `y_score`, `y_log_budget`). `feature_columns(df)` gives the model inputs and fails if any leaky column is present.
- Training-year state, such as the genre list, the top-10 languages and the overview TF-IDF+SVD topics, is fit on years ≤ 2015 only and held in a `FeatureSpec`. That spec must be reused for validation, test and `predict()`.
- `track_record()` builds the director, top-3 actor and lead-company stats from **all** films in the DB released strictly before the target. Revenue stats use only prior films with financials, and prior scores use only films with ≥ 50 votes. Unknown entities get `n_prior = 0` and NaN stats. A target doesn't need to exist in the DB, so `predict()` can pass a new film.

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
