# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

The project follows `Movie Prediction App 3-Week Project Plan.pdf` (a 15-working-day plan dated 2026-10-19), one day at a time. Day 1 (setup and data collection), Day 2 (cleaning → SQLite), Day 3 (feature engineering), Day 4 (time split, baselines, metrics), Day 5 (LightGBM revenue and score models), Day 6 (budget model), Day 7 (SHAP explanations, saved models, `predict()`), Day 8 (FastAPI backend), Day 9 (leakage tests) and Day 10 (React frontend with the creator form) are in place. Docker was dropped from Day 9 by the user's decision: the project has no Dockerfile or docker-compose, and Day 14 deploys without containers.

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
.venv/Scripts/python ml/src/baselines.py                 # baselines on the time split -> ml/reports/baselines.csv (committed)
.venv/Scripts/python ml/src/train.py                     # LightGBM models -> ml/reports/models.csv, coverage_by_year.csv, importance.csv, metrics.md (committed) + ml/models/predictor.joblib (git-ignored)
.venv/Scripts/python ml/src/predict.py                   # example prediction from the saved models
.venv/Scripts/python -m pip install -r backend/requirements.txt                              # API deps (includes ml/requirements.txt)
.venv/Scripts/python -m uvicorn app.main:app --reload --app-dir backend                      # API on :8000, docs at /docs
.venv/Scripts/python -m pytest ml/tests backend/tests -q  # tests (one test: add ::test_name to ml/tests/test_clean.py)
cd frontend && npm install                               # frontend deps (Node 22 LTS)
cd frontend && npm run dev                               # app on :5173, needs the API on :8000
cd frontend && npm run lint && npm run build             # ESLint, then a production build to frontend/dist
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
- Every feature must be something the creator form can supply the same way the data does. That's why the number of companies isn't a feature (the form takes one company) and why blank overviews get the average training topics.
- Training-year state, such as the genre list, the top-10 languages and the overview TF-IDF+SVD topics, is fit on years ≤ 2015 only and held in a `FeatureSpec`. That spec must be reused for validation, test and `predict()`.
- `track_record()` builds the director, top-3 actor and lead-company stats from **all** films in the DB released strictly before the target. Revenue and budget stats (`*_med_log_rev`, `*_med_log_budget`) use only prior films with financials, and prior scores use only films with ≥ 50 votes. Unknown entities get `n_prior = 0` and NaN stats. A target doesn't need to exist in the DB, so `predict()` can pass a new film.
- Training rows are films with financials from 1980 on (`MIN_TARGET_YEAR`); older films are only history for track records. `y_score` is NaN for films with fewer than 50 votes, so select rows per target with `target_rows(df, target)`. `split_frames(df)` gives train/valid/test.

## Evaluation (`ml/src/evaluate.py`, `ml/src/baselines.py`)

- Every model is fit on train only and scored with `evaluate.score()`: MAE, RMSE and R² on the log scale (score scale for `y_score`), plus `mdape` (median absolute % error in dollars) for revenue and budget, and `coverage` / `width` for 10–90% ranges (target coverage 80%).
- Baselines to beat: global median, median per genre (mean of a film's genre medians), and for revenue a straight line in log-budget. Their ranges come from train residual quantiles. Results are in `ml/reports/baselines.csv`.

## Models (`ml/src/train.py`)

- `RangeModel` is three LightGBM regressors on the same features:
  - an L2 model for the point prediction
  - q10 and q90 quantile models for the range
  - Crossed quantiles are sorted, and the point is clipped into the range.
  - `fit(..., calibrate=True)` widens the range by `margin`, the conformal (CQR) margin that makes 80% of valid films fall inside.
- `MODELS` maps (target, name) to the feature columns that model leaves out:
  - revenue `lgbm` uses `log_budget_adj`
  - revenue `lgbm_no_budget` leaves the budget out, because the creator's budget is optional
  - the score `lgbm` never uses the budget
  - the budget `lgbm` (`y_log_budget`) leaves out `log_budget_adj`, which is its own target
- `budget_quality(table)` applies the "rough estimate" rule, decided on valid only: the budget model is rough if valid R² < `ROUGH_MIN_R2` (0.30) or its MAE is less than `ROUGH_MIN_GAIN` (15%) below the genre median. The verdict goes into `metrics.md`; Day 7's `predict()` uses it for the label. The range itself is not widened further.
- `PARAMS` are fixed. Valid (2016–18) is used only for early stopping, which picks the number of trees, and for the range calibration. Test (2019+) is never used in training, tuning or calibration.
- `main()` writes `ml/reports/models.csv`, with the baselines next to the models, and `ml/reports/metrics.md` (valid and test, in markdown). The README's results table is copied from it by hand.
- `main()` also writes `ml/reports/coverage_by_year.csv`, test coverage per release year with the share of films below and above the range. `metrics.md` adds coverage without 2020–21 (`SHOCK_YEARS`, the cinema closures). These years are only reported, never used to tune anything.
- `RangeModel.contributions(df)` gives the point model's SHAP values (LightGBM `pred_contrib`, no `shap` package), summed into readable factors by `FACTOR_GROUPS` / `factor_of`, plus a `base` column. Per row they add up to the raw point prediction. `main()` writes `ml/reports/importance.csv`, the mean |SHAP| share per factor on test.
- The app's models come from `fit_live(history)`: same `PARAMS` and `MODELS`, with a `FeatureSpec` fit on ≤ 2023 (`fit_spec(..., train_end=)`), train ≤ 2023, early stopping and calibration on 2024–25. `main()` saves the bundle with joblib to `ml/models/predictor.joblib`: `spec`, `models`, `budget_rough` (from the evaluation split's `budget_quality`), `valid_years`, `valid_coverage` and `trained_on`. The README's numbers stay on the evaluation split. `train.py`'s `__main__` runs `train.main()` so the models pickle as `train.RangeModel`.

## Prediction (`ml/src/predict.py`)

- `Predictor(bundle=MODEL_PATH, history=None)` loads the bundle and `load_history()` once. `predict(FilmInput, top_k=5)` returns `budget`, `revenue` and `score`, each with `low`, `point`, `high` (dollars, or 0–10 for score), `model`, `label` and `drivers` (`factor`, `effect`, `text`), plus `prior_films`.
- `FilmInput` identifies the director, top-3 actors and lead company by TMDB id. The budget (today's dollars) and the overview are optional. A blank overview gets `OverviewEncoder.blank`, the average topics and word count of training films that have one; this applies in training too. Unknown genres or a non-positive budget/runtime raise `ValueError`.
- The film is added to a copy of the history as id −1 without outcomes and goes through `build_features`, so its track records only use earlier films, and unknown ids get `n_prior = 0`.
- Revenue uses `lgbm` when a budget is given and `lgbm_no_budget` otherwise. The budget's `label` is "rough estimate" when `budget_rough` is set, and a given budget is echoed as `given`.
- A call takes about 0.75 s on the full DB, mostly the history copy and track records.

## Backend (`backend/app/`)

- `app/__init__.py` puts `ml/src` on `sys.path`, so the API imports `predict`/`features` directly and the bundle unpickles as `train.RangeModel`. No ML code is copied.
- `create_app(settings)` loads the `Catalog` and one `Predictor` (bundle + history) at startup. `config.py` reads `MOVIES_DB`, `MODEL_PATH` and `CORS_ORIGINS` (comma-separated, default `http://localhost:5173`).
- `/predict` is a sync endpoint, so it runs in the thread pool. Genres are checked against the bundle's `spec.genres`, and `ValueError`s become 422s. The response adds `warnings` (`warnings_for`): no known director, actors or company, some of them unknown, a year after the last valid year, or a language outside `spec.languages`.
- `catalog.py` holds the browse and name-search SQL: read-only connections, parameterized queries, `LIKE` with `%`/`_` escaped. `GET /movies` sorts by `vote_count` by default, because Kaggle-only rows carry 2017 popularity values and no posters. Sorting by `vote_average` needs ≥ 50 votes. The name search (`/people?role=director|actor`, `/companies`) ranks by number of films and uses `NOT INDEXED`, which is about 25× faster than walking the id index.
- Endpoints: `GET /health`, `POST /predict`, `GET /movies`, `GET /movies/{id}`, `GET /meta/genres`, `GET /meta/languages`, `GET /people`, `GET /companies`.
- `backend/tests/conftest.py` builds a synthetic DB (with `clean_data.write_sqlite`) and a FAST `fit_live` bundle, reusing `ml/tests` helpers, so the API tests need neither the real data nor the trained models.

## Leakage tests

- `ml/tests/test_leakage.py` checks the no-leakage rule over the whole feature table, not one film:
  - Scrambling the outcomes of every film on or after a cutoff leaves all `feature_columns` of earlier rows unchanged, and so does dropping those films.
  - A row's own revenue and votes reach only its `y_*` targets, and its budget only `log_budget_adj` and `y_log_budget`.
  - `fit_spec`'s languages and overview encoder ignore films after `train_end`. The genre list is TMDB's fixed vocabulary from all years, which carries no outcomes.
- `backend/tests/test_api.py::test_predict_sees_only_earlier_films` checks the same rule through `POST /predict`.
- Breaking the date rule in `track_record` (`<` → `<=`, or no date filter at all) makes these tests fail. Keep it that way when changing features.

## Frontend (`frontend/`)

- React 19 + React Router 7 on Vite 6, in plain JavaScript and plain CSS. Routes are in `src/main.jsx`: `/` is the creator form (`PredictPage`), `/movies` is a placeholder for Day 12, and anything else is not found.
- `src/api.js` calls `import.meta.env.VITE_API_URL`, or `/api` by default. In development, `vite.config.js` proxies `/api/*` to `http://127.0.0.1:8000` and strips the prefix, so CORS isn't involved. A 422's `detail` becomes `ApiError.fields`, which the form shows next to the matching inputs. A 5xx without a JSON body, which is what the proxy returns when the API is down, reads "Is the API running?".
- `Autocomplete` searches `/people` or `/companies` from 2 characters, 250 ms after typing stops, and aborts the previous request. The search functions are defined at module level so the effect doesn't restart on every render. Only names picked from the list are sent as ids. A typed name that wasn't picked is dropped, and the API's warnings explain newcomers.
- The form sends `release_date` as `YYYY-MM-01`, since only the month and year reach the model, and the budget in millions × 1e6. Its client checks mirror `PredictRequest`. `RawResults` is the plain Day 10 view of the response; Day 11 replaces it with the results page.
## What the app does

A full-stack app with two audiences:
- **Creators** enter a movie's parameters (genres, runtime, release month/year, language, production company, director, top 3 billed actors, optional budget) and get **ranges** for budget, revenue and score, each with the top factors behind it.
- **The public** browses, searches and filters movies and views detail pages.

Stack: Python, scikit-learn/LightGBM, FastAPI, Pydantic, SQLite, React (Vite) with a router. Data comes from the TMDB API, with the Kaggle TMDB 5000 dataset as a fallback for older films.

## Planned layout

```
/ml        data collection (TMDB API pull), cleaning, feature engineering, training, evaluation
/backend   FastAPI service that loads the joblib models and serves the API
/frontend  React (Vite) app: creator form, results page, public browse/detail pages
```

Data flow: TMDB/Kaggle → cleaning → SQLite → feature pipeline → LightGBM models saved with joblib → a single `predict()` function → FastAPI `POST /predict` → React results page.

API (Day 8): `POST /predict`, `GET /movies` (search, filter, paginate), `GET /movies/{id}`, `GET /meta/genres`, plus `/meta/languages`, `/people`, `/companies` (the form's autocomplete) and `/health`.

## ML rules that must hold

- **No target leakage.** Never use `popularity`, `vote_count` or `vote_average` as features; they exist only after release. Director and actor track-record features must be computed **only from films released before the target film**. `ml/tests/test_leakage.py` and the feature/API tests enforce this rule.
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
