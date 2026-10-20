# Movie Prediction App

## Problem statement

Movie creators lack a quick, data-backed way to sanity-check a project before pitching it, and audiences lack one easy place to explore movies.

This app lets **creators** enter a movie's parameters (genres, runtime, release month, language, director, lead actors, optional budget) and get estimated **ranges** for budget, revenue and audience score, each with the top factors behind it. The **public** can browse, search and view movie information.

## Stack

- **ML:** Python, pandas, scikit-learn, LightGBM
- **Backend:** FastAPI, Pydantic, SQLite
- **Frontend:** React (Vite)
- **Data:** [TMDB API](https://developer.themoviedb.org/) and the Kaggle TMDB 5000 dataset

## Architecture

_TODO_

## How to run

The full guide comes with the final docs. For now, after building the database (`ml/src/clean_data.py`) and the models (`ml/src/train.py`), start the API:

```bash
.venv/Scripts/python -m pip install -r backend/requirements.txt
.venv/Scripts/python -m uvicorn app.main:app --reload --app-dir backend
```

The interactive API docs are at http://127.0.0.1:8000/docs. The endpoints are listed in [backend/README.md](backend/README.md).

Then start the frontend in a second terminal (Node 22 LTS) and open http://localhost:5173:

```bash
cd frontend
npm install
npm run dev
```

## Evaluation results

Results so far (Day 7). The split is by release year:
- **Train:** 1980–2015.
- **Valid:** 2016–2018, used only for early stopping and to calibrate the ranges.
- **Test:** 2019 onward, never used for training, tuning or calibration.

Revenue and budget are scored on log(2024 dollars). The columns are:
- **MAE:** mean absolute error.
- **MdAPE:** median absolute % error in dollars.
- **coverage:** the share of films whose true value falls inside the predicted 10–90% range. The target is 80%.
- **width:** the range's median width.

Score is TMDB's 0–10 vote average, for films with at least 50 votes. The full tables, including train, are in `ml/reports/models.csv` and `ml/reports/baselines.csv`.

| target | model | valid MAE | valid R² | test MAE | test R² | test MdAPE | test coverage |
|---|---|---|---|---|---|---|---|
| log-revenue | median per genre (baseline) | 1.662 | 0.044 | 1.954 | −0.121 | 138% | 78% |
| log-revenue | line in log-budget (baseline) | 1.232 | 0.430 | 1.372 | 0.397 | 76% | 77% |
| log-revenue | LightGBM, with budget | 1.042 | 0.569 | 1.240 | 0.483 | 77% | 74% |
| log-revenue | LightGBM, no budget | 1.138 | 0.501 | 1.335 | 0.418 | 81% | 77% |
| score | median per genre (baseline) | 0.583 | 0.073 | 0.617 | −0.062 | – | 77% |
| score | LightGBM | 0.491 | 0.339 | 0.526 | 0.204 | – | 78% |
| log-budget | median per genre (baseline) | 1.312 | −0.068 | 1.408 | −0.098 | 90% | 79% |
| log-budget | LightGBM | 0.652 | 0.700 | 0.702 | 0.715 | 49% | 78% |

- **Budget model.** It estimates the budget from everything except the budget itself: genres, runtime, release date, language, the overview topics, and the director, lead actors and lead company's track records. The strongest inputs are the median budgets of the cast's and director's earlier films; the overview topics add about 8% of the gain.
  - The "rough estimate" rule was fixed before training and is decided on valid only: the model needs a valid R² of at least 0.30 and an MAE at least 15% below the genre median. It reaches 0.700 and 50%, so its ranges are shown without the "rough estimate" label.
- **Budget track records.** Since Day 6 every model also sees the median budget of the director's, actors' and company's earlier films. Compared with Day 5, score improved slightly, revenue with budget is about the same, and revenue without budget is slightly worse on test (MAE 1.310 → 1.335).
- **Matching the creator form (Day 7).** Checking `predict()` on real 2024–25 films showed two inputs that the form can't fill the way the data does, so both were changed in training too:
  - **Number of companies:** this feature was dropped. Listed films name a median of 4 companies, but the form asks for one, which pulled every prediction down (budget by about 14%).
  - **Blank overview:** a blank overview now gets the average topics of training films that have one, instead of the topics of an empty text. Only 4% of training films have no overview, and they're mostly small films.
  - The test numbers barely moved: every model's test MAE changed by less than 0.01. On the 2024–25 films, run through `predict()` with their real inputs, revenue range coverage went from 75% to 79%.
- **Calibration.** Each LightGBM range comes from 10% and 90% quantile models. It's then widened by a single margin, chosen so that 80% of valid films fall inside; this method is conformalized quantile regression. Without calibration, test coverage is 71% for revenue, 70% for score and 73% for budget.
- **Range width.** The calibrated ranges are still narrower than the baselines': a median width of 3.44 vs 3.87 on log-revenue, 1.59 vs 1.87 on score, and 1.94 vs 3.91 on log-budget.
- **Test years.** All models except budget do worse on the test years than on valid, and test coverage is a few points below 80%. The per-year breakdown is in `ml/reports/coverage_by_year.csv`.
  - **Revenue:** most of the gap is the 2020–21 cinema closures. Coverage drops to 65–72% in 2020–21, because about 25–30% of those films earned less than their range. Without 2020–21, coverage is 77% with budget and 79% without.
  - **Budget:** coverage is 76–82% in every test year except 2020 (73%).
  - **Score:** the gap is steady drift. In every test year, 14–19% of films score above their range, where a 10–90% range should leave about 10% above. Recent films score higher than the training years predict.
  - The ranges were not re-tuned on the test years, which would make these numbers dishonest. The app's saved models are instead refit on films up to 2023 and calibrated on 2024–25, where their ranges cover 80% of films. Those 2024–25 numbers are calibration, not a test.
- **What drives the predictions.** Each prediction is explained with SHAP values, computed exactly by LightGBM, which are summed into readable factors. Across the test films the largest factors, by share of mean |SHAP|, are listed below. The full table is in `ml/reports/importance.csv`.
  - **Revenue, with budget:** budget 48%, production company 12%, runtime 10%.
  - **Revenue, no budget:** production company 27%, director 18%, runtime 16%.
  - **Score:** runtime 27%, genres 22%, director 18%.
  - **Budget:** lead actors 29%, director 15%, production company 14%.
  - These show what the models lean on, not cause and effect.

## Limitations

_TODO_

## Next steps

_TODO_

---

This product uses the TMDB API but is not endorsed or certified by TMDB.
