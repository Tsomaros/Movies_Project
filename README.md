# Movie Prediction App

## Problem statement

Movie creators lack a quick, data-backed way to sanity-check a project before pitching it, and audiences lack one easy place to explore movies.

This app lets **creators** enter a movie's parameters (genres, runtime, release month, language, director, lead actors, optional budget) and get estimated **ranges** for budget, revenue and audience score, each with the top factors behind it. The **public** can browse, search and view movie information.

## Stack

- **ML:** Python, pandas, scikit-learn, LightGBM
- **Backend:** FastAPI, Pydantic, SQLite
- **Frontend:** React (Vite)
- **Ops:** Docker, docker-compose
- **Data:** [TMDB API](https://developer.themoviedb.org/) and the Kaggle TMDB 5000 dataset

## Architecture

_TODO_

## How to run

_TODO_

## Evaluation results

Results so far (Day 6). The split is by release year:
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
| log-revenue | LightGBM, with budget | 1.033 | 0.576 | 1.247 | 0.481 | 78% | 74% |
| log-revenue | LightGBM, no budget | 1.142 | 0.500 | 1.326 | 0.426 | 81% | 76% |
| score | median per genre (baseline) | 0.583 | 0.073 | 0.617 | −0.062 | – | 77% |
| score | LightGBM | 0.489 | 0.348 | 0.525 | 0.210 | – | 78% |
| log-budget | median per genre (baseline) | 1.312 | −0.068 | 1.408 | −0.098 | 90% | 79% |
| log-budget | LightGBM | 0.650 | 0.706 | 0.693 | 0.725 | 48% | 79% |

- **Budget model.** It estimates the budget from everything except the budget itself: genres, runtime, release date, language, the overview topics, and the director, lead actors and lead company's track records. The strongest inputs are the median budgets of the cast's and director's earlier films; the overview topics add about 9% of the gain.
  - The "rough estimate" rule was fixed before training and is decided on valid only: the model needs a valid R² of at least 0.30 and an MAE at least 15% below the genre median. It reaches 0.706 and 50%, so its ranges are shown without the "rough estimate" label.
- **Budget track records.** Since Day 6 every model also sees the median budget of the director's, actors' and company's earlier films. Compared with Day 5, score improved slightly, revenue with budget is about the same, and revenue without budget is slightly worse on both valid and test (MAE 1.131 → 1.142 and 1.310 → 1.326).
- **Calibration.** Each LightGBM range comes from 10% and 90% quantile models. It's then widened by a single margin, chosen so that 80% of valid films fall inside; this method is conformalized quantile regression. Without calibration, test coverage is 72% for revenue, 70% for score and 73% for budget.
- **Range width.** The calibrated ranges are still narrower than the baselines': a median width of 3.43 vs 3.87 on log-revenue, 1.59 vs 1.87 on score, and 1.99 vs 3.91 on log-budget.
- **Test years.** All models except budget do worse on the test years than on valid, and test coverage is a few points below 80%. The per-year breakdown is in `ml/reports/coverage_by_year.csv`.
  - **Revenue:** most of the gap is the 2020–21 cinema closures. Coverage drops to 65% (2020) and 68% (2021), because about 30% of those films earned less than their range. Without 2020–21, coverage is 77% with and without budget.
  - **Budget:** coverage is 77–83% in every test year except 2020 (74%).
  - **Score:** the gap is steady drift. In every test year, 14–18% of films score above their range, where a 10–90% range should leave about 10% above. Recent films score higher than the training years predict.
  - The ranges were not re-tuned on the test years, which would make these numbers dishonest. The app's saved models (Day 7) will instead be refit and calibrated on the most recent years.

## Limitations

_TODO_

## Next steps

_TODO_

---

This product uses the TMDB API but is not endorsed or certified by TMDB.
