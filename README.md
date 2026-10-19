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

Results so far (Day 5). The split is by release year:
- **Train:** 1980–2015.
- **Valid:** 2016–2018, used only for early stopping and to calibrate the ranges.
- **Test:** 2019 onward, never used for training, tuning or calibration.

Revenue is scored on log(2024 dollars). The columns are:
- **MAE:** mean absolute error.
- **MdAPE:** median absolute % error in dollars.
- **coverage:** the share of films whose true value falls inside the predicted 10–90% range. The target is 80%.
- **width:** the range's median width.

Score is TMDB's 0–10 vote average, for films with at least 50 votes. The full tables, including train, are in `ml/reports/models.csv` and `ml/reports/baselines.csv`.

| target | model | valid MAE | valid R² | test MAE | test R² | test MdAPE | test coverage |
|---|---|---|---|---|---|---|---|
| log-revenue | median per genre (baseline) | 1.662 | 0.044 | 1.954 | −0.121 | 138% | 78% |
| log-revenue | line in log-budget (baseline) | 1.232 | 0.430 | 1.372 | 0.397 | 76% | 77% |
| log-revenue | LightGBM, with budget | 1.040 | 0.571 | 1.241 | 0.484 | 77% | 75% |
| log-revenue | LightGBM, no budget | 1.131 | 0.510 | 1.310 | 0.437 | 79% | 75% |
| score | median per genre (baseline) | 0.583 | 0.073 | 0.617 | −0.062 | – | 77% |
| score | LightGBM | 0.493 | 0.343 | 0.529 | 0.199 | – | 77% |

- **Calibration.** Each LightGBM range comes from 10% and 90% quantile models. It's then widened by a single margin, chosen so that 80% of valid films fall inside; this method is conformalized quantile regression. Without calibration, test coverage was 72% for revenue and 71% for score.
- **Range width.** The calibrated ranges are still narrower than the baselines': a median width of 3.43 vs 3.87 on log-revenue, and 1.54 vs 1.87 on score.
- **Test years.** All models do worse on the test years than on valid, likely in part because of the 2020–21 cinema closures. As a result, test coverage is still a few points below 80%.

## Limitations

_TODO_

## Next steps

_TODO_

---

This product uses the TMDB API but is not endorsed or certified by TMDB.
