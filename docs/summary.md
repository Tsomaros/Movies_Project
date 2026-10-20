# Movie Prediction App: one-page summary

**Problem.** Before a film is pitched, a creator has no quick, data-backed way to check whether its budget, box office and reception look plausible. Audiences, meanwhile, lack one easy place to explore movies.

**What it does.**
- **Creators** enter a film's genres, runtime, release month and year, language, production company, director, top 3 actors, and optionally a budget and overview.
- They get a **10–90% range** for budget, worldwide box office and audience score, each with its top 5 factors in plain English.
- They can change one input and compare the two predictions. Unknown people can be added as newcomers.
- **Visitors** search and filter about 36,000 films and open detail pages with poster, facts, crew and cast.

**How it's built.**
- **Data:** TMDB API pull, with the Kaggle TMDB 5000 dataset for older films. It's cleaned into SQLite, with amounts in 2024 dollars.
- **Features:** genres, runtime, release timing, language, overview topics (TF-IDF + SVD), and the director's, actors' and company's track records, built only from films released before the target.
- **Models:** LightGBM, with an L2 model for the point and q10/q90 models for the range, calibrated with conformalized quantile regression. There are separate models for revenue with and without a budget, for score and for budget. The factors are exact SHAP values grouped into readable factors.
- **App:** FastAPI + Pydantic serving a single `predict()` and the browse queries. React (Vite) frontend. 86 tests, including leakage tests on the whole feature table and through the API.

**Results** (time split: train ≤ 2015, validate 2016–18, test 2019+; test numbers):

| target | model | MAE | R² | 10–90% range coverage |
|---|---|---|---|---|
| log-revenue | median per genre (baseline) | 1.954 | −0.12 | 78% |
| log-revenue | line in log-budget (baseline) | 1.372 | 0.40 | 77% |
| log-revenue | **LightGBM, with budget** | **1.240** | **0.48** | 74% (77% without 2020–21) |
| log-revenue | LightGBM, no budget | 1.335 | 0.42 | 77% |
| score (0–10) | median per genre (baseline) | 0.617 | −0.06 | 77% |
| score (0–10) | **LightGBM** | **0.526** | **0.20** | 78% |
| log-budget | median per genre (baseline) | 1.408 | −0.10 | 79% |
| log-budget | **LightGBM** | **0.703** | **0.72** | 78% |

Every model beats its baselines on MAE and R², with narrower ranges. The revenue model's ranges cover a few points fewer test films than the baselines' wider ones. Test was never used for training, tuning or calibration.

**Honest limits.**
- Coverage on test is a little under 80%, because of the 2020–21 cinema closures and recent films scoring higher than expected.
- Score is hard to predict from pre-release inputs (R² 0.20).
- The data favours larger, US releases, and revenue is box office only.
- The factors show associations, not causes.

**Decisions along the way.**
- Docker was dropped and deployment (Day 14) was skipped, so the app runs locally.
- Two features that the form can't fill the way the data does were changed in training too: the number of companies (dropped) and blank overviews (given the average topics).

**Next steps.** Deploy on a free tier, schedule data refreshes and retraining, precompute track records for faster predictions, and let the compare panel swap people and companies.

*This product uses the TMDB API but is not endorsed or certified by TMDB.*
