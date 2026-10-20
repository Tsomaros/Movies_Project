**y_log_revenue**

| model | split | n | mae | r2 | mdape | coverage | width |
|---|---|---|---|---|---|---|---|
| global_median | valid | 1005 | 1.739 | -0.019 | 94% | 84% | 5.81 |
| global_median | test | 1683 | 1.986 | -0.137 | 105% | 78% | 5.81 |
| genre_median | valid | 1005 | 1.662 | 0.044 | 92% | 84% | 5.56 |
| genre_median | test | 1683 | 1.954 | -0.121 | 138% | 78% | 5.56 |
| budget_linear | valid | 1005 | 1.232 | 0.430 | 70% | 79% | 3.87 |
| budget_linear | test | 1683 | 1.372 | 0.397 | 76% | 77% | 3.87 |
| lgbm | valid | 1005 | 1.033 | 0.576 | 65% | 80% | 3.28 |
| lgbm | test | 1683 | 1.247 | 0.481 | 78% | 74% | 3.43 |
| lgbm_no_budget | valid | 1005 | 1.142 | 0.500 | 70% | 80% | 3.43 |
| lgbm_no_budget | test | 1683 | 1.326 | 0.426 | 81% | 76% | 3.60 |

**y_score**

| model | split | n | mae | r2 | coverage | width |
|---|---|---|---|---|---|---|
| global_median | valid | 875 | 0.622 | -0.037 | 80% | 2.00 |
| global_median | test | 1384 | 0.643 | -0.161 | 80% | 2.00 |
| genre_median | valid | 875 | 0.583 | 0.073 | 79% | 1.87 |
| genre_median | test | 1384 | 0.617 | -0.062 | 77% | 1.87 |
| lgbm | valid | 875 | 0.489 | 0.348 | 80% | 1.59 |
| lgbm | test | 1384 | 0.525 | 0.210 | 78% | 1.59 |

**y_log_budget**

| model | split | n | mae | r2 | mdape | coverage | width |
|---|---|---|---|---|---|---|---|
| global_median | valid | 1005 | 1.372 | -0.128 | 88% | 78% | 4.13 |
| global_median | test | 1683 | 1.453 | -0.151 | 89% | 79% | 4.13 |
| genre_median | valid | 1005 | 1.312 | -0.068 | 82% | 79% | 3.91 |
| genre_median | test | 1683 | 1.408 | -0.098 | 90% | 79% | 3.91 |
| lgbm | valid | 1005 | 0.650 | 0.706 | 46% | 80% | 1.89 |
| lgbm | test | 1683 | 0.693 | 0.725 | 48% | 79% | 1.99 |

**Test range coverage without 2020, 2021**

- y_log_revenue / lgbm: 77%
- y_log_revenue / lgbm_no_budget: 77%
- y_score / lgbm: 77%
- y_log_budget / lgbm: 79%

**Budget model: reliable**

Valid R² 0.706 (needs 0.30), MAE 50% below the genre median (needs 15%).
