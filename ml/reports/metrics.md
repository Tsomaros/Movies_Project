**y_log_revenue**

| model | split | n | mae | r2 | mdape | coverage | width |
|---|---|---|---|---|---|---|---|
| global_median | valid | 1005 | 1.739 | -0.019 | 94% | 84% | 5.81 |
| global_median | test | 1683 | 1.986 | -0.137 | 105% | 78% | 5.81 |
| genre_median | valid | 1005 | 1.662 | 0.044 | 92% | 84% | 5.56 |
| genre_median | test | 1683 | 1.954 | -0.121 | 138% | 78% | 5.56 |
| budget_linear | valid | 1005 | 1.232 | 0.430 | 70% | 79% | 3.87 |
| budget_linear | test | 1683 | 1.372 | 0.397 | 76% | 77% | 3.87 |
| lgbm | valid | 1005 | 1.040 | 0.571 | 65% | 80% | 3.26 |
| lgbm | test | 1683 | 1.241 | 0.484 | 77% | 75% | 3.43 |
| lgbm_no_budget | valid | 1005 | 1.131 | 0.510 | 69% | 80% | 3.46 |
| lgbm_no_budget | test | 1683 | 1.310 | 0.437 | 79% | 75% | 3.58 |

**y_score**

| model | split | n | mae | r2 | coverage | width |
|---|---|---|---|---|---|---|
| global_median | valid | 875 | 0.622 | -0.037 | 80% | 2.00 |
| global_median | test | 1384 | 0.643 | -0.161 | 80% | 2.00 |
| genre_median | valid | 875 | 0.583 | 0.073 | 79% | 1.87 |
| genre_median | test | 1384 | 0.617 | -0.062 | 77% | 1.87 |
| lgbm | valid | 875 | 0.493 | 0.343 | 80% | 1.55 |
| lgbm | test | 1384 | 0.529 | 0.199 | 77% | 1.54 |
