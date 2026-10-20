# 5-minute demo script

**Before the demo.** Start the API and the frontend (see [How to run](../README.md#how-to-run)) and open http://localhost:5173 in a wide browser window. Open `ml/reports/metrics.md` or the README's evaluation table in a second tab. Do the example once beforehand, so the name searches are warm.

The numbers below come from the saved models; the app shows them rounded the same way.

## 0:00–0:30 · The problem

> "Before a film is pitched, a creator has no quick, data-backed way to check whether its budget, box office and reception look plausible. This app gives a range for each, not a single guess, with the reasons behind it. It's trained on about 10,500 films from TMDB, and it also lets anyone browse the films."

## 0:30–1:45 · The creator form

Fill in the form while talking:

| field | value |
|---|---|
| Genres | Science Fiction, Drama |
| Runtime | 150 minutes |
| Release | December 2027 |
| Language | English |
| Director | Denis Villeneuve |
| Lead actors | Timothée Chalamet, Zendaya, Florence Pugh |
| Company | Legendary Pictures |
| Budget | 150 (millions of US dollars) |

> "The names come from an autocomplete over the database, so each one carries its real track record: only films released *before* this one, which is the rule that keeps the model honest. The budget is optional; without it, a second revenue model is used."

Click **Predict**.

## 1:45–2:45 · The results

Point at the note at the top first: "The models have seen films released up to 2025; a later year is treated like 2025." That's the API being upfront about a future release year. Then point at each card:

- **Budget:** "Your budget: $150M. Films like this usually cost $42.3M – $219M, most likely $110M." So $150M is inside the usual range.
- **Worldwide box office:** $67.9M – $660M, most likely $153M, about 1.0× the budget. "The range is wide on purpose: about 8 in 10 comparable films fell inside ranges like this."
- **Audience score:** 6.3 – 8.0 out of 10, most likely 7.4.
- **Factors:** "Each card lists the top 5 factors in plain English, from SHAP values. For box office the budget pushes up the most, then the 150-minute runtime. The genres pull it down a little."
- Scroll past the "Earlier films in our data" line to the **limitations** note at the end.

## 2:45–3:30 · Try changing one thing

In **Try changing one thing**, pick *Release month* → *July* → **Compare**.

> "Only the release month changes. Box office moves about +5% and the score doesn't change. The table also says the new box office is still inside the first range, so this isn't a big effect."

Then show the newcomer case. In the form, remove Denis Villeneuve, type "Jane Doe" in the director box, pick **Add “Jane Doe” as a newcomer**, and click **Predict** again.

> "People not in the data are handled as newcomers with no earlier films. The typical budget falls to $13.3M – $148M, most likely $49.7M, so the app now notes that $150M is above the usual range. The API also warns that the director has no earlier films."

## 3:30–4:15 · The public side

Open **Movies** in the header. Search a title, filter by a genre and years, and open a film.

> "Visitors can search and filter about 36,000 films. The filters stay in the URL, so back and reload keep them. The detail page has the poster, facts, key crew and the top 10 cast. Every page shows the TMDB attribution and logo."

## 4:15–5:00 · Is it any good?

Switch to the evaluation table.

> "The split is by year: train up to 2015, validate 2016–18, test 2019 on, so the test is the future. On test, revenue with a budget has an R² of 0.48 against 0.40 for a straight line in log-budget, and the genre-median baseline is below zero. The budget model reaches 0.72. Score is the hardest at 0.20: reception depends on things the form can't capture. The ranges were calibrated on the validation years only. They cover 74–78% of test films, a little under the 80% target, mostly because of the 2020–21 cinema closures and recent films scoring higher. I report that rather than tuning on test."

Close with:

> "Next would be deployment, which I skipped, a scheduled data refresh, and faster track records."

## Likely questions

- **How do you avoid leakage?** No `popularity`, `vote_count` or `vote_average` as features. Track records only use films released strictly before the target. `ml/tests/test_leakage.py` scrambles every film after a cutoff and checks that no earlier row's features change.
- **Why ranges?** One number would suggest a precision the data doesn't have. LightGBM fits 10% and 90% quantile models, and a conformal margin (CQR) widens them so 80% of validation films fall inside.
- **Why LightGBM?** Tabular data with missing values and a few thousand rows. It's fast, handles NaN track records for newcomers, gives exact SHAP values and has a quantile loss.
- **Why log revenue?** Box office spans from thousands to billions of dollars. On the log scale, errors are relative, and the model isn't dominated by a few blockbusters.
- **Inflation?** Budgets and revenue are converted to 2024 dollars with US CPI.
- **What would change the results most?** Better data on small and non-US films, and post-release signals, which are deliberately left out.
