# Backend

FastAPI service that serves the saved models and the movie database. Run it from the repo root after `ml/src/clean_data.py` and `ml/src/train.py` have built `ml/data/movies.db` and `ml/models/predictor.joblib`:

```bash
.venv/Scripts/python -m pip install -r backend/requirements.txt
.venv/Scripts/python -m uvicorn app.main:app --reload --app-dir backend
```

Interactive docs: http://127.0.0.1:8000/docs

| endpoint | what it does |
|---|---|
| `POST /predict` | budget, revenue and score ranges with their top factors and warnings |
| `GET /movies` | search (`q`), filter (`genre`, `year_from`, `year_to`, `language`), sort and paginate |
| `GET /movies/{id}` | one movie with genres, companies, directors and cast |
| `GET /meta/genres`, `GET /meta/languages` | the values the creator form can send |
| `GET /people?q=&role=director\|actor`, `GET /companies?q=` | name search for the form; ids go into `/predict` |
| `GET /health` | model training date and calibration years |

Settings (environment variables): `MOVIES_DB`, `MODEL_PATH`, `CORS_ORIGINS`.
