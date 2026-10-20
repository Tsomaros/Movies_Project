"""FastAPI service: predictions for creators, movie browsing for the public.

The models (ml/models/predictor.joblib) and the film history are loaded once at startup.

Usage (from the repo root):
    .venv/Scripts/python -m uvicorn app.main:app --reload --app-dir backend
    then open http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from features import load_history
from predict import FilmInput, Predictor

from .catalog import Catalog
from .config import Settings, get_settings
from .schemas import Health, Language, Languages, MovieDetail, MoviePage, NameMatch, PredictRequest, PredictResponse

LANGUAGE_NAMES = {"en": "English", "fr": "French", "ja": "Japanese", "es": "Spanish", "it": "Italian",
                  "zh": "Mandarin", "cn": "Cantonese", "de": "German", "ru": "Russian", "hi": "Hindi",
                  "ko": "Korean", "pt": "Portuguese", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
                  "sv": "Swedish", "da": "Danish", "tr": "Turkish", "ar": "Arabic"}

router = APIRouter()


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def get_catalog(request: Request) -> Catalog:
    return request.app.state.catalog


def join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def warnings_for(body: PredictRequest, out: dict, predictor: Predictor) -> list[str]:
    """Plain-language notes on inputs the models can say little about."""
    prior = out["prior_films"]
    found = {"director": prior["director"] > 0, "lead actors": prior["actors_max"] > 0,
             "production company": prior["company"] > 0}
    given = {"director": body.director_id is not None, "lead actors": bool(body.actor_ids),
             "production company": body.company_id is not None}
    notes = []
    if not any(found.values()):
        notes.append("None of the director, lead actors or production company has earlier films in our data, so "
                     "the estimate leans on genre, runtime and release date and tends toward small films.")
    else:
        missing = [role for role in given if given[role] and not found[role]]
        if missing:
            notes.append(f"No earlier films in our data for the {join(missing)}, so they count as newcomers.")
    last_year = predictor.bundle["valid_years"][1]
    if body.release_date.year > last_year:
        notes.append(f"The models have seen films released up to {last_year}; a later year is treated like {last_year}.")
    if body.original_language not in predictor.spec.languages:
        notes.append(f"'{body.original_language}' is not one of the {len(predictor.spec.languages)} most common "
                     "languages in the training data, so it is treated as 'other'.")
    return notes


@router.get("/health", response_model=Health)
def health(predictor: Predictor = Depends(get_predictor)):
    return {"status": "ok", "trained_on": predictor.bundle.get("trained_on"),
            "valid_years": predictor.bundle["valid_years"]}


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, predictor: Predictor = Depends(get_predictor)):
    unknown = sorted(set(body.genres) - set(predictor.spec.genres))
    if unknown:
        raise HTTPException(422, f"Unknown genres: {unknown}. See GET /meta/genres.")
    film = FilmInput(**body.model_dump(exclude={"release_date"}), release_date=body.release_date.isoformat())
    try:
        out = predictor.predict(film)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {**out, "warnings": warnings_for(body, out, predictor)}


@router.get("/movies", response_model=MoviePage)
def movies(q: str | None = Query(None, max_length=100, description="Title contains"),
           genre: str | None = None,
           year_from: int | None = Query(None, ge=1900, le=2100),
           year_to: int | None = Query(None, ge=1900, le=2100),
           language: str | None = Query(None, pattern=r"^[a-z]{2}$"),
           sort: Literal["vote_count", "popularity", "release_date", "vote_average", "title"] = Query(
               "vote_count", description="vote_count by default: Kaggle-only rows carry 2017 popularity values"),
           order: Literal["asc", "desc"] = "desc",
           page: int = Query(1, ge=1),
           page_size: int = Query(20, ge=1, le=100),
           catalog: Catalog = Depends(get_catalog)):
    return catalog.movies(q, genre, year_from, year_to, language, sort, order, page, page_size)


@router.get("/movies/{movie_id}", response_model=MovieDetail)
def movie(movie_id: int, catalog: Catalog = Depends(get_catalog)):
    found = catalog.movie(movie_id)
    if found is None:
        raise HTTPException(404, "Movie not found")
    return found


@router.get("/meta/genres", response_model=list[str])
def genres(predictor: Predictor = Depends(get_predictor)):
    return predictor.spec.genres


@router.get("/meta/languages", response_model=Languages)
def languages(predictor: Predictor = Depends(get_predictor)):
    return {"languages": [Language(code=c, name=LANGUAGE_NAMES.get(c, c)) for c in predictor.spec.languages],
            "note": "Any other language is treated as 'other'."}


@router.get("/people", response_model=list[NameMatch])
def people(q: str = Query(min_length=2, max_length=100), role: Literal["director", "actor"] = Query(),
           limit: int = Query(10, ge=1, le=25), catalog: Catalog = Depends(get_catalog)):
    return catalog.search_names(q, role, limit)


@router.get("/companies", response_model=list[NameMatch])
def companies(q: str = Query(min_length=2, max_length=100), limit: int = Query(10, ge=1, le=25),
              catalog: Catalog = Depends(get_catalog)):
    return catalog.search_names(q, "company", limit)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.catalog = Catalog(settings.db_path)
        app.state.predictor = Predictor(settings.model_path, load_history(settings.db_path))
        yield

    app = FastAPI(title="Movie Prediction API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])
    app.include_router(router)
    return app


app = create_app()
