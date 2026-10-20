"""Request and response models for the API."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    """What a creator enters. People and the company are TMDB ids, found with /people and /companies.

    An id not in our data counts as a newcomer with no earlier films. TMDB ids are positive, so the app sends
    negative ids for newcomers the creator names.
    """

    model_config = ConfigDict(extra="forbid")

    genres: list[str] = Field(min_length=1, max_length=5, examples=[["Action", "Science Fiction"]])
    runtime: int = Field(ge=40, le=300, description="Minutes")
    release_date: date
    original_language: str = Field("en", pattern=r"^[a-z]{2}$", description="ISO 639-1 code")
    company_id: int | None = Field(None, description="Lead production company; an unknown id is a newcomer")
    director_id: int | None = Field(None, description="An unknown id (for example a negative one) is a newcomer")
    actor_ids: list[int] = Field(default_factory=list, max_length=3,
                                 description="Top billed first; unknown ids (for example negative ones) are newcomers")
    budget: float | None = Field(None, gt=0, le=2e9, description="US dollars, today's prices")
    overview: str = Field("", max_length=1000)

    @field_validator("genres")
    @classmethod
    def unique_genres(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))

    @field_validator("actor_ids")
    @classmethod
    def unique_actors(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("actor_ids must not repeat")
        return v


class Driver(BaseModel):
    factor: str
    effect: float = Field(description="Log scale for budget and revenue, score points for score")
    text: str


class Range(BaseModel):
    low: float
    point: float
    high: float
    model: str
    label: str | None = None
    given: float | None = Field(None, description="The budget the creator entered, if any")
    drivers: list[Driver]


class PriorFilms(BaseModel):
    director: int
    actors_max: int
    company: int


class PredictResponse(BaseModel):
    budget: Range
    revenue: Range
    score: Range
    prior_films: PriorFilms = Field(description="Earlier films in the data for the director, best-known actor and company")
    warnings: list[str]


class MovieSummary(BaseModel):
    id: int
    title: str
    release_year: int | None
    genres: list[str]
    vote_average: float | None
    vote_count: int | None
    poster_url: str | None


class MoviePage(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[MovieSummary]


class Credit(BaseModel):
    id: int
    name: str


class CastCredit(Credit):
    character: str | None


class CrewCredit(Credit):
    job: str


class MovieDetail(BaseModel):
    id: int
    title: str
    original_title: str | None
    overview: str | None
    release_date: str | None
    release_year: int | None
    runtime: float | None
    original_language: str | None
    certification: str | None
    budget: float | None = Field(description="Nominal US dollars")
    revenue: float | None = Field(description="Nominal US dollars")
    vote_average: float | None
    vote_count: int | None
    poster_url: str | None
    genres: list[str]
    companies: list[Credit]
    directors: list[Credit]
    crew: list[CrewCredit] = Field(description="Key crew other than the directors: writers, producers, composer")
    cast: list[CastCredit]


class NameMatch(BaseModel):
    id: int
    name: str
    n_films: int
    known_for: str | None


class Language(BaseModel):
    code: str
    name: str


class Languages(BaseModel):
    languages: list[Language]
    note: str


class Health(BaseModel):
    status: str
    trained_on: str | None
    valid_years: tuple[int, int]
