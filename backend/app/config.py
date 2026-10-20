"""Settings from environment variables, with defaults for running from the repo."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    db_path: Path
    model_path: Path
    cors_origins: list[str]


def get_settings() -> Settings:
    origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    return Settings(db_path=Path(os.getenv("MOVIES_DB", ROOT / "ml" / "data" / "movies.db")),
                    model_path=Path(os.getenv("MODEL_PATH", ROOT / "ml" / "models" / "predictor.joblib")),
                    cors_origins=[o.strip() for o in origins.split(",") if o.strip()])
