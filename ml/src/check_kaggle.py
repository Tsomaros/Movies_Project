"""Download (if needed) and sanity-check the Kaggle TMDB 5000 fallback dataset.

Downloads "TMDB 5000 Movie Dataset" (tmdb/tmdb-movie-metadata) via kagglehub
into ml/data/raw/kaggle/. The dataset is public, so no credentials are needed;
KAGGLE_USERNAME/KAGGLE_KEY in .env are used if set. Alternatively, place tmdb_5000_movies.csv and
tmdb_5000_credits.csv there by hand and the download is skipped.

Usage:
    python ml/src/check_kaggle.py           # download if missing, then summarize
    python ml/src/check_kaggle.py --force   # re-download
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
KAGGLE_DIR = ROOT / "ml" / "data" / "raw" / "kaggle"
DATASET = "tmdb/tmdb-movie-metadata"
FILES = ["tmdb_5000_movies.csv", "tmdb_5000_credits.csv"]


def ensure_downloaded(force: bool = False) -> None:
    if not force and all((KAGGLE_DIR / name).exists() for name in FILES):
        print(f"CSVs already in {KAGGLE_DIR}; skipping download.")
        return

    load_dotenv(ROOT / ".env")  # must run before kagglehub reads credentials
    import kagglehub

    print(f"Downloading {DATASET} from Kaggle...")
    try:
        cache_dir = Path(kagglehub.dataset_download(DATASET, force_download=force))
    except Exception as exc:
        raise SystemExit(
            f"Kaggle download failed: {exc}\n"
            "Add KAGGLE_USERNAME and KAGGLE_KEY to .env (kaggle.com -> Settings -> API -> Create New Token),\n"
            f"or download the CSVs by hand into {KAGGLE_DIR}."
        )

    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(cache_dir / name, KAGGLE_DIR / name)
    print(f"Copied {', '.join(FILES)} -> {KAGGLE_DIR}")


def summarize() -> None:
    movies = pd.read_csv(KAGGLE_DIR / FILES[0])
    credits = pd.read_csv(KAGGLE_DIR / FILES[1])

    for name, df in [(FILES[0], movies), (FILES[1], credits)]:
        print(f"{name}: {len(df):,} rows")
        print(f"  columns: {', '.join(df.columns)}")

    years = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
    usable = ((movies["budget"] > 0) & (movies["revenue"] > 0)).sum()
    print(f"Release years: {int(years.min())}-{int(years.max())}")
    print(f"Movies with budget and revenue > 0: {usable:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if the CSVs exist")
    args = parser.parse_args()

    ensure_downloaded(force=args.force)
    summarize()


if __name__ == "__main__":
    main()
