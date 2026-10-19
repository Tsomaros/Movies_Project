"""Collect raw movie data from the TMDB API.

Phase 1 (discover): for each release year, collect the most-voted movie IDs.
Phase 2 (details):  fetch details + credits + release dates for each ID and
append the raw JSON to a JSONL file. Re-running resumes where it stopped.

Usage:
    python ml/src/collect_tmdb.py                 # full run (~10k movies)
    python ml/src/collect_tmdb.py --limit 20      # smoke test
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "ml" / "data" / "raw"
IDS_PATH = RAW_DIR / "movie_ids.json"
MOVIES_PATH = RAW_DIR / "tmdb_movies.jsonl"
MISSING_PATH = RAW_DIR / "missing_ids.json"

BASE_URL = "https://api.themoviedb.org/3"
PAGE_SIZE = 20
MAX_RETRIES = 6
TOP_CAST = 10
KEY_CREW_JOBS = {"Director", "Screenplay", "Writer", "Producer", "Original Music Composer"}

log = logging.getLogger("collect_tmdb")


class TMDBClient:
    """Small throttled TMDB client with retries."""

    def __init__(self, requests_per_second: float = 20.0):
        load_dotenv(ROOT / ".env")
        token = os.getenv("TMDB_READ_TOKEN", "").strip()
        api_key = os.getenv("TMDB_API_KEY", "").strip()
        if not token and not api_key:
            raise SystemExit("Set TMDB_READ_TOKEN or TMDB_API_KEY in .env (see .env.example).")

        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        self.default_params = {}
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        else:
            self.default_params["api_key"] = api_key

        self.min_interval = 1.0 / requests_per_second
        self._last_request = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(self, path: str, **params) -> dict | None:
        """GET a TMDB endpoint. Returns None on 404."""
        params = {**self.default_params, **params}
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=15)
            except (requests.ConnectionError, requests.Timeout) as exc:
                delay = 2**attempt
                log.warning("Network error on %s (%s); retrying in %ss", path, exc, delay)
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code == 401:
                raise SystemExit("TMDB rejected the credentials (401). Check your .env.")
            if resp.status_code == 429:
                delay = float(resp.headers.get("Retry-After", 2**attempt))
                log.warning("Rate limited; sleeping %ss", delay)
                time.sleep(delay)
                continue
            if resp.status_code >= 500:
                delay = 2**attempt
                log.warning("Server error %s on %s; retrying in %ss", resp.status_code, path, delay)
                time.sleep(delay)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Giving up on {path} after {MAX_RETRIES} attempts")


def discover_ids(client: TMDBClient, start_year: int, end_year: int, per_year: int) -> list[int]:
    """Most-voted movie IDs per release year, deduplicated, in order."""
    pages = -(-per_year // PAGE_SIZE)  # ceil division
    ids: list[int] = []
    seen: set[int] = set()
    for year in tqdm(range(start_year, end_year + 1), desc="Discover (years)"):
        year_ids: list[int] = []
        for page in range(1, pages + 1):
            data = client.get(
                "/discover/movie",
                primary_release_year=year,
                sort_by="vote_count.desc",
                include_adult="false",
                page=page,
            )
            if not data or not data.get("results"):
                break
            year_ids.extend(m["id"] for m in data["results"])
            if page >= data.get("total_pages", 0):
                break
        for movie_id in year_ids[:per_year]:
            if movie_id not in seen:
                seen.add(movie_id)
                ids.append(movie_id)
    return ids


def trim_credits(movie: dict) -> dict:
    """Keep only the top-billed cast and key crew to keep the file small."""
    credits = movie.get("credits") or {}
    cast = sorted(credits.get("cast", []), key=lambda c: c.get("order", 999))[:TOP_CAST]
    crew = [c for c in credits.get("crew", []) if c.get("job") in KEY_CREW_JOBS]
    movie["credits"] = {"cast": cast, "crew": crew}
    return movie


def load_done_ids() -> set[int]:
    done: set[int] = set()
    if MOVIES_PATH.exists():
        with MOVIES_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    if MISSING_PATH.exists():
        done.update(json.loads(MISSING_PATH.read_text()))
    return done


def fetch_details(client: TMDBClient, ids: list[int]) -> None:
    done = load_done_ids()
    todo = [i for i in ids if i not in done]
    log.info("%d IDs total, %d already collected, %d to fetch", len(ids), len(ids) - len(todo), len(todo))

    missing = json.loads(MISSING_PATH.read_text()) if MISSING_PATH.exists() else []
    with MOVIES_PATH.open("a", encoding="utf-8") as out:
        for movie_id in tqdm(todo, desc="Details"):
            movie = client.get(f"/movie/{movie_id}", append_to_response="credits,release_dates")
            if movie is None:
                log.info("Movie %s not found (404); skipping", movie_id)
                missing.append(movie_id)
                MISSING_PATH.write_text(json.dumps(missing))
                continue
            out.write(json.dumps(trim_credits(movie), ensure_ascii=False) + "\n")
            out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--per-year", type=int, default=220, help="movies per release year")
    parser.add_argument("--limit", type=int, help="only fetch details for the first N IDs (smoke test)")
    parser.add_argument("--refresh-ids", action="store_true", help="re-run discovery even if movie_ids.json exists")
    parser.add_argument("--rps", type=float, default=20.0, help="max requests per second")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = TMDBClient(requests_per_second=args.rps)

    if IDS_PATH.exists() and not args.refresh_ids:
        ids = json.loads(IDS_PATH.read_text())
        log.info("Loaded %d IDs from %s", len(ids), IDS_PATH.name)
    else:
        ids = discover_ids(client, args.start_year, args.end_year, args.per_year)
        IDS_PATH.write_text(json.dumps(ids))
        log.info("Discovered %d IDs -> %s", len(ids), IDS_PATH.name)

    if args.limit:
        ids = ids[: args.limit]
    fetch_details(client, ids)
    log.info("Done. Output: %s", MOVIES_PATH)


if __name__ == "__main__":
    main()
