"""Read-only SQLite queries for browsing movies and for the creator form's name search."""

import sqlite3
from contextlib import closing
from pathlib import Path

from features import MIN_SCORE_VOTES

POSTER_BASE = "https://image.tmdb.org/t/p/w342"
SORTS = {"popularity": "m.popularity", "vote_count": "m.vote_count", "release_date": "m.release_date",
         "vote_average": "m.vote_average", "title": "m.title COLLATE NOCASE"}
# table, id column and extra condition for each kind of name search
NAMES = {"director": ("crew", "person_id", "AND x.job = 'Director'"),
         "actor": ("cast_members", "person_id", ""),
         "company": ("companies", "company_id", "")}


def like(q: str) -> str:
    """A LIKE pattern for 'contains q', with q's own % and _ matched literally."""
    return "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def poster_url(path: str | None) -> str | None:
    return POSTER_BASE + path if path else None


def placeholders(values) -> str:
    return ", ".join("?" * len(values))


class Catalog:
    def __init__(self, path: Path):
        self.uri = f"file:{Path(path).as_posix()}?mode=ro"
        with self._connect() as conn:  # fail at startup rather than on the first request
            conn.execute("SELECT 1 FROM movies LIMIT 1")

    def _connect(self):
        conn = sqlite3.connect(self.uri, uri=True)
        conn.row_factory = sqlite3.Row
        return closing(conn)

    @staticmethod
    def _genres(conn, ids: list[int]) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        rows = conn.execute(f"SELECT movie_id, name FROM genres WHERE movie_id IN ({placeholders(ids)}) ORDER BY rowid",
                            ids)
        for r in rows:
            out.setdefault(r["movie_id"], []).append(r["name"])
        return out

    def movies(self, q: str | None = None, genre: str | None = None, year_from: int | None = None,
               year_to: int | None = None, language: str | None = None, sort: str = "vote_count",
               order: str = "desc", page: int = 1, page_size: int = 20) -> dict:
        where, args = [], []
        if q:
            where.append("m.title LIKE ? ESCAPE '\\'")
            args.append(like(q))
        if genre:
            where.append("EXISTS (SELECT 1 FROM genres g WHERE g.movie_id = m.id AND g.name = ?)")
            args.append(genre)
        if year_from is not None:
            where.append("m.release_year >= ?")
            args.append(year_from)
        if year_to is not None:
            where.append("m.release_year <= ?")
            args.append(year_to)
        if language:
            where.append("m.original_language = ?")
            args.append(language)
        if sort == "vote_average":  # a 10/10 from three votes shouldn't top the list
            where.append("m.vote_count >= ?")
            args.append(MIN_SCORE_VOTES)
        sql_where = "WHERE " + " AND ".join(where) if where else ""
        col, direction = SORTS[sort], "DESC" if order == "desc" else "ASC"
        with self._connect() as conn:
            total = conn.execute(f"SELECT count(*) FROM movies m {sql_where}", args).fetchone()[0]
            rows = conn.execute(
                f"SELECT m.id, m.title, m.release_year, m.vote_average, m.vote_count, m.poster_path FROM movies m "
                f"{sql_where} ORDER BY {col} IS NULL, {col} {direction}, m.id LIMIT ? OFFSET ?",
                [*args, page_size, (page - 1) * page_size]).fetchall()
            genres = self._genres(conn, [r["id"] for r in rows])
        items = [{"id": r["id"], "title": r["title"], "release_year": r["release_year"], "genres": genres.get(r["id"], []),
                  "vote_average": r["vote_average"], "vote_count": r["vote_count"],
                  "poster_url": poster_url(r["poster_path"])} for r in rows]
        return {"total": total, "page": page, "page_size": page_size, "items": items}

    def movie(self, movie_id: int) -> dict | None:
        with self._connect() as conn:
            m = conn.execute("SELECT * FROM movies WHERE id = ?", [movie_id]).fetchone()
            if m is None:
                return None
            companies = conn.execute("SELECT company_id AS id, name FROM companies WHERE movie_id = ? "
                                     "ORDER BY company_order", [movie_id]).fetchall()
            directors = conn.execute("SELECT person_id AS id, name FROM crew WHERE movie_id = ? AND job = 'Director' "
                                     "ORDER BY rowid", [movie_id]).fetchall()
            crew = conn.execute("SELECT person_id AS id, name, job FROM crew WHERE movie_id = ? AND job != 'Director' "
                                "ORDER BY rowid", [movie_id]).fetchall()
            cast = conn.execute("SELECT person_id AS id, name, character FROM cast_members WHERE movie_id = ? "
                                "ORDER BY cast_order", [movie_id]).fetchall()
            genres = self._genres(conn, [movie_id]).get(movie_id, [])
        keys = ["id", "title", "original_title", "overview", "release_date", "release_year", "runtime",
                "original_language", "certification", "budget", "revenue", "vote_average", "vote_count"]
        return {**{k: m[k] for k in keys}, "poster_url": poster_url(m["poster_path"]), "genres": genres,
                "companies": [dict(r) for r in companies], "directors": [dict(r) for r in directors],
                "crew": [dict(r) for r in crew], "cast": [dict(r) for r in cast]}

    def search_names(self, q: str, kind: str, limit: int = 10) -> list[dict]:
        """Directors, actors or companies whose name contains q, the ones with the most films first."""
        table, id_col, extra = NAMES[kind]
        with self._connect() as conn:
            # NOT INDEXED: walking the id index for the GROUP BY is ~25x slower than one scan of the table
            rows = conn.execute(
                f"SELECT x.{id_col} AS id, max(x.name) AS name, count(DISTINCT x.movie_id) AS n_films "
                f"FROM {table} x NOT INDEXED WHERE x.name LIKE ? ESCAPE '\\' {extra} "
                f"GROUP BY x.{id_col} ORDER BY n_films DESC, name LIMIT ?",
                [like(q), limit]).fetchall()
            ids = [r["id"] for r in rows]
            known_for: dict[int, str] = {}
            for r in conn.execute(f"SELECT x.{id_col} AS id, m.title FROM {table} x JOIN movies m ON m.id = x.movie_id "
                                  f"WHERE x.{id_col} IN ({placeholders(ids)}) {extra} ORDER BY m.vote_count DESC", ids):
                known_for.setdefault(r["id"], r["title"])  # their most-voted film tells namesakes apart
        return [{**dict(r), "known_for": known_for.get(r["id"])} for r in rows]
