"""IMDb facts for provider movies and series, joined by title and year.

Xtream catalogs carry little beyond a name: movie lists have no people at all, and
series lists only sometimes. IMDb's non-commercial datasets have ids, genres,
runtimes, ratings, and full credits. Only titles in the catalog are kept, in
``imdb_*`` tables of neTV's database.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from concurrent.futures import ProcessPoolExecutor
from typing import ParamSpec, TypedDict, TypeVar, cast
from urllib import parse

import functools
import gzip
import logging
import multiprocessing
import re
import sqlite3
import time
import unicodedata

from treekle import json_loads

from netv import db
from netv.util import safe_urlopen
from netv.xtream import ProviderRecord, text_field


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


class Info(TypedDict):
    """What IMDb says about one catalog title."""

    imdb_id: str
    title: str
    year: int
    genres: list[str]
    runtime_min: int
    rating: float
    votes: int
    directors: list[str]
    writers: list[str]
    cast: list[str]
    crew: dict[str, list[str]]


def title_year(name: str) -> tuple[str, int] | None:
    """Split a provider name like ``"Heat - 1995"`` or ``"Heat (1995)"``."""
    match = re.fullmatch(r"\s*(.+?)\s*(?:-\s*|\()((?:18|19|20)\d\d)\)?\s*", name)
    return (cast(str, match[1]), int(match[2])) if match else None


def title_key(title: str, year: int) -> str:
    """Return a key on which spelling variants of one title and year agree.

    Args:
        title: Title as the provider or IMDb spells it.
        year: Release year.

    Returns:
        key: Case-, accent-, and punctuation-free title, then the year.
    """
    folded = unicodedata.normalize(
        "NFKD",
        title.replace(
            "&",
            " and ",  # pragma: no mutate[case-insensitive]
        ),
    ).casefold()
    return "".join(c for c in folded if c.isalnum()) + f"|{year}"


def init() -> None:
    """Create the index tables in the open database."""
    conn = db.connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS imdb_titles (
            key TEXT PRIMARY KEY,
            imdb_id TEXT NOT NULL,
            title TEXT NOT NULL,
            year INTEGER NOT NULL,
            genres TEXT NOT NULL,
            runtime_min INTEGER NOT NULL,
            rating REAL NOT NULL,
            votes INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS imdb_names (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS imdb_credits (
            title_key TEXT NOT NULL,
            role TEXT NOT NULL,
            billing INTEGER NOT NULL,
            name_id INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_imdb_credits_title ON imdb_credits(title_key);
        CREATE INDEX IF NOT EXISTS idx_imdb_credits_name ON imdb_credits(name_id);
        CREATE TABLE IF NOT EXISTS imdb_builds (built_ts REAL NOT NULL);
    """)
    conn.commit()


def is_stale(*, max_age_sec: float) -> bool:
    """Whether the index was never built or is older than ``max_age_sec``.

    Args:
        max_age_sec: Oldest a build may be and still count as fresh.

    Returns:
        stale: True when the index needs a rebuild.
    """
    built = _optional_float(
        db.connection()
        .execute(
            "SELECT MAX(built_ts) FROM imdb_builds",  # pragma: no mutate[case-insensitive]
        )
        .fetchone()[0],
    )
    return built is None or time.time() - built > max_age_sec


def resolved_keys(records: Sequence[ProviderRecord]) -> list[str | None]:
    """Return each record's index key: its TMDB id's when indexed, else its title's."""
    present = {str(row[0]) for row in db.connection().execute("SELECT key FROM imdb_titles")}
    return [next((k for k in _catalog_keys(r) if k in present), None) for r in records]


def matching_keys(match: Callable[[str], bool]) -> set[str]:
    """Return the index keys of titles crediting anyone whose name ``match`` accepts.

    Args:
        match: Predicate on a person's name, run once per distinct name.

    Returns:
        keys: Index keys, as ``resolved_keys`` gives them.
    """
    conn = db.connection()
    # The file, plus a build's time and the newest saved name, identify the names.
    file = str(conn.execute("PRAGMA database_list").fetchone()[2])
    built, max_id = conn.execute(
        "SELECT (SELECT MAX(built_ts) FROM imdb_builds), (SELECT MAX(id) FROM imdb_names)",  # pragma: no mutate[case-insensitive]
    ).fetchone()
    names = _all_names(file, _optional_float(built), None if max_id is None else int(max_id))
    matched = [(i,) for i, name in names if match(name)]
    conn.execute(
        "CREATE TEMP TABLE IF NOT EXISTS matched_names (id INTEGER PRIMARY KEY)",  # pragma: no mutate[case-insensitive]
    )
    with conn:
        conn.execute("DELETE FROM matched_names")  # pragma: no mutate[case-insensitive]
        conn.executemany(
            "INSERT INTO matched_names VALUES (?)",  # pragma: no mutate[case-insensitive]
            matched,
        )
        rows = conn.execute(
            "SELECT DISTINCT title_key FROM imdb_credits"  # pragma: no mutate[case-insensitive]
            " WHERE name_id IN (SELECT id FROM matched_names)",  # pragma: no mutate[case-insensitive]
        ).fetchall()
    return {str(row[0]) for row in rows}


def info(record: ProviderRecord) -> Info | None:
    """Return what the index knows of a catalog movie or series.

    Args:
        record: Provider movie or series.

    Returns:
        found: IMDb's facts, or None for a title the index lacks.
    """
    conn = db.connection()
    return next(
        (found for key in filter(None, _catalog_keys(record)) if (found := _info_at(conn, key))),
        None,
    )


def credits(record: ProviderRecord, *, directors: str, cast: str) -> tuple[list[str], list[str]]:
    """Return the directors and cast a detail page shows, all of which search finds.

    IMDb's credits when the index has the title. Otherwise the provider's, which
    are first saved to the index, so that every name a page links finds the title.

    Args:
        record: Provider movie or series.
        directors: The provider's comma-separated directors.
        cast: The provider's comma-separated cast.

    Returns:
        directors: Names to show as directors.
        cast: Names to show as cast.
    """
    found = info(record)
    if found is not None:
        return found["directors"], found["cast"]
    provided = (_split(directors), _split(cast))
    key = next(filter(None, _catalog_keys(record)), None)
    if key is not None and any(provided):
        _add(key, text_field(record, "name"), *provided)
    return provided


def people(found: Info) -> list[str]:
    """Return everyone credited on a title, cast first."""
    return [*found["cast"], *found["directors"], *found["writers"]] + [
        name for names in found["crew"].values() for name in names
    ]


def build(
    records: Sequence[ProviderRecord],
    *,
    base_url: str = "https://datasets.imdbws.com",
    sparql_url: str = "https://query.wikidata.org/sparql",
) -> int:
    """Replace the index with what IMDb knows of every catalog movie and series.

    A record joins by its TMDB id, which Wikidata maps to an IMDb id. One without
    a mapping joins by title and year: a provider year may differ from IMDb's by
    one, so neighbors match too, and more votes, then an exact year, wins.
    Parsing is CPU-bound, so files that do not depend on each other are read in
    parallel processes.

    Args:
        records: Provider movies and series.
        base_url: Where IMDb's ``*.tsv.gz`` datasets are served.
        sparql_url: Wikidata's SPARQL endpoint.

    Returns:
        count: Catalog titles the index describes.
    """
    keys = [_catalog_keys(r) for r in records]
    wanted: dict[str, set[str]] = {}
    for record in records:
        found = _catalog_title(record)
        if found is not None:
            kind, title, year = found
            for near in (year, year - 1, year + 1):
                wanted.setdefault(f"{kind}:{title_key(title, near)}", set()).add(
                    f"{kind}:{title_key(title, year)}",
                )
    # Forking a threaded server can copy a held lock into the child.
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=spawn) as executor:
        pending_ratings = executor.submit(_ratings, base_url)
        by_tmdb = executor.submit(_tmdb_to_imdb, sparql_url).result()
        mapped = {k: by_tmdb[k] for k, _ in keys if k is not None and k in by_tmdb}
        scanned = executor.submit(_scan_basics, base_url, wanted, set(mapped.values()))
        candidates, basics = scanned.result()
        ratings = pending_ratings.result()
        chosen = {
            key: max(found, key=lambda f: (ratings.get(f[1], (0.0, 0))[1], f[0]))[1]
            for key, found in candidates.items()
        }
        # An id IMDb no longer lists falls back to the title join.
        chosen |= {k: tconst for k, tconst in mapped.items() if tconst in basics}
        needed = {
            key
            for by_id, by_title in keys
            for key in ((by_id,) if by_id in chosen else (by_id, by_title))
            if key is not None and key in chosen
        }
        kept = {chosen[k]: basics[chosen[k]] for k in needed}
        crew = executor.submit(_crew, base_url, kept)
        cast_ids, roles = executor.submit(_billing, base_url, kept).result()
        infos = _infos(base_url, kept, ratings, crew.result(), (cast_ids, roles))
    # A record IMDb now describes drops the credits its detail page saved.
    superseded = {k for pair in keys if not needed.isdisjoint(pair) for k in pair if k is not None}
    _write({key: infos[chosen[key]] for key in needed}, superseded)
    log.info("IMDb index: %d catalog titles", len(needed))
    return len(needed)


def _info_at(conn: sqlite3.Connection, key: str) -> Info | None:
    """Return the index entry at ``key``, or None."""
    row = conn.execute(
        "SELECT imdb_id, title, year, genres, runtime_min, rating, votes"  # pragma: no mutate[case-insensitive]
        " FROM imdb_titles WHERE key = ?",  # pragma: no mutate[case-insensitive]
        (key,),
    ).fetchone()
    if row is None:
        return None
    credits: dict[str, list[str]] = {}
    for role, name in conn.execute(
        "SELECT c.role, n.name FROM imdb_credits c JOIN imdb_names n ON n.id = c.name_id"  # pragma: no mutate[case-insensitive]
        " WHERE c.title_key = ? ORDER BY c.role, c.billing",  # pragma: no mutate[case-insensitive]
        (key,),
    ):
        credits.setdefault(str(role), []).append(str(name))
    imdb_id, title, year, genres, runtime_min, rating, votes = row
    return Info(
        imdb_id=str(imdb_id),
        title=str(title),
        year=int(year),
        genres=str(genres).split(",") if genres else [],
        runtime_min=int(runtime_min),
        rating=float(rating),
        votes=int(votes),
        directors=credits.pop("director", []),
        writers=credits.pop("writer", []),
        cast=credits.pop("cast", []),
        crew=credits,
    )


# A build keeps them until IMDb has the title.
#
# The title row's empty ``imdb_id`` marks it as the provider's.
def _add(key: str, title: str, directors: list[str], cast: list[str]) -> None:
    """Index one title's provider credits."""
    conn = db.connection()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO imdb_titles VALUES (?, '', ?, 0, '', 0, 0.0, 0)",  # pragma: no mutate[case-insensitive]
            (key, title),
        )
        conn.execute(
            "DELETE FROM imdb_credits WHERE title_key = ?",  # pragma: no mutate[case-insensitive]
            (key,),
        )
        names = [*directors, *cast]
        conn.executemany(
            "INSERT OR IGNORE INTO imdb_names (name) VALUES (?)",  # pragma: no mutate[case-insensitive]
            ((name,) for name in names),
        )
        ids = {
            str(name): int(name_id)
            for name, name_id in conn.execute(
                f"SELECT name, id FROM imdb_names WHERE name IN ({','.join('?' * len(names))})",  # noqa: S608 -- Only placeholders are interpolated.
                names,
            )
        }
        conn.executemany(
            "INSERT INTO imdb_credits VALUES (?, ?, ?, ?)",  # pragma: no mutate[case-insensitive]
            [
                (key, role, billing, ids[name])
                for role, credited in (("director", directors), ("cast", cast))
                for billing, name in enumerate(credited)
            ],
        )


def _write(index: Mapping[str, Info], superseded: Set[str]) -> None:
    """Replace the index tables in one transaction, so readers never see half."""
    conn = db.connection()
    saved = (
        str(row[0])
        for row in conn.execute(
            "SELECT key FROM imdb_titles WHERE imdb_id = ''",  # pragma: no mutate[case-insensitive]
        )
    )
    provided = [key for key in saved if key not in index and key not in superseded]
    index = {**{k: found for k in provided if (found := _info_at(conn, k))}, **index}
    names = sorted({name for found in index.values() for name in people(found)})
    name_ids = {name: i for i, name in enumerate(names)}
    credits = [
        (key, role, billing, name_ids[name])
        for key, found in index.items()
        for role, credited in (
            ("cast", found["cast"]),
            ("director", found["directors"]),
            ("writer", found["writers"]),
            *found["crew"].items(),
        )
        for billing, name in enumerate(credited)
    ]
    with conn:
        conn.execute("DELETE FROM imdb_titles")
        conn.execute("DELETE FROM imdb_names")
        conn.execute("DELETE FROM imdb_credits")
        conn.execute("DELETE FROM imdb_builds")
        conn.executemany(
            "INSERT INTO imdb_titles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    key,
                    f["imdb_id"],
                    f["title"],
                    f["year"],
                    ",".join(f["genres"]),
                    f["runtime_min"],
                    f["rating"],
                    f["votes"],
                )
                for key, f in index.items()
            ),
        )
        conn.executemany(
            "INSERT INTO imdb_names VALUES (?, ?)",
            ((i, n) for n, i in name_ids.items()),
        )
        conn.executemany("INSERT INTO imdb_credits VALUES (?, ?, ?, ?)", credits)
        conn.execute("INSERT INTO imdb_builds VALUES (?)", (time.time(),))


def _catalog_keys(record: ProviderRecord) -> tuple[str | None, str | None]:
    """The record's index keys: by TMDB id, and by title and year."""
    return _keys(
        "series" if "series_id" in record else "movies",
        text_field(record, "tmdb"),
        text_field(record, "name"),
        text_field(record, "releaseDate") or text_field(record, "release_date"),
    )


# Each search keys the whole catalog; normalizing 36k titles each time costs 0.2 s.
@functools.lru_cache(maxsize=1 << 17)
def _keys(kind: str, tmdb: str, name: str, aired: str) -> tuple[str | None, str | None]:
    found = _title_and_year(name, aired)
    return (
        f"{kind}:tmdb:{tmdb}" if tmdb.isdigit() else None,
        None if found is None else f"{kind}:{title_key(*found)}",
    )


# Search reads every name; loading them costs 0.16 s, so each index version loads once.
@functools.lru_cache(maxsize=1)
def _all_names(file: str, built: float | None, max_id: int | None) -> list[tuple[int, str]]:
    del file, built, max_id
    rows = db.connection().execute("SELECT id, name FROM imdb_names")
    return [(int(row[0]), str(row[1])) for row in rows]


def _catalog_title(record: ProviderRecord) -> tuple[str, str, int] | None:
    """The kind, title, and year of a catalog record; series fall back to air date."""
    found = _title_and_year(
        text_field(record, "name"),
        text_field(record, "releaseDate") or text_field(record, "release_date"),
    )
    kind = "series" if "series_id" in record else "movies"
    return None if found is None else (kind, *found)


def _title_and_year(name: str, aired: str) -> tuple[str, int] | None:
    """A name's title and year, the year from ``aired`` when the name lacks one."""
    parsed = title_year(name)
    if parsed is not None:
        return parsed
    return (name, int(aired[:4])) if aired[:4].isdigit() else None


def _retried(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Wrap ``fn`` to retry twice when the connection drops; IMDb's files are large."""

    @functools.wraps(fn)
    def retrying(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        for _ in range(2):
            try:
                return fn(*args, **kwargs)
            except OSError as exc:  # URLError and a reset connection are OSErrors.
                log.warning("IMDb download failed, retrying: %s", exc)
        return fn(*args, **kwargs)

    return retrying


@_retried
def _scan_basics(
    base_url: str,
    wanted: Mapping[str, set[str]],
    ids: Set[str],
) -> tuple[dict[str, list[tuple[bool, str]]], dict[str, list[str]]]:
    """Return each catalog key's IMDb candidates, flagged when the year is exact."""
    # Episodes are never a catalog title.
    kinds = {
        "movie": "movies",
        "tvMovie": "movies",
        "video": "movies",
        "tvSpecial": "movies",
        "short": "movies",
        "tvShort": "movies",
        "tvSeries": "series",
        "tvMiniSeries": "series",
    }
    candidates: dict[str, list[tuple[bool, str]]] = {}
    basics: dict[str, list[str]] = {}
    for row in _rows(f"{base_url}/title.basics.tsv.gz"):
        if row[0] in ids:
            basics[row[0]] = row
        kind = kinds.get(row[1])
        if kind is None or not row[5].isdigit():
            continue
        for title in {row[2], row[3]}:
            key = f"{kind}:{title_key(title, int(row[5]))}"
            for catalog_key in wanted.get(key, ()):
                candidates.setdefault(catalog_key, []).append((key == catalog_key, row[0]))
                basics[row[0]] = row
    return candidates, basics


def _tmdb_query_url(sparql_url: str) -> str:
    """The Wikidata query mapping TMDB movie and TV ids to IMDb ids."""
    query = (
        "SELECT ?kind ?tmdb ?imdb WHERE {"
        ' { ?i wdt:P4947 ?tmdb BIND("movies" AS ?kind) }'
        ' UNION { ?i wdt:P4983 ?tmdb BIND("series" AS ?kind) }'
        " ?i wdt:P345 ?imdb }"
    )
    return f"{sparql_url}?format=json&query={parse.quote(query)}"


class _Value(TypedDict):
    value: str


class _Binding(TypedDict):
    kind: _Value
    tmdb: _Value
    imdb: _Value


class _Bindings(TypedDict):
    bindings: list[_Binding]


class _SparqlResult(TypedDict):
    results: _Bindings


def _tmdb_to_imdb(sparql_url: str) -> dict[str, str]:
    """Map ``kind:tmdb:<id>`` to its IMDb id; empty when Wikidata is unreachable."""
    try:
        # Wikidata refuses requests without a descriptive User-Agent naming a contact.
        agent = "neTV (https://github.com/jvdillon/netv)"
        with safe_urlopen(_tmdb_query_url(sparql_url), timeout=120, user_agent=agent) as resp:
            rows = json_loads(resp.read(), _SparqlResult)["results"]["bindings"]
    except (OSError, ValueError) as exc:  # URLError is an OSError; ReadError a ValueError.
        log.warning("Wikidata TMDB ids unavailable, joining by title: %s", exc)
        return {}
    return {
        f"{row['kind']['value']}:tmdb:{row['tmdb']['value']}": row["imdb"]["value"] for row in rows
    }


@_retried
def _ratings(base_url: str) -> dict[str, tuple[float, int]]:
    """Every rated title's average and vote count."""
    return {
        row[0]: (float(row[1]), int(row[2])) for row in _rows(f"{base_url}/title.ratings.tsv.gz")
    }


@_retried
def _crew(base_url: str, basics: Mapping[str, list[str]]) -> dict[str, tuple[list[str], list[str]]]:
    """Each title's director and writer ids."""
    return {
        row[0]: (_ids(row[1]), _ids(row[2]))
        for row in _rows(f"{base_url}/title.crew.tsv.gz", basics)
    }


def _infos(
    base_url: str,
    basics: Mapping[str, list[str]],
    ratings: Mapping[str, tuple[float, int]],
    crew: Mapping[str, tuple[list[str], list[str]]],
    billing: tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]],
) -> dict[str, Info]:
    """Describe each chosen title, naming the people its credits list by id."""
    cast_ids, roles = billing
    everyone = {p for d, w in crew.values() for p in (*d, *w)}
    everyone |= {p for ids in cast_ids.values() for p in ids}
    everyone |= {p for by_role in roles.values() for ids in by_role.values() for p in ids}
    names = {row[0]: row[1] for row in _rows(f"{base_url}/name.basics.tsv.gz", everyone)}
    infos: dict[str, Info] = {}
    for tconst, row in basics.items():
        directors, writers = crew.get(tconst, ([], []))
        rating, votes = ratings.get(tconst, (0.0, 0))
        infos[tconst] = Info(
            imdb_id=tconst,
            title=row[2],
            year=int(row[5]) if row[5].isdigit() else 0,
            genres=_ids(row[8]),
            runtime_min=int(row[7]) if row[7].isdigit() else 0,
            rating=rating,
            votes=votes,
            directors=_names(directors, names),
            writers=_names(writers, names),
            cast=_names(cast_ids.get(tconst, []), names),
            crew={role: _names(ids, names) for role, ids in roles.get(tconst, {}).items()},
        )
    return infos


@_retried
def _billing(
    base_url: str,
    basics: Mapping[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    """Return each title's top-billed cast and its other principals by role."""
    billed: dict[str, list[tuple[int, str]]] = {}
    roles: dict[str, dict[str, list[str]]] = {}
    for row in _rows(f"{base_url}/title.principals.tsv.gz", basics):
        if row[3] in ("actor", "actress", "self"):
            billed.setdefault(row[0], []).append((int(row[1]), row[2]))
        # The crew file already lists directors and writers; archive footage is reused.
        elif row[3] not in {"director", "writer", "archive_footage", "archive_sound"}:
            roles.setdefault(row[0], {}).setdefault(row[3], []).append(row[2])
    # A series bills one actor once per character, so names repeat.
    cast_ids = {
        t: list(dict.fromkeys(p for _, p in sorted(found)))[:10] for t, found in billed.items()
    }
    return cast_ids, roles


def _rows(url: str, keep: Iterable[str] = ()) -> Iterator[list[str]]:
    """Stream a gzipped IMDb TSV, skipping its header and, given ``keep``, other ids."""
    wanted = frozenset(keep)
    with safe_urlopen(url, timeout=60) as resp, gzip.open(resp) as f:
        next(f)
        for raw in f:
            line = raw.decode()
            # Reading only the id first keeps the 100M-row principals file fast.
            if wanted and line.partition("\t")[0] not in wanted:
                continue
            yield line.rstrip("\n").split("\t")


def _optional_float(value: object) -> float | None:
    """A nullable SQLite number as a float."""
    return None if value is None else float(str(value))


def _split(names: str) -> list[str]:
    """Split a provider's comma-separated names, dropping blanks."""
    return [name for name in (n.strip() for n in names.split(",")) if name]


def _ids(field: str) -> list[str]:
    """Split a comma-separated IMDb field; ``\\N`` is IMDb's empty."""
    return [] if field == r"\N" else field.split(",")


def _names(ids: Iterable[str], names: Mapping[str, str]) -> list[str]:
    return [names[i] for i in ids if i in names]
