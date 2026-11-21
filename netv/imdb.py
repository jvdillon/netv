"""IMDb facts for provider movies and series, joined by title and year.

Xtream catalogs carry little beyond a name: movie lists have no people at all, and
series lists only sometimes. IMDb's non-commercial datasets have ids, genres,
runtimes, ratings, and full credits. Only titles in the catalog are kept, which
leaves a small JSON index next to the other caches.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from concurrent.futures import ProcessPoolExecutor
from typing import TypedDict, cast
from urllib import parse

import functools
import gzip
import logging
import multiprocessing
import pathlib
import re
import time
import unicodedata

from treekle import json_loads

from netv.util import atomic_write_json, safe_urlopen
from netv.xtream import ProviderRecord, text_field


log = logging.getLogger(__name__)


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


def is_stale(path: pathlib.Path, *, max_age_sec: float) -> bool:
    """Whether the index at ``path`` is missing, unreadable, or older than ``max_age_sec``."""
    if not path.exists():
        return True
    mtime = path.stat().st_mtime
    return time.time() - mtime > max_age_sec or _index(path, mtime) is None


def info(path: pathlib.Path, record: ProviderRecord) -> Info | None:
    """Return what the index at ``path`` knows of a catalog movie or series.

    Args:
        path: JSON index ``build`` wrote.
        record: Provider movie or series.

    Returns:
        found: IMDb's facts, or None for a title the index lacks.
    """
    if not path.exists():
        return None
    index = _index(path, path.stat().st_mtime)
    if index is None:
        return None
    by_id, by_title = _catalog_keys(record)
    return next((index[k] for k in (by_id, by_title) if k is not None and k in index), None)


def people(found: Info) -> list[str]:
    """Return everyone credited on a title, cast first."""
    return [*found["cast"], *found["directors"], *found["writers"]] + [
        name for names in found["crew"].values() for name in names
    ]


def build(
    path: pathlib.Path,
    records: Sequence[ProviderRecord],
    *,
    base_url: str = "https://datasets.imdbws.com",
    sparql_url: str = "https://query.wikidata.org/sparql",
) -> int:
    """Write what IMDb knows of every catalog movie and series to ``path``.

    A record joins by its TMDB id, which Wikidata maps to an IMDb id. One without
    a mapping joins by title and year: a provider year may differ from IMDb's by
    one, so neighbors match too, and more votes, then an exact year, wins.
    Parsing is CPU-bound, so files that do not depend on each other are read in
    parallel processes.

    Args:
        path: JSON index to replace.
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
            if key in chosen
        }
        kept = {chosen[k]: basics[chosen[k]] for k in needed}
        crew = executor.submit(_crew, base_url, kept)
        cast_ids, roles = executor.submit(_billing, base_url, kept).result()
        infos = _infos(base_url, kept, ratings, crew.result(), (cast_ids, roles))
    index = {key: infos[chosen[key]] for key in needed}
    atomic_write_json(path, index)
    _index.cache_clear()
    log.info("IMDb index: %d catalog titles", len(index))
    return len(index)


def _catalog_keys(record: ProviderRecord) -> tuple[str | None, str | None]:
    """The record's index keys: by TMDB id, and by title and year."""
    kind = "series" if "series_id" in record else "movies"
    tmdb = text_field(record, "tmdb")
    found = _catalog_title(record)
    return (
        f"{kind}:tmdb:{tmdb}" if tmdb.isdigit() else None,
        None if found is None else f"{kind}:{title_key(found[1], found[2])}",
    )


def _catalog_title(record: ProviderRecord) -> tuple[str, str, int] | None:
    """The kind, title, and year of a catalog record; series fall back to air date."""
    kind = "series" if "series_id" in record else "movies"
    name = text_field(record, "name")
    parsed = title_year(name)
    if parsed is not None:
        return kind, *parsed
    aired = text_field(record, "releaseDate") or text_field(record, "release_date")
    return (kind, name, int(aired[:4])) if aired[:4].isdigit() else None


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
        with safe_urlopen(_tmdb_query_url(sparql_url), timeout=120, user_agent="neTV") as resp:
            rows = json_loads(resp.read(), _SparqlResult)["results"]["bindings"]
    except (OSError, ValueError) as exc:  # URLError is an OSError; ReadError a ValueError.
        log.warning("Wikidata TMDB ids unavailable, joining by title: %s", exc)
        return {}
    return {
        f"{row['kind']['value']}:tmdb:{row['tmdb']['value']}": row["imdb"]["value"] for row in rows
    }


def _ratings(base_url: str) -> dict[str, tuple[float, int]]:
    """Every rated title's average and vote count."""
    return {
        row[0]: (float(row[1]), int(row[2])) for row in _rows(f"{base_url}/title.ratings.tsv.gz")
    }


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


def _ids(field: str) -> list[str]:
    """Split a comma-separated IMDb field; ``\\N`` is IMDb's empty."""
    return [] if field == r"\N" else field.split(",")


def _names(ids: Iterable[str], names: Mapping[str, str]) -> list[str]:
    return [names[i] for i in ids if i in names]


# ``mtime`` keys the cache so a rebuilt file is reread.
@functools.lru_cache(maxsize=1)
def _index(path: pathlib.Path, mtime: float) -> dict[str, Info] | None:
    """Load the index, or None when another release or a crash left it unreadable."""
    del mtime
    try:
        return json_loads(path.read_text(), dict[str, Info])
    except ValueError:  # ReadError is a ValueError; so is truncated JSON.
        return None
