"""Tests for imdb.py - IMDb facts joined to provider movies and series."""

from __future__ import annotations

from email.message import Message
from urllib import error, parse

import gzip
import json
import logging

import pytest

from netv import db, imdb, testing, util
from netv.xtream import ProviderRecord


_BASE = "https://imdb.test"
_SPARQL = "https://wikidata.test/sparql"
_TMDB = {
    ("movies", "500"): "tt2",
    ("series", "77"): "tt4",
    ("movies", "600"): "tt99",
    ("movies", "700"): "tt6",
}


@pytest.fixture(autouse=True)
def index(guide_db: None) -> None:
    """The index tables, in the per-test database."""
    del guide_db
    imdb.init()


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> testing.FakeWeb:
    # Threads share the test's fake internet; spawned processes would not.
    monkeypatch.setattr(imdb, "ProcessPoolExecutor", testing.thread_pool)
    web = testing.FakeWeb()
    testing.install_web(monkeypatch, web, util)
    for name, lines in _DATASETS.items():
        body = gzip.compress("\n".join(lines).encode() + b"\n")
        web.routes[f"{_BASE}/{name}.tsv.gz"] = lambda body=body: testing.FakeResponse(body)
    bindings = [
        {"kind": {"value": k}, "tmdb": {"value": t}, "imdb": {"value": i}}
        for (k, t), i in _TMDB.items()
    ]
    answer = json.dumps({"results": {"bindings": bindings}}).encode()
    web.routes[imdb._tmdb_query_url(_SPARQL)] = lambda: testing.FakeResponse(answer)
    return web


_DATASETS = {
    "title.basics": [
        "tconst\ttitleType\tprimaryTitle\toriginalTitle\tisAdult\tstartYear\tendYear\truntimeMinutes\tgenres",
        "tt1\tmovie\tHeat\tHeat\t0\t1995\t\\N\t170\tAction,Crime",
        "tt2\tmovie\tHeat\tHeat\t0\t1986\t\\N\t101\tAction",
        "tt3\tvideo\tHeat\tHeat\t0\t1995\t\\N\t\\N\t\\N",
        "tt4\ttvSeries\tAmélie & Co\tAmélie & Co\t0\t2001\t2003\t30\tComedy",
        "tt5\tmovie\tAmélie & Co\tLe Fabuleux\t0\t2001\t\\N\t\\N\t\\N",
        "tt6\tmovie\tNo Year\tNo Year\t0\t\\N\t\\N\t\\N\t\\N",
        "tt7\ttvEpisode\tHeat\tHeat\t0\t1995\t\\N\t\\N\t\\N",
        "tt8\tmovie\tThreehundred\tThreehundred\t0\t2006\t\\N\t\\N\t\\N",
        "tt9\tmovie\tThreehundred\tThreehundred\t0\t2007\t\\N\t\\N\t\\N",
    ],
    "title.ratings": [
        "tconst\taverageRating\tnumVotes",
        "tt1\t8.3\t700000",
        "tt3\t5\t9",
        "tt8\t7.6\t800000",
        "tt9\t3\t3",
    ],
    "title.crew": ["tconst\tdirectors\twriters", "tt1\tnm1\tnm7", "tt5\t\\N\t\\N"],
    "title.principals": [
        "tconst\tordering\tnconst\tcategory",
        "tt1\t2\tnm3\tactor",
        "tt1\t1\tnm2\tactor",
        "tt1\t3\tnm1\tdirector",
        "tt1\t4\tnm6\tarchive_footage",
        "tt1\t5\tnm5\tcomposer",
        "tt4\t1\tnm4\tactress",
        "tt5\t1\tnm4\tactress",
    ],
    "name.basics": [
        "nconst\tprimaryName",
        "nm1\tMichael Mann",
        "nm2\tAl Pacino",
        "nm3\tRobert De Niro",
        "nm4\tAudrey Tautou",
        "nm5\tElliot Goldenthal",
        "nm6\tOld Footage",
        "nm7\tAnne Writer",
    ],
}

_HEAT: imdb.Info = {
    "imdb_id": "tt1",
    "title": "Heat",
    "year": 1995,
    "genres": ["Action", "Crime"],
    "runtime_min": 170,
    "rating": 8.3,
    "votes": 700_000,
    "directors": ["Michael Mann"],
    "writers": ["Anne Writer"],
    "cast": ["Al Pacino", "Robert De Niro"],
    "crew": {"composer": ["Elliot Goldenthal"]},
}


def test_title_year_reads_both_provider_spellings():
    assert imdb.title_year("Heat - 1995") == ("Heat", 1995)
    assert imdb.title_year("Heat (1995)") == ("Heat", 1995)
    assert imdb.title_year("Heat") is None


def test_title_key_ignores_case_accents_and_punctuation():
    assert imdb.title_key("Heat!", 1995) == "heat|1995"
    assert imdb.title_key("Amélie & Co", 2001) == imdb.title_key("amelie and co.", 2001)
    assert imdb.title_key("Heat", 1995) != imdb.title_key("Heat", 1996)


def test_build_joins_movies_and_series_by_title_and_year(web: testing.FakeWeb):
    heat: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995"}
    show: ProviderRecord = {"series_id": 4, "name": "Amelie and Co", "releaseDate": "2001-05-01"}
    film: ProviderRecord = {"stream_id": 5, "name": "Amelie and Co - 2002"}
    unmatched: list[ProviderRecord] = [
        {"stream_id": 6, "name": "Unknown - 2000"},
        {"stream_id": 7, "name": "No year"},
        {"series_id": 8, "name": "Undated", "release_date": ""},
    ]

    records = [heat, show, film, *unmatched]
    assert imdb.build(records, base_url=_BASE, sparql_url=_SPARQL) == 3
    assert {f.timeout for f in web.fetches if f.url.startswith(_BASE)} == {60}
    # The exact-year movie with the most votes wins over a video, a remake, and an episode.
    assert imdb.info(heat) == _HEAT
    # A series dates by air date; a one-year difference still matches a movie.
    found_show = imdb.info(show)
    assert found_show is not None and found_show["imdb_id"] == "tt4"
    found_film = imdb.info(film)
    assert found_film is not None and found_film["imdb_id"] == "tt5"
    assert found_film["runtime_min"] == 0 and found_film["genres"] == []
    assert found_film["rating"] == 0.0 and found_film["directors"] == []
    assert [imdb.info(r) for r in unmatched] == [None, None, None]


def test_build_prefers_the_tmdb_id_then_votes(web: testing.FakeWeb):
    # Wikidata maps this TMDB id to the 1986 Heat, so the title join is not used.
    remake: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995", "tmdb": 500}
    show: ProviderRecord = {"series_id": 4, "name": "Renamed", "tmdb": "77"}
    # An id IMDb no longer lists, and an unmapped id, both join by title.
    gone: ProviderRecord = {"stream_id": 2, "name": "Heat - 1995", "tmdb": "600"}
    unmapped: ProviderRecord = {"stream_id": 3, "name": "Heat - 1995", "tmdb": "1"}
    # A popular film a year off beats an obscure one of the exact year.
    popular: ProviderRecord = {"stream_id": 4, "name": "Threehundred - 2007"}
    undated: ProviderRecord = {"stream_id": 5, "name": "Whatever", "tmdb": "700"}
    records = [remake, show, gone, unmapped, popular, undated]

    assert imdb.build(records, base_url=_BASE, sparql_url=_SPARQL) == 5
    found = [imdb.info(r) for r in records]
    assert [f and f["imdb_id"] for f in found] == ["tt2", "tt4", "tt1", "tt1", "tt8", "tt6"]
    assert found[5] is not None and found[5]["year"] == 0
    stored = {row[0] for row in db.connection().execute("SELECT key FROM imdb_titles")}
    # A record joined by id keeps no title key.
    assert stored == {
        "movies:tmdb:500",
        "series:tmdb:77",
        "movies:heat|1995",
        "movies:threehundred|2007",
        "movies:tmdb:700",
    }


def test_build_joins_by_title_when_wikidata_is_down(
    web: testing.FakeWeb,
    caplog: pytest.LogCaptureFixture,
):
    web.routes[imdb._tmdb_query_url(_SPARQL)] = error.HTTPError(
        _SPARQL,
        429,
        "slow",
        Message(),
        None,
    )
    remake: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995", "tmdb": "500"}
    with caplog.at_level(logging.WARNING, logger="netv.imdb"):
        assert imdb.build([remake], base_url=_BASE, sparql_url=_SPARQL) == 1
    found = imdb.info(remake)
    assert found is not None and found["imdb_id"] == "tt1"
    assert "joining by title" in caplog.text


def test_tmdb_query_names_movie_and_tv_ids():
    query = parse.unquote(imdb._tmdb_query_url(_SPARQL).partition("query=")[2])
    assert imdb._tmdb_query_url(_SPARQL).startswith(f"{_SPARQL}?format=json&query=")
    for clause in (
        'wdt:P4947 ?tmdb BIND("movies"',
        'wdt:P4983 ?tmdb BIND("series"',
        "wdt:P345 ?imdb",
    ):
        assert clause in query


def test_people_lists_cast_first():
    assert imdb.people(_HEAT) == [
        "Al Pacino",
        "Robert De Niro",
        "Michael Mann",
        "Anne Writer",
        "Elliot Goldenthal",
    ]


def test_an_unbuilt_index_is_empty_and_stale():
    assert imdb.info({"stream_id": 1, "name": "Heat - 1995"}) is None
    assert imdb.is_stale(max_age_sec=float("inf"))
    assert imdb.matching_keys(lambda name: True) == set()


def test_a_rebuild_replaces_the_index(monkeypatch: pytest.MonkeyPatch):
    heat: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995"}
    key = f"movies:{imdb.title_key('Heat', 1995)}"
    monkeypatch.setattr(imdb.time, "time", lambda: 1000.0)
    imdb._write({key: _HEAT})
    assert imdb.info(heat) == _HEAT
    monkeypatch.setattr(imdb.time, "time", lambda: 2000.0)
    imdb._write({key: {**_HEAT, "votes": 1, "cast": ["Val Kilmer"], "crew": {}}})
    found = imdb.info(heat)
    assert found is not None and found["votes"] == 1 and found["cast"] == ["Val Kilmer"]
    assert imdb.matching_keys(lambda name: "Pacino" in name) == set()
    # Only the latest build's age counts.
    assert not imdb.is_stale(max_age_sec=0)


def test_matching_and_resolved_keys():
    imdb._write(
        {
            "movies:tmdb:9": {**_HEAT, "cast": ["Al Pacino"], "crew": {}},
            "movies:heat|1995": {**_HEAT, "imdb_id": "tt2", "cast": ["Val Kilmer"], "crew": {}},
            "movies:ronin|1998": {**_HEAT, "cast": [], "writers": [], "directors": [], "crew": {}},
        },
    )
    ronin = imdb.info({"stream_id": 4, "name": "Ronin - 1998"})
    assert ronin is not None and ronin["cast"] == [] and ronin["writers"] == []
    assert imdb.matching_keys(lambda name: "pacino" in name.lower()) == {"movies:tmdb:9"}
    assert imdb.matching_keys(lambda name: "Mann" in name) == {"movies:tmdb:9", "movies:heat|1995"}
    records: list[ProviderRecord] = [
        {"stream_id": 1, "name": "Heat - 1995", "tmdb": "9"},
        {"stream_id": 2, "name": "Heat - 1995", "tmdb": "8"},
        {"stream_id": 3, "name": "Unknown"},
    ]
    # An indexed TMDB id wins over the title; an unindexed one falls back to it.
    assert imdb.resolved_keys(records) == ["movies:tmdb:9", "movies:heat|1995", None]
    found = [imdb.info(r) for r in records]
    assert [f and f["imdb_id"] for f in found] == ["tt1", "tt2", None]


def test_billing_keeps_the_top_ten_wanted_cast(web: testing.FakeWeb):
    rows = ["tconst\tordering\tnconst\tcategory"]
    rows += [f"tt1\t{i}\tnm{i}\t{('actor', 'actress', 'self')[i % 3]}" for i in range(12, 0, -1)]
    rows += [
        "tt1\t15\tnm1\tactor",
        "tt1\t13\tnm13\twriter",
        "tt1\t14\tnm14\tarchive_sound",
        "tt2\t1\tnm15\tproducer",
    ]
    web.routes[f"{_BASE}/title.principals.tsv.gz"] = testing.FakeResponse(
        gzip.compress("\n".join(rows).encode() + b"\n"),
    )
    cast_ids, roles = imdb._billing(_BASE, {"tt1": []})
    assert cast_ids == {"tt1": [f"nm{i}" for i in range(1, 11)]}
    assert roles == {}


def test_crew_keeps_wanted_titles(web: testing.FakeWeb):
    web.routes[f"{_BASE}/title.crew.tsv.gz"] = testing.FakeResponse(
        gzip.compress(b"tconst\tdirectors\twriters\ntt1\tnm1\tnm2\ntt2\tnm3\t\\N\n"),
    )
    assert imdb._crew(_BASE, {"tt1": []}) == {"tt1": (["nm1"], ["nm2"])}


def test_rows_keep_wanted_ids_and_empty_last_fields(web: testing.FakeWeb):
    web.routes[f"{_BASE}/t.tsv.gz"] = testing.FakeResponse(
        gzip.compress(b"id\tv\nb\t1\na\tx\t\nc\tX\nb\t2\n"),
    )
    assert list(imdb._rows(f"{_BASE}/t.tsv.gz", ["a", "c"])) == [["a", "x", ""], ["c", "X"]]


def test_is_stale_by_age(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(imdb.time, "time", lambda: 1000.0)
    imdb._write({})
    monkeypatch.setattr(imdb.time, "time", lambda: 1060.0)
    assert not imdb.is_stale(max_age_sec=60)
    assert imdb.is_stale(max_age_sec=59)


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
