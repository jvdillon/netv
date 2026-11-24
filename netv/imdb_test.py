"""Tests for imdb.py - IMDb facts joined to provider movies and series."""

from __future__ import annotations

from email.message import Message
from pathlib import Path
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
    networks = [{"imdb": {"value": i}, "network": {"value": n}} for i, n in _NETWORKS]
    on_air = json.dumps({"results": {"bindings": networks}}).encode()
    web.routes[imdb._networks_query_url(_SPARQL)] = lambda: testing.FakeResponse(on_air)
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
        # Wikidata names no network for this one; TMDB, given a key, does.
        "tt10\ttvSeries\tDuster\tDuster\t0\t2025\t\\N\t60\tCrime",
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
        "tconst\tordering\tnconst\tcategory\tjob\tcharacters",
        'tt1\t2\tnm3\tactor\t\\N\t["Neil McCauley"]',
        'tt1\t1\tnm2\tactor\t\\N\t["Vincent Hanna"]',
        "tt1\t3\tnm1\tdirector\t\\N\t\\N",
        'tt1\t4\tnm6\tarchive_footage\t\\N\t["Old Clip"]',
        "tt1\t5\tnm5\tcomposer\t\\N\t\\N",
        "tt4\t1\tnm4\tactress\t\\N\t\\N",
        "tt5\t1\tnm4\tactress\t\\N\t\\N",
        # IMDb credits a series' composers on its episodes, and its cast there too.
        "tt40\t1\tnm8\tcomposer\t\\N\t\\N",
        "tt41\t1\tnm8\tcomposer\t\\N\t\\N",
        "tt41\t2\tnm9\tcomposer\t\\N\t\\N",
        'tt41\t3\tnm2\tactor\t\\N\t["Guest"]',
        # An episode of a series not in the catalog.
        "tt70\t1\tnm5\tcomposer\t\\N\t\\N",
    ],
    "title.episode": [
        "tconst\tparentTconst\tseasonNumber\tepisodeNumber",
        "tt40\ttt4\t1\t1",
        "tt41\ttt4\t1\t2",
        "tt70\ttt7\t1\t1",
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
        "nm8\tYann Tiersen",
        "nm9\tSecond Composer",
    ],
}
# Wikidata's networks by IMDb id, a short name where it has one.
_NETWORKS = [
    ("tt4", "TF1"),
    ("tt4", "Netflix"),
    # One service under its rebrands is one network.
    ("tt4", "HBO Max"),
    ("tt4", "HBO"),
    ("tt4", "Max"),
    ("tt1", "Ignored Film Network"),
    ("tt99", "Gone"),
]

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
    "characters": ["Vincent Hanna", "Neil McCauley"],
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


def test_same_title_prefers_the_tmdb_id_then_title_and_year_then_name():
    by_id: ProviderRecord = {"stream_id": 1, "name": "Heat (1995)", "tmdb": "949"}
    reupload: ProviderRecord = {"stream_id": 2, "name": "Heat - 1995", "tmdb": 949}
    assert imdb.same_title(by_id) == imdb.same_title(reupload) == "movies:tmdb:949"
    # Without an id, spelling variants of one title and year still agree.
    spelled: list[ProviderRecord] = [
        {"stream_id": n, "name": name} for n, name in enumerate(["Heat (1995)", "HEAT - 1995"])
    ]
    assert imdb.same_title(spelled[0]) == imdb.same_title(spelled[1]) == "movies:heat|1995"
    # A zero id is a provider's "unknown", not one title every such record shares.
    zero: ProviderRecord = {"series_id": 3, "name": "Unknown Show (2020)", "tmdb": "0"}
    assert imdb.same_title(zero) == "series:unknownshow|2020"
    assert imdb.same_title({**zero, "tmdb": "1"}) == "series:tmdb:1"
    yearless: ProviderRecord = {"stream_id": 4, "name": "Some Film"}
    assert imdb.same_title(yearless) == "movies:name:some film"
    # A movie and a series of one name are different titles.
    assert imdb.same_title({"series_id": 4, "name": "Some Film"}) == "series:name:some film"


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
    # Search finds a title by the characters its cast play, as by the people.
    assert imdb.matching_keys(lambda name: name == "Vincent Hanna") == {"movies:heat|1995"}
    # Roles limit which credits count.
    hanna = imdb.matching_keys(lambda name: name == "Vincent Hanna", roles=("character",))
    assert hanna == {"movies:heat|1995"}
    assert (
        imdb.matching_keys(lambda name: name == "Vincent Hanna", roles=("cast", "director"))
        == set()
    )
    assert imdb.matching_keys(lambda name: name == "Al Pacino", roles=("cast", "director")) == {
        "movies:heat|1995",
    }
    # A series dates by air date; a one-year difference still matches a movie.
    found_show = imdb.info(show)
    assert found_show is not None and found_show["imdb_id"] == "tt4"
    # Its composers come from its episodes, most credited first; its networks from
    # Wikidata. A guest actor in one episode is not the series' cast.
    assert found_show["crew"] == {
        "composer": ["Yann Tiersen", "Second Composer"],
        "network": ["TF1", "Netflix", "HBO"],
    }
    assert found_show["cast"] == ["Audrey Tautou"]
    assert imdb.credited(show, "composer") == ["Yann Tiersen", "Second Composer"]
    assert imdb.credited(show, "network") == ["TF1", "Netflix", "HBO"]
    assert imdb.credited(heat, "writer") == ["Anne Writer"]
    assert imdb.credited(heat, "director") == ["Michael Mann"]
    # Only series have networks, whatever Wikidata says of a film.
    assert imdb.credited(heat, "network") == []
    assert imdb.credited({"stream_id": 99, "name": "Unknown - 1900"}, "writer") == []
    netflix = imdb.matching_keys(lambda name: name == "Netflix", roles=("network",))
    assert netflix == set(imdb.resolved_keys([show]))
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
    for url in (imdb._tmdb_query_url(_SPARQL), imdb._networks_query_url(_SPARQL)):
        web.routes[url] = error.HTTPError(_SPARQL, 429, "slow", Message(), None)
    remake: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995", "tmdb": "500"}
    show: ProviderRecord = {"series_id": 4, "name": "Amelie and Co", "releaseDate": "2001-05-01"}
    with caplog.at_level(logging.WARNING, logger="netv.imdb"):
        assert imdb.build([remake, show], base_url=_BASE, sparql_url=_SPARQL) == 2
    found = imdb.info(remake)
    assert found is not None and found["imdb_id"] == "tt1"
    assert "joining by title" in caplog.text
    # Without Wikidata a series has no network, and the rest of the build stands.
    assert imdb.credited(show, "network") == []
    assert imdb.credited(show, "composer") == ["Yann Tiersen", "Second Composer"]
    assert "networks unavailable" in caplog.text


_TMDB_API = "https://tmdb.test/3"


def test_build_asks_tmdb_only_for_networks_wikidata_lacks(web: testing.FakeWeb):
    duster: ProviderRecord = {
        "series_id": 9,
        "name": "Duster",
        "tmdb": "113959",
        "releaseDate": "2025-05-15",
    }
    # Wikidata has this one's networks, so TMDB is not asked about it.
    show: ProviderRecord = {"series_id": 4, "name": "Renamed", "tmdb": "77"}
    film: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995", "tmdb": "500"}
    records = [duster, show, film]
    # Without a key, TMDB is never asked: users need not set one up.
    assert imdb.build(records, base_url=_BASE, sparql_url=_SPARQL, tmdb_api=_TMDB_API) == 3
    assert imdb.credited(duster, "network") == []
    assert not [f for f in web.fetches if f.url.startswith(_TMDB_API)]

    web.routes[f"{_TMDB_API}/tv/113959"] = lambda: testing.FakeResponse(
        json.dumps({"id": 113_959, "networks": [{"id": 3186, "name": "Max"}]}).encode(),
    )
    imdb.build(records, base_url=_BASE, sparql_url=_SPARQL, tmdb_api=_TMDB_API, tmdb_token="t.o.k")
    # A rebrand merges as Wikidata's do: Max is HBO.
    assert imdb.credited(duster, "network") == ["HBO"]
    assert imdb.credited(show, "network") == ["TF1", "Netflix", "HBO"]
    asked = [f for f in web.fetches if f.url.startswith(_TMDB_API)]
    assert [(f.url, f.headers.get("Authorization")) for f in asked] == [
        (f"{_TMDB_API}/tv/113959", "Bearer t.o.k"),
    ]


def test_build_reads_a_v3_tmdb_key_and_survives_tmdb_failing(
    web: testing.FakeWeb,
    caplog: pytest.LogCaptureFixture,
):
    duster: ProviderRecord = {
        "series_id": 9,
        "name": "Duster",
        "tmdb": "113959",
        "releaseDate": "2025-05-15",
    }
    # A short v3 key goes in the query; TMDB refusing it costs only the networks.
    web.routes[f"{_TMDB_API}/tv/113959?api_key=abc123"] = error.HTTPError(
        _TMDB_API,
        401,
        "Unauthorized",
        Message(),
        None,
    )
    with caplog.at_level(logging.WARNING, logger="netv.imdb"):
        assert (
            imdb.build(
                [duster],
                base_url=_BASE,
                sparql_url=_SPARQL,
                tmdb_api=_TMDB_API,
                tmdb_token="abc123",
            )
            == 1
        )
    assert imdb.credited(duster, "network") == []
    assert "TMDB networks unavailable" in caplog.text


def test_build_skips_a_series_tmdb_no_longer_lists(web: testing.FakeWeb):
    duster: ProviderRecord = {
        "series_id": 9,
        "name": "Duster",
        "tmdb": "113959",
        "releaseDate": "2025-05-15",
    }
    gone: ProviderRecord = {
        "series_id": 8,
        "name": "Amelie and Co",
        "tmdb": "404404",
        "releaseDate": "2001-05-01",
    }
    web.routes[imdb._networks_query_url(_SPARQL)] = lambda: testing.FakeResponse(
        b'{"results": {"bindings": []}}',
    )
    web.routes[f"{_TMDB_API}/tv/113959"] = lambda: testing.FakeResponse(
        b'{"networks": [{"name": "HBO Max"}, {"name": "Max"}]}',
    )
    web.routes[f"{_TMDB_API}/tv/404404"] = error.HTTPError(
        _TMDB_API,
        404,
        "Not Found",
        Message(),
        None,
    )
    imdb.build(
        [duster, gone],
        base_url=_BASE,
        sparql_url=_SPARQL,
        tmdb_api=_TMDB_API,
        tmdb_token="t.o.k",
    )
    # One missing entry costs only its own networks.
    assert imdb.credited(duster, "network") == ["HBO"]
    assert imdb.credited(gone, "network") == []


def test_build_retries_tmdb_and_a_series_that_keeps_failing_loses_only_its_own(
    web: testing.FakeWeb,
):
    duster: ProviderRecord = {
        "series_id": 9,
        "name": "Duster",
        "tmdb": "113959",
        "releaseDate": "2025-05-15",
    }
    flaky: ProviderRecord = {
        "series_id": 8,
        "name": "Amelie and Co",
        "tmdb": "55",
        "releaseDate": "2001-05-01",
    }
    slowed: ProviderRecord = {"series_id": 7, "name": "Renamed", "tmdb": "77"}
    web.routes[imdb._networks_query_url(_SPARQL)] = lambda: testing.FakeResponse(
        b'{"results": {"bindings": []}}',
    )
    calls: list[int] = []

    def reset_once() -> testing.FakeResponse:
        # As TMDB did live: a connection reset by peer, once.
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionResetError(104, "Connection reset by peer")
        return testing.FakeResponse(b'{"networks": [{"name": "Max"}]}')

    web.routes[f"{_TMDB_API}/tv/113959"] = reset_once
    web.routes[f"{_TMDB_API}/tv/55"] = ConnectionResetError(104, "Connection reset by peer")
    web.routes[f"{_TMDB_API}/tv/77"] = error.HTTPError(_TMDB_API, 429, "Slow", Message(), None)
    imdb.build(
        [duster, flaky, slowed],
        base_url=_BASE,
        sparql_url=_SPARQL,
        tmdb_api=_TMDB_API,
        tmdb_token="t.o.k",
    )
    assert imdb.credited(duster, "network") == ["HBO"]
    assert len(calls) == 2
    # Three tries each, then each is left without a network; the rest stand.
    assert imdb.credited(flaky, "network") == []
    assert imdb.credited(slowed, "network") == []
    tries = sorted(f.url for f in web.fetches if f.url.endswith(("/tv/55", "/tv/77")))
    assert tries == [f"{_TMDB_API}/tv/55"] * 3 + [f"{_TMDB_API}/tv/77"] * 3


def _locale_answer(rows: list[tuple[str, str | None, str | None]]) -> testing.FakeResponse:
    bindings: list[dict[str, dict[str, str]]] = []
    for imdb_id, country, language in rows:
        binding = {"imdb": {"value": imdb_id}}
        if country is not None:
            binding["country"] = {"value": country}
        if language is not None:
            binding["language"] = {"value": language}
        bindings.append(binding)
    return testing.FakeResponse(json.dumps({"results": {"bindings": bindings}}).encode())


def test_build_adds_each_titles_country_and_language(web: testing.FakeWeb):
    heat: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995"}
    show: ProviderRecord = {"series_id": 4, "name": "Amelie and Co", "releaseDate": "2001-05-01"}
    film: ProviderRecord = {"stream_id": 5, "name": "Amelie and Co - 2002"}
    web.routes[imdb._locale_query_url(_SPARQL, ["tt1", "tt4", "tt5"])] = _locale_answer(
        [
            ("tt1", "United States", "English"),
            ("tt4", "France", "French"),
            ("tt4", "France", "English"),
        ],
    )
    imdb.build([heat, show, film], base_url=_BASE, sparql_url=_SPARQL)
    assert imdb.credited(heat, "country") == ["United States"]
    assert imdb.credited(heat, "language") == ["English"]
    assert imdb.credited(show, "country") == ["France"]
    assert imdb.credited(show, "language") == ["French", "English"]
    assert imdb.credited(film, "country") == []
    france = imdb.matching_keys(lambda name: name == "France", roles=("country",))
    assert france == set(imdb.resolved_keys([show]))


def test_locales_ask_in_batches_and_survive_a_failed_one(
    web: testing.FakeWeb,
    caplog: pytest.LogCaptureFixture,
):
    web.routes[imdb._locale_query_url(_SPARQL, ["tt1", "tt2"])] = _locale_answer(
        [("tt1", "Japan", None), ("tt2", None, "Korean")],
    )
    web.routes[imdb._locale_query_url(_SPARQL, ["tt3"])] = error.HTTPError(
        _SPARQL,
        504,
        "Gateway Timeout",
        Message(),
        None,
    )
    with caplog.at_level(logging.WARNING, logger="netv.imdb"):
        found = imdb._locales(_SPARQL, {"tt3", "tt2", "tt1"}, batch=2)
    # A batch that fails costs only its own titles.
    assert found == {"tt1": {"country": ["Japan"]}, "tt2": {"language": ["Korean"]}}
    assert "Wikidata countries and languages unavailable" in caplog.text
    query = parse.unquote(imdb._locale_query_url(_SPARQL, ["tt1"]).partition("query=")[2])
    for clause in ('VALUES ?imdb { "tt1" }', "wdt:P495 ?c", "wdt:P364 ?l"):
        assert clause in query, clause


def test_tmdb_query_names_movie_and_tv_ids():
    query = parse.unquote(imdb._tmdb_query_url(_SPARQL).partition("query=")[2])
    assert imdb._tmdb_query_url(_SPARQL).startswith(f"{_SPARQL}?format=json&query=")
    for clause in (
        'wdt:P4947 ?tmdb BIND("movies"',
        'wdt:P4983 ?tmdb BIND("series"',
        "wdt:P345 ?imdb",
    ):
        assert clause in query


def test_networks_query_asks_for_each_series_broadcaster_by_its_short_name():
    url = imdb._networks_query_url(_SPARQL)
    assert url.startswith(f"{_SPARQL}?format=json&query=")
    query = parse.unquote(url.partition("query=")[2])
    for clause in (
        "wdt:P4983 []",  # TV series only.
        "wdt:P345 ?imdb",
        "wdt:P449 ?n",  # Original broadcaster.
        'wdt:P1813 ?short FILTER(LANG(?short) = "en")',  # "ABC", not its full name.
        'rdfs:label ?label FILTER(LANG(?label) = "en")',
        "COALESCE(?short, ?label)",
    ):
        assert clause in query, clause


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
    imdb._write({key: _HEAT}, set())
    assert imdb.info(heat) == _HEAT
    monkeypatch.setattr(imdb.time, "time", lambda: 2000.0)
    imdb._write({key: {**_HEAT, "votes": 1, "cast": ["Val Kilmer"], "crew": {}}}, set())
    found = imdb.info(heat)
    assert found is not None and found["votes"] == 1 and found["cast"] == ["Val Kilmer"]
    assert imdb.matching_keys(lambda name: "Pacino" in name) == set()
    # Only the latest build's age counts.
    assert not imdb.is_stale(max_age_sec=0)


def test_rankings_give_imdb_rating_votes_and_year():
    imdb._write(
        {
            "movies:tmdb:9": {**_HEAT, "rating": 8.3, "votes": 700_000, "year": 1995},
            "movies:ronin|1998": {**_HEAT, "rating": 7.2, "votes": 9, "year": 1998, "genres": []},
            # Credits saved from a provider carry no rating, year, or genre.
            "movies:tmdb:7": {**_HEAT, "rating": 0.0, "votes": 0, "year": 0, "genres": []},
        },
        set(),
    )
    records: list[ProviderRecord] = [
        # IMDb's year wins over the provider's.
        {"stream_id": 1, "name": "Heat (1996)", "tmdb": "9"},
        {"stream_id": 2, "name": "Ronin - 1998"},
        {"stream_id": 6, "name": "Seven - 1995", "tmdb": "7"},
        # IMDb lacks these; one still has its own year, from its name or air date.
        {"stream_id": 3, "name": "Unknown - 2004"},
        {"series_id": 4, "name": "Show", "releaseDate": "2011-05-01"},
        {"stream_id": 5, "name": "No Year"},
    ]
    assert imdb.rankings(records) == [
        imdb.Ranking(rating=8.3, votes=700_000, year=1995, genres=("Action", "Crime")),
        imdb.Ranking(rating=7.2, votes=9, year=1998),
        imdb.Ranking(rating=0.0, votes=0, year=1995),
        imdb.Ranking(rating=0.0, votes=0, year=2004),
        imdb.Ranking(rating=0.0, votes=0, year=2011),
        imdb.Ranking(rating=0.0, votes=0, year=0),
    ]
    assert imdb.rankings([]) == []


def test_rankings_follow_a_rebuild_and_the_open_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    heat: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995", "tmdb": "9"}
    monkeypatch.setattr(imdb.time, "time", lambda: 1000.0)
    imdb._write({"movies:tmdb:9": {**_HEAT, "rating": 8.3}}, set())
    assert imdb.rankings([heat])[0].rating == 8.3
    # A rebuild, at a later time, is read again.
    monkeypatch.setattr(imdb.time, "time", lambda: 2000.0)
    imdb._write({"movies:tmdb:9": {**_HEAT, "rating": 5.0}}, set())
    assert imdb.rankings([heat])[0].rating == 5.0
    # So is another database, even one built at the same time.
    db.open_database(tmp_path / "other.db")
    imdb.init()
    imdb._write({"movies:tmdb:9": {**_HEAT, "rating": 6.5}}, set())
    assert imdb.rankings([heat])[0].rating == 6.5


def test_matching_and_resolved_keys():
    imdb._write(
        {
            "movies:tmdb:9": {**_HEAT, "cast": ["Al Pacino"], "crew": {}},
            "movies:heat|1995": {**_HEAT, "imdb_id": "tt2", "cast": ["Val Kilmer"], "crew": {}},
            "movies:ronin|1998": {**_HEAT, "cast": [], "writers": [], "directors": [], "crew": {}},
        },
        set(),
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


def test_billing_keeps_the_top_ten_wanted_cast_and_their_characters(web: testing.FakeWeb):
    rows = ["tconst\tordering\tnconst\tcategory\tjob\tcharacters"]
    rows += [
        f'tt1\t{i}\tnm{i}\t{("actor", "actress", "self")[i % 3]}\t\\N\t["Role {i}"]'
        for i in range(12, 0, -1)
    ]
    rows += [
        # One actor playing two characters, a character two actors share, none listed.
        'tt1\t15\tnm1\tactor\t\\N\t["Role 1","Twin"]',
        'tt1\t16\tnm2\tactor\t\\N\t["Twin"]',
        "tt1\t13\tnm13\twriter\t\\N\t\\N",
        'tt1\t14\tnm14\tarchive_sound\t\\N\t["Old Clip"]',
        "tt2\t1\tnm15\tproducer\t\\N\t\\N",
        "tt3\t1\tnm16\tactor\t\\N\t\\N",
    ]
    web.routes[f"{_BASE}/title.principals.tsv.gz"] = testing.FakeResponse(
        gzip.compress("\n".join(rows).encode() + b"\n"),
    )
    billing = imdb._billing(_BASE, {"tt1": [], "tt3": []}, {})
    assert billing.cast == {"tt1": [f"nm{i}" for i in range(1, 11)], "tt3": ["nm16"]}
    # The characters the top-billed cast play, each once, in billing order.
    assert billing.characters == {
        "tt1": [*(f"Role {i}" for i in range(1, 11)), "Twin"],
        "tt3": [],
    }
    assert billing.roles == {}


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


def test_rows_read_everything_given_none_and_nothing_given_an_empty_keep(web: testing.FakeWeb):
    url = f"{_BASE}/t.tsv.gz"
    body = gzip.compress(b"id\tv\na\t1\nb\t2\n")
    web.routes[url] = lambda: testing.FakeResponse(body)
    assert list(imdb._rows(url, None)) == [["a", "1"], ["b", "2"]]
    assert list(imdb._rows(url)) == [["a", "1"], ["b", "2"]]
    assert list(imdb._rows(url, {"b"})) == [["b", "2"]]
    web.fetches.clear()
    # Nothing wanted is not everything wanted; the file is not even opened.
    assert list(imdb._rows(url, set())) == []
    assert list(imdb._rows(url, iter(()))) == []
    assert web.fetches == []


def test_crew_billing_and_episodes_read_nothing_for_no_titles(web: testing.FakeWeb):
    assert imdb._crew(_BASE, {}) == {}
    assert imdb._episodes(_BASE, set()) == {}
    billing = imdb._billing(_BASE, {}, {})
    assert (billing.cast, billing.characters, billing.roles) == ({}, {}, {})
    assert web.fetches == []


def test_build_of_titles_imdb_lacks_writes_an_empty_index_without_reading_the_big_files(
    web: testing.FakeWeb,
):
    records: list[ProviderRecord] = [
        {"stream_id": 6, "name": "Unknown - 2000"},
        {"series_id": 8, "name": "Undated", "release_date": ""},
    ]
    assert imdb.build(records, base_url=_BASE, sparql_url=_SPARQL) == 0
    assert not imdb.is_stale(max_age_sec=60)
    assert imdb.info(records[0]) is None
    big = {f"{_BASE}/{name}.tsv.gz" for name in ("title.crew", "title.episode", "title.principals")}
    big.add(f"{_BASE}/name.basics.tsv.gz")
    assert big.isdisjoint(f.url for f in web.fetches)


def test_credits_prefer_imdb_and_index_the_providers():
    imdb._write({"movies:tmdb:9": _HEAT}, set())
    heat: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995", "tmdb": "9"}
    assert imdb.credits(heat, directors="Someone", cast="Else") == (
        ["Michael Mann"],
        ["Al Pacino", "Robert De Niro"],
    )
    assert imdb.matching_keys(lambda name: name == "Else") == set()

    film: ProviderRecord = {"stream_id": 2, "name": "Film", "tmdb": "5"}
    assert imdb.credits(film, directors="Ann Dir, ", cast=" Cy Star,Di Lead") == (
        ["Ann Dir"],
        ["Cy Star", "Di Lead"],
    )
    assert imdb.matching_keys(lambda name: name in {"Ann Dir", "Di Lead"}) == {"movies:tmdb:5"}
    found = imdb.info(film)
    assert found is not None and found["imdb_id"] == "" and found["title"] == "Film"

    # A record with no key cannot be indexed; its names are still shown.
    nameless: ProviderRecord = {"stream_id": 3, "name": "No year"}
    assert imdb.credits(nameless, directors="", cast="Solo") == ([], ["Solo"])
    assert imdb.matching_keys(lambda name: name == "Solo") == set()
    # Nor is a record whose provider credits nothing.
    blank: ProviderRecord = {"stream_id": 4, "name": "Blank", "tmdb": "4"}
    assert imdb.credits(blank, directors="", cast="") == ([], [])
    assert imdb.info(blank) is None


def test_saving_credits_again_replaces_only_that_title():
    film: ProviderRecord = {"stream_id": 2, "name": "Film", "tmdb": "5"}
    other: ProviderRecord = {"stream_id": 3, "name": "Other", "tmdb": "6"}
    imdb.credits(film, directors="", cast="Shared, Cy Star")
    imdb.credits(other, directors="", cast="Shared")
    imdb._add("movies:tmdb:5", "Film", [], ["Di Lead"])
    assert imdb.matching_keys(lambda name: name == "Shared") == {"movies:tmdb:6"}
    assert imdb.matching_keys(lambda name: name == "Di Lead") == {"movies:tmdb:5"}
    found = imdb.info(film)
    assert found is not None and found["cast"] == ["Di Lead"]


def test_a_build_keeps_saved_credits_until_imdb_has_the_title(web: testing.FakeWeb):
    del web
    film: ProviderRecord = {"stream_id": 2, "name": "Film", "tmdb": "5"}
    heat: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995", "tmdb": "1"}
    imdb.credits(film, directors="", cast="Cy Star")
    imdb.credits(heat, directors="", cast="Stale Name")
    assert imdb.build([film, heat], base_url=_BASE, sparql_url=_SPARQL) == 1
    # Heat now joins by title, so its saved entry under the TMDB key is dropped.
    found = imdb.info(heat)
    assert found is not None and found["imdb_id"] == "tt1"
    assert imdb.matching_keys(lambda name: name == "Stale Name") == set()
    assert imdb.matching_keys(lambda name: name == "Cy Star") == {"movies:tmdb:5"}


def test_retried_retries_dropped_connections(caplog: pytest.LogCaptureFixture):
    calls: list[int] = []

    def flaky(n: int) -> int:
        calls.append(n)
        if len(calls) < 3:
            raise ConnectionResetError("reset")
        return n * 2

    with caplog.at_level(logging.WARNING, logger="netv.imdb"):
        assert imdb._retried(flaky)(4) == 8
    assert calls == [4, 4, 4]
    assert caplog.text.count("retrying") == 2

    def broken() -> int:
        raise ConnectionResetError("reset")

    with pytest.raises(ConnectionResetError):
        imdb._retried(broken)()


def test_is_stale_by_age(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(imdb.time, "time", lambda: 1000.0)
    imdb._write({}, set())
    monkeypatch.setattr(imdb.time, "time", lambda: 1060.0)
    assert not imdb.is_stale(max_age_sec=60)
    assert imdb.is_stale(max_age_sec=59)


def test_an_index_built_before_characters_is_stale_until_rebuilt():
    # As a database from before characters were indexed holds it.
    conn = db.connection()
    conn.execute("DROP TABLE imdb_builds")
    conn.execute("CREATE TABLE imdb_builds (built_ts REAL NOT NULL)")
    conn.execute("INSERT INTO imdb_builds VALUES (1e12)")
    conn.commit()
    imdb.init()
    imdb.init()
    assert imdb.is_stale(max_age_sec=float("inf"))
    # Nor does one from before series composers and networks (format 2), before
    # rebranded networks were merged (3), before TMDB lookups were retried (4), or
    # before countries and languages (5).
    conn.execute("INSERT INTO imdb_builds VALUES (1e12, 2)")
    conn.execute("INSERT INTO imdb_builds VALUES (1e12, 3)")
    conn.execute("INSERT INTO imdb_builds VALUES (1e12, 4)")
    conn.execute("INSERT INTO imdb_builds VALUES (1e12, 5)")
    conn.commit()
    assert imdb.is_stale(max_age_sec=float("inf"))
    imdb._write({}, set())
    assert not imdb.is_stale(max_age_sec=float("inf"))


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
