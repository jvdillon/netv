"""Tests for imdb.py - IMDb facts joined to provider movies and series."""

from __future__ import annotations

from pathlib import Path

import gzip
import os
import time

import pytest

from netv import imdb, testing, util
from netv.xtream import ProviderRecord


_BASE = "https://imdb.test"


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> testing.FakeWeb:
    # Threads share the test's fake internet; spawned processes would not.
    monkeypatch.setattr(imdb, "ProcessPoolExecutor", testing.thread_pool)
    web = testing.FakeWeb()
    testing.install_web(monkeypatch, web, util)
    for name, lines in _DATASETS.items():
        body = gzip.compress("\n".join(lines).encode() + b"\n")
        web.routes[f"{_BASE}/{name}.tsv.gz"] = lambda body=body: testing.FakeResponse(body)
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
    ],
    "title.ratings": ["tconst\taverageRating\tnumVotes", "tt1\t8.3\t700000", "tt3\t5\t9"],
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


def test_build_joins_movies_and_series_by_title_and_year(tmp_path: Path, web: testing.FakeWeb):
    index = tmp_path / "imdb.json"
    heat: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995"}
    show: ProviderRecord = {"series_id": 4, "name": "Amelie and Co", "releaseDate": "2001-05-01"}
    film: ProviderRecord = {"stream_id": 5, "name": "Amelie and Co - 2002"}
    unmatched: list[ProviderRecord] = [
        {"stream_id": 6, "name": "Unknown - 2000"},
        {"stream_id": 7, "name": "No year"},
        {"series_id": 8, "name": "Undated", "release_date": ""},
    ]

    assert imdb.build(index, [heat, show, film, *unmatched], base_url=_BASE) == 3
    assert {f.timeout for f in web.fetches} == {60}
    # The exact-year movie with the most votes wins over a video, a remake, and an episode.
    assert imdb.info(index, heat) == _HEAT
    # A series dates by air date; a one-year difference still matches a movie.
    found_show = imdb.info(index, show)
    assert found_show is not None and found_show["imdb_id"] == "tt4"
    found_film = imdb.info(index, film)
    assert found_film is not None and found_film["imdb_id"] == "tt5"
    assert found_film["runtime_min"] == 0 and found_film["genres"] == []
    assert found_film["rating"] == 0.0 and found_film["directors"] == []
    assert [imdb.info(index, r) for r in unmatched] == [None, None, None]


def test_people_lists_cast_first():
    assert imdb.people(_HEAT) == [
        "Al Pacino",
        "Robert De Niro",
        "Michael Mann",
        "Anne Writer",
        "Elliot Goldenthal",
    ]


@pytest.mark.parametrize("text", ['{"heat|1995": {"director": "Michael Mann"}}', '{"trunc'])
def test_an_unreadable_index_is_absent_and_stale(tmp_path: Path, text: str):
    index = tmp_path / "imdb.json"
    index.write_text(text)
    assert imdb.info(index, {"stream_id": 1, "name": "Heat - 1995"}) is None
    assert imdb.is_stale(index, max_age_sec=60)


def test_a_repaired_index_is_fresh_again(tmp_path: Path):
    index = tmp_path / "imdb.json"
    index.write_text("{")
    os.utime(index, (time.time(), 1000))
    assert imdb.is_stale(index, max_age_sec=float("inf"))
    util.atomic_write_json(index, {})
    os.utime(index, (time.time(), 2000))
    assert not imdb.is_stale(index, max_age_sec=float("inf"))


def test_info_without_an_index_is_none(tmp_path: Path):
    assert imdb.info(tmp_path / "missing.json", {"stream_id": 1, "name": "Heat - 1995"}) is None


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


def test_info_rereads_an_index_rebuilt_elsewhere(tmp_path: Path):
    index = tmp_path / "imdb.json"
    heat: ProviderRecord = {"stream_id": 1, "name": "Heat - 1995"}
    key = f"movies:{imdb.title_key('Heat', 1995)}"
    util.atomic_write_json(index, {key: _HEAT})
    os.utime(index, (1000, 1000))
    assert imdb.info(index, heat) == _HEAT
    util.atomic_write_json(index, {key: {**_HEAT, "votes": 1}})
    os.utime(index, (2000, 2000))
    found = imdb.info(index, heat)
    assert found is not None and found["votes"] == 1


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


def test_is_stale_by_age(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    index = tmp_path / "imdb.json"
    assert imdb.is_stale(index, max_age_sec=60)
    index.write_text("{}")
    os.utime(index, (1000, 1000))
    monkeypatch.setattr(imdb.time, "time", lambda: 1060.0)
    assert not imdb.is_stale(index, max_age_sec=60)
    assert imdb.is_stale(index, max_age_sec=59)


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
