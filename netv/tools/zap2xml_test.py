"""Tests for the zap2it guide scraper."""

from __future__ import annotations

from collections.abc import Iterator
from email import message
from typing import Any
from urllib import error, request

import argparse
import gzip
import json
import os
import pathlib
import runpy
import sys
import time
import xml.etree.ElementTree as xml

import pytest

from netv.tools import zap2xml


SCRIPT = pathlib.Path(zap2xml.__file__).resolve()

CHANNEL = {
    "affiliateName": "AMERICAN BROADCASTING COMPANY",
    "callSign": "KXTVDT",
    "channelNo": "10.1",
    "thumbnail": "//zap2it.tmsimg.com/h3/s28708.png?w=55",
}


def _event(**overrides: Any) -> dict[str, Any]:
    program = {
        "title": "Game",
        "episodeTitle": "Finals",
        "shortDesc": "Desc",
        "season": "2",
        "episode": "9",
        "releaseYear": "2020",
        **overrides.pop("program", {}),
    }
    return {
        "startTime": "2025-04-20T18:00:00Z",
        "endTime": "2025-04-20T18:30:00Z",
        "duration": "30",
        "thumbnail": "p1_e",
        "filter": ["filter-sports"],
        "rating": "TV-G",
        "flag": ["New"],
        "tags": ["CC", "Stereo", "DVS"],
        "program": program,
        **overrides,
    }


@pytest.fixture(autouse=True)
def utc() -> Iterator[None]:
    old = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    yield
    if old is None:
        del os.environ["TZ"]
    else:
        os.environ["TZ"] = old
    time.tzset()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body


def _render(event: dict[str, Any], *, tvimate: bool) -> str:
    out = xml.Element("tv")
    f = zap2xml.add_programme_tvimate if tvimate else zap2xml.add_programme
    f(out, event, "K")
    return xml.tostring(out[0], encoding="unicode")


def _programme(event: dict[str, Any], *, tvimate: bool) -> xml.Element:
    out = xml.Element("tv")
    f = zap2xml.add_programme_tvimate if tvimate else zap2xml.add_programme
    f(out, event, "K")
    return out[0]


def test_main_builds_guide_from_api(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
) -> None:
    cache_dir = tmp_path / ".zap2xml"
    cache_dir.mkdir()
    (cache_dir / "1000.json.gz").write_bytes(b"")
    urls: list[str] = []

    def urlopen(req: request.Request) -> FakeResponse:
        urls.append(req.full_url)
        assert req.get_header("Accept") == "*/*"
        body = {"channels": [{**CHANNEL, "events": [_event(flag=[])]}]}
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(request, "urlopen", urlopen)
    argv = ["zap2xml.py", "--zip", "90210", "--path", str(tmp_path), "--delay", "0"]
    monkeypatch.setattr(sys, "argv", [*argv, "--days", "0.125"])
    assert zap2xml.main() == 0
    assert not (cache_dir / "1000.json.gz").exists()
    assert len(urls) == 1
    assert "postalCode=90210" in urls[0]
    assert "lineupId=USA-lineupId-DEFAULT" in urls[0]
    assert no_sleep == [0]

    tv = xml.parse(tmp_path / "xmltv.xml").getroot()
    assert tv.get("source-info-name") == "zap2it"
    channel = tv.find("channel")
    assert channel is not None
    assert channel.get("id") == "KXTVDT"
    assert channel.findtext("display-name") == ("American Broadcasting Company - KXTV-DT-1 - 10.1")
    icon = channel.find("icon")
    assert icon is not None
    assert icon.get("src") == "https://zap2it.tmsimg.com/h3/s28708.png"
    assert tv.findtext("programme/title") == "Game: Finals"

    assert "isOverride=True" in urls[0]

    monkeypatch.setattr(
        sys,
        "argv",
        [*argv, "--days", "0.25", "--no-tvimate", "--no-is-override"],
    )
    assert zap2xml.main() == 0
    assert len(urls) == 2
    assert "isOverride=False" in urls[1]
    titles = [p.findtext("sub-title") for p in xml.parse(tmp_path / "xmltv.xml").iter("programme")]
    assert titles == ["Finals", "Finals"]


def test_boolean_flags_parse_on_and_off() -> None:
    parser = argparse.ArgumentParser()
    zap2xml._add_arguments(parser)
    defaults = parser.parse_args(["--zip", "1"])
    assert (defaults.tvimate, defaults.zap_isOverride) == (True, True)
    off = parser.parse_args(["--zip", "1", "--no-tvimate", "--no-is-override"])
    assert (off.tvimate, off.zap_isOverride) == (False, False)
    on = parser.parse_args(["--zip", "1", "--tvimate", "--is-override"])
    assert (on.tvimate, on.zap_isOverride) == (True, True)


def test_main_defaults_to_cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["zap2xml", "--zip", "1", "--days", "0"])
    assert zap2xml.main() == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [".zap2xml", "xmltv.xml"]


def test_main_rejects_too_many_days(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["zap2xml.py", "--zip", "1", "--path", str(tmp_path), "--days", "16"],
    )
    with pytest.raises(ValueError, match="at most 15 days; 16.0 too large"):
        zap2xml.main()


def test_script_entrypoint(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--zip", "1", "--path", str(tmp_path), "--days", "0"],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 0
    assert (tmp_path / ".zap2xml").is_dir()
    assert (tmp_path / "xmltv.xml").read_bytes().startswith(b'<?xml version="1.0"')


def test_get_cached_paths(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with gzip.open(tmp_path / "5.json.gz", "wb") as f:
        f.write(b"cached")
    assert zap2xml.get_cached(tmp_path, 5, 3, "https://x") == b"cached"
    assert "Cached: https://x" in capsys.readouterr().out

    with pytest.raises(ValueError, match="must start with 'http:' or 'https:'"):
        zap2xml.get_cached(tmp_path, 6, 3, "ftp://x")

    def urlopen(req: request.Request) -> FakeResponse:
        code = int(req.full_url.rsplit("/", 1)[1])
        if code == 200:
            return FakeResponse(b"fresh")
        raise error.HTTPError(req.full_url, code, "err", message.Message(), None)

    monkeypatch.setattr(request, "urlopen", urlopen)
    assert zap2xml.get_cached(tmp_path, 7, 3, "http://x/200") == b"fresh"
    with gzip.open(tmp_path / "7.json.gz", "rb") as f:
        assert f.read() == b"fresh"

    result = zap2xml.get_cached(tmp_path, 8, 4, "https://x/400")
    assert json.loads(result)["channels"] == []
    assert "Got a 400 error! Ignoring it." in capsys.readouterr().out
    assert no_sleep == [3, 4]

    with pytest.raises(error.HTTPError) as exc:
        zap2xml.get_cached(tmp_path, 9, 3, "https://x/500")
    assert exc.value.__notes__ == ['Url is "https://x/500".']
    assert not (tmp_path / "9.json.gz").exists()


def test_remove_stale_cache(tmp_path: pathlib.Path) -> None:
    zap = 1_000_000
    day = 86_400
    files = {
        zap - 1: zap,  # Irrelevant: data predates zap time.
        zap + 100: zap,  # Fresh.
        zap + 101: zap - 2 * day,  # Near-term data fetched over a day ago.
        zap + 5 * day: zap - 2 * day,  # Far data still within a week.
        zap + 5 * day + 1: zap - 8 * day,  # Far data fetched over a week ago.
    }
    for data_time, mtime in files.items():
        p = tmp_path / f"{data_time}.json.gz"
        p.write_bytes(b"")
        os.utime(p, (mtime, mtime))
    zap2xml.remove_stale_cache(tmp_path, zap)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"{zap + 100}.json.gz",
        f"{zap + 5 * day}.json.gz",
    ]


def test_add_programme_full_event() -> None:
    assert _render(_event(), tvimate=False) == (
        '<programme start="20250420180000 +0000" stop="20250420183000 +0000" channel="K">'
        "<title>Game</title><sub-title>Finals</sub-title><desc>Desc</desc>"
        '<episode-num system="xmltv_ns">1.8.</episode-num>'
        '<rating system="VCHIP"><value>TV-G</value></rating>'
        '<length units="minutes">30</length><date>2020</date>'
        '<icon src="https://zap2it.tmsimg.com/assets/p1_e.jpg" />'
        "<category>Sports</category>"
        "<audio><present>yes</present><stereo>stereo</stereo><stereo>bilingual</stereo></audio>"
        '<subtitles type="teletext"><language>English</language></subtitles>'
        "<new /></programme>"
    )


def test_add_programme_sparse_movie(capsys: pytest.CaptureFixture[str]) -> None:
    event = _event(
        program={
            "title": "Movie",
            "episodeTitle": None,
            "shortDesc": None,
            "season": None,
            "releaseYear": None,
        },
        filter=["filter-movie", "novel", "filter-kids"],
        rating=None,
        thumbnail=None,
        tags=[],
        flag=[],
    )
    assert _render(event, tvimate=False) == (
        '<programme start="20250420180000 +0000" stop="20250420183000 +0000" channel="K">'
        "<title>Movie</title><sub-title>TBD</sub-title>"
        '<length units="minutes">30</length>'
        "<category>Movie</category><category>Kids</category>"
        "<audio><present>yes</present><stereo>stereo</stereo></audio></programme>"
    )
    assert capsys.readouterr().out == "Novel filter 'novel'.\nNovel filter 'filter-kids'.\n"


@pytest.mark.parametrize(
    ("title", "year", "expected"),
    [("Film", "1999", "Movie (1999)"), ("", None, "Movie")],
)
def test_add_programme_movie_subtitles(title: str, year: str | None, expected: str) -> None:
    event = _event(
        program={"title": title, "episodeTitle": "", "releaseYear": year},
        filter=["filter-movie"],
    )
    prog = _programme(event, tvimate=False)
    assert prog.findtext("sub-title") == expected
    assert (prog.find("title") is None) == (not title)


@pytest.mark.parametrize(
    ("tags", "audio"),
    [
        (["Dolby Digital", "Dolby"], "dolby digital"),
        (["Dolby", "Surround"], "dolby"),
        (["Surround", "Stereo"], "surround"),
        (["Stereo", "Mono"], "stereo"),
        (["Mono"], "mono"),
    ],
)
def test_add_programme_audio(tags: list[str], audio: str) -> None:
    prog = _programme(_event(tags=tags), tvimate=False)
    assert [e.text for e in prog.iterfind("audio/stereo")] == [audio]
    assert prog.find("subtitles") is None


def test_add_programme_tvimate_sports_live() -> None:
    assert _render(_event(flag=["Live", "New"]), tvimate=True) == (
        '<programme start="20250420180000 +0000" stop="20250420183000 +0000" channel="K">'
        "<title>Game: Finals ᴸⁱᵛᵉ</title><desc>S02E09\nDesc</desc>"
        "<category>Sports</category></programme>"
    )


@pytest.mark.parametrize(
    ("overrides", "title", "desc"),
    [
        (
            {"filter": ["filter-news"], "flag": ["Live"]},
            "Game",
            "S02E09 Finals\nDesc",
        ),
        ({"filter": [], "flag": ["New"]}, "Game ᴺᵉʷ", "S02E09 Finals\nDesc"),
        (
            {
                "filter": ["filter-movie"],
                "flag": [],
                "program": {"title": "Movie", "episodeTitle": None, "season": None},
            },
            "Movie",
            "Desc",
        ),
        (
            {
                "filter": ["filter-movie"],
                "flag": [],
                "program": {"episodeTitle": None, "episode": None, "shortDesc": None},
            },
            "Game",
            "Movie (2020)",
        ),
        (
            {
                "filter": ["filter-movie"],
                "flag": [],
                "program": {"episodeTitle": None, "releaseYear": None, "season": None},
            },
            "Game",
            "Movie\nDesc",
        ),
    ],
)
def test_add_programme_tvimate_variants(
    overrides: dict[str, Any],
    title: str,
    desc: str,
) -> None:
    prog = _programme(_event(**overrides), tvimate=True)
    assert prog.findtext("title") == title
    assert prog.findtext("desc") == desc


def test_add_programme_tvimate_empty(capsys: pytest.CaptureFixture[str]) -> None:
    event = _event(
        program={
            "title": "",
            "episodeTitle": None,
            "shortDesc": None,
            "season": None,
            "episode": None,
            "releaseYear": None,
        },
        filter=["odd", "filter-kids"],
    )
    prog = _programme(event, tvimate=True)
    assert [c.tag for c in prog] == ["category"]
    assert prog.findtext("category") == "Kids"
    assert capsys.readouterr().out == "Novel filter 'odd'.\nNovel filter 'filter-kids'.\n"


def test_parse_callsign() -> None:
    assert zap2xml.parse_callsign("KXTVDT") == "KXTV-DT-1"
    assert zap2xml.parse_callsign("kovrdt5") == "KOVR-DT-5"
    assert zap2xml.parse_callsign("KQSLD") == "KQSL-LD-1"
    with pytest.raises(ValueError, match="Expected result."):
        zap2xml.parse_callsign("WGN")
    with pytest.raises(ValueError, match='Expected num != "1".'):
        zap2xml.parse_callsign("KXTVDT1")


def test_time_and_xml_helpers() -> None:
    assert zap2xml.strf_time_int(0) == "1970-Jan-01 12:00am +0000"
    assert zap2xml.strf_time_str("2025-01-02T03:04:05Z", "%H%M") == "0304"
    assert zap2xml.toint(None, fail=-1) == -1
    assert zap2xml.toint("7") == 7
    root = zap2xml.add_xml_child(None, "tv", attrib={"a": "1"}, b="2")
    assert xml.tostring(root, encoding="unicode") == '<tv a="1" b="2" />'


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
