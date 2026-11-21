"""Tests for epg.py - EPG storage and parsing."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gzip
import logging
import tracemalloc
import zlib

import defusedxml.ElementTree as ET
import pytest

from netv import db, epg, testing, util
from netv.epg import Program


@pytest.fixture
def epg_db(tmp_path: Path) -> Iterator[None]:
    """Initialize EPG database in temp directory."""
    epg.init(tmp_path)
    yield
    db.close()


class TestInit:
    """Tests for database initialization."""

    def test_init_creates_tables(self, epg_db: None):
        conn = db.connection()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t["name"] for t in tables}
        assert "channels" in table_names
        assert "icons" in table_names
        assert "programs" in table_names


class TestReinit:
    """Tests for opening another guide database."""

    def test_reinit_reconnects_to_new_database(self, epg_db: None, tmp_path: Path):
        epg.insert_channel("ch1", "Channel One", "src1")
        epg.commit()
        epg.init(tmp_path / "other")
        assert epg.get_channel_count() == 0


class TestChannels:
    """Tests for channel operations."""

    def test_insert_channel(self, epg_db: None):
        epg.insert_channel("ch1", "Channel One", "src1")
        epg.commit()

        conn = db.connection()
        row = conn.execute("SELECT * FROM channels WHERE id = ?", ("ch1",)).fetchone()
        assert row["name"] == "Channel One"
        assert row["source_id"] == "src1"

    def test_insert_channel_upsert(self, epg_db: None):
        epg.insert_channel("ch1", "Old Name", "src1")
        epg.insert_channel("ch1", "New Name", "src1")
        epg.commit()

        conn = db.connection()
        rows = conn.execute("SELECT * FROM channels WHERE id = ?", ("ch1",)).fetchall()
        assert len(rows) == 1
        assert rows[0]["name"] == "New Name"


class TestIcons:
    """Tests for icon operations."""

    def test_insert_icon(self, epg_db: None):
        epg.insert_icon("ch1", "http://example.com/icon.png")
        epg.commit()

        result = epg.get_icon("ch1")
        assert result == "http://example.com/icon.png"

    def test_get_icon_not_found(self, epg_db: None):
        result = epg.get_icon("nonexistent")
        assert result == ""

    def test_get_icons_batch(self, epg_db: None):
        epg.insert_icon("ch1", "http://example.com/1.png")
        epg.insert_icon("ch2", "http://example.com/2.png")
        epg.insert_icon("ch3", "http://example.com/3.png")
        epg.commit()

        result = epg.get_icons_batch(["ch1", "ch3"])
        assert result == {
            "ch1": "http://example.com/1.png",
            "ch3": "http://example.com/3.png",
        }

    def test_get_icons_batch_empty(self, epg_db: None):
        result = epg.get_icons_batch([])
        assert result == {}


class TestPrograms:
    """Tests for program operations."""

    def test_insert_programs(self, epg_db: None):
        now = datetime.now(timezone.utc)
        programs = [
            (
                "ch1",
                "Show 1",
                now.timestamp(),
                (now + timedelta(hours=1)).timestamp(),
                "Desc 1",
                "src1",
            ),
            (
                "ch1",
                "Show 2",
                (now + timedelta(hours=1)).timestamp(),
                (now + timedelta(hours=2)).timestamp(),
                "Desc 2",
                "src1",
            ),
        ]
        epg.insert_programs(programs)
        epg.commit()

        count = epg.get_program_count()
        assert count == 2

    def test_get_programs_in_range(self, epg_db: None):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        programs = [
            (
                "ch1",
                "Show 1",
                now.timestamp(),
                (now + timedelta(hours=1)).timestamp(),
                "Desc 1",
                "src1",
            ),
            (
                "ch1",
                "Show 2",
                (now + timedelta(hours=1)).timestamp(),
                (now + timedelta(hours=2)).timestamp(),
                "Desc 2",
                "src1",
            ),
            (
                "ch1",
                "Show 3",
                (now + timedelta(hours=2)).timestamp(),
                (now + timedelta(hours=3)).timestamp(),
                "Desc 3",
                "src1",
            ),
        ]
        epg.insert_programs(programs)
        epg.commit()

        # Query for middle hour.
        result = epg.get_programs_in_range(
            "ch1",
            now + timedelta(minutes=30),
            now + timedelta(hours=1, minutes=30),
        )
        assert len(result) == 2
        assert result[0].title == "Show 1"
        assert result[1].title == "Show 2"

    def test_get_programs_in_range_empty(self, epg_db: None):
        now = datetime.now(timezone.utc)
        result = epg.get_programs_in_range("ch1", now, now + timedelta(hours=1))
        assert result == []

    def test_get_programs_batch(self, epg_db: None):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        programs = [
            ("ch1", "Show A", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1"),
            ("ch2", "Show B", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1"),
        ]
        epg.insert_programs(programs)
        epg.commit()

        result = epg.get_programs_batch(
            ["ch1", "ch2", "ch3"],
            now,
            now + timedelta(hours=1),
        )
        assert len(result["ch1"]) == 1
        assert len(result["ch2"]) == 1
        assert len(result["ch3"]) == 0
        assert result["ch1"][0].title == "Show A"
        assert result["ch2"][0].title == "Show B"

    def test_get_programs_batch_empty_channels(self, epg_db: None):
        result = epg.get_programs_batch([], datetime.now(timezone.utc), datetime.now(timezone.utc))
        assert result == {}

    def test_has_programs_false(self, epg_db: None):
        assert epg.has_programs() is False

    def test_has_programs_true(self, epg_db: None):
        now = datetime.now(timezone.utc)
        epg.insert_programs(
            [("ch1", "Show", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1")],
        )
        epg.commit()
        assert epg.has_programs() is True

    def test_get_program_count(self, epg_db: None):
        now = datetime.now(timezone.utc)
        assert epg.get_program_count() == 0

        epg.insert_programs(
            [
                (
                    "ch1",
                    "Show 1",
                    now.timestamp(),
                    (now + timedelta(hours=1)).timestamp(),
                    "",
                    "src1",
                ),
                (
                    "ch1",
                    "Show 2",
                    (now + timedelta(hours=1)).timestamp(),
                    (now + timedelta(hours=2)).timestamp(),
                    "",
                    "src1",
                ),
            ],
        )
        epg.commit()

        assert epg.get_program_count() == 2

    def test_get_channel_count(self, epg_db: None):
        assert epg.get_channel_count() == 0

        epg.insert_channel("ch1", "Channel 1", "src1")
        epg.insert_channel("ch2", "Channel 2", "src1")
        epg.commit()

        assert epg.get_channel_count() == 2


class TestClear:
    """Tests for clear operations."""

    def test_clear_all(self, epg_db: None):
        now = datetime.now(timezone.utc)
        epg.insert_channel("ch1", "Channel 1", "src1")
        epg.insert_icon("ch1", "http://example.com/icon.png")
        epg.insert_programs(
            [("ch1", "Show", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1")],
        )
        epg.commit()

        epg.clear()

        assert epg.get_channel_count() == 0
        assert epg.get_program_count() == 0
        assert epg.get_icon("ch1") == ""

    def test_clear_source(self, epg_db: None):
        now = datetime.now(timezone.utc)
        epg.insert_channel("ch1", "Channel 1", "src1")
        epg.insert_channel("ch2", "Channel 2", "src2")
        epg.insert_programs(
            [
                (
                    "ch1",
                    "Show 1",
                    now.timestamp(),
                    (now + timedelta(hours=1)).timestamp(),
                    "",
                    "src1",
                ),
                (
                    "ch2",
                    "Show 2",
                    now.timestamp(),
                    (now + timedelta(hours=1)).timestamp(),
                    "",
                    "src2",
                ),
            ],
        )
        epg.commit()

        epg.clear_source("src1")

        assert epg.get_channel_count() == 1
        assert epg.get_program_count() == 1


class TestPrune:
    """Tests for prune operations."""

    def test_has_programs_after(self, epg_db: None):
        now = datetime.now(timezone.utc)
        epg.insert_programs(
            [("ch1", "Show", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1")],
        )
        epg.commit()

        assert epg.has_programs_after(now)
        assert not epg.has_programs_after(now + timedelta(hours=2))

    def test_search_channels_ranks_channels_with_listings_first(self, epg_db: None):
        now = datetime.now(timezone.utc)
        epg.insert_channel("alpha.one", "XX| ALPHA ONE", "src1")
        epg.insert_channel("XX-Alpha One", "XX| ALPHA ONE HD", "src1")
        epg.insert_channel("beta.one", "XX| BETA", "src1")
        epg.insert_programs(
            [
                (
                    "XX-Alpha One",
                    "Show",
                    now.timestamp(),
                    (now + timedelta(hours=1)).timestamp(),
                    "",
                    "src1",
                ),
            ],
        )
        epg.commit()

        results = epg.search_channels(["ALPHA", "ONE"])

        assert [r["id"] for r in results] == ["XX-Alpha One", "alpha.one"]
        assert [r["has_listings"] for r in results] == [True, False]
        assert epg.search_channels([]) == []

    def test_prune_old_programs(self, epg_db: None):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=2)
        epg.insert_programs(
            [
                (
                    "ch1",
                    "Old Show",
                    old.timestamp(),
                    (old + timedelta(hours=1)).timestamp(),
                    "",
                    "src1",
                ),
                (
                    "ch1",
                    "New Show",
                    now.timestamp(),
                    (now + timedelta(hours=1)).timestamp(),
                    "",
                    "src1",
                ),
            ],
        )
        epg.commit()

        deleted = epg.prune_old_programs(now - timedelta(days=1))

        assert deleted == 1
        assert epg.get_program_count() == 1


class TestPreferredSource:
    """Tests for preferred source deduplication."""

    def test_prefer_source_in_range(self, epg_db: None):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        # Two overlapping programs from different sources.
        epg.insert_programs(
            [
                (
                    "ch1",
                    "From Src1",
                    now.timestamp(),
                    (now + timedelta(hours=1)).timestamp(),
                    "",
                    "src1",
                ),
                (
                    "ch1",
                    "From Src2",
                    now.timestamp(),
                    (now + timedelta(hours=1)).timestamp(),
                    "",
                    "src2",
                ),
            ],
        )
        epg.commit()

        # Prefer src2.
        result = epg.get_programs_in_range(
            "ch1",
            now,
            now + timedelta(hours=1),
            preferred_source_id="src2",
        )
        assert len(result) == 1
        assert result[0].title == "From Src2"

        # Prefer src1.
        result = epg.get_programs_in_range(
            "ch1",
            now,
            now + timedelta(hours=1),
            preferred_source_id="src1",
        )
        assert len(result) == 1
        assert result[0].title == "From Src1"


class TestProgram:
    """Tests for Program dataclass."""

    def test_program_dataclass(self):
        now = datetime.now(timezone.utc)
        p = Program(
            channel_id="ch1",
            title="Test Show",
            start=now,
            stop=now + timedelta(hours=1),
            desc="Description",
            source_id="src1",
        )
        assert p.channel_id == "ch1"
        assert p.title == "Test Show"
        assert p.desc == "Description"
        assert p.source_id == "src1"

    def test_program_defaults(self):
        now = datetime.now(timezone.utc)
        p = Program(channel_id="ch1", title="Test", start=now, stop=now + timedelta(hours=1))
        assert p.desc == ""
        assert p.source_id == ""


class TestBatchPreferredSources:
    """Tests for get_programs_batch deduplication."""

    def test_batch_dedupes_only_channels_with_preference(self, epg_db: None):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        hour = (now + timedelta(hours=1)).timestamp()
        later = (now + timedelta(hours=2)).timestamp()
        epg.insert_programs(
            [
                ("ch1", "Src1 Show", now.timestamp(), hour, "", "src1"),
                ("ch1", "Src2 Show", now.timestamp(), hour, "", "src2"),
                ("ch1", "Src1 Later", hour, later, "", "src1"),
                ("ch2", "A", now.timestamp(), hour, "", "src1"),
                ("ch2", "B", now.timestamp(), hour, "", "src2"),
                ("ch3", "Only", now.timestamp(), hour, "", "src1"),
            ],
        )
        epg.commit()

        result = epg.get_programs_batch(
            ["ch1", "ch2", "ch3", "ch4"],
            now,
            now + timedelta(hours=2),
            preferred_sources={"ch1": "src2", "ch3": "src2", "ch4": "src1"},
        )

        assert [p.title for p in result["ch1"]] == ["Src2 Show", "Src1 Later"]
        assert sorted(p.title for p in result["ch2"]) == ["A", "B"]
        assert [p.title for p in result["ch3"]] == ["Only"]
        assert result["ch4"] == []

    def test_batch_keeps_existing_when_preferred_comes_first(self, epg_db: None):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        hour = (now + timedelta(hours=1)).timestamp()
        epg.insert_programs(
            [
                ("ch1", "Pref", now.timestamp(), hour, "", "src1"),
                ("ch1", "Other", now.timestamp(), hour, "", "src2"),
            ],
        )
        epg.commit()
        result = epg.get_programs_batch(
            ["ch1"],
            now,
            now + timedelta(hours=1),
            preferred_sources={"ch1": "src1"},
        )
        assert [p.title for p in result["ch1"]] == ["Pref"]

    def test_dedupe_without_preference_is_identity(self):
        now = datetime.now(timezone.utc)
        programs = [
            Program(channel_id="c", title="a", start=now, stop=now + timedelta(hours=1)),
            Program(channel_id="c", title="b", start=now, stop=now + timedelta(hours=1)),
        ]
        assert epg._dedupe_programs(programs, "") is programs

    def test_batch_chunks_channel_ids(self, epg_db: None):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        hour = (now + timedelta(hours=1)).timestamp()
        epg.insert_programs([(f"ch{i}", f"S{i}", now.timestamp(), hour, "", "s") for i in range(5)])
        epg.insert_icon("ch4", "http://i/4.png")
        epg.commit()
        ids = [f"ch{i}" for i in range(5)]
        result = epg.get_programs_batch(ids, now, now + timedelta(hours=1), chunk_size=2)
        assert [result[i][0].title for i in ids] == [f"S{i}" for i in range(5)]
        assert epg.get_icons_batch(ids, chunk_size=2) == {"ch4": "http://i/4.png"}


class TestParseEpgTime:
    """Tests for XMLTV timestamp parsing."""

    def test_positive_offset(self):
        dt = epg._parse_epg_time("20241130120000 +0530")
        assert dt == datetime(2024, 11, 30, 6, 30, tzinfo=timezone.utc)

    def test_negative_offset(self):
        dt = epg._parse_epg_time("20241130120000-0800")
        assert dt == datetime(2024, 11, 30, 20, 0, tzinfo=timezone.utc)

    def test_hours_only_offset(self):
        dt = epg._parse_epg_time("20241130120000+05")
        assert dt.utcoffset() == timedelta(hours=5)

    def test_sign_only_offset_is_utc(self):
        dt = epg._parse_epg_time("20241130120000Z")
        assert dt.utcoffset() == timedelta(0)

    def test_no_offset_is_naive(self):
        assert epg._parse_epg_time("20241130120000") == datetime(2024, 11, 30, 12, 0)

    def test_short_string_returns_now(self):
        before = datetime.now(timezone.utc)
        dt = epg._parse_epg_time("2024")
        assert before <= dt <= datetime.now(timezone.utc)

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            epg._parse_epg_time("abcdefghijklmn")


def _serve(monkeypatch: pytest.MonkeyPatch, url: str, body: bytes) -> testing.FakeWeb:
    # A fresh response per request, as a server sends: a failed parse fetches again.
    web = testing.FakeWeb({url: lambda: testing.FakeResponse(body)})
    testing.install_web(monkeypatch, web, util)
    return web


def _requests(web: testing.FakeWeb) -> list[tuple[str, float | None, dict[str, str]]]:
    return [(f.url, f.timeout, f.headers) for f in web.fetches]


VLC = {"User-agent": "VLC/3.0.20 LibVLC/3.0.20"}


XMLTV = """<?xml version="1.0"?>
<tv>
  <channel id="ch1"><display-name>Channel One</display-name><icon src="http://i/1.png"/></channel>
  <channel id="ch2"><display-name></display-name></channel>
  <channel id="ch3"/>
  <programme start="20241130120000 +0000" stop="20241130130000 +0000" channel="ch1">
    <title>News</title><desc>Daily news</desc>
  </programme>
  <programme start="bogus-timestamp" stop="20241130140000 +0000" channel="ch2">
    <title>Broken</title>
  </programme>
  <programme start="20241130130000 +0000" stop="20241130140000 +0000" channel="ch1">
    <title></title>
  </programme>
</tv>
"""


def _guide(*programmes: tuple[str, str, str]) -> bytes:
    """An XMLTV feed of ``(channel, start hour on 2024-11-30, title)`` hour-long programmes."""
    rows = "".join(
        f'<programme start="20241130{hour}0000 +0000" stop="20241130{int(hour) + 1:02d}0000 +0000"'
        f' channel="{channel}"><title>{title}</title></programme>'
        for channel, hour, title in programmes
    )
    return f"<tv>{rows}</tv>".encode()


def _titles(source_id: str) -> list[tuple[str, str]]:
    rows = db.connection().execute(
        "SELECT channel_id, title FROM programs WHERE source_id = ? ORDER BY channel_id, start_ts",
        (source_id,),
    )
    return [(str(channel), str(title)) for channel, title in rows]


class TestRefetch:
    """A source fetched again replaces its schedule instead of adding a second copy."""

    def test_refetch_replaces_the_schedule_from_the_feed_start(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        url = "http://epg.example/guide.xml"
        _serve(
            monkeypatch,
            url,
            _guide(("c", "10", "Earlier"), ("c", "12", "Old"), ("d", "13", "Gone")),
        )
        epg.fetch_epg(url, tmp_path, source_id="src1")
        _serve(monkeypatch, url, _guide(("c", "12", "New"), ("c", "13", "Next")))
        assert epg.fetch_epg(url, tmp_path, source_id="src1") == 2
        # Listings before the new feed's first programme stay for catch-up; later ones are
        # replaced. Channels the new feed omits are kept: another guide may supply them.
        assert _titles("src1") == [
            ("c", "Earlier"),
            ("c", "New"),
            ("c", "Next"),
            ("d", "Gone"),
        ]

    def test_one_of_a_source_s_guides_leaves_the_others_channels_alone(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # A playlist can name several guide URLs, all loaded for one source.
        _serve(monkeypatch, "http://epg.example/a.xml", _guide(("a", "12", "From A")))
        epg.fetch_epg("http://epg.example/a.xml", tmp_path, source_id="src1")
        _serve(monkeypatch, "http://epg.example/b.xml", _guide(("b", "12", "From B")))
        epg.fetch_epg("http://epg.example/b.xml", tmp_path, source_id="src1")
        assert _titles("src1") == [("a", "From A"), ("b", "From B")]

    def test_refetch_leaves_other_sources_alone(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        url = "http://epg.example/guide.xml"
        _serve(monkeypatch, url, _guide(("c", "12", "Other")))
        epg.fetch_epg(url, tmp_path, source_id="src2")
        _serve(monkeypatch, url, _guide(("c", "12", "Mine")))
        epg.fetch_epg(url, tmp_path, source_id="src1")
        epg.fetch_epg(url, tmp_path, source_id="src1")
        assert _titles("src2") == [("c", "Other")]
        assert _titles("src1") == [("c", "Mine")]

    def test_a_feed_without_programmes_or_source_deletes_nothing(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        url = "http://epg.example/guide.xml"
        _serve(monkeypatch, url, _guide(("c", "12", "Kept")))
        epg.fetch_epg(url, tmp_path, source_id="src1")
        epg.fetch_epg(url, tmp_path)
        _serve(monkeypatch, url, _guide())
        assert epg.fetch_epg(url, tmp_path, source_id="src1") == 0
        assert _titles("src1") == [("c", "Kept")]
        # Feeds without a source share one, so none replaces another's listings.
        _serve(monkeypatch, url, _guide(("c", "12", "Second")))
        epg.fetch_epg(url, tmp_path)
        assert len(_titles("")) == 2


class TestFetchEpg:
    """Tests for fetch_epg against a fake EPG server."""

    def test_parses_channels_icons_and_programs(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        web = _serve(monkeypatch, "http://epg.example/guide.xml", XMLTV.encode())
        caplog.set_level(logging.DEBUG, logger="netv.epg")

        count = epg.fetch_epg(
            "http://epg.example/guide.xml",
            tmp_path,
            timeout=7,
            source_id="src1",
            user_agent="UA/1",
        )

        assert count == 2
        assert caplog.messages == [
            "EPG parsed: 3 channels, 2 unique program channel IDs, 2 programs",
        ]
        assert _requests(web) == [("http://epg.example/guide.xml", 7, {"User-agent": "UA/1"})]
        conn = db.connection()
        names = dict(conn.execute("SELECT id, name FROM channels").fetchall())
        assert names == {"ch1": "Channel One", "ch2": "ch2", "ch3": "ch3"}
        assert epg.get_icons_batch(["ch1", "ch2", "ch3"]) == {"ch1": "http://i/1.png"}
        start = datetime(2024, 11, 30, 12, tzinfo=timezone.utc)
        progs = epg.get_programs_in_range("ch1", start, start + timedelta(hours=2))
        assert [(p.title, p.desc, p.source_id) for p in progs] == [
            ("News", "Daily news", "src1"),
            ("Unknown", "", "src1"),
        ]
        hour = timedelta(hours=1)
        assert [(p.start, p.stop) for p in progs] == [
            (start, start + hour),
            (start + hour, start + 2 * hour),
        ]

    def test_decompresses_gzip(self, epg_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        url = "http://epg.example/guide.xml.gz"
        web = _serve(monkeypatch, url, gzip.compress(XMLTV.encode()))
        assert epg.fetch_epg(url, tmp_path) == 2
        assert _requests(web) == [(url, 120, VLC)]
        conn = db.connection()
        assert {row[0] for row in conn.execute("SELECT source_id FROM channels")} == {""}
        assert {row[0] for row in conn.execute("SELECT source_id FROM programs")} == {""}

    def test_missing_attributes_use_empty_defaults(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        start, stop = "20241130120000 +0000", "20241130130000 +0000"
        body = (
            "<tv><channel><icon/></channel>"
            '<programme channel="untimed"><title>Untimed</title></programme>'
            f'<programme start="{start}" channel="no-stop"><title>No stop</title></programme>'
            f'<programme stop="{stop}" channel="no-start"><title>No start</title></programme>'
            f'<programme start="{start}" stop="{stop}"/>'
            "</tv>"
        )
        _serve(monkeypatch, "http://epg.example/defaults.xml", body.encode())
        # A programme without both times is skipped rather than placed at "now".
        assert epg.fetch_epg("http://epg.example/defaults.xml", tmp_path) == 1
        conn = db.connection()
        assert [tuple(row) for row in conn.execute("SELECT * FROM channels")] == [("", "", "")]
        assert [tuple(row) for row in conn.execute("SELECT * FROM icons")] == [("", "")]
        noon = datetime(2024, 11, 30, 12, tzinfo=timezone.utc).timestamp()
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT channel_id, title, desc, source_id, start_ts, stop_ts FROM programs",
            )
        ] == [("", "Unknown", "", "", noon, noon + 3600)]

    def test_counts_rows_streamed_from_a_generator(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        n = 3
        prog = (
            '<programme start="20241130120000 +0000" stop="20241130130000 +0000" channel="c">'
            "<title>T</title></programme>"
        )
        _serve(monkeypatch, "http://epg.example/big.xml", f"<tv>{prog * n}</tv>".encode())

        assert epg.fetch_epg("http://epg.example/big.xml", tmp_path) == n
        assert epg.get_program_count() == n
        rows = (("c", f"T{i}", 0.0, 1.0, "", "") for i in range(2))
        assert epg.insert_programs(rows) == 2
        assert epg.get_program_count() == n + 2

    def test_sanitizes_corrupt_xml(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.INFO, logger="netv.epg")
        body = (
            '<?xml version="1.0"?><tv>'
            '<channel id="ch1"><display-name>One</display-name></channel>'
            "<garbage & stuff"
            '<programme start="20241130120000 +0000" stop="20241130130000 +0000" channel="ch1">'
            "<title>Kept</title></programme>"
            "</tv>"
        )
        web = _serve(monkeypatch, "http://epg.example/bad.xml", body.encode())

        assert epg.fetch_epg("http://epg.example/bad.xml", tmp_path) == 1
        # The streamed copy is gone when parsing fails, so the repair fetches it again.
        assert len(_requests(web)) == 2
        debug_files = list(tmp_path.glob("epg_debug_*.xml"))
        assert len(debug_files) == 1
        assert debug_files[0].read_text() == body
        assert epg.get_channel_count() == 1
        with pytest.raises(ET.ParseError) as original:
            ET.fromstring(body)
        assert caplog.messages == [
            f"EPG parse failed ({original.value}), attempting sanitization...",
            "Sanitized EPG: extracted 1 channels, 1 programmes",
            "Sanitized EPG parsed successfully",
        ]

    def test_raises_when_sanitized_xml_also_fails(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        body = '<tv><channel id="ch1"><bad></channel>'
        _serve(monkeypatch, "http://epg.example/worse.xml", body.encode())
        with pytest.raises(ET.ParseError) as error:
            epg.fetch_epg("http://epg.example/worse.xml", tmp_path)
        assert len(list(tmp_path.glob("epg_debug_*.xml"))) == 1
        assert caplog.messages[-1] == f"Sanitized EPG also failed: {error.value}"


class TestStreaming:
    def test_fetch_memory_does_not_grow_with_the_feed(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Detail the guide does not store, such as credits, is most of a real feed.
        credits = "<credits>" + "<actor>Someone Quite Famous</actor>" * 100 + "</credits>"
        prog = (
            '<programme start="20241130120000 +0000" stop="20241130130000 +0000" channel="c">'
            f"<title>T</title>{credits}</programme>"
        )
        peaks: list[int] = []
        sizes: list[int] = []
        for n in (20, 80):
            feed = f"<tv>{prog * n}</tv>".encode()
            url = f"http://epg.example/{n}.xml.gz"
            _serve(monkeypatch, url, gzip.compress(feed, 1))
            tracemalloc.start()
            try:
                assert epg.fetch_epg(url, tmp_path, chunk_size=4 << 10) == n
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
            sizes.append(len(feed))

        # Holding the feed, its text or its tree would grow the peak by the feed's growth.
        assert peaks[1] - peaks[0] < (sizes[1] - sizes[0]) / 8

    def test_a_feed_that_breaks_late_writes_only_the_repaired_parse(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        prog = (
            '<programme start="20241130120000 +0000" stop="20241130130000 +0000" channel="c">'
            "<title>T</title></programme>"
        )
        # Both programmes stream through before the parse fails at the garbage.
        body = f"<tv>{prog}{prog}<garbage & stuff</tv>"
        url = "http://epg.example/late.xml"
        web = _serve(monkeypatch, url, body.encode())
        count = epg.fetch_epg(url, tmp_path, timeout=7, source_id="src1", user_agent="UA/1")
        assert count == 2
        assert epg.get_program_count() == 2
        # The repair fetches the feed again exactly as the first request did.
        assert _requests(web) == [(url, 7, {"User-agent": "UA/1"})] * 2
        conn = db.connection()
        assert {row[0] for row in conn.execute("SELECT source_id FROM programs")} == {"src1"}

    def test_only_top_level_channels_and_programmes_are_read(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        prog = (
            '<programme start="20241130120000 +0000" stop="20241130130000 +0000" channel="{}">'
            "<title>T</title></programme>"
        )
        nested = f'<extra><channel id="inner"/>{prog.format("inner")}</extra>'
        body = f'<tv><channel id="outer"/>{nested}{prog.format("outer")}</tv>'
        _serve(monkeypatch, "http://epg.example/nested.xml", body.encode())
        assert epg.fetch_epg("http://epg.example/nested.xml", tmp_path) == 1
        conn = db.connection()
        assert [row[0] for row in conn.execute("SELECT id FROM channels")] == ["outer"]
        assert [row[0] for row in conn.execute("SELECT channel_id FROM programs")] == ["outer"]

    def test_gunzip_yields_no_chunk_larger_than_a_read(self):
        chunk_size = 64 << 10
        plain = b"x" * (5 * chunk_size)
        body = testing.FakeResponse(gzip.compress(plain))
        chunks = list(epg.iter_epg_body(body, 10 * chunk_size, chunk_size=chunk_size))
        assert b"".join(chunks) == plain
        assert max(len(chunk) for chunk in chunks) <= chunk_size

    def test_chunk_reader_returns_short_reads_until_the_end(self):
        reader = epg._ChunkReader(iter([b"abc", b"", b"defg"]))
        assert reader.read(2) == b"ab"
        assert reader.read(5) == b"c"
        assert reader.read(-1) == b"defg"
        assert reader.read(1) == b""


class TestSanitize:
    def test_extracts_only_well_formed_elements(self):
        xml = (
            '<channel id="a"><display-name>A</display-name></channel>'
            "junk<<<"
            '<programme start="1" stop="2" channel="a"><title>T</title></programme>'
            '<programme channel="a"><title>no times</title></programme>'
        )
        out = epg._sanitize_epg_xml(xml)
        assert out.startswith('<?xml version="1.0"?>\n<tv>\n')
        assert out.endswith("\n</tv>")
        assert '<channel id="a">' in out
        assert '<programme start="1"' in out
        assert "no times" not in out
        assert "junk" not in out


class _Chunks:
    """A response whose reads return the given chunks one at a time."""

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = list(chunks)

    def read(self, amt: int = -1, /) -> bytes:
        del amt
        return self._chunks.pop(0) if self._chunks else b""


class TestReadEpgBody:
    """Size-bounded reading and gunzipping of EPG feeds."""

    def test_gzip_bomb_stops_at_the_limit_without_expanding(self):
        bomb = gzip.compress(b" " * (8 << 20), 1)  # 8 MiB of spaces, ~36 KiB compressed.
        tracemalloc.start()
        try:
            with pytest.raises(epg.EPGTooLargeError):
                epg.read_epg_body(testing.FakeResponse(bomb), 64 << 10, chunk_size=16 << 10)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 1 << 20

    def test_plain_body_over_the_limit(self):
        message = (
            r"^EPG feed is larger than 100 bytes \(set NETV_EPG_MAX_BYTES to raise the limit\)$"
        )
        with pytest.raises(epg.EPGTooLargeError, match=message):
            epg.read_epg_body(testing.FakeResponse(b"x" * 101), 100)

    def test_bodies_at_the_limit_are_read(self):
        body = b"<tv>" + b" " * 96
        assert epg.read_epg_body(testing.FakeResponse(body), 100) == body
        assert epg.read_epg_body(testing.FakeResponse(gzip.compress(body)), 100) == body

    def test_gzip_members_split_across_reads(self):
        first, second = gzip.compress(b"<tv>"), gzip.compress(b"</tv>")
        # The first member ends exactly at the end of a read.
        assert epg.read_epg_body(_Chunks(first, second), 100) == b"<tv></tv>"
        # The second member's magic bytes are split across reads.
        joined = first + second
        cut = len(first) + 1
        assert epg.read_epg_body(_Chunks(joined[:cut], joined[cut:]), 100) == b"<tv></tv>"

    def test_trailing_padding_after_gzip_is_ignored(self):
        body = gzip.compress(b"<tv/>") + b"\x00" * 512
        assert epg.read_epg_body(testing.FakeResponse(body), 100) == b"<tv/>"

    def test_trailing_padding_in_a_later_read_is_ignored(self):
        assert epg.read_epg_body(_Chunks(gzip.compress(b"<tv/>"), b"\x00" * 512), 100) == b"<tv/>"

    def test_truncated_gzip_raises(self):
        body = gzip.compress(b"<tv>" + b"x" * 1000 + b"</tv>")
        with pytest.raises(EOFError):
            epg.read_epg_body(testing.FakeResponse(body[:-12]), 10_000)

    def test_invalid_gzip_raises(self):
        with pytest.raises(zlib.error):
            epg.read_epg_body(testing.FakeResponse(b"\x1f\x8bnot gzip at all"), 100)

    def test_limit_comes_from_the_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NETV_EPG_MAX_BYTES", "4096")
        assert epg.max_epg_bytes() == 4096
        for bad in ("", "lots", "0", "-5"):
            monkeypatch.setenv("NETV_EPG_MAX_BYTES", bad)
            assert epg.max_epg_bytes() == 1 << 30

    def test_fetch_epg_rejects_an_oversized_feed(
        self,
        epg_db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("NETV_EPG_MAX_BYTES", str(len(XMLTV) - 1))
        _serve(monkeypatch, "http://epg.example/guide.xml.gz", gzip.compress(XMLTV.encode()))
        with pytest.raises(epg.EPGTooLargeError):
            epg.fetch_epg("http://epg.example/guide.xml.gz", tmp_path)
        assert not epg.has_programs()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
