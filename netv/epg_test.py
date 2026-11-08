"""Tests for epg.py - EPG storage and parsing."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gc
import gzip
import sqlite3
import threading
import warnings

import defusedxml.ElementTree as ET
import pytest

from netv import epg
from netv.epg import Program


@pytest.fixture
def db(tmp_path: Path) -> Iterator[None]:
    """Initialize EPG database in temp directory."""
    epg.init(tmp_path)
    yield
    epg.close()


class TestInit:
    """Tests for database initialization."""

    def test_init_creates_tables(self, db: None):
        conn = epg._get_conn()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t["name"] for t in tables}
        assert "channels" in table_names
        assert "icons" in table_names
        assert "programs" in table_names


class TestConnectionLifecycle:
    """Tests that every per-thread connection is closed."""

    def test_connection_of_exited_thread_is_closed(self, db: None):
        worker = threading.Thread(target=epg.has_programs)
        worker.start()
        worker.join()
        del worker
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            gc.collect()
        assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []

    def test_close_closes_connections_of_every_thread(self, db: None):
        conns: list[sqlite3.Connection] = []
        worker = threading.Thread(target=_append_connection, args=(conns,))
        worker.start()
        worker.join()
        conns.append(epg._get_conn())
        epg.close()
        for conn in conns:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                conn.execute("SELECT 1")
        with pytest.raises(RuntimeError, match="not initialized"):
            epg.has_programs()

    def test_reinit_reconnects_to_new_database(self, db: None, tmp_path: Path):
        epg.insert_channel("ch1", "Channel One", "src1")
        epg.commit()
        epg.init(tmp_path / "other")
        assert epg.get_channel_count() == 0


class TestChannels:
    """Tests for channel operations."""

    def test_insert_channel(self, db: None):
        epg.insert_channel("ch1", "Channel One", "src1")
        epg.commit()

        conn = epg._get_conn()
        row = conn.execute("SELECT * FROM channels WHERE id = ?", ("ch1",)).fetchone()
        assert row["name"] == "Channel One"
        assert row["source_id"] == "src1"

    def test_insert_channel_upsert(self, db: None):
        epg.insert_channel("ch1", "Old Name", "src1")
        epg.insert_channel("ch1", "New Name", "src1")
        epg.commit()

        conn = epg._get_conn()
        rows = conn.execute("SELECT * FROM channels WHERE id = ?", ("ch1",)).fetchall()
        assert len(rows) == 1
        assert rows[0]["name"] == "New Name"


class TestIcons:
    """Tests for icon operations."""

    def test_insert_icon(self, db: None):
        epg.insert_icon("ch1", "http://example.com/icon.png")
        epg.commit()

        result = epg.get_icon("ch1")
        assert result == "http://example.com/icon.png"

    def test_get_icon_not_found(self, db: None):
        result = epg.get_icon("nonexistent")
        assert result == ""

    def test_get_icons_batch(self, db: None):
        epg.insert_icon("ch1", "http://example.com/1.png")
        epg.insert_icon("ch2", "http://example.com/2.png")
        epg.insert_icon("ch3", "http://example.com/3.png")
        epg.commit()

        result = epg.get_icons_batch(["ch1", "ch3"])
        assert result == {
            "ch1": "http://example.com/1.png",
            "ch3": "http://example.com/3.png",
        }

    def test_get_icons_batch_empty(self, db: None):
        result = epg.get_icons_batch([])
        assert result == {}


class TestPrograms:
    """Tests for program operations."""

    def test_insert_programs(self, db: None):
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

    def test_get_programs_in_range(self, db: None):
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

    def test_get_programs_in_range_empty(self, db: None):
        now = datetime.now(timezone.utc)
        result = epg.get_programs_in_range("ch1", now, now + timedelta(hours=1))
        assert result == []

    def test_get_programs_batch(self, db: None):
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

    def test_get_programs_batch_empty_channels(self, db: None):
        result = epg.get_programs_batch([], datetime.now(timezone.utc), datetime.now(timezone.utc))
        assert result == {}

    def test_has_programs_false(self, db: None):
        assert epg.has_programs() is False

    def test_has_programs_true(self, db: None):
        now = datetime.now(timezone.utc)
        epg.insert_programs(
            [("ch1", "Show", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1")],
        )
        epg.commit()
        assert epg.has_programs() is True

    def test_get_program_count(self, db: None):
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

    def test_get_channel_count(self, db: None):
        assert epg.get_channel_count() == 0

        epg.insert_channel("ch1", "Channel 1", "src1")
        epg.insert_channel("ch2", "Channel 2", "src1")
        epg.commit()

        assert epg.get_channel_count() == 2


class TestClear:
    """Tests for clear operations."""

    def test_clear_all(self, db: None):
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

    def test_clear_source(self, db: None):
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

    def test_has_programs_after(self, db: None):
        now = datetime.now(timezone.utc)
        epg.insert_programs(
            [("ch1", "Show", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1")],
        )
        epg.commit()

        assert epg.has_programs_after(now)
        assert not epg.has_programs_after(now + timedelta(hours=2))

    def test_search_channels_ranks_channels_with_listings_first(self, db: None):
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

    def test_prune_old_programs(self, db: None):
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

    def test_prefer_source_in_range(self, db: None):
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

    def test_batch_dedupes_only_channels_with_preference(self, db: None):
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

    def test_batch_keeps_existing_when_preferred_comes_first(self, db: None):
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

    def test_batch_chunks_channel_ids(self, db: None):
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


def _append_connection(conns: list[sqlite3.Connection]) -> None:
    conns.append(epg._get_conn())


class _Resp:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


def _fake_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
) -> list[tuple[str, int, str | None]]:
    calls: list[tuple[str, int, str | None]] = []

    def fake(url: str, timeout: int, user_agent: str | None) -> _Resp:
        calls.append((url, timeout, user_agent))
        return _Resp(body)

    monkeypatch.setattr(epg, "safe_urlopen", fake)
    return calls


XMLTV = """<?xml version="1.0"?>
<tv>
  <channel id="ch1"><display-name>Channel One</display-name><icon src="http://i/1.png"/></channel>
  <channel id="ch2"><display-name></display-name></channel>
  <channel id="ch3"/>
  <programme start="20241130120000 +0000" stop="20241130130000 +0000" channel="ch1">
    <title>News</title><desc>Daily news</desc>
  </programme>
  <programme start="20241130130000 +0000" stop="20241130140000 +0000" channel="ch1">
    <title></title>
  </programme>
  <programme start="bogus-timestamp" stop="20241130140000 +0000" channel="ch2">
    <title>Broken</title>
  </programme>
</tv>
"""


class TestFetchEpg:
    """Tests for fetch_epg with mocked HTTP."""

    def test_parses_channels_icons_and_programs(
        self,
        db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls = _fake_urlopen(monkeypatch, XMLTV.encode())

        count = epg.fetch_epg(
            "http://epg.example/guide.xml",
            tmp_path,
            timeout=7,
            source_id="src1",
            user_agent="UA/1",
        )

        assert count == 2
        assert calls == [("http://epg.example/guide.xml", 7, "UA/1")]
        conn = epg._get_conn()
        names = dict(conn.execute("SELECT id, name FROM channels").fetchall())
        assert names == {"ch1": "Channel One", "ch2": "ch2", "ch3": "ch3"}
        assert epg.get_icons_batch(["ch1", "ch2", "ch3"]) == {"ch1": "http://i/1.png"}
        start = datetime(2024, 11, 30, 12, tzinfo=timezone.utc)
        progs = epg.get_programs_in_range("ch1", start, start + timedelta(hours=2))
        assert [(p.title, p.desc, p.source_id) for p in progs] == [
            ("News", "Daily news", "src1"),
            ("Unknown", "", "src1"),
        ]

    def test_decompresses_gzip(self, db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _fake_urlopen(monkeypatch, gzip.compress(XMLTV.encode()))
        assert epg.fetch_epg("http://epg.example/guide.xml.gz", tmp_path) == 2

    def test_inserts_in_batches(self, db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        n = 10_001
        prog = (
            '<programme start="20241130120000 +0000" stop="20241130130000 +0000" channel="c">'
            "<title>T</title></programme>"
        )
        body = "<tv>" + prog * n + "</tv>"
        _fake_urlopen(monkeypatch, body.encode())
        inserted: list[int] = []
        real_insert = epg.insert_programs

        def counting_insert(batch: list[tuple[str, str, float, float, str, str]]) -> None:
            inserted.append(len(batch))
            real_insert(batch)

        monkeypatch.setattr(epg, "insert_programs", counting_insert)

        assert epg.fetch_epg("http://epg.example/big.xml", tmp_path) == n
        assert inserted == [10_000, 1]
        assert epg.get_program_count() == n

    def test_sanitizes_corrupt_xml(self, db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        body = (
            '<?xml version="1.0"?><tv>'
            '<channel id="ch1"><display-name>One</display-name></channel>'
            "<garbage & stuff"
            '<programme start="20241130120000 +0000" stop="20241130130000 +0000" channel="ch1">'
            "<title>Kept</title></programme>"
            "</tv>"
        )
        _fake_urlopen(monkeypatch, body.encode())

        assert epg.fetch_epg("http://epg.example/bad.xml", tmp_path) == 1
        debug_files = list(tmp_path.glob("epg_debug_*.xml"))
        assert len(debug_files) == 1
        assert debug_files[0].read_text() == body
        assert epg.get_channel_count() == 1

    def test_raises_when_sanitized_xml_also_fails(
        self,
        db: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        body = '<tv><channel id="ch1"><bad></channel>'
        _fake_urlopen(monkeypatch, body.encode())
        with pytest.raises(ET.ParseError):
            epg.fetch_epg("http://epg.example/worse.xml", tmp_path)
        assert len(list(tmp_path.glob("epg_debug_*.xml"))) == 1


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


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
