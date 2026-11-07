"""Tests for epg.py - EPG storage and parsing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import gc
import sqlite3
import threading
import warnings

import pytest

from epg import Program

import epg


@pytest.fixture
def db(tmp_path: Path):
    """Initialize EPG database in temp directory."""
    epg.init(tmp_path)
    yield epg
    epg.close()


class TestInit:
    """Tests for database initialization."""

    def test_init_creates_tables(self, db):
        conn = db._get_conn()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t["name"] for t in tables}
        assert "channels" in table_names
        assert "icons" in table_names
        assert "programs" in table_names


class TestConnectionLifecycle:
    """Tests that every per-thread connection is closed."""

    def test_connection_of_exited_thread_is_closed(self, db):
        worker = threading.Thread(target=db.has_programs)
        worker.start()
        worker.join()
        del worker
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            gc.collect()
        assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []

    def test_close_closes_connections_of_every_thread(self, db):
        conns: list[sqlite3.Connection] = []
        worker = threading.Thread(target=lambda: conns.append(db._get_conn()))
        worker.start()
        worker.join()
        conns.append(db._get_conn())
        db.close()
        for conn in conns:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                conn.execute("SELECT 1")
        with pytest.raises(RuntimeError, match="not initialized"):
            db.has_programs()

    def test_reinit_reconnects_to_new_database(self, db, tmp_path):
        db.insert_channel("ch1", "Channel One", "src1")
        db.commit()
        db.init(tmp_path / "other")
        assert db.get_channel_count() == 0


class TestChannels:
    """Tests for channel operations."""

    def test_insert_channel(self, db):
        db.insert_channel("ch1", "Channel One", "src1")
        db.commit()

        conn = db._get_conn()
        row = conn.execute("SELECT * FROM channels WHERE id = ?", ("ch1",)).fetchone()
        assert row["name"] == "Channel One"
        assert row["source_id"] == "src1"

    def test_insert_channel_upsert(self, db):
        db.insert_channel("ch1", "Old Name", "src1")
        db.insert_channel("ch1", "New Name", "src1")
        db.commit()

        conn = db._get_conn()
        rows = conn.execute("SELECT * FROM channels WHERE id = ?", ("ch1",)).fetchall()
        assert len(rows) == 1
        assert rows[0]["name"] == "New Name"


class TestIcons:
    """Tests for icon operations."""

    def test_insert_icon(self, db):
        db.insert_icon("ch1", "http://example.com/icon.png")
        db.commit()

        result = db.get_icon("ch1")
        assert result == "http://example.com/icon.png"

    def test_get_icon_not_found(self, db):
        result = db.get_icon("nonexistent")
        assert result == ""

    def test_get_icons_batch(self, db):
        db.insert_icon("ch1", "http://example.com/1.png")
        db.insert_icon("ch2", "http://example.com/2.png")
        db.insert_icon("ch3", "http://example.com/3.png")
        db.commit()

        result = db.get_icons_batch(["ch1", "ch3"])
        assert result == {
            "ch1": "http://example.com/1.png",
            "ch3": "http://example.com/3.png",
        }

    def test_get_icons_batch_empty(self, db):
        result = db.get_icons_batch([])
        assert result == {}


class TestPrograms:
    """Tests for program operations."""

    def test_insert_programs(self, db):
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
        db.insert_programs(programs)
        db.commit()

        count = db.get_program_count()
        assert count == 2

    def test_get_programs_in_range(self, db):
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
        db.insert_programs(programs)
        db.commit()

        # Query for middle hour.
        result = db.get_programs_in_range(
            "ch1",
            now + timedelta(minutes=30),
            now + timedelta(hours=1, minutes=30),
        )
        assert len(result) == 2
        assert result[0].title == "Show 1"
        assert result[1].title == "Show 2"

    def test_get_programs_in_range_empty(self, db):
        now = datetime.now(timezone.utc)
        result = db.get_programs_in_range("ch1", now, now + timedelta(hours=1))
        assert result == []

    def test_get_programs_batch(self, db):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        programs = [
            ("ch1", "Show A", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1"),
            ("ch2", "Show B", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1"),
        ]
        db.insert_programs(programs)
        db.commit()

        result = db.get_programs_batch(
            ["ch1", "ch2", "ch3"],
            now,
            now + timedelta(hours=1),
        )
        assert len(result["ch1"]) == 1
        assert len(result["ch2"]) == 1
        assert len(result["ch3"]) == 0
        assert result["ch1"][0].title == "Show A"
        assert result["ch2"][0].title == "Show B"

    def test_get_programs_batch_empty_channels(self, db):
        result = db.get_programs_batch([], datetime.now(timezone.utc), datetime.now(timezone.utc))
        assert result == {}

    def test_has_programs_false(self, db):
        assert db.has_programs() is False

    def test_has_programs_true(self, db):
        now = datetime.now(timezone.utc)
        db.insert_programs(
            [("ch1", "Show", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1")],
        )
        db.commit()
        assert db.has_programs() is True

    def test_get_program_count(self, db):
        now = datetime.now(timezone.utc)
        assert db.get_program_count() == 0

        db.insert_programs(
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
        db.commit()

        assert db.get_program_count() == 2

    def test_get_channel_count(self, db):
        assert db.get_channel_count() == 0

        db.insert_channel("ch1", "Channel 1", "src1")
        db.insert_channel("ch2", "Channel 2", "src1")
        db.commit()

        assert db.get_channel_count() == 2


class TestClear:
    """Tests for clear operations."""

    def test_clear_all(self, db):
        now = datetime.now(timezone.utc)
        db.insert_channel("ch1", "Channel 1", "src1")
        db.insert_icon("ch1", "http://example.com/icon.png")
        db.insert_programs(
            [("ch1", "Show", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1")],
        )
        db.commit()

        db.clear()

        assert db.get_channel_count() == 0
        assert db.get_program_count() == 0
        assert db.get_icon("ch1") == ""

    def test_clear_source(self, db):
        now = datetime.now(timezone.utc)
        db.insert_channel("ch1", "Channel 1", "src1")
        db.insert_channel("ch2", "Channel 2", "src2")
        db.insert_programs(
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
        db.commit()

        db.clear_source("src1")

        assert db.get_channel_count() == 1
        assert db.get_program_count() == 1


class TestPrune:
    """Tests for prune operations."""

    def test_has_programs_after(self, db):
        now = datetime.now(timezone.utc)
        db.insert_programs(
            [("ch1", "Show", now.timestamp(), (now + timedelta(hours=1)).timestamp(), "", "src1")],
        )
        db.commit()

        assert db.has_programs_after(now)
        assert not db.has_programs_after(now + timedelta(hours=2))

    def test_search_channels_ranks_channels_with_listings_first(self, db):
        now = datetime.now(timezone.utc)
        db.insert_channel("alpha.one", "XX| ALPHA ONE", "src1")
        db.insert_channel("XX-Alpha One", "XX| ALPHA ONE HD", "src1")
        db.insert_channel("beta.one", "XX| BETA", "src1")
        db.insert_programs(
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
        db.commit()

        results = db.search_channels(["ALPHA", "ONE"])

        assert [r["id"] for r in results] == ["XX-Alpha One", "alpha.one"]
        assert [r["has_listings"] for r in results] == [True, False]
        assert db.search_channels([]) == []

    def test_prune_old_programs(self, db):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=2)
        db.insert_programs(
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
        db.commit()

        deleted = db.prune_old_programs(now - timedelta(days=1))

        assert deleted == 1
        assert db.get_program_count() == 1


class TestPreferredSource:
    """Tests for preferred source deduplication."""

    def test_prefer_source_in_range(self, db):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        # Two overlapping programs from different sources.
        db.insert_programs(
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
        db.commit()

        # Prefer src2.
        result = db.get_programs_in_range(
            "ch1",
            now,
            now + timedelta(hours=1),
            preferred_source_id="src2",
        )
        assert len(result) == 1
        assert result[0].title == "From Src2"

        # Prefer src1.
        result = db.get_programs_in_range(
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


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
