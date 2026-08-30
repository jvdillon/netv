"""Tests for filtered native-player XMLTV output."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import sqlite3
import xml.etree.ElementTree as ET

import pytest

from gateway_catalog import GatewayStream
from gateway_epg import EpgUnavailableError, build_filtered_xmltv


def _stream(
    local_id: int,
    channel_id: str,
    source_id: str = "source-a",
) -> GatewayStream:
    return GatewayStream(
        local_id=local_id,
        source_id=source_id,
        source_type="xtream",
        upstream_id=str(local_id),
        direct_url="",
        access_group_ids=("category-a",),
        public={
            "name": f"Channel {local_id}",
            "stream_icon": f"https://images.example/{local_id}.png",
            "epg_channel_id": channel_id,
            "category_id": "1",
            "category_ids": ["1"],
        },
    )


def _create_epg_database(path: Path, now: datetime) -> None:
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE channels (id TEXT PRIMARY KEY, name TEXT, source_id TEXT);
        CREATE TABLE icons (channel_id TEXT PRIMARY KEY, url TEXT);
        CREATE TABLE programs (
            id INTEGER PRIMARY KEY,
            channel_id TEXT,
            title TEXT,
            start_ts REAL,
            stop_ts REAL,
            desc TEXT,
            source_id TEXT
        );
    """)
    connection.executemany(
        "INSERT INTO channels (id, name, source_id) VALUES (?, ?, ?)",
        [
            ("allowed.channel", "Allowed Channel", "source-a"),
            ("blocked.channel", "Blocked Channel", "source-a"),
        ],
    )
    connection.executemany(
        "INSERT INTO programs (channel_id, title, start_ts, stop_ts, desc, source_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                "allowed.channel",
                "Allowed Programme",
                now.timestamp(),
                (now + timedelta(hours=1)).timestamp(),
                "Allowed description",
                "source-a",
            ),
            (
                "blocked.channel",
                "Blocked Programme",
                now.timestamp(),
                (now + timedelta(hours=1)).timestamp(),
                "Blocked description",
                "source-a",
            ),
            (
                "allowed.channel",
                "Wrong Source Programme",
                now.timestamp(),
                (now + timedelta(hours=1)).timestamp(),
                "",
                "source-b",
            ),
            (
                "allowed.channel",
                "Standalone EPG Programme",
                (now + timedelta(hours=1)).timestamp(),
                (now + timedelta(hours=2)).timestamp(),
                "",
                "epg-source",
            ),
            (
                "allowed.channel",
                "Expired Programme",
                (now - timedelta(days=2)).timestamp(),
                (now - timedelta(days=2, hours=-1)).timestamp(),
                "",
                "source-a",
            ),
        ],
    )
    connection.commit()
    connection.close()


def test_filtered_xmltv_contains_only_allowed_current_programs(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=UTC)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)

    content = build_filtered_xmltv(
        database,
        [_stream(1, "allowed.channel")],
        now=now,
    )
    root = ET.fromstring(content)

    assert [channel.get("id") for channel in root.findall("channel")] == [
        "allowed.channel"
    ]
    assert [programme.findtext("title") for programme in root.findall("programme")] == [
        "Allowed Programme",
        "Standalone EPG Programme",
    ]
    assert b"blocked.channel" not in content
    assert b"Wrong Source Programme" not in content
    assert b"Expired Programme" not in content


def test_filtered_xmltv_accepts_nonoverlapping_standalone_epg_programme(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=UTC)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)

    content = build_filtered_xmltv(
        database,
        [_stream(1, "allowed.channel")],
        now=now,
    )
    titles = [
        programme.findtext("title")
        for programme in ET.fromstring(content).findall("programme")
    ]

    assert titles == ["Allowed Programme", "Standalone EPG Programme"]


def test_filtered_xmltv_preferred_programme_replaces_all_overlaps(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=UTC)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO channels (id, name, source_id) VALUES (?, ?, ?)",
        ("split.channel", "Split Channel", "source-a"),
    )
    connection.executemany(
        "INSERT INTO programs (channel_id, title, start_ts, stop_ts, desc, source_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                "split.channel",
                "External Part One",
                now.timestamp(),
                (now + timedelta(minutes=30)).timestamp(),
                "",
                "epg-source",
            ),
            (
                "split.channel",
                "External Part Two",
                (now + timedelta(minutes=30)).timestamp(),
                (now + timedelta(hours=1)).timestamp(),
                "",
                "epg-source",
            ),
            (
                "split.channel",
                "Preferred Full Programme",
                (now + timedelta(minutes=15)).timestamp(),
                (now + timedelta(hours=1, minutes=15)).timestamp(),
                "",
                "source-a",
            ),
        ],
    )
    connection.commit()
    connection.close()

    content = build_filtered_xmltv(
        database,
        [_stream(2, "split.channel")],
        now=now,
    )
    titles = [
        programme.findtext("title")
        for programme in ET.fromstring(content).findall("programme")
    ]

    assert titles == ["Preferred Full Programme"]


def test_filtered_xmltv_uses_stream_metadata_without_channel_row(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=UTC)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)

    content = build_filtered_xmltv(
        database,
        [_stream(3, "missing.channel")],
        now=now,
    )
    channel = ET.fromstring(content).find("channel")

    assert channel is not None
    assert channel.get("id") == "missing.channel"
    assert channel.findtext("display-name") == "Channel 3"


def test_filtered_xmltv_returns_empty_feed_without_channel_ids(tmp_path: Path):
    content = build_filtered_xmltv(
        tmp_path / "missing.db",
        [_stream(1, "")],
    )

    assert ET.fromstring(content).findall("channel") == []


def test_filtered_xmltv_requires_epg_database(tmp_path: Path):
    with pytest.raises(EpgUnavailableError, match="not ready"):
        build_filtered_xmltv(
            tmp_path / "missing.db",
            [_stream(1, "allowed.channel")],
        )
