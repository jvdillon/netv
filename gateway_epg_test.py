"""Tests for filtered native-player XMLTV output."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlite3
import xml.etree.ElementTree as ET

import pytest

from gateway_catalog import GatewayStream
from gateway_epg import EpgUnavailableError, build_filtered_xmltv


def test_filtered_xmltv_contains_only_allowed_current_programs(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=timezone.utc)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)

    content = build_filtered_xmltv(
        database,
        [_stream(1, "allowed.channel")],
        now=now,
    )
    root = ET.fromstring(content)

    assert [channel.get("id") for channel in root.findall("channel")] == ["allowed.channel"]
    assert [programme.findtext("title") for programme in root.findall("programme")] == [
        "Allowed Programme",
        "Standalone EPG Programme",
    ]
    assert b"blocked.channel" not in content
    assert b"Wrong Source Programme" not in content
    assert b"Expired Programme" not in content


def test_filtered_xmltv_accepts_nonoverlapping_standalone_epg_programme(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=timezone.utc)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)

    content = build_filtered_xmltv(
        database,
        [_stream(1, "allowed.channel")],
        now=now,
    )
    titles = [
        programme.findtext("title") for programme in ET.fromstring(content).findall("programme")
    ]

    assert titles == ["Allowed Programme", "Standalone EPG Programme"]


def test_filtered_xmltv_preferred_programme_replaces_all_overlaps(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=timezone.utc)
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
        programme.findtext("title") for programme in ET.fromstring(content).findall("programme")
    ]

    assert titles == ["Preferred Full Programme"]


def test_filtered_xmltv_uses_stream_metadata_without_channel_row(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=timezone.utc)
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


def test_filtered_xmltv_uses_database_channel_name_and_icon(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=timezone.utc)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO channels (id, name, source_id) VALUES (?, ?, ?)",
        ("nameless.channel", None, "source-a"),
    )
    connection.execute(
        "INSERT INTO icons (channel_id, url) VALUES (?, ?)",
        ("allowed.channel", "https://epg.example/allowed.png"),
    )
    connection.commit()
    connection.close()

    content = build_filtered_xmltv(
        database,
        [_stream(1, "allowed.channel"), _stream(2, "nameless.channel")],
        now=now,
    )
    channels = {channel.get("id"): channel for channel in ET.fromstring(content).findall("channel")}

    allowed = channels["allowed.channel"]
    assert allowed.findtext("display-name") == "Allowed Channel"
    icon = allowed.find("icon")
    assert icon is not None
    assert icon.get("src") == "https://epg.example/allowed.png"
    nameless = channels["nameless.channel"]
    assert nameless.findtext("display-name") == "nameless.channel"
    nameless_icon = nameless.find("icon")
    assert nameless_icon is not None
    assert nameless_icon.get("src") == "https://images.example/2.png"


def test_filtered_xmltv_merges_streams_sharing_channel(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=timezone.utc)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)

    content = build_filtered_xmltv(
        database,
        [_stream(1, "allowed.channel"), _stream(2, "allowed.channel", source_id="source-b")],
        now=now,
    )
    root = ET.fromstring(content)

    assert [channel.get("id") for channel in root.findall("channel")] == ["allowed.channel"]
    titles = [programme.findtext("title") for programme in root.findall("programme")]
    assert titles == ["Allowed Programme", "Standalone EPG Programme"]


def test_filtered_xmltv_omits_empty_icon_and_strips_control_characters(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=timezone.utc)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO programs (channel_id, title, start_ts, stop_ts, desc, source_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "bare.channel",
            None,
            now.timestamp(),
            (now + timedelta(hours=1)).timestamp(),
            "line\x01one\ttab",
            "source-a",
        ),
    )
    connection.commit()
    connection.close()
    stream = GatewayStream(
        local_id=9,
        source_id="source-a",
        source_type="xtream",
        upstream_id="9",
        direct_url="",
        access_group_ids=(),
        public={"epg_channel_id": "bare.channel"},
    )

    root = ET.fromstring(build_filtered_xmltv(database, [stream], now=now))

    channel = root.find("channel")
    assert channel is not None
    assert channel.findtext("display-name") == "bare.channel"
    assert channel.find("icon") is None
    programme = root.find("programme")
    assert programme is not None
    assert programme.get("start") == "20260829180000 +0000"
    assert programme.get("stop") == "20260829190000 +0000"
    assert programme.findtext("title") == "Unknown"
    assert programme.findtext("desc") == "lineone\ttab"


def test_filtered_xmltv_ignores_rows_for_unrequested_channel_ids(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=timezone.utc)
    database = tmp_path / "epg.db"
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE channels (id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE icons (channel_id TEXT PRIMARY KEY, url TEXT);
        CREATE TABLE programs (
            channel_id TEXT COLLATE NOCASE,
            title TEXT,
            start_ts REAL,
            stop_ts REAL,
            desc TEXT,
            source_id TEXT
        );
    """)
    connection.execute(
        "INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?)",
        (
            "ALLOWED.CHANNEL",
            "Case Mismatch",
            now.timestamp(),
            (now + timedelta(hours=1)).timestamp(),
            "",
            "source-a",
        ),
    )
    connection.commit()
    connection.close()

    root = ET.fromstring(build_filtered_xmltv(database, [_stream(1, "allowed.channel")], now=now))

    assert [channel.get("id") for channel in root.findall("channel")] == ["allowed.channel"]
    assert root.findall("programme") == []


def test_filtered_xmltv_queries_large_channel_sets_in_chunks(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=timezone.utc)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)
    streams = [_stream(index, f"filler.{index}") for index in range(450)]
    streams.append(_stream(999, "allowed.channel"))

    root = ET.fromstring(build_filtered_xmltv(database, streams, now=now))

    assert len(root.findall("channel")) == 451
    titles = [programme.findtext("title") for programme in root.findall("programme")]
    assert titles == ["Allowed Programme", "Standalone EPG Programme"]


def test_filtered_xmltv_opens_path_with_uri_metacharacters_read_only(tmp_path: Path):
    now = datetime(2026, 8, 29, 18, tzinfo=timezone.utc)
    directory = tmp_path / "a#b?c%d"
    directory.mkdir()
    database = directory / "epg.db"
    _create_epg_database(database, now)

    content = build_filtered_xmltv(database, [_stream(1, "allowed.channel")], now=now)

    assert b"Allowed Programme" in content
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a#b?c%d"]


def test_filtered_xmltv_reports_unopenable_database(tmp_path: Path):
    database = tmp_path / "epg.db"
    database.mkdir()

    with pytest.raises(EpgUnavailableError, match="not available"):
        build_filtered_xmltv(database, [_stream(1, "allowed.channel")])


def test_filtered_xmltv_reports_missing_schema(tmp_path: Path):
    database = tmp_path / "epg.db"
    sqlite3.connect(database).close()

    with pytest.raises(EpgUnavailableError, match="schema is not ready"):
        build_filtered_xmltv(database, [_stream(1, "allowed.channel")])


def test_filtered_xmltv_defaults_window_to_current_time(tmp_path: Path):
    now = datetime.now(timezone.utc)
    database = tmp_path / "epg.db"
    _create_epg_database(database, now)

    root = ET.fromstring(build_filtered_xmltv(database, [_stream(1, "allowed.channel")]))

    assert [programme.findtext("title") for programme in root.findall("programme")] == [
        "Allowed Programme",
        "Standalone EPG Programme",
    ]


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


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
