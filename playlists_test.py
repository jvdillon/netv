"""Tests for playlists.py - shared Live TV playlists."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import playlists


@pytest.fixture(autouse=True)
def cache_dir(tmp_path: Path):
    with patch("cache.CACHE_DIR", tmp_path):
        yield tmp_path


STREAMS = [
    {"stream_id": 1, "name": "US| ESPN HD", "source_id": "a", "epg_channel_id": "espn.us", "category_ids": ["10"]},
    {"stream_id": 2, "name": "US| FOX SPORTS 1 HD", "source_id": "a", "epg_channel_id": "", "category_ids": ["10"]},
    {"stream_id": 3, "name": "US| ABC 7 SAN FRANCISCO CA (KGO)", "source_id": "a", "epg_channel_id": "", "category_ids": ["20"]},
    {"stream_id": 4, "name": "UK| ESPN", "source_id": "a", "epg_channel_id": "", "category_ids": ["30"]},
    {"stream_id": 5, "name": "##### SPORTS #####", "source_id": "a", "epg_channel_id": "", "category_ids": ["10"]},
]  # fmt: skip
CATEGORY_NAMES = {"10": "US| SPORTS", "20": "US| ABC NETWORK", "30": "UK| SPORTS"}


def test_crud_and_order():
    first = playlists.create("Favorites")
    second = playlists.create("Kids")
    assert [p["name"] for p in playlists.load()] == ["Favorites", "Kids"]

    updated = playlists.update(
        first["id"],
        name="  YTTV  ",
        channels=[
            {"stream_id": 1, "name": "US| ESPN HD", "source_id": "a"},
            {"stream_id": "1", "name": "US| ESPN HD", "source_id": "a"},
            "garbage",
            {},
        ],
    )
    assert updated is not None
    assert updated["name"] == "YTTV"
    assert [c["stream_id"] for c in updated["channels"]] == ["1"]

    playlists.reorder([second["id"], first["id"]])
    assert [p["id"] for p in playlists.load()] == [second["id"], first["id"]]

    assert playlists.delete(second["id"])
    assert not playlists.delete(second["id"])
    assert playlists.update("missing", name="x") is None


def test_create_requires_name():
    with pytest.raises(ValueError):
        playlists.create("   ")


def test_resolve_survives_stream_id_changes():
    playlist = {
        "channels": [
            {"stream_id": "999", "name": "US| ESPN HD", "source_id": "a", "epg_channel_id": ""},
            {"stream_id": "3", "name": "US| ABC 7 SAN FRANCISCO CA (KGO)", "source_id": "a"},
            {"stream_id": "888", "name": "Renamed", "source_id": "a", "epg_channel_id": "espn.us"},
            {"stream_id": "777", "name": "Gone", "source_id": "a", "epg_channel_id": ""},
        ]
    }
    resolved = playlists.resolve(playlist, STREAMS)
    # ESPN resolves by name, KGO by id; the EPG-id fallback duplicates ESPN and is dropped.
    assert [s["stream_id"] for s in resolved] == [1, 3]


def test_resolve_prefers_name_when_stream_id_was_reused():
    playlist = {"channels": [{"stream_id": "2", "name": "US| ESPN HD", "source_id": "a"}]}
    assert [s["stream_id"] for s in playlists.resolve(playlist, STREAMS)] == [1]


def test_match_names_uses_aliases_and_category_prefix():
    matches = playlists.match_names(
        ["FS1", "KGO", "ESPN", "Nope Channel"], STREAMS, CATEGORY_NAMES, "US|"
    )
    best = {
        m["query"]: (m["candidates"][0]["stream_id"] if m["candidates"] else None) for m in matches
    }
    assert best == {"FS1": "2", "KGO": "3", "ESPN": "1", "Nope Channel": None}
    espn = next(m for m in matches if m["query"] == "ESPN")
    assert all(c["category"].startswith("US|") for c in espn["candidates"])


def test_search_skips_separators_and_filters():
    results = playlists.search(STREAMS, CATEGORY_NAMES, query="espn")
    assert [r["stream_id"] for r in results] == ["1", "4"]
    assert playlists.search(STREAMS, CATEGORY_NAMES, category_id="10") == [
        playlists.summarize(STREAMS[0], CATEGORY_NAMES),
        playlists.summarize(STREAMS[1], CATEGORY_NAMES),
    ]


def test_resolve_keeps_sources_apart():
    streams = [
        {"stream_id": 7, "name": "CNN", "source_id": "a"},
        {"stream_id": 7, "name": "BBC", "source_id": "b"},
    ]
    playlist = {
        "channels": [
            {"stream_id": "7", "name": "BBC", "source_id": "b"},
            {"stream_id": "7", "name": "CNN", "source_id": "a"},
        ]
    }
    assert [s["name"] for s in playlists.resolve(playlist, streams)] == ["BBC", "CNN"]


def test_resolve_entry_epg_overrides_stream():
    streams = [
        {"stream_id": 1, "name": "Alpha", "source_id": "a", "epg_channel_id": ""},
        {"stream_id": 2, "name": "Beta", "source_id": "a", "epg_channel_id": "beta.guide"},
    ]
    playlist = {
        "channels": [
            {"stream_id": "1", "name": "Alpha", "source_id": "a", "epg_channel_id": "alpha.guide"},
            {"stream_id": "2", "name": "Beta", "source_id": "a", "epg_channel_id": "old.beta"},
        ]
    }
    resolved = playlists.resolve(playlist, streams)
    assert [s["epg_channel_id"] for s in resolved] == ["alpha.guide", "old.beta"]
    assert streams[0]["epg_channel_id"] == ""


def test_resolve_prefers_epg_over_reused_id():
    streams = [
        {"stream_id": 1, "name": "Other", "source_id": "a", "epg_channel_id": ""},
        {"stream_id": 2, "name": "ESPN East", "source_id": "a", "epg_channel_id": "espn.us"},
    ]
    entry = {"stream_id": "1", "name": "ESPN", "source_id": "a", "epg_channel_id": "espn.us"}
    stream = playlists.resolve_channel(entry, streams)
    assert stream is not None and stream["stream_id"] == 2


def test_index_rebuilds_for_new_stream_list():
    entry = {"stream_id": "1", "name": "A", "source_id": "s"}
    for name in ("A", "B"):
        stream = playlists.resolve_channel(entry, [{"stream_id": 1, "name": name, "source_id": "s"}])
        assert stream is not None and stream["name"] == name
