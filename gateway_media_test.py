"""Tests for movie and series gateway catalogs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import json
import os

import pytest

from gateway_catalog import StreamIdRegistry
from gateway_media import GatewayMediaCatalog

import gateway_media


def test_movie_catalog_assigns_stable_ids_across_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    categories = [
        {"source_id": "source-a", "category_id": "7", "category_name": "Movies A"},
        {"source_id": "source-b", "category_id": "7", "category_name": "Movies B"},
    ]
    movies = [
        {
            "source_id": "source-a",
            "stream_id": 10,
            "name": "Movie A",
            "category_id": "7",
            "direct_source": "https://provider.example/private-a",
        },
        {
            "source_id": "source-b",
            "stream_id": 10,
            "name": "Movie B",
            "category_id": "7",
            "source_password": "provider-secret",
        },
    ]
    monkeypatch.setattr(
        gateway_media,
        "load_vod_data",
        lambda: (deepcopy(categories), deepcopy(movies)),
    )
    path = tmp_path / "ids.db"

    first = GatewayMediaCatalog(
        "movie",
        StreamIdRegistry(path),
        ttl_seconds=60,
    ).get()
    second = GatewayMediaCatalog(
        "movie",
        StreamIdRegistry(path),
        ttl_seconds=60,
    ).get()

    assert len(first.categories) == 2
    assert first.categories[0]["category_id"] != first.categories[1]["category_id"]
    assert first.items[0].local_id != first.items[1].local_id
    assert [item.local_id for item in second.items] == [
        item.local_id for item in first.items
    ]
    encoded = json.dumps([item.public for item in first.items])
    assert "provider.example" not in encoded
    assert "provider-secret" not in encoded


def test_series_info_remaps_episode_ids_and_removes_private_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        gateway_media,
        "load_series_data",
        lambda: (
            [{"source_id": "source-a", "category_id": "9", "category_name": "Drama"}],
            [
                {
                    "source_id": "source-a",
                    "series_id": 20,
                    "name": "A Show",
                    "category_id": "9",
                }
            ],
        ),
    )
    registry = StreamIdRegistry(tmp_path / "ids.db")
    catalog = GatewayMediaCatalog("series", registry, ttl_seconds=60)
    series = catalog.get().items[0]
    raw = {
        "info": {
            "name": "A Show",
            "series_id": 20,
            "category_id": "9",
            "source_password": "provider-secret",
        },
        "episodes": {
            "1": [
                {
                    "id": 30,
                    "title": "Pilot",
                    "container_extension": "mkv",
                    "direct_source": "https://provider.example/private",
                }
            ]
        },
    }

    result = catalog.remap_info(series, raw)

    local_episode_id = result["episodes"]["1"][0]["id"]
    assert result["info"]["series_id"] == series.local_id
    assert result["info"]["category_id"] == series.public["category_id"]
    assert local_episode_id != 30
    assert catalog.resolve_registered_id(
        local_episode_id,
        "series-episode",
    ) == ("source-a", "30")
    assert "provider-secret" not in json.dumps(result)
    assert "provider.example" not in json.dumps(result)
    assert raw["episodes"]["1"][0]["id"] == 30


def test_catalog_reloads_when_shared_cache_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cache_file = tmp_path / "vod_data.json"
    cache_file.write_text("{}")
    monkeypatch.setattr(gateway_media.cache, "CACHE_DIR", tmp_path)
    movies = [
        {
            "source_id": "source-a",
            "stream_id": 10,
            "name": "Before",
            "category_id": "7",
        }
    ]
    monkeypatch.setattr(
        gateway_media,
        "load_vod_data",
        lambda: (
            [{"source_id": "source-a", "category_id": "7", "category_name": "Movies"}],
            deepcopy(movies),
        ),
    )
    catalog = GatewayMediaCatalog(
        "movie",
        StreamIdRegistry(tmp_path / "ids.db"),
        ttl_seconds=3600,
    )
    assert catalog.get().items[0].public["name"] == "Before"

    movies[0]["name"] = "After"
    current_mtime = cache_file.stat().st_mtime_ns
    os.utime(cache_file, ns=(current_mtime + 1_000_000, current_mtime + 1_000_000))

    assert catalog.get().items[0].public["name"] == "After"


def test_uncategorized_items_receive_a_public_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        gateway_media,
        "load_vod_data",
        lambda: (
            [],
            [
                {
                    "source_id": "source-a",
                    "stream_id": 10,
                    "name": "Uncategorized",
                    "category_id": "missing",
                }
            ],
        ),
    )
    catalog = GatewayMediaCatalog(
        "movie",
        StreamIdRegistry(tmp_path / "ids.db"),
        ttl_seconds=60,
    )

    snapshot = catalog.get()

    assert snapshot.categories[0]["category_name"] == "Uncategorized"
    assert snapshot.items[0].public["category_id"] == snapshot.categories[0]["category_id"]


def test_registered_id_resolution_rejects_other_id_kinds(tmp_path: Path):
    registry = StreamIdRegistry(tmp_path / "ids.db")
    ids = registry.get_or_create_many(
        [
            "source-a:live:10",
            "source-a:movie-category:20",
            "source-a:series:30",
            "source-a:series-episode:40",
        ]
    )
    movies = GatewayMediaCatalog("movie", registry)
    series = GatewayMediaCatalog("series", registry)

    assert movies.resolve_registered_id(ids["source-a:live:10"]) is None
    assert movies.resolve_registered_id(ids["source-a:movie-category:20"]) is None
    assert series.resolve_registered_id(ids["source-a:series:30"], "series-episode") is None
    assert series.resolve_registered_id(
        ids["source-a:series-episode:40"],
        "series-episode",
    ) == ("source-a", "40")
