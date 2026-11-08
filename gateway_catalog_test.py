"""Tests for the native-player gateway catalog."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from urllib import error

import hashlib
import json

import pytest

from gateway_catalog import CatalogSnapshot, GatewayCatalog, StreamIdRegistry, _safe_int

import cache
import gateway_catalog


def _source(source_type: str = "xtream", source_id: str = "source-a") -> SimpleNamespace:
    return SimpleNamespace(id=source_id, name="Provider", type=source_type, epg_enabled=False)


def _catalog(tmp_path: Path, **kwargs: Any) -> GatewayCatalog:
    return GatewayCatalog(registry=StreamIdRegistry(tmp_path / "ids.json"), **kwargs)


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    categories: list[dict[str, Any]],
    streams: list[dict[str, Any]],
    sources: list[SimpleNamespace] | None = None,
) -> MagicMock:
    monkeypatch.setattr(cache, "get_sources", lambda: sources or [_source()])
    fetch = MagicMock(return_value=(categories, streams, None, 120))
    monkeypatch.setattr(gateway_catalog, "fetch_source_live_data", fetch)
    return fetch


def test_stream_id_registry_is_stable(tmp_path: Path):
    path = tmp_path / "stream_ids.json"
    first = StreamIdRegistry(path)
    assigned = first.get_or_create_many(["source-a:live:7", "source-b:live:7"])

    second = StreamIdRegistry(path)
    assert second.get_or_create_many(["source-b:live:7", "source-a:live:7"]) == assigned
    assert assigned["source-a:live:7"] != assigned["source-b:live:7"]


def test_stream_id_registry_persists_sorted_ids_and_continues_numbering(tmp_path: Path):
    path = tmp_path / "stream_ids.json"
    path.write_text(json.dumps({"ids": {"z": 4, "a": 2}}))

    assigned = StreamIdRegistry(path).get_or_create_many(["new", "a"])

    assert assigned == {"new": 5, "a": 2}
    assert json.loads(path.read_text()) == {"ids": {"a": 2, "new": 5, "z": 4}}
    assert list(json.loads(path.read_text())["ids"]) == ["a", "new", "z"]


def test_stream_id_registry_defaults_to_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    registry = StreamIdRegistry()

    assert registry.path == tmp_path / "gateway_stream_ids.json"
    assert registry.get_or_create_many(["k"]) == {"k": 1}
    assert json.loads((tmp_path / "gateway_stream_ids.json").read_text()) == {"ids": {"k": 1}}


def test_stream_id_registry_skips_save_when_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    registry = StreamIdRegistry(tmp_path / "ids.json")
    registry.get_or_create_many(["k"])
    write = MagicMock()
    monkeypatch.setattr(gateway_catalog, "atomic_write_json", write)

    assert registry.get_or_create_many(["k"]) == {"k": 1}
    write.assert_not_called()


def test_stream_id_registry_loads_file_only_once(tmp_path: Path):
    path = tmp_path / "ids.json"
    registry = StreamIdRegistry(path)
    registry.get_or_create_many(["k"])
    path.write_text("not json")

    assert registry.get_or_create_many(["k", "m"]) == {"k": 1, "m": 2}


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "{",
        json.dumps({"ids": []}),
        json.dumps({"ids": {"a": True}}),
        json.dumps({"ids": {"a": 0}}),
        json.dumps({"ids": {"a": "1"}}),
        json.dumps({"ids": {"a": 1, "b": 1}}),
    ],
)
def test_stream_id_registry_rejects_invalid_existing_file(tmp_path: Path, content: str):
    path = tmp_path / "stream_ids.json"
    path.write_text(content)

    with pytest.raises(RuntimeError, match="Invalid gateway stream ID registry"):
        StreamIdRegistry(path).get_or_create_many(["source-a:live:7"])

    assert path.read_text() == content


def test_stream_id_registry_rejects_unreadable_file(tmp_path: Path):
    path = tmp_path / "stream_ids.json"
    path.mkdir()

    with pytest.raises(RuntimeError, match="Invalid gateway stream ID registry"):
        StreamIdRegistry(path).get_or_create_many(["k"])


def test_catalog_uses_default_registry():
    catalog = GatewayCatalog()

    assert isinstance(catalog._registry, StreamIdRegistry)


def test_catalog_removes_upstream_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _serve(
        monkeypatch,
        [{"category_id": "source-a_9", "category_name": "News", "parent_id": 0}],
        [
            {
                "stream_id": 7,
                "name": "News",
                "category_ids": ["source-a_9"],
                "source_username": "remote-user",
                "source_password": "remote-password",
                "source_url": "https://provider.example",
            },
        ],
    )
    snapshot = _catalog(tmp_path, ttl_seconds=60).get()

    encoded = json.dumps(snapshot.streams[0].public)
    assert snapshot.categories[0]["category_id"].isdigit()
    assert snapshot.streams[0].public["category_id"] == snapshot.categories[0]["category_id"]
    assert "remote-user" not in encoded
    assert "remote-password" not in encoded
    assert "provider.example" not in encoded


def test_catalog_maps_full_public_stream_and_access_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fetch = _serve(
        monkeypatch,
        [
            {"category_id": "news", "category_name": "News"},
            {"category_id": "kids"},
            {"category_id": None, "category_name": "Ignored"},
            {"category_id": "", "category_name": "Ignored too"},
        ],
        [
            {
                "stream_id": 7,
                "num": 3,
                "name": "News HD",
                "stream_icon": "http://icon",
                "epg_channel_id": "news.hd",
                "added": 1_700_000_000,
                "category_ids": ["news", None, "", "extra"],
                "custom_sid": "sid",
                "tv_archive": "1",
                "tv_archive_duration": "bad",
            },
            {"stream_id": None, "name": "No id", "category_ids": ["news"]},
            {"stream_id": "", "name": "Empty id", "category_ids": ["news"]},
        ],
    )

    snapshot = _catalog(tmp_path).get()

    fetch.assert_called_once_with(_source(), persist_epg_url=False)
    by_name = {category["category_name"]: category for category in snapshot.categories}
    assert set(by_name) == {"News", "Uncategorized"}
    assert by_name["News"]["parent_id"] == 0
    news_id = by_name["News"]["category_id"]
    kids_id = by_name["Uncategorized"]["category_id"]
    assert snapshot.category_access_ids[news_id] == "news"
    assert snapshot.category_access_ids[kids_id] == "kids"
    assert set(snapshot.category_access_ids.values()) == {"news", "kids", "extra"}
    assert len(snapshot.streams) == 1
    stream = snapshot.streams[0]
    extra_id = next(k for k, v in snapshot.category_access_ids.items() if v == "extra")
    assert stream.source_id == "source-a"
    assert stream.source_type == "xtream"
    assert stream.upstream_id == "7"
    assert stream.direct_url == ""
    assert stream.access_group_ids == ("news", "None", "extra")
    assert stream.public == {
        "num": 3,
        "name": "News HD",
        "stream_type": "live",
        "stream_id": stream.local_id,
        "stream_icon": "http://icon",
        "epg_channel_id": "news.hd",
        "added": "1700000000",
        "category_id": news_id,
        "category_ids": [news_id, extra_id],
        "custom_sid": "sid",
        "tv_archive": 1,
        "tv_archive_duration": 0,
    }
    assert snapshot.streams_by_id == {stream.local_id: stream}


def test_catalog_falls_back_to_primary_category_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _serve(
        monkeypatch,
        [{"category_id": "news", "category_name": "News"}],
        [
            {"stream_id": 1, "category_id": "news", "category_ids": []},
            {"stream_id": 2, "category_id": "missing"},
        ],
    )

    snapshot = _catalog(tmp_path).get()

    names = {c["category_id"]: c["category_name"] for c in snapshot.categories}
    first, second = snapshot.streams
    assert names[first.public["category_id"]] == "News"
    assert first.public["category_ids"] == [first.public["category_id"]]
    assert first.access_group_ids == ("news",)
    assert first.public["name"] == "Unknown"
    assert first.public["num"] == first.local_id
    assert names[second.public["category_id"]] == "Uncategorized"
    assert second.access_group_ids == ()
    assert second.public["category_id"] not in snapshot.category_access_ids


def test_catalog_assigns_uncategorized_streams_numeric_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _serve(monkeypatch, [], [{"stream_id": 7, "name": "Uncategorized", "category_ids": []}])
    snapshot = _catalog(tmp_path, ttl_seconds=60).get()

    category_id = snapshot.categories[0]["category_id"]
    assert category_id.isdigit()
    assert snapshot.streams[0].public["category_id"] == category_id
    assert snapshot.streams[0].public["category_ids"] == [category_id]


def test_catalog_skips_unsupported_source_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fetch = _serve(monkeypatch, [], [], sources=[_source("vod"), _source("m3u", "m")])

    snapshot = _catalog(tmp_path).get()

    fetch.assert_called_once_with(_source("m3u", "m"), persist_epg_url=False)
    assert snapshot == CatalogSnapshot(
        categories=[],
        category_access_ids={},
        streams=[],
        streams_by_id={},
    )


def test_m3u_registry_key_uses_url_instead_of_playlist_position(tmp_path: Path):
    catalog = _catalog(tmp_path, ttl_seconds=60)
    first = catalog._stream_registry_key(
        "source-a",
        "m3u",
        {
            "stream_id": "source-a_1",
            "direct_url": "https://streams.example/news?token=first",
            "epg_channel_id": "news.example",
            "name": "News",
        },
    )
    reordered = catalog._stream_registry_key(
        "source-a",
        "m3u",
        {
            "stream_id": "source-a_42",
            "direct_url": "https://streams.example/news?token=rotated",
            "epg_channel_id": "news.example",
            "name": "News",
        },
    )
    different = catalog._stream_registry_key(
        "source-a",
        "m3u",
        {
            "stream_id": "source-a_1",
            "direct_url": "https://streams.example/sports",
            "epg_channel_id": "sports.example",
            "name": "Sports",
        },
    )

    assert first == reordered
    assert first != different


def test_registry_key_formats():
    assert GatewayCatalog._stream_registry_key("s", "xtream", {"stream_id": 9}) == "s:live:9"
    digest = hashlib.sha256(b"s_3").hexdigest()
    assert GatewayCatalog._stream_registry_key("s", "m3u", {"stream_id": "s_3"}) == (
        f"s:live-url:{digest}"
    )


def test_catalog_assigns_distinct_ids_to_colliding_m3u_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _serve(
        monkeypatch,
        [],
        [
            {
                "stream_id": "source-a_1",
                "name": "News",
                "direct_url": "https://streams.example/play?id=1",
                "category_ids": [],
            },
            {
                "stream_id": "source-a_2",
                "name": "News",
                "direct_url": "https://streams.example/play?id=2",
                "category_ids": [],
            },
        ],
        sources=[_source("m3u")],
    )

    snapshot = _catalog(tmp_path, ttl_seconds=60).get()

    assert len(snapshot.streams) == 2
    assert len(snapshot.streams_by_id) == 2
    assert snapshot.streams[0].local_id != snapshot.streams[1].local_id
    assert snapshot.streams[1].direct_url == "https://streams.example/play?id=2"


def test_map_stream_rejects_missing_upstream_id(tmp_path: Path):
    catalog = _catalog(tmp_path)

    assert catalog._map_stream("s", "xtream", {"stream_id": ""}, 1, {}, "u", "") is None


def test_map_stream_without_any_category(tmp_path: Path):
    stream = _catalog(tmp_path)._map_stream("s", "xtream", {"stream_id": 4}, 1, {}, "u", "")

    assert stream is not None
    assert stream.public["category_id"] == ""
    assert stream.public["category_ids"] == []
    assert stream.access_group_ids == ()


def test_catalog_get_returns_cached_snapshot_within_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fetch = _serve(monkeypatch, [], [{"stream_id": 1}])
    catalog = _catalog(tmp_path, ttl_seconds=60)

    first = catalog.get()

    assert catalog.get() is first
    assert fetch.call_count == 1
    assert catalog.get(force=True) is not first
    assert fetch.call_count == 2


def test_catalog_get_reloads_after_ttl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fetch = _serve(monkeypatch, [], [{"stream_id": 1}])
    catalog = _catalog(tmp_path, ttl_seconds=0)

    catalog.get()
    catalog.get()

    assert fetch.call_count == 2


def test_catalog_get_rechecks_cache_after_acquiring_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fetch = _serve(monkeypatch, [], [])
    catalog = _catalog(tmp_path, ttl_seconds=60)
    loaded_by_other_thread = CatalogSnapshot(
        categories=[],
        category_access_ids={},
        streams=[],
        streams_by_id={},
    )

    class RacingLock:
        def __enter__(self) -> None:
            catalog._snapshot = loaded_by_other_thread
            catalog._loaded_at = gateway_catalog.time.monotonic()

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(catalog, "_lock", RacingLock())

    assert catalog.get() is loaded_by_other_thread
    fetch.assert_not_called()


def test_catalog_retains_previous_snapshot_when_refresh_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    fetch = _serve(
        monkeypatch,
        [{"category_id": "source-a_9", "category_name": "News", "parent_id": 0}],
        [{"stream_id": 7, "name": "News", "category_ids": ["source-a_9"]}],
    )
    catalog = _catalog(tmp_path, ttl_seconds=60)
    original = catalog.get()
    fetch.side_effect = error.URLError("temporary failure")

    refreshed = catalog.get(force=True)

    assert refreshed is original
    assert len(refreshed.streams) == 1
    assert "Gateway failed to load source Provider (URLError)" in caplog.text
    assert "retaining the previous snapshot" in caplog.text
    assert catalog.get() is original
    assert fetch.call_count == 2


def test_catalog_failed_initial_load_retries_after_short_delay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fetch = _serve(monkeypatch, [], [])
    fetch.side_effect = KeyError("bad")
    catalog = _catalog(tmp_path, ttl_seconds=60, failed_load_retry_seconds=30)

    empty = catalog.get()

    assert empty.streams == []
    assert catalog.get() is empty
    assert fetch.call_count == 1

    catalog = _catalog(tmp_path, ttl_seconds=60, failed_load_retry_seconds=0)
    catalog.get()
    catalog.get()
    assert fetch.call_count == 3


@pytest.mark.parametrize(("value", "expected"), [("5", 5), (3.9, 3), (None, 0), ("x", 0)])
def test_safe_int(value: object, expected: int):
    assert _safe_int(value) == expected


def test_safe_int_custom_default():
    assert _safe_int("x", default=-1) == -1


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
