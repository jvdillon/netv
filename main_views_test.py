"""Tests for main.py routes and helpers from the settings refresh API onward."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict
from unittest.mock import MagicMock, patch
from urllib import error

import asyncio
import contextlib
import logging
import pathlib
import runpy
import sys
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient
from treekle import MutablePlainTree

import pytest

from cache import Source
from epg import Program
from main import PlayerInfo
from xtream import ProviderRecord

import auth
import cache as cache_module
import handoff
import m3u as m3u_module
import main
import main_test


mock_deps = main_test.mock_deps
auth_client = main_test.auth_client
clean_handoff = main_test.clean_handoff


class _SyncThread:
    """Thread stand-in that runs its target on start."""

    def __init__(self, target: Callable[[], None], daemon: bool = False):
        del daemon
        self.target = target

    def start(self) -> None:
        self.target()


class _RecordingThread:
    """Thread stand-in that only records what would have started."""

    started: list[Callable[[], None]] = []

    def __init__(self, target: Callable[[], None], daemon: bool = False):
        del daemon
        self.target = target

    def start(self) -> None:
        _RecordingThread.started.append(self.target)


@pytest.fixture
def sync_threads() -> Iterator[None]:
    with patch("main.threading", SimpleNamespace(Thread=_SyncThread)):
        yield


@pytest.fixture
def recorded_threads() -> Iterator[list[Callable[[], None]]]:
    _RecordingThread.started = []
    with patch("main.threading", SimpleNamespace(Thread=_RecordingThread)):
        yield _RecordingThread.started


@pytest.fixture
def refresh_flags() -> Iterator[set[str]]:
    flags = m3u_module.get_refresh_in_progress()
    flags.clear()
    yield flags
    flags.clear()


def _login_as(client: TestClient, username: str, password: str = "password123") -> None:
    auth.create_user(username, password)
    client.cookies.set("token", auth.create_token({"sub": username}))


def _source(source_id: str, source_type: str, **kwargs: Any) -> Source:
    return Source(id=source_id, name=source_id.upper(), type=source_type, url="http://up", **kwargs)


# ===========================================================================
# Guide and per-source refresh
# ===========================================================================


def test_guide_refresh_starts_each_refresh_once(
    auth_client: TestClient,
    refresh_flags: set[str],
    recorded_threads: list[Callable[[], None]],
):
    response = auth_client.get("/guide/refresh", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/guide?refreshing=1"
    assert refresh_flags == {"live_refresh", "epg_refresh"}
    assert len(recorded_threads) == 2

    auth_client.get("/guide/refresh", follow_redirects=False)
    assert len(recorded_threads) == 2


def test_source_refresh_unknown_source(auth_client: TestClient):
    with patch("main.get_sources", return_value=[]):
        assert auth_client.post("/settings/refresh/nope/live").json() == {
            "error": "Source not found",
        }


def test_source_refresh_already_running(auth_client: TestClient, refresh_flags: set[str]):
    refresh_flags.add("s1_live")
    with patch("main.get_sources", return_value=[_source("s1", "xtream")]):
        assert auth_client.post("/settings/refresh/s1/live").json() == {
            "status": "already_running",
        }


@pytest.mark.parametrize(("epg_url", "expected_epg"), [("http://g", True), ("", False)])
def test_source_refresh_live_replaces_only_that_source(
    auth_client: TestClient,
    refresh_flags: set[str],
    sync_threads: None,
    epg_url: str,
    expected_epg: bool,
):
    del sync_threads
    store = cache_module.get_cache()
    store["live_categories"] = [{"category_id": "o", "source_id": "other"}, {"source_id": "s1"}]
    store["live_streams"] = [{"stream_id": "o", "source_id": "other"}, {"source_id": "s1"}]
    store["epg_urls"] = [("http://o", 10, "other"), ("http://old", 10, "s1")]
    new_cat = {"category_id": "n", "source_id": "s1"}
    new_stream = {"stream_id": "n", "source_id": "s1"}
    with (
        patch("main.get_sources", return_value=[_source("s1", "xtream")]),
        patch(
            "main.fetch_source_live_data",
            return_value=([new_cat], [new_stream], epg_url, 30),
        ),
        patch("main.save_file_cache") as save,
    ):
        response = auth_client.post("/settings/refresh/s1/live")
    assert response.json() == {"status": "started", "key": "s1_live"}
    assert store["live_categories"] == [{"category_id": "o", "source_id": "other"}, new_cat]
    assert store["live_streams"] == [{"stream_id": "o", "source_id": "other"}, new_stream]
    expected = [("http://o", 10, "other")] + ([("http://g", 30, "s1")] if expected_epg else [])
    assert store["epg_urls"] == expected
    assert save.call_args.args[0] == "live_data"
    assert "s1_live" not in refresh_flags


@pytest.mark.parametrize(
    ("source", "fetched_url"),
    [
        (_source("s1", "xtream", epg_url="http://guide"), "http://guide"),
        (_source("s1", "epg"), "http://up"),
        (_source("s1", "m3u"), None),
    ],
)
def test_source_refresh_epg(
    auth_client: TestClient,
    refresh_flags: set[str],
    sync_threads: None,
    source: Source,
    fetched_url: str | None,
):
    del sync_threads
    with (
        patch("main.get_sources", return_value=[source]),
        patch("epg.clear_source") as clear,
        patch("main._fetch_all_epg", return_value=5) as fetch,
    ):
        auth_client.post("/settings/refresh/s1/epg")
    if fetched_url is not None:
        clear.assert_called_once_with("s1")
        fetch.assert_called_once_with([(fetched_url, source.epg_timeout, "s1")])
    else:
        fetch.assert_not_called()
    assert not refresh_flags


def test_source_refresh_vod_merges_and_drops_file_cache(
    auth_client: TestClient,
    refresh_flags: set[str],
    sync_threads: None,
    tmp_path: Path,
):
    del sync_threads, refresh_flags
    stale = tmp_path / "vod_data_x.json"
    stale.write_text("{}")
    cache_module.get_cache()["vod_streams"] = [{"stream_id": "stale"}]
    existing = (
        [{"category_id": "o", "source_id": "other"}, {"category_id": "x", "source_id": "s1"}],
        [{"stream_id": "o", "source_id": "other"}, {"stream_id": "x", "source_id": "s1"}],
    )
    with (
        patch("main.get_sources", return_value=[_source("s1", "xtream")]),
        patch("main.fetch_source_vod_data", return_value=([{"category_id": "n"}], [{"id": 1}])),
        patch("main.load_vod_data", return_value=existing),
        patch("main.save_file_cache") as save,
    ):
        auth_client.post("/settings/refresh/s1/vod")
    assert not stale.exists()
    assert "vod_streams" not in cache_module.get_cache()
    save.assert_called_once_with(
        "vod_data",
        {
            "cats": [{"category_id": "o", "source_id": "other"}, {"category_id": "n"}],
            "streams": [{"stream_id": "o", "source_id": "other"}, {"id": 1}],
        },
    )


def test_source_refresh_m3u(auth_client: TestClient, refresh_flags: set[str], sync_threads: None):
    del sync_threads, refresh_flags
    store = cache_module.get_cache()
    store["live_categories"] = [{"source_id": "s1"}]
    store["live_streams"] = [{"source_id": "s1"}, {"source_id": "other"}]
    store["epg_urls"] = [("http://o", 10, "other")]
    with (
        patch("main.get_sources", return_value=[_source("s1", "m3u")]),
        patch("main.fetch_m3u", return_value=([{"c": 1}], [{"s": 1}], "http://detected")),
        patch("main.update_source_epg_url") as update,
        patch("main.save_file_cache") as save,
    ):
        auth_client.post("/settings/refresh/s1/m3u")
    update.assert_called_once_with("s1", "http://detected")
    assert store["live_categories"] == [{"c": 1}]
    assert store["live_streams"] == [{"source_id": "other"}, {"s": 1}]
    assert save.call_args.args[1]["epg_urls"] == [("http://o", 10, "other")]


def test_source_refresh_failure_clears_flag(
    auth_client: TestClient,
    refresh_flags: set[str],
    sync_threads: None,
    caplog: pytest.LogCaptureFixture,
):
    del sync_threads
    with (
        patch("main.get_sources", return_value=[_source("s1", "xtream")]),
        patch("main.fetch_source_live_data", side_effect=RuntimeError("boom")),
        caplog.at_level(logging.ERROR, logger="main"),
    ):
        auth_client.post("/settings/refresh/s1/live")
    assert "boom" in caplog.text
    assert not refresh_flags


def test_settings_refresh_status_groups_keys(auth_client: TestClient, refresh_flags: set[str]):
    refresh_flags.update({"src_1_epg", "guide_load", "plain"})
    assert auth_client.get("/settings/refresh-status").json() == {
        "src_1": {"epg": True},
        "guide": {"load": True},
        "_global": {"live": True, "epg": True},
    }


# ===========================================================================
# Cast, preferences, playlists
# ===========================================================================


def test_cast_status_is_user_scoped(auth_client: TestClient):
    with patch("chromecast.manager.status", return_value={"active": False}) as status:
        assert auth_client.get("/api/cast/status").json() == {"active": False}
    status.assert_called_once_with("testuser")


def test_user_prefs_rejects_large_body(auth_client: TestClient):
    response = auth_client.post("/api/user-prefs", content=b"x" * (64 * 1024 + 1))
    assert response.status_code == 400


def test_category_filter_rejects_non_list(auth_client: TestClient):
    assert auth_client.post("/settings/vod-filter", json={"cats": "x"}).status_code == 400


def test_update_playlist_errors(auth_client: TestClient):
    with patch("playlists.update", side_effect=ValueError("bad name")):
        response = auth_client.put("/api/playlists/p1", json={"name": ""})
    assert response.status_code == 400
    assert main_test._body(response, main_test._Detail)["detail"] == "bad name"
    with patch("playlists.update", return_value=None):
        assert auth_client.put("/api/playlists/p1", json={}).status_code == 404


def test_delete_missing_playlist(auth_client: TestClient):
    with patch("playlists.delete", return_value=False):
        assert auth_client.delete("/api/playlists/p1").status_code == 404


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        (b"x" * (1024 * 1024 + 1), "Request too large"),
        (b"{not json", "Invalid JSON"),
        (b"[1]", "Expected a JSON object"),
    ],
)
def test_json_body_validation(auth_client: TestClient, body: bytes, detail: str):
    response = auth_client.post("/api/playlists/order", content=body)
    assert response.status_code == 400
    assert main_test._body(response, main_test._Detail)["detail"] == detail


# ===========================================================================
# Logo proxy
# ===========================================================================


def test_logo_requires_url(auth_client: TestClient):
    assert auth_client.get("/api/logo?url=").status_code == 400


def test_logo_served_from_cache(auth_client: TestClient, tmp_path: Path):
    cached = tmp_path / "logo.png"
    cached.write_bytes(b"png")
    with patch("main.get_cached_logo", return_value=cached):
        response = auth_client.get("/api/logo?url=http://x/logo.png")
    assert response.content == b"png"
    assert response.headers["cache-control"] == "max-age=86400"


def test_logo_rejects_non_http_scheme(auth_client: TestClient):
    with patch("main.get_cached_logo", return_value=None):
        assert auth_client.get("/api/logo?url=file:///etc/passwd").status_code == 400


def test_logo_fetches_and_caches(auth_client: TestClient, tmp_path: Path):
    saved = tmp_path / "saved.png"
    saved.write_bytes(b"img")
    with (
        patch("main.get_cached_logo", return_value=None),
        patch("main._fetch_logo", return_value=(b"img", "image/png")) as fetch,
        patch("main.save_logo", return_value=saved) as save,
    ):
        response = auth_client.get("/api/logo?url=http://x/l.png&source=src")
    assert response.content == b"img"
    fetch.assert_called_once_with("http://x/l.png")
    save.assert_called_once_with("src", "http://x/l.png", b"img", "image/png")


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (ValueError("URL is not an image"), 400),
        (error.URLError("down"), 502),
        (RuntimeError("weird"), 500),
    ],
)
def test_logo_fetch_errors(auth_client: TestClient, exc: Exception, status: int):
    with (
        patch("main.get_cached_logo", return_value=None),
        patch("main._fetch_logo", side_effect=exc),
    ):
        assert auth_client.get("/api/logo?url=https://x/l.png").status_code == status


@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeLogoResponse:
    headers: dict[str, str]
    data: bytes

    def read(self, amt: int) -> bytes:
        del amt
        return self.data


def _fake_response(content_type: str, data: bytes) -> contextlib.nullcontext[_FakeLogoResponse]:
    return contextlib.nullcontext(
        _FakeLogoResponse(headers={"Content-Type": content_type}, data=data),
    )


def test_fetch_logo_returns_image():
    with patch("main.safe_urlopen", return_value=_fake_response("image/png", b"ok")) as urlopen:
        assert main._fetch_logo("http://x") == (b"ok", "image/png")
    urlopen.assert_called_once_with("http://x", timeout=3)


@pytest.mark.parametrize(
    ("content_type", "data", "message"),
    [("text/html", b"", "not an image"), ("image/png", b"xxxx", "too large")],
)
def test_fetch_logo_rejects(content_type: str, data: bytes, message: str):
    with (
        patch("main.safe_urlopen", return_value=_fake_response(content_type, data)),
        pytest.raises(ValueError, match=message),
    ):
        main._fetch_logo("http://x", max_size=4)


# ===========================================================================
# Server settings
# ===========================================================================


def test_transcode_settings_store_custom_dir(auth_client: TestClient):
    with patch("main.get_sr_models", return_value=[]):
        response = auth_client.post(
            "/settings/transcode",
            data={
                "transcode_mode": "auto",
                "transcode_hw": "software",
                "transcode_dir": "/srv/transcode",
                "sr_model": "fake",
            },
        )
    assert response.json() == {"ok": True}
    settings = cache_module.load_server_settings()
    assert settings["transcode_dir"] == "/srv/transcode"
    assert settings.get("sr_model") == ""


def test_refresh_encoders(auth_client: TestClient):
    with patch("main.refresh_encoders", return_value={"nvenc": True}):
        assert auth_client.post("/settings/refresh-encoders").json() == {
            "ok": True,
            "encoders": {"nvenc": True},
        }


def test_user_agent_invalid_preset_falls_back(auth_client: TestClient):
    response = auth_client.post("/settings/user-agent", data={"preset": "evil", "custom": "UA"})
    assert response.json() == {"ok": True}
    settings = cache_module.load_server_settings()
    assert settings["user_agent_preset"] == "default"
    assert settings["user_agent_custom"] == "UA"


def test_probe_cache_enriched_with_series_names(auth_client: TestClient):
    stats: list[dict[str, object]] = [
        {
            "series_id": 1,
            "name": "",
            "episodes": [{"episode_id": 11}, {"episode_id": 99}],
        },
        {"series_id": 2, "name": "", "episodes": []},
        {"series_id": 3, "name": "Known", "episodes": []},
        {"series_id": 4, "name": "Preserved", "episodes": [{"episode_id": 11}]},
    ]
    series: ProviderRecord = {
        "info": {"name": "Show"},
        "episodes": {
            "1": [
                {"id": "11", "episode_num": 2, "title": "S01E02 - Show - Pilot"},
                {"id": None},
            ],
        },
    }

    def get_series_info(series_id: int) -> ProviderRecord:
        if series_id == 2:
            raise RuntimeError("upstream down")
        return series

    xtream = MagicMock()
    xtream.get_series_info.side_effect = get_series_info
    with (
        patch("main.ffmpeg_command.get_series_probe_cache_stats", return_value=stats),
        patch("main.get_first_xtream_client", return_value=xtream),
        patch("main.get_cached_info", side_effect=_fetch_uncached) as cached,
    ):
        data = main_test._body(auth_client.get("/settings/probe-cache"), _ProbeCache)["series"]
    assert data[0]["name"] == "Show"
    assert data[0]["episodes"] == [{"episode_id": 11, "name": "S01E02 Pilot"}, {"episode_id": 99}]
    assert data[1]["name"] == ""
    assert data[2]["name"] == "Known"
    assert data[3]["name"] == "Preserved"
    assert data[3]["episodes"] == [{"episode_id": 11, "name": "S01E02 Pilot"}]
    assert [c.args for c in xtream.get_series_info.call_args_list] == [(1,), (2,), (4,)]
    assert [c.args[0] for c in cached.call_args_list] == [
        "series_info_1",
        "series_info_2",
        "series_info_4",
    ]


def test_clear_series_mru(auth_client: TestClient):
    with patch("main.ffmpeg_command.clear_series_mru") as clear:
        assert auth_client.post("/settings/probe-cache/clear-mru/7").json() == {"ok": True}
    clear.assert_called_once_with(7)


def test_clear_data_cache(auth_client: TestClient):
    with patch("main.clear_all_file_caches", return_value=4):
        assert auth_client.post("/settings/data-cache/clear").json() == {
            "ok": True,
            "cleared": 4,
        }


# ===========================================================================
# Handoff API
# ===========================================================================


def test_handoff_watch_caps_position_at_now(auth_client: TestClient):
    with patch("handoff.report_watch") as report:
        response = auth_client.post(
            "/api/handoff/watch",
            json={"stream_id": "s1", "position": 9e12, "handoff_id": "h"},
        )
    assert response.json() == {"status": "ok"}
    username, stream_id, position, handoff_id = report.call_args.args
    assert (username, stream_id, handoff_id) == ("testuser", "s1", "h")
    assert position <= time.time()


def test_handoff_stop_rejects_hidden_stream(auth_client: TestClient):
    with patch("main._handoff_stream_allowed", return_value=False):
        response = auth_client.post("/api/handoff/stop", json={"stream_id": "s", "position": 1})
    assert response.status_code == 404


def test_handoff_dismiss(auth_client: TestClient):
    with patch("handoff.cancel") as cancel:
        assert auth_client.delete("/api/handoff").json() == {"status": "ok"}
    cancel.assert_called_once_with("testuser")


def test_handoff_file_routes(auth_client: TestClient, tmp_path: Path):
    playlist = tmp_path / "stream.m3u8"
    playlist.write_text("#EXTM3U\n")
    segment = tmp_path / "seg.ts"
    segment.write_bytes(b"ts")
    with patch("handoff.recording_file", return_value=None):
        assert auth_client.get("/handoff/r/stream.m3u8").status_code == 404
    with patch("handoff.recording_file", return_value=playlist):
        response = auth_client.get("/handoff/r/stream.m3u8")
        assert response.text == "#EXTM3U\n"
        assert response.headers["content-type"] == "application/vnd.apple.mpegurl"
        assert auth_client.get("/handoff/r/stream.txt").status_code == 404
    with patch("handoff.recording_file", return_value=segment):
        response = auth_client.get("/handoff/r/seg.ts")
    assert response.content == b"ts"
    assert response.headers["content-type"] == "video/mp2t"


# ===========================================================================
# User management
# ===========================================================================


def test_delete_own_account_with_password(auth_client: TestClient):
    response = auth_client.post(
        "/settings/users/delete/testuser",
        data={"password": "testpass123"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    assert "testuser" not in auth.get_all_usernames()


def test_delete_other_requires_admin(auth_client: TestClient):
    _login_as(auth_client, "bob")
    response = auth_client.post("/settings/users/delete/testuser")
    assert response.status_code == 403


def test_delete_missing_user(auth_client: TestClient):
    assert auth_client.post("/settings/users/delete/ghost").status_code == 404


@pytest.mark.parametrize(
    ("username", "password", "detail"),
    [
        (" a ", "password123", "Username must be at least 2 characters"),
        ("carol", "short", "Password must be at least 8 characters"),
        ("testuser", "password123", "User already exists"),
    ],
)
def test_add_user_validation(auth_client: TestClient, username: str, password: str, detail: str):
    response = auth_client.post(
        "/settings/users/add",
        data={"username": username, "password": password},
    )
    assert response.status_code == 400
    assert main_test._body(response, main_test._Detail)["detail"] == detail


def test_add_user_with_limits(auth_client: TestClient):
    response = auth_client.post(
        "/settings/users/add",
        data={
            "username": " carol ",
            "password": "password123",
            "admin": "on",
            "max_streams_per_source": '{"s1": 2}',
            "unavailable_groups": '["cat:1"]',
        },
    )
    assert response.json() == {"status": "ok"}
    assert auth.is_admin("carol")
    limits = auth.get_user_limits("carol")
    assert limits["max_streams_per_source"] == {"s1": 2}
    assert limits["unavailable_groups"] == ["cat:1"]


def test_add_user_ignores_malformed_limits(auth_client: TestClient):
    with patch("auth.set_user_limits") as set_limits:
        response = auth_client.post(
            "/settings/users/add",
            data={
                "username": "dave",
                "password": "password123",
                "max_streams_per_source": "{bad",
                "unavailable_groups": "[bad",
            },
        )
    assert response.json() == {"status": "ok"}
    assert "dave" in auth.get_all_usernames()
    assert not auth.is_admin("dave")
    set_limits.assert_not_called()


def test_change_own_password_errors(auth_client: TestClient):
    response = auth_client.post(
        "/settings/users/password",
        data={"current_password": "testpass123", "new_password": "short"},
    )
    assert response.status_code == 400
    with patch("auth.change_password", return_value=False):
        response = auth_client.post(
            "/settings/users/password",
            data={"current_password": "testpass123", "new_password": "longenough"},
        )
    assert response.status_code == 404


def test_change_other_password(auth_client: TestClient):
    auth.create_user("bob", "password123")
    url = "/settings/users/password/"
    assert auth_client.post(url + "bob", data={"new_password": "short"}).status_code == 400
    assert auth_client.post(url + "ghost", data={"new_password": "longenough"}).status_code == 404
    assert auth_client.post(url + "bob", data={"new_password": "longenough"}).status_code == 200
    assert auth.verify_password("bob", "longenough")
    _login_as(auth_client, "eve")
    assert auth_client.post(url + "bob", data={"new_password": "longenough"}).status_code == 403


def test_set_admin(auth_client: TestClient):
    auth.create_user("bob", "password123")
    assert auth_client.post("/settings/users/admin/bob", data={"admin": "on"}).status_code == 200
    assert auth.is_admin("bob")
    assert auth_client.post("/settings/users/admin/ghost").status_code == 404


def test_set_user_limits(auth_client: TestClient):
    auth.create_user("bob", "password123")
    response = auth_client.post(
        "/settings/users/limits/bob",
        data={"max_streams_per_source": '{"s1": 1}', "unavailable_groups": '["cat:2"]'},
    )
    assert response.json() == {"status": "ok"}
    assert auth.get_user_limits("bob")["unavailable_groups"] == ["cat:2"]
    assert auth_client.post("/settings/users/limits/ghost").status_code == 404


@pytest.mark.parametrize(
    ("data", "detail"),
    [
        ({"max_streams_per_source": "[]"}, "max_streams_per_source must be a JSON object"),
        ({"max_streams_per_source": "{bad"}, "Invalid JSON for max_streams_per_source"),
        ({"unavailable_groups": "{}"}, "unavailable_groups must be a JSON array"),
        ({"unavailable_groups": "[bad"}, "Invalid JSON for unavailable_groups"),
    ],
)
def test_set_user_limits_validation(auth_client: TestClient, data: dict[str, str], detail: str):
    response = auth_client.post("/settings/users/limits/testuser", data=data)
    assert response.status_code == 400
    assert main_test._body(response, main_test._Detail)["detail"] == detail


# ===========================================================================
# Small helpers
# ===========================================================================


def test_name_matcher_invalid_regex_matches_nothing():
    assert main._name_matcher("[", regex=True)("[") is False


def test_regex_matches_swallows_errors():
    pattern = MagicMock()
    pattern.search.side_effect = RuntimeError("catastrophic")
    assert main._regex_matches(pattern, "x") is False


def test_float_field_rejects_non_scalar():
    assert main._float_field({}, "position") == 0.0
    assert main._float_field({"position": "1.5"}, "position") == 1.5
    with pytest.raises(TypeError, match="position must be a number"):
        main._float_field({"position": [1]}, "position")


def test_unavailable_channel_keeps_only_saved_fields():
    assert main._unavailable_channel({"name": "Old"}) == {"name": "Old", "available": False}


def test_vod_details_without_client_keeps_movie():
    movie: ProviderRecord = {"stream_id": 5}
    with (
        patch("main.get_xtream_client_by_source", return_value=None),
        patch("main.get_first_xtream_client", return_value=None),
    ):
        assert asyncio.run(main._with_vod_details(movie, "s1", 5)) is movie


def test_vod_details_without_info_keeps_movie():
    movie: ProviderRecord = {"stream_id": 5}
    with (
        patch("main.get_xtream_client_by_source", return_value=_xtream({"info": {}})),
        patch("main.get_cached_info", side_effect=_fetch_uncached),
    ):
        assert asyncio.run(main._with_vod_details(movie, "s1", 5)) is movie


def test_user_prefs_saves_each_field(auth_client: TestClient):
    prefs: dict[str, MutablePlainTree] = {
        "favorites": {"movies": {"1": {"name": "Film"}}},
        "cc_lang": "eng",
        "cc_style": {"size": "large"},
        "cast_host": "192.168.1.5",
        "virtual_scroll": False,
        "sort_live_by_views": False,
        "guide_selected_cats": ["1"],
    }
    response = auth_client.post("/api/user-prefs", json=prefs)
    assert response.json() == {"ok": True}
    saved: dict[str, object] = {**cache_module.load_user_settings("testuser")}
    assert {key: saved.get(key) for key in prefs} == prefs


def test_user_prefs_rejects_mistyped_field(auth_client: TestClient):
    before = cache_module.load_user_settings("testuser")
    response = auth_client.post("/api/user-prefs", json={"cc_style": "large"})
    assert response.status_code == 400
    detail = main_test._body(response, main_test._Detail)["detail"]
    assert detail.startswith("Invalid preferences: ")
    assert cache_module.load_user_settings("testuser") == before


def test_api_settings_saves_each_field(auth_client: TestClient):
    update: dict[str, MutablePlainTree] = {
        "transcode_mode": "always",
        "transcode_hw": "software",
        "vod_transcode_cache_mins": 30,
        "probe_live": False,
        "probe_movies": False,
        "probe_series": False,
        "vod_order": ["2", "1"],
        "series_order": ["3"],
    }
    response = auth_client.post("/api/settings", json=update)
    assert response.json() == {"status": "ok"}
    saved: dict[str, object] = {**cache_module.load_server_settings()}
    assert {key: saved.get(key) for key in update} == update


def test_api_settings_rejects_mistyped_field(auth_client: TestClient):
    before = cache_module.load_server_settings()
    response = auth_client.post("/api/settings", json={"probe_live": "yes"})
    assert response.status_code == 400
    detail = main_test._body(response, main_test._Detail)["detail"]
    assert detail.startswith("Invalid settings: ")
    assert cache_module.load_server_settings() == before


def test_category_filter_rejects_non_text_ids(auth_client: TestClient):
    before = cache_module.load_user_settings("testuser")
    response = auth_client.post("/settings/vod-filter", json={"cats": ["1", 2]})
    assert response.status_code == 400
    assert cache_module.load_user_settings("testuser") == before


def test_safe_float():
    assert main._safe_float(None, 2.0) == 2.0
    assert main._safe_float("1.5") == 1.5
    assert main._safe_float("nope", 3.0) == 3.0


def _collect_events(queue: asyncio.Queue[str]) -> list[str]:
    async def collect() -> list[str]:
        return [event async for event in main._epg_event_stream(queue)]

    return asyncio.run(collect())


def test_epg_event_stream_ready_immediately():
    queue: asyncio.Queue[str] = asyncio.Queue()
    main._epg_subscribers.add(queue)
    with patch("epg.has_programs", return_value=True):
        assert _collect_events(queue) == ["data: epg_ready\n\n"]
    assert queue not in main._epg_subscribers


def test_epg_event_stream_waits_for_event():
    queue: asyncio.Queue[str] = asyncio.Queue()
    with (
        patch("epg.has_programs", return_value=False),
        patch("main._shutdown_event", None),
        patch("main._next_epg_event", side_effect=[None, "epg_ready"]),
    ):
        assert _collect_events(queue) == ["data: epg_ready\n\n"]


def test_epg_event_stream_stops_on_shutdown():
    queue: asyncio.Queue[str] = asyncio.Queue()
    main._epg_subscribers.add(queue)
    shutdown = asyncio.Event()
    shutdown.set()
    with patch("epg.has_programs", return_value=False), patch("main._shutdown_event", shutdown):
        assert _collect_events(queue) == []
    assert queue not in main._epg_subscribers


def test_next_epg_event():
    async def timeout(coro: Any, timeout: float) -> str:
        del timeout
        coro.close()
        raise asyncio.TimeoutError

    async def run() -> tuple[str | None, str | None]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        queue.put_nowait("epg_ready")
        ready = await main._next_epg_event(queue)
        with patch("asyncio.wait_for", timeout):
            missing = await main._next_epg_event(queue)
        return ready, missing

    assert asyncio.run(run()) == ("epg_ready", None)


def test_rate_limit_drops_expired_ip():
    with patch.dict(main._login_attempts, {"1.1.1.1": [0.0]}, clear=True):
        main._check_rate_limit("1.1.1.1")
        assert "1.1.1.1" not in main._login_attempts


def test_rate_limit_prunes_stale_ips_and_blocks():
    now = time.time()
    attempts: dict[str, list[float]] = {f"ip{i}": [0.0] for i in range(1001)}
    attempts["2.2.2.2"] = [now] * 10
    with patch.dict(main._login_attempts, attempts, clear=True):
        with pytest.raises(HTTPException) as exc:
            main._check_rate_limit("2.2.2.2")
        assert exc.value.status_code == 429
        assert len(main._login_attempts) == 902


def test_fetch_all_epg_sums_successes_and_prunes(caplog: pytest.LogCaptureFixture):
    def fetch(url: str, *_args: Any, **kwargs: Any) -> int:
        assert kwargs["user_agent"] == "UA"
        if "bad" in url:
            raise RuntimeError("timeout")
        return 3

    cache_module.get_cache()["live_streams"] = []
    with (
        patch("main.fetch_epg", side_effect=fetch),
        patch("main.cache.get_user_agent", return_value="UA"),
        patch("epg.prune_old_programs", return_value=2) as prune,
        caplog.at_level(logging.INFO, logger="main"),
    ):
        total = main._fetch_all_epg([("http://good", 5, "a"), ("http://bad", 5, "b")])
    assert total == 3
    cutoff = prune.call_args.args[0]
    assert cutoff < datetime.now(timezone.utc) - timedelta(hours=23)
    assert "Pruned 2 expired EPG programs" in caplog.text
    assert "EPG failed" in caplog.text


@pytest.mark.parametrize(
    ("flag", "starter"),
    [
        ("guide_load", main._start_guide_background_load),
        ("vod_load", main._start_vod_background_load),
        ("series_load", main._start_series_background_load),
    ],
)
def test_background_loads_skip_when_running(
    refresh_flags: set[str],
    recorded_threads: list[Callable[[], None]],
    flag: str,
    starter: Callable[[], None],
):
    refresh_flags.add(flag)
    starter()
    assert recorded_threads == []


def test_vod_background_load_fills_cache_and_clears_flag(
    refresh_flags: set[str],
    sync_threads: None,
):
    del sync_threads
    with patch("main.load_vod_data", return_value=(["vc"], ["vs"])):
        main._start_vod_background_load()
    assert cache_module.get_cache().get("vod_categories") == ["vc"]
    assert cache_module.get_cache().get("vod_streams") == ["vs"]
    assert "vod_load" not in refresh_flags


def test_guide_streams_ignore_unknown_playlist(auth_client: TestClient):
    del auth_client
    stream: ProviderRecord = {"stream_id": 1, "category_ids": ["1"], "source_id": "s"}
    cache_module.get_cache()["live_streams"] = [stream]
    with patch("playlists.load", return_value=[]):
        streams, ordered, selected = main._get_guide_streams("pl:gone,1", "testuser")
    assert streams == [stream]
    assert ordered == ["pl:gone", "1"]
    assert selected == {"pl:gone", "1"}


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"info": '{"plot": "From JSON"}'}, "From JSON"),
        ({"info": "{bad", "description": "Fallback"}, "Fallback"),
        ({"info": {"description": "Dict"}}, "Dict"),
        ({"plot": "Plain"}, "Plain"),
    ],
)
def test_episode_desc(info: ProviderRecord, expected: str):
    assert main._get_episode_desc(info) == expected


# ===========================================================================
# Player info
# ===========================================================================


def test_live_player_info_unknown_stream(auth_client: TestClient):
    del auth_client
    cache_module.get_cache()["live_streams"] = []
    with patch("main._ensure_live_cache"):
        assert main._get_live_player_info("missing") == PlayerInfo()


def test_live_player_info_uses_source_settings(auth_client: TestClient):
    del auth_client
    cache_module.get_cache()["live_streams"] = [main_test._archive_stream()]
    settings = {"sources": [{"id": "src1", "deinterlace_fallback": False}]}
    with (
        patch("main._ensure_live_cache"),
        patch("main.load_server_settings", return_value=settings),
        patch("main._get_current_program", return_value=None),
    ):
        info = main._get_live_player_info("src1_42")
    assert info.url == "http://upstream.test/live/user/pass/42.m3u8"
    assert info.deinterlace_fallback is False
    assert info.catchup_days == 2


def test_archived_program_needs_epg_id():
    with patch("playlists.load", return_value=[]):
        assert main._find_archived_program({"stream_id": 1}, datetime.now(timezone.utc)) is None


def test_catchup_player_info_unknown_stream():
    with patch("main._get_live_player_info", return_value=PlayerInfo()):
        assert main._get_catchup_player_info("x", 0) == PlayerInfo()


def _xtream(info: ProviderRecord | Exception | None = None) -> MagicMock:
    xtream = MagicMock()
    xtream.build_stream_url.side_effect = _stream_url
    for method in (xtream.get_vod_info, xtream.get_series_info):
        if isinstance(info, Exception):
            method.side_effect = info
        else:
            method.return_value = info
    return xtream


def test_movie_player_info_without_client():
    cache_module.get_cache().pop("vod_streams", None)
    with patch("main.get_first_xtream_client", return_value=None):
        assert main._get_movie_player_info("1", "") == PlayerInfo()


@pytest.mark.parametrize(
    ("movie", "name", "desc"),
    [
        ({"info": {"name": "Film", "releasedate": "1999-01-01", "plot": "P"}}, "Film (1999)", "P"),
        ({"info": {"name": "Film", "description": "D"}}, "Film", "D"),
        ({}, "", ""),
    ],
)
def test_movie_player_info(movie: ProviderRecord, name: str, desc: str):
    cache_module.get_cache()["vod_streams"] = [{"stream_id": 5, "source_id": "s1"}]
    xtream = _xtream(movie)
    with (
        patch("main.get_xtream_client_by_source", return_value=xtream) as by_source,
        patch("main.get_cached_info", side_effect=_fetch_uncached),
    ):
        info = main._get_movie_player_info("5", "")
    by_source.assert_called_once_with("s1")
    assert info.url == "http://x/movie/5.mkv"
    assert info.source_id == "s1"
    assert (info.channel_name, info.program_desc) == (name, desc)


def test_movie_player_info_survives_upstream_error():
    cache_module.get_cache()["vod_streams"] = []
    xtream = _xtream(error.URLError("down"))
    with (
        patch("main.get_first_xtream_client", return_value=xtream),
        patch("main.get_cached_info", side_effect=_fetch_uncached),
    ):
        info = main._get_movie_player_info("5", "mp4")
    assert info.url == "http://x/movie/5.mp4"
    assert info.channel_name == ""


_SERIES: ProviderRecord = {
    "info": {"name": "Show", "year": "2020"},
    "episodes": {
        "2": [{"id": 21, "episode_num": 1, "title": "S02E01 - Show - Season Two - Grand Finale"}],
        "10": [{"id": 101, "episode_num": 1, "title": "Last"}],
        "1": [
            {"id": 12, "episode_num": 2, "title": "Second", "container_extension": "mp4"},
            {"id": 11, "episode_num": 1, "title": "S01E01 -Pilot", "info": {"plot": "Start"}},
            {"id": 10, "title": "Prologue"},
        ],
        "3": None,
    },
}


def test_series_player_info_without_client():
    with patch("main.get_first_xtream_client", return_value=None):
        assert main._get_series_player_info("1", None, "") == (PlayerInfo(), None)


@pytest.mark.parametrize("series_id", [None, 0])
def test_series_player_info_without_series_id(series_id: int | None):
    cache_module.get_cache()["series"] = [{"series_id": series_id, "source_id": "ignored"}]
    with (
        patch("main.get_first_xtream_client", return_value=_xtream()),
        patch("main.get_xtream_client_by_source") as by_source,
        patch("main.get_cached_info") as cached,
    ):
        info, next_url = main._get_series_player_info("1", series_id, "")
    assert info == PlayerInfo(url="http://x/series/1.mkv")
    assert next_url is None
    by_source.assert_not_called()
    cached.assert_not_called()


@pytest.mark.parametrize(
    ("stream_id", "title", "desc", "next_url"),
    [
        ("10", "S01E00 -- Prologue", "", "/play/series/11?series_id=1&ext=mkv"),
        ("11", "S01E01 -- Pilot", "Start", "/play/series/12?series_id=1&ext=mp4"),
        ("12", "S01E02 -- Second", "", "/play/series/21?series_id=1&ext=mkv"),
        ("21", "S02E01 -- Grand Finale", "", "/play/series/101?series_id=1&ext=mkv"),
        ("101", "S10E01 -- Last", "", None),
    ],
)
def test_series_player_info_episodes(
    stream_id: str,
    title: str,
    desc: str,
    next_url: str | None,
):
    cache_module.get_cache()["series"] = [{"series_id": 1, "source_id": "s1"}]
    client = _xtream(_SERIES)
    with (
        patch("main.get_xtream_client_by_source", return_value=client) as by_source,
        patch("main.get_cached_info", side_effect=_fetch_uncached) as cached,
    ):
        info, url = main._get_series_player_info(stream_id, 1, "")
    assert info.source_id == "s1"
    assert info.channel_name == "Show (2020)"
    assert (info.program_title, info.program_desc, url) == (title, desc, next_url)
    by_source.assert_called_once_with("s1")
    assert cached.call_args.args[0] == "series_info_s1_1"
    client.get_series_info.assert_called_once_with(1)


@pytest.mark.parametrize(
    ("series", "name"),
    [
        (error.URLError("down"), ""),
        (None, ""),
        ({"info": {"name": "Untimed"}}, "Untimed"),
        ({"info": {"plot": "Unnamed"}}, ""),
    ],
)
def test_series_player_info_partial_data(
    series: ProviderRecord | Exception | None,
    name: str,
    caplog: pytest.LogCaptureFixture,
):
    cache_module.get_cache().pop("series", None)
    client = _xtream(series)
    with (
        patch("main.get_first_xtream_client", return_value=client),
        patch("main.get_xtream_client_by_source") as by_source,
        patch("main.get_cached_info", side_effect=_fetch_uncached) as cached,
        caplog.at_level(logging.WARNING, logger="main"),
    ):
        info, url = main._get_series_player_info("1", 9, "")
    assert info == PlayerInfo(url="http://x/series/1.mkv", channel_name=name)
    assert url is None
    by_source.assert_not_called()
    assert cached.call_args.args[0] == "series_info__9"
    client.get_series_info.assert_called_once_with(9)
    if isinstance(series, Exception):
        assert [record.getMessage() for record in caplog.records] == [
            f"Failed to fetch series info 9: {series}",
        ]


def test_ensure_live_cache_loads_streams_and_epg():
    cache_module.get_cache().pop("live_streams", None)
    with (
        patch("main.load_all_live_data", return_value=([{"c": 1}], [{"s": 1}], [("u", 1, "s")])),
        patch("epg.has_programs", return_value=False),
        patch("main.load_all_epg", side_effect=RuntimeError("offline")) as load_epg,
    ):
        main._ensure_live_cache()
    store = cache_module.get_cache()
    assert store.get("live_streams") == [{"s": 1}]
    assert store.get("live_categories") == [{"c": 1}]
    load_epg.assert_called_once_with([("u", 1, "s")])


# ===========================================================================
# Handoff helpers
# ===========================================================================


def _live_info(**overrides: Any) -> PlayerInfo:
    now = time.time()
    values: dict[str, Any] = {
        "url": "http://upstream.test/live/user/pass/42.m3u8",
        "channel_name": "Channel",
        "source_id": "src1",
        "catchup_days": 2,
        "program_title": "Now",
        "program_start": now - 600,
        "program_end": now + 600,
    }
    values.update(overrides)
    return PlayerInfo(**values)


def test_handoff_channel_unknown_stream():
    with (
        patch("main._get_live_player_info", return_value=PlayerInfo()),
        patch("main._find_live_stream") as find,
    ):
        assert main._handoff_channel("x", time.time()) is None
    find.assert_not_called()


@pytest.mark.parametrize("position", [100, 150])
def test_handoff_channel_within_current_program(position: int):
    stream = main_test._archive_stream()
    info = _live_info(program_start=100, program_end=200)
    with (
        patch("main._get_live_player_info", return_value=info) as live,
        patch("main._find_live_stream", return_value=stream) as find,
        patch("main._find_archived_program") as archived,
        patch("main.catchup.can_play", return_value=True) as can_play,
    ):
        channel = main._handoff_channel("src1_42", position)
    assert channel == handoff.Channel(
        stream_id="src1_42",
        name="Channel",
        url=info.url,
        source_id="src1",
        catchup_days=2,
        program_title="Now",
        program_start=100,
        program_end=200,
    )
    assert info.program_start == 100
    live.assert_called_once_with("src1_42")
    find.assert_called_once_with("src1_42")
    archived.assert_not_called()
    can_play.assert_called_once_with(stream, datetime.fromtimestamp(position - 60, timezone.utc))


@pytest.mark.parametrize("position", [50, 200])
def test_handoff_channel_looks_up_earlier_program(position: int):
    cache_module.get_cache()["live_streams"] = [main_test._archive_stream()]
    program = Program(
        channel_id="ch1",
        title="Earlier",
        start=datetime.fromtimestamp(position - 60, timezone.utc),
        stop=datetime.fromtimestamp(position + 60, timezone.utc),
    )
    with (
        patch(
            "main._get_live_player_info",
            return_value=_live_info(program_start=100, program_end=200),
        ),
        patch("main._find_archived_program", return_value=program) as archived,
        patch("main.catchup.can_play", return_value=True),
    ):
        channel = main._handoff_channel("src1_42", position)
    assert channel is not None
    assert channel.program_title == "Earlier"
    assert channel.program_start == program.start.timestamp()
    assert channel.program_end == program.stop.timestamp()
    archived.assert_called_once_with(
        main_test._archive_stream(),
        datetime.fromtimestamp(position, timezone.utc),
    )


def test_handoff_channel_outside_archive():
    cache_module.get_cache()["live_streams"] = [main_test._archive_stream()]
    with (
        patch("main._get_live_player_info", return_value=_live_info()),
        patch("main._find_archived_program", return_value=None),
    ):
        channel = main._handoff_channel("src1_42", time.time() - 30 * 86_400)
    assert channel is not None
    assert (channel.program_title, channel.program_start, channel.program_end) == ("", 0.0, 0.0)
    assert channel.catchup_days == 0


def test_handoff_player_info_without_live_url(clean_handoff: None):
    handoff._records["testuser"] = main_test._handoff_record()
    with patch("main._get_live_player_info", return_value=PlayerInfo()):
        info = main._get_handoff_player_info("testuser", "src1_42", "h1", None)
    assert info == PlayerInfo()


@pytest.mark.parametrize(("start", "position"), [(None, 150), (0, 0), (200, 200)])
@pytest.mark.parametrize("title", ["Saved program", ""])
def test_handoff_recording_publishes_a_replacement(
    start: int | None,
    position: int,
    title: str,
):
    record = main_test._handoff_record()
    record.update(
        position=150,
        recording_start=100,
        program_title=title,
        program_start=90,
        program_end=300,
    )
    original = PlayerInfo(url="upstream", channel_name="Channel", program_title="Live")
    with (
        patch("handoff.claim", return_value=record) as claim,
        patch("main._get_live_player_info", return_value=original) as live,
    ):
        result = main._get_handoff_player_info("viewer", "src1_42", "h1", start)
    assert result == replace(
        original,
        url="/handoff/rec1/stream.m3u8",
        program_title=title or "Live",
        program_start=90,
        program_end=300,
        catchup_start=100,
        catchup_seek=max(0, position - 100),
        handoff_recording=True,
    )
    assert original == PlayerInfo(url="upstream", channel_name="Channel", program_title="Live")
    claim.assert_called_once_with("viewer", "h1")
    live.assert_called_once_with("src1_42")


@pytest.mark.parametrize(("start", "expected"), [(None, 60), (0, 0), (200, 60)])
def test_handoff_catchup_respects_explicit_start_and_default_archive_lag(
    start: int | None,
    expected: int,
):
    original = PlayerInfo(url="archive")
    with (
        patch(
            "handoff.claim",
            return_value=main_test._handoff_record(mode="catchup", position=100),
        ),
        patch("main.time.time", return_value=120),
        patch("main._get_catchup_player_info", return_value=original) as catchup,
    ):
        assert main._get_handoff_player_info("viewer", "src1_42", "h1", start) is original
    catchup.assert_called_once_with("src1_42", expected)


@pytest.mark.parametrize("missing", [True, False])
def test_handoff_rejects_missing_or_wrong_stream(missing: bool):
    record = None if missing else main_test._handoff_record(stream_id="other")
    with patch("handoff.claim", return_value=record), pytest.raises(HTTPException) as error_info:
        main._get_handoff_player_info("viewer", "src1_42", "h1", None)
    assert error_info.value.status_code == 404
    assert error_info.value.detail == "This program is no longer available to resume"


def test_handoff_stream_allowed(auth_client: TestClient):
    del auth_client
    cache_module.get_cache()["live_streams"] = [{"stream_id": "1", "source_id": "s"}]
    with patch("main._ensure_live_cache"):
        assert main._handoff_stream_allowed("testuser", "1") is True
        assert main._handoff_stream_allowed("testuser", "2") is False


# ===========================================================================
# Sources, groups and access
# ===========================================================================


@pytest.mark.parametrize(
    ("url", "base", "username", "expected"),
    [
        ("http://h/live/u/p/1.ts", "http://h/", "u", True),
        ("http://h/live/u/p/1.ts", "http://h", "x", False),
        ("http://h/live/u/p/1.ts", "http://h", "", True),
        ("http://other/live/u/p/1.ts", "http://h", "u", False),
        ("http://h/1.ts", "", "", False),
    ],
)
def test_is_xtream_source_url(url: str, base: str, username: str, expected: bool):
    source = Source(id="s", name="S", type="xtream", url=base, username=username)
    assert main._is_xtream_source_url(url, source) is expected


def test_build_all_groups():
    cache_module.get_cache()["live_categories"] = [
        {"source_id": "s1", "category_id": "7", "category_name": "News"},
        {"source_id": "gone", "category_id": "8", "category_name": "Orphan"},
    ]
    with patch("main.get_sources", return_value=[_source("s1", "xtream"), _source("s2", "m3u")]):
        groups = main._build_all_groups()
    assert groups == [
        {"id": "cat:7", "name": "S1: News", "type": "live"},
        {"id": "movies:s1", "name": "S1: Movies", "type": "movies"},
        {"id": "series:s1", "name": "S1: Series", "type": "series"},
    ]


def test_source_users_rejects_unknown_access():
    with pytest.raises(HTTPException) as exc:
        main._source_users("bogus", None)
    assert exc.value.status_code == 400


def test_live_streams_loaded_from_file_cache():
    cache_module.clear_all_caches()
    cache_module.get_cache().pop("epg", None)
    data = {"cats": [{"category_id": 1, "category_name": "News"}], "streams": [{"stream_id": 1}]}
    with patch("m3u.load_file_cache", return_value=(data, 0.0)):
        streams, names = asyncio.run(main._live_streams_and_category_names())
    assert streams == [{"stream_id": 1}]
    assert names == {"1": "News"}
    assert cache_module.get_cache().get("epg_urls") == []


def test_live_streams_without_file_cache():
    cache_module.clear_all_caches()
    cache_module.get_cache().pop("epg", None)
    with patch("m3u.load_file_cache", return_value=None):
        assert asyncio.run(main._live_streams_and_category_names()) == ([], {})


@pytest.mark.parametrize(
    ("unavailable", "expected"),
    [
        ({"movies:s1", "series:s1", "series:s2"}, {"movies": True, "series": False}),
        ({"movies:s1", "movies:s2", "series:s1"}, {"movies": False, "series": True}),
    ],
)
def test_content_access_respects_restrictions(unavailable: set[str], expected: dict[str, bool]):
    sources = [_source("s1", "xtream"), _source("s2", "xtream")]
    with (
        patch("main.get_sources", return_value=sources),
        patch("main._unavailable_groups", return_value=unavailable),
    ):
        assert main._get_content_access("bob") == expected


def _request(cookies: dict[str, str]) -> Any:
    return SimpleNamespace(cookies=cookies)


@pytest.mark.parametrize(
    ("cookies", "payload"),
    [({}, None), ({"token": "garbage"}, None), ({"token": "t"}, {"sub": ""})],
)
def test_content_access_from_request_defaults(
    cookies: dict[str, str],
    payload: dict[str, str] | None,
):
    with patch("main.verify_token", return_value=payload):
        access = main._get_content_access_from_request(_request(cookies))
    assert access == {"movies": True, "series": True}


def test_content_access_from_request_uses_user():
    with (
        patch("main.verify_token", return_value={"sub": "bob"}),
        patch("main._get_content_access", return_value={"movies": False, "series": True}) as get,
    ):
        access = main._get_content_access_from_request(_request({"token": "t"}))
    assert access == {"movies": False, "series": True}
    get.assert_called_once_with("bob")


class _ProbeCache(TypedDict):
    series: list[dict[str, MutablePlainTree]]


def _stream_url(kind: str, stream_id: int, ext: str) -> str:
    return f"http://x/{kind}/{stream_id}.{ext}"


def _fetch_uncached(key: str, fetch: Callable[[], ProviderRecord]) -> ProviderRecord:
    del key
    return fetch()


# ===========================================================================
# Command line entry point
# ===========================================================================


def _run_cli(
    argv: list[str],
    live_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    log_level: str = "",
) -> dict[str, Any]:
    """Run the CLI entry point and return uvicorn.run keyword arguments."""
    real_path = pathlib.Path

    def redirected_path(path: str) -> pathlib.Path:
        return real_path(path.replace("/etc/letsencrypt/live", str(live_dir), 1))

    monkeypatch.setenv("LOG_LEVEL", log_level)
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    monkeypatch.setattr(pathlib, "Path", redirected_path)
    with patch("uvicorn.run") as run, patch("logging.basicConfig") as basic_config:
        assert main.main() == 0
    kwargs = dict(run.call_args.kwargs)
    kwargs["log_level_configured"] = basic_config.call_args.kwargs["level"]
    return kwargs


def test_cli_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    kwargs = _run_cli([], tmp_path, monkeypatch)
    assert kwargs["port"] == 8000
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["log_level"] == "info"
    assert kwargs["access_log"] is False
    assert kwargs["ssl_certfile"] is None
    assert kwargs["log_level_configured"] == logging.INFO


def test_cli_env_log_level_and_custom_certs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    kwargs = _run_cli(
        ["--port", "9000", "--debug", "--cert", "c.pem", "--key", "k.pem"],
        tmp_path,
        monkeypatch,
        log_level="warning",
    )
    assert kwargs["port"] == 9000
    assert kwargs["log_level"] == "debug"
    assert kwargs["access_log"] is True
    assert (kwargs["ssl_certfile"], kwargs["ssl_keyfile"]) == ("c.pem", "k.pem")
    assert kwargs["log_level_configured"] == logging.WARNING


def test_cli_debug_flag_without_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert _run_cli(["--debug"], tmp_path, monkeypatch)["log_level_configured"] == logging.DEBUG


@pytest.mark.parametrize("argv", [["--https", "tv.example"], ["--https"]])
def test_cli_lets_encrypt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    domain = tmp_path / "tv.example"
    domain.mkdir()
    (domain / "fullchain.pem").write_text("cert")
    (tmp_path / "README").write_text("not a domain")
    kwargs = _run_cli(argv, tmp_path, monkeypatch)
    assert kwargs["ssl_certfile"] == str(domain / "fullchain.pem")
    assert kwargs["ssl_keyfile"] == str(domain / "privkey.pem")


def test_cli_https_without_certs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(SystemExit, match="No Let's Encrypt certs"):
        _run_cli(["--https"], tmp_path, monkeypatch)


def test_cli_https_missing_domain_cert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(SystemExit, match="Cert not found"):
        _run_cli(["--https", "missing.example"], tmp_path, monkeypatch)


def test_script_guard_exits_with_main_status(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    with (
        patch("uvicorn.run") as run,
        patch("logging.basicConfig"),
        pytest.raises(SystemExit) as exit_info,
    ):
        runpy.run_module("main", run_name="__main__", alter_sys=True)
    assert exit_info.value.code == 0
    assert run.call_args.kwargs["port"] == 8000


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
