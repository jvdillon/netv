"""Tests for main.py - FastAPI routes."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterator, Sequence
from datetime import datetime, timedelta, timezone, tzinfo
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol, TypedDict, TypeVar, cast, override
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import asyncio
import json
import re
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient
from treekle import MutablePlainTree, json_loads

import pytest

from netv import (
    auth,
    cache as cache_module,
    fast_start,
    ffmpeg_session,
    handoff,
    m3u as m3u_module,
    main,
)
from netv.epg import Program
from netv.fast_start_test import video_packet
from netv.ffmpeg_session_lifecycle_test import LoopProcess
from netv.ffmpeg_session_test import FakeProcess
from netv.main import CatchupListing, GuideRowsResponse, PlayerInfo, PlaylistPayload
from netv.playback_policy import LiveRecoveryPolicy
from netv.playlists import NameMatch, StreamSummary
from netv.xtream import ProviderRecord


T = TypeVar("T")


@pytest.fixture
def mock_deps() -> Iterator[None]:
    """Mock all external dependencies before importing main."""
    with (
        patch.dict(
            "sys.modules",
            {"defusedxml": MagicMock(), "defusedxml.ElementTree": MagicMock()},
        ),
        patch("netv.cache.CACHE_DIR", Path("/tmp/test_cache")),
        patch("netv.cache.SERVER_SETTINGS_FILE", Path("/tmp/test_cache/server_settings.json")),
        patch("netv.cache.USERS_DIR", Path("/tmp/test_cache/users")),
    ):
        yield


@pytest.fixture
def client(tmp_path: Path, mock_deps: None) -> Iterator[TestClient]:
    """Create test client with mocked dependencies."""

    # Patch paths before importing main.
    with (
        patch("netv.cache.CACHE_DIR", tmp_path),
        patch("netv.cache.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("netv.cache.USERS_DIR", tmp_path / "users"),
        patch("netv.auth.CACHE_DIR", tmp_path),
        patch("netv.auth.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("netv.auth.USERS_DIR", tmp_path / "users"),
        patch("netv.epg.init"),
        patch("netv.ffmpeg_command.init"),
        patch("netv.ffmpeg_session.cleanup_and_recover_sessions"),
    ):
        (tmp_path / "users").mkdir(exist_ok=True)

        # Disable background loading.
        cache_module.clear_all_caches()
        cache_module.get_cache().pop("epg", None)
        yield TestClient(main.app)


@pytest.fixture
def auth_client(tmp_path: Path, mock_deps: None) -> Iterator[TestClient]:
    """Create test client with a logged-in user."""

    with (
        patch("netv.cache.CACHE_DIR", tmp_path),
        patch("netv.cache.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("netv.cache.USERS_DIR", tmp_path / "users"),
        patch("netv.auth.CACHE_DIR", tmp_path),
        patch("netv.auth.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("netv.auth.USERS_DIR", tmp_path / "users"),
        patch("netv.epg.init"),
        patch("netv.ffmpeg_command.init"),
        patch("netv.ffmpeg_session.cleanup_and_recover_sessions"),
    ):
        (tmp_path / "users").mkdir(exist_ok=True)

        cache_module.clear_all_caches()
        cache_module.get_cache().pop("epg", None)
        client = TestClient(main.app)

        # Create user and get token.
        auth.create_user("testuser", "testpass123")
        token = auth.create_token({"sub": "testuser"})
        client.cookies.set("token", token)

        yield client


def test_web_player_includes_resolution_badge(auth_client: TestClient):

    info = PlayerInfo(url="https://example.test/live.m3u8", channel_name="Test channel")
    with patch("netv.main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert 'id="quality-badge"' in response.text
    assert response.text.index('id="player-container"') < response.text.index('id="quality-badge"')
    assert response.text.index("/static/js/player-quality.js") < response.text.index(
        "/static/js/player.js",
    )
    assert response.text.index("/static/js/live-playback.js") < response.text.index(
        "/static/js/player.js",
    )
    script = auth_client.get("/static/js/player-quality.js")
    assert script.status_code == 200
    assert "videoWidth" in script.text
    assert 'id="cast-btn"' in response.text
    assert 'id="cast-dialog"' in response.text
    assert "cast_sender.js" not in response.text
    assert response.text.index("/static/js/cast.js") < response.text.index("/static/js/player.js")

    class CastPlacement(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.divs: list[dict[str, str | None]] = []
            self.parents: dict[str | None, str | None] = {}

        @override
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            attributes = dict(attrs)
            if attributes.get("id") in ("cast-overlay", "cast-dialog"):
                self.parents[attributes["id"]] = self.divs[-1].get("id")
            if tag == "div":
                self.divs.append(attributes)

        @override
        def handle_endtag(self, tag: str) -> None:
            if tag == "div":
                self.divs.pop()

    placement = CastPlacement()
    placement.feed(response.text)
    assert placement.parents == {
        "cast-overlay": "player-container",
        "cast-dialog": "player-container",
    }


def test_live_player_records_view(auth_client: TestClient):

    info = PlayerInfo(
        url="https://example.test/live.m3u8",
        channel_name="Test",
        source_id="source-a",
    )
    with patch("netv.main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/7")

    assert response.status_code == 200
    assert cache_module.load_live_view_counts("testuser") == {"source-a:7": 1}


@pytest.mark.parametrize("path", ["devices", "status"])
def test_cast_requires_authentication(client: TestClient, path: str):
    assert client.get(f"/api/cast/{path}").status_code == 401


@pytest.mark.parametrize("path", ["start", "control"])
def test_cast_commands_require_authentication(client: TestClient, path: str):
    assert client.post(f"/api/cast/{path}", json={}).status_code == 401


def test_cast_api_uses_http_and_remembers_lan_address(auth_client: TestClient):
    with patch("netv.chromecast.manager.start", return_value={"active": True}) as start:
        response = auth_client.post(
            "/api/cast/start",
            json={
                "host": "192.168.1.50",
                "session_id": "abc",
                "server_url": "http://192.168.1.10:9000/",
                "title": "News",
                "current_time": 15,
            },
        )
        assert response.status_code == 200
        start.assert_called_once_with(
            "testuser",
            "192.168.1.50",
            "abc",
            "http://192.168.1.10:9000",
            "News",
            15,
        )
        assert (
            cache_module.load_user_settings("testuser")["cast_host"] == "http://192.168.1.10:9000"
        )


def test_cast_api_uses_request_origin_and_supports_legacy_host_preference(auth_client: TestClient):
    with patch("netv.chromecast.manager.start", return_value={"active": True}) as start:
        response = auth_client.post(
            "/api/cast/start",
            json={"host": "192.168.1.50", "session_id": "abc"},
        )
        assert response.status_code == 200
        assert start.call_args.args[3] == "http://testserver"
        settings = cache_module.load_user_settings("testuser")
        settings["cast_host"] = "192.168.1.10:8000"
        cache_module.save_user_settings("testuser", settings)
        response = auth_client.post(
            "/api/cast/start",
            json={"host": "192.168.1.50", "session_id": "abc"},
        )
        assert response.status_code == 200
        assert start.call_args.args[3] == "http://192.168.1.10:8000"


def test_cast_rejects_localhost_media_address(auth_client: TestClient):
    with patch("netv.chromecast.manager.start") as start:
        response = auth_client.post(
            "/api/cast/start",
            json={
                "host": "192.168.1.50",
                "session_id": "abc",
                "server_url": "http://localhost:8000",
            },
        )
        assert response.status_code == 400
        start.assert_not_called()


def _release_and_report(
    username: str,
    action: str,
    volume: float | None,
    *,
    release: Callable[[str], None],
) -> dict[str, bool]:
    del username, action, volume
    release("cast-session")
    return {"active": True}


def test_cast_command_validation_and_user_scope(auth_client: TestClient):
    with (
        patch("netv.chromecast.manager.command", side_effect=_release_and_report) as command,
        patch("netv.main.ffmpeg_session.stop_session", new_callable=AsyncMock) as stop,
    ):
        assert (
            auth_client.post(
                "/api/cast/control",
                json={"action": "volume", "volume": 2},
            ).status_code
            == 422
        )
        assert auth_client.post("/api/cast/control", json={"action": "reboot"}).status_code == 422
        command.assert_not_called()
        assert auth_client.post("/api/cast/control", json={"action": "pause"}).status_code == 200
        assert command.call_args.args == ("testuser", "pause", None)
        stop.assert_awaited_once_with("cast-session", force=True)


def test_cast_discovery_reports_network_errors(auth_client: TestClient):
    with patch("netv.chromecast.manager.discover", side_effect=OSError("no network")):
        response = auth_client.get("/api/cast/devices")
    assert response.status_code == 503
    assert "manually" in _body(response, _Detail)["detail"]


def test_browser_cannot_stop_or_seek_cast_owned_session(auth_client: TestClient):
    with (
        patch("netv.chromecast.manager.owns_session", return_value=True),
        patch("netv.ffmpeg_session.stop_session") as stop,
    ):
        response = auth_client.post("/transcode/abc/stop?force=true")
        assert response.json() == {"status": "casting"}
        assert auth_client.delete("/transcode/abc?force=true").json() == {"status": "casting"}
        assert auth_client.get("/transcode/seek/abc?time=10").status_code == 409
        stop.assert_not_called()


def test_cast_source_cannot_be_restarted_while_receiver_owns_it(auth_client: TestClient):
    with (
        patch("netv.chromecast.manager.owns_url", return_value=True),
        patch("netv.ffmpeg_session.clear_url_session") as clear,
    ):
        assert auth_client.delete("/transcode-clear?url=http://provider/live").status_code == 409
        assert auth_client.get("/transcode/start?url=http://provider/live").status_code == 409
        clear.assert_not_called()


def test_receiver_hls_request_keeps_session_alive_without_cookie(
    client: TestClient,
    tmp_path: Path,
):
    (tmp_path / "stream.m3u8").write_text("#EXTM3U\n#EXTINF:6,\nseg001.ts\n")
    with (
        patch("netv.ffmpeg_session.get_session", return_value={"dir": str(tmp_path)}),
        patch("netv.ffmpeg_session.touch_session") as touch,
    ):
        response = client.get("/transcode/abc/stream.m3u8")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    touch.assert_called_once_with("abc")


def test_live_dvr_routes_auto_mode_through_server(auth_client: TestClient):

    info = PlayerInfo(url="https://example.test/live.ts", channel_name="Test channel")
    with patch("netv.main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert 'transcodeMode: "always"' in response.text
    assert "liveDvrMins: 120" in response.text


def test_live_dvr_disabled_keeps_direct_auto_mode(auth_client: TestClient):

    settings = cache_module.load_server_settings()
    settings["live_dvr_mins"] = 0
    cache_module.save_server_settings(settings)
    info = PlayerInfo(url="https://example.test/live.ts", channel_name="Test channel")
    with patch("netv.main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert 'transcodeMode: "auto"' in response.text
    assert "liveDvrMins: 0" in response.text


def test_live_player_passes_program_window_from_epg(auth_client: TestClient):

    info = PlayerInfo(
        url="https://example.test/live.ts",
        channel_name="Test channel",
        program_title="News",
        program_start=1749998200.0,
        program_end=1750000000.0,
    )
    with patch("netv.main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert "programStart: 1749998200.0" in response.text
    assert "programEnd: 1750000000.0" in response.text
    assert 'streamId: "test"' in response.text
    assert 'id="program-remaining"' not in response.text


def test_live_player_info_reads_program_window_from_epg():

    start = datetime.now(timezone.utc) - timedelta(minutes=30)
    stop = datetime.now(timezone.utc) + timedelta(minutes=30)
    program = Program(channel_id="ch1", title="News", start=start, stop=stop)
    stream = {
        "stream_id": "1",
        "name": "Channel",
        "direct_url": "https://example.test/live.ts",
        "epg_channel_id": "ch1",
    }
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": [stream]}),
        patch("netv.epg.get_programs_in_range", return_value=[program]),
    ):
        info = main._get_live_player_info("1")
    assert info.program_title == "News"
    assert info.program_start == start.timestamp()
    assert info.program_end == stop.timestamp()


def test_live_program_api_returns_current_program(auth_client: TestClient):

    start = datetime.now(timezone.utc) - timedelta(minutes=10)
    stop = start + timedelta(minutes=45)
    program = Program(
        channel_id="ch1",
        title="News",
        start=start,
        stop=stop,
        desc="Tonight's headlines",
    )
    stream = {"stream_id": "1", "name": "Channel", "epg_channel_id": "ch1"}
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": [stream]}),
        patch("netv.epg.get_programs_in_range", return_value=[program]),
    ):
        response = auth_client.get("/api/live/program/1")
    assert response.status_code == 200
    assert response.json() == {
        "title": "News",
        "desc": "Tonight's headlines",
        "start": start.timestamp(),
        "end": stop.timestamp(),
    }


def test_live_program_api_without_guide_data_returns_empty_program(auth_client: TestClient):
    stream = {"stream_id": "1", "name": "Channel"}
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": [stream]}),
    ):
        response = auth_client.get("/api/live/program/1")
    assert response.status_code == 200
    assert response.json() == {"title": "", "desc": "", "start": 0.0, "end": 0.0}


def test_live_program_api_unknown_stream_is_not_found(auth_client: TestClient):
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": []}),
    ):
        response = auth_client.get("/api/live/program/missing")
    assert response.status_code == 404


def test_catchup_player_info_builds_timeshift_url():

    start = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(second=0, microsecond=0)
    program = Program(
        channel_id="ch1",
        title="Earlier news",
        start=start,
        stop=start + timedelta(minutes=30),
    )
    client = MagicMock()
    client.get_server_info.return_value = {"server_info": {"timezone": "Europe/Amsterdam"}}
    tz_cache: dict[str, tuple[float, tzinfo]] = {}
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": [_archive_stream()]}),
        patch("netv.main.get_xtream_client_by_source", return_value=client),
        patch("netv.epg.get_programs_in_range", return_value=[program]),
        patch("netv.catchup._tz_cache", tz_cache),
    ):
        info = main._get_catchup_player_info("src1_42", start.timestamp())
    local = start.astimezone(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d:%H-%M")
    assert info.url == f"http://upstream.test/timeshift/user/pass/30/{local}/42.ts"
    assert info.catchup_start == start.timestamp()
    assert info.catchup_days == 2
    assert info.program_title == "Earlier news"
    assert info.program_end == program.stop.timestamp()


def test_catchup_player_info_starts_mid_program():

    program_start = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(
        second=0,
        microsecond=0,
    )
    program = Program(
        channel_id="ch1",
        title="Earlier news",
        start=program_start,
        stop=program_start + timedelta(minutes=30),
    )
    target = program_start + timedelta(minutes=10, seconds=25)
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": [_archive_stream()]}),
        patch("netv.main.get_xtream_client_by_source", return_value=None),
        patch("netv.epg.get_programs_in_range", return_value=[program]),
    ):
        info = main._get_catchup_player_info("src1_42", target.timestamp())
    begin = program_start + timedelta(minutes=10)
    assert f"/20/{begin:%Y-%m-%d:%H-%M}/42.ts" in info.url
    assert info.catchup_start == begin.timestamp()
    assert info.catchup_seek == 25
    assert info.program_start == program_start.timestamp()
    assert info.program_end == program.stop.timestamp()


def test_catchup_player_info_rejects_programs_outside_archive():

    too_old = datetime.now(timezone.utc) - timedelta(days=3)
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": [_archive_stream()]}),
        patch("netv.epg.get_programs_in_range", return_value=[]),
        pytest.raises(HTTPException) as excinfo,
    ):
        main._get_catchup_player_info("src1_42", too_old.timestamp())
    assert excinfo.value.status_code == 404


def test_catchup_player_page_plays_archive_as_vod(auth_client: TestClient):

    info = PlayerInfo(
        url="http://upstream.test/timeshift/user/pass/30/2026-01-01:10-00/42.ts",
        channel_name="Channel",
        catchup_days=2,
        catchup_start=1767261600.0,
        catchup_seek=25.0,
    )
    with patch("netv.main._get_catchup_player_info", return_value=info) as get_info:
        response = auth_client.get("/play/live/src1_42?start=1767261600")
    get_info.assert_called_once_with("src1_42", 1767261600.0)
    assert response.status_code == 200
    assert "catchup: true" in response.text
    assert "catchupStart: 1767261600.0" in response.text
    assert "catchupSeek: 25.0" in response.text
    assert 'id="go-live-btn"' in response.text
    assert 'id="jump-btn"' in response.text
    assert 'id="start-over-btn"' not in response.text


def test_live_player_offers_start_over_with_archive(auth_client: TestClient):

    info = PlayerInfo(url="https://example.test/live.ts", channel_name="Channel", catchup_days=1)
    with patch("netv.main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/test")
    assert "catchup: false" in response.text
    assert "catchupDays: 1" in response.text
    assert 'id="start-over-btn"' in response.text


def test_catchup_api_lists_archived_programs_newest_first(auth_client: TestClient):

    now = datetime.now(timezone.utc)
    programs = [
        Program(
            channel_id="ch1",
            title="Too old",
            start=now - timedelta(days=2, hours=1),
            stop=now - timedelta(days=2) + timedelta(minutes=1),
        ),
        Program(
            channel_id="ch1",
            title="Morning",
            start=now - timedelta(hours=5),
            stop=now - timedelta(hours=4),
        ),
        Program(
            channel_id="ch1",
            title="On now",
            start=now - timedelta(minutes=10),
            stop=now + timedelta(minutes=20),
        ),
    ]
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": [_archive_stream()]}),
        patch("netv.epg.get_programs_in_range", return_value=programs),
    ):
        response = auth_client.get("/api/live/catchup/src1_42")
    assert response.status_code == 200
    payload = _body(response, CatchupListing)
    assert payload["days"] == 2
    assert [p["title"] for p in payload["programs"]] == ["On now", "Morning"]
    assert payload["programs"][1]["start_timestamp"] == programs[1].start.timestamp()


def test_catchup_api_without_archive_is_empty(auth_client: TestClient):
    with (
        patch("netv.main._ensure_live_cache"),
        patch(
            "netv.main.get_cache",
            return_value={"live_streams": [_archive_stream(tv_archive=0)]},
        ),
    ):
        response = auth_client.get("/api/live/catchup/src1_42")
    assert response.json() == {"days": 0, "programs": []}


def test_catchup_api_uses_playlist_guide_assignment(auth_client: TestClient):

    now = datetime.now(timezone.utc)
    earlier = Program(
        channel_id="assigned",
        title="Morning",
        start=now - timedelta(hours=5),
        stop=now - timedelta(hours=4),
    )
    stream = _archive_stream(epg_channel_id="")
    playlist = {
        "id": "pl1",
        "channels": [
            {
                "stream_id": "src1_42",
                "source_id": "src1",
                "name": "Channel",
                "epg_channel_id": "assigned",
            },
        ],
    }
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": [stream]}),
        patch("netv.main.playlists.load", return_value=[playlist]),
        patch("netv.epg.get_programs_in_range", return_value=[earlier]) as get_programs,
    ):
        response = auth_client.get("/api/live/catchup/src1_42")
    assert [p["title"] for p in _body(response, CatchupListing)["programs"]] == ["Morning"]
    assert get_programs.call_args.args[0] == "assigned"


def test_catchup_player_page_rejects_invalid_start(auth_client: TestClient):
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": [_archive_stream()]}),
        patch("netv.epg.get_programs_in_range", return_value=[]),
    ):
        assert auth_client.get("/play/live/src1_42?start=1e300").status_code == 404


def test_catchup_api_unknown_stream_is_not_found(auth_client: TestClient):
    with (
        patch("netv.main._ensure_live_cache"),
        patch("netv.main.get_cache", return_value={"live_streams": []}),
    ):
        assert auth_client.get("/api/live/catchup/missing").status_code == 404


def test_adaptive_start_uses_shared_backend(auth_client: TestClient):
    result = {
        "session_id": "shared",
        "playlist": "/transcode/shared/low.m3u8",
        "master_playlist": "/transcode/shared/master.m3u8",
    }
    with patch("netv.ffmpeg_session.start_transcode", new=AsyncMock(return_value=result)) as start:
        response = auth_client.get(
            "/transcode/start",
            params={"url": "https://provider.example/live.m3u8", "fast_start": "true"},
        )
    assert response.status_code == 200
    assert response.json() == result
    assert start.call_args.kwargs["fast_start"] is True


def test_disconnected_adaptive_start_is_cleaned_up(auth_client: TestClient):
    with (
        patch(
            "netv.ffmpeg_session.start_transcode",
            new=AsyncMock(return_value={"session_id": "new"}),
        ),
        patch("starlette.requests.Request.is_disconnected", new=AsyncMock(return_value=True)),
        patch("netv.ffmpeg_session.stop_session") as stop,
    ):
        response = auth_client.get(
            "/transcode/start",
            params={"url": "https://provider.example/live.m3u8", "fast_start": "true"},
        )
    assert response.status_code == 499
    stop.assert_called_once_with("new", force=True)


@pytest.mark.parametrize("owner,status", [("testuser", 200), ("other", 404)])
def test_live_page_close_force_stop_checks_owner(auth_client: TestClient, owner: str, status: int):
    with (
        patch("netv.ffmpeg_session.get_session", return_value={"username": owner}),
        patch("netv.ffmpeg_session.stop_session") as stop,
    ):
        response = auth_client.post("/transcode/live/stop?force=true")
    assert response.status_code == status
    if status == 200:
        stop.assert_called_once_with("live", force=True)
    else:
        stop.assert_not_called()


@pytest.mark.parametrize("owner,status", [("testuser", 200), ("other", 404)])
def test_archive_stop_without_force_still_releases_only_owner_session(
    auth_client: TestClient,
    owner: str,
    status: int,
):
    with (
        patch(
            "netv.ffmpeg_session.get_session",
            return_value={"username": owner, "is_archive": True},
        ),
        patch("netv.ffmpeg_session.stop_session") as stop,
    ):
        response = auth_client.post("/transcode/archive/stop")
    assert response.status_code == status
    if status == 200:
        stop.assert_called_once_with("archive", force=True)
    else:
        stop.assert_not_called()


class TestLoadAllEpg:
    def test_refetches_when_listings_ran_out(self, mock_deps: None):

        urls = [("http://epg", 30, "s")]
        with (
            patch("netv.main.epg.has_programs_after", return_value=False),
            patch("netv.main._fetch_all_epg") as fetch,
        ):
            main.load_all_epg(urls)
        fetch.assert_called_once_with(urls)

    def test_skips_fetch_when_listings_are_current(self, mock_deps: None):

        with (
            patch("netv.main.epg.has_programs_after", return_value=True),
            patch("netv.main.epg.get_program_count", return_value=5),
            patch("netv.main._fetch_all_epg") as fetch,
        ):
            main.load_all_epg([("http://epg", 30, "s")])
        fetch.assert_not_called()


class TestSetup:
    """Tests for initial setup flow."""

    def test_setup_page_shown_when_no_users(self, client: TestClient):
        resp = client.get("/setup", follow_redirects=False)
        assert resp.status_code == 200
        assert b"setup" in resp.content.lower() or b"Create" in resp.content

    def test_setup_redirects_when_users_exist(self, client: TestClient, tmp_path: Path):

        auth.create_user("admin", "password123")

        resp = client.get("/setup", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_setup_creates_user(self, client: TestClient):
        resp = client.post(
            "/setup",
            data={"username": "admin", "password": "password123", "confirm": "password123"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

        assert auth.verify_password("admin", "password123")

    def test_setup_validates_username_length(self, client: TestClient):
        resp = client.post(
            "/setup",
            data={"username": "ab", "password": "password123", "confirm": "password123"},
        )
        assert resp.status_code == 200
        assert b"at least 3" in resp.content

    def test_setup_validates_password_length(self, client: TestClient):
        resp = client.post(
            "/setup",
            data={"username": "admin", "password": "short", "confirm": "short"},
        )
        assert resp.status_code == 200
        assert b"at least 8" in resp.content

    def test_setup_validates_password_match(self, client: TestClient):
        resp = client.post(
            "/setup",
            data={"username": "admin", "password": "password123", "confirm": "different"},
        )
        assert resp.status_code == 200
        assert b"do not match" in resp.content


class TestLogin:
    """Tests for login flow."""

    def test_login_page_redirects_to_setup_when_no_users(self, client: TestClient):
        resp = client.get("/login", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/setup"

    def test_login_page_shown_when_users_exist(self, client: TestClient, tmp_path: Path):

        auth.create_user("admin", "password123")

        resp = client.get("/login")
        assert resp.status_code == 200

    def test_login_success_sets_cookie(self, client: TestClient, tmp_path: Path):

        auth.create_user("admin", "password123")

        resp = client.post(
            "/login",
            data={"username": "admin", "password": "password123"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "token" in resp.cookies

    def test_login_failure_returns_401(self, client: TestClient, tmp_path: Path):

        auth.create_user("admin", "password123")

        resp = client.post(
            "/login",
            data={"username": "admin", "password": "wrongpassword"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "error=invalid" in resp.headers["location"]


class TestLogout:
    """Tests for logout."""

    def test_logout_clears_cookie(self, auth_client: TestClient):
        resp = auth_client.get("/logout", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"


class TestAuthRequired:
    """Tests for auth-protected routes."""

    def test_index_redirects_to_login(self, client: TestClient, tmp_path: Path):

        auth.create_user("admin", "password123")

        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_guide_redirects_to_login(self, client: TestClient, tmp_path: Path):

        auth.create_user("admin", "password123")

        resp = client.get("/guide", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_vod_redirects_to_login(self, client: TestClient, tmp_path: Path):

        auth.create_user("admin", "password123")

        resp = client.get("/vod", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_series_redirects_to_login(self, client: TestClient, tmp_path: Path):

        auth.create_user("admin", "password123")

        resp = client.get("/series", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"


class TestIndex:
    """Tests for index route."""

    def test_index_redirects_to_guide(self, auth_client: TestClient):
        resp = auth_client.get("/", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/guide"


class TestFavicon:
    """Tests for favicon."""

    def test_favicon_returns_204(self, client: TestClient):
        resp = client.get("/favicon.ico")
        assert resp.status_code == 204


class TestGuide:
    """Tests for guide page."""

    def test_guide_shows_loading_when_no_cache(self, auth_client: TestClient):
        with patch("netv.m3u.load_file_cache", return_value=None):
            resp = auth_client.get("/guide")
            assert resp.status_code == 200
            # Should show loading state.
            assert b"loading" in resp.content.lower() or b"Loading" in resp.content

    def test_guide_shows_channels_from_cache(self, auth_client: TestClient):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN", "category_ids": ["1"], "epg_channel_id": ""},
        ]

        with patch("netv.main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide?cats=1")
            assert resp.status_code == 200

    def test_guide_limits_initial_virtual_rows(self, auth_client: TestClient):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": stream_id,
                "name": f"Channel {stream_id}",
                "category_ids": ["1"],
                "epg_channel_id": "",
                "stream_icon": f"https://example.com/{stream_id}.png",
            }
            for stream_id in range(60)
        ]

        with patch("netv.main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide?cats=1")

        assert resp.status_code == 200
        assert resp.text.count('class="guide-row') == 50
        assert resp.text.count('loading="lazy"') == 50

    def test_guide_rows_include_category_metadata(self, auth_client: TestClient):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": 1,
                "name": "CNN",
                "category_ids": ["1"],
                "epg_channel_id": "",
            },
        ]

        resp = auth_client.get("/api/guide/rows?cats=1")

        assert resp.status_code == 200
        payload = _body(resp, GuideRowsResponse)
        assert payload["categories"] == [{"category_id": "1", "category_name": "News"}]
        assert payload["rows"][0]["channel"]["category_ids"] == ["1"]

    def test_guide_rows_categories_cover_all_pages(self, auth_client: TestClient):
        categories: list[ProviderRecord] = [
            {"category_id": "1", "category_name": "News"},
            {"category_id": "2", "category_name": "Sports"},
            {"category_id": "3", "category_name": "Unavailable"},
        ]
        cache_module.get_cache()["live_categories"] = categories
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": index,
                "name": f"Channel {index}",
                "category_ids": ["1" if index < 500 else "2"],
            }
            for index in range(501)
        ] + [{"stream_id": 999, "name": "Hidden", "category_ids": ["3"]}]

        with patch(
            "netv.main.auth.get_user_limits",
            return_value={"unavailable_groups": ["cat:3"]},
        ):
            first = _body(
                auth_client.get("/api/guide/rows?cats=1,2,3&count=500"),
                GuideRowsResponse,
            )
            second = _body(
                auth_client.get("/api/guide/rows?cats=1,2,3&start=500&count=500"),
                GuideRowsResponse,
            )
            reordered = _body(
                auth_client.get("/api/guide/rows?cats=2,1,3&count=1"),
                GuideRowsResponse,
            )

        assert first["total"] == second["total"] == 501
        assert len(first["rows"]) == 500
        assert len(second["rows"]) == 1
        assert second["rows"][0]["channel"]["stream_id"] == 500
        assert first["categories"] == second["categories"] == categories[:2]
        assert reordered["categories"] == categories[1::-1]
        assert reordered["rows"][0]["channel"]["stream_id"] == 500

    def test_guide_rows_timestamps_anchor_local_timeline(self, auth_client: TestClient):

        now = datetime(2026, 9, 14, 0, 15, tzinfo=timezone.utc)
        window_start = now.replace(minute=0)
        program_start = window_start - timedelta(minutes=30)
        program_end = window_start + timedelta(hours=1)
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "News", "category_ids": ["1"], "epg_channel_id": "news"},
        ]
        program = Program(
            channel_id="news",
            title="Overnight news",
            start=program_start,
            stop=program_end,
        )

        with (
            patch("netv.main.datetime") as clock,
            patch("netv.main.epg.get_icons_batch", return_value={}),
            patch("netv.main.epg.get_programs_batch", return_value={"news": [program]}),
        ):
            clock.now.return_value = now
            response = auth_client.get("/api/guide/rows?cats=1")

        assert response.status_code == 200
        payload = _body(response, GuideRowsResponse)
        listing = payload["rows"][0]["programs"][0]
        assert payload.get("window_start_timestamp") == window_start.timestamp()
        assert listing["start_timestamp"] == program_start.timestamp()
        assert listing["end_timestamp"] == program_end.timestamp()
        assert listing["start"] == "23:30"
        assert listing["end"] == "01:00"
        assert listing["left_pct"] == 0
        assert listing["width_pct"] == pytest.approx(100 / 3)

    def test_guide_rows_flag_archived_programs(self, auth_client: TestClient):

        now = datetime.now(timezone.utc)
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": 1,
                "name": "News",
                "category_ids": ["1"],
                "epg_channel_id": "news",
                "source_type": "xtream",
                "tv_archive": 1,
                "tv_archive_duration": 1,
            },
            {"stream_id": 2, "name": "Other", "category_ids": ["1"], "epg_channel_id": "other"},
        ]
        past = Program(
            channel_id="news",
            title="Earlier",
            start=now - timedelta(hours=1),
            stop=now - timedelta(minutes=5),
        )
        current = Program(
            channel_id="news",
            title="Now",
            start=now - timedelta(minutes=5),
            stop=now + timedelta(hours=1),
        )
        other = Program(channel_id="other", title="Earlier", start=past.start, stop=past.stop)

        with (
            patch("netv.main.epg.get_icons_batch", return_value={}),
            patch(
                "netv.main.epg.get_programs_batch",
                return_value={"news": [past, current], "other": [other]},
            ),
        ):
            rows = _body(auth_client.get("/api/guide/rows?cats=1&offset=-1"), GuideRowsResponse)[
                "rows"
            ]

        assert rows[0]["channel"]["catchup_days"] == 1
        assert [p["catchup"] for p in rows[0]["programs"]] == [True, False]
        assert [p.get("unavailable") for p in rows[0]["programs"]] == [False, False]
        assert rows[1]["channel"]["catchup_days"] == 0
        assert [p["catchup"] for p in rows[1]["programs"]] == [False]
        assert rows[1]["programs"][0].get("unavailable") is True
        assert rows[1]["programs_mobile"][0].get("unavailable") is True
        mobile = {p["title"]: p for p in rows[0]["programs_mobile"]}
        assert mobile["Earlier"]["catchup"] is True
        assert mobile["Earlier"]["start_timestamp"] == past.start.timestamp()
        assert mobile["Earlier"]["end_timestamp"] == past.stop.timestamp()

    @pytest.mark.parametrize("archive_days", [0, 1, 3])
    def test_guide_disables_unavailable_past_listings(
        self,
        auth_client: TestClient,
        archive_days: int,
    ):

        class ProgramLinks(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.links: list[dict[str, str]] = []

            @override
            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                attributes = {name: "" if value is None else value for name, value in attrs}
                if tag == "a" and "data-start" in attributes:
                    self.links.append(attributes)

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=2)
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "Group"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": 1,
                "name": "Stream",
                "category_ids": ["1"],
                "epg_channel_id": "guide",
                "source_type": "xtream",
                "tv_archive": 1,
                "tv_archive_duration": archive_days,
            },
        ]
        program = Program(
            channel_id="guide",
            title="Earlier program",
            start=start,
            stop=start + timedelta(hours=1),
        )
        with (
            patch("netv.main.epg.has_programs", return_value=True),
            patch("netv.main.epg.get_icons_batch", return_value={}),
            patch("netv.main.epg.get_programs_batch", return_value={"guide": [program]}),
        ):
            response = auth_client.get("/guide?cats=1&offset=-48")
            rows = _body(auth_client.get("/api/guide/rows?cats=1&offset=-48"), GuideRowsResponse)[
                "rows"
            ]
        assert response.status_code == 200
        parser = ProgramLinks()
        parser.feed(response.text)
        assert len(parser.links) == 2
        for link in parser.links:
            if archive_days == 3:
                assert link["href"] == f"/play/live/1?start={int(start.timestamp())}"
                assert link["data-nav"] == "epg"
                assert "aria-disabled" not in link
            else:
                assert "href" not in link
                assert "data-nav" not in link
                assert "focusable" not in link["class"]
                assert link["aria-disabled"] == "true"
                assert link["tabindex"] == "-1"
                assert "Not available in the upstream archive" in link["title"]
        for programs in (rows[0]["programs"], rows[0]["programs_mobile"]):
            assert programs[0].get("unavailable") is (archive_days != 3)

    def test_guide_page_exposes_timestamps_for_local_times(self, auth_client: TestClient):

        now = datetime.now(timezone.utc)
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "News", "category_ids": ["1"], "epg_channel_id": "news"},
        ]
        program = Program(
            channel_id="news",
            title="Headlines",
            start=now - timedelta(minutes=5),
            stop=now + timedelta(minutes=55),
        )
        with (
            patch("netv.main.epg.has_programs", return_value=True),
            patch("netv.main.epg.get_icons_batch", return_value={}),
            patch("netv.main.epg.get_programs_batch", return_value={"news": [program]}),
        ):
            html = auth_client.get("/guide?cats=1").text
        window_start = now.replace(minute=0, second=0, microsecond=0)
        assert f'data-clock="{window_start.timestamp()}"' in html
        assert f'data-start="{program.start.timestamp()}"' in html
        assert f'data-end="{program.stop.timestamp()}"' in html
        assert "localizeGuideTimes(document)" in html

    def test_guide_uses_saved_filter(self, auth_client: TestClient, tmp_path: Path):
        user_dir = tmp_path / "users" / "testuser"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "settings.json").write_text(json.dumps({"guide_filter": ["1", "2"]}))

        cache_module.get_cache()["live_categories"] = []
        cache_module.get_cache()["live_streams"] = []

        # Guide now renders directly using saved filter (no redirect)
        with patch("netv.main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide")
            assert resp.status_code == 200


class TestVod:
    """Tests for VOD page."""

    def test_vod_shows_loading_when_no_cache(self, auth_client: TestClient):
        with patch("netv.m3u.load_file_cache", return_value=None):
            resp = auth_client.get("/vod")
            assert resp.status_code == 200

    def test_vod_shows_movies_from_cache(self, auth_client: TestClient):
        cache_module.get_cache()["vod_categories"] = [
            {"category_id": "10", "category_name": "Movies", "source_id": "src1"},
        ]
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 100, "name": "Movie 1", "category_id": "10", "source_id": "src1"},
        ]

        resp = auth_client.get("/vod")
        assert resp.status_code == 200

    def test_vod_filters_by_category(self, auth_client: TestClient):
        cache_module.get_cache()["vod_categories"] = [
            {"category_id": "10", "category_name": "Action", "source_id": "src1"},
            {"category_id": "20", "category_name": "Comedy", "source_id": "src1"},
        ]
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 100, "name": "Action Movie", "category_id": "10", "source_id": "src1"},
            {"stream_id": 101, "name": "Comedy Movie", "category_id": "20", "source_id": "src1"},
        ]

        resp = auth_client.get("/vod?category=10")
        assert resp.status_code == 200

    def test_vod_sorts_by_alpha(self, auth_client: TestClient):
        cache_module.get_cache()["vod_categories"] = []
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 1, "name": "Zebra", "source_id": "src1"},
            {"stream_id": 2, "name": "Apple", "source_id": "src1"},
        ]

        resp = auth_client.get("/vod?sort=alpha")
        assert resp.status_code == 200


class TestSeries:
    """Tests for series page."""

    def test_series_shows_loading_when_no_cache(self, auth_client: TestClient):
        with patch("netv.m3u.load_file_cache", return_value=None):
            resp = auth_client.get("/series")
            assert resp.status_code == 200

    def test_series_shows_list_from_cache(self, auth_client: TestClient):
        cache_module.get_cache()["series_categories"] = [
            {"category_id": "30", "category_name": "Drama", "source_id": "src1"},
        ]
        cache_module.get_cache()["series"] = [
            {"series_id": 200, "name": "Show 1", "category_id": "30", "source_id": "src1"},
        ]

        resp = auth_client.get("/series")
        assert resp.status_code == 200


class TestSearch:
    """Tests for search page."""

    def test_search_page_renders(self, auth_client: TestClient):
        cache_module.get_cache()["live_streams"] = []
        cache_module.get_cache()["vod_streams"] = []
        cache_module.get_cache()["series"] = []

        resp = auth_client.get("/search")
        assert resp.status_code == 200

    def test_search_finds_live_streams(self, auth_client: TestClient):
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN News"},
            {"stream_id": 2, "name": "BBC World"},
        ]
        cache_module.get_cache()["live_categories"] = []
        cache_module.get_cache()["epg_urls"] = []
        cache_module.get_cache()["vod_streams"] = []
        cache_module.get_cache()["series"] = []

        resp = auth_client.get("/search?q=CNN&live=true")
        assert resp.status_code == 200

    def test_search_regex_mode(self, auth_client: TestClient):
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN News"},
            {"stream_id": 2, "name": "CNBC Finance"},
        ]
        cache_module.get_cache()["live_categories"] = []
        cache_module.get_cache()["epg_urls"] = []
        cache_module.get_cache()["vod_streams"] = []
        cache_module.get_cache()["series"] = []

        resp = auth_client.get("/search?q=CN.*&regex=true&live=true")
        assert resp.status_code == 200

    def test_search_rejects_long_regex(self, auth_client: TestClient):
        cache_module.get_cache()["live_streams"] = []

        resp = auth_client.get(f"/search?q={'a' * 101}&regex=true&live=true")
        assert resp.status_code == 400


class TestSettings:
    """Tests for settings page."""

    def test_only_nomos_engine_is_discovered(self, auth_client: TestClient, tmp_path: Path):

        (tmp_path / "retired-model_720p_fp16.engine").write_bytes(b"engine")
        with patch.dict("os.environ", {"SR_ENGINE_DIR": str(tmp_path)}):
            assert main.get_sr_models() == []

            (tmp_path / "2x-nomosuni-compact_720p_fp16.engine").write_bytes(b"engine")
            assert main.get_sr_models() == ["2x-nomosuni-compact"]

    def test_settings_page_renders(self, auth_client: TestClient):
        cache_module.get_cache()["live_categories"] = []

        with patch("netv.m3u.load_file_cache", return_value=None):
            resp = auth_client.get("/settings")
            assert resp.status_code == 200
            assert 'id="sort-live-by-views" checked' in resp.text

    def test_settings_page_maps_enabled_retired_model_to_nomos(self, auth_client: TestClient):
        cache_module.get_cache()["live_categories"] = []

        with (
            patch("netv.m3u.load_file_cache", return_value=None),
            patch(
                "netv.main.load_server_settings",
                return_value={"sr_model": "retired-model"},
            ),
            patch(
                "netv.main.get_sr_models",
                return_value=["2x-nomosuni-compact"],
            ),
        ):
            resp = auth_client.get("/settings")

        assert resp.status_code == 200
        assert resp.text.count('name="sr_model"') == 2
        assert 'value="2x-nomosuni-compact" checked' in resp.text
        assert "NomosUni 2x (480p/720p/1080p)" in resp.text

    def test_settings_guide_filter(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/guide-filter",
            json={"cats": ["1", "2", "3"]},
        )
        assert resp.status_code == 200
        assert _fields(resp)["status"] == "ok"

    def test_settings_captions(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/captions",
            data={"enabled": "on"},
        )
        assert resp.status_code == 200
        assert _fields(resp)["ok"] is True

    def test_settings_transcode(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/transcode",
            data={
                "transcode_mode": "auto",
                "transcode_hw": "nvidia",
                "vod_transcode_cache_mins": 60,
            },
        )
        assert resp.status_code == 200
        assert _fields(resp)["ok"] is True

    def test_settings_transcode_saves_long_live_dvr(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/transcode",
            data={
                "transcode_mode": "auto",
                "transcode_hw": "nvidia",
                "live_dvr_mins": 999,
            },
        )
        assert resp.status_code == 200
        assert cache_module.load_server_settings()["live_dvr_mins"] == 999


class TestAddSource:
    """Tests for adding sources."""

    def test_add_xtream_source(self, auth_client: TestClient):
        with patch("netv.main.clear_all_caches"):
            resp = auth_client.post(
                "/settings/add",
                data={
                    "name": "Test Provider",
                    "source_type": "xtream",
                    "url": "http://example.com",
                    "username": "user",
                    "password": "pass",
                    "epg_timeout": 120,
                },
                follow_redirects=False,
            )
            assert resp.status_code == 303

    def test_add_m3u_source(self, auth_client: TestClient):
        with patch("netv.main.clear_all_caches"):
            resp = auth_client.post(
                "/settings/add",
                data={
                    "name": "M3U Playlist",
                    "source_type": "m3u",
                    "url": "http://example.com/playlist.m3u",
                    "epg_timeout": 120,
                },
                follow_redirects=False,
            )
            assert resp.status_code == 303

    def test_add_source_validates_type(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/add",
            data={
                "name": "Bad Source",
                "source_type": "invalid",
                "url": "http://example.com",
            },
        )
        assert resp.status_code == 400

    def test_add_source_validates_url_scheme(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/add",
            data={
                "name": "Bad Source",
                "source_type": "xtream",
                "url": "ftp://example.com",
            },
        )
        assert resp.status_code == 400

    def test_add_source_validates_name_length(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/add",
            data={
                "name": "x" * 201,
                "source_type": "xtream",
                "url": "http://example.com",
            },
        )
        assert resp.status_code == 400


class TestDeleteSource:
    """Tests for deleting sources."""

    def test_delete_source(self, auth_client: TestClient, tmp_path: Path):
        settings_file = tmp_path / "server_settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "id": "src_123",
                            "name": "Test",
                            "type": "xtream",
                            "url": "http://example.com",
                        },
                    ],
                },
            ),
        )

        with patch("netv.main.clear_all_caches"):
            resp = auth_client.post("/settings/delete/src_123", follow_redirects=False)
            assert resp.status_code == 303


class TestUserPrefs:
    """Tests for user preferences API."""

    def test_get_user_prefs(self, auth_client: TestClient):
        resp = auth_client.get("/api/user-prefs")
        assert resp.status_code == 200
        data = _fields(resp)
        assert "favorites" in data
        assert "cc_lang" in data
        assert data["sort_live_by_views"] is True

    def test_save_user_prefs(self, auth_client: TestClient):
        resp = auth_client.post(
            "/api/user-prefs",
            json={
                "cc_lang": "eng",
                "cast_host": "192.168.1.100",
                "sort_live_by_views": False,
            },
        )
        assert resp.status_code == 200
        assert _fields(resp)["ok"] is True
        assert _fields(auth_client.get("/api/user-prefs"))["sort_live_by_views"] is False

    def test_view_sorting_preference_requires_boolean(self, auth_client: TestClient):
        resp = auth_client.post(
            "/api/user-prefs",
            json={"sort_live_by_views": "false"},
        )

        assert resp.status_code == 400


class TestWatchPosition:
    """Tests for watch position API."""

    def test_save_watch_position(self, auth_client: TestClient):
        resp = auth_client.post(
            "/api/watch-position",
            json={"url": "http://example.com/movie.mkv", "position": 1234.5, "duration": 7200},
        )
        assert resp.status_code == 200

    def test_get_watch_position(self, auth_client: TestClient):
        # Save first.
        auth_client.post(
            "/api/watch-position",
            json={"url": "http://example.com/movie.mkv", "position": 1234.5, "duration": 7200},
        )

        resp = auth_client.get("/api/watch-position?url=http://example.com/movie.mkv")
        assert resp.status_code == 200
        data = _fields(resp)
        assert data["position"] == 1234.5
        assert data["duration"] == 7200

    def test_get_watch_position_not_found(self, auth_client: TestClient):
        resp = auth_client.get("/api/watch-position?url=http://example.com/unknown.mkv")
        assert resp.status_code == 200
        data = _fields(resp)
        assert data["position"] == 0


class TestUserManagement:
    """Tests for user management endpoints."""

    def test_delete_user(self, auth_client: TestClient, tmp_path: Path):

        auth.create_user("otheruser", "password123")

        resp = auth_client.post("/settings/users/delete/otheruser", follow_redirects=False)
        assert resp.status_code == 303

    def test_cannot_delete_self(self, auth_client: TestClient):
        resp = auth_client.post("/settings/users/delete/testuser")
        assert resp.status_code == 400

    def test_change_password(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/users/password",
            data={"current_password": "testpass123", "new_password": "newpass456"},
        )
        assert resp.status_code == 200

    def test_change_password_wrong_current(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/users/password",
            data={"current_password": "wrongpass", "new_password": "newpass456"},
        )
        assert resp.status_code == 400


class TestPlaylistXspf:
    """Tests for XSPF playlist generation."""

    def test_playlist_xspf(self, auth_client: TestClient):
        resp = auth_client.get("/playlist.xspf?url=http://example.com/stream.m3u8")
        assert resp.status_code == 200
        assert b"<?xml" in resp.content
        assert b"http://example.com/stream.m3u8" in resp.content
        assert resp.headers["content-type"] == "application/xspf+xml"


class TestApiSettings:
    """Tests for settings API."""

    def test_get_settings(self, auth_client: TestClient):
        resp = auth_client.get("/api/settings")
        assert resp.status_code == 200
        data = _fields(resp)
        assert "transcode_mode" in data

    def test_update_settings(self, auth_client: TestClient):
        resp = auth_client.post(
            "/api/settings",
            json={"transcode_mode": "always"},
        )
        assert resp.status_code == 200


class TestTranscodeRoutes:
    """Tests for transcode routes (with mocked transcoding module)."""

    def test_transcode_file_not_found(self, auth_client: TestClient):
        with patch("netv.main.ffmpeg_session.get_session", return_value=None):
            resp = auth_client.get("/transcode/invalid-session/stream.m3u8")
            assert resp.status_code == 404

    def test_transcode_stop(self, auth_client: TestClient):
        with patch("netv.main.ffmpeg_session.stop_session"):
            resp = auth_client.delete("/transcode/test-session")
            assert resp.status_code == 200
            assert _fields(resp)["status"] == "stopped"

    def test_transcode_stop_post(self, auth_client: TestClient):
        with patch("netv.main.ffmpeg_session.stop_session"):
            resp = auth_client.post("/transcode/test-session/stop")
            assert resp.status_code == 200

    def test_transcode_progress_not_found(self, auth_client: TestClient):
        with patch("netv.main.ffmpeg_session.get_session_progress", return_value=None):
            resp = auth_client.get("/transcode/progress/invalid-session")
            assert resp.status_code == 404

    @pytest.mark.parametrize("filename", ["master.m3u8", "low.m3u8"])
    def test_adaptive_playlists_share_dates_but_master_is_not_rewritten(
        self,
        auth_client: TestClient,
        tmp_path: Path,
        filename: str,
    ):
        content = "#EXTM3U\n"
        (tmp_path / filename).write_text(content)
        session = {
            "dir": str(tmp_path),
            "fast_start": True,
            "origin_pts": 10,
            "origin_time": 0,
            "playback_buffer_seconds": 24,
        }
        with (
            patch("netv.ffmpeg_session.get_session", return_value=session),
            patch(
                "netv.main.ffmpeg_session.dated_fast_playlist",
                return_value=content + "#dated\n",
            ) as dates,
            patch(
                "netv.main.add_live_start_offset",
                side_effect=_mark_start,
            ) as start,
        ):
            response = auth_client.get(f"/transcode/shared/{filename}")
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"
        if filename == "master.m3u8":
            dates.assert_not_called()
            start.assert_not_called()
            assert response.text == content
        else:
            dates.assert_called_once_with("shared", content)
            start.assert_called_once_with(content + "#dated\n", 24)
            assert "#dated" in response.text
            assert "#start" in response.text

    def test_segmented_webvtt_is_mapped_to_the_video_clock(
        self,
        auth_client: TestClient,
        tmp_path: Path,
    ):
        filename = "sub0_000001.vtt"
        (tmp_path / filename).write_text("WEBVTT\n\n00:01.000 --> 00:03.000\nCaption\n")
        session = {"dir": str(tmp_path)}
        with (
            patch("netv.main.ffmpeg_session.get_session", return_value=session),
            patch(
                "netv.main.ffmpeg_session.get_caption_timestamp_origin",
                return_value=10.08,
            ),
        ):
            response = auth_client.get(f"/transcode/session/{filename}")

        assert response.status_code == 200
        assert "X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:907200" in response.text


def _fast_session(
    directory: Path,
    *,
    origin_pts: float | None = None,
    origin_time: float = 0.0,
    process: ffmpeg_session.ManagedProcess | None = None,
    ingest_process: ffmpeg_session.ManagedProcess | None = None,
    extra_processes: Sequence[ffmpeg_session.ManagedProcess] = (),
) -> ffmpeg_session.FastLiveSession:
    return {
        "dir": str(directory),
        "process": process if process is not None else FakeProcess(),
        "extra_processes": list(extra_processes),
        "ingest_process": ingest_process,
        "origin_pts": origin_pts,
        "origin_time": origin_time,
        "url": "https://upstream.example/live/1.m3u8",
        "username": "testuser",
        "started": 0,
        "last_access": 0,
        "fast_start": True,
        "live_recovery": LiveRecoveryPolicy(),
        "ingest_restart_command": ["ingest"],
        "low_restart_command": ["encoder", "low.m3u8"],
        "playlist_generation": 0,
    }


def _write_segments(directory: Path, names_and_pts: list[tuple[str, float]]) -> str:
    for name, pts in names_and_pts:
        (directory / name).write_bytes(video_packet(pts) * 10)
    return "".join(f"#EXTINF:2.000000,\n{name}\n" for name, _ in names_and_pts)


def _program_dates(text: str) -> dict[str, float]:
    dates = {}
    stamp = 0.0
    for line in text.splitlines():
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            stamp = datetime.fromisoformat(
                line.removeprefix("#EXT-X-PROGRAM-DATE-TIME:"),
            ).timestamp()
        elif line.endswith(".ts"):
            dates[line] = stamp
    return dates


def _close_background(coro: Coroutine[object, object, None]) -> None:
    coro.close()


class TestFastStartRoutes:
    def test_program_dates_stay_monotonic_after_pipeline_restart(
        self,
        auth_client: TestClient,
        tmp_path: Path,
    ):
        directory = tmp_path / "fast"
        directory.mkdir()
        header = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n"
        old = _write_segments(directory, [(f"low_{i}.ts", 5000 + 2 * i) for i in range(3)])
        (directory / "low.m3u8").write_text(header + old)
        session = _fast_session(directory, origin_pts=5000.0, origin_time=time.time() - 100)
        restarted = [(f"low_{i}.ts", 1.48 + 2 * (i - 100)) for i in range(100, 103)]

        async def launch(command: list[str]) -> FakeProcess:
            if command[0] == "ingest":
                (directory / "input_000001.ts").write_bytes(video_packet(1.48) * 10)
                (directory / "input.m3u8").write_text("#EXTINF:2,\ninput_000001.ts\n")
            else:
                text = (directory / "low.m3u8").read_text()
                (directory / "low.m3u8").write_text(
                    text + "#EXT-X-DISCONTINUITY\n" + _write_segments(directory, restarted),
                )
            return FakeProcess()

        with patch.dict(ffmpeg_session._transcode_sessions, {"fast": session}, clear=True):
            before = _program_dates(auth_client.get("/transcode/fast/low.m3u8").text)
            session["live_recovery"], permitted = session["live_recovery"].permit_restart(
                "pipeline",
                time.monotonic(),
            )
            assert permitted
            with (
                patch("netv.ffmpeg_session._launch_ffmpeg", side_effect=launch),
                patch(
                    "netv.ffmpeg_session._wait_for_fresh_fast_playlist",
                    new=AsyncMock(return_value=True),
                ),
                patch("netv.ffmpeg_session._spawn_background_task", side_effect=_close_background),
            ):
                asyncio.run(ffmpeg_session._restart_fast_pipeline("fast", session, "input"))
            restart_time = time.time()
            during = _program_dates(auth_client.get("/transcode/fast/low.m3u8").text)
            (directory / "low.m3u8").write_text(
                header + "".join(f"#EXTINF:2.000000,\n{name}\n" for name, _ in restarted),
            )
            after = _program_dates(auth_client.get("/transcode/fast/low.m3u8").text)

        assert during["low_0.ts"] == pytest.approx(before["low_0.ts"])
        assert before["low_2.ts"] < during["low_100.ts"] <= restart_time + 1
        assert after == {name: during[name] for name, _ in restarted}

    def test_listing_reads_each_segment_timestamp_once(
        self,
        auth_client: TestClient,
        tmp_path: Path,
    ):
        directory = tmp_path / "fast"
        directory.mkdir()
        segments = _write_segments(directory, [(f"low_{i}.ts", 10 + 2 * i) for i in range(4)])
        (directory / "low.m3u8").write_text("#EXTM3U\n" + segments)
        session = _fast_session(directory, origin_pts=10.0, origin_time=0.0)
        with (
            patch.dict(ffmpeg_session._transcode_sessions, {"fast": session}, clear=True),
            patch("netv.fast_start.segment_pts", wraps=fast_start.segment_pts) as reads,
        ):
            first = auth_client.get("/transcode/fast/low.m3u8")
            second = auth_client.get("/transcode/fast/low.m3u8")
        assert first.text == second.text
        assert reads.call_count == 4

    @pytest.mark.parametrize("force", [False, True])
    def test_stop_removes_fast_session_and_lets_encoders_exit(
        self,
        auth_client: TestClient,
        tmp_path: Path,
        force: bool,
    ):
        directory = tmp_path / "fast"
        directory.mkdir()
        processes = [LoopProcess(), LoopProcess(), LoopProcess()]
        session = _fast_session(
            directory,
            process=processes[0],
            ingest_process=processes[1],
            extra_processes=processes[1:],
        )
        with (
            patch.dict(ffmpeg_session._transcode_sessions, {"fast": session}, clear=True),
            patch("netv.ffmpeg_session.get_live_cache_timeout", return_value=30),
        ):
            response = auth_client.delete(f"/transcode/fast?force={str(force).lower()}")
            assert response.status_code == 200
            assert "fast" not in ffmpeg_session._transcode_sessions
        assert [process.signals for process in processes] == [["TERM"]] * 3
        assert [process.returncode for process in processes] == [-15] * 3
        assert not directory.exists()


class TestSubtitleRoutes:
    """Tests for subtitle routes."""

    def test_subtitle_invalid_filename(self, auth_client: TestClient):
        resp = auth_client.get("/subs/session/notavtt.txt")
        assert resp.status_code == 400

    def test_subtitle_session_not_found(self, auth_client: TestClient):
        with patch("netv.main.ffmpeg_session.get_session", return_value=None):
            resp = auth_client.get("/subs/invalid-session/sub0.vtt")
            assert resp.status_code == 404


class TestProbeCache:
    """Tests for probe cache endpoints."""

    def test_get_probe_cache(self, auth_client: TestClient):
        with patch("netv.main.ffmpeg_command.get_series_probe_cache_stats", return_value=[]):
            resp = auth_client.get("/settings/probe-cache")
            assert resp.status_code == 200
            assert "series" in _fields(resp)

    def test_clear_probe_cache(self, auth_client: TestClient):
        with patch("netv.main.ffmpeg_command.clear_all_probe_cache", return_value=5):
            resp = auth_client.post("/settings/probe-cache/clear")
            assert resp.status_code == 200
            assert _fields(resp)["cleared"] == 5

    def test_clear_series_probe_cache(self, auth_client: TestClient):
        with patch("netv.main.ffmpeg_command.invalidate_series_probe_cache"):
            resp = auth_client.post("/settings/probe-cache/clear/123")
            assert resp.status_code == 200


class TestRefreshStatus:
    """Tests for refresh status endpoints."""

    def test_guide_refresh_status(self, auth_client: TestClient):
        m3u_module.get_refresh_in_progress().clear()

        resp = auth_client.get("/guide/refresh-status")
        assert resp.status_code == 200
        data = _fields(resp)
        assert data["live"] is False
        assert data["epg"] is False

    def test_settings_refresh_status(self, auth_client: TestClient):
        m3u_module.get_refresh_in_progress().clear()

        resp = auth_client.get("/settings/refresh-status")
        assert resp.status_code == 200


class TestPlaybackHealth:
    def test_invalid_feedback_is_rejected(self, auth_client: TestClient):
        response = auth_client.post(
            "/transcode/missing/health",
            json={"buffer_seconds": -1, "waiting": True},
        )
        assert response.status_code == 422

    def test_feedback_is_passed_to_backend(self, auth_client: TestClient):
        with patch(
            "netv.ffmpeg_session.report_playback_health",
            return_value={"bandwidth_saver": True},
        ) as report:
            response = auth_client.post(
                "/transcode/live/health",
                json={"buffer_seconds": 0, "waiting": True},
            )
        assert response.json() == {"bandwidth_saver": True}
        assert report.call_args.args[:2] == ("live", "testuser")

    def test_force_stop_releases_current_stream(self, auth_client: TestClient):
        with (
            patch("netv.ffmpeg_session.get_session", return_value={"username": "testuser"}),
            patch("netv.ffmpeg_session.stop_session") as stop,
        ):
            response = auth_client.delete("/transcode/live?force=true")
        assert response.status_code == 200
        stop.assert_called_once_with("live", force=True)

    def test_force_stop_requires_owner(self, auth_client: TestClient):
        with (
            patch("netv.ffmpeg_session.get_session", return_value={"username": "other"}),
            patch("netv.ffmpeg_session.stop_session") as stop,
        ):
            assert auth_client.delete("/transcode/live?force=true").status_code == 404
        stop.assert_not_called()


class TestPlaylists:
    @pytest.fixture(autouse=True)
    def live_data(self, auth_client: TestClient):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
            {"category_id": "2", "category_name": "Sports"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": 1,
                "name": "CNN",
                "category_ids": ["1"],
                "epg_channel_id": "",
                "source_id": "s",
            },
            {
                "stream_id": 2,
                "name": "ESPN",
                "category_ids": ["2"],
                "epg_channel_id": "",
                "source_id": "s",
            },
            {
                "stream_id": 3,
                "name": "FS1",
                "category_ids": ["2"],
                "epg_channel_id": "",
                "source_id": "s",
            },
        ]

    def _create(
        self,
        client: TestClient,
        name: str = "Favorites",
        ids: tuple[int, ...] = (3, 1),
    ) -> PlaylistPayload:
        store = cache_module.get_cache()
        assert "live_streams" in store
        streams = {s["stream_id"]: s for s in store["live_streams"]}
        channels = [{"stream_id": i, "name": streams[i]["name"], "source_id": "s"} for i in ids]
        resp = client.post("/api/playlists", json={"name": name, "channels": channels})
        assert resp.status_code == 200
        return _body(resp, PlaylistPayload)

    def test_guide_rows_prepend_playlists_for_app_requests(self, auth_client: TestClient):
        playlist = self._create(auth_client)

        payload = _body(auth_client.get("/api/guide/rows?cats=1,2"), GuideRowsResponse)

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [3, 1, 2]
        assert payload["rows"][0]["channel"]["category_ids"] == ["2", playlist["category_id"]]
        assert payload["categories"][0] == {
            "category_id": playlist["category_id"],
            "category_name": "Favorites",
            "kind": "playlist",
        }
        assert [c["category_id"] for c in payload["categories"][1:]] == ["1", "2"]

    def test_guide_rows_exact_skips_playlists(self, auth_client: TestClient):
        self._create(auth_client)

        payload = _body(auth_client.get("/api/guide/rows?cats=2&exact=1"), GuideRowsResponse)

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [2, 3]
        assert [c["category_id"] for c in payload["categories"]] == ["2"]

    def test_guide_rows_single_playlist_keeps_playlist_order(self, auth_client: TestClient):
        playlist = self._create(auth_client, ids=(2, 3, 1))

        payload = _body(
            auth_client.get(f"/api/guide/rows?cats={playlist['category_id']}"),
            GuideRowsResponse,
        )

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [2, 3, 1]

    def test_guide_rows_order_most_viewed_streams_within_category(self, auth_client: TestClient):
        cache_module.record_live_view("testuser", "s", 3)
        cache_module.record_live_view("testuser", "s", 3)

        payload = _body(auth_client.get("/api/guide/rows?cats=1,2&exact=1"), GuideRowsResponse)

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [1, 3, 2]

    def test_guide_rows_order_most_viewed_streams_within_playlist(self, auth_client: TestClient):
        playlist = self._create(auth_client, ids=(2, 3, 1))
        cache_module.record_live_view("testuser", "s", 1)

        payload = _body(
            auth_client.get(f"/api/guide/rows?cats={playlist['category_id']}"),
            GuideRowsResponse,
        )

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [1, 2, 3]

    def test_guide_rows_preserve_order_when_view_sorting_is_disabled(self, auth_client: TestClient):
        cache_module.record_live_view("testuser", "s", 3)
        settings = cache_module.load_user_settings("testuser")
        settings["sort_live_by_views"] = False
        cache_module.save_user_settings("testuser", settings)

        payload = _body(auth_client.get("/api/guide/rows?cats=2&exact=1"), GuideRowsResponse)

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [2, 3]

    def test_guide_page_defaults_to_playlists_then_filter(self, auth_client: TestClient):
        playlist = self._create(auth_client)
        auth_client.post("/settings/guide-filter", json={"cats": ["1"]})
        with patch("netv.main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide")
        assert resp.status_code == 200
        assert "★ Favorites" in resp.text
        assert f'"{playlist["category_id"]},1"' in resp.text

    def test_guide_page_shows_section_headings(self, auth_client: TestClient):
        self._create(auth_client)
        auth_client.post("/settings/guide-filter", json={"cats": ["2"]})
        with patch("netv.main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide")

        match = re.search(r"sections: (\[.*?\]),\n", resp.text)
        assert match
        sections = json.loads(match.group(1))
        assert sections == [{"start": 0, "name": "★ Favorites"}, {"start": 2, "name": "Sports"}]
        headings = re.findall(
            r'class="guide-section[^"]*">\s*<span class="truncate">([^<]*)</span>',
            resp.text,
        )
        assert headings == ["★ Favorites", "Sports"]
        rows = re.findall(r'<div class="guide-row[^"]*" data-row="(\d+)"', resp.text)
        assert rows == ["0", "1", "2"]

    def _sidebar(
        self,
        client: TestClient,
        url: str = "/guide",
    ) -> tuple[list[tuple[str, str]], list[str], int]:
        with patch("netv.main.epg.has_programs", return_value=True):
            text = client.get(url).text
        aside = text[text.index("<aside") : text.index("</aside>")]
        links = cast(
            list[tuple[str, str]],
            re.findall(
                r'data-nav="cat".*?title="([^"]*)">.*?tabular-nums[^>]*>(\d+)<',
                aside,
                re.S,
            ),
        )
        current = cast(list[str], re.findall(r'aria-current="page" title="([^"]*)"', aside))
        rows = re.findall(r'<div class="guide-row[^"]*" data-row="\d+"', text)
        return links, current, len(rows)

    def test_guide_sidebar_lists_playlists_and_categories(self, auth_client: TestClient):

        auth_client.post("/settings/guide-filter", json={"cats": ["1", "2"]})
        auth_client.post("/api/user-prefs", json={"guide_selected_cats": ["2"]})
        playlist = self._create(auth_client)

        links, current, rows = self._sidebar(auth_client)
        assert links == [("All Channels", "3"), ("Favorites", "2"), ("News", "1"), ("Sports", "2")]
        assert current == ["Sports"]
        assert rows == 2

        links, current, rows = self._sidebar(auth_client, f"/guide?cats={playlist['category_id']}")
        assert current == ["Favorites"]
        assert rows == 2
        assert main.load_user_settings("testuser").get("guide_selected_cats") == [
            playlist["category_id"],
        ]

        _, current, rows = self._sidebar(auth_client, "/guide?cats=")
        assert current == ["All Channels"]
        assert rows == 3
        settings = main.load_user_settings("testuser")
        assert "guide_selected_cats" in settings
        assert settings["guide_selected_cats"] is None

    def test_old_multi_category_view_falls_back_to_all(self, auth_client: TestClient):
        auth_client.post("/settings/guide-filter", json={"cats": ["1", "2"]})
        auth_client.post("/api/user-prefs", json={"guide_selected_cats": ["1", "2"]})
        self._create(auth_client)

        _, current, rows = self._sidebar(auth_client)
        assert current == ["All Channels"]
        assert rows == 3

    def test_guide_rows_use_earliest_selected_category(self, auth_client: TestClient):
        store = cache_module.get_cache()
        assert "live_streams" in store
        store["live_streams"].append(
            {
                "stream_id": 4,
                "name": "Both",
                "category_ids": ["2", "1"],
                "epg_channel_id": "",
                "source_id": "s",
            },
        )

        payload = _body(auth_client.get("/api/guide/rows?cats=1,2"), GuideRowsResponse)

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [1, 4, 2, 3]

    def test_empty_playlists_are_hidden_from_guide(self, auth_client: TestClient):
        auth_client.post("/api/playlists", json={"name": "Empty"})
        payload = _body(auth_client.get("/api/guide/rows?cats=1"), GuideRowsResponse)
        assert [c["category_id"] for c in payload["categories"]] == ["1"]

    def test_update_rename_reorder_delete(self, auth_client: TestClient):
        first = self._create(auth_client, "A")
        second = self._create(auth_client, "B")

        resp = auth_client.put(f"/api/playlists/{first['id']}", json={"name": "A2", "channels": []})
        assert _fields(resp)["name"] == "A2"
        auth_client.post("/api/playlists/order", json={"ids": [second["id"], first["id"]]})
        assert [
            p["name"] for p in _body(auth_client.get("/api/playlists"), _PlaylistList)["playlists"]
        ] == [
            "B",
            "A2",
        ]
        assert auth_client.delete(f"/api/playlists/{second['id']}").status_code == 200
        assert auth_client.get(f"/api/playlists/{second['id']}").status_code == 404

    def test_playlist_reports_unavailable_channels(self, auth_client: TestClient):
        playlist = self._create(auth_client)
        auth_client.put(
            f"/api/playlists/{playlist['id']}",
            json={"channels": [{"stream_id": 42, "name": "Gone", "source_id": "s"}]},
        )
        channels = _body(auth_client.get(f"/api/playlists/{playlist['id']}"), PlaylistPayload)[
            "channels"
        ]
        assert channels == [
            {
                "stream_id": "42",
                "name": "Gone",
                "source_id": "s",
                "epg_channel_id": "",
                "available": False,
            },
        ]

    def test_assigned_guide_survives_round_trip_and_feeds_guide(self, auth_client: TestClient):
        playlist = self._create(auth_client, ids=(1,))
        channels = _body(auth_client.get(f"/api/playlists/{playlist['id']}"), PlaylistPayload)[
            "channels"
        ]
        assert channels[0].get("guide_assignable") is True
        channels[0]["epg_channel_id"] = "guide.one"
        auth_client.put(f"/api/playlists/{playlist['id']}", json={"channels": channels})

        channels = _body(auth_client.get(f"/api/playlists/{playlist['id']}"), PlaylistPayload)[
            "channels"
        ]
        assert channels[0].get("epg_channel_id") == "guide.one"
        with (
            patch("netv.main.epg.get_icons_batch", return_value={}),
            patch("netv.main.epg.get_programs_batch", return_value={}),
        ):
            rows = _body(
                auth_client.get(f"/api/guide/rows?cats={playlist['category_id']}"),
                GuideRowsResponse,
            )["rows"]
        assert rows[0]["channel"]["epg_id"] == "guide.one"

    def test_epg_channel_search(self, auth_client: TestClient):
        with patch("netv.main.epg.search_channels", return_value=[]) as search:
            resp = auth_client.get("/api/playlists/epg-channels?q=XX| ALPHA ONE HD")
        assert resp.status_code == 200
        search.assert_called_once_with(["ALPHA", "ONE"], 20)

    def test_match_and_search(self, auth_client: TestClient):
        matches = _body(
            auth_client.post("/api/playlists/match", json={"lines": ["ESPN", ""]}),
            _Matches,
        )["matches"]
        assert [(m["query"], m["candidates"][0]["stream_id"]) for m in matches] == [("ESPN", "2")]
        results = _body(auth_client.get("/api/playlists/search?category_id=2"), _SearchResults)[
            "results"
        ]
        assert [r["stream_id"] for r in results] == ["2", "3"]

    def test_non_admin_can_read_but_not_edit(self, auth_client: TestClient):

        playlist = self._create(auth_client)
        auth.create_user("viewer", "viewerpass1")
        auth_client.cookies.set("token", auth.create_token({"sub": "viewer"}))

        assert auth_client.get("/api/playlists").status_code == 200
        assert auth_client.get(f"/api/playlists/{playlist['id']}").status_code == 200
        assert auth_client.post("/api/playlists", json={"name": "X"}).status_code == 403
        assert (
            auth_client.put(f"/api/playlists/{playlist['id']}", json={"name": "X"}).status_code
            == 403
        )
        assert auth_client.delete(f"/api/playlists/{playlist['id']}").status_code == 403
        assert auth_client.get("/playlists").status_code == 403
        rows = _body(auth_client.get("/api/guide/rows?cats=1"), GuideRowsResponse)["rows"]
        assert [r["channel"]["stream_id"] for r in rows] == [3, 1]

    def test_admin_page_renders(self, auth_client: TestClient):
        resp = auth_client.get("/playlists")
        assert resp.status_code == 200
        assert 'id="playlist-list"' in resp.text

    def test_same_stream_id_from_two_sources_is_kept(self, auth_client: TestClient):
        store = cache_module.get_cache()
        assert "live_streams" in store
        store["live_streams"].append(
            {
                "stream_id": 1,
                "name": "BBC",
                "category_ids": ["1"],
                "epg_channel_id": "",
                "source_id": "t",
            },
        )
        resp = auth_client.post(
            "/api/playlists",
            json={"name": "Mix", "channels": [{"stream_id": 1, "name": "BBC", "source_id": "t"}]},
        )
        playlist = _body(resp, PlaylistPayload)

        rows = _body(auth_client.get("/api/guide/rows?cats=1"), GuideRowsResponse)["rows"]

        assert [(r["channel"]["stream_id"], r["channel"]["name"]) for r in rows] == [
            (1, "BBC"),
            (1, "CNN"),
        ]
        assert playlist["category_id"] in rows[0]["channel"]["category_ids"]
        assert playlist["category_id"] not in rows[1]["channel"]["category_ids"]

    def test_restricted_viewer_does_not_see_restricted_playlist_channels(
        self,
        auth_client: TestClient,
    ):

        playlist = self._create(auth_client, ids=(3, 1))
        only_sports = self._create(auth_client, "Sports", ids=(3,))
        auth.create_user("viewer", "viewerpass1")
        auth.set_user_limits("viewer", unavailable_groups=["cat:2"])
        auth_client.cookies.set("token", auth.create_token({"sub": "viewer"}))

        listed = _body(auth_client.get("/api/playlists"), _PlaylistList)["playlists"]
        assert [(p["id"], p["channel_count"]) for p in listed] == [(playlist["id"], 1)]
        detail = _body(auth_client.get(f"/api/playlists/{playlist['id']}"), PlaylistPayload)
        assert [c.get("name") for c in detail["channels"]] == ["CNN"]
        assert (
            _body(auth_client.get(f"/api/playlists/{only_sports['id']}"), PlaylistPayload)[
                "channels"
            ]
            == []
        )
        rows = _body(auth_client.get("/api/guide/rows?cats=1"), GuideRowsResponse)["rows"]
        assert [r["channel"]["stream_id"] for r in rows] == [1]

    def test_invalid_create_does_not_persist(self, auth_client: TestClient):
        too_many = [{"stream_id": i, "name": str(i)} for i in range(2001)]
        assert (
            auth_client.post(
                "/api/playlists",
                json={"name": "Big", "channels": too_many},
            ).status_code
            == 400
        )
        assert _body(auth_client.get("/api/playlists"), _PlaylistList)["playlists"] == []

    def test_saved_view_drops_deleted_playlist(self, auth_client: TestClient):

        playlist = self._create(auth_client)
        auth_client.post("/settings/guide-filter", json={"cats": ["1"]})
        auth_client.post("/api/user-prefs", json={"guide_selected_cats": [playlist["category_id"]]})
        auth_client.delete(f"/api/playlists/{playlist['id']}")

        with patch("netv.main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide")

        assert resp.status_code == 200
        assert "CNN" in resp.text
        assert main.load_user_settings("testuser").get("guide_selected_cats") is None


class TestSourceAssignment:
    """Sources can be limited to selected users; admins always see everything."""

    @classmethod
    def _setup(cls, auth_client: TestClient, family_users: list[str] | None) -> None:
        del auth_client
        settings = cache_module.load_server_settings()
        settings["sources"] = [
            {
                "id": "fam",
                "name": "Family",
                "type": "m3u",
                "url": "http://a",
                "users": family_users,
            },
            {"id": "pub", "name": "Shared", "type": "m3u", "url": "http://b"},
        ]
        cache_module.save_server_settings(settings)
        cache_module.get_cache().update(
            {
                "live_categories": [
                    {"category_id": "fam_1", "category_name": "Fam", "source_id": "fam"},
                    {"category_id": "pub_1", "category_name": "Pub", "source_id": "pub"},
                ],
                "live_streams": [
                    {
                        "stream_id": "fam_1",
                        "name": "A",
                        "category_ids": ["fam_1"],
                        "source_id": "fam",
                    },
                    {
                        "stream_id": "pub_1",
                        "name": "B",
                        "category_ids": ["pub_1"],
                        "source_id": "pub",
                    },
                ],
            },
        )
        auth.create_user("friend", "friendpass1")
        auth.create_user("kid", "kidpass123")

    def _rows(self, auth_client: TestClient, user: str) -> list[MutablePlainTree]:

        auth_client.cookies.set("token", auth.create_token({"sub": user}))
        rows = _body(
            auth_client.get("/api/guide/rows?cats=fam_1,pub_1&exact=1"),
            GuideRowsResponse,
        )["rows"]
        return [r["channel"]["name"] for r in rows]

    def test_assigned_source_is_hidden_from_other_users(self, auth_client: TestClient):
        self._setup(auth_client, ["kid"])
        assert self._rows(auth_client, "testuser") == ["A", "B"]  # Admin.
        assert self._rows(auth_client, "kid") == ["A", "B"]
        assert self._rows(auth_client, "friend") == ["B"]

    def test_unassigned_source_is_shared(self, auth_client: TestClient):
        self._setup(auth_client, None)
        assert self._rows(auth_client, "friend") == ["A", "B"]

    def test_hidden_source_blocks_transcode(self, auth_client: TestClient):

        self._setup(auth_client, [])
        auth_client.cookies.set("token", auth.create_token({"sub": "kid"}))
        resp = auth_client.get("/transcode/start?url=http://a/x.ts&source_id=fam")
        assert resp.status_code == 403

    def test_edit_source_saves_selected_users(self, auth_client: TestClient):
        self._setup(auth_client, None)
        page = auth_client.get("/settings")
        assert page.status_code == 200
        assert 'name="users" value="kid"' in page.text
        resp = auth_client.post(
            "/settings/edit/fam",
            data={
                "name": "Family",
                "source_type": "m3u",
                "url": "http://a",
                "access": "selected",
                "users": ["kid", "ghost"],
            },
        )
        assert resp.status_code == 200
        fam = next(s for s in cache_module.get_sources() if s.id == "fam")
        assert fam.users == ["kid"]

        auth_client.post(
            "/settings/edit/fam",
            data={"name": "Family", "source_type": "m3u", "url": "http://a", "access": "all"},
        )
        fam = next(s for s in cache_module.get_sources() if s.id == "fam")
        assert fam.users is None

    def test_deleting_user_keeps_source_restricted(self, auth_client: TestClient):
        self._setup(auth_client, ["kid"])
        auth.delete_user("kid")
        fam = next(s for s in cache_module.get_sources() if s.id == "fam")
        assert fam.users == []

    def test_settings_api_hides_private_settings(self, auth_client: TestClient):
        self._setup(auth_client, ["kid"])
        data = _fields(auth_client.get("/api/settings"))
        assert "sources" not in data
        assert "users" not in data


# ===========================================================================
# Handoff
# ===========================================================================


@pytest.fixture
def clean_handoff() -> Iterator[None]:

    handoff._records.clear()
    handoff._watches.clear()
    yield
    handoff._records.clear()
    handoff._watches.clear()


def test_handoff_api_offers_the_users_handoff(auth_client: TestClient, clean_handoff: None):
    handoff._records["testuser"] = _handoff_record()
    body = _body(auth_client.get("/api/handoff"), _HandoffBody)["handoff"]
    assert body is not None
    assert body["id"] == "h1"
    assert body["stream_id"] == "src1_42"
    assert "recording_id" not in body
    handoff._records["testuser"]["resumed"] = True
    assert auth_client.get("/api/handoff").json() == {"handoff": None}


def test_handoff_recording_resumes_from_where_it_was_left(
    auth_client: TestClient,
    clean_handoff: None,
):

    handoff._records["testuser"] = _handoff_record()
    info = PlayerInfo(url="http://upstream.test/live/u/p/42.m3u8", channel_name="Channel")
    with patch("netv.main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/src1_42?handoff=h1")
    assert response.status_code == 200
    assert '"/handoff/rec1/stream.m3u8"' in response.text
    assert 'transcodeMode: "never"' in response.text
    assert "catchupSeek: 10.0" in response.text
    assert "handoffRecording: true" in response.text
    assert handoff._records["testuser"]["resumed"] is True


def test_handoff_catchup_resumes_through_the_archive(auth_client: TestClient, clean_handoff: None):

    handoff._records["testuser"] = _handoff_record(mode="catchup", recording_id="")
    info = PlayerInfo(url="http://upstream.test/timeshift/x.ts", catchup_start=1767264000.0)
    with patch("netv.main._get_catchup_player_info", return_value=info) as get_info:
        response = auth_client.get("/play/live/src1_42?handoff=h1")
    assert response.status_code == 200
    get_info.assert_called_once_with("src1_42", 1767264000.0)


def test_playing_something_else_cancels_the_handoff(auth_client: TestClient, clean_handoff: None):

    handoff._records["testuser"] = _handoff_record(mode="catchup", recording_id="")
    info = PlayerInfo(url="https://example.test/live.m3u8", channel_name="Other")
    with patch("netv.main._get_live_player_info", return_value=info):
        auth_client.get("/play/live/other")
    assert "testuser" not in handoff._records


def test_stale_handoff_cannot_be_resumed(auth_client: TestClient, clean_handoff: None):
    handoff._records["testuser"] = _handoff_record(mode="catchup", recording_id="")
    response = auth_client.get("/play/live/src1_42?handoff=gone")
    assert response.status_code == 404
    assert handoff._records["testuser"]["id"] == "h1", "An old link keeps the newer handoff"


def test_handoff_stop_hands_off_the_watched_channel(auth_client: TestClient, clean_handoff: None):

    now = datetime.now(timezone.utc).timestamp()
    handoff._watches["testuser"] = handoff._Watch(
        stream_id="src1_42",
        position=now - 10,
        first_seen=now - 600,
        last_seen=now - 15,
    )
    channel = handoff.Channel(
        stream_id="src1_42",
        name="Channel",
        url="http://x/42.m3u8",
        catchup_days=2,
    )
    with (
        patch("netv.main._handoff_stream_allowed", return_value=True),
        patch("netv.main._handoff_channel", return_value=channel),
    ):
        response = auth_client.post(
            "/api/handoff/stop",
            json={"stream_id": "src1_42", "position": now - 5},
        )
    assert response.status_code == 200
    assert handoff._records["testuser"]["mode"] == "catchup"
    assert handoff._records["testuser"]["position"] == pytest.approx(now - 5)


def test_handoff_recording_first_releases_the_live_transcode(
    auth_client: TestClient,
    clean_handoff: None,
):
    now = datetime.now(timezone.utc).timestamp()
    handoff._watches["testuser"] = handoff._Watch(
        stream_id="src1_42",
        position=now - 10,
        first_seen=now - 600,
        last_seen=now - 15,
    )
    channel = handoff.Channel(stream_id="src1_42", name="Channel", url="http://x/42.m3u8")
    with (
        patch("netv.main._handoff_stream_allowed", return_value=True),
        patch("netv.main._handoff_channel", return_value=channel),
        patch("netv.main.ffmpeg_session.stop_user_stream", new_callable=AsyncMock) as release,
        patch("netv.handoff._start_recorder", side_effect=OSError("no ffmpeg")),
    ):
        response = auth_client.post(
            "/api/handoff/stop",
            json={"stream_id": "src1_42", "position": now - 5},
        )
    assert response.status_code == 200
    release.assert_awaited_once_with("testuser", "http://x/42.m3u8")


class TestManagedDirectGuide:
    def test_direct_export_uses_saved_filter_and_never_proxies(self, auth_client: TestClient):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "Allowed"},
            {"category_id": "2", "category_name": "Other"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": "src_42",
                "name": "Xtream",
                "category_ids": ["1"],
                "source_type": "xtream",
                "source_url": "https://provider.test/",
                "source_username": "user@name",
                "source_password": "p#ss",
            },
            {
                "stream_id": "m3u_1",
                "name": "M3U",
                "category_ids": ["1"],
                "direct_url": "https://media.test/channel.m3u8",
            },
            {
                "stream_id": "hidden",
                "name": "Other",
                "category_ids": ["2"],
                "direct_url": "https://media.test/hidden.m3u8",
            },
        ]
        with (
            patch("netv.main.load_user_settings", return_value={"guide_filter": ["1"]}),
            patch("netv.main._playlist_categories", return_value=[]),
        ):
            normal = _body(auth_client.get("/api/guide/rows"), GuideRowsResponse)
            response = auth_client.get("/api/guide/rows?direct=true")
        assert all("direct_url" not in row["channel"] for row in normal["rows"])
        payload = _body(response, GuideRowsResponse)
        assert payload["direct_playback"] is True
        assert all(row["channel"]["catchup_days"] == 0 for row in payload["rows"])
        assert response.headers["cache-control"] == "no-store"
        urls = {
            row["channel"]["stream_id"]: row["channel"].get("direct_url") for row in payload["rows"]
        }
        assert urls == {
            "src_42": "https://provider.test/live/user%40name/p%23ss/42.m3u8",
            "m3u_1": "https://media.test/channel.m3u8",
        }

    def test_direct_export_excludes_restricted_channels(self, auth_client: TestClient):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": "secret",
                "name": "Secret",
                "category_ids": ["1"],
                "direct_url": "https://provider.test/private.m3u8",
            },
        ]
        with patch("netv.main._unavailable_groups", return_value={"cat:1"}):
            payload = _body(
                auth_client.get("/api/guide/rows?cats=1&direct=true"),
                GuideRowsResponse,
            )
        assert payload["rows"] == []
        assert payload["direct_playback"] is True


def _archive_stream(**overrides: str | int) -> ProviderRecord:
    stream: ProviderRecord = {
        "stream_id": "src1_42",
        "name": "Channel",
        "source_id": "src1",
        "source_type": "xtream",
        "source_url": "http://upstream.test",
        "source_username": "user",
        "source_password": "pass",
        "epg_channel_id": "ch1",
        "tv_archive": 1,
        "tv_archive_duration": 2,
    }
    stream.update(overrides)
    return stream


def _handoff_record(**overrides: str | float | bool) -> dict[str, str | float | bool]:
    record: dict[str, str | float | bool] = {
        "id": "h1",
        "stream_id": "src1_42",
        "channel_name": "Channel",
        "program_title": "Match",
        "program_start": 1767261600.0,
        "program_end": 1767268800.0,
        "position": 1767264000.0,
        "mode": "recording",
        "recording_id": "rec1",
        "recording_start": 1767263990.0,
        "resumed": False,
        "expires": 9e9,
    }
    record.update(overrides)
    return record


def _mark_start(content: str, seconds: float) -> str:
    del seconds
    return content + "#start\n"


class _JsonResponse(Protocol):
    @property
    def content(self) -> bytes: ...


def _body(response: _JsonResponse, target: type[T]) -> T:
    return json_loads(response.content, target)


def _fields(response: _JsonResponse) -> dict[str, MutablePlainTree]:
    return _body(response, dict[str, MutablePlainTree])


class _Detail(TypedDict):
    detail: str


class _PlaylistSummary(TypedDict):
    id: str
    category_id: str
    name: str
    channel_count: int


class _PlaylistList(TypedDict):
    playlists: list[_PlaylistSummary]


class _Matches(TypedDict):
    matches: list[NameMatch]


class _SearchResults(TypedDict):
    results: list[StreamSummary]


class _HandoffBody(TypedDict):
    handoff: dict[str, MutablePlainTree] | None


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
