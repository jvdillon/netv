"""Tests for main.py - FastAPI routes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol, TypedDict, TypeVar, cast, override
from urllib import parse
from zoneinfo import ZoneInfo

import asyncio
import json
import re
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.types import Message, Scope
from treekle import MutablePlainTree, json_loads

import pytest
import pytest_asyncio

from netv import (
    auth,
    cache as cache_module,
    catchup,
    chromecast,
    db,
    epg,
    ffmpeg_command,
    ffmpeg_session,
    handoff,
    imdb,
    m3u as m3u_module,
    main,
    testing,
    util,
)
from netv.auth import TokenClaims
from netv.epg import Program
from netv.ffmpeg_command import MediaInfo
from netv.main import CatchupListing, GuideRowsResponse, PlayerInfo, PlaylistPayload
from netv.playback_policy import PlaybackHealth
from netv.playlists import NameMatch, StreamSummary
from netv.xtream import ProviderRecord


T = TypeVar("T")

SERVER_INFO_URL = "http://upstream.test/player_api.php?username=user&password=pass"


@pytest.fixture
def mock_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test fresh session, cast and login state."""
    monkeypatch.setattr(ffmpeg_session, "_transcode_sessions", {})
    monkeypatch.setattr(ffmpeg_session, "_url_to_session", {})
    monkeypatch.setattr(chromecast, "manager", chromecast.CastManager())
    monkeypatch.setattr(m3u_module, "_refresh_in_progress", set())
    monkeypatch.setattr(main, "_login_attempts", {})
    monkeypatch.setattr(catchup, "_tz_cache", {})


@pytest.fixture
def transcode_dir(tmp_path: Path) -> Path:
    """Keep transcodes and handoff recordings out of the system temp dir."""
    path = tmp_path / "transcode"
    cache_module.save_server_settings(
        {**cache_module.load_server_settings(), "transcode_dir": str(path)},
    )
    return path


@pytest.fixture
def client(mock_deps: None) -> TestClient:
    """Create a test client with no users signed in."""
    return TestClient(main.app)


@pytest.fixture
def auth_client(mock_deps: None) -> TestClient:
    """Create test client with a logged-in user."""
    client = TestClient(main.app)
    auth.create_user("testuser", "testpass123")
    client.cookies.set("token", auth.create_token({"sub": "testuser"}))
    return client


@pytest.fixture
def threads(monkeypatch: pytest.MonkeyPatch) -> testing.ThreadRecorder:
    """Hold main's background loaders instead of running them."""
    recorder = testing.ThreadRecorder(run_inline=False)
    testing.install_threads(monkeypatch, recorder, main)
    return recorder


@pytest.mark.usefixtures("guide_db")
def test_web_player_includes_resolution_badge(auth_client: TestClient):
    _set_live({"stream_id": "test", "name": "Test channel", "direct_url": "https://e.test/l.m3u8"})
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


@pytest.mark.usefixtures("guide_db")
def test_live_player_records_view(auth_client: TestClient):
    _set_live(
        {"stream_id": "7", "name": "Test", "direct_url": "https://e.test/l.m3u8", "source_id": "a"},
    )
    response = auth_client.get("/play/live/7")

    assert response.status_code == 200
    assert cache_module.load_live_view_counts("testuser") == {"a:7": 1}


@pytest.mark.parametrize("path", ["devices", "status"])
def test_cast_requires_authentication(client: TestClient, path: str):
    assert client.get(f"/api/cast/{path}").status_code == 401


@pytest.mark.parametrize("path", ["start", "control"])
def test_cast_commands_require_authentication(client: TestClient, path: str):
    assert client.post(f"/api/cast/{path}", json={}).status_code == 401


def test_cast_api_uses_http_and_remembers_lan_address(
    auth_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receiver = _cast_lan(monkeypatch, tmp_path)
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
    assert _fields(response)["host"] == "192.168.1.50"
    load = receiver.media_controller.loaded[0]
    assert load.url == "http://192.168.1.10:9000/transcode/abc/stream.m3u8"
    assert load.options["title"] == "News"
    assert cache_module.load_user_settings("testuser")["cast_host"] == "http://192.168.1.10:9000"


def test_cast_api_uses_request_origin_and_supports_legacy_host_preference(
    auth_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receiver = _cast_lan(monkeypatch, tmp_path)
    response = auth_client.post(
        "/api/cast/start",
        json={"host": "192.168.1.50", "session_id": "abc"},
    )
    assert response.status_code == 200
    assert receiver.media_controller.loaded[0].url == "http://testserver/transcode/abc/stream.m3u8"
    assert cache_module.load_user_settings("testuser").get("cast_host", "") == ""

    monkeypatch.setattr(chromecast, "manager", chromecast.CastManager())
    settings = cache_module.load_user_settings("testuser")
    settings["cast_host"] = "192.168.1.10:8000"
    cache_module.save_user_settings("testuser", settings)
    response = auth_client.post(
        "/api/cast/start",
        json={"host": "192.168.1.50", "session_id": "abc"},
    )
    assert response.status_code == 200
    assert receiver.media_controller.loaded[1].url == (
        "http://192.168.1.10:8000/transcode/abc/stream.m3u8"
    )


def test_cast_rejects_localhost_media_address(
    auth_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _cast_lan(monkeypatch, tmp_path)
    network = testing.CastNetwork({"192.168.1.50": testing.FakeChromecast()})
    testing.install_chromecast(monkeypatch, network, chromecast)
    response = auth_client.post(
        "/api/cast/start",
        json={
            "host": "192.168.1.50",
            "session_id": "abc",
            "server_url": "http://localhost:8000",
        },
    )
    assert response.status_code == 400
    assert "localhost" in _body(response, _Detail)["detail"]
    assert network.connects == []
    assert cache_module.load_user_settings("testuser").get("cast_host", "") == ""


def test_cast_command_validation_and_user_scope(
    auth_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receiver = _cast_lan(monkeypatch, tmp_path)
    assert (
        auth_client.post("/api/cast/start", json={"host": "192.168.1.50", "session_id": "abc"})
    ).status_code == 200
    process = ffmpeg_session._transcode_sessions["abc"]["process"]
    assert (
        auth_client.post("/api/cast/control", json={"action": "volume", "volume": 2}).status_code
        == 422
    )
    assert auth_client.post("/api/cast/control", json={"action": "reboot"}).status_code == 422
    assert receiver.media_controller.commands == ["load"]

    assert auth_client.post("/api/cast/control", json={"action": "pause"}).status_code == 200
    assert receiver.media_controller.commands == ["load", "pause"]
    assert "abc" in ffmpeg_session._transcode_sessions

    auth.create_user("other", "otherpass1")
    other = TestClient(main.app)
    other.cookies.set("token", auth.create_token({"sub": "other"}))
    assert other.post("/api/cast/control", json={"action": "stop"}).status_code == 404

    assert auth_client.post("/api/cast/control", json={"action": "stop"}).status_code == 200
    assert receiver.media_controller.commands == ["load", "pause", "stop"]
    assert "abc" not in ffmpeg_session._transcode_sessions
    assert isinstance(process, testing.FakeProcess)
    assert process.signals == ["TERM"]


def test_cast_discovery_reports_network_errors(
    auth_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    network = testing.CastNetwork(discovery_error=OSError("no network"))
    testing.install_chromecast(monkeypatch, network, chromecast)
    response = auth_client.get("/api/cast/devices")
    assert response.status_code == 503
    assert "manually" in _body(response, _Detail)["detail"]


def test_browser_cannot_stop_or_seek_cast_owned_session(
    auth_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session = _add_session("abc", tmp_path)
    _cast_owns(monkeypatch, session_id="abc", url=session["url"])
    response = auth_client.post("/transcode/abc/stop?force=true")
    assert response.json() == {"status": "casting"}
    assert auth_client.delete("/transcode/abc?force=true").json() == {"status": "casting"}
    assert auth_client.get("/transcode/seek/abc?time=10").status_code == 409
    assert ffmpeg_session._transcode_sessions["abc"] is session
    assert _signals(session) == []


def test_cast_source_cannot_be_restarted_while_receiver_owns_it(
    auth_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session = _add_session("abc", tmp_path, url="http://provider/live")
    _cast_owns(monkeypatch, session_id="abc", url="http://provider/live")
    assert auth_client.delete("/transcode-clear?url=http://provider/live").status_code == 409
    assert (
        auth_client.get("/transcode/start", params=_start("http://provider/live")).status_code
        == 409
    )
    assert ffmpeg_session._url_to_session == {"http://provider/live": "abc"}
    assert _signals(session) == []


def test_receiver_hls_request_keeps_session_alive_without_cookie(
    client: TestClient,
    tmp_path: Path,
):
    (tmp_path / "stream.m3u8").write_text("#EXTM3U\n#EXTINF:6,\nseg001.ts\n")
    session = _add_session("abc", tmp_path, last_access=0.0)
    before = time.time()
    response = client.get("/transcode/abc/stream.m3u8")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert session.get("last_access", 0) >= before


@pytest.mark.usefixtures("guide_db")
def test_live_dvr_routes_auto_mode_through_server(auth_client: TestClient):
    _set_live(_archive_stream(stream_id="test", tv_archive=0))
    response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert 'transcodeMode: "always"' in response.text
    assert "liveDvrMins: 120" in response.text


@pytest.mark.usefixtures("guide_db")
def test_live_dvr_disabled_keeps_direct_auto_mode(auth_client: TestClient):
    settings = cache_module.load_server_settings()
    settings["live_dvr_mins"] = 0
    cache_module.save_server_settings(settings)
    _set_live(_archive_stream(stream_id="test", tv_archive=0))
    response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert 'transcodeMode: "auto"' in response.text
    assert "liveDvrMins: 0" in response.text


@pytest.mark.usefixtures("guide_db")
def test_live_player_passes_program_window_from_epg(auth_client: TestClient):
    start = _now() - timedelta(minutes=30)
    stop = start + timedelta(hours=1)
    _set_live(
        {
            "stream_id": "test",
            "name": "Test channel",
            "direct_url": "https://e.test/l.ts",
            "epg_channel_id": "ch1",
        },
    )
    _air("ch1", "News", start, stop)
    response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert f"programStart: {start.timestamp()}" in response.text
    assert f"programEnd: {stop.timestamp()}" in response.text
    assert 'streamId: "test"' in response.text
    assert 'id="program-remaining"' not in response.text


@pytest.mark.usefixtures("guide_db")
def test_live_player_info_reads_program_window_from_epg(mock_deps: None):
    start = _now() - timedelta(minutes=30)
    stop = _now() + timedelta(minutes=30)
    _air("ch1", "News", start, stop, desc="Headlines")
    settings = cache_module.load_server_settings()
    settings["sources"] = [{"id": "src1", "name": "P", "type": "m3u", "url": "http://p"}]
    cache_module.save_server_settings(settings)
    _set_live(
        {
            "stream_id": "1",
            "name": "Channel",
            "direct_url": "https://example.test/live.ts",
            "epg_channel_id": "ch1",
            "source_id": "src1",
            "category_ids": "not-a-list",
        },
    )
    info = main._get_live_player_info("1")
    assert (info.url, info.is_m3u) == ("https://example.test/live.ts", True)
    assert (info.program_title, info.program_desc) == ("News", "Headlines")
    assert info.program_start == start.timestamp()
    assert info.program_end == stop.timestamp()
    # A source saved without the setting keeps deinterlacing on.
    assert info.deinterlace_fallback is True
    assert info.category_ids is None


@pytest.mark.usefixtures("guide_db")
def test_live_program_api_returns_current_program(auth_client: TestClient):
    start = _now() - timedelta(minutes=10)
    stop = start + timedelta(minutes=45)
    _air("ch1", "News", start, stop, desc="Tonight's headlines")
    _set_live({"stream_id": "1", "name": "Channel", "epg_channel_id": "ch1"})
    response = auth_client.get("/api/live/program/1")
    assert response.status_code == 200
    assert response.json() == {
        "title": "News",
        "desc": "Tonight's headlines",
        "start": start.timestamp(),
        "end": stop.timestamp(),
    }


@pytest.mark.usefixtures("guide_db")
def test_live_program_api_without_guide_data_returns_empty_program(auth_client: TestClient):
    _set_live({"stream_id": "1", "name": "Channel"})
    response = auth_client.get("/api/live/program/1")
    assert response.status_code == 200
    assert response.json() == {"title": "", "desc": "", "start": 0.0, "end": 0.0}


@pytest.mark.usefixtures("guide_db")
def test_live_program_api_unknown_stream_is_not_found(auth_client: TestClient):
    _set_live()
    response = auth_client.get("/api/live/program/missing")
    assert response.status_code == 404


class _CatchupWorld:
    """Live cache, guide and upstream server info for ``_get_catchup_player_info``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.stream = _archive_stream()
        _set_live(self.stream)
        self.web = testing.FakeWeb()
        testing.install_web(monkeypatch, self.web, util)

    def serve_timezone(self, name: str) -> None:
        """Configure the archive's Xtream source and the time zone its server reports."""
        settings = cache_module.load_server_settings()
        settings["sources"] = [
            {
                "id": "src1",
                "name": "Upstream",
                "type": "xtream",
                "url": "http://upstream.test",
                "username": "user",
                "password": "pass",
            },
        ]
        cache_module.save_server_settings(settings)
        body = json.dumps({"server_info": {"timezone": name}}).encode()
        self.web.routes[SERVER_INFO_URL] = testing.FakeResponse(body)


@pytest.fixture
def catchup_world(mock_deps: None, monkeypatch: pytest.MonkeyPatch) -> _CatchupWorld:
    return _CatchupWorld(monkeypatch)


def _hours_ago(hours: int) -> datetime:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).replace(second=0, microsecond=0)


@pytest.mark.usefixtures("guide_db")
def test_catchup_player_info_builds_timeshift_url(catchup_world: _CatchupWorld):
    start = _hours_ago(3)
    program = _air("ch1", "Earlier news", start, start + timedelta(minutes=30), desc="Headlines")
    # A direct playlist URL makes live info M3U; the archive is always Xtream timeshift.
    catchup_world.stream["direct_url"] = "http://m3u.test/42.ts"
    catchup_world.serve_timezone("Europe/Amsterdam")

    info = main._get_catchup_player_info("src1_42", start.timestamp())

    local = start.astimezone(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d:%H-%M")
    assert info.url == f"http://upstream.test/timeshift/user/pass/30/{local}/42.ts"
    assert info.is_m3u is False
    assert info.catchup_start == start.timestamp()
    assert info.catchup_days == 2
    assert (info.program_title, info.program_desc) == ("Earlier news", "Headlines")
    assert info.program_end == program.stop.timestamp()
    assert list(catchup._tz_cache) == ["src1"]
    assert catchup_world.web.urls == [SERVER_INFO_URL]


@pytest.mark.usefixtures("guide_db")
def test_catchup_player_info_starts_mid_program(catchup_world: _CatchupWorld):
    program_start = _hours_ago(3)
    program = _air("ch1", "Earlier news", program_start, program_start + timedelta(minutes=30))
    target = program_start + timedelta(minutes=10, seconds=25, microseconds=500_000)

    info = main._get_catchup_player_info("src1_42", target.timestamp())

    begin = program_start + timedelta(minutes=10)
    assert f"/20/{begin:%Y-%m-%d:%H-%M}/42.ts" in info.url
    assert info.catchup_start == begin.timestamp()
    assert info.catchup_seek == 25.5
    assert info.program_start == program_start.timestamp()
    assert info.program_end == program.stop.timestamp()
    assert catchup_world.web.urls == []


@pytest.mark.usefixtures("guide_db")
def test_catchup_player_info_without_guide_plays_one_hour(catchup_world: _CatchupWorld):
    del catchup_world
    start = _hours_ago(3) + timedelta(seconds=40)
    # What airs now must not label an archive the guide does not cover.
    now = _now()
    _air("ch1", "Live now", now - timedelta(minutes=5), now + timedelta(minutes=25), desc="Current")

    info = main._get_catchup_player_info("src1_42", start.timestamp())

    begin = start.replace(second=0)
    assert f"/timeshift/user/pass/61/{begin:%Y-%m-%d:%H-%M}/42.ts" in info.url
    assert (info.program_title, info.program_desc) == ("", "")
    assert info.program_start == begin.timestamp()
    assert info.program_end == (start + timedelta(hours=1)).timestamp()


@pytest.mark.usefixtures("guide_db")
def test_catchup_player_info_needs_a_live_url(catchup_world: _CatchupWorld):
    catchup_world.stream["source_type"] = "m3u"
    assert main._get_catchup_player_info("src1_42", _hours_ago(3).timestamp()) == PlayerInfo()


@pytest.mark.usefixtures("guide_db")
def test_catchup_player_info_rejects_programs_outside_archive(catchup_world: _CatchupWorld):
    del catchup_world
    too_old = datetime.now(timezone.utc) - timedelta(days=3)
    with pytest.raises(HTTPException) as excinfo:
        main._get_catchup_player_info("src1_42", too_old.timestamp())
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "This program is not available to watch from the start"


@pytest.mark.usefixtures("guide_db")
def test_catchup_player_page_plays_archive_as_vod(auth_client: TestClient):
    _set_live(_archive_stream())
    begin = _hours_ago(3)
    start = int(begin.timestamp()) + 25
    response = auth_client.get(f"/play/live/src1_42?start={start}")
    assert response.status_code == 200
    assert "catchup: true" in response.text
    assert f"catchupStart: {begin.timestamp()}" in response.text
    assert "catchupSeek: 25.0" in response.text
    assert "/timeshift/user/pass/" in response.text
    assert 'id="go-live-btn"' in response.text
    assert 'id="jump-btn"' in response.text
    assert 'id="start-over-btn"' not in response.text


@pytest.mark.usefixtures("guide_db")
def test_live_player_offers_start_over_with_archive(auth_client: TestClient):
    _set_live(
        _archive_stream(stream_id="test", direct_url="https://e.test/l.ts", tv_archive_duration=1),
    )
    response = auth_client.get("/play/live/test")
    assert "catchup: false" in response.text
    assert "catchupDays: 1" in response.text
    assert 'id="start-over-btn"' in response.text


@pytest.mark.usefixtures("guide_db")
def test_catchup_api_lists_archived_programs_newest_first(auth_client: TestClient):
    now = _now()
    _air("ch1", "Too old", now - timedelta(days=2, hours=1), now - timedelta(days=2, minutes=-1))
    morning = _air("ch1", "Morning", now - timedelta(hours=5), now - timedelta(hours=4))
    _air("ch1", "On now", now - timedelta(minutes=10), now + timedelta(minutes=20))
    _set_live(_archive_stream())
    response = auth_client.get("/api/live/catchup/src1_42")
    assert response.status_code == 200
    payload = _body(response, CatchupListing)
    assert payload["days"] == 2
    assert [p["title"] for p in payload["programs"]] == ["On now", "Morning"]
    assert payload["programs"][1]["start_timestamp"] == morning.start.timestamp()


@pytest.mark.usefixtures("guide_db")
def test_catchup_api_without_archive_is_empty(auth_client: TestClient):
    _set_live(_archive_stream(tv_archive=0))
    response = auth_client.get("/api/live/catchup/src1_42")
    assert response.json() == {"days": 0, "programs": []}


@pytest.mark.usefixtures("guide_db")
def test_catchup_api_uses_playlist_guide_assignment(auth_client: TestClient):
    now = _now()
    _air("assigned", "Morning", now - timedelta(hours=5), now - timedelta(hours=4))
    _air("ch1", "Unassigned guide", now - timedelta(hours=3), now - timedelta(hours=2))
    _set_live(_archive_stream(epg_channel_id=""))
    auth_client.post(
        "/api/playlists",
        json={
            "name": "pl1",
            "channels": [
                {
                    "stream_id": "src1_42",
                    "source_id": "src1",
                    "name": "Channel",
                    "epg_channel_id": "assigned",
                },
            ],
        },
    )
    response = auth_client.get("/api/live/catchup/src1_42")
    assert [p["title"] for p in _body(response, CatchupListing)["programs"]] == ["Morning"]


@pytest.mark.usefixtures("guide_db")
def test_catchup_player_page_rejects_invalid_start(auth_client: TestClient):
    _set_live(_archive_stream())
    assert auth_client.get("/play/live/src1_42?start=1e300").status_code == 404


@pytest.mark.usefixtures("guide_db")
def test_catchup_api_unknown_stream_is_not_found(auth_client: TestClient):
    _set_live()
    assert auth_client.get("/api/live/catchup/missing").status_code == 404


def _start(url: str, *, user: str = "testuser", **params: str) -> dict[str, str]:
    """Query for /transcode/start with the token a player page gives ``user``."""
    return {"url": url, "url_token": auth.stream_url_token(user, url), **params}


class _StartRecorder:
    """Stands in for ``ffmpeg_session.start_transcode`` and records the URLs it was asked to open."""

    def __init__(self) -> None:
        self.urls: list[str] = []
        self.options: list[dict[str, object]] = []

    async def __call__(self, url: str, *args: object, **kwargs: object) -> dict[str, str]:
        del args
        self.urls.append(url)
        self.options.append(kwargs)
        return {"session_id": "s"}


@pytest.fixture
def start_recorder(monkeypatch: pytest.MonkeyPatch) -> _StartRecorder:
    recorder = _StartRecorder()
    monkeypatch.setattr(ffmpeg_session, "start_transcode", recorder)
    return recorder


class _Prestarts:
    """Stands in for ``ffmpeg_session.prestart_live``: play pages start no real stream."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, url: str, **options: object) -> None:
        self.calls.append((url, options))


@pytest.fixture(autouse=True)
def prestarts(monkeypatch: pytest.MonkeyPatch) -> _Prestarts:
    recorder = _Prestarts()
    monkeypatch.setattr(ffmpeg_session, "prestart_live", recorder)
    return recorder


_OTA = "http://tuner.test:5004/auto/v9.1"


@pytest.mark.usefixtures("guide_db")
def test_a_live_page_starts_its_stream_while_its_scripts_load(
    auth_client: TestClient,
    prestarts: _Prestarts,
):
    _set_live({"stream_id": "ota", "name": "KQED", "direct_url": _OTA, "source_id": "tuner"})
    assert auth_client.get("/play/live/ota").status_code == 200
    assert prestarts.calls == [
        (
            _OTA,
            {
                "username": "testuser",
                "source_id": "tuner",
                "deinterlace_fallback": True,
                "user_max_streams": 0,
                "source_max_streams": 0,
            },
        ),
    ]


@pytest.mark.usefixtures("guide_db")
def test_a_page_that_will_not_transcode_starts_nothing(
    auth_client: TestClient,
    prestarts: _Prestarts,
):
    _set_live({"stream_id": "ota", "name": "KQED", "direct_url": _OTA, "source_id": "tuner"})
    cache_module.save_server_settings(
        {**cache_module.load_server_settings(), "transcode_mode": "never"},
    )
    assert auth_client.get("/play/live/ota").status_code == 200
    assert prestarts.calls == []


@pytest.mark.usefixtures("guide_db")
def test_a_stream_playing_on_a_tv_is_not_started_again(
    auth_client: TestClient,
    prestarts: _Prestarts,
    monkeypatch: pytest.MonkeyPatch,
):
    _set_live({"stream_id": "ota", "name": "KQED", "direct_url": _OTA, "source_id": "tuner"})

    def casting(url: str) -> bool:
        return url == _OTA

    monkeypatch.setattr(chromecast.manager, "owns_url", casting)
    assert auth_client.get("/play/live/ota").status_code == 200
    assert prestarts.calls == []


@pytest.mark.parametrize("case", ["missing", "other-user", "other-url", "forged", "not-hex"])
def test_transcode_start_refuses_urls_the_server_did_not_issue(
    auth_client: TestClient,
    start_recorder: _StartRecorder,
    case: str,
):
    url = "http://169.254.169.254/x"
    params = {"url": url}
    # Tokens are made here, inside the fixture: the secret key lives in its settings.
    tokens = {
        "other-user": auth.stream_url_token("other", url),
        "other-url": auth.stream_url_token("testuser", "http://provider/live"),
        "forged": "0" * 64,
        "not-hex": "\u00e9" * 64,
    }
    if case in tokens:
        params["url_token"] = tokens[case]
    response = auth_client.get("/transcode/start", params=params)
    assert response.status_code == 403
    assert start_recorder.urls == []


@pytest.mark.usefixtures("guide_db")
def test_player_page_token_starts_that_stream_for_that_user(
    auth_client: TestClient,
    start_recorder: _StartRecorder,
):
    _set_live(
        {
            "stream_id": "test",
            "name": "Test channel",
            "direct_url": "https://provider.example/live.m3u8?user=a&pass=b",
        },
    )
    page = auth_client.get("/play/live/test")
    url_match = re.search(r"rawUrl: (\"[^\"]*\")", page.text)
    token_match = re.search(r"rawUrlToken: (\"[0-9a-f]+\")", page.text)
    assert url_match is not None and token_match is not None
    url, token = json.loads(url_match.group(1)), json.loads(token_match.group(1))
    assert url == "https://provider.example/live.m3u8?user=a&pass=b"
    ok = auth_client.get("/transcode/start", params={"url": url, "url_token": token})
    auth_client.cookies.set("token", auth.create_token({"sub": "other"}))
    other = auth_client.get("/transcode/start", params={"url": url, "url_token": token})
    assert ok.status_code == 200
    assert other.status_code == 403
    assert start_recorder.urls == [url]


@pytest.mark.parametrize(("query", "expected"), [({}, 2), ({"min_segments": "1"}, 1)])
def test_a_start_can_ask_to_be_answered_at_the_first_segment(
    auth_client: TestClient,
    start_recorder: _StartRecorder,
    query: dict[str, str],
    expected: int,
):
    url = "http://provider/live"
    response = auth_client.get("/transcode/start", params=_start(url, **query))
    assert response.status_code == 200
    assert start_recorder.options[0].get("min_segments") == expected


@pytest.mark.parametrize("value", ["0", "3"])
def test_a_start_cannot_ask_for_other_segment_counts(
    auth_client: TestClient,
    start_recorder: _StartRecorder,
    value: str,
):
    response = auth_client.get(
        "/transcode/start",
        params=_start("http://provider/live", min_segments=value),
    )
    assert response.status_code == 422
    assert start_recorder.urls == []


@pytest.mark.asyncio
async def test_adaptive_start_uses_shared_backend(auth_client: TestClient, fast_live: _FastLive):
    session_id = await fast_live.start()
    response = auth_client.get("/transcode/start", params=_start(PROVIDER))
    assert response.status_code == 200
    body = _fields(response)
    assert re.fullmatch(r"[0-9a-f]{32}", str(body.pop("viewer_id")))
    assert body == {
        "session_id": session_id,
        "playlist": f"/transcode/{session_id}/low.m3u8",
        "master_playlist": f"/transcode/{session_id}/master.m3u8",
        "playback_buffer_seconds": 12.0,
        "subtitles": [],
        "duration": 0,
        "seek_offset": 0,
    }
    # The viewer's session is reused: still one worker and one upstream connection.
    assert len(fast_live.launcher.launches) == 1
    assert len(fast_live.upstreams.opened) == 1
    await ffmpeg_session.stop_session(session_id, force=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnected", [True, False])
async def test_disconnected_adaptive_start_is_cleaned_up(fast_live: _FastLive, disconnected: bool):
    session_id = await fast_live.start()
    # The start above was a server prestart, which is no viewer of its own.
    session = ffmpeg_session._transcode_sessions[session_id]
    assert ffmpeg_session._is_fast_live(session)
    session["viewers"].clear()
    [worker] = fast_live.launcher.processes
    scope: Scope = {"type": "http", "method": "GET", "headers": [], "query_string": b""}
    request = Request(scope, _gone if disconnected else _connected)
    start = main.transcode_start(
        request,
        {"sub": "testuser"},
        PROVIDER,
        auth.stream_url_token("testuser", PROVIDER),
    )
    if disconnected:
        with pytest.raises(HTTPException) as excinfo:
            await start
        assert excinfo.value.status_code == 499
        # Only this request's viewing ends; with no one else, the stream follows.
        await _end_grace(fast_live)
        assert session_id not in ffmpeg_session._transcode_sessions
        assert worker.signals == ["TERM"]
        assert fast_live.upstreams.opened[0].closed
        assert worker.stdin.closed
    else:
        assert (await start)["session_id"] == session_id
        assert session_id in ffmpeg_session._transcode_sessions
        assert worker.signals == []
        await ffmpeg_session.stop_session(session_id, force=True)


@pytest.mark.parametrize("owner,status", [("testuser", 200), ("other", 404)])
def test_live_page_close_force_stop_checks_owner(
    auth_client: TestClient,
    tmp_path: Path,
    owner: str,
    status: int,
):
    session = _add_session("live", tmp_path, username=owner)
    response = auth_client.post("/transcode/live/stop?force=true")
    assert response.status_code == status
    _assert_stopped("live", session, stopped=status == 200)


@pytest.mark.parametrize("owner,status", [("testuser", 200), ("other", 404)])
def test_archive_stop_without_force_still_releases_only_owner_session(
    auth_client: TestClient,
    tmp_path: Path,
    owner: str,
    status: int,
):
    session = _add_session("archive", tmp_path, username=owner, is_archive=True)
    response = auth_client.post("/transcode/archive/stop")
    assert response.status_code == status
    _assert_stopped("archive", session, stopped=status == 200)


XMLTV_URL = "http://epg.test/guide.xml"


class TestLoadAllEpg:
    @pytest.mark.usefixtures("guide_db")
    def test_refetches_when_listings_ran_out(
        self,
        mock_deps: None,
        monkeypatch: pytest.MonkeyPatch,
    ):
        now = _now()
        _air("news", "Ended", now - timedelta(hours=2), now - timedelta(hours=1))
        web = testing.FakeWeb({XMLTV_URL: testing.FakeResponse(_xmltv("news", now, hours=12))})
        testing.install_web(monkeypatch, web, util)

        main.load_all_epg([(XMLTV_URL, 30, "s")])

        assert web.urls == [XMLTV_URL]
        assert web.fetches[0].timeout == 30
        assert epg.has_programs_after(now + timedelta(hours=6))

    @pytest.mark.usefixtures("guide_db")
    def test_skips_fetch_when_listings_are_current(
        self,
        mock_deps: None,
        monkeypatch: pytest.MonkeyPatch,
    ):
        now = _now()
        _air("news", "Tonight", now, now + timedelta(hours=12))
        web = testing.FakeWeb()
        testing.install_web(monkeypatch, web, util)

        main.load_all_epg([(XMLTV_URL, 30, "s")])

        assert web.fetches == []
        assert epg.get_program_count() == 1


class TestSetup:
    """Tests for initial setup flow."""

    def test_setup_page_shown_when_no_users(self, client: TestClient):
        resp = client.get("/setup", follow_redirects=False)
        assert resp.status_code == 200
        assert b"setup" in resp.content.lower() or b"Create" in resp.content

    def test_setup_redirects_when_users_exist(self, client: TestClient):
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

    def test_login_page_shown_when_users_exist(self, client: TestClient):
        auth.create_user("admin", "password123")

        resp = client.get("/login")
        assert resp.status_code == 200

    def test_login_success_sets_cookie(self, client: TestClient):
        auth.create_user("admin", "password123")

        resp = client.post(
            "/login",
            data={"username": "admin", "password": "password123"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "token" in resp.cookies

    def test_login_failure_returns_401(self, client: TestClient):
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

    @pytest.mark.parametrize("path", ["/", "/guide", "/vod", "/series"])
    def test_redirects_to_login(self, client: TestClient, path: str):
        auth.create_user("admin", "password123")

        resp = client.get(path, follow_redirects=False)
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

    @pytest.mark.usefixtures("guide_db")
    @pytest.mark.parametrize("guide_db_open", [True, False])
    def test_guide_shows_loading_when_no_cache(
        self,
        auth_client: TestClient,
        threads: testing.ThreadRecorder,
        guide_db_open: bool,
    ):
        resp = auth_client.get("/guide")
        assert resp.status_code == 200
        assert b"Loading channel data" in resp.content
        assert [t.is_alive() for t in threads.threads] == [True]
        assert "guide_load" in m3u_module.get_refresh_in_progress()
        assert auth_client.get("/guide").status_code == 200
        assert len(threads.threads) == 1, "A load in progress is not started twice"

        if not guide_db_open:
            db.close()
        threads.run_all()
        assert threads.errors == []
        assert cache_module.get_cache().get("live_streams") == []
        assert ("epg_error" in cache_module.get_cache()) is not guide_db_open
        assert "guide_load" not in m3u_module.get_refresh_in_progress()

    @pytest.mark.usefixtures("guide_db")
    def test_guide_shows_channels_from_cache(self, auth_client: TestClient):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN", "category_ids": ["1"], "epg_channel_id": ""},
        ]

        resp = auth_client.get("/guide?cats=1")
        assert resp.status_code == 200
        assert "CNN" in resp.text

    @pytest.mark.usefixtures("guide_db")
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

    def test_guide_rows_carry_raw_icon_and_signed_proxy(self, auth_client: TestClient):
        icon = "https://logos.example/cnn.png"
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": 1,
                "name": "CNN",
                "category_ids": ["1"],
                "epg_channel_id": "",
                "stream_icon": icon,
            },
        ]

        payload = _body(auth_client.get("/api/guide/rows?cats=1"), GuideRowsResponse)

        channel = payload["rows"][0]["channel"]
        # Apps load the raw icon themselves; the web guide uses the signed proxy.
        assert channel["icon"] == icon
        proxy = parse.urlparse(channel["icon_proxy"])
        assert proxy.path == "/api/logo"
        query = parse.parse_qs(proxy.query)
        assert query["url"] == [icon]
        assert auth.verify_logo_url_signature(icon, query["sig"][0])

    def test_guide_rows_categories_cover_all_pages(self, auth_client: TestClient):
        categories: list[ProviderRecord] = [
            {"category_id": "1", "category_name": "News"},
            {"category_id": "2", "category_name": "Sports"},
            {"category_id": "3", "category_name": "Unavailable"},
        ]
        cache_module.get_cache()["live_categories"] = categories
        # Page 1 holds only News channels; Sports first appears on page 2.
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": index,
                "name": f"Channel {index}",
                "category_ids": ["1" if index < 2 else "2"],
            }
            for index in range(3)
        ] + [{"stream_id": 999, "name": "Hidden", "category_ids": ["3"]}]
        auth.set_user_limits("testuser", unavailable_groups=["cat:3"])

        first = _body(
            auth_client.get("/api/guide/rows?cats=1,2,3&count=2"),
            GuideRowsResponse,
        )
        second = _body(
            auth_client.get("/api/guide/rows?cats=1,2,3&start=2&count=2"),
            GuideRowsResponse,
        )
        reordered = _body(
            auth_client.get("/api/guide/rows?cats=2,1,3&count=1"),
            GuideRowsResponse,
        )

        assert first["total"] == second["total"] == 3
        assert len(first["rows"]) == 2
        assert len(second["rows"]) == 1
        assert second["rows"][0]["channel"]["stream_id"] == 2
        assert first["categories"] == second["categories"] == categories[:2]
        assert reordered["categories"] == categories[1::-1]
        assert reordered["rows"][0]["channel"]["stream_id"] == 2
        past_end = _body(
            auth_client.get("/api/guide/rows?cats=1,2,3&start=10&count=10"),
            GuideRowsResponse,
        )
        assert (past_end["rows"], past_end["total"]) == ([], 3)

    @pytest.mark.usefixtures("guide_db")
    def test_guide_rows_timestamps_anchor_local_timeline(
        self,
        auth_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        now = datetime(2026, 9, 14, 0, 15, tzinfo=timezone.utc)
        window_start = now.replace(minute=0)
        program_start = window_start - timedelta(minutes=30)
        program_end = window_start + timedelta(hours=1)
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "News", "category_ids": ["1"], "epg_channel_id": "news"},
        ]
        _air("news", "Overnight news", program_start, program_end)
        testing.install_clock(monkeypatch, testing.FakeClock(now.timestamp()), main)

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

    @pytest.mark.usefixtures("guide_db")
    def test_guide_rows_flag_archived_programs(self, auth_client: TestClient):
        now = _now()
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
        past = _air("news", "Earlier", now - timedelta(hours=1), now - timedelta(minutes=5))
        _air("news", "Now", now - timedelta(minutes=5), now + timedelta(hours=1))
        _air("other", "Earlier", past.start, past.stop)

        rows = _body(auth_client.get("/api/guide/rows?cats=1&offset=-1"), GuideRowsResponse)["rows"]

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

    @pytest.mark.usefixtures("guide_db")
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

        start = _now() - timedelta(days=2)
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
        _air("guide", "Earlier program", start, start + timedelta(hours=1))
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

    @pytest.mark.usefixtures("guide_db")
    def test_guide_page_exposes_timestamps_for_local_times(self, auth_client: TestClient):
        now = datetime.now(timezone.utc)
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "News", "category_ids": ["1"], "epg_channel_id": "news"},
        ]
        program = _air(
            "news",
            "Headlines",
            _now() - timedelta(minutes=5),
            _now() + timedelta(minutes=55),
        )
        html = auth_client.get("/guide?cats=1").text
        window_start = now.replace(minute=0, second=0, microsecond=0)
        assert f'data-clock="{window_start.timestamp()}"' in html
        assert f'data-start="{program.start.timestamp()}"' in html
        assert f'data-end="{program.stop.timestamp()}"' in html
        assert "localizeGuideTimes(document)" in html

    @pytest.mark.usefixtures("guide_db")
    def test_guide_uses_saved_filter(self, auth_client: TestClient):
        _save_guide_filter(["2"])
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
            {"category_id": "2", "category_name": "Sports"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN", "category_ids": ["1"]},
            {"stream_id": 2, "name": "ESPN", "category_ids": ["2"]},
        ]

        # Guide now renders directly using saved filter (no redirect)
        resp = auth_client.get("/guide")
        assert resp.status_code == 200
        assert "ESPN" in resp.text
        assert "CNN" not in resp.text


class TestVod:
    """Tests for VOD page."""

    def test_vod_shows_loading_when_no_cache(
        self,
        auth_client: TestClient,
        threads: testing.ThreadRecorder,
    ):
        resp = auth_client.get("/vod")
        assert resp.status_code == 200
        assert len(threads.threads) == 1
        assert "vod_load" in m3u_module.get_refresh_in_progress()
        auth_client.get("/vod")
        assert len(threads.threads) == 1
        threads.run_all()
        assert cache_module.get_cache().get("vod_streams") == []
        assert "vod_load" not in m3u_module.get_refresh_in_progress()

    def test_vod_shows_movies_from_cache(self, auth_client: TestClient):
        cache_module.get_cache()["vod_categories"] = [
            {"category_id": "10", "category_name": "Movies", "source_id": "src1"},
        ]
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 100, "name": "Movie 1", "category_id": "10", "source_id": "src1"},
        ]

        resp = auth_client.get("/vod?sort=alpha")
        assert resp.status_code == 200
        assert "Movie 1" in resp.text

    def test_vod_loads_movies_from_file_cache(self, auth_client: TestClient):
        cats: list[ProviderRecord] = [{"category_id": "10", "category_name": "Westerns"}]
        streams: list[ProviderRecord] = [{"stream_id": 100, "name": "Movie 1", "category_id": "10"}]
        cache_module.save_file_cache("vod_data", {"cats": cats, "streams": streams})

        resp = auth_client.get("/vod?sort=alpha")

        assert resp.status_code == 200
        assert "Westerns" in resp.text
        assert "Movie 1" in resp.text
        store = cache_module.get_cache()
        assert (store.get("vod_categories"), store.get("vod_streams")) == (cats, streams)

    def test_vod_filters_by_category(self, auth_client: TestClient):
        cache_module.get_cache()["vod_categories"] = [
            {"category_id": "10", "category_name": "Action", "source_id": "src1"},
            {"category_id": "20", "category_name": "Comedy", "source_id": "src1"},
        ]
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 100, "name": "Action Movie", "category_id": "10", "source_id": "src1"},
            {"stream_id": 101, "name": "Comedy Movie", "category_id": "20", "source_id": "src1"},
        ]

        resp = auth_client.get("/vod?category=10&sort=alpha")
        assert resp.status_code == 200
        assert "Action Movie" in resp.text
        assert "Comedy Movie" not in resp.text

    def test_vod_sorts_by_alpha(self, auth_client: TestClient):
        cache_module.get_cache()["vod_categories"] = []
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 1, "name": "Zebra", "source_id": "src1"},
            {"stream_id": 2, "name": "apple", "source_id": "src1"},
            # "_" sorts between the upper- and lowercase letters, so case folding shows.
            {"stream_id": 3, "name": "_Extras", "source_id": "src1"},
        ]

        resp = auth_client.get("/vod?sort=alpha")
        assert resp.status_code == 200
        assert resp.text.index("_Extras") < resp.text.index("apple") < resp.text.index("Zebra")

    def test_vod_sorts_unrated_movies_last(self, auth_client: TestClient):
        cache_module.get_cache()["vod_categories"] = []
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 1, "name": "Unrated", "rating": None},
            {"stream_id": 2, "name": "Poor", "rating": "0.5"},
        ]

        resp = auth_client.get("/vod?sort=rating")
        assert resp.text.index("Poor") < resp.text.index("Unrated")


class TestSeries:
    """Tests for series page."""

    def test_series_shows_loading_when_no_cache(
        self,
        auth_client: TestClient,
        threads: testing.ThreadRecorder,
    ):
        resp = auth_client.get("/series")
        assert resp.status_code == 200
        assert len(threads.threads) == 1
        assert "series_load" in m3u_module.get_refresh_in_progress()
        auth_client.get("/series")
        assert len(threads.threads) == 1
        threads.run_all()
        assert cache_module.get_cache().get("series") == []
        assert "series_load" not in m3u_module.get_refresh_in_progress()

    def test_series_shows_list_from_cache(self, auth_client: TestClient):
        cache_module.get_cache()["series_categories"] = [
            {"category_id": "30", "category_name": "Drama", "source_id": "src1"},
        ]
        cache_module.get_cache()["series"] = [
            {"series_id": 200, "name": "Show 1", "category_id": "30", "source_id": "src1"},
        ]

        resp = auth_client.get("/series?sort=alpha")
        assert resp.status_code == 200
        assert "Show 1" in resp.text


@pytest.mark.usefixtures("imdb_index")
class TestSearch:
    """Tests for search page."""

    @pytest.fixture
    def imdb_index(self, guide_db: None) -> None:
        del guide_db
        imdb.init()

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
        assert "CNN News" in resp.text
        assert "BBC World" not in resp.text

    def test_search_regex_mode(self, auth_client: TestClient):
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN News"},
            {"stream_id": 2, "name": "CNBC Finance"},
            {"stream_id": 3, "name": "BBC World"},
        ]
        cache_module.get_cache()["live_categories"] = []
        cache_module.get_cache()["epg_urls"] = []
        cache_module.get_cache()["vod_streams"] = []
        cache_module.get_cache()["series"] = []

        resp = auth_client.get("/search?q=CN.*&regex=true&live=true")
        assert resp.status_code == 200
        assert "CNN News" in resp.text
        assert "CNBC Finance" in resp.text
        assert "BBC World" not in resp.text

    def test_search_matches_cast_and_director(self, auth_client: TestClient):
        cache_module.get_cache()["live_streams"] = []
        cache_module.get_cache()["vod_streams"] = []
        cache_module.get_cache()["series"] = [
            {"series_id": 1, "name": "Fargo", "director": "Noah Hawley", "cast": "Billy Bob"},
            {"series_id": 2, "name": "Legion", "director": "Noah Hawley", "cast": "Dan Stevens"},
            {"series_id": 3, "name": "Lost", "director": "J. J. Abrams", "cast": "Matthew Fox"},
        ]

        by_director = auth_client.get("/search?q=hawley&series=true").text
        assert "Fargo" in by_director and "Legion" in by_director and "Lost" not in by_director
        by_cast = auth_client.get("/search?q=stevens&series=true").text
        assert "Legion" in by_cast and "Fargo" not in by_cast

    def test_search_matches_credits_from_imdb(self, auth_client: TestClient):
        cache_module.get_cache()["live_streams"] = [{"stream_id": 9, "name": "Mann TV - 1995"}]
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 1, "name": "Heat - 1995", "container_extension": "mkv"},
            {"stream_id": 2, "name": "Ronin - 1998", "container_extension": "mkv"},
        ]
        cache_module.get_cache()["series"] = [
            {"series_id": 3, "name": "Miami Vice (1984)", "cast": "Don Johnson"},
        ]
        heat: imdb.Info = {
            "imdb_id": "tt1",
            "title": "Heat",
            "year": 1995,
            "genres": [],
            "runtime_min": 0,
            "rating": 0.0,
            "votes": 0,
            "directors": ["Michael Mann"],
            "writers": [],
            "cast": ["Al Pacino"],
            "crew": {},
        }
        vice: imdb.Info = {**heat, "directors": ["Mann"]}
        index = {
            f"movies:{imdb.title_key('Heat', 1995)}": heat,
            f"series:{imdb.title_key('Miami Vice', 1984)}": vice,
            # Live channels are never looked up, even when a key would match.
            f"movies:{imdb.title_key('Mann TV', 1995)}": heat,
        }
        imdb._write(index, set())

        by_director = auth_client.get("/search?q=mann").text
        assert "Heat - 1995" in by_director and "Miami Vice" in by_director
        assert "Ronin" not in by_director and "Mann TV" in by_director
        assert "Heat - 1995" in auth_client.get("/search?q=pacino&vod=true").text
        assert "Mann TV" not in auth_client.get("/search?q=pacino&live=true").text

    def test_search_rejects_long_regex(self, auth_client: TestClient):
        cache_module.get_cache()["live_streams"] = []

        resp = auth_client.get(f"/search?q={'a' * 101}&regex=true&live=true")
        assert resp.status_code == 400


class TestSettings:
    """Tests for settings page."""

    def test_only_nomos_engine_is_discovered(
        self,
        auth_client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        (tmp_path / "retired-model_720p_fp16.engine").write_bytes(b"engine")
        monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path))
        assert main.get_sr_models() == []

        (tmp_path / "2x-nomosuni-compact_720p_fp16.engine").write_bytes(b"engine")
        assert main.get_sr_models() == ["2x-nomosuni-compact"]

    def test_settings_page_renders(
        self,
        auth_client: TestClient,
        threads: testing.ThreadRecorder,
    ):
        resp = auth_client.get("/settings")
        assert resp.status_code == 200
        assert 'id="sort-live-by-views" checked' in resp.text
        assert len(threads.threads) == 1

    def test_settings_page_maps_enabled_retired_model_to_nomos(
        self,
        auth_client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        cache_module.get_cache()["live_categories"] = []
        settings = cache_module.load_server_settings()
        settings["sr_model"] = "retired-model"
        cache_module.save_server_settings(settings)
        (tmp_path / "2x-nomosuni-compact_720p_fp16.engine").write_bytes(b"engine")
        monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path))

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
        assert cache_module.load_user_settings("testuser")["guide_filter"] == ["1", "2", "3"]

    def test_settings_captions(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/captions",
            data={"enabled": "on"},
        )
        assert resp.status_code == 200
        assert _fields(resp)["ok"] is True
        assert cache_module.load_user_settings("testuser")["captions_enabled"] is True

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
        cache_module.get_cache()["live_streams"] = []
        resp = auth_client.post(
            "/settings/add",
            data={
                "name": "Test Provider",
                "source_type": "xtream",
                "url": "http://example.com/",
                "username": "user",
                "password": "pass",
                "epg_timeout": 120,
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        [source] = cache_module.get_sources()
        assert (source.name, source.type, source.url) == (
            "Test Provider",
            "xtream",
            "http://example.com",
        )
        assert "live_streams" not in cache_module.get_cache()

    def test_add_m3u_source(self, auth_client: TestClient):
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
        assert [s.type for s in cache_module.get_sources()] == ["m3u"]

    @pytest.mark.parametrize(
        "data",
        [
            {"name": "Bad Source", "source_type": "invalid", "url": "http://example.com"},
            {"name": "Bad Source", "source_type": "xtream", "url": "ftp://example.com"},
            {"name": "x" * 201, "source_type": "xtream", "url": "http://example.com"},
        ],
    )
    def test_add_source_validates(self, auth_client: TestClient, data: dict[str, str]):
        resp = auth_client.post("/settings/add", data=data)
        assert resp.status_code == 400
        assert cache_module.get_sources() == []


class TestDeleteSource:
    """Tests for deleting sources."""

    @pytest.mark.usefixtures("guide_db")
    def test_delete_source(self, auth_client: TestClient):
        settings = cache_module.load_server_settings()
        settings["sources"] = [
            {"id": "src_123", "name": "Test", "type": "xtream", "url": "http://example.com"},
        ]
        cache_module.save_server_settings(settings)
        epg.insert_channel("ch", "Channel", "src_123")
        _air("ch", "Show", _now(), _now() + timedelta(hours=1), source_id="src_123")

        resp = auth_client.post("/settings/delete/src_123", follow_redirects=False)
        assert resp.status_code == 303
        assert cache_module.get_sources() == []
        assert epg.get_program_count() == 0


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

    def test_audio_language_defaults_to_english_and_saves(self, auth_client: TestClient):
        assert _fields(auth_client.get("/api/user-prefs"))["audio_lang"] == "en"
        auth_client.post("/api/user-prefs", json={"audio_lang": "fr"})
        assert _fields(auth_client.get("/api/user-prefs"))["audio_lang"] == "fr"

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

    def test_delete_user(self, auth_client: TestClient):
        auth.create_user("otheruser", "password123")

        resp = auth_client.post("/settings/users/delete/otheruser", follow_redirects=False)
        assert resp.status_code == 303
        assert "otheruser" not in auth.get_all_usernames()

    def test_cannot_delete_self(self, auth_client: TestClient):
        resp = auth_client.post("/settings/users/delete/testuser")
        assert resp.status_code == 400

    def test_change_password(self, auth_client: TestClient):
        resp = auth_client.post(
            "/settings/users/password",
            data={"current_password": "testpass123", "new_password": "newpass456"},
        )
        assert resp.status_code == 200
        assert auth.verify_password("testuser", "newpass456")

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
        assert cache_module.load_server_settings()["transcode_mode"] == "always"


class TestTranscodeRoutes:
    """Tests for transcode routes."""

    def test_transcode_file_not_found(self, auth_client: TestClient):
        resp = auth_client.get("/transcode/invalid-session/stream.m3u8")
        assert resp.status_code == 404

    @pytest.mark.parametrize("method", ["DELETE", "POST"])
    def test_transcode_stop_keeps_movie_segments(
        self,
        auth_client: TestClient,
        tmp_path: Path,
        method: str,
    ):
        session = _add_session("test-session", tmp_path, last_access=0.0)
        path = "/transcode/test-session" + ("/stop" if method == "POST" else "")
        resp = auth_client.request(method, path)
        assert resp.status_code == 200
        assert _fields(resp)["status"] == "stopped"
        assert _signals(session) == ["TERM"]
        # The movie keeps its segments to resume from; live is covered by TestFastStartRoutes.
        assert "test-session" in ffmpeg_session._transcode_sessions
        assert tmp_path.exists()

    def test_transcode_progress_not_found(self, auth_client: TestClient):
        resp = auth_client.get("/transcode/progress/invalid-session")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.parametrize("filename", ["master.m3u8", "low.m3u8"])
    async def test_adaptive_playlists_share_dates_but_master_is_not_rewritten(
        self,
        client: TestClient,
        fast_live: _FastLive,
        filename: str,
    ):
        session_id = await fast_live.start()
        session = ffmpeg_session._transcode_sessions[session_id]
        directory = Path(session["dir"])
        response = client.get(f"/transcode/{session_id}/{filename}")
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"
        if filename == "master.m3u8":
            assert response.text == (directory / "master.m3u8").read_text()
        else:
            # Two 2 s segments call for the 12 s minimum playback buffer.
            assert session.get("playback_buffer_seconds") == 12
            assert "#EXT-X-START:TIME-OFFSET=-12,PRECISE=NO\n" in response.text
            origin = session.get("origin_time", 0)
            assert _program_dates(response.text) == {
                "low_0.ts": pytest.approx(origin),
                "low_1.ts": pytest.approx(origin + 2),
            }
        # Only the web player asks for blocking reload; other clients poll as before.
        assert "SERVER-CONTROL" not in response.text
        assert "?blocking=1" not in response.text
        await ffmpeg_session.stop_session(session_id, force=True)

    @pytest.mark.asyncio
    async def test_the_web_player_reloads_its_playlist_by_blocking(
        self,
        client: TestClient,
        fast_live: _FastLive,
    ):
        session_id = await fast_live.start()

        master = client.get(f"/transcode/{session_id}/master.m3u8?blocking=1").text
        playlist = client.get(f"/transcode/{session_id}/low.m3u8?blocking=1").text
        # Both segments are listed, so asking for the second returns at once.
        listed = client.get(f"/transcode/{session_id}/low.m3u8?blocking=1&_HLS_msn=1").text

        assert "\nlow.m3u8?blocking=1\n" in master
        assert "#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES\n" in playlist
        assert "low_1.ts" in listed
        await ffmpeg_session.stop_session(session_id, force=True)

    def test_segmented_webvtt_is_mapped_to_the_video_clock(
        self,
        auth_client: TestClient,
        tmp_path: Path,
    ):
        filename = "sub0_000001.vtt"
        (tmp_path / filename).write_text("WEBVTT\n\n00:01.000 --> 00:03.000\nCaption\n")
        _add_session("session", tmp_path, caption_timestamp_origin=10.08)

        response = auth_client.get(f"/transcode/session/{filename}")

        assert response.status_code == 200
        assert "X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:907200" in response.text


def _write_segments(directory: Path, names_and_pts: list[tuple[str, float]]) -> str:
    for name, pts in names_and_pts:
        (directory / name).write_bytes(testing.video_packet(pts) * 10)
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


PROVIDER = "https://provider/live"
LOW_HEADER = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n"


@pytest_asyncio.fixture
async def upstreams(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[testing.FakeUpstreams]:
    """The provider's live connections; closed before the loop ends to free reader threads."""
    opened = testing.FakeUpstreams()
    testing.install_web(monkeypatch, testing.FakeWeb({PROVIDER: opened}), util)
    yield opened
    opened.close()


class _FastLive:
    """The provider, probe and ffmpeg workers behind a real fast-start session.

    Each worker launch publishes the next entry of ``renditions`` to low.m3u8, after
    a discontinuity when an earlier worker already published there; launches past
    the last entry publish nothing.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, upstreams: testing.FakeUpstreams) -> None:
        self.upstreams = upstreams
        self.clock = testing.FakeClock()
        testing.install_clock(monkeypatch, self.clock, ffmpeg_session)
        probe = testing.FakeRun()
        probe.program("ffprobe", testing.Program(stdout=testing.probed_video_json()))
        testing.install_subprocess(monkeypatch, ffmpeg_command, run=probe)
        self.tasks = testing.BackgroundTasks()
        monkeypatch.setattr(ffmpeg_session, "_spawn_background_task", self.tasks)
        monkeypatch.setattr(ffmpeg_session, "_startup_guards", {})
        monkeypatch.setattr(ffmpeg_session, "_recordings", set())
        self.renditions: list[list[tuple[str, float]]] = [[("low_0.ts", 10.0), ("low_1.ts", 12.0)]]
        self.launcher = testing.ProcessLauncher(on_launch=self._publish)
        testing.install_launcher(monkeypatch, self.launcher, ffmpeg_session)

    async def start(self, *, seconds: float = 10.0) -> str:
        """Start testuser's session of the provider, whose source opens at ``seconds``."""
        task = asyncio.create_task(
            ffmpeg_session.start_transcode(PROVIDER, username="testuser"),
        )
        await self.feed(task, connection=0, data=testing.ts_source(seconds=seconds))
        return (await task)["session_id"]

    async def feed(self, task: asyncio.Task[object], *, connection: int, data: bytes) -> None:
        """Wait until ``task`` opens upstream ``connection``, then deliver ``data`` on it."""
        await _until(lambda: len(self.upstreams.opened) > connection, task)
        self.upstreams.opened[connection].feed(data)

    def _publish(self, command: list[str]) -> None:
        if not self.renditions:  # A recorder, which publishes nothing startup waits for.
            return
        directory = Path(command[-1]).parent
        playlist = directory / "low.m3u8"
        listed = (
            playlist.read_text() + "#EXT-X-DISCONTINUITY\n" if playlist.exists() else LOW_HEADER
        )
        playlist.write_text(listed + _write_segments(directory, self.renditions.pop(0)))


@pytest.fixture
def fast_live(
    mock_deps: None,
    monkeypatch: pytest.MonkeyPatch,
    upstreams: testing.FakeUpstreams,
    transcode_dir: Path,
) -> Iterator[_FastLive]:
    del transcode_dir
    fake = _FastLive(monkeypatch, upstreams)
    yield fake
    fake.tasks.discard()


async def _until(condition: Callable[[], bool], task: asyncio.Task[object]) -> None:
    """Yield to ``task`` until ``condition`` holds, surfacing the task's failure."""
    for _ in range(100_000):
        if condition():
            return
        if task.done():
            task.result()
            raise AssertionError("task finished before the condition held")
        await asyncio.sleep(0)
    raise AssertionError("condition never held")


async def _end_grace(fast_live: _FastLive) -> None:
    """Let each pending grace for an unwatched live session run out."""
    for task in fast_live.tasks.tasks:
        if not task.done() and getattr(task.coroutine, "__name__", "") == "stop_when_unwatched":
            await task.run()


async def _open_page(fast_live: _FastLive) -> ffmpeg_session.TranscodeResponse:
    """A player page starting testuser's provider stream, as the first viewer."""
    task = asyncio.create_task(ffmpeg_session.start_transcode(PROVIDER, username="testuser"))
    await fast_live.feed(task, connection=0, data=testing.ts_source(seconds=10.0))
    return await task


class TestSharedLiveSessions:
    """Each page load is one viewer; a stream stops only when its last viewer leaves."""

    @pytest.mark.asyncio
    async def test_a_reloaded_page_keeps_the_stream_its_old_page_stops(self, fast_live: _FastLive):
        old_page = await _open_page(fast_live)
        new_page = await ffmpeg_session.start_transcode(PROVIDER, username="testuser")
        session_id = old_page["session_id"]
        assert new_page["session_id"] == session_id
        assert old_page.get("viewer_id") != new_page.get("viewer_id")
        [worker] = fast_live.launcher.processes
        user: TokenClaims = {"sub": "testuser"}

        # The old page's close beacon lands after the new page joined.
        await main.transcode_stop_post(
            session_id,
            user,
            force=True,
            viewer=old_page.get("viewer_id", ""),
        )
        assert session_id in ffmpeg_session._transcode_sessions
        assert worker.signals == []

        await main.transcode_stop(
            session_id,
            user,
            force=True,
            viewer=new_page.get("viewer_id", ""),
        )
        await _end_grace(fast_live)
        assert session_id not in ffmpeg_session._transcode_sessions
        assert worker.signals == ["TERM"]

    @pytest.mark.asyncio
    async def test_a_reload_whose_old_stop_lands_first_rejoins_the_stream(
        self,
        fast_live: _FastLive,
    ):
        """The old page's close beacon usually arrives before the new page asks."""
        old_page = await _open_page(fast_live)
        session_id = old_page["session_id"]
        user: TokenClaims = {"sub": "testuser"}
        await main.transcode_stop_post(
            session_id,
            user,
            force=True,
            viewer=old_page.get("viewer_id", ""),
        )
        new_page = await ffmpeg_session.start_transcode(PROVIDER, username="testuser")
        await _end_grace(fast_live)
        assert new_page["session_id"] == session_id
        assert session_id in ffmpeg_session._transcode_sessions
        assert len(fast_live.launcher.launches) == 1
        await ffmpeg_session.stop_session(session_id, force=True)

    @pytest.mark.asyncio
    async def test_a_viewer_that_stopped_reporting_does_not_hold_the_stream(
        self,
        fast_live: _FastLive,
    ):
        """A closed tab whose beacon never arrived expires with its heartbeat."""
        watching = await _open_page(fast_live)
        await ffmpeg_session.start_transcode(PROVIDER, username="testuser")
        session_id = watching["session_id"]
        user: TokenClaims = {"sub": "testuser"}
        health = PlaybackHealth(buffer_seconds=12, waiting=False)
        fast_live.clock.advance(31)
        await main.transcode_health(session_id, health, user, viewer=watching.get("viewer_id", ""))

        await main.transcode_stop(
            session_id,
            user,
            force=True,
            viewer=watching.get("viewer_id", ""),
        )
        await _end_grace(fast_live)

        assert session_id not in ffmpeg_session._transcode_sessions

    @pytest.mark.asyncio
    async def test_a_stop_without_a_viewer_ends_the_stream(self, fast_live: _FastLive):
        """Clients that predate viewer ids, and the server's own cleanup, stop it outright."""
        page = await _open_page(fast_live)
        await ffmpeg_session.start_transcode(PROVIDER, username="testuser")
        await main.transcode_stop(page["session_id"], {"sub": "testuser"}, force=True)
        assert page["session_id"] not in ffmpeg_session._transcode_sessions


class TestFastStartRoutes:
    @pytest.mark.asyncio
    async def test_program_dates_stay_monotonic_after_pipeline_restart(
        self,
        client: TestClient,
        fast_live: _FastLive,
    ):
        restarted = [(f"low_{i}.ts", 1.48 + 2 * (i - 100)) for i in range(100, 103)]
        fast_live.renditions = [[(f"low_{i}.ts", 5000 + 2 * i) for i in range(3)], restarted]
        session_id = await fast_live.start(seconds=5000)
        session = ffmpeg_session._transcode_sessions[session_id]
        assert ffmpeg_session._is_fast_live(session)
        listing = f"/transcode/{session_id}/low.m3u8"
        # No client listed the old segments yet; they still keep the old clock's dates.
        first_byte_at = session.get("origin_time", 0)

        # The upstream stalls for a minute, then the watchdog reconnects it.
        fast_live.clock.advance(60)
        session["live_recovery"], permitted = session["live_recovery"].permit_restart(
            "pipeline",
            fast_live.clock.monotonic(),
        )
        assert permitted
        restart = asyncio.create_task(
            ffmpeg_session._restart_fast_pipeline(session_id, session, "input"),
        )
        await fast_live.feed(restart, connection=1, data=testing.ts_source(seconds=1.48))
        await restart
        during = _program_dates(client.get(listing).text)
        # The old segments age out of the window.
        (Path(session["dir"]) / "low.m3u8").write_text(
            LOW_HEADER + "".join(f"#EXTINF:2.000000,\n{name}\n" for name, _ in restarted),
        )
        after = _program_dates(client.get(listing).text)

        assert len(fast_live.launcher.launches) == 2
        assert fast_live.upstreams.opened[0].closed
        assert session.get("playlist_generation") == 1
        assert during["low_0.ts"] == pytest.approx(first_byte_at)
        assert during["low_2.ts"] == pytest.approx(first_byte_at + 4)
        assert during["low_2.ts"] < during["low_100.ts"] <= fast_live.clock.now
        assert after == {name: during[name] for name, _ in restarted}
        await ffmpeg_session.stop_session(session_id, force=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("switched", [False, True])
    async def test_low_playlist_continues_into_the_upscaled_rendition(
        self,
        client: TestClient,
        fast_live: _FastLive,
        switched: bool,
    ):
        # The plain worker published low_0 (PTS 10) and low_1 (PTS 12).
        session_id = await fast_live.start()
        session = ffmpeg_session._transcode_sessions[session_id]
        assert ffmpeg_session._is_fast_live(session)
        directory = Path(session["dir"])
        if switched:
            # The upscaler caught up and took over, as _upgrade_to_upscale leaves it.
            upscaled = _write_segments(directory, [("up_1.ts", 12.0), ("up_2.ts", 14.0)])
            (directory / "up.m3u8").write_text(LOW_HEADER + upscaled)
            session["rendition"] = "up"
            session["splice_low"] = (directory / "low.m3u8").read_text()

        # Clients keep requesting low.m3u8 across the switch.
        text = client.get(f"/transcode/{session_id}/low.m3u8").text

        # The upscaled rendition continues where the plain playlist ended (PTS 14).
        names = ["low_0.ts", "low_1.ts", *(["up_2.ts"] if switched else [])]
        origin = session.get("origin_time", 0)
        # One source clock dates both renditions.
        assert _program_dates(text) == {
            name: pytest.approx(origin + 2 * i) for i, name in enumerate(names)
        }
        if switched:
            assert (
                text.index("low_1.ts") < text.index("#EXT-X-DISCONTINUITY") < text.index("up_2.ts")
            )
        else:
            assert "#EXT-X-DISCONTINUITY" not in text
        assert "#EXT-X-START:TIME-OFFSET=-12,PRECISE=NO\n" in text
        await ffmpeg_session.stop_session(session_id, force=True)

    @pytest.mark.asyncio
    async def test_listing_reads_each_segment_timestamp_once(
        self,
        client: TestClient,
        fast_live: _FastLive,
    ):
        fast_live.renditions = [[(f"low_{i}.ts", 10 + 2 * i) for i in range(4)]]
        session_id = await fast_live.start()
        directory = Path(ffmpeg_session._transcode_sessions[session_id]["dir"])
        listing = f"/transcode/{session_id}/low.m3u8"
        first = client.get(listing)
        # A second listing that re-read the segments would now date them differently.
        _write_segments(directory, [(f"low_{i}.ts", 500 + 2 * i) for i in range(4)])
        second = client.get(listing)
        assert len(_program_dates(first.text)) == 4
        assert first.text == second.text
        await ffmpeg_session.stop_session(session_id, force=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("force", [False, True])
    @pytest.mark.parametrize("route", [main.transcode_stop, main.transcode_stop_post])
    async def test_stop_removes_fast_session_and_lets_its_worker_exit(
        self,
        fast_live: _FastLive,
        force: bool,
        route: Callable[..., Coroutine[object, object, dict[str, str]]],
    ):
        # Even with a VOD cache window a fast-start session cannot resume, so it is not kept.
        settings = cache_module.load_server_settings()
        settings["vod_transcode_cache_mins"] = 30
        cache_module.save_server_settings(settings)
        session_id = await fast_live.start()
        directory = Path(ffmpeg_session._transcode_sessions[session_id]["dir"])
        [worker] = fast_live.launcher.processes

        stopped = await route(session_id, {"sub": "testuser"}, force=force)

        assert stopped == {"status": "stopped"}
        assert session_id not in ffmpeg_session._transcode_sessions
        assert (worker.signals, worker.returncode) == (["TERM"], -15)
        assert worker.stdin.closed
        assert fast_live.upstreams.opened[0].closed
        assert not directory.exists()


class TestSubtitleRoutes:
    """Tests for subtitle routes."""

    def test_subtitle_invalid_filename(self, auth_client: TestClient):
        resp = auth_client.get("/subs/session/notavtt.txt")
        assert resp.status_code == 400

    def test_subtitle_session_not_found(self, auth_client: TestClient):
        resp = auth_client.get("/subs/invalid-session/sub0.vtt")
        assert resp.status_code == 404


@pytest.fixture
def probe_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """One cached episode probe, for series 123."""
    monkeypatch.setattr(ffmpeg_command, "_probe_cache", {})
    monkeypatch.setattr(ffmpeg_command, "_series_probe_cache", {})
    media = MediaInfo(video_codec="h264", audio_codec="aac", pix_fmt="yuv420p", height=720)
    ffmpeg_command.restore_probe_cache_entry("http://x/7.mkv", media, [], 123, 7)


class TestProbeCache:
    """Tests for probe cache endpoints."""

    def test_get_probe_cache(self, auth_client: TestClient, probe_cache: None):
        resp = auth_client.get("/settings/probe-cache")
        assert resp.status_code == 200
        series = _body(resp, _ProbeCache)["series"]
        assert [s["series_id"] for s in series] == [123]

    def test_clear_probe_cache(self, auth_client: TestClient, probe_cache: None):
        resp = auth_client.post("/settings/probe-cache/clear")
        assert resp.status_code == 200
        assert _fields(resp)["cleared"] == 2
        assert _body(auth_client.get("/settings/probe-cache"), _ProbeCache)["series"] == []

    def test_clear_series_probe_cache(self, auth_client: TestClient, probe_cache: None):
        resp = auth_client.post("/settings/probe-cache/clear/123")
        assert resp.status_code == 200
        assert _body(auth_client.get("/settings/probe-cache"), _ProbeCache)["series"] == []

    def test_url_probes_are_listed_by_movie_and_channel_name_and_cleared(
        self,
        auth_client: TestClient,
        probe_cache: None,
    ):
        del probe_cache
        media = MediaInfo(video_codec="h264", audio_codec="ac3", pix_fmt="yuv420p", duration=60)
        ffmpeg_command.remember_probe("http://h/movie/u/p/42.mkv", media)
        ffmpeg_command.remember_probe("http://h/live/u/p/9.ts", media)
        ffmpeg_command.remember_probe("http://hdhr/auto/v2.1", media)
        ffmpeg_command.remember_probe("http://h/movie/u/p/43.mkv", None)
        store = cache_module.get_cache()
        store["vod_streams"] = [{"stream_id": 42, "name": "Answer"}]
        store["live_streams"] = [
            {"stream_id": 9, "name": "News"},
            {"stream_id": "ota_2", "name": "KTVU", "direct_url": "http://hdhr/auto/v2.1"},
        ]
        probes = _body(auth_client.get("/settings/probe-cache"), _ProbeCache)["probes"]
        labels = {(p["kind"], p["stream_id"], p["name"]) for p in probes}
        assert labels == {
            ("movie", "42", "Answer"),
            ("live", "9", "News"),
            ("live", "ota_2", "KTVU"),
            ("other", "", ""),
        }
        assert all("url" not in p for p in probes)
        resp = auth_client.post("/settings/probe-cache/clear-url?kind=live&stream_id=ota_2")
        assert _fields(resp)["cleared"] == 1
        probes = _body(auth_client.get("/settings/probe-cache"), _ProbeCache)["probes"]
        assert ("live", "ota_2") not in {(p["kind"], p["stream_id"]) for p in probes}


class TestRefreshStatus:
    """Tests for refresh status endpoints."""

    def test_guide_refresh_status(self, auth_client: TestClient):
        resp = auth_client.get("/guide/refresh-status")
        assert resp.status_code == 200
        data = _fields(resp)
        assert data["live"] is False
        assert data["epg"] is False

    def test_settings_refresh_status(self, auth_client: TestClient):
        resp = auth_client.get("/settings/refresh-status")
        assert resp.status_code == 200


class TestPlaybackHealth:
    def test_invalid_feedback_is_rejected(self, auth_client: TestClient):
        response = auth_client.post(
            "/transcode/missing/health",
            json={"buffer_seconds": -1, "waiting": True},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_feedback_is_passed_to_backend(
        self,
        auth_client: TestClient,
        fast_live: _FastLive,
    ):
        session_id = await fast_live.start()
        session = ffmpeg_session._transcode_sessions[session_id]
        assert ffmpeg_session._is_fast_live(session)
        policy = session["playback_policy"]
        fast_live.clock.advance(10)
        health = f"/transcode/{session_id}/health"
        response = auth_client.post(health, json={"buffer_seconds": 0, "waiting": True})
        assert response.json() == {
            "bandwidth_saver": False,
            "playlist": f"/transcode/{session_id}/low.m3u8",
        }
        assert session.get("last_access") == fast_live.clock.now
        assert session["playback_policy"] is not policy

        auth.create_user("other", "otherpass1")
        auth_client.cookies.set("token", auth.create_token({"sub": "other"}))
        response = auth_client.post(health, json={"buffer_seconds": 0, "waiting": True})
        assert response.status_code == 404
        await ffmpeg_session.stop_session(session_id, force=True)

    @pytest.mark.parametrize("owner,status", [("testuser", 200), ("other", 404)])
    def test_force_stop_requires_owner(
        self,
        auth_client: TestClient,
        tmp_path: Path,
        owner: str,
        status: int,
    ):
        session = _add_session("live", tmp_path, username=owner)
        assert auth_client.delete("/transcode/live?force=true").status_code == status
        _assert_stopped("live", session, stopped=status == 200)


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

    @pytest.mark.usefixtures("guide_db")
    def test_guide_page_defaults_to_playlists_then_filter(self, auth_client: TestClient):
        playlist = self._create(auth_client)
        auth_client.post("/settings/guide-filter", json={"cats": ["1"]})
        resp = auth_client.get("/guide")
        assert resp.status_code == 200
        assert "★ Favorites" in resp.text
        assert f'"{playlist["category_id"]},1"' in resp.text

    @pytest.mark.usefixtures("guide_db")
    def test_guide_page_shows_section_headings(self, auth_client: TestClient):
        self._create(auth_client)
        auth_client.post("/settings/guide-filter", json={"cats": ["2"]})
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
        text = client.get(url).text
        start = text.index('aria-label="Channel categories"')
        chips = text[start : text.index("</div>", start)]
        links = cast(
            list[tuple[str, str]],
            re.findall(
                r'data-nav="cat".*?title="([^"]*)">.*?tabular-nums[^>]*>(\d+)<',
                chips,
                re.S,
            ),
        )
        current = cast(list[str], re.findall(r'aria-current="page" title="([^"]*)"', chips))
        rows = re.findall(r'<div class="guide-row[^"]*" data-row="\d+"', text)
        return links, current, len(rows)

    @pytest.mark.usefixtures("guide_db")
    def test_guide_sidebar_lists_playlists_and_categories(self, auth_client: TestClient):
        auth_client.post("/settings/guide-filter", json={"cats": ["1", "2"]})
        auth_client.post("/api/user-prefs", json={"guide_selected_cats": ["2"]})
        playlist = self._create(auth_client)

        links, current, rows = self._sidebar(auth_client)
        assert links == [("All", "3"), ("★ Favorites", "2"), ("News", "1"), ("Sports", "2")]
        assert current == ["Sports"]
        assert rows == 2

        links, current, rows = self._sidebar(auth_client, f"/guide?cats={playlist['category_id']}")
        assert current == ["★ Favorites"]
        assert rows == 2
        assert main.load_user_settings("testuser").get("guide_selected_cats") == [
            playlist["category_id"],
        ]

        _, current, rows = self._sidebar(auth_client, "/guide?cats=")
        assert current == ["All"]
        assert rows == 3
        settings = main.load_user_settings("testuser")
        assert "guide_selected_cats" in settings
        assert settings["guide_selected_cats"] is None

    @pytest.mark.usefixtures("guide_db")
    def test_old_multi_category_view_falls_back_to_all(self, auth_client: TestClient):
        auth_client.post("/settings/guide-filter", json={"cats": ["1", "2"]})
        auth_client.post("/api/user-prefs", json={"guide_selected_cats": ["1", "2"]})
        self._create(auth_client)

        _, current, rows = self._sidebar(auth_client)
        assert current == ["All"]
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

    @pytest.mark.usefixtures("guide_db")
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
        _air("guide.one", "Assigned show", _now(), _now() + timedelta(hours=1))
        rows = _body(
            auth_client.get(f"/api/guide/rows?cats={playlist['category_id']}"),
            GuideRowsResponse,
        )["rows"]
        assert rows[0]["channel"]["epg_id"] == "guide.one"
        assert [p["title"] for p in rows[0]["programs"]] == ["Assigned show"]

    @pytest.mark.usefixtures("guide_db")
    def test_epg_channel_search(self, auth_client: TestClient):
        epg.insert_channel("alpha.one.hd", "ALPHA ONE HD", "s")
        epg.insert_channel("alpha.one", "Alpha One", "s")
        epg.insert_channel("alpha.two", "Alpha Two", "s")
        epg.commit()
        resp = auth_client.get("/api/playlists/epg-channels?q=XX| ALPHA ONE HD")
        assert resp.status_code == 200
        results = _body(resp, _EpgChannels)["results"]
        # Provider prefixes and quality tags are not search terms.
        assert sorted(r["id"] for r in results) == ["alpha.one", "alpha.one.hd"]

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
        denied = auth_client.post("/api/playlists", json={"name": "X"})
        assert denied.status_code == 403
        assert _body(denied, _Detail)["detail"] == "Admin access required"
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

    @pytest.mark.usefixtures("guide_db")
    def test_saved_view_drops_deleted_playlist(self, auth_client: TestClient):
        playlist = self._create(auth_client)
        auth_client.post("/settings/guide-filter", json={"cats": ["1"]})
        auth_client.post("/api/user-prefs", json={"guide_selected_cats": [playlist["category_id"]]})
        auth_client.delete(f"/api/playlists/{playlist['id']}")

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
        resp = auth_client.get(
            "/transcode/start",
            params=_start("http://a/x.ts", user="kid", source_id="fam"),
        )
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
def clean_handoff(monkeypatch: pytest.MonkeyPatch, transcode_dir: Path) -> None:
    del transcode_dir
    monkeypatch.setattr(handoff, "_users_dir", None)
    monkeypatch.setattr(handoff, "_records", {})
    monkeypatch.setattr(handoff, "_watches", {})
    monkeypatch.setattr(handoff, "_recorders", {})
    monkeypatch.setattr(handoff, "_generations", {})


def test_handoff_api_offers_the_users_handoff(auth_client: TestClient, clean_handoff: None):
    handoff._records["testuser"] = _handoff_record()
    body = _body(auth_client.get("/api/handoff"), _HandoffBody)["handoff"]
    assert body is not None
    assert body["id"] == "h1"
    assert body["stream_id"] == "src1_42"
    assert "recording_id" not in body
    handoff._records["testuser"]["resumed"] = True
    assert auth_client.get("/api/handoff").json() == {"handoff": None}


@pytest.mark.usefixtures("guide_db")
def test_handoff_recording_resumes_from_where_it_was_left(
    auth_client: TestClient,
    clean_handoff: None,
):
    handoff._records["testuser"] = _handoff_record()
    _set_live(_archive_stream())
    response = auth_client.get("/play/live/src1_42?handoff=h1")
    assert response.status_code == 200
    assert '"/handoff/rec1/stream.m3u8"' in response.text
    assert 'transcodeMode: "never"' in response.text
    assert "catchupSeek: 10.0" in response.text
    assert "handoffRecording: true" in response.text
    assert handoff._records["testuser"]["resumed"] is True


@pytest.mark.usefixtures("guide_db")
def test_handoff_catchup_resumes_through_the_archive(auth_client: TestClient, clean_handoff: None):
    position = _hours_ago(3).timestamp()
    handoff._records["testuser"] = _handoff_record(
        mode="catchup",
        recording_id="",
        position=position,
    )
    _set_live(_archive_stream())
    response = auth_client.get("/play/live/src1_42?handoff=h1")
    assert response.status_code == 200
    assert "/timeshift/user/pass/" in response.text
    assert f"catchupStart: {position}" in response.text
    assert "catchupSeek: 0.0" in response.text
    assert handoff._records["testuser"]["resumed"] is True


@pytest.mark.usefixtures("guide_db")
def test_playing_something_else_cancels_the_handoff(auth_client: TestClient, clean_handoff: None):
    handoff._records["testuser"] = _handoff_record(mode="catchup", recording_id="")
    _set_live({"stream_id": "other", "name": "Other", "direct_url": "https://e.test/l.m3u8"})
    assert auth_client.get("/play/live/other").status_code == 200
    assert "testuser" not in handoff._records


def test_stale_handoff_cannot_be_resumed(auth_client: TestClient, clean_handoff: None):
    handoff._records["testuser"] = _handoff_record(mode="catchup", recording_id="")
    response = auth_client.get("/play/live/src1_42?handoff=gone")
    assert response.status_code == 404
    assert handoff._records["testuser"]["id"] == "h1", "An old link keeps the newer handoff"


@pytest.mark.usefixtures("guide_db")
def test_handoff_stop_hands_off_the_watched_channel(auth_client: TestClient, clean_handoff: None):
    now = time.time()
    _watching("src1_42", now)
    _set_live(_archive_stream())
    response = auth_client.post(
        "/api/handoff/stop",
        json={"stream_id": "src1_42", "position": now - 5},
    )
    assert response.status_code == 200
    assert handoff._records["testuser"]["mode"] == "catchup"
    assert handoff._records["testuser"]["position"] == pytest.approx(now - 5)


@pytest.mark.usefixtures("guide_db")
def test_handoff_stop_rejects_streams_the_user_cannot_see(
    auth_client: TestClient,
    clean_handoff: None,
):
    now = time.time()
    _watching("src1_42", now)
    _set_live(_archive_stream(category_ids=["9"]))
    auth.set_user_limits("testuser", unavailable_groups=["cat:9"])
    response = auth_client.post(
        "/api/handoff/stop",
        json={"stream_id": "src1_42", "position": now - 5},
    )
    assert response.status_code == 404
    assert handoff._records == {}


@pytest.mark.usefixtures("guide_db")
@pytest.mark.asyncio
async def test_handoff_recording_takes_over_the_live_session(
    auth_client: TestClient,
    clean_handoff: None,
    fast_live: _FastLive,
):
    session_id = await fast_live.start()
    [viewer] = fast_live.launcher.processes
    now = time.time()
    _watching("src1_42", now)
    _set_live({"stream_id": "src1_42", "name": "Channel", "direct_url": PROVIDER})
    report = main.HandoffReport(stream_id="src1_42", position=now - 5)

    # On the loop that owns the session's relay, as the server runs it.
    assert await main.handoff_stop_api(report, {"sub": "testuser"}) == {"status": "ok"}

    record = handoff._records["testuser"]
    assert record["mode"] == "recording"
    # The recording reads the viewer's relay: still one upstream connection, still open.
    [upstream] = fast_live.upstreams.opened
    assert not upstream.closed
    assert session_id not in ffmpeg_session._transcode_sessions
    assert viewer.signals == ["TERM"]
    _, recording = fast_live.launcher.launches
    command = recording.command
    assert command[command.index("-i") + 1] == "pipe:0"
    assert command[-1].startswith(str(handoff.recording_dir(record["recording_id"])))
    await recording.process.stdin.wait_for(len(testing.ts_source()))
    assert bytes(recording.process.stdin.written) == testing.ts_source()
    await ffmpeg_session.shutdown()
    assert recording.process.signals == ["TERM"]
    assert upstream.closed


@pytest.mark.usefixtures("guide_db")
@pytest.mark.asyncio
async def test_handoff_recording_without_a_session_reads_the_panels_ts_stream(
    auth_client: TestClient,
    clean_handoff: None,
    fast_live: _FastLive,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = cache_module.load_server_settings()
    settings["sources"] = [
        {
            "id": "src1",
            "name": "Upstream",
            "type": "xtream",
            "url": "http://upstream.test",
            "username": "user",
            "password": "pass",
        },
    ]
    cache_module.save_server_settings(settings)
    server_info = json.dumps({"user_info": {"allowed_output_formats": ["m3u8", "ts"]}})
    ts_url = "http://upstream.test/live/user/pass/42.ts"
    web = testing.FakeWeb(
        {
            SERVER_INFO_URL: testing.FakeResponse(server_info.encode()),
            ts_url: fast_live.upstreams,
        },
    )
    testing.install_web(monkeypatch, web, util)
    # An HLS URL would need a remuxer; none may run here.
    testing.install_subprocess(monkeypatch, ffmpeg_session, popen=testing.PopenRecorder())
    _set_live(_archive_stream(tv_archive=0))
    now = time.time()
    _watching("src1_42", now)
    report = main.HandoffReport(stream_id="src1_42", position=now - 5)

    stopping = asyncio.create_task(main.handoff_stop_api(report, {"sub": "testuser"}))
    await fast_live.feed(stopping, connection=0, data=testing.ts_source())

    assert await stopping == {"status": "ok"}
    assert handoff._records["testuser"]["mode"] == "recording"
    # The panel serves live TS, so the recording reads the channel as one TS stream.
    assert web.urls == [SERVER_INFO_URL, ts_url]
    [recording] = fast_live.launcher.launches
    await recording.process.stdin.wait_for(len(testing.ts_source()))
    await ffmpeg_session.shutdown()
    assert fast_live.upstreams.opened[0].closed


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
        _save_guide_filter(["1"])
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
        auth.set_user_limits("testuser", unavailable_groups=["cat:1"])
        payload = _body(
            auth_client.get("/api/guide/rows?cats=1&direct=true"),
            GuideRowsResponse,
        )
        assert payload["rows"] == []
        assert payload["direct_playback"] is True


def _archive_stream(**overrides: str | int | list[MutablePlainTree]) -> ProviderRecord:
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


def _now() -> datetime:
    """The current UTC time in whole seconds, which the guide stores exactly."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _save_guide_filter(cats: list[str]) -> None:
    settings = cache_module.load_user_settings("testuser")
    settings["guide_filter"] = cats
    cache_module.save_user_settings("testuser", settings)


def _set_live(*streams: ProviderRecord) -> None:
    store = cache_module.get_cache()
    store["live_streams"] = list(streams)
    store["live_categories"] = []
    store["epg_urls"] = []


def _air(
    channel_id: str,
    title: str,
    start: datetime,
    stop: datetime,
    *,
    desc: str = "",
    source_id: str = "",
) -> Program:
    epg.insert_programs([(channel_id, title, start.timestamp(), stop.timestamp(), desc, source_id)])
    epg.commit()
    return Program(channel_id=channel_id, title=title, start=start, stop=stop, desc=desc)


def _xmltv(channel_id: str, start: datetime, *, hours: int) -> bytes:
    stop = start + timedelta(hours=hours)
    stamp = "%Y%m%d%H%M%S +0000"
    return (
        f'<tv><channel id="{channel_id}"><display-name>News</display-name></channel>'
        f'<programme channel="{channel_id}" start="{start:{stamp}}" stop="{stop:{stamp}}">'
        "<title>Tonight</title></programme></tv>"
    ).encode()


def _add_session(
    session_id: str,
    directory: Path,
    *,
    username: str = "testuser",
    url: str = "http://provider/movie.mkv",
    last_access: float | None = None,
    is_archive: bool = False,
    caption_timestamp_origin: float | None = None,
) -> ffmpeg_session.TranscodeSession:
    """Register a running VOD transcode (a movie, or catchup when ``is_archive``)."""
    session: ffmpeg_session.TranscodeSession = {
        "dir": str(directory),
        "process": testing.FakeProcess(exit_code=None),
        "started": time.time(),
        "url": url,
        "username": username,
        "last_access": time.time() if last_access is None else last_access,
        "is_vod": True,
        "is_archive": is_archive,
    }
    if caption_timestamp_origin is not None:
        session["caption_timestamp_origin"] = caption_timestamp_origin
    ffmpeg_session._transcode_sessions[session_id] = session
    ffmpeg_session._url_to_session[url] = session_id
    return session


def _signals(session: ffmpeg_session.TranscodeSession) -> list[str]:
    process = session["process"]
    assert isinstance(process, testing.FakeProcess)
    return process.signals


def _assert_stopped(
    session_id: str,
    session: ffmpeg_session.TranscodeSession,
    *,
    stopped: bool,
) -> None:
    if stopped:
        assert session_id not in ffmpeg_session._transcode_sessions
        assert _signals(session) == ["TERM"]
    else:
        assert ffmpeg_session._transcode_sessions[session_id] is session
        assert _signals(session) == []


def _cast_lan(monkeypatch: pytest.MonkeyPatch, directory: Path) -> testing.FakeChromecast:
    """A TV at 192.168.1.50 and a playback session ``abc`` of testuser to cast."""
    receiver = testing.FakeChromecast()
    testing.install_chromecast(
        monkeypatch,
        testing.CastNetwork({"192.168.1.50": receiver}),
        chromecast,
    )
    testing.install_threads(monkeypatch, testing.ThreadRecorder(run_inline=False), chromecast)
    _add_session("abc", directory)
    return receiver


def _cast_owns(monkeypatch: pytest.MonkeyPatch, *, session_id: str, url: str) -> None:
    record = chromecast.CastSession(
        username="testuser",
        host="192.168.1.50",
        session_id=session_id,
        url=f"http://192.168.1.10:8000/transcode/{session_id}/stream.m3u8",
        source_url=url,
        name="TV",
        starting=False,
    )
    monkeypatch.setattr(chromecast.manager, "_sessions", {record.host: record})


def _watching(stream_id: str, now: float) -> None:
    handoff._watches["testuser"] = handoff._Watch(
        stream_id=stream_id,
        position=now - 10,
        first_seen=now - 600,
        last_seen=now - 15,
    )


async def _gone() -> Message:
    return {"type": "http.disconnect"}


async def _connected() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


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


class _EpgChannels(TypedDict):
    results: list[epg.ChannelMatch]


class _ProbeSeries(TypedDict):
    series_id: int


class _ProbeCache(TypedDict):
    probes: list[dict[str, str | float | list[str]]]
    series: list[_ProbeSeries]


class _HandoffBody(TypedDict):
    handoff: dict[str, MutablePlainTree] | None


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
