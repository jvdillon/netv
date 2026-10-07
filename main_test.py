"""Tests for main.py - FastAPI routes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import json
import re

import pytest

import cache as cache_module
import m3u as m3u_module


@pytest.fixture
def mock_deps():
    """Mock all external dependencies before importing main."""
    with (
        patch.dict(
            "sys.modules", {"defusedxml": MagicMock(), "defusedxml.ElementTree": MagicMock()}
        ),
        patch("cache.CACHE_DIR", Path("/tmp/test_cache")),
        patch("cache.SERVER_SETTINGS_FILE", Path("/tmp/test_cache/server_settings.json")),
        patch("cache.USERS_DIR", Path("/tmp/test_cache/users")),
    ):
        yield


@pytest.fixture
def client(tmp_path: Path, mock_deps):
    """Create test client with mocked dependencies."""
    from fastapi.testclient import TestClient

    # Patch paths before importing main
    with (
        patch("cache.CACHE_DIR", tmp_path),
        patch("cache.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("cache.USERS_DIR", tmp_path / "users"),
        patch("auth.CACHE_DIR", tmp_path),
        patch("auth.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("auth.USERS_DIR", tmp_path / "users"),
        patch("epg.init"),
        patch("ffmpeg_command.init"),
        patch("ffmpeg_session.cleanup_and_recover_sessions"),
    ):
        (tmp_path / "users").mkdir(exist_ok=True)
        import main

        # Disable background loading
        cache_module.get_cache().clear()
        yield TestClient(main.app)


@pytest.fixture
def auth_client(tmp_path: Path, mock_deps):
    """Create test client with a logged-in user."""
    from fastapi.testclient import TestClient

    with (
        patch("cache.CACHE_DIR", tmp_path),
        patch("cache.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("cache.USERS_DIR", tmp_path / "users"),
        patch("auth.CACHE_DIR", tmp_path),
        patch("auth.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("auth.USERS_DIR", tmp_path / "users"),
        patch("epg.init"),
        patch("ffmpeg_command.init"),
        patch("ffmpeg_session.cleanup_and_recover_sessions"),
    ):
        (tmp_path / "users").mkdir(exist_ok=True)
        import auth
        import main

        cache_module.get_cache().clear()
        client = TestClient(main.app)

        # Create user and get token
        auth.create_user("testuser", "testpass123")
        token = auth.create_token({"sub": "testuser"})
        client.cookies.set("token", token)

        yield client


def test_web_player_includes_resolution_badge(auth_client):
    from main import PlayerInfo

    info = PlayerInfo(url="https://example.test/live.m3u8", channel_name="Test channel")
    with patch("main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert 'id="quality-badge"' in response.text
    assert response.text.index('id="player-container"') < response.text.index('id="quality-badge"')
    assert response.text.index("/static/js/player-quality.js") < response.text.index(
        "/static/js/player.js"
    )
    assert response.text.index("/static/js/live-playback.js") < response.text.index(
        "/static/js/player.js"
    )
    script = auth_client.get("/static/js/player-quality.js")
    assert script.status_code == 200
    assert "videoWidth" in script.text
    assert 'id="cast-btn"' in response.text
    assert 'id="cast-dialog"' in response.text
    assert "cast_sender.js" not in response.text
    assert response.text.index("/static/js/cast.js") < response.text.index("/static/js/player.js")
    from html.parser import HTMLParser

    class CastPlacement(HTMLParser):
        def __init__(self):
            super().__init__()
            self.divs = []
            self.parents = {}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if attrs.get("id") in ("cast-overlay", "cast-dialog"):
                self.parents[attrs["id"]] = self.divs[-1].get("id")
            if tag == "div":
                self.divs.append(attrs)

        def handle_endtag(self, tag):
            if tag == "div":
                self.divs.pop()

    placement = CastPlacement()
    placement.feed(response.text)
    assert placement.parents == {"cast-overlay": "player-container", "cast-dialog": "player-container"}


def test_live_player_records_view(auth_client):
    from main import PlayerInfo

    info = PlayerInfo(
        url="https://example.test/live.m3u8",
        channel_name="Test",
        source_id="source-a",
    )
    with patch("main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/7")

    assert response.status_code == 200
    assert cache_module.load_live_view_counts("testuser") == {"source-a:7": 1}


@pytest.mark.parametrize("path", ["devices", "status"])
def test_cast_requires_authentication(client, path):
    assert client.get(f"/api/cast/{path}").status_code == 401


@pytest.mark.parametrize("path", ["start", "control"])
def test_cast_commands_require_authentication(client, path):
    assert client.post(f"/api/cast/{path}", json={}).status_code == 401


def test_cast_api_uses_http_and_remembers_lan_address(auth_client):
    with patch("casting.manager.start", return_value={"active": True}) as start:
        response = auth_client.post("/api/cast/start", json={
            "host": "192.168.1.50", "session_id": "abc", "server_url": "http://192.168.1.10:9000/",
            "title": "News", "current_time": 15,
        })
        assert response.status_code == 200
        start.assert_called_once_with("testuser", "192.168.1.50", "abc", "http://192.168.1.10:9000", "News", 15)
        assert cache_module.load_user_settings("testuser")["cast_host"] == "http://192.168.1.10:9000"


def test_cast_api_uses_request_origin_and_supports_legacy_host_preference(auth_client):
    with patch("casting.manager.start", return_value={"active": True}) as start:
        response = auth_client.post("/api/cast/start", json={"host": "192.168.1.50", "session_id": "abc"})
        assert response.status_code == 200
        assert start.call_args.args[3] == "http://testserver"
        cache_module.save_user_settings("testuser", {"cast_host": "192.168.1.10:8000"})
        response = auth_client.post("/api/cast/start", json={"host": "192.168.1.50", "session_id": "abc"})
        assert response.status_code == 200
        assert start.call_args.args[3] == "http://192.168.1.10:8000"


def test_cast_rejects_localhost_media_address(auth_client):
    with patch("casting.manager.start") as start:
        response = auth_client.post("/api/cast/start", json={
            "host": "192.168.1.50", "session_id": "abc", "server_url": "http://localhost:8000",
        })
        assert response.status_code == 400
        start.assert_not_called()


def test_cast_command_validation_and_user_scope(auth_client):
    with patch("casting.manager.command", return_value={"active": True}) as command:
        assert auth_client.post("/api/cast/control", json={"action": "volume", "volume": 2}).status_code == 422
        assert auth_client.post("/api/cast/control", json={"action": "reboot"}).status_code == 422
        command.assert_not_called()
        assert auth_client.post("/api/cast/control", json={"action": "pause"}).status_code == 200
        command.assert_called_once_with("testuser", "pause", None)


def test_cast_discovery_reports_network_errors(auth_client):
    with patch("casting.manager.discover", side_effect=OSError("no network")):
        response = auth_client.get("/api/cast/devices")
    assert response.status_code == 503
    assert "manually" in response.json()["detail"]


def test_browser_cannot_stop_or_seek_cast_owned_session(auth_client):
    with patch("casting.manager.owns_session", return_value=True), patch("ffmpeg_session.stop_session") as stop:
        response = auth_client.post("/transcode/abc/stop?force=true")
        assert response.json() == {"status": "casting"}
        assert auth_client.delete("/transcode/abc?force=true").json() == {"status": "casting"}
        assert auth_client.get("/transcode/seek/abc?time=10").status_code == 409
        stop.assert_not_called()


def test_cast_source_cannot_be_restarted_while_receiver_owns_it(auth_client):
    with patch("casting.manager.owns_url", return_value=True), patch("ffmpeg_session.clear_url_session") as clear:
        assert auth_client.delete("/transcode-clear?url=http://provider/live").status_code == 409
        assert auth_client.get("/transcode/start?url=http://provider/live").status_code == 409
        clear.assert_not_called()


def test_receiver_hls_request_keeps_session_alive_without_cookie(client, tmp_path):
    (tmp_path / "stream.m3u8").write_text("#EXTM3U\n#EXTINF:6,\nseg001.ts\n")
    with patch("ffmpeg_session.get_session", return_value={"dir": str(tmp_path)}), patch("ffmpeg_session.touch_session") as touch:
        response = client.get("/transcode/abc/stream.m3u8")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    touch.assert_called_once_with("abc")


def test_live_dvr_routes_auto_mode_through_server(auth_client):
    from main import PlayerInfo

    info = PlayerInfo(url="https://example.test/live.ts", channel_name="Test channel")
    with patch("main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert 'transcodeMode: "always"' in response.text
    assert 'liveDvrMins: 120' in response.text


def test_live_dvr_disabled_keeps_direct_auto_mode(auth_client):
    from main import PlayerInfo

    settings = cache_module.load_server_settings()
    settings["live_dvr_mins"] = 0
    cache_module.save_server_settings(settings)
    info = PlayerInfo(url="https://example.test/live.ts", channel_name="Test channel")
    with patch("main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert 'transcodeMode: "auto"' in response.text
    assert 'liveDvrMins: 0' in response.text


def test_live_player_passes_program_window_from_epg(auth_client):
    from main import PlayerInfo

    info = PlayerInfo(
        url="https://example.test/live.ts",
        channel_name="Test channel",
        program_title="News",
        program_start=1749998200.0,
        program_end=1750000000.0,
    )
    with patch("main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/test")
    assert response.status_code == 200
    assert "programStart: 1749998200.0" in response.text
    assert "programEnd: 1750000000.0" in response.text
    assert 'streamId: "test"' in response.text
    assert 'id="program-remaining"' not in response.text


def test_live_player_info_reads_program_window_from_epg():
    from datetime import UTC, datetime, timedelta

    from epg import Program

    import main

    start = datetime.now(UTC) - timedelta(minutes=30)
    stop = datetime.now(UTC) + timedelta(minutes=30)
    program = Program("ch1", "News", start, stop)
    stream = {
        "stream_id": "1",
        "name": "Channel",
        "direct_url": "https://example.test/live.ts",
        "epg_channel_id": "ch1",
    }
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": [stream]}),
        patch("epg.get_programs_in_range", return_value=[program]),
    ):
        info = main._get_live_player_info("1")
    assert info.program_title == "News"
    assert info.program_start == start.timestamp()
    assert info.program_end == stop.timestamp()


def test_live_program_api_returns_current_program(auth_client):
    from datetime import UTC, datetime, timedelta

    from epg import Program

    start = datetime.now(UTC) - timedelta(minutes=10)
    stop = start + timedelta(minutes=45)
    program = Program("ch1", "News", start, stop, desc="Tonight's headlines")
    stream = {"stream_id": "1", "name": "Channel", "epg_channel_id": "ch1"}
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": [stream]}),
        patch("epg.get_programs_in_range", return_value=[program]),
    ):
        response = auth_client.get("/api/live/program/1")
    assert response.status_code == 200
    assert response.json() == {
        "title": "News",
        "desc": "Tonight's headlines",
        "start": start.timestamp(),
        "end": stop.timestamp(),
    }


def test_live_program_api_without_guide_data_returns_empty_program(auth_client):
    stream = {"stream_id": "1", "name": "Channel"}
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": [stream]}),
    ):
        response = auth_client.get("/api/live/program/1")
    assert response.status_code == 200
    assert response.json() == {"title": "", "desc": "", "start": 0.0, "end": 0.0}


def test_live_program_api_unknown_stream_is_not_found(auth_client):
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": []}),
    ):
        response = auth_client.get("/api/live/program/missing")
    assert response.status_code == 404


def _archive_stream(**overrides):
    stream = {
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


def test_catchup_player_info_builds_timeshift_url():
    from zoneinfo import ZoneInfo

    from epg import Program

    import main

    start = (datetime.now(UTC) - timedelta(hours=3)).replace(second=0, microsecond=0)
    program = Program("ch1", "Earlier news", start, start + timedelta(minutes=30))
    client = MagicMock()
    client.get_server_info.return_value = {"server_info": {"timezone": "Europe/Amsterdam"}}
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": [_archive_stream()]}),
        patch("main.get_xtream_client_by_source", return_value=client),
        patch("epg.get_programs_in_range", return_value=[program]),
        patch("catchup._tz_cache", {}),
    ):
        info = main._get_catchup_player_info("src1_42", start.timestamp())
    local = start.astimezone(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d:%H-%M")
    assert info.url == f"http://upstream.test/timeshift/user/pass/30/{local}/42.ts"
    assert info.catchup_start == start.timestamp()
    assert info.catchup_days == 2
    assert info.program_title == "Earlier news"
    assert info.program_end == program.stop.timestamp()


def test_catchup_player_info_starts_mid_program():
    from epg import Program

    import main

    program_start = (datetime.now(UTC) - timedelta(hours=3)).replace(second=0, microsecond=0)
    program = Program("ch1", "Earlier news", program_start, program_start + timedelta(minutes=30))
    target = program_start + timedelta(minutes=10, seconds=25)
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": [_archive_stream()]}),
        patch("main.get_xtream_client_by_source", return_value=None),
        patch("epg.get_programs_in_range", return_value=[program]),
    ):
        info = main._get_catchup_player_info("src1_42", target.timestamp())
    begin = program_start + timedelta(minutes=10)
    assert f"/20/{begin:%Y-%m-%d:%H-%M}/42.ts" in info.url
    assert info.catchup_start == begin.timestamp()
    assert info.catchup_seek == 25
    assert info.program_start == program_start.timestamp()
    assert info.program_end == program.stop.timestamp()


def test_catchup_player_info_rejects_programs_outside_archive():
    from fastapi import HTTPException

    import main

    too_old = datetime.now(UTC) - timedelta(days=3)
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": [_archive_stream()]}),
        patch("epg.get_programs_in_range", return_value=[]),
        pytest.raises(HTTPException) as excinfo,
    ):
        main._get_catchup_player_info("src1_42", too_old.timestamp())
    assert excinfo.value.status_code == 404


def test_catchup_player_page_plays_archive_as_vod(auth_client):
    from main import PlayerInfo

    info = PlayerInfo(
        url="http://upstream.test/timeshift/user/pass/30/2026-01-01:10-00/42.ts",
        channel_name="Channel",
        catchup_days=2,
        catchup_start=1767261600.0,
        catchup_seek=25.0,
    )
    with patch("main._get_catchup_player_info", return_value=info) as get_info:
        response = auth_client.get("/play/live/src1_42?start=1767261600")
    get_info.assert_called_once_with("src1_42", 1767261600.0)
    assert response.status_code == 200
    assert "catchup: true" in response.text
    assert "catchupStart: 1767261600.0" in response.text
    assert "catchupSeek: 25.0" in response.text
    assert 'id="go-live-btn"' in response.text
    assert 'id="jump-btn"' in response.text
    assert 'id="start-over-btn"' not in response.text


def test_live_player_offers_start_over_with_archive(auth_client):
    from main import PlayerInfo

    info = PlayerInfo(url="https://example.test/live.ts", channel_name="Channel", catchup_days=1)
    with patch("main._get_live_player_info", return_value=info):
        response = auth_client.get("/play/live/test")
    assert "catchup: false" in response.text
    assert "catchupDays: 1" in response.text
    assert 'id="start-over-btn"' in response.text


def test_catchup_api_lists_archived_programs_newest_first(auth_client):
    from epg import Program

    now = datetime.now(UTC)
    programs = [
        Program("ch1", "Too old", now - timedelta(days=2, hours=1), now - timedelta(days=2) + timedelta(minutes=1)),
        Program("ch1", "Morning", now - timedelta(hours=5), now - timedelta(hours=4)),
        Program("ch1", "On now", now - timedelta(minutes=10), now + timedelta(minutes=20)),
    ]
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": [_archive_stream()]}),
        patch("epg.get_programs_in_range", return_value=programs),
    ):
        response = auth_client.get("/api/live/catchup/src1_42")
    assert response.status_code == 200
    payload = response.json()
    assert payload["days"] == 2
    assert [p["title"] for p in payload["programs"]] == ["On now", "Morning"]
    assert payload["programs"][1]["start_timestamp"] == programs[1].start.timestamp()


def test_catchup_api_without_archive_is_empty(auth_client):
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": [_archive_stream(tv_archive=0)]}),
    ):
        response = auth_client.get("/api/live/catchup/src1_42")
    assert response.json() == {"days": 0, "programs": []}


def test_catchup_api_uses_playlist_guide_assignment(auth_client):
    from epg import Program

    now = datetime.now(UTC)
    earlier = Program("assigned", "Morning", now - timedelta(hours=5), now - timedelta(hours=4))
    stream = _archive_stream(epg_channel_id="")
    playlist = {
        "id": "pl1",
        "channels": [
            {"stream_id": "src1_42", "source_id": "src1", "name": "Channel", "epg_channel_id": "assigned"}
        ],
    }
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": [stream]}),
        patch("main.playlists.load", return_value=[playlist]),
        patch("epg.get_programs_in_range", return_value=[earlier]) as get_programs,
    ):
        response = auth_client.get("/api/live/catchup/src1_42")
    assert [p["title"] for p in response.json()["programs"]] == ["Morning"]
    assert get_programs.call_args.args[0] == "assigned"


def test_catchup_player_page_rejects_invalid_start(auth_client):
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": [_archive_stream()]}),
        patch("epg.get_programs_in_range", return_value=[]),
    ):
        assert auth_client.get("/play/live/src1_42?start=1e300").status_code == 404


def test_catchup_api_unknown_stream_is_not_found(auth_client):
    with (
        patch("main._ensure_live_cache"),
        patch("main.get_cache", return_value={"live_streams": []}),
    ):
        assert auth_client.get("/api/live/catchup/missing").status_code == 404


def test_adaptive_start_uses_shared_backend(auth_client):
    result = {
        "session_id": "shared",
        "playlist": "/transcode/shared/low.m3u8",
        "master_playlist": "/transcode/shared/master.m3u8",
    }
    with patch("ffmpeg_session.start_transcode", new=AsyncMock(return_value=result)) as start:
        response = auth_client.get(
            "/transcode/start",
            params={"url": "https://provider.example/live.m3u8", "fast_start": "true"},
        )
    assert response.status_code == 200
    assert response.json() == result
    assert start.call_args.kwargs["fast_start"] is True


def test_disconnected_adaptive_start_is_cleaned_up(auth_client):
    with (
        patch("ffmpeg_session.start_transcode", new=AsyncMock(return_value={"session_id": "new"})),
        patch("starlette.requests.Request.is_disconnected", new=AsyncMock(return_value=True)),
        patch("ffmpeg_session.stop_session") as stop,
    ):
        response = auth_client.get(
            "/transcode/start",
            params={"url": "https://provider.example/live.m3u8", "fast_start": "true"},
        )
    assert response.status_code == 499
    stop.assert_called_once_with("new", force=True)


@pytest.mark.parametrize("owner,status", [("testuser", 200), ("other", 404)])
def test_live_page_close_force_stop_checks_owner(auth_client, owner, status):
    with (
        patch("ffmpeg_session.get_session", return_value={"username": owner}),
        patch("ffmpeg_session.stop_session") as stop,
    ):
        response = auth_client.post("/transcode/live/stop?force=true")
    assert response.status_code == status
    if status == 200:
        stop.assert_called_once_with("live", force=True)
    else:
        stop.assert_not_called()


@pytest.mark.parametrize("owner,status", [("testuser", 200), ("other", 404)])
def test_archive_stop_without_force_still_releases_only_owner_session(auth_client, owner, status):
    with (
        patch("ffmpeg_session.get_session", return_value={"username": owner, "is_archive": True}),
        patch("ffmpeg_session.stop_session") as stop,
    ):
        response = auth_client.post("/transcode/archive/stop")
    assert response.status_code == status
    if status == 200:
        stop.assert_called_once_with("archive", force=True)
    else:
        stop.assert_not_called()


class TestLoadAllEpg:
    def test_refetches_when_listings_ran_out(self, mock_deps):
        import main

        urls = [("http://epg", 30, "s")]
        with (
            patch("main.epg.has_programs_after", return_value=False),
            patch("main._fetch_all_epg") as fetch,
        ):
            main.load_all_epg(urls)
        fetch.assert_called_once_with(urls)

    def test_skips_fetch_when_listings_are_current(self, mock_deps):
        import main

        with (
            patch("main.epg.has_programs_after", return_value=True),
            patch("main.epg.get_program_count", return_value=5),
            patch("main._fetch_all_epg") as fetch,
        ):
            main.load_all_epg([("http://epg", 30, "s")])
        fetch.assert_not_called()


class TestSetup:
    """Tests for initial setup flow."""

    def test_setup_page_shown_when_no_users(self, client):
        resp = client.get("/setup", follow_redirects=False)
        assert resp.status_code == 200
        assert b"setup" in resp.content.lower() or b"Create" in resp.content

    def test_setup_redirects_when_users_exist(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/setup", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_setup_creates_user(self, client):
        resp = client.post(
            "/setup",
            data={"username": "admin", "password": "password123", "confirm": "password123"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

        import auth

        assert auth.verify_password("admin", "password123")

    def test_setup_validates_username_length(self, client):
        resp = client.post(
            "/setup",
            data={"username": "ab", "password": "password123", "confirm": "password123"},
        )
        assert resp.status_code == 200
        assert b"at least 3" in resp.content

    def test_setup_validates_password_length(self, client):
        resp = client.post(
            "/setup",
            data={"username": "admin", "password": "short", "confirm": "short"},
        )
        assert resp.status_code == 200
        assert b"at least 8" in resp.content

    def test_setup_validates_password_match(self, client):
        resp = client.post(
            "/setup",
            data={"username": "admin", "password": "password123", "confirm": "different"},
        )
        assert resp.status_code == 200
        assert b"do not match" in resp.content


class TestLogin:
    """Tests for login flow."""

    def test_login_page_redirects_to_setup_when_no_users(self, client):
        resp = client.get("/login", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/setup"

    def test_login_page_shown_when_users_exist(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/login")
        assert resp.status_code == 200

    def test_login_success_sets_cookie(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.post(
            "/login",
            data={"username": "admin", "password": "password123"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "token" in resp.cookies

    def test_login_failure_returns_401(self, client, tmp_path):
        import auth

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

    def test_logout_clears_cookie(self, auth_client):
        resp = auth_client.get("/logout", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"


class TestAuthRequired:
    """Tests for auth-protected routes."""

    def test_index_redirects_to_login(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_guide_redirects_to_login(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/guide", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_vod_redirects_to_login(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/vod", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_series_redirects_to_login(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/series", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"


class TestIndex:
    """Tests for index route."""

    def test_index_redirects_to_guide(self, auth_client):
        resp = auth_client.get("/", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/guide"


class TestFavicon:
    """Tests for favicon."""

    def test_favicon_returns_204(self, client):
        resp = client.get("/favicon.ico")
        assert resp.status_code == 204


class TestGuide:
    """Tests for guide page."""

    def test_guide_shows_loading_when_no_cache(self, auth_client):
        with patch("main.load_file_cache", return_value=None):
            resp = auth_client.get("/guide")
            assert resp.status_code == 200
            # Should show loading state
            assert b"loading" in resp.content.lower() or b"Loading" in resp.content

    def test_guide_shows_channels_from_cache(self, auth_client):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"}
        ]
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN", "category_ids": ["1"], "epg_channel_id": ""}
        ]

        with patch("main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide?cats=1")
            assert resp.status_code == 200

    def test_guide_limits_initial_virtual_rows(self, auth_client):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"}
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

        with patch("main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide?cats=1")

        assert resp.status_code == 200
        assert resp.text.count('class="guide-row') == 50
        assert resp.text.count('loading="lazy"') == 50

    def test_guide_rows_include_category_metadata(self, auth_client):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"}
        ]
        cache_module.get_cache()["live_streams"] = [
            {
                "stream_id": 1,
                "name": "CNN",
                "category_ids": ["1"],
                "epg_channel_id": "",
            }
        ]

        resp = auth_client.get("/api/guide/rows?cats=1")

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["categories"] == [{"category_id": "1", "category_name": "News"}]
        assert payload["rows"][0]["channel"]["category_ids"] == ["1"]

    def test_guide_rows_categories_cover_all_pages(self, auth_client):
        categories = [
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

        with patch("main.auth.get_user_limits", return_value={"unavailable_groups": ["cat:3"]}):
            first = auth_client.get("/api/guide/rows?cats=1,2,3&count=500").json()
            second = auth_client.get("/api/guide/rows?cats=1,2,3&start=500&count=500").json()
            reordered = auth_client.get("/api/guide/rows?cats=2,1,3&count=1").json()

        assert first["total"] == second["total"] == 501
        assert len(first["rows"]) == 500
        assert len(second["rows"]) == 1
        assert second["rows"][0]["channel"]["stream_id"] == 500
        assert first["categories"] == second["categories"] == categories[:2]
        assert reordered["categories"] == categories[1::-1]
        assert reordered["rows"][0]["channel"]["stream_id"] == 500

    def test_guide_rows_timestamps_anchor_local_timeline(self, auth_client):
        from epg import Program

        now = datetime(2026, 9, 14, 0, 15, tzinfo=UTC)
        window_start = now.replace(minute=0)
        program_start = window_start - timedelta(minutes=30)
        program_end = window_start + timedelta(hours=1)
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "News", "category_ids": ["1"], "epg_channel_id": "news"}
        ]
        program = Program("news", "Overnight news", program_start, program_end)

        with (
            patch("main.datetime") as clock,
            patch("main.epg.get_icons_batch", return_value={}),
            patch("main.epg.get_programs_batch", return_value={"news": [program]}),
        ):
            clock.now.return_value = now
            response = auth_client.get("/api/guide/rows?cats=1")

        assert response.status_code == 200
        payload = response.json()
        listing = payload["rows"][0]["programs"][0]
        assert payload["window_start_timestamp"] == window_start.timestamp()
        assert listing["start_timestamp"] == program_start.timestamp()
        assert listing["end_timestamp"] == program_end.timestamp()
        assert listing["start"] == "23:30"
        assert listing["end"] == "01:00"
        assert listing["left_pct"] == 0
        assert listing["width_pct"] == pytest.approx(100 / 3)

    def test_guide_rows_flag_archived_programs(self, auth_client):
        from epg import Program

        now = datetime.now(UTC)
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
        past = Program("news", "Earlier", now - timedelta(hours=1), now - timedelta(minutes=5))
        current = Program("news", "Now", now - timedelta(minutes=5), now + timedelta(hours=1))
        other = Program("other", "Earlier", past.start, past.stop)

        with (
            patch("main.epg.get_icons_batch", return_value={}),
            patch(
                "main.epg.get_programs_batch",
                return_value={"news": [past, current], "other": [other]},
            ),
        ):
            rows = auth_client.get("/api/guide/rows?cats=1&offset=-1").json()["rows"]

        assert rows[0]["channel"]["catchup_days"] == 1
        assert [p["catchup"] for p in rows[0]["programs"]] == [True, False]
        assert [p["unavailable"] for p in rows[0]["programs"]] == [False, False]
        assert rows[1]["channel"]["catchup_days"] == 0
        assert [p["catchup"] for p in rows[1]["programs"]] == [False]
        assert rows[1]["programs"][0]["unavailable"] is True
        assert rows[1]["programs_mobile"][0]["unavailable"] is True
        mobile = {p["title"]: p for p in rows[0]["programs_mobile"]}
        assert mobile["Earlier"]["catchup"] is True
        assert mobile["Earlier"]["start_timestamp"] == past.start.timestamp()
        assert mobile["Earlier"]["end_timestamp"] == past.stop.timestamp()

    @pytest.mark.parametrize("archive_days", [0, 1, 3])
    def test_guide_disables_unavailable_past_listings(self, auth_client, archive_days):
        from html.parser import HTMLParser

        from epg import Program

        class ProgramLinks(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and "data-start" in attrs:
                    self.links.append(attrs)

        now = datetime.now(UTC)
        start = now - timedelta(days=2)
        cache_module.get_cache()["live_categories"] = [{"category_id": "1", "category_name": "Group"}]
        cache_module.get_cache()["live_streams"] = [{
            "stream_id": 1, "name": "Stream", "category_ids": ["1"], "epg_channel_id": "guide",
            "source_type": "xtream", "tv_archive": 1, "tv_archive_duration": archive_days,
        }]
        program = Program("guide", "Earlier program", start, start + timedelta(hours=1))
        with (
            patch("main.epg.has_programs", return_value=True),
            patch("main.epg.get_icons_batch", return_value={}),
            patch("main.epg.get_programs_batch", return_value={"guide": [program]}),
        ):
            response = auth_client.get("/guide?cats=1&offset=-48")
            rows = auth_client.get("/api/guide/rows?cats=1&offset=-48").json()["rows"]
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
        for programs in ("programs", "programs_mobile"):
            assert rows[0][programs][0]["unavailable"] is (archive_days != 3)

    def test_guide_page_exposes_timestamps_for_local_times(self, auth_client):
        from epg import Program

        now = datetime.now(UTC)
        cache_module.get_cache()["live_categories"] = [{"category_id": "1", "category_name": "News"}]
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "News", "category_ids": ["1"], "epg_channel_id": "news"}
        ]
        program = Program("news", "Headlines", now - timedelta(minutes=5), now + timedelta(minutes=55))
        with (
            patch("main.epg.has_programs", return_value=True),
            patch("main.epg.get_icons_batch", return_value={}),
            patch("main.epg.get_programs_batch", return_value={"news": [program]}),
        ):
            html = auth_client.get("/guide?cats=1").text
        window_start = now.replace(minute=0, second=0, microsecond=0)
        assert f'data-clock="{window_start.timestamp()}"' in html
        assert f'data-start="{program.start.timestamp()}"' in html
        assert f'data-end="{program.stop.timestamp()}"' in html
        assert "localizeGuideTimes(document)" in html

    def test_guide_uses_saved_filter(self, auth_client, tmp_path):
        user_dir = tmp_path / "users" / "testuser"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "settings.json").write_text(json.dumps({"guide_filter": ["1", "2"]}))

        cache_module.get_cache()["live_categories"] = []
        cache_module.get_cache()["live_streams"] = []

        # Guide now renders directly using saved filter (no redirect)
        with patch("main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide")
            assert resp.status_code == 200


class TestVod:
    """Tests for VOD page."""

    def test_vod_shows_loading_when_no_cache(self, auth_client):
        with patch("main.load_file_cache", return_value=None):
            resp = auth_client.get("/vod")
            assert resp.status_code == 200

    def test_vod_shows_movies_from_cache(self, auth_client):
        cache_module.get_cache()["vod_categories"] = [
            {"category_id": "10", "category_name": "Movies", "source_id": "src1"}
        ]
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 100, "name": "Movie 1", "category_id": "10", "source_id": "src1"}
        ]

        resp = auth_client.get("/vod")
        assert resp.status_code == 200

    def test_vod_filters_by_category(self, auth_client):
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

    def test_vod_sorts_by_alpha(self, auth_client):
        cache_module.get_cache()["vod_categories"] = []
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 1, "name": "Zebra", "source_id": "src1"},
            {"stream_id": 2, "name": "Apple", "source_id": "src1"},
        ]

        resp = auth_client.get("/vod?sort=alpha")
        assert resp.status_code == 200


class TestSeries:
    """Tests for series page."""

    def test_series_shows_loading_when_no_cache(self, auth_client):
        with patch("main.load_file_cache", return_value=None):
            resp = auth_client.get("/series")
            assert resp.status_code == 200

    def test_series_shows_list_from_cache(self, auth_client):
        cache_module.get_cache()["series_categories"] = [
            {"category_id": "30", "category_name": "Drama", "source_id": "src1"}
        ]
        cache_module.get_cache()["series"] = [
            {"series_id": 200, "name": "Show 1", "category_id": "30", "source_id": "src1"}
        ]

        resp = auth_client.get("/series")
        assert resp.status_code == 200


class TestSearch:
    """Tests for search page."""

    def test_search_page_renders(self, auth_client):
        cache_module.get_cache()["live_streams"] = []
        cache_module.get_cache()["vod_streams"] = []
        cache_module.get_cache()["series"] = []

        resp = auth_client.get("/search")
        assert resp.status_code == 200

    def test_search_finds_live_streams(self, auth_client):
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

    def test_search_regex_mode(self, auth_client):
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

    def test_search_rejects_long_regex(self, auth_client):
        cache_module.get_cache()["live_streams"] = []

        resp = auth_client.get(f"/search?q={'a' * 101}&regex=true&live=true")
        assert resp.status_code == 400


class TestSettings:
    """Tests for settings page."""

    def test_only_nomos_engine_is_discovered(self, auth_client, tmp_path: Path):
        import main

        (tmp_path / "retired-model_720p_fp16.engine").write_bytes(b"engine")
        with patch("main.SR_ENGINE_DIR", tmp_path):
            assert main.get_sr_models() == []

            (tmp_path / "2x-nomosuni-compact_720p_fp16.engine").write_bytes(b"engine")
            assert main.get_sr_models() == ["2x-nomosuni-compact"]

    def test_settings_page_renders(self, auth_client):
        cache_module.get_cache()["live_categories"] = []

        with patch("main.load_file_cache", return_value=None):
            resp = auth_client.get("/settings")
            assert resp.status_code == 200
            assert 'id="sort-live-by-views" checked' in resp.text

    def test_settings_page_maps_enabled_retired_model_to_nomos(self, auth_client):
        cache_module.get_cache()["live_categories"] = []

        with (
            patch("main.load_file_cache", return_value=None),
            patch(
                "main.load_server_settings",
                return_value={"sr_model": "retired-model"},
            ),
            patch(
                "main.get_sr_models",
                return_value=["2x-nomosuni-compact"],
            ),
        ):
            resp = auth_client.get("/settings")

        assert resp.status_code == 200
        assert resp.text.count('name="sr_model"') == 2
        assert 'value="2x-nomosuni-compact" checked' in resp.text
        assert "NomosUni 2x (480p/720p/1080p)" in resp.text

    def test_settings_guide_filter(self, auth_client):
        resp = auth_client.post(
            "/settings/guide-filter",
            json={"cats": ["1", "2", "3"]},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_settings_captions(self, auth_client):
        resp = auth_client.post(
            "/settings/captions",
            data={"enabled": "on"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_settings_transcode(self, auth_client):
        resp = auth_client.post(
            "/settings/transcode",
            data={
                "transcode_mode": "auto",
                "transcode_hw": "nvidia",
                "vod_transcode_cache_mins": 60,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_settings_transcode_caps_live_dvr_at_two_hours(self, auth_client):
        resp = auth_client.post(
            "/settings/transcode",
            data={
                "transcode_mode": "auto",
                "transcode_hw": "nvidia",
                "live_dvr_mins": 999,
            },
        )
        assert resp.status_code == 200
        assert cache_module.load_server_settings()["live_dvr_mins"] == 120


class TestAddSource:
    """Tests for adding sources."""

    def test_add_xtream_source(self, auth_client):
        with patch("main.clear_all_caches"):
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

    def test_add_m3u_source(self, auth_client):
        with patch("main.clear_all_caches"):
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

    def test_add_source_validates_type(self, auth_client):
        resp = auth_client.post(
            "/settings/add",
            data={
                "name": "Bad Source",
                "source_type": "invalid",
                "url": "http://example.com",
            },
        )
        assert resp.status_code == 400

    def test_add_source_validates_url_scheme(self, auth_client):
        resp = auth_client.post(
            "/settings/add",
            data={
                "name": "Bad Source",
                "source_type": "xtream",
                "url": "ftp://example.com",
            },
        )
        assert resp.status_code == 400

    def test_add_source_validates_name_length(self, auth_client):
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

    def test_delete_source(self, auth_client, tmp_path):
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
                        }
                    ]
                }
            )
        )

        with patch("main.clear_all_caches"):
            resp = auth_client.post("/settings/delete/src_123", follow_redirects=False)
            assert resp.status_code == 303


class TestUserPrefs:
    """Tests for user preferences API."""

    def test_get_user_prefs(self, auth_client):
        resp = auth_client.get("/api/user-prefs")
        assert resp.status_code == 200
        data = resp.json()
        assert "favorites" in data
        assert "cc_lang" in data
        assert data["sort_live_by_views"] is True

    def test_save_user_prefs(self, auth_client):
        resp = auth_client.post(
            "/api/user-prefs",
            json={
                "cc_lang": "eng",
                "cast_host": "192.168.1.100",
                "sort_live_by_views": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert auth_client.get("/api/user-prefs").json()["sort_live_by_views"] is False

    def test_view_sorting_preference_requires_boolean(self, auth_client):
        resp = auth_client.post(
            "/api/user-prefs",
            json={"sort_live_by_views": "false"},
        )

        assert resp.status_code == 400


class TestWatchPosition:
    """Tests for watch position API."""

    def test_save_watch_position(self, auth_client):
        resp = auth_client.post(
            "/api/watch-position",
            json={"url": "http://example.com/movie.mkv", "position": 1234.5, "duration": 7200},
        )
        assert resp.status_code == 200

    def test_get_watch_position(self, auth_client):
        # Save first
        auth_client.post(
            "/api/watch-position",
            json={"url": "http://example.com/movie.mkv", "position": 1234.5, "duration": 7200},
        )

        resp = auth_client.get("/api/watch-position?url=http://example.com/movie.mkv")
        assert resp.status_code == 200
        data = resp.json()
        assert data["position"] == 1234.5
        assert data["duration"] == 7200

    def test_get_watch_position_not_found(self, auth_client):
        resp = auth_client.get("/api/watch-position?url=http://example.com/unknown.mkv")
        assert resp.status_code == 200
        data = resp.json()
        assert data["position"] == 0


class TestUserManagement:
    """Tests for user management endpoints."""

    def test_delete_user(self, auth_client, tmp_path):
        import auth

        auth.create_user("otheruser", "password123")

        resp = auth_client.post("/settings/users/delete/otheruser", follow_redirects=False)
        assert resp.status_code == 303

    def test_cannot_delete_self(self, auth_client):
        resp = auth_client.post("/settings/users/delete/testuser")
        assert resp.status_code == 400

    def test_change_password(self, auth_client):
        resp = auth_client.post(
            "/settings/users/password",
            data={"current_password": "testpass123", "new_password": "newpass456"},
        )
        assert resp.status_code == 200

    def test_change_password_wrong_current(self, auth_client):
        resp = auth_client.post(
            "/settings/users/password",
            data={"current_password": "wrongpass", "new_password": "newpass456"},
        )
        assert resp.status_code == 400


class TestPlaylistXspf:
    """Tests for XSPF playlist generation."""

    def test_playlist_xspf(self, auth_client):
        resp = auth_client.get("/playlist.xspf?url=http://example.com/stream.m3u8")
        assert resp.status_code == 200
        assert b"<?xml" in resp.content
        assert b"http://example.com/stream.m3u8" in resp.content
        assert resp.headers["content-type"] == "application/xspf+xml"


class TestApiSettings:
    """Tests for settings API."""

    def test_get_settings(self, auth_client):
        resp = auth_client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "transcode_mode" in data

    def test_update_settings(self, auth_client):
        resp = auth_client.post(
            "/api/settings",
            json={"transcode_mode": "always"},
        )
        assert resp.status_code == 200


class TestTranscodeRoutes:
    """Tests for transcode routes (with mocked transcoding module)."""

    def test_transcode_file_not_found(self, auth_client):
        with patch("main.ffmpeg_session.get_session", return_value=None):
            resp = auth_client.get("/transcode/invalid-session/stream.m3u8")
            assert resp.status_code == 404

    def test_transcode_stop(self, auth_client):
        with patch("main.ffmpeg_session.stop_session"):
            resp = auth_client.delete("/transcode/test-session")
            assert resp.status_code == 200
            assert resp.json()["status"] == "stopped"

    def test_transcode_stop_post(self, auth_client):
        with patch("main.ffmpeg_session.stop_session"):
            resp = auth_client.post("/transcode/test-session/stop")
            assert resp.status_code == 200

    def test_transcode_progress_not_found(self, auth_client):
        with patch("main.ffmpeg_session.get_session_progress", return_value=None):
            resp = auth_client.get("/transcode/progress/invalid-session")
            assert resp.status_code == 404

    @pytest.mark.parametrize("filename", ["master.m3u8", "low.m3u8", "high.m3u8"])
    def test_adaptive_playlists_share_dates_but_master_is_not_rewritten(
        self, auth_client, tmp_path, filename
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
            patch("ffmpeg_session.get_session", return_value=session),
            patch("main.dated_playlist", return_value=content + "#dated\n") as dates,
            patch(
                "main.add_live_start_offset",
                side_effect=lambda value, _seconds: value + "#start\n",
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
            dates.assert_called_once_with(str(tmp_path), content, 10, 0)
            start.assert_called_once_with(content + "#dated\n", 24)
            assert "#dated" in response.text
            assert "#start" in response.text

    def test_segmented_webvtt_is_mapped_to_the_video_clock(self, auth_client, tmp_path):
        filename = "sub0_000001.vtt"
        (tmp_path / filename).write_text(
            "WEBVTT\n\n00:01.000 --> 00:03.000\nCaption\n"
        )
        session = {"dir": str(tmp_path)}
        with (
            patch("main.ffmpeg_session.get_session", return_value=session),
            patch(
                "main.ffmpeg_session.get_caption_timestamp_origin",
                return_value=10.08,
            ),
        ):
            response = auth_client.get(f"/transcode/session/{filename}")

        assert response.status_code == 200
        assert (
            "X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:907200"
            in response.text
        )


class TestSubtitleRoutes:
    """Tests for subtitle routes."""

    def test_subtitle_invalid_filename(self, auth_client):
        resp = auth_client.get("/subs/session/notavtt.txt")
        assert resp.status_code == 400

    def test_subtitle_session_not_found(self, auth_client):
        with patch("main.ffmpeg_session.get_session", return_value=None):
            resp = auth_client.get("/subs/invalid-session/sub0.vtt")
            assert resp.status_code == 404


class TestProbeCache:
    """Tests for probe cache endpoints."""

    def test_get_probe_cache(self, auth_client):
        with patch("main.ffmpeg_command.get_series_probe_cache_stats", return_value=[]):
            resp = auth_client.get("/settings/probe-cache")
            assert resp.status_code == 200
            assert "series" in resp.json()

    def test_clear_probe_cache(self, auth_client):
        with patch("main.ffmpeg_command.clear_all_probe_cache", return_value=5):
            resp = auth_client.post("/settings/probe-cache/clear")
            assert resp.status_code == 200
            assert resp.json()["cleared"] == 5

    def test_clear_series_probe_cache(self, auth_client):
        with patch("main.ffmpeg_command.invalidate_series_probe_cache"):
            resp = auth_client.post("/settings/probe-cache/clear/123")
            assert resp.status_code == 200


class TestRefreshStatus:
    """Tests for refresh status endpoints."""

    def test_guide_refresh_status(self, auth_client):
        m3u_module.get_refresh_in_progress().clear()

        resp = auth_client.get("/guide/refresh-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["live"] is False
        assert data["epg"] is False

    def test_settings_refresh_status(self, auth_client):
        m3u_module.get_refresh_in_progress().clear()

        resp = auth_client.get("/settings/refresh-status")
        assert resp.status_code == 200


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)


class TestPlaybackHealth:
    def test_invalid_feedback_is_rejected(self, auth_client):
        response = auth_client.post("/transcode/missing/health", json={"buffer_seconds": -1, "waiting": True})
        assert response.status_code == 422

    def test_feedback_is_passed_to_backend(self, auth_client):
        with patch("ffmpeg_session.report_playback_health", return_value={"bandwidth_saver": True}) as report:
            response = auth_client.post("/transcode/live/health", json={"buffer_seconds": 0, "waiting": True})
        assert response.json() == {"bandwidth_saver": True}
        assert report.call_args.args[:2] == ("live", "testuser")

    def test_force_stop_releases_current_stream(self, auth_client):
        with (
            patch("ffmpeg_session.get_session", return_value={"username": "testuser"}),
            patch("ffmpeg_session.stop_session") as stop,
        ):
            response = auth_client.delete("/transcode/live?force=true")
        assert response.status_code == 200
        stop.assert_called_once_with("live", force=True)

    def test_force_stop_requires_owner(self, auth_client):
        with (
            patch("ffmpeg_session.get_session", return_value={"username": "other"}),
            patch("ffmpeg_session.stop_session") as stop,
        ):
            assert auth_client.delete("/transcode/live?force=true").status_code == 404
        stop.assert_not_called()


class TestPlaylists:
    @pytest.fixture(autouse=True)
    def live_data(self, auth_client):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"},
            {"category_id": "2", "category_name": "Sports"},
        ]
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN", "category_ids": ["1"], "epg_channel_id": "", "source_id": "s"},
            {"stream_id": 2, "name": "ESPN", "category_ids": ["2"], "epg_channel_id": "", "source_id": "s"},
            {"stream_id": 3, "name": "FS1", "category_ids": ["2"], "epg_channel_id": "", "source_id": "s"},
        ]

    def _create(self, client, name="Favorites", ids=(3, 1)):
        streams = {s["stream_id"]: s for s in cache_module.get_cache()["live_streams"]}
        channels = [{"stream_id": i, "name": streams[i]["name"], "source_id": "s"} for i in ids]
        resp = client.post("/api/playlists", json={"name": name, "channels": channels})
        assert resp.status_code == 200
        return resp.json()

    def test_guide_rows_prepend_playlists_for_app_requests(self, auth_client):
        playlist = self._create(auth_client)

        payload = auth_client.get("/api/guide/rows?cats=1,2").json()

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [3, 1, 2]
        assert payload["rows"][0]["channel"]["category_ids"] == ["2", playlist["category_id"]]
        assert payload["categories"][0] == {
            "category_id": playlist["category_id"],
            "category_name": "Favorites",
            "kind": "playlist",
        }
        assert [c["category_id"] for c in payload["categories"][1:]] == ["1", "2"]

    def test_guide_rows_exact_skips_playlists(self, auth_client):
        self._create(auth_client)

        payload = auth_client.get("/api/guide/rows?cats=2&exact=1").json()

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [2, 3]
        assert [c["category_id"] for c in payload["categories"]] == ["2"]

    def test_guide_rows_single_playlist_keeps_playlist_order(self, auth_client):
        playlist = self._create(auth_client, ids=(2, 3, 1))

        payload = auth_client.get(f"/api/guide/rows?cats={playlist['category_id']}").json()

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [2, 3, 1]

    def test_guide_rows_order_most_viewed_streams_within_category(self, auth_client):
        cache_module.record_live_view("testuser", "s", 3)
        cache_module.record_live_view("testuser", "s", 3)

        payload = auth_client.get("/api/guide/rows?cats=1,2&exact=1").json()

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [1, 3, 2]

    def test_guide_rows_order_most_viewed_streams_within_playlist(self, auth_client):
        playlist = self._create(auth_client, ids=(2, 3, 1))
        cache_module.record_live_view("testuser", "s", 1)

        payload = auth_client.get(f"/api/guide/rows?cats={playlist['category_id']}").json()

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [1, 2, 3]

    def test_guide_rows_preserve_order_when_view_sorting_is_disabled(self, auth_client):
        cache_module.record_live_view("testuser", "s", 3)
        settings = cache_module.load_user_settings("testuser")
        settings["sort_live_by_views"] = False
        cache_module.save_user_settings("testuser", settings)

        payload = auth_client.get("/api/guide/rows?cats=2&exact=1").json()

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [2, 3]

    def test_guide_page_defaults_to_playlists_then_filter(self, auth_client):
        playlist = self._create(auth_client)
        auth_client.post("/settings/guide-filter", json={"cats": ["1"]})
        with patch("main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide")
        assert resp.status_code == 200
        assert "★ Favorites" in resp.text
        assert f'"{playlist["category_id"]},1"' in resp.text

    def test_guide_page_shows_section_headings(self, auth_client):
        self._create(auth_client)
        auth_client.post("/settings/guide-filter", json={"cats": ["2"]})
        with patch("main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide")

        match = re.search(r"sections: (\[.*?\]),\n", resp.text)
        assert match
        sections = json.loads(match.group(1))
        assert sections == [{"start": 0, "name": "★ Favorites"}, {"start": 2, "name": "Sports"}]
        headings = re.findall(r'class="guide-section[^"]*">\s*<span class="truncate">([^<]*)</span>', resp.text)
        assert headings == ["★ Favorites", "Sports"]
        rows = re.findall(r'<div class="guide-row[^"]*" data-row="(\d+)"', resp.text)
        assert rows == ["0", "1", "2"]

    def _sidebar(self, client, url="/guide"):
        with patch("main.epg.has_programs", return_value=True):
            text = client.get(url).text
        aside = text[text.index("<aside") : text.index("</aside>")]
        links = re.findall(r'data-nav="cat".*?title="([^"]*)">.*?tabular-nums[^>]*>(\d+)<', aside, re.S)
        current = re.findall(r'aria-current="page" title="([^"]*)"', aside)
        rows = re.findall(r'<div class="guide-row[^"]*" data-row="\d+"', text)
        return links, current, len(rows)

    def test_guide_sidebar_lists_playlists_and_categories(self, auth_client):
        import main

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
        assert main.load_user_settings("testuser")["guide_selected_cats"] == [playlist["category_id"]]

        _, current, rows = self._sidebar(auth_client, "/guide?cats=")
        assert current == ["All Channels"]
        assert rows == 3
        assert main.load_user_settings("testuser")["guide_selected_cats"] is None

    def test_old_multi_category_view_falls_back_to_all(self, auth_client):
        auth_client.post("/settings/guide-filter", json={"cats": ["1", "2"]})
        auth_client.post("/api/user-prefs", json={"guide_selected_cats": ["1", "2"]})
        self._create(auth_client)

        _, current, rows = self._sidebar(auth_client)
        assert current == ["All Channels"]
        assert rows == 3

    def test_guide_rows_use_earliest_selected_category(self, auth_client):
        cache_module.get_cache()["live_streams"].append(
            {"stream_id": 4, "name": "Both", "category_ids": ["2", "1"], "epg_channel_id": "", "source_id": "s"}
        )

        payload = auth_client.get("/api/guide/rows?cats=1,2").json()

        assert [r["channel"]["stream_id"] for r in payload["rows"]] == [1, 4, 2, 3]

    def test_empty_playlists_are_hidden_from_guide(self, auth_client):
        auth_client.post("/api/playlists", json={"name": "Empty"})
        payload = auth_client.get("/api/guide/rows?cats=1").json()
        assert [c["category_id"] for c in payload["categories"]] == ["1"]

    def test_update_rename_reorder_delete(self, auth_client):
        first = self._create(auth_client, "A")
        second = self._create(auth_client, "B")

        resp = auth_client.put(f"/api/playlists/{first['id']}", json={"name": "A2", "channels": []})
        assert resp.json()["name"] == "A2"
        auth_client.post("/api/playlists/order", json={"ids": [second["id"], first["id"]]})
        assert [p["name"] for p in auth_client.get("/api/playlists").json()["playlists"]] == ["B", "A2"]
        assert auth_client.delete(f"/api/playlists/{second['id']}").status_code == 200
        assert auth_client.get(f"/api/playlists/{second['id']}").status_code == 404

    def test_playlist_reports_unavailable_channels(self, auth_client):
        playlist = self._create(auth_client)
        auth_client.put(
            f"/api/playlists/{playlist['id']}",
            json={"channels": [{"stream_id": 42, "name": "Gone", "source_id": "s"}]},
        )
        channels = auth_client.get(f"/api/playlists/{playlist['id']}").json()["channels"]
        assert channels == [
            {"stream_id": "42", "name": "Gone", "source_id": "s", "epg_channel_id": "", "available": False}
        ]

    def test_assigned_guide_survives_round_trip_and_feeds_guide(self, auth_client):
        playlist = self._create(auth_client, ids=(1,))
        channels = auth_client.get(f"/api/playlists/{playlist['id']}").json()["channels"]
        assert channels[0]["guide_assignable"] is True
        channels[0]["epg_channel_id"] = "guide.one"
        auth_client.put(f"/api/playlists/{playlist['id']}", json={"channels": channels})

        channels = auth_client.get(f"/api/playlists/{playlist['id']}").json()["channels"]
        assert channels[0]["epg_channel_id"] == "guide.one"
        with (
            patch("main.epg.get_icons_batch", return_value={}),
            patch("main.epg.get_programs_batch", return_value={}),
        ):
            rows = auth_client.get(f"/api/guide/rows?cats={playlist['category_id']}").json()["rows"]
        assert rows[0]["channel"]["epg_id"] == "guide.one"

    def test_epg_channel_search(self, auth_client):
        with patch("main.epg.search_channels", return_value=[]) as search:
            resp = auth_client.get("/api/playlists/epg-channels?q=XX| ALPHA ONE HD")
        assert resp.status_code == 200
        search.assert_called_once_with(["ALPHA", "ONE"], 20)

    def test_match_and_search(self, auth_client):
        matches = auth_client.post("/api/playlists/match", json={"lines": ["ESPN", ""]}).json()["matches"]
        assert [(m["query"], m["candidates"][0]["stream_id"]) for m in matches] == [("ESPN", "2")]
        results = auth_client.get("/api/playlists/search?category_id=2").json()["results"]
        assert [r["stream_id"] for r in results] == ["2", "3"]

    def test_non_admin_can_read_but_not_edit(self, auth_client):
        import auth

        playlist = self._create(auth_client)
        auth.create_user("viewer", "viewerpass1")
        auth_client.cookies.set("token", auth.create_token({"sub": "viewer"}))

        assert auth_client.get("/api/playlists").status_code == 200
        assert auth_client.get(f"/api/playlists/{playlist['id']}").status_code == 200
        assert auth_client.post("/api/playlists", json={"name": "X"}).status_code == 403
        assert auth_client.put(f"/api/playlists/{playlist['id']}", json={"name": "X"}).status_code == 403
        assert auth_client.delete(f"/api/playlists/{playlist['id']}").status_code == 403
        assert auth_client.get("/playlists").status_code == 403
        rows = auth_client.get("/api/guide/rows?cats=1").json()["rows"]
        assert [r["channel"]["stream_id"] for r in rows] == [3, 1]

    def test_admin_page_renders(self, auth_client):
        resp = auth_client.get("/playlists")
        assert resp.status_code == 200
        assert 'id="playlist-list"' in resp.text

    def test_same_stream_id_from_two_sources_is_kept(self, auth_client):
        cache_module.get_cache()["live_streams"].append(
            {"stream_id": 1, "name": "BBC", "category_ids": ["1"], "epg_channel_id": "", "source_id": "t"}
        )
        resp = auth_client.post(
            "/api/playlists",
            json={"name": "Mix", "channels": [{"stream_id": 1, "name": "BBC", "source_id": "t"}]},
        )
        playlist = resp.json()

        rows = auth_client.get("/api/guide/rows?cats=1").json()["rows"]

        assert [(r["channel"]["stream_id"], r["channel"]["name"]) for r in rows] == [(1, "BBC"), (1, "CNN")]
        assert playlist["category_id"] in rows[0]["channel"]["category_ids"]
        assert playlist["category_id"] not in rows[1]["channel"]["category_ids"]

    def test_restricted_viewer_does_not_see_restricted_playlist_channels(self, auth_client):
        import auth

        playlist = self._create(auth_client, ids=(3, 1))
        only_sports = self._create(auth_client, "Sports", ids=(3,))
        auth.create_user("viewer", "viewerpass1")
        auth.set_user_limits("viewer", unavailable_groups=["cat:2"])
        auth_client.cookies.set("token", auth.create_token({"sub": "viewer"}))

        listed = auth_client.get("/api/playlists").json()["playlists"]
        assert [(p["id"], p["channel_count"]) for p in listed] == [(playlist["id"], 1)]
        detail = auth_client.get(f"/api/playlists/{playlist['id']}").json()
        assert [c["name"] for c in detail["channels"]] == ["CNN"]
        assert auth_client.get(f"/api/playlists/{only_sports['id']}").json()["channels"] == []
        rows = auth_client.get("/api/guide/rows?cats=1").json()["rows"]
        assert [r["channel"]["stream_id"] for r in rows] == [1]

    def test_invalid_create_does_not_persist(self, auth_client):
        too_many = [{"stream_id": i, "name": str(i)} for i in range(2001)]
        assert auth_client.post("/api/playlists", json={"name": "Big", "channels": too_many}).status_code == 400
        assert auth_client.get("/api/playlists").json()["playlists"] == []

    def test_saved_view_drops_deleted_playlist(self, auth_client):
        import main

        playlist = self._create(auth_client)
        auth_client.post("/settings/guide-filter", json={"cats": ["1"]})
        auth_client.post("/api/user-prefs", json={"guide_selected_cats": [playlist["category_id"]]})
        auth_client.delete(f"/api/playlists/{playlist['id']}")

        with patch("main.epg.has_programs", return_value=True):
            resp = auth_client.get("/guide")

        assert resp.status_code == 200
        assert "CNN" in resp.text
        assert main.load_user_settings("testuser").get("guide_selected_cats") is None


class TestSourceAssignment:
    """Sources can be limited to selected users; admins always see everything."""

    @staticmethod
    def _setup(auth_client, family_users):
        import auth

        settings = cache_module.load_server_settings()
        settings["sources"] = [
            {"id": "fam", "name": "Family", "type": "m3u", "url": "http://a", "users": family_users},
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
                    {"stream_id": "fam_1", "name": "A", "category_ids": ["fam_1"], "source_id": "fam"},
                    {"stream_id": "pub_1", "name": "B", "category_ids": ["pub_1"], "source_id": "pub"},
                ],
            }
        )
        auth.create_user("friend", "friendpass1")
        auth.create_user("kid", "kidpass123")
        return auth

    def _rows(self, auth_client, user):
        import auth

        auth_client.cookies.set("token", auth.create_token({"sub": user}))
        rows = auth_client.get("/api/guide/rows?cats=fam_1,pub_1&exact=1").json()["rows"]
        return [r["channel"]["name"] for r in rows]

    def test_assigned_source_is_hidden_from_other_users(self, auth_client):
        self._setup(auth_client, ["kid"])
        assert self._rows(auth_client, "testuser") == ["A", "B"]  # admin
        assert self._rows(auth_client, "kid") == ["A", "B"]
        assert self._rows(auth_client, "friend") == ["B"]

    def test_unassigned_source_is_shared(self, auth_client):
        self._setup(auth_client, None)
        assert self._rows(auth_client, "friend") == ["A", "B"]

    def test_hidden_source_blocks_transcode(self, auth_client):
        import auth

        self._setup(auth_client, [])
        auth_client.cookies.set("token", auth.create_token({"sub": "kid"}))
        resp = auth_client.get("/transcode/start?url=http://a/x.ts&source_id=fam")
        assert resp.status_code == 403

    def test_edit_source_saves_selected_users(self, auth_client):
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

    def test_deleting_user_keeps_source_restricted(self, auth_client):
        auth = self._setup(auth_client, ["kid"])
        auth.delete_user("kid")
        fam = next(s for s in cache_module.get_sources() if s.id == "fam")
        assert fam.users == []

    def test_settings_api_hides_private_settings(self, auth_client):
        self._setup(auth_client, ["kid"])
        data = auth_client.get("/api/settings").json()
        assert "sources" not in data
        assert "users" not in data
