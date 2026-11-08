"""Tests for the native-player gateway."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from urllib import parse

import asyncio
import logging
import runpy
import sys

from fastapi import Request
from fastapi.testclient import TestClient

import pytest
import uvicorn

from gateway_catalog import CatalogSnapshot, GatewayStream

import auth
import cache
import gateway


@pytest.fixture
def gateway_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings_file = tmp_path / "server_settings.json"
    users_dir = tmp_path / "users"
    users_dir.mkdir()
    monkeypatch.setattr(auth, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(auth, "SERVER_SETTINGS_FILE", settings_file)
    monkeypatch.setattr(auth, "USERS_DIR", users_dir)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "SERVER_SETTINGS_FILE", settings_file)
    monkeypatch.setattr(cache, "USERS_DIR", users_dir)
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path / "engines"))
    monkeypatch.delenv("NETV_GATEWAY_PUBLIC_URL", raising=False)
    auth.create_user("player", "local-pass")
    monkeypatch.setattr(gateway, "authenticator", gateway.GatewayAuthenticator())

    public = {
        "num": 1,
        "name": "News One",
        "stream_type": "live",
        "stream_id": 41,
        "stream_icon": "https://images.example/news.png",
        "epg_channel_id": "news.one",
        "added": "",
        "category_id": "src_1_news",
        "category_ids": ["src_1_news"],
        "custom_sid": "",
        "tv_archive": 0,
        "tv_archive_duration": 0,
    }
    stream = GatewayStream(
        local_id=41,
        source_id="src_1",
        source_type="xtream",
        upstream_id="987",
        direct_url="",
        access_group_ids=("src_1_news",),
        public=public,
    )
    snapshot = CatalogSnapshot(
        categories=[
            {
                "category_id": "src_1_news",
                "category_name": "News",
                "parent_id": 0,
            },
        ],
        category_access_ids={"src_1_news": "src_1_news"},
        streams=[stream],
        streams_by_id={41: stream},
    )
    monkeypatch.setattr(gateway, "catalog", _FakeCatalog(snapshot))
    return TestClient(gateway.app), snapshot


def test_healthcheck_does_not_require_auth(gateway_client):
    client, _ = gateway_client
    assert client.get("/healthz").json() == {"status": "ok"}


def test_player_api_rejects_invalid_credentials(gateway_client):
    client, _ = gateway_client
    response = client.get(
        "/player_api.php",
        params={"username": "player", "password": "wrong"},
    )
    assert response.status_code == 200
    assert response.json()["user_info"]["auth"] == 0


def test_player_api_exposes_active_server_profile(gateway_client):
    client, _ = gateway_client
    response = client.get(
        "/player_api.php",
        params={"username": "player", "password": "local-pass"},
    )
    assert response.status_code == 200
    assert response.json()["user_info"] == {
        "username": "player",
        "password": "local-pass",
        "message": "neTV native-player gateway",
        "auth": 1,
        "status": "Active",
        "exp_date": None,
        "is_trial": "0",
        "active_cons": "0",
        "created_at": "0",
        "max_connections": "1",
        "allowed_output_formats": ["ts"],
    }


def test_server_profile_uses_configured_public_url(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    monkeypatch.setenv("NETV_GATEWAY_PUBLIC_URL", "https://tv.example:8443")

    response = client.get(
        "/player_api.php",
        params={"username": "player", "password": "local-pass"},
    )

    assert response.json()["server_info"] == {
        "url": "tv.example",
        "port": "8443",
        "https_port": "8443",
        "server_protocol": "https",
        "rtmp_port": "0",
        "timezone": "UTC",
        "timestamp_now": response.json()["server_info"]["timestamp_now"],
        "time_now": response.json()["server_info"]["time_now"],
    }


def test_successful_authentication_is_cached(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    verify_password = MagicMock(return_value=True)
    monkeypatch.setattr(auth, "verify_password", verify_password)

    for _ in range(2):
        response = client.get(
            "/player_api.php",
            params={"username": "player", "password": "local-pass"},
        )
        assert response.status_code == 200

    verify_password.assert_called_once_with("player", "local-pass")


def test_failed_authentication_is_rate_limited(gateway_client):
    client, _ = gateway_client
    params = {"username": "player", "password": "wrong"}

    for _ in range(10):
        assert client.get("/player_api.php", params=params).status_code == 200

    assert client.get("/player_api.php", params=params).status_code == 429


def test_success_cache_is_checked_before_client_rate_limit(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    valid = {"username": "player", "password": "local-pass"}
    assert client.get("/player_api.php", params=valid).status_code == 200
    gateway.authenticator.max_login_failures = 1
    assert (
        client.get(
            "/player_api.php",
            params={"username": "player", "password": "wrong"},
        ).status_code
        == 200
    )

    assert client.get("/player_api.php", params=valid).status_code == 200


def test_player_api_exposes_local_live_catalog(gateway_client):
    client, _ = gateway_client
    response = client.get(
        "/player_api.php",
        params={
            "username": "player",
            "password": "local-pass",
            "action": "get_live_streams",
        },
    )
    assert response.status_code == 200
    assert response.json() == [
        {
            "num": 1,
            "name": "News One",
            "stream_type": "live",
            "stream_id": 41,
            "stream_icon": "https://images.example/news.png",
            "epg_channel_id": "news.one",
            "added": "",
            "category_id": "src_1_news",
            "category_ids": ["src_1_news"],
            "custom_sid": "",
            "tv_archive": 0,
            "tv_archive_duration": 0,
        },
    ]


def test_player_api_orders_most_viewed_streams_within_category(gateway_client):
    client, snapshot = gateway_client
    second = GatewayStream(
        local_id=42,
        source_id="src_1",
        source_type="xtream",
        upstream_id="654",
        direct_url="",
        access_group_ids=("src_1_news",),
        public={
            **snapshot.streams[0].public,
            "num": 2,
            "name": "News Two",
            "stream_id": 42,
        },
    )
    snapshot.streams.append(second)
    snapshot.streams_by_id[42] = second
    cache.record_live_view("player", "src_1", "654")

    response = client.get(
        "/player_api.php",
        params={
            "username": "player",
            "password": "local-pass",
            "action": "get_live_streams",
            "category_id": "src_1_news",
        },
    )

    assert [stream["stream_id"] for stream in response.json()] == [42, 41]

    settings = cache.load_user_settings("player")
    settings["sort_live_by_views"] = False
    cache.save_user_settings("player", settings)
    response = client.get(
        "/player_api.php",
        params={
            "username": "player",
            "password": "local-pass",
            "action": "get_live_streams",
            "category_id": "src_1_news",
        },
    )
    assert [stream["stream_id"] for stream in response.json()] == [41, 42]


def test_playlist_contains_only_local_playback_urls(gateway_client):
    client, _ = gateway_client
    response = client.get(
        "/get.php",
        params={"username": "player", "password": "local-pass", "output": "ts"},
    )
    assert response.status_code == 200
    assert "http://testserver/live/player/local-pass/41.ts" in response.text
    assert "provider.example/live" not in response.text
    assert 'group-title="News"' in response.text


def test_upscale_setting_controls_gateway_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    engine = tmp_path / "2x-nomosuni-compact_720p_fp16.engine"
    engine.write_bytes(b"engine")
    (tmp_path / "retired-model_720p_fp16.engine").write_bytes(b"engine")
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path))
    settings = {
        "sr_model": "",
        "transcode_hw": "nvenc+software",
        "max_resolution": "4k",
        "quality": "high",
    }
    monkeypatch.setattr(gateway.cache, "load_server_settings", lambda: settings)

    assert gateway._upscale_enabled() is False
    assert gateway._available_upscale_models() == ["2x-nomosuni-compact"]

    settings["sr_model"] = "retired-model"

    assert gateway._upscale_enabled() is True
    assert gateway._upscale_settings()["sr_model"] == "2x-nomosuni-compact"
    assert gateway._upscale_settings()["max_resolution"] == "4k"


def test_disabled_upscale_accepts_480p_max_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path))
    monkeypatch.setattr(
        gateway.cache,
        "load_server_settings",
        lambda: {
            "sr_model": "",
            "transcode_hw": "nvenc+software",
            "max_resolution": "480p",
            "quality": "high",
        },
    )

    assert gateway._upscale_enabled() is False
    assert gateway._upscale_settings()["max_resolution"] == "480p"


def test_upscale_playlist_uses_hls_urls(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    monkeypatch.setattr(gateway, "_upscale_enabled", lambda: True)

    response = client.get(
        "/get.php",
        params={"username": "player", "password": "local-pass", "output": "m3u8"},
    )

    assert response.status_code == 200
    assert "http://testserver/live/player/local-pass/41.m3u8" in response.text


def test_upscale_live_stream_starts_session(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    monkeypatch.setattr(gateway, "_upscale_enabled", lambda: True)
    monkeypatch.setattr(
        gateway,
        "_resolve_upstream",
        lambda _stream, _extension: "https://provider.example/live/987.ts",
    )
    start = AsyncMock(return_value={"session_id": "upscale-session"})
    monkeypatch.setattr(gateway.ffmpeg_session, "start_transcode", start)

    response = client.get(
        "/live/player/local-pass/41.m3u8",
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["location"] == "/upscale/upscale-session/stream.m3u8"
    start.assert_awaited_once()
    await_args = start.await_args
    assert await_args is not None
    assert await_args.args[0] == "https://provider.example/live/987.ts"
    assert await_args.kwargs["content_type"] == "live"
    assert await_args.kwargs["username"] == "player"


def test_upscale_files_are_served_from_session(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    client, _ = gateway_client
    (tmp_path / "stream.m3u8").write_text("#EXTM3U\n#EXTINF:2.0,\nsegment000.ts\n")
    (tmp_path / "segment000.ts").write_bytes(b"video")
    monkeypatch.setattr(
        gateway.ffmpeg_session,
        "get_session",
        lambda _session_id: {"dir": str(tmp_path)},
    )
    touch = MagicMock(return_value=True)
    monkeypatch.setattr(gateway.ffmpeg_session, "touch_session", touch)

    manifest = client.get("/upscale/session/stream.m3u8")
    segment = client.get("/upscale/session/segment000.ts")

    assert manifest.status_code == 200
    assert manifest.headers["content-type"].startswith("application/vnd.apple.mpegurl")
    assert "segment000.ts" in manifest.text
    assert segment.status_code == 200
    assert segment.content == b"video"
    assert touch.call_count == 2


def test_user_category_restrictions_apply_to_gateway(gateway_client):
    client, _ = gateway_client
    assert auth.set_user_limits("player", unavailable_groups=["cat:src_1_news"])
    response = client.get(
        "/player_api.php",
        params={
            "username": "player",
            "password": "local-pass",
            "action": "get_live_streams",
        },
    )
    assert response.json() == []


def test_source_assignment_applies_to_gateway(gateway_client):
    client, _ = gateway_client
    auth.create_user("guest", "guest-pass")
    settings = cache.load_server_settings()
    settings["sources"] = [
        {"id": "src_1", "name": "A", "type": "xtream", "url": "http://a", "users": []},
    ]
    cache.save_server_settings(settings)

    def api(username: str, password: str, action: str):
        params = {"username": username, "password": password, "action": action}
        return client.get("/player_api.php", params=params).json()

    assert api("guest", "guest-pass", "get_live_streams") == []
    assert api("guest", "guest-pass", "get_live_categories") == []
    assert len(api("player", "local-pass", "get_live_streams")) == 1


def test_uncategorized_streams_hidden_for_restricted_user(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    _, snapshot = gateway_client
    stream = snapshot.streams[0]
    uncategorized = GatewayStream(
        local_id=stream.local_id,
        source_id=stream.source_id,
        source_type=stream.source_type,
        upstream_id=stream.upstream_id,
        direct_url=stream.direct_url,
        access_group_ids=(),
        public={**stream.public, "category_id": "99", "category_ids": ["99"]},
    )
    restricted_snapshot = CatalogSnapshot(
        categories=[{"category_id": "99", "category_name": "Uncategorized", "parent_id": 0}],
        category_access_ids={},
        streams=[uncategorized],
        streams_by_id={uncategorized.local_id: uncategorized},
    )
    monkeypatch.setattr(
        auth,
        "get_user_limits",
        lambda _username: {"unavailable_groups": ["cat:some-other-category"]},
    )

    assert gateway._allowed_category_ids("player", restricted_snapshot) == set()
    assert gateway._visible_streams("player", restricted_snapshot) == []


def test_live_stream_proxies_resolved_upstream(gateway_client, monkeypatch: pytest.MonkeyPatch):
    client, _ = gateway_client
    upstream = MagicMock()
    upstream.headers = {"Content-Type": "video/mp2t"}
    upstream.read.side_effect = [b"video-data", b""]
    fake_xtream = MagicMock()
    fake_xtream.build_stream_url.return_value = (
        "https://provider.example/live/remote-user/remote-pass/987.ts"
    )
    monkeypatch.setattr(gateway, "get_xtream_client_by_source", lambda _source_id: fake_xtream)
    open_mock = MagicMock(return_value=upstream)
    monkeypatch.setattr(gateway, "safe_urlopen", open_mock)

    response = client.get("/live/player/local-pass/41.ts")

    assert response.status_code == 200
    assert response.content == b"video-data"
    fake_xtream.build_stream_url.assert_called_once_with("live", 987, "ts")
    assert open_mock.call_args.args[0].endswith("/987.ts")
    upstream.close.assert_called_once()
    assert cache.load_live_view_counts("player") == {"src_1:987": 1}


def test_live_stream_enforces_category_restrictions(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    assert auth.set_user_limits("player", unavailable_groups=["cat:src_1_news"])
    open_mock = MagicMock()
    monkeypatch.setattr(gateway, "safe_urlopen", open_mock)

    response = client.get("/live/player/local-pass/41.ts")

    assert response.status_code == 404
    open_mock.assert_not_called()


def test_xmltv_builds_feed_for_visible_streams(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, snapshot = gateway_client
    build_mock = MagicMock(return_value=b'<?xml version="1.0"?><tv />')
    monkeypatch.setattr(gateway, "build_filtered_xmltv", build_mock)

    response = client.get(
        "/xmltv.php",
        params={"username": "player", "password": "local-pass"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert build_mock.call_args.args[1] == snapshot.streams


def test_xmltv_enforces_category_restrictions(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    assert auth.set_user_limits("player", unavailable_groups=["cat:src_1_news"])
    build_mock = MagicMock(return_value=b'<?xml version="1.0"?><tv />')
    monkeypatch.setattr(gateway, "build_filtered_xmltv", build_mock)

    response = client.get(
        "/xmltv.php",
        params={"username": "player", "password": "local-pass"},
    )

    assert response.status_code == 200
    assert build_mock.call_args.args[1] == []


def test_playlist_accepts_mpegts_output(gateway_client):
    client, _ = gateway_client
    response = client.get(
        "/get.php",
        params={
            "username": "player",
            "password": "local-pass",
            "output": "mpegts",
        },
    )
    assert response.status_code == 200


def test_live_stream_password_may_contain_slash(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    upstream = MagicMock()
    upstream.headers = {"Content-Type": "video/mp2t"}
    upstream.read.side_effect = [b"video-data", b""]
    monkeypatch.setattr(
        gateway.authenticator,
        "verify",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(gateway, "_resolve_upstream", lambda _stream, _extension: "http://upstream")
    monkeypatch.setattr(gateway, "safe_urlopen", MagicMock(return_value=upstream))

    response = client.get("/live/player/local%2Fpass/41.ts")

    assert response.status_code == 200
    assert response.content == b"video-data"


def _settings(**overrides):
    return {
        "sr_model": "",
        "transcode_hw": "nvenc+software",
        "max_resolution": "1080p",
        "quality": "high",
        **overrides,
    }


def _request(host: str | None, base_url: str = "http://host/") -> Request:
    url = parse.urlsplit(base_url)
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": url.scheme,
            "server": (url.hostname, url.port or 80),
            "root_path": "",
            "path": "/",
            "query_string": b"",
            "headers": [(b"host", url.netloc.encode())],
            "client": (host, 0) if host else None,
        },
    )


def _params(**extra):
    return {"username": "player", "password": "local-pass", **extra}


def test_upscale_settings_rejects_missing_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(gateway.cache, "load_server_settings", lambda: _settings(sr_model="x"))

    with pytest.raises(RuntimeError, match="not installed; available models: none"):
        gateway._upscale_settings()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"max_resolution": "8k"}, "max_resolution must be one of"),
        ({"quality": "ultra"}, "quality must be one of"),
    ],
)
def test_upscale_settings_rejects_invalid_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override,
    message,
):
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path))
    monkeypatch.setattr(gateway.cache, "load_server_settings", lambda: _settings(**override))

    with pytest.raises(RuntimeError, match=message):
        gateway._upscale_settings()


def test_upscale_settings_fills_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path))
    monkeypatch.setattr(gateway.cache, "load_server_settings", dict)

    assert gateway._upscale_settings() == _settings()


def test_cleanup_upscale_sessions_runs_every_minute(monkeypatch: pytest.MonkeyPatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    class Stop(Exception):
        pass

    cleanup = MagicMock(side_effect=[None, Stop()])
    monkeypatch.setattr(gateway.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(gateway.ffmpeg_session, "cleanup_expired_sessions", cleanup)

    with pytest.raises(Stop):
        asyncio.run(gateway._cleanup_upscale_sessions())

    assert sleeps == [60, 60]
    assert cleanup.call_count == 2


@pytest.mark.parametrize(
    ("sr_model", "message"),
    [
        ("2x-nomosuni-compact", "Native upscaling enabled: model=2x-nomosuni-compact"),
        ("", "Native upscaling disabled"),
    ],
)
def test_lifespan_initializes_and_shuts_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    sr_model,
    message,
):
    calls: list[str] = []
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path))
    monkeypatch.setattr(gateway.cache, "initialize", lambda: calls.append("initialize"))
    init = MagicMock()
    monkeypatch.setattr(gateway.ffmpeg_command, "init", init)
    monkeypatch.setattr(
        gateway.ffmpeg_session,
        "cleanup_and_recover_sessions",
        lambda: calls.append("recover"),
    )
    monkeypatch.setattr(gateway.ffmpeg_session, "shutdown", lambda: calls.append("shutdown"))
    upscale_settings = MagicMock(return_value=_settings(sr_model=sr_model))
    monkeypatch.setattr(gateway, "_upscale_settings", upscale_settings)

    with caplog.at_level("INFO", logger="gateway"), TestClient(gateway.app) as client:
        assert calls == ["initialize", "recover"]
        assert client.get("/healthz").status_code == 200

    assert calls == ["initialize", "recover", "shutdown"]
    init.assert_called_once_with(upscale_settings, sr_engine_dir=str(tmp_path))
    assert message in caplog.text


def test_authenticator_evicts_expired_success_entries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(auth, "verify_password", lambda _u, _p: True)
    authenticator = gateway.GatewayAuthenticator(max_auth_cache_entries=2)
    authenticator._success_cache = {"expired": 0.0, "live": 1e12}

    assert asyncio.run(authenticator.verify(_request("1.1.1.1"), "u", "p"))

    fingerprint = gateway.GatewayAuthenticator._fingerprint("u", "p")
    assert set(authenticator._success_cache) == {"live", fingerprint}


def test_authenticator_evicts_oldest_success_entry_when_full(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(auth, "verify_password", lambda _u, _p: True)
    authenticator = gateway.GatewayAuthenticator(max_auth_cache_entries=2)
    authenticator._success_cache = {"expired": 0.0, "old": 1e12, "new": 2e12}

    assert asyncio.run(authenticator.verify(_request(None), "u", "p"))

    fingerprint = gateway.GatewayAuthenticator._fingerprint("u", "p")
    assert set(authenticator._success_cache) == {"new", fingerprint}


def test_authenticator_bounds_failure_clients(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(auth, "verify_password", lambda _u, _p: False)
    authenticator = gateway.GatewayAuthenticator(max_failure_clients=1)

    assert not asyncio.run(authenticator.verify(_request("1.1.1.1"), "u", "bad"))
    assert not asyncio.run(authenticator.verify(_request("2.2.2.2"), "u", "bad"))
    assert not asyncio.run(authenticator.verify(_request("2.2.2.2"), "u", "bad"))

    assert list(authenticator._failures) == ["2.2.2.2"]
    assert len(authenticator._failures["2.2.2.2"]) == 2
    assert len(authenticator._global_failures) == 3


def test_authenticator_global_rate_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(auth, "verify_password", lambda _u, _p: False)
    authenticator = gateway.GatewayAuthenticator(max_global_login_failures=1)
    asyncio.run(authenticator.verify(_request("1.1.1.1"), "u", "bad"))

    with pytest.raises(gateway.HTTPException) as exc_info:
        asyncio.run(authenticator.verify(_request("2.2.2.2"), "u", "bad"))
    assert exc_info.value.status_code == 429


def test_authenticator_forgets_failures_outside_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(auth, "verify_password", lambda _u, _p: True)
    authenticator = gateway.GatewayAuthenticator(max_login_failures=1)
    authenticator._failures = {"1.1.1.1": [0.0], "2.2.2.2": [0.0, 1e12]}
    authenticator._global_failures = [0.0]

    assert asyncio.run(authenticator.verify(_request("3.3.3.3"), "u", "p"))

    assert authenticator._failures == {"2.2.2.2": [1e12]}
    assert authenticator._global_failures == []


def test_healthcheck_reports_upscale_mode(gateway_client, monkeypatch: pytest.MonkeyPatch):
    client, _ = gateway_client
    monkeypatch.setattr(gateway, "_upscale_enabled", lambda: True)

    assert client.get("/healthz").json() == {"status": "ok", "mode": "upscale"}


@pytest.mark.parametrize(
    ("sr_model", "expected"),
    [
        (
            "on",
            {
                "upscale": True,
                "models": ["2x-nomosuni-compact"],
                "selected_model": "2x-nomosuni-compact",
                "max_resolution": "4k",
                "quality": "medium",
                "output_format": "hls",
            },
        ),
        (
            "",
            {
                "upscale": False,
                "models": ["2x-nomosuni-compact"],
                "selected_model": "",
                "max_resolution": "4k",
                "quality": "medium",
                "output_format": "mpegts",
            },
        ),
    ],
)
def test_capabilities(
    gateway_client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sr_model,
    expected,
):
    client, _ = gateway_client
    engines = tmp_path / "engines"
    engines.mkdir()
    (engines / "2x-nomosuni-compact_1080p_fp16.engine").write_bytes(b"engine")
    monkeypatch.setattr(
        gateway.cache,
        "load_server_settings",
        lambda: _settings(sr_model=sr_model, max_resolution="4k", quality="medium"),
    )

    assert client.get("/capabilities").json() == expected


def test_available_upscale_models_requires_matching_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path))
    (tmp_path / "other_720p_fp16.engine").write_bytes(b"engine")

    assert gateway._available_upscale_models() == []


def test_player_api_live_categories_include_unmapped_category(gateway_client):
    client, snapshot = gateway_client
    snapshot.categories.append({"category_id": "77", "category_name": "Misc", "parent_id": 0})

    response = client.get("/player_api.php", params=_params(action="get_live_categories"))

    assert [category["category_id"] for category in response.json()] == ["src_1_news", "77"]


def test_player_api_hides_unmapped_category_for_restricted_user(gateway_client):
    client, snapshot = gateway_client
    snapshot.categories.append({"category_id": "77", "category_name": "Misc", "parent_id": 0})
    assert auth.set_user_limits("player", unavailable_groups=["cat:other"])

    response = client.get("/player_api.php", params=_params(action="get_live_categories"))

    assert [category["category_id"] for category in response.json()] == ["src_1_news"]


@pytest.mark.parametrize(
    "action",
    ["get_vod_categories", "get_vod_streams", "get_series_categories", "get_series"],
)
def test_player_api_returns_empty_vod_and_series(gateway_client, action):
    client, _ = gateway_client

    assert client.get("/player_api.php", params=_params(action=action)).json() == []


def test_player_api_rejects_unsupported_action(gateway_client):
    client, _ = gateway_client

    response = client.get("/player_api.php", params=_params(action="bogus"))

    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported action: bogus"}


def test_short_epg_requires_stream_id(gateway_client):
    client, _ = gateway_client

    response = client.get("/player_api.php", params=_params(action="get_short_epg"))

    assert response.status_code == 400
    assert response.json() == {"detail": "stream_id is required"}


def test_short_epg_rejects_unknown_stream(gateway_client):
    client, _ = gateway_client

    response = client.get(
        "/player_api.php",
        params=_params(action="get_short_epg", stream_id=999),
    )

    assert response.status_code == 404


def test_short_epg_rejects_hidden_stream(gateway_client):
    client, _ = gateway_client
    assert auth.set_user_limits("player", unavailable_groups=["cat:src_1_news"])

    response = client.get(
        "/player_api.php",
        params=_params(action="get_short_epg", stream_id=41),
    )

    assert response.status_code == 404


def test_short_epg_without_source_client_is_empty(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    monkeypatch.setattr(gateway, "get_xtream_client_by_source", lambda _source_id: None)

    response = client.get(
        "/player_api.php",
        params=_params(action="get_short_epg", stream_id=41),
    )

    assert response.json() == {"epg_listings": []}


@pytest.mark.parametrize(("limit", "expected_limit"), [(500, 100), (0, 1), (5, 5)])
def test_short_epg_proxies_upstream_with_clamped_limit(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
    limit,
    expected_limit,
):
    client, _ = gateway_client
    xtream = MagicMock()
    xtream.get_short_epg.return_value = {"epg_listings": [{"title": "News"}]}
    monkeypatch.setattr(gateway, "get_xtream_client_by_source", lambda _source_id: xtream)

    response = client.get(
        "/player_api.php",
        params=_params(action="get_short_epg", stream_id=41, limit=limit),
    )

    assert response.json() == {"epg_listings": [{"title": "News"}]}
    xtream.get_short_epg.assert_called_once_with(987, expected_limit)


def test_short_epg_upstream_failure_is_bad_gateway(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    client, _ = gateway_client
    xtream = MagicMock()
    xtream.get_short_epg.side_effect = OSError("boom")
    monkeypatch.setattr(gateway, "get_xtream_client_by_source", lambda _source_id: xtream)

    response = client.get(
        "/player_api.php",
        params=_params(action="get_short_epg", stream_id=41),
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Unable to load upstream EPG data"}
    assert "Gateway EPG lookup failed (OSError)" in caplog.text


def test_player_api_orders_by_views_preserving_category_order(gateway_client):
    client, snapshot = gateway_client

    def add(local_id: int, upstream_id: str, category: str) -> None:
        stream = GatewayStream(
            local_id=local_id,
            source_id="src_1",
            source_type="xtream",
            upstream_id=upstream_id,
            direct_url="",
            access_group_ids=("src_1_news",),
            public={
                **snapshot.streams[0].public,
                "stream_id": local_id,
                "category_id": category,
                "category_ids": [category],
            },
        )
        snapshot.streams.append(stream)
        snapshot.streams_by_id[local_id] = stream

    add(42, "500", "sports")
    add(43, "600", "src_1_news")
    cache.record_live_view("player", "src_1", "500")
    cache.record_live_view("player", "src_1", "500")
    cache.record_live_view("player", "src_1", "600")

    response = client.get("/player_api.php", params=_params(action="get_live_streams"))

    assert [stream["stream_id"] for stream in response.json()] == [43, 41, 42]


def test_playlist_rejects_invalid_credentials(gateway_client):
    client, _ = gateway_client

    response = client.get("/get.php", params={"username": "player", "password": "wrong"})

    assert response.status_code == 401
    assert response.text == "Invalid username or password\n"


@pytest.mark.parametrize(
    ("params", "detail"),
    [
        ({"type": "xspf"}, "Unsupported playlist type"),
        ({"output": "mkv"}, "Unsupported playlist output"),
    ],
)
def test_playlist_rejects_unsupported_options(gateway_client, params, detail):
    client, _ = gateway_client

    response = client.get("/get.php", params=_params(**params))

    assert response.status_code == 400
    assert response.json() == {"detail": detail}


def test_playlist_sanitizes_values_and_defaults_group(gateway_client):
    client, snapshot = gateway_client
    stream = snapshot.streams[0]
    snapshot.streams[0] = GatewayStream(
        local_id=stream.local_id,
        source_id=stream.source_id,
        source_type=stream.source_type,
        upstream_id=stream.upstream_id,
        direct_url=stream.direct_url,
        access_group_ids=stream.access_group_ids,
        public={**stream.public, "name": 'Bad\r\n"Name"', "stream_icon": None, "category_id": "x"},
    )

    response = client.get("/get.php", params=_params())

    assert (
        '#EXTINF:-1 tvg-id="news.one" tvg-name="Bad  \'Name\'" tvg-logo="" '
        "group-title=\"Uncategorized\",Bad  'Name'"
    ) in response.text


def test_live_stream_rejects_malformed_path(gateway_client):
    client, _ = gateway_client

    for path in ("/live/player", "/live/player/pass", "/live/player/pass/41", "/live/u/p/x.ts"):
        response = client.get(path)
        assert response.status_code == 400, path
        assert response.json() == {"detail": "Invalid live stream path"}


def test_live_stream_rejects_invalid_credentials(gateway_client):
    client, _ = gateway_client

    assert client.get("/live/player/wrong/41.ts").status_code == 401


def test_live_stream_rejects_unsupported_format(gateway_client):
    client, _ = gateway_client

    response = client.get("/live/player/local-pass/41.mp4")

    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported stream format"}


def test_live_stream_rejects_unknown_stream(gateway_client):
    client, _ = gateway_client

    assert client.get("/live/player/local-pass/999.ts").status_code == 404


def test_upscale_live_stream_applies_source_limits(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    settings = cache.load_server_settings()
    settings["sources"] = [
        {
            "id": "src_1",
            "name": "A",
            "type": "xtream",
            "url": "http://a",
            "max_streams": 3,
            "deinterlace_fallback": False,
        },
    ]
    cache.save_server_settings(settings)
    assert auth.set_user_limits("player", max_streams_per_source={"src_1": 2})
    monkeypatch.setattr(gateway, "_upscale_enabled", lambda: True)
    monkeypatch.setattr(gateway, "_resolve_upstream", lambda _stream, _extension: "http://up")
    start = AsyncMock(return_value={"session_id": "s1"})
    monkeypatch.setattr(gateway.ffmpeg_session, "start_transcode", start)

    response = client.get("/live/player/local-pass/41.ts", follow_redirects=False)

    assert response.headers["location"] == "/upscale/s1/stream.m3u8"
    start.assert_awaited_once_with(
        "http://up",
        content_type="live",
        deinterlace_fallback=False,
        username="player",
        source_id="src_1",
        user_max_streams=2,
        source_max_streams=3,
    )


def test_live_stream_upstream_failure_is_bad_gateway(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    monkeypatch.setattr(gateway, "_resolve_upstream", lambda _stream, _extension: "http://up")
    monkeypatch.setattr(gateway, "safe_urlopen", MagicMock(side_effect=OSError("refused")))

    response = client.get("/live/player/local-pass/41.ts")

    assert response.status_code == 502
    assert response.json() == {"detail": "Unable to connect to upstream stream"}


def test_proxy_url_forwards_length_and_encoding(monkeypatch: pytest.MonkeyPatch):
    upstream = MagicMock()
    upstream.headers = {"Content-Length": "5", "Content-Encoding": "gzip"}
    upstream.read.side_effect = [b"abcde", b""]
    opener = MagicMock(return_value=upstream)
    monkeypatch.setattr(gateway, "safe_urlopen", opener)
    monkeypatch.setattr(gateway.cache, "get_user_agent", lambda: "agent/1")

    response = asyncio.run(gateway._proxy_url("http://up"))

    opener.assert_called_once_with("http://up", 30, "agent/1")
    assert response.media_type == "video/mp2t"
    assert response.headers["content-length"] == "5"
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["cache-control"] == "no-store"
    assert list(gateway._iter_upstream(upstream)) == [b"abcde"]


def _bare_stream(source_type: str, direct_url: str) -> GatewayStream:
    return GatewayStream(
        local_id=1,
        source_id="s",
        source_type=source_type,
        upstream_id="7",
        direct_url=direct_url,
        access_group_ids=(),
        public={},
    )


def test_resolve_upstream_uses_m3u_direct_url():
    stream = _bare_stream("m3u", "http://direct/1.ts")

    assert gateway._resolve_upstream(stream, "ts") == "http://direct/1.ts"


def test_resolve_upstream_rejects_m3u_without_url():
    with pytest.raises(gateway.HTTPException) as exc_info:
        gateway._resolve_upstream(_bare_stream("m3u", ""), "ts")
    assert exc_info.value.status_code == 502


def test_resolve_upstream_rejects_removed_xtream_source(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gateway, "get_xtream_client_by_source", lambda _source_id: None)

    with pytest.raises(gateway.HTTPException) as exc_info:
        gateway._resolve_upstream(_bare_stream("xtream", ""), "ts")
    assert exc_info.value.status_code == 404


def test_upscale_file_rejects_dotted_filename(gateway_client):
    client, _ = gateway_client

    response = client.get("/upscale/session/..stream.m3u8")

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid filename"}


def test_upscale_file_rejects_unknown_session(gateway_client, monkeypatch: pytest.MonkeyPatch):
    client, _ = gateway_client
    monkeypatch.setattr(gateway.ffmpeg_session, "touch_session", lambda _sid: False)

    assert client.get("/upscale/session/stream.m3u8").status_code == 404


def test_upscale_file_rejects_vanished_session(gateway_client, monkeypatch: pytest.MonkeyPatch):
    client, _ = gateway_client
    monkeypatch.setattr(gateway.ffmpeg_session, "touch_session", lambda _sid: True)
    monkeypatch.setattr(gateway.ffmpeg_session, "get_session", lambda _sid: None)

    response = client.get("/upscale/session/stream.m3u8")

    assert response.status_code == 404
    assert response.json() == {"detail": "Upscale session not found"}


def test_upscale_file_rejects_missing_and_unsupported_files(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    client, _ = gateway_client
    (tmp_path / "notes.txt").write_text("x")
    monkeypatch.setattr(gateway.ffmpeg_session, "touch_session", lambda _sid: True)
    monkeypatch.setattr(gateway.ffmpeg_session, "get_session", lambda _sid: {"dir": str(tmp_path)})

    missing = client.get("/upscale/session/segment999.ts")
    unsupported = client.get("/upscale/session/notes.txt")

    assert missing.status_code == 404
    assert missing.json() == {"detail": "File not found"}
    assert unsupported.status_code == 400
    assert unsupported.json() == {"detail": "Unsupported upscale file"}


def test_xmltv_rejects_invalid_credentials(gateway_client):
    client, _ = gateway_client

    response = client.get("/xmltv.php", params={"username": "player", "password": "wrong"})

    assert response.status_code == 401


def test_xmltv_reports_unavailable_epg(gateway_client, monkeypatch: pytest.MonkeyPatch):
    client, _ = gateway_client
    monkeypatch.setattr(
        gateway,
        "build_filtered_xmltv",
        MagicMock(side_effect=gateway.EpgUnavailableError("EPG database is not ready")),
    )

    response = client.get("/xmltv.php", params=_params())

    assert response.status_code == 503
    assert response.json() == {"detail": "EPG database is not ready"}


def test_server_info_defaults_ports_from_scheme(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NETV_GATEWAY_PUBLIC_URL", raising=False)
    monkeypatch.setattr(gateway, "_upscale_enabled", lambda: False)

    info = gateway._server_info(_request(None), "u", "p")

    assert info["server_info"]["port"] == "80"
    assert info["server_info"]["https_port"] == ""


def test_server_info_https_without_port(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NETV_GATEWAY_PUBLIC_URL", "https://tv.example/")
    monkeypatch.setattr(gateway, "_upscale_enabled", lambda: True)

    info = gateway._server_info(_request(None, "http://ignored/"), "u", "p")

    assert info["server_info"]["url"] == "tv.example"
    assert info["server_info"]["port"] == "443"
    assert info["server_info"]["https_port"] == "443"
    assert info["user_info"]["allowed_output_formats"] == ["m3u8"]


@pytest.mark.parametrize(
    ("argv", "env", "expected_level", "uvicorn_level"),
    [
        (["gateway.py", "--debug", "--port", "9001"], {}, logging.DEBUG, "debug"),
        (["gateway.py"], {"LOG_LEVEL": "warning"}, logging.WARNING, "info"),
        (["gateway.py"], {"LOG_LEVEL": "bogus"}, logging.INFO, "info"),
    ],
)
def test_main_runs_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
    argv,
    env,
    expected_level,
    uvicorn_level,
):

    run = MagicMock()
    basic_config = MagicMock()
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(logging, "basicConfig", basic_config)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delenv("NETV_GATEWAY_PORT", raising=False)
    monkeypatch.setenv("NETV_GATEWAY_TRUSTED_PROXIES", "10.0.0.1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(gateway.__file__, run_name="__main__")

    assert exit_info.value.code == 0
    assert basic_config.call_args.kwargs["level"] == expected_level
    kwargs = run.call_args.kwargs
    assert kwargs["port"] == (9001 if "--port" in argv else 8100)
    assert kwargs["log_level"] == uvicorn_level
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["forwarded_allow_ips"] == "10.0.0.1"
    assert kwargs["proxy_headers"] is True


class _FakeCatalog:
    def __init__(self, snapshot: CatalogSnapshot) -> None:
        self.snapshot = snapshot

    def get(self) -> CatalogSnapshot:
        return self.snapshot


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
