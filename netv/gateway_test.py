"""Tests for the native-player gateway."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Protocol, TypedDict, TypeVar
from urllib import error, parse

import asyncio
import ipaddress
import json
import logging
import runpy
import sys
import time

from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from treekle import MutablePlainTree, ReadError, json_loads

import pytest

from netv import auth, cache, db, epg, gateway, testing, util
from netv.cache import SourceConfig
from netv.ffmpeg import commands, probe, transcode
from netv.ffmpeg.transcode import TranscodeSession
from netv.gateway_catalog import CatalogSnapshot, GatewayCatalog, GatewayStream, StreamIdRegistry


GatewayClient = tuple[TestClient, CatalogSnapshot]
T = TypeVar("T")

MODEL = "2x-nomosuni-compact"
UPSTREAM = "https://provider.example/live/remote-user/remote-pass/987.ts"
API = "https://provider.example/player_api.php?username=remote-user&password=remote-pass"


@pytest.fixture
def engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An AI Upscale engine directory holding the default model's engine."""
    directory = tmp_path / "engines"
    directory.mkdir()
    (directory / f"{MODEL}_1080p_fp16.engine").write_bytes(b"engine")
    monkeypatch.setenv("SR_ENGINE_DIR", str(directory))
    return directory


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> dict[str, TranscodeSession]:
    """Start with no transcode sessions; return the live session table."""
    table: dict[str, TranscodeSession] = {}
    monkeypatch.setattr(transcode, "_transcode_sessions", table)
    monkeypatch.setattr(transcode, "_url_to_session", {})
    return table


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> testing.FakeWeb:
    """The provider, reachable only through the fake internet."""
    web = testing.FakeWeb()
    testing.install_web(monkeypatch, web, util)
    return web


@pytest.fixture
def gateway_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engines: Path,
    sessions: dict[str, TranscodeSession],
    web: testing.FakeWeb,
) -> GatewayClient:
    monkeypatch.delenv("NETV_GATEWAY_PUBLIC_URL", raising=False)
    monkeypatch.setattr(commands, "_load_settings", cache.load_server_settings)
    auth.create_user("player", "local-pass")
    _set_sources(_source())
    monkeypatch.setattr(gateway, "authenticator", gateway.GatewayAuthenticator())

    public: dict[str, object] = {
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
    _serve_catalog(monkeypatch, tmp_path, snapshot)
    return TestClient(gateway.app), snapshot


def test_healthcheck_does_not_require_auth(gateway_client: GatewayClient):
    client, _ = gateway_client
    assert client.get("/healthz").json() == {"status": "ok"}


def test_player_api_rejects_invalid_credentials(gateway_client: GatewayClient):
    client, _ = gateway_client
    response = client.get(
        "/player_api.php",
        params={"username": "player", "password": "wrong"},
    )
    assert response.status_code == 200
    assert _body(response, _UserInfo)["user_info"]["auth"] == 0


def test_player_api_exposes_active_server_profile(gateway_client: GatewayClient):
    client, _ = gateway_client
    response = client.get(
        "/player_api.php",
        params={"username": "player", "password": "local-pass"},
    )
    assert response.status_code == 200
    assert _body(response, _UserInfo)["user_info"] == {
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
    gateway_client: GatewayClient,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    monkeypatch.setenv("NETV_GATEWAY_PUBLIC_URL", "https://tv.example:8443")

    response = client.get(
        "/player_api.php",
        params={"username": "player", "password": "local-pass"},
    )

    server_info = _body(response, _Profile)["server_info"]
    assert server_info == {
        "url": "tv.example",
        "port": "8443",
        "https_port": "8443",
        "server_protocol": "https",
        "rtmp_port": "0",
        "timezone": "UTC",
        "timestamp_now": server_info["timestamp_now"],
        "time_now": server_info["time_now"],
    }


def test_successful_authentication_is_cached(gateway_client: GatewayClient):
    client, _ = gateway_client
    params = {"username": "player", "password": "local-pass"}
    assert _body(client.get("/player_api.php", params=params), _UserInfo)["user_info"]["auth"] == 1
    assert auth.change_password("player", "new-pass")

    # Within the cache window the old password is not checked again.
    assert _body(client.get("/player_api.php", params=params), _UserInfo)["user_info"]["auth"] == 1


def test_failed_authentication_is_rate_limited(gateway_client: GatewayClient):
    client, _ = gateway_client
    params = {"username": "player", "password": "wrong"}
    gateway.authenticator.max_login_failures = 2

    for _ in range(2):
        assert client.get("/player_api.php", params=params).status_code == 200

    assert client.get("/player_api.php", params=params).status_code == 429


def test_success_cache_is_checked_before_client_rate_limit(gateway_client: GatewayClient):
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


def test_player_api_exposes_local_live_catalog(gateway_client: GatewayClient):
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


def test_player_api_orders_most_viewed_streams_within_category(gateway_client: GatewayClient):
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

    assert [stream["stream_id"] for stream in _records(response)] == [42, 41]

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
    assert [stream["stream_id"] for stream in _records(response)] == [41, 42]


def test_playlist_contains_only_local_playback_urls(gateway_client: GatewayClient):
    client, _ = gateway_client
    response = client.get(
        "/get.php",
        params={"username": "player", "password": "local-pass", "output": "ts"},
    )
    assert response.status_code == 200
    assert "http://testserver/live/player/local-pass/41.ts" in response.text
    assert "provider.example/live" not in response.text
    assert 'group-title="News"' in response.text
    assert 'url-tvg="http://testserver/xmltv.php?username=player&password=local-pass"' in (
        response.text
    )


def test_upscale_setting_controls_gateway_mode(engines: Path):
    (engines / "retired-model_720p_fp16.engine").write_bytes(b"engine")
    _set_upscale(sr_model="", max_resolution="4k")

    assert gateway._upscale_enabled() is False
    assert gateway._available_upscale_models() == [MODEL]

    _set_upscale(sr_model="retired-model", max_resolution="4k")

    assert gateway._upscale_enabled() is True
    assert gateway._upscale_settings().get("sr_model") == MODEL
    assert gateway._upscale_settings().get("max_resolution") == "4k"


def test_disabled_upscale_accepts_480p_max_resolution(engines: Path):
    _set_upscale(sr_model="", max_resolution="480p")

    assert gateway._upscale_enabled() is False
    assert gateway._upscale_settings().get("max_resolution") == "480p"


def test_upscale_playlist_uses_hls_urls(gateway_client: GatewayClient):
    client, _ = gateway_client
    _set_upscale(sr_model=MODEL)

    response = client.get(
        "/get.php",
        params={"username": "player", "password": "local-pass", "output": "m3u8"},
    )

    assert response.status_code == 200
    assert "http://testserver/live/player/local-pass/41.m3u8" in response.text


def test_upscale_live_stream_starts_session(
    gateway_client: GatewayClient,
    sessions: dict[str, TranscodeSession],
    tmp_path: Path,
):
    client, _ = gateway_client
    _set_upscale(sr_model=MODEL)
    # The player already watches this channel, so its running live session is reused.
    running = _session(tmp_path / "upscale", url=UPSTREAM)
    running.update(process=testing.FakeProcess(exit_code=None), fast_start=True)
    sessions["upscale-session"] = running
    transcode._url_to_session[UPSTREAM] = "upscale-session"

    response = client.get(
        "/live/player/local-pass/41.m3u8",
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["location"] == "/upscale/upscale-session/low.m3u8"
    assert cache.load_live_view_counts("player") == {"src_1:987": 1}


def test_upscale_files_are_served_from_session(
    gateway_client: GatewayClient,
    sessions: dict[str, TranscodeSession],
    tmp_path: Path,
):
    client, _ = gateway_client
    sessions["session"] = _session(tmp_path / "session", url=UPSTREAM)
    (tmp_path / "session" / "stream.m3u8").write_text("#EXTM3U\n#EXTINF:2.0,\nsegment000.ts\n")
    (tmp_path / "session" / "segment000.ts").write_bytes(b"video")

    manifest = client.get("/upscale/session/stream.m3u8")
    segment = client.get("/upscale/session/segment000.ts")

    assert manifest.status_code == 200
    assert manifest.headers["content-type"].startswith("application/vnd.apple.mpegurl")
    assert manifest.headers["cache-control"] == "no-cache, no-store, must-revalidate"
    assert "segment000.ts" in manifest.text
    assert segment.status_code == 200
    assert segment.headers["content-type"] == "video/mp2t"
    assert segment.content == b"video"
    assert sessions["session"].get("last_access", 0.0) > 0.0


def test_upscale_files_follow_their_owner_s_lan_rule(
    gateway_client: GatewayClient,
    sessions: dict[str, TranscodeSession],
    tmp_path: Path,
):
    del gateway_client
    auth.set_lan_only("player", lan_only=True)
    sessions["session"] = _session(tmp_path / "session", url=UPSTREAM)
    sessions["session"]["last_access"] = 0.0
    (tmp_path / "session" / "segment000.ts").write_bytes(b"video")
    home = TestClient(gateway.app, client=("192.168.1.20", 5000))
    away = TestClient(gateway.app, client=("8.8.8.8", 5000))
    assert home.get("/upscale/session/segment000.ts").status_code == 200
    # Started at home, it neither plays nor stays alive from away.
    sessions["session"]["last_access"] = 0.0
    assert away.get("/upscale/session/segment000.ts").status_code == 403
    assert sessions["session"].get("last_access") == 0.0
    auth.delete_user("player")
    assert home.get("/upscale/session/segment000.ts").status_code == 403


def test_user_category_restrictions_apply_to_gateway(gateway_client: GatewayClient):
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


def test_source_assignment_applies_to_gateway(gateway_client: GatewayClient):
    client, _ = gateway_client
    auth.create_user("guest", "guest-pass")
    _set_sources({"id": "src_1", "name": "A", "type": "xtream", "url": "http://a", "users": []})

    def api(username: str, password: str, action: str) -> list[dict[str, MutablePlainTree]]:
        params = {"username": username, "password": password, "action": action}
        return _records(client.get("/player_api.php", params=params))

    assert api("guest", "guest-pass", "get_live_streams") == []
    assert api("guest", "guest-pass", "get_live_categories") == []
    assert len(api("player", "local-pass", "get_live_streams")) == 1
    assert len(api("player", "local-pass", "get_live_categories")) == 1


def test_uncategorized_streams_hidden_for_restricted_user(gateway_client: GatewayClient):
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
    assert gateway._allowed_category_ids("player", restricted_snapshot) == {"99"}
    assert gateway._visible_streams("player", restricted_snapshot) == [uncategorized]
    assert auth.set_user_limits("player", unavailable_groups=["cat:some-other-category"])

    assert gateway._allowed_category_ids("player", restricted_snapshot) == set()
    assert gateway._visible_streams("player", restricted_snapshot) == []


def test_live_stream_proxies_resolved_upstream(
    gateway_client: GatewayClient,
    web: testing.FakeWeb,
):
    client, _ = gateway_client
    upstream = testing.FakeResponse(b"video-data", headers={"Content-Type": "video/mp2t"})
    web.routes[UPSTREAM] = upstream

    response = client.get("/live/player/local-pass/41.ts")

    assert response.status_code == 200
    assert response.content == b"video-data"
    assert response.headers["content-type"] == "video/mp2t"
    assert web.urls == [UPSTREAM]
    assert upstream.closed
    assert cache.load_live_view_counts("player") == {"src_1:987": 1}


def test_live_stream_enforces_category_restrictions(
    gateway_client: GatewayClient,
    web: testing.FakeWeb,
):
    client, _ = gateway_client
    assert auth.set_user_limits("player", unavailable_groups=["cat:src_1_news"])

    response = client.get("/live/player/local-pass/41.ts")

    assert response.status_code == 404
    assert web.fetches == []


def test_xmltv_builds_feed_for_visible_streams(gateway_client: GatewayClient):
    client, _ = gateway_client
    epg.init(cache.CACHE_DIR)
    try:
        epg.insert_channel("news.one", "News One HD", "src_1")
        epg.commit()
    finally:
        db.close()

    response = client.get(
        "/xmltv.php",
        params={"username": "player", "password": "local-pass"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert response.headers["cache-control"] == "private, max-age=300"
    assert '<channel id="news.one"><display-name>News One HD</display-name>' in response.text


def test_xmltv_enforces_category_restrictions(gateway_client: GatewayClient):
    client, _ = gateway_client
    assert auth.set_user_limits("player", unavailable_groups=["cat:src_1_news"])

    response = client.get(
        "/xmltv.php",
        params={"username": "player", "password": "local-pass"},
    )

    # No visible channel, so the feed is empty even without a guide database.
    assert response.status_code == 200
    assert "<channel" not in response.text


def test_playlist_accepts_mpegts_output(gateway_client: GatewayClient):
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
    gateway_client: GatewayClient,
    web: testing.FakeWeb,
):
    client, _ = gateway_client
    auth.create_user("slash", "local/pass")
    web.routes[UPSTREAM] = testing.FakeResponse(b"video-data")

    response = client.get("/live/slash/local%2Fpass/41.ts")

    assert response.status_code == 200
    assert response.content == b"video-data"


def _set_upscale(
    *,
    sr_model: str,
    max_resolution: cache.MaxResolution = "1080p",
    quality: str = "high",
) -> None:
    settings = cache.load_server_settings()
    settings["sr_model"] = sr_model
    settings["max_resolution"] = max_resolution
    settings["quality"] = quality
    cache.save_server_settings(settings)


def _set_sources(*sources: SourceConfig) -> None:
    settings = cache.load_server_settings()
    settings["sources"] = list(sources)
    cache.save_server_settings(settings)


def _source() -> SourceConfig:
    return {
        "id": "src_1",
        "name": "Provider",
        "type": "xtream",
        "url": "https://provider.example",
        "username": "remote-user",
        "password": "remote-pass",
    }


def _session(
    directory: Path,
    *,
    url: str,
    username: str = "player",
    source_id: str = "src_1",
    started: float = 0.0,
) -> TranscodeSession:
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "dir": str(directory),
        "process": None,
        "started": started,
        "last_access": time.time(),
        "url": url,
        "username": username,
        "source_id": source_id,
    }


def _serve_catalog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    snapshot: CatalogSnapshot,
) -> None:
    """Serve ``snapshot`` from a real catalog, as if it had just been loaded."""
    catalog = GatewayCatalog(registry=StreamIdRegistry(tmp_path / "ids.json"))
    catalog._snapshot = snapshot
    catalog._loaded_at = time.monotonic()
    monkeypatch.setattr(gateway, "catalog", catalog)


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
            "client": None if host is None or host == "" else (host, 0),
        },
    )


def _params(**extra: str | int) -> dict[str, str | int]:
    return {"username": "player", "password": "local-pass", **extra}


def test_upscale_settings_rejects_missing_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path / "missing"))
    _set_upscale(sr_model="x")

    with pytest.raises(
        RuntimeError,
        match=f"^Configured AI Upscale model '{MODEL}' is not installed$",
    ):
        gateway._upscale_settings()


@pytest.mark.parametrize("quality", ["ultra", "LOW"])
def test_upscale_settings_rejects_invalid_quality(engines: Path, quality: str):
    _set_upscale(sr_model="", quality=quality)

    with pytest.raises(
        RuntimeError,
        match=r"^Configured quality must be one of: high, medium, low$",
    ):
        gateway._upscale_settings()


@pytest.mark.parametrize("max_resolution", ["8k", "720P", "1440p"])
def test_unsupported_max_resolution_fails_to_load(max_resolution: str):
    cache.SERVER_SETTINGS_FILE.write_text(json.dumps({"max_resolution": max_resolution}))

    with pytest.raises(ReadError, match="max_resolution"):
        gateway._upscale_settings()


@pytest.mark.parametrize(
    ("max_resolution", "quality"),
    [("4k", "high"), ("1080p", "medium"), ("720p", "low"), ("480p", "high")],
)
def test_upscale_settings_accepts_supported_values(
    engines: Path,
    max_resolution: cache.MaxResolution,
    quality: str,
):
    _set_upscale(sr_model="", max_resolution=max_resolution, quality=quality)

    settings = gateway._upscale_settings()

    assert (settings.get("max_resolution"), settings.get("quality")) == (max_resolution, quality)


def test_upscale_settings_fills_defaults(engines: Path):
    settings = cache.load_server_settings()

    assert gateway._upscale_settings() == {
        **settings,
        "max_resolution": "1080p",
        "quality": "high",
        "sr_model": "",
    }

    _set_upscale(sr_model="", quality="")
    assert gateway._upscale_settings().get("quality") == "high"


@pytest.mark.asyncio
async def test_cleanup_upscale_sessions_runs_every_minute(
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, TranscodeSession],
    tmp_path: Path,
):
    clock = testing.FakeClock()
    testing.install_clock(monkeypatch, clock, gateway)
    expired = _session(tmp_path / "expired", url="http://up/old")
    expired["last_access"] = 0.0
    sessions["expired"] = expired
    ticks: list[float] = []
    second_tick, never = asyncio.Event(), asyncio.Event()

    async def tick(seconds: float) -> None:
        ticks.append(seconds)
        if len(ticks) == 2:
            second_tick.set()
            await never.wait()

    task = asyncio.ensure_future(gateway._cleanup_upscale_sessions(sleep=tick))
    # The second tick follows the first pass, which removes the directory on a thread.
    await asyncio.wait_for(second_tick.wait(), timeout=5)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert sessions == {}
    assert not (tmp_path / "expired").exists()
    assert ticks == [60, 60]


@pytest.mark.parametrize(
    ("sr_model", "message"),
    [
        (MODEL, f"Native upscaling enabled: model={MODEL}"),
        ("", "Native upscaling disabled"),
    ],
)
def test_lifespan_initializes_and_shuts_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    engines: Path,
    sessions: dict[str, TranscodeSession],
    sr_model: str,
    message: str,
):
    for name in ("_load_settings", "_sr_engine_dir"):
        monkeypatch.setattr(commands, name, getattr(commands, name))
    transcodes = tmp_path / "transcodes"
    (transcodes / "netv_transcode_orphan").mkdir(parents=True)
    _set_upscale(sr_model=sr_model)
    settings = cache.load_server_settings()
    settings["transcode_dir"] = str(transcodes)
    cache.save_server_settings(settings)
    sessions["running"] = _session(tmp_path / "running", url="http://up/live")

    with caplog.at_level("INFO", logger="netv.gateway"), TestClient(gateway.app) as client:
        assert not (transcodes / "netv_transcode_orphan").exists()
        assert commands.get_settings().get("sr_model") == sr_model
        assert commands._sr_engine_dir == str(engines)
        assert client.get("/healthz").status_code == 200
        assert "running" in sessions

    assert sessions == {}
    assert message in caplog.text


def _authenticator_user(monkeypatch: pytest.MonkeyPatch, now: float = 1000.0) -> None:
    auth.create_user("u", "p")
    testing.install_clock(monkeypatch, testing.FakeClock(now=now), gateway)


def test_authenticator_evicts_expired_success_entries(monkeypatch: pytest.MonkeyPatch):
    _authenticator_user(monkeypatch)
    authenticator = gateway.GatewayAuthenticator(max_auth_cache_entries=2)
    authenticator._success_cache = {"expired": 1000.0, "live": 1001.0}

    assert asyncio.run(authenticator.verify(_request("1.1.1.1"), "u", "p"))

    fingerprint = gateway.GatewayAuthenticator._fingerprint("u", "p")
    assert set(authenticator._success_cache) == {"live", fingerprint}
    assert authenticator._success_cache[fingerprint] == 1030.0


def test_authenticator_admits_a_lan_only_user_only_from_home(monkeypatch: pytest.MonkeyPatch):
    _authenticator_user(monkeypatch)
    auth.set_lan_only("u", lan_only=True)
    authenticator = gateway.GatewayAuthenticator()
    assert asyncio.run(authenticator.verify(_request("192.168.1.20"), "u", "p"))
    # A success cached at home does not admit the same password from away.
    assert not asyncio.run(authenticator.verify(_request("8.8.8.8"), "u", "p"))
    assert not asyncio.run(authenticator.verify(_request(None), "u", "p"))
    # Refused from away like a wrong password: it counts toward the limit, and the
    # password is checked all the same, so timing does not mark LAN-only accounts.
    checked: list[str] = []
    real = auth.verify_password

    def verify_password(username: str, password: str) -> bool:
        checked.append(username)
        return real(username, password)

    monkeypatch.setattr(auth, "verify_password", verify_password)
    assert not asyncio.run(authenticator.verify(_request("8.8.4.4"), "u", "p"))
    assert checked == ["u"]
    assert len(authenticator._failures["8.8.4.4"]) == 1


def test_authenticator_evicts_oldest_success_entry_when_full(monkeypatch: pytest.MonkeyPatch):
    _authenticator_user(monkeypatch)
    authenticator = gateway.GatewayAuthenticator(max_auth_cache_entries=2)
    authenticator._success_cache = {"expired": 1000.0, "old": 1001.0, "new": 1002.0}

    assert asyncio.run(authenticator.verify(_request(None), "u", "p"))

    fingerprint = gateway.GatewayAuthenticator._fingerprint("u", "p")
    assert set(authenticator._success_cache) == {"new", fingerprint}


def test_authenticator_bounds_failure_clients(monkeypatch: pytest.MonkeyPatch):
    _authenticator_user(monkeypatch)
    authenticator = gateway.GatewayAuthenticator(max_failure_clients=1)

    assert not asyncio.run(authenticator.verify(_request("1.1.1.1"), "u", "bad"))
    assert not asyncio.run(authenticator.verify(_request("2.2.2.2"), "u", "bad"))
    assert not asyncio.run(authenticator.verify(_request("2.2.2.2"), "u", "bad"))

    assert list(authenticator._failures) == ["2.2.2.2"]
    assert len(authenticator._failures["2.2.2.2"]) == 2
    assert len(authenticator._account_failures["u"]) == 3


def test_authenticator_limits_an_account_guessed_from_away(monkeypatch: pytest.MonkeyPatch):
    _authenticator_user(monkeypatch)
    auth.create_user("v", "q")
    authenticator = gateway.GatewayAuthenticator(max_account_failures=2)
    # Guesses from many addresses add up against the account they target.
    for ip in ("1.1.1.1", "2.2.2.2"):
        assert not asyncio.run(authenticator.verify(_request(ip), "u", "bad"))
    with pytest.raises(gateway.HTTPException) as exc_info:
        asyncio.run(authenticator.verify(_request("3.3.3.3"), "u", "p"))
    assert exc_info.value.status_code == 429
    # No one locks everyone out: other accounts sign in, and so does this one at home.
    assert asyncio.run(authenticator.verify(_request("4.4.4.4"), "v", "q"))
    assert asyncio.run(authenticator.verify(_request("192.168.1.5"), "u", "p"))
    # Nor do failures at home count against the account.
    for _ in range(3):
        assert not asyncio.run(authenticator.verify(_request("192.168.1.6"), "v", "bad"))
    assert asyncio.run(authenticator.verify(_request("5.5.5.5"), "v", "q"))


@pytest.mark.parametrize("now", [0.0, 1000.0])
def test_authenticator_forgets_failures_outside_window(
    monkeypatch: pytest.MonkeyPatch,
    now: float,
):
    _authenticator_user(monkeypatch, now)
    authenticator = gateway.GatewayAuthenticator(max_login_failures=1)
    cutoff = now - authenticator.login_window_seconds
    authenticator._failures = {
        "1.1.1.1": [cutoff - 1, cutoff],
        "2.2.2.2": [cutoff - 1, cutoff, cutoff + 1],
    }
    authenticator._account_failures = {"u": [cutoff - 1, cutoff], "w": [cutoff + 1]}

    assert asyncio.run(authenticator.verify(_request("3.3.3.3"), "u", "p"))

    assert authenticator._failures == {"2.2.2.2": [cutoff + 1]}
    assert authenticator._account_failures == {"w": [cutoff + 1]}


def test_healthcheck_reports_upscale_mode(gateway_client: GatewayClient):
    client, _ = gateway_client
    _set_upscale(sr_model=MODEL)

    assert client.get("/healthz").json() == {"status": "ok", "mode": "upscale"}


@pytest.mark.parametrize(
    ("sr_model", "expected"),
    [
        (
            "on",
            {
                "upscale": True,
                "models": [MODEL],
                "selected_model": MODEL,
                "max_resolution": "4k",
                "quality": "medium",
                "output_format": "hls",
            },
        ),
        (
            "",
            {
                "upscale": False,
                "models": [MODEL],
                "selected_model": "",
                "max_resolution": "4k",
                "quality": "medium",
                "output_format": "mpegts",
            },
        ),
    ],
)
def test_capabilities(
    gateway_client: GatewayClient,
    sr_model: str,
    expected: dict[str, object],
):
    client, _ = gateway_client
    _set_upscale(sr_model=sr_model, max_resolution="4k", quality="medium")

    assert client.get("/capabilities").json() == expected


def test_available_upscale_models_requires_matching_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path))
    (tmp_path / "other_720p_fp16.engine").write_bytes(b"engine")

    assert gateway._available_upscale_models() == []


def test_player_api_live_categories_include_unmapped_category(gateway_client: GatewayClient):
    client, snapshot = gateway_client
    snapshot.categories.append({"category_id": "77", "category_name": "Misc", "parent_id": 0})

    response = client.get("/player_api.php", params=_params(action="get_live_categories"))

    assert [category["category_id"] for category in _records(response)] == ["src_1_news", "77"]


def test_player_api_hides_unmapped_category_for_restricted_user(gateway_client: GatewayClient):
    client, snapshot = gateway_client
    snapshot.categories.append({"category_id": "77", "category_name": "Misc", "parent_id": 0})
    assert auth.set_user_limits("player", unavailable_groups=["cat:other"])

    response = client.get("/player_api.php", params=_params(action="get_live_categories"))

    assert [category["category_id"] for category in _records(response)] == ["src_1_news"]


@pytest.mark.parametrize(
    "action",
    ["get_vod_categories", "get_vod_streams", "get_series_categories", "get_series"],
)
def test_player_api_returns_empty_vod_and_series(gateway_client: GatewayClient, action: str):
    client, _ = gateway_client

    assert client.get("/player_api.php", params=_params(action=action)).json() == []


def test_player_api_rejects_unsupported_action(gateway_client: GatewayClient):
    client, _ = gateway_client

    response = client.get("/player_api.php", params=_params(action="bogus"))

    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported action: bogus"}


def test_short_epg_requires_stream_id(gateway_client: GatewayClient):
    client, _ = gateway_client

    response = client.get("/player_api.php", params=_params(action="get_short_epg"))

    assert response.status_code == 400
    assert response.json() == {"detail": "stream_id is required"}


def test_short_epg_rejects_unknown_stream(gateway_client: GatewayClient):
    client, _ = gateway_client

    response = client.get(
        "/player_api.php",
        params=_params(action="get_short_epg", stream_id=999),
    )

    assert response.status_code == 404


def test_short_epg_rejects_hidden_stream(gateway_client: GatewayClient):
    client, _ = gateway_client
    assert auth.set_user_limits("player", unavailable_groups=["cat:src_1_news"])

    response = client.get(
        "/player_api.php",
        params=_params(action="get_short_epg", stream_id=41),
    )

    assert response.status_code == 404


def test_short_epg_without_source_client_is_empty(
    gateway_client: GatewayClient,
    web: testing.FakeWeb,
):
    client, _ = gateway_client
    _set_sources()

    response = client.get(
        "/player_api.php",
        params=_params(action="get_short_epg", stream_id=41),
    )

    assert response.json() == {"epg_listings": []}
    assert web.fetches == []


@pytest.mark.parametrize(("limit", "expected_limit"), [(500, 100), (0, 1), (5, 5)])
def test_short_epg_proxies_upstream_with_clamped_limit(
    gateway_client: GatewayClient,
    web: testing.FakeWeb,
    limit: int,
    expected_limit: int,
):
    client, _ = gateway_client
    url = f"{API}&action=get_short_epg&stream_id=987&limit={expected_limit}"
    web.routes[url] = testing.FakeResponse(b'{"epg_listings": [{"title": "News"}]}')

    response = client.get(
        "/player_api.php",
        params=_params(action="get_short_epg", stream_id=41, limit=limit),
    )

    assert response.json() == {"epg_listings": [{"title": "News"}]}
    assert web.urls == [url]


def test_short_epg_upstream_failure_is_bad_gateway(
    gateway_client: GatewayClient,
    web: testing.FakeWeb,
    caplog: pytest.LogCaptureFixture,
):
    client, _ = gateway_client
    web.routes[f"{API}&action=get_short_epg&stream_id=987&limit=10"] = error.URLError("boom")

    response = client.get(
        "/player_api.php",
        params=_params(action="get_short_epg", stream_id=41),
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Unable to load upstream EPG data"}
    assert "Gateway EPG lookup failed (URLError)" in caplog.text


def test_player_api_orders_by_views_preserving_category_order(gateway_client: GatewayClient):
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

    assert [stream["stream_id"] for stream in _records(response)] == [43, 41, 42]


def test_playlist_rejects_invalid_credentials(gateway_client: GatewayClient):
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
def test_playlist_rejects_unsupported_options(
    gateway_client: GatewayClient,
    params: dict[str, str],
    detail: str,
):
    client, _ = gateway_client

    response = client.get("/get.php", params=_params(**params))

    assert response.status_code == 400
    assert response.json() == {"detail": detail}


def test_playlist_sanitizes_values_and_defaults_group(gateway_client: GatewayClient):
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


def test_live_stream_rejects_malformed_path(gateway_client: GatewayClient):
    client, _ = gateway_client

    for path in ("/live/player", "/live/player/pass", "/live/player/pass/41", "/live/u/p/x.ts"):
        response = client.get(path)
        assert response.status_code == 400, path
        assert response.json() == {"detail": "Invalid live stream path"}


def test_live_stream_rejects_invalid_credentials(gateway_client: GatewayClient):
    client, _ = gateway_client

    assert client.get("/live/player/wrong/41.ts").status_code == 401


def test_live_stream_rejects_unsupported_format(gateway_client: GatewayClient):
    client, _ = gateway_client

    response = client.get("/live/player/local-pass/41.mp4")

    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported stream format"}


def test_live_stream_rejects_unknown_stream(gateway_client: GatewayClient):
    client, _ = gateway_client

    assert client.get("/live/player/local-pass/999.ts").status_code == 404


def _start_upscale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    client: TestClient,
    web: testing.FakeWeb,
    launcher: testing.ProcessLauncher | None = None,
    probe_json: str = testing.probed_video_json(),
) -> tuple[testing.ProcessLauncher, int]:
    """Request the upscaled stream, by default with a scripted ffmpeg that exits at once."""
    _set_upscale(sr_model=MODEL)
    settings = cache.load_server_settings()
    settings["transcode_dir"] = str(tmp_path / "transcodes")
    cache.save_server_settings(settings)
    web.routes[UPSTREAM] = testing.ended_upstream
    ffprobe = testing.FakeRun()
    ffprobe.program("ffprobe", testing.Program(stdout=probe_json))
    testing.install_subprocess(monkeypatch, probe, commands, run=ffprobe)
    if launcher is None:
        launcher = testing.ProcessLauncher([testing.FakeProcess(exited=True, exit_code=1)])
    testing.install_launcher(monkeypatch, launcher, transcode)
    testing.install_clock(monkeypatch, testing.FakeClock(), transcode)
    tasks = testing.BackgroundTasks()
    monkeypatch.setattr(transcode, "_spawn_background_task", tasks)
    response = client.get("/live/player/local-pass/41.ts", follow_redirects=False)
    tasks.discard()
    return launcher, response.status_code


def _publish_rendition(command: list[str], name: str = "low", first: int = 0) -> None:
    """Write two playable 2 s segments from number ``first`` (PTS 10 + 2 * number)."""
    directory = Path(command[-1]).parent
    lines = ["#EXTM3U"]
    for i in range(first, first + 2):
        (directory / f"{name}_{i}.ts").write_bytes(testing.video_packet(10 + 2 * i) * 10)
        lines += ["#EXTINF:2,", f"{name}_{i}.ts"]
    (directory / f"{name}.m3u8").write_text("\n".join(lines) + "\n")


def test_upscale_playlist_follows_the_rendition_players_should_see(
    gateway_client: GatewayClient,
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, TranscodeSession],
    tmp_path: Path,
    web: testing.FakeWeb,
):
    client, _ = gateway_client
    launcher = testing.ProcessLauncher(on_launch=_publish_rendition)

    _, status = _start_upscale(monkeypatch, tmp_path, client, web, launcher)

    assert status == 302
    [(session_id, session)] = sessions.items()
    playlist = f"/upscale/{session_id}/low.m3u8"
    plain = client.get(playlist).text
    assert _segment_uris(plain) == ["low_0.ts", "low_1.ts"]
    # The upscaler caught up and took over, as _upgrade_to_upscale leaves it; players
    # keep requesting low.m3u8 and continue into it where the plain playlist ended.
    _publish_rendition(launcher.commands[0], "up", first=1)
    assert transcode._is_fast_live(session)
    session["rendition"] = "up"
    session["splice_low"] = plain
    spliced = client.get(playlist)
    assert spliced.status_code == 200
    assert _segment_uris(spliced.text) == ["low_0.ts", "low_1.ts", "up_2.ts"]
    assert spliced.text.index("low_1.ts") < spliced.text.index("#EXT-X-DISCONTINUITY")
    assert spliced.text.index("#EXT-X-DISCONTINUITY") < spliced.text.index("up_2.ts")


def _segment_uris(playlist: str) -> list[str]:
    return [line for line in playlist.splitlines() if line and not line.startswith("#")]


def test_upscale_live_stream_transcodes_resolved_upstream(
    gateway_client: GatewayClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    web: testing.FakeWeb,
):
    client, _ = gateway_client

    interlaced = '{"streams": [{"codec_type": "video", "codec_name": "mpeg2video", "height": 1080, "field_order": "tt"}]}'
    launcher, status = _start_upscale(monkeypatch, tmp_path, client, web, probe_json=interlaced)

    # The scripted ffmpeg exited at once, so the start itself fails.
    assert status == 500
    # The session's relay reads the resolved upstream; the worker reads the relay.
    assert UPSTREAM in web.urls
    [command] = launcher.commands
    assert UPSTREAM not in command
    assert command[command.index("-i") + 1] == "pipe:0"
    # The probe found interlaced video, so the worker deinterlaces it.
    assert any("yadif" in arg for arg in command)


def test_upscale_live_stream_applies_source_settings(
    gateway_client: GatewayClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    web: testing.FakeWeb,
):
    client, _ = gateway_client
    _set_sources({**_source(), "max_streams": 3, "deinterlace_fallback": False})

    # Without a probe of the video, the source's setting decides deinterlacing.
    no_video = '{"streams": []}'
    launcher, status = _start_upscale(monkeypatch, tmp_path, client, web, probe_json=no_video)

    assert status == 500
    [command] = launcher.commands
    assert not any("yadif" in arg for arg in command)


def test_upscale_live_stream_applies_source_limit(
    gateway_client: GatewayClient,
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, TranscodeSession],
    tmp_path: Path,
    web: testing.FakeWeb,
):
    client, _ = gateway_client
    _set_sources({**_source(), "max_streams": 3})
    for name in ("a", "b", "c"):
        sessions[name] = _session(tmp_path / name, url=f"http://up/{name}", username="other")

    launcher, status = _start_upscale(monkeypatch, tmp_path, client, web)

    assert status == 429
    assert launcher.launches == []
    assert set(sessions) == {"a", "b", "c"}


def test_upscale_live_stream_applies_user_limit(
    gateway_client: GatewayClient,
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, TranscodeSession],
    tmp_path: Path,
    web: testing.FakeWeb,
):
    client, _ = gateway_client
    assert auth.set_user_limits("player", max_streams_per_source={"src_1": 2})
    sessions["old"] = _session(tmp_path / "old", url="http://up/old", started=1.0)
    sessions["new"] = _session(tmp_path / "new", url="http://up/new", started=2.0)

    launcher, _ = _start_upscale(monkeypatch, tmp_path, client, web)

    # The user's limit of two stopped their oldest stream to make room.
    assert "old" not in sessions
    assert "new" in sessions
    assert len(launcher.launches) == 1


def test_live_stream_upstream_failure_is_bad_gateway(
    gateway_client: GatewayClient,
    web: testing.FakeWeb,
    caplog: pytest.LogCaptureFixture,
):
    client, _ = gateway_client
    web.routes[UPSTREAM] = error.URLError("refused")

    response = client.get("/live/player/local-pass/41.ts")

    assert response.status_code == 502
    assert response.json() == {"detail": "Unable to connect to upstream stream"}
    # The log names the failure's type, never the URL with its credentials.
    assert [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "netv.gateway"] == [
        ("WARNING", "Gateway upstream connection failed (URLError)"),
    ]


def test_proxy_url_forwards_length_and_encoding(web: testing.FakeWeb):
    settings = cache.load_server_settings()
    settings["user_agent_preset"] = "custom"
    settings["user_agent_custom"] = "agent/1"
    cache.save_server_settings(settings)
    upstream = testing.FakeResponse(
        b"abcde",
        headers={"Content-Length": "5", "Content-Encoding": "gzip"},
    )
    web.routes["http://up"] = upstream

    async def proxy() -> tuple[StreamingResponse, list[object]]:
        response = await gateway._proxy_url("http://up", "s")
        return response, [chunk async for chunk in response.body_iterator]

    response, body = asyncio.run(proxy())

    [fetch] = web.fetches
    assert (fetch.url, fetch.timeout, fetch.headers) == ("http://up", 30, {"User-agent": "agent/1"})
    # A provider's redirect cannot turn the relay on this server or the home network.
    assert testing.dials_only_the_internet(fetch)
    assert response.media_type == "video/mp2t"
    assert response.headers["content-length"] == "5"
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["cache-control"] == "no-store"
    assert body == [b"abcde"]
    assert upstream.closed


def test_proxy_url_omits_absent_length_and_encoding(web: testing.FakeWeb):
    web.routes["http://up"] = testing.FakeResponse(
        b"x",
        headers={"Content-Type": "video/x", "Content-Length": "", "Content-Encoding": ""},
    )

    response = asyncio.run(gateway._proxy_url("http://up", "s"))

    assert response.media_type == "video/x"
    assert "content-encoding" not in response.headers
    assert "content-length" not in response.headers


@pytest.mark.parametrize(
    ("lineup", "home"),
    [("http://192.168.1.24/lineup.m3u", True), ("http://8.8.8.8/list.m3u", False)],
)
def test_live_stream_reaches_a_tuner_only_from_a_home_lineup(
    gateway_client: GatewayClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    web: testing.FakeWeb,
    lineup: str,
    home: bool,
):
    client, snapshot = gateway_client
    tuner = "http://192.168.1.87:5004/auto/v5.1"
    web.routes[tuner] = testing.FakeResponse(b"ts", headers={"Content-Type": "video/mp2t"})
    _set_sources(_source(), {"id": "t", "name": "Tuner", "type": "m3u", "url": lineup})
    stream = GatewayStream(
        local_id=42,
        source_id="t",
        source_type="m3u",
        upstream_id="5.1",
        direct_url=tuner,
        access_group_ids=("src_1_news",),
        public={**snapshot.streams[0].public, "stream_id": 42},
    )
    _serve_catalog(
        monkeypatch,
        tmp_path,
        CatalogSnapshot(
            categories=snapshot.categories,
            category_access_ids=snapshot.category_access_ids,
            streams=[stream],
            streams_by_id={42: stream},
        ),
    )
    assert client.get("/live/player/local-pass/42.ts").status_code == 200
    [fetch] = web.fetches
    vets = testing.address_vets(fetch)
    assert vets and all(vet(ipaddress.ip_address("192.168.1.87")) is home for vet in vets)


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


def test_resolve_upstream_rejects_removed_xtream_source():
    with pytest.raises(gateway.HTTPException) as exc_info:
        gateway._resolve_upstream(_bare_stream("xtream", ""), "ts")
    assert exc_info.value.status_code == 404


def test_upscale_file_rejects_dotted_filename(gateway_client: GatewayClient):
    client, _ = gateway_client

    response = client.get("/upscale/session/..stream.m3u8")

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid filename"}


def test_upscale_file_rejects_unknown_session(gateway_client: GatewayClient):
    client, _ = gateway_client

    response = client.get("/upscale/session/stream.m3u8")

    assert response.status_code == 404
    assert response.json() == {"detail": "Upscale session not found"}


def test_upscale_file_rejects_vanished_session(
    gateway_client: GatewayClient,
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, TranscodeSession],
    tmp_path: Path,
):
    client, _ = gateway_client
    sessions["session"] = _session(tmp_path / "session", url=UPSTREAM)
    # The session ends between the heartbeat and the lookup that follows it.
    ended = testing.ContendedLock(sessions.clear)
    monkeypatch.setattr(
        transcode,
        "_transcode_lock",
        testing.ContendedLock(
            lambda: monkeypatch.setattr(transcode, "_transcode_lock", ended),
        ),
    )

    response = client.get("/upscale/session/stream.m3u8")

    assert response.status_code == 404
    assert response.json() == {"detail": "Upscale session not found"}
    assert ended.acquired == 1


def test_upscale_file_rejects_missing_and_unsupported_files(
    gateway_client: GatewayClient,
    sessions: dict[str, TranscodeSession],
    tmp_path: Path,
):
    client, _ = gateway_client
    sessions["session"] = _session(tmp_path / "session", url=UPSTREAM)
    (tmp_path / "session" / "notes.txt").write_text("x")

    missing = client.get("/upscale/session/segment999.ts")
    unsupported = client.get("/upscale/session/notes.txt")

    assert missing.status_code == 404
    assert missing.json() == {"detail": "File not found"}
    assert unsupported.status_code == 400
    assert unsupported.json() == {"detail": "Unsupported upscale file"}


def test_xmltv_rejects_invalid_credentials(gateway_client: GatewayClient):
    client, _ = gateway_client

    response = client.get("/xmltv.php", params={"username": "player", "password": "wrong"})

    assert response.status_code == 401


def test_xmltv_reports_unavailable_epg(gateway_client: GatewayClient):
    client, _ = gateway_client

    response = client.get("/xmltv.php", params=_params())

    assert response.status_code == 503
    assert response.json() == {"detail": "EPG database is not ready"}


def test_server_info_defaults_ports_from_scheme(
    monkeypatch: pytest.MonkeyPatch,
    engines: Path,
):
    monkeypatch.delenv("NETV_GATEWAY_PUBLIC_URL", raising=False)

    info = gateway._server_info(_request(None), "u", "p")

    assert info["server_info"]["port"] == "80"
    assert info["server_info"]["https_port"] == ""
    assert info["user_info"]["allowed_output_formats"] == ["ts"]


def test_server_info_https_without_port(monkeypatch: pytest.MonkeyPatch, engines: Path):
    monkeypatch.setenv("NETV_GATEWAY_PUBLIC_URL", "https://tv.example/")
    _set_upscale(sr_model=MODEL)

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
    argv: list[str],
    env: dict[str, str],
    expected_level: int,
    uvicorn_level: str,
):
    server = testing.ServerRecorder()
    testing.install_server(monkeypatch, server)
    monkeypatch.setattr(logging.root, "handlers", list(logging.root.handlers))
    monkeypatch.setattr(logging.root, "level", logging.root.level)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delenv("NETV_GATEWAY_PORT", raising=False)
    monkeypatch.setenv("NETV_GATEWAY_TRUSTED_PROXIES", "10.0.0.1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(gateway.__file__, run_name="__main__")

    assert exit_info.value.code == 0
    assert logging.root.level == expected_level
    [handler] = logging.root.handlers
    record = logging.makeLogRecord(
        {"msg": "Stream http://p.example/live/remote-user/remote-pass/1.ts"},
    )
    assert handler.format(record).endswith(" | Stream http://p.example/live/***/***/1.ts")
    [run] = server.runs
    assert run.options["port"] == (9001 if "--port" in argv else 8100)
    assert run.options["log_level"] == uvicorn_level
    assert run.options["host"] == "0.0.0.0"
    assert run.options["forwarded_allow_ips"] == "10.0.0.1"
    assert run.options["proxy_headers"] is True
    assert run.options["access_log"] is False


class _JsonResponse(Protocol):
    @property
    def content(self) -> bytes: ...


def _body(response: _JsonResponse, target: type[T]) -> T:
    return json_loads(response.content, target)


def _records(response: _JsonResponse) -> list[dict[str, MutablePlainTree]]:
    return _body(response, list[dict[str, MutablePlainTree]])


class _UserInfo(TypedDict):
    user_info: dict[str, MutablePlainTree]


class _Profile(_UserInfo):
    server_info: dict[str, MutablePlainTree]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
