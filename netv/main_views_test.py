"""Tests for main.py routes and helpers from the settings refresh API onward."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures
from datetime import datetime, timezone
from pathlib import Path
from typing import SupportsIndex, TypedDict, override
from urllib import error, parse

import asyncio
import json
import logging
import pathlib
import re
import sys
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from treekle import MutablePlainTree, json_loads

import pytest

from netv import (
    auth,
    cache as cache_module,
    chromecast,
    db,
    epg,
    ffmpeg_command,
    ffmpeg_session,
    handoff,
    imdb,
    m3u as m3u_module,
    main,
    main_test,
    playlists,
    testing,
    util,
)
from netv.cache import SourceConfig
from netv.chromecast import CastSession
from netv.ffmpeg_command import MediaInfo, SeriesProbeStats
from netv.main import PlayerInfo
from netv.testing import (
    FakeClock,
    FakeResponse,
    FakeRun,
    FakeWeb,
    Program,
    Route,
    ServerRecorder,
    ThreadRecorder,
    install_clock,
    install_server,
    install_subprocess,
    install_threads,
    install_web,
)
from netv.xtream import ProviderRecord, XtreamClient


mock_deps = main_test.mock_deps
auth_client = main_test.auth_client
clean_handoff = main_test.clean_handoff
transcode_dir = main_test.transcode_dir


@pytest.fixture(autouse=True)
def web(monkeypatch: pytest.MonkeyPatch) -> FakeWeb:
    """Serve every ``safe_urlopen`` fetch from a table, so no test reaches the network."""
    fake = FakeWeb()
    install_web(monkeypatch, fake, util)
    return fake


@pytest.fixture
def threads(monkeypatch: pytest.MonkeyPatch) -> ThreadRecorder:
    recorder = ThreadRecorder()
    install_threads(monkeypatch, recorder, main)
    return recorder


@pytest.fixture
def held_threads(monkeypatch: pytest.MonkeyPatch) -> ThreadRecorder:
    recorder = ThreadRecorder(run_inline=False)
    install_threads(monkeypatch, recorder, main)
    return recorder


@pytest.fixture
def refresh_flags() -> Iterator[set[str]]:
    flags = m3u_module.get_refresh_in_progress()
    flags.clear()
    yield flags
    flags.clear()


@pytest.fixture
def root_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Let ``configure_logging`` reset a private root handler list, not pytest's."""
    level = logging.root.level
    monkeypatch.setattr(logging.root, "handlers", [])
    yield
    for handler in logging.root.handlers:
        handler.close()
    logging.root.setLevel(level)


def _login_as(client: TestClient, username: str, password: str = "password123") -> None:
    auth.create_user(username, password)
    client.cookies.set("token", auth.create_token({"sub": username}))


def _source(
    source_id: str,
    source_type: str,
    *,
    url: str = "http://up",
    epg_url: str = "",
    epg_timeout: int = 120,
    epg_enabled: bool = True,
    deinterlace_fallback: bool = True,
) -> SourceConfig:
    return {
        "id": source_id,
        "name": source_id.upper(),
        "type": source_type,
        "url": url,
        "username": "u",
        "password": "p",
        "epg_url": epg_url,
        "epg_timeout": epg_timeout,
        "epg_enabled": epg_enabled,
        "deinterlace_fallback": deinterlace_fallback,
    }


def _use_sources(*sources: SourceConfig) -> None:
    cache_module.save_server_settings(
        {**cache_module.load_server_settings(), "sources": list(sources)},
    )


def _api(base: str, action: str, **params: int) -> str:
    query = {"username": "u", "password": "p", "action": action}
    query.update({key: str(value) for key, value in params.items()})
    return f"{base}/player_api.php?{parse.urlencode(query)}"


def _json(value: object) -> FakeResponse:
    return FakeResponse(json.dumps(value).encode())


def _xmltv(*programs: tuple[str, float, float]) -> FakeResponse:
    def stamp(ts: float) -> str:
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y%m%d%H%M%S +0000")

    rows = "".join(
        f'<programme start="{stamp(start)}" stop="{stamp(stop)}" channel="ch1">'
        f"<title>{title}</title></programme>"
        for title, start, stop in programs
    )
    return FakeResponse(f"<tv>{rows}</tv>".encode())


def _add_program(title: str, start: float, stop: float, source_id: str = "src1") -> None:
    epg.insert_programs([("ch1", title, start, stop, "", source_id)])
    epg.commit()


def _titles_by_source() -> list[tuple[str, str]]:
    now = time.time()
    programs = epg.get_programs_in_range(
        "ch1",
        datetime.fromtimestamp(now - 86_400, timezone.utc),
        datetime.fromtimestamp(now + 86_400, timezone.utc),
    )
    return sorted((p.source_id, p.title) for p in programs)


def _now() -> float:
    return float(int(time.time()))


# ===========================================================================
# Guide and per-source refresh
# ===========================================================================


def test_guide_refresh_starts_each_refresh_once(
    auth_client: TestClient,
    refresh_flags: set[str],
    held_threads: ThreadRecorder,
):
    response = auth_client.get("/guide/refresh", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/guide?refreshing=1"
    assert refresh_flags == {"live_refresh", "epg_refresh"}
    assert len(held_threads.threads) == 2

    auth_client.get("/guide/refresh", follow_redirects=False)
    assert len(held_threads.threads) == 2


def test_source_refresh_unknown_source(auth_client: TestClient):
    assert auth_client.post("/settings/refresh/nope/live").json() == {
        "error": "Source not found",
    }


def test_source_refresh_already_running(auth_client: TestClient, refresh_flags: set[str]):
    refresh_flags.add("s1_live")
    _use_sources(_source("s1", "xtream"))
    assert auth_client.post("/settings/refresh/s1/live").json() == {
        "status": "already_running",
    }


@pytest.mark.parametrize("epg_enabled", [True, False])
def test_source_refresh_live_replaces_only_that_source(
    auth_client: TestClient,
    refresh_flags: set[str],
    threads: ThreadRecorder,
    web: FakeWeb,
    epg_enabled: bool,
):
    del threads
    _use_sources(_source("s1", "xtream", epg_timeout=30, epg_enabled=epg_enabled))
    store = cache_module.get_cache()
    other_cat: ProviderRecord = {"category_id": "o", "source_id": "other"}
    other_stream: ProviderRecord = {"stream_id": "o", "source_id": "other"}
    store["live_categories"] = [other_cat, {"source_id": "s1"}]
    store["live_streams"] = [other_stream, {"source_id": "s1"}]
    store["epg_urls"] = [("http://o", 10, "other"), ("http://old", 10, "s1")]
    web.routes[_api("http://up", "get_live_categories")] = _json([{"category_id": "n"}])
    web.routes[_api("http://up", "get_live_streams")] = _json(
        [{"stream_id": "n", "category_id": "7"}],
    )
    response = auth_client.post("/settings/refresh/s1/live")
    assert response.json() == {"status": "started", "key": "s1_live"}
    new_cat: ProviderRecord = {"category_id": "s1_n", "source_id": "s1"}
    new_stream: ProviderRecord = {
        "stream_id": "n",
        "category_id": "7",
        "source_id": "s1",
        "source_type": "xtream",
        "source_url": "http://up",
        "source_username": "u",
        "source_password": "p",
        "category_ids": ["s1_7"],
    }
    guide = "http://up/xmltv.php?username=u&password=p"
    expected_epg = [("http://o", 10, "other")] + ([(guide, 30, "s1")] if epg_enabled else [])
    assert store["live_categories"] == [other_cat, new_cat]
    assert store["live_streams"] == [other_stream, new_stream]
    assert store["epg_urls"] == expected_epg
    cached = m3u_module.cached_live_data()
    assert cached is not None
    assert cached[0] == {
        "cats": [other_cat, new_cat],
        "streams": [other_stream, new_stream],
        "epg_urls": expected_epg,
    }
    assert cache_module.get_sources()[0].epg_url == guide
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
    guide_db: None,
    auth_client: TestClient,
    refresh_flags: set[str],
    threads: ThreadRecorder,
    web: FakeWeb,
    source: SourceConfig,
    fetched_url: str | None,
):
    del threads, guide_db
    now = _now()
    _use_sources(source)
    _add_program("Old", now - 600, now + 600, "s1")
    _add_program("Kept", now - 600, now + 600, "other")
    if fetched_url is not None:
        web.routes[fetched_url] = _xmltv(("New", now, now + 600))
    auth_client.post("/settings/refresh/s1/epg")
    if fetched_url is not None:
        assert _titles_by_source() == [("other", "Kept"), ("s1", "New")]
        assert [(f.url, f.timeout) for f in web.fetches] == [(fetched_url, 120)]
    else:
        assert _titles_by_source() == [("other", "Kept"), ("s1", "Old")]
        assert web.fetches == []
    assert not refresh_flags


def test_source_refresh_vod_merges_and_drops_file_cache(
    auth_client: TestClient,
    refresh_flags: set[str],
    threads: ThreadRecorder,
    web: FakeWeb,
):
    del threads, refresh_flags
    _use_sources(_source("s1", "xtream"))
    stale = cache_module.CACHE_DIR / "vod_data_x.json"
    stale.write_text("{}")
    cache_module.get_cache()["vod_streams"] = [{"stream_id": "stale"}]
    cache_module.save_file_cache(
        "vod_data",
        {
            "cats": [
                {"category_id": "o", "source_id": "other"},
                {"category_id": "x", "source_id": "s1"},
            ],
            "streams": [
                {"stream_id": "o", "source_id": "other"},
                {"stream_id": "x", "source_id": "s1"},
            ],
        },
    )
    web.routes[_api("http://up", "get_vod_categories")] = _json([{"category_id": "n"}])
    web.routes[_api("http://up", "get_vod_streams")] = _json([{"stream_id": 1}])
    auth_client.post("/settings/refresh/s1/vod")
    assert not stale.exists()
    assert "vod_streams" not in cache_module.get_cache()
    cached = m3u_module.cached_vod_data()
    assert cached is not None
    assert cached[0] == {
        "cats": [
            {"category_id": "o", "source_id": "other"},
            {"category_id": "n", "source_id": "s1"},
        ],
        "streams": [
            {"stream_id": "o", "source_id": "other"},
            {"stream_id": 1, "source_id": "s1"},
        ],
    }


def test_source_refresh_m3u(
    auth_client: TestClient,
    refresh_flags: set[str],
    threads: ThreadRecorder,
    web: FakeWeb,
):
    del threads, refresh_flags
    _use_sources(_source("s1", "m3u"))
    store = cache_module.get_cache()
    store["live_categories"] = [{"source_id": "s1"}]
    store["live_streams"] = [{"source_id": "s1"}, {"source_id": "other"}]
    store["epg_urls"] = [("http://o", 10, "other")]
    web.routes["http://up"] = FakeResponse(
        b'#EXTM3U url-tvg="http://detected"\n'
        b'#EXTINF:-1 group-title="News",One\n'
        b"http://media/1.ts\n",
    )
    auth_client.post("/settings/refresh/s1/m3u")
    assert cache_module.get_sources()[0].epg_url == "http://detected"
    assert store["live_categories"] == [
        {"category_id": "s1_news", "category_name": "News", "parent_id": 0, "source_id": "s1"},
    ]
    assert store["live_streams"] == [
        {"source_id": "other"},
        {
            "stream_id": "s1:1.ts",
            "name": "One",
            "stream_icon": "",
            "epg_channel_id": "",
            "category_ids": ["s1_news"],
            "direct_url": "http://media/1.ts",
            "source_id": "s1",
        },
    ]
    cached = m3u_module.cached_live_data()
    assert cached is not None
    assert cached[0]["epg_urls"] == [("http://o", 10, "other")]


def test_source_refresh_failure_clears_flag(
    auth_client: TestClient,
    refresh_flags: set[str],
    threads: ThreadRecorder,
    web: FakeWeb,
    caplog: pytest.LogCaptureFixture,
):
    del threads
    _use_sources(_source("s1", "xtream"))
    web.routes[_api("http://up", "get_live_categories")] = RuntimeError("boom")
    with caplog.at_level(logging.ERROR, logger="netv.main"):
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


def _cast(username: str, host: str, *, starting: bool) -> CastSession:
    return CastSession(
        username=username,
        host=host,
        session_id=f"{username}-session",
        url="http://tv/stream.m3u8",
        source_url="http://up/1.ts",
        name="TV",
        starting=starting,
    )


def test_cast_status_is_user_scoped(auth_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        chromecast.manager,
        "_sessions",
        {"10.0.0.2": _cast("bob", "10.0.0.2", starting=False)},
    )
    assert auth_client.get("/api/cast/status").json() == {"active": False}
    monkeypatch.setitem(
        chromecast.manager._sessions,
        "10.0.0.3",
        _cast("testuser", "10.0.0.3", starting=True),
    )
    assert auth_client.get("/api/cast/status").json() == {"active": False, "connecting": True}


def test_user_prefs_rejects_large_body(auth_client: TestClient):
    response = auth_client.post("/api/user-prefs", content=b"x" * (64 * 1024 + 1))
    assert response.status_code == 400


def test_category_filter_rejects_non_list(auth_client: TestClient):
    assert auth_client.post("/settings/vod-filter", json={"cats": "x"}).status_code == 400


def test_update_playlist_errors(auth_client: TestClient):
    playlist = playlists.create("P")
    response = auth_client.put(f"/api/playlists/{playlist['id']}", json={"name": ""})
    assert response.status_code == 400
    assert main_test._body(response, main_test._Detail)["detail"] == "Playlist name is required"
    assert playlists.get(playlist["id"]) == playlist
    assert auth_client.put("/api/playlists/missing", json={}).status_code == 404


def test_delete_missing_playlist(auth_client: TestClient):
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


def _logo(url: str, **params: str) -> str:
    """A signed /api/logo request path, as _logo_url_filter would emit it."""
    query = {"url": url, **params, "sig": auth.logo_url_signature(url)}
    return "/api/logo?" + parse.urlencode(query)


def test_logo_requires_url(auth_client: TestClient):
    assert auth_client.get("/api/logo?url=").status_code == 400


def test_logo_served_from_cache(auth_client: TestClient, web: FakeWeb):
    cache_module.save_logo("default", "http://x/logo.png", b"png", "image/png")
    response = auth_client.get(_logo("http://x/logo.png"))
    assert response.content == b"png"
    assert response.headers["cache-control"] == "max-age=86400"
    assert web.fetches == []


def test_logo_rejects_non_http_scheme(auth_client: TestClient):
    assert auth_client.get(_logo("file:///etc/passwd")).status_code == 400


@pytest.mark.parametrize(
    "sig",
    [
        None,
        "",
        "0" * 64,
        "zz" * 32,
        "é" * 64,
        "ABCDEF" * 10 + "abcd",
    ],
)
def test_logo_refuses_unsigned_urls(auth_client: TestClient, web: FakeWeb, sig: str | None):
    # Any viewer could otherwise make the server fetch internal addresses.
    url = "http://127.0.0.1:8080/snapshot.jpg"
    web.routes[url] = FakeResponse(b"img", headers={"Content-Type": "image/png"})
    query = {"url": url} if sig is None else {"url": url, "sig": sig}
    response = auth_client.get("/api/logo?" + parse.urlencode(query))
    assert response.status_code == 403
    assert web.fetches == []


def test_logo_signature_is_bound_to_the_url(auth_client: TestClient, web: FakeWeb):
    signed = auth.logo_url_signature("http://x/l.png")
    response = auth_client.get(
        "/api/logo?" + parse.urlencode({"url": "http://127.0.0.1/l.png", "sig": signed}),
    )
    assert response.status_code == 403
    assert web.fetches == []


def test_logo_refuses_unsigned_urls_even_when_cached(auth_client: TestClient):
    cache_module.save_logo("default", "http://x/logo.png", b"png", "image/png")
    assert auth_client.get("/api/logo?url=http://x/logo.png").status_code == 403


@pytest.mark.parametrize(
    ("rating", "shown"),
    [("8.049", "8.0"), (7.25, "7.2"), (9, "9.0"), ("N/A", "N/A"), ("nan", "nan"), (None, "None")],
)
def test_rating_filter_rounds_to_a_tenth(rating: MutablePlainTree, shown: str):
    assert main._rating_filter(rating) == shown


def test_logo_url_filter_signs_proxy_urls():
    url = "http://logos.example:8080/c.png?x=1"
    proxied = main._logo_url_filter(url)
    query = parse.parse_qs(parse.urlparse(proxied).query)
    assert query["url"] == [url]
    assert query["source"] == ["logos.example"]
    assert auth.verify_logo_url_signature(url, query["sig"][0])
    for local in ("", "/static/x.png", "data:image/png;base64,AA=="):
        assert main._logo_url_filter(local) == local


def test_logo_fetches_and_caches(auth_client: TestClient, web: FakeWeb):
    web.routes["http://x/l.png"] = FakeResponse(b"img", headers={"Content-Type": "image/png"})
    response = auth_client.get(_logo("http://x/l.png", source="src"))
    assert response.content == b"img"
    assert [(f.url, f.timeout) for f in web.fetches] == [("http://x/l.png", 3)]
    saved = cache_module.get_cached_logo("src", "http://x/l.png")
    assert saved is not None
    assert saved.suffix == ".png"
    assert saved.read_bytes() == b"img"


@pytest.mark.parametrize(
    ("route", "status"),
    [
        (FakeResponse(b"<html>", headers={"Content-Type": "text/html"}), 400),
        (None, 502),
        (RuntimeError("weird"), 500),
    ],
)
def test_logo_fetch_errors(auth_client: TestClient, web: FakeWeb, route: Route | None, status: int):
    if route is not None:
        web.routes["https://x/l.png"] = route
    assert auth_client.get(_logo("https://x/l.png")).status_code == status


def test_fetch_logo_returns_image(web: FakeWeb):
    web.routes["http://x"] = FakeResponse(b"ok", headers={"Content-Type": "image/png"})
    assert main._fetch_logo("http://x") == (b"ok", "image/png")
    assert [(f.url, f.timeout) for f in web.fetches] == [("http://x", 3)]
    # A provider names logos, so one may not point the server inward.
    assert testing.dials_only_the_internet(web.fetches[0])


@pytest.mark.parametrize(
    ("content_type", "data", "message"),
    [("text/html", b"", "not an image"), ("image/png", b"xxxx", "too large")],
)
def test_fetch_logo_rejects(web: FakeWeb, content_type: str, data: bytes, message: str):
    web.routes["http://x"] = FakeResponse(data, headers={"Content-Type": content_type})
    with pytest.raises(ValueError, match=message):
        main._fetch_logo("http://x", max_size=4)


# ===========================================================================
# Server settings
# ===========================================================================


def test_transcode_settings_store_custom_dir(
    auth_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path / "engines"))
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


def test_refresh_encoders(auth_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    run = FakeRun()
    run.program("ffmpeg", Program(returncode=1, stderr="Unknown encoder"))
    run.program("ffmpeg", Program(), when="h264_nvenc")
    install_subprocess(monkeypatch, cache_module, run=run)
    monkeypatch.setattr(cache_module, "VAAPI_DEVICE", None)
    assert auth_client.post("/settings/refresh-encoders").json() == {
        "ok": True,
        "encoders": {"nvenc": True, "amf": False, "qsv": False, "vaapi": False},
    }
    assert len(run.calls) == 3


def test_user_agent_invalid_preset_falls_back(auth_client: TestClient):
    response = auth_client.post("/settings/user-agent", data={"preset": "evil", "custom": "UA"})
    assert response.json() == {"ok": True}
    settings = cache_module.load_server_settings()
    assert settings["user_agent_preset"] == "default"
    assert settings["user_agent_custom"] == "UA"


_SHOW: ProviderRecord = {
    "info": {"name": "Show"},
    "episodes": {
        "1": [
            {"id": "11", "episode_num": 2, "title": "S01E02 - Show - Pilot"},
            {"id": None},
        ],
    },
}


def _probe_stats(series_id: int, name: str, *episode_ids: int) -> SeriesProbeStats:
    return {
        "series_id": series_id,
        "name": name,
        "mru": None,
        "episode_count": len(episode_ids),
        "video_codec": "h264",
        "audio_codec": "aac",
        "subtitle_count": 0,
        "episodes": [
            {"episode_id": eid, "duration": 60.0, "subtitle_count": 0} for eid in episode_ids
        ],
    }


def test_probe_cache_route_names_series(
    auth_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    web: FakeWeb,
):
    _use_sources(_source("s1", "xtream"))
    media = MediaInfo(video_codec="h264", audio_codec="aac", pix_fmt="yuv420p", duration=60.0)
    monkeypatch.setattr(
        ffmpeg_command,
        "_series_probe_cache",
        {1: {"name": "", "mru": None, "episodes": {11: (1.0, media, [])}}},
    )
    web.routes[_api("http://up", "get_series_info", series_id=1)] = _json(_SHOW)
    response = auth_client.get("/settings/probe-cache")
    assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"
    [series] = main_test._body(response, _ProbeCache)["series"]
    assert series["name"] == "Show"
    assert series["episodes"] == [
        {"episode_id": 11, "duration": 60.0, "subtitle_count": 0, "name": "S01E02 Pilot"},
    ]


def test_enrich_probe_cache_names_series_and_episodes(web: FakeWeb):
    stats = [
        _probe_stats(1, "", 11, 99),
        _probe_stats(2, "", 21),
        _probe_stats(3, "Known"),
        _probe_stats(4, "Preserved", 11),
    ]
    urls = [_api("http://up", "get_series_info", series_id=i) for i in (1, 2, 4)]
    web.routes.update({urls[0]: _json(_SHOW), urls[1]: RuntimeError("down"), urls[2]: _json(_SHOW)})
    client = XtreamClient(base_url="http://up", username="u", password="p")
    data = main._enrich_probe_cache_stats(stats, client)
    named = {"episode_id": 11, "duration": 60.0, "subtitle_count": 0, "name": "S01E02 Pilot"}
    assert [entry["name"] for entry in data] == ["Show", "", "Known", "Preserved"]
    assert data[0]["episodes"] == [named, {"episode_id": 99, "duration": 60.0, "subtitle_count": 0}]
    assert data[1]["episodes"] == [{"episode_id": 21, "duration": 60.0, "subtitle_count": 0}]
    assert data[3]["episodes"] == [named]
    assert web.urls == urls
    assert [(cache_module.CACHE_DIR / f"series_info_{i}.json").exists() for i in (1, 2, 4)] == [
        True,
        False,
        True,
    ]


def test_clear_series_mru(auth_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        ffmpeg_command,
        "_series_probe_cache",
        {7: {"name": "Show", "mru": 3, "episodes": {}}},
    )
    assert auth_client.post("/settings/probe-cache/clear-mru/7").json() == {"ok": True}
    assert ffmpeg_command._series_probe_cache == {7: {"name": "Show", "episodes": {}}}
    saved = json.loads((cache_module.CACHE_DIR / "series_probe_cache.json").read_text())
    assert saved["7"]["mru"] is None


def test_clear_data_cache(auth_client: TestClient):
    cache_module.save_file_cache("live_data", {})
    cache_module.save_file_cache("series_data", {})
    assert auth_client.post("/settings/data-cache/clear").json() == {
        "ok": True,
        "cleared": 2,
    }
    assert not (cache_module.CACHE_DIR / "live_data.json").exists()
    assert not (cache_module.CACHE_DIR / "series_data.json").exists()


# ===========================================================================
# Handoff API
# ===========================================================================


def test_handoff_watch_caps_position_at_now(auth_client: TestClient, clean_handoff: None):
    del clean_handoff
    response = auth_client.post(
        "/api/handoff/watch",
        json={"stream_id": "s1", "position": 9e12, "handoff_id": "h"},
    )
    assert response.json() == {"status": "ok"}
    watch = handoff._watches["testuser"]
    assert (watch.stream_id, watch.handoff_id) == ("s1", "h")
    assert watch.position <= time.time()


def test_handoff_stop_rejects_hidden_stream(guide_db: None, auth_client: TestClient):
    del guide_db
    cache_module.get_cache()["live_streams"] = [
        {"stream_id": "s", "source_id": "x", "category_ids": ["1"]},
    ]
    auth.set_user_limits("testuser", unavailable_groups=["cat:1"])
    for stream_id in ("s", "missing"):
        response = auth_client.post(
            "/api/handoff/stop",
            json={"stream_id": stream_id, "position": 1},
        )
        assert response.status_code == 404


def test_handoff_dismiss(auth_client: TestClient, clean_handoff: None):
    del clean_handoff
    handoff._records["testuser"] = main_test._handoff_record(mode="catchup", recording_id="")
    assert auth_client.delete("/api/handoff").json() == {"status": "ok"}
    assert "testuser" not in handoff._records


def test_handoff_file_routes(
    auth_client: TestClient,
    clean_handoff: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    del clean_handoff
    monkeypatch.setattr(ffmpeg_command, "_load_settings", cache_module.load_server_settings)
    cache_module.save_server_settings(
        {**cache_module.load_server_settings(), "transcode_dir": str(tmp_path)},
    )
    handoff._records["testuser"] = main_test._handoff_record()
    recording = tmp_path / "netv_handoff_rec1"
    recording.mkdir()
    (recording / "stream.m3u8").write_text("#EXTM3U\n")
    (recording / "stream.txt").write_text("text")
    (recording / "seg.ts").write_bytes(b"ts")
    assert auth_client.get("/handoff/other/stream.m3u8").status_code == 404
    response = auth_client.get("/handoff/rec1/stream.m3u8")
    assert response.text == "#EXTM3U\n"
    assert response.headers["content-type"] == "application/vnd.apple.mpegurl"
    assert auth_client.get("/handoff/rec1/stream.txt").status_code == 404
    response = auth_client.get("/handoff/rec1/seg.ts")
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
    response = auth_client.post("/settings/data-cache/clear")
    assert response.status_code == 403
    assert main_test._body(response, main_test._Detail)["detail"] == "Admin access required"


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
            "lan_only": "on",
            "max_streams_per_source": '{"s1": 2}',
            "unavailable_groups": '["cat:1"]',
        },
    )
    assert response.json() == {"status": "ok"}
    assert auth.is_admin("carol")
    assert not auth.allowed_from("carol", "8.8.8.8")
    assert 'name="lan_only"' in auth_client.get("/settings").text
    limits = auth.get_user_limits("carol")
    assert limits["max_streams_per_source"] == {"s1": 2}
    assert limits["unavailable_groups"] == ["cat:1"]


def test_add_user_ignores_malformed_limits(auth_client: TestClient):
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
    assert not auth.is_admin("dave")
    users = json_loads(cache_module.SERVER_SETTINGS_FILE.read_text(), _StoredUsers)["users"]
    assert set(users["dave"]) == {"password", "admin"}


@pytest.mark.parametrize(
    ("current", "new", "detail"),
    [
        ("testpass123", "short", "Password must be at least 8 characters"),
        ("wrongpass1", "longenough", "Current password is incorrect"),
    ],
)
def test_change_own_password_errors(auth_client: TestClient, current: str, new: str, detail: str):
    response = auth_client.post(
        "/settings/users/password",
        data={"current_password": current, "new_password": new},
    )
    assert response.status_code == 400
    assert main_test._body(response, main_test._Detail)["detail"] == detail
    assert auth.verify_password("testuser", "testpass123")


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


def test_api_keys_are_set_by_admins_and_never_shown(auth_client: TestClient):
    assert auth_client.post(
        "/settings/api-keys",
        data={"service": "tmdb", "key": "s.e.c"},
    ).json() == {
        "status": "ok",
    }
    assert cache_module.load_server_settings().get("api_keys") == {"tmdb": "s.e.c"}
    page = auth_client.get("/settings").text
    section = re.search(r'<div id="api-keys".*?</details>', page, re.DOTALL)
    assert section is not None
    # A stored key shows as a filled password field that can be removed; no badge.
    assert "TMDB" in section.group(0) and "api-key-remove" in section.group(0)
    assert 'placeholder="••••••••••••"' in section.group(0)
    assert "Saved" not in section.group(0)
    # The key itself never leaves the server, on the page or through the settings API.
    assert "s.e.c" not in page
    assert "s.e.c" not in auth_client.get("/api/settings").text
    assert (
        auth_client.post("/settings/api-keys", data={"service": "imdb", "key": "x"}).status_code
        == 400
    )
    # An empty key removes it.
    auth_client.post("/settings/api-keys", data={"service": "tmdb", "key": " "})
    assert cache_module.load_server_settings().get("api_keys") == {}
    page = auth_client.get("/settings").text
    assert "api-key-remove" not in page and 'placeholder="Paste the key"' in page
    assert "Not set" not in page
    auth.create_user("viewer", "password123")
    viewer = TestClient(main.app)
    viewer.cookies.set("token", auth.create_token({"sub": "viewer"}))
    assert (
        viewer.post("/settings/api-keys", data={"service": "tmdb", "key": "x"}).status_code == 403
    )
    assert 'id="api-keys"' not in viewer.get("/settings").text


def test_tmdb_token_prefers_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    assert main._tmdb_token() == ""
    cache_module.save_server_settings(
        {**cache_module.load_server_settings(), "api_keys": {"tmdb": "saved"}},
    )
    assert main._tmdb_token() == "saved"
    monkeypatch.setenv("TMDB_API_KEY", "from-env")
    assert main._tmdb_token() == "from-env"


def test_lan_only_users_sign_in_and_stay_signed_in_only_from_home(auth_client: TestClient):
    auth.create_user("kid", "password123")
    # Only an admin marks a user LAN-only.
    kid = TestClient(main.app, client=("192.168.1.20", 5000))
    kid.cookies.set("token", auth.create_token({"sub": "kid"}))
    assert kid.post("/settings/users/lan/kid", data={"lan_only": "on"}).status_code == 403
    assert auth_client.post("/settings/users/lan/kid", data={"lan_only": "on"}).status_code == 200
    assert auth_client.post("/settings/users/lan/ghost").status_code == 404
    page = auth_client.get("/settings").text
    assert re.search(
        r'<details[^>]*data-username="kid".*?class="lan-only-toggle[^>]*checked',
        page,
        re.DOTALL,
    )

    # From home: signed in, and able to sign in again.
    assert kid.get("/settings", follow_redirects=False).status_code == 200
    login = kid.post(
        "/login",
        data={"username": "kid", "password": "password123"},
        follow_redirects=False,
    )
    assert login.headers["location"] == "/"
    # From away: a session started at home stops working, and signing in is refused.
    away = TestClient(main.app, client=("8.8.8.8", 5000))
    away.cookies.set("token", auth.create_token({"sub": "kid"}))
    assert away.get("/settings", follow_redirects=False).headers["location"].startswith("/login")
    refused = away.post(
        "/login",
        data={"username": "kid", "password": "password123"},
        follow_redirects=False,
    )
    wrong = away.post(
        "/login",
        data={"username": "kid", "password": "not-it-at-all"},
        follow_redirects=False,
    )
    # From away, the right password gets the wrong one's answer: no confirmation.
    assert refused.headers["location"] == wrong.headers["location"] == "/login?error=invalid"
    assert refused.headers.get("set-cookie") is None
    assert "works only at home" in away.get("/login?error=invalid").text
    # And each counts toward the address's limit.
    assert len(main._login_attempts["8.8.8.8"]) == 2

    assert auth_client.post("/settings/users/lan/kid").status_code == 200
    assert away.get("/settings", follow_redirects=False).status_code == 200


def test_an_account_guessed_from_many_addresses_is_locked_only_from_away(
    auth_client: TestClient,
):
    del auth_client
    wrong = {"username": "testuser", "password": "wrong"}
    for i in range(20):
        TestClient(main.app, client=(f"8.8.{i}.1", 5000)).post("/login", data=wrong)
    right = {"username": "testuser", "password": "testpass123"}
    away = TestClient(main.app, client=("9.9.9.9", 5000))
    # Twenty guesses at one account from away lock it from away, even with the password.
    assert away.post("/login", data=right, follow_redirects=False).status_code == 429
    # Its owner still signs in at home, so strangers cannot lock the owner out.
    home = TestClient(main.app, client=("192.168.1.2", 5000))
    assert home.post("/login", data=right, follow_redirects=False).headers["location"] == "/"
    # And failures at home never count against it.
    assert "account:testuser" in main._login_attempts
    for _ in range(3):
        home.post("/login", data=wrong)
    assert len(main._login_attempts["account:testuser"]) == 20


def test_admins_name_the_home_network(auth_client: TestClient):
    saved = auth_client.post(
        "/settings/home-networks",
        data={"networks": "192.168.1.0/24, fd00::/8"},
    )
    assert saved.json() == {"status": "ok"}
    assert auth.home_networks() == ["192.168.1.0/24", "fd00::/8"]
    page = auth_client.get("/settings").text
    assert re.search(r'<input[^>]*name="networks"[^>]*value="192\.168\.1\.0/24, fd00::/8"', page)
    bad = auth_client.post("/settings/home-networks", data={"networks": "192.168.1.0/24 nope"})
    assert bad.status_code == 400 and "nope" in bad.text
    assert auth_client.post("/settings/home-networks", data={"networks": ""}).status_code == 200
    assert auth.home_networks() == []
    _login_as(viewer := TestClient(main.app), "viewer")
    assert viewer.post("/settings/home-networks", data={"networks": "0.0.0.0/0"}).status_code == 403


def test_a_fresh_install_is_set_up_only_from_home():
    assert auth.is_setup_required()
    away = TestClient(main.app, client=("8.8.8.8", 5000))
    form = {"username": "admin", "password": "password123", "confirm": "password123"}
    # The first account is the administrator, so a stranger must not make it.
    refused = away.get("/setup")
    assert refused.status_code == 403
    assert refused.json() == {"detail": "Finish setting up neTV from your home network"}
    assert away.post("/setup", data=form).status_code == 403
    assert auth.is_setup_required()
    home = TestClient(main.app, client=("192.168.1.20", 5000))
    assert home.get("/setup").status_code == 200
    assert home.post("/setup", data=form, follow_redirects=False).status_code == 303
    assert auth.get_all_usernames() == ["admin"]


def test_a_new_password_ends_every_other_session(auth_client: TestClient):
    elsewhere = TestClient(main.app)
    elsewhere.cookies.set("token", auth.create_token({"sub": "testuser"}))
    assert elsewhere.get("/settings", follow_redirects=False).status_code == 200
    changed = auth_client.post(
        "/settings/users/password",
        data={"current_password": "testpass123", "new_password": "newpass456"},
    )
    assert changed.json() == {"status": "ok"}
    # A stolen cookie dies with the old password; the changer stays signed in.
    assert elsewhere.get("/settings", follow_redirects=False).status_code == 303
    assert auth_client.get("/settings", follow_redirects=False).status_code == 200
    # Likewise through the per-user route, which also renews only its caller.
    again = auth_client.post("/settings/users/password/testuser", data={"new_password": "third789"})
    assert again.status_code == 200
    assert auth_client.get("/settings", follow_redirects=False).status_code == 200
    auth.create_user("bob", "password123")
    bob = TestClient(main.app)
    bob.cookies.set("token", auth.create_token({"sub": "bob"}))
    reset = auth_client.post("/settings/users/password/bob", data={"new_password": "reset1234"})
    assert reset.headers.get("set-cookie") is None
    assert bob.get("/settings", follow_redirects=False).status_code == 303


def test_the_last_admin_stays(auth_client: TestClient):
    auth.create_user("kid", "password123")
    auth.set_lan_only("kid", lan_only=True)
    refused = auth_client.post("/settings/users/admin/testuser", data={})
    assert refused.status_code == 400
    assert "another user an admin" in refused.text
    deleted = auth_client.post(
        "/settings/users/delete/testuser",
        data={"password": "testpass123"},
        follow_redirects=False,
    )
    assert deleted.status_code == 400
    assert auth.is_admin("testuser") and not auth.is_admin("kid")


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
    assert main._name_matcher("[", regex=True, whole=False)("[") is False


def test_name_matcher_ignores_case_and_a_whole_name_is_one_of_a_list():
    assert main._name_matcher("PAC", regex=True, whole=False)("Al Pacino")
    assert main._name_matcher("pac", regex=False, whole=False)("Al PACINO")
    whole = main._name_matcher("Al Pacino", regex=False, whole=True)
    assert whole("Robert De Niro, al pacino") and not whole("Al Pacinos")


def test_search_terms_split_fields_from_plain_text():
    assert main._search_terms('cast:pacino writer:"Ann W"') == [
        ("cast", "pacino", False),
        ("writer", "Ann W", True),
    ]
    # Plain text, quotes trimmed, joins the fields; with no field it is the term.
    assert main._search_terms('cast:x "Xmen"') == [("cast", "x", False), ("any", "Xmen", False)]
    assert main._search_terms("") == [("any", "", False)]


def _search_record(name: str, **fields: str) -> ProviderRecord:
    return {"name": name, **fields}


def test_matching_names_sorts_by_name_and_keeps_at_most_the_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    looked_up: list[str] = []

    def resolved_keys(records: list[ProviderRecord]) -> list[str | None]:
        looked_up.append("keys")
        return [None] * len(records)

    def rankings(records: list[ProviderRecord]) -> list[main.imdb.Ranking]:
        looked_up.append("ranks")
        return []

    monkeypatch.setattr(main.imdb, "resolved_keys", resolved_keys)
    monkeypatch.setattr(main.imdb, "rankings", rankings)
    records = [
        _search_record("b", director="Mann"),
        _search_record("_c", director="Mann"),
        _search_record("A", director="Mann"),
        _search_record("d", director="Lean"),
    ]
    terms = [("director", "mann", False)]
    found = main._matching_names(records, terms, 0, regex=False, imdb_credits=False)
    # The provider's own director field counts; names sort without regard to case.
    assert [r["name"] for r in found] == ["_c", "A", "b"]
    assert len(main._matching_names(records, terms, 2, regex=False, imdb_credits=False)) == 2
    # Nothing here needs IMDb: no title is resolved or ranked.
    assert looked_up == []


def test_fact_terms_read_ranges_and_regex_genres(monkeypatch: pytest.MonkeyPatch):
    rank = main.imdb.Ranking(6.0, 1, 1999)

    def rankings(records: list[ProviderRecord]) -> list[main.imdb.Ranking]:
        return [rank] * len(records)

    monkeypatch.setattr(main.imdb, "rankings", rankings)
    record = _search_record("x", genre="Thriller")
    assert main._fact_matches("imdb", "5-6", rank, record, regex=False, whole=False)
    assert main._fact_matches("genre", "thr.ller", rank, record, regex=True, whole=False)
    assert not main._fact_matches("genre", "thr.ller", rank, record, regex=False, whole=False)
    found = main._matching_names(
        [record],
        [("genre", "^thr", False)],
        0,
        regex=True,
        imdb_credits=True,
    )
    assert found == [record]


class _FailingName(str):
    """A name whose truncation fails, as a pathological regex search would."""

    @override
    def __getitem__(self, key: SupportsIndex | slice) -> str:
        del key
        raise RuntimeError("catastrophic")


def test_regex_matches_swallows_errors():
    assert main._regex_matches(re.compile("x"), _FailingName("x")) is False


def test_float_field_rejects_non_scalar():
    assert main._float_field({}, "position") == 0.0
    assert main._float_field({"position": "1.5"}, "position") == 1.5
    with pytest.raises(TypeError, match="position must be a number"):
        main._float_field({"position": [1]}, "position")


def test_unavailable_channel_keeps_only_saved_fields():
    assert main._unavailable_channel({"name": "Old"}) == {"name": "Old", "available": False}


def test_vod_details_without_client_keeps_movie():
    movie: ProviderRecord = {"stream_id": 5}
    assert asyncio.run(main._with_vod_details(movie, "s1", 5)) is movie


def test_vod_details_without_info_keeps_movie(web: FakeWeb):
    _use_sources(_source("s1", "xtream"))
    url = _api("http://up", "get_vod_info", vod_id=5)
    web.routes[url] = _json({"info": {}})
    movie: ProviderRecord = {"stream_id": 5}
    assert asyncio.run(main._with_vod_details(movie, "s1", 5)) is movie
    assert web.urls == [url]


def test_user_prefs_saves_each_field(auth_client: TestClient):
    prefs: dict[str, MutablePlainTree] = {
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


class _FavoriteEntry(TypedDict):
    cover: str


class _FavoritePrefs(TypedDict):
    favorites: dict[str, dict[str, _FavoriteEntry]]


def _favorite_covers(auth_client: TestClient) -> dict[str, dict[str, str]]:
    """Each favorite's cover, as ``/api/user-prefs`` hands it to a page."""
    favorites = main_test._body(auth_client.get("/api/user-prefs"), _FavoritePrefs)["favorites"]
    return {
        kind: {fid: entry["cover"] for fid, entry in entries.items()}
        for kind, entries in favorites.items()
    }


def _set_favorite(
    auth_client: TestClient,
    kind: str,
    fid: str,
    entry: dict[str, str] | None,
) -> dict[str, dict[str, MutablePlainTree]]:
    response = auth_client.post("/api/favorites", json={"kind": kind, "id": fid, "entry": entry})
    assert response.status_code == 200
    return main_test._body(response, _PageFavorites)["favorites"]


class _PageFavorites(TypedDict):
    favorites: dict[str, dict[str, MutablePlainTree]]


def test_favorite_covers_are_saved_as_provider_urls(auth_client: TestClient):
    raw = "https://img.example/film.jpg"
    legacy = "/api/logo?" + parse.urlencode({"source": "img.example", "url": raw})
    _set_favorite(auth_client, "movies", "1", {"name": "Film", "cover": _logo(raw), "ext": "mkv"})
    _set_favorite(auth_client, "series", "2", {"name": "Show", "cover": legacy})
    saved = cache_module.load_user_settings("testuser")["favorites"]
    assert saved == {
        "movies": {"1": {"name": "Film", "cover": raw, "ext": "mkv"}},
        "series": {"2": {"name": "Show", "cover": raw}},
    }


def test_favorites_without_a_text_cover_are_saved_as_sent(auth_client: TestClient):
    _set_favorite(auth_client, "movies", "1", {"name": "No cover"})
    auth_client.post(
        "/api/favorites",
        json={"kind": "series", "id": "2", "entry": {"name": "Odd", "cover": 5}},
    )
    saved = cache_module.load_user_settings("testuser")["favorites"]
    assert saved["movies"]["1"] == {"name": "No cover"}
    assert saved["series"]["2"] == {"name": "Odd", "cover": 5}


def test_each_favorite_change_keeps_the_others(auth_client: TestClient):
    stale_page = _set_favorite(auth_client, "movies", "1", {"name": "One", "cover": ""})
    # Another tab adds a favorite this page never saw.
    _set_favorite(auth_client, "movies", "2", {"name": "Two", "cover": ""})
    assert set(stale_page["movies"]) == {"1"}
    page = _set_favorite(auth_client, "series", "3", {"name": "Three", "cover": ""})
    assert (set(page["movies"]), set(page["series"])) == ({"1", "2"}, {"3"})
    page = _set_favorite(auth_client, "movies", "1", None)
    assert set(page["movies"]) == {"2"}
    assert _set_favorite(auth_client, "movies", "missing", None)["movies"].keys() == {"2"}
    saved = cache_module.load_user_settings("testuser")["favorites"]
    assert (set(saved["movies"]), set(saved["series"])) == ({"2"}, {"3"})


@pytest.mark.parametrize(
    "body",
    [b"{bad", b'{"kind": "live", "id": "1", "entry": null}', b"x" * (64 * 1024 + 1)],
)
def test_favorite_update_rejects_bad_bodies(auth_client: TestClient, body: bytes):
    assert auth_client.post("/api/favorites", content=body).status_code == 400


def test_favorites_of_other_shapes_are_shown_as_they_are(auth_client: TestClient):
    raw = "https://img.example/film.jpg"
    favorites: dict[str, dict[str, MutablePlainTree]] = {
        "movies": {"1": {"name": "Film", "cover": raw}, "2": "Film"},
        "series": {"3": 3},
    }
    settings = cache_module.load_user_settings("testuser")
    settings["favorites"] = favorites
    cache_module.save_user_settings("testuser", settings)
    cache_module.get_cache()["vod_streams"] = [{"stream_id": 2, "stream_icon": raw}]
    page = main_test._fields(auth_client.get("/api/user-prefs"))["favorites"]
    assert page == {
        "movies": {"1": {"name": "Film", "cover": ""}, "2": "Film"},
        "series": {"3": 3},
    }


def test_favorite_saved_before_signing_still_loads_its_cover(auth_client: TestClient, web: FakeWeb):
    raw = "https://img.example/film.jpg"
    web.routes[raw] = FakeResponse(b"img", headers={"Content-Type": "image/png"})
    # Saved before /api/logo needed a signature: an unsigned proxy link.
    settings = cache_module.load_user_settings("testuser")
    settings["favorites"] = {
        "series": {},
        "movies": {"7": {"name": "Film", "cover": "/api/logo?source=img.example&url=" + raw}},
    }
    cache_module.save_user_settings("testuser", settings)
    cache_module.get_cache()["vod_streams"] = [{"stream_id": 7, "stream_icon": raw}]

    cover = _favorite_covers(auth_client)["movies"]["7"]

    assert cover == main._logo_url_filter(raw)
    response = auth_client.get(cover)
    assert response.status_code == 200
    assert response.content == b"img"


def test_favorite_covers_are_signed_only_from_the_catalog(auth_client: TestClient):
    # Favorites are written by viewers, so their stored covers must not be signed.
    internal = "http://169.254.169.254/latest/meta-data"
    settings = cache_module.load_user_settings("testuser")
    settings["favorites"] = {
        "movies": {
            "7": {"name": "Listed", "cover": internal},
            "8": {"name": "Not listed", "cover": internal},
        },
        "series": {"3": {"name": "Show", "cover": internal}},
    }
    cache_module.save_user_settings("testuser", settings)
    cache_module.get_cache()["vod_streams"] = [
        {"stream_id": 7, "stream_icon": "https://img.example/7.jpg"},
    ]
    cache_module.get_cache()["series"] = [{"series_id": 3, "cover": ""}]

    covers = _favorite_covers(auth_client)

    assert covers == {
        "movies": {"7": main._logo_url_filter("https://img.example/7.jpg"), "8": ""},
        "series": {"3": ""},
    }


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
    assert main._safe_float([]) == 0.0
    assert main._safe_float(None, 2.0) == 2.0
    assert main._safe_float("1.5") == 1.5
    assert main._safe_float("nope", 3.0) == 3.0


async def _collect_events(queue: asyncio.Queue[str]) -> list[str]:
    return [event async for event in main._epg_event_stream(queue)]


def test_epg_event_stream_ready_immediately(guide_db: None):
    del guide_db
    now = _now()
    _add_program("Now", now, now + 600)
    queue: asyncio.Queue[str] = asyncio.Queue()
    main._epg_subscribers.add(queue)
    assert asyncio.run(_collect_events(queue)) == ["data: epg_ready\n\n"]
    assert queue not in main._epg_subscribers


def test_epg_event_stream_waits_for_event(guide_db: None, monkeypatch: pytest.MonkeyPatch):
    del guide_db
    clock = FakeClock()
    install_clock(monkeypatch, clock, main)
    monkeypatch.setattr(main, "_shutdown_event", None)

    async def run() -> list[str]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        task = asyncio.ensure_future(_collect_events(queue))
        while not clock.sleeps:
            await asyncio.sleep(0)
        queue.put_nowait("epg_ready")
        return await task

    assert asyncio.run(run()) == ["data: epg_ready\n\n"]
    assert clock.sleeps[0] == 1


def test_epg_event_stream_stops_on_shutdown(guide_db: None, monkeypatch: pytest.MonkeyPatch):
    del guide_db
    queue: asyncio.Queue[str] = asyncio.Queue()
    main._epg_subscribers.add(queue)
    shutdown = asyncio.Event()
    shutdown.set()
    monkeypatch.setattr(main, "_shutdown_event", shutdown)
    assert asyncio.run(_collect_events(queue)) == []
    assert queue not in main._epg_subscribers


def test_next_epg_event(monkeypatch: pytest.MonkeyPatch):
    clock = FakeClock()
    install_clock(monkeypatch, clock, main)

    async def run() -> tuple[str | None, str | None]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        queue.put_nowait("epg_ready")
        return await main._next_epg_event(queue), await main._next_epg_event(queue)

    assert asyncio.run(run()) == ("epg_ready", None)
    assert clock.sleeps == [1]


def test_rate_limit_drops_expired_ip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "_login_attempts", {"1.1.1.1": [0.0]})
    main._check_rate_limit("1.1.1.1")
    assert "1.1.1.1" not in main._login_attempts


def test_rate_limit_prunes_stale_ips_and_blocks(monkeypatch: pytest.MonkeyPatch):
    now = time.time()
    attempts: dict[str, list[float]] = {f"ip{i}": [0.0] for i in range(1001)}
    attempts["2.2.2.2"] = [now] * 10
    monkeypatch.setattr(main, "_login_attempts", attempts)
    with pytest.raises(HTTPException) as exc:
        main._check_rate_limit("2.2.2.2")
    assert exc.value.status_code == 429
    assert len(main._login_attempts) == 902


def test_fetch_all_epg_sums_successes_and_prunes(
    guide_db: None,
    web: FakeWeb,
    caplog: pytest.LogCaptureFixture,
):
    del guide_db
    now = _now()
    cache_module.save_server_settings(
        {
            **cache_module.load_server_settings(),
            "user_agent_preset": "custom",
            "user_agent_custom": "UA",
        },
    )
    _add_program("Expired", now - 90_000, now - 88_000)
    _add_program("Expired too", now - 90_000, now - 89_000)
    _add_program("Recent", now - 84_000, now - 82_800)
    # URLs past the 80 characters logged, so each message shows where it cuts.
    good, other, bad, messy = (
        f"http://{name}/{'x' * 90}" for name in ("good", "other", "bad", "messy")
    )
    web.routes[good] = _xmltv(*((f"P{i}", now + i * 60, now + i * 60 + 60) for i in range(3)))
    web.routes[other] = _xmltv(("Q", now, now + 120))
    web.routes[bad] = RuntimeError("timeout")
    # Junk between programmes makes the feed fetch again, whole, and keep a debug
    # copy; sanitizing salvages the programme.
    junk = _xmltv(("M", now, now + 60)).read().replace(b"</tv>", b"<<</tv>")
    web.routes[messy] = lambda: FakeResponse(junk)
    # Two days ago this feed already listed further than it does now.
    quiet = "http://quiet"
    # Starting before the refetched listing, it is kept, not replaced.
    _add_program("Long", now - 60, now + 5 * 86_400, source_id="q")
    epg.record_horizons(now=now - 2 * 86_400)
    web.routes[quiet] = _xmltv(("R", now, now + 60))
    cache_module.get_cache()["live_streams"] = []
    feeds = [(good, 5, "a"), (other, 5, "c"), (bad, 5, "b"), (messy, 5, "d"), (quiet, 5, "q")]
    with caplog.at_level(logging.INFO, logger="netv.main"):
        total = main._fetch_all_epg(feeds)
    assert total == 6
    assert epg.get_program_count() == 8
    # Each fetched source's horizon is recorded for the next fetch to compare.
    assert set(epg.horizons_advanced()) >= {"a", "c", "d", "q"}
    assert {f.headers["User-agent"] for f in web.fetches} == {"UA"}
    assert list(cache_module.CACHE_DIR.glob("epg_debug_*.xml"))

    def until(stop: float) -> str:
        return datetime.fromtimestamp(stop, timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    stale = "is its feed updating?"
    assert {r.getMessage() for r in caplog.records if r.name == "netv.main"} == {
        *(f"Fetching EPG (timeout=5s): {url[:80]}" for url in (good, other, bad, messy, quiet)),
        f"EPG done: 3 programs from {good[:50]}",
        f"EPG done: 1 programs from {other[:50]}",
        f"EPG done: 1 programs from {messy[:50]}",
        f"EPG done: 1 programs from {quiet[:50]}",
        f"EPG failed: {bad[:50]} - timeout",
        "Pruned 2 expired EPG programs",
        f"Guide for b has no listings; {stale}",
        f"Guide for q has not reached further since {until(now - 2 * 86_400)}"
        f" (listings end {until(now + 5 * 86_400)}); {stale}",
        "EPG fetch complete: 6 programs total",
    }


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
    held_threads: ThreadRecorder,
    flag: str,
    starter: object,
):
    refresh_flags.add(flag)
    assert callable(starter)
    starter()
    assert held_threads.threads == []


def test_vod_background_load_fills_cache_and_clears_flag(
    refresh_flags: set[str],
    threads: ThreadRecorder,
):
    del threads
    cache_module.save_file_cache(
        "vod_data",
        {"cats": [{"category_id": "vc"}], "streams": [{"stream_id": "vs"}]},
    )
    main._start_vod_background_load()
    assert cache_module.get_cache().get("vod_categories") == [{"category_id": "vc"}]
    assert cache_module.get_cache().get("vod_streams") == [{"stream_id": "vs"}]
    assert "vod_load" not in refresh_flags


def test_series_background_load_fills_cache_and_clears_flag(
    refresh_flags: set[str],
    threads: ThreadRecorder,
):
    del threads
    cache_module.save_file_cache(
        "series_data",
        {"cats": [{"category_id": "sc"}], "series": [{"series_id": 1}]},
    )
    main._start_series_background_load()
    assert cache_module.get_cache().get("series_categories") == [{"category_id": "sc"}]
    assert cache_module.get_cache().get("series") == [{"series_id": 1}]
    assert "series_load" not in refresh_flags


def test_guide_background_load_fetches_guide(
    guide_db: None,
    refresh_flags: set[str],
    threads: ThreadRecorder,
    web: FakeWeb,
):
    del guide_db, threads
    now = _now()
    cache_module.save_file_cache(
        "live_data",
        {"cats": [{"category_id": "c"}], "streams": [], "epg_urls": [["http://g", 5, "s"]]},
    )
    web.routes["http://g"] = _xmltv(("P", now, now + 60))
    main._start_guide_background_load()
    store = cache_module.get_cache()
    assert store.get("live_categories") == [{"category_id": "c"}]
    assert store.get("epg_urls") == [("http://g", 5, "s")]
    assert "epg_error" not in store
    assert epg.get_program_count() == 1
    assert "guide_load" not in refresh_flags


def test_guide_background_load_records_guide_error(
    refresh_flags: set[str],
    threads: ThreadRecorder,
    monkeypatch: pytest.MonkeyPatch,
):
    del threads
    monkeypatch.setattr(db, "_path", None)
    cache_module.save_file_cache("live_data", {"cats": [], "streams": [], "epg_urls": []})
    main._start_guide_background_load()
    assert cache_module.get_cache().get("epg_error") == "Database not initialized"
    assert "guide_load" not in refresh_flags


def test_guide_rows_for_empty_range():
    now = datetime.now(timezone.utc)
    assert main._build_guide_rows([{"stream_id": 1}], 1, 10, now, now) == []


def test_guide_streams_ignore_unknown_playlist(auth_client: TestClient):
    del auth_client
    stream: ProviderRecord = {"stream_id": 1, "category_ids": ["1"], "source_id": "s"}
    cache_module.get_cache()["live_streams"] = [stream]
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


def test_live_player_info_unknown_stream(guide_db: None):
    del guide_db
    cache_module.get_cache()["live_streams"] = []
    assert main._get_live_player_info("missing") == PlayerInfo()


def test_live_player_info_uses_source_settings(guide_db: None):
    del guide_db
    now = _now()
    cache_module.get_cache()["live_streams"] = [
        {**main_test._archive_stream(), "category_ids": ["1"]},
    ]
    _use_sources(_source("src1", "xtream", deinterlace_fallback=False))
    _add_program("Now", now - 60, now + 60)
    info = main._get_live_player_info("src1_42")
    assert info.url == "http://upstream.test/live/user/pass/42.m3u8"
    assert info.is_m3u is False
    assert info.deinterlace_fallback is False
    assert info.category_ids == ["1"]
    assert info.catchup_days == 2
    assert (info.program_title, info.program_start, info.program_end) == ("Now", now - 60, now + 60)


def test_live_player_info_direct_url_and_source_defaults(guide_db: None):
    del guide_db
    now = _now()
    cache_module.get_cache()["live_streams"] = [
        {
            "stream_id": "m1",
            "name": "M3U",
            "direct_url": "http://media/1.m3u8",
            "source_id": "m",
            "category_ids": "1",
            "epg_channel_id": "ch1",
        },
    ]
    _use_sources({"id": "m", "name": "M", "type": "m3u", "url": "http://up"})
    epg.insert_programs([("ch1", "Now", now - 60, now + 60, "Plot", "m")])
    epg.commit()
    assert main._get_live_player_info("m1") == PlayerInfo(
        url="http://media/1.m3u8",
        is_m3u=True,
        channel_name="M3U",
        source_id="m",
        deinterlace_fallback=True,
        program_title="Now",
        program_desc="Plot",
        program_start=now - 60,
        program_end=now + 60,
    )


def test_archived_program_needs_epg_id():
    assert main._find_archived_program({"stream_id": 1}, datetime.now(timezone.utc)) is None


def test_catchup_player_info_unknown_stream(guide_db: None):
    del guide_db
    cache_module.get_cache()["live_streams"] = []
    assert main._get_catchup_player_info("x", 0) == PlayerInfo()


def test_movie_player_info_without_client():
    cache_module.get_cache().pop("vod_streams", None)
    assert main._get_movie_player_info("1") == PlayerInfo()


@pytest.mark.parametrize(
    ("movie", "name", "desc"),
    [
        ({"info": {"name": "Film", "releasedate": "1999-01-01", "plot": "P"}}, "Film (1999)", "P"),
        ({"info": {"name": "Film", "description": "D"}}, "Film", "D"),
        ({}, "", ""),
    ],
)
@pytest.mark.usefixtures("guide_db")
def test_movie_player_info(web: FakeWeb, movie: ProviderRecord, name: str, desc: str):
    imdb.init()
    cache_module.get_cache()["vod_streams"] = [{"stream_id": 5, "source_id": "s1"}]
    _use_sources(
        _source("s0", "xtream", url="http://zero"),
        _source("s1", "xtream", url="http://x"),
    )
    web.routes[_api("http://x", "get_vod_info", vod_id=5)] = _json(movie)
    info = main._get_movie_player_info("5")
    assert info.url == "http://x/movie/u/p/5.mkv"
    assert info.source_id == "s1"
    assert (info.channel_name, info.program_desc) == (name, desc)


def test_movie_player_info_plays_the_catalog_s_container(web: FakeWeb):
    cache_module.get_cache()["vod_streams"] = [
        {"stream_id": 5, "source_id": "s0", "container_extension": "mp4"},
    ]
    _use_sources(_source("s0", "xtream", url="http://x"))
    info = main._get_movie_player_info("5")
    assert info.url == "http://x/movie/u/p/5.mp4"
    assert info.channel_name == ""
    assert web.urls == [_api("http://x", "get_vod_info", vod_id=5)]


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
    assert main._get_series_player_info("1", 0) == (PlayerInfo(), None)


def test_series_player_info_without_series_id(web: FakeWeb):
    cache_module.get_cache()["series"] = [{"series_id": 0, "source_id": "ignored"}]
    _use_sources(
        _source("s0", "xtream", url="http://x"),
        _source("ignored", "xtream", url="http://ignored"),
    )
    info, next_url = main._get_series_player_info("1", 0)
    assert info == PlayerInfo(url="http://x/series/u/p/1.mkv")
    assert next_url is None
    assert web.fetches == []


@pytest.mark.parametrize(
    ("stream_id", "ext", "title", "desc", "next_id"),
    [
        ("10", "mkv", "S01E00 -- Prologue", "", "11"),
        ("11", "mkv", "S01E01 -- Pilot", "Start", "12"),
        ("12", "mp4", "S01E02 -- Second", "", "21"),
        ("21", "mkv", "S02E01 -- Grand Finale", "", "101"),
        ("101", "mkv", "S10E01 -- Last", "", None),
    ],
)
@pytest.mark.usefixtures("guide_db")
def test_series_player_info_episodes(
    web: FakeWeb,
    stream_id: str,
    ext: str,
    title: str,
    desc: str,
    next_id: str | None,
):
    next_url = None if next_id is None else main._play_path("series", next_id, series_id=1)
    imdb.init()
    cache_module.get_cache()["series"] = [{"series_id": 1, "source_id": "s1"}]
    _use_sources(
        _source("s0", "xtream", url="http://zero"),
        _source("s1", "xtream", url="http://x"),
    )
    url = _api("http://x", "get_series_info", series_id=1)
    web.routes[url] = _json(_SERIES)
    info, found_next = main._get_series_player_info(stream_id, 1)
    assert info.url == f"http://x/series/u/p/{stream_id}.{ext}"
    assert info.source_id == "s1"
    assert info.channel_name == "Show (2020)"
    assert (info.program_title, info.program_desc, found_next) == (title, desc, next_url)
    assert web.urls == [url]
    assert (cache_module.CACHE_DIR / "series_info_s1_1.json").exists()


def test_stale_guides_name_sources_whose_feeds_stopped_updating(guide_db: None):
    del guide_db
    now = _now()
    day = 86_400.0
    _use_sources(_source("ota", "m3u", url="http://ota"))
    # Two days ago OTA listed two weeks ahead and the daily feed one day back.
    _add_program("Old OTA", now - 2 * day, now + 12 * day, source_id="ota")
    _add_program("Old daily", now - 2 * day, now - day, source_id="daily")
    epg.record_horizons(now=now - 2 * day)
    # Since then the daily feed reaches further, as a refreshing feed does, and
    # OTA's has not, as when its generator stopped (Nov 8 to 21).
    _add_program("New daily", now, now + day, source_id="daily")
    epg.record_horizons(now=now)
    # Listings never recorded are not judged yet.
    _add_program("Unrecorded", now, now + 3600, source_id="new")
    epg_urls = [
        ("http://ota/a", 5, "ota"),
        ("http://ota/b", 5, "ota"),
        ("http://daily", 5, "daily"),
        ("http://new", 5, "new"),
        ("http://empty", 5, "empty"),
    ]
    ota: list[main.StaleGuide] = [
        {
            "name": "OTA",
            "horizon": {
                "ends": datetime.fromtimestamp(now + 12 * day, timezone.utc),
                "since": datetime.fromtimestamp(now - 2 * day, timezone.utc),
            },
        },
    ]

    # Named once, by its configured name; a source with no listings, by id.
    assert main._stale_guides(epg_urls, now) == [
        *ota,
        {"name": "empty", "horizon": None},
    ]
    # One missed daily refresh is not yet a dead feed; 36 h without one is.
    assert main._stale_guides(epg_urls[:1], now - 2 * day + 36 * 3600) == []
    assert main._stale_guides(epg_urls[:1], now - 2 * day + 36 * 3600 + 1) == ota
    assert main._stale_guides([], now) == []


def test_fetch_all_epg_caps_its_workers(
    guide_db: None,
    monkeypatch: pytest.MonkeyPatch,
):
    del guide_db
    workers: list[int | None] = []
    real_pool = futures.ThreadPoolExecutor

    def pool(max_workers: int | None) -> futures.ThreadPoolExecutor:
        workers.append(max_workers)
        return real_pool(max_workers)

    monkeypatch.setattr(main.futures, "ThreadPoolExecutor", pool)
    cache_module.get_cache()["live_streams"] = []
    main._fetch_all_epg([(f"http://down{i}", 1, "x") for i in range(9)])
    main._fetch_all_epg([])
    assert workers == [8, 1]


@pytest.mark.parametrize(
    ("route", "name", "cached"),
    [
        (error.URLError("down"), "", False),
        # A provider answer without info is used once, never cached.
        ({}, "", False),
        ({"info": {"name": "Untimed"}}, "Untimed", True),
        ({"info": {"plot": "Unnamed"}}, "", True),
    ],
)
def test_series_player_info_partial_data(
    web: FakeWeb,
    route: dict[str, object] | Exception,
    name: str,
    cached: bool,
    caplog: pytest.LogCaptureFixture,
):
    cache_module.get_cache().pop("series", None)
    _use_sources(_source("s0", "xtream", url="http://x"))
    url = _api("http://x", "get_series_info", series_id=9)
    # A fresh response each run: a body is read once, and a test may run twice.
    web.routes[url] = route if isinstance(route, Exception) else _json(route)
    with caplog.at_level(logging.WARNING, logger="netv.main"):
        info, next_url = main._get_series_player_info("1", 9)
    assert info == PlayerInfo(url="http://x/series/u/p/1.mkv", channel_name=name)
    assert next_url is None
    assert web.urls == [url]
    assert (cache_module.CACHE_DIR / "series_info__9.json").exists() is cached
    failed = isinstance(route, Exception)
    warnings = [r.getMessage() for r in caplog.records if r.name == "netv.main"]
    assert warnings == ([f"Failed to fetch series info 9: {route}"] if failed else [])


def test_ensure_live_cache_loads_streams_and_epg(guide_db: None, web: FakeWeb):
    del guide_db
    cache_module.get_cache().pop("live_streams", None)
    cache_module.save_file_cache(
        "live_data",
        {"cats": [{"c": 1}], "streams": [{"s": 1}], "epg_urls": [["http://u", 1, "s"]]},
    )
    web.routes["http://u"] = RuntimeError("offline")
    assert main._ensure_live_cache() == [{"s": 1}]
    store = cache_module.get_cache()
    assert store.get("live_streams") == [{"s": 1}]
    assert store.get("live_categories") == [{"c": 1}]
    assert web.urls == ["http://u"]


def test_ensure_live_cache_survives_a_failed_guide_load(
    guide_db: None,
    monkeypatch: pytest.MonkeyPatch,
):
    del guide_db
    # Channels cached without guide URLs; the guide loader then fails outright.
    cache_module.get_cache()["live_streams"] = [{"s": 1}]
    cache_module.get_cache().pop("epg_urls", None)
    asked: list[list[tuple[str, int, str]]] = []

    def failing_load(urls: list[tuple[str, int, str]]) -> None:
        asked.append(urls)
        raise RuntimeError("guide down")

    monkeypatch.setattr(main, "load_all_epg", failing_load)
    assert main._ensure_live_cache() == [{"s": 1}]
    assert asked == [[]]


# ===========================================================================
# Handoff helpers
# ===========================================================================


def _handoff_guide(now: float) -> None:
    """Current program from another source; archived neighbors from the stream's own."""
    cache_module.get_cache()["live_streams"] = [main_test._archive_stream()]
    _add_program("Now", now - 600, now + 600, "other")
    _add_program("Earlier", now - 900, now - 300)
    _add_program("Later", now - 300, now + 900)


def test_handoff_channel_unknown_stream(guide_db: None):
    del guide_db
    cache_module.get_cache()["live_streams"] = []
    assert main._handoff_channel("x", time.time()) is None


@pytest.mark.parametrize("offset", [-600, -100])
def test_handoff_channel_within_current_program(guide_db: None, offset: int):
    del guide_db
    now = _now()
    _handoff_guide(now)
    channel = main._handoff_channel("src1_42", now + offset)
    assert channel == handoff.Channel(
        stream_id="src1_42",
        name="Channel",
        url="http://upstream.test/live/user/pass/42.m3u8",
        source_id="src1",
        catchup_days=2,
        program_title="Now",
        program_start=now - 600,
        program_end=now + 600,
    )


@pytest.mark.parametrize(
    ("offset", "title", "start", "stop", "catchup_days"),
    [(-700, "Earlier", -900, -300, 2), (600, "Later", -300, 900, 0)],
)
def test_handoff_channel_looks_up_earlier_program(
    guide_db: None,
    offset: int,
    title: str,
    start: int,
    stop: int,
    catchup_days: int,
):
    del guide_db
    now = _now()
    _handoff_guide(now)
    channel = main._handoff_channel("src1_42", now + offset)
    assert channel is not None
    assert channel.program_title == title
    assert (channel.program_start, channel.program_end) == (now + start, now + stop)
    assert channel.catchup_days == catchup_days


def test_handoff_channel_outside_archive(guide_db: None):
    del guide_db
    cache_module.get_cache()["live_streams"] = [main_test._archive_stream()]
    channel = main._handoff_channel("src1_42", time.time() - 30 * 86_400)
    assert channel is not None
    assert (channel.program_title, channel.program_start, channel.program_end) == ("", 0.0, 0.0)
    assert channel.catchup_days == 0


def test_handoff_player_info_without_live_url(guide_db: None, clean_handoff: None):
    del guide_db, clean_handoff
    cache_module.get_cache()["live_streams"] = []
    handoff._records["testuser"] = main_test._handoff_record()
    info = main._get_handoff_player_info("testuser", "src1_42", "h1", None)
    assert info == PlayerInfo()
    assert handoff._records["testuser"]["resumed"] is True


@pytest.mark.parametrize(("start", "position"), [(None, 150), (0, 0), (200, 200)])
@pytest.mark.parametrize("title", ["Saved program", ""])
def test_handoff_recording_publishes_a_replacement(
    guide_db: None,
    clean_handoff: None,
    start: int | None,
    position: int,
    title: str,
):
    del guide_db, clean_handoff
    now = _now()
    cache_module.get_cache()["live_streams"] = [main_test._archive_stream()]
    _add_program("Live", now - 60, now + 60)
    handoff._records["viewer"] = main_test._handoff_record(
        position=150,
        recording_start=100,
        program_title=title,
        program_start=90,
        program_end=300,
    )
    result = main._get_handoff_player_info("viewer", "src1_42", "h1", start)
    assert result == PlayerInfo(
        url="/handoff/rec1/stream.m3u8",
        channel_name="Channel",
        source_id="src1",
        catchup_days=2,
        program_title=title or "Live",
        program_start=90,
        program_end=300,
        catchup_start=100,
        catchup_seek=max(0, position - 100),
        handoff_recording=True,
    )
    assert handoff._records["viewer"]["resumed"] is True


@pytest.mark.parametrize(("start", "expected"), [(None, -60), (-3600, -3600), (80, -60)])
def test_handoff_catchup_respects_explicit_start_and_default_archive_lag(
    guide_db: None,
    clean_handoff: None,
    monkeypatch: pytest.MonkeyPatch,
    start: int | None,
    expected: int,
):
    del guide_db, clean_handoff
    now = _now()
    install_clock(monkeypatch, FakeClock(now), main)
    cache_module.get_cache()["live_streams"] = [main_test._archive_stream()]
    handoff._records["viewer"] = main_test._handoff_record(mode="catchup", position=now - 20)
    info = main._get_handoff_player_info(
        "viewer",
        "src1_42",
        "h1",
        None if start is None else now + start,
    )
    assert info.catchup_start + info.catchup_seek == now + expected
    assert "/timeshift/user/pass/" in info.url


@pytest.mark.parametrize("missing", [True, False])
def test_handoff_rejects_missing_or_wrong_stream(clean_handoff: None, missing: bool):
    del clean_handoff
    if not missing:
        handoff._records["viewer"] = main_test._handoff_record(stream_id="other")
    with pytest.raises(HTTPException) as error_info:
        main._get_handoff_player_info("viewer", "src1_42", "h1", None)
    assert error_info.value.status_code == 404
    assert error_info.value.detail == "This program is no longer available to resume"


def test_handoff_stream_allowed(guide_db: None, auth_client: TestClient):
    del guide_db, auth_client
    cache_module.get_cache()["live_streams"] = [{"stream_id": "1", "source_id": "s"}]
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
    source = cache_module.Source(id="s", name="S", type="xtream", url=base, username=username)
    assert main._is_xtream_source_url(url, source) is expected


def test_build_all_groups():
    cache_module.get_cache()["live_categories"] = [
        {"source_id": "s1", "category_id": "7", "category_name": "News"},
        {"source_id": "gone", "category_id": "8", "category_name": "Orphan"},
    ]
    _use_sources(_source("s1", "xtream"), _source("s2", "m3u"))
    assert main._build_all_groups() == [
        {"id": "cat:7", "name": "S1: News", "type": "live"},
        {"id": "movies:s1", "name": "S1: Movies", "type": "movies"},
        {"id": "series:s1", "name": "S1: Series", "type": "series"},
    ]


def test_source_users_rejects_unknown_access():
    with pytest.raises(HTTPException) as exc:
        main._source_users("bogus", None)
    assert exc.value.status_code == 400


def test_live_streams_loaded_from_file_cache():
    cache_module.save_file_cache(
        "live_data",
        {"cats": [{"category_id": 1, "category_name": "News"}], "streams": [{"stream_id": 1}]},
    )
    streams, names = asyncio.run(main._live_streams_and_category_names())
    assert streams == [{"stream_id": 1}]
    assert names == {"1": "News"}
    assert cache_module.get_cache().get("epg_urls") == []


def test_live_streams_without_file_cache():
    assert asyncio.run(main._live_streams_and_category_names()) == ([], {})


@pytest.mark.parametrize(
    ("unavailable", "expected"),
    [
        (["movies:s1", "series:s1", "series:s2"], {"movies": True, "series": False}),
        (["movies:s1", "movies:s2", "series:s1"], {"movies": False, "series": True}),
    ],
)
def test_content_access_respects_restrictions(unavailable: list[str], expected: dict[str, bool]):
    _use_sources(_source("s1", "xtream"), _source("s2", "xtream"))
    auth.create_user("bob", "password123")
    auth.set_user_limits("bob", unavailable_groups=unavailable)
    assert main._get_content_access("bob") == expected


def _request(token: str | None) -> Request:
    headers: list[tuple[bytes, bytes]] = (
        [] if token is None else [(b"cookie", f"token={token}".encode())]
    )
    return Request({"type": "http", "headers": headers})


@pytest.mark.parametrize("token", [None, "garbage", "empty-subject"])
def test_content_access_from_request_defaults(token: str | None):
    _use_sources(_source("s1", "xtream"))
    if token == "empty-subject":
        token = auth.create_token({"sub": ""})
    access = main._get_content_access_from_request(_request(token))
    assert access == {"movies": True, "series": True}


def test_content_access_from_request_uses_user():
    _use_sources(_source("s1", "xtream"))
    auth.create_user("bob", "password123")
    auth.set_user_limits("bob", unavailable_groups=["movies:s1"])
    access = main._get_content_access_from_request(_request(auth.create_token({"sub": "bob"})))
    assert access == {"movies": False, "series": True}


class _ProbeCache(TypedDict):
    series: list[dict[str, MutablePlainTree]]


class _StoredUsers(TypedDict):
    users: dict[str, dict[str, MutablePlainTree]]


# ===========================================================================
# Command line entry point
# ===========================================================================


def _run_cli(
    argv: list[str],
    live_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    log_level: str = "",
) -> dict[str, object]:
    """Run the CLI entry point and return the options uvicorn would serve with."""
    real_path = pathlib.Path

    def redirected_path(path: str) -> pathlib.Path:
        return real_path(path.replace("/etc/letsencrypt/live", str(live_dir), 1))

    server = ServerRecorder()
    install_server(monkeypatch, server, main)
    monkeypatch.setenv("LOG_LEVEL", log_level)
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    monkeypatch.setattr(pathlib, "Path", redirected_path)
    assert main.main() == 0
    [run] = server.runs
    assert run.app is main.app
    return run.options


def test_cli_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root_logging: None):
    del root_logging
    monkeypatch.delenv("NETV_TRUSTED_PROXIES", raising=False)
    options = _run_cli([], tmp_path, monkeypatch)
    assert options["port"] == 8000
    assert options["host"] == "0.0.0.0"
    # A forwarded address is believed only from a proxy on this machine: from anyone
    # else, "X-Forwarded-For: 192.168.1.5" would pass a LAN-only check.
    assert options["proxy_headers"] is True
    assert options["forwarded_allow_ips"] == "127.0.0.1"
    monkeypatch.setenv("NETV_TRUSTED_PROXIES", "10.0.0.2")
    assert _run_cli([], tmp_path, monkeypatch)["forwarded_allow_ips"] == "10.0.0.2"
    assert options["log_level"] == "info"
    assert options["access_log"] is False
    assert options["ssl_certfile"] is None
    assert logging.root.level == logging.INFO
    [handler] = logging.root.handlers
    record = logging.makeLogRecord(
        {"msg": "Play http://p.example/live/remote-user/remote-pass/1.ts"},
    )
    assert handler.format(record).endswith(" | Play http://p.example/live/***/***/1.ts")


def test_cli_env_log_level_and_custom_certs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_logging: None,
):
    del root_logging
    options = _run_cli(
        ["--port", "9000", "--debug", "--cert", "c.pem", "--key", "k.pem"],
        tmp_path,
        monkeypatch,
        log_level="warning",
    )
    assert options["port"] == 9000
    assert options["log_level"] == "debug"
    assert options["access_log"] is True
    assert (options["ssl_certfile"], options["ssl_keyfile"]) == ("c.pem", "k.pem")
    assert logging.root.level == logging.WARNING


def test_cli_debug_flag_without_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_logging: None,
):
    del root_logging
    _run_cli(["--debug"], tmp_path, monkeypatch)
    assert logging.root.level == logging.DEBUG


@pytest.mark.parametrize("argv", [["--https", "tv.example"], ["--https"]])
def test_cli_lets_encrypt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_logging: None,
    argv: list[str],
):
    del root_logging
    domain = tmp_path / "tv.example"
    domain.mkdir()
    (domain / "fullchain.pem").write_text("cert")
    (tmp_path / "README").write_text("not a domain")
    options = _run_cli(argv, tmp_path, monkeypatch)
    assert options["ssl_certfile"] == str(domain / "fullchain.pem")
    assert options["ssl_keyfile"] == str(domain / "privkey.pem")


def test_cli_https_without_certs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_logging: None,
):
    del root_logging
    with pytest.raises(SystemExit, match="No Let's Encrypt certs"):
        _run_cli(["--https"], tmp_path, monkeypatch)


def test_cli_https_missing_domain_cert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_logging: None,
):
    del root_logging
    with pytest.raises(SystemExit, match="Cert not found"):
        _run_cli(["--https", "missing.example"], tmp_path, monkeypatch)


def test_script_guard_exits_with_main_status(monkeypatch: pytest.MonkeyPatch, root_logging: None):
    del root_logging
    server = ServerRecorder()
    install_server(monkeypatch, server)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    # Run only the guard, at its own lines; re-importing all of main costs ~0.1 s.
    path = Path(main.__file__)
    lines = path.read_text().splitlines(keepends=True)
    start = lines.index('if __name__ == "__main__":\n')
    guard = compile("\n" * start + "".join(lines[start:]), str(path), "exec")
    with pytest.raises(SystemExit) as exit_info:
        exec(guard, {**vars(main), "__name__": "__main__"})  # noqa: S102 -- Runs main's own guard lines, compiled from its source.
    assert exit_info.value.code == 0
    [run] = server.runs
    assert run.options["port"] == 8000


def test_created_app_serves_static_files_that_browsers_revalidate() -> None:
    application = main._create_app()
    assert application.title == "neTV"
    assert application.router.lifespan_context is main.lifespan
    # Without a context manager the client skips the lifespan's preloads.
    client = TestClient(application)
    first = client.get("/static/js/app.js")
    assert first.status_code == 200
    assert first.headers["cache-control"] == "no-cache"
    assert first.content == (main.APP_DIR / "static" / "js" / "app.js").read_bytes()
    again = client.get("/static/js/app.js", headers={"if-none-match": first.headers["etag"]})
    assert again.status_code == 304
    assert again.headers["cache-control"] == "no-cache"
    assert application.url_path_for("static", path="/js/app.js") == "/static/js/app.js"


_PLAY_TARGETS: list[tuple[ffmpeg_session.ContentType, str, int]] = [
    ("live", "ota:v2.1", 0),
    ("live", "20080", 0),
    ("movie", "486947", 0),
    ("series", "12", 15_137),
    ("series", "12", 0),
]


def test_a_play_link_is_short_opaque_and_survives_a_restart(tmp_path: pathlib.Path):
    epg.init(tmp_path)
    paths = [main._play_path(k, s, series_id=n) for k, s, n in _PLAY_TARGETS]
    # Ten letters and digits that say nothing of what plays.
    assert all(re.fullmatch(r"/play/[a-z2-7]{10}", p) for p in paths)
    assert len(set(paths)) == len(paths)
    assert not any(s in p for p, (_, s, _) in zip(paths, _PLAY_TARGETS, strict=True))
    # After a restart the links still play, and the same streams get the same links.
    db.close()
    epg.init(tmp_path)
    found = [main._play_target(p.removeprefix("/play/")) for p in paths]
    assert [(t.kind, t.stream_id, t.series_id) for t in found] == _PLAY_TARGETS
    assert [main._play_path(k, s, series_id=n) for k, s, n in _PLAY_TARGETS] == paths
    db.close()


@pytest.mark.usefixtures("guide_db")
@pytest.mark.parametrize("link", ["", "aaaaaaaaaa", "!!"])
def test_an_unissued_play_link_names_nothing(link: str):
    # A channel's link, built without a series, names none.
    channel = main._play_target(main._play_path("live", "1").removeprefix("/play/"))
    assert (channel.kind, channel.stream_id, channel.series_id) == ("live", "1", 0)
    with pytest.raises(ValueError, match="Unknown play path"):
        main._play_target(link)


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
