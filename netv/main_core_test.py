"""Tests for main.py startup, page, transcode and source-settings routes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, cast
from urllib import parse

import asyncio
import json
import logging
import math
import re
import signal
import threading
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient

import pytest

from netv import (
    auth,
    cache,
    chromecast,
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
from netv.cache import SourceConfig
from netv.chromecast import CastSession
from netv.ffmpeg_session import TranscodeSession
from netv.xtream import ProviderRecord


_XTREAM: SourceConfig = {
    "id": "s1",
    "name": "Prov",
    "type": "xtream",
    "url": "http://x",
    "username": "u",
    "password": "p",
}


@pytest.fixture
def api() -> TestClient:
    """A client signed in as ``testuser``, the first and therefore admin account."""
    auth.create_user("testuser", "testpass123")
    return _client("testuser")


@pytest.fixture(autouse=True)
def prestarts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Live play pages start their stream early; record it instead of opening one."""
    urls: list[str] = []

    def prestart(url: str, **options: object) -> None:
        del options
        urls.append(url)

    monkeypatch.setattr(ffmpeg_session, "prestart_live", prestart)
    return urls


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> testing.FakeWeb:
    """The internet as seen by provider, guide and playlist fetches."""
    fake = testing.FakeWeb()
    testing.install_web(monkeypatch, fake, util, ffmpeg_command)
    return fake


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> dict[str, TranscodeSession]:
    """Start with no transcode sessions; return the live session table."""
    table: dict[str, TranscodeSession] = {}
    monkeypatch.setattr(ffmpeg_session, "_transcode_sessions", table)
    monkeypatch.setattr(ffmpeg_session, "_url_to_session", {})
    return table


@pytest.fixture
def flags(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Start with no refresh in progress; return the shared flag set."""
    in_progress: set[str] = set()
    monkeypatch.setattr(m3u_module, "_refresh_in_progress", in_progress)
    return in_progress


def _client(username: str) -> TestClient:
    client = TestClient(main.app)
    client.cookies.set("token", auth.create_token({"sub": username}))
    return client


def _set_sources(*sources: SourceConfig) -> None:
    cache.save_server_settings({**cache.load_server_settings(), "sources": list(sources)})


def _corrupt_server_settings() -> None:
    cache.SERVER_SETTINGS_FILE.write_text(json.dumps({"live_dvr_mins": "soon"}))


def _api_url(base: str, action: str, **params: str) -> str:
    query = {"username": "u", "password": "p", "action": action, **params}
    return f"{base}/player_api.php?{parse.urlencode(query)}"


def _xmltv(channel: str, title: str, start: datetime, stop: datetime) -> bytes:
    stamp = "%Y%m%d%H%M%S +0000"
    return (
        f'<tv><channel id="{channel}"><display-name>{channel}</display-name></channel>'
        f'<programme start="{start.astimezone(timezone.utc).strftime(stamp)}" '
        f'stop="{stop.astimezone(timezone.utc).strftime(stamp)}" channel="{channel}">'
        f"<title>{title}</title></programme></tv>"
    ).encode()


def _xmltv_route(channel: str, title: str, now: datetime) -> testing.Route:
    body = _xmltv(channel, title, now - timedelta(hours=1), now + timedelta(hours=8))
    return lambda: testing.FakeResponse(body)


def _insert_program(
    channel: str,
    source_id: str,
    start: datetime,
    stop: datetime,
    desc: str = "",
) -> None:
    epg.insert_programs([(channel, "Old", start.timestamp(), stop.timestamp(), desc, source_id)])
    epg.commit()


def _publish_rendition(directory: Path) -> None:
    """Write two playable segments and low.m3u8, as a started live worker would."""
    lines = ["#EXTM3U"]
    for i in range(2):
        (directory / f"low_{i}.ts").write_bytes(testing.video_packet(10 + 2 * i) * 10)
        lines += ["#EXTINF:2,", f"low_{i}.ts"]
    (directory / "low.m3u8").write_text("\n".join(lines) + "\n")


def _session(
    directory: Path,
    *,
    url: str = "http://up/m.mkv",
    username: str = "testuser",
    source_id: str = "",
    started: float = 0.0,
    **extra: object,
) -> TranscodeSession:
    directory.mkdir(parents=True, exist_ok=True)
    session: TranscodeSession = {
        "dir": str(directory),
        "process": None,
        "started": started,
        "url": url,
        "username": username,
        "source_id": source_id,
    }
    if "is_vod" in extra:
        session["is_vod"] = bool(extra["is_vod"])
    if "last_access" in extra:
        session["last_access"] = cast(float, extra["last_access"])
    return session


# =============================================================================
# Startup and shutdown
# =============================================================================


def test_lifespan_runs_workers_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    web: testing.FakeWeb,
    sessions: dict[str, TranscodeSession],
    caplog: pytest.LogCaptureFixture,
) -> None:
    noon = datetime(2026, 1, 1, 12, 0)
    # The scheduler polls every 30 s, so its first two passes land at 12:00:00 and 12:00:30.
    clock = testing.FakeClock(now=noon.timestamp() - 30)
    testing.install_clock(monkeypatch, clock, main)
    threads = testing.ThreadRecorder(run_inline=False)
    testing.install_threads(monkeypatch, threads, main)
    run = testing.FakeRun()
    run.program("pgrep", testing.Program(stdout="77\n"))
    testing.install_subprocess(monkeypatch, main, run=run)
    signals = testing.SignalRecorder()
    testing.install_signals(monkeypatch, signals, main)
    probe = testing.FakeRun()
    probe.program("ffprobe", testing.Program(stdout=testing.probed_video_json()))
    testing.install_subprocess(monkeypatch, ffmpeg_command, run=probe)

    def publish(command: list[str]) -> None:
        if not launcher.launches:  # The viewer's worker; the recorder publishes nothing awaited.
            _publish_rendition(Path(command[-1]).parent)

    launcher = testing.ProcessLauncher(on_launch=publish)
    testing.install_launcher(monkeypatch, launcher, ffmpeg_session)
    tasks = testing.BackgroundTasks()
    monkeypatch.setattr(ffmpeg_session, "_spawn_background_task", tasks)
    monkeypatch.setattr(ffmpeg_session, "_startup_guards", {})
    monkeypatch.setattr(ffmpeg_session, "_recordings", set())
    upstreams = testing.FakeUpstreams()
    empty: dict[str, object] = {}
    for name, value in (
        ("_records", dict(empty)),
        ("_watches", dict(empty)),
        ("_recorders", dict(empty)),
        ("_generations", dict(empty)),
        ("_users_dir", None),
    ):
        monkeypatch.setattr(handoff, name, value)
    monkeypatch.setattr(main, "_shutdown_event", None)
    # Threads share the test's fake internet; spawned processes would not.
    monkeypatch.setattr(imdb, "ProcessPoolExecutor", testing.thread_pool)
    monkeypatch.setattr(chromecast.manager, "_sessions", {})
    monkeypatch.setattr(chromecast.manager, "_stop", threading.Event())
    monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", None)
    monkeypatch.setattr(ffmpeg_command, "_load_settings", cache.load_server_settings)
    engines = tmp_path / "engines"
    engines.mkdir()
    (engines / "2x-nomosuni-compact_1080p_fp16.engine").touch()
    monkeypatch.setenv("SR_ENGINE_DIR", str(engines))
    flags: set[str] = set()
    monkeypatch.setattr(m3u_module, "_refresh_in_progress", flags)

    transcodes = tmp_path / "transcodes"
    sources: list[SourceConfig] = [
        {**_XTREAM, "id": "x", "username": "", "password": "", "epg_schedule": ["12:00"]},
        {"id": "m", "name": "M", "type": "m3u", "url": "http://m/list", "epg_schedule": ["12:00"]},
        {"id": "e", "name": "E", "type": "epg", "url": "http://e/epg", "epg_schedule": ["12:00"]},
        {"id": "o", "name": "O", "type": "other", "url": "http://o", "epg_schedule": ["12:00"]},
        {"id": "bad", "name": "Bad", "type": "m3u", "url": "http://bad", "epg_schedule": ["12:00"]},
        {
            "id": "late",
            "name": "Late",
            "type": "epg",
            "url": "http://late",
            "epg_schedule": ["13:00"],
        },
    ]
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "sources": sources,
            "transcode_dir": str(transcodes),
            "probe_live": False,
        },
    )
    (transcodes / "netv_transcode_orphan").mkdir(parents=True)
    (transcodes / "netv_handoff_orphan").mkdir()
    now = noon.astimezone(timezone.utc)
    xtream_epg = "http://x/xmltv.php?username=&password="
    web.routes.update(
        {
            "http://guide/epg": _xmltv_route("guide", "Startup", now),
            xtream_epg: _xmltv_route("xch", "X", now),
            "http://m/list": lambda: testing.FakeResponse(b'#EXTM3U url-tvg="http://m/epg"\n'),
            "http://m/epg": _xmltv_route("mch", "M", now),
            "http://e/epg": _xmltv_route("ech", "E", now),
        },
    )
    stream_url = "http://up/one.ts"
    web.routes[stream_url] = upstreams
    stream = {"stream_id": "1", "name": "One", "source_id": "x", "direct_url": stream_url}
    cache.save_file_cache(
        "live_data",
        {"cats": [], "streams": [stream], "epg_urls": [["http://guide/epg", 5, "x"]]},
    )
    cache.save_file_cache("vod_data", {"cats": [{"category_id": "vc"}], "streams": []})
    cache.save_file_cache("series_data", {"cats": [{"category_id": "sc"}], "series": []})
    watcher = testing.FakeProcess(exit_code=None)
    sessions["stale"] = _session(tmp_path / "stale", username="gone", is_vod=True)
    sessions["watcher"] = {
        **_session(tmp_path / "watcher", username="watcher", is_vod=True),
        "process": watcher,
        "last_access": time.time(),
    }
    started = time.time() - 1000
    for user, position in (("viewer", started), ("ghost", math.nan)):
        handoff.report_watch(user, "1", position, now=started)
        handoff.report_watch(user, "1", position, now=started + 200)
    viewer_sessions: list[str] = []

    async def serve() -> None:
        async with main.lifespan(main.app):
            load_all, load_vod, load_series, cleanup, handoff_loop, imdb_loop, scheduler = (
                threads.threads[:7]
            )
            load_all.run()
            load_vod.run()
            load_series.run()
            # Each deadline lets a worker loop pass the given number of times.
            clock.deadline = clock.now + 2 * 30 + 1
            scheduler.run()
            for refresh in threads.threads[7:]:
                refresh.run()
            # The viewer is still playing the channel it will be handed off from.
            viewing = asyncio.create_task(
                ffmpeg_session.start_transcode(stream_url, username="viewer"),
            )
            while not upstreams.opened:
                assert not viewing.done()
                await asyncio.sleep(0)
            upstreams.opened[0].feed(testing.ts_source())
            viewer_sessions.append((await viewing)["session_id"])
            clock.deadline = clock.now + 60 + 1
            await asyncio.to_thread(cleanup.run)
            # The handoff loop shares the cleanup loop's stop event.
            cleanup_stop, _ = clock.events
            cleanup_stop.clear()
            clock.deadline = clock.now + 15 + 1
            with caplog.at_level(logging.ERROR, logger="netv.main"):
                await asyncio.to_thread(handoff_loop.run)
                # A failed IMDb build is logged; the loop then ends with the server.
                cache.get_cache()["vod_streams"] = [{"name": "Heat - 1995"}]
                imdb_loop.run()
            viewer, recording = launcher.launches
            await recording.process.stdin.wait_for(len(testing.ts_source()))
            # The recording took over before shutdown, which would stop the viewer anyway.
            assert viewer_sessions[0] not in sessions
            assert viewer.process.signals == ["TERM"]
            assert main._shutdown_event is None

    try:
        asyncio.run(serve())
    finally:
        tasks.discard()
        upstreams.close()

    shutdown_event = main._shutdown_event
    assert shutdown_event is not None and shutdown_event.is_set()
    assert signals.kills == [(77, signal.SIGKILL)]
    assert ffmpeg_command._sr_engine_dir == str(engines)
    assert not (transcodes / "netv_transcode_orphan").exists()
    assert not (transcodes / "netv_handoff_orphan").exists()
    data = cache.get_cache()
    assert data.get("live_streams") == [stream]
    assert data.get("vod_categories") == [{"category_id": "vc"}]
    assert data.get("series_categories") == [{"category_id": "sc"}]
    assert [p.title for p in epg.get_programs_in_range("guide", now, now)] == ["Startup"]
    # The cleanup loop expired the idle session. The sweep recorded the channel the
    # viewer left from its session's relay, stopping that session, then hit the
    # broken watch.
    assert "stale" not in sessions
    _, recording = launcher.launches
    assert recording.command[recording.command.index("-i") + 1] == "pipe:0"
    assert Path(recording.command[-1]).parent.name.startswith("netv_handoff_")
    assert bytes(recording.process.stdin.written) == testing.ts_source()
    assert "Handoff sweep failed" in caplog.text
    # Five due sources, triggered once each even though the scheduler looped twice.
    assert "IMDb index build failed" in caplog.text
    assert len(threads.threads) == 7 + 5
    assert sorted(web.urls[1:6]) == sorted(
        ["http://bad", "http://e/epg", "http://m/epg", "http://m/list", xtream_epg],
    )
    # The recording reused the viewer's one upstream connection.
    assert web.urls[6] == stream_url
    imdb_urls = {f"https://datasets.imdbws.com/title.{n}.tsv.gz" for n in ("basics", "ratings")}
    # Wikidata is unreachable too; the build falls back to titles, then fails on IMDb.
    imdb_urls.add(imdb._tmdb_query_url("https://query.wikidata.org/sparql"))
    assert set(web.urls[7:]) == imdb_urls
    assert not flags
    # Shutdown stopped the remaining transcode, the recorder, its upstream and the
    # cast monitor.
    assert watcher.signals == ["TERM"]
    assert recording.process.signals == ["TERM"]
    assert upstreams.opened[0].closed
    assert chromecast.manager._stop.is_set()


def test_kill_orphaned_ffmpeg_tolerates_missing_pgrep(monkeypatch: pytest.MonkeyPatch) -> None:
    run = testing.FakeRun()
    testing.install_subprocess(monkeypatch, main, run=run)
    signals = testing.SignalRecorder()
    testing.install_signals(monkeypatch, signals, main)
    main._kill_orphaned_ffmpeg()
    assert run.calls == [["pgrep", "-f", "ffmpeg.*iptv_transcode"]]
    assert signals.kills == []


def test_kill_orphaned_ffmpeg_skips_bad_pids_and_stops_on_permission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = testing.FakeRun()
    run.program("pgrep", testing.Program(stdout="12\n\nabc\n13\n14\n15\n"))
    testing.install_subprocess(monkeypatch, main, run=run)
    signals = testing.SignalRecorder(errors={13: ProcessLookupError(), 14: PermissionError()})
    testing.install_signals(monkeypatch, signals, main)
    main._kill_orphaned_ffmpeg()
    assert signals.kills == [(12, signal.SIGKILL), (13, signal.SIGKILL), (14, signal.SIGKILL)]


def test_preload_live_fills_cache_and_clears_flag(
    web: testing.FakeWeb,
    flags: set[str],
) -> None:
    _set_sources({"id": "m", "name": "M", "type": "m3u", "url": "http://m/list"})
    loading: list[bool] = []

    def playlist() -> testing.FakeResponse:
        loading.append("guide_load" in flags)
        return testing.FakeResponse(
            b'#EXTM3U url-tvg="http://m/epg"\n#EXTINF:-1 group-title="News",CNN\nhttp://m/1\n',
        )

    web.routes["http://m/list"] = playlist
    main._preload_live()
    assert loading == [True]
    data = cache.get_cache()
    assert data.get("live_categories", [{}])[0]["category_name"] == "News"
    assert [s["name"] for s in data.get("live_streams", [])] == ["CNN"]
    assert data.get("epg_urls") == [("http://m/epg", 120, "m")]
    assert "guide_load" not in flags


def test_preload_epg_notifies_subscribers(
    monkeypatch: pytest.MonkeyPatch,
    guide_db: None,
    web: testing.FakeWeb,
) -> None:
    del guide_db
    now = datetime.now(timezone.utc)
    web.routes["http://e/epg"] = _xmltv_route("ch", "News", now)
    cache.get_cache()["epg_urls"] = [("http://e/epg", 5, "s")]
    queue: asyncio.Queue[str] = asyncio.Queue()
    full: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    full.put_nowait("stale")
    monkeypatch.setattr(main, "_epg_subscribers", {queue, full})
    main._preload_epg()
    assert web.urls == ["http://e/epg"]
    assert epg.get_program_count() == 1
    assert queue.get_nowait() == "epg_ready"
    assert full.get_nowait() == "stale"


def test_preload_epg_logs_errors(caplog: pytest.LogCaptureFixture) -> None:
    cache.get_cache()["epg_urls"] = [("http://e/epg", 5, "s")]
    with caplog.at_level(logging.ERROR):
        main._preload_epg()
    assert "EPG load error: Database not initialized" in caplog.text


def test_load_all_epg_skips_when_another_thread_filled_it(
    monkeypatch: pytest.MonkeyPatch,
    guide_db: None,
    web: testing.FakeWeb,
) -> None:
    del guide_db
    now = datetime.now(timezone.utc)
    lock = testing.ContendedLock(lambda: _insert_program("ch", "s", now, now + timedelta(days=1)))
    monkeypatch.setattr(main, "_epg_fetch_lock", lock)
    main.load_all_epg([("http://e/epg", 5, "s")])
    assert lock.acquired == 1
    assert web.urls == []


def test_load_all_epg_records_fetch_error(guide_db: None) -> None:
    del guide_db
    _corrupt_server_settings()
    main.load_all_epg([("http://e/epg", 5, "s")])
    assert cache.get_cache().get("epg_error") == "live_dvr_mins must be an integer"


def test_epg_events_registers_subscriber(monkeypatch: pytest.MonkeyPatch) -> None:
    subscribers: set[asyncio.Queue[str]] = set()
    monkeypatch.setattr(main, "_epg_subscribers", subscribers)
    response = asyncio.run(main.epg_events({"sub": "u"}))
    assert response.media_type == "text/event-stream"
    assert len(subscribers) == 1


def test_epg_events_rejects_when_full(monkeypatch: pytest.MonkeyPatch) -> None:
    full: set[asyncio.Queue[str]] = {asyncio.Queue() for _ in range(100)}
    monkeypatch.setattr(main, "_epg_subscribers", full)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.epg_events({"sub": "u"}))
    assert exc.value.status_code == 503
    assert len(full) == 100


# =============================================================================
# Auth and error pages
# =============================================================================


def test_http_error_renders_html_for_browsers(api: TestClient) -> None:
    resp = api.get("/play/bogus/1", headers={"accept": "text/html"})
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("text/html")
    assert "Invalid stream type" in resp.text


def test_setup_post_redirects_once_users_exist(api: TestClient) -> None:
    resp = api.post(
        "/setup",
        data={"username": "other", "password": "password1", "confirm": "password1"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"


# =============================================================================
# Guide, VOD and series pages
# =============================================================================

_LIVE_FILE_CACHE: dict[str, list[dict[str, object]]] = {
    "cats": [{"category_id": "1", "category_name": "News"}],
    "streams": [
        {"stream_id": 1, "name": "CachedCNN", "category_ids": ["1"], "epg_channel_id": ""},
    ],
    "epg_urls": [],
}


def test_guide_page_loads_file_cache(api: TestClient, guide_db: None) -> None:
    del guide_db
    cache.save_file_cache("live_data", _LIVE_FILE_CACHE)
    resp = api.get("/guide?cats=1")
    assert resp.status_code == 200
    assert "CachedCNN" in resp.text
    assert cache.get_cache().get("live_streams", [{}])[0]["name"] == "CachedCNN"


def test_guide_rows_api_loads_file_cache(api: TestClient) -> None:
    cache.save_file_cache("live_data", _LIVE_FILE_CACHE)
    resp = api.get("/api/guide/rows?cats=1")
    assert resp.status_code == 200
    body = cast(dict[str, object], resp.json())
    assert body["total"] == 1
    assert cast(list[dict[str, dict[str, object]]], body["rows"])[0]["channel"]["name"] == (
        "CachedCNN"
    )
    past_end = cast(dict[str, object], api.get("/api/guide/rows?cats=1&start=5").json())
    assert (past_end["total"], past_end["rows"]) == (1, [])


def _vod_streams() -> list[ProviderRecord]:
    return [
        # A rating below one sorts above an unparsable one, which counts as zero.
        {"stream_id": 1, "name": "MovieAlpha", "category_id": "1", "rating": "0.5", "added": "1"},
        {"stream_id": 2, "name": "MovieBeta", "category_id": "1", "rating": "bad", "added": "3"},
        {"stream_id": 3, "name": "MovieGamma", "category_id": "2", "rating": "9", "added": "9"},
    ]


def test_vod_page_loads_file_cache(api: TestClient) -> None:
    cats = [{"category_id": "1", "category_name": "Films"}]
    cache.save_file_cache("vod_data", {"cats": cats, "streams": _vod_streams()})
    resp = api.get("/vod?sort=alpha")
    assert resp.status_code == 200
    assert "MovieAlpha" in resp.text
    assert cache.get_cache().get("vod_streams") == _vod_streams()


def test_vod_page_restricted(api: TestClient) -> None:
    _set_sources(_XTREAM)
    auth.set_user_limits("testuser", None, ["movies:s1"])
    cache.get_cache().update(vod_categories=[], vod_streams=[])
    assert api.get("/vod").status_code == 403


def test_vod_page_applies_filter_and_sorts(api: TestClient) -> None:
    cache.get_cache().update(
        vod_categories=[
            {"category_id": "1", "category_name": "CatOne"},
            {"category_id": "2", "category_name": "CatTwo"},
        ],
        vod_streams=_vod_streams(),
    )
    settings = cache.load_user_settings("testuser")
    settings["vod_filter"] = ["1"]
    cache.save_user_settings("testuser", settings)
    by_rating = api.get("/vod?sort=rating").text
    newest = api.get("/vod?sort=newest").text
    assert "MovieGamma" not in by_rating
    assert "CatTwo" not in by_rating
    assert by_rating.index("MovieAlpha") < by_rating.index("MovieBeta")
    assert newest.index("MovieBeta") < newest.index("MovieAlpha")


def test_vod_browse_defers_rows_past_the_first_few(api: TestClient) -> None:
    cats: list[ProviderRecord] = [
        {"category_id": str(i), "category_name": f"Cat{i}"} for i in range(6)
    ]
    films: list[ProviderRecord] = [
        {"stream_id": i, "name": f"Film{i}", "category_id": str(i)} for i in range(6)
    ]
    cache.get_cache().update(vod_categories=cats, vod_streams=films)
    page = api.get("/vod?sort=alpha").text
    assert all(f"Film{i}" in page for i in range(4))
    assert "Film4" not in page
    assert "Cat5" in page
    assert "/api/vod/cards?category=5&sort=alpha&limit=20" in page


def test_vod_browse_row_links_to_full_category(api: TestClient) -> None:
    films: list[ProviderRecord] = [
        {"stream_id": i, "name": f"F{i:02}", "category_id": "1"} for i in range(21)
    ]
    cache.get_cache().update(
        vod_categories=[{"category_id": "1", "category_name": "Big"}],
        vod_streams=films,
    )
    page = api.get("/vod?sort=alpha").text
    assert "F19" in page
    assert "F20" not in page
    assert 'href="/vod?category=1&sort=alpha"' in page
    full = api.get("/vod?category=1&sort=alpha").text
    assert "F20" in full
    assert 'data-total="21"' in full


def test_vod_cards_returns_a_slice_of_one_category(api: TestClient) -> None:
    cache.get_cache().update(vod_categories=[], vod_streams=_vod_streams())
    resp = api.get("/api/vod/cards?category=1&sort=alpha&offset=1&limit=1")
    assert resp.status_code == 200
    assert "MovieBeta" in resp.text
    assert "MovieAlpha" not in resp.text
    assert "MovieGamma" not in resp.text
    assert api.get("/api/vod/cards?category=1&limit=999").status_code == 422


def test_cards_load_file_caches(api: TestClient) -> None:
    cats = [{"category_id": "1", "category_name": "Films"}]
    cache.save_file_cache("vod_data", {"cats": cats, "streams": _vod_streams()})
    cache.save_file_cache("series_data", {"cats": cats, "series": _series_list()})
    movies = api.get("/api/vod/cards?category=1&sort=unknown")
    series = api.get("/api/series/cards?category=1&sort=alpha")
    # An unknown sort keeps provider order.
    assert movies.text.index("MovieAlpha") < movies.text.index("MovieBeta")
    assert series.text.index("ShowA") < series.text.index("ShowB")
    assert "ShowC" not in series.text


def test_browsable_drops_unavailable_groups_and_filtered_categories() -> None:
    record: ProviderRecord = {"source_id": "s1", "category_id": "7"}
    assert main._browsable(record, "movies", set(), [])
    assert main._browsable(record, "movies", {"series:s1"}, ["7"])
    assert not main._browsable(record, "movies", {"movies:s1"}, [])
    assert not main._browsable(record, "movies", set(), ["8"])


def test_series_cards_restricted(api: TestClient) -> None:
    _set_sources(_XTREAM)
    auth.set_user_limits("testuser", None, ["series:s1"])
    cache.get_cache().update(series_categories=[], series=[])
    assert api.get("/api/series/cards?category=1").status_code == 403


def _series_list() -> list[ProviderRecord]:
    return [
        {
            "series_id": 1,
            "name": "ShowB",
            "category_id": "1",
            "rating": "0.5",
            "last_modified": "1",
        },
        {"series_id": 2, "name": "ShowA", "category_id": "1", "rating": "x", "last_modified": "3"},
        {"series_id": 3, "name": "ShowC", "category_id": "2", "rating": "9", "last_modified": "9"},
    ]


def test_series_page_loads_file_cache(api: TestClient) -> None:
    cats = [{"category_id": "1", "category_name": "Drama"}]
    cache.save_file_cache("series_data", {"cats": cats, "series": _series_list()})
    resp = api.get("/series?sort=alpha")
    assert resp.status_code == 200
    assert "ShowA" in resp.text
    assert cache.get_cache().get("series") == _series_list()


def test_series_page_without_cache_loads_in_background(
    api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    flags: set[str],
) -> None:
    threads = testing.ThreadRecorder(run_inline=False)
    testing.install_threads(monkeypatch, threads, main)
    first = api.get("/series")
    again = api.get("/series")
    assert first.status_code == again.status_code == 200
    (load,) = threads.threads
    assert flags == {"series_load"}
    cache.save_file_cache("series_data", {"cats": [], "series": _series_list()})
    load.run()
    assert cache.get_cache().get("series") == _series_list()
    assert not flags


def test_series_page_restricted(api: TestClient) -> None:
    _set_sources(_XTREAM)
    auth.set_user_limits("testuser", None, ["series:s1"])
    cache.get_cache().update(series_categories=[], series=[])
    assert api.get("/series").status_code == 403


def test_series_page_applies_filter_category_and_sorts(api: TestClient) -> None:
    cache.get_cache().update(
        series_categories=[
            {"category_id": "1", "category_name": "CatOne"},
            {"category_id": "2", "category_name": "CatTwo"},
        ],
        series=_series_list(),
    )
    settings = cache.load_user_settings("testuser")
    settings["series_filter"] = ["1", "2"]
    cache.save_user_settings("testuser", settings)
    alpha = api.get("/series?sort=alpha&category=1").text
    rating = api.get("/series?sort=rating&category=1").text
    newest = api.get("/series?sort=newest&category=1").text
    assert "ShowC" not in alpha
    assert alpha.index("ShowA") < alpha.index("ShowB")
    assert rating.index("ShowB") < rating.index("ShowA")
    assert newest.index("ShowA") < newest.index("ShowB")
    settings = cache.load_user_settings("testuser")
    settings["series_filter"] = ["2"]
    cache.save_user_settings("testuser", settings)
    filtered = api.get("/series?sort=alpha").text
    assert "ShowC" in filtered
    assert "ShowA" not in filtered


def test_series_detail_restricted(api: TestClient) -> None:
    auth.set_user_limits("testuser", None, ["series:s1"])
    cache.get_cache()["series"] = [{"series_id": 5, "source_id": "s1"}]
    assert api.get("/series/5").status_code == 403


def test_series_detail_without_xtream_source(api: TestClient) -> None:
    cache.get_cache()["series"] = []
    assert api.get("/series/5").status_code == 404


def test_series_detail_provider_error(api: TestClient, web: testing.FakeWeb) -> None:
    _set_sources({**_XTREAM, "id": "other"})
    cache.get_cache()["series"] = [{"series_id": 5, "source_id": "s1"}]
    resp = api.get("/series/5")
    assert resp.status_code == 502
    assert "Failed to load series info" in resp.text
    assert web.urls == [_api_url("http://x", "get_series_info", series_id="5")]


def test_series_detail_cleans_episode_metadata(api: TestClient, web: testing.FakeWeb) -> None:
    _set_sources(_XTREAM)
    cache.get_cache()["series"] = [{"series_id": 5, "source_id": "s1"}]
    cache.save_file_cache("series_info_s1_5", {"info": {"name": "Stale"}, "episodes": {}})
    series = {
        "info": {
            "name": "Show",
            "releaseDate": "2019-05-01",
            "director": "Ann Dir",
            "cast": "Cy Star, Di Lead",
        },
        "episodes": {
            "1": [
                {
                    "id": "11",
                    "episode_num": 1,
                    "title": "S01E01 - Show - Pilot",
                    "info": '{"plot": "Plot A"}',
                },
                {"id": "12", "episode_num": 2, "title": "Plain", "info": "not json"},
                {"id": "13", "episode_num": 3, "title": "", "info": {"description": "Desc C"}},
                {"id": "14", "episode_num": 4, "title": "Listy", "info": "[1, 2]"},
            ],
        },
    }
    web.routes[_api_url("http://x", "get_series_info", series_id="5")] = testing.FakeResponse(
        json.dumps(series).encode(),
    )
    resp = api.get("/series/5?refresh=true")
    assert resp.status_code == 200
    # A refresh bypasses the cached copy and replaces it under the source's key.
    stored = cache.load_file_cache("series_info_s1_5")
    assert stored is not None and stored[0] == series
    assert "Stale" not in resp.text
    assert "S01E01 - Show" not in resp.text
    assert ">Pilot<" in resp.text
    assert "Plot A" in resp.text
    assert "Desc C" in resp.text
    assert "2019" in resp.text
    people = re.findall(r'<a href="/search\?q=([^"]*)"[^>]*>([^<]*)</a>', resp.text)
    assert people == [("Ann%20Dir", "Ann Dir"), ("Cy%20Star", "Cy Star"), ("Di%20Lead", "Di Lead")]


def test_movie_detail_loads_vod_and_restricts(api: TestClient) -> None:
    movie = {"stream_id": 7, "name": "Film", "source_id": "s1"}
    cache.save_file_cache("vod_data", {"cats": [], "streams": [movie]})
    auth.set_user_limits("testuser", None, ["movies:s1"])
    assert api.get("/movie/7").status_code == 403
    assert cache.get_cache().get("vod_streams") == [movie]


def test_movie_detail_tolerates_provider_error(api: TestClient, web: testing.FakeWeb) -> None:
    _set_sources({**_XTREAM, "id": "other"})
    url = _api_url("http://x", "get_vod_info", vod_id="7")
    web.routes[url] = TimeoutError()
    cache.get_cache()["vod_streams"] = [{"stream_id": 7, "name": "Film", "source_id": "s1"}]
    resp = api.get("/movie/7")
    assert resp.status_code == 200
    assert "Film" in resp.text
    assert web.urls == [url]


def test_movie_detail_merges_provider_info(api: TestClient, web: testing.FakeWeb) -> None:
    _set_sources(_XTREAM)
    cache.get_cache()["vod_streams"] = [{"stream_id": 7, "name": "Film", "source_id": "s1"}]
    info = {
        "info": {
            "plot": "Great plot",
            "director": "Dir",
            "actors": "Actor A, Bé & Co,",
            "genre": "Drama",
            "rating": "7",
            "releasedate": "2001-02-03",
            "duration": "01:30",
            "movie_image": "/img.png",
            "youtube_trailer": "yt123",
        },
    }
    web.routes[_api_url("http://x", "get_vod_info", vod_id="7")] = testing.FakeResponse(
        json.dumps(info).encode(),
    )
    resp = api.get("/movie/7")
    assert resp.status_code == 200
    stored = cache.load_file_cache("vod_info_s1_7")
    assert stored is not None and stored[0] == info
    for text in ("Great plot", "Drama", "2001", "01:30", "yt123"):
        assert text in resp.text
    # Each credited name searches for itself; empty list entries are dropped.
    people = re.findall(r'<a href="/search\?q=([^"]*)"[^>]*>([^<]*)</a>', resp.text)
    assert people == [
        ("Dir", "Dir"),
        ("Actor%20A", "Actor A"),
        ("B%C3%A9%20%26%20Co", "Bé &amp; Co"),
    ]


# =============================================================================
# Player page
# =============================================================================


def test_player_movie_resumes_watch_position(api: TestClient, web: testing.FakeWeb) -> None:
    del web
    _set_sources(_XTREAM)
    cache.get_cache()["vod_streams"] = [{"stream_id": 9, "name": "Film", "source_id": "s1"}]
    cache.save_watch_position("testuser", "http://x/movie/u/p/9.mkv", 42.5, 100.0)
    auth.set_user_limits("testuser", None, ["cat:other"])
    resp = api.get("/play/movie/9?ext=mkv")
    assert resp.status_code == 200
    assert 'rawUrl: "http://x/movie/u/p/9.mkv"' in resp.text
    assert "serverResumePosition: 42.5" in resp.text


def test_player_series_strips_year_from_name(api: TestClient, web: testing.FakeWeb) -> None:
    _set_sources(_XTREAM)
    cache.get_cache()["series"] = [{"series_id": 5, "source_id": "s1"}]
    web.routes[_api_url("http://x", "get_series_info", series_id="5")] = testing.FakeResponse(
        json.dumps(
            {
                "info": {"name": "Great Show", "year": "2020"},
                "episodes": {
                    "1": [
                        {
                            "id": "12",
                            "episode_num": 2,
                            "title": "Next",
                            "container_extension": "mp4",
                        },
                        {"id": "11", "episode_num": 1, "title": "Pilot"},
                    ],
                },
            },
        ).encode(),
    )
    auth.set_user_limits("testuser", None, ["movies:s1"])
    resp = api.get("/play/series/11?series_id=5&ext=mkv")
    assert resp.status_code == 200
    assert 'rawUrl: "http://x/series/u/p/11.mkv"' in resp.text
    assert 'seriesName: "Great Show"' in resp.text
    assert "/play/series/12?series_id=5\\u0026ext=mp4" in resp.text


@pytest.mark.parametrize(
    ("path", "group"),
    [
        ("/play/live/1", "cat:3"),
        ("/play/movie/1", "movies:s1"),
        ("/play/series/1?series_id=7", "series:s1"),
    ],
)
def test_player_enforces_group_restrictions(
    api: TestClient,
    guide_db: None,
    web: testing.FakeWeb,
    path: str,
    group: str,
) -> None:
    del guide_db, web
    _set_sources(_XTREAM)
    cache.get_cache().update(
        live_streams=[
            {"stream_id": 1, "source_id": "s1", "category_ids": ["3"], "direct_url": "http://up/x"},
        ],
        vod_streams=[{"stream_id": 1, "source_id": "s1"}],
        series=[{"series_id": 7, "source_id": "s1"}],
    )
    assert api.get(path).status_code == 200
    auth.set_user_limits("testuser", None, [group])
    assert api.get(path).status_code == 403


def test_player_live_playlist_stream_transcodes_with_source_and_guide(
    api: TestClient,
    guide_db: None,
    prestarts: list[str],
) -> None:
    del guide_db
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "live_dvr_mins": 0,
            "sources": [{"id": "m", "name": "M", "type": "m3u", "url": "http://m"}],
        },
    )
    cache.get_cache()["live_streams"] = [
        {
            "stream_id": "m_1",
            "name": "CNN",
            "source_id": "m",
            "direct_url": "http://m/1.ts",
            "epg_channel_id": "cnn",
        },
    ]
    now = datetime.now(timezone.utc)
    _insert_program("cnn", "m", now - timedelta(minutes=5), now + timedelta(hours=1), "Breaking")
    resp = api.get("/play/live/m_1")
    assert resp.status_code == 200
    # Playlist streams always go through the server; the source keeps its default.
    assert 'transcodeMode: "always"' in resp.text
    # So the server starts the stream while the page loads.
    assert prestarts == ["http://m/1.ts"]
    assert "deinterlaceFallback: true" in resp.text
    assert ">Breaking</div>" in resp.text


def test_player_missing_stream(api: TestClient, guide_db: None) -> None:
    del guide_db
    cache.get_cache()["live_streams"] = []
    resp = api.get("/play/live/1")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Stream not found"}


# =============================================================================
# Search and stream redirects
# =============================================================================


@pytest.mark.usefixtures("guide_db")
def test_search_loads_data_and_applies_filters(api: TestClient) -> None:
    imdb.init()
    live = [
        {"stream_id": 1, "name": "News One", "category_ids": ["1"]},
        {"stream_id": 2, "name": "News Two", "category_ids": ["2"]},
    ]
    vod = [
        {"stream_id": 3, "name": "News Movie", "category_id": "1"},
        {"stream_id": 4, "name": "News Film", "category_id": "2"},
    ]
    series = [
        {"series_id": 5, "name": "News Show", "category_id": "1"},
        {"series_id": 6, "name": "News Serial", "category_id": "2"},
    ]
    cache.save_file_cache("live_data", {"cats": [], "streams": live, "epg_urls": []})
    cache.save_file_cache("vod_data", {"cats": [], "streams": vod})
    cache.save_file_cache("series_data", {"cats": [], "series": series})
    settings = cache.load_user_settings("testuser")
    settings["guide_filter"] = ["1"]
    settings["vod_filter"] = ["1"]
    settings["series_filter"] = ["1"]
    cache.save_user_settings("testuser", settings)
    resp = api.get("/search?q=news")
    assert resp.status_code == 200
    for shown in ("News One", "News Movie", "News Show"):
        assert shown in resp.text
    for hidden in ("News Two", "News Film", "News Serial"):
        assert hidden not in resp.text
    assert cache.get_cache().get("live_streams") == live
    assert cache.get_cache().get("vod_streams") == vod
    assert cache.get_cache().get("series") == series


def test_stream_redirect_without_xtream(api: TestClient) -> None:
    assert api.get("/stream/movie/1").status_code == 404


def test_stream_redirect_hidden_source(api: TestClient) -> None:
    del api
    _set_sources({**_XTREAM, "users": ["someone"]})
    auth.create_user("viewer", "password1")
    assert _client("viewer").get("/stream/movie/1").status_code == 403


def test_stream_redirect_to_provider(api: TestClient) -> None:
    _set_sources({**_XTREAM, "url": "http://up"})
    resp = api.get("/stream/movie/1?ext=mkv", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "http://up/movie/u/p/1.mkv"


# =============================================================================
# Transcode routes
# =============================================================================


def _limited_source(tmp_path: Path) -> None:
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "sources": [
                {"id": "s1", "name": "S", "type": "m3u", "url": "http://up", "max_streams": 3},
            ],
            "transcode_dir": str(tmp_path / "transcodes"),
            "probe_live": False,
        },
    )
    auth.set_user_limits("testuser", {"s1": 2}, None)


def _start(url: str, **params: str) -> dict[str, str]:
    """Query for /transcode/start with the token a player page gives ``testuser``."""
    return {"url": url, "url_token": auth.stream_url_token("testuser", url), **params}


def test_transcode_start_enforces_source_limit(
    api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sessions: dict[str, TranscodeSession],
) -> None:
    _limited_source(tmp_path)
    for name in ("a", "b", "c"):
        sessions[name] = _session(tmp_path / name, username="other", source_id="s1")
    launcher = testing.ProcessLauncher()
    testing.install_launcher(monkeypatch, launcher, ffmpeg_session)
    resp = api.get("/transcode/start", params=_start("http://up/l", source_id="s1"))
    assert resp.status_code == 429
    assert resp.json() == {"detail": "Source at capacity (3 streams)"}
    assert set(sessions) == {"a", "b", "c"}
    assert launcher.launches == []


def test_transcode_start_rotates_users_oldest_stream(
    api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sessions: dict[str, TranscodeSession],
    web: testing.FakeWeb,
) -> None:
    _limited_source(tmp_path)
    sessions["old"] = _session(tmp_path / "old", source_id="s1", started=1.0)
    sessions["new"] = _session(tmp_path / "new", source_id="s1", started=2.0)
    web.routes["http://up/l"] = testing.ended_upstream
    probe = testing.FakeRun()
    probe.program("ffprobe", testing.Program(stdout=testing.probed_video_json()))
    testing.install_subprocess(monkeypatch, ffmpeg_command, run=probe)
    launcher = testing.ProcessLauncher([testing.FakeProcess(exited=True, exit_code=1)])
    testing.install_launcher(monkeypatch, launcher, ffmpeg_session)
    testing.install_clock(monkeypatch, testing.FakeClock(), ffmpeg_session)
    tasks = testing.BackgroundTasks()
    monkeypatch.setattr(ffmpeg_session, "_spawn_background_task", tasks)
    resp = api.get("/transcode/start", params=_start("http://up/l", source_id="s1"))
    tasks.discard()
    # The user's limit of two stopped their oldest stream before the start, which
    # then failed because the scripted ffmpeg exited at once.
    assert resp.status_code == 500
    assert set(sessions) == {"new"}
    assert not (tmp_path / "old").exists()
    # The new session's relay read the upstream; its worker read the relay.
    assert "http://up/l" in web.urls
    [command] = launcher.commands
    assert command[command.index("-i") + 1] == "pipe:0"


def test_transcode_seek_and_progress(
    api: TestClient,
    tmp_path: Path,
    sessions: dict[str, TranscodeSession],
) -> None:
    sessions["abc"] = _session(tmp_path / "abc", is_vod=True)
    sessions["abc"]["process"] = testing.FakeProcess(exit_code=None)
    for segment in ("seg003.ts", "seg004.ts"):
        (tmp_path / "abc" / segment).write_bytes(b"\x47" * 1_001)
    seek = api.get("/transcode/seek/abc?time=10")
    progress = api.get("/transcode/progress/abc")
    assert seek.json() == {"session_id": "abc", "playlist": "/transcode/abc/stream.m3u8"}
    assert sessions["abc"].get("seek_offset") == 10.0
    assert progress.json() == {"segment_count": 2, "duration": 6.0}


def test_transcode_file_rejects_dotted_name(api: TestClient) -> None:
    assert api.get("/transcode/abc/x..m3u8").status_code == 400


def test_transcode_file_missing_file(
    api: TestClient,
    tmp_path: Path,
    sessions: dict[str, TranscodeSession],
) -> None:
    sessions["abc"] = _session(tmp_path / "abc")
    resp = api.get("/transcode/abc/seg1.ts")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "File not found"}


def test_transcode_file_serves_segment_to_chromecast(
    api: TestClient,
    tmp_path: Path,
    sessions: dict[str, TranscodeSession],
) -> None:
    sessions["abc"] = _session(tmp_path / "abc", last_access=0.0)
    (tmp_path / "abc" / "seg1.ts").write_bytes(b"\x47segment")
    resp = api.get("/transcode/abc/seg1.ts", headers={"user-agent": "CrKey/1.0"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp2t"
    assert resp.headers["access-control-allow-origin"] == "*"
    assert resp.content == b"\x47segment"
    assert sessions["abc"].get("last_access", 0.0) > 0.0


def test_subtitle_file_rejects_dotted_name(api: TestClient) -> None:
    assert api.get("/subs/abc/x..vtt").status_code == 400


def _subtitle(
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, TranscodeSession],
    directory: Path,
) -> tuple[bytes, list[float]]:
    """Serve ``sub0.vtt`` from a session in ``directory``; return it and the sleeps."""
    sessions["abc"] = _session(directory)
    clock = testing.FakeClock()
    testing.install_clock(monkeypatch, clock, main)
    response = asyncio.run(main.subtitle_file("abc", "sub0.vtt"))
    assert response.headers["access-control-allow-origin"] == "*"
    return bytes(response.body), clock.sleeps


def test_subtitle_file_serves_ready_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sessions: dict[str, TranscodeSession],
) -> None:
    body = "WEBVTT\n\n00:00.000 --> 00:01.000\nHello\n"
    tmp_path.joinpath("sub0.vtt").write_text(body)
    assert _subtitle(monkeypatch, sessions, tmp_path) == (body.encode(), [])


def test_subtitle_file_missing_returns_empty_vtt_at_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sessions: dict[str, TranscodeSession],
) -> None:
    # The player polls again shortly; holding the request would only delay it.
    assert _subtitle(monkeypatch, sessions, tmp_path) == (b"WEBVTT\n\n", [])


def test_subtitle_file_undecodable_returns_empty_vtt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sessions: dict[str, TranscodeSession],
) -> None:
    tmp_path.joinpath("sub0.vtt").write_bytes(b"\xff" * 30)
    assert _subtitle(monkeypatch, sessions, tmp_path)[0] == b"WEBVTT\n\n"


def test_transcode_clear_refuses_cast_url(
    api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    casting = CastSession(
        username="testuser",
        host="192.168.1.5",
        session_id="sid",
        url="http://lan/transcode/sid/stream.m3u8",
        source_url="http://up/x",
        name="TV",
    )
    monkeypatch.setattr(chromecast.manager, "_sessions", {"testuser": casting})
    resp = api.delete("/transcode-clear", params={"url": "http://up/x"})
    assert resp.status_code == 409


def test_transcode_clear_stops_cached_session(
    api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sessions: dict[str, TranscodeSession],
) -> None:
    sessions["sid"] = _session(tmp_path / "sid", url="http://up/x", is_vod=True)
    urls = {"http://up/x": "sid"}
    monkeypatch.setattr(ffmpeg_session, "_url_to_session", urls)
    resp = api.delete("/transcode-clear", params={"url": "http://up/x"})
    assert resp.json() == {"status": "cleared", "session_id": "sid"}
    assert sessions == {}
    assert urls == {}
    assert not (tmp_path / "sid").exists()


# =============================================================================
# Settings
# =============================================================================


def test_settings_page_loads_file_caches(
    api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path / "no-engines"))
    cache.save_file_cache(
        "live_data",
        {"cats": [{"category_id": "1", "category_name": "LiveCat"}], "streams": []},
    )
    cache.save_file_cache(
        "vod_data",
        {"cats": [{"category_id": "2", "category_name": "VodCat"}], "streams": []},
    )
    cache.save_file_cache(
        "series_data",
        {"cats": [{"category_id": "3", "category_name": "SeriesCat"}], "series": []},
    )
    resp = api.get("/settings")
    assert resp.status_code == 200
    for name in ("LiveCat", "VodCat", "SeriesCat"):
        assert name in resp.text
    data = cache.get_cache()
    assert data.get("epg_urls") == []
    assert data.get("vod_streams") == []
    assert data.get("series") == []


def test_settings_page_starts_background_load_without_cache(
    api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flags: set[str],
) -> None:
    monkeypatch.setenv("SR_ENGINE_DIR", str(tmp_path / "no-engines"))
    threads = testing.ThreadRecorder(run_inline=False)
    testing.install_threads(monkeypatch, threads, main)
    resp = api.get("/settings")
    assert resp.status_code == 200
    (load,) = threads.threads
    assert load.daemon
    assert flags == {"guide_load"}
    load.run()
    # No sources load no channels, and the guide database is not open.
    assert cache.get_cache().get("live_streams") == []
    assert cache.get_cache().get("epg_error") == "Database not initialized"
    assert not flags


@pytest.mark.parametrize(("path", "setting"), [("vod", "vod_filter"), ("series", "series_filter")])
def test_settings_category_filters(
    api: TestClient,
    path: str,
    setting: Literal["vod_filter", "series_filter"],
) -> None:
    resp = api.post(f"/settings/{path}-filter", json={"cats": ["1", "2"]})
    assert resp.json() == {"status": "ok"}
    assert cache.load_user_settings("testuser").get(setting) == ["1", "2"]


def _source_form(**overrides: str | int) -> dict[str, str | int]:
    return {"name": "Prov", "source_type": "xtream", "url": "http://up.example/"} | overrides


def test_add_source_requires_admin(api: TestClient) -> None:
    del api
    auth.create_user("viewer", "password1")
    resp = _client("viewer").post("/settings/add", data=_source_form())
    assert resp.status_code == 403
    assert resp.json() == {"detail": "Admin access required"}
    assert cache.load_server_settings()["sources"] == []


def test_add_source_rejects_blank_name(api: TestClient) -> None:
    assert api.post("/settings/add", data=_source_form(name="   ")).status_code == 400


def test_add_source_normalizes_schedule(api: TestClient) -> None:
    cache.get_cache()["live_streams"] = [{"stream_id": 1}]
    resp = api.post(
        "/settings/add",
        data=_source_form(epg_schedule="8:00, bad, 23:30"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "live_streams" not in cache.get_cache()
    source = cache.load_server_settings()["sources"][-1]
    assert source.get("epg_schedule") == ["08:00", "23:30"]
    assert source["url"] == "http://up.example"


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": " "},
        {"source_type": "bogus"},
        {"url": "ftp://up.example"},
        {"name": "x" * 201},
    ],
)
def test_edit_source_validates(api: TestClient, overrides: dict[str, str]) -> None:
    assert api.post("/settings/edit/s1", data=_source_form(**overrides)).status_code == 400


def test_edit_source_updates_matching_source(api: TestClient) -> None:
    _set_sources(
        {"id": "other", "name": "Other", "type": "m3u", "url": "http://other"},
        {"id": "s1", "name": "Old", "type": "m3u", "url": "http://old"},
    )
    cache.get_cache()["live_streams"] = [{"stream_id": 1}]
    resp = api.post(
        "/settings/edit/s1",
        data=_source_form(
            name="New",
            epg_schedule="7:15,nope",
            epg_url=" http://epg.example ",
            max_streams=-4,
        ),
    )
    assert resp.json() == {"ok": True}
    assert "live_streams" not in cache.get_cache()
    other, edited = cache.load_server_settings()["sources"]
    assert other["name"] == "Other"
    assert edited["name"] == "New"
    assert edited["type"] == "xtream"
    assert edited["url"] == "http://up.example"
    assert edited.get("epg_schedule") == ["07:15"]
    assert edited.get("epg_url") == "http://epg.example"
    assert edited.get("max_streams") == 0


def test_delete_source_clears_its_data(api: TestClient, guide_db: None) -> None:
    del guide_db
    _set_sources(
        {"id": "keep", "name": "Keep", "type": "m3u", "url": "http://keep"},
        {"id": "s1", "name": "Gone", "type": "m3u", "url": "http://gone"},
    )
    now = datetime.now(timezone.utc)
    for source_id in ("keep", "s1"):
        _insert_program("ch", source_id, now, now + timedelta(hours=1))
    cache.save_file_cache("live_data", _LIVE_FILE_CACHE)
    resp = api.post("/settings/delete/s1", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings"
    assert [s["id"] for s in cache.load_server_settings()["sources"]] == ["keep"]
    remaining = epg.get_programs_in_range("ch", now, now + timedelta(hours=1))
    assert [p.source_id for p in remaining] == ["keep"]
    assert cache.load_file_cache("live_data") is None


# =============================================================================
# Guide refresh
# =============================================================================


def test_guide_refresh_workers(
    api: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    guide_db: None,
    web: testing.FakeWeb,
    flags: set[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    del guide_db
    threads = testing.ThreadRecorder(run_inline=False)
    testing.install_threads(monkeypatch, threads, main, m3u_module)
    live = {
        "cats": [{"category_id": "c"}],
        "streams": [{"stream_id": 1, "name": "S"}],
        "epg_urls": [["http://e/epg", 5, "x"]],
    }
    # A stale file cache is served while it refreshes, then rewritten fresh.
    (cache.CACHE_DIR / "live_data.json").write_text(json.dumps({"data": live, "timestamp": 0}))
    resp = api.get("/guide/refresh", follow_redirects=False)
    again = api.get("/guide/refresh", follow_redirects=False)
    assert resp.status_code == again.status_code == 303
    assert resp.headers["location"] == "/guide?refreshing=1"
    assert {"live_refresh", "epg_refresh"} <= flags
    refresh_live, refresh_epg = threads.threads
    data = cache.get_cache()

    refresh_live.run()
    assert data.get("live_streams") == live["streams"]
    assert data.get("epg_urls") == [("http://e/epg", 5, "x")]
    saved = cache.load_file_cache("live_data")
    assert saved is not None and saved[0] == live and saved[1] > 0
    assert "live_refresh" not in flags

    now = datetime.now(timezone.utc)
    _insert_program("old", "x", now, now + timedelta(hours=1))
    web.routes["http://e/epg"] = _xmltv_route("ch", "Fresh", now)
    data["epg_error"] = "old"
    refresh_epg.run()
    assert web.urls == ["http://e/epg"]
    assert epg.get_program_count() == 1
    assert "epg_error" not in data
    assert "epg_refresh" not in flags

    (cache.CACHE_DIR / "live_data.json").unlink()
    _corrupt_server_settings()
    assert refresh_live.target is not None and refresh_epg.target is not None
    with caplog.at_level(logging.WARNING, logger="netv.main"):
        refresh_live.target()
        refresh_epg.target()
        assert data.get("epg_error") == "live_dvr_mins must be an integer"
        no_epg_urls: list[tuple[str, int, str]] = []
        data["epg_urls"] = no_epg_urls
        refresh_epg.target()
    assert "Live refresh failed: live_dvr_mins must be an integer" in caplog.text
    assert "EPG refresh: no EPG URLs available" in caplog.text
    assert not {"live_refresh", "epg_refresh"} & flags


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
