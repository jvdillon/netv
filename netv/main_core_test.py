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
from treekle import json_loads

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
    built: list[list[ProviderRecord]] = []
    tokens: list[str] = []
    # A TMDB key, when the environment has one, reaches the build; none is needed.
    monkeypatch.setenv("TMDB_API_KEY", "t.o.k")

    def failing_build(records: list[ProviderRecord], *, tmdb_token: str) -> int:
        built.append(records)
        tokens.append(tmdb_token)
        raise OSError("IMDb unreachable")

    monkeypatch.setattr(imdb, "build", failing_build)
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
                # Once both catalogs load, a failed IMDb build is logged; the loop then
                # ends with the server.
                cache.get_cache()["vod_streams"] = [{"name": "Heat - 1995"}]
                imdb_loop.run()
            assert built == [[{"name": "Heat - 1995"}]]
            assert tokens == ["t.o.k"]
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
    assert web.urls[6:] == [stream_url]
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


@pytest.fixture
def imdb_index(guide_db: None) -> None:
    """The IMDb index tables cards read their ratings from, in the per-test database."""
    del guide_db
    imdb.init()


def _vod_streams() -> list[ProviderRecord]:
    return [
        # A rating below one sorts above an unparsable one, which counts as zero.
        {"stream_id": 1, "name": "MovieAlpha", "category_id": "1", "rating": "0.5", "added": "1"},
        {"stream_id": 2, "name": "MovieBeta", "category_id": "1", "rating": "bad", "added": "3"},
        {"stream_id": 3, "name": "MovieGamma", "category_id": "2", "rating": "9", "added": "9"},
    ]


@pytest.mark.usefixtures("imdb_index")
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


@pytest.mark.usefixtures("imdb_index")
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


@pytest.mark.usefixtures("imdb_index")
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


@pytest.mark.usefixtures("imdb_index")
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


@pytest.mark.usefixtures("imdb_index")
def test_vod_cards_returns_a_slice_of_one_category(api: TestClient) -> None:
    cache.get_cache().update(vod_categories=[], vod_streams=_vod_streams())
    resp = api.get("/api/vod/cards?category=1&sort=alpha&offset=1&limit=1")
    assert resp.status_code == 200
    assert "MovieBeta" in resp.text
    assert "MovieAlpha" not in resp.text
    assert "MovieGamma" not in resp.text
    assert api.get("/api/vod/cards?category=1&limit=999").status_code == 422


def _heat_versions() -> list[ProviderRecord]:
    """One movie the provider lists three times, under two spellings, and another."""
    return [
        {"stream_id": 1, "name": "Heat - 1995", "tmdb": 949, "category_id": "1", "added": "100"},
        {"stream_id": 2, "name": "Heat (1995)", "tmdb": "949", "category_id": "1", "added": "300"},
        {"stream_id": 3, "name": "Heat - 1995", "tmdb": 949, "category_id": "2", "added": "200"},
        {"stream_id": 4, "name": "Ronin - 1998", "tmdb": 8195, "category_id": "1", "added": "50"},
    ]


def _movie_links(page: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'href="/movie/(\d+)"', page)]


def test_one_per_title_keeps_the_earliest_added_movie_then_lowest_id() -> None:
    listings: list[ProviderRecord] = [
        {"stream_id": 6, "name": "Heat - 1995", "tmdb": 949, "added": "7"},
        {"stream_id": 5, "name": "Heat - 1995", "tmdb": 949, "added": "7"},
        {"stream_id": 8, "name": "Heat - 1995", "tmdb": 949, "added": "7"},
    ]
    assert main._one_per_title(listings, per_category=False) == [listings[1]]
    # When it was added counts before its id.
    earlier: ProviderRecord = {"stream_id": 9, "name": "Heat - 1995", "tmdb": 949, "added": "6"}
    assert main._one_per_title([*listings, earlier], per_category=False) == [earlier]


def _house_of_cards() -> list[ProviderRecord]:
    """One series the provider lists twice (as House of Cards is), and another."""
    common: ProviderRecord = {"tmdb": "1425", "source_id": "s1", "releaseDate": "2013-02-01"}
    july = int(datetime(2026, 7, 30, 12, tzinfo=timezone.utc).timestamp())
    october = int(datetime(2026, 10, 9, 12, tzinfo=timezone.utc).timestamp())
    return [
        {
            **common,
            "series_id": 16_566,
            "name": "House of Cards (US) (2013)",
            "category_id": "1",
            "last_modified": str(july),
        },
        {
            **common,
            "series_id": 16_486,
            "name": "House of Cards",
            "category_id": "2",
            "last_modified": str(october),
        },
        {"series_id": 7, "name": "Homeland", "tmdb": "1407", "source_id": "s1", "category_id": "1"},
    ]


def test_one_per_title_keeps_the_most_recently_updated_series() -> None:
    # A series listing gains episodes as it updates, so the newest has the most.
    older, newer, other = _house_of_cards()
    assert main._one_per_title([older, other, newer], per_category=False) == [newer, other]
    tie: ProviderRecord = {**newer, "series_id": 16_999}
    assert main._one_per_title([newer, tie], per_category=False) == [tie]


@pytest.mark.usefixtures("imdb_index")
def test_series_show_once_in_search_and_once_per_category(api: TestClient) -> None:
    cache.get_cache().update(
        live_streams=[],
        vod_streams=[],
        series_categories=[
            {"category_id": "1", "category_name": "US"},
            {"category_id": "2", "category_name": "Drama"},
        ],
        series=_house_of_cards(),
    )
    assert _series_links(api.get("/search?q=house&series=true&min_imdb=").text) == ["16486"]
    assert _series_links(api.get("/series?category=1&sort=alpha").text) == ["7", "16566"]


@pytest.mark.usefixtures("imdb_index")
def test_series_detail_offers_each_listing(api: TestClient, web: testing.FakeWeb) -> None:
    _set_sources(_XTREAM)
    older, newer, other = _house_of_cards()
    # Listings whose details cannot load: one from a source since removed, one whose
    # provider times out. Each still says when it changed.
    orphan: ProviderRecord = {
        **older,
        "series_id": 17_001,
        "source_id": "gone",
        "last_modified": "1",
    }
    slow: ProviderRecord = {**older, "series_id": 17_002, "last_modified": "2"}
    cache.get_cache()["series"] = [older, newer, other, orphan, slow]
    episodes = {"1": [{"id": "1", "episode_num": 1}, {"id": "2", "episode_num": 2}]}
    for series_id, eps in (("16566", {"1": episodes["1"][:1]}), ("16486", episodes)):
        web.routes[_api_url("http://x", "get_series_info", series_id=series_id)] = (
            testing.FakeResponse(json.dumps({"info": {"name": "HoC"}, "episodes": eps}).encode())
        )
    web.routes[_api_url("http://x", "get_series_info", series_id="17002")] = TimeoutError()
    page = api.get("/series/16566").text
    versions = re.findall(r'data-version="(\d+)"( aria-current="true")?>\s*([^<]*?)\s*<', page)
    # Newest first, as cards open it; each says when it last changed and how much it has.
    stamp = f"{datetime.fromtimestamp(2):%b %-d, %Y}"
    assert versions == [
        ("16486", "", "Updated Oct 9, 2026 · 2 episodes"),
        ("16566", ' aria-current="true"', "Updated Jul 30, 2026 · 1 episode"),
        ("17002", "", f"Updated {stamp}"),
        ("17001", "", f"Updated {datetime.fromtimestamp(1):%b %-d, %Y}"),
    ]
    assert 'href="/series/16486"' in page
    assert "data-version" not in api.get("/series/7").text
    # A series missing from the catalog has no other listings to offer.
    web.routes[_api_url("http://x", "get_series_info", series_id="999")] = testing.FakeResponse(
        b'{"info": {"name": "Gone"}, "episodes": {}}',
    )
    assert "data-version" not in api.get("/series/999").text


@pytest.mark.usefixtures("imdb_index")
def test_vod_browse_shows_each_movie_once_per_category(api: TestClient) -> None:
    cache.get_cache().update(
        vod_categories=[
            {"category_id": "1", "category_name": "CatOne"},
            {"category_id": "2", "category_name": "CatTwo"},
        ],
        vod_streams=_heat_versions(),
    )
    # The oldest version stands for the movie in each category that lists it: the
    # provider's re-uploads are often dead (37 of 40 sampled), its originals rarely.
    assert _movie_links(api.get("/vod?sort=alpha").text) == ["1", "4", "3"]
    full = api.get("/vod?category=1&sort=alpha").text
    assert 'data-total="2"' in full
    assert _movie_links(full) == ["1", "4"]
    assert _movie_links(api.get("/api/vod/cards?category=1&sort=newest").text) == ["1", "4"]


@pytest.mark.usefixtures("imdb_index")
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


@pytest.mark.usefixtures("imdb_index")
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


def _imdb_facts(rating: float, votes: int, year: int) -> imdb.Info:
    return {
        "imdb_id": "tt1",
        "title": "",
        "year": year,
        "genres": [],
        "runtime_min": 0,
        "rating": rating,
        "votes": votes,
        "directors": [],
        "writers": [],
        "cast": [],
        "characters": [],
        "crew": {},
    }


def _ranked_catalog() -> tuple[list[ProviderRecord], list[ProviderRecord]]:
    """Movies and series IMDb ranks; Delta and Epsilon it does not, nor their years."""
    imdb.init()
    imdb._write(
        {
            f"{kind}:tmdb:{n}": facts
            for kind in ("movies", "series")
            for n, facts in (
                (1, _imdb_facts(6.0, 5, 1990)),
                (2, _imdb_facts(8.0, 10, 2010)),
                (3, _imdb_facts(8.0, 99, 2001)),
            )
        },
        set(),
    )
    # The provider rates the titles IMDb does not.
    titles = (
        ("Alpha - 1990", ""),
        ("Beta - 2010", ""),
        ("Gamma - 2001", ""),
        ("Delta", "3"),
        ("Epsilon", "5"),
    )
    movies: list[ProviderRecord] = [
        {"stream_id": n, "name": name, "tmdb": n, "category_id": "1", "rating": rating}
        for n, (name, rating) in enumerate(titles, 1)
    ]
    series: list[ProviderRecord] = [
        {"series_id": n, "name": name, "tmdb": n, "category_id": "1", "rating": rating}
        for n, (name, rating) in enumerate(titles, 1)
    ]
    return movies, series


def _series_links(page: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'href="/series/(\d+)"', page)]


@pytest.mark.usefixtures("guide_db")
def test_browse_sorts_by_imdb_rating_and_by_year(api: TestClient) -> None:
    movies, series = _ranked_catalog()
    cats: list[ProviderRecord] = [{"category_id": "1", "category_name": "CatOne"}]
    cache.get_cache().update(
        vod_categories=cats,
        vod_streams=movies,
        series_categories=cats,
        series=series,
    )
    # Best rated first, more votes breaking a tie; what IMDb has not rated last, by
    # the provider's rating.
    by_imdb = api.get("/vod?category=1&sort=imdb").text
    assert _movie_links(by_imdb) == ["3", "2", "1", "5", "4"]
    assert 'value="imdb" selected' in by_imdb
    # Cards show the IMDb rating they sort by; unrated ones, the provider's.
    shown = re.findall(r'<span class="text-yellow-400">([^<]*)<', by_imdb)
    assert shown == ["IMDb 8.0", "IMDb 8.0", "IMDb 6.0", "★ 5.0", "★ 3.0"]
    # Newest year first; titles without one last.
    by_year = api.get("/api/vod/cards?category=1&sort=year").text
    assert _movie_links(by_year) == ["2", "3", "1", "4", "5"]
    assert _series_links(api.get("/series?category=1&sort=imdb").text) == ["3", "2", "1", "5", "4"]
    assert _series_links(api.get("/series?category=1&sort=year").text) == ["2", "3", "1", "4", "5"]


@pytest.mark.usefixtures("guide_db")
def test_search_sorts_movies_and_series_before_limiting(api: TestClient) -> None:
    movies, series = _ranked_catalog()
    cache.get_cache().update(live_streams=[], vod_streams=movies, series=series)

    # The limit applies to movies and series together, after one sort of them all.
    by_imdb = api.get("/search?q=a&sort=imdb&limit=4").text
    # The four best rated, not the first four by name.
    assert (_movie_links(by_imdb), _series_links(by_imdb)) == (["3", "2"], ["3", "2"])
    assert 'value="imdb" selected' in by_imdb
    by_year = api.get("/search?q=a&sort=year&limit=4").text
    assert (_movie_links(by_year), _series_links(by_year)) == (["2", "3"], ["2", "3"])
    by_name = api.get("/search?q=a&sort=alpha&limit=4").text
    assert (_movie_links(by_name), _series_links(by_name)) == (["1", "2"], ["1", "2"])
    assert 'value="alpha" selected' in by_name
    assert len(_movie_links(api.get("/search?q=a&sort=imdb&limit=1").text)) == 1


@pytest.mark.usefixtures("guide_db")
def test_sort_choice_is_remembered_across_search_and_browse(api: TestClient) -> None:
    movies, series = _ranked_catalog()
    cats: list[ProviderRecord] = [{"category_id": "1", "category_name": "CatOne"}]
    cache.get_cache().update(
        live_streams=[],
        vod_categories=cats,
        vod_streams=movies,
        series_categories=cats,
        series=series,
    )
    # Before any choice, search sorts by name and Browse All by IMDb rating.
    assert _movie_links(api.get("/search?q=a").text) == ["1", "2", "4", "3"]
    assert 'href="/vod?sort=imdb"' in api.get("/vod").text

    api.get("/search?q=a&sort=year")

    # Later searches and browsing use the choice until another is made.
    assert cache.load_user_settings("testuser").get("results_sort") == "year"
    assert _movie_links(api.get("/search?q=a").text) == ["2", "3", "1", "4"]
    assert 'href="/series?sort=year"' in api.get("/series").text
    api.get("/vod?category=1&sort=imdb")
    assert 'value="imdb" selected' in api.get("/search?q=a").text
    # A sort the pages do not offer is not remembered.
    api.get("/vod?sort=bogus")
    assert cache.load_user_settings("testuser").get("results_sort") == "imdb"
    # "Rating" sorted by the provider's number, which no card shows. Old links and a
    # remembered "rating" now mean the rating cards show.
    api.get("/vod?category=1&sort=year")
    by_old_link = api.get("/vod?category=1&sort=rating").text
    assert _movie_links(by_old_link) == ["3", "2", "1", "5", "4"]
    assert 'value="imdb" selected' in by_old_link
    assert 'value="rating"' not in by_old_link
    settings = cache.load_user_settings("testuser")
    assert settings.get("results_sort") == "imdb"
    settings["results_sort"] = "rating"
    cache.save_user_settings("testuser", settings)
    remembered = api.get("/search?q=a").text
    assert 'value="imdb" selected' in remembered
    assert _movie_links(remembered) == ["3", "2", "1", "4"]


@pytest.mark.usefixtures("imdb_index")
def test_pages_show_one_rating_and_year_per_title(api: TestClient, web: testing.FakeWeb) -> None:
    _set_sources(_XTREAM)
    imdb._write(
        {
            "movies:tmdb:5": _imdb_facts(7.0, 127_509, 1976),
            "series:tmdb:6": _imdb_facts(8.5, 10, 2007),
        },
        set(),
    )
    rated: ProviderRecord = {
        "stream_id": 5,
        "name": "Spy - 1977",
        "tmdb": 5,
        "source_id": "s1",
        "category_id": "1",
        "rating": "6.82",
    }
    # IMDb knows nothing of this one, so its rating and year are the provider's.
    unrated: ProviderRecord = {**rated, "stream_id": 7, "name": "Other - 1977", "tmdb": 7}
    show: ProviderRecord = {"series_id": 6, "name": "Show", "tmdb": 6, "source_id": "s1"}
    cache.get_cache().update(live_streams=[], vod_streams=[rated, unrated], series=[show])
    for stream_id in ("5", "7"):
        info = {"info": {"name": "Spy", "rating": "6.82", "releasedate": "1977-07-07"}}
        web.routes[_api_url("http://x", "get_vod_info", vod_id=stream_id)] = testing.FakeResponse(
            json.dumps(info).encode(),
        )
    series_info: ProviderRecord = {
        "info": {"name": "Show", "rating": "6.1", "releaseDate": "2008-01-01"},
        "episodes": {},
    }
    web.routes[_api_url("http://x", "get_series_info", series_id="6")] = testing.FakeResponse(
        json.dumps(series_info).encode(),
    )

    def year_and_rating(page: str) -> list[str]:
        found = re.finditer(r'<span(?: class="text-yellow-400")?>([^<]*)</span>', page)
        return [str(m.group(1)) for m in found][:2]

    # Detail pages, cards, and the player all show IMDb's rating and year.
    card = r'<span class="text-gray-400">([^<]*)</span><span class="text-yellow-400">([^<]*)<'
    assert year_and_rating(api.get("/movie/5").text) == ["1976", "IMDb 7.0"]
    assert re.findall(card, api.get("/search?q=spy&vod=true").text) == [("1976", "IMDb 7.0")]
    assert main._get_movie_player_info("5", "mkv").channel_name == "Spy (1976)"
    assert year_and_rating(api.get("/series/6").text) == ["2007", "IMDb 8.5"]
    assert main._get_series_player_info("1", 6, "")[0].channel_name == "Show (2007)"
    # Without IMDb's, every page shows the provider's.
    assert year_and_rating(api.get("/movie/7").text) == ["1977", "★ 6.8"]
    assert re.findall(card, api.get("/search?q=other&vod=true").text) == [("1977", "★ 6.8")]
    assert main._get_movie_player_info("7", "mkv").channel_name == "Spy (1977)"


@pytest.mark.usefixtures("imdb_index")
def test_detail_pages_and_favorites_use_the_cards_name_and_poster(
    api: TestClient,
    web: testing.FakeWeb,
) -> None:
    _set_sources(_XTREAM)
    film: ProviderRecord = {
        "stream_id": 5,
        "name": "Film - 1999",
        "source_id": "s1",
        "stream_icon": "http://img/film-small.jpg",
    }
    show: ProviderRecord = {
        "series_id": 6,
        "name": "Show (2007)",
        "source_id": "s1",
        "cover": "http://img/show-small.jpg",
    }
    # Catalog records without a poster, whose provider details have one.
    bare_film: ProviderRecord = {"stream_id": 7, "name": "Bare - 1999", "source_id": "s1"}
    bare_show: ProviderRecord = {"series_id": 8, "name": "Plain Series", "source_id": "s1"}
    cache.get_cache().update(
        live_streams=[],
        vod_streams=[film, bare_film],
        series=[show, bare_show],
    )
    for stream_id in ("5", "7"):
        details = {"info": {"name": "Film", "cover_big": f"http://img/big-{stream_id}.jpg"}}
        web.routes[_api_url("http://x", "get_vod_info", vod_id=stream_id)] = testing.FakeResponse(
            json.dumps(details).encode(),
        )
    for series_id in ("6", "8"):
        series_details: ProviderRecord = {
            "info": {"name": "Provider Show", "cover": f"http://img/big-{series_id}.jpg"},
            "episodes": {},
        }
        web.routes[_api_url("http://x", "get_series_info", series_id=series_id)] = (
            testing.FakeResponse(json.dumps(series_details).encode())
        )

    def card(kind: str, query: str) -> tuple[str, str]:
        page = api.get(f"/search?q={query}&{kind}=true").text
        found = re.search(rf'data-{kind}-name="([^"]*)"\s+data-{kind}-cover="([^"]*)"', page)
        assert found is not None
        return str(found.group(1)), str(found.group(2))

    def detail(path: str) -> tuple[str, str]:
        page = api.get(path).text
        name = re.search(r'<h2 class="[^"]*">([^<]*)</h2>', page)
        cover = re.search(r'<img id="cover" src="([^"]*)"', page)
        assert name is not None and cover is not None
        return str(name.group(1)), str(cover.group(1))

    # The detail page, and so a favorite starred there, shows what the card shows.
    film_card, show_card = card("movie", "film"), card("series", "show")
    # Names lose their year suffix; pages show the year on its own.
    assert detail("/movie/5") == film_card and film_card[0] == "Film"
    assert "film-small.jpg" in film_card[1]
    assert detail("/series/6") == show_card and show_card[0] == "Show"
    assert "show-small.jpg" in show_card[1]
    # The provider's poster fills in one the catalog lacks.
    bare_name, bare_cover = detail("/movie/7")
    assert bare_name == "Bare" and "big-7.jpg" in bare_cover
    bare_name, bare_cover = detail("/series/8")
    assert bare_name == "Plain Series" and "big-8.jpg" in bare_cover


@pytest.mark.usefixtures("imdb_index")
def test_search_and_detail_pages_scroll_whole(
    api: TestClient,
    web: testing.FakeWeb,
) -> None:
    _set_sources(_XTREAM)
    cache.get_cache().update(
        vod_streams=[{"stream_id": 7, "name": "Film", "source_id": "s1"}],
        series=[{"series_id": 5, "name": "Show", "source_id": "s1"}],
    )
    web.routes[_api_url("http://x", "get_vod_info", vod_id="7")] = TimeoutError()
    web.routes[_api_url("http://x", "get_series_info", series_id="5")] = testing.FakeResponse(
        b'{"info": {"name": "Show"}, "episodes": {}}',
    )
    for path in ("/search", "/movie/7", "/series/5"):
        page = api.get(path).text
        outer = re.search(r'<main[^>]*>\s*<div class="([^"]*)"', page)
        assert outer is not None, path
        # The whole page scrolls at every size; held to the window height, its list
        # scrolled in a box of its own, on phones a thin strip.
        assert not [c for c in outer.group(1).split() if c.endswith("h-full")], path
        assert re.findall(r'class="[^"]*overflow-auto[^"]*"', page) == [], path


def _mixed_catalog() -> None:
    """A channel, two movies, and a series that "heat" finds, in named categories."""
    cache.get_cache().update(
        live_categories=[{"category_id": "7", "category_name": "Action"}],
        live_streams=[{"stream_id": 31, "name": "Heat TV", "category_ids": [7]}],
        vod_categories=[
            {"category_id": "1", "category_name": "Action"},
            {"category_id": "2", "category_name": "Drama"},
        ],
        vod_streams=[
            {"stream_id": 1, "name": "Heat - 1995", "category_id": "1", "added": "5"},
            {"stream_id": 2, "name": "Heat Wave - 2001", "category_id": "2", "added": "9"},
        ],
        series_categories=[{"category_id": "3", "category_name": "Drama"}],
        series=[
            {"series_id": 4, "name": "Heatwave (2019)", "category_id": "3", "last_modified": "7"},
        ],
    )
    imdb._write(
        {
            "movies:heat|1995": _imdb_facts(8.3, 1, 1995),
            "movies:heatwave|2001": _imdb_facts(5.0, 1, 2001),
            "series:heatwave|2019": _imdb_facts(7.0, 1, 2019),
        },
        set(),
    )


def _cards_shown(page: str) -> list[tuple[str, str, str]]:
    """Each card's kind, id, and the kind label it shows, in page order."""
    found = re.finditer(
        r'<div class="(movie|series|live)-card[^>]*data-\1-id="(\d+)".*?data-kind-label>([^<]*)<',
        page,
        re.DOTALL,
    )
    return [(str(m[1]), str(m[2]), str(m[3])) for m in found]


@pytest.mark.usefixtures("imdb_index")
def test_search_lists_every_kind_in_one_grid(api: TestClient) -> None:
    _mixed_catalog()
    page = api.get("/search?q=heat&sort=alpha&min_imdb=").text
    # One grid, one sort, one limit: each card says what it is.
    assert page.count('<div id="results"') == 1 and 'id="vod-results"' not in page
    assert _cards_shown(page) == [
        ("movie", "1", "Movie"),
        ("live", "31", "Live"),
        ("movie", "2", "Movie"),
        ("series", "4", "Series"),
    ]
    assert 'href="/play/live/31"' in page
    # A channel has no IMDb rating or year, so those sorts put it last.
    by_imdb = _cards_shown(api.get("/search?q=heat&sort=imdb").text)
    assert [(k, i) for k, i, _ in by_imdb] == [
        ("movie", "1"),
        ("series", "4"),
        ("movie", "2"),
        ("live", "31"),
    ]
    by_newest = _cards_shown(api.get("/search?q=heat&sort=newest").text)
    assert [i for _, i, _ in by_newest] == ["2", "4", "1", "31"]
    assert len(_cards_shown(api.get("/search?q=heat&sort=alpha&limit=2").text)) == 2


@pytest.mark.usefixtures("imdb_index")
def test_search_filters_by_category_across_kinds_and_remembers_it(api: TestClient) -> None:
    _mixed_catalog()
    page = api.get("/search?q=heat&min_imdb=").text
    # One list of the categories results fall in, same names merged.
    select = re.search(r'<select name="category".*?</select>', page, re.DOTALL)
    assert select is not None
    options = re.findall(r'<option value="([^"]*)"[^>]*>[^<]*</option>', select.group(0))
    assert options == ["", "Action", "Drama"]
    drama = api.get("/search?q=heat&category=Drama&sort=alpha").text
    assert [(k, i) for k, i, _ in _cards_shown(drama)] == [("movie", "2"), ("series", "4")]
    # Remembered, like the year and IMDb filters, until cleared.
    assert len(_cards_shown(api.get("/search?q=heat").text)) == 2
    assert 'value="Drama" selected' in api.get("/search?q=heat").text
    assert len(_cards_shown(api.get("/search?q=heat&category=").text)) == 4
    # Only the kinds searched offer their categories.
    live_only = api.get("/search?q=heat&live=true&category=").text
    select = re.search(r'<select name="category".*?</select>', live_only, re.DOTALL)
    assert select is not None and "Drama" not in select.group(0)


@pytest.mark.usefixtures("imdb_index")
def test_search_says_how_many_its_filters_hide(api: TestClient) -> None:
    _mixed_catalog()
    page = api.get("/search?q=heat&min_imdb=7.5").text
    # Year and IMDb bounds judge movies and series; a channel always passes.
    assert [(k, i) for k, i, _ in _cards_shown(page)] == [("movie", "1"), ("live", "31")]
    note = re.search(r'id="hidden-by-filters"[^>]*>(.*?)</p>', page, re.DOTALL)
    assert note is not None and "2 more hidden by your filters" in note.group(1)
    link = re.search(r'id="hidden-by-filters".*?href="([^"]*)"', page, re.DOTALL)
    assert link is not None
    assert all(
        f"{f}=&" in link.group(1) or link.group(1).endswith(f"{f}=")
        for f in ("year_from", "year_to", "min_imdb", "category")
    )
    assert 'id="hidden-by-filters"' not in api.get("/search?q=heat&min_imdb=").text


@pytest.mark.usefixtures("imdb_index")
def test_keyboard_focus_is_plainly_visible(api: TestClient) -> None:
    page = api.get("/search").text
    style = re.search(r"<style>(.*?)</style>", page, re.DOTALL)
    assert style is not None
    css = re.sub(r"\s+", " ", style.group(1))
    # A thin blue ring vanished on dark posters; one bold yellow outline marks the
    # focused control everywhere, and inside cards, where nothing can cover or clip it.
    assert ":focus-visible { outline: 3px solid #facc15; outline-offset: 2px; }" in css
    assert '[data-nav="grid"]:focus-visible { outline-width: 4px; outline-offset: -4px; }' in css
    assert "outline: none" not in css


@pytest.mark.usefixtures("imdb_index")
def test_detail_pages_name_the_composer(api: TestClient, web: testing.FakeWeb) -> None:
    _set_sources(_XTREAM)
    imdb._write(
        {
            "movies:tmdb:5": {
                **_imdb_facts(7.0, 1, 1995),
                "writers": ["Michael Mann"],
                "crew": {
                    "composer": ["Elliot Goldenthal"],
                    "country": ["United States"],
                    "language": ["English", "Spanish"],
                },
            },
        },
        set(),
    )
    cache.get_cache().update(
        vod_streams=[
            {"stream_id": 5, "name": "Heat", "tmdb": 5, "source_id": "s1"},
            {"stream_id": 6, "name": "Ronin", "tmdb": 6, "source_id": "s1"},
        ],
    )
    for stream_id in ("5", "6"):
        web.routes[_api_url("http://x", "get_vod_info", vod_id=stream_id)] = TimeoutError()
    # Linked like every credit, to a search for its own field.
    page = api.get("/movie/5").text
    assert '<span class="text-gray-500">Composer:</span>' in page
    assert 'href="/search?q=composer%3A%22Elliot%20Goldenthal%22"' in page
    assert '<span class="text-gray-500">Writer:</span>' in page
    assert 'href="/search?q=writer%3A%22Michael%20Mann%22"' in page
    assert '<span class="text-gray-500">Country:</span>' in page
    assert 'href="/search?q=country%3A%22United%20States%22"' in page
    assert '<span class="text-gray-500">Language:</span>' in page
    assert 'href="/search?q=language%3A%22Spanish%22"' in page
    # Each name is a finger's height to tap on a phone, not a 19px line of text.
    link = re.search(r'<a href="/search\?q=writer%3A[^>]*>', page)
    assert link is not None and "inline-block py-1" in link.group(0)
    other = api.get("/movie/6").text
    assert "Composer:" not in other and "Country:" not in other and "Language:" not in other


@pytest.mark.usefixtures("imdb_index")
def test_series_pages_name_the_network(api: TestClient, web: testing.FakeWeb) -> None:
    _set_sources(_XTREAM)
    imdb._write(
        {
            "series:tmdb:5": {
                **_imdb_facts(9.2, 1, 2011),
                "crew": {"network": ["HBO"], "country": ["United States"], "language": ["English"]},
            },
        },
        set(),
    )
    cache.get_cache().update(
        series=[
            {"series_id": 5, "name": "Game of Thrones", "tmdb": 5, "source_id": "s1"},
            {"series_id": 6, "name": "Unaired", "tmdb": 6, "source_id": "s1"},
        ],
    )
    episodes = b'{"1": [{"id": "11", "episode_num": 1, "title": "Pilot"}]}'
    for series_id in ("5", "6"):
        web.routes[_api_url("http://x", "get_series_info", series_id=series_id)] = (
            testing.FakeResponse(b'{"info": {"name": "Show"}, "episodes": ' + episodes + b"}")
        )
    page = api.get("/series/5").text
    assert '<span class="text-gray-500">Network:</span>' in page
    # The first episode takes focus without the browser's autofocus, which scrolled
    # the page past the title to show it.
    first = re.search(r'<a href="/play/series/11[^>]*>', page)
    assert first is not None
    assert "data-initial-focus" in first.group(0) and "autofocus" not in first.group(0)
    assert 'href="/search?q=network%3A%22HBO%22"' in page
    assert 'href="/search?q=country%3A%22United%20States%22"' in page
    assert 'href="/search?q=language%3A%22English%22"' in page
    other = api.get("/series/6").text
    assert "Network:" not in other and "Country:" not in other and "Language:" not in other


@pytest.mark.usefixtures("imdb_index")
def test_search_page_explains_its_fields(api: TestClient) -> None:
    page = api.get("/search").text
    box = re.search(r'<input type="text" name="q"[^>]*title="([^"]*)"', page)
    overlay = re.search(r'<div id="search-help".*?</div>\s*</div>', page, re.DOTALL)
    assert box is not None and overlay is not None
    fields = (
        "title",
        "cast",
        "role",
        "director",
        "writer",
        "composer",
        "network",
        "country",
        "language",
        "year",
        "genre",
        "imdb",
    )
    # Hovering the box names every field; the overlay shows how to use each.
    assert all(f"{field}:" in box.group(1) for field in fields)
    assert all(f"<code>{field}:" in overlay.group(0) for field in fields)
    # A button opens it, and so does "?" typed outside a text field.
    assert 'aria-controls="search-help"' in page
    assert "event.key === '?'" in page


@pytest.mark.usefixtures("imdb_index")
def test_every_page_draws_a_title_with_one_card(api: TestClient) -> None:
    heat: ProviderRecord = {
        "stream_id": 1,
        "name": "Heat - 1995",
        "rating": "8",
        "stream_icon": "http://img/heat.jpg",
        "category_id": "1",
        "container_extension": "mp4",
    }
    show: ProviderRecord = {
        "series_id": 2,
        "name": "Hshow (2001)",
        "cover": "http://img/show.jpg",
        "category_id": "1",
        "plot": "Plot.",
    }
    cats: list[ProviderRecord] = [{"category_id": "1", "category_name": "One"}]
    cache.get_cache().update(
        live_streams=[],
        vod_categories=cats,
        vod_streams=[heat],
        series_categories=cats,
        series=[show],
    )
    settings = cache.load_user_settings("testuser")
    settings["favorites"] = {
        "movies": {
            "1": {"name": "Heat", "cover": "", "ext": "mp4"},
            # Gone from the catalog: its saved name, and no cover, since saved
            # covers are never signed.
            "9": {"name": "Gone", "cover": "http://img/gone.jpg", "ext": "mkv"},
        },
        "series": {"2": {"name": "Hshow", "cover": ""}},
    }
    cache.save_user_settings("testuser", settings)
    server = cache.load_server_settings()
    server["vod_order"] = ["9", "1"]
    cache.save_server_settings(server)

    def card(page: str, noun: str, item_id: int) -> str:
        found = re.search(
            rf'<div class="{noun}-card [^>]*data-{noun}-id="{item_id}".*?</button>\s*</div>',
            page,
            re.DOTALL,
        )
        assert found is not None, (noun, item_id)
        return found.group(0)

    search = api.get("/search?q=h").text
    # Search's mix of kinds labels each card with its kind; that label is the only
    # difference from the same card elsewhere.
    unlabeled = re.sub(r'<span class="text-gray-500" data-kind-label>[^<]*</span>', "", search)
    assert unlabeled != search
    movie, series = card(unlabeled, "movie", 1), card(unlabeled, "series", 2)
    # Favorites toggle from the card's own data, so no card repeats it in script;
    # every card can be dragged to a new place.
    for drawn in (movie, series):
        assert 'onclick="toggleFavorite(this)"' in drawn and 'draggable="true"' in drawn
    movie_pages = (
        "/vod?sort=imdb",
        "/vod?category=1&sort=imdb",
        "/api/vod/cards?category=1&sort=imdb",
        "/api/vod/favorites",
    )
    for path in movie_pages:
        assert card(api.get(path).text, "movie", 1) == movie, path
    series_pages = (
        "/series?sort=imdb",
        "/series?category=1&sort=imdb",
        "/api/series/cards?category=1&sort=imdb",
        "/api/series/favorites",
    )
    for path in series_pages:
        assert card(api.get(path).text, "series", 2) == series, path
    # Favorites keep the saved order.
    favorites = api.get("/api/vod/favorites").text
    assert re.findall(r'data-movie-id="(\d+)"', favorites) == ["9", "1"]
    assert 'data-movie-name="Gone" data-movie-cover=""' in card(favorites, "movie", 9)


@pytest.mark.usefixtures("imdb_index")
def test_favorite_cards_load_the_catalog_after_a_restart(api: TestClient) -> None:
    cache.save_file_cache("vod_data", {"cats": [], "streams": [{"stream_id": 1, "name": "Heat"}]})
    cache.save_file_cache("series_data", {"cats": [], "series": [{"series_id": 2, "name": "Vice"}]})
    settings = cache.load_user_settings("testuser")
    settings["favorites"] = {
        "movies": {"1": {"name": "Saved", "cover": "", "ext": "mkv"}},
        "series": {"2": {"name": "Saved", "cover": ""}},
    }
    cache.save_user_settings("testuser", settings)
    # The catalog's name, so the catalog was loaded, not only the saved favorite.
    assert 'data-movie-name="Heat"' in api.get("/api/vod/favorites").text
    assert 'data-series-name="Vice"' in api.get("/api/series/favorites").text


@pytest.mark.usefixtures("imdb_index")
def test_favorite_cards_respect_access(api: TestClient) -> None:
    _set_sources(_XTREAM, {**_XTREAM, "id": "s2"})
    cache.get_cache().update(
        vod_streams=[
            {"stream_id": 1, "name": "Heat", "source_id": "s1"},
            {"stream_id": 2, "name": "Ronin", "source_id": "s2"},
        ],
    )
    settings = cache.load_user_settings("testuser")
    settings["favorites"] = {
        "movies": {
            "1": {"name": "Heat", "cover": "", "ext": "mkv"},
            "2": {"name": "Ronin", "cover": "", "ext": "mkv"},
        },
        "series": {},
    }
    cache.save_user_settings("testuser", settings)
    # A favorite from a source the user can no longer use is not shown.
    auth.set_user_limits("testuser", None, ["movies:s1"])
    shown = api.get("/api/vod/favorites").text
    assert re.findall(r'data-movie-id="(\d+)"', shown) == ["2"]
    auth.set_user_limits("testuser", None, ["movies:s1", "movies:s2"])
    assert api.get("/api/vod/favorites").status_code == 403


@pytest.mark.usefixtures("imdb_index")
def test_cards_show_a_one_line_title_then_year_and_rating(api: TestClient) -> None:
    rated: ProviderRecord = {"stream_id": 1, "name": "A - 1999", "rating": "7"}
    long_name = "A " + "very long title " * 6
    unrated: ProviderRecord = {"stream_id": 2, "name": long_name}
    show: ProviderRecord = {"series_id": 3, "name": "A show (2001)", "plot": "Two go west."}
    cache.get_cache().update(live_streams=[], vod_streams=[rated, unrated], series=[show])
    page = api.get("/search?q=a&sort=alpha").text
    # Hovering shows the whole title, its year and rating, and its description.
    # Search sorts movies and series together, by name.
    assert re.findall(r'data-nav="grid" title="([^"]*)"', page) == [
        "A&#10;1999 · ★ 7.0",
        "A show&#10;2001&#10;Two go west.",
        long_name,
    ]
    cards = re.findall(
        r'<div class="p-2[^"]*">\s*<div class="([^"]*)">([^<]*)</div>'
        r'\s*<div class="([^"]*)"><span[^>]*data-kind-label>[^<]*</span>'
        r"<span[^>]*>([^<]*)</span><span[^>]*>([^<]*)</span></div>",
        page,
    )
    # The year leaves the name for the line below, beside the rating.
    assert [(c[1], c[3], c[4]) for c in cards] == [
        ("A", "1999", "★ 7.0"),
        ("A show", "2001", ""),
        (long_name, "", ""),
    ]
    # A long title is cut to one line, and the line below keeps its height when
    # empty, so every card is one size.
    assert len({(c[0], c[2]) for c in cards}) == 1
    assert "truncate" in cards[0][0] and "h-4" in cards[0][2]


def test_enter_submits_every_form_with_several_fields() -> None:
    # Browsers ignore Enter in a form with several text-like fields and no submit
    # button; the search form's year and IMDb fields once left typing a search dead.
    templates = Path(main.__file__).parent / "templates"
    checked: list[str] = []
    for template in sorted(templates.glob("*.html")):
        for form in re.findall(r"<form\b.*?</form>", template.read_text(), re.DOTALL):
            fields = re.findall(
                r'<input[^>]*type="(?:text|number|password|url|email|search)"',
                form,
            )
            # settings.js saves each source field as it changes; Enter has nothing to submit.
            if len(fields) < 2 or "source-edit-form" in form:
                continue
            checked.append(template.name)
            assert re.search(r'<button[^>]*type="submit"', form), (template.name, form[:80])
    assert checked == [
        "login.html",
        "player.html",
        "search.html",
        "settings.html",
        "settings.html",
        "setup.html",
    ]


@pytest.mark.usefixtures("guide_db")
def test_search_filters_by_year_and_imdb_score_and_remembers_them(api: TestClient) -> None:
    movies, series = _ranked_catalog()
    cache.get_cache().update(live_streams=[], vod_streams=movies, series=series)

    def shown(query: str) -> tuple[list[str], list[str]]:
        page = api.get(f"/search?q=a&sort=alpha{query}").text
        return _movie_links(page), _series_links(page)

    # Bounds include their own year and score; a year filter leaves out titles
    # without a known year.
    assert shown("&year_from=2001&year_to=2005") == (["3"], ["3"])
    assert shown("&year_from=&year_to=1990") == (["1"], ["1"])
    # Filters persist: this keeps the year bound the last search set.
    assert shown("&min_imdb=8") == ([], [])
    # An emptied field clears that filter; an IMDb score leaves out the unrated.
    assert shown("&year_from=&year_to=&min_imdb=8") == (["2", "3"], ["2", "3"])
    assert cache.load_user_settings("testuser").get("search_filters") == {
        "year_from": None,
        "year_to": None,
        "min_imdb": 8.0,
    }
    page = api.get("/search?q=a").text
    assert (_movie_links(page), _series_links(page)) == (["2", "3"], ["2", "3"])
    assert re.search(r'name="min_imdb"[^>]*value="8.0"', page)
    assert re.search(r'name="year_from"[^>]*value=""', page)
    assert api.get("/search?q=a&year_from=abc").status_code == 400
    assert api.get("/search?q=a&min_imdb=nan").status_code == 400


@pytest.mark.usefixtures("imdb_index")
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


@pytest.mark.usefixtures("guide_db")
def test_series_detail_cleans_episode_metadata(api: TestClient, web: testing.FakeWeb) -> None:
    imdb.init()
    _set_sources(_XTREAM)
    cache.get_cache()["series"] = [{"series_id": 5, "source_id": "s1", "tmdb": "77"}]
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
    # Each name searches its own role.
    assert [(parse.unquote(q), name) for q, name in people] == [
        ('director:"Ann Dir"', "Ann Dir"),
        ('cast:"Cy Star"', "Cy Star"),
        ('cast:"Di Lead"', "Di Lead"),
    ]
    # Every name the page links is in the index search reads.
    assert imdb.matching_keys(lambda name: name == "Di Lead") == {"series:tmdb:77"}


def test_seasons_read_both_shapes_providers_send() -> None:
    first = {"id": "11", "season": 1, "episode_num": 1}
    second = {"id": "21", "season": 2, "episode_num": 1}
    keyed = json_loads(
        json.dumps({"episodes": {"1": [first], "2": [second, "junk"], "3": None}}),
        ProviderRecord,
    )
    # Columbo's provider sends a list of seasons; each episode names its own.
    listed = json_loads(
        json.dumps({"episodes": [[second], [first], None, ["junk"]]}),
        ProviderRecord,
    )
    unnamed = json_loads(json.dumps({"episodes": [[{"id": "31"}]]}), ProviderRecord)
    assert main._seasons(keyed) == {"1": [first], "2": [second]}
    assert main._seasons(listed) == {"2": [second], "1": [first]}
    # An episode that names no season takes its place in the list.
    assert main._seasons(unnamed) == {"1": [{"id": "31"}]}
    assert main._seasons({"episodes": "none"}) == main._seasons({}) == {}


def test_templates_carry_the_app_filters_and_globals() -> None:
    env = main._create_templates().env
    # Edited templates show without a restart.
    assert env.auto_reload is True
    assert env.get_template("_cards.html").name == "_cards.html"
    assert env.filters["logo_url"] is main._logo_url_filter
    assert env.filters["rating"] is main._rating_filter
    assert env.filters["youtube_id"] is main._youtube_id
    assert env.globals["with_shown"] is main._shown
    assert env.globals["get_content_access"] is main._get_content_access
    assert env.globals["get_content_access_from_request"] is main._get_content_access_from_request


@pytest.mark.parametrize(
    ("trailer", "video"),
    [
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=3", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        # Anything else could add its own embed options, such as autoplay.
        ("dQw4w9WgXcQ?autoplay=1", None),
        ("", None),
    ],
)
@pytest.mark.usefixtures("imdb_index")
def test_detail_pages_embed_the_trailer_paused(
    api: TestClient,
    web: testing.FakeWeb,
    trailer: str,
    video: str | None,
) -> None:
    _set_sources(_XTREAM)
    cache.get_cache().update(
        vod_streams=[{"stream_id": 7, "name": "Film", "source_id": "s1"}],
        series=[{"series_id": 5, "name": "Show", "source_id": "s1"}],
    )
    web.routes[_api_url("http://x", "get_vod_info", vod_id="7")] = testing.FakeResponse(
        json.dumps({"info": {"name": "Film", "youtube_trailer": trailer}}).encode(),
    )
    series: ProviderRecord = {"info": {"name": "Show", "youtube_trailer": trailer}, "episodes": {}}
    web.routes[_api_url("http://x", "get_series_info", series_id="5")] = testing.FakeResponse(
        json.dumps(series).encode(),
    )
    for path in ("/movie/7", "/series/5"):
        page = api.get(path).text
        frames = re.findall(r"<iframe[^>]*>", page)
        if video is None:
            assert frames == [], path
            continue
        [frame] = frames
        # No autoplay: it waits, paused, for a click; and it loads only when near.
        assert f'src="https://www.youtube-nocookie.com/embed/{video}"' in frame, path
        assert 'loading="lazy"' in frame and "autoplay" not in frame.split("allow=")[0], path
        assert "youtube.com/watch" not in page, path
        # One place at every width: its own row under the poster and details. Inside the
        # details it crowded Play; as a column it moved as the window widened.
        holder = re.search(r'<div class="([^"]*)">\s*<iframe', page)
        assert holder is not None, path
        assert {"w-full", "max-w-[40rem]", "mb-6"} <= set(holder.group(1).split()), path
        assert not [
            c for c in holder.group(1).split() if c.startswith(("sm:", "md:", "lg:", "xl:"))
        ]
        assert page.index("<!-- end header -->") < page.index("<iframe"), path
    # The Favorite button keeps one label, so toggling it never resizes the row.
    for path in ("/movie/7", "/series/5"):
        page = api.get(path).text
        assert re.search(r'id="fav-btn"[^>]*>\s*☆ Favorite\s*<', page), path
        assert "isFav ? '★ Favorite' : '☆ Favorite'" in page, path
    # Title, then its year, rating and length, then Play and Favorite, then the rest:
    # the buttons no longer split a title from its facts.
    for path, first in (("/movie/7", 'href="/play/movie/7'), ("/series/5", 'href="?refresh=1"')):
        page = api.get(path).text
        facts = page.index('<div class="flex gap-4 text-gray-400')
        assert page.index("</h2>") < facts < page.index(first) < page.index('id="fav-btn"'), path
        assert page.index('id="fav-btn"') < page.index("<!-- end header -->"), path


@pytest.mark.usefixtures("imdb_index")
@pytest.mark.parametrize(
    ("run_time", "durations", "shown"),
    [
        # Locke & Key's provider says "0" while its episodes run 57 and 50 minutes.
        ("0", [3420, 3015, None], "54 min/ep"),
        ("45", [], "45 min/ep"),
        # Any length counts, however short; only a missing or zero one is skipped.
        ("0", [1, 119], "1 min/ep"),
        ("0", [0], None),
        (None, [], None),
    ],
)
def test_series_detail_shows_the_average_episode_length(
    api: TestClient,
    web: testing.FakeWeb,
    run_time: str | None,
    durations: list[int | None],
    shown: str | None,
) -> None:
    _set_sources(_XTREAM)
    cache.get_cache()["series"] = [{"series_id": 5, "name": "Show", "source_id": "s1"}]
    episodes = [
        {"id": str(i), "episode_num": i, "info": {} if d is None else {"duration_secs": d}}
        for i, d in enumerate(durations, 1)
    ]
    series = {"info": {"name": "Show", "episode_run_time": run_time}, "episodes": {"1": episodes}}
    web.routes[_api_url("http://x", "get_series_info", series_id="5")] = testing.FakeResponse(
        json.dumps(series).encode(),
    )
    lengths = re.findall(r"<span>([^<]*min/ep)</span>", api.get("/series/5").text)
    assert lengths == ([] if shown is None else [shown])


@pytest.mark.usefixtures("imdb_index")
def test_series_with_listed_seasons_plays_and_shows(api: TestClient, web: testing.FakeWeb) -> None:
    _set_sources(_XTREAM)
    cache.get_cache()["series"] = [{"series_id": 5, "name": "Columbo", "source_id": "s1"}]
    series = json_loads(
        json.dumps(
            {
                "info": {"name": "Columbo"},
                "episodes": [
                    [{"id": "11", "season": 1, "episode_num": 1, "title": "Murder by the Book"}],
                    [{"id": "21", "season": 2, "episode_num": 1, "title": "Etude in Black"}],
                ],
            },
        ),
        ProviderRecord,
    )
    web.routes[_api_url("http://x", "get_series_info", series_id="5")] = testing.FakeResponse(
        json.dumps(series).encode(),
    )
    page = api.get("/series/5")
    assert page.status_code == 200
    assert re.findall(r"Season (\d+)", page.text) == ["1", "2"]
    assert ">Murder by the Book<" in page.text and ">Etude in Black<" in page.text
    _, next_url = main._get_series_player_info("11", 5, "")
    assert next_url == "/play/series/21?series_id=5&ext=mkv"
    assert main._episode_labels(series) == {
        11: "S01E01 Murder by the Book",
        21: "S02E01 Etude in Black",
    }


def test_movie_detail_loads_vod_and_restricts(api: TestClient) -> None:
    movie = {"stream_id": 7, "name": "Film", "source_id": "s1"}
    cache.save_file_cache("vod_data", {"cats": [], "streams": [movie]})
    auth.set_user_limits("testuser", None, ["movies:s1"])
    assert api.get("/movie/7").status_code == 403
    assert cache.get_cache().get("vod_streams") == [movie]


@pytest.mark.usefixtures("guide_db")
def test_movie_detail_tolerates_provider_error(api: TestClient, web: testing.FakeWeb) -> None:
    imdb.init()
    _set_sources({**_XTREAM, "id": "other"})
    url = _api_url("http://x", "get_vod_info", vod_id="7")
    web.routes[url] = TimeoutError()
    cache.get_cache()["vod_streams"] = [{"stream_id": 7, "name": "Film", "source_id": "s1"}]
    resp = api.get("/movie/7")
    assert resp.status_code == 200
    assert "Film" in resp.text
    assert web.urls == [url]


@pytest.mark.usefixtures("guide_db")
def test_movie_detail_merges_provider_info(api: TestClient, web: testing.FakeWeb) -> None:
    imdb.init()
    _set_sources(_XTREAM)
    movie: ProviderRecord = {"stream_id": 7, "name": "Film", "source_id": "s1", "tmdb": 55}
    cache.get_cache()["vod_streams"] = [movie]
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
    for text in ("Great plot", "Drama", "2001", "01:30"):
        assert text in resp.text
    # Each credited name searches its own role; empty list entries are dropped.
    people = re.findall(r'<a href="/search\?q=([^"]*)"[^>]*>([^<]*)</a>', resp.text)
    assert [(parse.unquote(q), name) for q, name in people] == [
        ('director:"Dir"', "Dir"),
        ('cast:"Actor A"', "Actor A"),
        ('cast:"Bé & Co"', "Bé &amp; Co"),
    ]
    assert imdb.matching_keys(lambda name: name == "Bé & Co") == {"movies:tmdb:55"}
    # Following a name's link finds the movie it was clicked on.
    for q, _ in people:
        assert _movie_links(api.get(f"/search?q={q}&vod=true").text) == ["7"]


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


@pytest.mark.usefixtures("imdb_index")
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


@pytest.mark.usefixtures("guide_db")
def test_search_lists_each_movie_once(api: TestClient) -> None:
    imdb.init()
    cache.get_cache().update(live_streams=[], vod_streams=_heat_versions(), series=[])
    # Search is not by category, so a movie listed in two shows once, as its oldest.
    assert _movie_links(api.get("/search?q=heat&vod=true").text) == ["1"]


@pytest.mark.usefixtures("guide_db")
def test_movie_detail_offers_each_version(api: TestClient, web: testing.FakeWeb) -> None:
    imdb.init()
    _set_sources(_XTREAM)
    june = int(datetime(2026, 6, 10, 12, tzinfo=timezone.utc).timestamp())
    may = int(datetime(2026, 5, 4, 12, tzinfo=timezone.utc).timestamp())
    common: ProviderRecord = {"name": "Heat - 1995", "tmdb": 949, "source_id": "s1"}
    cache.get_cache()["vod_streams"] = [
        {**common, "stream_id": 1, "added": str(may), "container_extension": "mkv"},
        {**common, "stream_id": 2, "added": str(june), "container_extension": "mp4"},
        {**common, "stream_id": 3, "added": str(june + 1), "source_id": "hidden"},
        {"stream_id": 4, "name": "Ronin - 1998", "tmdb": 8195, "source_id": "s1"},
    ]
    auth.set_user_limits("testuser", None, ["movies:hidden"])
    info = {"info": {"bitrate": 10_750, "duration": "01:38:47"}}
    web.routes[_api_url("http://x", "get_vod_info", vod_id="1")] = testing.FakeResponse(
        json.dumps(info).encode(),
    )
    web.routes[_api_url("http://x", "get_vod_info", vod_id="2")] = TimeoutError()

    page = api.get("/movie/1").text

    # Oldest first, as cards open it; the one on show is marked; a version the user may
    # not see is left out; one whose details fail to load shows what the catalog says.
    versions = re.findall(r'data-version="(\d+)"( aria-current="true")?>\s*([^<]*?)\s*<', page)
    assert versions == [
        ("1", ' aria-current="true"', "Added May 4, 2026 · MKV · 10.8 Mbps · 01:38:47"),
        ("2", "", "Added Jun 10, 2026 · MP4"),
    ]
    assert "/play/movie/1?ext=mkv" in page
    assert sorted(web.urls) == sorted(
        _api_url("http://x", "get_vod_info", vod_id=v) for v in ("1", "2")
    )
    # A movie with one version offers no choice.
    assert "data-version" not in api.get("/movie/4").text


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
