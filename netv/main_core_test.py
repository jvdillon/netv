"""Tests for main.py startup, page, transcode and source-settings routes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, tzinfo
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, override
from unittest.mock import AsyncMock, MagicMock, patch
from urllib import error

import asyncio
import contextlib
import logging

from fastapi import HTTPException
from fastapi.testclient import TestClient

import pytest

from netv import (
    cache as cache_module,
    m3u as m3u_module,
    main,
    main_test,
)
from netv.cache import MemoryCache, Source, SourceConfig
from netv.handoff import Releaser, Resolver
from netv.main import PlayerInfo


# Shared fixtures from main_test: ``api`` is its logged-in ``api``.
mock_deps = main_test.mock_deps
api = main_test.auth_client

Target = Callable[[], None]


def _thread_recorder() -> tuple[type, list[Target]]:
    """Return a Thread stand-in that records targets instead of running them."""
    targets: list[Target] = []

    class FakeThread:
        def __init__(self, target: Target, daemon: bool = False) -> None:
            del daemon
            self.target = target

        def start(self) -> None:
            targets.append(self.target)

    return FakeThread, targets


class _ScriptedEvent:
    """Event whose ``wait`` reports a timeout ``falses`` times, then stop."""

    def __init__(self) -> None:
        self.falses = 0
        self.flag = False

    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        if self.falses:
            self.falses -= 1
            return False
        return True

    def set(self) -> None:
        self.flag = True


class _Noon(datetime):
    @override
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> _Noon:
        del tz
        return cls(2026, 1, 1, 12, 0)


def _json(resp: Any) -> dict[str, Any]:
    return resp.json()


@pytest.fixture
def fresh_cache() -> Iterator[MemoryCache]:
    cache_module.clear_all_caches()
    cache_module.get_cache().pop("epg", None)
    yield cache_module.get_cache()
    cache_module.clear_all_caches()
    cache_module.get_cache().pop("epg", None)


def _settings_with_sources(*sources: SourceConfig) -> None:
    settings = cache_module.load_server_settings()
    settings["sources"] = list(sources)
    cache_module.save_server_settings(settings)


# =============================================================================
# Startup and shutdown
# =============================================================================


def test_lifespan_runs_workers_and_shuts_down(fresh_cache: MemoryCache) -> None:
    fake_thread, targets = _thread_recorder()
    events: list[_ScriptedEvent] = []

    class RecordedEvent(_ScriptedEvent):
        def __init__(self) -> None:
            super().__init__()
            events.append(self)

    sources = [
        Source(id="x", name="X", type="xtream", url="http://x", epg_schedule=["12:00"]),
        Source(id="m", name="M", type="m3u", url="http://m", epg_schedule=["12:00"]),
        Source(id="e", name="E", type="epg", url="http://e/epg", epg_schedule=["12:00"]),
        Source(id="o", name="O", type="other", url="http://o", epg_schedule=["12:00"]),
        Source(id="bad", name="Bad", type="m3u", url="http://bad", epg_schedule=["12:00"]),
        Source(id="late", name="Late", type="epg", url="http://late", epg_schedule=["13:00"]),
    ]

    def fake_fetch_m3u(url: str, source_id: str) -> tuple[None, None, str]:
        del source_id
        if url == "http://bad":
            raise OSError("unreachable")
        return None, None, "http://m/epg"

    xtream = MagicMock(epg_url="http://x/epg")

    def sweep_then_fail(resolve: Resolver, release: Releaser) -> None:
        del resolve
        release("viewer", "http://x/1.m3u8")
        raise RuntimeError("sweep failed")

    sweep = MagicMock(side_effect=sweep_then_fail)
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch("netv.main.threading", SimpleNamespace(Thread=fake_thread, Event=RecordedEvent)),
        )
        stack.enter_context(patch.object(main, "_shutdown_event", None))
        stack.enter_context(patch("netv.main.is_sr_available", return_value=True))
        stack.enter_context(patch("netv.main.handoff.sweep", sweep))
        stack.enter_context(patch("netv.main.load_vod_data", return_value=(["vc"], ["vs"])))
        stack.enter_context(patch("netv.main.load_series_data", return_value=(["sc"], ["ss"])))
        stack.enter_context(patch("netv.main.get_sources", return_value=sources))
        stack.enter_context(patch("netv.main.XtreamClient", return_value=xtream))
        stack.enter_context(patch("netv.main.fetch_m3u", side_effect=fake_fetch_m3u))
        for name in (
            "netv.main.epg.init",
            "netv.main.ffmpeg_session.cleanup_and_recover_sessions",
            "netv.main.handoff.load",
            "netv.main.handoff.cleanup_orphans",
        ):
            stack.enter_context(patch(name))
        initialize = stack.enter_context(patch("netv.main.cache.initialize"))
        command_init = stack.enter_context(patch("netv.main.ffmpeg_command.init"))
        kill_orphans = stack.enter_context(patch("netv.main._kill_orphaned_ffmpeg"))
        cleanup_expired = stack.enter_context(
            patch("netv.main.ffmpeg_session.cleanup_expired_sessions", new_callable=AsyncMock),
        )
        session_shutdown = stack.enter_context(
            patch("netv.main.ffmpeg_session.shutdown", new_callable=AsyncMock),
        )
        release = stack.enter_context(
            patch("netv.main.ffmpeg_session.stop_user_stream", new_callable=AsyncMock),
        )
        handoff_shutdown = stack.enter_context(patch("netv.main.handoff.shutdown"))
        cast_close = stack.enter_context(patch("netv.main.chromecast.manager.close"))
        preload_live = stack.enter_context(patch("netv.main._preload_live"))
        prune = stack.enter_context(patch("netv.main._prune_epg"))
        preload_epg = stack.enter_context(patch("netv.main._preload_epg"))
        fetch_all = stack.enter_context(patch("netv.main._fetch_all_epg"))

        async def run() -> None:
            async with main.lifespan(main.app):
                load_all, load_vod, load_series, cleanup, handoff_loop, scheduler = targets
                load_all()
                load_vod()
                load_series()
                cleanup_stop, scheduler_stop = events
                # Worker loops run in threads and hand session work to the loop.
                cleanup_stop.falses = 1
                await asyncio.to_thread(cleanup)
                cleanup_stop.falses = 1
                await asyncio.to_thread(handoff_loop)
                scheduler_stop.falses = 2
                with patch("netv.main.datetime", _Noon):
                    scheduler()
                for refresh in targets[6:]:
                    refresh()
                assert main._shutdown_event is None
            shutdown_event = main._shutdown_event
            assert shutdown_event is not None
            assert shutdown_event.is_set()
            assert cleanup_stop.flag
            assert scheduler_stop.flag

        asyncio.run(run())

        initialize.assert_called_once()
        assert command_init.call_args.kwargs["sr_engine_dir"] == str(main.sr_engine_dir())
        kill_orphans.assert_called_once()
        preload_live.assert_called_once()
        prune.assert_called_once()
        preload_epg.assert_called_once()
        assert fresh_cache.get("vod_categories") == ["vc"]
        assert fresh_cache.get("vod_streams") == ["vs"]
        assert fresh_cache.get("series_categories") == ["sc"]
        assert fresh_cache.get("series") == ["ss"]
        cleanup_expired.assert_called_once()
        sweep.assert_called_once()
        release.assert_awaited_once_with("viewer", "http://x/1.m3u8")
        # Five due sources, triggered once each even though the loop ran twice.
        assert len(targets) == 6 + 5
        assert [c.args[0] for c in fetch_all.call_args_list] == [
            [("http://x/epg", 120, "x")],
            [("http://m/epg", 120, "m")],
            [("http://e/epg", 120, "e")],
        ]
        assert not {f"{s.id}_epg" for s in sources} & m3u_module.get_refresh_in_progress()
        cast_close.assert_called_once()
        session_shutdown.assert_called_once()
        handoff_shutdown.assert_called_once()


def test_kill_orphaned_ffmpeg_tolerates_missing_pgrep() -> None:
    with (
        patch("netv.main.subprocess.run", side_effect=OSError("no pgrep")),
        patch("netv.main.os.kill") as kill,
    ):
        main._kill_orphaned_ffmpeg()
    kill.assert_not_called()


def test_kill_orphaned_ffmpeg_skips_bad_pids_and_stops_on_permission_error() -> None:
    def fake_kill(pid: int, sig: int) -> None:
        del sig
        if pid == 13:
            raise ProcessLookupError
        if pid == 14:
            raise PermissionError

    with (
        patch(
            "netv.main.subprocess.run",
            return_value=SimpleNamespace(stdout="12\n\nabc\n13\n14\n15\n"),
        ),
        patch("netv.main.os.kill", side_effect=fake_kill) as kill,
    ):
        main._kill_orphaned_ffmpeg()
    assert [c.args[0] for c in kill.call_args_list] == [12, 13, 14]


def test_preload_live_fills_cache_and_clears_flag(fresh_cache: MemoryCache) -> None:
    def load() -> tuple[list[str], list[str], list[str]]:
        assert "guide_load" in m3u_module.get_refresh_in_progress()
        return ["c"], ["s"], ["u"]

    with patch("netv.main.load_all_live_data", side_effect=load):
        main._preload_live()
    assert fresh_cache.get("live_categories") == ["c"]
    assert fresh_cache.get("live_streams") == ["s"]
    assert fresh_cache.get("epg_urls") == ["u"]
    assert "guide_load" not in m3u_module.get_refresh_in_progress()


def test_preload_epg_notifies_subscribers(fresh_cache: MemoryCache) -> None:
    fresh_cache["epg_urls"] = [("http://e", 5, "s")]
    queue: asyncio.Queue[str] = asyncio.Queue()
    broken = MagicMock()
    broken.put_nowait.side_effect = RuntimeError("closed")
    with (
        patch.object(main, "_epg_subscribers", {queue, broken}),
        patch("netv.main.load_all_epg") as load_all,
        patch("netv.main.epg.get_program_count", return_value=3),
    ):
        main._preload_epg()
    load_all.assert_called_once_with([("http://e", 5, "s")])
    assert queue.get_nowait() == "epg_ready"


def test_preload_epg_logs_errors(
    fresh_cache: MemoryCache,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fresh_cache["epg_urls"] = [("http://e", 5, "s")]
    with (
        caplog.at_level(logging.ERROR),
        patch("netv.main.load_all_epg", side_effect=RuntimeError("db locked")),
    ):
        main._preload_epg()
    assert "EPG load error: db locked" in caplog.text


def test_load_all_epg_skips_when_another_thread_filled_it() -> None:
    with (
        patch("netv.main._epg_is_fresh", side_effect=[False, True]),
        patch("netv.main._fetch_all_epg") as fetch_all,
    ):
        main.load_all_epg([("http://e", 5, "s")])
    fetch_all.assert_not_called()


def test_load_all_epg_records_fetch_error(fresh_cache: MemoryCache) -> None:
    with (
        patch("netv.main._epg_is_fresh", return_value=False),
        patch("netv.main._fetch_all_epg", side_effect=RuntimeError("boom")),
    ):
        main.load_all_epg([("http://e", 5, "s")])
    assert fresh_cache.get("epg_error") == "boom"


def test_epg_events_registers_subscriber() -> None:
    subscribers: set[asyncio.Queue[str]] = set()
    with patch.object(main, "_epg_subscribers", subscribers):
        response = asyncio.run(main.epg_events({"sub": "u"}))
    assert response.media_type == "text/event-stream"
    assert len(subscribers) == 1


def test_epg_events_rejects_when_full() -> None:
    full: set[Any] = {object() for _ in range(100)}
    with patch.object(main, "_epg_subscribers", full), pytest.raises(HTTPException) as exc:
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

_LIVE_FILE_CACHE: tuple[dict[str, list[dict[str, object]]], int] = (
    {
        "cats": [{"category_id": "1", "category_name": "News"}],
        "streams": [
            {"stream_id": 1, "name": "CachedCNN", "category_ids": ["1"], "epg_channel_id": ""},
        ],
        "epg_urls": [],
    },
    0,
)


def test_guide_page_loads_file_cache(api: TestClient) -> None:
    with (
        patch("netv.m3u.load_file_cache", return_value=_LIVE_FILE_CACHE),
        patch("netv.main.epg.has_programs", return_value=True),
    ):
        resp = api.get("/guide?cats=1")
    assert resp.status_code == 200
    assert "CachedCNN" in resp.text
    store = cache_module.get_cache()
    assert "live_streams" in store
    assert store["live_streams"][0]["name"] == "CachedCNN"


def test_guide_rows_api_loads_file_cache(api: TestClient) -> None:
    with patch("netv.m3u.load_file_cache", return_value=_LIVE_FILE_CACHE):
        resp = api.get("/api/guide/rows?cats=1")
    assert resp.status_code == 200
    assert _json(resp)["total"] == 1
    assert _json(resp)["rows"][0]["channel"]["name"] == "CachedCNN"


def _vod_streams() -> list[dict[str, Any]]:
    return [
        {"stream_id": 1, "name": "MovieAlpha", "category_id": "1", "rating": "5", "added": "1"},
        {"stream_id": 2, "name": "MovieBeta", "category_id": "1", "rating": "bad", "added": "3"},
        {"stream_id": 3, "name": "MovieGamma", "category_id": "2", "rating": "9", "added": "9"},
    ]


def test_vod_page_loads_file_cache(api: TestClient) -> None:
    data = {"cats": [{"category_id": "1", "category_name": "Films"}], "streams": _vod_streams()}
    with patch("netv.m3u.load_file_cache", return_value=(data, 0)):
        resp = api.get("/vod?sort=alpha")
    assert resp.status_code == 200
    assert "MovieAlpha" in resp.text
    assert cache_module.get_cache().get("vod_streams") == _vod_streams()


def test_vod_page_restricted(api: TestClient) -> None:
    cache_module.get_cache().update(vod_categories=[], vod_streams=[])
    with patch("netv.main._get_content_access", return_value={"movies": False, "series": True}):
        resp = api.get("/vod")
    assert resp.status_code == 403


def test_vod_page_applies_filter_and_sorts(api: TestClient) -> None:
    cache_module.get_cache().update(
        vod_categories=[
            {"category_id": "1", "category_name": "CatOne"},
            {"category_id": "2", "category_name": "CatTwo"},
        ],
        vod_streams=_vod_streams(),
    )
    settings = cache_module.load_user_settings("testuser")
    settings["vod_filter"] = ["1"]
    cache_module.save_user_settings("testuser", settings)
    by_rating = api.get("/vod?sort=rating").text
    newest = api.get("/vod?sort=newest").text
    assert "MovieGamma" not in by_rating
    assert "CatTwo" not in by_rating
    assert by_rating.index("MovieAlpha") < by_rating.index("MovieBeta")
    assert newest.index("MovieBeta") < newest.index("MovieAlpha")


def _series_list() -> list[dict[str, Any]]:
    return [
        {"series_id": 1, "name": "ShowB", "category_id": "1", "rating": "5", "last_modified": "1"},
        {"series_id": 2, "name": "ShowA", "category_id": "1", "rating": "x", "last_modified": "3"},
        {"series_id": 3, "name": "ShowC", "category_id": "2", "rating": "9", "last_modified": "9"},
    ]


def test_series_page_loads_file_cache(api: TestClient) -> None:
    data = {"cats": [{"category_id": "1", "category_name": "Drama"}], "series": _series_list()}
    with patch("netv.m3u.load_file_cache", return_value=(data, 0)):
        resp = api.get("/series?sort=alpha")
    assert resp.status_code == 200
    assert "ShowA" in resp.text
    assert cache_module.get_cache().get("series") == _series_list()


def test_series_page_restricted(api: TestClient) -> None:
    cache_module.get_cache().update(series_categories=[], series=[])
    with patch("netv.main._get_content_access", return_value={"movies": True, "series": False}):
        resp = api.get("/series")
    assert resp.status_code == 403


def test_series_page_applies_filter_category_and_sorts(api: TestClient) -> None:
    cache_module.get_cache().update(
        series_categories=[
            {"category_id": "1", "category_name": "CatOne"},
            {"category_id": "2", "category_name": "CatTwo"},
        ],
        series=_series_list(),
    )
    settings = cache_module.load_user_settings("testuser")
    settings["series_filter"] = ["1", "2"]
    cache_module.save_user_settings("testuser", settings)
    alpha = api.get("/series?sort=alpha&category=1").text
    rating = api.get("/series?sort=rating&category=1").text
    newest = api.get("/series?sort=newest&category=1").text
    assert "ShowC" not in alpha
    assert alpha.index("ShowA") < alpha.index("ShowB")
    assert rating.index("ShowB") < rating.index("ShowA")
    assert newest.index("ShowA") < newest.index("ShowB")
    settings = cache_module.load_user_settings("testuser")
    settings["series_filter"] = ["2"]
    cache_module.save_user_settings("testuser", settings)
    filtered = api.get("/series?sort=alpha").text
    assert "ShowC" in filtered
    assert "ShowA" not in filtered


def _call_fetch(key: str, fetch: Callable[[], Any], force: bool = False) -> Any:
    del key, force
    return fetch()


def test_series_detail_restricted(api: TestClient) -> None:
    cache_module.get_cache()["series"] = [{"series_id": 5, "source_id": "s1"}]
    with patch("netv.main._unavailable_groups", return_value={"series:s1"}):
        resp = api.get("/series/5")
    assert resp.status_code == 403


def test_series_detail_without_xtream_source(api: TestClient) -> None:
    cache_module.get_cache()["series"] = []
    with patch("netv.main.get_first_xtream_client", return_value=None):
        resp = api.get("/series/5")
    assert resp.status_code == 404


def test_series_detail_provider_error(api: TestClient) -> None:
    cache_module.get_cache()["series"] = [{"series_id": 5, "source_id": "s1"}]
    with (
        patch("netv.main.get_xtream_client_by_source", return_value=None),
        patch("netv.main.get_first_xtream_client", return_value=MagicMock()),
        patch("netv.main.get_cached_info", side_effect=error.URLError("down")),
    ):
        resp = api.get("/series/5")
    assert resp.status_code == 502
    assert "Failed to load series info" in resp.text


def test_series_detail_cleans_episode_metadata(api: TestClient) -> None:
    cache_module.get_cache()["series"] = [{"series_id": 5, "source_id": "s1"}]
    xtream = MagicMock()
    xtream.get_series_info.return_value = {
        "info": {"name": "Show", "releaseDate": "2019-05-01"},
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
    with (
        patch("netv.main.get_xtream_client_by_source", return_value=xtream),
        patch("netv.main.get_cached_info", side_effect=_call_fetch) as cached_info,
    ):
        resp = api.get("/series/5?refresh=true")
    assert resp.status_code == 200
    assert cached_info.call_args.args[0] == "series_info_s1_5"
    assert cached_info.call_args.args[2] is True
    xtream.get_series_info.assert_called_once_with(5)
    assert "Pilot" in resp.text
    assert "S01E01 - Show" not in resp.text
    assert ">Pilot<" in resp.text
    assert "Plot A" in resp.text
    assert "Desc C" in resp.text
    assert "2019" in resp.text


def test_movie_detail_loads_vod_and_restricts(api: TestClient) -> None:
    movie = {"stream_id": 7, "name": "Film", "source_id": "s1"}
    with (
        patch("netv.main.load_vod_data", return_value=([], [movie])),
        patch("netv.main._unavailable_groups", return_value={"movies:s1"}),
    ):
        resp = api.get("/movie/7")
    assert resp.status_code == 403
    assert cache_module.get_cache().get("vod_streams") == [movie]


def test_movie_detail_tolerates_provider_error(api: TestClient) -> None:
    cache_module.get_cache()["vod_streams"] = [{"stream_id": 7, "name": "Film", "source_id": "s1"}]
    with (
        patch("netv.main.get_xtream_client_by_source", return_value=None),
        patch("netv.main.get_first_xtream_client", return_value=MagicMock()),
        patch("netv.main.get_cached_info", side_effect=TimeoutError),
    ):
        resp = api.get("/movie/7")
    assert resp.status_code == 200
    assert "Film" in resp.text


def test_movie_detail_merges_provider_info(api: TestClient) -> None:
    cache_module.get_cache()["vod_streams"] = [{"stream_id": 7, "name": "Film", "source_id": "s1"}]
    xtream = MagicMock()
    xtream.get_vod_info.return_value = {
        "info": {
            "plot": "Great plot",
            "director": "Dir",
            "actors": "Actor A",
            "genre": "Drama",
            "rating": "7",
            "releasedate": "2001-02-03",
            "duration": "01:30",
            "movie_image": "/img.png",
            "youtube_trailer": "yt123",
        },
    }
    with (
        patch("netv.main.get_xtream_client_by_source", return_value=xtream),
        patch("netv.main.get_cached_info", side_effect=_call_fetch) as cached_info,
    ):
        resp = api.get("/movie/7")
    assert resp.status_code == 200
    assert cached_info.call_args.args[0] == "vod_info_s1_7"
    for text in ("Great plot", "Dir", "Actor A", "Drama", "2001", "01:30", "yt123"):
        assert text in resp.text


# =============================================================================
# Player page
# =============================================================================


def test_player_movie_resumes_watch_position(api: TestClient) -> None:
    info = PlayerInfo(url="http://up/movie.mkv", channel_name="Film", source_id="s1")
    with (
        patch("netv.main._get_movie_player_info", return_value=info) as get_info,
        patch("netv.main.get_watch_position", return_value={"position": 42.5}),
        patch("netv.main._unavailable_groups", return_value={"cat:other"}),
    ):
        resp = api.get("/play/movie/9?ext=mkv")
    assert resp.status_code == 200
    get_info.assert_called_once_with("9", "mkv")
    assert "42.5" in resp.text


def test_player_series_strips_year_from_name(api: TestClient) -> None:
    info = PlayerInfo(url="http://up/ep.mkv", channel_name="Great Show (2020)", source_id="s1")
    with (
        patch(
            "netv.main._get_series_player_info",
            return_value=(info, "/play/series/12"),
        ) as get_info,
        patch("netv.main._unavailable_groups", return_value={"movies:s1"}),
    ):
        resp = api.get("/play/series/11?series_id=5&ext=mkv")
    assert resp.status_code == 200
    get_info.assert_called_once_with("11", 5, "mkv")
    assert "Great Show" in resp.text
    assert "/play/series/12" in resp.text


@pytest.mark.parametrize(
    ("stream_type", "patched", "group"),
    [
        ("live", "netv.main._get_live_player_info", "cat:3"),
        ("movie", "netv.main._get_movie_player_info", "movies:s1"),
        ("series", "netv.main._get_series_player_info", "series:s1"),
    ],
)
def test_player_enforces_group_restrictions(
    api: TestClient,
    stream_type: str,
    patched: str,
    group: str,
) -> None:
    info = PlayerInfo(url="http://up/x", source_id="s1", category_ids=["3"])
    value: Any = (info, None) if stream_type == "series" else info
    with (
        patch(patched, return_value=value),
        patch("netv.main._unavailable_groups", return_value={group}),
    ):
        resp = api.get(f"/play/{stream_type}/1")
    assert resp.status_code == 403


def test_player_missing_stream(api: TestClient) -> None:
    with patch("netv.main._get_live_player_info", return_value=PlayerInfo()):
        resp = api.get("/play/live/1")
    assert resp.status_code == 404
    assert _json(resp)["detail"] == "Stream not found"


# =============================================================================
# Search and stream redirects
# =============================================================================


def test_search_loads_data_and_applies_filters(api: TestClient) -> None:
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
    settings = cache_module.load_user_settings("testuser")
    settings["guide_filter"] = ["1"]
    settings["vod_filter"] = ["1"]
    settings["series_filter"] = ["1"]
    cache_module.save_user_settings("testuser", settings)
    with (
        patch("netv.main.load_all_live_data", return_value=([], live, [])),
        patch("netv.main.load_vod_data", return_value=([], vod)),
        patch("netv.main.load_series_data", return_value=([], series)),
    ):
        resp = api.get("/search?q=news")
    assert resp.status_code == 200
    for shown in ("News One", "News Movie", "News Show"):
        assert shown in resp.text
    for hidden in ("News Two", "News Film", "News Serial"):
        assert hidden not in resp.text
    assert cache_module.get_cache().get("live_streams") == live
    assert cache_module.get_cache().get("vod_streams") == vod
    assert cache_module.get_cache().get("series") == series


def test_stream_redirect_without_xtream(api: TestClient) -> None:
    with patch("netv.main.get_first_xtream_client", return_value=None):
        resp = api.get("/stream/movie/1")
    assert resp.status_code == 404


def test_stream_redirect_hidden_source(api: TestClient) -> None:
    source = Source(id="s1", name="S", type="xtream", url="http://up")
    with (
        patch("netv.main.get_first_xtream_client", return_value=MagicMock()),
        patch("netv.main.get_sources", return_value=[source]),
        patch("netv.main._hidden_sources", return_value={"s1"}),
    ):
        resp = api.get("/stream/movie/1")
    assert resp.status_code == 403


def test_stream_redirect_to_provider(api: TestClient) -> None:
    xtream = MagicMock()
    xtream.build_stream_url.return_value = "http://up/movie/u/p/1.mkv"
    with (
        patch("netv.main.get_first_xtream_client", return_value=xtream),
        patch("netv.main.get_sources", return_value=[]),
    ):
        resp = api.get("/stream/movie/1?ext=mkv", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "http://up/movie/u/p/1.mkv"
    xtream.build_stream_url.assert_called_once_with("movie", 1, "mkv")


# =============================================================================
# Transcode routes
# =============================================================================


def test_transcode_start_passes_stream_limits(api: TestClient) -> None:
    start = AsyncMock(return_value={"session_id": "abc"})
    with (
        patch("netv.main.chromecast.manager.owns_url", return_value=False),
        patch("netv.main.auth.get_user_limits", return_value={"max_streams_per_source": {"s1": 2}}),
        patch(
            "netv.main.load_server_settings",
            return_value={"sources": [{"id": "s1", "max_streams": 3}]},
        ),
        patch("netv.main.ffmpeg_session.start_transcode", start),
    ):
        resp = api.get("/transcode/start", params={"url": "http://up/l", "source_id": "s1"})
    assert resp.status_code == 200
    assert resp.json() == {"session_id": "abc"}
    assert start.call_args.args[8:10] == (2, 3)


def test_transcode_seek_and_progress(api: TestClient) -> None:
    with (
        patch("netv.main.chromecast.manager.owns_session", return_value=False),
        patch(
            "netv.main.ffmpeg_session.seek_transcode",
            AsyncMock(return_value={"ok": True}),
        ) as seek,
        patch("netv.main.ffmpeg_session.get_session_progress", return_value={"segments": 3}),
    ):
        seek_resp = api.get("/transcode/seek/abc?time=10")
        progress = api.get("/transcode/progress/abc")
    assert seek_resp.json() == {"ok": True}
    seek.assert_awaited_once_with("abc", 10.0)
    assert progress.json() == {"segments": 3}


def test_transcode_file_rejects_dotted_name(api: TestClient) -> None:
    assert api.get("/transcode/abc/x..m3u8").status_code == 400


def test_transcode_file_missing_file(api: TestClient, tmp_path: Path) -> None:
    with patch("netv.main.ffmpeg_session.get_session", return_value={"dir": str(tmp_path)}):
        resp = api.get("/transcode/abc/seg1.ts")
    assert resp.status_code == 404
    assert _json(resp)["detail"] == "File not found"


def test_transcode_file_serves_segment_to_chromecast(
    api: TestClient,
    tmp_path: Path,
) -> None:
    (tmp_path / "seg1.ts").write_bytes(b"\x47segment")
    with (
        patch("netv.main.ffmpeg_session.get_session", return_value={"dir": str(tmp_path)}),
        patch("netv.main.ffmpeg_session.touch_session") as touch,
    ):
        resp = api.get("/transcode/abc/seg1.ts", headers={"user-agent": "CrKey/1.0"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp2t"
    assert resp.headers["access-control-allow-origin"] == "*"
    assert resp.content == b"\x47segment"
    touch.assert_called_once_with("abc")


def test_subtitle_file_rejects_dotted_name(api: TestClient) -> None:
    assert api.get("/subs/abc/x..vtt").status_code == 400


def _subtitle(tmp_path: Path, filename: str, sleep: AsyncMock | None = None) -> bytes:
    with (
        patch("netv.main.ffmpeg_session.get_session", return_value={"dir": str(tmp_path)}),
        patch("netv.main.asyncio.sleep", sleep or AsyncMock()),
    ):
        response = asyncio.run(main.subtitle_file("abc", filename))
    assert response.headers["access-control-allow-origin"] == "*"
    return bytes(response.body)


def test_subtitle_file_serves_ready_file(tmp_path: Path) -> None:
    body = "WEBVTT\n\n00:00.000 --> 00:01.000\nHello\n"
    (tmp_path / "sub0.vtt").write_text(body)
    sleep = AsyncMock()
    assert _subtitle(tmp_path, "sub0.vtt", sleep).decode() == body
    sleep.assert_not_awaited()


def test_subtitle_file_missing_returns_empty_vtt(tmp_path: Path) -> None:
    sleep = AsyncMock()
    assert _subtitle(tmp_path, "sub0.vtt", sleep) == b"WEBVTT\n\n"
    assert sleep.await_count == 15


def test_subtitle_file_undecodable_returns_empty_vtt(tmp_path: Path) -> None:
    (tmp_path / "sub0.vtt").write_bytes(b"\xff" * 30)
    assert _subtitle(tmp_path, "sub0.vtt") == b"WEBVTT\n\n"


def test_transcode_clear_refuses_cast_url(api: TestClient) -> None:
    with patch("netv.main.chromecast.manager.owns_url", return_value=True):
        resp = api.delete("/transcode-clear", params={"url": "http://up/x"})
    assert resp.status_code == 409


def test_transcode_clear_stops_cached_session(api: TestClient) -> None:
    with (
        patch("netv.main.chromecast.manager.owns_url", return_value=False),
        patch("netv.main.ffmpeg_session.clear_url_session", return_value="sid") as clear,
        patch("netv.main.ffmpeg_session.stop_session") as stop,
    ):
        resp = api.delete("/transcode-clear", params={"url": "http://up/x"})
    assert resp.json() == {"status": "cleared", "session_id": "sid"}
    clear.assert_called_once_with("http://up/x")
    stop.assert_called_once_with("sid", force=True)


# =============================================================================
# Settings
# =============================================================================


def test_settings_page_loads_file_caches(api: TestClient) -> None:
    files: dict[str, tuple[dict[str, list[dict[str, str]]], int]] = {
        "live_data": (
            {"cats": [{"category_id": "1", "category_name": "LiveCat"}], "streams": []},
            0,
        ),
        "vod_data": ({"cats": [{"category_id": "2", "category_name": "VodCat"}], "streams": []}, 0),
        "series_data": (
            {"cats": [{"category_id": "3", "category_name": "SeriesCat"}], "series": []},
            0,
        ),
    }
    with patch("netv.m3u.load_file_cache", side_effect=files.get):
        resp = api.get("/settings")
    assert resp.status_code == 200
    for name in ("LiveCat", "VodCat", "SeriesCat"):
        assert name in resp.text
    data = cache_module.get_cache()
    assert data.get("epg_urls") == []
    assert data.get("vod_streams") == []
    assert data.get("series") == []


def test_settings_page_starts_background_load_without_cache(
    api: TestClient,
) -> None:
    with (
        patch("netv.m3u.load_file_cache", return_value=None),
        patch("netv.main._start_guide_background_load") as start,
    ):
        resp = api.get("/settings")
    assert resp.status_code == 200
    start.assert_called_once()


@pytest.mark.parametrize(("path", "setting"), [("vod", "vod_filter"), ("series", "series_filter")])
def test_settings_category_filters(
    api: TestClient,
    path: str,
    setting: Literal["vod_filter", "series_filter"],
) -> None:
    resp = api.post(f"/settings/{path}-filter", json={"cats": ["1", "2"]})
    assert resp.json() == {"status": "ok"}
    assert cache_module.load_user_settings("testuser").get(setting) == ["1", "2"]


def _source_form(**overrides: Any) -> dict[str, Any]:
    return {"name": "Prov", "source_type": "xtream", "url": "http://up.example/"} | overrides


def test_add_source_rejects_blank_name(api: TestClient) -> None:
    assert api.post("/settings/add", data=_source_form(name="   ")).status_code == 400


def test_add_source_normalizes_schedule(api: TestClient) -> None:
    with patch("netv.main.clear_all_caches") as clear:
        resp = api.post(
            "/settings/add",
            data=_source_form(epg_schedule="8:00, bad, 23:30"),
            follow_redirects=False,
        )
    assert resp.status_code == 303
    clear.assert_called_once()
    source = cache_module.load_server_settings()["sources"][-1]
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
def test_edit_source_validates(api: TestClient, overrides: dict[str, Any]) -> None:
    assert api.post("/settings/edit/s1", data=_source_form(**overrides)).status_code == 400


def test_edit_source_updates_matching_source(api: TestClient) -> None:
    _settings_with_sources(
        {"id": "other", "name": "Other", "type": "m3u", "url": "http://other"},
        {"id": "s1", "name": "Old", "type": "m3u", "url": "http://old"},
    )
    with patch("netv.main.clear_all_caches") as clear:
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
    clear.assert_called_once()
    other, edited = cache_module.load_server_settings()["sources"]
    assert other["name"] == "Other"
    assert edited["name"] == "New"
    assert edited["type"] == "xtream"
    assert edited["url"] == "http://up.example"
    assert edited.get("epg_schedule") == ["07:15"]
    assert edited.get("epg_url") == "http://epg.example"
    assert edited.get("max_streams") == 0


def test_delete_source_clears_its_data(api: TestClient) -> None:
    _settings_with_sources(
        {"id": "keep", "name": "Keep", "type": "m3u", "url": "http://keep"},
        {"id": "s1", "name": "Gone", "type": "m3u", "url": "http://gone"},
    )
    with (
        patch("netv.main.epg.clear_source") as clear_source,
        patch("netv.main.clear_all_file_caches") as clear_files,
    ):
        resp = api.post("/settings/delete/s1", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings"
    assert [s["id"] for s in cache_module.load_server_settings()["sources"]] == ["keep"]
    clear_source.assert_called_once_with("s1")
    clear_files.assert_called_once()


# =============================================================================
# Guide refresh
# =============================================================================


def test_guide_refresh_workers(
    api: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    flags = m3u_module.get_refresh_in_progress()
    flags.difference_update({"live_refresh", "epg_refresh"})
    fake_thread, targets = _thread_recorder()
    with patch("netv.main.threading", SimpleNamespace(Thread=fake_thread)):
        resp = api.get("/guide/refresh", follow_redirects=False)
        again = api.get("/guide/refresh", follow_redirects=False)
    assert resp.status_code == again.status_code == 303
    assert resp.headers["location"] == "/guide?refreshing=1"
    assert {"live_refresh", "epg_refresh"} <= flags
    refresh_live, refresh_epg = targets
    data = cache_module.get_cache()

    with (
        patch("netv.main.load_all_live_data", return_value=(["c"], ["s"], [("http://e", 5, "x")])),
        patch("netv.main.save_file_cache") as save,
    ):
        refresh_live()
    assert data.get("live_streams") == ["s"]
    save.assert_called_once_with(
        "live_data",
        {"cats": ["c"], "streams": ["s"], "epg_urls": [("http://e", 5, "x")]},
    )
    assert "live_refresh" not in flags

    data["epg_error"] = "old"
    with (
        patch("netv.main.epg.clear") as clear,
        patch("netv.main._fetch_all_epg", return_value=5) as fetch,
    ):
        refresh_epg()
    clear.assert_called_once()
    fetch.assert_called_once_with([("http://e", 5, "x")])
    assert "epg_error" not in data
    assert "epg_refresh" not in flags

    with caplog.at_level(logging.WARNING):
        with patch("netv.main.load_all_live_data", side_effect=RuntimeError("live down")):
            refresh_live()
        with (
            patch("netv.main.epg.clear"),
            patch("netv.main._fetch_all_epg", side_effect=RuntimeError("x")),
        ):
            refresh_epg()
        assert data["epg_error"] == "x"
        no_epg_urls: list[tuple[str, int, str]] = []
        data["epg_urls"] = no_epg_urls
        refresh_epg()
    assert "Live refresh failed: live down" in caplog.text
    assert "EPG refresh: no EPG URLs available" in caplog.text
    assert not {"live_refresh", "epg_refresh"} & flags


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
