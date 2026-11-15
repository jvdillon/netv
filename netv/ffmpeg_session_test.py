"""Tests for ffmpeg session management."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from types import CoroutineType
from typing import cast

import asyncio
import json
import logging
import os
import pathlib
import threading
import time
import uuid

from fastapi import HTTPException

import pytest

from netv import cache, fast_start, ffmpeg_command, ffmpeg_session
from netv.fast_start import master_playlist
from netv.fast_start_test import playlist, video_packet
from netv.ffmpeg_command import MediaInfo, SubtitleStream
from netv.playback_policy import LiveRecoveryPolicy, PlaybackHealth, PlaybackPolicy
from netv.testing import (
    BackgroundTasks,
    FakeClock,
    FakeProcess,
    FakeResponse,
    FakeRun,
    FakeWeb,
    ProcessLauncher,
    Program,
    install_clock,
    install_launcher,
    install_subprocess,
    install_web,
)


_NOW = 1_700_000_000.0
ARCHIVE_URL = "https://upstream.test/timeshift/user/pass/60/2026-09-28:10-00/1.ts"
_MOVIE_PROBE = json.dumps(
    {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "height": 1080,
                "field_order": "tt",
                "bit_rate": "8000000",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "sample_rate": "48000",
            },
            {
                "index": 2,
                "codec_type": "subtitle",
                "codec_name": "subrip",
                "tags": {"language": "eng"},
            },
        ],
        "format": {"format_name": "mpegts", "duration": "120"},
    },
)
# Interlaced 720p with Dolby 5.1 and two extractable subtitle tracks beside a bitmap one.
_DOLBY_PROBE = json.dumps(
    {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "height": 720,
                "field_order": "tt",
                "bit_rate": "1050000",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "ac3",
                "channels": 6,
                "sample_rate": "48000",
            },
            {
                "index": 2,
                "codec_type": "subtitle",
                "codec_name": "subrip",
                "tags": {"language": "eng"},
            },
            {
                "index": 3,
                "codec_type": "subtitle",
                "codec_name": "subrip",
                "tags": {"language": "spa"},
            },
            {"index": 4, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
        ],
        "format": {"format_name": "mpegts", "duration": "1500"},
    },
)


@dataclass(kw_only=True, slots=True, frozen=True)
class _Env:
    sessions: dict[str, ffmpeg_session.TranscodeSession]
    urls: dict[str, str]
    root: pathlib.Path
    launcher: ProcessLauncher
    run: FakeRun
    web: FakeWeb
    clock: FakeClock


def _media_playlist(
    path: pathlib.Path,
    segments: Sequence[str],
    *,
    duration: float = 3.0,
    start_pts: float = 10.0,
) -> None:
    lines: list[str] = []
    for index, name in enumerate(segments):
        (path.parent / name).write_bytes(video_packet(start_pts + index * duration) * 10)
        lines += [f"#EXTINF:{duration},", name]
    path.write_text("\n".join(lines) + "\n")


def _ffmpeg_output(command: list[str]) -> None:
    """Write the playlist and segments the launched ffmpeg command would publish."""
    playlists = [arg for arg in command if arg.endswith(".m3u8")]
    if not playlists:
        return
    output = pathlib.Path(playlists[-1])
    if output.name == "input.m3u8":
        _media_playlist(output, ["input_000009.ts"], duration=2.0, start_pts=1.5)
    elif output.name == "low.m3u8":
        _media_playlist(output, ["low_000009.ts", "low_000010.ts"], duration=4.0)
    else:
        start = (
            int(command[command.index("-start_number") + 1]) if "-start_number" in command else 0
        )
        _media_playlist(output, [f"seg{start:03d}.ts", f"seg{start + 1:03d}.ts"])


def _input_only(command: list[str]) -> None:
    if command[-1].endswith("input.m3u8"):
        _ffmpeg_output(command)


def _encoder_only(command: list[str]) -> None:
    if command[-1].endswith("low.m3u8"):
        _ffmpeg_output(command)


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> _Env:
    sessions: dict[str, ffmpeg_session.TranscodeSession] = {}
    urls: dict[str, str] = {}
    monkeypatch.setattr(ffmpeg_session, "_transcode_sessions", sessions)
    monkeypatch.setattr(ffmpeg_session, "_url_to_session", urls)
    monkeypatch.setattr(ffmpeg_session, "_startup_guards", {})
    monkeypatch.setattr(ffmpeg_command, "_load_settings", cache.load_server_settings)
    monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", None)
    monkeypatch.setattr(ffmpeg_command, "_probe_cache", {})
    monkeypatch.setattr(ffmpeg_command, "_series_probe_cache", {})
    monkeypatch.setattr(ffmpeg_command, "_gpu_nvdec_codecs", set())
    root = tmp_path / "transcode"
    cache.save_server_settings({**cache.load_server_settings(), "transcode_dir": str(root)})
    run = FakeRun()
    install_subprocess(monkeypatch, ffmpeg_command, run=run)
    web = FakeWeb()
    install_web(monkeypatch, web, ffmpeg_command)
    clock = FakeClock(now=_NOW)
    install_clock(monkeypatch, clock, ffmpeg_session)
    launcher = ProcessLauncher(on_launch=_ffmpeg_output)
    install_launcher(monkeypatch, launcher, ffmpeg_session)
    return _Env(
        sessions=sessions,
        urls=urls,
        root=root,
        launcher=launcher,
        run=run,
        web=web,
        clock=clock,
    )


@pytest.fixture
def tasks(monkeypatch: pytest.MonkeyPatch) -> Iterator[BackgroundTasks]:
    background = BackgroundTasks()
    monkeypatch.setattr(ffmpeg_session, "_spawn_background_task", background)
    yield background
    background.discard()


def _relaunch(
    monkeypatch: pytest.MonkeyPatch,
    processes: Sequence[FakeProcess],
    *,
    error: OSError | None = None,
) -> ProcessLauncher:
    launcher = ProcessLauncher(processes, error=error, on_launch=_ffmpeg_output)
    install_launcher(monkeypatch, launcher, ffmpeg_session)
    return launcher


def _proc(process: FakeProcess) -> asyncio.subprocess.Process:
    return cast(asyncio.subprocess.Process, process)


def _coroutine_names(background: BackgroundTasks) -> list[str]:
    names: list[str] = []
    for task in background.tasks:
        assert isinstance(task.coroutine, CoroutineType)
        names.append(task.coroutine.__name__)
    return names


async def _idle() -> None:
    await asyncio.Event().wait()


async def _connected() -> bool:
    return False


async def _disconnected() -> bool:
    return True


async def until(condition: Callable[[], object]) -> None:
    """Yield to the event loop until ``condition`` holds; real worker threads may run."""
    deadline = time.monotonic() + 1
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        await asyncio.sleep(0)


def _live(directory: pathlib.Path, url: str = "http://live") -> ffmpeg_session.TranscodeSession:
    return ffmpeg_session.TranscodeSession(
        dir=str(directory),
        process=FakeProcess(exit_code=None),
        started=_NOW,
        last_access=_NOW,
        url=url,
    )


def _save_settings_cache(*, live_secs: int = 0, vod_mins: int = 60) -> None:
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "live_transcode_cache_secs": live_secs,
            "vod_transcode_cache_mins": vod_mins,
        },
    )


# =============================================================================
# Process liveness, cache timeouts, validity
# =============================================================================


class TestIsProcessAlive:
    """Tests for _is_process_alive."""

    def test_none_is_dead(self):
        assert ffmpeg_session._is_process_alive(None) is False

    def test_dead_process_placeholder(self):
        assert ffmpeg_session._is_process_alive(ffmpeg_session._DeadProcess()) is False

    def test_alive_process(self):
        assert ffmpeg_session._is_process_alive(FakeProcess(exit_code=None)) is True

    def test_dead_process(self):
        assert ffmpeg_session._is_process_alive(FakeProcess(exited=True)) is False

    def test_object_without_process_state_is_dead(self):
        assert ffmpeg_session._is_process_alive(object()) is False


class TestCacheTimeouts:
    """Tests for cache timeout getters."""

    def test_vod_cache_timeout_default(self):
        """VOD cache default is 60 min = 3600 sec."""
        assert ffmpeg_session.get_vod_cache_timeout() == 3600

    def test_vod_cache_timeout_custom(self):
        _save_settings_cache(vod_mins=30)
        assert ffmpeg_session.get_vod_cache_timeout() == 1800

    def test_live_cache_timeout_default(self):
        """Live cache default is 0 (no caching)."""
        assert ffmpeg_session.get_live_cache_timeout() == 0

    def test_live_cache_timeout_custom(self):
        _save_settings_cache(live_secs=30)
        assert ffmpeg_session.get_live_cache_timeout() == 30


class TestIsSessionValid:
    """Tests for is_session_valid with heartbeat timeout."""

    def _session(
        self,
        tmp_path: pathlib.Path,
        *,
        alive: bool,
        age: float,
        is_vod: bool = False,
    ) -> ffmpeg_session.TranscodeSession:
        return ffmpeg_session.TranscodeSession(
            process=FakeProcess(exit_code=None) if alive else FakeProcess(exited=True),
            started=_NOW - age,
            last_access=_NOW - age,
            is_vod=is_vod,
            dir=str(tmp_path),
            url="http://stream.test/live",
        )

    def test_active_process_with_recent_heartbeat(self, tmp_path: pathlib.Path):
        assert ffmpeg_session.is_session_valid(self._session(tmp_path, alive=True, age=0)) is True

    def test_active_process_stale_heartbeat(self, tmp_path: pathlib.Path):
        session = self._session(tmp_path, alive=True, age=400)
        assert ffmpeg_session.is_session_valid(session) is False

    def test_dead_process_live_session_no_cache(self, tmp_path: pathlib.Path):
        _save_settings_cache(live_secs=0)
        session = self._session(tmp_path, alive=False, age=0)
        assert ffmpeg_session.is_session_valid(session) is False

    def test_dead_process_live_session_within_cache(self, tmp_path: pathlib.Path):
        _save_settings_cache(live_secs=30)
        session = self._session(tmp_path, alive=False, age=10)
        assert ffmpeg_session.is_session_valid(session) is True

    def test_dead_process_vod_session_within_cache(self, tmp_path: pathlib.Path):
        _save_settings_cache(vod_mins=60)
        session = self._session(tmp_path, alive=False, age=10, is_vod=True)
        assert ffmpeg_session.is_session_valid(session, heartbeat_timeout_sec=3600) is True

    def test_dead_process_vod_session_expired_cache(self, tmp_path: pathlib.Path):
        _save_settings_cache(vod_mins=1)
        session = self._session(tmp_path, alive=False, age=120, is_vod=True)
        assert ffmpeg_session.is_session_valid(session, heartbeat_timeout_sec=3600) is False

    def test_heartbeat_timeout_boundary(self, tmp_path: pathlib.Path):
        session = self._session(tmp_path, alive=True, age=30)
        assert ffmpeg_session.is_session_valid(session) is True
        session["last_access"] = _NOW - 30.5
        assert ffmpeg_session.is_session_valid(session) is False

    def test_missing_last_access_uses_started(self, tmp_path: pathlib.Path):
        session = ffmpeg_session.TranscodeSession(
            process=FakeProcess(exit_code=None),
            started=_NOW - 31,
            dir=str(tmp_path),
            url="http://stream.test/live",
        )
        assert ffmpeg_session.is_session_valid(session) is False
        session["started"] = _NOW
        assert ffmpeg_session.is_session_valid(session) is True


def test_fast_start_session_with_recovery_stays_valid_without_process(tmp_path: pathlib.Path):
    session = ffmpeg_session.FastLiveSession(
        started=_NOW,
        fast_start=True,
        live_recovery=LiveRecoveryPolicy(),
        process=None,
        dir=str(tmp_path),
        url="http://stream.test/live",
        ingest_restart_command=["ffmpeg", "ingest"],
    )
    assert ffmpeg_session.is_session_valid(session) is True


# =============================================================================
# Stop, cleanup, shutdown
# =============================================================================


class TestStopSession:
    """Tests for stop_session."""

    def test_stop_nonexistent_session(self):
        asyncio.run(ffmpeg_session.stop_session("nonexistent"))

    def test_stop_session_force(self, env: _Env, tmp_path: pathlib.Path):
        process = FakeProcess(exit_code=None)
        env.sessions["s"] = ffmpeg_session.TranscodeSession(
            process=process,
            dir=str(tmp_path),
            url="http://test",
            last_access=_NOW,
            started=_NOW,
        )
        env.urls["http://test"] = "s"
        asyncio.run(ffmpeg_session.stop_session("s", force=True))
        assert "s" not in env.sessions
        assert "http://test" not in env.urls
        assert process.signals == ["TERM"]
        assert not tmp_path.exists()

    def test_stop_keeps_url_mapped_to_another_session(self, env: _Env, tmp_path: pathlib.Path):
        env.sessions["s"] = _live(tmp_path, url="http://test")
        env.urls["http://test"] = "newer"
        asyncio.run(ffmpeg_session.stop_session("s", force=True))
        assert env.urls == {"http://test": "newer"}

    @pytest.mark.parametrize("is_vod", [True, False])
    def test_stop_skips_recently_accessed_session(
        self,
        env: _Env,
        tmp_path: pathlib.Path,
        is_vod: bool,
    ):
        """Seeking or another viewer touched the session within the last 5 s."""
        process = FakeProcess(exit_code=None)
        env.sessions["s"] = ffmpeg_session.TranscodeSession(
            process=process,
            dir=str(tmp_path),
            url="http://test",
            is_vod=is_vod,
            last_access=_NOW - 4.9,
            started=_NOW,
        )
        asyncio.run(ffmpeg_session.stop_session("s", force=False))
        assert "s" in env.sessions
        assert process.returncode is None

    def test_stop_session_multi_user_grace_period(self, env: _Env, tmp_path: pathlib.Path):
        """Stopping session while another user watching should preserve session."""
        process = FakeProcess(exit_code=None)
        env.sessions["s"] = ffmpeg_session.TranscodeSession(
            process=process,
            dir=str(tmp_path),
            url="http://shared-stream",
            last_access=_NOW - 10,
            started=_NOW,
        )
        ffmpeg_session.touch_session("s")
        asyncio.run(ffmpeg_session.stop_session("s", force=False))
        assert "s" in env.sessions
        assert process.returncode is None

    def test_stop_session_caches_vod(self, env: _Env, tmp_path: pathlib.Path):
        """Stop caches VOD session instead of removing it."""
        process = FakeProcess(exit_code=None)
        env.sessions["s"] = ffmpeg_session.TranscodeSession(
            process=process,
            dir=str(tmp_path),
            url="http://vod",
            is_vod=True,
            last_access=_NOW - 5,
            started=_NOW,
        )
        _save_settings_cache(vod_mins=60)
        asyncio.run(ffmpeg_session.stop_session("s", force=False))
        assert "s" in env.sessions
        assert env.sessions["s"].get("last_access") == _NOW
        assert process.signals == ["TERM"]
        assert tmp_path.exists()

    def test_stale_live_session_without_cache_is_removed(self, env: _Env, tmp_path: pathlib.Path):
        env.sessions["s"] = _live(tmp_path)
        env.sessions["s"]["last_access"] = _NOW - 10
        asyncio.run(ffmpeg_session.stop_session("s"))
        assert not env.sessions
        assert not tmp_path.exists()


def test_cached_live_stop_keeps_session_but_stops_its_process(env: _Env, tmp_path: pathlib.Path):
    cache.save_server_settings({**cache.load_server_settings(), "live_transcode_cache_secs": 30})
    env.sessions["s"] = _live(tmp_path)
    env.sessions["s"]["last_access"] = 0
    process = env.sessions["s"]["process"]
    asyncio.run(ffmpeg_session.stop_session("s"))
    assert process is not None
    assert process.returncode == -15
    assert env.sessions["s"].get("last_access") == _NOW
    assert tmp_path.exists()


@pytest.mark.asyncio
async def test_stop_session_cancels_pending_tasks_and_kills_extra_processes(
    env: _Env,
    tmp_path: pathlib.Path,
):
    watchdog = asyncio.create_task(_idle())
    extra = FakeProcess(exit_code=None)
    env.sessions["s"] = _live(tmp_path)
    env.sessions["s"].update({"watchdog_task": watchdog, "extra_processes": [extra]})
    await ffmpeg_session.stop_session("s", force=True)
    await asyncio.sleep(0)
    assert watchdog.cancelled()
    assert extra.returncode == -15
    assert "s" not in env.sessions
    assert not tmp_path.exists()


def test_stop_user_stream_matches_url_or_request_url_for_owner_only(
    env: _Env,
    tmp_path: pathlib.Path,
):
    # "b" matches only through the URL the client asked for, not the resolved one.
    for sid, user, url, request_url in [
        ("a", "alice", "http://stream", None),
        ("b", "alice", "http://resolved/b", "http://stream"),
        ("c", "bob", "http://stream", None),
    ]:
        directory = tmp_path / sid
        directory.mkdir()
        env.sessions[sid] = ffmpeg_session.TranscodeSession(
            dir=str(directory),
            process=FakeProcess(exit_code=None),
            started=_NOW,
            url=url,
            username=user,
        )
        if request_url is not None:
            env.sessions[sid]["request_url"] = request_url
    asyncio.run(ffmpeg_session.stop_user_stream("alice", "http://stream"))
    assert set(env.sessions) == {"c"}


@pytest.mark.asyncio
async def test_watchdog_stopping_its_own_session_is_not_cancelled(
    env: _Env,
    tmp_path: pathlib.Path,
):
    finished = asyncio.Event()

    async def watchdog() -> None:
        await ffmpeg_session.stop_session("s", force=True)
        finished.set()

    env.sessions["s"] = _live(tmp_path)
    env.sessions["s"]["fast_start"] = True
    task = asyncio.create_task(watchdog())
    env.sessions["s"]["watchdog_task"] = task
    await task
    assert finished.is_set()
    assert not env.sessions


class TestCleanupExpiredSessions:
    """Tests for cleanup_expired_sessions."""

    def test_cleanup_removes_expired_and_keeps_valid(self, env: _Env, tmp_path: pathlib.Path):
        expired, valid = tmp_path / "expired", tmp_path / "valid"
        expired.mkdir()
        valid.mkdir()
        env.sessions["expired"] = ffmpeg_session.TranscodeSession(
            process=FakeProcess(exited=True),
            dir=str(expired),
            url="http://expired",
            started=_NOW - 400,
            last_access=_NOW - 400,
        )
        env.sessions["valid"] = _live(valid, url="http://valid")
        asyncio.run(ffmpeg_session.cleanup_expired_sessions())
        assert set(env.sessions) == {"valid"}
        assert not expired.exists()
        assert valid.exists()


@pytest.mark.asyncio
async def test_shutdown_cancels_tasks_and_kills_every_process(env: _Env, tmp_path: pathlib.Path):
    task = asyncio.create_task(_idle())
    extra, first, second = (FakeProcess(exit_code=None) for _ in range(3))
    env.sessions["s"] = ffmpeg_session.TranscodeSession(
        watchdog_task=task,
        extra_processes=[extra],
        process=first,
        dir=str(tmp_path),
        started=_NOW,
        url="http://stream.test/live",
    )
    env.sessions["t"] = _live(tmp_path)
    env.sessions["t"]["process"] = second
    await ffmpeg_session.shutdown()
    await asyncio.sleep(0)
    assert task.cancelled()
    assert [p.returncode for p in (extra, first, second)] == [-15, -15, -15]
    assert not env.sessions


# =============================================================================
# Stream limits and session queries
# =============================================================================


def _owned(
    tmp_path: pathlib.Path,
    *,
    started: float,
    username: str = "",
    source_id: str = "",
) -> ffmpeg_session.TranscodeSession:
    return ffmpeg_session.TranscodeSession(
        username=username,
        source_id=source_id,
        started=started,
        dir=str(tmp_path),
        process=None,
        url="http://stream.test/live",
    )


def test_user_and_source_sessions_filter_and_sort_by_start(
    env: _Env,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env.sessions["s3"] = _owned(tmp_path, started=3, username="alice", source_id="src1")
    env.sessions["s1"] = _owned(tmp_path, started=1, username="alice", source_id="src1")
    env.sessions["s2"] = _owned(tmp_path, started=2, username="bob", source_id="src2")
    legacy = _owned(tmp_path, started=99, username="alice", source_id="src1")
    # Sessions recovered from before start times were recorded lack "started".
    monkeypatch.delitem(cast(Mapping[str, object], legacy), "started")
    env.sessions["legacy"] = legacy
    assert [sid for sid, _ in ffmpeg_session.get_user_sessions("alice")] == [
        "legacy",
        "s1",
        "s3",
    ]
    assert [sid for sid, _ in ffmpeg_session.get_source_sessions("src1")] == [
        "legacy",
        "s1",
        "s3",
    ]
    assert ffmpeg_session.get_user_sessions("alice")[0][1] is legacy
    assert ffmpeg_session.get_user_sessions("nobody") == []


class TestEnforceStreamLimits:
    """Tests for enforce_stream_limits."""

    def test_no_limits_returns_none(self):
        assert asyncio.run(ffmpeg_session.enforce_stream_limits("alice", None, 0, 0)) is None

    def test_user_limit_stops_oldest(self, env: _Env, tmp_path: pathlib.Path):
        for sid, started in [("s1", 1), ("s2", 2)]:
            env.sessions[sid] = _live(tmp_path / sid)
            env.sessions[sid].update(username="alice", started=started)
        assert asyncio.run(ffmpeg_session.enforce_stream_limits("alice", None, 2, 0)) is None
        assert set(env.sessions) == {"s2"}

    def test_user_below_limit_keeps_sessions(self, env: _Env, tmp_path: pathlib.Path):
        env.sessions["s1"] = _live(tmp_path)
        env.sessions["s1"]["username"] = "alice"
        assert asyncio.run(ffmpeg_session.enforce_stream_limits("alice", None, 2, 0)) is None
        assert set(env.sessions) == {"s1"}

    def test_source_limit_stops_user_session(self, env: _Env, tmp_path: pathlib.Path):
        env.sessions["s1"] = _live(tmp_path)
        env.sessions["s1"].update(username="alice", source_id="src1")
        assert asyncio.run(ffmpeg_session.enforce_stream_limits("alice", "src1", 0, 1)) is None
        assert not env.sessions

    def test_source_limit_returns_error_for_other_user(self, env: _Env, tmp_path: pathlib.Path):
        env.sessions["s1"] = _live(tmp_path)
        env.sessions["s1"].update(username="bob", source_id="src1")
        result = asyncio.run(ffmpeg_session.enforce_stream_limits("alice", "src1", 0, 1))
        assert result == "Source at capacity (1 streams)"
        assert "s1" in env.sessions


def test_get_session_returns_a_copy(env: _Env, tmp_path: pathlib.Path):
    env.sessions["test"] = _live(tmp_path)
    session = ffmpeg_session.get_session("test")
    assert session == env.sessions["test"]
    assert session is not env.sessions["test"]
    assert ffmpeg_session.get_session("nonexistent") is None


def test_touch_session_records_heartbeat(env: _Env, tmp_path: pathlib.Path):
    env.sessions["test"] = _live(tmp_path)
    env.sessions["test"]["last_access"] = _NOW - 100
    assert ffmpeg_session.touch_session("test") is True
    assert env.sessions["test"].get("last_access") == _NOW
    assert ffmpeg_session.touch_session("nonexistent") is False


class TestGetSessionProgress:
    """Tests for get_session_progress."""

    def test_progress_with_playlist(self, env: _Env, tmp_path: pathlib.Path):
        (tmp_path / "stream.m3u8").write_text(
            "#EXTM3U\n#EXTINF:3.0,\nseg0.ts\n#EXTINF:3.5,\nseg1.ts\n",
        )
        env.sessions["test"] = _live(tmp_path)
        env.sessions["test"]["last_access"] = 0
        assert ffmpeg_session.get_session_progress("test") == {
            "segment_count": 2,
            "duration": 6.5,
        }
        assert env.sessions["test"].get("last_access") == _NOW

    def test_progress_no_playlist(self, env: _Env, tmp_path: pathlib.Path):
        env.sessions["test"] = _live(tmp_path)
        assert ffmpeg_session.get_session_progress("test") == {
            "segment_count": 0,
            "duration": 0.0,
        }

    def test_progress_nonexistent_session(self):
        assert ffmpeg_session.get_session_progress("nonexistent") is None

    def test_progress_reads_fast_start_playlist(self, env: _Env, tmp_path: pathlib.Path):
        (tmp_path / "low.m3u8").write_text("#EXTINF:2.0,\nlow_0.ts\n")
        env.sessions["s"] = _live(tmp_path)
        env.sessions["s"]["fast_start"] = True
        assert ffmpeg_session.get_session_progress("s") == {"segment_count": 1, "duration": 2.0}


def test_clear_url_session(env: _Env):
    env.urls["http://test"] = "session-123"
    assert ffmpeg_session.clear_url_session("http://test") == "session-123"
    assert not env.urls
    assert ffmpeg_session.clear_url_session("http://nonexistent") is None


@pytest.mark.parametrize("seek_time", [None, 0, 30])
def test_shared_session_process_update(env: _Env, seek_time: int | None, tmp_path: pathlib.Path):
    session = _live(tmp_path)
    session.update(seek_offset=15, caption_timestamp_origin=3.0)
    env.sessions["resume"] = session
    process = FakeProcess(exit_code=None)
    assert ffmpeg_session._update_session_process(
        "resume",
        process,
        url="https://provider/movie",
        seek_time=seek_time,
    )
    assert session["process"] is process
    assert session.get("seek_offset") == (15 if seek_time is None else seek_time)
    # A seek moves the timeline, so the caption clock must be measured again.
    assert ("caption_timestamp_origin" in session) is (seek_time is None)
    assert env.urls == {"https://provider/movie": "resume"}
    assert not ffmpeg_session._update_session_process(
        "missing",
        process,
        url="https://provider/missing",
        seek_time=0,
    )
    assert "https://provider/missing" not in env.urls


def test_update_session_process_without_url_keeps_mapping(env: _Env, tmp_path: pathlib.Path):
    env.sessions["s"] = _live(tmp_path)
    assert ffmpeg_session._update_session_process("s", FakeProcess(exit_code=None))
    assert not env.urls


# =============================================================================
# Startup recovery
# =============================================================================


def _vod_dir(root: pathlib.Path, name: str, info: object, *, segment: bool = True) -> pathlib.Path:
    directory = root / f"netv_transcode_{name}"
    directory.mkdir(parents=True)
    (directory / "session.json").write_text(info if isinstance(info, str) else json.dumps(info))
    if segment:
        (directory / "seg000.ts").write_bytes(b"x" * 2000)
    return directory


def test_recovery_removes_orphaned_dirs(env: _Env):
    env.root.mkdir()
    orphan = env.root / "netv_transcode_orphan"
    orphan.mkdir()
    (orphan / "seg000.ts").write_bytes(b"data")
    ffmpeg_session.cleanup_and_recover_sessions()
    assert not orphan.exists()


def test_recovery_restores_valid_vod_session(env: _Env):
    _vod_dir(
        env.root,
        "vod123",
        {
            "session_id": "vod123",
            "url": "http://movie.mp4",
            "is_vod": True,
            "started": 5,
            "duration": 3600,
            "subtitle_indices": [2],
            "username": "alice",
        },
    )
    ffmpeg_session.cleanup_and_recover_sessions()
    session = env.sessions["vod123"]
    assert env.urls == {"http://movie.mp4": "vod123"}
    assert isinstance(session["process"], ffmpeg_session._DeadProcess)
    assert (session["started"], session.get("last_access")) == (5, _NOW)
    assert (session.get("duration"), session.get("subtitles")) == (3600, [2])
    assert (session.get("is_vod"), session.get("username")) == (True, "alice")


def test_recovery_removes_expired_vod_session(env: _Env):
    directory = _vod_dir(
        env.root,
        "expired",
        {"session_id": "expired", "url": "http://old.mp4", "is_vod": True},
    )
    os.utime(directory, (_NOW - 61, _NOW - 61))
    cache.save_server_settings({**cache.load_server_settings(), "vod_transcode_cache_mins": 1})
    ffmpeg_session.cleanup_and_recover_sessions()
    assert not directory.exists()
    assert not env.sessions


def test_recovery_discards_legacy_cached_archives(env: _Env):
    directory = _vod_dir(
        env.root,
        "archive",
        {"session_id": "archive", "url": ARCHIVE_URL, "is_vod": True},
    )
    ffmpeg_session.cleanup_and_recover_sessions()
    assert not directory.exists()
    assert not env.sessions


def test_recovery_ignores_plain_files_and_dangling_links(env: _Env):
    env.root.mkdir()
    stray = env.root / "netv_transcode_file"
    stray.write_text("x")
    dangling = env.root / "netv_transcode_link"
    dangling.symlink_to(env.root / "missing")
    ffmpeg_session.cleanup_and_recover_sessions()
    assert stray.exists()
    assert dangling.is_symlink()
    assert not env.sessions


def test_recovery_removes_session_without_segments(env: _Env):
    directory = _vod_dir(
        env.root,
        "empty",
        {"session_id": "e", "url": "http://m", "is_vod": True},
        segment=False,
    )
    ffmpeg_session.cleanup_and_recover_sessions()
    assert not directory.exists()
    assert not env.sessions


def test_recovery_removes_corrupt_session_json(env: _Env):
    directory = _vod_dir(env.root, "bad", "{not json")
    ffmpeg_session.cleanup_and_recover_sessions()
    assert not directory.exists()


@pytest.mark.parametrize(
    "existing_seek,new_seek,winner",
    [(0, 5, "new"), (5, 0, "old"), (0, 0, "new")],
)
def test_recovery_prefers_seeked_or_newer_duplicate(
    env: _Env,
    tmp_path: pathlib.Path,
    existing_seek: int,
    new_seek: int,
    winner: str,
):
    env.sessions["old"] = ffmpeg_session.TranscodeSession(
        seek_offset=existing_seek,
        last_access=0,
        dir=str(tmp_path),
        process=None,
        started=_NOW,
        url="http://stream.test/live",
    )
    env.urls["http://m"] = "old"
    _vod_dir(
        env.root,
        "new",
        {"session_id": "new", "url": "http://m", "is_vod": True, "seek_offset": new_seek},
    )
    ffmpeg_session.cleanup_and_recover_sessions()
    assert "new" in env.sessions
    assert env.urls["http://m"] == winner


def test_recovery_restores_probe_cache(env: _Env):
    info = {
        "session_id": "p",
        "url": "http://m",
        "is_vod": True,
        "duration": 99,
        "series_id": 7,
        "episode_id": 8,
        "subtitles": [{"index": 2, "lang": "eng"}, 3],
        "probe": {"video_codec": "hevc", "height": 2160, "interlaced": True},
    }
    _vod_dir(env.root, "p", info)
    ffmpeg_session.cleanup_and_recover_sessions()
    media, subs = ffmpeg_command.probe_media("http://m", 7, 8)
    assert media is not None
    assert (media.video_codec, media.height, media.duration, media.interlaced) == (
        "hevc",
        2160,
        99,
        True,
    )
    assert subs == [SubtitleStream(index=2, lang="eng", name="")]
    assert ffmpeg_command.probe_media("http://m") == (media, subs)
    assert env.run.calls == []


# =============================================================================
# Playlist and caption helpers
# =============================================================================


def test_calc_hls_duration_reads_playlist_or_estimates(tmp_path: pathlib.Path):
    playlist_path = tmp_path / "stream.m3u8"
    assert ffmpeg_session._calc_hls_duration(playlist_path, 5) == 15.0
    playlist_path.write_text("#EXTM3U\n")
    assert ffmpeg_session._calc_hls_duration(playlist_path, 2) == 6.0
    playlist_path.write_text("#EXTM3U\n#EXTINF:3.5,\nseg0.ts\n#EXTINF:3.0,\nseg1.ts\n")
    assert ffmpeg_session._calc_hls_duration(playlist_path, 2) == 6.5


class TestBuildSubtitleTracks:
    """Tests for _build_subtitle_tracks."""

    def test_builds_track_list(self):
        sub_info: list[ffmpeg_session.StoredSubtitle | int] = [
            {"index": 2, "lang": "eng", "name": "English"},
            {"index": 3, "lang": "jpn", "name": "Japanese"},
        ]
        assert ffmpeg_session._build_subtitle_tracks("session-123", sub_info) == [
            {
                "url": "/subs/session-123/sub0.vtt",
                "lang": "eng",
                "label": "English",
                "default": True,
            },
            {
                "url": "/subs/session-123/sub1.vtt",
                "lang": "jpn",
                "label": "Japanese",
                "default": False,
            },
        ]

    def test_empty_sub_info(self):
        assert ffmpeg_session._build_subtitle_tracks("s", []) == []
        assert ffmpeg_session._build_subtitle_tracks("s", None) == []

    def test_non_dict_sub_info(self):
        """Returns empty list for old format (indices only)."""
        assert ffmpeg_session._build_subtitle_tracks("s", [2, 3]) == []


class TestCaptionMasterPlaylist:
    def test_links_native_hls_subtitle_renditions(self):
        result = ffmpeg_session._caption_master_playlist(
            [
                {"lang": "eng", "name": "English"},
                {"lang": "spa", "name": 'Español "CC"'},
                {"lang": "", "name": "  "},
            ],
            "1080p",
        )
        assert result == (
            "#EXTM3U\n#EXT-X-VERSION:3\n"
            '#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="captions",NAME="English",LANGUAGE="eng",'
            'DEFAULT=NO,AUTOSELECT=YES,FORCED=NO,URI="sub0.m3u8"\n'
            '#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="captions",NAME="Español \'CC\'",'
            'LANGUAGE="spa",DEFAULT=NO,AUTOSELECT=YES,FORCED=NO,URI="sub1.m3u8"\n'
            '#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="captions",NAME="",LANGUAGE="und",'
            'DEFAULT=NO,AUTOSELECT=YES,FORCED=NO,URI="sub2.m3u8"\n'
            '#EXT-X-STREAM-INF:BANDWIDTH=9440000,SUBTITLES="captions"\nstream.m3u8\n'
        )

    def test_names_unnamed_tracks_and_disambiguates_duplicates(self):
        result = ffmpeg_session._caption_master_playlist(
            [
                {"lang": "eng", "name": "English"},
                {"lang": "eng", "name": "English"},
                {"lang": "eng", "name": ""},
            ],
            "720p",
        )
        assert 'NAME="English",' in result
        assert 'NAME="English 2",' in result
        assert 'NAME="Captions 3",' in result


class TestWebvttTimestampMap:
    def test_derives_the_clock_origin_from_a_later_video_segment(self, tmp_path: pathlib.Path):
        (tmp_path / "stream.m3u8").write_text(
            "#EXTM3U\n#EXTINF:4.0,\nseg000.ts\n#EXTINF:4.0,\nseg001.ts\n",
        )
        (tmp_path / "seg000.ts").touch()
        (tmp_path / "seg001.ts").write_bytes(video_packet(14.08))
        assert ffmpeg_session._caption_timestamp_origin(str(tmp_path)) == pytest.approx(10.08)

    def test_maps_webvtt_to_the_mpeg_ts_clock(self):
        mapped = ffmpeg_session.add_webvtt_timestamp_map(
            "WEBVTT\n\n00:01.000 --> 00:03.000\nCaption\n",
            10.08,
        )
        assert mapped == (
            "WEBVTT\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:907200\n"
            "\n00:01.000 --> 00:03.000\nCaption\n"
        )
        crlf = ffmpeg_session.add_webvtt_timestamp_map("WEBVTT\r\n\r\nWEBVTT\r\n", 1)
        assert crlf == "WEBVTT\r\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:90000\r\n\r\nWEBVTT\r\n"

    def test_wraps_the_timestamp_at_33_bits(self):
        mapped = ffmpeg_session.add_webvtt_timestamp_map("WEBVTT\n", (1 << 33) / 90_000 + 1)
        assert "MPEGTS:90000\n" in mapped

    def test_preserves_existing_mapping_and_non_webvtt_content(self):
        mapped = "WEBVTT\r\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:90000\r\n"
        assert ffmpeg_session.add_webvtt_timestamp_map(mapped, 10.08) == mapped
        assert ffmpeg_session.add_webvtt_timestamp_map("not WebVTT", 10.08) == "not WebVTT"
        assert ffmpeg_session.add_webvtt_timestamp_map("WEBVTT\n", None) == "WEBVTT\n"


def test_caption_timestamp_origin_paths(env: _Env, tmp_path: pathlib.Path):
    assert ffmpeg_session.get_caption_timestamp_origin("missing") is None
    env.sessions["cached"] = _live(tmp_path)
    env.sessions["cached"]["caption_timestamp_origin"] = 3
    assert ffmpeg_session.get_caption_timestamp_origin("cached") == 3.0
    env.sessions["none"] = _live(tmp_path)
    assert ffmpeg_session.get_caption_timestamp_origin("none") is None
    _media_playlist(tmp_path / "stream.m3u8", ["seg0.ts"], start_pts=4.5)
    assert ffmpeg_session.get_caption_timestamp_origin("none") == 4.5
    assert env.sessions["none"].get("caption_timestamp_origin") == 4.5


def test_caption_timestamp_origin_none_when_session_stops_while_reading(
    env: _Env,
    tmp_path: pathlib.Path,
):
    # Opening a FIFO blocks the reader until a writer opens it: the session
    # is removed between the two locked sections of the lookup.
    (tmp_path / "stream.m3u8").write_text("#EXTINF:2,\nseg0.ts\n")
    fifo = tmp_path / "seg0.ts"
    os.mkfifo(fifo)
    env.sessions["gone"] = _live(tmp_path)
    results: list[float | None] = []
    reader = threading.Thread(
        target=lambda: results.append(ffmpeg_session.get_caption_timestamp_origin("gone")),
    )
    reader.start()
    with fifo.open("wb") as writer:
        env.sessions.pop("gone")
        writer.write(video_packet(1.0))
    reader.join(timeout=5)
    assert results == [None]


def test_caption_timestamp_origin_skips_nested_names_and_counts_elapsed(tmp_path: pathlib.Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.ts").write_bytes(video_packet(99))
    (tmp_path / "seg0.ts").write_bytes(b"not mpeg-ts")
    (tmp_path / "stream.m3u8").write_text("#EXTINF:bad,\n#EXTINF:4,\nsub/x.ts\nseg0.ts\n")
    assert ffmpeg_session._caption_timestamp_origin(str(tmp_path)) is None
    (tmp_path / "seg1.ts").write_bytes(video_packet(10))
    with (tmp_path / "stream.m3u8").open("a") as playlist_file:
        playlist_file.write("#EXTINF:2,\nseg1.ts\n")
    assert ffmpeg_session._caption_timestamp_origin(str(tmp_path)) == 6.0


class TestRegeneratePlaylist:
    """Tests for _regenerate_playlist."""

    def test_regenerates_playlist_from_segments(self, tmp_path: pathlib.Path):
        for i in range(3):
            (tmp_path / f"seg00{i}.ts").write_bytes(b"x" * 2000)
        ffmpeg_session._regenerate_playlist(tmp_path, start_segment=1)
        assert (tmp_path / "stream.m3u8").read_text() == (
            "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:4\n#EXT-X-MEDIA-SEQUENCE:1\n"
            "#EXT-X-PLAYLIST-TYPE:EVENT\n#EXTINF:3.000000,\nseg001.ts\n"
            "#EXTINF:3.000000,\nseg002.ts\n"
        )

    def test_regenerate_skips_small_segments(self, tmp_path: pathlib.Path):
        (tmp_path / "seg000.ts").write_bytes(b"x" * 1000)  # Not above the threshold.
        (tmp_path / "seg001.ts").write_bytes(b"x" * 1001)
        ffmpeg_session._regenerate_playlist(tmp_path, start_segment=0)
        content = (tmp_path / "stream.m3u8").read_text()
        assert "seg001.ts" in content
        assert "seg000.ts" not in content

    def test_regenerate_playlist_skips_bad_names_and_empty(self, tmp_path: pathlib.Path):
        (tmp_path / "segabc.ts").write_bytes(b"x" * 2000)
        ffmpeg_session._regenerate_playlist(tmp_path, 0)
        assert not (tmp_path / "stream.m3u8").exists()


# =============================================================================
# start_transcode routing
# =============================================================================


def _existing(
    env: _Env,
    tmp_path: pathlib.Path,
    url: str,
    *,
    username: str | None = None,
    last_access: float | None = None,
    audio_passthrough: bool = False,
    fast_start: bool = False,
) -> None:
    directory = tmp_path / "old"
    directory.mkdir()
    session = _live(directory, url=url)
    if username is not None:
        session["username"] = username
    if last_access is not None:
        session["last_access"] = last_access
    # Absent and False are distinct for these flags: set only what the case needs.
    if audio_passthrough:
        session["audio_passthrough"] = True
    if fast_start:
        session["fast_start"] = True
    env.sessions["old"] = session
    env.urls[url] = "old"


@pytest.mark.asyncio
async def test_start_rejects_bandwidth_saver_for_vod():
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode("http://m", "movie", bandwidth_saver=True)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_start_enforces_stream_limits(env: _Env, tmp_path: pathlib.Path):
    env.sessions["bob"] = _live(tmp_path, url="http://other")
    env.sessions["bob"].update(username="bob", source_id="src")
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode(
            "http://live",
            username="alice",
            source_id="src",
            source_max_streams=1,
        )
    assert (error.value.status_code, error.value.detail) == (429, "Source at capacity (1 streams)")
    assert env.launcher.commands == []


@pytest.mark.asyncio
async def test_start_rejects_archive_owned_by_other_user(env: _Env, tmp_path: pathlib.Path):
    _existing(env, tmp_path, ARCHIVE_URL, username="bob")
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode(ARCHIVE_URL, "movie", username="alice")
    assert (error.value.status_code, error.value.detail) == (
        409,
        "This archive is playing on another device.",
    )


@pytest.mark.asyncio
async def test_start_reuses_own_archive_session(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    del tasks
    _existing(env, tmp_path, ARCHIVE_URL, username="alice")
    result = await ffmpeg_session.start_transcode(ARCHIVE_URL, "movie", username="alice")
    assert result["session_id"] == "old"
    assert env.launcher.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bandwidth_saver,audio_passthrough",
    [(True, False), (False, True)],
)
async def test_start_replaces_own_incompatible_session(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    bandwidth_saver: bool,
    audio_passthrough: bool,
):
    del tasks
    _existing(env, tmp_path, "http://live", username="alice", audio_passthrough=audio_passthrough)
    env.sessions["old"]["seek_offset"] = 30
    result = await ffmpeg_session.start_transcode(
        "http://live",
        username="alice",
        bandwidth_saver=bandwidth_saver,
    )
    sid = result["session_id"]
    assert set(env.sessions) == {sid}
    # A quality change starts over; only the passthrough swap keeps the position.
    assert env.sessions[sid].get("seek_offset") == (0.0 if bandwidth_saver else 30)
    assert env.sessions[sid].get("bandwidth_saver") is bandwidth_saver
    ((command),) = env.launcher.commands
    assert ("-ss" in command) is not bandwidth_saver


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bandwidth_saver,audio_passthrough,fast_start,status,detail",
    [
        (True, False, False, 404, "Session not found"),
        (False, True, False, 409, "This stream is playing on another device."),
        (False, False, True, 404, "Session not found"),
    ],
)
async def test_start_refuses_other_users_session(
    env: _Env,
    tmp_path: pathlib.Path,
    bandwidth_saver: bool,
    audio_passthrough: bool,
    fast_start: bool,
    status: int,
    detail: str,
):
    _existing(
        env,
        tmp_path,
        "http://live",
        username="bob",
        audio_passthrough=audio_passthrough,
        fast_start=fast_start,
    )
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode(
            "http://live",
            username="alice",
            bandwidth_saver=bandwidth_saver,
        )
    assert (error.value.status_code, error.value.detail) == (status, detail)
    assert "old" in env.sessions
    assert env.launcher.commands == []


@pytest.mark.asyncio
async def test_start_rejoins_own_fast_start_session(env: _Env, tmp_path: pathlib.Path):
    _existing(
        env,
        tmp_path,
        "http://live",
        username="alice",
        last_access=_NOW - 1,
        fast_start=True,
    )
    result = await ffmpeg_session.start_transcode("http://live", username="alice")
    assert result["playlist"] == "/transcode/old/low.m3u8"
    assert env.sessions["old"].get("last_access") == _NOW
    assert env.launcher.commands == []


@pytest.mark.asyncio
async def test_start_reuses_valid_session(
    env: _Env,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
):
    _existing(env, tmp_path, "http://live")
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        result = await ffmpeg_session.start_transcode("http://live")
    assert (result["session_id"], result["playlist"]) == ("old", "/transcode/old/stream.m3u8")
    assert env.launcher.commands == []
    assert [r.getMessage() for r in caplog.records] == [
        "Found valid existing session old (vod=False)",
    ]


@pytest.mark.asyncio
async def test_start_replaces_an_expired_session(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
):
    del tasks
    _existing(env, tmp_path, "http://live", username="alice", last_access=_NOW - 60)
    old = env.sessions["old"]["process"]
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        result = await ffmpeg_session.start_transcode("http://live", username="alice")
    sid = result["session_id"]
    assert set(env.sessions) == {sid} != {"old"}
    assert env.urls == {"http://live": sid}
    assert old is not None
    assert old.returncode == -15
    messages = [r.getMessage() for r in caplog.records]
    assert "Cleaning up invalid session old" in messages
    assert not any(message.startswith("Found valid") for message in messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type,probes", [("movie", 2), ("series", 0)])
async def test_start_resumes_a_cached_vod_session(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    content_type: ffmpeg_session.ContentType,
    probes: int,
):
    del tasks
    _existing(env, tmp_path, "http://movie")
    env.sessions["old"].update(process=FakeProcess(exited=True), is_vod=True, seek_offset=0)
    (tmp_path / "old" / "seg000.ts").write_bytes(b"x" * 2000)
    result = await ffmpeg_session.start_transcode("http://movie", content_type)
    assert result["session_id"] == "old"
    (command,) = env.launcher.commands
    assert _option(command, "-i") == "http://movie"
    # Movies are probed by default and series are not; ffprobe tries each URL twice.
    assert [call[-1] for call in env.run.calls] == ["http://movie"] * probes


@pytest.mark.asyncio
async def test_start_rotates_the_users_oldest_stream(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    del tasks
    env.sessions["older"] = _live(tmp_path, url="http://other")
    env.sessions["older"].update(username="alice", started=1)
    result = await ffmpeg_session.start_transcode(
        "http://live",
        username="alice",
        user_max_streams=1,
    )
    assert set(env.sessions) == {result["session_id"]}


@pytest.mark.asyncio
async def test_start_cleans_up_unreusable_session(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    del tasks
    _existing(env, tmp_path, "http://movie")
    env.sessions["old"].update(process=FakeProcess(exited=True), is_vod=True)
    result = await ffmpeg_session.start_transcode("http://movie", "movie")
    assert set(env.sessions) == {result["session_id"]}
    assert len(env.launcher.commands) == 1


@pytest.mark.asyncio
async def test_start_fast_live_without_dvr(env: _Env, tasks: BackgroundTasks):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "live_dvr_mins": 0})
    result = await ffmpeg_session.start_transcode("http://live", fast_start=True)
    assert result.get("master_playlist") == f"/transcode/{result['session_id']}/master.m3u8"
    session = env.sessions[result["session_id"]]
    assert (session.get("username"), session.get("source_id")) == ("", "")
    ingest_cmd, encoder_cmd = env.launcher.commands
    assert ingest_cmd[ingest_cmd.index("-i") + 1] == "http://live"
    assert any("yadif" in arg for arg in encoder_cmd)
    assert all(call[-1] != "http://live" for call in env.run.calls)


@pytest.mark.asyncio
async def test_start_retries_series_after_clearing_probe_cache(
    env: _Env,
    tasks: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "probe_series": True})
    env.run.program("ffprobe", Program(stdout=_DOLBY_PROBE))
    media = MediaInfo(video_codec="h264", audio_codec="aac", pix_fmt="yuv420p", duration=1)
    ffmpeg_command.restore_probe_cache_entry("http://other", media, [], 3, 4)
    ffmpeg_command.restore_probe_cache_entry("http://other", media, [], 3, 5)
    launcher = _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        result = await ffmpeg_session.start_transcode(
            "http://ep",
            "series",
            series_id=3,
            episode_id=4,
            series_name="Show",
            username="alice",
            source_id="src",
            audio_passthrough=True,
        )
    sid = result["session_id"]
    assert set(env.sessions) == {sid}
    session = env.sessions[sid]
    assert (session.get("is_vod"), session.get("series_id"), session.get("episode_id")) == (
        True,
        3,
        4,
    )
    assert (session.get("username"), session.get("source_id")) == ("alice", "src")
    # The retry probed afresh and found Dolby audio this client can decode.
    assert session.get("audio_passthrough") is True
    assert len(launcher.commands) == 2
    (series,) = ffmpeg_command.get_series_probe_cache_stats()
    assert series["name"] == "Show"
    assert sorted(episode["episode_id"] for episode in series["episodes"]) == [4, 5]
    assert "Transcode failed, clearing probe cache and retrying" in [
        r.getMessage() for r in caplog.records
    ]


@pytest.mark.asyncio
async def test_retry_keeps_the_requested_quality_and_deinterlacing(
    env: _Env,
    tasks: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
):
    del tasks
    cache.save_server_settings(
        {**cache.load_server_settings(), "probe_live": False, "transcode_hw": "software"},
    )
    launcher = _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    result = await ffmpeg_session.start_transcode(
        "http://live",
        "live",
        series_id=3,
        bandwidth_saver=True,
        deinterlace_fallback=False,
    )
    assert env.sessions[result["session_id"]].get("bandwidth_saver") is True
    _, retried = launcher.commands
    assert "min(ih\\,720)" in _option(retried, "-vf")
    assert "yadif" not in _option(retried, "-vf")


@pytest.mark.asyncio
async def test_retry_still_honours_a_disconnected_archive_client(
    env: _Env,
    tasks: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
):
    del tasks
    answers = iter([False, False, True])

    async def disconnected() -> bool:
        return next(answers)

    launcher = _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode(
            ARCHIVE_URL,
            "series",
            series_id=3,
            is_disconnected=disconnected,
        )
    assert error.value.status_code == 499
    assert len(launcher.commands) == 1
    assert not env.sessions


@pytest.mark.asyncio
async def test_failed_movie_start_is_not_retried(
    env: _Env,
    tasks: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
):
    del tasks
    launcher = _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode("http://movie", "movie")
    assert error.value.status_code == 500
    assert len(launcher.commands) == 1
    assert not env.sessions


def _save_max_resolution(
    max_resolution: cache.MaxResolution | None,
    *,
    live_dvr_mins: int = 0,
) -> None:
    """Set the resolution cap, or clear it with None so the 1080p default applies."""
    settings = cache.load_server_settings()
    if max_resolution is None:
        settings.pop("max_resolution", None)
    else:
        settings["max_resolution"] = max_resolution
    settings["live_dvr_mins"] = live_dvr_mins
    cache.save_server_settings(settings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_resolution,fast",
    [(None, True), ("1080p", True), ("4k", True), ("720p", False)],
)
async def test_start_uses_fast_start_only_at_full_resolutions(
    env: _Env,
    fast_env: BackgroundTasks,
    max_resolution: cache.MaxResolution | None,
    fast: bool,
):
    del fast_env
    _save_max_resolution(max_resolution)
    await ffmpeg_session.start_transcode("http://live", fast_start=True)
    assert len(env.launcher.commands) == (2 if fast else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_resolution,height,audio,passthrough,dvr_mins,remux",
    [
        ("4k", 2160, "aac", False, 1, True),
        (None, 1440, "aac", False, 60, False),
        ("1080p", 2160, "aac", False, 60, False),
        ("1080p", 1080, "ac3", True, 60, True),
    ],
)
async def test_dvr_start_remuxes_only_copyable_sources(
    env: _Env,
    fast_env: BackgroundTasks,
    caplog: pytest.LogCaptureFixture,
    max_resolution: cache.MaxResolution | None,
    height: int,
    audio: str,
    passthrough: bool,
    dvr_mins: int,
    remux: bool,
):
    del fast_env
    _save_max_resolution(max_resolution, live_dvr_mins=dvr_mins)
    source = {
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "height": height},
            {
                "codec_type": "audio",
                "codec_name": audio,
                "channels": 2,
                "sample_rate": "48000",
                "profile": "LC",
            },
        ],
    }
    env.run.program("ffprobe", Program(stdout=json.dumps(source)), when="http://live")
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        await ffmpeg_session.start_transcode(
            "http://live",
            fast_start=True,
            audio_passthrough=passthrough,
        )
    assert len(env.launcher.commands) == (1 if remux else 2)
    remux_log = "DVR live session uses a stream-copy remux instead of fast-start: http://live"
    assert (remux_log in [r.getMessage() for r in caplog.records]) is remux


@pytest.mark.asyncio
async def test_start_passes_viewer_choices_to_fast_start(env: _Env, fast_env: BackgroundTasks):
    del fast_env
    _save_max_resolution("4k")
    env.run.program(
        "ffprobe",
        Program(
            stdout='{"streams": [{"codec_name": "eac3", "channels": 6, "sample_rate": "48000"}]}',
        ),
        when="a:0",
    )
    result = await ffmpeg_session.start_transcode(
        "http://live",
        fast_start=True,
        deinterlace_fallback=False,
        audio_passthrough=True,
        bandwidth_saver=True,
        username="alice",
        source_id="src",
    )
    session = env.sessions[result["session_id"]]
    assert (session.get("username"), session.get("source_id")) == ("alice", "src")
    assert session.get("bandwidth_saver") is True
    assert session.get("audio_passthrough") is True
    _, encoder = env.launcher.commands
    assert "yadif" not in _option(encoder, "-vf")
    assert "min(ih\\,720)" in _option(encoder, "-vf")
    assert _option(encoder, "-c:a") == "copy"


@pytest.mark.asyncio
async def test_fast_start_gives_up_when_the_client_disconnects(
    env: _Env,
    fast_env: BackgroundTasks,
):
    del fast_env
    _save_max_resolution("4k")
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode(
            "http://live",
            fast_start=True,
            is_disconnected=_disconnected,
        )
    assert error.value.status_code == 499
    assert not env.sessions


@pytest.mark.asyncio
async def test_start_passes_the_request_to_the_transcoder(env: _Env, tasks: BackgroundTasks):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "probe_live": False})
    result = await ffmpeg_session.start_transcode(
        "http://live",
        series_id=3,
        episode_id=4,
        deinterlace_fallback=False,
        username="alice",
        source_id="src",
    )
    session = env.sessions[result["session_id"]]
    assert (session.get("series_id"), session.get("episode_id")) == (3, 4)
    assert (session.get("username"), session.get("source_id")) == ("alice", "src")
    (command,) = env.launcher.commands
    # Live streams are deinterlaced unless the viewer turned it off.
    assert "yadif" not in _option(command, "-vf")


@pytest.mark.asyncio
async def test_start_passes_probe_choices_to_the_transcoder(env: _Env, tasks: BackgroundTasks):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "probe_series": True})
    env.run.program("ffprobe", Program(stdout=_DOLBY_PROBE))
    result = await ffmpeg_session.start_transcode(
        "http://ep",
        "series",
        series_id=3,
        episode_id=4,
        series_name="Show",
        audio_passthrough=True,
    )
    assert env.sessions[result["session_id"]].get("audio_passthrough") is True
    (series,) = ffmpeg_command.get_series_probe_cache_stats()
    assert series["name"] == "Show"


# =============================================================================
# _do_start_transcode decisions
# =============================================================================


def _session_id(command: list[str]) -> str:
    """The session id in the name of the output directory a command writes to."""
    return pathlib.Path(command[-1]).parent.name.removeprefix("netv_transcode_")[:36]


@pytest.mark.asyncio
async def test_do_start_records_the_probed_session(
    env: _Env,
    tasks: BackgroundTasks,
    caplog: pytest.LogCaptureFixture,
):
    del tasks
    cache.save_server_settings(
        {**cache.load_server_settings(), "probe_series": True, "max_resolution": "1080p"},
    )
    env.run.program("ffprobe", Program(stdout=_DOLBY_PROBE))
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        result = await ffmpeg_session._do_start_transcode(
            "http://ep",
            "series",
            3,
            4,
            0,
            "Show",
            True,
            "alice",
            "src",
            audio_passthrough=True,
        )
    sid = result["session_id"]
    assert uuid.UUID(sid).version == 4
    session = env.sessions[sid]
    directory = pathlib.Path(session["dir"])
    assert directory.parent == env.root
    assert directory.name.startswith(f"netv_transcode_{sid}_")
    subtitles: list[ffmpeg_session.StoredSubtitle | int] = [
        {"index": 2, "lang": "eng", "name": "English"},
        {"index": 3, "lang": "spa", "name": "Spanish"},
    ]
    assert session == {
        "dir": str(directory),
        "process": env.launcher.processes[0],
        "started": _NOW,
        "url": "http://ep",
        "request_url": "http://ep",
        "is_vod": True,
        "is_archive": False,
        "last_access": _NOW,
        "subtitles": subtitles,
        "duration": 1500,
        "seek_offset": 0,
        "series_id": 3,
        "episode_id": 4,
        "username": "alice",
        "source_id": "src",
        "bandwidth_saver": False,
        "audio_passthrough": True,
        "playback_policy": PlaybackPolicy(),
        "recovery_bitrate": pytest.approx(8_800_000),
        "caption_timestamp_origin": 10.0,
    }
    assert result == {
        "session_id": sid,
        "playlist": f"/transcode/{sid}/stream.m3u8",
        "subtitles": [
            {"url": f"/subs/{sid}/sub0.vtt", "lang": "eng", "label": "English", "default": True},
            {"url": f"/subs/{sid}/sub1.vtt", "lang": "spa", "label": "Spanish", "default": False},
        ],
        "duration": 1500,
        "seek_offset": 0,
        "caption_playlist": f"/transcode/{sid}/captions.m3u8",
    }
    (command,) = env.launcher.commands
    assert _option(command, "-user_agent") == "TiviMate/4.7.0"
    assert "0:3" in command
    assert _option(command, "-c:a") == "copy"
    assert "BANDWIDTH=9440000," in (directory / "captions.m3u8").read_text()
    (series,) = ffmpeg_command.get_series_probe_cache_stats()
    assert (series["series_id"], series["name"]) == (3, "Show")
    assert [episode["episode_id"] for episode in series["episodes"]] == [4]
    assert [r.getMessage() for r in caplog.records] == [
        "Probe: video=h264/yuv420p/720p/1.1Mbps/interlaced audio=ac3/6ch/48000Hz "
        "duration=1500s subs=subrip,subrip,hdmv_pgs_subtitle [extract:eng,spa]",
        f"Starting transcode session {sid} (vod=True): {' '.join(command)}",
    ]


@pytest.mark.asyncio
async def test_do_start_saver_caps_resolution_and_logs_sparse_probe(
    env: _Env,
    tasks: BackgroundTasks,
    caplog: pytest.LogCaptureFixture,
):
    del tasks
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "max_resolution": "480p",
            "probe_live": True,
            "transcode_hw": "software",
        },
    )
    sparse = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "pix_fmt": "yuv420p10le",
                "height": 2160,
            },
            {"codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "44100"},
        ],
    }
    env.run.program("ffprobe", Program(stdout=json.dumps(sparse)))
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        result = await ffmpeg_session._do_start_transcode(
            "http://live",
            "live",
            None,
            None,
            0,
            bandwidth_saver=True,
        )
    session = env.sessions[result["session_id"]]
    assert session.get("playback_policy") == PlaybackPolicy(bandwidth_saver=True)
    assert session.get("bandwidth_saver") is True
    # Recovery still targets the configured full-quality cap.
    assert session.get("recovery_bitrate") == pytest.approx(3_300_000)
    (command,) = env.launcher.commands
    assert "min(ih\\,480)" in _option(command, "-vf")
    assert _option(command, "-crf") == "32"
    assert caplog.records[0].getMessage() == (
        "Probe: video=hevc/yuv420p10le/2160p/? audio=aac/2ch/44100Hz duration=0s subs=none"
    )


@pytest.mark.asyncio
async def test_do_start_defaults(env: _Env, tasks: BackgroundTasks):
    del tasks
    cache.save_server_settings(
        {**cache.load_server_settings(), "probe_movies": False, "transcode_hw": "software"},
    )
    result = await ffmpeg_session._do_start_transcode("http://movie", "movie", None, None, 0)
    session = env.sessions[result["session_id"]]
    assert (session.get("username"), session.get("source_id")) == ("", "")
    assert (session.get("duration"), session.get("audio_passthrough")) == (0.0, False)
    assert session.get("recovery_bitrate") == pytest.approx(8_800_000)
    (command,) = env.launcher.commands
    # Unprobed movies are deinterlaced, capped at 1080p, and encoded at high quality.
    assert "yadif" in _option(command, "-vf")
    assert "min(ih\\,1080)" in _option(command, "-vf")
    assert _option(command, "-crf") == "20"


@pytest.mark.asyncio
async def test_do_start_series_defaults_keep_dolby_from_aac_clients(
    env: _Env,
    tasks: BackgroundTasks,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "probe_series": True})
    env.run.program("ffprobe", Program(stdout=_DOLBY_PROBE))
    result = await ffmpeg_session._do_start_transcode("http://ep", "series", 9, None, 0)
    session = env.sessions[result["session_id"]]
    assert session.get("is_vod") is True
    assert session.get("audio_passthrough") is False
    (command,) = env.launcher.commands
    assert _option(command, "-c:a") == "aac"
    assert _option(command, "-hls_list_size") == "0"
    (series,) = ffmpeg_command.get_series_probe_cache_stats()
    assert (series["name"], [e["episode_id"] for e in series["episodes"]]) == ("", [0])


@pytest.mark.asyncio
async def test_do_start_quality_setting_applies_without_saver(env: _Env, tasks: BackgroundTasks):
    del tasks
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "probe_live": False,
            "quality": "low",
            "transcode_hw": "software",
        },
    )
    result = await ffmpeg_session._do_start_transcode("http://live", "live", None, None, 0)
    assert env.run.calls == []
    assert env.sessions[result["session_id"]].get("is_vod") is False
    (command,) = env.launcher.commands
    assert _option(command, "-crf") == "32"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seek_offset,exit_code,waited,logged_exit",
    [(0, None, 31.1, -1), (0.5, 0, 40.1, -1), (42.0, 3, 40.1, 3)],
)
async def test_failed_start_reports_the_last_stderr_lines(
    env: _Env,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    seek_offset: float,
    exit_code: int | None,
    waited: float,
    logged_exit: int,
):
    cache.save_server_settings({**cache.load_server_settings(), "probe_movies": False})
    stderr = [f"line{i}\n".encode() for i in range(11)]
    launcher = _relaunch(monkeypatch, [FakeProcess(exit_code=exit_code, stderr=stderr)])
    launcher.on_launch = None
    with (
        caplog.at_level(logging.DEBUG, logger="netv.ffmpeg_session"),
        pytest.raises(HTTPException) as error,
    ):
        await ffmpeg_session._do_start_transcode("http://movie", "movie", None, None, seek_offset)
    assert (error.value.status_code, error.value.detail) == (
        500,
        "Transcode failed - check server logs for details",
    )
    assert sum(env.clock.sleeps) == pytest.approx(waited, abs=0.25)
    (command,) = launcher.commands
    sid = _session_id(command)
    assert ("-ss" in command) is (seek_offset > 0)
    messages = [r.getMessage() for r in caplog.records]
    tail = "\n".join(f"line{i}" for i in range(1, 11))
    assert f"ffmpeg:{sid} failed (exit {logged_exit}): {tail}" in messages
    assert f"ffmpeg:{sid} line0" in messages
    applied = f"Applying seek_offset={seek_offset:.1f} from previous session"
    assert (applied in messages) is (seek_offset > 0)
    assert not env.sessions
    assert not list(env.root.iterdir())


@pytest.mark.asyncio
async def test_do_start_needs_two_published_segments(
    env: _Env,
    tasks: BackgroundTasks,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "probe_movies": False})
    env.launcher.on_launch = lambda command: _media_playlist(
        pathlib.Path(command[-1]),
        ["seg000.ts"],
    )
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session._do_start_transcode("http://movie", "movie", None, None, 0)
    assert error.value.status_code == 500


@pytest.mark.asyncio
async def test_do_start_ignores_disconnects_for_ordinary_streams(
    env: _Env,
    tasks: BackgroundTasks,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "probe_movies": False})
    result = await ffmpeg_session._do_start_transcode(
        "http://movie",
        "movie",
        None,
        None,
        0,
        is_disconnected=_disconnected,
    )
    assert set(env.sessions) == {result["session_id"]}


@pytest.mark.asyncio
async def test_do_start_cancelled_while_waiting_releases_a_recent_session(
    env: _Env,
    tasks: BackgroundTasks,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "probe_movies": False})
    env.launcher.on_launch = None
    task = asyncio.create_task(
        ffmpeg_session._do_start_transcode("http://movie", "movie", None, None, 0),
    )
    await until(lambda: env.sessions)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not env.sessions
    assert env.launcher.processes[0].signals == ["TERM"]
    assert not list(env.root.iterdir())


# =============================================================================
# Archives and transcode startup
# =============================================================================


@pytest.fixture
def archive(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Env:
    """An NVENC host with a 1080p AI Upscale engine and probing off."""
    del tasks
    env.root.mkdir()
    engines = tmp_path / "engines"
    engines.mkdir()
    (engines / "2x-nomosuni-compact_1080p_fp16.engine").write_bytes(b"engine")
    monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", str(engines))
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "transcode_hw": "nvenc+software",
            "max_resolution": "4k",
            "quality": "high",
            "sr_model": "upscale-model",
            "probe_live": False,
            "probe_movies": False,
            "probe_series": False,
        },
    )
    return env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,content_type,upscale",
    [
        (ARCHIVE_URL, "movie", False),
        ("https://upstream.test/movie/1.mp4", "movie", True),
        ("https://upstream.test/live/1.ts", "live", True),
    ],
)
async def test_archive_bypasses_upscale_without_changing_live_or_movies(
    archive: _Env,
    url: str,
    content_type: ffmpeg_session.ContentType,
    upscale: bool,
):
    result = await ffmpeg_session.start_transcode(url, content_type)
    [command] = archive.launcher.commands
    video_filter = command[command.index("-vf") + 1]
    assert ("dnn_processing=" in video_filter) is upscale
    assert "2160" in video_filter
    if content_type == "movie":
        assert command[command.index("-qp") + 1] == "20"
    session = archive.sessions[result["session_id"]]
    assert session.get("is_archive") is (not upscale)
    assert session.get("request_url") == url
    assert (pathlib.Path(session["dir"]) / "session.json").exists() is (
        content_type == "movie" and upscale
    )


@pytest.mark.asyncio
async def test_dvr_remux_start_resolves_master_playlist_once(archive: _Env):
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "max_resolution": "1080p",
            "live_dvr_mins": 60,
            "probe_live": True,
            "sr_model": "",
        },
    )
    master = "https://upstream.test/live/1.m3u8"
    variant = "https://upstream.test/live/1_hi.m3u8"
    archive.web.routes[master] = FakeResponse(
        b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=8000000\n1_hi.m3u8\n",
    )
    probe = json.loads(_MOVIE_PROBE)
    probe["streams"][0].pop("field_order")
    archive.run.program("ffprobe", Program(stdout=json.dumps(probe)))
    result = await ffmpeg_session.start_transcode(master, "live", fast_start=True)
    assert archive.web.urls == [master]
    assert [call[-1] for call in archive.run.calls] == [variant]
    [command] = archive.launcher.commands
    assert command[command.index("-i") + 1] == variant
    assert command[command.index("-c:v") + 1] == "copy"
    session = archive.sessions[result["session_id"]]
    assert (session["url"], session.get("request_url")) == (variant, master)


@pytest.mark.asyncio
async def test_archive_stop_releases_recent_session_without_caching(archive: _Env):
    result = await ffmpeg_session.start_transcode(ARCHIVE_URL, "movie")
    directory = pathlib.Path(archive.sessions[result["session_id"]]["dir"])
    await ffmpeg_session.stop_session(result["session_id"])
    assert archive.launcher.processes[0].signals == ["TERM"]
    assert not archive.sessions
    assert ARCHIVE_URL not in archive.urls
    assert not directory.exists()


@pytest.mark.asyncio
async def test_disconnected_archive_does_not_launch_encoder(archive: _Env):
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode(ARCHIVE_URL, "movie", is_disconnected=_disconnected)
    assert (error.value.status_code, error.value.detail) == (499, "Playback request disconnected")
    assert archive.launcher.launches == []
    assert not list(archive.root.iterdir())


@pytest.mark.asyncio
async def test_archive_disconnect_during_playlist_wait_releases_encoder(archive: _Env):
    answers = iter([False, True])

    async def disconnected() -> bool:
        return next(answers)

    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode(ARCHIVE_URL, "movie", is_disconnected=disconnected)
    assert error.value.status_code == 499
    [process] = archive.launcher.processes
    assert process.signals == ["TERM"]
    assert not archive.sessions
    assert not list(archive.root.iterdir())


@pytest.mark.asyncio
async def test_archive_cancelled_startup_releases_encoder(archive: _Env):
    archive.launcher.on_launch = None
    task = asyncio.create_task(ffmpeg_session.start_transcode(ARCHIVE_URL, "movie"))
    await until(lambda: archive.launcher.launches)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [process] = archive.launcher.processes
    assert process.signals == ["TERM"]
    assert not archive.sessions
    assert not list(archive.root.iterdir())


@pytest.mark.asyncio
async def test_archive_rejects_file_offset_seek_without_interrupting_playback(archive: _Env):
    result = await ffmpeg_session.start_transcode(ARCHIVE_URL, "movie")
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.seek_transcode(result["session_id"], 600)
    assert error.value.status_code == 400
    assert archive.launcher.processes[0].returncode is None


@pytest.mark.asyncio
async def test_dead_archive_is_not_resumed_using_byte_seeking(archive: _Env):
    result = await ffmpeg_session.start_transcode(ARCHIVE_URL, "movie")
    archive.launcher.processes[0].kill()
    resumed = await ffmpeg_session._handle_existing_vod_session(
        result["session_id"],
        ARCHIVE_URL,
        "software",
        False,
    )
    assert resumed is None
    assert len(archive.launcher.launches) == 1
    assert not archive.sessions
    assert not archive.urls


@pytest.mark.asyncio
async def test_transcode_response_includes_native_caption_playlist(archive: _Env):
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "transcode_hw": "software",
            "max_resolution": "1080p",
            "probe_movies": True,
        },
    )
    archive.run.program("ffprobe", Program(stdout=_MOVIE_PROBE))
    result = await ffmpeg_session.start_transcode(ARCHIVE_URL, "movie")
    sid = result["session_id"]
    assert result.get("caption_playlist") == f"/transcode/{sid}/captions.m3u8"
    assert result["subtitles"] == [
        {"url": f"/subs/{sid}/sub0.vtt", "lang": "eng", "label": "English", "default": True},
    ]
    assert (result["duration"], result["seek_offset"]) == (120, 0)
    directory = pathlib.Path(archive.sessions[sid]["dir"])
    assert 'URI="sub0.m3u8"' in (directory / "captions.m3u8").read_text()
    placeholder = (directory / "sub0.m3u8").read_text()
    assert "#EXT-X-TARGETDURATION:3" in placeholder
    assert "#EXTINF" not in placeholder


# =============================================================================
# Progress, snapshots, responses
# =============================================================================


def test_adaptive_response_with_captions_and_generation(env: _Env, tmp_path: pathlib.Path):
    env.sessions["s"] = _live(tmp_path)
    env.sessions["s"].update(playlist_generation=3, closed_captions=True, playback_buffer_seconds=8)
    response = ffmpeg_session._adaptive_session_response("s")
    assert response == {
        "session_id": "s",
        "playlist": "/transcode/s/low.m3u8?generation=3",
        "master_playlist": "/transcode/s/master.m3u8",
        "playback_buffer_seconds": 8.0,
        "subtitles": [],
        "duration": 0,
        "seek_offset": 0,
        "caption_playlist": "/transcode/s/low-captions.m3u8?generation=3",
    }
    missing = ffmpeg_session._adaptive_session_response("missing")
    assert missing["playlist"] == "/transcode/missing/low.m3u8"
    assert missing.get("playback_buffer_seconds") == 12.0
    assert "caption_playlist" not in missing


def test_get_existing_session_variants(env: _Env, tmp_path: pathlib.Path):
    assert ffmpeg_session._get_existing_session("http://none") is None
    env.urls["http://dangling"] = "gone"
    assert ffmpeg_session._get_existing_session("http://dangling") is None
    env.sessions["s"] = _live(tmp_path)
    env.sessions["s"]["seek_offset"] = 7
    env.urls["http://live"] = "s"
    existing = ffmpeg_session._get_existing_session("http://live")
    assert existing is not None
    assert (existing.session_id, existing.is_valid) == ("s", True)
    assert existing.session == env.sessions["s"]
    assert existing.session is not env.sessions["s"]
    env.sessions["s"]["last_access"] = _NOW - 60
    existing = ffmpeg_session._get_existing_session("http://live")
    assert existing is not None
    assert existing.is_valid is False


@pytest.mark.asyncio
async def test_start_resumes_at_the_previous_seek_offset_except_for_archives(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "probe_movies": False})
    for url in ("http://movie", ARCHIVE_URL):
        directory = tmp_path / str(len(env.sessions))
        directory.mkdir()
        env.sessions[url] = ffmpeg_session.TranscodeSession(
            dir=str(directory),
            process=FakeProcess(exited=True),
            started=_NOW,
            last_access=_NOW - 60,
            url=url,
            seek_offset=30,
            username="",
        )
        env.urls[url] = url
    await ffmpeg_session.start_transcode("http://movie", "movie")
    await ffmpeg_session.start_transcode(ARCHIVE_URL, "movie")
    movie, archive = env.launcher.commands
    assert _option(movie, "-ss") == "30"
    assert "-ss" not in archive


@pytest.mark.asyncio
async def test_try_reuse_live_session_builds_caption_response(env: _Env, tmp_path: pathlib.Path):
    (tmp_path / "stream.m3u8").write_text("#EXTINF:4.0,\nseg000.ts\n")
    (tmp_path / "seg000.ts").write_bytes(b"x")
    (tmp_path / "captions.m3u8").write_text("#EXTM3U\n")
    subs: list[ffmpeg_session.StoredSubtitle | int] = [
        {"index": 2, "lang": "eng", "name": "English"},
    ]
    env.sessions["s"] = _live(tmp_path)
    env.sessions["s"].update(subtitles=subs, duration=60, seek_offset=5, last_access=0)
    result = await ffmpeg_session._try_reuse_session("s", "http://live", False, "live")
    assert result == {
        "session_id": "s",
        "playlist": "/transcode/s/stream.m3u8",
        "subtitles": [
            {"url": "/subs/s/sub0.vtt", "lang": "eng", "label": "English", "default": True},
        ],
        "duration": 60,
        "seek_offset": 5,
        "transcoded_duration": 4.0,
        "caption_playlist": "/transcode/s/captions.m3u8",
    }
    assert env.sessions["s"].get("last_access") == _NOW
    assert await ffmpeg_session._try_reuse_session("missing", "http://live", False, "live") is None


@pytest.mark.asyncio
async def test_try_reuse_without_captions_omits_caption_playlist(
    env: _Env,
    tmp_path: pathlib.Path,
):
    (tmp_path / "captions.m3u8").write_text("#EXTM3U\n")
    env.sessions["s"] = _live(tmp_path)
    result = await ffmpeg_session._try_reuse_session("s", "http://live", False, "live")
    assert result is not None
    assert "caption_playlist" not in result
    assert (result["subtitles"], result["duration"], result["seek_offset"]) == ([], 0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type,probe_movies,probe_series,probed",
    [
        ("series", False, True, True),
        ("series", True, False, False),
        ("movie", True, False, True),
        ("movie", False, True, False),
    ],
)
async def test_try_reuse_vod_uses_settings(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    content_type: ffmpeg_session.ContentType,
    probe_movies: bool,
    probe_series: bool,
    probed: bool,
):
    del tasks
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "transcode_hw": "nvenc+software",
            "probe_movies": probe_movies,
            "probe_series": probe_series,
            "max_resolution": "4k",
            "quality": "low",
        },
    )
    (tmp_path / "seg000.ts").write_bytes(b"x" * 2000)
    env.sessions["s"] = ffmpeg_session.TranscodeSession(
        dir=str(tmp_path),
        process=FakeProcess(exited=True),
        url="http://ep",
        is_vod=True,
        seek_offset=0,
        started=_NOW,
    )
    assert await ffmpeg_session._try_reuse_session("s", "http://ep", True, content_type) is not None
    (command,) = env.launcher.commands
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-qp") + 1] == "35"
    assert "min(ih\\,2160)" in command[command.index("-vf") + 1]
    assert [call[-1] for call in env.run.calls] == (["http://ep", "http://ep"] if probed else [])


@pytest.mark.asyncio
async def test_try_reuse_vod_defaults_without_quality_settings(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "transcode_hw": "software"})
    (tmp_path / "seg000.ts").write_bytes(b"x" * 2000)
    env.sessions["s"] = ffmpeg_session.TranscodeSession(
        dir=str(tmp_path),
        process=FakeProcess(exited=True),
        url="http://ep",
        is_vod=True,
        started=_NOW,
    )
    assert await ffmpeg_session._try_reuse_session("s", "http://ep", True, "series") is not None
    (command,) = env.launcher.commands
    assert command[command.index("-crf") + 1] == "20"
    assert "min(ih\\,1080)" in command[command.index("-vf") + 1]
    assert env.run.calls == []


# =============================================================================
# Seek
# =============================================================================


@pytest.fixture
def seek_dir(env: _Env, tasks: BackgroundTasks, tmp_path: pathlib.Path) -> pathlib.Path:
    del tasks
    directory = tmp_path / "v"
    directory.mkdir()
    env.sessions["v"] = ffmpeg_session.TranscodeSession(
        dir=str(directory),
        process=FakeProcess(exit_code=None),
        url="http://movie",
        is_vod=True,
        subtitles=[{"index": 2, "lang": "eng", "name": "English"}, 7],
        series_id=None,
        episode_id=None,
        last_access=_NOW,
        started=_NOW,
    )
    return directory


@pytest.mark.asyncio
async def test_seek_unknown_or_live_session_is_404(env: _Env, tmp_path: pathlib.Path):
    env.sessions["live"] = _live(tmp_path)
    for sid in ("missing", "live"):
        with pytest.raises(HTTPException) as error:
            await ffmpeg_session.seek_transcode(sid, 10)
        assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_smart_seek_reuses_existing_segment(env: _Env, seek_dir: pathlib.Path):
    old = env.sessions["v"]["process"]
    (seek_dir / "seg001.ts").write_bytes(b"x" * 2000)
    result = await ffmpeg_session.seek_transcode("v", 4.0)
    assert result == {"session_id": "v", "playlist": "/transcode/v/stream.m3u8"}
    assert env.sessions["v"].get("seek_offset") == 4.0
    assert "#EXT-X-MEDIA-SEQUENCE:1" in (seek_dir / "stream.m3u8").read_text()
    assert old is not None
    assert old.returncode is None
    assert env.launcher.commands == []


@pytest.mark.asyncio
async def test_seek_restarts_ffmpeg_at_target(env: _Env, seek_dir: pathlib.Path):
    old = env.sessions["v"]["process"]
    (seek_dir / "seg000.ts").write_bytes(b"x" * 2000)
    for name in ("seg005.ts", "segxyz.ts", "sub0.vtt", "stream.m3u8"):
        (seek_dir / name).write_bytes(b"x")
    (seek_dir / "session.json").write_text(json.dumps({"seek_offset": 0}))
    result = await ffmpeg_session.seek_transcode("v", 9.0)
    assert result == {"ok": True, "segment": 3, "time": 9.0}
    assert old is not None
    assert old.returncode == -15
    (command,) = env.launcher.commands
    i = command.index("-i")
    assert command[i - 2 : i + 2] == ["-ss", "9.0", "-i", "http://movie"]
    delay = command.index("-max_delay")
    assert command[delay - 2 : delay] == ["-output_ts_offset", "-9.0"]
    assert command[-2:] == ["-start_number", "3"]
    assert command[command.index("-map") + 1] == "0:2"
    assert (seek_dir / "seg000.ts").exists()
    assert (seek_dir / "segxyz.ts").exists()
    assert not (seek_dir / "seg005.ts").exists()
    assert not (seek_dir / "sub0.vtt").exists()
    assert "#EXT-X-TARGETDURATION:3" in (seek_dir / "sub0.m3u8").read_text()
    assert json.loads((seek_dir / "session.json").read_text())["seek_offset"] == 9.0
    assert env.sessions["v"]["process"] is env.launcher.processes[0]
    assert [call[-1] for call in env.run.calls] == ["http://movie", "http://movie"]
    assert env.sessions["v"].get("caption_timestamp_origin") == 10.0


@pytest.mark.asyncio
async def test_seek_timeout_without_probe_tolerates_bad_session_json(
    env: _Env,
    seek_dir: pathlib.Path,
):
    env.sessions["v"].update(series_id=5, subtitles=[])
    (seek_dir / "session.json").write_text("{bad")
    env.launcher.on_launch = None
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.seek_transcode("v", 1.0)
    assert error.value.status_code == 500
    assert env.run.calls == []
    assert "caption_timestamp_origin" not in env.sessions["v"]
    assert (seek_dir / "session.json").read_text() == "{bad"


@pytest.mark.asyncio
async def test_seek_kills_new_process_when_session_disappears(
    env: _Env,
    seek_dir: pathlib.Path,
):
    del seek_dir
    env.launcher.on_launch = lambda command: env.sessions.pop("v")
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.seek_transcode("v", 1.0)
    assert error.value.status_code == 404
    assert env.launcher.processes[0].returncode == -15


@pytest.mark.asyncio
async def test_seek_caption_preparation_failure(env: _Env, seek_dir: pathlib.Path):
    seek_dir.chmod(0o555)
    try:
        with pytest.raises(HTTPException) as error:
            await ffmpeg_session.seek_transcode("v", 1.0)
    finally:
        seek_dir.chmod(0o755)
    assert error.value.status_code == 500
    assert env.launcher.commands == []


# =============================================================================
# Playback health
# =============================================================================


def test_feedback_decision_and_heartbeat(env: _Env, tmp_path: pathlib.Path):
    env.sessions["live"] = _live(tmp_path)
    env.sessions["live"].update(username="viewer", playback_policy=PlaybackPolicy(), last_access=0)
    health = PlaybackHealth(buffer_seconds=0, waiting=True)
    decisions: list[ffmpeg_session.HealthResponse] = []
    for now in range(0, 26, 2):
        env.clock.now = now
        decisions.append(ffmpeg_session.report_playback_health("live", "viewer", health))
    assert decisions[-2] == {"bandwidth_saver": False}
    assert decisions[-1] == {"bandwidth_saver": True}
    assert env.sessions["live"].get("last_access") == 24


def test_feedback_requires_owner_and_live_session(env: _Env, tmp_path: pathlib.Path):
    health = PlaybackHealth(buffer_seconds=0, waiting=True)
    env.sessions["vod"] = _live(tmp_path)
    env.sessions["vod"].update(username="viewer", is_vod=True)
    for session_id, username, status in [
        ("missing", "viewer", 404),
        ("vod", "other", 404),
        ("vod", "viewer", 400),
    ]:
        with pytest.raises(HTTPException) as error:
            ffmpeg_session.report_playback_health(session_id, username, health)
        assert error.value.status_code == status


def test_feedback_remembers_the_full_quality_requirement(env: _Env, tmp_path: pathlib.Path):
    env.sessions["live"] = _live(tmp_path)
    env.sessions["live"].update(username="u", playback_policy=PlaybackPolicy())
    for required in (3e6, 2e6):
        health = PlaybackHealth(buffer_seconds=10, waiting=False, required_bitrate=required)
        assert ffmpeg_session.report_playback_health("live", "u", health) == {
            "bandwidth_saver": False,
        }
    assert env.sessions["live"].get("recovery_bitrate") == 3e6
    env.sessions["live"]["bandwidth_saver"] = True
    health = PlaybackHealth(buffer_seconds=10, waiting=False, required_bitrate=9e6)
    ffmpeg_session.report_playback_health("live", "u", health)
    assert env.sessions["live"].get("recovery_bitrate") == 3e6


def _low_playlist(directory: pathlib.Path, segment_bytes: int) -> None:
    """Two segments whose bitrate is segment_bytes * 8 / 0.002."""
    for name in ("low_0.ts", "low_1.ts"):
        (directory / name).write_bytes(b"x" * segment_bytes)
    (directory / "low.m3u8").write_text("#EXTINF:0.002,\nlow_0.ts\n#EXTINF:0.002,\nlow_1.ts\n")


def test_fast_start_health_learns_bitrate_from_live_encoder(env: _Env, tmp_path: pathlib.Path):
    encoder = FakeProcess(exit_code=None)
    env.sessions["f"] = ffmpeg_session.TranscodeSession(
        username="u",
        dir=str(tmp_path),
        fast_start=True,
        playback_policy=PlaybackPolicy(),
        process=encoder,
        closed_captions=True,
        playlist_generation=2,
        recovery_bitrate=1e6,
        started=_NOW,
        url="http://stream.test/live",
    )
    health = PlaybackHealth(buffer_seconds=10, waiting=False, observed_bitrate=1e8)
    _low_playlist(tmp_path, 1250)
    assert ffmpeg_session.report_playback_health("f", "u", health) == {
        "bandwidth_saver": False,
        "playlist": "/transcode/f/low.m3u8?generation=2",
        "caption_playlist": "/transcode/f/low-captions.m3u8?generation=2",
    }
    assert env.sessions["f"].get("recovery_bitrate") == 5e6
    encoder.returncode = 1
    _low_playlist(tmp_path, 2250)
    ffmpeg_session.report_playback_health("f", "u", health)
    assert env.sessions["f"].get("recovery_bitrate") == 5e6


def test_health_without_playback_policy_raises_key_error(env: _Env, tmp_path: pathlib.Path):
    env.sessions["f"] = _live(tmp_path)
    env.sessions["f"]["username"] = "u"
    with pytest.raises(KeyError, match="playback_policy"):
        ffmpeg_session.report_playback_health(
            "f",
            "u",
            PlaybackHealth(buffer_seconds=1, waiting=False),
        )


def _fast_health_session(
    tmp_path: pathlib.Path,
    *,
    bandwidth_saver: bool,
    recovery_bitrate: float = 0,
) -> ffmpeg_session.TranscodeSession:
    return ffmpeg_session.TranscodeSession(
        username="u",
        dir=str(tmp_path),
        fast_start=True,
        bandwidth_saver=bandwidth_saver,
        playback_policy=PlaybackPolicy(bandwidth_saver=bandwidth_saver),
        process=FakeProcess(exited=True),
        closed_captions=True,
        recovery_bitrate=recovery_bitrate,
        started=_NOW,
        url="http://stream.test/live",
    )


def test_congestion_on_full_quality_session_omits_playlist_so_clients_retune(
    env: _Env,
    tmp_path: pathlib.Path,
):
    env.sessions["f"] = _fast_health_session(tmp_path, bandwidth_saver=False)
    stalled = PlaybackHealth(buffer_seconds=0, waiting=True)
    results: list[ffmpeg_session.HealthResponse] = []
    for now in range(0, 26, 2):
        env.clock.now = _NOW + now
        results.append(ffmpeg_session.report_playback_health("f", "u", stalled))
    assert results[0] == {
        "bandwidth_saver": False,
        "playlist": "/transcode/f/low.m3u8",
        "caption_playlist": "/transcode/f/low-captions.m3u8",
    }
    assert results[-1] == {"bandwidth_saver": True}


def test_recovery_on_saver_session_omits_playlist_so_clients_retune(
    env: _Env,
    tmp_path: pathlib.Path,
):
    env.sessions["f"] = _fast_health_session(tmp_path, bandwidth_saver=True, recovery_bitrate=1000)
    good = PlaybackHealth(buffer_seconds=12, waiting=False, observed_bitrate=100_000)
    results: list[ffmpeg_session.HealthResponse] = []
    for now in range(0, 62, 2):
        env.clock.now = _NOW + now
        results.append(ffmpeg_session.report_playback_health("f", "u", good))
    assert results[0] == {
        "bandwidth_saver": True,
        "playlist": "/transcode/f/low.m3u8",
        "caption_playlist": "/transcode/f/low-captions.m3u8",
    }
    assert results[-1] == {"bandwidth_saver": False}


# =============================================================================
# Fast-start startup
# =============================================================================


def _startup(tmp_path: pathlib.Path) -> ffmpeg_session._FastLiveStartup:
    return ffmpeg_session._FastLiveStartup(
        session_id="f",
        directory=str(tmp_path),
        processes=[],
        started=_NOW,
        poll_interval_sec=0.2,
        is_disconnected=_connected,
    )


@pytest.mark.asyncio
async def test_startup_launch_tracks_process(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    startup = _startup(tmp_path)
    process = await startup.launch(["ffmpeg"], "ingest")
    assert startup.processes == [process]
    assert env.launcher.processes == [process]
    assert _coroutine_names(tasks) == ["_monitor_fast_process"]


@pytest.mark.asyncio
async def test_startup_wait_ready_paths(env: _Env, tmp_path: pathlib.Path):
    startup = _startup(tmp_path)
    _media_playlist(tmp_path / "low.m3u8", ["low_0.ts", "low_1.ts"], duration=5.0)
    await startup.wait_ready("low.m3u8", _proc(FakeProcess(exit_code=None)), 10)
    assert env.clock.sleeps == []

    with pytest.raises(HTTPException) as error:
        await startup.wait_ready("low.m3u8", _proc(FakeProcess(exit_code=None)), 10.5)
    assert error.value.status_code == 500
    assert sum(env.clock.sleeps) == pytest.approx(30, abs=0.25)

    with pytest.raises(HTTPException) as error:
        await startup.wait_ready("low.m3u8", _proc(FakeProcess(exited=True)))
    assert error.value.status_code == 500

    startup = replace(startup, is_disconnected=_disconnected)
    with pytest.raises(HTTPException) as error:
        await startup.wait_ready("low.m3u8", _proc(FakeProcess(exit_code=None)))
    assert error.value.status_code == 499


@pytest.fixture
def fast_env(env: _Env, tasks: BackgroundTasks) -> BackgroundTasks:
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "transcode_hw": "software",
            "max_resolution": "4k",
            "quality": "high",
        },
    )
    env.run.program("ffprobe", Program(stdout='{"streams": []}'), when="a:0")
    env.run.program(
        "ffprobe",
        Program(stdout='{"streams": [{"codec_name": "h264"}]}'),
        when="stream=codec_name",
    )
    env.run.program(
        "ffprobe",
        Program(stdout='{"frames": [{"side_data_list": [{"side_data_type": "Closed Captions"}]}]}'),
        when="frame=side_data_list",
    )
    return tasks


def _option(command: Sequence[str], flag: str) -> str:
    return command[command.index(flag) + 1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bandwidth_saver,resolution,bitrate,height",
    [(False, "4k", "16000000", "2160"), (True, "720p", "4000000", "720")],
)
async def test_start_fast_live_runs_ingest_and_one_encoder(
    env: _Env,
    fast_env: BackgroundTasks,
    bandwidth_saver: bool,
    resolution: str,
    bitrate: str,
    height: str,
):
    result = await ffmpeg_session._start_fast_live(
        "http://live",
        "alice",
        "src",
        True,
        bandwidth_saver=bandwidth_saver,
    )
    sid = result["session_id"]
    session = env.sessions[sid]
    assert ffmpeg_session._is_fast_live(session)
    ingest_cmd, low_cmd = env.launcher.commands
    ingest, low = env.launcher.processes
    assert _option(ingest_cmd, "-i") == "http://live"
    assert "-hls_start_number_source" not in ingest_cmd
    assert "-hls_start_number_source" in session["ingest_restart_command"]
    assert _option(low_cmd, "-live_start_index") == "0"
    assert _option(low_cmd, "-b:v") == bitrate
    assert f"min(ih\\,{height})" in _option(low_cmd, "-vf")
    restart = session.get("low_restart_command")
    assert restart is not None
    assert _option(restart, "-live_start_index") == "-3"
    assert session["process"] is low
    assert session.get("extra_processes") == [ingest]
    assert session.get("watchdog_task") is fast_env.tasks[-1]
    assert _coroutine_names(fast_env)[-1] == "_watch_fast_live"
    assert session.get("closed_captions") is True
    assert session.get("origin_pts") == 1.5
    assert session.get("watchdog_stale_after") == 45.0
    assert session.get("master_resolution") == resolution
    assert session.get("recovery_bitrate") == pytest.approx(22_000_000)
    directory = pathlib.Path(session["dir"])
    assert (directory / "master.m3u8").read_text() == master_playlist(
        resolution,
        audio_bitrate=384_000,
    )
    assert "low.m3u8" in (directory / "low-captions.m3u8").read_text()
    assert result.get("caption_playlist") == f"/transcode/{sid}/low-captions.m3u8"


@pytest.mark.asyncio
async def test_start_fast_live_cancels_watchdog_for_replaced_session(
    env: _Env,
    fast_env: BackgroundTasks,
):
    env.launcher.on_launch = _input_only

    async def replace_while_encoding() -> None:
        # The stale-after budget is the last thing startup records before it
        # waits for the encoder, so the replacement lands during that wait.
        while not any("watchdog_stale_after" in s for s in env.sessions.values()):
            await asyncio.sleep(0)
        (sid,) = env.sessions
        env.sessions[sid] = env.sessions[sid].copy()
        _ffmpeg_output(env.launcher.commands[1])

    replacer = asyncio.create_task(replace_while_encoding())
    result = await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    await replacer
    assert result["playlist"].endswith("/low.m3u8")
    watchdog = fast_env.tasks[_coroutine_names(fast_env).index("_watch_fast_live")]
    assert watchdog.cancelled()
    assert "watchdog_task" not in env.sessions[result["session_id"]]


@pytest.mark.asyncio
async def test_start_fast_live_failure_releases_everything(env: _Env, fast_env: BackgroundTasks):
    del fast_env

    def fail_encoder(command: list[str]) -> None:
        if command[-1].endswith("low.m3u8"):
            raise RuntimeError("boom")
        _ffmpeg_output(command)

    env.launcher.on_launch = fail_encoder
    with pytest.raises(RuntimeError):
        await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    (ingest,) = env.launcher.processes
    assert ingest.returncode == -15
    assert not env.sessions
    assert not list(env.root.iterdir())


@pytest.mark.asyncio
async def test_start_fast_live_aborts_when_session_stopped_during_probe(
    env: _Env,
    fast_env: BackgroundTasks,
):
    del fast_env

    async def stop_once_registered() -> None:
        while not env.sessions:
            await asyncio.sleep(0)
        env.sessions.clear()

    stopper = asyncio.create_task(stop_once_registered())
    with pytest.raises(KeyError):
        await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    await stopper
    ((ingest_cmd),) = env.launcher.commands
    assert ingest_cmd[-1].endswith("input.m3u8")
    assert env.launcher.processes[0].returncode == -15
    assert not env.sessions
    assert not list(env.root.iterdir())


def test_startup_rejects_a_replaced_session_with_its_identifier(
    env: _Env,
    tmp_path: pathlib.Path,
):
    session = _live(tmp_path)
    env.sessions["replaced"] = _live(tmp_path)
    with pytest.raises(KeyError) as error:
        ffmpeg_session._require_current("replaced", session)
    assert error.value.args == ("replaced",)


# =============================================================================
# Concurrent fast-start requests for one URL
# =============================================================================


_FAST_URL = "https://upstream.example/live/1.m3u8"


@pytest.fixture
def startups(env: _Env, tasks: BackgroundTasks) -> ProcessLauncher:
    """The encoder publishes at once; each ingest waits for the test to release it."""
    del tasks
    cache.save_server_settings(
        {**cache.load_server_settings(), "max_resolution": "1080p", "live_dvr_mins": 0},
    )
    # Fake segments carry no media, so ffprobe finds no streams.
    env.run.program("ffprobe", Program(stdout="{}"))
    env.launcher.on_launch = _encoder_only
    return env.launcher


def _start(user_max_streams: int = 0) -> asyncio.Task[ffmpeg_session.TranscodeResponse]:
    return asyncio.create_task(
        ffmpeg_session.start_transcode(
            _FAST_URL,
            fast_start=True,
            username="viewer",
            user_max_streams=user_max_streams,
        ),
    )


def _ingests(launcher: ProcessLauncher) -> int:
    return sum(command[-1].endswith("/input.m3u8") for command in launcher.commands)


async def _yield_to_startups() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_second_request_waits_for_startup_then_reuses_it(startups: ProcessLauncher):
    first = _start()
    await until(lambda: startups.launches)
    second = _start()
    await _yield_to_startups()
    assert not second.done(), "An unfinished startup must not be handed to another request"
    _ffmpeg_output(startups.commands[0])
    first_result, second_result = await asyncio.gather(first, second)
    assert second_result["session_id"] == first_result["session_id"]
    assert _ingests(startups) == 1


@pytest.mark.asyncio
async def test_failed_startup_is_not_handed_to_a_waiting_request(
    env: _Env,
    startups: ProcessLauncher,
):
    first = _start()
    await until(lambda: startups.launches)
    second = _start()
    await _yield_to_startups()
    startups.processes[0].kill()
    await until(lambda: len(startups.launches) == 2)
    _ffmpeg_output(startups.commands[1])
    first_result, second_result = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(first_result, HTTPException)
    assert first_result.status_code == 500
    assert isinstance(second_result, dict)
    assert set(env.sessions) == {second_result["session_id"]}
    assert _ingests(startups) == 2


@pytest.mark.asyncio
async def test_concurrent_starts_open_one_upstream_within_stream_limits(
    env: _Env,
    startups: ProcessLauncher,
):
    requests = [_start(user_max_streams=1), _start(user_max_streams=1)]
    await until(lambda: startups.launches)
    _ffmpeg_output(startups.commands[0])
    results = await asyncio.gather(*requests)
    assert _ingests(startups) == 1
    assert {result["session_id"] for result in results} == set(env.sessions)
    assert not ffmpeg_session._startup_guards


# =============================================================================
# Process helpers and monitors
# =============================================================================


def test_dead_process_signals_are_noops():
    process = ffmpeg_session._DeadProcess()
    process.terminate()
    process.kill()
    assert process.returncode == -1
    assert asyncio.run(process.wait()) == -1


@pytest.mark.asyncio
async def test_shared_ffmpeg_launcher(env: _Env):
    cmd = ["ffmpeg", "-i", "https://provider/live.ts", "/tmp/stream.m3u8"]
    env.launcher.on_launch = None
    assert await ffmpeg_session._launch_ffmpeg(cmd) is env.launcher.processes[0]
    [launch] = env.launcher.launches
    assert launch.command == cmd
    assert launch.options == {
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.DEVNULL,
        "stderr": asyncio.subprocess.PIPE,
    }


@pytest.mark.asyncio
async def test_monitor_stderr_collects_lines(caplog: pytest.LogCaptureFixture):
    process = FakeProcess(stderr=[b"frame=1\n", b"Fatal error\n", b"Aborting now\n"])
    lines: list[str] = []
    with caplog.at_level(logging.DEBUG, logger="netv.ffmpeg_session"):
        await ffmpeg_session._monitor_ffmpeg_stderr(_proc(process), "s", lines)
    assert lines == ["frame=1", "Fatal error", "Aborting now"]
    assert [
        (r.levelno, r.getMessage()) for r in caplog.records if "ffmpeg:s" in r.getMessage()
    ] == [
        (logging.DEBUG, "ffmpeg:s frame=1"),
        (logging.WARNING, "ffmpeg:s Fatal error"),
        (logging.WARNING, "ffmpeg:s Aborting now"),
    ]
    process.stderr = None
    with pytest.raises(ValueError):
        await ffmpeg_session._monitor_ffmpeg_stderr(_proc(process), "s")


@pytest.mark.asyncio
async def test_monitor_resume_invalidates_quick_failure(env: _Env, tmp_path: pathlib.Path):
    env.sessions["s"] = _live(tmp_path)
    env.urls["http://m"] = "s"
    await ffmpeg_session._monitor_resume_ffmpeg(_proc(FakeProcess(exit_code=1)), "s", "http://m")
    assert not env.sessions
    assert not env.urls
    assert not tmp_path.exists()


@pytest.mark.asyncio
async def test_monitor_resume_keeps_session_after_clean_or_slow_exit(
    env: _Env,
    tmp_path: pathlib.Path,
):
    env.sessions["s"] = _live(tmp_path)
    env.urls["http://m"] = "s"
    await ffmpeg_session._monitor_resume_ffmpeg(_proc(FakeProcess(exit_code=0)), "s", "http://m")
    await ffmpeg_session._monitor_resume_ffmpeg(
        _proc(FakeProcess(exit_code=1)),
        "s",
        "http://m",
        quick_failure_threshold_sec=0,
    )
    assert set(env.sessions) == {"s"}
    assert env.urls == {"http://m": "s"}


@pytest.mark.asyncio
async def test_monitor_seek_logs_failure(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await ffmpeg_session._monitor_seek_ffmpeg(_proc(FakeProcess(exit_code=0)), "s")
        await ffmpeg_session._monitor_seek_ffmpeg(_proc(FakeProcess(exit_code=2)), "s")
    assert [r.getMessage() for r in caplog.records] == [
        "Seek ffmpeg exited with code 2 for session s",
    ]


@pytest.mark.asyncio
async def test_spawn_background_task_tracks_until_done():
    async def work() -> None:
        return None

    task = ffmpeg_session._spawn_background_task(work())
    assert task in ffmpeg_session._background_tasks
    await task
    await asyncio.sleep(0)
    assert task not in ffmpeg_session._background_tasks


@pytest.mark.asyncio
async def test_wait_for_playlist_paths(env: _Env, tmp_path: pathlib.Path):
    playlist_path = tmp_path / "stream.m3u8"
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session._wait_for_playlist(
            playlist_path,
            FakeProcess(exit_code=None),
            is_disconnected=_disconnected,
        )
    assert error.value.status_code == 499
    assert await ffmpeg_session._wait_for_playlist(playlist_path, FakeProcess(exited=True)) is False
    playlist_path.write_text("#EXTINF:3,\nseg000.ts\n")
    assert (
        await ffmpeg_session._wait_for_playlist(
            playlist_path,
            FakeProcess(exit_code=None),
            min_segments=2,
            timeout_sec=2,
        )
        is False
    )
    assert sum(env.clock.sleeps) == pytest.approx(2, abs=0.25)
    playlist_path.write_text("#EXTINF:3,\nseg000.ts\n#EXTINF:3,\nseg001.ts\n")
    (tmp_path / "seg000.ts").write_bytes(b"x" * 1000)
    (tmp_path / "seg001.ts").write_bytes(b"x")
    assert (
        await ffmpeg_session._wait_for_playlist(
            playlist_path,
            FakeProcess(exit_code=None),
            min_segments=2,
            timeout_sec=1,
        )
        is False
    )
    (tmp_path / "seg000.ts").write_bytes(b"x" * 1001)
    assert (
        await ffmpeg_session._wait_for_playlist(
            playlist_path,
            FakeProcess(exit_code=None),
            min_segments=2,
        )
        is True
    )


# =============================================================================
# Existing VOD sessions
# =============================================================================


@pytest.fixture
def vod_dir(env: _Env, tasks: BackgroundTasks, tmp_path: pathlib.Path) -> pathlib.Path:
    del tasks
    directory = tmp_path / "v"
    directory.mkdir()
    (directory / "seg000.ts").write_bytes(b"x" * 2000)
    (directory / "seg001.ts").write_bytes(b"x" * 2000)
    (directory / "stream.m3u8").write_text("#EXTINF:3.0,\nseg000.ts\n#EXTINF:3.0,\nseg001.ts\n")
    env.sessions["v"] = ffmpeg_session.TranscodeSession(
        dir=str(directory),
        process=FakeProcess(exited=True),
        url="http://movie",
        is_vod=True,
        seek_offset=0,
        duration=100,
        started=_NOW,
    )
    return directory


@pytest.mark.asyncio
async def test_existing_vod_missing_session():
    assert (
        await ffmpeg_session._handle_existing_vod_session("missing", "http://m", "software", False)
        is None
    )


@pytest.mark.asyncio
async def test_existing_vod_active_session_waits_for_playlist(env: _Env, vod_dir: pathlib.Path):
    env.sessions["v"]["process"] = FakeProcess(exit_code=None)
    (vod_dir / "stream.m3u8").unlink()
    result = await ffmpeg_session._handle_existing_vod_session(
        "v",
        "http://movie",
        "software",
        False,
    )
    assert result is not None
    assert result.get("transcoded_duration") == 6.0
    assert sum(env.clock.sleeps) == pytest.approx(15, abs=0.25)
    assert env.launcher.commands == []


@pytest.mark.asyncio
async def test_existing_vod_without_segments_is_released(env: _Env, vod_dir: pathlib.Path):
    for segment in vod_dir.glob("seg*.ts"):
        segment.unlink()
    env.urls["http://movie"] = "v"
    assert (
        await ffmpeg_session._handle_existing_vod_session("v", "http://movie", "software", False)
        is None
    )
    assert not env.sessions
    assert not env.urls
    assert env.launcher.commands == []


@pytest.mark.asyncio
async def test_existing_vod_with_seek_offset_returns_cache(env: _Env, vod_dir: pathlib.Path):
    del vod_dir
    env.sessions["v"]["seek_offset"] = 30
    result = await ffmpeg_session._handle_existing_vod_session(
        "v",
        "http://movie",
        "software",
        False,
    )
    assert result is not None
    assert result["seek_offset"] == 30
    assert env.clock.sleeps == []
    assert env.launcher.commands == []


@pytest.mark.asyncio
async def test_existing_vod_resume_appends_to_playlist(env: _Env, vod_dir: pathlib.Path):
    del vod_dir
    result = await ffmpeg_session._handle_existing_vod_session(
        "v",
        "http://movie",
        "software",
        True,
    )
    assert result is not None
    (command,) = env.launcher.commands
    i = command.index("-i")
    assert command[i - 2 : i + 2] == ["-ss", "6.0", "-i", "http://movie"]
    assert _option(command, "-hls_flags") == "independent_segments+append_list"
    assert command[-2:] == ["-start_number", "2"]
    assert env.sessions["v"]["process"] is env.launcher.processes[0]
    assert [call[-1] for call in env.run.calls] == ["http://movie", "http://movie"]
    assert env.clock.sleeps == []


@pytest.mark.asyncio
async def test_existing_vod_resume_polls_for_next_segment(env: _Env, vod_dir: pathlib.Path):
    del vod_dir
    env.launcher.on_launch = None
    result = await ffmpeg_session._handle_existing_vod_session(
        "v",
        "http://movie",
        "software",
        False,
    )
    assert result is not None
    assert env.run.calls == []
    assert sum(env.clock.sleeps) == pytest.approx(5, abs=0.25)


@pytest.mark.asyncio
async def test_existing_vod_resume_dies_immediately(
    monkeypatch: pytest.MonkeyPatch,
    vod_dir: pathlib.Path,
):
    del vod_dir
    launcher = _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    launcher.on_launch = None
    assert (
        await ffmpeg_session._handle_existing_vod_session("v", "http://movie", "software", False)
        is None
    )


@pytest.mark.asyncio
async def test_existing_vod_resume_session_vanishes(env: _Env, vod_dir: pathlib.Path):
    del vod_dir
    env.launcher.on_launch = lambda command: env.sessions.pop("v")
    assert (
        await ffmpeg_session._handle_existing_vod_session("v", "http://movie", "software", False)
        is None
    )
    assert env.launcher.processes[0].returncode == -15


# =============================================================================
# _do_start_transcode
# =============================================================================


@pytest.fixture
def do_env(env: _Env, tasks: BackgroundTasks) -> None:
    del tasks
    cache.save_server_settings(
        {**cache.load_server_settings(), "max_resolution": "4k", "probe_live": True},
    )
    env.run.program("ffprobe", Program(stdout=_MOVIE_PROBE))


@pytest.mark.asyncio
@pytest.mark.usefixtures("do_env")
async def test_do_start_vod_with_seek_offset_persists_probe(env: _Env):
    result = await ffmpeg_session._do_start_transcode(
        "http://movie",
        "movie",
        1,
        2,
        42.0,
        username="alice",
        source_id="src",
    )
    (command,) = env.launcher.commands
    i = command.index("-i")
    assert command[i - 2 : i + 2] == ["-ss", "42.0", "-i", "http://movie"]
    assert "min(ih\\,2160)" in _option(command, "-vf")
    assert "yadif" in _option(command, "-vf")
    assert _option(command, "-crf") == "20"
    session = env.sessions[result["session_id"]]
    assert env.urls == {"http://movie": result["session_id"]}
    assert (session.get("username"), session.get("source_id")) == ("alice", "src")
    assert (session.get("series_id"), session.get("episode_id")) == (1, 2)
    assert (session.get("duration"), session.get("seek_offset")) == (120, 42.0)
    info = json.loads((pathlib.Path(session["dir"]) / "session.json").read_text())
    assert info == {
        "session_id": result["session_id"],
        "url": "http://movie",
        "is_vod": True,
        "started": _NOW,
        "subtitles": [{"index": 2, "lang": "eng", "name": "English"}],
        "duration": 120,
        "seek_offset": 42.0,
        "series_id": 1,
        "episode_id": 2,
        "username": "alice",
        "source_id": "src",
        "audio_passthrough": False,
        "probe": {
            "video_codec": "h264",
            "audio_codec": "aac",
            "pix_fmt": "yuv420p",
            "audio_channels": 2,
            "audio_sample_rate": 48_000,
            "subtitle_codecs": ["subrip"],
            "height": 1080,
            "video_bitrate": 8_000_000,
            "interlaced": True,
        },
    }
    assert result.get("caption_playlist", "").endswith("/captions.m3u8")
    assert session.get("caption_timestamp_origin") == 10.0


@pytest.mark.asyncio
@pytest.mark.usefixtures("do_env")
async def test_do_start_bandwidth_saver_lowers_quality(env: _Env):
    await ffmpeg_session._do_start_transcode(
        "http://live",
        "live",
        None,
        None,
        0,
        bandwidth_saver=True,
    )
    (command,) = env.launcher.commands
    assert "min(ih\\,720)" in _option(command, "-vf")
    assert _option(command, "-crf") == "32"


@pytest.mark.asyncio
@pytest.mark.usefixtures("do_env")
async def test_do_start_launch_failure_removes_output(env: _Env, monkeypatch: pytest.MonkeyPatch):
    _relaunch(monkeypatch, [], error=FileNotFoundError("ffmpeg"))
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session._do_start_transcode("http://movie", "movie", None, None, 0)
    assert (error.value.status_code, error.value.detail) == (500, "Could not start the transcoder")
    assert not list(env.root.iterdir())
    assert not env.sessions


@pytest.mark.asyncio
@pytest.mark.usefixtures("do_env")
async def test_do_start_cancelled_mid_launch_removes_output(env: _Env):
    gate = asyncio.Event()

    async def blocked(command: list[str]) -> None:
        del command
        await gate.wait()

    env.launcher.on_launch = blocked
    task = asyncio.create_task(
        ffmpeg_session._do_start_transcode("http://movie", "movie", None, None, 0),
    )
    while not env.root.exists() or not list(env.root.iterdir()):
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list(env.root.iterdir())
    assert not env.sessions
    assert env.launcher.processes[0].returncode == -15


@pytest.mark.asyncio
@pytest.mark.usefixtures("do_env")
async def test_do_start_reports_failed_startup(
    env: _Env,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    launcher = _relaunch(monkeypatch, [FakeProcess(exit_code=1)])
    launcher.on_launch = None
    with (
        caplog.at_level(logging.ERROR, logger="netv.ffmpeg_session"),
        pytest.raises(HTTPException) as error,
    ):
        await ffmpeg_session._do_start_transcode("http://movie", "movie", None, None, 42.0)
    assert error.value.status_code == 500
    assert "failed (exit 1): unknown" in caplog.text
    # A resumed start waits out the longer seek deadline before giving up.
    assert sum(env.clock.sleeps) == pytest.approx(40.1, abs=0.25)
    assert not env.sessions
    assert not list(env.root.iterdir())


# =============================================================================
# Fast-start recovery helpers
# =============================================================================


def test_playlist_marker_and_segments(tmp_path: pathlib.Path):
    path = tmp_path / "low.m3u8"
    assert ffmpeg_session._playlist_marker(path) is None
    assert ffmpeg_session._playlist_segments(path) == set()
    path.write_text("#EXTM3U\n\nlow_1.ts\n../escape.ts\n#EXTINF:2,\nlow_2.ts\n")
    stat = path.stat()
    assert ffmpeg_session._playlist_marker(path) == (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    assert ffmpeg_session._playlist_segments(path) == {"low_1.ts", "low_2.ts"}


def test_prepare_encoder_restart():
    command = ["ffmpeg", "-i", "x", "out.m3u8"]
    assert ffmpeg_session._prepare_encoder_restart(command, {"a.ts"}) is command
    numbered = ["ffmpeg", "-hls_start_number_source", "x", "out.m3u8"]
    assert ffmpeg_session._prepare_encoder_restart(numbered, set()) is numbered
    assert ffmpeg_session._prepare_encoder_restart(command, set()) == [
        "ffmpeg",
        "-i",
        "x",
        "-hls_start_number_source",
        "epoch_us",
        "out.m3u8",
    ]
    assert command == ["ffmpeg", "-i", "x", "out.m3u8"]


@pytest.mark.asyncio
async def test_wait_for_fresh_fast_playlist(env: _Env, tmp_path: pathlib.Path):
    path = tmp_path / "low.m3u8"
    _media_playlist(path, ["low_1.ts", "low_2.ts"])
    marker = ffmpeg_session._playlist_marker(path)
    assert not await ffmpeg_session._wait_for_fresh_fast_playlist(
        str(tmp_path),
        "low.m3u8",
        FakeProcess(exited=True),
        None,
    )
    assert not await ffmpeg_session._wait_for_fresh_fast_playlist(
        str(tmp_path),
        "low.m3u8",
        FakeProcess(exit_code=None),
        marker,
        timeout_sec=2,
    )
    assert sum(env.clock.sleeps) == pytest.approx(2, abs=0.25)
    assert not await ffmpeg_session._wait_for_fresh_fast_playlist(
        str(tmp_path),
        "low.m3u8",
        FakeProcess(exit_code=None),
        None,
        minimum_duration=6.5,
        timeout_sec=1,
    )
    assert await ffmpeg_session._wait_for_fresh_fast_playlist(
        str(tmp_path),
        "low.m3u8",
        FakeProcess(exit_code=None),
        None,
        previous_segments={"old.ts"},
        minimum_duration=6,
    )


@pytest.mark.asyncio
async def test_fresh_playlist_waits_for_two_new_segments(tmp_path: pathlib.Path):
    playlist(tmp_path, "low")
    path = tmp_path / "low.m3u8"
    marker = ffmpeg_session._playlist_marker(path)
    previous_segments = ffmpeg_session._playlist_segments(path)

    def publish_segment(number: int) -> None:
        filename = f"low_{number}.ts"
        (tmp_path / filename).write_bytes(video_packet(20 + number * 2) * 10)
        with path.open("a") as playlist_file:
            playlist_file.write(f"#EXTINF:2,\n{filename}\n")

    waiter = asyncio.create_task(
        ffmpeg_session._wait_for_fresh_fast_playlist(
            str(tmp_path),
            "low.m3u8",
            FakeProcess(exit_code=None),
            marker,
            previous_segments=previous_segments,
        ),
    )
    publish_segment(3)
    await _yield_to_startups()
    assert not waiter.done(), "One new segment must not satisfy the restart"
    publish_segment(4)
    assert await waiter


@pytest.mark.asyncio
async def test_terminate_process_simple_paths():
    dead = FakeProcess(exited=True)
    await ffmpeg_session._terminate_process(dead)
    assert (dead.returncode, dead.signals) == (0, [])
    await ffmpeg_session._terminate_process(None)
    vanished = FakeProcess(exit_code=None, reaped=True)
    await ffmpeg_session._terminate_process(vanished)
    assert vanished.returncode is None
    graceful = FakeProcess(exit_code=None)
    await ffmpeg_session._terminate_process(graceful)
    assert (graceful.returncode, graceful.signals) == (-15, ["TERM"])


@pytest.mark.asyncio
async def test_terminate_process_escalates_after_timeout(env: _Env):
    process = FakeProcess(exit_code=None, ignores_terminate=True)
    await ffmpeg_session._terminate_process(process)
    assert (process.returncode, process.signals) == (-9, ["TERM", "KILL"])
    assert env.clock.sleeps == [5]


@pytest.mark.asyncio
async def test_terminate_process_escalates_when_cancelled():
    process = FakeProcess(exit_code=None, ignores_terminate=True)
    task = asyncio.create_task(ffmpeg_session._terminate_process(process))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (process.returncode, process.signals) == (-9, ["TERM", "KILL"])


@pytest.mark.asyncio
async def test_launch_ffmpeg_terminates_a_process_launched_after_cancellation(env: _Env):
    gate = asyncio.Event()

    async def blocked(command: list[str]) -> None:
        del command
        await gate.wait()

    env.launcher.on_launch = blocked
    task = asyncio.create_task(ffmpeg_session._launch_ffmpeg(["ffmpeg"]))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    (late,) = env.launcher.processes
    assert late.returncode == -15


@pytest.mark.asyncio
async def test_monitor_fast_process(
    env: _Env,
    caplog: pytest.LogCaptureFixture,
    tmp_path: pathlib.Path,
):
    process = FakeProcess(stderr=[b"open http://user:secret@host/x failed\n"], exit_code=1)
    ingest = FakeProcess(exit_code=3)
    env.sessions["f"] = ffmpeg_session.TranscodeSession(
        process=process,
        ingest_process=ingest,
        dir=str(tmp_path),
        started=_NOW,
        url="http://stream.test/live",
    )
    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await ffmpeg_session._monitor_fast_process(_proc(process), "720p", "f")
        await ffmpeg_session._monitor_fast_process(_proc(ingest), "ingest", "f")
        await ffmpeg_session._monitor_fast_process(_proc(FakeProcess(exit_code=1)), "720p", "f")
        await ffmpeg_session._monitor_fast_process(_proc(FakeProcess(exit_code=1)), "720p", "x")
    warnings = [r.getMessage() for r in caplog.records]
    assert warnings == [
        "Fast-start 720p exited for f (code 1): open http://user:secret@host/x failed",
        "Fast-start ingest exited for f (code 3): no stderr",
    ]
    process.stderr = None
    with pytest.raises(ValueError):
        await ffmpeg_session._monitor_fast_process(_proc(process), "720p", "f")


def test_log_watchdog_delay_rate_limited(caplog: pytest.LogCaptureFixture, tmp_path: pathlib.Path):
    session = _live(tmp_path)
    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        ffmpeg_session._log_watchdog_delay("f", session, "pipeline", 100)
        ffmpeg_session._log_watchdog_delay("f", session, "pipeline", 129.9)
        ffmpeg_session._log_watchdog_delay("f", session, "pipeline", 130)
    assert [r.getMessage() for r in caplog.records] == [
        "Live watchdog f is delaying pipeline recovery because its restart budget is cooling down",
    ] * 2
    assert session.get("pipeline_restart_log_at") == 130


def _fast_session(
    directory: pathlib.Path,
    process: ffmpeg_session.ManagedProcess | None = None,
    *,
    live_recovery: LiveRecoveryPolicy | None = None,
) -> ffmpeg_session.FastLiveSession:
    return ffmpeg_session.FastLiveSession(
        dir=str(directory),
        process=process,
        started=_NOW,
        last_access=_NOW,
        url="http://live",
        live_recovery=live_recovery or LiveRecoveryPolicy(),
        ingest_restart_command=["ffmpeg", "-i", "http://live", f"{directory}/input.m3u8"],
        low_restart_command=["ffmpeg", "-i", f"{directory}/input.m3u8", f"{directory}/low.m3u8"],
        playlist_generation=0,
    )


@pytest.mark.asyncio
async def test_restart_clock_dates_old_segments_then_follows_new_ingest(
    env: _Env,
    tmp_path: pathlib.Path,
):
    (tmp_path / "low_1.ts").write_bytes(video_packet(5002) * 10)
    (tmp_path / "input_000001.ts").write_bytes(video_packet(1.5) * 10)
    session = _fast_session(tmp_path)
    session.update(origin_pts=5000.0, origin_time=100.0)
    env.sessions["f"] = session
    await ffmpeg_session._restart_session_clock("f", session, tmp_path / "missing.m3u8")
    assert "segment_dates" not in session
    (tmp_path / "low.m3u8").write_text("#EXTINF:2,\nlow_1.ts\n")
    session.update(origin_pts=5000.0, origin_time=100.0)
    await ffmpeg_session._restart_session_clock("f", session, tmp_path / "low.m3u8")
    assert session.get("segment_dates") == {"low_1.ts": 102.0}
    assert session.get("origin_pts") == 1.5
    assert session.get("origin_time") == _NOW
    env.sessions.clear()
    session.update(origin_pts=7.0)
    await ffmpeg_session._restart_session_clock("f", session, tmp_path / "low.m3u8")
    assert session.get("origin_pts") == 7.0


def test_dated_fast_playlist_needs_a_known_clock(env: _Env, tmp_path: pathlib.Path):
    assert ffmpeg_session.dated_fast_playlist("missing", "#EXTM3U\n") == "#EXTM3U\n"
    env.sessions["f"] = _fast_session(tmp_path)
    assert ffmpeg_session.dated_fast_playlist("f", "#EXTM3U\n") == "#EXTM3U\n"
    env.sessions["f"]["origin_pts"] = None
    assert ffmpeg_session.dated_fast_playlist("f", "#EXTM3U\n") == "#EXTM3U\n"
    env.sessions["f"]["origin_pts"] = 1.0
    with pytest.raises(KeyError, match="origin_time"):
        ffmpeg_session.dated_fast_playlist("f", "#EXTM3U\n")


def _pipeline_dir(tmp_path: pathlib.Path) -> None:
    (tmp_path / "input.m3u8").write_text("#EXTINF:2,\ninput_1.ts\n")
    (tmp_path / "input_1.ts").write_bytes(b"x")
    (tmp_path / "input_2.ts.tmp").write_bytes(b"x")
    (tmp_path / "low.m3u8").write_text("#EXTINF:2,\nlow_1.ts\n")


@pytest.mark.asyncio
async def test_restart_pipeline_skips_replaced_session(env: _Env, tmp_path: pathlib.Path):
    session = _fast_session(tmp_path)
    await ffmpeg_session._restart_fast_pipeline("f", session, "input")
    assert "watchdog_recovering" not in session
    assert env.launcher.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize("generation", [0, 7])
async def test_restart_pipeline_success(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    generation: int,
):
    _pipeline_dir(tmp_path)
    low_old, ingest_old = FakeProcess(exit_code=None), FakeProcess(exit_code=None)
    recovery, permitted = LiveRecoveryPolicy(failures={"input": _NOW - 5}).permit_restart(
        "pipeline",
        _NOW,
    )
    assert permitted
    session = _fast_session(tmp_path, process=low_old, live_recovery=recovery)
    session.update(
        {
            "ingest_process": ingest_old,
            "extra_processes": [ingest_old],
            "playlist_generation": generation,
        },
    )
    env.sessions["f"] = session
    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await ffmpeg_session._restart_fast_pipeline("f", session, "input")
    assert env.launcher.commands == [
        session["ingest_restart_command"],
        session.get("low_restart_command"),
    ]
    ingest, low = env.launcher.processes
    assert all(p.returncode == -15 for p in (low_old, ingest_old))
    assert session["process"] is low
    assert session.get("ingest_process") is ingest
    assert session.get("extra_processes") == [ingest]
    assert session.get("playlist_generation") == generation + 1
    assert session.get("origin_pts") == 1.5
    assert dict(session["live_recovery"].failures) == {}
    assert session["live_recovery"].pending == frozenset()
    assert "watchdog_recovering" not in session
    assert not (tmp_path / "input_1.ts").exists()
    assert not (tmp_path / "input_2.ts.tmp").exists()
    assert _coroutine_names(tasks) == ["_monitor_fast_process", "_monitor_fast_process"]
    assert [r.getMessage() for r in caplog.records] == [
        f"Live watchdog f recovered pipeline generation {generation + 1} after input stalled",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["ingest", "low"])
async def test_restart_pipeline_timeout_discards_replacements(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    failing: str,
    caplog: pytest.LogCaptureFixture,
):
    del tasks
    _pipeline_dir(tmp_path)
    recovery, _ = LiveRecoveryPolicy(failures={"low": _NOW - 5}).permit_restart("pipeline", _NOW)
    session = _fast_session(tmp_path, process=FakeProcess(exit_code=None), live_recovery=recovery)
    session["playlist_generation"] = 5
    env.sessions["f"] = session
    env.launcher.on_launch = None if failing == "ingest" else _input_only
    with caplog.at_level(logging.ERROR, logger="netv.ffmpeg_session"):
        await ffmpeg_session._restart_fast_pipeline("f", session, "low output")
    assert "failed to recover the pipeline after low output stalled" in caplog.text
    assert isinstance(session["process"], ffmpeg_session._DeadProcess)
    assert session.get("ingest_process", "absent") is None
    assert session.get("extra_processes") == []
    assert [p.returncode for p in env.launcher.processes] == (
        [-15] if failing == "ingest" else [-15, -15]
    )
    assert session.get("playlist_generation") == 5
    # The stall stays on record and the budget frees for the next attempt.
    assert dict(session["live_recovery"].failures) == {"low": _NOW - 5}
    assert session["live_recovery"].pending == frozenset()
    assert "watchdog_recovering" not in session


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["ingest", "low", "ready"])
async def test_restart_pipeline_session_replaced(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    stage: str,
):
    del tasks
    _pipeline_dir(tmp_path)
    session = _fast_session(tmp_path, process=FakeProcess(exit_code=None))
    env.sessions["f"] = session

    def launch(command: list[str]) -> None:
        role = "ingest" if command[-1].endswith("input.m3u8") else "low"
        if role == "ingest" or stage == "low":
            _ffmpeg_output(command)
        if role == stage:
            env.sessions.clear()

    async def stop_once_encoding() -> None:
        while (
            len(env.launcher.processes) < 2 or session["process"] is not env.launcher.processes[1]
        ):
            await asyncio.sleep(0)
        env.sessions.clear()
        _ffmpeg_output(env.launcher.commands[1])

    env.launcher.on_launch = launch
    stopper = asyncio.create_task(stop_once_encoding())
    with pytest.raises(asyncio.CancelledError):
        await ffmpeg_session._restart_fast_pipeline("f", session, "input")
    if stage == "ready":
        await stopper
    else:
        stopper.cancel()
    assert [p.returncode for p in env.launcher.processes] == (
        [-15] if stage == "ingest" else [-15, -15]
    )
    assert session.get("playlist_generation") == 0
    assert session.get("watchdog_recovering") == "input"


# =============================================================================
# Watchdog
# =============================================================================


def _aged_playlist(directory: pathlib.Path, name: str, *, segments: int, mtime: float) -> None:
    lines = ["#EXTM3U"]
    for i in range(segments):
        (directory / f"{name}_{i}.ts").write_bytes(video_packet(10 + 2 * i) * 10)
        lines += ["#EXTINF:2,", f"{name}_{i}.ts"]
    path = directory / f"{name}.m3u8"
    path.write_text("\n".join(lines) + "\n")
    os.utime(path, (mtime, mtime))


def _process(alive: bool | None) -> FakeProcess | None:
    if alive is None:
        return None
    return FakeProcess(exit_code=None) if alive else FakeProcess(exited=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_after", [None, 47.5])
@pytest.mark.parametrize(
    "ingest_alive,low_alive,input_age,low_age",
    [
        (True, True, 5, 5),
        (True, True, 60, 5),
        (True, True, 5, 60),
        (True, True, 30, 30),
        (False, True, 5, 5),
        (True, False, 5, 5),
        (None, None, 5, 5),
    ],
)
async def test_watchdog_checks_live_media_and_persists_each_observation(
    env: _Env,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stale_after: float | None,
    ingest_alive: bool | None,
    low_alive: bool | None,
    input_age: float,
    low_age: float,
):
    now = 123.5
    env.clock.now = now
    install_clock(monkeypatch, env.clock, fast_start)
    # The input needs one complete segment; the encoder output needs two.
    _aged_playlist(tmp_path, "input", segments=1, mtime=now - input_age)
    _aged_playlist(tmp_path, "low", segments=2, mtime=now - low_age)
    session = ffmpeg_session.FastLiveSession(
        dir=str(tmp_path),
        process=_process(low_alive),
        started=0,
        url="https://stream.test/live",
        live_recovery=LiveRecoveryPolicy(),
        ingest_restart_command=["ingest"],
    )
    if ingest_alive is not None:
        session["ingest_process"] = _process(ingest_alive)
    if stale_after is not None:
        session["watchdog_stale_after"] = stale_after
    env.sessions["media-check"] = session

    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await ffmpeg_session._check_fast_live_session("media-check", session)

    max_age = 10 if stale_after is None else stale_after
    ready = {
        "input": bool(ingest_alive) and input_age <= max_age,
        "low": bool(low_alive) and low_age <= max_age,
    }
    recovery = session["live_recovery"]
    assert dict(recovery.failures) == {stage: now for stage, ok in ready.items() if not ok}
    assert dict(recovery.attempts) == {}
    assert recovery.pending == frozenset()
    assert env.launcher.launches == []
    assert env.sessions == {"media-check": session}
    assert not caplog.records


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_stalled,low_stalled,attempts,pending,action",
    [
        (True, False, (), False, "restart"),
        (False, True, (), False, "restart"),
        (True, True, (), False, "restart"),
        (True, False, (281.5, 291.5, 306.5), True, "pending"),
        (False, True, (281.5, 291.5, 306.5), False, "stop"),
        (True, False, (281.5, 291.5, 306.5), False, "stop"),
        (False, True, (320.5,), False, "delay"),
    ],
)
async def test_watchdog_dispatches_recovery_with_updated_policy(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    input_stalled: bool,
    low_stalled: bool,
    attempts: tuple[float, ...],
    pending: bool,
    action: str,
):
    del tasks
    now = 321.5
    env.clock.now = now
    # The replacement ingest dies at once, so a permitted restart fails fast.
    launcher = _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    stalled = {"input": input_stalled, "low": low_stalled}
    session = ffmpeg_session.FastLiveSession(
        dir=str(tmp_path),
        process=None,
        started=0,
        url="https://stream.test/live",
        live_recovery=LiveRecoveryPolicy(
            failures={stage: now - 10 for stage, value in stalled.items() if value},
            attempts={"pipeline": attempts} if attempts else {},
            pending=frozenset({"pipeline"}) if pending else frozenset(),
        ),
        ingest_restart_command=["ffmpeg", "ingest"],
        low_restart_command=["ffmpeg", "low"],
    )
    env.sessions["dispatch-check"] = session
    env.urls["https://stream.test/live"] = "dispatch-check"

    with caplog.at_level(logging.ERROR, logger="netv.ffmpeg_session"):
        await ffmpeg_session._check_fast_live_session("dispatch-check", session)

    stage = "input" if input_stalled else "low output"
    recovery = session["live_recovery"]
    assert dict(recovery.failures) == {
        "input": now - 10 if input_stalled else now,
        "low": now - 10 if low_stalled else now,
    }
    messages = [(record.levelno, record.getMessage()) for record in caplog.records]
    if action == "restart":
        assert launcher.commands == [["ffmpeg", "ingest"]]
        assert dict(recovery.attempts) == {"pipeline": (now,)}
        assert recovery.pending == frozenset()
        assert messages == [
            (
                logging.ERROR,
                f"Live watchdog dispatch-check failed to recover the pipeline after {stage} "
                "stalled",
            ),
        ]
    else:
        assert launcher.launches == []
        assert dict(recovery.attempts) == {"pipeline": attempts}
    if action == "pending":
        assert recovery.pending == frozenset({"pipeline"})
    if action == "stop":
        # Without a pipeline, polling clients would otherwise keep reusing it.
        assert not env.sessions
        assert ffmpeg_session._get_existing_session("https://stream.test/live") is None
        assert not env.urls
        assert not tmp_path.exists()
        assert messages == [
            (
                logging.ERROR,
                f"Live watchdog dispatch-check stopped the session: {stage} stalled "
                "and its restart budget is spent",
            ),
        ]
    else:
        assert env.sessions == {"dispatch-check": session}
    if action in {"pending", "delay"}:
        assert not messages
    assert session.get("pipeline_restart_log_at") == (now if action == "delay" else None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failures,input_ready,stage",
    [
        ({"input": _NOW - 5}, False, "input"),
        ({"input": _NOW - 5, "low": _NOW - 5}, True, "low output"),
    ],
)
async def test_check_restarts_stalled_pipeline(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    failures: dict[str, float],
    input_ready: bool,
    stage: str,
):
    del tasks
    if input_ready:
        _media_playlist(tmp_path / "input.m3u8", ["input_000001.ts"], duration=2.0)
    session = _fast_session(tmp_path, live_recovery=LiveRecoveryPolicy(failures=failures))
    session["ingest_process"] = FakeProcess(exit_code=None)
    env.sessions["f"] = session
    env.launcher.on_launch = None
    with caplog.at_level(logging.ERROR, logger="netv.ffmpeg_session"):
        await ffmpeg_session._check_fast_live_session("f", session)
    assert f"failed to recover the pipeline after {stage} stalled" in caplog.text
    assert env.launcher.commands[0] == session["ingest_restart_command"]
    # A live ingest with fresh output clears its stall record.
    assert ("input" in session["live_recovery"].failures) is not input_ready


@pytest.mark.asyncio
async def test_watchdog_logs_errors_and_propagates_cancel(
    env: _Env,
    tasks: BackgroundTasks,
    caplog: pytest.LogCaptureFixture,
    tmp_path: pathlib.Path,
):
    del tasks
    session = _fast_session(tmp_path, live_recovery=LiveRecoveryPolicy(failures={"low": _NOW - 5}))
    env.sessions["f"] = session
    launched: list[list[str]] = []

    def launch(command: list[str]) -> None:
        launched.append(command)
        if len(launched) == 1:
            raise RuntimeError("boom")
        env.sessions.clear()

    env.launcher.on_launch = launch
    with (
        caplog.at_level(logging.ERROR, logger="netv.ffmpeg_session"),
        pytest.raises(asyncio.CancelledError),
    ):
        await ffmpeg_session._watch_fast_live("f", watchdog_interval_sec=6)
    assert "failed while checking pipeline health" in caplog.text
    assert len(launched) == 2


@pytest.mark.asyncio
async def test_watchdog_skips_idle_session_and_exits_when_removed(
    env: _Env,
    tmp_path: pathlib.Path,
):
    policy = LiveRecoveryPolicy()
    session = _fast_session(tmp_path, live_recovery=policy)
    session["last_access"] = _NOW - 30.5
    env.sessions["f"] = session
    task = asyncio.create_task(ffmpeg_session._watch_fast_live("f", watchdog_interval_sec=0))
    for _ in range(3):
        await asyncio.sleep(0)
    env.sessions.clear()
    await asyncio.wait_for(task, timeout=1)
    assert session["live_recovery"] is policy
    assert env.launcher.commands == []


@pytest.mark.asyncio
async def test_watchdog_missing_recovery_names_the_required_field(tmp_path: pathlib.Path):
    session = ffmpeg_session.TranscodeSession(dir=str(tmp_path), process=None, started=0, url="x")
    with pytest.raises(KeyError) as error:
        await ffmpeg_session._check_fast_live_session("missing-policy", session)
    assert error.value.args == ("live_recovery",)


# =============================================================================
# Live recovery across watchdog checks
# =============================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class _Recovery:
    session: ffmpeg_session.FastLiveSession
    launcher: ProcessLauncher
    clock: FakeClock
    ingest: FakeProcess
    low: FakeProcess


@pytest.fixture
def recovery(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Recovery:
    """A running fast-start session whose replacement ffmpeg publishes on launch."""
    del tasks
    for name in ("input", "low"):
        playlist(tmp_path, name)
    ingest, low = FakeProcess(exit_code=None), FakeProcess(exit_code=None)
    session = _fast_session(tmp_path, process=low)
    session.update(
        {
            "ingest_process": ingest,
            "extra_processes": [ingest],
            "username": "viewer",
            "fast_start": True,
            "watchdog_stale_after": 45,
        },
    )
    env.sessions["recover"] = session
    install_clock(monkeypatch, env.clock, fast_start)
    return _Recovery(
        session=session,
        launcher=env.launcher,
        clock=env.clock,
        ingest=ingest,
        low=low,
    )


async def _check(recovery: _Recovery, now: float, fresh: dict[str, bool]) -> None:
    """Republish each playlist, fresh or 100 s stale, then run one watchdog check at ``now``."""
    recovery.clock.now = now
    directory = pathlib.Path(recovery.session["dir"])
    for name, is_fresh in fresh.items():
        playlist(directory, name)
        mtime = now if is_fresh else now - 100
        os.utime(directory / f"{name}.m3u8", (mtime, mtime))
    await ffmpeg_session._check_fast_live_session("recover", recovery.session)


@pytest.mark.asyncio
async def test_input_or_low_stall_restarts_pipeline_once(
    recovery: _Recovery,
    caplog: pytest.LogCaptureFixture,
):
    fresh = {"input": False, "low": True}
    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await _check(recovery, 1000, fresh)
        assert recovery.launcher.launches == []
        await _check(recovery, 1002, fresh)
        assert len(recovery.launcher.launches) == 2
        await _check(recovery, 1004, fresh)
    assert len(recovery.launcher.launches) == 2
    assert recovery.ingest.signals == ["TERM"]
    assert recovery.low.signals == ["TERM"]
    assert recovery.session.get("playlist_generation") == 1
    assert [record.getMessage() for record in caplog.records] == [
        "Live watchdog recover recovered pipeline generation 1 after input stalled",
    ]


@pytest.mark.asyncio
async def test_short_publication_gap_does_not_restart(recovery: _Recovery):
    await _check(recovery, 1000, {"input": True, "low": False})
    await _check(recovery, 1002, {"input": True, "low": True})
    assert recovery.launcher.launches == []
    assert recovery.session["process"] is recovery.low
    assert dict(recovery.session["live_recovery"].failures) == {}


@pytest.mark.asyncio
async def test_running_pipeline_tolerates_thirty_second_publication_gap(recovery: _Recovery):
    for name in ("input", "low"):
        os.utime(pathlib.Path(recovery.session["dir"]) / f"{name}.m3u8", (70, 70))
    recovery.clock.now = 100
    await ffmpeg_session._check_fast_live_session("recover", recovery.session)
    assert recovery.launcher.launches == []
    assert dict(recovery.session["live_recovery"].failures) == {}


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
