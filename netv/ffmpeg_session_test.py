"""Tests for ffmpeg session management."""

from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import CoroutineType
from typing import cast

import asyncio
import ipaddress
import json
import logging
import os
import pathlib
import subprocess
import threading
import time
import uuid

from fastapi import HTTPException

import pytest
import pytest_asyncio

from netv import cache, fast_start, ffmpeg_command, ffmpeg_session, proxy, testing, util
from netv.fast_start import master_playlist
from netv.fast_start_test import playlist
from netv.ffmpeg_command import MediaInfo, SubtitleStream
from netv.playback_policy import LiveRecoveryPolicy, PlaybackHealth, PlaybackPolicy
from netv.testing import (
    BackgroundTasks,
    FakeClock,
    FakeProcess,
    FakeResponse,
    FakeRun,
    FakeTask,
    FakeUpstream,
    FakeUpstreams,
    FakeWeb,
    PopenRecorder,
    ProcessLauncher,
    Program,
    install_clock,
    install_launcher,
    install_subprocess,
    install_web,
    pat_packet,
    pmt_packet,
    probe_capture,
    probed_video_json,
    ts_source,
    video_packet,
)


_NOW = 1_700_000_000.0
# A live upstream that sends nothing until the test feeds it.
_TUNER = "http://tuner"
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
    if output.name in ("low.m3u8", "up.m3u8"):
        segments = [f"{output.stem}_000009.ts", f"{output.stem}_000010.ts"]
        _media_playlist(output, segments, duration=4.0)
    else:
        start = (
            int(command[command.index("-start_number") + 1]) if "-start_number" in command else 0
        )
        _media_playlist(output, [f"seg{start:03d}.ts", f"seg{start + 1:03d}.ts"])


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> _Env:
    sessions: dict[str, ffmpeg_session.TranscodeSession] = {}
    urls: dict[str, str] = {}
    monkeypatch.setattr(ffmpeg_session, "_transcode_sessions", sessions)
    monkeypatch.setattr(ffmpeg_session, "_url_to_session", urls)
    monkeypatch.setattr(ffmpeg_session, "_startup_guards", {})
    monkeypatch.setattr(ffmpeg_session, "_recordings", set())
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
    install_web(monkeypatch, web, util)
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


@pytest_asyncio.fixture(autouse=True)
async def upstreams(env: _Env) -> AsyncIterator[FakeUpstreams]:
    """Live connections: http://live sends a keyframe-led source at once; the tuner waits.

    Closed before the event loop ends, so no relay reader thread outlives the test.
    """
    opened = FakeUpstreams()

    def tuned() -> FakeUpstream:
        upstream = opened()
        upstream.feed(ts_source())
        return upstream

    env.web.routes.update({"http://live": tuned, _TUNER: opened})
    yield opened
    opened.close()


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


def _save_sources(*sources: cache.SourceConfig) -> None:
    cache.save_server_settings({**cache.load_server_settings(), "sources": list(sources)})


def _plan(
    source_url: str = "http://live",
    *,
    hw: cache.HwAccel = "software",
    deinterlace: bool = False,
    source_id: str = "",
) -> ffmpeg_session._FastLivePlan:
    return ffmpeg_session._FastLivePlan(
        source_url=source_url,
        source_id=source_id,
        user_agent=None,
        hw=hw,
        resolution="1080p",
        quality="high",
        deinterlace=deinterlace,
        bandwidth_saver=False,
        audio_passthrough=False,
        dvr=False,
    )


# Captured probes (testing.probe_capture). Each row is what the live encoder must do:
# (capture, deinterlaced, tone-mapped, GPU-decoded on an h264-only NVDEC, 5.1 audio).
_PROBED_SOURCES = [
    ("ota_1080i_mpeg2", True, False, False, True),
    # The startup sample was progressive, but the channel switches to 60i content.
    ("ota_1080_mpeg2_progressive_segment", True, False, False, True),
    ("ota_480i_mpeg2", True, False, False, False),
    ("ota_720p_mpeg2", False, False, False, True),
    ("ota_720p_h264", False, False, True, False),
    # 1080 H.264 may carry fields later; frames marked progressive pass untouched.
    ("iptv_1080p_h264_aac", True, False, True, False),
    ("iptv_2160p_hevc_hdr", False, True, False, True),
]


@pytest.mark.parametrize(
    ("capture", "deinterlaced", "tone_mapped", "gpu_decoded", "surround"),
    _PROBED_SOURCES,
)
@pytest.mark.parametrize("hw", ["software", "nvenc+software"])
def test_the_live_encoder_acts_on_what_the_startup_probe_found(
    env: _Env,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capture: str,
    deinterlaced: bool,
    tone_mapped: bool,
    gpu_decoded: bool,
    surround: bool,
    hw: cache.HwAccel,
):
    """The probe the worker takes at startup decides decoding, filters and audio.

    The source's deinterlace setting is off here: it applies only when no probe is had.
    """
    monkeypatch.setattr(ffmpeg_command, "_gpu_nvdec_codecs", {"h264"})
    env.run.program("ffprobe", Program(stdout=probe_capture(capture)), when="-show_streams")
    source = ffmpeg_command.probe_segment(str(tmp_path / "probe.ts"))
    assert source is not None

    cmd, role = _plan(hw=hw).worker(str(tmp_path), source, restarting=False)

    assert role == "encoder"
    video_filter = cmd[cmd.index("-vf") + 1]
    assert ("yadif" in video_filter) == deinterlaced
    assert ("tonemap" in video_filter or "libplacebo" in video_filter) == tone_mapped
    assert ("-hwaccel" in cmd) == (gpu_decoded and hw == "nvenc+software")
    assert ("aformat=channel_layouts=5.1" in cmd) == surround


@pytest.mark.parametrize("fallback", [False, True])
def test_without_a_probe_the_source_setting_decides_deinterlacing(
    env: _Env,
    tmp_path: pathlib.Path,
    fallback: bool,
):
    del env
    cmd, _ = _plan(deinterlace=fallback).worker(str(tmp_path), None, restarting=False)
    assert ("yadif" in cmd[cmd.index("-vf") + 1]) == fallback


def _fast_session(
    directory: pathlib.Path,
    process: ffmpeg_session.ManagedProcess | None = None,
    *,
    live_recovery: LiveRecoveryPolicy | None = None,
    source_url: str = "http://live",
) -> ffmpeg_session.FastLiveSession:
    """A fast-start session whose relay is built but not started."""
    plan = _plan(source_url)
    return ffmpeg_session.FastLiveSession(
        dir=str(directory),
        process=process,
        started=_NOW,
        last_access=_NOW,
        url="http://live",
        viewers={},
        viewer_joined=asyncio.Event(),
        live_recovery=live_recovery or LiveRecoveryPolicy(),
        playback_policy=PlaybackPolicy(),
        plan=plan,
        relay=plan.relay(),
        playlist_generation=0,
    )


async def _receiving(session: ffmpeg_session.FastLiveSession) -> None:
    """Start the session's relay and wait until it holds the upstream's first bytes."""
    await session["relay"].start()
    await session["relay"].wait_for_bytes(1)


def _save_settings_cache(*, vod_mins: int = 60) -> None:
    cache.save_server_settings(
        {**cache.load_server_settings(), "vod_transcode_cache_mins": vod_mins},
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


class TestIsSessionValid:
    """Tests for is_session_valid with heartbeat timeout."""

    def _session(
        self,
        tmp_path: pathlib.Path,
        *,
        alive: bool,
        age: float,
    ) -> ffmpeg_session.TranscodeSession:
        return ffmpeg_session.TranscodeSession(
            process=FakeProcess(exit_code=None) if alive else FakeProcess(exited=True),
            started=_NOW - age,
            last_access=_NOW - age,
            is_vod=True,
            dir=str(tmp_path),
            url="http://stream.test/movie",
        )

    def test_active_process_with_recent_heartbeat(self, tmp_path: pathlib.Path):
        assert ffmpeg_session.is_session_valid(self._session(tmp_path, alive=True, age=0)) is True

    def test_active_process_stale_heartbeat(self, tmp_path: pathlib.Path):
        session = self._session(tmp_path, alive=True, age=400)
        assert ffmpeg_session.is_session_valid(session) is False

    def test_dead_process_vod_session_without_cache(self, tmp_path: pathlib.Path):
        _save_settings_cache(vod_mins=0)
        session = self._session(tmp_path, alive=False, age=0)
        assert ffmpeg_session.is_session_valid(session) is False

    def test_dead_process_vod_session_within_cache(self, tmp_path: pathlib.Path):
        _save_settings_cache(vod_mins=60)
        session = self._session(tmp_path, alive=False, age=10)
        assert ffmpeg_session.is_session_valid(session, heartbeat_timeout_sec=3600) is True

    def test_dead_process_vod_session_expired_cache(self, tmp_path: pathlib.Path):
        _save_settings_cache(vod_mins=1)
        session = self._session(tmp_path, alive=False, age=120)
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
    session = _fast_session(tmp_path)
    session["fast_start"] = True
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

    def test_stop_skips_recently_accessed_session(self, env: _Env, tmp_path: pathlib.Path):
        """Seeking or another viewer touched the session within the last 5 s."""
        process = FakeProcess(exit_code=None)
        env.sessions["s"] = ffmpeg_session.TranscodeSession(
            process=process,
            dir=str(tmp_path),
            url="http://test",
            is_vod=True,
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
            is_vod=True,
            last_access=_NOW - 10,
            started=_NOW,
        )
        ffmpeg_session.touch_session("s")
        asyncio.run(ffmpeg_session.stop_session("s", force=False))
        assert "s" in env.sessions
        assert process.returncode is None

    def test_stop_spares_a_recently_accessed_session_by_default(
        self,
        env: _Env,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ):
        env.sessions["s"] = _live(tmp_path)
        env.sessions["s"]["is_vod"] = True
        with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
            asyncio.run(ffmpeg_session.stop_session("s"))
        assert "s" in env.sessions
        assert "Ignoring stop for recently-accessed session s" in caplog.messages

    def test_a_session_never_accessed_stops(
        self,
        env: _Env,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ):
        _save_settings_cache(vod_mins=0)
        session = _live(tmp_path, url="http://vod")
        del session["last_access"]
        env.sessions["s"] = session
        env.urls["http://vod"] = "s"
        with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
            asyncio.run(ffmpeg_session.stop_session("s"))
        assert not env.sessions
        assert not env.urls
        assert "Stopped transcode session s" in caplog.messages

    def test_stop_session_caches_vod(
        self,
        env: _Env,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """Stop caches VOD session instead of removing it, even for one minute."""
        process = FakeProcess(exit_code=None)
        env.sessions["s"] = ffmpeg_session.TranscodeSession(
            process=process,
            dir=str(tmp_path),
            url="http://vod",
            is_vod=True,
            last_access=_NOW - 5,
            started=_NOW,
        )
        _save_settings_cache(vod_mins=1)
        with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
            asyncio.run(ffmpeg_session.stop_session("s", force=False))
        assert "s" in env.sessions
        assert env.sessions["s"].get("last_access") == _NOW
        assert process.signals == ["TERM"]
        assert tmp_path.exists()
        assert "Session s cached (ffmpeg stopped, segments kept)" in caplog.messages

    def test_stale_vod_session_without_cache_is_removed(self, env: _Env, tmp_path: pathlib.Path):
        _save_settings_cache(vod_mins=0)
        env.sessions["s"] = _live(tmp_path)
        env.sessions["s"].update(is_vod=True, last_access=_NOW - 10)
        asyncio.run(ffmpeg_session.stop_session("s"))
        assert not env.sessions
        assert not tmp_path.exists()


@pytest.mark.asyncio
async def test_stop_fast_session_cancels_the_watchdog_and_closes_the_relay(
    env: _Env,
    upstreams: FakeUpstreams,
    tmp_path: pathlib.Path,
):
    """A fast-start session cannot resume, so even a just-polled one stops."""
    watchdog = asyncio.create_task(_idle())
    worker = FakeProcess(exit_code=None)
    session = _fast_session(tmp_path, worker)
    session.update(watchdog_task=watchdog, fast_start=True)
    env.sessions["s"] = session
    await _receiving(session)
    session["subscription"] = session["relay"].subscribe(worker.stdin)
    await ffmpeg_session.stop_session("s")
    await asyncio.sleep(0)
    assert watchdog.cancelled()
    assert worker.signals == ["TERM"]
    assert upstreams.opened[0].closed
    assert session["relay"].finished
    assert worker.stdin.closed
    assert "s" not in env.sessions
    assert not tmp_path.exists()


@pytest.mark.asyncio
async def test_a_short_session_keeps_the_burst_gap_a_longer_one_learned(
    env: _Env,
    tmp_path: pathlib.Path,
):
    cache.remember_burst_gap("live", 9.5, 30)
    session = _fast_session(tmp_path, FakeProcess(exit_code=None))
    session["fast_start"] = True
    env.sessions["s"] = session
    await _receiving(session)
    await ffmpeg_session.stop_session("s")
    assert cache.relay_hints()["recurring_gaps"] == {"live": 9.5}


def _watched(
    env: _Env,
    directory: pathlib.Path,
    viewers: dict[str, float],
) -> ffmpeg_session.FastLiveSession:
    """A running fast-start session ``s`` whose pages last reported at ``viewers``."""
    session = _fast_session(directory, FakeProcess(exit_code=None))
    session.update(fast_start=True, viewers=viewers)
    env.sessions["s"] = session
    env.urls["http://live"] = "s"
    return session


@pytest.mark.asyncio
async def test_a_page_leaving_a_shared_session_keeps_it_for_the_others(
    env: _Env,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
):
    """A reloaded page joins before its old page's stop arrives."""
    session = _watched(env, tmp_path, {"old": _NOW, "new": _NOW})
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        await ffmpeg_session.stop_session("s", force=True, viewer="old")
    assert env.sessions == {"s": session}
    assert session["viewers"] == {"new": _NOW}
    assert "Viewer left session s; 1 remain" in caplog.messages


@pytest.mark.asyncio
async def test_the_last_viewer_leaving_stops_the_session_after_a_grace(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
):
    """A reloaded page's old stop often lands before the new page asks again."""
    _watched(env, tmp_path, {"page": _NOW})
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        await ffmpeg_session.stop_session("s", force=True, viewer="page")
        assert "s" in env.sessions
        assert caplog.messages == ["Last viewer left session s"]
        await tasks.run_all()
    assert env.clock.now == _NOW + 10
    assert not env.sessions
    assert not tmp_path.exists()
    assert "No viewer joined session s within 10s" in caplog.messages


@pytest.mark.asyncio
async def test_a_session_already_gone_is_not_waited_on(env: _Env):
    await ffmpeg_session.stop_when_unwatched("gone")
    assert env.clock.now == _NOW
    assert not env.sessions


@pytest.mark.asyncio
async def test_a_session_gone_during_the_grace_needs_no_stop(
    env: _Env,
    tmp_path: pathlib.Path,
):
    _watched(env, tmp_path, {})
    grace = asyncio.ensure_future(ffmpeg_session.stop_when_unwatched("s"))
    asyncio.get_running_loop().call_soon(env.sessions.clear)
    await grace
    assert env.clock.now == _NOW + 10
    assert tmp_path.exists()


@pytest.mark.asyncio
async def test_a_page_joining_during_the_grace_ends_it_at_once(
    env: _Env,
    tmp_path: pathlib.Path,
):
    session = _watched(env, tmp_path, {})
    grace = asyncio.ensure_future(ffmpeg_session.stop_when_unwatched("s"))
    # The join lands while the grace waits; the wait ends then, not 10 s later.
    loop = asyncio.get_running_loop()
    loop.call_soon(ffmpeg_session._adaptive_session_response, "s")
    await grace
    assert env.clock.now == _NOW
    assert env.sessions == {"s": session}
    assert len(session["viewers"]) == 1


@pytest.mark.asyncio
async def test_a_page_joining_within_the_grace_keeps_the_session(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    session = _watched(env, tmp_path, {"old": _NOW})
    await ffmpeg_session.stop_session("s", force=True, viewer="old")
    # A new page joins as start hands it the session.
    response = ffmpeg_session._adaptive_session_response("s")
    assert session["viewers"] == {response.get("viewer_id"): _NOW}
    await tasks.run_all()
    assert env.sessions == {"s": session}


@pytest.mark.asyncio
async def test_pages_silent_past_the_heartbeat_no_longer_hold_the_session(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    _watched(env, tmp_path, {"leaving": _NOW, "silent": _NOW - 30.5, "recent": _NOW - 30})
    await ffmpeg_session.stop_session("s", force=True, viewer="leaving")
    assert env.sessions["s"].get("viewers") == {"recent": _NOW - 30}
    await ffmpeg_session.stop_session("s", force=True, viewer="unknown", heartbeat_timeout_sec=29)
    await tasks.run_all()
    assert not env.sessions


@pytest.mark.asyncio
async def test_only_live_sessions_count_viewers(env: _Env, tmp_path: pathlib.Path):
    env.sessions["s"] = _live(tmp_path)
    await ffmpeg_session.stop_session("s", force=True, viewer="page")
    assert not env.sessions


def test_a_reporting_page_counts_as_watching(env: _Env, tmp_path: pathlib.Path):
    session = _watched(env, tmp_path, {"page": _NOW - 20})
    env.clock.advance(5)
    ffmpeg_session.note_viewer("s", "page")
    ffmpeg_session.note_viewer("s", "")
    ffmpeg_session.note_viewer("missing", "page")
    env.sessions["vod"] = _live(tmp_path)
    ffmpeg_session.note_viewer("vod", "page")
    assert session["viewers"] == {"page": _NOW + 5}
    assert "viewers" not in env.sessions["vod"]


@pytest.mark.asyncio
async def test_a_report_landing_after_its_page_left_does_not_bring_it_back(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    """A page's last health report can be in flight when its stop arrives."""
    session = _watched(env, tmp_path, {"page": _NOW})
    await ffmpeg_session.stop_session("s", force=True, viewer="page")
    ffmpeg_session.note_viewer("s", "page")
    assert session["viewers"] == {}
    await tasks.run_all()
    assert not env.sessions


class _Sleeps:
    """Records each poll's sleep and runs ``on_sleep`` instead of waiting."""

    def __init__(self, on_sleep: Callable[[], object] = lambda: None) -> None:
        self.calls: list[float] = []
        self.on_sleep = on_sleep

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.on_sleep()


@pytest.mark.asyncio
async def test_a_blocking_reload_returns_once_the_segment_is_listed(
    env: _Env,
    tmp_path: pathlib.Path,
):
    session = _fast_session(tmp_path)
    env.sessions["s"] = session
    playlist = tmp_path / "low.m3u8"
    playlist.write_text("#EXTM3U\n#EXTINF:1,\nlow_0.ts\n")
    sleeps = _Sleeps(lambda: playlist.write_text(playlist.read_text() + "#EXTINF:1,\nlow_1.ts\n"))

    await ffmpeg_session.wait_for_segment("s", 1, sleep=sleeps)

    assert sleeps.calls == [0.02]
    # A listed segment returns at once.
    await ffmpeg_session.wait_for_segment("s", 0, sleep=sleeps)
    assert sleeps.calls == [0.02]


@pytest.mark.asyncio
async def test_a_blocking_reload_gives_up_after_its_timeout(env: _Env, tmp_path: pathlib.Path):
    env.sessions["s"] = _fast_session(tmp_path)
    (tmp_path / "low.m3u8").write_text("#EXTM3U\n#EXTINF:1,\nlow_0.ts\n")
    sleeps = _Sleeps()
    await ffmpeg_session.wait_for_segment("s", 5, sleep=sleeps)
    assert sleeps.calls == [0.02] * 150
    # Unknown sessions have nothing to wait for.
    await ffmpeg_session.wait_for_segment("missing", 0, sleep=sleeps)
    assert len(sleeps.calls) == 150


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
    first, second = FakeProcess(exit_code=None), FakeProcess(exit_code=None)
    env.sessions["s"] = ffmpeg_session.TranscodeSession(
        watchdog_task=task,
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
    assert [p.returncode for p in (first, second)] == [-15, -15]
    assert not env.sessions


@pytest.mark.asyncio
async def test_shutdown_closes_every_relay(
    env: _Env,
    upstreams: FakeUpstreams,
    tmp_path: pathlib.Path,
):
    """An open relay would keep its upstream connection, and for HLS a remuxer, after exit."""
    worker = FakeProcess(exit_code=None)
    session = _fast_session(tmp_path, worker)
    env.sessions["f"] = session
    await _receiving(session)
    await ffmpeg_session.shutdown()
    assert worker.signals == ["TERM"]
    assert upstreams.opened[0].closed
    assert session["relay"].finished
    assert not env.sessions


# =============================================================================
# Live recording
# =============================================================================


@pytest.fixture
def recording_dir(env: _Env, tasks: BackgroundTasks, tmp_path: pathlib.Path) -> pathlib.Path:
    """A recording directory; ffprobe finds 1080p video in the relay's sample."""
    del tasks
    env.run.program("ffprobe", Program(stdout=probed_video_json()))
    directory = tmp_path / "recording"
    directory.mkdir()
    return directory


async def _viewer(
    env: _Env,
    tmp_path: pathlib.Path,
    username: str,
) -> tuple[ffmpeg_session.FastLiveSession, FakeProcess]:
    """A user's playing fast-start session of http://live, its worker fed by the relay."""
    directory = tmp_path / username
    directory.mkdir()
    worker = FakeProcess(exit_code=None)
    session = _fast_session(directory, worker)
    session.update(username=username, fast_start=True)
    env.sessions[username] = session
    await _receiving(session)
    session["subscription"] = session["relay"].subscribe(worker.stdin)
    return session, worker


def _finish_task(background: BackgroundTasks) -> FakeTask:
    (task,) = [
        task
        for task, name in zip(background.tasks, _coroutine_names(background), strict=True)
        if name == "_finish_recording"
    ]
    return task


@pytest.mark.asyncio
async def test_recording_takes_over_the_users_playing_session(
    env: _Env,
    tasks: BackgroundTasks,
    upstreams: FakeUpstreams,
    recording_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """One upstream connection: the recorder reads the viewer's relay, which outlives the viewer."""
    viewer, worker = await _viewer(env, tmp_path, "alice")
    (recording_dir / "ffmpeg.log").write_bytes(b"earlier\n")
    recorder = FakeProcess(stderr=[b"frame=1\n", b"frame=2\n"])
    launcher = _relaunch(monkeypatch, [recorder])
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        recording = await ffmpeg_session.record_live(
            "alice",
            "http://live",
            "src",
            str(recording_dir),
            60,
        )
    assert f"Recording {recording_dir} for 60s from its session" in caplog.messages
    relay = viewer["relay"]
    assert recording.relay is relay
    assert len(upstreams.opened) == 1
    assert "alice" not in env.sessions
    assert worker.signals == ["TERM"]
    assert worker.stdin.closed
    assert not relay.finished
    assert not upstreams.opened[0].closed
    (launch,) = launcher.launches
    assert launch.options["stdin"] == asyncio.subprocess.PIPE
    probe = recording_dir / "probe.ts"
    assert probe.read_bytes() == ts_source()
    assert launch.command == fast_start.recorder_command(
        str(recording_dir),
        cache.load_server_settings()["transcode_hw"],
        "1080p",
        "high",
        60,
        ffmpeg_command.probe_segment(str(probe)),
    )
    # The recorder starts at the source's keyframe.
    await recorder.stdin.wait_for(len(ts_source()))
    assert bytes(recorder.stdin.written) == ts_source()
    assert ffmpeg_session._recordings == {recording}
    assert not recording.finished()
    await _finish_task(tasks).run()
    assert recording.finished()
    assert not ffmpeg_session._recordings
    assert (recording_dir / "ffmpeg.log").read_bytes() == b"earlier\nframe=1\nframe=2\n"
    # With its last subscriber gone, the released relay closes its upstream.
    await until(lambda: relay.finished and upstreams.opened[0].closed)
    assert recorder.stdin.closed


@pytest.mark.asyncio
async def test_recording_without_a_session_opens_its_own_relay_and_stops_from_a_thread(
    env: _Env,
    tasks: BackgroundTasks,
    upstreams: FakeUpstreams,
    recording_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Another user's session of the channel, or the user's of another, is left alone."""
    other, worker = await _viewer(env, tmp_path, "bob")
    elsewhere, _ = await _viewer(env, tmp_path, "alice")
    elsewhere["url"] = "http://other"
    recorder = FakeProcess(exit_code=None)
    _relaunch(monkeypatch, [recorder])
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        recording = await ffmpeg_session.record_live(
            "alice",
            "http://live",
            "src",
            str(recording_dir),
            60,
        )
    assert f"Recording {recording_dir} for 60s from upstream" in caplog.messages
    assert recording.relay not in (other["relay"], elsewhere["relay"])
    assert len(upstreams.opened) == 3
    assert env.sessions == {"bob": other, "alice": elsewhere}
    assert worker.signals == []
    assert not recording.relay.finished
    await asyncio.to_thread(recording.stop)
    assert recorder.signals == ["TERM"]
    assert recording.finished()
    await _finish_task(tasks).run()
    await until(lambda: recording.relay.finished)
    assert [upstream.closed for upstream in upstreams.opened] == [False, False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("viewing", [True, False])
async def test_recording_launch_failure_closes_only_a_relay_it_opened(
    env: _Env,
    upstreams: FakeUpstreams,
    recording_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    viewing: bool,
):
    if viewing:
        await _viewer(env, tmp_path, "alice")
    _relaunch(monkeypatch, [], error=OSError("no ffmpeg"))
    with pytest.raises(OSError, match="no ffmpeg"):
        await ffmpeg_session.record_live("alice", "http://live", "src", str(recording_dir), 60)
    (upstream,) = upstreams.opened
    assert upstream.closed is not viewing
    assert set(env.sessions) == ({"alice"} if viewing else set())
    assert not ffmpeg_session._recordings


@pytest.mark.asyncio
async def test_recording_guards_the_recorder_pipes(
    env: _Env,
    tasks: BackgroundTasks,
    recording_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    deaf = FakeProcess(exit_code=None)
    monkeypatch.setattr(deaf, "stdin", None)
    mute = FakeProcess(exit_code=None)
    _relaunch(monkeypatch, [deaf, mute])
    with pytest.raises(ValueError, match=r"^Expected process\.stdin is not None\.$"):
        await ffmpeg_session.record_live("alice", "http://live", "src", str(recording_dir), 60)
    await ffmpeg_session.record_live("alice", "http://live", "src", str(recording_dir), 60)
    mute.stderr = None
    with pytest.raises(ValueError, match=r"^Expected process\.stderr is not None\.$"):
        await _finish_task(tasks).run()


@pytest.mark.asyncio
async def test_shutdown_stops_recordings_and_closes_their_relays(
    env: _Env,
    upstreams: FakeUpstreams,
    recording_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    viewer, worker = await _viewer(env, tmp_path, "bob")
    recorder = FakeProcess(exit_code=None)
    _relaunch(monkeypatch, [recorder])
    recording = await ffmpeg_session.record_live(
        "alice",
        "http://live",
        "src",
        str(recording_dir),
        60,
    )
    await ffmpeg_session.shutdown()
    assert (recorder.signals, worker.signals) == (["TERM"], ["TERM"])
    assert recording.relay.finished
    assert viewer["relay"].finished
    assert [upstream.closed for upstream in upstreams.opened] == [True, True]
    assert not ffmpeg_session._recordings
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
    env.sessions["a3"] = _owned(tmp_path, started=3, username="alice", source_id="src1")
    env.sessions["m1"] = _owned(tmp_path, started=1, username="alice", source_id="src1")
    env.sessions["s2"] = _owned(tmp_path, started=2, username="bob", source_id="src2")
    # Ids sort opposite to start order, so only the start key orders them.
    legacy = _owned(tmp_path, started=99, username="alice", source_id="src1")
    # Sessions recovered from before start times were recorded lack "started".
    monkeypatch.delitem(cast(Mapping[str, object], legacy), "started")
    env.sessions["z-legacy"] = legacy
    assert [sid for sid, _ in ffmpeg_session.get_user_sessions("alice")] == [
        "z-legacy",
        "m1",
        "a3",
    ]
    assert [sid for sid, _ in ffmpeg_session.get_source_sessions("src1")] == [
        "z-legacy",
        "m1",
        "a3",
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
        env.sessions["test"].update(last_access=0, username="alice")
        assert ffmpeg_session.get_session_progress("test", "alice") == {
            "segment_count": 2,
            "duration": 6.5,
        }
        assert env.sessions["test"].get("last_access") == _NOW

    def test_progress_no_playlist(self, env: _Env, tmp_path: pathlib.Path):
        env.sessions["test"] = _live(tmp_path)
        env.sessions["test"]["username"] = "alice"
        assert ffmpeg_session.get_session_progress("test", "alice") == {
            "segment_count": 0,
            "duration": 0.0,
        }

    def test_progress_nonexistent_session(self):
        assert ffmpeg_session.get_session_progress("nonexistent", "alice") is None

    def test_another_users_progress_reads_as_absent_and_keeps_no_heartbeat(
        self,
        env: _Env,
        tmp_path: pathlib.Path,
    ):
        env.sessions["test"] = _live(tmp_path)
        env.sessions["test"].update(last_access=0, username="alice")
        assert ffmpeg_session.get_session_progress("test", "eve") is None
        assert env.sessions["test"].get("last_access") == 0

    def test_progress_reads_the_fast_start_rendition(self, env: _Env, tmp_path: pathlib.Path):
        (tmp_path / "low.m3u8").write_text("#EXTINF:2.0,\nlow_0.ts\n")
        (tmp_path / "up.m3u8").write_text("#EXTINF:1.5,\nup_0.ts\n#EXTINF:1.5,\nup_1.ts\n")
        env.sessions["s"] = _fast_session(tmp_path)
        env.sessions["s"]["username"] = "alice"
        assert ffmpeg_session.get_session_progress("s", "alice") == {
            "segment_count": 1,
            "duration": 2.0,
        }
        env.sessions["s"]["rendition"] = "up"
        assert ffmpeg_session.get_session_progress("s", "alice") == {
            "segment_count": 2,
            "duration": 3.0,
        }


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
        "probe": {
            "video_codec": "hevc",
            "height": 2160,
            "interlaced": True,
            "audio_tracks": [
                {
                    "index": 1,
                    "lang": "eng",
                    "codec": "ac3",
                    "channels": 6,
                    "sample_rate": 48_000,
                    "profile": "",
                    "default": True,
                },
            ],
        },
    }
    _vod_dir(env.root, "p", info)
    ffmpeg_session.cleanup_and_recover_sessions()
    media, subs = ffmpeg_command.probe_media("http://m", 7, 8, source_id="")
    assert media is not None
    assert (media.video_codec, media.height, media.duration, media.interlaced) == (
        "hevc",
        2160,
        99,
        True,
    )
    assert subs == [SubtitleStream(index=2, lang="eng", name="")]
    assert [t.lang for t in media.audio_tracks] == ["eng"]
    assert ffmpeg_command.probe_media("http://m", source_id="") == (media, subs)
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
async def test_start_replaces_own_live_session_for_bandwidth_saver(
    env: _Env,
    tasks: BackgroundTasks,
    upstreams: FakeUpstreams,
    tmp_path: pathlib.Path,
):
    del tasks
    env.run.program("ffprobe", Program(stdout=probed_video_json()))
    _existing(env, tmp_path, "http://live", username="alice")
    old = env.sessions["old"]["process"]
    result = await ffmpeg_session.start_transcode(
        "http://live",
        username="alice",
        bandwidth_saver=True,
    )
    sid = result["session_id"]
    assert set(env.sessions) == {sid}
    assert old is not None
    assert old.returncode == -15
    assert env.sessions[sid].get("bandwidth_saver") is True
    (worker,) = env.launcher.commands
    assert "min(ih\\,720)" in _option(worker, "-vf")
    assert len(upstreams.opened) == 1


@pytest.mark.asyncio
async def test_start_replaces_own_dolby_movie_for_an_aac_client_at_its_position(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "probe_movies": False})
    _existing(env, tmp_path, "http://movie", username="alice", audio_passthrough=True)
    env.sessions["old"]["seek_offset"] = 30
    result = await ffmpeg_session.start_transcode("http://movie", "movie", username="alice")
    sid = result["session_id"]
    assert set(env.sessions) == {sid}
    assert env.sessions[sid].get("seek_offset") == 30
    assert env.sessions[sid].get("audio_passthrough") is False
    (command,) = env.launcher.commands
    assert _option(command, "-ss") == "30"


@pytest.mark.asyncio
async def test_start_replaces_own_movie_in_another_audio_language_at_its_position(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "probe_movies": False})
    _existing(env, tmp_path, "http://movie", username="alice")
    env.sessions["old"].update(seek_offset=30, audio_lang="fr", is_vod=True)
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        result = await ffmpeg_session.start_transcode("http://movie", "movie", username="alice")
    sid = result["session_id"]
    assert set(env.sessions) == {sid}
    assert env.sessions[sid].get("audio_lang") == "en"
    (command,) = env.launcher.commands
    assert _option(command, "-ss") == "30"
    assert "Restarting session old for audio language" in [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_start_reuses_own_movie_in_the_users_audio_language(
    env: _Env,
    tmp_path: pathlib.Path,
):
    _existing(env, tmp_path, "http://movie", username="alice")
    env.sessions["old"].update(audio_lang="en", is_vod=True)
    result = await ffmpeg_session.start_transcode("http://movie", "movie", username="alice")
    assert result["session_id"] == "old"
    assert env.launcher.commands == []


@pytest.mark.asyncio
async def test_start_replaces_own_plain_live_session_without_reusing_it(
    env: _Env,
    tasks: BackgroundTasks,
    upstreams: FakeUpstreams,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
):
    del tasks
    env.run.program("ffprobe", Program(stdout=probed_video_json()))
    _existing(env, tmp_path, "http://live", username="alice")
    env.sessions["old"]["audio_lang"] = "fr"
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        result = await ffmpeg_session.start_transcode("http://live", username="alice")
    assert result["session_id"] != "old"
    assert len(upstreams.opened) == 1
    about_old = [m for m in (r.getMessage() for r in caplog.records) if m.endswith(" old")]
    assert about_old == ["Cleaning up invalid session old", "Stopped transcode session old"]


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
    _existing(env, tmp_path, "http://movie")
    env.sessions["old"]["is_vod"] = True
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        result = await ffmpeg_session.start_transcode("http://movie", "movie")
    assert (result["session_id"], result["playlist"]) == ("old", "/transcode/old/stream.m3u8")
    assert env.launcher.commands == []
    assert [r.getMessage() for r in caplog.records] == [
        "Found valid existing VOD session old",
        "Reusing active session old",
    ]


@pytest.mark.asyncio
async def test_a_shared_vod_session_admits_only_those_it_was_handed_to(
    env: _Env,
    tmp_path: pathlib.Path,
):
    _existing(env, tmp_path, "http://movie", username="bob")
    env.sessions["old"]["is_vod"] = True
    assert ffmpeg_session.user_session("old", "alice") is None
    result = await ffmpeg_session.start_transcode("http://movie", "movie", username="alice")
    assert result["session_id"] == "old"
    assert [
        ffmpeg_session.user_session(session_id, name) is not None
        for session_id, name in [("old", "alice"), ("old", "bob"), ("old", "eve"), ("gone", "bob")]
    ] == [True, True, False, False]


@pytest.mark.asyncio
async def test_start_replaces_an_expired_session(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
):
    del tasks
    _existing(env, tmp_path, "http://movie", username="alice", last_access=_NOW - 60)
    env.sessions["old"]["is_vod"] = True
    old = env.sessions["old"]["process"]
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        result = await ffmpeg_session.start_transcode("http://movie", "movie", username="alice")
    sid = result["session_id"]
    assert set(env.sessions) == {sid} != {"old"}
    assert env.urls == {"http://movie": sid}
    assert old is not None
    assert old.returncode == -15
    messages = [r.getMessage() for r in caplog.records]
    assert "Cleaning up invalid session old" in messages
    assert not any(message.startswith("Found valid") for message in messages)


@pytest.mark.asyncio
async def test_start_replaces_a_dead_session_even_when_just_accessed(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
):
    """A recent heartbeat protects a session from casual stops, not from replacement."""
    del tasks
    _existing(env, tmp_path, "http://movie", username="alice", last_access=_NOW - 1)
    env.sessions["old"].update(process=FakeProcess(exited=True), is_vod=True)
    result = await ffmpeg_session.start_transcode("http://movie", "movie", username="alice")
    sid = result["session_id"]
    assert set(env.sessions) == {sid}
    assert not (tmp_path / "old").exists()
    # A session without a recorded position starts from the beginning.
    assert "-ss" not in env.launcher.commands[0]


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
    cache.save_server_settings({**cache.load_server_settings(), "probe_series": False})
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
async def test_start_fast_live_without_dvr(
    env: _Env,
    tasks: BackgroundTasks,
    upstreams: FakeUpstreams,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "live_dvr_mins": 0})
    result = await ffmpeg_session.start_transcode("http://live")
    assert result.get("master_playlist") == f"/transcode/{result['session_id']}/master.m3u8"
    session = env.sessions[result["session_id"]]
    assert (session.get("username"), session.get("source_id")) == ("", "")
    (worker,) = env.launcher.commands
    assert _option(worker, "-i") == "pipe:0"
    assert "yadif" in _option(worker, "-vf")
    # The relay is the upstream's only reader; probes read its local sample.
    assert len(upstreams.opened) == 1
    assert env.run.calls
    assert all(call[-1].endswith("/probe.ts") for call in env.run.calls)


@pytest.mark.asyncio
async def test_start_retries_movie_with_a_stale_cached_probe(
    env: _Env,
    tasks: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
):
    del tasks
    # The retry probes because it is a movie, not because series are probed.
    cache.save_server_settings({**cache.load_server_settings(), "probe_series": False})
    env.run.program("ffprobe", Program(stdout=_DOLBY_PROBE))
    stale = MediaInfo(video_codec="h264", audio_codec="aac", pix_fmt="yuv420p", height=720)
    ffmpeg_command.remember_probe("http://movie", stale)
    launcher = _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    result = await ffmpeg_session.start_transcode(
        "http://movie",
        "movie",
        username="alice",
        audio_passthrough=True,
    )
    assert env.sessions[result["session_id"]].get("audio_passthrough") is True
    assert len(launcher.commands) == 2
    assert ffmpeg_command.probe_media("http://movie", source_id="")[0] != stale


@pytest.mark.asyncio
async def test_start_does_not_retry_a_freshly_probed_movie(
    env: _Env,
    monkeypatch: pytest.MonkeyPatch,
):
    env.run.program("ffprobe", Program(stdout=_DOLBY_PROBE))
    launcher = _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    with pytest.raises(HTTPException):
        await ffmpeg_session.start_transcode("http://movie", "movie", username="alice")
    assert len(launcher.commands) == 1


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
@pytest.mark.parametrize("deinterlace", [False, True])
async def test_retry_keeps_the_requested_deinterlacing(
    env: _Env,
    tasks: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
    deinterlace: bool,
):
    del tasks
    cache.save_server_settings({**cache.load_server_settings(), "transcode_hw": "software"})
    ffmpeg_command.restore_probe_cache_entry(
        "http://other",
        MediaInfo(video_codec="h264", audio_codec="aac", pix_fmt="yuv420p"),
        [],
        3,
        1,
    )
    launcher = _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    result = await ffmpeg_session.start_transcode(
        "http://ep",
        "series",
        series_id=3,
        deinterlace_fallback=deinterlace,
    )
    assert env.sessions[result["session_id"]].get("series_id") == 3
    first, retried = launcher.commands
    assert ("yadif" in _option(first, "-vf"), "yadif" in _option(retried, "-vf")) == (
        deinterlace,
        deinterlace,
    )


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

    ffmpeg_command.restore_probe_cache_entry(
        "http://other",
        MediaInfo(video_codec="h264", audio_codec="aac", pix_fmt="yuv420p"),
        [],
        3,
        1,
    )
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


@pytest.mark.parametrize(
    "max_resolution,bandwidth_saver,resolution",
    [(None, False, "1080p"), ("4k", False, "4k"), ("4k", True, "720p"), ("480p", True, "480p")],
)
def test_live_resolution_caps_saver_at_720p_without_raising_480p(
    max_resolution: cache.MaxResolution | None,
    bandwidth_saver: bool,
    resolution: str,
):
    _save_max_resolution(max_resolution)
    settings = cache.load_server_settings()
    assert ffmpeg_session._live_resolution(settings, bandwidth_saver=bandwidth_saver) == resolution


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_resolution,height,audio,passthrough,dvr_mins,remux",
    [
        ("4k", 2160, "aac", False, 1, True),
        (None, 1440, "aac", False, 60, False),
        ("1080p", 2160, "aac", False, 60, False),
        ("1080p", 1080, "ac3", True, 60, True),
        ("1080p", 1080, "aac", False, 0, False),
    ],
)
async def test_dvr_start_copies_only_copyable_sources(
    env: _Env,
    fast_env: BackgroundTasks,
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
    env.run.program("ffprobe", Program(stdout=json.dumps(source)), when="-show_streams")
    await ffmpeg_session.start_transcode("http://live", audio_passthrough=passthrough)
    (worker,) = env.launcher.commands
    assert _option(worker, "-i") == "pipe:0"
    assert ("-c" in worker and _option(worker, "-c") == "copy") is remux
    # Routing never probes the upstream URL.
    assert all(call[-1] != "http://live" for call in env.run.calls)


@pytest.mark.asyncio
async def test_start_passes_viewer_choices_to_fast_start(env: _Env, fast_env: BackgroundTasks):
    del fast_env
    _save_max_resolution("4k")
    env.run.program(
        "ffprobe",
        Program(
            stdout=json.dumps(
                {
                    "streams": [
                        {"codec_type": "video", "codec_name": "h264"},
                        {
                            "codec_type": "audio",
                            "codec_name": "eac3",
                            "channels": 6,
                            "sample_rate": "48000",
                        },
                    ],
                },
            ),
        ),
        when="-show_streams",
    )
    result = await ffmpeg_session.start_transcode(
        "http://live",
        deinterlace_fallback=False,
        audio_passthrough=True,
        bandwidth_saver=True,
        username="alice",
        source_id="src",
    )
    session = env.sessions[result["session_id"]]
    assert (session.get("username"), session.get("source_id")) == ("alice", "src")
    # Its relay dials by its source's rule.
    assert ffmpeg_session._is_fast_live(session)
    assert session["plan"].source_id == "src"
    assert session.get("bandwidth_saver") is True
    assert session.get("audio_passthrough") is True
    (encoder,) = env.launcher.commands
    assert "yadif" not in _option(encoder, "-vf")
    assert "min(ih\\,720)" in _option(encoder, "-vf")
    assert _option(encoder, "-c:a") == "copy"


@pytest.mark.asyncio
async def test_fast_start_gives_up_when_the_client_disconnects(
    env: _Env,
    fast_env: BackgroundTasks,
    upstreams: FakeUpstreams,
):
    del fast_env
    _save_max_resolution("4k")
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode("http://live", is_disconnected=_disconnected)
    assert (error.value.status_code, error.value.detail) == (499, "Playback request disconnected")
    (worker,) = env.launcher.processes
    assert worker.signals == ["TERM"]
    assert upstreams.opened[0].closed
    assert not env.sessions
    assert not list(env.root.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,upstream",
    [("http://live", "open"), ("http://gone", "<urlopen error no route to http://gone>")],
)
async def test_failed_fast_start_reports_the_worker_exit_and_the_upstream(
    env: _Env,
    fast_env: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    url: str,
    upstream: str,
):
    del fast_env
    _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode(url)
    assert (error.value.status_code, error.value.detail) == (
        500,
        "Fast-start stream failed to become ready",
    )
    assert f"(exit=1, upstream={upstream})" in caplog.text
    assert not env.sessions
    assert not list(env.root.iterdir())


@pytest.mark.asyncio
async def test_start_passes_the_request_to_the_transcoder(env: _Env, tasks: BackgroundTasks):
    del tasks
    result = await ffmpeg_session.start_transcode(
        "http://ep",
        "series",
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
    # Unprobed video is deinterlaced unless the viewer turned it off.
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
        "audio_passthrough": True,
        "audio_lang": "en",
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
        f"Starting transcode session {sid}: {' '.join(command)}",
    ]


@pytest.mark.asyncio
async def test_do_start_logs_a_sparse_probe(
    env: _Env,
    tasks: BackgroundTasks,
    caplog: pytest.LogCaptureFixture,
):
    del tasks
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
        await ffmpeg_session._do_start_transcode("http://movie", "movie", None, None, 0)
    # No bitrate, no format duration, and no subtitle streams.
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
async def test_do_start_applies_the_quality_setting(env: _Env, tasks: BackgroundTasks):
    del tasks
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "probe_movies": False,
            "quality": "low",
            "transcode_hw": "software",
        },
    )
    result = await ffmpeg_session._do_start_transcode("http://movie", "movie", None, None, 0)
    assert env.run.calls == []
    assert env.sessions[result["session_id"]].get("is_vod") is True
    (command,) = env.launcher.commands
    assert _option(command, "-crf") == "32"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seek_offset,exit_code,waited,logged_exit",
    # Past the deadline, only a process still running holds the report back (1 s).
    [(0, None, 31.0, -1), (0.5, 0, 40.0, -1), (42.0, 3, 40.0, 3)],
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
    assert sum(env.clock.sleeps) == pytest.approx(waited, abs=0.05)
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
            "probe_movies": False,
            "probe_series": False,
        },
    )
    return env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,upscale",
    [(ARCHIVE_URL, False), ("https://upstream.test/movie/1.mp4", True)],
)
async def test_archive_bypasses_upscale_without_changing_movies(
    archive: _Env,
    url: str,
    upscale: bool,
):
    result = await ffmpeg_session.start_transcode(url, "movie")
    [command] = archive.launcher.commands
    video_filter = command[command.index("-vf") + 1]
    assert ("dnn_processing=" in video_filter) is upscale
    assert "2160" in video_filter
    assert command[command.index("-qp") + 1] == "20"
    session = archive.sessions[result["session_id"]]
    assert session.get("is_archive") is (not upscale)
    assert session.get("request_url") == url
    # Archives are never resumed after a restart, so they persist no session.json.
    assert (pathlib.Path(session["dir"]) / "session.json").exists() is upscale


@pytest.mark.asyncio
async def test_fast_start_reads_the_best_variant_through_one_remuxer(
    env: _Env,
    fast_env: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
):
    del fast_env
    # A home-network HLS source, so its remuxer may reach the home network.
    _save_sources({"id": "h", "name": "Home", "type": "m3u", "url": "http://192.168.1.50/l.m3u"})
    master = "http://192.168.1.50/live/1.m3u8"
    variant = "http://192.168.1.50/live/1_hi.m3u8"
    env.web.routes[master] = FakeResponse(
        b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=2000000\n1_lo.m3u8\n"
        b"#EXT-X-STREAM-INF:BANDWIDTH=8000000\n1_hi.m3u8\n",
    )
    remuxers = PopenRecorder()
    install_subprocess(monkeypatch, ffmpeg_session, popen=remuxers)
    task = asyncio.create_task(ffmpeg_session.start_transcode(master, "live", source_id="h"))
    await until(lambda: remuxers.started)
    (remuxer,) = remuxers.started
    try:
        remuxer.stdout.feed(ts_source())
        result = await task
        assert env.web.urls == [master]
        assert remuxer.args == fast_start.source_command(variant, cache.get_user_agent(), "h")
        assert remuxer.args[remuxer.args.index("-http_proxy") + 1] == proxy.egress().url(lan=True)
        assert remuxer.options == {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "env": proxy.ffmpeg_env(),
        }
        session = env.sessions[result["session_id"]]
        assert ffmpeg_session._is_fast_live(session)
        # Each connection resolves anew: variant URLs carry tokens that expire.
        assert session["plan"].source_url == master
        # Reuse and casting match the URL the client asked for.
        assert session["url"] == master
        assert env.urls[master] == result["session_id"]
        await ffmpeg_session.stop_session(result["session_id"], force=True)
        # Closing the relay stops the remuxer and its pipe.
        assert set(remuxer.signals) == {"KILL"}
        assert remuxer.stdout.closed
    finally:
        remuxer.stdout.close()


def test_direct_sources_open_once_with_a_burst_tolerant_timeout(
    env: _Env,
    upstreams: FakeUpstreams,
):
    """Panels send TS in ~10 s segment bursts; 30 s of silence ends the relay for the watchdog."""
    body = ffmpeg_session._open_source("http://live", "agent/1", "s")
    assert body is upstreams.opened[0]
    (fetch,) = env.web.fetches
    assert (fetch.url, fetch.timeout, fetch.headers) == (
        "http://live",
        30,
        {"User-agent": "agent/1"},
    )
    # A provider's redirect cannot point the stream at this server or the home network.
    assert testing.dials_only_the_internet(fetch)


@pytest.mark.asyncio
async def test_a_tuner_a_home_lineup_lists_is_read_on_the_home_network(
    env: _Env,
    upstreams: FakeUpstreams,
    recording_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    tuner = "http://192.168.1.87:5004/auto/v5.1"

    def tuned() -> FakeUpstream:
        upstream = upstreams()
        upstream.feed(ts_source())
        return upstream

    env.web.routes[tuner] = tuned
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "sources": [
                {
                    "id": "t",
                    "name": "Tuner",
                    "type": "m3u",
                    "url": "http://192.168.1.24/lineup.m3u",
                },
            ],
        },
    )
    # A live session's relay, and a recording's, may dial the tuner.
    stream = _plan(tuner, source_id="t").relay()
    await stream.start()
    await until(lambda: upstreams.opened)
    await stream.close()
    _relaunch(monkeypatch, [], error=OSError("no ffmpeg"))
    with pytest.raises(OSError, match="no ffmpeg"):
        await ffmpeg_session.record_live("alice", tuner, "t", str(recording_dir), 60)
    assert len(env.web.fetches) == 2
    for fetch in env.web.fetches:
        vets = testing.address_vets(fetch)
        assert vets and all(vet(ipaddress.ip_address("192.168.1.87")) for vet in vets)


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
    content_type: ffmpeg_session.VodType,
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
    assert await ffmpeg_session._try_reuse_session("s", "http://ep", content_type) is not None
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
    cache.save_server_settings(
        {**cache.load_server_settings(), "transcode_hw": "software", "probe_series": False},
    )
    (tmp_path / "seg000.ts").write_bytes(b"x" * 2000)
    env.sessions["s"] = ffmpeg_session.TranscodeSession(
        dir=str(tmp_path),
        process=FakeProcess(exited=True),
        url="http://ep",
        is_vod=True,
        started=_NOW,
    )
    assert await ffmpeg_session._try_reuse_session("s", "http://ep", "series") is not None
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
async def test_seek_into_a_dead_transcode_restarts_ffmpeg(env: _Env, seek_dir: pathlib.Path):
    # Ffmpeg stopped (a full disk); a segment it left behind cannot carry playback on.
    env.sessions["v"]["process"] = FakeProcess(exited=True, exit_code=134)
    (seek_dir / "seg001.ts").write_bytes(b"x" * 2000)
    result = await ffmpeg_session.seek_transcode("v", 4.0)
    assert result == {"ok": True, "segment": 1, "time": 4.0}
    (command,) = env.launcher.commands
    # Before the playlist: ffmpeg ignores options after its last output.
    assert command[-3:-1] == ["-start_number", "1"] and command[-1].endswith("stream.m3u8")


def _served(session_id: str, number: int) -> int:
    return ffmpeg_session.note_segment_served(session_id, f"seg{number:03d}.ts", keep_behind_sec=6)


def test_segments_far_behind_playback_are_deleted(seek_dir: pathlib.Path):
    for number in range(8):
        (seek_dir / f"seg{number:03d}.ts").write_bytes(b"x")
    assert _served("v", 1) == 0
    assert _served("v", 4) == 2
    assert sorted(p.name for p in seek_dir.glob("seg*.ts"))[:2] == ["seg002.ts", "seg003.ts"]
    assert _served("v", 6) == 2
    assert not (seek_dir / "seg003.ts").exists()
    # A seek back regenerates earlier segments; moving forward again evicts those too.
    for number in (0, 1, 2):
        (seek_dir / f"seg{number:03d}.ts").write_bytes(b"x")
    assert _served("v", 1) == 0
    assert _served("v", 5) == 3
    assert {p.name for p in seek_dir.glob("seg*.ts")} == {
        "seg004.ts",
        "seg005.ts",
        "seg006.ts",
        "seg007.ts",
    }


@pytest.mark.parametrize(
    "filename",
    ["stream.m3u8", "sub0_000001.vtt", "segxyz.ts", "seg001.vtt", "missing", "seg.ts"],
)
def test_only_vod_segments_count_as_served(seek_dir: pathlib.Path, filename: str):
    (seek_dir / "seg000.ts").write_bytes(b"x")
    assert ffmpeg_session.note_segment_served("v", filename, keep_behind_sec=0) == 0
    assert (seek_dir / "seg000.ts").exists()


def test_served_segments_of_live_or_unknown_sessions_are_kept(
    env: _Env,
    seek_dir: pathlib.Path,
):
    (seek_dir / "seg000.ts").write_bytes(b"x")
    assert ffmpeg_session.note_segment_served("missing", "seg009.ts", keep_behind_sec=0) == 0
    env.sessions["v"]["fast_start"] = True
    assert ffmpeg_session.note_segment_served("v", "seg009.ts", keep_behind_sec=0) == 0
    env.sessions["v"]["fast_start"] = False
    env.sessions["v"]["is_vod"] = False
    assert ffmpeg_session.note_segment_served("v", "seg009.ts", keep_behind_sec=0) == 0
    assert (seek_dir / "seg000.ts").exists()


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
    # Before the playlist: ffmpeg ignores options after its last output.
    assert command[-3:-1] == ["-start_number", "3"] and command[-1].endswith("stream.m3u8")
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
    cache.save_server_settings({**cache.load_server_settings(), "probe_series": False})
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
    env.sessions["live"] = _fast_health_session(tmp_path, bandwidth_saver=False)
    env.sessions["live"].update(username="viewer", last_access=0, closed_captions=False)
    health = PlaybackHealth(buffer_seconds=0, waiting=True)
    decisions: list[ffmpeg_session.HealthResponse] = []
    for now in range(0, 26, 2):
        env.clock.now = now
        decisions.append(ffmpeg_session.report_playback_health("live", "viewer", health))
    assert decisions[-2] == {"bandwidth_saver": False, "playlist": "/transcode/live/low.m3u8"}
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
    env.sessions["live"] = _fast_health_session(tmp_path, bandwidth_saver=False)
    env.sessions["live"]["closed_captions"] = False
    for required in (3e6, 2e6):
        health = PlaybackHealth(buffer_seconds=10, waiting=False, required_bitrate=required)
        assert ffmpeg_session.report_playback_health("live", "u", health) == {
            "bandwidth_saver": False,
            "playlist": "/transcode/live/low.m3u8",
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
    env.sessions["f"] = _fast_health_session(tmp_path, bandwidth_saver=False, recovery_bitrate=1e6)
    env.sessions["f"].update(process=encoder, playlist_generation=2)
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


def _fast_health_session(
    tmp_path: pathlib.Path,
    *,
    bandwidth_saver: bool,
    recovery_bitrate: float = 0,
) -> ffmpeg_session.FastLiveSession:
    """A captioned fast-start session owned by "u" whose worker has exited."""
    session = _fast_session(tmp_path, FakeProcess(exited=True))
    session.update(
        username="u",
        fast_start=True,
        bandwidth_saver=bandwidth_saver,
        playback_policy=PlaybackPolicy(bandwidth_saver=bandwidth_saver),
        closed_captions=True,
        recovery_bitrate=recovery_bitrate,
    )
    return session


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


@pytest.fixture
def fast_env(env: _Env, tasks: BackgroundTasks) -> BackgroundTasks:
    """Software 4k fast start; ffprobe finds 1080p video with embedded captions."""
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "transcode_hw": "software",
            "max_resolution": "4k",
            "quality": "high",
        },
    )
    env.run.program("ffprobe", Program(stdout=probed_video_json()), when="-show_streams")
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
async def test_start_fast_live_runs_one_worker_fed_by_the_relay(
    env: _Env,
    fast_env: BackgroundTasks,
    upstreams: FakeUpstreams,
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
    (launch,) = env.launcher.launches
    worker = launch.command
    assert launch.options["stdin"] == asyncio.subprocess.PIPE
    assert _option(worker, "-i") == "pipe:0"
    assert _option(worker, "-b:v") == bitrate
    assert f"min(ih\\,{height})" in _option(worker, "-vf")
    assert "append_list" not in _option(worker, "-hls_flags")
    # The worker gets the stream from its keyframe: here, everything since the first byte.
    await launch.process.stdin.wait_for(len(ts_source()))
    assert bytes(launch.process.stdin.written) == ts_source()
    directory = pathlib.Path(session["dir"])
    assert (directory / "probe.ts").read_bytes() == ts_source()
    assert session["process"] is launch.process
    assert "subscription" in session
    audio_info = session.get("audio_info")
    assert audio_info is not None
    assert (audio_info.video_codec, audio_info.height) == ("mpeg2video", 1080)
    assert session.get("audio_passthrough") is False
    assert session.get("watchdog_task") is fast_env.tasks[-1]
    assert _coroutine_names(fast_env) == ["_monitor_fast_process", "_watch_fast_live"]
    assert session.get("closed_captions") is True
    # The session clock starts at the source's first timestamp, when its first byte arrived.
    assert (session.get("origin_pts"), session.get("origin_time")) == (10, _NOW)
    assert session.get("watchdog_stale_after") == 45.0
    assert session.get("playback_buffer_seconds") == 12.0
    assert session["plan"].resolution == resolution
    assert session.get("recovery_bitrate") == pytest.approx(22_000_000)
    assert (directory / "master.m3u8").read_text() == master_playlist(
        resolution,
        audio_bitrate=384_000,
    )
    assert (directory / "low-captions.m3u8").read_text() == (
        fast_start.closed_caption_master_playlist(resolution, audio_bitrate=384_000)
    )
    assert result.get("caption_playlist") == f"/transcode/{sid}/low-captions.m3u8"
    assert len(upstreams.opened) == 1


def _panel() -> str:
    """Configure an Xtream panel and an M3U list; return the panel's player_api URL."""
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "sources": [
                {
                    "id": "x",
                    "name": "Panel",
                    "type": "xtream",
                    "url": "https://panel",
                    "username": "u",
                    "password": "p",
                },
                {"id": "m", "name": "List", "type": "m3u", "url": "https://list"},
            ],
        },
    )
    return "https://panel/player_api.php?username=u&password=p"


def test_xtream_live_reads_mpegts_where_the_panel_allows_it(env: _Env):
    """Panels serve live as continuous MPEG-TS: no playlist round trips, no remuxer."""
    api = _panel()
    info = {"user_info": {"allowed_output_formats": ["m3u8", "ts"]}}
    env.web.routes[api] = lambda: FakeResponse(json.dumps(info).encode())
    hls = "https://panel/live/u/p/7.m3u8"
    assert ffmpeg_session._relay_url(hls, "x") == "https://panel/live/u/p/7.ts"
    assert ffmpeg_session._relay_url("https://panel/live/u/p/8.m3u8", "x").endswith("/8.ts")
    # The panel's formats are asked once per source.
    assert env.web.urls == [api]
    # A list's HLS URL is the only format it offers; other URLs are already TS.
    assert ffmpeg_session._relay_url(hls, "m") == hls
    assert ffmpeg_session._relay_url(hls, "gone") == hls
    assert ffmpeg_session._relay_url("https://panel/live/u/p/7.ts", "x").endswith("/7.ts")
    assert env.web.urls == [api]


@pytest.mark.parametrize(
    "answer",
    [
        {"user_info": {"allowed_output_formats": ["m3u8"]}},
        {"user_info": {"allowed_output_formats": "ts"}},
        {"user_info": []},
        {},
    ],
)
def test_xtream_live_stays_hls_where_ts_is_not_offered(env: _Env, answer: object):
    api = _panel()
    env.web.routes[api] = lambda: FakeResponse(json.dumps(answer).encode())
    hls = "https://panel/live/u/p/7.m3u8"
    assert ffmpeg_session._relay_url(hls, "x") == hls
    assert ffmpeg_session._relay_url(hls, "x") == hls
    assert env.web.urls == [api]


def test_unreachable_panel_api_keeps_hls_and_asks_again(
    env: _Env,
    caplog: pytest.LogCaptureFixture,
):
    api = _panel()
    hls = "https://panel/live/u/p/7.m3u8"
    assert ffmpeg_session._relay_url(hls, "x") == hls
    assert ffmpeg_session._relay_url(hls, "x") == hls
    assert env.web.urls == [api, api]
    assert [record.getMessage() for record in caplog.records] == [
        "Xtream source x did not report its output formats",
    ] * 2


@pytest.mark.asyncio
async def test_sources_that_deliver_in_bursts_start_with_a_reserve_that_bridges_them(
    env: _Env,
    fast_env: BackgroundTasks,
    upstreams: FakeUpstreams,
):
    """A source that went silent for 10 s between bursts gets 10 s of output before playing."""
    del fast_env
    first = await ffmpeg_session._start_fast_live("http://live", "alice", "src", False)
    session = env.sessions[first["session_id"]]
    assert ffmpeg_session._is_fast_live(session)
    # Continuous so far: two 4 s segments suffice and clients hold the 12 s default.
    assert first.get("playback_buffer_seconds") == 12.0
    # A pattern: one silence alone could be a stall.
    for pts in (11, 21, 31):
        env.clock.advance(10)
        upstreams.opened[0].feed(video_packet(pts))
        await session["relay"].wait_for_bytes(len(session["relay"].history()) + 1)
    await ffmpeg_session.stop_session(first["session_id"], force=True)
    # The next session on this source waits for output covering the gap.
    task = asyncio.create_task(ffmpeg_session._start_fast_live("http://live", "bob", "src", False))
    await until(lambda: len(env.launcher.launches) == 2)
    directory = pathlib.Path(env.launcher.launches[1].command[-1]).parent
    for _ in range(20):
        await asyncio.sleep(0)
    assert not task.done()
    _media_playlist(directory / "low.m3u8", ["low_1.ts", "low_2.ts", "low_3.ts"], duration=4.0)
    second = await task
    assert second.get("playback_buffer_seconds") == 20.0
    assert env.sessions[second["session_id"]].get("watchdog_stale_after") == 45.0
    await ffmpeg_session.stop_session(second["session_id"], force=True)


@pytest.mark.asyncio
async def test_a_client_holding_its_own_lead_is_answered_at_the_first_segment(
    env: _Env,
    fast_env: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
):
    """The web player waits for its own buffer, so one segment is enough to answer it."""
    del fast_env

    def first_segment(command: list[str]) -> None:
        _media_playlist(pathlib.Path(command[-1]), ["low_000000.ts"], duration=0.25)

    install_launcher(monkeypatch, ProcessLauncher(on_launch=first_segment), ffmpeg_session)
    result = await ffmpeg_session.start_transcode("http://live", min_segments=1)
    assert result["session_id"] in env.sessions
    await ffmpeg_session.stop_session(result["session_id"], force=True)


@pytest.mark.asyncio
async def test_a_prestarted_live_session_waits_for_its_page(env: _Env, fast_env: BackgroundTasks):
    """The play page starts the stream while its scripts load; the page then joins it."""
    ffmpeg_session.prestart_live(
        "http://live",
        username="alice",
        source_id="src",
        deinterlace_fallback=False,
        user_max_streams=0,
        source_max_streams=0,
    )
    assert _coroutine_names(fast_env) == ["_prestart"]
    await fast_env.tasks[0].run()
    [session_id] = env.sessions
    # Nobody watches yet: the prestart is not a viewer that would hold the stream.
    assert env.sessions[session_id].get("viewers") == {}

    page = await ffmpeg_session.start_transcode(
        "http://live",
        username="alice",
        source_id="src",
        min_segments=1,
    )

    assert page["session_id"] == session_id
    session = env.sessions[session_id]
    assert ffmpeg_session._is_fast_live(session)
    assert list(session["viewers"]) == [page.get("viewer_id")]
    assert len(env.launcher.launches) == 1
    await ffmpeg_session.stop_session(session_id, force=True)


class _Starts:
    """Stands in for ``start_transcode``, answering with ``response``."""

    def __init__(self, response: ffmpeg_session.TranscodeResponse) -> None:
        self.response = response
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args: object, **kwargs: object) -> ffmpeg_session.TranscodeResponse:
        self.calls.append((args, kwargs))
        return self.response


@pytest.mark.asyncio
async def test_a_prestart_asks_for_the_page_stream_answered_at_one_segment(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    session = _watched(env, tmp_path, {"prestart": _NOW, "page": _NOW})
    response: ffmpeg_session.TranscodeResponse = {
        "session_id": "s",
        "playlist": "/transcode/s/low.m3u8",
        "subtitles": [],
        "duration": 0,
        "seek_offset": 0,
        "viewer_id": "prestart",
    }
    starts = _Starts(response)
    monkeypatch.setattr(ffmpeg_session, "start_transcode", starts)
    ffmpeg_session.prestart_live(
        "http://live",
        username="alice",
        source_id="src",
        deinterlace_fallback=False,
        user_max_streams=2,
        source_max_streams=3,
    )
    await tasks.tasks[0].run()
    assert starts.calls == [
        (
            ("http://live", "live"),
            {
                "deinterlace_fallback": False,
                "username": "alice",
                "source_id": "src",
                "user_max_streams": 2,
                "source_max_streams": 3,
                "min_segments": 1,
            },
        ),
    ]
    # Its own viewer is released; the page keeps watching.
    assert session["viewers"] == {"page": _NOW}


@pytest.mark.asyncio
async def test_a_prestart_answered_without_a_viewer_releases_nothing(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    session = _watched(env, tmp_path, {"page": _NOW})
    response: ffmpeg_session.TranscodeResponse = {
        "session_id": "s",
        "playlist": "/transcode/s/low.m3u8",
        "subtitles": [],
        "duration": 0,
        "seek_offset": 0,
    }
    monkeypatch.setattr(ffmpeg_session, "start_transcode", _Starts(response))
    ffmpeg_session.prestart_live(
        "http://live",
        username="alice",
        source_id="src",
        deinterlace_fallback=True,
        user_max_streams=0,
        source_max_streams=0,
    )
    await tasks.tasks[0].run()
    assert env.sessions == {"s": session}
    assert session["viewers"] == {"page": _NOW}
    assert len(tasks.tasks) == 1


@pytest.mark.asyncio
async def test_a_prestart_that_cannot_start_leaves_it_to_the_page(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
):
    env.sessions["bob"] = _live(tmp_path)
    env.sessions["bob"].update(username="bob", source_id="src")
    ffmpeg_session.prestart_live(
        "http://live",
        username="alice",
        source_id="src",
        deinterlace_fallback=False,
        user_max_streams=0,
        source_max_streams=1,
    )
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        await tasks.tasks[0].run()
    assert set(env.sessions) == {"bob"}
    assert "Live prestart did not start: Source at capacity (1 streams)" in caplog.messages


def test_master_without_known_captions_declares_none(tmp_path: pathlib.Path):
    session = _fast_session(tmp_path)
    session["master_audio_bitrate"] = 0
    ffmpeg_session._publish_fast_master(session)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["master.m3u8"]
    assert (tmp_path / "master.m3u8").read_text() == master_playlist("1080p")


# =============================================================================
# AI Upscale rendition
# =============================================================================


@pytest.fixture
def upscaler(
    fast_env: BackgroundTasks,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> BackgroundTasks:
    """An NVENC host at 4k with an AI Upscale engine for the 1080p source ffprobe finds."""
    engines = tmp_path / "engines"
    engines.mkdir()
    (engines / "2x-nomosuni-compact_1080p_fp16.engine").write_bytes(b"engine")
    monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", str(engines))
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "transcode_hw": "nvenc+software",
            "sr_model": "2x-nomosuni-compact",
        },
    )
    return fast_env


def _upgrade_task(background: BackgroundTasks) -> FakeTask:
    (task,) = [
        task
        for task, name in zip(background.tasks, _coroutine_names(background), strict=True)
        if name == "_upgrade_to_upscale"
    ]
    return task


def _upscales(command: Sequence[str]) -> bool:
    return "dnn_processing=" in _option(command, "-vf")


@pytest.mark.asyncio
async def test_startup_switches_to_the_ai_upscale_rendition_once_it_caught_up(
    env: _Env,
    upscaler: BackgroundTasks,
    caplog: pytest.LogCaptureFixture,
):
    """TensorRT loads slowly, so playback starts plain and moves to "up" at a new generation."""
    result = await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    session = env.sessions[result["session_id"]]
    assert ffmpeg_session._is_fast_live(session)
    (plain,) = env.launcher.launches
    assert plain.command[-1].endswith("/low.m3u8")
    assert not _upscales(plain.command)
    assert _coroutine_names(upscaler) == [
        "_monitor_fast_process",
        "_upgrade_to_upscale",
        "_watch_fast_live",
    ]
    plain_subscription = session.get("subscription")
    assert plain_subscription is not None
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        await _upgrade_task(upscaler).run()
    _, up = env.launcher.launches
    assert up.command[-1].endswith("/up.m3u8")
    assert _option(up.command, "-hls_segment_filename").endswith("/up_%06d.ts")
    assert _upscales(up.command)
    assert up.options["stdin"] == asyncio.subprocess.PIPE
    await up.process.stdin.wait_for(len(ts_source()))
    assert bytes(up.process.stdin.written) == ts_source()
    assert session["process"] is up.process
    assert session.get("subscription") is not plain_subscription
    assert session.get("rendition") == "up"
    assert session.get("playlist_generation") == 1
    assert ffmpeg_session.playlist_file(session) == "up.m3u8"
    assert plain.process.signals == ["TERM"]
    await plain_subscription.wait_closed()
    assert plain.process.stdin.closed
    assert _coroutine_names(upscaler)[-1] == "_monitor_fast_process"
    assert (
        f"Fast-start session {result['session_id']} switched to its AI Upscale rendition"
        in caplog.messages
    )
    # Clients keep reading low.m3u8: the plain rendition's final playlist, continued
    # by the first upscaled segment past its end.
    directory = pathlib.Path(session["dir"])
    low = (directory / "low.m3u8").read_text()
    assert session.get("splice_low") == low
    assert ffmpeg_session.live_playlist(result["session_id"]) == low
    _media_playlist(
        directory / "up.m3u8",
        ["up_000009.ts", "up_000010.ts", "up_000011.ts"],
        duration=4.0,
    )
    assert ffmpeg_session.live_playlist(result["session_id"]) == fast_start.splice_playlists(
        low,
        (directory / "up.m3u8").read_text(),
        2,
        list_size=ffmpeg_command.get_live_hls_list_size(1.0),
    )
    assert session.get("splice_point") == 2


@pytest.mark.asyncio
async def test_switch_waits_until_the_upscaled_rendition_catches_up(
    env: _Env,
    tmp_path: pathlib.Path,
):
    _media_playlist(tmp_path / "low.m3u8", ["low_1.ts", "low_2.ts"], duration=2.0)
    _media_playlist(tmp_path / "up.m3u8", ["up_1.ts"], duration=2.0)
    running = FakeProcess(exit_code=None)
    directory = str(tmp_path)
    assert not await ffmpeg_session._caught_up(directory, FakeProcess(exited=True))
    assert not await ffmpeg_session._caught_up(directory, running, timeout_sec=1)
    assert set(env.clock.sleeps) == {0.2}
    assert 1 <= sum(env.clock.sleeps) < 1.2
    env.clock.sleeps.clear()
    caught_up = asyncio.create_task(ffmpeg_session._caught_up(directory, running))
    await until(lambda: env.clock.sleeps)
    assert not caught_up.done()
    _media_playlist(tmp_path / "up.m3u8", ["up_1.ts", "up_2.ts"], duration=2.0)
    assert await caught_up
    # A playlist without a dated segment has no end yet.
    (tmp_path / "up.m3u8").write_text("#EXTM3U\n")
    assert not await ffmpeg_session._caught_up(directory, running, timeout_sec=0.1)
    env.clock.sleeps.clear()
    # Quarter-second polls land exactly on the 30 s deadline, which ends the wait.
    assert not await ffmpeg_session._caught_up(directory, running, poll_interval_sec=0.25)
    assert len(env.clock.sleeps) == 120


def test_live_playlist_continues_the_plain_rendition_across_the_switch(
    env: _Env,
    tmp_path: pathlib.Path,
):
    env.sessions["vod"] = _live(tmp_path)
    for session_id in ("missing", "vod"):
        assert ffmpeg_session.live_playlist(session_id) is None
    session = _fast_session(tmp_path)
    env.sessions["f"] = session
    assert ffmpeg_session.live_playlist("f") is None
    _media_playlist(tmp_path / "low.m3u8", ["low_1.ts", "low_2.ts"], duration=2.0)
    low = (tmp_path / "low.m3u8").read_text()
    assert ffmpeg_session.live_playlist("f") == low
    # Until the upscaler cuts a segment past the plain list's end, clients keep the plain list.
    _media_playlist(tmp_path / "up.m3u8", ["up_1.ts"], duration=2.0, start_pts=13)
    session.update(rendition="up", splice_low=low)
    assert ffmpeg_session.live_playlist("f") == low
    assert "splice_point" not in session
    _media_playlist(tmp_path / "up.m3u8", ["up_1.ts", "up_2.ts"], duration=2.0, start_pts=12)
    up = (tmp_path / "up.m3u8").read_text()
    list_size = ffmpeg_command.get_live_hls_list_size(1.0)
    assert ffmpeg_session.live_playlist("f") == fast_start.splice_playlists(
        low,
        up,
        1,
        list_size=list_size,
    )
    assert session.get("splice_point") == 1
    # The splice point holds once found, even when it could no longer be found.
    (tmp_path / "up_2.ts").unlink()
    assert fast_start.splice_point(str(tmp_path), low, up) is None
    assert ffmpeg_session.live_playlist("f") == fast_start.splice_playlists(
        low,
        up,
        1,
        list_size=list_size,
    )


@pytest.mark.asyncio
async def test_failed_ai_upscale_keeps_the_plain_rendition(
    env: _Env,
    upscaler: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    plain_worker = FakeProcess(exit_code=None)
    launcher = _relaunch(monkeypatch, [plain_worker, FakeProcess(exited=True, exit_code=1)])
    result = await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    session = env.sessions[result["session_id"]]
    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await _upgrade_task(upscaler).run()
    _, up = launcher.processes
    assert session["process"] is plain_worker
    assert "rendition" not in session
    assert session.get("playlist_generation") == 0
    assert ffmpeg_session.playlist_file(session) == "low.m3u8"
    assert plain_worker.signals == []
    assert (
        f"Fast-start session {result['session_id']} keeps its plain rendition: AI Upscale failed"
        in caplog.messages
    )
    await until(lambda: up.stdin.closed)


@pytest.mark.asyncio
async def test_ai_upscale_stops_when_the_session_ends_first(
    env: _Env,
    upscaler: BackgroundTasks,
    caplog: pytest.LogCaptureFixture,
):
    def plain_only(command: list[str]) -> None:
        if command[-1].endswith("/low.m3u8"):
            _ffmpeg_output(command)

    env.launcher.on_launch = plain_only
    result = await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        upgrade = asyncio.create_task(_upgrade_task(upscaler).run())
        await until(lambda: len(env.launcher.launches) == 2)
        # Shutdown drops the session but leaves its directory, so the worker catches up.
        await ffmpeg_session.shutdown()
        _ffmpeg_output(env.launcher.commands[1])
        await upgrade
    up = env.launcher.processes[1]
    assert up.signals == ["TERM"]
    assert up.stdin.closed
    assert (
        f"Fast-start session {result['session_id']} ended before AI Upscale caught up"
        in caplog.messages
    )


@pytest.mark.asyncio
async def test_ai_upscale_rejects_a_worker_without_stdin(
    upscaler: BackgroundTasks,
    monkeypatch: pytest.MonkeyPatch,
):
    deaf = FakeProcess(exit_code=None)
    monkeypatch.setattr(deaf, "stdin", None)
    _relaunch(monkeypatch, [FakeProcess(exit_code=None), deaf])
    await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    with pytest.raises(ValueError, match=r"^Expected process\.stdin is not None\.$"):
        await _upgrade_task(upscaler).run()


@pytest.mark.asyncio
async def test_saver_session_never_upscales(env: _Env, upscaler: BackgroundTasks):
    await ffmpeg_session._start_fast_live(
        "http://live",
        "alice",
        "src",
        True,
        bandwidth_saver=True,
    )
    (worker,) = env.launcher.commands
    assert not _upscales(worker)
    assert "_upgrade_to_upscale" not in _coroutine_names(upscaler)


@pytest.mark.asyncio
async def test_restart_after_the_switch_relaunches_the_upscaled_rendition(
    env: _Env,
    upscaler: BackgroundTasks,
):
    result = await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    session = env.sessions[result["session_id"]]
    assert ffmpeg_session._is_fast_live(session)
    await _upgrade_task(upscaler).run()
    directory = pathlib.Path(session["dir"])
    _media_playlist(directory / "up.m3u8", ["up_000001.ts"])
    up = session["process"]
    await ffmpeg_session._restart_fast_pipeline(result["session_id"], session, "low output")
    restart = env.launcher.launches[-1]
    assert len(env.launcher.launches) == 3
    assert restart.command[-1].endswith("/up.m3u8")
    assert _upscales(restart.command)
    assert "append_list" in _option(restart.command, "-hls_flags")
    assert session["process"] is restart.process
    assert up is not None
    assert up.returncode == -15
    assert session.get("playlist_generation") == 2
    # Only a fresh start spawns the upgrade; the restart keeps "up".
    assert _coroutine_names(upscaler).count("_upgrade_to_upscale") == 1


@pytest.mark.asyncio
async def test_start_fast_live_launch_failure_releases_everything(
    env: _Env,
    fast_env: BackgroundTasks,
    upstreams: FakeUpstreams,
):
    del fast_env
    env.launcher.error = OSError("no ffmpeg")
    with pytest.raises(OSError, match="no ffmpeg"):
        await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    assert upstreams.opened[0].closed
    assert not env.sessions
    assert not env.urls
    assert not list(env.root.iterdir())


@pytest.mark.asyncio
async def test_start_fast_live_rejects_a_worker_without_stdin(
    env: _Env,
    fast_env: BackgroundTasks,
    upstreams: FakeUpstreams,
    monkeypatch: pytest.MonkeyPatch,
):
    del fast_env
    worker = FakeProcess(exit_code=None)
    monkeypatch.setattr(worker, "stdin", None)
    _relaunch(monkeypatch, [worker])
    with pytest.raises(ValueError, match=r"^Expected process\.stdin is not None\.$"):
        await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    assert worker.signals == ["TERM"]
    assert upstreams.opened[0].closed
    assert not env.sessions
    assert not list(env.root.iterdir())


@pytest.mark.asyncio
async def test_start_fast_live_aborts_when_the_session_stops_during_launch(
    env: _Env,
    fast_env: BackgroundTasks,
    upstreams: FakeUpstreams,
):
    del fast_env
    stopped: list[str] = []

    async def stop_during_launch(command: list[str]) -> None:
        del command
        stopped.extend(env.sessions)
        await ffmpeg_session.stop_session(stopped[0], force=True)

    env.launcher.on_launch = stop_during_launch
    with pytest.raises(KeyError) as error:
        await ffmpeg_session._start_fast_live("http://live", "alice", "src", True)
    assert error.value.args == (stopped[0],)
    (worker,) = env.launcher.processes
    assert worker.signals == ["TERM"]
    assert upstreams.opened[0].closed
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


@pytest.fixture
def startups(env: _Env, tasks: BackgroundTasks) -> ProcessLauncher:
    """Each worker publishes only when the test releases it."""
    del tasks
    cache.save_server_settings(
        {**cache.load_server_settings(), "max_resolution": "1080p", "live_dvr_mins": 0},
    )
    env.run.program("ffprobe", Program(stdout=probed_video_json()))
    env.launcher.on_launch = None
    return env.launcher


def _start(user_max_streams: int = 0) -> asyncio.Task[ffmpeg_session.TranscodeResponse]:
    return asyncio.create_task(
        ffmpeg_session.start_transcode(
            "http://live",
            username="viewer",
            user_max_streams=user_max_streams,
        ),
    )


async def _yield_to_startups() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_second_request_waits_for_startup_then_reuses_it(
    startups: ProcessLauncher,
    upstreams: FakeUpstreams,
):
    first = _start()
    await until(lambda: startups.launches)
    second = _start()
    await _yield_to_startups()
    assert not second.done(), "An unfinished startup must not be handed to another request"
    _ffmpeg_output(startups.commands[0])
    first_result, second_result = await asyncio.gather(first, second)
    assert second_result["session_id"] == first_result["session_id"]
    assert len(upstreams.opened) == 1


@pytest.mark.asyncio
async def test_failed_startup_is_not_handed_to_a_waiting_request(
    env: _Env,
    startups: ProcessLauncher,
    upstreams: FakeUpstreams,
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
    assert [upstream.closed for upstream in upstreams.opened] == [True, False]


@pytest.mark.asyncio
async def test_concurrent_starts_open_one_upstream_within_stream_limits(
    env: _Env,
    startups: ProcessLauncher,
    upstreams: FakeUpstreams,
):
    requests = [_start(user_max_streams=1), _start(user_max_streams=1)]
    await until(lambda: startups.launches)
    _ffmpeg_output(startups.commands[0])
    results = await asyncio.gather(*requests)
    assert len(upstreams.opened) == 1
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
async def test_shared_ffmpeg_launcher(env: _Env, monkeypatch: pytest.MonkeyPatch):
    cmd = ["ffmpeg", "-i", "https://provider/live.ts", "/tmp/stream.m3u8"]
    env.launcher.on_launch = None
    # A no_proxy variable would send matching hosts around the egress proxy.
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    assert await ffmpeg_session._launch_ffmpeg(cmd) is env.launcher.processes[0]
    [launch] = env.launcher.launches
    assert launch.command == cmd
    assert launch.options == {
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.DEVNULL,
        "stderr": asyncio.subprocess.PIPE,
        "env": proxy.ffmpeg_env(),
    }
    assert "no_proxy" not in proxy.ffmpeg_env()


@pytest.mark.asyncio
async def test_monitor_stderr_collects_lines(caplog: pytest.LogCaptureFixture):
    process = FakeProcess(stderr=[b"frame=1\n", b"Fatal error\n", b"Aborting now\n"])
    lines: list[str] = []
    done = asyncio.Event()
    with caplog.at_level(logging.DEBUG, logger="netv.ffmpeg_session"):
        await ffmpeg_session._monitor_ffmpeg_stderr(_proc(process), "s", lines, done)
    assert lines == ["frame=1", "Fatal error", "Aborting now"]
    # Set only once stderr closed, so a reader of ``lines`` sees the last of them.
    assert done.is_set()
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
@pytest.mark.parametrize("playlist", ["#EXTINF:3.0,\nseg000.ts\n", "", None])
async def test_existing_vod_that_died_after_a_seek_is_replaced(
    env: _Env,
    vod_dir: pathlib.Path,
    playlist: str | None,
):
    # Ffmpeg stopped partway (a full disk left an empty playlist); its segments end early.
    if playlist is None:
        (vod_dir / "stream.m3u8").unlink()
    else:
        (vod_dir / "stream.m3u8").write_text(playlist)
    env.sessions["v"].update(seek_offset=30)
    env.urls["http://movie"] = "v"
    result = await ffmpeg_session._handle_existing_vod_session(
        "v",
        "http://movie",
        "software",
        False,
    )
    assert result is None
    assert "v" not in env.sessions
    assert "http://movie" not in env.urls


@pytest.mark.asyncio
@pytest.mark.parametrize("captioned", [True, False])
async def test_existing_vod_with_seek_offset_returns_cache(
    env: _Env,
    vod_dir: pathlib.Path,
    captioned: bool,
):
    """A cached session links its caption master only when it has subtitle tracks."""
    (vod_dir / "captions.m3u8").write_text("#EXTM3U\n")
    with (vod_dir / "stream.m3u8").open("a") as playlist:
        playlist.write("#EXT-X-ENDLIST\n")
    subs: list[ffmpeg_session.StoredSubtitle | int] = [
        {"index": 2, "lang": "eng", "name": "English"},
    ]
    env.sessions["v"].update(seek_offset=30, last_access=0, subtitles=subs if captioned else [])
    result = await ffmpeg_session._handle_existing_vod_session(
        "v",
        "http://movie",
        "software",
        False,
    )
    expected: ffmpeg_session.TranscodeResponse = {
        "session_id": "v",
        "playlist": "/transcode/v/stream.m3u8",
        "subtitles": [],
        "duration": 100,
        "seek_offset": 30,
        "transcoded_duration": 6.0,
    }
    if captioned:
        expected["subtitles"] = [
            {"url": "/subs/v/sub0.vtt", "lang": "eng", "label": "English", "default": True},
        ]
        expected["caption_playlist"] = "/transcode/v/captions.m3u8"
    assert result == expected
    assert env.sessions["v"].get("last_access") == _NOW
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
    # Before the playlist: ffmpeg ignores options after its last output.
    assert command[-3:-1] == ["-start_number", "2"] and command[-1].endswith("stream.m3u8")
    assert env.sessions["v"]["process"] is env.launcher.processes[0]
    assert [call[-1] for call in env.run.calls] == ["http://movie", "http://movie"]
    assert env.clock.sleeps == []


# A home-network HLS movie: its master names the variant ffmpeg reads.
_HOME_MOVIE = "http://192.168.1.50/movie/1.m3u8"
_HOME_VARIANT = "http://192.168.1.50/movie/1_hd.m3u8"


def _home_movie(env: _Env) -> None:
    _save_sources({"id": "h", "name": "Home", "type": "m3u", "url": "http://192.168.1.50/l.m3u"})
    env.web.routes[_HOME_MOVIE] = lambda: FakeResponse(
        b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=9\n1_hd.m3u8\n",
    )


def _read_home(env: _Env) -> bool:
    """Whether every fetch, probe and encoder reached the home movie by the home rule."""
    home = proxy.egress().url(lan=True)
    fetched = [fetch for fetch in env.web.fetches if fetch.url == _HOME_MOVIE]
    return (
        bool(fetched and env.run.calls and env.launcher.commands)
        and all(
            vet(ipaddress.ip_address("192.168.1.50"))
            for fetch in fetched
            for vet in testing.address_vets(fetch)
        )
        and all(
            command[-1] == _HOME_VARIANT and _option(command, "-http_proxy") == home
            for command in env.run.calls
        )
        and all(
            _option(command, "-i") == _HOME_VARIANT and _option(command, "-http_proxy") == home
            for command in env.launcher.commands
        )
    )


@pytest.mark.asyncio
async def test_a_vod_start_reads_its_source_by_that_source_s_rule(
    env: _Env,
    tasks: BackgroundTasks,
):
    del tasks
    _home_movie(env)
    env.run.program("ffprobe", Program(stdout=_DOLBY_PROBE))
    await ffmpeg_session.start_transcode(_HOME_MOVIE, "movie", source_id="h")
    assert _read_home(env)


@pytest.mark.asyncio
async def test_a_seek_reads_its_source_by_that_source_s_rule(env: _Env, seek_dir: pathlib.Path):
    _home_movie(env)
    env.sessions["v"].update(url=_HOME_MOVIE, source_id="h")
    (seek_dir / "session.json").write_text(json.dumps({"seek_offset": 0}))
    await ffmpeg_session.seek_transcode("v", 9.0)
    assert _read_home(env)


@pytest.mark.asyncio
async def test_a_resume_reads_its_source_by_that_source_s_rule(env: _Env, vod_dir: pathlib.Path):
    del vod_dir
    _home_movie(env)
    env.sessions["v"].update(url=_HOME_MOVIE, source_id="h")
    await ffmpeg_session._handle_existing_vod_session("v", _HOME_MOVIE, "software", True)
    assert _read_home(env)


def test_a_session_snapshot_carries_every_field_it_resumes_with(env: _Env, tmp_path: pathlib.Path):
    process = FakeProcess(exit_code=None)
    env.sessions["full"] = ffmpeg_session.TranscodeSession(
        dir=str(tmp_path),
        process=process,
        url="http://movie",
        is_vod=True,
        seek_offset=12.5,
        subtitles=[{"index": 2, "lang": "eng", "name": "English"}],
        duration=99.0,
        audio_passthrough=True,
        username="alice",
        source_id="h",
        started=_NOW,
    )
    env.sessions["bare"] = ffmpeg_session.TranscodeSession(
        dir=str(tmp_path),
        process=None,
        url="http://movie",
        is_vod=True,
        started=_NOW,
    )
    assert ffmpeg_session._get_session_snapshot("full") == ffmpeg_session._SessionSnapshot(
        output_dir=str(tmp_path),
        process=process,
        seek_offset=12.5,
        subtitles=[{"index": 2, "lang": "eng", "name": "English"}],
        duration=99.0,
        audio_passthrough=True,
        username="alice",
        source_id="h",
    )
    assert ffmpeg_session._get_session_snapshot("bare") == ffmpeg_session._SessionSnapshot(
        output_dir=str(tmp_path),
        process=None,
        seek_offset=0,
        subtitles=[],
        duration=0,
        audio_passthrough=False,
        username="",
        source_id="",
    )
    assert ffmpeg_session._get_session_snapshot("missing") is None


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
        {**cache.load_server_settings(), "max_resolution": "4k", "probe_movies": True},
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
        "audio_lang": "en",
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
            "audio_tracks": [
                {
                    "index": 1,
                    "lang": "und",
                    "codec": "aac",
                    "channels": 2,
                    "sample_rate": 48_000,
                    "profile": "",
                    "default": False,
                },
            ],
        },
    }
    assert result.get("caption_playlist", "").endswith("/captions.m3u8")
    assert session.get("caption_timestamp_origin") == 10.0


@pytest.mark.asyncio
@pytest.mark.usefixtures("do_env")
async def test_do_start_plays_the_users_audio_language(env: _Env):
    probe = json.loads(_MOVIE_PROBE)
    probe["streams"].append(
        {
            "index": 3,
            "codec_type": "audio",
            "codec_name": "ac3",
            "channels": 6,
            "sample_rate": "48000",
            "tags": {"language": "fre"},
        },
    )
    env.run.program("ffprobe", Program(stdout=json.dumps(probe)))
    cache.save_user_settings("alice", {**cache.load_user_settings("alice"), "audio_lang": "fr"})
    result = await ffmpeg_session._do_start_transcode(
        "http://movie",
        "movie",
        None,
        None,
        0,
        username="alice",
    )
    session = env.sessions[result["session_id"]]
    info = json.loads((pathlib.Path(session["dir"]) / "session.json").read_text())
    assert (info["probe"]["audio_codec"], info["probe"]["audio_channels"]) == ("ac3", 6)


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
    # A resumed start waits out the longer seek deadline, then up to 1 s for stderr
    # to close: its reader is held here, so it never does.
    assert sum(env.clock.sleeps) == pytest.approx(41.0, abs=0.05)
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
    directory = str(tmp_path)
    running = FakeProcess(exit_code=None)
    wait = ffmpeg_session._wait_for_fresh_fast_playlist
    assert not await wait(directory, "low.m3u8", FakeProcess(exited=True), None)
    assert env.clock.sleeps == []
    # The playlist has not changed since the old worker published it.
    assert not await wait(directory, "low.m3u8", running, marker, timeout_sec=2)
    assert set(env.clock.sleeps) == {0.05}
    assert 2 <= sum(env.clock.sleeps) < 2.1
    assert not await wait(directory, "low.m3u8", running, None, minimum_segments=3, timeout_sec=1)
    # Only segments the old worker did not publish count.
    assert not await wait(
        directory,
        "low.m3u8",
        running,
        None,
        previous_segments={"low_1.ts"},
        timeout_sec=1,
    )
    env.clock.sleeps.clear()
    assert await wait(directory, "low.m3u8", running, None, previous_segments={"old.ts"})
    assert await wait(directory, "low.m3u8", running, None, is_disconnected=_connected)
    assert env.clock.sleeps == []
    with pytest.raises(HTTPException) as error:
        await wait(directory, "low.m3u8", running, None, is_disconnected=_disconnected)
    assert error.value.status_code == 499


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
async def test_worker_sample_starts_at_the_first_keyframe(upstreams: FakeUpstreams):
    """Bytes before the first keyframe cannot decode; the probe and worker never see them."""
    stream = _plan(_TUNER).relay()
    await stream.start()
    await until(lambda: upstreams.opened)
    sample = asyncio.create_task(ffmpeg_session._wait_for_keyframe(stream))
    upstreams.opened[0].feed(pat_packet() + pmt_packet() + video_packet(9) * 5)
    await until(lambda: stream.stats.bytes_received)
    await _yield_to_startups()
    assert not sample.done()
    upstreams.opened[0].feed(ts_source())
    assert await sample == ts_source()
    await stream.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ends", [False, True])
async def test_a_source_without_keyframe_marks_is_used_whole(
    upstreams: FakeUpstreams,
    ends: bool,
):
    """Without random-access marks, the sample is everything: at the cap, or when upstream ends."""
    unmarked = video_packet(10) * 10
    stream = _plan(_TUNER).relay()
    await stream.start()
    await until(lambda: upstreams.opened)
    sample = asyncio.create_task(
        ffmpeg_session._wait_for_keyframe(stream, max_bytes=len(unmarked) * (2 if ends else 1)),
    )
    upstreams.opened[0].feed(unmarked)
    if ends:
        upstreams.opened[0].end()
    assert await sample == unmarked
    await stream.close()


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
    # Eleven lines, one not UTF-8: the warning keeps the last ten, readably.
    stderr = [f"line {i}\n".encode() for i in range(9)] + [b"bad \xff byte\n", b"failed\n"]
    process = FakeProcess(stderr=stderr, exit_code=1)
    quiet = FakeProcess(exit_code=3)
    for sid, worker in (("f", process), ("g", quiet)):
        env.sessions[sid] = ffmpeg_session.TranscodeSession(
            process=worker,
            dir=str(tmp_path),
            started=_NOW,
            url="http://stream.test/live",
        )
    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await ffmpeg_session._monitor_fast_process(_proc(process), "encoder", "f")
        await ffmpeg_session._monitor_fast_process(_proc(quiet), "remux", "g")
        # A replaced worker, or one whose session ended, exits quietly.
        await ffmpeg_session._monitor_fast_process(_proc(FakeProcess(exit_code=1)), "encoder", "f")
        await ffmpeg_session._monitor_fast_process(_proc(FakeProcess(exit_code=1)), "encoder", "x")
    warnings = [r.getMessage() for r in caplog.records]
    assert warnings == [
        "Fast-start encoder exited for f (code 1): "
        + " | ".join([f"line {i}" for i in range(1, 9)] + ["bad \ufffd byte", "failed"]),
        "Fast-start remux exited for g (code 3): no stderr",
    ]
    process.stderr = None
    with pytest.raises(ValueError, match=r"^Expected process\.stderr is not None\.$"):
        await ffmpeg_session._monitor_fast_process(_proc(process), "encoder", "f")


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


@pytest.mark.asyncio
async def test_restart_clock_dates_old_segments_then_follows_the_new_relay(
    env: _Env,
    tmp_path: pathlib.Path,
):
    (tmp_path / "low_1.ts").write_bytes(video_packet(5002) * 10)
    session = _fast_session(tmp_path)
    session.update(origin_pts=5000.0, origin_time=100.0)
    env.sessions["f"] = session
    # Before the first worker publishes, there is nothing to date.
    await ffmpeg_session._restart_session_clock("f", session, 1.5, 200.0)
    assert "segment_dates" not in session
    assert (session.get("origin_pts"), session.get("origin_time")) == (1.5, 200.0)
    (tmp_path / "low.m3u8").write_text("#EXTINF:2,\nlow_1.ts\n")
    session.update(origin_pts=5000.0, origin_time=100.0)
    await ffmpeg_session._restart_session_clock("f", session, None, 300.0)
    assert session.get("segment_dates") == {"low_1.ts": 102.0}
    assert (session.get("origin_pts"), session.get("origin_time")) == (None, 300.0)
    env.sessions.clear()
    with pytest.raises(KeyError):
        await ffmpeg_session._restart_session_clock("f", session, 7.0, 400.0)
    assert (session.get("origin_pts"), session.get("origin_time")) == (None, 300.0)


def test_dated_fast_playlist_needs_a_known_clock(env: _Env, tmp_path: pathlib.Path):
    assert ffmpeg_session.dated_fast_playlist("missing", "#EXTM3U\n") == "#EXTM3U\n"
    env.sessions["f"] = _fast_session(tmp_path)
    assert ffmpeg_session.dated_fast_playlist("f", "#EXTM3U\n") == "#EXTM3U\n"
    env.sessions["f"]["origin_pts"] = None
    assert ffmpeg_session.dated_fast_playlist("f", "#EXTM3U\n") == "#EXTM3U\n"
    env.sessions["f"]["origin_pts"] = 1.0
    with pytest.raises(KeyError, match="origin_time"):
        ffmpeg_session.dated_fast_playlist("f", "#EXTM3U\n")


def _published_low(directory: pathlib.Path) -> None:
    """The old worker's rendition: one segment, dated by the old clock at 102."""
    (directory / "low_1.ts").write_bytes(video_packet(5002) * 10)
    (directory / "low.m3u8").write_text("#EXTINF:2,\nlow_1.ts\n")


@pytest.mark.asyncio
async def test_restart_pipeline_skips_replaced_session(env: _Env, tmp_path: pathlib.Path):
    session = _fast_session(tmp_path)
    await ffmpeg_session._restart_fast_pipeline("f", session, "input")
    assert "watchdog_recovering" not in session
    assert env.launcher.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage,generation",
    [("input", 0), ("input", 7), ("low output", 0)],
)
async def test_restart_pipeline_success(
    env: _Env,
    tasks: BackgroundTasks,
    upstreams: FakeUpstreams,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    stage: str,
    generation: int,
):
    """A stalled input reconnects the relay and restarts the clock; a stalled worker only restarts."""
    env.run.program("ffprobe", Program(stdout=probed_video_json()))
    _published_low(tmp_path)
    old = FakeProcess(exit_code=None)
    recovery, permitted = LiveRecoveryPolicy(failures={"input": _NOW - 5}).permit_restart(
        "pipeline",
        _NOW,
    )
    assert permitted
    session = _fast_session(tmp_path, old, live_recovery=recovery)
    session.update(playlist_generation=generation, origin_pts=5000.0, origin_time=100.0)
    env.sessions["f"] = session
    first = session["relay"]
    await _receiving(session)
    session["subscription"] = first.subscribe(old.stdin)
    env.clock.now = _NOW + 30
    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await ffmpeg_session._restart_fast_pipeline("f", session, stage)
    (launch,) = env.launcher.launches
    # The replacement appends to the rendition after a discontinuity.
    assert "append_list" in _option(launch.command, "-hls_flags")
    assert "-hls_start_number_source" not in launch.command
    await launch.process.stdin.wait_for(len(ts_source()))
    assert bytes(launch.process.stdin.written) == ts_source()
    assert old.signals == ["TERM"]
    assert old.stdin.closed
    assert session["process"] is launch.process
    assert session.get("playlist_generation") == generation + 1
    assert dict(session["live_recovery"].failures) == {}
    assert session["live_recovery"].pending == frozenset()
    assert "watchdog_recovering" not in session
    assert _coroutine_names(tasks) == ["_monitor_fast_process"]
    assert [r.getMessage() for r in caplog.records] == [
        f"Live watchdog f recovered pipeline generation {generation + 1} after {stage} stalled",
    ]
    reconnect = stage == "input"
    assert (session["relay"] is not first, first.finished) == (reconnect, reconnect)
    assert [upstream.closed for upstream in upstreams.opened] == (
        [True, False] if reconnect else [False]
    )
    # A new connection may bring a new source clock; listed segments keep their dates.
    assert (session.get("origin_pts"), session.get("origin_time")) == (
        (10, _NOW + 30) if reconnect else (5000.0, 100.0)
    )
    assert session.get("segment_dates") == ({"low_1.ts": 102.0} if reconnect else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("silent", [True, False])
async def test_restart_pipeline_failure_discards_the_replacement(
    env: _Env,
    tasks: BackgroundTasks,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    silent: bool,
):
    """A replacement that never publishes, or never launches, leaves the stall for the next try."""
    del tasks
    _published_low(tmp_path)
    recovery, _ = LiveRecoveryPolicy(failures={"low": _NOW - 5}).permit_restart("pipeline", _NOW)
    session = _fast_session(tmp_path, FakeProcess(exit_code=None), live_recovery=recovery)
    session["playlist_generation"] = 5
    env.sessions["f"] = session
    await _receiving(session)
    env.launcher.on_launch = None
    env.launcher.error = None if silent else OSError("no ffmpeg")
    with caplog.at_level(logging.ERROR, logger="netv.ffmpeg_session"):
        await ffmpeg_session._restart_fast_pipeline("f", session, "low output")
    assert "failed to recover the pipeline after low output stalled" in caplog.text
    assert isinstance(session["process"], ffmpeg_session._DeadProcess)
    assert [p.signals for p in env.launcher.processes] == ([["TERM"]] if silent else [])
    assert session.get("playlist_generation") == 5
    # The stall stays on record and the budget frees for the next attempt.
    assert dict(session["live_recovery"].failures) == {"low": _NOW - 5}
    assert session["live_recovery"].pending == frozenset()
    assert "watchdog_recovering" not in session


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["input", "low output"])
async def test_restart_pipeline_stops_when_the_session_ends(
    env: _Env,
    tasks: BackgroundTasks,
    upstreams: FakeUpstreams,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    stage: str,
):
    del tasks
    _published_low(tmp_path)
    session = _fast_session(tmp_path, FakeProcess(exit_code=None))
    env.sessions["f"] = session
    await _receiving(session)

    async def stop_during_launch(command: list[str]) -> None:
        del command
        await ffmpeg_session.stop_session("f", force=True)

    env.launcher.on_launch = stop_during_launch
    with caplog.at_level(logging.INFO, logger="netv.ffmpeg_session"):
        await ffmpeg_session._restart_fast_pipeline("f", session, stage)
    (replacement,) = env.launcher.processes
    assert replacement.signals == ["TERM"]
    assert "Live watchdog f stopped recovering: the session ended" in caplog.messages
    assert all(upstream.closed for upstream in upstreams.opened)
    assert not env.sessions
    # The ended session's state is left as the stop found it.
    assert session.get("playlist_generation") == 0
    assert session.get("watchdog_recovering") == stage


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


async def _relay_last_heard(
    session: ffmpeg_session.FastLiveSession,
    clock: FakeClock,
    at: float,
) -> None:
    """Start the session's relay with its upstream's first bytes arriving at time ``at``."""
    clock.now = at
    await _receiving(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_after", [None, 47.5])
@pytest.mark.parametrize(
    "input_state,low_alive,input_age,low_age",
    [
        ("receiving", True, 5, 5),
        ("receiving", True, 60, 5),
        ("receiving", True, 5, 60),
        ("receiving", True, 30, 30),
        ("receiving", True, 10.5, 9.5),
        ("receiving", True, 10, 5),
        ("ended", True, 5, 5),
        ("receiving", False, 5, 5),
        ("silent", None, 5, 5),
    ],
)
async def test_watchdog_checks_live_media_and_persists_each_observation(
    env: _Env,
    upstreams: FakeUpstreams,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stale_after: float | None,
    input_state: str,
    low_alive: bool | None,
    input_age: float,
    low_age: float,
):
    """The input is ready while the relay hears from upstream; the output needs two segments."""
    now = 123.5
    install_clock(monkeypatch, env.clock, fast_start)
    _aged_playlist(tmp_path, "low", segments=2, mtime=now - low_age)
    session = _fast_session(
        tmp_path,
        _process(low_alive),
        source_url=_TUNER if input_state == "silent" else "http://live",
    )
    if stale_after is not None:
        session["watchdog_stale_after"] = stale_after
    env.sessions["media-check"] = session
    if input_state == "silent":
        await session["relay"].start()
        await until(lambda: upstreams.opened)
    else:
        await _relay_last_heard(session, env.clock, now - input_age)
    if input_state == "ended":
        upstreams.opened[0].end()
        await session["relay"].wait_finished()
    env.clock.now = now

    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await ffmpeg_session._check_fast_live_session("media-check", session)

    max_age = 10 if stale_after is None else stale_after
    ready = {
        "input": input_state == "receiving" and input_age <= max_age,
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
    upstreams: FakeUpstreams,
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
    # The replacement worker dies at once, so a permitted restart fails fast.
    launcher = _relaunch(monkeypatch, [FakeProcess(exited=True, exit_code=1)])
    stalled = {"input": input_stalled, "low": low_stalled}
    session = _fast_session(
        tmp_path,
        live_recovery=LiveRecoveryPolicy(
            failures={stage: now - 10 for stage, value in stalled.items() if value},
            attempts={"pipeline": attempts} if attempts else {},
            pending=frozenset({"pipeline"}) if pending else frozenset(),
        ),
    )
    # A just-polled session survives an unforced stop.
    session["last_access"] = now
    env.sessions["dispatch-check"] = session
    env.urls["http://live"] = "dispatch-check"
    # Upstream went quiet 100 s ago and no worker runs: both stages look stalled.
    await _relay_last_heard(session, env.clock, now - 100)
    env.clock.now = now

    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await ffmpeg_session._check_fast_live_session("dispatch-check", session)

    stage = "input" if input_stalled else "low output"
    recovery = session["live_recovery"]
    assert dict(recovery.failures) == {
        "input": now - 10 if input_stalled else now,
        "low": now - 10 if low_stalled else now,
    }
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    messages = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.levelno >= logging.ERROR
    ]
    if action == "restart":
        (command,) = launcher.commands
        assert _option(command, "-i") == "pipe:0"
        # Only a stalled input reconnects upstream.
        assert len(upstreams.opened) == (2 if input_stalled else 1)
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
        assert ffmpeg_session._get_existing_session("http://live") is None
        assert not env.urls
        assert upstreams.opened[0].closed
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
    if action == "delay":
        assert warnings == [
            "Live watchdog dispatch-check is delaying pipeline recovery because its "
            "restart budget is cooling down",
        ]
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
    session = _fast_session(tmp_path, live_recovery=LiveRecoveryPolicy(failures=failures))
    env.sessions["f"] = session
    await _relay_last_heard(session, env.clock, _NOW if input_ready else _NOW - 100)
    env.clock.now = _NOW
    env.launcher.on_launch = None
    with caplog.at_level(logging.ERROR, logger="netv.ffmpeg_session"):
        await ffmpeg_session._check_fast_live_session("f", session)
    assert f"failed to recover the pipeline after {stage} stalled" in caplog.text
    (command,) = env.launcher.commands
    assert _option(command, "-i") == "pipe:0"
    # An upstream heard from recently clears its stall record.
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
    await _receiving(session)
    launched: list[list[str]] = []
    release = asyncio.Event()

    async def launch(command: list[str]) -> None:
        launched.append(command)
        if len(launched) == 1:
            raise RuntimeError("boom")
        await release.wait()

    env.launcher.on_launch = launch
    with caplog.at_level(logging.ERROR, logger="netv.ffmpeg_session"):
        watchdog = asyncio.create_task(
            ffmpeg_session._watch_fast_live("f", watchdog_interval_sec=6),
        )
        await until(lambda: len(launched) == 2)
        watchdog.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await watchdog
    assert "failed while checking pipeline health" in caplog.text
    (late,) = env.launcher.processes
    assert late.signals == ["TERM"]
    assert session["live_recovery"].pending == frozenset()
    assert "watchdog_recovering" not in session


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
    upstreams: FakeUpstreams
    low: FakeProcess


@pytest_asyncio.fixture
async def recovery(
    env: _Env,
    tasks: BackgroundTasks,
    upstreams: FakeUpstreams,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Recovery:
    """A running fast-start session, its first bytes at time 0; replacements publish on launch."""
    del tasks
    env.run.program("ffprobe", Program(stdout=probed_video_json()))
    install_clock(monkeypatch, env.clock, fast_start)
    playlist(tmp_path, "low")
    low = FakeProcess(exit_code=None)
    session = _fast_session(tmp_path, low)
    session.update(username="viewer", watchdog_stale_after=45)
    env.sessions["recover"] = session
    env.clock.now = 0
    await _receiving(session)
    session["subscription"] = session["relay"].subscribe(low.stdin)
    return _Recovery(
        session=session,
        launcher=env.launcher,
        clock=env.clock,
        upstreams=upstreams,
        low=low,
    )


async def _deliver(recovery: _Recovery, now: float) -> None:
    """Have the session's current upstream deliver a packet at ``now``."""
    recovery.clock.now = now
    stream = recovery.session["relay"]
    received = stream.stats.bytes_received
    recovery.upstreams.opened[-1].feed(video_packet(now))
    await until(lambda: stream.stats.bytes_received > received)


async def _check(recovery: _Recovery, now: float, fresh: dict[str, bool]) -> None:
    """Deliver input or republish low output, fresh or 100 s stale, then check at ``now``."""
    if fresh["input"]:
        await _deliver(recovery, now)
    recovery.clock.now = now
    directory = pathlib.Path(recovery.session["dir"])
    playlist(directory, "low")
    mtime = now if fresh["low"] else now - 100
    os.utime(directory / "low.m3u8", (mtime, mtime))
    await ffmpeg_session._check_fast_live_session("recover", recovery.session)


@pytest.mark.asyncio
async def test_input_stall_restarts_pipeline_once(
    recovery: _Recovery,
    caplog: pytest.LogCaptureFixture,
):
    fresh = {"input": False, "low": True}
    with caplog.at_level(logging.WARNING, logger="netv.ffmpeg_session"):
        await _check(recovery, 1000, fresh)
        assert recovery.launcher.launches == []
        await _check(recovery, 1002, fresh)
        assert len(recovery.launcher.launches) == 1
        # The reconnected relay delivers again, so the input is no longer stalled.
        await _check(recovery, 1004, fresh)
    assert len(recovery.launcher.launches) == 1
    assert recovery.low.signals == ["TERM"]
    assert [upstream.closed for upstream in recovery.upstreams.opened] == [True, False]
    assert recovery.session.get("playlist_generation") == 1
    assert [r.getMessage() for r in caplog.records if r.name == "netv.ffmpeg_session"] == [
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
    await _deliver(recovery, 70)
    os.utime(pathlib.Path(recovery.session["dir"]) / "low.m3u8", (70, 70))
    recovery.clock.now = 100
    await ffmpeg_session._check_fast_live_session("recover", recovery.session)
    assert recovery.launcher.launches == []
    assert dict(recovery.session["live_recovery"].failures) == {}


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
