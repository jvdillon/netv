"""Coverage-focused tests for ffmpeg session management."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import asyncio
import json
import logging
import pathlib
import time

from fastapi import HTTPException

import pytest

from ffmpeg_command import MediaInfo
from ffmpeg_session_test import ARCHIVE_URL, FakeProcess
from playback_policy import LiveRecoveryPolicy, PlaybackHealth

import ffmpeg_session as fs


class FakeStderr:
    def __init__(self, lines=()):
        self._lines = list(lines)

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


class AsyncProcess(FakeProcess):
    """Fake process with async stderr and wait."""

    def __init__(self, alive: bool = True, lines=(), exit_code: int = 0):
        super().__init__(alive=alive)
        self.stderr: FakeStderr | None = FakeStderr(lines)
        self.exit_code = exit_code
        self.pid = 4242
        self.waited = 0

    async def wait(self) -> int:
        self.waited += 1
        if self.returncode is None:
            self.returncode = self.exit_code
        return self.returncode


class StubbornProcess:
    """Ignores SIGTERM."""

    returncode = None

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        self.returncode = -9


def _close(coro):
    coro.close()


def _proc(process: FakeProcess) -> asyncio.subprocess.Process:
    return cast(asyncio.subprocess.Process, process)


def _call(mock: Mock) -> Any:
    assert mock.call_args is not None
    return mock.call_args


@pytest.fixture
def state():
    with (
        patch.dict(fs._transcode_sessions, {}, clear=True),
        patch.dict(fs._url_to_session, {}, clear=True),
    ):
        yield fs._transcode_sessions, fs._url_to_session


def _live(tmp_path, **extra):
    session = {
        "dir": str(tmp_path),
        "process": FakeProcess(),
        "started": time.time(),
        "last_access": time.time(),
        "url": "http://live",
    }
    session.update(extra)
    return session


# =============================================================================
# Validity, stop, shutdown
# =============================================================================


def test_fast_start_session_with_recovery_stays_valid_without_process():
    session = {
        "started": time.time(),
        "fast_start": True,
        "live_recovery": object(),
        "process": None,
    }
    assert fs.is_session_valid(session) is True


def test_stop_session_cancels_pending_tasks_and_kills_extra_processes(state, tmp_path):
    sessions, _ = state
    pending = MagicMock(done=MagicMock(return_value=False))
    finished = MagicMock(done=MagicMock(return_value=True))
    extra = FakeProcess()
    sessions["s"] = _live(
        tmp_path,
        watchdog_task=pending,
        high_recovery_task=finished,
        extra_processes=[extra],
    )
    fs.stop_session("s", force=True)
    pending.cancel.assert_called_once()
    finished.cancel.assert_not_called()
    assert extra.returncode == -15
    assert "s" not in sessions
    assert not tmp_path.exists()


def test_stop_user_stream_matches_url_or_request_url_for_owner_only(state, tmp_path):
    sessions, _ = state
    for sid, user, key in [
        ("a", "alice", "url"),
        ("b", "alice", "request_url"),
        ("c", "bob", "url"),
    ]:
        directory = tmp_path / sid
        directory.mkdir()
        sessions[sid] = {"dir": str(directory), "process": FakeProcess(), "username": user}
        sessions[sid][key] = "http://stream"
    fs.stop_user_stream("alice", "http://stream")
    assert set(sessions) == {"c"}


def test_shutdown_cancels_tasks_and_kills_extras(state):
    sessions, _ = state
    task = MagicMock(done=MagicMock(return_value=False))
    extra = FakeProcess()
    sessions["s"] = {"watchdog_task": task, "extra_processes": [extra], "process": None}
    fs.shutdown()
    task.cancel.assert_called_once()
    assert extra.returncode == -15
    assert not sessions


# =============================================================================
# Startup recovery
# =============================================================================


def _vod_dir(root: pathlib.Path, name: str, info, *, segment: bool = True) -> pathlib.Path:
    directory = root / f"netv_transcode_{name}"
    directory.mkdir()
    (directory / "session.json").write_text(info if isinstance(info, str) else json.dumps(info))
    if segment:
        (directory / "seg000.ts").write_bytes(b"x" * 2000)
    return directory


def _recover(root):
    with (
        patch("ffmpeg_session.get_transcode_dir", return_value=root),
        patch("ffmpeg_session.get_vod_cache_timeout", return_value=3600),
    ):
        fs.cleanup_and_recover_sessions()


def test_recovery_ignores_plain_files(state, tmp_path):
    stray = tmp_path / "netv_transcode_file"
    stray.write_text("x")
    _recover(tmp_path)
    assert stray.exists()


def test_recovery_removes_dir_that_cannot_be_stat(state):
    directory = MagicMock()
    directory.is_dir.return_value = True
    directory.stat.side_effect = OSError
    root = MagicMock()
    root.glob.return_value = [directory]
    with patch("ffmpeg_session.shutil.rmtree") as rmtree:
        _recover(root)
    rmtree.assert_called_once_with(directory, ignore_errors=True)


def test_recovery_removes_session_without_segments(state, tmp_path):
    directory = _vod_dir(
        tmp_path,
        "empty",
        {"session_id": "e", "url": "http://m", "is_vod": True},
        segment=False,
    )
    _recover(tmp_path)
    assert not directory.exists()
    assert not state[0]


def test_recovery_removes_corrupt_session_json(state, tmp_path):
    directory = _vod_dir(tmp_path, "bad", "{not json")
    _recover(tmp_path)
    assert not directory.exists()


@pytest.mark.parametrize(
    "existing_seek,new_seek,winner",
    [(0, 5, "new"), (5, 0, "old"), (0, 0, "new")],
)
def test_recovery_prefers_seeked_or_newer_duplicate(
    state,
    tmp_path,
    existing_seek,
    new_seek,
    winner,
):
    sessions, urls = state
    sessions["old"] = {"seek_offset": existing_seek, "last_access": 0}
    urls["http://m"] = "old"
    _vod_dir(
        tmp_path,
        "new",
        {"session_id": "new", "url": "http://m", "is_vod": True, "seek_offset": new_seek},
    )
    _recover(tmp_path)
    assert "new" in sessions
    assert urls["http://m"] == winner


def test_recovery_restores_probe_cache(state, tmp_path):
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
    _vod_dir(tmp_path, "p", info)
    with patch("ffmpeg_session.restore_probe_cache_entry") as restore:
        _recover(tmp_path)
    url, media, subs, series, episode = restore.call_args.args
    assert (url, series, episode) == ("http://m", 7, 8)
    assert (media.video_codec, media.height, media.duration, media.interlaced) == (
        "hevc",
        2160,
        99,
        True,
    )
    assert [(s.index, s.lang, s.name) for s in subs] == [(2, "eng", "")]


# =============================================================================
# Caption timestamps
# =============================================================================


def test_caption_timestamp_origin_paths(state, tmp_path):
    sessions, _ = state
    assert fs.get_caption_timestamp_origin("missing") is None
    sessions["cached"] = {"dir": str(tmp_path), "caption_timestamp_origin": 3}
    assert fs.get_caption_timestamp_origin("cached") == 3.0
    sessions["none"] = {"dir": str(tmp_path)}
    assert fs.get_caption_timestamp_origin("none") is None
    with patch("ffmpeg_session._caption_timestamp_origin", return_value=4.5):
        assert fs.get_caption_timestamp_origin("none") == 4.5
    assert sessions["none"]["caption_timestamp_origin"] == 4.5

    def vanish(_):
        sessions.pop("gone")
        return 1.0

    sessions["gone"] = {"dir": str(tmp_path)}
    with patch("ffmpeg_session._caption_timestamp_origin", side_effect=vanish):
        assert fs.get_caption_timestamp_origin("gone") is None


def test_caption_timestamp_origin_none_without_pts(tmp_path):
    (tmp_path / "stream.m3u8").write_text("#EXTINF:bad,\n#EXTINF:4,\nsub/x.ts\nseg0.ts\n")
    with patch("ffmpeg_session.segment_pts", return_value=None) as pts:
        assert fs._caption_timestamp_origin(str(tmp_path)) is None
    pts.assert_called_once_with(tmp_path / "seg0.ts")


# =============================================================================
# start_transcode routing
# =============================================================================


@pytest.fixture
def routing(state):
    do_start = AsyncMock(return_value={"fresh": True})
    with (
        patch("ffmpeg_session.get_settings", return_value={}),
        patch("ffmpeg_session._do_start_transcode", new=do_start),
    ):
        yield state, do_start


def _existing(state, tmp_path, url, **extra):
    sessions, urls = state
    sessions["old"] = _live(tmp_path, url=url, **extra)
    urls[url] = "old"
    return sessions


@pytest.mark.asyncio
async def test_start_rejects_bandwidth_saver_for_vod(routing):
    with pytest.raises(HTTPException) as error:
        await fs.start_transcode("http://m", "movie", bandwidth_saver=True)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_start_enforces_stream_limits(routing):
    with (
        patch("ffmpeg_session.enforce_stream_limits", return_value="full"),
        pytest.raises(HTTPException) as error,
    ):
        await fs.start_transcode("http://live", username="alice")
    assert (error.value.status_code, error.value.detail) == (429, "full")


@pytest.mark.asyncio
async def test_start_rejects_archive_owned_by_other_user(routing, tmp_path):
    state, _ = routing
    _existing(state, tmp_path, ARCHIVE_URL, username="bob")
    with pytest.raises(HTTPException) as error:
        await fs.start_transcode(ARCHIVE_URL, "movie", username="alice")
    assert error.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,extra",
    [
        ({"bandwidth_saver": True}, {}),
        ({}, {"audio_passthrough": True}),
    ],
)
async def test_start_replaces_own_incompatible_session(routing, tmp_path, kwargs, extra):
    state, do_start = routing
    sessions = _existing(state, tmp_path, "http://live", username="alice", **extra)
    assert await fs.start_transcode("http://live", username="alice", **kwargs) == {"fresh": True}
    assert "old" not in sessions
    assert do_start.await_args.args[4] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,extra,status",
    [
        ({"bandwidth_saver": True}, {}, 404),
        ({}, {"audio_passthrough": True}, 409),
        ({}, {"fast_start": True}, 404),
    ],
)
async def test_start_refuses_other_users_session(routing, tmp_path, kwargs, extra, status):
    state, _ = routing
    sessions = _existing(state, tmp_path, "http://live", username="bob", **extra)
    with pytest.raises(HTTPException) as error:
        await fs.start_transcode("http://live", username="alice", **kwargs)
    assert error.value.status_code == status
    assert "old" in sessions


@pytest.mark.asyncio
async def test_start_rejoins_own_fast_start_session(routing, tmp_path):
    state, do_start = routing
    sessions = _existing(
        state,
        tmp_path,
        "http://live",
        username="alice",
        fast_start=True,
        last_access=time.time(),
    )
    result = await fs.start_transcode("http://live", username="alice")
    assert result["playlist"] == "/transcode/old/low.m3u8"
    assert sessions["old"]["last_access"] > 0
    do_start.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_reuses_valid_session(routing, tmp_path):
    state, do_start = routing
    _existing(state, tmp_path, "http://live")
    with patch("ffmpeg_session._try_reuse_session", new=AsyncMock(return_value={"reused": 1})):
        assert await fs.start_transcode("http://live") == {"reused": 1}
    do_start.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_cleans_up_unreusable_session(routing, tmp_path):
    state, do_start = routing
    sessions = _existing(state, tmp_path, "http://live")
    with patch("ffmpeg_session._try_reuse_session", new=AsyncMock(return_value=None)):
        assert await fs.start_transcode("http://live") == {"fresh": True}
    assert "old" not in sessions
    do_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_fast_live_without_dvr(routing):
    with patch("ffmpeg_session._start_fast_live", new=AsyncMock(return_value={"fast": 1})) as fast:
        assert await fs.start_transcode("http://live", fast_start=True) == {"fast": 1}
    assert _call(fast).args == ("http://live", "", "", True)


@pytest.mark.asyncio
async def test_start_retries_series_after_clearing_probe_cache(routing):
    _, do_start = routing
    do_start.side_effect = [HTTPException(500), {"retried": 1}]
    with patch("ffmpeg_session.invalidate_series_probe_cache") as invalidate:
        result = await fs.start_transcode("http://ep", "series", series_id=3, episode_id=4)
    assert result == {"retried": 1}
    invalidate.assert_called_once_with(3, 4)
    assert do_start.await_count == 2


# =============================================================================
# Progress, snapshots, responses
# =============================================================================


def test_progress_reads_selected_fast_start_playlist(state, tmp_path):
    (tmp_path / "high.m3u8").write_text("#EXTINF:2.0,\nh0.ts\n")
    state[0]["s"] = {"dir": str(tmp_path), "fast_start": True, "high_selected": True}
    assert fs.get_session_progress("s") == {"segment_count": 1, "duration": 2.0}


def test_adaptive_response_with_captions_and_generation(state):
    state[0]["s"] = {
        "playlist_generation": 3,
        "closed_captions": True,
        "playback_buffer_seconds": 8,
    }
    response = fs._adaptive_session_response("s")
    assert response["playlist"] == "/transcode/s/low.m3u8?generation=3"
    assert response["caption_playlist"] == "/transcode/s/low-captions.m3u8?generation=3"
    assert response["playback_buffer_seconds"] == 8.0
    missing = fs._adaptive_session_response("missing")
    assert missing["playlist"] == "/transcode/missing/low.m3u8"
    assert missing["playback_buffer_seconds"] == 12.0
    assert "caption_playlist" not in missing


def test_get_existing_session_variants(state, tmp_path):
    sessions, urls = state
    assert fs._get_existing_session("http://none") == (None, False, 0.0)
    urls["http://dangling"] = "gone"
    assert fs._get_existing_session("http://dangling") == (None, False, 0.0)
    sessions["s"] = _live(tmp_path, seek_offset=7)
    urls["http://live"] = "s"
    assert fs._get_existing_session("http://live") == ("s", True, 7)


@pytest.mark.asyncio
async def test_try_reuse_live_session_builds_caption_response(state, tmp_path):
    sessions, _ = state
    (tmp_path / "stream.m3u8").write_text("#EXTINF:4.0,\nseg000.ts\n")
    (tmp_path / "seg000.ts").write_bytes(b"x")
    (tmp_path / "captions.m3u8").write_text("#EXTM3U\n")
    subs = [{"index": 2, "lang": "eng", "name": "English"}]
    sessions["s"] = _live(tmp_path, subtitles=subs, duration=60, seek_offset=0)
    result = await fs._try_reuse_session("s", "http://live", False, "live")
    assert result is not None
    assert result["transcoded_duration"] == 4.0
    assert result["caption_playlist"] == "/transcode/s/captions.m3u8"
    assert result["subtitles"][0]["url"] == "/subs/s/sub0.vtt"
    assert await fs._try_reuse_session("missing", "http://live", False, "live") is None


@pytest.mark.asyncio
async def test_try_reuse_vod_uses_settings(state):
    settings = {
        "transcode_hw": "nvenc",
        "probe_series": True,
        "max_resolution": "4k",
        "quality": "low",
    }
    with (
        patch("ffmpeg_session.get_settings", return_value=settings),
        patch(
            "ffmpeg_session._handle_existing_vod_session",
            new=AsyncMock(return_value={"v": 1}),
        ) as handle,
    ):
        assert await fs._try_reuse_session("s", "http://ep", True, "series") == {"v": 1}
    handle.assert_awaited_once_with("s", "http://ep", "nvenc", True, "4k", "low")


def test_cleanup_invalid_session(state, tmp_path):
    sessions, urls = state
    sessions["s"] = _live(tmp_path)
    urls["http://live"] = "s"
    fs._cleanup_invalid_session("http://live", "s")
    assert not sessions
    assert not urls


# =============================================================================
# Seek
# =============================================================================


@pytest.fixture
def seek_env(state, tmp_path):
    sessions, _ = state
    old = FakeProcess()
    sessions["v"] = {
        "dir": str(tmp_path),
        "process": old,
        "url": "http://movie",
        "is_vod": True,
        "subtitles": [{"index": 2, "lang": "eng", "name": "English"}, 7],
        "series_id": None,
        "episode_id": None,
        "last_access": time.time(),
    }
    process = FakeProcess()
    launch = AsyncMock(return_value=process)
    with (
        patch("ffmpeg_session.get_settings", return_value={"probe_movies": True}),
        patch("ffmpeg_session.get_hls_segment_duration", return_value=3.0),
        patch("ffmpeg_session.resolve_hls_master_playlist", side_effect=lambda url: url),
        patch("ffmpeg_session.probe_media", return_value=(None, [])) as probe,
        patch("ffmpeg_session.get_user_agent", return_value="ua"),
        patch(
            "ffmpeg_session.build_hls_ffmpeg_cmd",
            side_effect=lambda *a, **k: ["ffmpeg", "-i", a[0], "-max_delay", "0", "out"],
        ),
        patch("ffmpeg_session._launch_ffmpeg", new=launch),
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
        patch("ffmpeg_session.get_caption_timestamp_origin") as origin,
    ):
        yield sessions, tmp_path, old, process, launch, probe, origin


@pytest.mark.asyncio
async def test_seek_unknown_or_live_session_is_404(state, tmp_path):
    state[0]["live"] = _live(tmp_path)
    for sid in ("missing", "live"):
        with pytest.raises(HTTPException) as error:
            await fs.seek_transcode(sid, 10)
        assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_smart_seek_reuses_existing_segment(seek_env):
    sessions, directory, old, _, launch, _, _ = seek_env
    (directory / "seg001.ts").write_bytes(b"x" * 2000)
    result = await fs.seek_transcode("v", 4.0)
    assert result == {"session_id": "v", "playlist": "/transcode/v/stream.m3u8"}
    assert sessions["v"]["seek_offset"] == 4.0
    assert "#EXT-X-MEDIA-SEQUENCE:1" in (directory / "stream.m3u8").read_text()
    assert old.returncode is None
    launch.assert_not_awaited()


@pytest.mark.asyncio
async def test_seek_restarts_ffmpeg_at_target(seek_env):
    sessions, directory, old, process, launch, probe, origin = seek_env
    for name in ("seg000.ts", "seg005.ts", "segxyz.ts", "sub0.vtt", "stream.m3u8"):
        (directory / name).write_bytes(b"x")
    (directory / "session.json").write_text(json.dumps({"seek_offset": 0}))
    with patch("ffmpeg_session._wait_for_playlist", new=AsyncMock(return_value=True)):
        result = await fs.seek_transcode("v", 9.0)
    assert result == {"ok": True, "segment": 3, "time": 9.0}
    assert old.returncode == -15
    assert launch.await_args.args[0] == [
        "ffmpeg",
        "-ss",
        "9.0",
        "-i",
        "http://movie",
        "-output_ts_offset",
        "-9.0",
        "-max_delay",
        "0",
        "out",
        "-start_number",
        "3",
    ]
    assert (directory / "seg000.ts").exists()
    assert (directory / "segxyz.ts").exists()
    assert not (directory / "seg005.ts").exists()
    assert not (directory / "sub0.vtt").exists()
    assert "#EXT-X-TARGETDURATION:3" in (directory / "sub0.m3u8").read_text()
    assert json.loads((directory / "session.json").read_text())["seek_offset"] == 9.0
    assert sessions["v"]["process"] is process
    probe.assert_called_once_with("http://movie", None, None)
    origin.assert_called_once_with("v")


@pytest.mark.asyncio
async def test_seek_timeout_without_probe_tolerates_bad_session_json(seek_env):
    sessions, directory, _, _, _, probe, origin = seek_env
    sessions["v"].update(series_id=5, subtitles=[])
    (directory / "session.json").write_text("{bad")
    with (
        patch("ffmpeg_session._wait_for_playlist", new=AsyncMock(return_value=False)),
        pytest.raises(HTTPException) as error,
    ):
        await fs.seek_transcode("v", 1.0)
    assert error.value.status_code == 500
    probe.assert_not_called()
    origin.assert_not_called()
    assert (directory / "session.json").read_text() == "{bad"


@pytest.mark.asyncio
async def test_seek_kills_new_process_when_session_disappears(seek_env):
    _, _, _, process, _, _, _ = seek_env
    with (
        patch("ffmpeg_session._update_session_process", return_value=False),
        pytest.raises(HTTPException) as error,
    ):
        await fs.seek_transcode("v", 1.0)
    assert error.value.status_code == 404
    assert process.returncode == -15


@pytest.mark.asyncio
async def test_seek_caption_preparation_failure(seek_env):
    _, _, _, _, launch, _, _ = seek_env
    with (
        patch("ffmpeg_session._write_empty_caption_playlists", side_effect=OSError),
        pytest.raises(HTTPException) as error,
    ):
        await fs.seek_transcode("v", 1.0)
    assert error.value.status_code == 500
    launch.assert_not_awaited()


# =============================================================================
# Playback health for fast start
# =============================================================================


def test_fast_start_health_selects_and_drops_high(state, tmp_path):
    sessions, _ = state
    policy = MagicMock(bandwidth_saver=False)
    policy.observe.return_value = False
    upgrade = MagicMock(reason="ready", samples=[])
    upgrade.observe.return_value = True
    high = FakeProcess()
    sessions["f"] = {
        "username": "u",
        "dir": str(tmp_path),
        "fast_start": True,
        "playback_policy": policy,
        "upgrade_policy": upgrade,
        "high_process": high,
        "closed_captions": True,
        "playlist_generation": 2,
    }
    health = PlaybackHealth(buffer_seconds=10, waiting=False, observed_bitrate=1e8)
    with (
        patch("ffmpeg_session.ready_bitrate", return_value=5e6),
        patch("ffmpeg_session.aligned", return_value=True),
    ):
        result = fs.report_playback_health("f", "u", health)
        assert result == {
            "bandwidth_saver": False,
            "playlist": "/transcode/f/high.m3u8?generation=2",
            "caption_playlist": "/transcode/f/high-captions.m3u8?generation=2",
        }
        assert sessions["f"]["recovery_bitrate"] == 5e6
        assert policy.observe.call_args.kwargs["recovery_ready"] is True
        high.returncode = 1
        result = fs.report_playback_health("f", "u", health)
    assert result["playlist"] == "/transcode/f/low.m3u8?generation=2"
    assert sessions["f"]["high_selected"] is False


# =============================================================================
# Fast-start startup
# =============================================================================


def _startup(tmp_path, **kwargs):
    return fs._FastLiveStartup(
        session_id="f",
        directory=str(tmp_path),
        processes=[],
        started=time.monotonic(),
        poll_interval_sec=0,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_startup_launch_tracks_process(tmp_path):
    process = FakeProcess()
    startup = _startup(tmp_path, is_disconnected=None)
    with (
        patch("ffmpeg_session._launch_ffmpeg", new=AsyncMock(return_value=process)),
        patch("ffmpeg_session._spawn_background_task", side_effect=_close) as spawn,
    ):
        assert await startup.launch(["ffmpeg"], "ingest") is process
    assert startup.processes == [process]
    spawn.assert_called_once()


@pytest.mark.asyncio
async def test_startup_wait_ready_paths(tmp_path):
    startup = _startup(tmp_path, is_disconnected=AsyncMock(return_value=False))
    with (
        patch("ffmpeg_session.ready_bitrate", side_effect=[0, 1]),
        patch("ffmpeg_session.playlist_duration", return_value=10),
    ):
        await startup.wait_ready("low.m3u8", _proc(FakeProcess()), 5)

    with pytest.raises(HTTPException) as error:
        await startup.wait_ready("low.m3u8", _proc(FakeProcess(alive=False)))
    assert error.value.status_code == 500

    startup.is_disconnected = AsyncMock(return_value=True)
    with pytest.raises(HTTPException) as error:
        await startup.wait_ready("low.m3u8", _proc(FakeProcess()))
    assert error.value.status_code == 499


@pytest.fixture
def fast_env(state, tmp_path):
    processes = []

    def ingest_command(url, directory, user_agent, restarting=False):
        directory = pathlib.Path(directory)
        (directory / "input_000.ts").write_bytes(b"x")
        (directory / "input.m3u8").write_text("#EXTINF:2.0,\ninput_000.ts\n")
        return ["ingest", url, str(restarting)]

    async def launch(cmd):
        process = FakeProcess()
        processes.append((cmd, process))
        return process

    with (
        patch("ffmpeg_session.get_settings", return_value={"transcode_hw": "software"}),
        patch("ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch("ffmpeg_session.get_user_agent", return_value="ua"),
        patch("ffmpeg_session.ingest_command", side_effect=ingest_command),
        patch(
            "ffmpeg_session.encoder_command",
            side_effect=lambda d, hw, res, q, deint, high, **k: [
                "enc",
                res,
                str(k.get("restarting")),
            ],
        ),
        patch("ffmpeg_session._launch_ffmpeg", side_effect=launch) as launch_mock,
        patch("ffmpeg_session.ready_bitrate", return_value=1),
        patch("ffmpeg_session.playlist_duration", return_value=100),
        patch("ffmpeg_session.segment_pts", return_value=1.5),
        patch("ffmpeg_session.probe_audio", return_value=None),
        patch("ffmpeg_session.probe_closed_captions", return_value=True),
        patch("ffmpeg_session.probe_video_codec", return_value="h264"),
    ):
        yield state[0], tmp_path, processes, launch_mock


@pytest.mark.asyncio
async def test_start_fast_live_builds_pipeline(fast_env):
    sessions, _, processes, _ = fast_env
    watchdog = MagicMock()

    def spawn(coro):
        coro.close()
        return watchdog

    with patch("ffmpeg_session._spawn_background_task", side_effect=spawn):
        result = await fs._start_fast_live("http://live", "alice", "src", True)
    sid = result["session_id"]
    session = sessions[sid]
    ingest, low, high = (process for _, process in processes)
    assert [cmd[0:2] for cmd, _ in processes] == [
        ["ingest", "http://live"],
        ["enc", "720p"],
        ["enc", "1080p"],
    ]
    assert session["process"] is low
    assert session["high_process"] is high
    assert session["extra_processes"] == [ingest, high]
    assert session["watchdog_task"] is watchdog
    assert session["closed_captions"] is True
    assert session["origin_pts"] == 1.5
    assert session["low_restart_command"] == ["enc", "720p", "True"]
    assert session["watchdog_stale_after"] == 45.0
    directory = pathlib.Path(session["dir"])
    assert "high.m3u8" in (directory / "master.m3u8").read_text()
    assert (directory / "high-captions.m3u8").exists()
    assert result["caption_playlist"] == f"/transcode/{sid}/low-captions.m3u8"


@pytest.mark.asyncio
async def test_start_fast_live_cancels_watchdog_for_replaced_session(fast_env):
    sessions, _, _, _ = fast_env
    watchdog = MagicMock()

    def spawn(coro):
        name = coro.__name__
        coro.close()
        if name == "_watch_fast_live":
            sessions.clear()
        return watchdog

    with patch("ffmpeg_session._spawn_background_task", side_effect=spawn):
        result = await fs._start_fast_live("http://live", "alice", "src", True)
    watchdog.cancel.assert_called_once()
    assert result["playlist"].endswith("/low.m3u8")


@pytest.mark.asyncio
async def test_start_fast_live_continues_without_high_encoder(fast_env):
    sessions, _, processes, launch = fast_env
    real = launch.side_effect

    async def launch_or_fail(cmd):
        if cmd[:2] == ["enc", "1080p"]:
            raise OSError("no gpu")
        return await real(cmd)

    launch.side_effect = launch_or_fail
    with patch("ffmpeg_session._spawn_background_task", side_effect=_close):
        result = await fs._start_fast_live("http://live", "alice", "src", True)
    session = sessions[result["session_id"]]
    assert "high_process" not in session
    directory = pathlib.Path(session["dir"])
    assert "high.m3u8" not in (directory / "master.m3u8").read_text()
    assert not (directory / "high-captions.m3u8").exists()
    assert len(processes) == 2


@pytest.mark.asyncio
async def test_start_fast_live_failure_releases_everything(fast_env):
    sessions, root, processes, launch = fast_env
    real = launch.side_effect

    async def launch_or_fail(cmd):
        if cmd[:2] == ["enc", "1080p"]:
            raise RuntimeError("boom")
        return await real(cmd)

    launch.side_effect = launch_or_fail
    with (
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
        pytest.raises(RuntimeError),
    ):
        await fs._start_fast_live("http://live", "alice", "src", True)
    assert all(process.returncode == -15 for _, process in processes)
    assert not sessions
    assert not list(root.iterdir())


# =============================================================================
# Process helpers and monitors
# =============================================================================


def test_dead_process_signals_are_noops():
    process = fs._DeadProcess()
    process.terminate()
    process.kill()
    assert process.returncode == -1


def test_kill_process_escalates_to_sigkill():
    process = StubbornProcess()
    with patch("ffmpeg_session.time.sleep") as sleep:
        assert fs._kill_process(process) is True
    assert process.returncode == -9
    assert sleep.call_count == 10


@pytest.mark.asyncio
async def test_monitor_stderr_collects_lines(caplog):
    process = AsyncProcess(lines=[b"frame=1\n", b"Fatal error\n"])
    lines: list[str] = []
    with caplog.at_level(logging.DEBUG, logger="ffmpeg_session"):
        await fs._monitor_ffmpeg_stderr(_proc(process), "s", lines)
    assert lines == ["frame=1", "Fatal error"]
    assert [r.levelno for r in caplog.records if "ffmpeg:s" in r.getMessage()] == [
        logging.DEBUG,
        logging.WARNING,
    ]
    process.stderr = None
    with pytest.raises(ValueError):
        await fs._monitor_ffmpeg_stderr(_proc(process), "s")


@pytest.mark.asyncio
async def test_monitor_resume_invalidates_quick_failure(state, tmp_path):
    sessions, urls = state
    sessions["s"] = {"dir": str(tmp_path)}
    urls["http://m"] = "s"
    await fs._monitor_resume_ffmpeg(_proc(AsyncProcess(exit_code=1)), "s", "http://m")
    assert not sessions
    assert not urls
    assert not tmp_path.exists()


@pytest.mark.asyncio
async def test_monitor_seek_logs_failure(caplog):
    with caplog.at_level(logging.WARNING, logger="ffmpeg_session"):
        await fs._monitor_seek_ffmpeg(_proc(AsyncProcess(exit_code=2)), "s")
    assert "Seek ffmpeg exited with code 2 for session s" in caplog.text


@pytest.mark.asyncio
async def test_spawn_background_task_tracks_until_done():
    async def work():
        return None

    task = fs._spawn_background_task(work())
    assert task in fs._background_tasks
    await task
    await asyncio.sleep(0)
    assert task not in fs._background_tasks


@pytest.mark.asyncio
async def test_wait_for_playlist_paths(tmp_path):
    playlist = tmp_path / "stream.m3u8"
    with pytest.raises(HTTPException) as error:
        await fs._wait_for_playlist(
            playlist,
            _proc(FakeProcess()),
            is_disconnected=AsyncMock(return_value=True),
        )
    assert error.value.status_code == 499
    assert await fs._wait_for_playlist(playlist, _proc(FakeProcess(alive=False))) is False
    playlist.write_text("#EXTINF:3,\nseg000.ts\n")
    assert (
        await fs._wait_for_playlist(
            playlist,
            _proc(FakeProcess()),
            min_segments=2,
            timeout_sec=0.02,
            poll_interval_sec=0.001,
        )
        is False
    )
    playlist.write_text("#EXTINF:3,\nseg000.ts\n#EXTINF:3,\nseg001.ts\n")
    (tmp_path / "seg000.ts").write_bytes(b"x" * 2000)
    (tmp_path / "seg001.ts").write_bytes(b"x")
    assert await fs._wait_for_playlist(playlist, _proc(FakeProcess()), min_segments=2) is True


def test_regenerate_playlist_skips_bad_names_and_empty(tmp_path):
    (tmp_path / "segabc.ts").write_bytes(b"x" * 2000)
    with patch("ffmpeg_session.get_hls_segment_duration", return_value=3.0):
        fs._regenerate_playlist(tmp_path, 0)
    assert not (tmp_path / "stream.m3u8").exists()


# =============================================================================
# Existing VOD sessions
# =============================================================================


@pytest.fixture
def vod_env(state, tmp_path):
    sessions, _ = state
    (tmp_path / "seg000.ts").write_bytes(b"x" * 2000)
    (tmp_path / "seg001.ts").write_bytes(b"x" * 2000)
    (tmp_path / "stream.m3u8").write_text("#EXTINF:3.0,\nseg000.ts\n#EXTINF:3.0,\nseg001.ts\n")
    sessions["v"] = {
        "dir": str(tmp_path),
        "process": FakeProcess(alive=False),
        "url": "http://movie",
        "is_vod": True,
        "seek_offset": 0,
        "duration": 100,
    }
    with (
        patch("ffmpeg_session.resolve_hls_master_playlist", side_effect=lambda url: url),
        patch("ffmpeg_session.probe_media", return_value=(None, [])),
        patch("ffmpeg_session.get_user_agent", return_value="ua"),
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
        patch("ffmpeg_session._wait_for_playlist", new=AsyncMock(return_value=True)) as wait,
    ):
        yield sessions, tmp_path, wait


@pytest.mark.asyncio
async def test_existing_vod_missing_session(state):
    assert await fs._handle_existing_vod_session("missing", "http://m", "software", False) is None


@pytest.mark.asyncio
async def test_existing_vod_active_session_is_reused(vod_env):
    sessions, _, wait = vod_env
    sessions["v"]["process"] = FakeProcess()
    result = await fs._handle_existing_vod_session("v", "http://movie", "software", False)
    assert result is not None
    assert result["transcoded_duration"] == 6.0
    wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_vod_with_seek_offset_returns_cache(vod_env):
    sessions, _, wait = vod_env
    sessions["v"]["seek_offset"] = 30
    result = await fs._handle_existing_vod_session("v", "http://movie", "software", False)
    assert result is not None
    assert result["seek_offset"] == 30
    wait.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_vod_resume_appends_to_playlist(vod_env):
    sessions, directory, _ = vod_env
    process = AsyncProcess()

    async def launch(cmd):
        (directory / "seg002.ts").write_bytes(b"x")
        return process

    cmd = ["ffmpeg", "-i", "http://movie", "-hls_flags", "independent_segments", "out"]
    with (
        patch("ffmpeg_session.build_hls_ffmpeg_cmd", return_value=cmd),
        patch("ffmpeg_session._launch_ffmpeg", side_effect=launch) as launch_mock,
    ):
        result = await fs._handle_existing_vod_session("v", "http://movie", "software", True)
    assert result is not None
    assert launch_mock.call_args.args[0] == [
        "ffmpeg",
        "-ss",
        "6.0",
        "-i",
        "http://movie",
        "-hls_flags",
        "independent_segments+append_list",
        "out",
        "-start_number",
        "2",
    ]
    assert sessions["v"]["process"] is process


@pytest.mark.asyncio
async def test_existing_vod_resume_adds_hls_flags_and_polls(vod_env):
    _, _, wait = vod_env
    with (
        patch("ffmpeg_session.build_hls_ffmpeg_cmd", return_value=["ffmpeg", "-i", "u", "out"]),
        patch(
            "ffmpeg_session._launch_ffmpeg",
            new=AsyncMock(return_value=AsyncProcess()),
        ) as launch,
    ):
        result = await fs._handle_existing_vod_session(
            "v",
            "http://movie",
            "software",
            False,
            resume_segment_wait_timeout_sec=0.02,
            poll_interval_sec=0.001,
        )
    assert result is not None
    assert _call(launch).args[0][-4:] == ["-hls_flags", "append_list", "-start_number", "2"]
    wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_vod_resume_dies_immediately(vod_env):
    with (
        patch("ffmpeg_session.build_hls_ffmpeg_cmd", return_value=["ffmpeg", "-i", "u", "out"]),
        patch(
            "ffmpeg_session._launch_ffmpeg",
            new=AsyncMock(return_value=AsyncProcess(alive=False)),
        ),
    ):
        assert await fs._handle_existing_vod_session("v", "http://movie", "software", False) is None


@pytest.mark.asyncio
async def test_existing_vod_resume_session_vanishes(vod_env):
    process = AsyncProcess()
    with (
        patch("ffmpeg_session.build_hls_ffmpeg_cmd", return_value=["ffmpeg", "-i", "u", "out"]),
        patch("ffmpeg_session._launch_ffmpeg", new=AsyncMock(return_value=process)),
        patch("ffmpeg_session._update_session_process", return_value=False),
    ):
        assert await fs._handle_existing_vod_session("v", "http://movie", "software", False) is None
    assert process.returncode == -15


# =============================================================================
# _do_start_transcode
# =============================================================================


@pytest.fixture
def do_env(state, tmp_path):
    media = MediaInfo(
        video_codec="h264",
        audio_codec="aac",
        pix_fmt="yuv420p",
        height=1080,
        duration=120,
        video_bitrate=8_000_000,
        interlaced=True,
        subtitle_codecs=["subrip"],
    )
    subs = [fs.SubtitleStream(index=2, lang="eng", name="English")]
    process = AsyncProcess()
    settings = {"max_resolution": "4k", "probe_movies": True, "probe_live": True}
    with (
        patch("ffmpeg_session.get_settings", return_value=settings),
        patch("ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch("ffmpeg_session.resolve_hls_master_playlist", side_effect=lambda url: url),
        patch("ffmpeg_session.probe_media", return_value=(media, subs)),
        patch("ffmpeg_session.get_user_agent", return_value="ua"),
        patch(
            "ffmpeg_session.build_hls_ffmpeg_cmd",
            side_effect=lambda *a, **k: ["ffmpeg", "-i", a[0], "out"],
        ) as build,
        patch("ffmpeg_session._launch_ffmpeg", new=AsyncMock(return_value=process)) as launch,
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
        patch("ffmpeg_session.get_caption_timestamp_origin"),
    ):
        yield state[0], tmp_path, process, build, launch


@pytest.mark.asyncio
async def test_do_start_vod_with_seek_offset_persists_probe(do_env):
    sessions, _, _, build, launch = do_env
    with patch("ffmpeg_session._wait_for_playlist", new=AsyncMock(return_value=True)) as wait:
        result = await fs._do_start_transcode("http://movie", "movie", 1, 2, 42.0)
    assert _call(launch).args[0] == ["ffmpeg", "-ss", "42.0", "-i", "http://movie", "out"]
    assert _call(wait).kwargs["timeout_sec"] == 40.0
    session = sessions[result["session_id"]]
    info = json.loads((pathlib.Path(session["dir"]) / "session.json").read_text())
    assert info["probe"]["video_codec"] == "h264"
    assert info["probe"]["interlaced"] is True
    assert info["seek_offset"] == 42.0
    assert result["caption_playlist"].endswith("/captions.m3u8")
    assert build.call_args.args[6:8] == ("4k", "high")


@pytest.mark.asyncio
async def test_do_start_bandwidth_saver_lowers_quality(do_env):
    _, _, _, build, _ = do_env
    with patch("ffmpeg_session._wait_for_playlist", new=AsyncMock(return_value=True)):
        await fs._do_start_transcode("http://live", "live", None, None, 0, bandwidth_saver=True)
    assert build.call_args.args[6:8] == ("720p", "low")
    assert build.call_args.kwargs["allow_upscale"] is False


@pytest.mark.asyncio
async def test_do_start_caption_failure_removes_output(do_env):
    sessions, root, _, _, launch = do_env
    with (
        patch("ffmpeg_session._write_caption_master", side_effect=OSError),
        pytest.raises(HTTPException) as error,
    ):
        await fs._do_start_transcode("http://movie", "movie", None, None, 0)
    assert error.value.status_code == 500
    assert not list(root.iterdir())
    launch.assert_not_awaited()
    assert not sessions


@pytest.mark.asyncio
async def test_do_start_reports_failed_startup(do_env, caplog):
    sessions, root, process, _, _ = do_env
    process.exit_code = 1
    with (
        patch("ffmpeg_session._wait_for_playlist", new=AsyncMock(return_value=False)),
        caplog.at_level(logging.ERROR, logger="ffmpeg_session"),
        pytest.raises(HTTPException) as error,
    ):
        await fs._do_start_transcode("http://movie", "movie", None, None, 0)
    assert error.value.status_code == 500
    assert "failed (exit 1): unknown" in caplog.text
    assert not sessions
    assert not list(root.iterdir())


# =============================================================================
# Fast-start recovery helpers
# =============================================================================


def test_playlist_marker_and_segments(tmp_path):
    path = tmp_path / "low.m3u8"
    assert fs._playlist_marker(path) is None
    assert fs._playlist_segments(path) == set()
    path.write_text("#EXTM3U\n\nlow_1.ts\n../escape.ts\n#EXTINF:2,\nlow_2.ts\n")
    stat = path.stat()
    assert fs._playlist_marker(path) == (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    assert fs._playlist_segments(path) == {"low_1.ts", "low_2.ts"}


def test_prepare_encoder_restart():
    command = ["ffmpeg", "-i", "x", "out.m3u8"]
    assert fs._prepare_encoder_restart(command, {"a.ts"}) is command
    numbered = ["ffmpeg", "-hls_start_number_source", "x", "out.m3u8"]
    assert fs._prepare_encoder_restart(numbered, set()) is numbered
    assert fs._prepare_encoder_restart(command, set()) == [
        "ffmpeg",
        "-i",
        "x",
        "-hls_start_number_source",
        "epoch_us",
        "out.m3u8",
    ]
    assert command == ["ffmpeg", "-i", "x", "out.m3u8"]


@pytest.mark.asyncio
async def test_wait_for_fresh_fast_playlist(tmp_path):
    path = tmp_path / "low.m3u8"
    path.write_text("#EXTINF:2,\nlow_1.ts\n#EXTINF:2,\nlow_2.ts\n")
    marker = fs._playlist_marker(path)
    assert not await fs._wait_for_fresh_fast_playlist(
        str(tmp_path),
        "low.m3u8",
        FakeProcess(alive=False),
        None,
    )
    assert not await fs._wait_for_fresh_fast_playlist(
        str(tmp_path),
        "low.m3u8",
        FakeProcess(),
        marker,
        timeout_sec=0.02,
        poll_interval_sec=0.001,
    )
    with (
        patch("ffmpeg_session.ready_bitrate", return_value=1),
        patch("ffmpeg_session.playlist_duration", return_value=4),
    ):
        assert await fs._wait_for_fresh_fast_playlist(
            str(tmp_path),
            "low.m3u8",
            FakeProcess(),
            None,
            previous_segments={"old.ts"},
        )


@pytest.mark.asyncio
async def test_terminate_fast_process_simple_paths():
    dead = FakeProcess(alive=False)
    await fs._terminate_fast_process(dead)
    assert dead.returncode == 0
    vanished = FakeProcess(killed=True)
    await fs._terminate_fast_process(vanished)
    assert vanished.returncode is None
    no_wait = FakeProcess()
    await fs._terminate_fast_process(no_wait)
    assert no_wait.returncode == -15
    graceful = AsyncProcess()
    await fs._terminate_fast_process(graceful)
    assert (graceful.returncode, graceful.waited) == (-15, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError, asyncio.CancelledError])
async def test_terminate_fast_process_escalates(error):
    process = AsyncProcess()
    process.terminate = lambda: None

    async def wait_for(coro, timeout):
        coro.close()
        raise error

    with patch.object(fs.asyncio, "wait_for", wait_for):
        if error is asyncio.CancelledError:
            with pytest.raises(asyncio.CancelledError):
                await fs._terminate_fast_process(process)
        else:
            await fs._terminate_fast_process(process)
    assert process.returncode == -9
    assert process.waited == 1


@pytest.mark.asyncio
async def test_launch_recovery_ffmpeg_success_and_cancel():
    process = AsyncProcess()
    with patch("ffmpeg_session._launch_ffmpeg", new=AsyncMock(return_value=process)):
        assert await fs._launch_recovery_ffmpeg(["ffmpeg"]) is process

    gate = asyncio.Event()
    late = AsyncProcess()

    async def slow(cmd):
        await gate.wait()
        return late

    with patch("ffmpeg_session._launch_ffmpeg", side_effect=slow):
        task = asyncio.create_task(fs._launch_recovery_ffmpeg(["ffmpeg"]))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert late.returncode == -15


@pytest.mark.asyncio
async def test_monitor_fast_process(state, caplog):
    sessions, _ = state
    process = AsyncProcess(lines=[b"open http://user:secret@host/x failed\n"], exit_code=1)
    sessions["f"] = {"process": process}
    with caplog.at_level(logging.WARNING, logger="ffmpeg_session"):
        await fs._monitor_fast_process(_proc(process), "720p", "f")
        await fs._monitor_fast_process(_proc(AsyncProcess(exit_code=1)), "720p", "f")
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "Fast-start 720p exited for f (code 1)" in warnings[0]
    assert "secret" not in warnings[0]
    process.stderr = None
    with pytest.raises(ValueError):
        await fs._monitor_fast_process(_proc(process), "720p", "f")


@pytest.mark.asyncio
async def test_cancel_high_recovery(state):
    sessions, _ = state
    session = {}
    await fs._cancel_high_recovery("f", session)
    task = asyncio.create_task(asyncio.sleep(10))
    session["high_recovery_task"] = task
    sessions["f"] = session
    await fs._cancel_high_recovery("f", session)
    assert task.cancelled()
    assert "high_recovery_task" not in session


def test_log_watchdog_delay_rate_limited(caplog):
    session = {}
    with caplog.at_level(logging.WARNING, logger="ffmpeg_session"):
        fs._log_watchdog_delay("f", session, "pipeline", 100)
        fs._log_watchdog_delay("f", session, "pipeline", 110)
    assert len(caplog.records) == 1
    assert session["pipeline_restart_log_at"] == 100


def _fast_session(tmp_path, **extra) -> dict[str, Any]:
    session: dict[str, Any] = {
        "dir": str(tmp_path),
        "live_recovery": LiveRecoveryPolicy(high_started=0),
        "master_resolution": "1080p",
        "master_audio_bitrate": 0,
        "high_restart_command": ["enc", "high", "out"],
        "ingest_restart_command": ["ingest", "out"],
        "low_restart_command": ["enc", "low", "out"],
        "playlist_generation": 0,
    }
    session.update(extra)
    return session


@pytest.fixture
def recovery_env(state):
    with patch("ffmpeg_session._spawn_background_task", side_effect=_close):
        yield state[0]


@pytest.mark.asyncio
async def test_restart_high_skips_replaced_session(recovery_env, tmp_path):
    session = _fast_session(tmp_path)
    session["live_recovery"].pending.add("high")
    await fs._restart_high_encoder("f", session)
    assert session["live_recovery"].pending == set()
    assert "high_recovering" not in session


@pytest.mark.asyncio
async def test_restart_high_success(recovery_env, tmp_path):
    sessions = recovery_env
    old, ingest, replacement = FakeProcess(), FakeProcess(), AsyncProcess()
    session = _fast_session(tmp_path, high_process=old, extra_processes=[ingest, old])
    sessions["f"] = session
    with (
        patch("ffmpeg_session._launch_ffmpeg", new=AsyncMock(return_value=replacement)) as launch,
        patch("ffmpeg_session._wait_for_fresh_fast_playlist", new=AsyncMock(return_value=True)),
    ):
        task = asyncio.create_task(fs._restart_high_encoder("f", session))
        session["high_recovery_task"] = task
        await task
    assert _call(launch).args[0] == [
        "enc",
        "high",
        "-hls_start_number_source",
        "epoch_us",
        "out",
    ]
    assert old.returncode == -15
    assert session["high_process"] is replacement
    assert session["extra_processes"] == [ingest, replacement]
    assert session["high_recovering"] is False
    assert "high_recovery_task" not in session
    assert "high.m3u8" in (tmp_path / "master.m3u8").read_text()


@pytest.mark.asyncio
async def test_restart_high_timeout_discards_replacement(recovery_env, tmp_path, caplog):
    sessions = recovery_env
    replacement = AsyncProcess()
    session = _fast_session(tmp_path, extra_processes=[])
    sessions["f"] = session
    with (
        patch("ffmpeg_session._launch_ffmpeg", new=AsyncMock(return_value=replacement)),
        patch("ffmpeg_session._wait_for_fresh_fast_playlist", new=AsyncMock(return_value=False)),
        caplog.at_level(logging.ERROR, logger="ffmpeg_session"),
    ):
        await fs._restart_high_encoder("f", session)
    assert "could not restart high-quality output" in caplog.text
    assert session["high_process"] is None
    assert session["extra_processes"] == []
    assert replacement.returncode == -15
    assert "high.m3u8" not in (tmp_path / "master.m3u8").read_text()


@pytest.mark.asyncio
async def test_restart_high_session_replaced_during_launch(recovery_env, tmp_path):
    sessions = recovery_env
    replacement = AsyncProcess()
    session = _fast_session(tmp_path)
    sessions["f"] = session

    async def launch(cmd):
        sessions.clear()
        return replacement

    with (
        patch("ffmpeg_session._launch_ffmpeg", side_effect=launch),
        pytest.raises(asyncio.CancelledError),
    ):
        await fs._restart_high_encoder("f", session)
    assert replacement.returncode == -15
    assert session["high_recovering"] is True


def _pipeline_dir(tmp_path):
    (tmp_path / "input.m3u8").write_text("#EXTINF:2,\ninput_1.ts\n")
    (tmp_path / "input_1.ts").write_bytes(b"x")
    (tmp_path / "input_2.ts.tmp").write_bytes(b"x")
    (tmp_path / "low.m3u8").write_text("#EXTINF:2,\nlow_1.ts\n")


@pytest.mark.asyncio
async def test_restart_pipeline_skips_replaced_session(recovery_env, tmp_path):
    session = _fast_session(tmp_path)
    await fs._restart_fast_pipeline("f", session, "input")
    assert "watchdog_recovering" not in session


@pytest.mark.asyncio
async def test_restart_pipeline_success(recovery_env, tmp_path):
    sessions = recovery_env
    _pipeline_dir(tmp_path)
    low_old, ingest_old, high_old = FakeProcess(), FakeProcess(), FakeProcess()
    session = _fast_session(
        tmp_path,
        process=low_old,
        ingest_process=ingest_old,
        high_process=high_old,
        extra_processes=[ingest_old, high_old],
        high_selected=True,
    )
    sessions["f"] = session
    ingest, low = AsyncProcess(), AsyncProcess()
    with (
        patch("ffmpeg_session._launch_ffmpeg", new=AsyncMock(side_effect=[ingest, low])) as launch,
        patch("ffmpeg_session._wait_for_fresh_fast_playlist", new=AsyncMock(return_value=True)),
    ):
        await fs._restart_fast_pipeline("f", session, "input")
    assert [call.args[0] for call in launch.await_args_list] == [
        ["ingest", "out"],
        ["enc", "low", "out"],
    ]
    assert all(p.returncode == -15 for p in (low_old, ingest_old, high_old))
    assert (session["process"], session["ingest_process"]) == (low, ingest)
    assert session["extra_processes"] == [ingest]
    assert session["high_process"] is None
    assert session["high_selected"] is False
    assert session["playlist_generation"] == 1
    assert "watchdog_recovering" not in session
    assert not (tmp_path / "input.m3u8").exists()
    assert not list(tmp_path.glob("input_*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["ingest", "low"])
async def test_restart_pipeline_timeout_discards_replacements(
    recovery_env,
    tmp_path,
    failing,
    caplog,
):
    sessions = recovery_env
    _pipeline_dir(tmp_path)
    session = _fast_session(tmp_path, process=FakeProcess())
    sessions["f"] = session
    ingest, low = AsyncProcess(), AsyncProcess()
    ready = [False] if failing == "ingest" else [True, False]
    with (
        patch("ffmpeg_session._launch_ffmpeg", new=AsyncMock(side_effect=[ingest, low])),
        patch("ffmpeg_session._wait_for_fresh_fast_playlist", new=AsyncMock(side_effect=ready)),
        caplog.at_level(logging.ERROR, logger="ffmpeg_session"),
    ):
        await fs._restart_fast_pipeline("f", session, "low output")
    assert "failed to recover the pipeline after low output stalled" in caplog.text
    assert isinstance(session["process"], fs._DeadProcess)
    assert session["ingest_process"] is None
    assert session["extra_processes"] == []
    assert ingest.returncode == -15
    assert low.returncode == (-15 if failing == "low" else None)
    assert session["playlist_generation"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["ingest", "low", "ready"])
async def test_restart_pipeline_session_replaced(recovery_env, tmp_path, stage):
    sessions = recovery_env
    _pipeline_dir(tmp_path)
    session = _fast_session(tmp_path, process=FakeProcess())
    sessions["f"] = session
    ingest, low = AsyncProcess(), AsyncProcess()
    launches = iter([ingest, low])

    async def launch(cmd):
        process = next(launches)
        if (stage, process) in (("ingest", ingest), ("low", low)):
            sessions.clear()
        return process

    waits = iter([True, True])

    async def wait(*args, **kwargs):
        result = next(waits)
        if stage == "ready" and args[1] == "low.m3u8":
            sessions.clear()
        return result

    with (
        patch("ffmpeg_session._launch_ffmpeg", side_effect=launch),
        patch("ffmpeg_session._wait_for_fresh_fast_playlist", side_effect=wait),
        pytest.raises(asyncio.CancelledError),
    ):
        await fs._restart_fast_pipeline("f", session, "input")
    assert ingest.returncode == -15
    assert low.returncode == (None if stage == "ingest" else -15)
    assert session["playlist_generation"] == 0
    assert session["watchdog_recovering"] == "input"


# =============================================================================
# Watchdog
# =============================================================================


def _recovery(stalled=(), permit=False, pending=()):
    recovery = MagicMock()
    recovery.stalled.side_effect = lambda stage, alive, ready, now: stage in stalled
    recovery.permit_restart.return_value = permit
    recovery.pending = set(pending)
    return recovery


@pytest.fixture
def watchdog_env(state, tmp_path):
    with (
        patch("ffmpeg_session.ready_bitrate", return_value=1),
        patch("ffmpeg_session._restart_fast_pipeline", new=AsyncMock()) as restart,
        patch("ffmpeg_session._log_watchdog_delay") as delay,
    ):
        yield state[0], tmp_path, restart, delay


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled,stage", [({"input"}, "input"), ({"low"}, "low output")])
async def test_check_restarts_stalled_pipeline(watchdog_env, stalled, stage):
    _, tmp_path, restart, delay = watchdog_env
    recovery = _recovery(stalled=stalled, permit=True)
    session = {"dir": str(tmp_path), "live_recovery": recovery, "ingest_process": FakeProcess()}
    await fs._check_fast_live_session("f", session)
    restart.assert_awaited_once_with("f", session, stage)
    delay.assert_not_called()
    assert recovery.stalled.call_args_list[0].args[1:3] == (True, True)


@pytest.mark.asyncio
async def test_check_delays_pipeline_restart_on_cooldown(watchdog_env):
    _, tmp_path, restart, delay = watchdog_env
    session = {"dir": str(tmp_path), "live_recovery": _recovery(stalled={"low"})}
    await fs._check_fast_live_session("f", session)
    restart.assert_not_awaited()
    delay.assert_called_once()
    assert delay.call_args.args[2] == "pipeline"
    delay.reset_mock()
    session["live_recovery"] = _recovery(stalled={"low"}, pending={"pipeline"})
    await fs._check_fast_live_session("f", session)
    delay.assert_not_called()


@pytest.mark.asyncio
async def test_check_healthy_session_does_nothing(watchdog_env):
    _, tmp_path, restart, delay = watchdog_env
    with patch("ffmpeg_session._spawn_background_task") as spawn:
        await fs._check_fast_live_session("f", {"dir": str(tmp_path), "live_recovery": _recovery()})
    spawn.assert_not_called()
    restart.assert_not_awaited()
    delay.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("current", [True, False])
async def test_check_spawns_high_restart(watchdog_env, current):
    sessions, tmp_path, _, _ = watchdog_env
    session = {"dir": str(tmp_path), "live_recovery": _recovery(stalled={"high"}, permit=True)}
    if current:
        sessions["f"] = session
    task = MagicMock()

    def spawn(coro):
        assert coro.__name__ == "_restart_high_encoder"
        coro.close()
        return task

    with patch("ffmpeg_session._spawn_background_task", side_effect=spawn):
        await fs._check_fast_live_session("f", session)
    assert (session.get("high_recovery_task") is task) is current
    assert task.cancel.called is (not current)


@pytest.mark.asyncio
async def test_check_delays_high_restart_on_cooldown(watchdog_env):
    _, tmp_path, _, delay = watchdog_env
    session = {"dir": str(tmp_path), "live_recovery": _recovery(stalled={"high"})}
    await fs._check_fast_live_session("f", session)
    assert delay.call_args.args[2] == "high-quality"


@pytest.mark.asyncio
async def test_watchdog_logs_errors_and_propagates_cancel(state, caplog):
    sessions, _ = state
    sessions["f"] = {"last_access": time.time()}
    check = AsyncMock(side_effect=[RuntimeError("boom"), asyncio.CancelledError])
    with (
        patch("ffmpeg_session._check_fast_live_session", new=check),
        caplog.at_level(logging.ERROR, logger="ffmpeg_session"),
        pytest.raises(asyncio.CancelledError),
    ):
        await fs._watch_fast_live("f", watchdog_interval_sec=0)
    assert "failed while checking pipeline health" in caplog.text
    assert check.await_count == 2


@pytest.mark.asyncio
async def test_watchdog_skips_idle_session_and_exits_when_removed(state):
    sessions, _ = state
    sessions["f"] = {"last_access": 0}
    check = AsyncMock()
    with patch("ffmpeg_session._check_fast_live_session", new=check):
        task = asyncio.create_task(fs._watch_fast_live("f", watchdog_interval_sec=0))
        for _ in range(3):
            await asyncio.sleep(0)
        sessions.clear()
        await asyncio.wait_for(task, timeout=1)
    check.assert_not_awaited()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
