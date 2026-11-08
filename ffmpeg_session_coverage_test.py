"""Coverage-focused tests for ffmpeg session management."""

from collections.abc import Awaitable, Callable, Collection, Coroutine, Iterable, Iterator
from dataclasses import replace
from types import CoroutineType
from typing import Any, cast, override
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import asyncio
import json
import logging
import pathlib
import time

from fastapi import HTTPException

import pytest

from fast_start_test import video_packet
from ffmpeg_command import MediaInfo
from ffmpeg_session_test import ARCHIVE_URL, FakeProcess
from playback_policy import LiveRecoveryPolicy, PlaybackHealth, PlaybackPolicy

import ffmpeg_session as fs


class FakeStderr:
    def __init__(self, lines: Iterable[bytes] = ()) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


class AsyncProcess(FakeProcess):
    """Fake process with async stderr and wait."""

    def __init__(
        self,
        alive: bool = True,
        lines: Iterable[bytes] = (),
        exit_code: int = 0,
    ) -> None:
        super().__init__(alive=alive)
        self.stderr: FakeStderr | None = FakeStderr(lines)
        self.exit_code = exit_code
        self.pid = 4242
        self.waited = 0

    @override
    async def wait(self) -> int:
        self.waited += 1
        if self.returncode is None:
            self.returncode = self.exit_code
        return self.returncode


class _IgnoresTerminate(AsyncProcess):
    """Ignores SIGTERM so termination must escalate."""

    @override
    def terminate(self) -> None:
        pass


def _close(coro: Coroutine[object, object, None]) -> None:
    coro.close()


def _same_url(url: str) -> str:
    return url


def _proc(process: FakeProcess) -> asyncio.subprocess.Process:
    return cast(asyncio.subprocess.Process, process)


def _call(mock: Mock) -> Any:
    assert mock.call_args is not None
    return mock.call_args


_Sessions = dict[str, fs.TranscodeSession]
# What `patch(...) as name` binds without `new=`.
_Patched = MagicMock | AsyncMock
_State = tuple[_Sessions, dict[str, str]]


@pytest.fixture
def state() -> Iterator[_State]:
    with (
        patch.dict(fs._transcode_sessions, {}, clear=True),
        patch.dict(fs._url_to_session, {}, clear=True),
    ):
        yield fs._transcode_sessions, fs._url_to_session


def _live(tmp_path: pathlib.Path, url: str = "http://live") -> fs.TranscodeSession:
    return fs.TranscodeSession(
        dir=str(tmp_path),
        process=FakeProcess(),
        started=time.time(),
        last_access=time.time(),
        url=url,
    )


# =============================================================================
# Validity, stop, shutdown
# =============================================================================


def test_fast_start_session_with_recovery_stays_valid_without_process(tmp_path: pathlib.Path):
    session = fs.FastLiveSession(
        started=time.time(),
        fast_start=True,
        live_recovery=LiveRecoveryPolicy(),
        process=None,
        dir=str(tmp_path),
        url="http://stream.test/live",
        ingest_restart_command=["ffmpeg", "ingest"],
    )
    assert fs.is_session_valid(session) is True


def test_stop_session_cancels_pending_tasks_and_kills_extra_processes(
    state: _State,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    pending = MagicMock(done=MagicMock(return_value=False))
    extra = FakeProcess()
    sessions["s"] = _live(tmp_path)
    sessions["s"].update({"watchdog_task": pending, "extra_processes": [extra]})
    asyncio.run(fs.stop_session("s", force=True))
    pending.cancel.assert_called_once()
    assert extra.returncode == -15
    assert "s" not in sessions
    assert not tmp_path.exists()


def test_stop_user_stream_matches_url_or_request_url_for_owner_only(
    state: _State,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    # "b" matches only through the URL the client asked for, not the resolved one.
    for sid, user, url, request_url in [
        ("a", "alice", "http://stream", None),
        ("b", "alice", "http://resolved/b", "http://stream"),
        ("c", "bob", "http://stream", None),
    ]:
        directory = tmp_path / sid
        directory.mkdir()
        sessions[sid] = fs.TranscodeSession(
            dir=str(directory),
            process=FakeProcess(),
            started=time.time(),
            url=url,
            username=user,
        )
        if request_url is not None:
            sessions[sid]["request_url"] = request_url
    asyncio.run(fs.stop_user_stream("alice", "http://stream"))
    assert set(sessions) == {"c"}


def test_shutdown_cancels_tasks_and_kills_extras(state: _State, tmp_path: pathlib.Path):
    sessions, _ = state
    task = MagicMock(done=MagicMock(return_value=False))
    extra = FakeProcess()
    sessions["s"] = fs.TranscodeSession(
        watchdog_task=task,
        extra_processes=[extra],
        process=None,
        dir=str(tmp_path),
        started=time.time(),
        url="http://stream.test/live",
    )
    asyncio.run(fs.shutdown())
    task.cancel.assert_called_once()
    assert extra.returncode == -15
    assert not sessions


@pytest.mark.asyncio
async def test_watchdog_stopping_its_own_session_is_not_cancelled(
    state: _State,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    finished = asyncio.Event()

    async def watchdog() -> None:
        await fs.stop_session("s", force=True)
        finished.set()

    sessions["s"] = _live(tmp_path)
    sessions["s"]["fast_start"] = True
    task = asyncio.create_task(watchdog())
    sessions["s"]["watchdog_task"] = task
    await task
    assert finished.is_set()
    assert not sessions


def test_cached_live_stop_keeps_session_but_stops_its_process(
    state: _State,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    sessions["s"] = _live(tmp_path)
    sessions["s"]["last_access"] = 0
    process = sessions["s"]["process"]
    with patch("ffmpeg_session.get_live_cache_timeout", return_value=30):
        asyncio.run(fs.stop_session("s"))
    assert process is not None
    assert process.returncode == -15
    assert sessions["s"]["last_access"] > 0
    assert tmp_path.exists()


# =============================================================================
# Startup recovery
# =============================================================================


def _vod_dir(root: pathlib.Path, name: str, info: object, *, segment: bool = True) -> pathlib.Path:
    directory = root / f"netv_transcode_{name}"
    directory.mkdir()
    (directory / "session.json").write_text(info if isinstance(info, str) else json.dumps(info))
    if segment:
        (directory / "seg000.ts").write_bytes(b"x" * 2000)
    return directory


def _recover(root: object) -> None:
    with (
        patch("ffmpeg_session.get_transcode_dir", return_value=root),
        patch("ffmpeg_session.get_vod_cache_timeout", return_value=3600),
    ):
        fs.cleanup_and_recover_sessions()


def test_recovery_ignores_plain_files(state: _State, tmp_path: pathlib.Path):
    stray = tmp_path / "netv_transcode_file"
    stray.write_text("x")
    _recover(tmp_path)
    assert stray.exists()


def test_recovery_removes_dir_that_cannot_be_stat(state: _State):
    directory = MagicMock()
    directory.is_dir.return_value = True
    directory.stat.side_effect = OSError
    root = MagicMock()
    root.glob.return_value = [directory]
    with patch("ffmpeg_session.shutil.rmtree") as rmtree:
        _recover(root)
    rmtree.assert_called_once_with(directory, ignore_errors=True)


def test_recovery_removes_session_without_segments(state: _State, tmp_path: pathlib.Path):
    directory = _vod_dir(
        tmp_path,
        "empty",
        {"session_id": "e", "url": "http://m", "is_vod": True},
        segment=False,
    )
    _recover(tmp_path)
    assert not directory.exists()
    assert not state[0]


def test_recovery_removes_corrupt_session_json(state: _State, tmp_path: pathlib.Path):
    directory = _vod_dir(tmp_path, "bad", "{not json")
    _recover(tmp_path)
    assert not directory.exists()


@pytest.mark.parametrize(
    "existing_seek,new_seek,winner",
    [(0, 5, "new"), (5, 0, "old"), (0, 0, "new")],
)
def test_recovery_prefers_seeked_or_newer_duplicate(
    state: _State,
    tmp_path: pathlib.Path,
    existing_seek: int,
    new_seek: int,
    winner: str,
):
    sessions, urls = state
    sessions["old"] = fs.TranscodeSession(
        seek_offset=existing_seek,
        last_access=0,
        dir=str(tmp_path),
        process=None,
        started=time.time(),
        url="http://stream.test/live",
    )
    urls["http://m"] = "old"
    _vod_dir(
        tmp_path,
        "new",
        {"session_id": "new", "url": "http://m", "is_vod": True, "seek_offset": new_seek},
    )
    _recover(tmp_path)
    assert "new" in sessions
    assert urls["http://m"] == winner


def test_recovery_restores_probe_cache(state: _State, tmp_path: pathlib.Path):
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


def test_caption_timestamp_origin_paths(state: _State, tmp_path: pathlib.Path):
    sessions, _ = state
    assert fs.get_caption_timestamp_origin("missing") is None
    sessions["cached"] = fs.TranscodeSession(
        dir=str(tmp_path),
        caption_timestamp_origin=3,
        process=None,
        started=time.time(),
        url="http://stream.test/live",
    )
    assert fs.get_caption_timestamp_origin("cached") == 3.0
    sessions["none"] = fs.TranscodeSession(
        dir=str(tmp_path),
        process=None,
        started=time.time(),
        url="http://stream.test/live",
    )
    assert fs.get_caption_timestamp_origin("none") is None
    with patch("ffmpeg_session._caption_timestamp_origin", return_value=4.5):
        assert fs.get_caption_timestamp_origin("none") == 4.5
    assert "caption_timestamp_origin" in sessions["none"]
    assert sessions["none"]["caption_timestamp_origin"] == 4.5

    def vanish(_: str) -> float:
        sessions.pop("gone")
        return 1.0

    sessions["gone"] = fs.TranscodeSession(
        dir=str(tmp_path),
        process=None,
        started=time.time(),
        url="http://stream.test/live",
    )
    with patch("ffmpeg_session._caption_timestamp_origin", side_effect=vanish):
        assert fs.get_caption_timestamp_origin("gone") is None


def test_caption_timestamp_origin_none_without_pts(tmp_path: pathlib.Path):
    (tmp_path / "stream.m3u8").write_text("#EXTINF:bad,\n#EXTINF:4,\nsub/x.ts\nseg0.ts\n")
    with patch("ffmpeg_session.segment_pts", return_value=None) as pts:
        assert fs._caption_timestamp_origin(str(tmp_path)) is None
    pts.assert_called_once_with(tmp_path / "seg0.ts")


# =============================================================================
# start_transcode routing
# =============================================================================


@pytest.fixture
def routing(state: _State) -> Iterator[tuple[_State, AsyncMock]]:
    do_start = AsyncMock(return_value={"fresh": True})
    with (
        patch("ffmpeg_session.get_settings", return_value={}),
        patch("ffmpeg_session._do_start_transcode", new=do_start),
    ):
        yield state, do_start


def _existing(
    state: _State,
    tmp_path: pathlib.Path,
    url: str,
    *,
    username: str | None = None,
    last_access: float | None = None,
    audio_passthrough: bool = False,
    fast_start: bool = False,
) -> _Sessions:
    sessions, urls = state
    session = _live(tmp_path, url=url)
    if username is not None:
        session["username"] = username
    if last_access is not None:
        session["last_access"] = last_access
    # Absent and False are distinct for these flags: set only what the case needs.
    if audio_passthrough:
        session["audio_passthrough"] = True
    if fast_start:
        session["fast_start"] = True
    sessions["old"] = session
    urls[url] = "old"
    return sessions


@pytest.mark.asyncio
async def test_start_rejects_bandwidth_saver_for_vod(routing: tuple[_State, AsyncMock]):
    with pytest.raises(HTTPException) as error:
        await fs.start_transcode("http://m", "movie", bandwidth_saver=True)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_start_enforces_stream_limits(routing: tuple[_State, AsyncMock]):
    with (
        patch("ffmpeg_session.enforce_stream_limits", return_value="full"),
        pytest.raises(HTTPException) as error,
    ):
        await fs.start_transcode("http://live", username="alice")
    assert (error.value.status_code, error.value.detail) == (429, "full")


@pytest.mark.asyncio
async def test_start_rejects_archive_owned_by_other_user(
    routing: tuple[_State, AsyncMock],
    tmp_path: pathlib.Path,
):
    state, _ = routing
    _existing(state, tmp_path, ARCHIVE_URL, username="bob")
    with pytest.raises(HTTPException) as error:
        await fs.start_transcode(ARCHIVE_URL, "movie", username="alice")
    assert error.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bandwidth_saver,audio_passthrough",
    [(True, False), (False, True)],
)
async def test_start_replaces_own_incompatible_session(
    routing: tuple[_State, AsyncMock],
    tmp_path: pathlib.Path,
    bandwidth_saver: bool,
    audio_passthrough: bool,
):
    state, do_start = routing
    sessions = _existing(
        state,
        tmp_path,
        "http://live",
        username="alice",
        audio_passthrough=audio_passthrough,
    )
    result = await fs.start_transcode(
        "http://live",
        username="alice",
        bandwidth_saver=bandwidth_saver,
    )
    assert result == {"fresh": True}
    assert "old" not in sessions
    assert do_start.await_args is not None
    assert do_start.await_args.args[4] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bandwidth_saver,audio_passthrough,fast_start,status",
    [
        (True, False, False, 404),
        (False, True, False, 409),
        (False, False, True, 404),
    ],
)
async def test_start_refuses_other_users_session(
    routing: tuple[_State, AsyncMock],
    tmp_path: pathlib.Path,
    bandwidth_saver: bool,
    audio_passthrough: bool,
    fast_start: bool,
    status: int,
):
    state, _ = routing
    sessions = _existing(
        state,
        tmp_path,
        "http://live",
        username="bob",
        audio_passthrough=audio_passthrough,
        fast_start=fast_start,
    )
    with pytest.raises(HTTPException) as error:
        await fs.start_transcode(
            "http://live",
            username="alice",
            bandwidth_saver=bandwidth_saver,
        )
    assert error.value.status_code == status
    assert "old" in sessions


@pytest.mark.asyncio
async def test_start_rejoins_own_fast_start_session(
    routing: tuple[_State, AsyncMock],
    tmp_path: pathlib.Path,
):
    state, do_start = routing
    sessions = _existing(
        state,
        tmp_path,
        "http://live",
        username="alice",
        last_access=time.time(),
        fast_start=True,
    )
    result = await fs.start_transcode("http://live", username="alice")
    assert result["playlist"] == "/transcode/old/low.m3u8"
    assert "last_access" in sessions["old"]
    assert sessions["old"]["last_access"] > 0
    do_start.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_reuses_valid_session(
    routing: tuple[_State, AsyncMock],
    tmp_path: pathlib.Path,
):
    state, do_start = routing
    _existing(state, tmp_path, "http://live")
    with patch("ffmpeg_session._try_reuse_session", new=AsyncMock(return_value={"reused": 1})):
        assert await fs.start_transcode("http://live") == {"reused": 1}
    do_start.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_cleans_up_unreusable_session(
    routing: tuple[_State, AsyncMock],
    tmp_path: pathlib.Path,
):
    state, do_start = routing
    sessions = _existing(state, tmp_path, "http://live")
    with patch("ffmpeg_session._try_reuse_session", new=AsyncMock(return_value=None)):
        assert await fs.start_transcode("http://live") == {"fresh": True}
    assert "old" not in sessions
    do_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_fast_live_without_dvr(routing: tuple[_State, AsyncMock]):
    with patch("ffmpeg_session._start_fast_live", new=AsyncMock(return_value={"fast": 1})) as fast:
        assert await fs.start_transcode("http://live", fast_start=True) == {"fast": 1}
    assert _call(fast).args == ("http://live", "", "", True)


@pytest.mark.asyncio
async def test_start_retries_series_after_clearing_probe_cache(routing: tuple[_State, AsyncMock]):
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


def test_progress_reads_fast_start_playlist(state: _State, tmp_path: pathlib.Path):
    (tmp_path / "low.m3u8").write_text("#EXTINF:2.0,\nlow_0.ts\n")
    state[0]["s"] = fs.TranscodeSession(
        dir=str(tmp_path),
        fast_start=True,
        process=None,
        started=time.time(),
        url="http://stream.test/live",
    )
    assert fs.get_session_progress("s") == {"segment_count": 1, "duration": 2.0}


def test_adaptive_response_with_captions_and_generation(state: _State, tmp_path: pathlib.Path):
    state[0]["s"] = fs.TranscodeSession(
        playlist_generation=3,
        closed_captions=True,
        playback_buffer_seconds=8,
        dir=str(tmp_path),
        process=None,
        started=time.time(),
        url="http://stream.test/live",
    )
    response = fs._adaptive_session_response("s")
    assert response["playlist"] == "/transcode/s/low.m3u8?generation=3"
    assert "caption_playlist" in response
    assert response["caption_playlist"] == "/transcode/s/low-captions.m3u8?generation=3"
    assert "playback_buffer_seconds" in response
    assert response["playback_buffer_seconds"] == 8.0
    missing = fs._adaptive_session_response("missing")
    assert missing["playlist"] == "/transcode/missing/low.m3u8"
    assert "playback_buffer_seconds" in missing
    assert missing["playback_buffer_seconds"] == 12.0
    assert "caption_playlist" not in missing


def test_get_existing_session_variants(state: _State, tmp_path: pathlib.Path):
    sessions, urls = state
    assert fs._get_existing_session("http://none") == (None, False, 0.0)
    urls["http://dangling"] = "gone"
    assert fs._get_existing_session("http://dangling") == (None, False, 0.0)
    sessions["s"] = _live(tmp_path)
    sessions["s"]["seek_offset"] = 7
    urls["http://live"] = "s"
    assert fs._get_existing_session("http://live") == ("s", True, 7)


@pytest.mark.asyncio
async def test_try_reuse_live_session_builds_caption_response(
    state: _State,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    (tmp_path / "stream.m3u8").write_text("#EXTINF:4.0,\nseg000.ts\n")
    (tmp_path / "seg000.ts").write_bytes(b"x")
    (tmp_path / "captions.m3u8").write_text("#EXTM3U\n")
    subs: list[fs.StoredSubtitle | int] = [{"index": 2, "lang": "eng", "name": "English"}]
    sessions["s"] = _live(tmp_path)
    sessions["s"]["subtitles"] = subs
    sessions["s"]["duration"] = 60
    sessions["s"]["seek_offset"] = 0
    result = await fs._try_reuse_session("s", "http://live", False, "live")
    assert result is not None
    assert "transcoded_duration" in result
    assert result["transcoded_duration"] == 4.0
    assert "caption_playlist" in result
    assert result["caption_playlist"] == "/transcode/s/captions.m3u8"
    assert result["subtitles"][0]["url"] == "/subs/s/sub0.vtt"
    assert await fs._try_reuse_session("missing", "http://live", False, "live") is None


@pytest.mark.asyncio
async def test_try_reuse_vod_uses_settings(state: _State):
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


def test_cleanup_invalid_session(state: _State, tmp_path: pathlib.Path):
    sessions, urls = state
    sessions["s"] = _live(tmp_path)
    urls["http://live"] = "s"
    asyncio.run(fs._cleanup_invalid_session("http://live", "s"))
    assert not sessions
    assert not urls


# =============================================================================
# Seek
# =============================================================================


def _seek_command(*args: str, **kwargs: object) -> list[str]:
    del kwargs
    return ["ffmpeg", "-i", args[0], "-max_delay", "0", "out"]


def _vod_command(*args: str, **kwargs: object) -> list[str]:
    del kwargs
    return ["ffmpeg", "-i", args[0], "out"]


@pytest.fixture
def seek_env(
    state: _State,
    tmp_path: pathlib.Path,
) -> Iterator[
    tuple[_Sessions, pathlib.Path, FakeProcess, FakeProcess, AsyncMock, _Patched, _Patched]
]:
    sessions, _ = state
    old = FakeProcess()
    sessions["v"] = fs.TranscodeSession(
        dir=str(tmp_path),
        process=old,
        url="http://movie",
        is_vod=True,
        subtitles=[{"index": 2, "lang": "eng", "name": "English"}, 7],
        series_id=None,
        episode_id=None,
        last_access=time.time(),
        started=time.time(),
    )
    process = FakeProcess()
    launch = AsyncMock(return_value=process)
    with (
        patch("ffmpeg_session.get_settings", return_value={"probe_movies": True}),
        patch("ffmpeg_session.get_hls_segment_duration", return_value=3.0),
        patch("ffmpeg_session.resolve_hls_master_playlist", side_effect=_same_url),
        patch("ffmpeg_session.probe_media", return_value=(None, [])) as probe,
        patch("ffmpeg_session.get_user_agent", return_value="ua"),
        patch(
            "ffmpeg_session.build_hls_ffmpeg_cmd",
            side_effect=_seek_command,
        ),
        patch("ffmpeg_session._launch_ffmpeg", new=launch),
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
        patch("ffmpeg_session.get_caption_timestamp_origin") as origin,
    ):
        yield sessions, tmp_path, old, process, launch, probe, origin


@pytest.mark.asyncio
async def test_seek_unknown_or_live_session_is_404(state: _State, tmp_path: pathlib.Path):
    state[0]["live"] = _live(tmp_path)
    for sid in ("missing", "live"):
        with pytest.raises(HTTPException) as error:
            await fs.seek_transcode(sid, 10)
        assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_smart_seek_reuses_existing_segment(
    seek_env: tuple[
        _Sessions,
        pathlib.Path,
        FakeProcess,
        FakeProcess,
        AsyncMock,
        _Patched,
        _Patched,
    ],
):
    sessions, directory, old, _, launch, _, _ = seek_env
    (directory / "seg001.ts").write_bytes(b"x" * 2000)
    result = await fs.seek_transcode("v", 4.0)
    assert result == {"session_id": "v", "playlist": "/transcode/v/stream.m3u8"}
    assert "seek_offset" in sessions["v"]
    assert sessions["v"]["seek_offset"] == 4.0
    assert "#EXT-X-MEDIA-SEQUENCE:1" in (directory / "stream.m3u8").read_text()
    assert old.returncode is None
    launch.assert_not_awaited()


@pytest.mark.asyncio
async def test_seek_restarts_ffmpeg_at_target(
    seek_env: tuple[
        _Sessions,
        pathlib.Path,
        FakeProcess,
        FakeProcess,
        AsyncMock,
        _Patched,
        _Patched,
    ],
):
    sessions, directory, old, process, launch, probe, origin = seek_env
    for name in ("seg000.ts", "seg005.ts", "segxyz.ts", "sub0.vtt", "stream.m3u8"):
        (directory / name).write_bytes(b"x")
    (directory / "session.json").write_text(json.dumps({"seek_offset": 0}))
    with patch("ffmpeg_session._wait_for_playlist", new=AsyncMock(return_value=True)):
        result = await fs.seek_transcode("v", 9.0)
    assert result == {"ok": True, "segment": 3, "time": 9.0}
    assert old.returncode == -15
    assert launch.await_args is not None
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
async def test_seek_timeout_without_probe_tolerates_bad_session_json(
    seek_env: tuple[
        _Sessions,
        pathlib.Path,
        FakeProcess,
        FakeProcess,
        AsyncMock,
        _Patched,
        _Patched,
    ],
):
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
async def test_seek_kills_new_process_when_session_disappears(
    seek_env: tuple[
        _Sessions,
        pathlib.Path,
        FakeProcess,
        FakeProcess,
        AsyncMock,
        _Patched,
        _Patched,
    ],
):
    _, _, _, process, _, _, _ = seek_env
    with (
        patch("ffmpeg_session._update_session_process", return_value=False),
        pytest.raises(HTTPException) as error,
    ):
        await fs.seek_transcode("v", 1.0)
    assert error.value.status_code == 404
    assert process.returncode == -15


@pytest.mark.asyncio
async def test_seek_caption_preparation_failure(
    seek_env: tuple[
        _Sessions,
        pathlib.Path,
        FakeProcess,
        FakeProcess,
        AsyncMock,
        _Patched,
        _Patched,
    ],
):
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


def test_fast_start_health_learns_bitrate_from_live_encoder(state: _State, tmp_path: pathlib.Path):
    sessions, _ = state
    policy = MagicMock(spec=PlaybackPolicy, bandwidth_saver=False)
    policy.observe.return_value = PlaybackPolicy()
    encoder = FakeProcess()
    sessions["f"] = fs.TranscodeSession(
        username="u",
        dir=str(tmp_path),
        fast_start=True,
        playback_policy=policy,
        process=encoder,
        closed_captions=True,
        playlist_generation=2,
        recovery_bitrate=1e6,
        started=time.time(),
        url="http://stream.test/live",
    )
    health = PlaybackHealth(buffer_seconds=10, waiting=False, observed_bitrate=1e8)
    with patch("ffmpeg_session.ready_bitrate", return_value=5e6) as ready:
        result = fs.report_playback_health("f", "u", health)
        assert result == {
            "bandwidth_saver": False,
            "playlist": "/transcode/f/low.m3u8?generation=2",
            "caption_playlist": "/transcode/f/low-captions.m3u8?generation=2",
        }
        ready.assert_called_once_with(str(tmp_path), "low.m3u8")
        assert policy.observe.call_args.args[2] == 5e6
        encoder.returncode = 1
        ready.return_value = 9e6
        fs.report_playback_health("f", "u", health)
    assert "recovery_bitrate" in sessions["f"]
    assert sessions["f"]["recovery_bitrate"] == 5e6
    ready.assert_called_once()


def test_health_without_playback_policy_raises_key_error(state: _State, tmp_path: pathlib.Path):
    sessions, _ = state
    sessions["f"] = fs.TranscodeSession(
        username="u",
        dir=str(tmp_path),
        process=None,
        started=time.time(),
        url="http://stream.test/live",
    )
    with pytest.raises(KeyError, match="playback_policy"):
        fs.report_playback_health("f", "u", PlaybackHealth(buffer_seconds=1, waiting=False))


def _fast_health_session(
    tmp_path: pathlib.Path,
    *,
    bandwidth_saver: bool,
    recovery_bitrate: float = 0,
) -> fs.TranscodeSession:
    return fs.TranscodeSession(
        username="u",
        dir=str(tmp_path),
        fast_start=True,
        bandwidth_saver=bandwidth_saver,
        playback_policy=PlaybackPolicy(bandwidth_saver=bandwidth_saver),
        process=FakeProcess(alive=False),
        closed_captions=True,
        recovery_bitrate=recovery_bitrate,
        started=time.time(),
        url="http://stream.test/live",
    )


def test_congestion_on_full_quality_session_omits_playlist_so_clients_retune(
    state: _State,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    sessions["f"] = _fast_health_session(tmp_path, bandwidth_saver=False)
    stalled = PlaybackHealth(buffer_seconds=0, waiting=True)
    results = []
    with patch("ffmpeg_session.time.monotonic") as clock:
        for now in range(0, 26, 2):
            clock.return_value = now
            results.append(fs.report_playback_health("f", "u", stalled))
    assert results[0] == {
        "bandwidth_saver": False,
        "playlist": "/transcode/f/low.m3u8",
        "caption_playlist": "/transcode/f/low-captions.m3u8",
    }
    assert results[-1] == {"bandwidth_saver": True}


def test_recovery_on_saver_session_omits_playlist_so_clients_retune(
    state: _State,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    sessions["f"] = _fast_health_session(tmp_path, bandwidth_saver=True, recovery_bitrate=1000)
    good = PlaybackHealth(buffer_seconds=12, waiting=False, observed_bitrate=100_000)
    results = []
    with patch("ffmpeg_session.time.monotonic") as clock:
        for now in range(0, 62, 2):
            clock.return_value = now
            results.append(fs.report_playback_health("f", "u", good))
    assert results[0] == {
        "bandwidth_saver": True,
        "playlist": "/transcode/f/low.m3u8",
        "caption_playlist": "/transcode/f/low-captions.m3u8",
    }
    assert results[-1] == {"bandwidth_saver": False}


# =============================================================================
# Fast-start startup
# =============================================================================


def _startup(
    tmp_path: pathlib.Path,
    *,
    is_disconnected: Callable[[], Awaitable[bool]] | None,
) -> fs._FastLiveStartup:
    return fs._FastLiveStartup(
        session_id="f",
        directory=str(tmp_path),
        processes=[],
        started=time.monotonic(),
        poll_interval_sec=0,
        is_disconnected=is_disconnected,
    )


@pytest.mark.asyncio
async def test_startup_launch_tracks_process(tmp_path: pathlib.Path):
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
async def test_startup_wait_ready_paths(tmp_path: pathlib.Path):
    startup = _startup(tmp_path, is_disconnected=AsyncMock(return_value=False))
    with (
        patch("ffmpeg_session.ready_bitrate", side_effect=[0, 1]),
        patch("ffmpeg_session.playlist_duration", return_value=10),
    ):
        await startup.wait_ready("low.m3u8", _proc(FakeProcess()), 5)

    with pytest.raises(HTTPException) as error:
        await startup.wait_ready("low.m3u8", _proc(FakeProcess(alive=False)))
    assert error.value.status_code == 500

    startup = replace(startup, is_disconnected=AsyncMock(return_value=True))
    with pytest.raises(HTTPException) as error:
        await startup.wait_ready("low.m3u8", _proc(FakeProcess()))
    assert error.value.status_code == 499


_Launched = list[tuple[list[str], FakeProcess]]


def _encoder_command(
    directory: str,
    hw: str,
    resolution: str,
    quality: str,
    deinterlace: bool,
    **kwargs: object,
) -> list[str]:
    del directory, hw, deinterlace
    return [
        "enc",
        resolution,
        quality,
        str(kwargs["allow_upscale"]),
        str(kwargs.get("restarting", False)),
    ]


@pytest.fixture
def fast_env(
    state: _State,
    tmp_path: pathlib.Path,
) -> Iterator[tuple[_Sessions, pathlib.Path, _Launched, _Patched]]:
    processes: _Launched = []

    def ingest_command(
        url: str,
        directory: str,
        user_agent: str | None,
        restarting: bool = False,
    ) -> list[str]:
        output = pathlib.Path(directory)
        (output / "input_000.ts").write_bytes(b"x")
        (output / "input.m3u8").write_text("#EXTINF:2.0,\ninput_000.ts\n")
        return ["ingest", url, str(restarting)]

    async def launch(cmd: list[str]) -> FakeProcess:
        process = FakeProcess()
        processes.append((cmd, process))
        return process

    with (
        patch(
            "ffmpeg_session.get_settings",
            return_value={"transcode_hw": "software", "max_resolution": "4k", "quality": "high"},
        ),
        patch("ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch("ffmpeg_session.get_user_agent", return_value="ua"),
        patch("ffmpeg_session.ingest_command", side_effect=ingest_command),
        patch(
            "ffmpeg_session.encoder_command",
            side_effect=_encoder_command,
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
@pytest.mark.parametrize(
    "bandwidth_saver,encoder",
    [(False, ["enc", "4k", "high", "True"]), (True, ["enc", "720p", "low", "False"])],
)
async def test_start_fast_live_runs_ingest_and_one_encoder(
    fast_env: tuple[_Sessions, pathlib.Path, _Launched, _Patched],
    bandwidth_saver: bool,
    encoder: list[str],
):
    sessions, _, processes, _ = fast_env
    watchdog = MagicMock()

    def spawn(coro: Coroutine[object, object, None]) -> MagicMock:
        coro.close()
        return watchdog

    with patch("ffmpeg_session._spawn_background_task", side_effect=spawn):
        result = await fs._start_fast_live(
            "http://live",
            "alice",
            "src",
            True,
            bandwidth_saver=bandwidth_saver,
        )
    sid = result["session_id"]
    session = sessions[sid]
    (ingest_cmd, ingest), (low_cmd, low) = processes
    assert ingest_cmd == ["ingest", "http://live", "False"]
    assert low_cmd == [*encoder, "False"]
    assert fs._is_fast_live(session)
    assert "low_restart_command" in session
    assert session["low_restart_command"] == [*encoder, "True"]
    assert session["process"] is low
    assert "extra_processes" in session
    assert session["extra_processes"] == [ingest]
    assert "watchdog_task" in session
    assert session["watchdog_task"] is watchdog
    assert "closed_captions" in session
    assert session["closed_captions"] is True
    assert "origin_pts" in session
    assert session["origin_pts"] == 1.5
    assert "watchdog_stale_after" in session
    assert session["watchdog_stale_after"] == 45.0
    assert "master_resolution" in session
    assert session["master_resolution"] == encoder[1]
    assert "recovery_bitrate" in session
    assert session["recovery_bitrate"] == pytest.approx(22_000_000)
    directory = pathlib.Path(session["dir"])
    assert (directory / "master.m3u8").read_text() == fs.master_playlist(
        encoder[1],
        audio_bitrate=384_000,
    )
    assert "low.m3u8" in (directory / "low-captions.m3u8").read_text()
    assert "caption_playlist" in result
    assert result["caption_playlist"] == f"/transcode/{sid}/low-captions.m3u8"


@pytest.mark.asyncio
async def test_start_fast_live_cancels_watchdog_for_replaced_session(
    fast_env: tuple[_Sessions, pathlib.Path, _Launched, _Patched],
):
    sessions, _, _, _ = fast_env
    watchdog = MagicMock()

    def spawn(coro: Coroutine[object, object, None]) -> MagicMock:
        assert isinstance(coro, CoroutineType)
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
async def test_start_fast_live_failure_releases_everything(
    fast_env: tuple[_Sessions, pathlib.Path, _Launched, _Patched],
):
    sessions, root, processes, launch = fast_env
    real = launch.side_effect

    async def launch_or_fail(cmd: list[str]) -> FakeProcess:
        if cmd[0] == "enc":
            raise RuntimeError("boom")
        process = await real(cmd)
        assert isinstance(process, FakeProcess)
        return process

    launch.side_effect = launch_or_fail
    with (
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
        pytest.raises(RuntimeError),
    ):
        await fs._start_fast_live("http://live", "alice", "src", True)
    assert all(process.returncode == -15 for _, process in processes)
    assert not sessions
    assert not list(root.iterdir())


@pytest.mark.asyncio
async def test_start_fast_live_aborts_when_session_stopped_during_probe(
    fast_env: tuple[_Sessions, pathlib.Path, _Launched, _Patched],
):
    sessions, root, processes, _ = fast_env

    def stop_during_probe(path: str) -> None:
        del path
        sessions.clear()

    with (
        patch("ffmpeg_session.probe_audio", side_effect=stop_during_probe),
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
        pytest.raises(KeyError),
    ):
        await fs._start_fast_live("http://live", "alice", "src", True)
    ((ingest_cmd, ingest),) = processes
    assert ingest_cmd[0] == "ingest"
    assert ingest.returncode == -15
    assert not sessions
    assert not list(root.iterdir())


def test_startup_rejects_a_replaced_session_with_its_identifier(
    state: _State,
    tmp_path: pathlib.Path,
):
    session = _live(tmp_path)
    state[0]["replaced"] = _live(tmp_path)
    with pytest.raises(KeyError) as error:
        fs._require_current("replaced", session)
    assert error.value.args == ("replaced",)


# =============================================================================
# Process helpers and monitors
# =============================================================================


def test_dead_process_signals_are_noops():
    process = fs._DeadProcess()
    process.terminate()
    process.kill()
    assert process.returncode == -1
    assert asyncio.run(process.wait()) == -1


@pytest.mark.asyncio
async def test_monitor_stderr_collects_lines(caplog: pytest.LogCaptureFixture):
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
async def test_monitor_resume_invalidates_quick_failure(state: _State, tmp_path: pathlib.Path):
    sessions, urls = state
    sessions["s"] = fs.TranscodeSession(
        dir=str(tmp_path),
        process=None,
        started=time.time(),
        url="http://stream.test/live",
    )
    urls["http://m"] = "s"
    await fs._monitor_resume_ffmpeg(_proc(AsyncProcess(exit_code=1)), "s", "http://m")
    assert not sessions
    assert not urls
    assert not tmp_path.exists()


@pytest.mark.asyncio
async def test_monitor_seek_logs_failure(caplog: pytest.LogCaptureFixture):
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
async def test_wait_for_playlist_paths(tmp_path: pathlib.Path):
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


def test_regenerate_playlist_skips_bad_names_and_empty(tmp_path: pathlib.Path):
    (tmp_path / "segabc.ts").write_bytes(b"x" * 2000)
    with patch("ffmpeg_session.get_hls_segment_duration", return_value=3.0):
        fs._regenerate_playlist(tmp_path, 0)
    assert not (tmp_path / "stream.m3u8").exists()


# =============================================================================
# Existing VOD sessions
# =============================================================================


@pytest.fixture
def vod_env(
    state: _State,
    tmp_path: pathlib.Path,
) -> Iterator[tuple[_Sessions, pathlib.Path, AsyncMock]]:
    sessions, _ = state
    (tmp_path / "seg000.ts").write_bytes(b"x" * 2000)
    (tmp_path / "seg001.ts").write_bytes(b"x" * 2000)
    (tmp_path / "stream.m3u8").write_text("#EXTINF:3.0,\nseg000.ts\n#EXTINF:3.0,\nseg001.ts\n")
    sessions["v"] = fs.TranscodeSession(
        dir=str(tmp_path),
        process=FakeProcess(alive=False),
        url="http://movie",
        is_vod=True,
        seek_offset=0,
        duration=100,
        started=time.time(),
    )
    with (
        patch("ffmpeg_session.resolve_hls_master_playlist", side_effect=_same_url),
        patch("ffmpeg_session.probe_media", return_value=(None, [])),
        patch("ffmpeg_session.get_user_agent", return_value="ua"),
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
        patch("ffmpeg_session._wait_for_playlist", new=AsyncMock(return_value=True)) as wait,
    ):
        yield sessions, tmp_path, wait


@pytest.mark.asyncio
async def test_existing_vod_missing_session(state: _State):
    assert await fs._handle_existing_vod_session("missing", "http://m", "software", False) is None


@pytest.mark.asyncio
async def test_existing_vod_active_session_is_reused(
    vod_env: tuple[_Sessions, pathlib.Path, AsyncMock],
):
    sessions, _, wait = vod_env
    sessions["v"]["process"] = FakeProcess()
    result = await fs._handle_existing_vod_session("v", "http://movie", "software", False)
    assert result is not None
    assert "transcoded_duration" in result
    assert result["transcoded_duration"] == 6.0
    wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_vod_with_seek_offset_returns_cache(
    vod_env: tuple[_Sessions, pathlib.Path, AsyncMock],
):
    sessions, _, wait = vod_env
    sessions["v"]["seek_offset"] = 30
    result = await fs._handle_existing_vod_session("v", "http://movie", "software", False)
    assert result is not None
    assert result["seek_offset"] == 30
    wait.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_vod_resume_appends_to_playlist(
    vod_env: tuple[_Sessions, pathlib.Path, AsyncMock],
):
    sessions, directory, _ = vod_env
    process = AsyncProcess()

    async def launch(cmd: list[str]) -> AsyncProcess:
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
async def test_existing_vod_resume_adds_hls_flags_and_polls(
    vod_env: tuple[_Sessions, pathlib.Path, AsyncMock],
):
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
async def test_existing_vod_resume_dies_immediately(
    vod_env: tuple[_Sessions, pathlib.Path, AsyncMock],
):
    with (
        patch("ffmpeg_session.build_hls_ffmpeg_cmd", return_value=["ffmpeg", "-i", "u", "out"]),
        patch(
            "ffmpeg_session._launch_ffmpeg",
            new=AsyncMock(return_value=AsyncProcess(alive=False)),
        ),
    ):
        assert await fs._handle_existing_vod_session("v", "http://movie", "software", False) is None


@pytest.mark.asyncio
async def test_existing_vod_resume_session_vanishes(
    vod_env: tuple[_Sessions, pathlib.Path, AsyncMock],
):
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
def do_env(
    state: _State,
    tmp_path: pathlib.Path,
) -> Iterator[tuple[_Sessions, pathlib.Path, AsyncProcess, _Patched, AsyncMock]]:
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
        patch("ffmpeg_session.resolve_hls_master_playlist", side_effect=_same_url),
        patch("ffmpeg_session.probe_media", return_value=(media, subs)),
        patch("ffmpeg_session.get_user_agent", return_value="ua"),
        patch(
            "ffmpeg_session.build_hls_ffmpeg_cmd",
            side_effect=_vod_command,
        ) as build,
        patch("ffmpeg_session._launch_ffmpeg", new=AsyncMock(return_value=process)) as launch,
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
        patch("ffmpeg_session.get_caption_timestamp_origin"),
    ):
        yield state[0], tmp_path, process, build, launch


@pytest.mark.asyncio
async def test_do_start_vod_with_seek_offset_persists_probe(
    do_env: tuple[_Sessions, pathlib.Path, AsyncProcess, _Patched, AsyncMock],
):
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
    assert "caption_playlist" in result
    assert result["caption_playlist"].endswith("/captions.m3u8")
    assert build.call_args.args[6:8] == ("4k", "high")


@pytest.mark.asyncio
async def test_do_start_bandwidth_saver_lowers_quality(
    do_env: tuple[_Sessions, pathlib.Path, AsyncProcess, _Patched, AsyncMock],
):
    _, _, _, build, _ = do_env
    with patch("ffmpeg_session._wait_for_playlist", new=AsyncMock(return_value=True)):
        await fs._do_start_transcode("http://live", "live", None, None, 0, bandwidth_saver=True)
    assert build.call_args.args[6:8] == ("720p", "low")
    assert build.call_args.kwargs["allow_upscale"] is False


@pytest.mark.asyncio
async def test_do_start_caption_failure_removes_output(
    do_env: tuple[_Sessions, pathlib.Path, AsyncProcess, _Patched, AsyncMock],
):
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
async def test_do_start_reports_failed_startup(
    do_env: tuple[_Sessions, pathlib.Path, AsyncProcess, _Patched, AsyncMock],
    caplog: pytest.LogCaptureFixture,
):
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


def test_playlist_marker_and_segments(tmp_path: pathlib.Path):
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
async def test_wait_for_fresh_fast_playlist(tmp_path: pathlib.Path):
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
async def test_terminate_process_simple_paths():
    dead = FakeProcess(alive=False)
    await fs._terminate_process(dead)
    assert dead.returncode == 0
    vanished = FakeProcess(killed=True)
    await fs._terminate_process(vanished)
    assert vanished.returncode is None
    graceful = AsyncProcess()
    await fs._terminate_process(graceful)
    assert (graceful.returncode, graceful.waited) == (-15, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError, asyncio.CancelledError])
async def test_terminate_process_escalates(error: type[BaseException]):
    process = _IgnoresTerminate()

    async def wait_for(coro: Coroutine[object, object, object], timeout: float) -> None:
        del timeout
        coro.close()
        raise error

    with patch.object(fs.asyncio, "wait_for", wait_for):
        if error is asyncio.CancelledError:
            with pytest.raises(asyncio.CancelledError):
                await fs._terminate_process(process)
        else:
            await fs._terminate_process(process)
    assert process.returncode == -9
    assert process.waited == 1


@pytest.mark.asyncio
async def test_launch_ffmpeg_terminates_a_process_launched_after_cancellation():
    gate = asyncio.Event()
    late = AsyncProcess()

    async def slow(*cmd: str, **kwargs: object) -> AsyncProcess:
        del cmd, kwargs
        await gate.wait()
        return late

    with patch("ffmpeg_session.asyncio.create_subprocess_exec", side_effect=slow):
        task = asyncio.create_task(fs._launch_ffmpeg(["ffmpeg"]))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert late.returncode == -15


@pytest.mark.asyncio
async def test_monitor_fast_process(
    state: _State,
    caplog: pytest.LogCaptureFixture,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    process = AsyncProcess(lines=[b"open http://user:secret@host/x failed\n"], exit_code=1)
    sessions["f"] = fs.TranscodeSession(
        process=process,
        dir=str(tmp_path),
        started=time.time(),
        url="http://stream.test/live",
    )
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


def test_log_watchdog_delay_rate_limited(caplog: pytest.LogCaptureFixture, tmp_path: pathlib.Path):
    session = _live(tmp_path)
    with caplog.at_level(logging.WARNING, logger="ffmpeg_session"):
        fs._log_watchdog_delay("f", session, "pipeline", 100)
        fs._log_watchdog_delay("f", session, "pipeline", 110)
    assert len(caplog.records) == 1
    assert "pipeline_restart_log_at" in session
    assert session["pipeline_restart_log_at"] == 100


def _fast_session(
    tmp_path: pathlib.Path,
    process: fs.ManagedProcess | None = None,
) -> fs.FastLiveSession:
    return fs.FastLiveSession(
        dir=str(tmp_path),
        process=process,
        started=time.time(),
        url="http://live",
        live_recovery=LiveRecoveryPolicy(),
        ingest_restart_command=["ingest", "out"],
        low_restart_command=["enc", "low", "out"],
        playlist_generation=0,
    )


@pytest.fixture
def recovery_env(state: _State) -> Iterator[_Sessions]:
    with (
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
        patch("ffmpeg_session._restart_session_clock", new=AsyncMock()),
    ):
        yield state[0]


@pytest.mark.asyncio
async def test_restart_clock_dates_old_segments_then_follows_new_ingest(
    state: _State,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    (tmp_path / "low_1.ts").write_bytes(video_packet(5002) * 10)
    (tmp_path / "input_000001.ts").write_bytes(video_packet(1.5) * 10)
    session = _fast_session(tmp_path)
    session.update(origin_pts=5000.0, origin_time=100.0)
    sessions["f"] = session
    await fs._restart_session_clock("f", session, tmp_path / "missing.m3u8")
    assert "segment_dates" not in session
    (tmp_path / "low.m3u8").write_text("#EXTINF:2,\nlow_1.ts\n")
    session.update(origin_pts=5000.0, origin_time=100.0)
    await fs._restart_session_clock("f", session, tmp_path / "low.m3u8")
    assert session.get("segment_dates") == {"low_1.ts": 102.0}
    assert session.get("origin_pts") == 1.5
    assert session.get("origin_time") == pytest.approx(time.time(), abs=5)
    sessions.clear()
    await fs._restart_session_clock("f", session, tmp_path / "low.m3u8")
    assert session.get("origin_pts") == 1.5


def test_dated_fast_playlist_needs_a_known_clock(state: _State, tmp_path: pathlib.Path):
    sessions, _ = state
    assert fs.dated_fast_playlist("missing", "#EXTM3U\n") == "#EXTM3U\n"
    sessions["f"] = _fast_session(tmp_path)
    sessions["f"]["origin_pts"] = None
    assert fs.dated_fast_playlist("f", "#EXTM3U\n") == "#EXTM3U\n"
    sessions["f"]["origin_pts"] = 1.0
    with pytest.raises(KeyError, match="origin_time"):
        fs.dated_fast_playlist("f", "#EXTM3U\n")


def _pipeline_dir(tmp_path: pathlib.Path) -> None:
    (tmp_path / "input.m3u8").write_text("#EXTINF:2,\ninput_1.ts\n")
    (tmp_path / "input_1.ts").write_bytes(b"x")
    (tmp_path / "input_2.ts.tmp").write_bytes(b"x")
    (tmp_path / "low.m3u8").write_text("#EXTINF:2,\nlow_1.ts\n")


@pytest.mark.asyncio
async def test_restart_pipeline_skips_replaced_session(
    recovery_env: _Sessions,
    tmp_path: pathlib.Path,
):
    session = _fast_session(tmp_path)
    await fs._restart_fast_pipeline("f", session, "input")
    assert "watchdog_recovering" not in session


@pytest.mark.asyncio
async def test_restart_pipeline_success(recovery_env: _Sessions, tmp_path: pathlib.Path):
    sessions = recovery_env
    _pipeline_dir(tmp_path)
    low_old, ingest_old = FakeProcess(), FakeProcess()
    session = _fast_session(tmp_path, process=low_old)
    session.update({"ingest_process": ingest_old, "extra_processes": [ingest_old]})
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
    assert all(p.returncode == -15 for p in (low_old, ingest_old))
    assert "ingest_process" in session
    assert (session["process"], session["ingest_process"]) == (low, ingest)
    assert "extra_processes" in session
    assert session["extra_processes"] == [ingest]
    assert "playlist_generation" in session
    assert session["playlist_generation"] == 1
    assert "watchdog_recovering" not in session
    assert not (tmp_path / "input.m3u8").exists()
    assert not list(tmp_path.glob("input_*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["ingest", "low"])
async def test_restart_pipeline_timeout_discards_replacements(
    recovery_env: _Sessions,
    tmp_path: pathlib.Path,
    failing: str,
    caplog: pytest.LogCaptureFixture,
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
    assert "ingest_process" in session
    assert session["ingest_process"] is None
    assert "extra_processes" in session
    assert session["extra_processes"] == []
    assert ingest.returncode == -15
    assert low.returncode == (-15 if failing == "low" else None)
    assert "playlist_generation" in session
    assert session["playlist_generation"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["ingest", "low", "ready"])
async def test_restart_pipeline_session_replaced(
    recovery_env: _Sessions,
    tmp_path: pathlib.Path,
    stage: str,
):
    sessions = recovery_env
    _pipeline_dir(tmp_path)
    session = _fast_session(tmp_path, process=FakeProcess())
    sessions["f"] = session
    ingest, low = AsyncProcess(), AsyncProcess()
    launches = iter([ingest, low])

    async def launch(cmd: list[str]) -> AsyncProcess:
        process = next(launches)
        if (stage, process) in (("ingest", ingest), ("low", low)):
            sessions.clear()
        return process

    waits = iter([True, True])

    async def wait(*args: object, **kwargs: object) -> bool:
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
    assert "playlist_generation" in session
    assert session["playlist_generation"] == 0
    assert "watchdog_recovering" in session
    assert session["watchdog_recovering"] == "input"


# =============================================================================
# Watchdog
# =============================================================================


def _recovery(
    stalled: Collection[str] = (),
    permit: bool = False,
    pending: Collection[str] = (),
    exhausted: bool = False,
) -> MagicMock:
    recovery = MagicMock()

    def is_stalled(stage: str, alive: bool, ready: bool, now: float) -> tuple[MagicMock, bool]:
        del alive, ready, now
        return recovery, stage in stalled

    recovery.stalled.side_effect = is_stalled
    recovery.permit_restart.return_value = recovery, permit
    recovery.exhausted.return_value = exhausted
    recovery.pending = set(pending)
    return recovery


@pytest.fixture
def watchdog_env(
    state: _State,
    tmp_path: pathlib.Path,
) -> Iterator[tuple[_Sessions, pathlib.Path, AsyncMock, _Patched]]:
    with (
        patch("ffmpeg_session.ready_bitrate", return_value=1),
        patch("ffmpeg_session._restart_fast_pipeline", new=AsyncMock()) as restart,
        patch("ffmpeg_session._log_watchdog_delay") as delay,
    ):
        yield state[0], tmp_path, restart, delay


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled,stage", [({"input"}, "input"), ({"low"}, "low output")])
async def test_check_restarts_stalled_pipeline(
    watchdog_env: tuple[_Sessions, pathlib.Path, AsyncMock, _Patched],
    stalled: set[str],
    stage: str,
):
    _, tmp_path, restart, delay = watchdog_env
    recovery = _recovery(stalled=stalled, permit=True)
    session = fs.FastLiveSession(
        dir=str(tmp_path),
        live_recovery=recovery,
        ingest_process=FakeProcess(),
        process=None,
        started=time.time(),
        url="http://stream.test/live",
        ingest_restart_command=["ffmpeg", "ingest"],
    )
    await fs._check_fast_live_session("f", session)
    restart.assert_awaited_once_with("f", session, stage)
    delay.assert_not_called()
    assert recovery.stalled.call_args_list[0].args[1:3] == (True, True)


@pytest.mark.asyncio
async def test_check_delays_pipeline_restart_on_cooldown(
    watchdog_env: tuple[_Sessions, pathlib.Path, AsyncMock, _Patched],
):
    _, tmp_path, restart, delay = watchdog_env
    session = fs.FastLiveSession(
        dir=str(tmp_path),
        live_recovery=_recovery(stalled={"low"}),
        process=None,
        started=time.time(),
        url="http://stream.test/live",
        ingest_restart_command=["ffmpeg", "ingest"],
    )
    await fs._check_fast_live_session("f", session)
    restart.assert_not_awaited()
    delay.assert_called_once()
    assert delay.call_args.args[2] == "pipeline"
    delay.reset_mock()
    session["live_recovery"] = _recovery(stalled={"low"}, pending={"pipeline"})
    await fs._check_fast_live_session("f", session)
    delay.assert_not_called()


@pytest.mark.asyncio
async def test_check_healthy_session_does_nothing(
    watchdog_env: tuple[_Sessions, pathlib.Path, AsyncMock, _Patched],
):
    _, tmp_path, restart, delay = watchdog_env
    await fs._check_fast_live_session(
        "f",
        fs.FastLiveSession(
            dir=str(tmp_path),
            live_recovery=_recovery(),
            process=None,
            started=time.time(),
            url="http://stream.test/live",
            ingest_restart_command=["ffmpeg", "ingest"],
        ),
    )
    restart.assert_not_awaited()
    delay.assert_not_called()


@pytest.mark.asyncio
async def test_watchdog_logs_errors_and_propagates_cancel(
    state: _State,
    caplog: pytest.LogCaptureFixture,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    sessions["f"] = fs.FastLiveSession(
        last_access=time.time(),
        dir=str(tmp_path),
        process=None,
        started=time.time(),
        url="http://stream.test/live",
        live_recovery=LiveRecoveryPolicy(),
        ingest_restart_command=["ingest", "out"],
    )
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
async def test_watchdog_skips_idle_session_and_exits_when_removed(
    state: _State,
    tmp_path: pathlib.Path,
):
    sessions, _ = state
    sessions["f"] = fs.FastLiveSession(
        last_access=0,
        dir=str(tmp_path),
        process=None,
        started=time.time(),
        url="http://stream.test/live",
        live_recovery=LiveRecoveryPolicy(),
        ingest_restart_command=["ingest", "out"],
    )
    check = AsyncMock()
    with patch("ffmpeg_session._check_fast_live_session", new=check):
        task = asyncio.create_task(fs._watch_fast_live("f", watchdog_interval_sec=0))
        for _ in range(3):
            await asyncio.sleep(0)
        sessions.clear()
        await asyncio.wait_for(task, timeout=1)
    check.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_fast_live_session_requires_live_recovery(tmp_path: pathlib.Path):
    session = fs.TranscodeSession(
        dir=str(tmp_path),
        process=None,
        started=time.time(),
        url="http://x",
    )
    with pytest.raises(KeyError, match="live_recovery"):
        await fs._check_fast_live_session("s", session)


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
