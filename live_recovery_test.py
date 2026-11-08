"""Live recovery must detect stale media and rebuild the pipeline once."""

from collections.abc import Coroutine, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio
import os
import pathlib

import pytest

from fast_start_test import playlist, video_packet
from ffmpeg_session_test import FakeProcess
from playback_policy import LiveRecoveryPolicy, PlaybackHealth, PlaybackPolicy

import ffmpeg_session


class RecoveryProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__()
        self.wait = AsyncMock(return_value=0)
        self.stderr = MagicMock(readline=AsyncMock(return_value=b""))


def process() -> RecoveryProcess:
    return RecoveryProcess()


def _close(coroutine: Coroutine[object, object, None]) -> None:
    coroutine.close()


@pytest.fixture
def recovery_session(tmp_path: pathlib.Path) -> Iterator[ffmpeg_session.FastLiveSession]:
    for name in ("input", "low"):
        playlist(tmp_path, name)
    ingest, low = process(), process()
    session: ffmpeg_session.FastLiveSession = {
        "dir": str(tmp_path),
        "process": low,
        "ingest_process": ingest,
        "extra_processes": [ingest],
        "username": "viewer",
        "url": "https://upstream.example/live.m3u8",
        "started": 0,
        "last_access": 0,
        "fast_start": True,
        "closed_captions": True,
        "playback_policy": PlaybackPolicy(),
        "live_recovery": LiveRecoveryPolicy(),
        "ingest_restart_command": ["ffmpeg", "ingest"],
        "low_restart_command": ["ffmpeg", "low"],
        "master_resolution": "1080p",
        "master_audio_bitrate": 0,
        "playlist_generation": 0,
        "watchdog_stale_after": 45,
    }
    with patch.dict(ffmpeg_session._transcode_sessions, {"recover": session}, clear=True):
        yield session


async def check(
    session: ffmpeg_session.FastLiveSession,
    now: float,
    rates: dict[str, int],
) -> None:
    def ready_bitrate(directory: str, name: str, **kwargs: object) -> int:
        del directory, kwargs
        return rates[name]

    with (
        patch("ffmpeg_session.time.monotonic", return_value=now),
        patch("ffmpeg_session.ready_bitrate", side_effect=ready_bitrate),
    ):
        await ffmpeg_session._check_fast_live_session("recover", session)


@pytest.mark.asyncio
async def test_input_or_low_stall_restarts_pipeline_once(
    recovery_session: ffmpeg_session.FastLiveSession,
):
    rates = {"input.m3u8": 0, "low.m3u8": 4_000_000}
    with patch(
        "ffmpeg_session._restart_fast_pipeline",
        new=AsyncMock(),
    ) as restart:
        await check(recovery_session, 100, rates)
        restart.assert_not_awaited()
        await check(recovery_session, 102, rates)
        restart.assert_awaited_once_with("recover", recovery_session, "input")
        await check(recovery_session, 104, rates)
        restart.assert_awaited_once()


@pytest.mark.asyncio
async def test_short_publication_gap_does_not_restart(
    recovery_session: ffmpeg_session.FastLiveSession,
):
    stale = {"input.m3u8": 4_000_000, "low.m3u8": 0}
    healthy = {**stale, "low.m3u8": 4_000_000}
    with patch("ffmpeg_session._restart_fast_pipeline", new=AsyncMock()) as pipeline_restart:
        await check(recovery_session, 100, stale)
        await check(recovery_session, 102, healthy)
        pipeline_restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_running_pipeline_tolerates_thirty_second_publication_gap(
    recovery_session: ffmpeg_session.FastLiveSession,
):
    for name in ("input", "low"):
        os.utime(pathlib.Path(recovery_session["dir"]) / f"{name}.m3u8", (70, 70))

    with (
        patch("fast_start.time.time", return_value=100),
        patch("ffmpeg_session.time.monotonic", return_value=100),
        patch("ffmpeg_session._restart_fast_pipeline", new=AsyncMock()) as pipeline_restart,
    ):
        await ffmpeg_session._check_fast_live_session("recover", recovery_session)

    pipeline_restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_playlist_waits_for_two_new_segments(tmp_path: pathlib.Path):
    playlist(tmp_path, "low")
    path = tmp_path / "low.m3u8"
    marker = ffmpeg_session._playlist_marker(path)
    previous_segments = ffmpeg_session._playlist_segments(path)
    sleep_calls = 0

    async def publish_segment(delay: float) -> None:
        del delay
        nonlocal sleep_calls
        sleep_calls += 1
        number = sleep_calls + 2
        filename = f"low_{number}.ts"
        (tmp_path / filename).write_bytes(video_packet(20 + number * 2) * 10)
        with path.open("a") as playlist_file:
            playlist_file.write(f"#EXTINF:2,\n{filename}\n")

    with patch("ffmpeg_session.asyncio.sleep", side_effect=publish_segment):
        ready = await ffmpeg_session._wait_for_fresh_fast_playlist(
            str(tmp_path),
            "low.m3u8",
            process(),
            marker,
            previous_segments=previous_segments,
        )

    assert ready
    assert sleep_calls == 2


def test_restart_without_playlist_history_uses_epoch_sequence():
    command = ["ffmpeg", "-hls_flags", "append_list", "/tmp/low.m3u8"]
    prepared = ffmpeg_session._prepare_encoder_restart(command, set())
    assert prepared[prepared.index("-hls_start_number_source") + 1] == "epoch_us"
    assert ffmpeg_session._prepare_encoder_restart(command, {"low_1.ts"}) == command


@pytest.mark.asyncio
async def test_cancelled_launch_terminates_late_replacement():
    started = asyncio.Event()
    release = asyncio.Event()
    replacement = process()

    async def launch(*command: str, **kwargs: object) -> RecoveryProcess:
        del command, kwargs
        started.set()
        await release.wait()
        return replacement

    with patch("ffmpeg_session.asyncio.create_subprocess_exec", side_effect=launch):
        task = asyncio.create_task(
            ffmpeg_session._launch_ffmpeg(["ffmpeg", "replacement"]),
        )
        await started.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert replacement.returncode is not None


@pytest.mark.asyncio
async def test_pipeline_restart_replaces_dependencies_without_second_upstream_reader(
    recovery_session: ffmpeg_session.FastLiveSession,
):
    assert "ingest_process" in recovery_session
    old_processes = {
        recovery_session["ingest_process"],
        recovery_session["process"],
    }
    new_ingest, new_low = process(), process()
    recovery_session["live_recovery"].permit_restart("pipeline", 100)

    with (
        patch(
            "ffmpeg_session._launch_ffmpeg",
            new=AsyncMock(side_effect=[new_ingest, new_low]),
        ) as launch,
        patch(
            "ffmpeg_session._wait_for_fresh_fast_playlist",
            new=AsyncMock(return_value=True),
        ),
        patch("ffmpeg_session._restart_session_clock", new=AsyncMock()),
        patch(
            "ffmpeg_session._terminate_process",
            new=AsyncMock(),
        ) as terminate,
        patch(
            "ffmpeg_session._spawn_background_task",
            side_effect=_close,
        ),
    ):
        await ffmpeg_session._restart_fast_pipeline(
            "recover",
            recovery_session,
            "input",
        )

    assert [call.args[0] for call in launch.await_args_list] == [
        ["ffmpeg", "ingest"],
        ["ffmpeg", "low"],
    ]
    assert {call.args[0] for call in terminate.await_args_list} == old_processes
    assert "ingest_process" in recovery_session
    assert recovery_session["ingest_process"] is new_ingest
    assert recovery_session["process"] is new_low
    assert "extra_processes" in recovery_session
    assert recovery_session["extra_processes"] == [new_ingest]
    assert "playlist_generation" in recovery_session
    assert recovery_session["playlist_generation"] == 1


@pytest.mark.asyncio
async def test_repeated_pipeline_recovery_advances_generation(
    recovery_session: ffmpeg_session.FastLiveSession,
):
    recovery_session["playlist_generation"] = 7
    recovery_session["live_recovery"].permit_restart("pipeline", 100)
    new_ingest, new_low = process(), process()

    with (
        patch(
            "ffmpeg_session._launch_ffmpeg",
            new=AsyncMock(side_effect=[new_ingest, new_low]),
        ),
        patch(
            "ffmpeg_session._wait_for_fresh_fast_playlist",
            new=AsyncMock(return_value=True),
        ),
        patch("ffmpeg_session._restart_session_clock", new=AsyncMock()),
        patch(
            "ffmpeg_session._terminate_process",
            new=AsyncMock(),
        ),
        patch(
            "ffmpeg_session._spawn_background_task",
            side_effect=_close,
        ),
    ):
        await ffmpeg_session._restart_fast_pipeline(
            "recover",
            recovery_session,
            "low",
        )

    assert recovery_session["playlist_generation"] == 8


@pytest.mark.asyncio
async def test_failed_pipeline_recovery_keeps_generation(
    recovery_session: ffmpeg_session.FastLiveSession,
):
    recovery_session["playlist_generation"] = 5
    recovery_session["live_recovery"].permit_restart("pipeline", 100)
    new_ingest, new_low = process(), process()

    with (
        patch(
            "ffmpeg_session._launch_ffmpeg",
            new=AsyncMock(side_effect=[new_ingest, new_low]),
        ),
        patch(
            "ffmpeg_session._wait_for_fresh_fast_playlist",
            new=AsyncMock(side_effect=[True, False]),
        ),
        patch("ffmpeg_session._restart_session_clock", new=AsyncMock()),
        patch(
            "ffmpeg_session._terminate_process",
            new=AsyncMock(),
        ),
        patch(
            "ffmpeg_session._spawn_background_task",
            side_effect=_close,
        ),
    ):
        await ffmpeg_session._restart_fast_pipeline(
            "recover",
            recovery_session,
            "input",
        )

    assert recovery_session["playlist_generation"] == 5


def test_health_reports_current_recovery_generation(
    recovery_session: ffmpeg_session.FastLiveSession,
):
    recovery_session["playlist_generation"] = 3
    with patch("ffmpeg_session.ready_bitrate", return_value=4_000_000):
        result = ffmpeg_session.report_playback_health(
            "recover",
            "viewer",
            PlaybackHealth(
                buffer_seconds=10,
                waiting=False,
                observed_bitrate=10_000_000,
                required_bitrate=4_000_000,
            ),
        )

    assert "playlist" in result
    assert result["playlist"] == "/transcode/recover/low.m3u8?generation=3"
    assert "caption_playlist" in result
    assert result["caption_playlist"] == "/transcode/recover/low-captions.m3u8?generation=3"


def test_reused_session_response_includes_current_generation(
    recovery_session: ffmpeg_session.FastLiveSession,
):
    assert (
        ffmpeg_session._adaptive_session_response("recover")["playlist"]
        == "/transcode/recover/low.m3u8"
    )
    recovery_session["playlist_generation"] = 4
    response = ffmpeg_session._adaptive_session_response("recover")

    assert response["playlist"] == "/transcode/recover/low.m3u8?generation=4"
    assert "caption_playlist" in response
    assert response["caption_playlist"] == "/transcode/recover/low-captions.m3u8?generation=4"
    assert "master_playlist" in response
    assert response["master_playlist"] == "/transcode/recover/master.m3u8"


def test_stopping_session_cancels_watchdog(recovery_session: ffmpeg_session.FastLiveSession):
    watchdog = MagicMock()
    watchdog.done.return_value = False
    recovery_session["watchdog_task"] = watchdog

    asyncio.run(ffmpeg_session.stop_session("recover", force=True))

    watchdog.cancel.assert_called_once_with()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
