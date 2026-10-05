"""Live recovery must detect stale media and isolate high-quality failures."""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncio
import pathlib

import pytest

from fast_start_test import playlist, video_packet
from ffmpeg_session_test import FakeProcess
from playback_policy import LiveRecoveryPolicy, PlaybackPolicy, UpgradePolicy

import ffmpeg_session


class RecoveryProcess(FakeProcess):
    def __init__(self):
        super().__init__()
        self.wait = AsyncMock(return_value=0)
        self.stderr = MagicMock(readline=AsyncMock(return_value=b""))


def process():
    return RecoveryProcess()


@pytest.fixture
def recovery_session(tmp_path):
    for name in ("input", "low", "high"):
        playlist(tmp_path, name)
    ingest, low, high = process(), process(), process()
    session = {
        "dir": str(tmp_path),
        "process": low,
        "ingest_process": ingest,
        "high_process": high,
        "extra_processes": [ingest, high],
        "username": "viewer",
        "url": "https://upstream.example/live.m3u8",
        "started": 0,
        "last_access": 0,
        "fast_start": True,
        "high_selected": True,
        "playback_policy": PlaybackPolicy(),
        "upgrade_policy": UpgradePolicy(samples=[(90, 100_000_000)]),
        "live_recovery": LiveRecoveryPolicy(high_started=0),
        "ingest_restart_command": ["ffmpeg", "ingest"],
        "low_restart_command": ["ffmpeg", "low"],
        "high_restart_command": ["ffmpeg", "high"],
        "master_resolution": "1080p",
        "master_audio_bitrate": 0,
    }
    with patch.dict(ffmpeg_session._transcode_sessions, {"recover": session}, clear=True):
        yield session


async def check(session, now, rates):
    with (
        patch("ffmpeg_session.time.monotonic", return_value=now),
        patch(
            "ffmpeg_session.ready_bitrate",
            side_effect=lambda _directory, name, **_kwargs: rates[name],
        ),
    ):
        await ffmpeg_session._check_fast_live_session("recover", session)


@pytest.mark.asyncio
async def test_input_or_low_stall_restarts_pipeline_once(recovery_session):
    rates = {"input.m3u8": 0, "low.m3u8": 4_000_000, "high.m3u8": 8_000_000}
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
async def test_isolated_high_stall_schedules_only_high_restart(recovery_session):
    rates = {"input.m3u8": 4_000_000, "low.m3u8": 4_000_000, "high.m3u8": 0}
    with patch(
        "ffmpeg_session._restart_high_encoder",
        new=AsyncMock(),
    ) as restart:
        await check(recovery_session, 100, rates)
        restart.assert_not_awaited()
        await check(recovery_session, 102, rates)
        task = recovery_session["high_recovery_task"]
        await task
        restart.assert_awaited_once_with("recover", recovery_session)
        assert recovery_session["process"].returncode is None
        assert recovery_session["ingest_process"].returncode is None


@pytest.mark.asyncio
async def test_short_publication_gap_does_not_restart(recovery_session):
    stale = {"input.m3u8": 4_000_000, "low.m3u8": 4_000_000, "high.m3u8": 0}
    healthy = {**stale, "high.m3u8": 8_000_000}
    with (
        patch("ffmpeg_session._restart_high_encoder", new=AsyncMock()) as high_restart,
        patch("ffmpeg_session._restart_fast_pipeline", new=AsyncMock()) as pipeline_restart,
    ):
        await check(recovery_session, 100, stale)
        await check(recovery_session, 102, healthy)
        high_restart.assert_not_awaited()
        pipeline_restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_playlist_waits_for_two_new_segments(tmp_path):
    playlist(tmp_path, "high")
    path = tmp_path / "high.m3u8"
    marker = ffmpeg_session._playlist_marker(path)
    previous_segments = ffmpeg_session._playlist_segments(path)
    sleep_calls = 0

    async def publish_segment(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        number = sleep_calls + 2
        filename = f"high_{number}.ts"
        (tmp_path / filename).write_bytes(video_packet(20 + number * 2) * 10)
        with path.open("a") as playlist_file:
            playlist_file.write(f"#EXTINF:2,\n{filename}\n")

    with patch("ffmpeg_session.asyncio.sleep", side_effect=publish_segment):
        ready = await ffmpeg_session._wait_for_fresh_fast_playlist(
            str(tmp_path),
            "high.m3u8",
            process(),
            marker,
            previous_segments=previous_segments,
        )

    assert ready
    assert sleep_calls == 2


def test_restart_without_playlist_history_uses_epoch_sequence():
    command = ["ffmpeg", "-hls_flags", "append_list", "/tmp/high.m3u8"]
    prepared = ffmpeg_session._prepare_encoder_restart(command, set())
    assert prepared[prepared.index("-hls_start_number_source") + 1] == "epoch_us"
    assert ffmpeg_session._prepare_encoder_restart(command, {"high_1.ts"}) == command


@pytest.mark.asyncio
async def test_cancelled_launch_terminates_late_replacement():
    started = asyncio.Event()
    release = asyncio.Event()
    replacement = process()

    async def launch(_command):
        started.set()
        await release.wait()
        return replacement

    with patch("ffmpeg_session._launch_ffmpeg", side_effect=launch):
        task = asyncio.create_task(
            ffmpeg_session._launch_recovery_ffmpeg(["ffmpeg", "replacement"])
        )
        await started.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert replacement.returncode is not None


@pytest.mark.asyncio
async def test_high_restart_preserves_ingest_and_low(recovery_session):
    old_ingest = recovery_session["ingest_process"]
    old_low = recovery_session["process"]
    old_high = recovery_session["high_process"]
    replacement = process()
    recovery_session["live_recovery"].permit_restart("high", 100)

    with (
        patch(
            "ffmpeg_session._launch_recovery_ffmpeg",
            new=AsyncMock(return_value=replacement),
        ) as launch,
        patch(
            "ffmpeg_session._wait_for_fresh_fast_playlist",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "ffmpeg_session._terminate_fast_process",
            new=AsyncMock(),
        ) as terminate,
        patch(
            "ffmpeg_session._spawn_background_task",
            side_effect=lambda coroutine: coroutine.close(),
        ),
    ):
        await ffmpeg_session._restart_high_encoder("recover", recovery_session)

    launch.assert_awaited_once_with(["ffmpeg", "high"])
    terminate.assert_awaited_once_with(old_high)
    assert recovery_session["ingest_process"] is old_ingest
    assert recovery_session["process"] is old_low
    assert recovery_session["high_process"] is replacement
    assert recovery_session["extra_processes"] == [old_ingest, replacement]
    assert recovery_session["upgrade_policy"].samples == []
    master = pathlib.Path(recovery_session["dir"]) / "master.m3u8"
    assert "high.m3u8" in master.read_text()


@pytest.mark.asyncio
async def test_pipeline_restart_replaces_dependencies_without_second_upstream_reader(
    recovery_session,
):
    old_processes = {
        recovery_session["ingest_process"],
        recovery_session["process"],
        recovery_session["high_process"],
    }
    new_ingest, new_low = process(), process()
    recovery_session["live_recovery"].permit_restart("pipeline", 100)

    with (
        patch(
            "ffmpeg_session._launch_recovery_ffmpeg",
            new=AsyncMock(side_effect=[new_ingest, new_low]),
        ) as launch,
        patch(
            "ffmpeg_session._wait_for_fresh_fast_playlist",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "ffmpeg_session._terminate_fast_process",
            new=AsyncMock(),
        ) as terminate,
        patch(
            "ffmpeg_session._spawn_background_task",
            side_effect=lambda coroutine: coroutine.close(),
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
    assert recovery_session["ingest_process"] is new_ingest
    assert recovery_session["process"] is new_low
    assert recovery_session["high_process"] is None
    assert recovery_session["extra_processes"] == [new_ingest]
    assert recovery_session["upgrade_policy"].samples == []


def test_stopping_session_cancels_recovery_tasks(recovery_session):
    watchdog = MagicMock()
    watchdog.done.return_value = False
    high_recovery = MagicMock()
    high_recovery.done.return_value = False
    recovery_session["watchdog_task"] = watchdog
    recovery_session["high_recovery_task"] = high_recovery

    ffmpeg_session.stop_session("recover", force=True)

    watchdog.cancel.assert_called_once_with()
    high_recovery.cancel.assert_called_once_with()
