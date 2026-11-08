"""Fast-start sessions start once per URL and never outlive their pipeline."""

from collections.abc import Coroutine, Iterator
from unittest.mock import patch

import asyncio
import pathlib
import time

from fastapi import HTTPException

import pytest

from fast_start_test import video_packet
from playback_policy import LiveRecoveryPolicy, PlaybackPolicy

import ffmpeg_session


URL = "https://upstream.example/live/1.m3u8"


class LoopProcess:
    """Fake subprocess whose exit, like asyncio's, is delivered by the event loop."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.signals: list[str] = []
        self._exited = asyncio.Event()

    def terminate(self) -> None:
        self.signals.append("TERM")
        asyncio.get_running_loop().call_soon(self._exit, -15)

    def kill(self) -> None:
        self.signals.append("KILL")
        self._exit(-9)

    async def wait(self) -> int:
        await self._exited.wait()
        assert isinstance(self.returncode, int)
        return self.returncode

    def _exit(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
            self._exited.set()


class Startup:
    """Gate fast-start readiness so tests can interleave concurrent requests."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.failures = 0
        self.launches: list[tuple[list[str], LoopProcess]] = []

    async def launch(self, cmd: list[str]) -> LoopProcess:
        process = LoopProcess()
        self.launches.append((cmd, process))
        return process

    async def wait_ready(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        await self.gate.wait()
        if self.failures:
            self.failures -= 1
            raise HTTPException(500, "Fast-start stream failed to become ready")

    @property
    def ingests(self) -> int:
        return sum(cmd[0] == "ingest" for cmd, _ in self.launches)


def _ingest_command(
    url: str,
    directory: str,
    user_agent: str | None,
    **kwargs: object,
) -> list[str]:
    del user_agent, kwargs
    path = pathlib.Path(directory)
    (path / "input_000.ts").write_bytes(video_packet(10) * 10)
    (path / "input.m3u8").write_text("#EXTINF:2.0,\ninput_000.ts\n")
    return ["ingest", url]


def _close(coro: Coroutine[object, object, None]) -> None:
    coro.close()


@pytest.fixture
def startup(tmp_path: pathlib.Path) -> Iterator[Startup]:
    runtime = Startup()
    with (
        patch.dict(ffmpeg_session._transcode_sessions, {}, clear=True),
        patch.dict(ffmpeg_session._url_to_session, {}, clear=True),
        patch("ffmpeg_session.get_settings", return_value={"max_resolution": "1080p"}),
        patch("ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch("ffmpeg_session.get_user_agent", return_value=None),
        patch("ffmpeg_session.ingest_command", side_effect=_ingest_command),
        patch("ffmpeg_session.encoder_command", return_value=["encoder"]),
        patch("ffmpeg_session._launch_ffmpeg", side_effect=runtime.launch),
        patch.object(ffmpeg_session._FastLiveStartup, "wait_ready", runtime.wait_ready),
        patch("ffmpeg_session.probe_audio", return_value=None),
        patch("ffmpeg_session.probe_closed_captions", return_value=False),
        patch("ffmpeg_session.probe_video_codec", return_value="h264"),
        patch("ffmpeg_session._spawn_background_task", side_effect=_close),
    ):
        yield runtime


def _start(user_max_streams: int = 0) -> asyncio.Task[ffmpeg_session.TranscodeResponse]:
    return asyncio.create_task(
        ffmpeg_session.start_transcode(
            URL,
            fast_start=True,
            username="viewer",
            user_max_streams=user_max_streams,
        ),
    )


@pytest.mark.asyncio
async def test_second_request_waits_for_startup_then_reuses_it(startup: Startup):
    first = _start()
    await asyncio.sleep(0.01)
    second = _start()
    await asyncio.sleep(0.01)
    assert not second.done(), "An unfinished startup must not be handed to another request"
    startup.gate.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert second_result["session_id"] == first_result["session_id"]
    assert startup.ingests == 1


@pytest.mark.asyncio
async def test_failed_startup_is_not_handed_to_a_waiting_request(startup: Startup):
    startup.failures = 1
    first = _start()
    await asyncio.sleep(0.01)
    second = _start()
    await asyncio.sleep(0.01)
    startup.gate.set()
    first_result, second_result = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(first_result, HTTPException)
    assert isinstance(second_result, dict)
    assert second_result["session_id"] in ffmpeg_session._transcode_sessions
    assert startup.ingests == 2


@pytest.mark.asyncio
async def test_concurrent_starts_open_one_upstream_within_stream_limits(startup: Startup):
    requests = [_start(user_max_streams=1), _start(user_max_streams=1)]
    await asyncio.sleep(0.01)
    startup.gate.set()
    results = await asyncio.gather(*requests)
    assert startup.ingests == 1
    assert {result["session_id"] for result in results} == set(ffmpeg_session._transcode_sessions)
    assert not ffmpeg_session._startup_guards


def _dead_fast_session(tmp_path: pathlib.Path) -> ffmpeg_session.FastLiveSession:
    now = time.monotonic()
    return {
        "dir": str(tmp_path),
        "process": ffmpeg_session._DeadProcess(),
        "ingest_process": None,
        "extra_processes": [],
        "url": URL,
        "username": "viewer",
        "started": time.time(),
        "last_access": time.time(),
        "fast_start": True,
        "playback_policy": PlaybackPolicy(),
        "ingest_restart_command": ["ingest"],
        # The restart budget is spent: three attempts inside the window.
        "live_recovery": LiveRecoveryPolicy(
            failures={"input": now - 10},
            attempts={"pipeline": (now - 40, now - 30, now - 15)},
        ),
    }


@pytest.mark.asyncio
async def test_watchdog_releases_a_pipeline_it_can_no_longer_restart(
    startup: Startup,
    tmp_path: pathlib.Path,
):
    ffmpeg_session._transcode_sessions["dead"] = _dead_fast_session(tmp_path)
    ffmpeg_session._url_to_session[URL] = "dead"
    with patch("ffmpeg_session.ready_bitrate", return_value=0):
        await ffmpeg_session._check_fast_live_session(
            "dead",
            ffmpeg_session._transcode_sessions["dead"],
        )
    assert "dead" not in ffmpeg_session._transcode_sessions
    assert ffmpeg_session._get_existing_session(URL) == (None, False, 0.0)
    assert not tmp_path.exists()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
