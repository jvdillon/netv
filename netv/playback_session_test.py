"""Verify the fallback changes the actual encoder configuration."""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import asyncio

from fastapi import HTTPException

import pytest
import pytest_asyncio

from netv import cache, fast_start, ffmpeg_command, ffmpeg_session, util
from netv.fast_start_test import ffprobe_source, output_dir, serve_fast_start, serve_transcode
from netv.playback_policy import PlaybackHealth
from netv.testing import (
    BackgroundTasks,
    FakeClock,
    FakeProcess,
    FakeRun,
    FakeUpstreams,
    FakeWeb,
    ProcessLauncher,
    Program,
    install_clock,
    install_launcher,
    install_subprocess,
    install_web,
    probed_video_json,
)


LIVE = "http://example.com/live.ts"


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Each readiness poll advances fake time instead of sleeping 0.2s."""
    fake = FakeClock()
    install_clock(monkeypatch, fake, ffmpeg_session)
    return fake


@pytest.fixture(autouse=True)
def programs(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    """No external program exists unless a test scripts it."""
    run = FakeRun()
    install_subprocess(monkeypatch, ffmpeg_command, run=run)
    return run


@pytest.fixture(autouse=True)
def fresh_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg_session, "_transcode_sessions", {})
    monkeypatch.setattr(ffmpeg_session, "_url_to_session", {})
    monkeypatch.setattr(ffmpeg_session, "_startup_guards", {})
    monkeypatch.setattr(ffmpeg_command, "_probe_cache", {})


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> Iterator[ProcessLauncher]:
    """Record ffmpeg launches; stderr monitors stay unscheduled."""
    tasks = BackgroundTasks()
    monkeypatch.setattr(ffmpeg_session, "_spawn_background_task", tasks)
    fake = ProcessLauncher()
    install_launcher(monkeypatch, fake, ffmpeg_session)
    yield fake
    tasks.discard()


@pytest_asyncio.fixture
async def upstreams(
    monkeypatch: pytest.MonkeyPatch,
    programs: FakeRun,
) -> AsyncIterator[FakeUpstreams]:
    """The live provider, probed as 1080p video; closed before the loop ends to free readers."""
    programs.program("ffprobe", Program(stdout=probed_video_json()))
    opened = FakeUpstreams()
    install_web(monkeypatch, FakeWeb({LIVE: opened}), util)
    yield opened
    opened.close()


@pytest.fixture
def transcode_dir(tmp_path: Path) -> Path:
    """Configure plain live transcodes into a fresh directory, without probing."""
    directory = tmp_path / "transcode"
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "transcode_dir": str(directory),
            "max_resolution": "1080p",
            "probe_live": False,
            "probe_movies": False,
        },
    )
    return directory


def existing_session(
    directory: Path,
    process: FakeProcess | None,
    *,
    username: str = "viewer",
    audio_passthrough: bool = False,
    session_id: str = "old",
    url: str = "stream",
    now: float,
) -> None:
    """Register a running live session for ``url``."""
    directory.mkdir()
    ffmpeg_session._transcode_sessions[session_id] = {
        "dir": str(directory),
        "process": process,
        "started": now,
        "last_access": now,
        "url": url,
        "username": username,
        "audio_passthrough": audio_passthrough,
    }
    ffmpeg_session._url_to_session[url] = session_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_resolution,expected_resolution",
    [("4k", "720p"), ("480p", "480p")],
)
async def test_fallback_command_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transcode_dir: Path,
    programs: FakeRun,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
    configured_resolution: cache.MaxResolution,
    expected_resolution: str,
):
    # A 240p source on an installed and selected AI Upscale: only the saver keeps it off.
    small = '{"streams": [{"codec_type": "video", "codec_name": "mpeg2video", "height": 240}]}'
    programs.program("ffprobe", Program(stdout=small))
    engines = tmp_path / "engines"
    engines.mkdir()
    (engines / "2x-nomosuni-compact_1080p_fp16.engine").touch()
    monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", str(engines))
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "transcode_hw": "nvenc+software",
            "max_resolution": configured_resolution,
            "quality": "high",
            "sr_model": "nomos",
        },
    )
    task = asyncio.create_task(ffmpeg_session.start_transcode(LIVE, bandwidth_saver=True))
    result = await serve_fast_start(launcher, upstreams, task)
    try:
        [command] = launcher.commands
        session = ffmpeg_session.get_session(result["session_id"])
        assert session is not None
        encoder = (str(output_dir(launcher, 0)), "nvenc+software", expected_resolution, "low", True)
        source = session.get("audio_info")
        assert command == fast_start.encoder_command(
            *encoder,
            allow_upscale=False,
            audio=source,
            video_codec="mpeg2video",
        )
        assert not any("dnn_processing=" in arg for arg in command)
        upscaled = fast_start.encoder_command(
            *encoder,
            allow_upscale=True,
            audio=source,
            video_codec="mpeg2video",
            rendition="up",
        )
        assert any("dnn_processing=" in arg for arg in upscaled)
        # Nor does the session start an AI Upscale worker beside it.
        assert ffmpeg_session._is_fast_live(session)
        assert not session["plan"].upscales(encoder[0], source)
        settings = cache.load_server_settings()
        assert settings["transcode_hw"] == "nvenc+software"
        assert settings.get("max_resolution") == configured_resolution
        assert settings.get("quality") == "high"
        assert settings.get("sr_model") == "nomos"
        assert session.get("bandwidth_saver") is True
    finally:
        await ffmpeg_session.stop_session(result["session_id"], force=True)
    assert not list(transcode_dir.iterdir())


@pytest.mark.asyncio
async def test_fallback_does_not_reuse_high_quality_stream(
    tmp_path: Path,
    clock: FakeClock,
    transcode_dir: Path,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
):
    old = FakeProcess(exit_code=None)
    existing_session(tmp_path / "old", old, url=LIVE, now=clock.now)
    task = asyncio.create_task(
        ffmpeg_session.start_transcode(LIVE, username="viewer", bandwidth_saver=True),
    )
    result = await serve_fast_start(launcher, upstreams, task)
    assert result["session_id"] != "old"
    assert old.signals == ["TERM"]
    assert not (tmp_path / "old").exists()
    assert "old" not in ffmpeg_session._transcode_sessions
    assert ffmpeg_session._url_to_session[LIVE] == result["session_id"]
    assert len(launcher.launches) == 1
    session = ffmpeg_session.get_session(result["session_id"])
    assert session is not None
    assert session.get("bandwidth_saver") is True
    await ffmpeg_session.stop_session(result["session_id"], force=True)
    assert not list(transcode_dir.iterdir())


@pytest.mark.asyncio
async def test_saver_session_can_request_recovery(
    clock: FakeClock,
    transcode_dir: Path,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
):
    task = asyncio.create_task(ffmpeg_session.start_transcode(LIVE, bandwidth_saver=True))
    result = await serve_fast_start(launcher, upstreams, task)
    session_id = result["session_id"]
    try:
        good = PlaybackHealth(buffer_seconds=12, waiting=False, observed_bitrate=20_000_000)
        start = clock.now
        for offset in range(0, 60, 2):
            clock.now = start + offset
            assert ffmpeg_session.report_playback_health(session_id, "", good) == {
                "bandwidth_saver": True,
                "playlist": result["playlist"],
            }
        clock.now = start + 60
        # The saver encoder cannot serve full quality: the client retunes to a new session.
        assert ffmpeg_session.report_playback_health(session_id, "", good) == {
            "bandwidth_saver": False,
        }
    finally:
        await ffmpeg_session.stop_session(session_id, force=True)
    assert not list(transcode_dir.iterdir())


@pytest.mark.asyncio
async def test_aac_client_does_not_reuse_dolby_session(
    tmp_path: Path,
    clock: FakeClock,
    programs: FakeRun,
    transcode_dir: Path,
    launcher: ProcessLauncher,
):
    """A client without Dolby decoding replaces its own passthrough session."""
    cache.save_server_settings({**cache.load_server_settings(), "probe_movies": True})
    programs.program("ffprobe", Program(stdout=ffprobe_source("h264", "eac3", 6)), when="stream")
    old = FakeProcess(exit_code=None)
    existing_session(tmp_path / "old", old, audio_passthrough=True, now=clock.now)
    ffmpeg_session._transcode_sessions["old"].update(is_vod=True, seek_offset=12.0)
    task = asyncio.create_task(ffmpeg_session.start_transcode("stream", "movie", username="viewer"))
    result = await serve_transcode(launcher, task)
    assert result["session_id"] != "old"
    assert old.signals == ["TERM"]
    assert not (tmp_path / "old").exists()
    command = launcher.commands[0]
    # The VOD resume position survives the restart.
    assert command[command.index("-i") - 2 : command.index("-i")] == ["-ss", "12.0"]
    assert result["seek_offset"] == 12.0
    # The replacement encodes the Dolby source to AAC.
    assert command[command.index("-c:a") + 1] == "aac"
    session = ffmpeg_session.get_session(result["session_id"])
    assert session is not None
    assert session.get("audio_passthrough") is False
    await ffmpeg_session.stop_session(result["session_id"], force=True)
    assert not list(transcode_dir.iterdir())


@pytest.mark.asyncio
async def test_dolby_session_of_another_user_is_not_stopped(
    tmp_path: Path,
    clock: FakeClock,
    launcher: ProcessLauncher,
):
    old = FakeProcess(exit_code=None)
    existing_session(tmp_path / "old", old, username="other", audio_passthrough=True, now=clock.now)
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode("stream", username="viewer")
    assert error.value.status_code == 409
    assert old.signals == []
    assert "old" in ffmpeg_session._transcode_sessions
    assert (tmp_path / "old").exists()
    assert launcher.launches == []


@pytest.mark.asyncio
async def test_passthrough_client_reuses_dolby_session(
    tmp_path: Path,
    clock: FakeClock,
    launcher: ProcessLauncher,
):
    old = FakeProcess(exit_code=None)
    existing_session(tmp_path / "old", old, audio_passthrough=True, now=clock.now)
    ffmpeg_session._transcode_sessions["old"]["fast_start"] = True
    result = await ffmpeg_session.start_transcode(
        "stream",
        username="viewer",
        audio_passthrough=True,
    )
    assert result["session_id"] == "old"
    assert result["playlist"] == "/transcode/old/low.m3u8"
    assert old.signals == []
    assert launcher.launches == []


@pytest.mark.asyncio
async def test_starting_passthrough_session_is_not_reused_by_aac_client(
    tmp_path: Path,
    clock: FakeClock,
    transcode_dir: Path,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
):
    """While the audio probe is pending, a passthrough request's session counts as Dolby."""
    pending = tmp_path / "pending"
    worker = FakeProcess(exit_code=None)
    existing_session(
        pending,
        worker,
        audio_passthrough=True,
        session_id="pending",
        url=LIVE,
        now=clock.now,
    )
    ffmpeg_session._transcode_sessions["pending"]["fast_start"] = True
    task = asyncio.create_task(ffmpeg_session.start_transcode(LIVE, username="viewer"))
    result = await serve_fast_start(launcher, upstreams, task)
    assert result["session_id"] != "pending"
    assert worker.signals == ["TERM"]
    assert "pending" not in ffmpeg_session._transcode_sessions
    assert not pending.exists()
    assert ffmpeg_session._url_to_session[LIVE] == result["session_id"]
    await ffmpeg_session.stop_session(result["session_id"], force=True)
    assert not list(transcode_dir.iterdir())


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
