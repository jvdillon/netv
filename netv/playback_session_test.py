"""Verify the fallback changes the actual encoder configuration."""

from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import pytest

from netv import ffmpeg_session
from netv.ffmpeg_session_test import FakeProcess
from netv.playback_policy import PlaybackHealth


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_resolution,expected_resolution",
    [("4k", "720p"), ("480p", "480p")],
)
async def test_fallback_command_and_cleanup(
    tmp_path: Path,
    configured_resolution: str,
    expected_resolution: str,
):
    settings = {"max_resolution": configured_resolution, "quality": "high", "sr_model": "nomos"}
    with (
        patch("netv.ffmpeg_session.get_settings", return_value=settings),
        patch("netv.ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch("netv.ffmpeg_session.resolve_hls_master_playlist", side_effect=_same_url),
        patch(
            "netv.ffmpeg_session.build_hls_ffmpeg_cmd",
            return_value=["ffmpeg", "-i", "stream"],
        ) as command,
        patch(
            "netv.ffmpeg_session.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=FakeProcess(),
        ),
        patch(
            "netv.ffmpeg_session._spawn_background_task",
            side_effect=_close_coroutine,
        ),
        patch("netv.ffmpeg_session._wait_for_playlist", new_callable=AsyncMock, return_value=True),
    ):
        result = await ffmpeg_session.start_transcode(
            "http://example.com/live.ts",
            bandwidth_saver=True,
        )
        try:
            assert command.call_args.args[6:8] == (expected_resolution, "low")
            assert command.call_args.kwargs["allow_upscale"] is False
            assert settings == {
                "max_resolution": configured_resolution,
                "quality": "high",
                "sr_model": "nomos",
            }
            session = ffmpeg_session.get_session(result["session_id"])
            assert session is not None
            assert session.get("bandwidth_saver") is True
        finally:
            await ffmpeg_session.stop_session(result["session_id"], force=True)
        assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_fallback_does_not_reuse_high_quality_stream():
    with (
        patch("netv.ffmpeg_session._get_existing_session", return_value=("old", True, 0)),
        patch("netv.ffmpeg_session.get_session", return_value={"username": "viewer"}),
        patch("netv.ffmpeg_session.stop_session", new_callable=AsyncMock) as stop,
        patch("netv.ffmpeg_session.enforce_stream_limits", new=AsyncMock(return_value=None)),
        patch("netv.ffmpeg_session._try_reuse_session", new_callable=AsyncMock) as reuse,
        patch(
            "netv.ffmpeg_session._do_start_transcode",
            new_callable=AsyncMock,
            return_value={"session_id": "new"},
        ) as start,
    ):
        result = await ffmpeg_session.start_transcode(
            "stream",
            username="viewer",
            bandwidth_saver=True,
        )
    assert result == {"session_id": "new"}
    stop.assert_called_once_with("old", force=True)
    reuse.assert_not_called()
    assert start.call_args.args[-1] is True


@pytest.mark.asyncio
async def test_legacy_saver_session_can_request_recovery(tmp_path: Path):

    with (
        patch("netv.ffmpeg_session.get_settings", return_value={"max_resolution": "1080p"}),
        patch("netv.ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch("netv.ffmpeg_session.resolve_hls_master_playlist", side_effect=_same_url),
        patch("netv.ffmpeg_session.build_hls_ffmpeg_cmd", return_value=["ffmpeg", "-i", "stream"]),
        patch(
            "netv.ffmpeg_session.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=FakeProcess()),
        ),
        patch("netv.ffmpeg_session._spawn_background_task", side_effect=_close_coroutine),
        patch("netv.ffmpeg_session._wait_for_playlist", new=AsyncMock(return_value=True)),
    ):
        result = await ffmpeg_session.start_transcode(
            "http://example.com/live.ts",
            bandwidth_saver=True,
        )
        session_id = result["session_id"]
        try:
            good = PlaybackHealth(buffer_seconds=12, waiting=False, observed_bitrate=20_000_000)
            with patch("netv.ffmpeg_session.time.monotonic") as clock:
                for now in range(0, 60, 2):
                    clock.return_value = now
                    assert ffmpeg_session.report_playback_health(session_id, "", good) == {
                        "bandwidth_saver": True,
                    }
                clock.return_value = 60
                assert ffmpeg_session.report_playback_health(session_id, "", good) == {
                    "bandwidth_saver": False,
                }
        finally:
            await ffmpeg_session.stop_session(session_id, force=True)


@pytest.mark.asyncio
async def test_aac_client_does_not_reuse_dolby_session():
    """A client without Dolby decoding replaces its own passthrough session."""
    with (
        patch("netv.ffmpeg_session._get_existing_session", return_value=("old", True, 12.0)),
        patch(
            "netv.ffmpeg_session.get_session",
            return_value={"username": "viewer", "audio_passthrough": True},
        ),
        patch("netv.ffmpeg_session.stop_session", new_callable=AsyncMock) as stop,
        patch("netv.ffmpeg_session.enforce_stream_limits", new=AsyncMock(return_value=None)),
        patch("netv.ffmpeg_session._try_reuse_session", new_callable=AsyncMock) as reuse,
        patch(
            "netv.ffmpeg_session._do_start_transcode",
            new_callable=AsyncMock,
            return_value={"session_id": "new"},
        ) as start,
    ):
        result = await ffmpeg_session.start_transcode("stream", "movie", username="viewer")
    assert result == {"session_id": "new"}
    stop.assert_called_once_with("old", force=True)
    reuse.assert_not_called()
    assert start.call_args.args[4] == 12.0  # VOD resume position survives the restart.
    assert start.call_args.kwargs["audio_passthrough"] is False


@pytest.mark.asyncio
async def test_dolby_session_of_another_user_is_not_stopped():
    with (
        patch("netv.ffmpeg_session._get_existing_session", return_value=("old", True, 0)),
        patch(
            "netv.ffmpeg_session.get_session",
            return_value={"username": "other", "audio_passthrough": True},
        ),
        patch("netv.ffmpeg_session.stop_session", new_callable=AsyncMock) as stop,
        patch("netv.ffmpeg_session.enforce_stream_limits", new=AsyncMock(return_value=None)),
        pytest.raises(HTTPException) as error,
    ):
        await ffmpeg_session.start_transcode("stream", username="viewer")
    assert error.value.status_code == 409
    stop.assert_not_called()


@pytest.mark.asyncio
async def test_passthrough_client_reuses_dolby_session():
    session = {"username": "viewer", "audio_passthrough": True}
    with (
        patch("netv.ffmpeg_session._get_existing_session", return_value=("old", True, 0)),
        patch("netv.ffmpeg_session.get_session", return_value=session),
        patch("netv.ffmpeg_session.stop_session", new_callable=AsyncMock) as stop,
        patch("netv.ffmpeg_session.enforce_stream_limits", new=AsyncMock(return_value=None)),
        patch(
            "netv.ffmpeg_session._try_reuse_session",
            new_callable=AsyncMock,
            return_value={"session_id": "old"},
        ),
    ):
        result = await ffmpeg_session.start_transcode(
            "stream",
            username="viewer",
            audio_passthrough=True,
        )
    assert result == {"session_id": "old"}
    stop.assert_not_called()


@pytest.mark.asyncio
async def test_starting_passthrough_session_is_not_reused_by_aac_client():
    """While the audio probe is pending, a passthrough request's session counts as Dolby."""
    ffmpeg_session._transcode_sessions["pending"] = {
        "username": "viewer",
        "fast_start": True,
        "audio_passthrough": True,
        "url": "stream",
        "process": None,
        "dir": "/nonexistent",
        "last_access": 0,
        "started": 0,
    }
    ffmpeg_session._url_to_session["stream"] = "pending"
    try:
        with (
            patch("netv.ffmpeg_session._get_existing_session", return_value=("pending", True, 0)),
            patch("netv.ffmpeg_session.stop_session", new_callable=AsyncMock) as stop,
            patch("netv.ffmpeg_session.enforce_stream_limits", new=AsyncMock(return_value=None)),
            patch(
                "netv.ffmpeg_session._do_start_transcode",
                new_callable=AsyncMock,
                return_value={"session_id": "new"},
            ),
        ):
            await ffmpeg_session.start_transcode("stream", username="viewer")
        stop.assert_called_once_with("pending", force=True)
    finally:
        ffmpeg_session._transcode_sessions.pop("pending", None)
        ffmpeg_session._url_to_session.pop("stream", None)


def _same_url(url: str) -> str:
    return url


def _close_coroutine(coroutine: Coroutine[object, object, object]) -> None:
    coroutine.close()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
