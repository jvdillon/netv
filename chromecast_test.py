from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from pychromecast.error import NotConnected

import pytest

import chromecast


@dataclass(kw_only=True, slots=True)
class MediaStatus:
    """Mutable stand-in for a receiver's media status."""

    content_id: str | None = None
    player_state: str = "IDLE"
    current_time: float = 30
    adjusted_current_time: float = 30
    idle_reason: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CastEnv:
    """Cast manager wired to a fake receiver and fake playback session."""

    manager: chromecast.CastManager
    cast: MagicMock
    status: MediaStatus
    session: dict[str, object]
    connect: MagicMock
    touch: MagicMock
    stop: MagicMock
    save: MagicMock


@pytest.fixture
def cast_env() -> Iterator[CastEnv]:
    manager = chromecast.CastManager()
    cast = MagicMock()
    cast.cast_info.friendly_name = "Living Room"
    cast.status.volume_level = 0.5
    cast.socket_client.is_connected = True
    status = MediaStatus()
    cast.media_controller.status = status

    def load(url: str, *_args: object, **_kwargs: object) -> None:
        status.content_id = url
        status.player_state = "PLAYING"

    cast.media_controller.play_media.side_effect = load
    session: dict[str, object] = {
        "username": "alice",
        "url": "http://provider/live",
        "is_vod": False,
        "duration": 3600,
        "seek_offset": 120,
    }
    with (
        patch("chromecast.get_chromecast_from_host", return_value=cast) as connect,
        patch("chromecast.ffmpeg_session.get_session", return_value=session),
        patch("chromecast.ffmpeg_session.touch_session") as touch,
        patch("chromecast.save_watch_position") as save,
        patch.object(manager, "_ensure_worker"),
    ):
        yield CastEnv(
            manager=manager,
            cast=cast,
            status=status,
            session=session,
            connect=connect,
            touch=touch,
            stop=MagicMock(),
            save=save,
        )
    manager.close()


def start(
    env: CastEnv,
    username: str = "alice",
    host: str = "192.168.1.50",
) -> chromecast.CastStatus:
    return env.manager.start(username, host, "session123", "http://192.168.1.10:8000", "News", 42)


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "0.0.0.0", "8.8.8.8", "169.254.1.1", "::1", "224.0.0.1"],
)
def test_device_must_be_private_lan_ip(host: str):
    with pytest.raises(HTTPException) as error:
        chromecast.lan_host(host)
    assert error.value.status_code == 400


@pytest.mark.parametrize("host", ["192.168.1.50", "10.0.0.2", "172.16.0.2", "fd00::1"])
def test_private_device_addresses(host: str):
    assert chromecast.lan_host(host) == host


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://[::1]:8000",
        "http://0.0.0.0:8000",
        "file:///etc/passwd",
        "http://user:secret@server",
        "http://server/path",
        "http://server:bad",
        "http://server?token=1",
        "http://host.docker.internal:8000",
    ],
)
def test_unreachable_or_invalid_media_origin(origin: str):
    with pytest.raises(HTTPException):
        chromecast.media_origin(origin)


def test_http_cast_reuses_owned_session_and_live_edge(cast_env: CastEnv):
    result = start(cast_env)
    assert result["active"] is True
    assert result.get("session_id") == "session123"
    assert result.get("name") == "Living Room"
    args, kwargs = cast_env.cast.media_controller.play_media.call_args
    assert args == (
        "http://192.168.1.10:8000/transcode/session123/stream.m3u8",
        "application/x-mpegurl",
    )
    assert kwargs["stream_type"] == "LIVE"
    assert kwargs["current_time"] is None
    assert cast_env.cast.media_controller.app_must_match is True
    assert cast_env.manager.owns_session("session123")
    assert cast_env.manager.owns_url("http://provider/live")


def test_vod_and_series_use_buffered_media_and_relative_position(cast_env: CastEnv):
    cast_env.session["is_vod"] = True
    start(cast_env)
    kwargs = cast_env.cast.media_controller.play_media.call_args.kwargs
    assert kwargs["stream_type"] == "BUFFERED"
    assert kwargs["current_time"] == 42
    cast_env.manager.maintain_sessions()
    cast_env.save.assert_called_with("alice", "http://provider/live", 150, 3600)


def test_adaptive_cast_uses_master_playlist(cast_env: CastEnv):
    cast_env.session["fast_start"] = True
    start(cast_env)
    assert cast_env.cast.media_controller.play_media.call_args.args[0].endswith("/master.m3u8")


def test_other_user_cannot_cast_or_control_session(cast_env: CastEnv):
    with pytest.raises(HTTPException) as error:
        start(cast_env, username="bob")
    assert error.value.status_code == 404
    cast_env.connect.assert_not_called()
    start(cast_env)
    assert cast_env.manager.status("bob") == {"active": False}
    with pytest.raises(HTTPException) as error:
        cast_env.manager.command("bob", "stop", release=cast_env.stop)
    assert error.value.status_code == 404
    cast_env.cast.media_controller.stop.assert_not_called()


def test_existing_cast_cannot_be_overwritten(cast_env: CastEnv):
    start(cast_env)
    with pytest.raises(HTTPException) as error:
        start(cast_env, host="192.168.1.51")
    assert error.value.status_code == 409
    cast_env.session["username"] = "bob"
    with pytest.raises(HTTPException) as error:
        start(cast_env, username="bob")
    assert error.value.status_code == 409


def test_connection_failure_releases_reservation_but_keeps_local_stream(cast_env: CastEnv):
    cast_env.cast.wait.side_effect = NotConnected()
    with pytest.raises(HTTPException) as error:
        start(cast_env)
    assert error.value.status_code == 502
    assert not cast_env.manager.owns_session("session123")
    cast_env.cast.disconnect.assert_called_once()
    cast_env.stop.assert_not_called()


def test_load_timeout_does_not_report_success(cast_env: CastEnv):
    cast_env.cast.media_controller.play_media.side_effect = None
    with (
        patch("chromecast.time.monotonic", side_effect=[0, 16]),
        pytest.raises(HTTPException) as error,
    ):
        start(cast_env)
    assert error.value.status_code == 502
    assert cast_env.manager.status("alice") == {"active": False}
    cast_env.cast.disconnect.assert_called_once()


def test_receiver_load_error_fails_fast(cast_env: CastEnv):
    def load(url: str, *_args: object, **_kwargs: object) -> None:
        cast_env.status.content_id = url
        cast_env.status.idle_reason = "ERROR"

    cast_env.cast.media_controller.play_media.side_effect = load
    with (
        patch("chromecast.time.monotonic", side_effect=[0, 1]),
        pytest.raises(HTTPException) as error,
    ):
        start(cast_env)
    assert error.value.status_code == 502
    assert "could not load" in error.value.detail
    assert not cast_env.manager.owns_session("session123")


def test_start_polls_until_receiver_plays(cast_env: CastEnv):
    status = cast_env.status
    cast_env.cast.media_controller.play_media.side_effect = None

    def wait(secs: float) -> None:
        del secs
        status.content_id = "http://192.168.1.10:8000/transcode/session123/stream.m3u8"
        status.player_state = "BUFFERING"

    with (
        patch("chromecast.time.monotonic", side_effect=[0, 1, 2]),
        patch.object(cast_env.manager._stop, "wait", side_effect=wait) as waited,
    ):
        assert start(cast_env).get("state") == "BUFFERING"
    waited.assert_called_once_with(0.1)


def test_status_while_connecting_and_unknown(cast_env: CastEnv):
    record = chromecast.CastSession(
        username="alice",
        host="192.168.1.50",
        session_id="s",
        url="u",
        source_url="p",
        name="TV",
    )
    cast_env.manager._sessions[record.host] = record
    assert cast_env.manager.status("alice") == {"active": False, "connecting": True}
    record.starting = False
    assert cast_env.manager.status("alice") == {
        "active": True,
        "host": "192.168.1.50",
        "name": "TV",
        "session_id": "s",
        "state": "UNKNOWN",
        "current_time": 0,
        "volume": 1,
    }


def test_starting_session_is_kept_alive(cast_env: CastEnv):
    record = chromecast.CastSession(
        username="alice",
        host="192.168.1.50",
        session_id="s",
        url="u",
        source_url="p",
        name="TV",
    )
    cast_env.manager._sessions[record.host] = record
    cast_env.manager.maintain_sessions()
    cast_env.touch.assert_called_once_with("s")
    assert cast_env.manager._sessions == {record.host: record}


def test_volume_requires_a_level(cast_env: CastEnv):
    start(cast_env)
    with pytest.raises(HTTPException) as error:
        cast_env.manager.command("alice", "volume", release=cast_env.stop)
    assert error.value.status_code == 400
    cast_env.cast.set_volume.assert_not_called()


def test_unresponsive_receiver_command_fails(cast_env: CastEnv):
    start(cast_env)
    cast_env.cast.media_controller.pause.side_effect = NotConnected()
    with pytest.raises(HTTPException) as error:
        cast_env.manager.command("alice", "pause", release=cast_env.stop)
    assert error.value.status_code == 502


def test_save_position_skips_foreign_media_and_early_positions(cast_env: CastEnv):
    cast_env.session["is_vod"] = True
    start(cast_env)
    record = next(iter(cast_env.manager._sessions.values()))
    cast_env.status.content_id = "another-app"
    cast_env.manager._save_position(record)
    cast_env.status.content_id = record.url
    cast_env.session["seek_offset"] = 0
    cast_env.status.adjusted_current_time = 2
    cast_env.manager._save_position(record)
    cast_env.save.assert_not_called()
    cast_env.status.adjusted_current_time = 9999
    cast_env.manager._save_position(record)
    cast_env.save.assert_called_once_with("alice", "http://provider/live", 3600, 3600)


def test_save_position_write_failure_is_logged(
    cast_env: CastEnv,
    caplog: pytest.LogCaptureFixture,
):
    cast_env.session["is_vod"] = True
    start(cast_env)
    record = next(iter(cast_env.manager._sessions.values()))
    cast_env.save.side_effect = OSError("disk full")
    cast_env.manager._save_position(record)
    assert "Could not save Chromecast watch position" in caplog.text


def test_worker_started_once_and_joined_on_close():
    manager = chromecast.CastManager()
    with patch("chromecast.threading.Thread") as thread_type:
        worker = thread_type.return_value
        worker.is_alive.return_value = True
        manager._ensure_worker()
        manager._ensure_worker()
    thread_type.assert_called_once_with(target=manager._monitor, daemon=True, name="cast-monitor")
    worker.start.assert_called_once()
    manager.close()
    worker.join.assert_called_once_with(timeout=6)
    assert manager._stop.is_set()


def test_monitor_maintains_until_stopped():
    manager = chromecast.CastManager()
    with (
        patch.object(manager._stop, "wait", side_effect=[False, False, True]),
        patch.object(manager, "maintain_sessions") as maintain,
    ):
        manager._monitor()
    assert maintain.call_count == 2


def test_server_keeps_cast_alive_without_browser(cast_env: CastEnv):
    start(cast_env)
    cast_env.touch.reset_mock()
    cast_env.status.player_state = "PAUSED"
    cast_env.manager.maintain_sessions()
    cast_env.touch.assert_called_once_with("session123")
    assert cast_env.manager.status("alice")["active"]


def test_disconnected_receiver_releases_ownership(cast_env: CastEnv):
    start(cast_env)
    cast_env.cast.socket_client.is_connected = False
    cast_env.manager.maintain_sessions()
    assert cast_env.manager.status("alice") == {"active": False}


def test_other_receiver_app_is_not_stopped(cast_env: CastEnv):
    start(cast_env)
    cast_env.status.content_id = "another-app"
    with pytest.raises(HTTPException) as error:
        cast_env.manager.command("alice", "stop", release=cast_env.stop)
    assert error.value.status_code == 409
    cast_env.cast.media_controller.stop.assert_not_called()


def test_remote_controls_and_stop_release_transcode(cast_env: CastEnv):
    start(cast_env)
    cast_env.manager.command("alice", "pause", release=cast_env.stop)
    cast_env.manager.command("alice", "play", release=cast_env.stop)
    cast_env.manager.command("alice", "volume", 0.25, release=cast_env.stop)
    assert cast_env.manager.command("alice", "stop", release=cast_env.stop) == {"active": False}
    cast_env.cast.media_controller.pause.assert_called_once()
    cast_env.cast.media_controller.play.assert_called_once()
    cast_env.cast.set_volume.assert_called_once_with(0.25)
    cast_env.stop.assert_called_once_with("session123")


def test_discovery_closes_browser_and_returns_lan_devices():
    manager = chromecast.CastManager()
    with (
        patch("chromecast.Zeroconf"),
        patch("chromecast.CastBrowser") as browser_type,
        patch.object(manager._stop, "wait"),
    ):
        browser = browser_type.return_value
        browser.devices = {
            "a": SimpleNamespace(host="192.168.1.50", friendly_name="TV"),
            "b": SimpleNamespace(host="127.0.0.1", friendly_name="Not a TV"),
        }
        assert manager.discover() == [{"host": "192.168.1.50", "name": "TV"}]
        browser.stop_discovery.assert_called_once()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
