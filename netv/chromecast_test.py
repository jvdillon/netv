"""Tests for chromecast.py."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fastapi import HTTPException
from pychromecast.error import NotConnected

import pytest

from netv import cache, chromecast, testing
from netv.ffmpeg import transcode
from netv.ffmpeg.transcode import TranscodeSession


HOST = "192.168.1.50"
SOURCE_URL = "http://provider/live"
CAST_URL = "http://192.168.1.10:8000/transcode/session123/stream.m3u8"


@dataclass(frozen=True, kw_only=True, slots=True)
class CastEnv:
    """Cast manager on a fake LAN, playing a real transcode session."""

    manager: chromecast.CastManager
    network: testing.CastNetwork
    receiver: testing.FakeChromecast
    session: TranscodeSession
    clock: testing.FakeClock
    threads: testing.ThreadRecorder
    released: list[str]

    def release(self, session_id: str) -> None:
        self.released.append(session_id)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> testing.FakeClock:
    clock = testing.FakeClock()
    testing.install_clock(monkeypatch, clock, chromecast, transcode)
    return clock


@pytest.fixture
def threads(monkeypatch: pytest.MonkeyPatch) -> testing.ThreadRecorder:
    threads = testing.ThreadRecorder(run_inline=False)
    testing.install_threads(monkeypatch, threads, chromecast)
    return threads


@pytest.fixture
def cast_env(
    monkeypatch: pytest.MonkeyPatch,
    clock: testing.FakeClock,
    threads: testing.ThreadRecorder,
) -> CastEnv:
    receiver = testing.FakeChromecast(friendly_name="Living Room", clock=clock)
    assert receiver.status is not None
    receiver.status.volume_level = 0.5
    network = testing.CastNetwork({HOST: receiver})
    testing.install_chromecast(monkeypatch, network, chromecast)
    session: TranscodeSession = {
        "dir": "",
        "process": None,
        "started": 0.0,
        "username": "alice",
        "url": SOURCE_URL,
        "is_vod": False,
        "duration": 3600,
        "seek_offset": 120,
    }
    monkeypatch.setattr(transcode, "_transcode_sessions", {"session123": session})
    return CastEnv(
        manager=chromecast.CastManager(),
        network=network,
        receiver=receiver,
        session=session,
        clock=clock,
        threads=threads,
        released=[],
    )


def start(
    env: CastEnv,
    username: str = "alice",
    host: str = HOST,
) -> chromecast.CastStatus:
    return env.manager.start(username, host, "session123", "http://192.168.1.10:8000", "News", 42)


def watched(username: str = "alice") -> dict[str, tuple[float, float]]:
    history = cache.load_user_settings(username)["watch_history"]
    return {url: (entry["position"], entry["duration"]) for url, entry in history.items()}


def starting_record(session_id: str = "s") -> chromecast.CastSession:
    return chromecast.CastSession(
        username="alice",
        host=HOST,
        session_id=session_id,
        url="u",
        source_url="p",
        name="TV",
    )


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
    assert result.get("volume") == 0.5
    [load] = cast_env.receiver.media_controller.loaded
    assert (load.url, load.content_type) == (CAST_URL, "application/x-mpegurl")
    assert load.options["stream_type"] == "LIVE"
    assert load.options["current_time"] is None
    assert load.options["title"] == "News"
    assert cast_env.receiver.media_controller.app_must_match is True
    assert cast_env.network.connects == [HOST]
    assert cast_env.session.get("last_access") == cast_env.clock.now
    assert cast_env.manager.owns_session("session123")
    assert cast_env.manager.owns_url(SOURCE_URL)


def test_vod_and_series_use_buffered_media_and_relative_position(cast_env: CastEnv):
    cast_env.session["is_vod"] = True
    start(cast_env)
    [load] = cast_env.receiver.media_controller.loaded
    assert load.options["stream_type"] == "BUFFERED"
    assert load.options["current_time"] == 42
    cast_env.receiver.media_controller.status.adjusted_current_time = 30
    cast_env.manager.maintain_sessions()
    assert watched() == {SOURCE_URL: (150, 3600)}


def test_adaptive_cast_uses_master_playlist(cast_env: CastEnv):
    cast_env.session["fast_start"] = True
    start(cast_env)
    assert cast_env.receiver.media_controller.loaded[0].url.endswith("/master.m3u8")


def test_receiver_without_a_name_is_named_by_its_address(cast_env: CastEnv):
    cast_env.receiver.cast_info.friendly_name = None
    assert start(cast_env).get("name") == HOST


def test_other_user_cannot_cast_or_control_session(cast_env: CastEnv):
    with pytest.raises(HTTPException) as error:
        start(cast_env, username="bob")
    assert error.value.status_code == 404
    assert cast_env.network.connects == []
    start(cast_env)
    assert cast_env.manager.status("bob") == {"active": False}
    with pytest.raises(HTTPException) as error:
        cast_env.manager.command("bob", "stop", release=cast_env.release)
    assert error.value.status_code == 404
    assert "stop" not in cast_env.receiver.media_controller.commands
    assert cast_env.released == []


def test_missing_session_cannot_be_cast(cast_env: CastEnv):
    transcode._transcode_sessions.clear()
    with pytest.raises(HTTPException) as error:
        start(cast_env)
    assert error.value.status_code == 404


def test_existing_cast_cannot_be_overwritten(cast_env: CastEnv):
    start(cast_env)
    with pytest.raises(HTTPException) as error:
        start(cast_env, host="192.168.1.51")
    assert error.value.status_code == 409
    cast_env.session["username"] = "bob"
    with pytest.raises(HTTPException) as error:
        start(cast_env, username="bob")
    assert error.value.status_code == 409
    assert cast_env.network.connects == [HOST]


def test_unreachable_receiver_releases_reservation(cast_env: CastEnv):
    with pytest.raises(HTTPException) as error:
        start(cast_env, host="192.168.1.99")
    assert error.value.status_code == 502
    assert cast_env.network.connects == ["192.168.1.99"]
    assert not cast_env.manager.owns_session("session123")


def test_connection_failure_releases_reservation_but_keeps_local_stream(cast_env: CastEnv):
    cast_env.receiver.media_controller.error = NotConnected()
    with pytest.raises(HTTPException) as error:
        start(cast_env)
    assert error.value.status_code == 502
    assert not cast_env.manager.owns_session("session123")
    assert cast_env.receiver.disconnected
    assert cast_env.released == []
    assert transcode.get_session("session123") is not None


def test_load_timeout_does_not_report_success(cast_env: CastEnv):
    cast_env.receiver.media_controller.outcome = "hangs"
    started = cast_env.clock.now
    with pytest.raises(HTTPException) as error:
        start(cast_env)
    assert error.value.status_code == 502
    assert cast_env.clock.now - started == pytest.approx(15, abs=0.1)
    assert cast_env.manager.status("alice") == {"active": False}
    assert cast_env.receiver.disconnected


def test_receiver_load_error_fails_fast(cast_env: CastEnv):
    cast_env.receiver.media_controller.outcome = "fails"
    started = cast_env.clock.now
    with pytest.raises(HTTPException) as error:
        start(cast_env)
    assert error.value.status_code == 502
    assert "could not load" in error.value.detail
    assert cast_env.clock.now == started
    assert not cast_env.manager.owns_session("session123")


def test_start_polls_until_receiver_plays(cast_env: CastEnv):
    cast_env.receiver.media_controller.load_secs = 0.1
    cast_env.receiver.media_controller.started_state = "BUFFERING"
    started = cast_env.clock.now
    assert start(cast_env).get("state") == "BUFFERING"
    assert cast_env.clock.sleeps == [0.1]
    assert cast_env.clock.now == started + 0.1


def test_status_while_connecting_and_unknown(cast_env: CastEnv):
    record = starting_record()
    cast_env.manager._sessions[record.host] = record
    assert cast_env.manager.status("alice") == {"active": False, "connecting": True}
    cast_env.manager._sessions[record.host] = replace(record, starting=False)
    assert cast_env.manager.status("alice") == {
        "active": True,
        "host": HOST,
        "name": "TV",
        "session_id": "s",
        "state": "UNKNOWN",
        "current_time": 0,
        "volume": 1,
    }


def test_status_without_receiver_volume(cast_env: CastEnv):
    start(cast_env)
    cast_env.receiver.status = None
    assert cast_env.manager.status("alice").get("volume") == 1


def test_session_replacement_keeps_previous_snapshots_and_does_not_restore_removed_sessions(
    cast_env: CastEnv,
):
    start(cast_env)
    record = cast_env.manager._sessions[HOST]
    updated = replace(record, name="Bedroom")
    assert cast_env.manager._replace_session(record, updated) is updated
    assert cast_env.manager._sessions[record.host] is updated
    assert record.name == "Living Room"
    cast_env.manager._forget(updated)
    assert cast_env.manager._replace_session(record, updated) is updated
    assert cast_env.manager._sessions == {}


def test_starting_session_is_kept_alive(cast_env: CastEnv):
    record = starting_record("session123")
    cast_env.manager._sessions[record.host] = record
    cast_env.clock.advance(30)
    cast_env.manager.maintain_sessions()
    assert cast_env.session.get("last_access") == cast_env.clock.now
    assert cast_env.manager._sessions == {record.host: record}


def test_volume_requires_a_level(cast_env: CastEnv):
    start(cast_env)
    with pytest.raises(HTTPException) as error:
        cast_env.manager.command("alice", "volume", release=cast_env.release)
    assert error.value.status_code == 400
    assert "volume" not in cast_env.receiver.media_controller.commands


def test_unresponsive_receiver_command_fails(cast_env: CastEnv):
    start(cast_env)
    cast_env.receiver.media_controller.error = NotConnected()
    with pytest.raises(HTTPException) as error:
        cast_env.manager.command("alice", "pause", release=cast_env.release)
    assert error.value.status_code == 502


def test_save_position_skips_foreign_media_and_early_positions(cast_env: CastEnv):
    cast_env.session["is_vod"] = True
    start(cast_env)
    record = cast_env.manager._sessions[HOST]
    controller = cast_env.receiver.media_controller
    controller.status = testing.FakeMediaStatus(content_id="another-app", player_state="PLAYING")
    cast_env.manager._save_position(record)
    cast_env.session["seek_offset"] = 0
    controller.status = testing.FakeMediaStatus(content_id=record.url, adjusted_current_time=2)
    cast_env.manager._save_position(record)
    controller.status = testing.FakeMediaStatus(content_id=record.url, adjusted_current_time=None)
    cast_env.manager._save_position(record)
    assert watched() == {}
    controller.status = testing.FakeMediaStatus(content_id=record.url, adjusted_current_time=9999)
    cast_env.manager._save_position(record)
    assert watched() == {SOURCE_URL: (3600, 3600)}


def test_save_position_skips_live_and_ended_sessions(cast_env: CastEnv):
    start(cast_env)
    record = cast_env.manager._sessions[HOST]
    cast_env.receiver.media_controller.status.adjusted_current_time = 600
    cast_env.manager._save_position(record)
    transcode._transcode_sessions.clear()
    cast_env.manager._save_position(record)
    cast_env.manager._save_position(replace(record, cast=None))
    assert watched() == {}


def test_save_position_without_duration_keeps_position(cast_env: CastEnv):
    cast_env.session.update(is_vod=True, duration=0, seek_offset=0)
    start(cast_env)
    cast_env.receiver.media_controller.status.adjusted_current_time = 7200
    cast_env.manager._save_position(cast_env.manager._sessions[HOST])
    assert watched() == {SOURCE_URL: (7200, 0)}


def test_save_position_write_failure_is_logged(
    cast_env: CastEnv,
    caplog: pytest.LogCaptureFixture,
):
    cast_env.session["is_vod"] = True
    start(cast_env)
    record = cast_env.manager._sessions[HOST]
    (cache.USERS_DIR / "alice").write_text("")  # A file where the user directory belongs.
    cast_env.manager._save_position(record)
    assert "Could not save Chromecast watch position" in caplog.text


def test_worker_started_once_and_joined_on_close(
    clock: testing.FakeClock,
    threads: testing.ThreadRecorder,
):
    manager = chromecast.CastManager()
    manager._ensure_worker()
    manager._ensure_worker()
    [worker] = threads.threads
    assert worker.target == manager._monitor
    assert (worker.daemon, worker.name) == (True, "cast-monitor")
    manager.close()
    assert worker.joins == [6]
    assert manager._stop.is_set()


def test_finished_worker_is_restarted(
    clock: testing.FakeClock,
    threads: testing.ThreadRecorder,
):
    manager = chromecast.CastManager()
    manager._ensure_worker()
    manager._stop.set()
    threads.run_all()
    manager._ensure_worker()
    assert len(threads.threads) == 2
    assert not manager._stop.is_set()


def test_monitor_maintains_until_stopped(cast_env: CastEnv, monkeypatch: pytest.MonkeyPatch):
    start = testing.FakeClock().now
    clock = testing.FakeClock(now=start, deadline=start + 12)
    testing.install_clock(monkeypatch, clock, chromecast, transcode)
    manager = chromecast.CastManager()
    record = starting_record("session123")
    manager._sessions[record.host] = record
    manager._monitor()
    assert clock.sleeps == [5, 5, 2]
    # The second maintenance pass, at +10s, was the last heartbeat.
    assert cast_env.session.get("last_access") == start + 10


def test_server_keeps_cast_alive_without_browser(cast_env: CastEnv):
    start(cast_env)
    cast_env.receiver.media_controller.status.player_state = "PAUSED"
    cast_env.clock.advance(30)
    cast_env.manager.maintain_sessions()
    assert cast_env.session.get("last_access") == cast_env.clock.now
    assert cast_env.manager.status("alice")["active"]


def test_disconnected_receiver_releases_ownership(cast_env: CastEnv):
    start(cast_env)
    cast_env.receiver.socket_client.is_connected = False
    cast_env.manager.maintain_sessions()
    assert cast_env.manager.status("alice") == {"active": False}
    assert cast_env.receiver.disconnected


@pytest.mark.parametrize("state", ["IDLE", "UNKNOWN"])
def test_stopped_receiver_releases_ownership(cast_env: CastEnv, state: str):
    start(cast_env)
    cast_env.receiver.media_controller.status.player_state = state
    cast_env.manager.maintain_sessions()
    assert cast_env.manager.status("alice") == {"active": False}


def test_other_receiver_app_is_not_stopped(cast_env: CastEnv):
    start(cast_env)
    cast_env.receiver.media_controller.status = testing.FakeMediaStatus(
        content_id="another-app",
        player_state="PLAYING",
    )
    with pytest.raises(HTTPException) as error:
        cast_env.manager.command("alice", "stop", release=cast_env.release)
    assert error.value.status_code == 409
    assert "stop" not in cast_env.receiver.media_controller.commands
    assert cast_env.manager.status("alice") == {"active": False}


def test_remote_controls_and_stop_release_transcode(cast_env: CastEnv):
    start(cast_env)
    manager = cast_env.manager
    assert manager.command("alice", "pause", release=cast_env.release).get("state") == "PAUSED"
    assert manager.command("alice", "play", release=cast_env.release).get("state") == "PLAYING"
    assert manager.command("alice", "volume", 0.25, release=cast_env.release).get("volume") == 0.25
    assert manager.command("alice", "stop", release=cast_env.release) == {"active": False}
    assert cast_env.receiver.media_controller.commands == [
        "load",
        "pause",
        "play",
        "volume",
        "stop",
    ]
    assert cast_env.released == ["session123"]
    assert cast_env.receiver.disconnected


def test_close_disconnects_every_cast(cast_env: CastEnv):
    start(cast_env)
    cast_env.manager.close()
    assert cast_env.receiver.disconnected
    assert cast_env.manager._sessions == {}


def test_discovery_closes_browser_and_returns_lan_devices(
    monkeypatch: pytest.MonkeyPatch,
    clock: testing.FakeClock,
):
    network = testing.CastNetwork(
        advertised=[
            testing.FakeCastDevice(host="192.168.1.51", friendly_name="tv b"),
            testing.FakeCastDevice(host=HOST, friendly_name="TV"),
            testing.FakeCastDevice(host="192.168.1.52", friendly_name=None),
            testing.FakeCastDevice(host="127.0.0.1", friendly_name="Not a TV"),
        ],
    )
    testing.install_chromecast(monkeypatch, network, chromecast)
    manager = chromecast.CastManager()
    assert manager.discover() == [
        {"host": "192.168.1.52", "name": "192.168.1.52"},
        {"host": HOST, "name": "TV"},
        {"host": "192.168.1.51", "name": "tv b"},
    ]
    assert clock.sleeps == [5]
    assert not network.browsing


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
