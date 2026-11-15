"""Tests for handoff.py."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar
from uuid import UUID

import json
import subprocess
import uuid

import pytest

from netv import cache, ffmpeg_command, handoff, testing


NOW = 1_800_000_000.0
CHANNEL_URL = "http://upstream.test/live/u/p/7.m3u8"
T = TypeVar("T")


@dataclass(frozen=True, kw_only=True, slots=True)
class Env:
    """Handoff state in a temp dir, with fake ffmpeg launches and upstream."""

    users: Path
    transcode_dir: Path
    popen: testing.PopenRecorder
    web: testing.FakeWeb
    released: list[tuple[str, str]] = field(default_factory=list)

    def release(self, username: str, url: str) -> None:
        self.released.append((username, url))

    @property
    def recorders(self) -> list[testing.FakePopen]:
        return self.popen.started


class TestStop:
    def test_short_views_are_not_handed_off(self, env: Env):
        _watch(60)
        assert _stop(env, _channel(catchup_days=2)) is None
        assert handoff.current("ann", now=NOW) is None

    def test_stop_without_a_watch_is_ignored(self, env: Env):
        assert _stop(env, _channel(catchup_days=2)) is None

    def test_stop_of_another_stream_is_ignored(self, env: Env):
        _watch(600)
        assert handoff.stop_watch("ann", "8", NOW, lambda *_: _channel(), env.release) is None
        assert "ann" in handoff._watches

    def test_archived_channel_resumes_as_catchup(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel(catchup_days=2)))
        assert record["mode"] == "catchup"
        assert record["position"] == NOW - 5
        assert record["program_title"] == "Evening News"
        assert record["expires"] == NOW + 24 * 3600
        assert _some(handoff.current("ann", now=NOW))["id"] == _str(record["id"])
        saved = json.loads((env.users / "ann" / "handoff.json").read_text())
        assert saved["id"] == _str(record["id"])
        assert env.recorders == []
        assert env.released == []

    def test_channel_without_archive_records_until_program_end(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel()))
        assert env.released == [("ann", CHANNEL_URL)]
        [recorder] = env.recorders
        directory = env.transcode_dir / f"netv_handoff_{record['recording_id']}"
        i = recorder.args.index("-i")
        assert recorder.args[i - 2 : i + 2] == ["-t", "1800", "-i", CHANNEL_URL]
        assert recorder.args[-1] == f"{directory}/stream.m3u8"
        assert record["mode"] == "recording"
        assert record["recording_live"] is True
        assert record["recording_start"] == NOW
        assert record["expires"] == NOW + 1800 + 24 * 3600
        assert handoff._recorders == {_str(record["recording_id"]): recorder}

    def test_recording_without_guide_end_is_capped(self, env: Env):
        _watch(600)
        _stop(env, _channel(program_end=0.0))
        assert _seconds(env.recorders[0]) == 3 * 3600

    def test_recording_of_a_long_program_is_capped(self, env: Env):
        _watch(600)
        _stop(env, _channel(program_end=NOW + 5 * 3600))
        assert _seconds(env.recorders[0]) == 3 * 3600

    def test_stopping_at_program_end_leaves_nothing(self, env: Env):
        _watch(600)
        assert _stop(env, _channel(catchup_days=2, program_end=NOW + 10)) is None

    def test_failed_recorder_leaves_nothing(self, env: Env, monkeypatch: pytest.MonkeyPatch):
        missing = testing.PopenRecorder(error=FileNotFoundError(2, "No such file", "ffmpeg"))
        testing.install_subprocess(monkeypatch, handoff, popen=missing)
        _watch(600)
        assert _stop(env, _channel()) is None
        assert handoff.current("ann", now=NOW) is None
        assert list(env.transcode_dir.glob("netv_handoff_*")) == []


class TestResume:
    def _recorded(self, env: Env) -> Mapping[str, object]:
        _watch(600)
        return _some(_stop(env, _channel()))

    def test_claim_hides_the_card_but_keeps_the_recording(self, env: Env):
        record = self._recorded(env)
        assert _some(handoff.claim("ann", _str(record["id"])))["recording_id"] == _str(
            record["recording_id"],
        )
        assert _some(handoff.claim("ann", _str(record["id"])))["resumed"] is True
        assert handoff.current("ann", now=NOW) is None
        assert _str(record["recording_id"]) in handoff._recorders
        assert handoff.claim("ann", "other") is None
        assert handoff.claim("bob", _str(record["id"])) is None

    def test_stopping_a_resumed_recording_moves_the_bookmark(self, env: Env):
        record = self._recorded(env)
        handoff.claim("ann", _str(record["id"]))
        _watch(30, position=NOW + 100, handoff_id=_str(record["id"]), now=NOW + 110)
        moved = _some(_stop(env, _channel(), NOW + 100, _str(record["id"]), now=NOW + 110))
        assert len(env.recorders) == 1
        assert moved["id"] == _str(record["id"])
        assert moved["position"] == NOW + 100
        assert _some(handoff.current("ann", now=NOW))["position"] == NOW + 100

    def test_finishing_a_resumed_recording_ends_it(self, env: Env):
        record = self._recorded(env)
        handoff.claim("ann", _str(record["id"]))
        _watch(30, position=NOW + 1790, handoff_id=_str(record["id"]), now=NOW + 1800)
        assert _stop(env, _channel(), NOW + 1790, _str(record["id"]), now=NOW + 1800) is None
        assert env.recorders[0].signals == ["TERM"]
        assert handoff._records == {}
        assert handoff._recorders == {}

    def test_catchup_resume_rolling_into_the_next_program_hands_off_again(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel(catchup_days=2)))
        handoff.claim("ann", _str(record["id"]))
        later = NOW + 3000
        _watch(30, position=later - 10, handoff_id=_str(record["id"]), now=later)
        next_program = _channel(
            catchup_days=2,
            program_title="Late News",
            program_start=NOW + 1800,
            program_end=NOW + 5400,
        )
        moved = _some(_stop(env, next_program, later - 10, _str(record["id"]), now=later))
        assert moved["mode"] == "catchup"
        assert moved["program_title"] == "Late News"
        assert moved["position"] == later - 10

    def test_position_ahead_of_now_is_clamped(self, env: Env):
        _watch(600, position=NOW + 60)
        assert _some(_stop(env, _channel(catchup_days=2), NOW + 60))["position"] == NOW


class TestCancel:
    def test_cancel_stops_the_recording(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel()))
        directory = handoff.recording_dir(_str(record["recording_id"]))
        assert (directory / "ffmpeg.log").exists()
        handoff.cancel("ann")
        assert env.recorders[0].signals == ["TERM"]
        assert handoff.current("ann", now=NOW) is None
        assert _str(record["recording_id"]) not in handoff._recorders
        assert not directory.exists()
        assert not (env.users / "ann" / "handoff.json").exists()

    def test_cancel_keeps_the_handoff_being_resumed(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel(catchup_days=2)))
        handoff.cancel("ann", keep=_str(record["id"]))
        assert _some(handoff.current("ann", now=NOW))["id"] == _str(record["id"])

    def test_cancel_during_recorder_launch_discards_it(self, env: Env):
        _watch(600)

        def release(username: str, url: str) -> None:
            del url
            handoff.cancel(username)  # Another device starts playing meanwhile.

        assert (
            handoff.stop_watch("ann", "7", NOW - 5, lambda *_: _channel(), release, now=NOW) is None
        )
        assert env.recorders[0].signals == ["TERM"]
        assert list(env.transcode_dir.glob("netv_handoff_*")) == []
        assert handoff._records == {}
        assert handoff._recorders == {}


class TestSweep:
    def test_lapsed_watch_is_handed_off_at_its_last_position(self, env: Env):
        handoff.report_watch("ann", "7", NOW - 700, now=NOW - 700)
        handoff.report_watch("ann", "7", NOW - 100, now=NOW - 100)
        handoff.sweep(lambda *_: _channel(catchup_days=2), env.release, now=NOW)
        assert _some(handoff.current("ann", now=NOW))["position"] == NOW - 100
        assert handoff._watches == {}

    def test_fresh_watch_is_left_alone(self, env: Env):
        handoff.report_watch("ann", "7", NOW - 700, now=NOW - 700)
        handoff.report_watch("ann", "7", NOW - 10, now=NOW - 10)
        handoff.sweep(lambda *_: _channel(catchup_days=2), env.release, now=NOW)
        assert handoff.current("ann", now=NOW) is None
        assert "ann" in handoff._watches

    def test_expired_handoff_is_dropped(self, env: Env):
        _watch(600)
        _stop(env, _channel(catchup_days=2))
        handoff.sweep(_unresolvable, env.release, now=NOW + 24 * 3600 + 1)
        assert handoff._records == {}

    def test_finished_recorder_is_marked_and_kept(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel()))
        directory = handoff.recording_dir(_str(record["recording_id"]))
        (directory / "stream.m3u8").write_text("#EXTM3U\n")
        env.recorders[0].finish(0)
        handoff.sweep(_unresolvable, env.release, now=NOW + 1)
        assert handoff._records["ann"]["recording_live"] is False
        assert handoff._recorders == {}
        saved = json.loads((env.users / "ann" / "handoff.json").read_text())
        assert saved["recording_live"] is False
        assert handoff.recording_file(_str(record["recording_id"]), "stream.m3u8")

    def test_running_recorder_is_left_alone(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel()))
        handoff.sweep(_unresolvable, env.release, now=NOW + 1)
        assert handoff._records["ann"]["recording_live"] is True
        assert _str(record["recording_id"]) in handoff._recorders

    def test_recorder_that_produced_nothing_is_dropped(self, env: Env):
        _watch(600)
        _stop(env, _channel())
        env.recorders[0].finish(1)
        handoff.sweep(_unresolvable, env.release, now=NOW + 1)
        assert handoff._records == {}


class TestRecordingFiles:
    def test_only_known_recordings_are_served(self, env: Env):
        directory = handoff.recording_dir("abc123")
        directory.mkdir()
        (directory / "stream.m3u8").write_text("#EXTM3U\n")
        assert handoff.recording_file("abc123", "stream.m3u8") is None
        handoff._records["ann"] = {"id": "h", "recording_id": "abc123"}
        assert handoff.recording_file("abc123", "stream.m3u8") == directory / "stream.m3u8"
        assert handoff.recording_file("abc123", "missing.ts") is None
        assert handoff.recording_file("abc123", "../stream.m3u8") is None
        assert handoff.recording_file("../x", "stream.m3u8") is None

    def test_orphaned_recordings_are_deleted(self, env: Env):
        kept = handoff.recording_dir("kept")
        orphan = handoff.recording_dir("orphan")
        kept.mkdir()
        orphan.mkdir()
        handoff._records["ann"] = {"id": "h", "recording_id": "kept"}
        handoff.cleanup_orphans()
        assert kept.exists()
        assert not orphan.exists()

    def test_recorder_records_the_best_variant_until_the_requested_time(
        self,
        env: Env,
        tmp_path: Path,
    ):
        master = "http://upstream.test/7.m3u8"
        env.web.routes[master] = testing.FakeResponse(
            b"#EXTM3U\n"
            b"#EXT-X-STREAM-INF:BANDWIDTH=800000\nlow.m3u8\n"
            b"#EXT-X-STREAM-INF:BANDWIDTH=5000000\nhigh.m3u8\n",
        )
        cmd = handoff._recorder_command(master, tmp_path, 1800.4)
        i = cmd.index("-i")
        assert cmd[i - 2 : i + 2] == ["-t", "1800", "-i", "http://upstream.test/high.m3u8"]
        assert "event" in cmd
        assert cmd[-1] == f"{tmp_path}/stream.m3u8"
        assert env.web.urls == [master]


class TestLoad:
    def test_load_restores_valid_records_only(self, env: Env):
        (env.users / "ann" / "handoff.json").write_text(
            json.dumps({"id": "h1", "recording_live": True}),
        )
        for name, text in [("bob", "not json"), ("cat", "[1]"), ("dan", '{"x": 1}')]:
            (env.users / name).mkdir()
            (env.users / name / "handoff.json").write_text(text)
        (env.users / "eve").mkdir()
        (env.users / "eve" / "handoff.json").mkdir()  # Reading a directory raises OSError.
        handoff.load(env.users)
        assert handoff._records == {"ann": {"id": "h1", "recording_live": False}}


class TestShutdown:
    def test_terminates_only_running_recorders(self, env: Env):
        handoff._recorders.update(
            a=handoff._start_recorder(CHANNEL_URL, "a", 60),
            b=handoff._start_recorder(CHANNEL_URL, "b", 60),
        )
        running, done = env.recorders
        done.finish(0)
        handoff.shutdown()
        assert running.signals == ["TERM"]
        assert done.signals == []
        assert handoff._recorders == {}


class TestSave:
    def test_without_users_dir_is_noop(self, env: Env):
        handoff._users_dir = None
        handoff._records["ann"] = {"id": "h"}
        handoff._save("ann")
        assert not (env.users / "ann" / "handoff.json").exists()

    def test_write_failure_is_logged(self, env: Env, caplog: pytest.LogCaptureFixture):
        (env.users / "zed").write_text("")  # A file where the user directory belongs.
        handoff._records["zed"] = {"id": "h"}
        handoff._save("zed")
        assert "Could not save handoff for zed" in caplog.text


class TestRecorderProcess:
    def test_start_recorder_launches_ffmpeg_with_log(self, env: Env):
        started = handoff._start_recorder("http://u.test/7", "rid", 60)
        directory = env.transcode_dir / "netv_handoff_rid"
        [process] = env.recorders
        assert process is started
        i = process.args.index("-i")
        assert process.args[0] == "ffmpeg"
        assert process.args[i - 2 : i + 2] == ["-t", "60", "-i", "http://u.test/7"]
        assert process.args[-1] == f"{directory}/stream.m3u8"
        assert process.options["stdin"] == subprocess.DEVNULL
        assert process.options["stdout"] == subprocess.DEVNULL
        assert (directory / "ffmpeg.log").exists()

    def test_stop_recorder_kills_unresponsive_process(self, env: Env):
        handoff._recorders["rid"] = handoff._start_recorder(CHANNEL_URL, "rid", 60)
        [process] = env.recorders
        process.ignores_terminate = True
        handoff._stop_recorder("rid")
        assert process.signals == ["TERM", "KILL"]
        assert not handoff.recording_dir("rid").exists()

    def test_stop_recorder_leaves_finished_process_alone(self, env: Env):
        handoff._recorders["rid"] = handoff._start_recorder(CHANNEL_URL, "rid", 60)
        [process] = env.recorders
        process.finish(0)
        handoff._stop_recorder("rid")
        assert process.signals == []
        assert handoff._recorders == {}


class TestHandOffRaces:
    def test_unresolvable_channel_cancels(self, env: Env):
        handoff._records["ann"] = {"id": "old", "mode": "catchup", "expires": NOW + 100}
        _watch(600)
        assert _stop(env, None) is None
        assert handoff._records == {}

    def test_cancel_while_building_record_discards_it(
        self,
        env: Env,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _watch(600)

        def uuid4() -> UUID:
            handoff.cancel("ann")  # Another device starts playing meanwhile.
            return uuid.UUID(int=1)

        monkeypatch.setattr(handoff, "uuid", testing.ModuleView(uuid, {"uuid4": uuid4}))
        assert _stop(env, _channel(catchup_days=2)) is None
        assert handoff._records == {}


def _some(value: T | None) -> T:
    assert value is not None
    return value


def _str(value: object) -> str:
    assert isinstance(value, str)
    return value


@pytest.fixture(autouse=True)
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    users = tmp_path / "users"
    (users / "ann").mkdir(parents=True)
    transcode_dir = tmp_path / "transcode"
    cache.save_server_settings(
        {**cache.load_server_settings(), "transcode_dir": str(transcode_dir), "probe_live": False},
    )
    popen = testing.PopenRecorder()
    testing.install_subprocess(monkeypatch, handoff, popen=popen)
    web = testing.FakeWeb({CHANNEL_URL: lambda: testing.FakeResponse(b"#EXTM3U\n")})
    testing.install_web(monkeypatch, web, ffmpeg_command)
    for name in ("_records", "_watches", "_recorders", "_generations"):
        monkeypatch.setattr(handoff, name, {})
    monkeypatch.setattr(handoff, "_users_dir", None)
    handoff.load(users)
    return Env(users=users, transcode_dir=transcode_dir, popen=popen, web=web)


def _channel(
    catchup_days: int = 0,
    *,
    program_title: str = "Evening News",
    program_start: float = NOW - 1800,
    program_end: float = NOW + 1800,
) -> handoff.Channel:
    return handoff.Channel(
        stream_id="7",
        name="News",
        url=CHANNEL_URL,
        catchup_days=catchup_days,
        program_title=program_title,
        program_start=program_start,
        program_end=program_end,
    )


def _unresolvable(stream_id: str, position: float) -> handoff.Channel | None:
    del stream_id, position


def _seconds(recorder: testing.FakePopen) -> int:
    return int(recorder.args[recorder.args.index("-t") + 1])


def _watch(
    seconds: float,
    position: float = NOW - 5,
    handoff_id: str = "",
    now: float = NOW,
) -> None:
    handoff.report_watch("ann", "7", now - seconds, handoff_id, now=now - seconds)
    handoff.report_watch("ann", "7", position, handoff_id, now=now)


def _stop(
    env: Env,
    channel: handoff.Channel | None,
    position: float = NOW - 5,
    handoff_id: str = "",
    now: float = NOW,
) -> Mapping[str, object] | None:
    return handoff.stop_watch(
        "ann",
        "7",
        position,
        lambda _stream_id, _position: channel,
        env.release,
        handoff_id,
        now=now,
    )


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
