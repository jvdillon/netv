"""Tests for handoff.py."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar
from uuid import UUID

import json
import uuid

import pytest

from netv import cache, handoff, testing


NOW = 1_800_000_000.0
CHANNEL_URL = "http://upstream.test/live/u/p/7.m3u8"
T = TypeVar("T")


@dataclass(kw_only=True, slots=True)
class FakeRecording:
    """A recording whose end the test decides."""

    done: bool = False
    stopped: bool = False

    def finished(self) -> bool:
        return self.done

    def stop(self) -> None:
        self.stopped = True


@dataclass(frozen=True, kw_only=True, slots=True)
class RecordCall:
    """One request to record a channel."""

    username: str
    channel: handoff.Channel
    directory: Path
    seconds: float
    directory_existed: bool


@dataclass(kw_only=True, slots=True)
class FakeRecorder:
    """Stands in for the live-session recorder ``handoff`` is given.

    Args:
        error: Raise this instead of recording, as when ffmpeg cannot start.
        during: Work another request does while the recording starts.
    """

    error: Exception | None = None
    during: Callable[[], object] | None = None
    calls: list[RecordCall] = field(default_factory=list)
    recordings: list[FakeRecording] = field(default_factory=list)

    def __call__(
        self,
        username: str,
        channel: handoff.Channel,
        directory: Path,
        seconds: float,
    ) -> FakeRecording:
        self.calls.append(
            RecordCall(
                username=username,
                channel=channel,
                directory=directory,
                seconds=seconds,
                directory_existed=directory.is_dir(),
            ),
        )
        if self.during is not None:
            self.during()
        if self.error is not None:
            raise self.error
        recording = FakeRecording()
        self.recordings.append(recording)
        return recording


@dataclass(frozen=True, kw_only=True, slots=True)
class Env:
    """Handoff state in a temp dir, recording through a fake recorder."""

    users: Path
    transcode_dir: Path
    recorder: FakeRecorder


class TestStop:
    def test_short_views_are_not_handed_off(self, env: Env):
        _watch(60)
        assert _stop(env, _channel(catchup_days=2)) is None
        assert handoff.current("ann", now=NOW) is None

    def test_stop_without_a_watch_is_ignored(self, env: Env):
        assert _stop(env, _channel(catchup_days=2)) is None

    def test_stop_of_another_stream_is_ignored(self, env: Env):
        _watch(600)
        assert handoff.stop_watch("ann", "8", NOW, lambda *_: _channel(), env.recorder) is None
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
        assert env.recorder.calls == []

    def test_channel_without_archive_records_until_program_end(self, env: Env):
        _watch(600)
        channel = _channel()
        record = _some(_stop(env, channel))
        recording_id = _str(record["recording_id"])
        assert env.recorder.calls == [
            RecordCall(
                username="ann",
                channel=channel,
                directory=env.transcode_dir / f"netv_handoff_{recording_id}",
                seconds=1800,
                directory_existed=True,
            ),
        ]
        assert record["mode"] == "recording"
        assert record["recording_live"] is True
        assert record["recording_start"] == NOW
        assert record["expires"] == NOW + 1800 + 24 * 3600
        assert handoff._recorders == {recording_id: env.recorder.recordings[0]}

    def test_recording_without_guide_end_is_capped(self, env: Env):
        _watch(600)
        _stop(env, _channel(program_end=0.0))
        assert env.recorder.calls[0].seconds == 3 * 3600

    def test_recording_of_a_long_program_is_capped(self, env: Env):
        _watch(600)
        _stop(env, _channel(program_end=NOW + 5 * 3600))
        assert env.recorder.calls[0].seconds == 3 * 3600

    def test_stopping_at_program_end_leaves_nothing(self, env: Env):
        _watch(600)
        assert _stop(env, _channel(catchup_days=2, program_end=NOW + 10)) is None

    def test_failed_recorder_leaves_nothing(self, env: Env, caplog: pytest.LogCaptureFixture):
        env.recorder.error = OSError("no ffmpeg")
        _watch(600)
        assert _stop(env, _channel()) is None
        assert handoff.current("ann", now=NOW) is None
        assert list(env.transcode_dir.glob("netv_handoff_*")) == []
        assert "Could not record handoff for ann: no ffmpeg" in caplog.text


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
        assert len(env.recorder.calls) == 1
        assert moved["id"] == _str(record["id"])
        assert moved["position"] == NOW + 100
        assert moved["updated"] == NOW + 110
        assert _some(handoff.current("ann", now=NOW))["position"] == NOW + 100

    def test_finishing_a_resumed_recording_ends_it(self, env: Env):
        record = self._recorded(env)
        handoff.claim("ann", _str(record["id"]))
        _watch(30, position=NOW + 1770, handoff_id=_str(record["id"]), now=NOW + 1800)
        assert _stop(env, _channel(), NOW + 1770, _str(record["id"]), now=NOW + 1800) is None
        assert env.recorder.recordings[0].stopped
        assert handoff._records == {}
        assert handoff._recorders == {}

    def test_resumed_recording_just_before_its_end_margin_is_kept(self, env: Env):
        record = self._recorded(env)
        handoff.claim("ann", _str(record["id"]))
        _watch(30, position=NOW + 1769, handoff_id=_str(record["id"]), now=NOW + 1800)
        moved = _some(_stop(env, _channel(), NOW + 1769, _str(record["id"]), now=NOW + 1800))
        assert moved["position"] == NOW + 1769
        assert not env.recorder.recordings[0].stopped

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
        assert directory.is_dir()
        handoff.cancel("ann")
        assert env.recorder.recordings[0].stopped
        assert handoff.current("ann", now=NOW) is None
        assert _str(record["recording_id"]) not in handoff._recorders
        assert not directory.exists()
        assert not (env.users / "ann" / "handoff.json").exists()

    def test_cancel_leaves_a_finished_recording_alone(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel()))
        env.recorder.recordings[0].done = True
        handoff.cancel("ann")
        assert not env.recorder.recordings[0].stopped
        assert not handoff.recording_dir(_str(record["recording_id"])).exists()

    def test_cancel_of_a_restored_recording_whose_files_are_gone(self, env: Env):
        # Recorders do not survive a restart, and the orphan sweep may delete the files.
        (env.users / "ann" / "handoff.json").write_text(
            json.dumps({"id": "h", "mode": "recording", "recording_id": "gone"}),
        )
        handoff.load(env.users)
        handoff.cancel("ann")
        assert handoff._records == {}
        assert not (env.users / "ann" / "handoff.json").exists()

    def test_cancel_keeps_the_handoff_being_resumed(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel(catchup_days=2)))
        handoff.cancel("ann", keep=_str(record["id"]))
        assert _some(handoff.current("ann", now=NOW))["id"] == _str(record["id"])

    def test_cancel_during_recorder_launch_discards_it(self, env: Env):
        _watch(600)
        # Another device starts playing while the recording starts.
        env.recorder.during = lambda: handoff.cancel("ann")
        assert _stop(env, _channel()) is None
        assert env.recorder.recordings[0].stopped
        assert list(env.transcode_dir.glob("netv_handoff_*")) == []
        assert handoff._records == {}
        assert handoff._recorders == {}


class TestSweep:
    def test_lapsed_watch_is_handed_off_at_its_last_position(self, env: Env):
        handoff.report_watch("ann", "7", NOW - 700, now=NOW - 700)
        handoff.report_watch("ann", "7", NOW - 100, now=NOW - 100)
        handoff.sweep(lambda *_: _channel(catchup_days=2), env.recorder, now=NOW)
        assert _some(handoff.current("ann", now=NOW))["position"] == NOW - 100
        assert handoff._watches == {}

    def test_lapsed_watch_of_a_live_only_channel_is_recorded(self, env: Env):
        handoff.report_watch("ann", "7", NOW - 700, now=NOW - 700)
        handoff.report_watch("ann", "7", NOW - 100, now=NOW - 100)
        handoff.sweep(lambda *_: _channel(), env.recorder, now=NOW)
        assert _some(handoff.current("ann", now=NOW))["mode"] == "recording"
        assert env.recorder.calls[0].seconds == 1800

    def test_fresh_watch_is_left_alone(self, env: Env):
        handoff.report_watch("ann", "7", NOW - 700, now=NOW - 700)
        handoff.report_watch("ann", "7", NOW - 10, now=NOW - 10)
        handoff.sweep(lambda *_: _channel(catchup_days=2), env.recorder, now=NOW)
        assert handoff.current("ann", now=NOW) is None
        assert "ann" in handoff._watches

    def test_watch_exactly_at_the_timeout_is_left_alone(self, env: Env):
        handoff.report_watch("ann", "7", NOW - 700, now=NOW - 700)
        handoff.report_watch("ann", "7", NOW - 45, now=NOW - 45)
        handoff.sweep(lambda *_: _channel(catchup_days=2), env.recorder, now=NOW)
        assert "ann" in handoff._watches

    def test_expired_handoff_is_dropped(self, env: Env):
        _watch(600)
        _stop(env, _channel(catchup_days=2))
        handoff.sweep(_unresolvable, env.recorder, now=NOW + 24 * 3600 + 1)
        assert handoff._records == {}

    def test_finished_recorder_is_marked_and_kept(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel()))
        directory = handoff.recording_dir(_str(record["recording_id"]))
        (directory / "stream.m3u8").write_text("#EXTM3U\n")
        env.recorder.recordings[0].done = True
        handoff.sweep(_unresolvable, env.recorder, now=NOW + 1)
        assert handoff._records["ann"]["recording_live"] is False
        assert handoff._recorders == {}
        saved = json.loads((env.users / "ann" / "handoff.json").read_text())
        assert saved["recording_live"] is False
        assert handoff.recording_file(_str(record["recording_id"]), "stream.m3u8")
        assert not env.recorder.recordings[0].stopped

    def test_running_recorder_is_left_alone(self, env: Env):
        _watch(600)
        record = _some(_stop(env, _channel()))
        handoff.sweep(_unresolvable, env.recorder, now=NOW + 1)
        assert handoff._records["ann"]["recording_live"] is True
        assert _str(record["recording_id"]) in handoff._recorders

    def test_recorder_that_produced_nothing_is_dropped(self, env: Env):
        _watch(600)
        _stop(env, _channel())
        env.recorder.recordings[0].done = True
        handoff.sweep(_unresolvable, env.recorder, now=NOW + 1)
        assert handoff._records == {}


class TestRecordingFiles:
    def test_only_known_recordings_are_served(self, env: Env):
        directory = handoff.recording_dir("abc123")
        directory.mkdir()
        (directory / "stream.m3u8").write_text("#EXTM3U\n")
        assert handoff.recording_file("abc123", "stream.m3u8") is None
        handoff._records["ann"] = {"id": "h", "recording_id": "abc123"}
        # With its owner, whose LAN rule the server applies to the file.
        assert handoff.recording_file("abc123", "stream.m3u8") == (directory / "stream.m3u8", "ann")
        assert handoff.recording_file("abc123", "missing.ts") is None
        assert handoff.recording_file("abc123", "../stream.m3u8") is None
        assert handoff.recording_file("../x", "stream.m3u8") is None
        # A file beside the recording is out of reach, though it exists.
        (directory.parent / "secret.ts").write_bytes(b"x")
        assert handoff.recording_file("abc123", "../secret.ts") is None

    def test_orphaned_recordings_are_deleted(self, env: Env):
        kept = handoff.recording_dir("kept")
        orphan = handoff.recording_dir("orphan")
        kept.mkdir()
        orphan.mkdir()
        handoff._records["ann"] = {"id": "h", "recording_id": "kept"}
        handoff.cleanup_orphans()
        assert kept.exists()
        assert not orphan.exists()


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
        {**cache.load_server_settings(), "transcode_dir": str(transcode_dir)},
    )
    for name in ("_records", "_watches", "_recorders", "_generations"):
        monkeypatch.setattr(handoff, name, {})
    monkeypatch.setattr(handoff, "_users_dir", None)
    handoff.load(users)
    return Env(users=users, transcode_dir=transcode_dir, recorder=FakeRecorder())


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
        env.recorder,
        handoff_id,
        now=now,
    )


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
