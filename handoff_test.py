"""Tests for handoff.py."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar, cast
from unittest.mock import MagicMock, patch

import json

import pytest

import handoff


NOW = 1_800_000_000.0
T = TypeVar("T")


class TestStop:
    def test_short_views_are_not_handed_off(self):
        _watch(60)
        assert _stop(_channel(catchup_days=2)) is None
        assert handoff.current("ann", now=NOW) is None

    def test_stop_without_a_watch_is_ignored(self):
        assert _stop(_channel(catchup_days=2)) is None

    def test_archived_channel_resumes_as_catchup(self, state: Path):
        _watch(600)
        record = _some(_stop(_channel(catchup_days=2)))
        assert record["mode"] == "catchup"
        assert record["position"] == NOW - 5
        assert record["program_title"] == "Evening News"
        assert _some(handoff.current("ann", now=NOW))["id"] == record["id"]
        saved = json.loads((state / "ann" / "handoff.json").read_text())
        assert saved["id"] == record["id"]

    def test_channel_without_archive_records_until_program_end(self):
        release = MagicMock()
        _watch(600)
        with patch("handoff._start_recorder", return_value=_process()) as start:
            record = _some(_stop(_channel(), release=release))
        release.assert_called_once_with("ann", "http://upstream.test/live/u/p/7.m3u8")
        url, recording_id, seconds = start.call_args.args
        assert seconds == 1800
        assert record["mode"] == "recording"
        assert record["recording_id"] == recording_id
        assert record["recording_start"] == NOW
        assert record["expires"] == NOW + 1800 + 24 * 3600

    def test_recording_without_guide_end_is_capped(self):
        _watch(600)
        with patch("handoff._start_recorder", return_value=_process()) as start:
            _stop(_channel(program_end=0.0))
        assert start.call_args.args[2] == 3 * 3600

    def test_stopping_at_program_end_leaves_nothing(self):
        _watch(600)
        assert _stop(_channel(catchup_days=2, program_end=NOW + 10)) is None

    def test_failed_recorder_leaves_nothing(self):
        _watch(600)
        with patch("handoff._start_recorder", side_effect=OSError("no ffmpeg")):
            assert _stop(_channel()) is None
        assert handoff.current("ann", now=NOW) is None


class TestResume:
    def _recorded(self):
        _watch(600)
        with patch("handoff._start_recorder", return_value=_process()):
            return _some(_stop(_channel()))

    def test_claim_hides_the_card_but_keeps_the_recording(self):
        record = self._recorded()
        assert _some(handoff.claim("ann", record["id"]))["recording_id"] == record["recording_id"]
        assert handoff.current("ann", now=NOW) is None
        assert record["recording_id"] in handoff._recorders
        assert handoff.claim("ann", "other") is None

    def test_stopping_a_resumed_recording_moves_the_bookmark(self):
        record = self._recorded()
        handoff.claim("ann", record["id"])
        _watch(30, position=NOW + 100, handoff_id=record["id"], now=NOW + 110)
        with patch("handoff._start_recorder") as start:
            moved = _some(_stop(_channel(), NOW + 100, record["id"], now=NOW + 110))
        start.assert_not_called()
        assert moved["id"] == record["id"]
        assert moved["position"] == NOW + 100
        assert _some(handoff.current("ann", now=NOW))["position"] == NOW + 100

    def test_finishing_a_resumed_recording_ends_it(self):
        record = self._recorded()
        handoff.claim("ann", record["id"])
        _watch(30, position=NOW + 1790, handoff_id=record["id"], now=NOW + 1800)
        with patch("handoff._stop_recorder") as stop:
            assert _stop(_channel(), NOW + 1790, record["id"], now=NOW + 1800) is None
        stop.assert_called_once_with(record["recording_id"])
        assert handoff._records == {}

    def test_catchup_resume_rolling_into_the_next_program_hands_off_again(self):
        _watch(600)
        record = _some(_stop(_channel(catchup_days=2)))
        handoff.claim("ann", record["id"])
        later = NOW + 3000
        _watch(30, position=later - 10, handoff_id=record["id"], now=later)
        next_program = _channel(
            catchup_days=2,
            program_title="Late News",
            program_start=NOW + 1800,
            program_end=NOW + 5400,
        )
        moved = _some(_stop(next_program, later - 10, record["id"], now=later))
        assert moved["mode"] == "catchup"
        assert moved["program_title"] == "Late News"
        assert moved["position"] == later - 10


class TestCancel:
    def test_cancel_stops_the_recording(self):
        _watch(600)
        process = _process()
        with patch("handoff._start_recorder", return_value=process):
            record = _some(_stop(_channel()))
        handoff.cancel("ann")
        process.terminate.assert_called_once()
        assert handoff.current("ann", now=NOW) is None
        assert record["recording_id"] not in handoff._recorders

    def test_cancel_keeps_the_handoff_being_resumed(self):
        _watch(600)
        record = _some(_stop(_channel(catchup_days=2)))
        handoff.cancel("ann", keep=record["id"])
        assert _some(handoff.current("ann", now=NOW))["id"] == record["id"]

    def test_cancel_during_recorder_launch_discards_it(self):
        _watch(600)
        process = _process()

        def launch(*_args):
            handoff.cancel("ann")  # Another device starts playing meanwhile.
            return process

        with patch("handoff._start_recorder", side_effect=launch):
            assert _stop(_channel()) is None
        process.terminate.assert_called_once()
        assert handoff._records == {}
        assert handoff._recorders == {}


class TestSweep:
    def test_lapsed_watch_is_handed_off_at_its_last_position(self):
        handoff.report_watch("ann", "7", NOW - 700, now=NOW - 700)
        handoff.report_watch("ann", "7", NOW - 100, now=NOW - 100)
        handoff.sweep(lambda *_: _channel(catchup_days=2), MagicMock(), now=NOW)
        assert _some(handoff.current("ann", now=NOW))["position"] == NOW - 100
        assert handoff._watches == {}

    def test_fresh_watch_is_left_alone(self):
        handoff.report_watch("ann", "7", NOW - 700, now=NOW - 700)
        handoff.report_watch("ann", "7", NOW - 10, now=NOW - 10)
        handoff.sweep(lambda *_: _channel(catchup_days=2), MagicMock(), now=NOW)
        assert handoff.current("ann", now=NOW) is None
        assert "ann" in handoff._watches

    def test_expired_handoff_is_dropped(self):
        _watch(600)
        _stop(_channel(catchup_days=2))
        handoff.sweep(MagicMock(), MagicMock(), now=NOW + 24 * 3600 + 1)
        assert handoff._records == {}

    def test_finished_recorder_is_marked_and_kept(self, tmp_path: Path):
        _watch(600)
        with patch("handoff._start_recorder", return_value=_process()):
            record = _some(_stop(_channel()))
        directory = handoff.recording_dir(record["recording_id"])
        directory.mkdir()
        (directory / "stream.m3u8").write_text("#EXTM3U\n")
        cast(MagicMock, handoff._recorders[record["recording_id"]]).poll.return_value = 0
        handoff.sweep(MagicMock(), MagicMock(), now=NOW + 1)
        assert handoff._records["ann"]["recording_live"] is False
        assert handoff.recording_file(record["recording_id"], "stream.m3u8")

    def test_recorder_that_produced_nothing_is_dropped(self):
        _watch(600)
        with patch("handoff._start_recorder", return_value=_process()):
            _stop(_channel())
        cast(MagicMock, next(iter(handoff._recorders.values()))).poll.return_value = 1
        handoff.sweep(MagicMock(), MagicMock(), now=NOW + 1)
        assert handoff._records == {}


class TestRecordingFiles:
    def test_only_known_recordings_are_served(self, tmp_path: Path):
        directory = handoff.recording_dir("abc123")
        directory.mkdir()
        (directory / "stream.m3u8").write_text("#EXTM3U\n")
        assert handoff.recording_file("abc123", "stream.m3u8") is None
        handoff._records["ann"] = {"id": "h", "recording_id": "abc123"}
        assert handoff.recording_file("abc123", "stream.m3u8") == directory / "stream.m3u8"
        assert handoff.recording_file("abc123", "../stream.m3u8") is None
        assert handoff.recording_file("../x", "stream.m3u8") is None

    def test_orphaned_recordings_are_deleted(self, tmp_path: Path):
        kept = handoff.recording_dir("kept")
        orphan = handoff.recording_dir("orphan")
        kept.mkdir()
        orphan.mkdir()
        handoff._records["ann"] = {"id": "h", "recording_id": "kept"}
        handoff.cleanup_orphans()
        assert kept.exists()
        assert not orphan.exists()

    def test_recorder_stops_at_the_requested_time(self, tmp_path: Path):
        with (
            patch("ffmpeg_command.get_settings", return_value={"probe_live": False}),
            patch("ffmpeg_command.resolve_hls_master_playlist", side_effect=_same_url),
            patch("ffmpeg_command.get_user_agent", return_value=None),
        ):
            cmd = handoff._recorder_command("http://upstream.test/7.m3u8", tmp_path, 1800.4)
        i = cmd.index("-i")
        assert cmd[i - 2 : i + 2] == ["-t", "1800", "-i", "http://upstream.test/7.m3u8"]
        assert "event" in cmd
        assert cmd[-1] == f"{tmp_path}/stream.m3u8"


class TestLoad:
    def test_load_restores_valid_records_only(self, state: Path):
        (state / "ann" / "handoff.json").write_text(
            json.dumps({"id": "h1", "recording_live": True}),
        )
        for name, text in [("bob", "not json"), ("cat", "[1]"), ("dan", '{"x": 1}')]:
            (state / name).mkdir()
            (state / name / "handoff.json").write_text(text)
        (state / "eve").mkdir()
        (state / "eve" / "handoff.json").mkdir()  # Reading a directory raises OSError.
        handoff.load(state)
        assert handoff._records == {"ann": {"id": "h1", "recording_live": False}}


class TestShutdown:
    def test_terminates_only_running_recorders(self):
        running, done = _process(), _process(running=False)
        handoff._recorders.update(a=running, b=done)
        handoff.shutdown()
        running.terminate.assert_called_once()
        done.terminate.assert_not_called()
        assert handoff._recorders == {}


class TestSave:
    def test_without_users_dir_is_noop(self, state: Path):
        handoff._users_dir = None
        handoff._records["ann"] = {"id": "h"}
        handoff._save("ann")
        assert not (state / "ann" / "handoff.json").exists()

    def test_write_failure_is_logged(self, state: Path, caplog: pytest.LogCaptureFixture):
        handoff._records["zed"] = {"id": "h"}  # No user directory to write into.
        with patch("handoff.atomic_write_json", side_effect=OSError("disk full")):
            handoff._save("zed")
        assert "Could not save handoff for zed" in caplog.text


class TestRecorderProcess:
    def test_start_recorder_launches_ffmpeg_with_log(self, tmp_path: Path):
        with (
            patch("handoff._recorder_command", return_value=["ffmpeg", "-x"]) as command,
            patch("subprocess.Popen") as popen,
        ):
            assert handoff._start_recorder("http://u.test/7", "rid", 60) is popen.return_value
        directory = tmp_path / "netv_handoff_rid"
        command.assert_called_once_with("http://u.test/7", directory, 60)
        assert popen.call_args.args == (["ffmpeg", "-x"],)
        assert (directory / "ffmpeg.log").exists()

    def test_stop_recorder_kills_unresponsive_process(self, tmp_path: Path):
        directory = handoff.recording_dir("rid")
        directory.mkdir()
        process = _process()
        process.wait.side_effect = handoff.subprocess.TimeoutExpired("ffmpeg", 5)
        handoff._recorders["rid"] = process
        handoff._stop_recorder("rid")
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        assert not directory.exists()


class TestHandOffRaces:
    def test_unresolvable_channel_cancels(self):
        handoff._records["ann"] = {"id": "old", "mode": "catchup", "expires": NOW + 100}
        _watch(600)
        assert _stop(None) is None
        assert handoff._records == {}

    def test_cancel_while_building_record_discards_it(self):
        _watch(600)
        real_uuid4 = handoff.uuid.uuid4

        def uuid4():
            handoff.cancel("ann")  # Another device starts playing meanwhile.
            return real_uuid4()

        with patch("handoff.uuid.uuid4", side_effect=uuid4):
            assert _stop(_channel(catchup_days=2)) is None
        assert handoff._records == {}


def _some(value: T | None) -> T:
    assert value is not None
    return value


@pytest.fixture(autouse=True)
def state(tmp_path: Path) -> Iterator[Path]:
    users = tmp_path / "users"
    (users / "ann").mkdir(parents=True)
    handoff.load(users)
    handoff._watches.clear()
    handoff._recorders.clear()
    handoff._generations.clear()
    with patch("ffmpeg_command.get_transcode_dir", return_value=tmp_path):
        yield users
    handoff._users_dir = None
    handoff._records.clear()
    handoff._watches.clear()
    handoff._recorders.clear()


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
        url="http://upstream.test/live/u/p/7.m3u8",
        catchup_days=catchup_days,
        program_title=program_title,
        program_start=program_start,
        program_end=program_end,
    )


def _process(running: bool = True) -> MagicMock:
    process = MagicMock()
    process.poll.return_value = None if running else 0
    return process


def _watch(
    seconds: float,
    position: float = NOW - 5,
    handoff_id: str = "",
    now: float = NOW,
) -> None:
    handoff.report_watch("ann", "7", now - seconds, handoff_id, now=now - seconds)
    handoff.report_watch("ann", "7", position, handoff_id, now=now)


def _stop(
    channel: handoff.Channel | None,
    position: float = NOW - 5,
    handoff_id: str = "",
    release: handoff.Releaser | None = None,
    now: float = NOW,
):
    return handoff.stop_watch(
        "ann",
        "7",
        position,
        lambda _stream_id, _position: channel,
        release or MagicMock(),
        handoff_id,
        now=now,
    )


def _same_url(url: str) -> str:
    return url


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
