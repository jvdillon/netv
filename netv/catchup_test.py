"""Tests for catchup.py."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from treekle import MutablePlainTree

import pytest

from netv import catchup
from netv.xtream import ProviderRecord


class TestArchiveDays:
    def test_reads_archive_length(self):
        assert catchup.archive_days(_stream()) == 3
        assert catchup.archive_days(_stream(tv_archive="1", tv_archive_duration=1)) == 1

    @pytest.mark.parametrize(
        "overrides",
        [
            {"tv_archive": 0},
            {"tv_archive": None},
            {"source_type": "m3u"},
            {"tv_archive_duration": "bad"},
            {"tv_archive_duration": None},
            {"tv_archive_duration": -2},
            {"tv_archive_duration": ""},
            {"tv_archive_duration": [3]},
        ],
    )
    def test_no_archive(self, overrides: ProviderRecord):
        assert catchup.archive_days(_stream(**overrides)) == 0

    def test_clamps_long_archives(self):
        assert catchup.archive_days(_stream(tv_archive_duration=90)) == 14


class TestCanPlay:
    now = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)

    def test_inside_window(self):
        assert catchup.can_play(_stream(), self.now - timedelta(hours=5), self.now)
        assert catchup.can_play(_stream(), self.now - timedelta(days=3), self.now)

    def test_outside_window(self):
        stream = _stream()
        assert not catchup.can_play(stream, self.now - timedelta(days=3, seconds=1), self.now)
        assert not catchup.can_play(stream, self.now, self.now)
        assert not catchup.can_play(stream, self.now + timedelta(hours=1), self.now)

    def test_without_archive(self):
        assert not catchup.can_play(_stream(tv_archive=0), self.now - timedelta(hours=1), self.now)

    def test_defaults_to_current_time(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        assert catchup.can_play(_stream(), recent)
        assert not catchup.can_play(_stream(), recent + timedelta(hours=2))


def test_history_covers_longest_archive():
    assert catchup.history([]) == timedelta(hours=24)
    assert catchup.history([_stream(tv_archive=0)]) == timedelta(hours=24)
    assert catchup.history([_stream(tv_archive_duration=1), _stream()]) == timedelta(days=3)


def test_duration_minutes_rounds_up_and_clamps():
    start = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
    assert catchup.duration_minutes(start, start + timedelta(minutes=29, seconds=30)) == 30
    assert catchup.duration_minutes(start, start) == 1
    assert catchup.duration_minutes(start, start + timedelta(days=1)) == 12 * 60


class TestTimeshiftUrl:
    def test_uses_upstream_time_zone(self):
        start = datetime(2026, 7, 1, 18, 30, tzinfo=timezone.utc)
        url = catchup.timeshift_url(_stream(), start, 60, ZoneInfo("Europe/Amsterdam"))
        assert url == "http://upstream.test:8080/timeshift/user/p%23ss/60/2026-07-01:20-30/42.ts"

    def test_winter_offset_and_extension(self):
        start = datetime(2026, 1, 15, 23, 15, tzinfo=timezone.utc)
        url = catchup.timeshift_url(
            _stream(stream_id="7"),
            start,
            45,
            ZoneInfo("Europe/Amsterdam"),
            ext="m3u8",
        )
        assert url.endswith("/45/2026-01-16:00-15/7.m3u8")


class TestServerTimezone:
    def test_reads_and_caches(self):
        fetch = MagicMock(return_value={"server_info": {"timezone": "Europe/Amsterdam"}})
        assert catchup.server_timezone("src1", fetch) == ZoneInfo("Europe/Amsterdam")
        assert catchup.server_timezone("src1", fetch) == ZoneInfo("Europe/Amsterdam")
        fetch.assert_called_once()

    def test_unknown_zone_falls_back_to_utc(self):
        fetch = MagicMock(return_value={"server_info": {"timezone": "Nowhere/Special"}})
        assert catchup.server_timezone("src1", fetch) == timezone.utc
        catchup.server_timezone("src1", fetch)
        fetch.assert_called_once()

    def test_missing_zone_is_utc(self):
        assert catchup.server_timezone("src1", MagicMock(return_value={})) == timezone.utc

    def test_transient_failure_is_retried(self):
        fetch = MagicMock(side_effect=[OSError("down"), {"server_info": {"timezone": "UTC"}}])
        assert catchup.server_timezone("src1", fetch) == timezone.utc
        assert catchup.server_timezone("src1", fetch) == ZoneInfo("UTC")
        assert fetch.call_count == 2


def _stream(**overrides: MutablePlainTree) -> ProviderRecord:
    stream: ProviderRecord = {
        "stream_id": "src1_42",
        "source_type": "xtream",
        "source_url": "http://upstream.test:8080/",
        "source_username": "user",
        "source_password": "p#ss",
        "tv_archive": 1,
        "tv_archive_duration": "3",
    }
    stream.update(overrides)
    return stream


@pytest.fixture(autouse=True)
def _clear_tz_cache() -> Iterator[None]:
    catchup._tz_cache.clear()
    yield
    catchup._tz_cache.clear()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
