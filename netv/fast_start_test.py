"""Fast start must keep one upstream reader and exactly one local encoder."""

from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import replace

import asyncio
import math
import os
import pathlib
import subprocess
import time

from fastapi import HTTPException

import pytest
import pytest_asyncio

from netv import cache, fast_start, ffmpeg_command, ffmpeg_session, util
from netv.cache import HwAccel
from netv.ffmpeg_command import MediaInfo
from netv.playback_policy import LiveRecoveryPolicy, PlaybackHealth, PlaybackPolicy
from netv.testing import (
    BackgroundTasks,
    FakeClock,
    FakeProcess,
    FakeResponse,
    FakeRun,
    FakeUpstreams,
    FakeWeb,
    PopenRecorder,
    ProcessLauncher,
    Program,
    install_clock,
    install_launcher,
    install_subprocess,
    install_web,
    pat_packet,
    pmt_packet,
    probed_video_json,
    ts_source,
    video_packet,
)


CAPTIONED_FRAMES = '{"frames": [{"side_data_list": [{"side_data_type": "Closed Captions"}]}]}'
PROVIDER = "https://provider/live"


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Each readiness poll advances fake time instead of sleeping 0.2s."""
    fake = FakeClock()
    install_clock(monkeypatch, fake, ffmpeg_session)
    return fake


@pytest.fixture(autouse=True)
def programs(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    """Fake samples carry no media; ffprobe reports a 1080p video unless scripted."""
    run = FakeRun()
    run.program("ffprobe", Program(stdout=probed_video_json()))
    install_subprocess(monkeypatch, ffmpeg_command, run=run)
    return run


@pytest_asyncio.fixture(autouse=True)
async def upstreams(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeUpstreams]:
    """The provider's live connections; closed before the loop ends to free reader threads."""
    opened = FakeUpstreams()
    install_web(monkeypatch, FakeWeb({PROVIDER: opened}), util)
    yield opened
    opened.close()


@pytest.fixture(autouse=True)
def fresh_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg_session, "_transcode_sessions", {})
    monkeypatch.setattr(ffmpeg_session, "_url_to_session", {})
    monkeypatch.setattr(ffmpeg_session, "_startup_guards", {})
    monkeypatch.setattr(ffmpeg_command, "_probe_cache", {})


@pytest.fixture
def tasks(monkeypatch: pytest.MonkeyPatch) -> Iterator[BackgroundTasks]:
    background = BackgroundTasks()
    monkeypatch.setattr(ffmpeg_session, "_spawn_background_task", background)
    yield background
    background.discard()


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch, tasks: BackgroundTasks) -> ProcessLauncher:
    """Record ffmpeg launches; monitors and watchdogs stay unscheduled in ``tasks``."""
    del tasks
    fake = ProcessLauncher()
    install_launcher(monkeypatch, fake, ffmpeg_session)
    return fake


def configure(
    transcode_dir: pathlib.Path,
    *,
    live_dvr_mins: int = 0,
    max_resolution: cache.MaxResolution = "4k",
) -> None:
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "transcode_dir": str(transcode_dir),
            "max_resolution": max_resolution,
            "live_dvr_mins": live_dvr_mins,
        },
    )


def ffprobe_source(video_codec: str, audio_codec: str = "aac", channels: int = 2) -> str:
    """ffprobe JSON for a 1080p source with one audio stream."""
    return (
        '{"streams": ['
        f'{{"codec_type": "video", "codec_name": "{video_codec}", '
        '"pix_fmt": "yuv420p", "height": 1080}, '
        f'{{"codec_type": "audio", "codec_name": "{audio_codec}", "channels": {channels}, '
        '"sample_rate": "48000", "profile": "LC"}], '
        '"format": {"format_name": "mpegts"}}'
    )


async def until(
    condition: Callable[[], bool],
    task: asyncio.Task[ffmpeg_session.TranscodeResponse],
) -> None:
    """Yield to ``task`` until ``condition`` holds, surfacing the task's failure.

    The deadline is wall time: relay readers and provider lookups run in real threads.
    """
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if condition():
            return
        if task.done():
            task.result()
            raise AssertionError("startup finished before the condition held")
        await asyncio.sleep(0)
    raise AssertionError("condition never held")


async def polled_again(
    clock: FakeClock,
    task: asyncio.Task[ffmpeg_session.TranscodeResponse],
) -> None:
    """Wait until ``task`` slept after re-reading its playlists at least once."""
    polls = len(clock.sleeps)
    await until(lambda: len(clock.sleeps) > polls + 1, task)


def output_dir(launcher: ProcessLauncher, index: int) -> pathlib.Path:
    return pathlib.Path(launcher.commands[index][-1]).parent


async def feed_source(
    upstreams: FakeUpstreams,
    task: asyncio.Task[ffmpeg_session.TranscodeResponse],
    data: bytes | None = None,
) -> None:
    """Wait for the session to connect upstream, then deliver ``data``."""
    await until(lambda: bool(upstreams.opened), task)
    upstreams.opened[-1].feed(ts_source() if data is None else data)


async def serve_fast_start(
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
    task: asyncio.Task[ffmpeg_session.TranscodeResponse],
) -> ffmpeg_session.TranscodeResponse:
    """Deliver the source, publish what the worker would write, then finish startup."""
    await feed_source(upstreams, task)
    await until(lambda: len(launcher.launches) == 1, task)
    media_playlist(output_dir(launcher, 0), "low", [1, 1])
    return await task


async def serve_transcode(
    launcher: ProcessLauncher,
    task: asyncio.Task[ffmpeg_session.TranscodeResponse],
) -> ffmpeg_session.TranscodeResponse:
    """Publish what a single HLS transcode would write, then finish startup."""
    await until(lambda: len(launcher.launches) == 1, task)
    directory = output_dir(launcher, 0)
    lines = ["#EXTM3U"]
    for i in range(2):
        (directory / f"seg{i:03d}.ts").write_bytes(video_packet(10 + 3 * i) * 10)
        lines += ["#EXTINF:3,", f"seg{i:03d}.ts"]
    (directory / "stream.m3u8").write_text("\n".join(lines) + "\n")
    return await task


def media_playlist(
    directory: pathlib.Path,
    name: str,
    durations: Sequence[float],
    start: float = 10,
) -> None:
    """Write complete segments of the given durations and their playlist."""
    lines = ["#EXTM3U"]
    for i, duration in enumerate(durations):
        filename = f"{name}_{i}.ts"
        (directory / filename).write_bytes(video_packet(start + sum(durations[:i])) * 10)
        lines += [f"#EXTINF:{duration},", filename]
    (directory / f"{name}.m3u8").write_text("\n".join(lines) + "\n")


def playlist(directory: pathlib.Path, name: str, start: float = 10) -> None:
    lines = ["#EXTM3U", "#EXT-X-TARGETDURATION:2"]
    for i in range(3):
        filename = f"{name}_{i}.ts"
        (directory / filename).write_bytes(video_packet(start + i * 2) * 10)
        lines += ["#EXTINF:2,", filename]
    (directory / f"{name}.m3u8").write_text("\n".join(lines) + "\n")


def test_timeline_and_readiness(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    playlist(tmp_path, "low")
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 7520
    assert fast_start.segment_pts(tmp_path / "low_0.ts") == 10
    content = (tmp_path / "low.m3u8").read_text()
    dated, dates = fast_start.dated_playlist(str(tmp_path), content, 10, 0, {})
    assert "#EXT-X-PROGRAM-DATE-TIME:1970-01-01T00:00:00.000+00:00" in dated
    assert dates == {"low_0.ts": 0, "low_1.ts": 2, "low_2.ts": 4}
    os.utime(tmp_path / "low.m3u8", (80, 80))
    install_clock(monkeypatch, FakeClock(now=100), fast_start)
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 0
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8", max_age=30) == 7520
    (tmp_path / "low_2.ts").unlink()
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 0


def test_segment_pts_skips_non_video_packets(tmp_path: pathlib.Path):
    assert fast_start.segment_pts(tmp_path / "absent.ts") is None
    no_sync = b"\x00" * 188
    no_payload_start = b"\x47\x01\x00\x10".ljust(188, b"\xff")
    adaptation_only = b"\x47\x41\x00\x20\xb7".ljust(188, b"\xff")
    no_pts = (b"\x47\x41\x00\x10\x00\x00\x01\xe0\x00\x00\x80\x00\x00").ljust(188, b"\xff")
    audio = (b"\x47\x41\x00\x10\x00\x00\x01\xc0\x00\x00\x80\x80\x05").ljust(188, b"\xff")
    junk = tmp_path / "junk.ts"
    junk.write_bytes(no_sync + no_payload_start + adaptation_only + no_pts + audio)
    assert fast_start.segment_pts(junk) is None
    # Adaptation field plus payload: PES starts after the field's length byte.
    packet = video_packet(7)
    with_adaptation = (b"\x47\x41\x00\x30\x01\x00" + packet[4:]).ljust(188, b"\xff")[:188]
    junk.write_bytes(no_pts + with_adaptation)
    assert fast_start.segment_pts(junk) == 7


def test_dated_playlist_keeps_earlier_dates_across_a_clock_change(tmp_path: pathlib.Path):
    playlist(tmp_path, "low")
    content = "#EXTM3U\n#EXTINF:2,\nlow_1.ts\n#EXTINF:2,\nlow_2.ts\n#EXTINF:2,\ngone.ts\n"
    earlier = {"low_1.ts": 500.0, "rolled_off.ts": 400.0}
    dated, dates = fast_start.dated_playlist(str(tmp_path), content, 14, 1000, earlier)
    assert dates == {"low_1.ts": 500.0, "low_2.ts": 1000.0}
    assert "#EXT-X-PROGRAM-DATE-TIME:1970-01-01T00:08:20.000+00:00\n#EXTINF" not in dated
    assert dated.count("#EXT-X-PROGRAM-DATE-TIME:") == 2
    assert dated.endswith("#EXTINF:2,\ngone.ts\n")


def test_dated_playlist_keeps_clock_after_discontinuity(tmp_path: pathlib.Path):
    (tmp_path / "next.ts").write_bytes(video_packet(12))
    content = "#EXTM3U\n#EXT-X-DISCONTINUITY\n#EXTINF:2,\nnext.ts\n"
    dated, dates = fast_start.dated_playlist(str(tmp_path), content, 10, 0, {})
    assert "#EXT-X-PROGRAM-DATE-TIME:1970-01-01T00:00:02.000+00:00" in dated
    assert dates == {"next.ts": 2.0}


def test_dated_playlist_unwraps_33_bit_pts_rollover(tmp_path: pathlib.Path):
    rollover = (1 << 33) / 90_000
    (tmp_path / "wrapped.ts").write_bytes(video_packet(1))
    content = "#EXTM3U\n#EXTINF:2,\nwrapped.ts\n"
    _, dates = fast_start.dated_playlist(str(tmp_path), content, rollover - 1, 100, {})
    assert dates == {"wrapped.ts": pytest.approx(102)}


def test_dated_playlist_never_dates_tags(tmp_path: pathlib.Path):
    content = "#EXTM3U\n#EXT-X-VERSION:3\n"
    stale = {"#EXTM3U": 5.0, "#EXT-X-VERSION:3": 5.0}
    assert fast_start.dated_playlist(str(tmp_path), content, 10, 0, stale) == (content, {})


def test_ready_bitrate_rejects_unsafe_or_empty_entries(tmp_path: pathlib.Path):
    playlist(tmp_path, "low")
    path = tmp_path / "low.m3u8"
    content = path.read_text()
    path.write_text(content.replace("low_2.ts", "../low_2.ts"))
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 0
    path.write_text(content.replace("#EXTINF:2,\nlow_2.ts", "#EXTINF:0,\nlow_2.ts"))
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 0


def test_live_start_offset_requires_playlist_header():
    content = "#EXT-X-VERSION:3\n#EXTM3U\n"
    assert fast_start.add_live_start_offset(content, 24) == content


def test_last_media_sequence_counts_from_the_playlist_start():
    assert fast_start.last_media_sequence("#EXTM3U\n#EXT-X-TARGETDURATION:1\n") == -1
    assert fast_start.last_media_sequence("#EXTM3U\n#EXTINF:1,\na.ts\n#EXTINF:1,\nb.ts\n") == 1
    deleted = "#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:40\n#EXT-X-DISCONTINUITY\n#EXTINF:1,\nc.ts\n"
    assert fast_start.last_media_sequence(deleted) == 40


def test_blocking_reload_is_advertised_once_after_the_header():
    content = "#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:1,\na.ts\n"
    blocking = fast_start.with_blocking_reload(content)
    assert blocking == (
        "#EXTM3U\n#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES\n#EXT-X-VERSION:3\n#EXTINF:1,\na.ts\n"
    )
    assert fast_start.with_blocking_reload(blocking) == blocking
    assert fast_start.with_blocking_reload("#EXT-X-VERSION:3\n") == "#EXT-X-VERSION:3\n"


def test_a_blocking_master_points_its_rendition_at_the_blocking_playlist():
    master = fast_start.master_playlist("1080p")
    assert fast_start.blocking_master(master) == master.replace(
        "\nlow.m3u8\n",
        "\nlow.m3u8?blocking=1\n",
    )
    assert "low.m3u8?blocking=1" in fast_start.blocking_master(
        fast_start.closed_caption_master_playlist("1080p"),
    )


def test_live_buffers_follow_upstream_segment_cadence():
    # A steady tuner delivers short segments: two of them bridge any gap, so
    # startup waits seconds of real time, not a fixed 8s.
    assert fast_start.live_buffer_durations([1.97, 1.53, 1.52]) == (3.94, 12)
    assert fast_start.live_buffer_durations([0.5]) == (1, 12)
    # No usable duration falls back to a 4s upstream cadence.
    assert fast_start.live_buffer_durations([]) == (8, 12)
    assert fast_start.live_buffer_durations([0, -1, math.inf, math.nan]) == (8, 12)
    assert fast_start.live_buffer_durations([4]) == (8, 12)
    assert fast_start.live_buffer_durations([6]) == (12, 12)
    assert fast_start.live_buffer_durations([10]) == (20, 20)
    assert fast_start.live_buffer_durations([30]) == (60, 60)
    assert fast_start.live_buffer_durations([45]) == (90, 60)


def test_live_playlist_prefers_the_measured_buffer_offset():
    content = "#EXTM3U\n#EXT-X-VERSION:3\n"
    updated = fast_start.add_live_start_offset(content, 24)
    assert updated == ("#EXTM3U\n#EXT-X-START:TIME-OFFSET=-24,PRECISE=NO\n#EXT-X-VERSION:3\n")
    assert fast_start.add_live_start_offset(updated, 30) == updated


def _workers(directory: str, *, restarting: bool = False) -> list[list[str]]:
    encoders = [
        fast_start.encoder_command(
            directory,
            hardware,
            "1080p" if upscale else "720p",
            "low",
            False,
            allow_upscale=upscale,
            restarting=restarting,
        )
        for hardware in ("software", "nvenc+software", "amf+software", "qsv", "vaapi")
        for upscale in (False, True)
    ]
    return [*encoders, fast_start.remux_command(directory, restarting=restarting)]


def test_workers_read_the_relay_from_stdin(tmp_path: pathlib.Path):
    """Workers never open upstream: the relay is its only reader."""
    for cmd in _workers(str(tmp_path)):
        assert cmd[cmd.index("-i") - 2 : cmd.index("-i") + 2] == ["-f", "mpegts", "-i", "pipe:0"]
        assert "-reconnect" not in cmd
        assert "-live_start_index" not in cmd
        assert "-copyts" in cmd
        assert cmd[-1] == f"{tmp_path}/low.m3u8"
        assert cmd[cmd.index("-hls_segment_filename") + 1] == f"{tmp_path}/low_%06d.ts"
        assert (
            float(cmd[cmd.index("-hls_time") + 1])
            * int(
                cmd[cmd.index("-hls_list_size") + 1],
            )
            >= 30
        )


def test_workers_start_playback_with_short_segments(tmp_path: pathlib.Path):
    """Quarter-second first segments, then one-second segments."""
    for cmd in _workers(str(tmp_path)):
        assert cmd[cmd.index("-hls_time") + 1] == "1"
        assert cmd[cmd.index("-hls_init_time") + 1] == "0.25"


def test_workers_analyze_only_as_long_as_they_must():
    """Encoders decode audio whenever it arrives; a copy must see it before writing."""
    *encoders, remux = _workers("/s")
    for cmd in encoders:
        assert cmd[cmd.index("-analyzeduration") + 1] == "300000"
    assert remux[remux.index("-analyzeduration") + 1] == "800000"


def test_encoders_add_no_startup_delay(tmp_path: pathlib.Path):
    """No B-frames or lookahead; quarter-second keyframes cut the short first segments."""
    for cmd in _workers(str(tmp_path))[:-1]:
        assert cmd.count("-bf") == 1
        assert cmd[cmd.index("-bf") + 1] == "0"
        assert not {"-rc-lookahead", "-look_ahead", "-look_ahead_depth"} & set(cmd)
        assert cmd[cmd.index("-force_key_frames") + 1] == (
            "expr:if(isnan(prev_forced_t),1,if(lt(n_forced,4),"
            "gte(floor((t+0.000000)*4),floor((prev_forced_t+0.000000)*4)+1),"
            "gte(floor(t+0.000000),floor(prev_forced_t+0.000000)+1)))"
        )
        assert cmd[cmd.index("-probesize") + 1] == "5000000"
        # Only encoders that flag forced non-IDR frames as non-key are asked for IDR.
        encoder = cmd[cmd.index("-c:v") + 1]
        assert ("-forced-idr" in cmd) == (encoder in ("h264_nvenc", "libx264"))


def test_the_upscale_rendition_publishes_beside_the_plain_one():
    """The AI Upscale worker writes up.m3u8 so the plain rendition keeps playing meanwhile."""
    args = ("/s", "nvenc+software", "1080p", "high", False)
    cmd = fast_start.encoder_command(*args, allow_upscale=True, rendition="up")
    assert cmd[-1] == "/s/up.m3u8"
    assert cmd[cmd.index("-hls_segment_filename") + 1] == "/s/up_%06d.ts"
    assert fast_start.encoder_command(*args, allow_upscale=False)[-1] == "/s/low.m3u8"


def _rendition(directory: pathlib.Path, name: str, starts: Sequence[float], msn: int) -> str:
    """A rendition's playlist of 1 s segments whose video starts at ``starts``."""
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-TARGETDURATION:1",
        f"#EXT-X-MEDIA-SEQUENCE:{msn}",
    ]
    for i, start in enumerate(starts):
        filename = f"{name}_{msn + i:06d}.ts"
        (directory / filename).write_bytes(video_packet(start) * 10)
        lines += ["#EXTINF:1.000000,", filename]
    return "\n".join(lines) + "\n"


def test_the_upscaled_rendition_continues_the_plain_one_after_a_discontinuity(
    tmp_path: pathlib.Path,
):
    """A playing client sees low's segments, a discontinuity, then up's from where low ended."""
    low = _rendition(tmp_path, "low", [10, 11, 12], msn=40)
    # The upscaler, numbered on its own, cuts at the same source timestamps.
    up = _rendition(tmp_path, "up", [11, 12, 13, 14], msn=0)
    assert fast_start.splice_point(str(tmp_path), low, up) == 2
    spliced = fast_start.splice_playlists(low, up, 2, list_size=10)
    assert spliced == (
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:40\n"
        "#EXTINF:1.000000,\nlow_000040.ts\n#EXTINF:1.000000,\nlow_000041.ts\n"
        "#EXTINF:1.000000,\nlow_000042.ts\n"
        "#EXT-X-DISCONTINUITY\n#EXTINF:1.000000,\nup_000002.ts\n"
        "#EXTINF:1.000000,\nup_000003.ts\n"
    )
    # Program dates follow the shared source timeline across the switch.
    _, dates = fast_start.dated_playlist(str(tmp_path), spliced, 10, 1000, {})
    assert list(dates.values()) == [1000, 1001, 1002, 1003, 1004]


def test_misaligned_grids_continue_at_the_next_upscaled_segment(tmp_path: pathlib.Path):
    low = _rendition(tmp_path, "low", [10, 11], msn=0)
    up = _rendition(tmp_path, "up", [11.5, 12.5], msn=3)
    assert fast_start.splice_point(str(tmp_path), low, up) == 4
    # A plain playlist with no segments yet is continued from the upscaler's first.
    assert fast_start.splice_point(str(tmp_path), "#EXTM3U\n", up) == 3


def test_a_spliced_rendition_slides_its_window_and_counts_rolled_off_discontinuities(
    tmp_path: pathlib.Path,
):
    low = _rendition(tmp_path, "low", [10, 11], msn=5)
    up = _rendition(tmp_path, "up", [12, 13, 14], msn=7)
    spliced = fast_start.splice_playlists(low, up, 7, list_size=4)
    assert "#EXT-X-MEDIA-SEQUENCE:6\n" in spliced
    assert "low_000006.ts\n#EXT-X-DISCONTINUITY\n#EXTINF:1.000000,\nup_000007.ts" in spliced
    # Once low has rolled off, the discontinuity it ended with is counted instead.
    rolled = fast_start.splice_playlists(low, up, 7, list_size=2)
    assert "#EXT-X-MEDIA-SEQUENCE:8\n#EXT-X-DISCONTINUITY-SEQUENCE:1\n" in rolled
    assert "low_" not in rolled
    assert "#EXT-X-DISCONTINUITY\n" not in rolled
    # The upscaler's own window may already have passed the splice point.
    later = _rendition(tmp_path, "up", [14, 15], msn=9)
    assert "#EXT-X-MEDIA-SEQUENCE:9\n#EXT-X-DISCONTINUITY-SEQUENCE:1\n" in (
        fast_start.splice_playlists(low, later, 7, list_size=2)
    )


def test_nothing_continues_before_the_upscaler_reaches_the_plain_end(tmp_path: pathlib.Path):
    low = _rendition(tmp_path, "low", [10, 11], msn=0)
    behind = _rendition(tmp_path, "up", [8, 9], msn=0)
    assert fast_start.splice_point(str(tmp_path), low, behind) is None
    assert fast_start.splice_playlists(low, behind, None, list_size=10) == low
    # Segments whose files are gone reveal no timestamp.
    gone = "#EXTM3U\n#EXTINF:1,\ngone.ts\n"
    assert fast_start.splice_point(str(tmp_path), low, gone) is None
    assert fast_start.playlist_end_pts(str(tmp_path), "#EXTM3U\n") is None
    assert fast_start.playlist_end_pts(str(tmp_path), gone) is None
    assert fast_start.playlist_end_pts(str(tmp_path), low) == 12


def test_recorder_reads_the_relay_and_stops_with_the_program():
    source = MediaInfo(video_codec="h264", audio_codec="aac", pix_fmt="yuv420p", height=720)
    cmd = fast_start.recorder_command("/r", "software", "1080p", "high", 1800.7, source)
    i = cmd.index("-i")
    assert cmd[i - 4 : i + 2] == ["-f", "mpegts", "-t", "1800", "-i", "pipe:0"]
    assert cmd[-1] == "/r/stream.m3u8"
    # Kept like VOD: every segment stays and the playlist grows as an event.
    assert cmd[cmd.index("-hls_playlist_type") + 1] == "event"
    assert cmd[cmd.index("-hls_list_size") + 1] == "0"
    assert not any("dnn_processing" in arg for arg in cmd)
    # Without probe facts ffmpeg analyzes the stream itself.
    unprobed = fast_start.recorder_command("/r", "software", "1080p", "high", 60, None)
    assert "-analyzeduration" in unprobed


def test_restart_commands_append_after_a_discontinuity(tmp_path: pathlib.Path):
    for fresh, restart in zip(
        _workers(str(tmp_path)),
        _workers(str(tmp_path), restarting=True),
        strict=True,
    ):
        assert "append_list" not in fresh[fresh.index("-hls_flags") + 1]
        flags = restart[restart.index("-hls_flags") + 1]
        assert "append_list" in flags
        assert "discont_start" in flags
        assert "-hls_start_number_source" not in restart


def test_dated_playlist_replaces_encoder_dates_after_discontinuity(tmp_path: pathlib.Path):
    playlist(tmp_path, "low")
    content = (
        (tmp_path / "low.m3u8")
        .read_text()
        .replace(
            "#EXTINF:2,\n",
            "#EXTINF:2,\n#EXT-X-DISCONTINUITY\n#EXT-X-PROGRAM-DATE-TIME:2030-01-01T00:00:00Z\n",
        )
    )
    dated, dates = fast_start.dated_playlist(str(tmp_path), content, 10, 0, {})
    assert dated.count("#EXT-X-PROGRAM-DATE-TIME:") == 3
    assert "2030-" not in dated
    assert fast_start.dated_playlist(str(tmp_path), dated, 10, 0, dates) == (dated, dates)
    (tmp_path / "low.m3u8").write_text(content)
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 7520


def test_discontinuity_keeps_shared_clock_when_pts_remain_continuous(tmp_path: pathlib.Path):
    playlist(tmp_path, "low")
    content = (
        (tmp_path / "low.m3u8")
        .read_text()
        .replace(
            "#EXTINF:2,\nlow_1.ts",
            "#EXTINF:2,\n#EXT-X-DISCONTINUITY\nlow_1.ts",
        )
    )
    os.utime(tmp_path / "low_1.ts", (32, 32))
    dated, _ = fast_start.dated_playlist(str(tmp_path), content, 10, 0, {})
    assert "#EXT-X-PROGRAM-DATE-TIME:1970-01-01T00:00:02.000+00:00" in dated


@pytest.mark.parametrize("upscale", [False, True])
def test_encoder_keeps_surround_audio(tmp_path: pathlib.Path, upscale: bool):
    """The encoder keeps 5.1; Dolby is copied only for passthrough clients."""

    audio = MediaInfo(
        video_codec="",
        audio_codec="eac3",
        pix_fmt="",
        audio_channels=6,
        audio_sample_rate=48_000,
    )
    args = (str(tmp_path), "software", "1080p" if upscale else "720p", "low", False)
    aac = fast_start.encoder_command(*args, allow_upscale=upscale, source=audio)
    assert aac[aac.index("-c:a") + 1] == "aac"
    assert aac[aac.index("-af") + 1] == "aformat=channel_layouts=5.1"
    dolby = fast_start.encoder_command(
        *args,
        allow_upscale=upscale,
        source=audio,
        audio_passthrough=True,
    )
    assert dolby[dolby.index("-c:a") + 1] == "copy"
    assert dolby[dolby.index("-c:v") + 1] != "copy"


def test_encoder_command_for_an_interlaced_surround_broadcast(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The whole command a 1080i, 5.1 broadcast gets on an NVENC host."""
    configure(tmp_path)
    monkeypatch.setattr(ffmpeg_command, "_gpu_nvdec_codecs", {"h264"})
    source = MediaInfo(
        video_codec="h264",
        audio_codec="ac3",
        pix_fmt="yuv420p",
        audio_channels=6,
        audio_sample_rate=48_000,
        height=1080,
        interlaced=True,
    )
    cmd = fast_start.encoder_command(
        "/s",
        "nvenc+software",
        "1080p",
        "high",
        False,
        allow_upscale=False,
        source=source,
    )
    keyframes = (
        "expr:if(isnan(prev_forced_t),1,if(lt(n_forced,4),"
        "gte(floor((t+0.000000)*4),floor((prev_forced_t+0.000000)*4)+1),"
        "gte(floor(t+0.000000),floor(prev_forced_t+0.000000)+1)))"
    )
    assert cmd == [
        "ffmpeg",
        "-copyts",
        "-hide_banner",
        "-loglevel",
        "error",
        "-noautorotate",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-extra_hw_frames",
        "3",
        "-probesize",
        "5000000",
        "-analyzeduration",
        "300000",
        "-fflags",
        "+discardcorrupt+genpts",
        "-err_detect",
        "ignore_err",
        "-f",
        "mpegts",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-vf",
        "yadif_cuda=0:-1:1,scale_cuda=-2:min(ih\\,1080):format=nv12",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-spatial-aq",
        "1",
        "-temporal-aq",
        "1",
        "-g",
        "60",
        "-c:a",
        "aac",
        "-af",
        "aformat=channel_layouts=5.1",
        "-ar",
        "48000",
        "-b:a",
        "384k",
        "-profile:a",
        "aac_low",
        "-max_delay",
        "5000000",
        "-bf",
        "0",
        "-force_key_frames",
        keyframes,
        "-forced-idr",
        "1",
        "-f",
        "hls",
        "-hls_time",
        "1",
        "-hls_init_time",
        "0.25",
        "-hls_list_size",
        "30",
        "-hls_segment_filename",
        "/s/low_%06d.ts",
        "-hls_flags",
        "delete_segments+temp_file",
        "-rc",
        "vbr",
        "-b:v",
        "6000000",
        "-maxrate",
        "8000000",
        "-bufsize",
        "8000000",
        "/s/low.m3u8",
    ]


def test_ready_bitrate_counts_one_segment_when_asked(tmp_path: pathlib.Path):
    segment = tmp_path / "low_0.ts"
    segment.write_bytes(video_packet(10) * 10)
    path = tmp_path / "low.m3u8"
    path.write_text("#EXTM3U\n#EXTINF:4,\nlow_0.ts\n")
    assert fast_start.ready_bitrate(str(tmp_path), path.name, minimum_segments=1) == 3760
    assert fast_start.ready_bitrate(str(tmp_path), path.name) == 0
    segment.write_bytes(b"partial")
    assert fast_start.ready_bitrate(str(tmp_path), path.name, minimum_segments=1) == 0
    segment.unlink()
    assert fast_start.ready_bitrate(str(tmp_path), path.name, minimum_segments=1) == 0


@pytest.mark.parametrize("hardware", ["nvenc+software", "software", "amf+software", "qsv"])
@pytest.mark.parametrize(
    "resolution,target,maximum",
    [
        ("1080p", "6000000", "8000000"),
        ("4k", "16000000", "20000000"),
    ],
)
def test_encoder_uses_bounded_bitrate(
    tmp_path: pathlib.Path,
    hardware: HwAccel,
    resolution: str,
    target: str,
    maximum: str,
):
    cmd = fast_start.encoder_command(
        str(tmp_path),
        hardware,
        resolution,
        "high",
        False,
        allow_upscale=True,
    )
    assert cmd[cmd.index("-b:v") + 1] == target
    assert cmd[cmd.index("-maxrate") + 1] == maximum
    assert cmd[cmd.index("-bufsize") + 1] == maximum
    assert not set(cmd) & {"-qp", "-qp_i", "-qp_p", "-global_quality", "-crf", "constqp", "cqp"}
    if hardware == "nvenc+software":
        assert cmd[cmd.index("-rc") + 1] == "vbr"
    low = fast_start.encoder_command(
        str(tmp_path),
        hardware,
        "720p",
        "low",
        False,
        allow_upscale=False,
    )
    assert low[low.index("-b:v") + 1] == "4000000"
    assert low[low.index("-maxrate") + 1] == "6000000"
    assert not set(low) & {"-qp", "-qp_i", "-qp_p", "-global_quality", "-crf", "constqp"}


def test_encoders_decode_on_gpu_when_source_codec_is_known(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    programs: FakeRun,
):
    # Maxwell decodes H.264 but not HEVC.
    programs.program("nvidia-smi", Program(stdout="NVIDIA GeForce GTX TITAN X, 5.2\n"))
    monkeypatch.setattr(ffmpeg_command, "_gpu_nvdec_codecs", None)
    for upscale in (False, True):
        args = (str(tmp_path), "nvenc+software", "1080p", "high", False)
        h264 = MediaInfo(video_codec="h264", audio_codec="", pix_fmt="")
        cmd = fast_start.encoder_command(*args, allow_upscale=upscale, source=h264)
        assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
        assert "hwupload_cuda" not in cmd[cmd.index("-vf") + 1]
        assert "-hwaccel" not in fast_start.encoder_command(*args, allow_upscale=upscale)
        assert "-hwaccel" not in fast_start.encoder_command(
            *args,
            allow_upscale=upscale,
            source=MediaInfo(video_codec="hevc", audio_codec="", pix_fmt=""),
        )


def test_master_playlist_reserves_surround_audio_bandwidth():

    def audio(codec: str, channels: int) -> MediaInfo:
        return MediaInfo(video_codec="", audio_codec=codec, pix_fmt="", audio_channels=channels)

    assert fast_start.surround_audio_bitrate(audio("aac", 2), True) == 0
    assert fast_start.surround_audio_bitrate(audio("aac", 1), False) == 0
    assert fast_start.surround_audio_bitrate(audio("aac", 0), False) == 384_000
    assert fast_start.surround_audio_bitrate(audio("aac", 6), False) == 384_000
    assert fast_start.surround_audio_bitrate(None, False) == 384_000
    assert fast_start.surround_audio_bitrate(audio("eac3", 6), False) == 384_000
    assert fast_start.surround_audio_bitrate(audio("eac3", 6), True) == 640_000
    content = fast_start.master_playlist("720p", audio_bitrate=640_000)
    assert "BANDWIDTH=7240000\nlow.m3u8" in content


@pytest.mark.parametrize("resolution,bandwidth", [("720p", 6_600_000), ("4k", 22_000_000)])
def test_master_playlist_labels_the_single_rendition_at_encoder_resolution(
    resolution: str,
    bandwidth: int,
):
    assert fast_start.master_playlist(resolution) == (
        f"#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-STREAM-INF:BANDWIDTH={bandwidth}\nlow.m3u8\n"
    )


def test_closed_caption_master_declares_instream_cc1():
    assert fast_start.closed_caption_master_playlist("720p") == (
        "#EXTM3U\n#EXT-X-VERSION:3\n"
        '#EXT-X-MEDIA:TYPE=CLOSED-CAPTIONS,GROUP-ID="cc",NAME="CC1",'
        'DEFAULT=NO,AUTOSELECT=YES,INSTREAM-ID="CC1"\n'
        '#EXT-X-STREAM-INF:BANDWIDTH=6600000,CLOSED-CAPTIONS="cc"\n'
        "low.m3u8\n"
    )


def fast_session(
    directory: pathlib.Path,
    started: float,
    *,
    bandwidth_saver: bool = False,
    recovery_bitrate: float = 0,
) -> ffmpeg_session.FastLiveSession:
    """A running fast-live session with a live worker; its relay never starts."""
    plan = ffmpeg_session._FastLivePlan(
        source_url="http://stream.test/live",
        user_agent=None,
        hw="software",
        resolution="720p" if bandwidth_saver else "1080p",
        quality="high",
        deinterlace=False,
        bandwidth_saver=bandwidth_saver,
        audio_passthrough=False,
        dvr=False,
    )
    return ffmpeg_session.FastLiveSession(
        dir=str(directory),
        username="u",
        fast_start=True,
        viewers={},
        process=FakeProcess(exit_code=None),
        playback_policy=PlaybackPolicy(bandwidth_saver=bandwidth_saver),
        live_recovery=LiveRecoveryPolicy(),
        plan=plan,
        relay=plan.relay(),
        bandwidth_saver=bandwidth_saver,
        recovery_bitrate=recovery_bitrate,
        started=started,
        url=plan.source_url,
    )


def test_health_learns_full_quality_bitrate_then_saver_recovers(
    tmp_path: pathlib.Path,
    clock: FakeClock,
):
    playlist(tmp_path, "low")
    session = fast_session(tmp_path, clock.time())
    ffmpeg_session._transcode_sessions["fast"] = session
    start = clock.now
    good = PlaybackHealth(buffer_seconds=10, waiting=False, observed_bitrate=20_000)
    result = ffmpeg_session.report_playback_health("fast", "u", good)
    assert result == {"bandwidth_saver": False, "playlist": "/transcode/fast/low.m3u8"}
    assert "recovery_bitrate" in session
    assert session["recovery_bitrate"] == 7520
    session["playback_policy"] = replace(session["playback_policy"], bandwidth_saver=True)
    for offset in range(2, 62, 2):
        clock.now = start + offset
        result = ffmpeg_session.report_playback_health("fast", "u", good)
        # The full-quality session cannot serve saver: clients must retune.
        assert result == {"bandwidth_saver": True}
    clock.now = start + 62
    result = ffmpeg_session.report_playback_health("fast", "u", good)
    assert result == {"bandwidth_saver": False, "playlist": "/transcode/fast/low.m3u8"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bandwidth_saver", "resolution", "quality", "bitrate", "upscale"),
    [(False, "4k", "high", "16000000", True), (True, "720p", "low", "4000000", False)],
)
@pytest.mark.parametrize("segment_duration", [0.5, 2, 30])
async def test_shared_session_runs_one_worker_fed_by_the_relay(
    tmp_path: pathlib.Path,
    clock: FakeClock,
    programs: FakeRun,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
    segment_duration: float,
    bandwidth_saver: bool,
    resolution: str,
    quality: str,
    bitrate: str,
    upscale: bool,
):
    configure(tmp_path)
    programs.program("ffprobe", Program(stdout=CAPTIONED_FRAMES), when="frame=side_data_list")
    task = asyncio.create_task(
        ffmpeg_session.start_transcode(PROVIDER, bandwidth_saver=bandwidth_saver),
    )
    first_byte_at = clock.now
    await feed_source(upstreams, task)
    await until(lambda: len(launcher.launches) == 1, task)
    directory = output_dir(launcher, 0)
    # One segment cannot play; two can, however short.
    media_playlist(directory, "low", [segment_duration])
    await polled_again(clock, task)
    assert not task.done()
    media_playlist(directory, "low", [segment_duration, segment_duration])
    result = await task

    [launch] = launcher.launches
    worker = launch.command
    assert PROVIDER not in worker
    assert launch.options["stdin"] == asyncio.subprocess.PIPE
    # The worker replays everything since the first byte.
    await launch.process.stdin.wait_for(len(ts_source()))
    assert bytes(launch.process.stdin.written) == ts_source()
    assert worker[worker.index("-b:v") + 1] == bitrate
    session = ffmpeg_session.get_session(result["session_id"])
    assert session is not None
    assert "audio_info" in session
    assert session["audio_info"] is not None
    assert session["audio_info"].height == 1080
    encoder = (str(directory), "software", resolution, quality, True)
    assert worker == fast_start.encoder_command(
        *encoder,
        allow_upscale=upscale,
        source=session["audio_info"],
    )
    # The session clock starts at the source's first timestamp, when it arrived.
    assert (session.get("origin_pts"), session.get("origin_time")) == (10, first_byte_at)
    assert result["playlist"].endswith("/low.m3u8")
    assert "master_playlist" in result
    assert result["master_playlist"].endswith("/master.m3u8")
    assert "caption_playlist" in result
    assert result["caption_playlist"].endswith("/low-captions.m3u8")
    expected_playback_buffer = min(60, max(12, 2 * segment_duration))
    assert "playback_buffer_seconds" in result
    assert result["playback_buffer_seconds"] == expected_playback_buffer
    assert ffmpeg_session._is_fast_live(session)
    assert session["playback_policy"].bandwidth_saver is bandwidth_saver
    assert "watchdog_stale_after" in session
    assert session["watchdog_stale_after"] == max(45, 2 * segment_duration + 2)
    assert "playback_buffer_seconds" in session
    assert session["playback_buffer_seconds"] == expected_playback_buffer
    master = (directory / "master.m3u8").read_text()
    assert master == fast_start.master_playlist(resolution, audio_bitrate=384_000)
    assert 'INSTREAM-ID="CC1"' in (directory / "low-captions.m3u8").read_text()
    assert sorted(path.name for path in directory.iterdir()) == [
        "low-captions.m3u8",
        "low.m3u8",
        "low_0.ts",
        "low_1.ts",
        "master.m3u8",
        "probe.ts",
    ]
    reused = await ffmpeg_session.start_transcode(PROVIDER)
    # The same session, handed to a second page: its own viewer id.
    assert {**reused, "viewer_id": ""} == {**result, "viewer_id": ""}
    assert reused.get("viewer_id") not in (None, result.get("viewer_id"))
    [upstream] = upstreams.opened
    await ffmpeg_session.stop_session(result["session_id"], force=True)
    assert launch.process.returncode is not None
    assert upstream.closed
    assert launch.process.stdin.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_worker_starts_at_the_first_keyframe(
    tmp_path: pathlib.Path,
    programs: FakeRun,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
):
    """Bytes before the first keyframe cannot decode; neither probe nor worker sees them."""
    configure(tmp_path)
    task = asyncio.create_task(ffmpeg_session.start_transcode(PROVIDER))
    joined_mid_gop = pat_packet() + pmt_packet() + video_packet(9) * 50
    await feed_source(upstreams, task, joined_mid_gop)
    for _ in range(1000):
        await asyncio.sleep(0)
    assert not programs.calls
    assert not launcher.launches
    upstreams.opened[0].feed(ts_source(seconds=10))
    await until(lambda: len(launcher.launches) == 1, task)
    probed = (output_dir(launcher, 0) / "probe.ts").read_bytes()
    assert probed == ts_source(seconds=10)
    stdin = launcher.processes[0].stdin
    await stdin.wait_for(len(probed))
    assert bytes(stdin.written) == probed
    media_playlist(output_dir(launcher, 0), "low", [1, 1])
    result = await task
    session = ffmpeg_session.get_session(result["session_id"])
    assert session is not None
    # The clock still starts from the first byte's timestamp.
    assert session.get("origin_pts") == 9
    await ffmpeg_session.stop_session(result["session_id"], force=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("ends", [False, True])
async def test_a_source_without_keyframe_marks_is_used_whole(
    tmp_path: pathlib.Path,
    programs: FakeRun,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
    ends: bool,
):
    """Without random-access marks, the worker gets everything: at 4 MB, or when upstream ends."""
    configure(tmp_path)
    task = asyncio.create_task(ffmpeg_session.start_transcode(PROVIDER))
    unmarked = video_packet(10) * (10 if ends else (4 << 20) // 188 + 1)
    await feed_source(upstreams, task, unmarked)
    if ends:
        upstreams.opened[0].end()
    await until(lambda: len(launcher.launches) == 1, task)
    assert (output_dir(launcher, 0) / "probe.ts").read_bytes() == unmarked
    await launcher.processes[0].stdin.wait_for(len(unmarked))
    media_playlist(output_dir(launcher, 0), "low", [1, 1])
    result = await task
    await ffmpeg_session.stop_session(result["session_id"], force=True)


def test_keyframe_replay_starts_at_the_last_keyframe_with_the_tables():
    assert fast_start.keyframe_replay(b"") is None
    # No keyframe, or a keyframe without both tables, is not yet replayable.
    assert fast_start.keyframe_replay(pat_packet() + pmt_packet() + video_packet(1) * 3) is None
    assert fast_start.keyframe_replay(pat_packet() + video_packet(1, keyframe=True) * 3) is None
    assert fast_start.keyframe_replay(pmt_packet() + video_packet(1, keyframe=True) * 3) is None
    first, second = video_packet(1, keyframe=True), video_packet(2, keyframe=True)
    stream = pat_packet() + pmt_packet() + first + video_packet(1) + second + video_packet(2)
    assert fast_start.keyframe_replay(
        stream,
    ) == pat_packet() + pmt_packet() + second + video_packet(2)
    # History trimmed mid-packet resynchronizes; tables after the keyframe still count.
    trimmed = b"\x00" * 100 + first + video_packet(1) + pat_packet() + pmt_packet()
    assert (
        fast_start.keyframe_replay(trimmed)
        == pat_packet() + pmt_packet() + first + video_packet(1) + pat_packet() + pmt_packet()
    )


def test_keyframe_replay_skips_marks_off_the_packet_grid_and_a_partial_last_packet():
    keyframe = video_packet(1, keyframe=True)
    # A sync byte and unit-start flag inside a payload are not a packet start.
    payload_mark = bytearray(video_packet(1))
    payload_mark[100:102] = b"\x47\x40"
    stream = pat_packet() + pmt_packet() + keyframe + bytes(payload_mark)
    assert fast_start.keyframe_replay(stream) == stream
    # A keyframe still arriving is not replayed.
    assert fast_start.keyframe_replay(stream + keyframe[:100]) == stream + keyframe[:100]


def test_keyframe_replay_ignores_audio_and_continuation_packets():
    audio = bytearray(video_packet(1, keyframe=True))
    audio[9] = 0xC0  # An audio stream id, random access marked.
    continuation = bytearray(video_packet(1, keyframe=True))
    continuation[1] &= ~0x40  # Not a unit start.
    no_adaptation_body = bytearray(video_packet(1, keyframe=True))
    no_adaptation_body[4] = 0  # An empty adaptation field carries no flags.
    short = bytearray(video_packet(1, keyframe=True))
    short[4] = 182  # Adaptation fills the packet; no PES header follows.
    for packet in (audio, continuation, no_adaptation_body, short):
        assert fast_start.keyframe_replay(pat_packet() + pmt_packet() + bytes(packet) * 3) is None
    # A pat_packet() whose pointer field skips past its program entry names no pmt_packet().
    pointer_past_entry = pat_packet()[:4] + b"\xb4" + pat_packet()[5:]
    keyframe = video_packet(1, keyframe=True)
    assert fast_start.keyframe_replay(pointer_past_entry + pmt_packet() + keyframe) is None


@pytest.mark.asyncio
async def test_hls_sources_are_read_once_through_a_remuxer(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    launcher: ProcessLauncher,
):
    """The relay's reader for HLS is one ffmpeg remuxing to MPEG-TS on stdout."""
    configure(tmp_path)
    url = "https://provider/live.m3u8"
    web = FakeWeb({url: lambda: FakeResponse(b"#EXTM3U\n#EXTINF:6,\nsegment.ts\n")})
    install_web(monkeypatch, web, ffmpeg_command)
    remuxers = PopenRecorder()
    install_subprocess(monkeypatch, ffmpeg_session, popen=remuxers)
    task = asyncio.create_task(ffmpeg_session.start_transcode(url))
    await until(lambda: bool(remuxers.started), task)
    [remuxer] = remuxers.started
    try:
        assert remuxer.args == fast_start.source_command(url, cache.get_user_agent())
        assert remuxer.options == {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE}
        remuxer.stdout.feed(ts_source())
        await until(lambda: len(launcher.launches) == 1, task)
        media_playlist(output_dir(launcher, 0), "low", [1, 1])
        result = await task
        await ffmpeg_session.stop_session(result["session_id"], force=True)
        # Both the relay's close and its reader's exit stop the remuxer.
        assert set(remuxer.signals) == {"KILL"}
        assert remuxer.stdout.closed
    finally:
        remuxer.stdout.close()


@pytest.mark.asyncio
async def test_xtream_live_is_read_as_one_mpegts_stream(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    launcher: ProcessLauncher,
):
    """Xtream panels serve live as continuous MPEG-TS: no playlist, no remuxer, ~3 s sooner."""
    configure(tmp_path)
    settings = cache.load_server_settings()
    settings["sources"] = [
        {
            "id": "x",
            "name": "Panel",
            "type": "xtream",
            "url": "https://panel",
            "username": "u",
            "password": "p",
        },
        {"id": "m", "name": "List", "type": "m3u", "url": "https://list"},
    ]
    cache.save_server_settings(settings)
    stream = FakeUpstreams()
    formats = b'{"user_info": {"allowed_output_formats": ["m3u8", "ts"]}}'
    api = "https://panel/player_api.php?username=u&password=p"
    web = FakeWeb({api: lambda: FakeResponse(formats), "https://panel/live/u/p/7.ts": stream})
    install_web(monkeypatch, web, util)
    remuxers = PopenRecorder()
    install_subprocess(monkeypatch, ffmpeg_session, popen=remuxers)
    url = "https://panel/live/u/p/7.m3u8"
    task = asyncio.create_task(ffmpeg_session.start_transcode(url, source_id="x"))
    try:
        await until(lambda: bool(stream.opened), task)
        stream.opened[0].feed(ts_source())
        await until(lambda: len(launcher.launches) == 1, task)
        media_playlist(output_dir(launcher, 0), "low", [1, 1])
        result = await task
        assert web.urls == [api, "https://panel/live/u/p/7.ts"]
        assert not remuxers.started
        session = ffmpeg_session.get_session(result["session_id"])
        assert session is not None
        # Clients, reuse, and casting keep the URL that was asked for.
        assert session["url"] == url
        await ffmpeg_session.stop_session(result["session_id"], force=True)
    finally:
        stream.close()


def test_hls_source_command_remuxes_to_stdout():
    url = "https://provider/live.m3u8"
    cmd = fast_start.source_command(url, "agent/1")
    assert cmd[cmd.index("-user_agent") + 1] == "agent/1"
    assert cmd[cmd.index("-i") - 8 : cmd.index("-i") + 2] == [
        "-f",
        "hls",
        "-extension_picky",
        "0",
        "-probesize",
        "5000000",
        "-analyzeduration",
        "1000000",
        "-i",
        url,
    ]
    assert cmd[-9:] == ["-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-f", "mpegts", "pipe:1"]
    assert "-reconnect" in cmd
    assert "-user_agent" not in fast_start.source_command(url, None)


@pytest.mark.asyncio
async def test_startup_cancellation_closes_the_upstream(
    tmp_path: pathlib.Path,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
):
    configure(tmp_path)
    task = asyncio.create_task(ffmpeg_session.start_transcode(PROVIDER))
    await until(lambda: bool(upstreams.opened), task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not launcher.launches
    assert upstreams.opened[0].closed
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_disconnected_startup_stops_waiting_and_cleans_up(
    tmp_path: pathlib.Path,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
):
    configure(tmp_path)

    async def disconnected() -> bool:
        return True

    task = asyncio.create_task(
        ffmpeg_session.start_transcode(PROVIDER, is_disconnected=disconnected),
    )
    await feed_source(upstreams, task)
    with pytest.raises(HTTPException) as error:
        await task
    assert error.value.status_code == 499
    assert len(launcher.launches) == 1
    assert launcher.processes[0].returncode is not None
    assert upstreams.opened[0].closed
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_worker_startup_failure_cleans_up(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    tasks: BackgroundTasks,
    upstreams: FakeUpstreams,
    caplog: pytest.LogCaptureFixture,
    cancelled: bool,
):
    del tasks
    configure(tmp_path)
    # A cancelled client leaves the worker running; otherwise it dies at startup.
    worker = FakeProcess(exit_code=None) if cancelled else FakeProcess(exited=True, exit_code=1)
    launcher = ProcessLauncher([worker])
    install_launcher(monkeypatch, launcher, ffmpeg_session)
    task = asyncio.create_task(ffmpeg_session.start_transcode(PROVIDER))
    await feed_source(upstreams, task)
    await until(lambda: len(launcher.launches) == 1, task)
    if cancelled:
        await polled_again(clock, task)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(HTTPException) as error:
            await task
        assert error.value.status_code == 500
        assert "(exit=1, upstream=open)" in caplog.text
    assert worker.returncode is not None
    assert upstreams.opened[0].closed
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_unreachable_upstream_fails_startup_and_reports_why(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    launcher: ProcessLauncher,
):
    configure(tmp_path)
    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode("https://provider/gone")
    assert error.value.status_code == 500
    assert "upstream=<urlopen error no route to https://provider/gone>" in caplog.text
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("restarting", "flags"),
    [
        (False, "delete_segments+temp_file"),
        (True, "delete_segments+temp_file+append_list+discont_start"),
    ],
)
def test_remux_command_copies_the_relay_into_the_dvr_window(
    tmp_path: pathlib.Path,
    restarting: bool,
    flags: str,
):
    configure(tmp_path, live_dvr_mins=30)
    command = (
        fast_start.remux_command("/s", restarting=True)
        if restarting
        else fast_start.remux_command("/s")
    )
    assert command == [
        "ffmpeg",
        "-copyts",
        "-hide_banner",
        "-loglevel",
        "error",
        "-analyzeduration",
        "800000",
        "-f",
        "mpegts",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c",
        "copy",
        "-max_delay",
        "0",
        "-f",
        "hls",
        "-hls_time",
        "1",
        "-hls_init_time",
        "0.25",
        "-hls_list_size",
        "1800",
        "-hls_segment_filename",
        "/s/low_%06d.ts",
        "-hls_flags",
        flags,
        "/s/low.m3u8",
    ]


@pytest.mark.asyncio
async def test_dvr_copyable_source_stream_copies_the_relay(
    tmp_path: pathlib.Path,
    programs: FakeRun,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
):
    """DVR of a copy-compatible source remuxes the relay's bytes; upstream is opened once."""
    configure(tmp_path, live_dvr_mins=60)
    programs.program("ffprobe", Program(stdout=ffprobe_source("h264")), when="-show_streams")
    task = asyncio.create_task(ffmpeg_session.start_transcode(PROVIDER))
    result = await serve_fast_start(launcher, upstreams, task)
    directory = output_dir(launcher, 0)
    assert result["playlist"] == f"/transcode/{result['session_id']}/low.m3u8"
    assert "master_playlist" in result
    [remux_cmd] = launcher.commands
    assert remux_cmd == fast_start.remux_command(str(directory))
    # The DVR window: 60 minutes of 1-second segments.
    assert remux_cmd[remux_cmd.index("-hls_list_size") + 1] == "3600"
    # Routing probes the relay's sample; nothing else opens the upstream URL.
    assert len(upstreams.opened) == 1
    assert programs.calls
    assert all(call[-1].endswith("/probe.ts") for call in programs.calls)
    await ffmpeg_session.stop_session(result["session_id"], force=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("video_codec", "bandwidth_saver", "live_dvr_mins"),
    [("hevc", False, 60), ("h264", True, 60), ("h264", False, 0)],
)
async def test_live_encodes_when_copy_is_unavailable_or_unneeded(
    tmp_path: pathlib.Path,
    programs: FakeRun,
    launcher: ProcessLauncher,
    upstreams: FakeUpstreams,
    video_codec: str,
    bandwidth_saver: bool,
    live_dvr_mins: int,
):
    """Incompatible sources, bandwidth saver, and DVR-off sessions keep the encoder."""
    configure(tmp_path, live_dvr_mins=live_dvr_mins)
    programs.program("ffprobe", Program(stdout=ffprobe_source(video_codec)), when="-show_streams")
    task = asyncio.create_task(
        ffmpeg_session.start_transcode(PROVIDER, bandwidth_saver=bandwidth_saver),
    )
    result = await serve_fast_start(launcher, upstreams, task)
    assert result["playlist"] == f"/transcode/{result['session_id']}/low.m3u8"
    [encoder_cmd] = launcher.commands
    assert encoder_cmd[encoder_cmd.index("-c:v") + 1] == "libx264"
    # One stream probe and one caption probe of the relay's sample.
    assert len(programs.calls) == 2
    assert all(call[-1].endswith("/probe.ts") for call in programs.calls)
    await ffmpeg_session.stop_session(result["session_id"], force=True)


def test_saver_session_keeps_full_quality_recovery_target(
    tmp_path: pathlib.Path,
    clock: FakeClock,
):
    playlist(tmp_path, "low")
    session = fast_session(tmp_path, clock.time(), bandwidth_saver=True, recovery_bitrate=50_000)
    ffmpeg_session._transcode_sessions["saver"] = session
    start = clock.now
    # Enough for the 720p saver output (7520 bps), not the full-quality target.
    good = PlaybackHealth(buffer_seconds=12, waiting=False, observed_bitrate=60_000)
    for offset in range(0, 120, 2):
        clock.now = start + offset
        result = ffmpeg_session.report_playback_health("saver", "u", good)
        assert result["bandwidth_saver"]
        assert "playlist" in result
        assert result["playlist"].endswith("/low.m3u8")
    assert "recovery_bitrate" in session
    assert session["recovery_bitrate"] == 50_000


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
