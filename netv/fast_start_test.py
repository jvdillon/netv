"""Fast start must keep one upstream reader and exactly one local encoder."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace

import asyncio
import os
import pathlib
import shutil
import subprocess

from fastapi import HTTPException

import pytest

from netv import cache, fast_start, ffmpeg_command, ffmpeg_session
from netv.cache import HwAccel
from netv.ffmpeg_command import MediaInfo
from netv.playback_policy import PlaybackHealth, PlaybackPolicy
from netv.testing import (
    BackgroundTasks,
    FakeClock,
    FakeProcess,
    FakeRun,
    ProcessLauncher,
    Program,
    install_clock,
    install_launcher,
    install_subprocess,
)


CAPTIONED_FRAMES = '{"frames": [{"side_data_list": [{"side_data_type": "Closed Captions"}]}]}'


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Each readiness poll advances fake time instead of sleeping 0.2s."""
    fake = FakeClock()
    install_clock(monkeypatch, fake, ffmpeg_session)
    return fake


@pytest.fixture(autouse=True)
def programs(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    """Fake segments carry no media, so ffprobe reports no streams unless scripted."""
    run = FakeRun()
    run.program("ffprobe", Program(stdout="{}"))
    install_subprocess(monkeypatch, ffmpeg_command, run=run)
    return run


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
    """Yield to ``task`` until ``condition`` holds, surfacing the task's failure."""
    for _ in range(100_000):
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


async def serve_fast_start(
    launcher: ProcessLauncher,
    task: asyncio.Task[ffmpeg_session.TranscodeResponse],
    *,
    source_duration: float = 4,
) -> ffmpeg_session.TranscodeResponse:
    """Publish what the fast-start ingest and encoder would write, then finish startup."""
    await until(lambda: len(launcher.launches) == 1, task)
    media_playlist(output_dir(launcher, 0), "input", [source_duration])
    await until(lambda: len(launcher.launches) == 2, task)
    media_playlist(output_dir(launcher, 0), "low", [source_duration, source_duration])
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


def video_packet(seconds: float) -> bytes:
    pts = int(seconds * 90_000)
    encoded = bytes(
        [
            0x21 | ((pts >> 29) & 14),
            (pts >> 22) & 255,
            ((pts >> 14) & 254) | 1,
            (pts >> 7) & 255,
            ((pts << 1) & 254) | 1,
        ],
    )
    pes = b"\x00\x00\x01\xe0\x00\x00\x80\x80\x05" + encoded
    return (b"\x47\x41\x00\x10" + pes).ljust(188, b"\xff")


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


def test_playlist_duration_sums_entries_and_tolerates_missing(tmp_path: pathlib.Path):
    playlist(tmp_path, "low")
    assert fast_start.playlist_duration(str(tmp_path), "low.m3u8") == 6
    assert fast_start.playlist_duration(str(tmp_path), "absent.m3u8") == 0


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


def test_live_buffers_follow_upstream_segment_cadence():
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


def test_commands_only_ingest_opens_upstream(tmp_path: pathlib.Path):
    url = "https://upstream.example/live.ts"
    ingest = fast_start.ingest_command(url, str(tmp_path), "neTV")
    assert ingest[ingest.index("-i") + 1] == url
    assert "-rw_timeout" not in ingest
    assert ingest[ingest.index("-reconnect_on_network_error") + 1] == "1"
    assert ingest[ingest.index("-analyzeduration") + 1] == "1000000"
    assert ingest[ingest.index("-probesize") + 1] == "5000000"
    assert ingest.index("-analyzeduration") < ingest.index("-i")
    assert "-fflags" not in ingest  # Do not drop probed packets with nobuffer.
    assert "-reconnect" not in fast_start.ingest_command("/tmp/source.ts", str(tmp_path), None)
    for upscale in (False, True):
        cmd = fast_start.encoder_command(
            str(tmp_path),
            "software",
            "1080p" if upscale else "720p",
            "low",
            False,
            allow_upscale=upscale,
        )
        assert url not in cmd
        assert cmd[-1] == f"{tmp_path}/low.m3u8"
        assert cmd[cmd.index("-hls_segment_filename") + 1] == f"{tmp_path}/low_%06d.ts"
        assert cmd[cmd.index("-i") + 1] == f"{tmp_path}/input.m3u8"
        assert "-reconnect" not in cmd
        assert "-copyts" in cmd
        assert cmd[cmd.index("-probesize") + 1] == "5000000"
        assert cmd[cmd.index("-analyzeduration") + 1] == "2000000"
        duration = float(cmd[cmd.index("-hls_time") + 1])
        assert duration * int(cmd[cmd.index("-hls_list_size") + 1]) >= 30


def test_ingest_allows_extensionless_hls_segments_only_for_hls(tmp_path: pathlib.Path):
    for url in (
        "https://upstream.example/live/u/p/1.m3u8",
        "https://upstream.example/live/u/p/1.M3U8?token=x",
    ):
        ingest = fast_start.ingest_command(url, str(tmp_path), None)
        assert ingest[ingest.index("-f") + 1] == "hls"
        assert ingest[ingest.index("-extension_picky") + 1] == "0"
        assert ingest.index("-extension_picky") < ingest.index("-i")
    # The option only exists on the HLS demuxer; raw TS would fail to open.
    ingest = fast_start.ingest_command("https://upstream.example/live/1.ts", str(tmp_path), None)
    assert "-extension_picky" not in ingest


def test_restart_commands_use_unique_sequences_and_discontinuities(tmp_path: pathlib.Path):
    ingest = fast_start.ingest_command(
        "https://upstream.example/live.m3u8",
        str(tmp_path),
        None,
        restarting=True,
    )
    assert ingest[ingest.index("-hls_start_number_source") + 1] == "epoch_us"
    assert "discont_start" in ingest[ingest.index("-hls_flags") + 1]

    encoder = fast_start.encoder_command(
        str(tmp_path),
        "software",
        "1080p",
        "high",
        False,
        allow_upscale=True,
        restarting=True,
    )
    assert encoder[encoder.index("-live_start_index") + 1] == "-3"
    assert "-hls_start_number_source" not in encoder
    flags = encoder[encoder.index("-hls_flags") + 1]
    assert "append_list" in flags
    assert "discont_start" in flags
    assert encoder[encoder.index("-i") + 1] == f"{tmp_path}/input.m3u8"


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
    aac = fast_start.encoder_command(*args, allow_upscale=upscale, audio=audio)
    assert aac[aac.index("-c:a") + 1] == "aac"
    assert aac[aac.index("-af") + 1] == "aformat=channel_layouts=5.1"
    dolby = fast_start.encoder_command(
        *args,
        allow_upscale=upscale,
        audio=audio,
        audio_passthrough=True,
    )
    assert dolby[dolby.index("-c:a") + 1] == "copy"
    assert dolby[dolby.index("-c:v") + 1] != "copy"


def test_ingest_can_warm_encoders_before_playback_is_ready(tmp_path: pathlib.Path):
    segment = tmp_path / "input_0.ts"
    segment.write_bytes(video_packet(10) * 10)
    path = tmp_path / "input.m3u8"
    path.write_text("#EXTM3U\n#EXTINF:4,\ninput_0.ts\n")
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
        cmd = fast_start.encoder_command(*args, allow_upscale=upscale, video_codec="h264")
        assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
        assert "hwupload_cuda" not in cmd[cmd.index("-vf") + 1]
        assert "-hwaccel" not in fast_start.encoder_command(*args, allow_upscale=upscale)
        assert "-hwaccel" not in fast_start.encoder_command(
            *args,
            allow_upscale=upscale,
            video_codec="hevc",
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


def test_health_learns_full_quality_bitrate_then_saver_recovers(
    tmp_path: pathlib.Path,
    clock: FakeClock,
):
    playlist(tmp_path, "low")
    policy = PlaybackPolicy()
    session = ffmpeg_session.TranscodeSession(
        dir=str(tmp_path),
        username="u",
        fast_start=True,
        process=FakeProcess(exit_code=None),
        playback_policy=policy,
        bandwidth_saver=False,
        started=clock.time(),
        url="http://stream.test/live",
    )
    ffmpeg_session._transcode_sessions["fast"] = session
    start = clock.now
    good = PlaybackHealth(buffer_seconds=10, waiting=False, observed_bitrate=20_000)
    result = ffmpeg_session.report_playback_health("fast", "u", good)
    assert result == {"bandwidth_saver": False, "playlist": "/transcode/fast/low.m3u8"}
    assert "recovery_bitrate" in session
    assert session["recovery_bitrate"] == 7520
    assert "playback_policy" in session
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
@pytest.mark.parametrize("source_duration", [4, 6, 30])
async def test_shared_session_runs_one_encoder(
    tmp_path: pathlib.Path,
    clock: FakeClock,
    programs: FakeRun,
    launcher: ProcessLauncher,
    source_duration: int,
    bandwidth_saver: bool,
    resolution: str,
    quality: str,
    bitrate: str,
    upscale: bool,
):
    configure(tmp_path)
    programs.program("ffprobe", Program(stdout=CAPTIONED_FRAMES), when="frame=side_data_list")
    url = "https://provider/live"
    task = asyncio.create_task(
        ffmpeg_session.start_transcode(url, fast_start=True, bandwidth_saver=bandwidth_saver),
    )
    await until(lambda: len(launcher.launches) == 1, task)
    directory = output_dir(launcher, 0)
    # The encoder warms on one complete source segment.
    media_playlist(directory, "input", [source_duration])
    await until(lambda: len(launcher.launches) == 2, task)
    # Encoder warm-up must not reduce the playback reserve, including for
    # sources whose long keyframe intervals require more than eight seconds:
    # playback needs two output segments covering it.
    reserve = 2 * source_duration
    for durations in ([reserve], [1, 1], [source_duration - 1, source_duration - 1]):
        media_playlist(directory, "low", durations)
        await polled_again(clock, task)
        assert not task.done()
    media_playlist(directory, "low", [source_duration, source_duration])
    result = await task

    assert len(launcher.launches) == 2
    ingest_cmd, encoder_cmd = launcher.commands
    assert url in ingest_cmd
    assert url not in encoder_cmd
    assert encoder_cmd[encoder_cmd.index("-b:v") + 1] == bitrate
    encoder = (str(directory), "software", resolution, quality, True)
    assert encoder_cmd == fast_start.encoder_command(*encoder, allow_upscale=upscale)
    assert result["playlist"].endswith("/low.m3u8")
    assert "master_playlist" in result
    assert result["master_playlist"].endswith("/master.m3u8")
    assert "caption_playlist" in result
    assert result["caption_playlist"].endswith("/low-captions.m3u8")
    expected_playback_buffer = min(60, max(12, 2 * source_duration))
    assert "playback_buffer_seconds" in result
    assert result["playback_buffer_seconds"] == expected_playback_buffer
    session = ffmpeg_session.get_session(result["session_id"])
    assert session is not None
    assert "playback_policy" in session
    assert session["playback_policy"].bandwidth_saver is bandwidth_saver
    assert "watchdog_stale_after" in session
    assert session["watchdog_stale_after"] == max(45, 2 * source_duration + 2)
    assert "playback_buffer_seconds" in session
    assert session["playback_buffer_seconds"] == expected_playback_buffer
    assert ffmpeg_session._is_fast_live(session)
    assert session.get("low_restart_command") == fast_start.encoder_command(
        *encoder,
        allow_upscale=upscale,
        restarting=True,
    )
    master = (directory / "master.m3u8").read_text()
    assert master == fast_start.master_playlist(resolution, audio_bitrate=384_000)
    assert 'INSTREAM-ID="CC1"' in (directory / "low-captions.m3u8").read_text()
    assert sorted(path.name for path in directory.iterdir()) == [
        "input.m3u8",
        "input_0.ts",
        "low-captions.m3u8",
        "low.m3u8",
        "low_0.ts",
        "low_1.ts",
        "master.m3u8",
    ]
    reused = await ffmpeg_session.start_transcode(url, fast_start=True)
    assert reused == result
    assert sum(url in cmd for cmd in launcher.commands) == 1
    await ffmpeg_session.stop_session(result["session_id"], force=True)
    assert all(proc.returncode is not None for proc in launcher.processes)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_startup_cancellation_cleans_ingest(
    tmp_path: pathlib.Path,
    clock: FakeClock,
    launcher: ProcessLauncher,
):
    configure(tmp_path)
    task = asyncio.create_task(
        ffmpeg_session.start_transcode("https://provider/cancel", fast_start=True),
    )
    await until(lambda: bool(clock.sleeps), task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(launcher.launches) == 1
    assert launcher.processes[0].returncode is not None
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_disconnected_startup_stops_waiting_and_cleans_ingest(
    tmp_path: pathlib.Path,
    launcher: ProcessLauncher,
):
    configure(tmp_path)

    async def disconnected() -> bool:
        return True

    with pytest.raises(HTTPException) as error:
        await ffmpeg_session.start_transcode(
            "https://provider/disconnected",
            fast_start=True,
            is_disconnected=disconnected,
        )
    assert error.value.status_code == 499
    assert len(launcher.launches) == 1
    assert launcher.processes[0].returncode is not None
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_encoder_startup_failure_cleans_up(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    tasks: BackgroundTasks,
    cancelled: bool,
):
    del tasks
    configure(tmp_path)
    # A cancelled client leaves the encoder running; otherwise it dies at startup.
    encoder = FakeProcess(exit_code=None) if cancelled else FakeProcess(exited=True, exit_code=1)
    launcher = ProcessLauncher([FakeProcess(exit_code=None), encoder])
    install_launcher(monkeypatch, launcher, ffmpeg_session)
    task = asyncio.create_task(
        ffmpeg_session.start_transcode("https://provider/failed-low", fast_start=True),
    )
    await until(lambda: len(launcher.launches) == 1, task)
    media_playlist(output_dir(launcher, 0), "input", [4])
    await until(lambda: len(launcher.launches) == 2, task)
    if cancelled:
        await polled_again(clock, task)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(HTTPException) as error:
            await task
        assert error.value.status_code == 500
    assert len(launcher.launches) == 2
    assert all(proc.returncode is not None for proc in launcher.processes)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("restarting", "start_index", "flags"),
    [
        (False, "0", "delete_segments+temp_file"),
        (True, "-3", "delete_segments+temp_file+append_list+discont_start"),
    ],
)
def test_remux_command_copies_input_into_the_dvr_window(
    tmp_path: pathlib.Path,
    restarting: bool,
    start_index: str,
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
        "-live_start_index",
        start_index,
        "-i",
        "/s/input.m3u8",
        "-map",
        "0",
        "-c",
        "copy",
        "-max_delay",
        "0",
        "-f",
        "hls",
        "-hls_time",
        "2",
        "-hls_list_size",
        "900",
        "-hls_segment_filename",
        "/s/low_%06d.ts",
        "-hls_flags",
        flags,
        "/s/low.m3u8",
    ]


@pytest.mark.asyncio
async def test_dvr_copyable_source_stream_copies_the_local_input(
    tmp_path: pathlib.Path,
    programs: FakeRun,
    launcher: ProcessLauncher,
):
    """DVR of a copy-compatible source remuxes the ingest output; upstream is opened once."""
    url = "https://provider/live"
    configure(tmp_path, live_dvr_mins=60)
    programs.program("ffprobe", Program(stdout=ffprobe_source("h264")), when="-show_streams")
    task = asyncio.create_task(ffmpeg_session.start_transcode(url, fast_start=True))
    result = await serve_fast_start(launcher, task)
    directory = output_dir(launcher, 0)
    assert result["playlist"] == f"/transcode/{result['session_id']}/low.m3u8"
    assert "master_playlist" in result
    ingest_cmd, remux_cmd = launcher.commands
    assert ingest_cmd[ingest_cmd.index("-i") + 1] == url
    assert remux_cmd == fast_start.remux_command(str(directory))
    assert remux_cmd[remux_cmd.index("-i") + 1] == f"{directory}/input.m3u8"
    assert remux_cmd[remux_cmd.index("-c") + 1] == "copy"
    # The DVR window: 60 minutes of 2-second segments.
    assert remux_cmd[remux_cmd.index("-hls_list_size") + 1] == "1800"
    session = ffmpeg_session.get_session(result["session_id"])
    assert session is not None
    assert session.get("low_restart_command") == fast_start.remux_command(
        str(directory),
        restarting=True,
    )
    # Routing reads the local segment; no probe opens the upstream URL.
    assert programs.calls
    assert all(call[-1].endswith("/input_0.ts") for call in programs.calls)
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
    video_codec: str,
    bandwidth_saver: bool,
    live_dvr_mins: int,
):
    """Incompatible sources, bandwidth saver, and DVR-off sessions keep the encoder."""
    url = "https://provider/live"
    configure(tmp_path, live_dvr_mins=live_dvr_mins)
    programs.program("ffprobe", Program(stdout=ffprobe_source(video_codec)), when="-show_streams")
    task = asyncio.create_task(
        ffmpeg_session.start_transcode(url, fast_start=True, bandwidth_saver=bandwidth_saver),
    )
    result = await serve_fast_start(launcher, task)
    assert result["playlist"] == f"/transcode/{result['session_id']}/low.m3u8"
    encoder_cmd = launcher.commands[1]
    assert encoder_cmd[encoder_cmd.index("-c:v") + 1] == "libx264"
    # One segment probe and one caption probe, both local.
    assert len(programs.calls) == 2
    assert all(call[-1].endswith("/input_0.ts") for call in programs.calls)
    await ffmpeg_session.stop_session(result["session_id"], force=True)


@pytest.mark.cli_ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires FFmpeg")
def test_real_remux_copies_ingest_onto_the_source_timeline(tmp_path: pathlib.Path):
    """The real remux command restates the ingest output without re-encoding or retiming."""
    source = tmp_path / "source.ts"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=25:duration=6",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=48000:duration=6",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "25",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    subprocess.run(fast_start.ingest_command(str(source), str(tmp_path), None), check=True)
    first_input = min(tmp_path.glob("input_*.ts"))
    result = subprocess.run(
        fast_start.remux_command(str(tmp_path)),
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert fast_start.playlist_duration(str(tmp_path), "low.m3u8") > 4
    first_low = min(tmp_path.glob("low_*.ts"))
    assert fast_start.segment_pts(first_low) == fast_start.segment_pts(first_input)
    # Copied, not re-encoded: the source's codecs and size survive.
    streams = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,height",
            "-of",
            "csv=p=0",
            str(first_low),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert streams == ["h264,180", "aac"]


@pytest.mark.cli_ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires FFmpeg")
def test_real_local_encoders_share_timestamps(tmp_path: pathlib.Path):
    """Exercise the actual generated commands, demuxing, and timestamp extraction."""
    source = tmp_path / "source.ts"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "6",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "25",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    subprocess.run(fast_start.ingest_command(str(source), str(tmp_path), None), check=True)
    process = subprocess.Popen(
        fast_start.encoder_command(
            str(tmp_path),
            "software",
            "720p",
            "low",
            False,
            allow_upscale=False,
        ),
        stderr=subprocess.PIPE,
    )
    try:
        _, error = process.communicate(timeout=30)
        assert process.returncode == 0, error.decode()
        assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") > 0
        first = (tmp_path / "low.m3u8").read_text().split("low_", 1)[1].split("\n", 1)[0]
        assert fast_start.segment_pts(tmp_path / f"low_{first}") is not None

        low_playlist = tmp_path / "low.m3u8"
        previous_segments = {
            line for line in low_playlist.read_text().splitlines() if line.endswith(".ts")
        }
        subprocess.run(
            fast_start.encoder_command(
                str(tmp_path),
                "software",
                "720p",
                "low",
                False,
                allow_upscale=False,
                restarting=True,
            ),
            check=True,
        )
        restarted = low_playlist.read_text()
        restarted_segments = {line for line in restarted.splitlines() if line.endswith(".ts")}
        assert previous_segments < restarted_segments
        assert "#EXT-X-DISCONTINUITY" in restarted
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


@pytest.mark.cli_ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires FFmpeg")
def test_real_encoder_probes_past_late_mpeg2_sequence_header(tmp_path: pathlib.Path):
    """HDHomeRun MPEG-2 joined mid-GOP: the first sequence header is >500KB in."""
    source = tmp_path / "source.ts"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            # Small frames padded to broadcast bitrate keep a 1 s GOP ~1.9 MB,
            # so the next sequence header still lands >500 KB past the cut.
            "testsrc2=size=320x180:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=48000",
            "-t",
            "4",
            "-c:v",
            "mpeg2video",
            "-b:v",
            "15M",
            "-minrate",
            "15M",
            "-maxrate",
            "15M",
            "-bufsize",
            "8M",
            "-g",
            "30",
            "-c:a",
            "ac3",
            str(source),
        ],
        check=True,
    )
    joined = tmp_path / "joined.ts"
    joined.write_bytes(source.read_bytes()[188 * 3000 :])
    subprocess.run(fast_start.ingest_command(str(joined), str(tmp_path), None), check=True)
    result = subprocess.run(
        fast_start.encoder_command(
            str(tmp_path),
            "software",
            "720p",
            "low",
            False,
            allow_upscale=False,
        ),
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert b"frames duplicated" not in result.stderr
    assert fast_start.playlist_duration(str(tmp_path), "low.m3u8") > 2


def test_saver_session_keeps_full_quality_recovery_target(
    tmp_path: pathlib.Path,
    clock: FakeClock,
):
    playlist(tmp_path, "low")
    session = ffmpeg_session.TranscodeSession(
        dir=str(tmp_path),
        username="u",
        fast_start=True,
        process=FakeProcess(exit_code=None),
        bandwidth_saver=True,
        playback_policy=PlaybackPolicy(bandwidth_saver=True),
        recovery_bitrate=50_000,
        started=clock.time(),
        url="http://stream.test/live",
    )
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
