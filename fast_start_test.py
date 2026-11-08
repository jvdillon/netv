"""Fast start must keep one upstream reader and exactly one local encoder."""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncio
import os
import pathlib
import shutil
import subprocess

from fastapi import HTTPException

import pytest

from ffmpeg_command import HwAccel, MediaInfo
from ffmpeg_session_test import FakeProcess
from playback_policy import PlaybackHealth, PlaybackPolicy

import fast_start
import ffmpeg_session


def video_packet(seconds):
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


def playlist(directory, name, start=10):
    lines = ["#EXTM3U", "#EXT-X-TARGETDURATION:2"]
    for i in range(3):
        filename = f"{name}_{i}.ts"
        (directory / filename).write_bytes(video_packet(start + i * 2) * 10)
        lines += ["#EXTINF:2,", filename]
    (directory / f"{name}.m3u8").write_text("\n".join(lines) + "\n")


def test_timeline_and_readiness(tmp_path):
    playlist(tmp_path, "low")
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 7520
    assert fast_start.segment_pts(tmp_path / "low_0.ts") == 10
    content = (tmp_path / "low.m3u8").read_text()
    dated = fast_start.dated_playlist(str(tmp_path), content, 10, 0)
    assert "#EXT-X-PROGRAM-DATE-TIME:1970-01-01T00:00:00.000+00:00" in dated
    os.utime(tmp_path / "low.m3u8", (80, 80))
    with patch("fast_start.time.time", return_value=100):
        assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 0
        assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8", max_age=30) == 7520
    (tmp_path / "low_2.ts").unlink()
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 0


def test_playlist_duration_sums_entries_and_tolerates_missing(tmp_path):
    playlist(tmp_path, "low")
    assert fast_start.playlist_duration(str(tmp_path), "low.m3u8") == 6
    assert fast_start.playlist_duration(str(tmp_path), "absent.m3u8") == 0


def test_segment_pts_skips_non_video_packets(tmp_path):
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


def test_dated_playlist_keeps_clock_when_discontinuous_segment_vanishes(tmp_path):
    content = "#EXTM3U\n#EXT-X-DISCONTINUITY\n#EXTINF:2,\ngone.ts\n"
    with patch("fast_start.segment_pts", return_value=12):
        dated = fast_start.dated_playlist(str(tmp_path), content, 10, 0)
    assert "#EXT-X-PROGRAM-DATE-TIME:1970-01-01T00:00:02.000+00:00" in dated


def test_ready_bitrate_rejects_unsafe_or_empty_entries(tmp_path):
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


def test_commands_only_ingest_opens_upstream(tmp_path):
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


def test_ingest_allows_extensionless_hls_segments_only_for_hls(tmp_path):
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


def test_restart_commands_use_unique_sequences_and_discontinuities(tmp_path):
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


def test_dated_playlist_replaces_encoder_dates_after_discontinuity(tmp_path):
    playlist(tmp_path, "low")
    content = (
        (tmp_path / "low.m3u8")
        .read_text()
        .replace(
            "#EXTINF:2,\n",
            "#EXTINF:2,\n#EXT-X-DISCONTINUITY\n#EXT-X-PROGRAM-DATE-TIME:2030-01-01T00:00:00Z\n",
        )
    )
    dated = fast_start.dated_playlist(str(tmp_path), content, 10, 0)
    assert dated.count("#EXT-X-PROGRAM-DATE-TIME:") == 3
    assert "2030-" not in dated
    assert fast_start.dated_playlist(str(tmp_path), dated, 10, 0) == dated
    (tmp_path / "low.m3u8").write_text(content)
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 7520


def test_discontinuity_keeps_shared_clock_when_pts_remain_continuous(tmp_path):
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
    dated = fast_start.dated_playlist(str(tmp_path), content, 10, 0)
    assert "#EXT-X-PROGRAM-DATE-TIME:1970-01-01T00:00:02.000+00:00" in dated


@pytest.mark.parametrize("upscale", [False, True])
def test_encoder_keeps_surround_audio(tmp_path, upscale):
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


def test_ingest_can_warm_encoders_before_playback_is_ready(tmp_path):
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
        ("1440p", "10000000", "14000000"),
        ("4k", "16000000", "20000000"),
    ],
)
def test_encoder_uses_bounded_bitrate(tmp_path, hardware: HwAccel, resolution, target, maximum):
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


def test_encoders_decode_on_gpu_when_source_codec_is_known(tmp_path):
    with patch("ffmpeg_command._get_gpu_nvdec_codecs", return_value={"h264"}):
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

    def audio(codec, channels):
        return MediaInfo(video_codec="", audio_codec=codec, pix_fmt="", audio_channels=channels)

    assert fast_start.surround_audio_bitrate(audio("aac", 2), True) == 0
    assert fast_start.surround_audio_bitrate(audio("aac", 6), False) == 384_000
    assert fast_start.surround_audio_bitrate(None, False) == 384_000
    assert fast_start.surround_audio_bitrate(audio("eac3", 6), False) == 384_000
    assert fast_start.surround_audio_bitrate(audio("eac3", 6), True) == 640_000
    content = fast_start.master_playlist("720p", audio_bitrate=640_000)
    assert "BANDWIDTH=7240000\nlow.m3u8" in content


@pytest.mark.parametrize("resolution,bandwidth", [("720p", 6_600_000), ("4k", 22_000_000)])
def test_master_playlist_labels_the_single_rendition_at_encoder_resolution(
    resolution,
    bandwidth,
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


def test_health_learns_full_quality_bitrate_then_saver_recovers(tmp_path):
    playlist(tmp_path, "low")
    policy = PlaybackPolicy()
    session = dict(
        dir=str(tmp_path),
        username="u",
        fast_start=True,
        process=FakeProcess(),
        playback_policy=policy,
        bandwidth_saver=False,
    )
    good = PlaybackHealth(buffer_seconds=10, waiting=False, observed_bitrate=20_000)
    with (
        patch.dict(ffmpeg_session._transcode_sessions, {"fast": session}),
        patch("ffmpeg_session.time.monotonic") as clock,
    ):
        clock.return_value = 0
        result = ffmpeg_session.report_playback_health("fast", "u", good)
        assert result == {"bandwidth_saver": False, "playlist": "/transcode/fast/low.m3u8"}
        assert session["recovery_bitrate"] == 7520
        policy.bandwidth_saver = True
        for now in range(2, 62, 2):
            clock.return_value = now
            result = ffmpeg_session.report_playback_health("fast", "u", good)
            # The full-quality session cannot serve saver: clients must retune.
            assert result == {"bandwidth_saver": True}
        clock.return_value = 62
        result = ffmpeg_session.report_playback_health("fast", "u", good)
        assert result == {"bandwidth_saver": False, "playlist": "/transcode/fast/low.m3u8"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bandwidth_saver", "resolution", "bitrate", "upscale"),
    [(False, "4k", "16000000", True), (True, "720p", "4000000", False)],
)
@pytest.mark.parametrize("source_duration", [4, 6, 30])
async def test_shared_session_runs_one_encoder(
    tmp_path,
    source_duration,
    bandwidth_saver,
    resolution,
    bitrate,
    upscale,
):
    launched = []
    durations = iter([source_duration, 2, 2 * source_duration - 2, 2 * source_duration])

    async def launch(*cmd, **kwargs):
        process = FakeProcess()
        launched.append((cmd, process))
        return process

    def ready(directory, name, *, minimum_segments=2):
        if name == "input.m3u8":
            assert minimum_segments == 1
            (pathlib.Path(directory) / "input_0.ts").write_bytes(video_packet(10))
            (pathlib.Path(directory) / name).write_text(
                f"#EXTM3U\n#EXTINF:{source_duration},\ninput_0.ts\n",
            )
        else:
            assert minimum_segments == 2
        return 10_000

    def measure_duration(directory, name):
        if name == "low.m3u8":
            assert len(launched) == 2
        return next(durations)

    with (
        patch("ffmpeg_session.get_settings", return_value={"max_resolution": "4k"}),
        patch("ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch("ffmpeg_session.asyncio.create_subprocess_exec", side_effect=launch),
        patch("ffmpeg_session.probe_audio", return_value=None),
        patch(
            "fast_start.build_hls_ffmpeg_cmd",
            wraps=fast_start.build_hls_ffmpeg_cmd,
        ) as build,
        patch("ffmpeg_session._spawn_background_task", side_effect=lambda coro: coro.close()),
        patch("ffmpeg_session.probe_closed_captions", return_value=True),
        patch("ffmpeg_session.ready_bitrate", side_effect=ready),
        # Encoder warm-up must not reduce the playback reserve, including for
        # sources whose long keyframe intervals require more than eight seconds.
        patch(
            "ffmpeg_session.playlist_duration",
            side_effect=measure_duration,
        ) as duration,
    ):
        result = await ffmpeg_session.start_transcode(
            "https://provider/live",
            fast_start=True,
            bandwidth_saver=bandwidth_saver,
        )
        assert duration.call_count == 4
        assert len(launched) == 2
        (ingest_cmd, _), (encoder_cmd, _) = launched
        assert "https://provider/live" in ingest_cmd
        assert encoder_cmd[encoder_cmd.index("-b:v") + 1] == bitrate
        assert encoder_cmd[-1].endswith("/low.m3u8")
        # Launched encoder plus its stored restart command.
        assert {(call.args[6], call.kwargs["allow_upscale"]) for call in build.call_args_list} == {
            (resolution, upscale),
        }
        assert result["playlist"].endswith("/low.m3u8")
        assert result["master_playlist"].endswith("/master.m3u8")
        assert result["caption_playlist"].endswith("/low-captions.m3u8")
        expected_playback_buffer = min(60, max(12, 2 * source_duration))
        assert result["playback_buffer_seconds"] == expected_playback_buffer
        session = ffmpeg_session.get_session(result["session_id"])
        assert session is not None
        assert session["playback_policy"].bandwidth_saver is bandwidth_saver
        assert session["watchdog_stale_after"] == max(45, 2 * source_duration + 2)
        assert session["playback_buffer_seconds"] == expected_playback_buffer
        master = (pathlib.Path(session["dir"]) / "master.m3u8").read_text()
        assert master == fast_start.master_playlist(resolution, audio_bitrate=384_000)
        low_captions = pathlib.Path(session["dir"]) / "low-captions.m3u8"
        assert 'INSTREAM-ID="CC1"' in low_captions.read_text()
        assert sorted(path.name for path in pathlib.Path(session["dir"]).iterdir()) == [
            "input.m3u8",
            "input_0.ts",
            "low-captions.m3u8",
            "master.m3u8",
        ]
        reused = await ffmpeg_session.start_transcode("https://provider/live", fast_start=True)
        assert reused == result
        assert sum("https://provider/live" in cmd for cmd, _ in launched) == 1
        ffmpeg_session.stop_session(result["session_id"], force=True)
        assert all(proc.returncode is not None for _, proc in launched)
        assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_startup_cancellation_cleans_ingest(tmp_path):
    proc = FakeProcess()
    with (
        patch("ffmpeg_session.get_settings", return_value={"max_resolution": "4k"}),
        patch("ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch("ffmpeg_session.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch("ffmpeg_session._spawn_background_task", side_effect=lambda coro: coro.close()),
        patch("ffmpeg_session.ready_bitrate", side_effect=asyncio.CancelledError),
    ):
        with pytest.raises(asyncio.CancelledError):
            await ffmpeg_session.start_transcode("https://provider/cancel", fast_start=True)
        assert proc.returncode is not None
        assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_disconnected_startup_stops_waiting_and_cleans_ingest(tmp_path):
    proc = FakeProcess()
    disconnected = AsyncMock(return_value=True)
    with (
        patch("ffmpeg_session.get_settings", return_value={"max_resolution": "4k"}),
        patch("ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch(
            "ffmpeg_session.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ) as launch,
        patch("ffmpeg_session._spawn_background_task", side_effect=lambda coro: coro.close()),
    ):
        with pytest.raises(HTTPException) as error:
            await ffmpeg_session.start_transcode(
                "https://provider/disconnected",
                fast_start=True,
                is_disconnected=disconnected,
            )
        assert error.value.status_code == 499
        assert launch.call_count == 1
        assert proc.returncode is not None
        assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_encoder_startup_failure_cleans_up(tmp_path, cancelled):
    processes = [FakeProcess(), FakeProcess()]

    def ready(directory, name, **kwargs):
        if name == "input.m3u8":
            path = pathlib.Path(directory)
            (path / "input_0.ts").write_bytes(video_packet(10) * 10)
            (path / name).write_text("#EXTM3U\n#EXTINF:4,\ninput_0.ts\n")
            return 10_000
        if cancelled:
            raise asyncio.CancelledError
        processes[1].returncode = 1
        return 0

    with (
        patch("ffmpeg_session.get_settings", return_value={"max_resolution": "4k"}),
        patch("ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch(
            "ffmpeg_session.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=processes),
        ) as launch,
        patch("ffmpeg_session._spawn_background_task", side_effect=lambda coro: coro.close()),
        patch("ffmpeg_session.ready_bitrate", side_effect=ready),
    ):
        expected = asyncio.CancelledError if cancelled else HTTPException
        with pytest.raises(expected):
            await ffmpeg_session.start_transcode("https://provider/failed-low", fast_start=True)
        assert launch.call_count == 2
        assert all(proc.returncode is not None for proc in processes)
        assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_dvr_compatible_source_remuxes_without_fast_start():
    """A DVR request for a copy-compatible source must not spawn encoders."""
    media = MediaInfo(
        video_codec="h264",
        audio_codec="aac",
        pix_fmt="yuv420p",
        audio_channels=2,
        audio_sample_rate=48_000,
        audio_profile="LC",
        height=1080,
    )
    remux = {"session_id": "remux", "playlist": "/transcode/remux/stream.m3u8"}
    with (
        patch(
            "ffmpeg_session.get_settings",
            return_value={"max_resolution": "4k", "live_dvr_mins": 60, "probe_live": True},
        ),
        patch("ffmpeg_session.resolve_hls_master_playlist", return_value="https://provider/live"),
        patch("ffmpeg_session.probe_media", return_value=(media, [])),
        patch("ffmpeg_session._do_start_transcode", new=AsyncMock(return_value=remux)) as do_start,
        patch("ffmpeg_session._start_fast_live", new=AsyncMock()) as fast_live,
    ):
        result = await ffmpeg_session.start_transcode("https://provider/live", fast_start=True)
    assert result == remux
    assert do_start.await_count == 1
    assert fast_live.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("media", "bandwidth_saver"),
    [
        (
            MediaInfo(
                video_codec="hevc",
                audio_codec="aac",
                pix_fmt="yuv420p",
                audio_channels=2,
                audio_sample_rate=48_000,
                audio_profile="LC",
                height=1080,
            ),
            False,
        ),
        (
            MediaInfo(
                video_codec="h264",
                audio_codec="aac",
                pix_fmt="yuv420p",
                audio_channels=2,
                audio_sample_rate=48_000,
                audio_profile="LC",
                height=1080,
            ),
            True,
        ),
    ],
)
async def test_dvr_keeps_fast_start_when_remux_is_unavailable(media, bandwidth_saver):
    """Incompatible sources and bandwidth-saver requests keep fast start."""
    fast = {"session_id": "fast", "playlist": "/transcode/fast/low.m3u8"}
    with (
        patch(
            "ffmpeg_session.get_settings",
            return_value={"max_resolution": "4k", "live_dvr_mins": 60, "probe_live": True},
        ),
        patch("ffmpeg_session.resolve_hls_master_playlist", return_value="https://provider/live"),
        patch("ffmpeg_session.probe_media", return_value=(media, [])),
        patch("ffmpeg_session._start_fast_live", new=AsyncMock(return_value=fast)) as fast_live,
    ):
        result = await ffmpeg_session.start_transcode(
            "https://provider/live",
            fast_start=True,
            bandwidth_saver=bandwidth_saver,
        )
    assert result == fast
    assert fast_live.await_count == 1


@pytest.mark.asyncio
async def test_dvr_disabled_skips_probe_and_uses_fast_start():
    """Without DVR the fast-start path must not pay a probe round trip."""
    fast = {"session_id": "fast", "playlist": "/transcode/fast/low.m3u8"}
    with (
        patch(
            "ffmpeg_session.get_settings",
            return_value={"max_resolution": "4k", "live_dvr_mins": 0, "probe_live": True},
        ),
        patch("ffmpeg_session.probe_media", new=MagicMock()) as probe,
        patch("ffmpeg_session._start_fast_live", new=AsyncMock(return_value=fast)) as fast_live,
    ):
        result = await ffmpeg_session.start_transcode("https://provider/live", fast_start=True)
    assert result == fast
    assert fast_live.await_count == 1
    assert probe.call_count == 0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_real_local_encoders_share_timestamps(tmp_path):
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
            "12",
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


def test_real_encoder_probes_past_late_mpeg2_sequence_header(tmp_path):
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
            "testsrc2=size=1280x720:rate=60",
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
            "60",
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


def test_saver_session_keeps_full_quality_recovery_target(tmp_path):
    playlist(tmp_path, "low")
    session = dict(
        dir=str(tmp_path),
        username="u",
        fast_start=True,
        process=FakeProcess(),
        bandwidth_saver=True,
        playback_policy=PlaybackPolicy(bandwidth_saver=True),
        recovery_bitrate=50_000,
    )
    # Enough for the 720p saver output (7520 bps), not the full-quality target.
    good = PlaybackHealth(buffer_seconds=12, waiting=False, observed_bitrate=60_000)
    with (
        patch.dict(ffmpeg_session._transcode_sessions, {"saver": session}),
        patch("ffmpeg_session.time.monotonic") as clock,
    ):
        for now in range(0, 120, 2):
            clock.return_value = now
            result = ffmpeg_session.report_playback_health("saver", "u", good)
            assert result["bandwidth_saver"]
            assert result["playlist"].endswith("/low.m3u8")
    assert session["recovery_bitrate"] == 50_000


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
