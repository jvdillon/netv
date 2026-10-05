"""Fast start must keep one upstream reader and survive a failed high encoder."""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncio
import os
import pathlib
import shutil
import subprocess

import pytest

from ffmpeg_command import HwAccel, MediaInfo
from ffmpeg_session_test import FakeProcess
from playback_policy import PlaybackHealth, PlaybackPolicy

import fast_start
import ffmpeg_session


def video_packet(seconds):
    pts = int(seconds * 90000)
    encoded = bytes(
        [
            0x21 | ((pts >> 29) & 14),
            (pts >> 22) & 255,
            ((pts >> 14) & 254) | 1,
            (pts >> 7) & 255,
            ((pts << 1) & 254) | 1,
        ]
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
    playlist(tmp_path, "high")
    assert fast_start.aligned(str(tmp_path))
    assert fast_start.ready_bitrate(str(tmp_path), "high.m3u8") == 7520
    assert fast_start.segment_pts(tmp_path / "low_0.ts") == 10
    content = (tmp_path / "high.m3u8").read_text()
    dated = fast_start.dated_playlist(str(tmp_path), content, 10, 0)
    assert "#EXT-X-PROGRAM-DATE-TIME:1970-01-01T00:00:00.000+00:00" in dated
    playlist(tmp_path, "high", start=0)
    assert not fast_start.aligned(str(tmp_path))
    os.utime(tmp_path / "high.m3u8", (80, 80))
    with patch("fast_start.time.time", return_value=100):
        assert fast_start.ready_bitrate(str(tmp_path), "high.m3u8") == 0
        assert fast_start.ready_bitrate(str(tmp_path), "high.m3u8", max_age=30) == 7520
    (tmp_path / "low_2.ts").unlink()
    assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") == 0


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
    for high in (False, True):
        cmd = fast_start.encoder_command(
            str(tmp_path), "software", "1080p" if high else "720p", "low", False, high
        )
        assert url not in cmd
        assert cmd[cmd.index("-i") + 1] == f"{tmp_path}/input.m3u8"
        assert "-reconnect" not in cmd
        assert "-copyts" in cmd
        duration = float(cmd[cmd.index("-hls_time") + 1])
        assert duration * int(cmd[cmd.index("-hls_list_size") + 1]) >= 30


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
        True,
        restarting=True,
    )
    assert encoder[encoder.index("-live_start_index") + 1] == "-3"
    assert "-hls_start_number_source" not in encoder
    flags = encoder[encoder.index("-hls_flags") + 1]
    assert "append_list" in flags
    assert "discont_start" in flags
    assert encoder[encoder.index("-i") + 1] == f"{tmp_path}/input.m3u8"


def test_dated_playlist_replaces_encoder_dates_after_discontinuity(tmp_path):
    playlist(tmp_path, "high")
    content = (
        (tmp_path / "high.m3u8")
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
    (tmp_path / "high.m3u8").write_text(content)
    assert fast_start.ready_bitrate(str(tmp_path), "high.m3u8") == 7520


def test_discontinuity_keeps_shared_clock_when_pts_remain_continuous(tmp_path):
    playlist(tmp_path, "high")
    content = (
        (tmp_path / "high.m3u8")
        .read_text()
        .replace(
            "#EXTINF:2,\nhigh_1.ts",
            "#EXTINF:2,\n#EXT-X-DISCONTINUITY\nhigh_1.ts",
        )
    )
    os.utime(tmp_path / "high_1.ts", (32, 32))
    dated = fast_start.dated_playlist(str(tmp_path), content, 10, 0)
    assert "#EXT-X-PROGRAM-DATE-TIME:1970-01-01T00:00:02.000+00:00" in dated


@pytest.mark.parametrize("high", [False, True])
def test_encoders_keep_surround_audio(tmp_path, high):
    """Both renditions keep 5.1; Dolby is copied only for passthrough clients."""
    from ffmpeg_command import MediaInfo

    audio = MediaInfo(video_codec="", audio_codec="eac3", pix_fmt="", audio_channels=6,
                      audio_sample_rate=48000)  # fmt: skip
    args = (str(tmp_path), "software", "1080p" if high else "720p", "low", False, high)
    aac = fast_start.encoder_command(*args, audio=audio)
    assert aac[aac.index("-c:a") + 1] == "aac"
    assert aac[aac.index("-af") + 1] == "aformat=channel_layouts=5.1"
    dolby = fast_start.encoder_command(*args, audio=audio, audio_passthrough=True)
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
def test_upgrade_uses_bounded_bitrate(tmp_path, hardware: HwAccel, resolution, target, maximum):
    cmd = fast_start.encoder_command(str(tmp_path), hardware, resolution, "high", False, True)
    assert cmd[cmd.index("-b:v") + 1] == target
    assert cmd[cmd.index("-maxrate") + 1] == maximum
    assert cmd[cmd.index("-bufsize") + 1] == maximum
    assert not set(cmd) & {"-qp", "-qp_i", "-qp_p", "-global_quality", "-crf", "constqp", "cqp"}
    if hardware == "nvenc+software":
        assert cmd[cmd.index("-rc") + 1] == "vbr"
    low = fast_start.encoder_command(str(tmp_path), hardware, "720p", "low", False, False)
    assert low[low.index("-b:v") + 1] == "4000000"
    assert low[low.index("-maxrate") + 1] == "6000000"
    assert not set(low) & {"-qp", "-qp_i", "-qp_p", "-global_quality", "-crf", "constqp"}


def test_master_playlist_reserves_surround_audio_bandwidth():
    from ffmpeg_command import MediaInfo

    def audio(codec, channels):
        return MediaInfo(video_codec="", audio_codec=codec, pix_fmt="", audio_channels=channels)

    assert fast_start.surround_audio_bitrate(audio("aac", 2), True) == 0
    assert fast_start.surround_audio_bitrate(audio("aac", 6), False) == 384_000
    assert fast_start.surround_audio_bitrate(None, False) == 384_000
    assert fast_start.surround_audio_bitrate(audio("eac3", 6), False) == 384_000
    assert fast_start.surround_audio_bitrate(audio("eac3", 6), True) == 640_000
    content = fast_start.master_playlist("4k", include_high=False, audio_bitrate=640_000)
    assert "BANDWIDTH=7240000\nlow.m3u8" in content


def test_master_playlist_only_exposes_local_renditions():
    content = fast_start.master_playlist("4k", include_high=True)
    assert content == (
        "#EXTM3U\n#EXT-X-VERSION:3\n"
        "#EXT-X-STREAM-INF:BANDWIDTH=6600000\nlow.m3u8\n"
        "#EXT-X-STREAM-INF:BANDWIDTH=22000000\nhigh.m3u8\n"
    )
    assert "high.m3u8" not in fast_start.master_playlist("4k", include_high=False)


def test_upgrade_fallback_and_recovery_keep_same_encoder_and_session(tmp_path):
    playlist(tmp_path, "low")
    playlist(tmp_path, "high")
    high = FakeProcess()
    policy = PlaybackPolicy()
    session = dict(
        dir=str(tmp_path),
        username="u",
        fast_start=True,
        high_process=high,
        playback_policy=policy,
    )
    good = PlaybackHealth(buffer_seconds=10, waiting=False, observed_bitrate=20000)
    weak = PlaybackHealth(buffer_seconds=10, waiting=False, observed_bitrate=8000)
    with (
        patch.dict(ffmpeg_session._transcode_sessions, {"fast": session}),
        patch("ffmpeg_session.time.monotonic") as clock,
    ):
        for now in (0, 2, 4, 6):
            clock.return_value = now
            result = ffmpeg_session.report_playback_health("fast", "u", weak)
            assert result["playlist"].endswith("/low.m3u8")
        for now in (8, 10, 12):
            clock.return_value = now
            assert ffmpeg_session.report_playback_health("fast", "u", good)["playlist"].endswith(
                "/low.m3u8"
            )
        clock.return_value = 14
        assert ffmpeg_session.report_playback_health("fast", "u", good)["playlist"].endswith(
            "/high.m3u8"
        )
        with patch("ffmpeg_session.ready_bitrate", return_value=0):
            assert ffmpeg_session.report_playback_health("fast", "u", good)["playlist"].endswith(
                "/high.m3u8"
            )
        policy.bandwidth_saver = True
        clock.return_value = 16
        result = ffmpeg_session.report_playback_health("fast", "u", good)
        assert result["bandwidth_saver"]
        assert result["playlist"].endswith("/low.m3u8")
        assert high.returncode is None
        for now in range(18, 76, 2):
            clock.return_value = now
            result = ffmpeg_session.report_playback_health("fast", "u", good)
            assert result["bandwidth_saver"]
            assert result["playlist"].endswith("/low.m3u8")
        clock.return_value = 76
        result = ffmpeg_session.report_playback_health("fast", "u", good)
        assert not result["bandwidth_saver"]
        assert result["playlist"].endswith("/high.m3u8")
        assert session["high_process"] is high
        assert high.returncode is None


@pytest.mark.asyncio
@pytest.mark.parametrize("bandwidth_saver", [False, True])
@pytest.mark.parametrize("fail_high", [False, True])
@pytest.mark.parametrize("source_duration", [4, 6, 30])
async def test_shared_session_cleanup(tmp_path, fail_high, source_duration, bandwidth_saver):
    launched = []
    startup_ready = False
    durations = iter([source_duration, 2, 2 * source_duration - 2, 2 * source_duration])

    async def launch(*cmd, **kwargs):
        if len(launched) == 2:
            assert startup_ready, "High encoder must not compete with startup buffering"
        if fail_high and len(launched) == 2:
            raise OSError("no encoder")
        process = FakeProcess()
        launched.append((cmd, process))
        return process

    def ready(directory, name, *, minimum_segments=2):
        if name == "input.m3u8":
            assert minimum_segments == 1
            (pathlib.Path(directory) / "input_0.ts").write_bytes(video_packet(10))
            (pathlib.Path(directory) / name).write_text(
                f"#EXTM3U\n#EXTINF:{source_duration},\ninput_0.ts\n"
            )
        else:
            assert minimum_segments == 2
        return 10000

    def measure_duration(directory, name):
        nonlocal startup_ready
        value = next(durations)
        if name == "low.m3u8":
            assert len(launched) == 2
            startup_ready = value >= 2 * source_duration
        return value

    with (
        patch("ffmpeg_session.get_settings", return_value={"max_resolution": "4k"}),
        patch("ffmpeg_session.get_transcode_dir", return_value=tmp_path),
        patch("ffmpeg_session.asyncio.create_subprocess_exec", side_effect=launch),
        patch("ffmpeg_session._spawn_background_task", side_effect=lambda coro: coro.close()),
        patch("ffmpeg_session.ready_bitrate", side_effect=ready),
        # Encoder warm-up must not reduce the playback reserve, including for
        # sources whose long keyframe intervals require more than eight seconds.
        patch(
            "ffmpeg_session.playlist_duration",
            side_effect=measure_duration,
        ) as duration,
    ):
        result = await ffmpeg_session.start_transcode(
            "https://provider/live", fast_start=True, bandwidth_saver=bandwidth_saver
        )
        assert duration.call_count == 4
        assert len(launched) == (2 if fail_high else 3)
        assert result["playlist"].endswith("/low.m3u8")
        assert result["master_playlist"].endswith("/master.m3u8")
        session = ffmpeg_session.get_session(result["session_id"])
        assert session is not None
        assert session["playback_policy"].bandwidth_saver is bandwidth_saver
        assert session["watchdog_stale_after"] == max(45, 2 * source_duration + 2)
        master = pathlib.Path(session["dir"]) / "master.m3u8"
        assert ("high.m3u8" in master.read_text()) is not fail_high
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
            "ffmpeg_session.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)
        ) as launch,
        patch("ffmpeg_session._spawn_background_task", side_effect=lambda coro: coro.close()),
    ):
        with pytest.raises(ffmpeg_session.HTTPException) as error:
            await ffmpeg_session.start_transcode(
                "https://provider/disconnected", fast_start=True, is_disconnected=disconnected
            )
        assert error.value.status_code == 499
        assert launch.call_count == 1
        assert proc.returncode is not None
        assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_low_startup_failure_never_launches_high(tmp_path, cancelled):
    processes = [FakeProcess(), FakeProcess()]

    def ready(directory, name, **kwargs):
        if name == "input.m3u8":
            path = pathlib.Path(directory)
            (path / "input_0.ts").write_bytes(video_packet(10) * 10)
            (path / name).write_text("#EXTM3U\n#EXTINF:4,\ninput_0.ts\n")
            return 10000
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
        expected = asyncio.CancelledError if cancelled else ffmpeg_session.HTTPException
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
        audio_sample_rate=48000,
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
                audio_sample_rate=48000,
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
                audio_sample_rate=48000,
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
            "https://provider/live", fast_start=True, bandwidth_saver=bandwidth_saver
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
    processes = [
        subprocess.Popen(
            fast_start.encoder_command(str(tmp_path), "software", "720p", "low", False, high),
            stderr=subprocess.PIPE,
        )
        for high in (False, True)
    ]
    try:
        for process in processes:
            _, error = process.communicate(timeout=30)
            assert process.returncode == 0, error.decode()
        assert fast_start.ready_bitrate(str(tmp_path), "low.m3u8") > 0
        assert fast_start.ready_bitrate(str(tmp_path), "high.m3u8") > 0
        assert fast_start.aligned(str(tmp_path))

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
                False,
                restarting=True,
            ),
            check=True,
        )
        restarted = low_playlist.read_text()
        restarted_segments = {line for line in restarted.splitlines() if line.endswith(".ts")}
        assert previous_segments < restarted_segments
        assert "#EXT-X-DISCONTINUITY" in restarted
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()


def test_saver_does_not_recover_when_high_encoder_has_failed(tmp_path):
    playlist(tmp_path, "low")
    playlist(tmp_path, "high")
    high = FakeProcess()
    high.returncode = 1
    session = dict(
        dir=str(tmp_path),
        username="u",
        fast_start=True,
        high_process=high,
        playback_policy=PlaybackPolicy(bandwidth_saver=True),
        recovery_bitrate=7520,
    )
    good = PlaybackHealth(buffer_seconds=12, waiting=False, observed_bitrate=100000)
    with (
        patch.dict(ffmpeg_session._transcode_sessions, {"failed": session}),
        patch("ffmpeg_session.time.monotonic") as clock,
    ):
        for now in range(0, 120, 2):
            clock.return_value = now
            result = ffmpeg_session.report_playback_health("failed", "u", good)
            assert result["bandwidth_saver"]
            assert result["playlist"].endswith("/low.m3u8")
