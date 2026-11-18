"""Tests for ffmpeg command generation and media probing."""

from pathlib import Path

import json
import subprocess
import tempfile
import time

import pytest

from netv import cache, ffmpeg_command
from netv.cache import HwAccel, ServerSettings, get_user_agent
from netv.ffmpeg_command import (
    MediaInfo,
    SubtitleStream,
    _build_audio_args,
    _build_sr_filter,
    _build_video_args,
    _find_sr_engine,
    _get_gpu_nvdec_codecs,
    build_hls_ffmpeg_cmd,
    can_remux_live,
    clear_all_probe_cache,
    clear_series_mru,
    get_live_hls_list_size,
    get_series_probe_cache_stats,
    get_transcode_dir,
    http_reconnect_args,
    invalidate_series_probe_cache,
    limit_live_video_bitrate,
    probe_closed_captions,
    probe_media,
    probe_segment,
    resolve_hls_master_playlist,
    restore_probe_cache_entry,
)
from netv.testing import FakeResponse, FakeRun, FakeWeb, Program, install_subprocess, install_web


@pytest.fixture(autouse=True)
def command_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with empty probe caches, the real settings loader, and a VAAPI device."""
    monkeypatch.setattr(cache, "VAAPI_DEVICE", "/dev/dri/renderD128")
    monkeypatch.setattr(ffmpeg_command, "_load_settings", cache.load_server_settings)
    monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", None)
    monkeypatch.setattr(ffmpeg_command, "_probe_cache", {})
    monkeypatch.setattr(ffmpeg_command, "_series_probe_cache", {})
    monkeypatch.setattr(ffmpeg_command, "_gpu_nvdec_codecs", None)
    monkeypatch.setattr(ffmpeg_command, "_has_libplacebo", None)


@pytest.fixture(autouse=True)
def fake_run(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> FakeRun:
    """Serve ffprobe, ffmpeg, and nvidia-smi from scripts, except in real-ffmpeg tests."""
    run = FakeRun()
    if request.node.get_closest_marker("cli_ffmpeg") is None:
        install_subprocess(monkeypatch, ffmpeg_command, run=run)
    return run


def _save_dvr(minutes: int) -> None:
    cache.save_server_settings({**cache.load_server_settings(), "live_dvr_mins": minutes})


def _enable_sr(monkeypatch: pytest.MonkeyPatch, engine_dir: Path, *, model: str = "on") -> Path:
    """Install one 720p AI Upscale engine and select a model; return the engine path."""
    engine = engine_dir / "2x-nomosuni-compact_720p_fp16.engine"
    engine.write_bytes(b"engine")
    monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", str(engine_dir))
    cache.save_server_settings({**cache.load_server_settings(), "sr_model": model})
    return engine


def _script_libplacebo(run: FakeRun, *, available: bool) -> None:
    run.program(
        "ffmpeg",
        Program(stdout="... libplacebo ..." if available else ""),
        when="-filters",
    )


def _ffprobe(
    run: FakeRun,
    payload: object,
    *,
    returncode: int = 0,
    when: str | None = None,
) -> None:
    run.program("ffprobe", Program(stdout=json.dumps(payload), returncode=returncode), when=when)


def _between(cmd: list[str], start: str, end: str) -> list[str]:
    return cmd[cmd.index(start) + 1 : cmd.index(end)]


# =============================================================================
# Video Args Tests
# =============================================================================


class TestBuildVideoArgs:
    """Tests for _build_video_args."""

    @pytest.mark.parametrize(
        "hw",
        ["nvenc+vaapi", "nvenc+software", "amf+vaapi", "amf+software", "qsv", "vaapi", "software"],
    )
    @pytest.mark.parametrize("deinterlace", [True, False])
    @pytest.mark.parametrize("max_resolution", ["1080p", "720p", "4k"])
    def test_all_hw_combinations(self, hw: HwAccel, deinterlace: bool, max_resolution: str):
        """Test all hardware/deinterlace/resolution combinations produce valid args."""
        pre, post = _build_video_args(
            copy_video=False,
            hw=hw,
            deinterlace=deinterlace,
            use_hw_pipeline=(hw not in ("software", "nvenc+software", "amf+software")),
            max_resolution=max_resolution,
            quality="high",
        )

        if hw in ("nvenc+vaapi", "nvenc+software") or hw in ("amf+vaapi", "amf+software"):
            assert pre == [] or "-hwaccel" in pre
        elif hw == "qsv":
            assert "-hwaccel" in pre
            assert "qsv" in pre
        elif hw == "vaapi":
            assert "-hwaccel" in pre
            assert "vaapi" in pre
        else:
            assert pre == []

        assert "-vf" in post
        assert "-c:v" in post
        assert "-g" in post
        assert "60" in post

    @pytest.mark.parametrize(
        "hw",
        ["nvenc+vaapi", "nvenc+software", "amf+vaapi", "amf+software", "qsv", "vaapi", "software"],
    )
    def test_copy_video(self, hw: HwAccel):
        """Test copy_video returns minimal args."""
        pre, post = _build_video_args(
            copy_video=True,
            hw=hw,
            deinterlace=False,
            use_hw_pipeline=False,
            max_resolution="1080p",
            quality="high",
        )
        assert pre == []
        assert post == ["-c:v", "copy"]

    def test_nvenc_hw_pipeline_filters(self):
        """Test NVENC with hw pipeline uses CUDA filters."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="nvenc+software",
            deinterlace=True,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
        )
        assert "-hwaccel" in pre
        vf = post[post.index("-vf") + 1]
        assert "yadif_cuda" in vf
        assert "scale_cuda" in vf

    def test_nvenc_sw_fallback_filters(self):
        """Test NVENC without hw pipeline uses SW decode + GPU processing."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="nvenc+software",
            deinterlace=True,
            use_hw_pipeline=False,
            max_resolution="1080p",
            quality="high",
        )
        assert pre == []
        vf = post[post.index("-vf") + 1]
        # Upload to GPU, then deinterlace (mode=0 for original framerate) and scale on GPU.
        assert "hwupload_cuda" in vf
        assert "yadif_cuda=0" in vf
        assert "scale_cuda" in vf

    def test_nvenc_sr_uses_nomos(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        engine = _enable_sr(monkeypatch, tmp_path, model="retired-model")
        retired_engine = tmp_path / "retired-model_720p_fp16.engine"
        retired_engine.write_bytes(b"engine")

        pre, post = _build_video_args(
            copy_video=False,
            hw="nvenc+software",
            deinterlace=False,
            use_hw_pipeline=False,
            max_resolution="4k",
            quality="high",
            source_height=720,
        )

        assert pre[-2:] == ["-filter_threads", "4"]
        assert post[post.index("-preset") + 1] == "p5"
        assert post[post.index("-bf") + 1] == "0"
        assert "-rc-lookahead" not in post
        video_filter = post[post.index("-vf") + 1]
        assert f"dnn_processing=dnn_backend=tensorrt:model={engine}" in video_filter
        assert str(retired_engine) not in video_filter

    @pytest.mark.parametrize(
        ("source_height", "expected_height"),
        [(480, 480), (576, 720), (720, 720), (1080, 1080)],
    )
    def test_sr_selects_fixed_nomos_engine(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        source_height: int,
        expected_height: int,
    ):
        for height in (480, 720, 1080):
            (tmp_path / f"2x-nomosuni-compact_{height}p_fp16.engine").write_bytes(b"engine")
        (tmp_path / "retired-model_576p_fp16.engine").write_bytes(b"engine")
        monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", str(tmp_path))

        result = _find_sr_engine(source_height)

        assert result is not None
        engine_path, engine_height, _, scale = result
        assert engine_path.endswith(f"2x-nomosuni-compact_{expected_height}p_fp16.engine")
        assert engine_height == expected_height
        assert scale == 2

    def test_sr_maps_enabled_retired_setting_to_nomos(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        engine = _enable_sr(monkeypatch, tmp_path, model="retired-model")

        video_filter = _build_sr_filter(1080, 2160)

        assert f"model={engine}" in video_filter

    def test_vaapi_filters(self):
        """Test VAAPI uses VAAPI filters."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="vaapi",
            deinterlace=True,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
        )
        vf = post[post.index("-vf") + 1]
        assert "deinterlace_vaapi=auto=1," in vf
        assert "scale_vaapi" in vf

    def test_qsv_filters(self):
        """Test QSV uses QSV filters."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="qsv",
            deinterlace=True,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
        )
        vf = post[post.index("-vf") + 1]
        assert "vpp_qsv" in vf
        assert "scale_qsv" in vf

    def test_software_filters(self):
        """Test software uses yadif (mode=0 for original framerate) and scale."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="software",
            deinterlace=True,
            use_hw_pipeline=False,
            max_resolution="1080p",
            quality="high",
        )
        assert pre == []
        vf = post[post.index("-vf") + 1]
        assert "yadif=0" in vf

    @pytest.mark.parametrize(
        "quality,expected_qp",
        [("high", "20"), ("medium", "28"), ("low", "35")],
    )
    def test_quality_presets(self, quality: str, expected_qp: str):
        """Test quality presets map to correct QP values."""
        _, post = _build_video_args(
            copy_video=False,
            hw="vaapi",
            deinterlace=False,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality=quality,
        )
        assert expected_qp in post

    def test_sr_live_hls_uses_two_second_segments(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        _enable_sr(monkeypatch, tmp_path, model="2x-nomosuni-compact")
        _save_dvr(0)
        media_info = MediaInfo(
            video_codec="h264",
            audio_codec="aac",
            pix_fmt="yuv420p",
            audio_channels=2,
            audio_sample_rate=48_000,
            audio_profile="LC",
            height=720,
        )

        command = build_hls_ffmpeg_cmd(
            "https://example.test/live.m3u8",
            "nvenc+software",
            str(tmp_path / "output"),
            media_info=media_info,
            max_resolution="4k",
            deinterlace_fallback=False,
        )

        assert "dnn_processing=" in command[command.index("-vf") + 1]
        assert command[command.index("-hls_time") + 1] == "2"
        assert command[command.index("-hls_list_size") + 1] == "15"
        assert command[command.index("-hls_init_time") + 1] == "1"

    def test_invalid_hw_raises(self):
        """Test invalid hardware raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized hardware"):
            _build_video_args(
                copy_video=False,
                hw="invalid",
                deinterlace=False,
                use_hw_pipeline=False,
                max_resolution="1080p",
                quality="high",
            )

    def test_nvenc_hdr_with_libplacebo(self, fake_run: FakeRun):
        """Test NVENC HDR uses libplacebo when available."""
        _script_libplacebo(fake_run, available=True)
        _, post = _build_video_args(
            copy_video=False,
            hw="nvenc+software",
            deinterlace=False,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
            is_hdr=True,
        )
        vf = post[post.index("-vf") + 1]
        assert "libplacebo" in vf
        assert "tonemapping=hable" in vf
        # Should download from CUDA, process, re-upload.
        assert "hwdownload" in vf
        assert "hwupload_cuda" in vf

    def test_nvenc_hdr_zscale_fallback(self, fake_run: FakeRun):
        """Test NVENC HDR falls back to zscale when libplacebo unavailable."""
        _script_libplacebo(fake_run, available=False)
        _, post = _build_video_args(
            copy_video=False,
            hw="nvenc+software",
            deinterlace=False,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
            is_hdr=True,
        )
        vf = post[post.index("-vf") + 1]
        assert "zscale" in vf
        assert "tonemap=hable" in vf
        assert "libplacebo" not in vf

    def test_nvenc_hdr_deinterlace_order(self, fake_run: FakeRun):
        """Test NVENC HDR hw decode deinterlaces BEFORE tonemap."""
        _script_libplacebo(fake_run, available=True)
        _, post = _build_video_args(
            copy_video=False,
            hw="nvenc+software",
            deinterlace=True,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
            is_hdr=True,
        )
        vf = post[post.index("-vf") + 1]
        # Deinterlace should come before tonemap in hw decode path.
        deint_pos = vf.find("yadif_cuda")
        tonemap_pos = vf.find("libplacebo")
        assert deint_pos < tonemap_pos, f"deinterlace should come before tonemap: {vf}"

    def test_nvenc_sw_hdr_deinterlace_order(self, fake_run: FakeRun):
        """Test NVENC HDR sw decode uses CPU deinterlace before tonemap."""
        _script_libplacebo(fake_run, available=True)
        _, post = _build_video_args(
            copy_video=False,
            hw="nvenc+software",
            deinterlace=True,
            use_hw_pipeline=False,
            max_resolution="1080p",
            quality="high",
            is_hdr=True,
        )
        vf = post[post.index("-vf") + 1]
        # SW decode HDR should use CPU yadif before tonemap.
        assert "yadif=0" in vf
        deint_pos = vf.find("yadif=0")
        tonemap_pos = vf.find("libplacebo")
        assert deint_pos < tonemap_pos, f"CPU deinterlace should come before tonemap: {vf}"

    def test_vaapi_hdr_tonemap(self):
        """Test VAAPI HDR uses tonemap_vaapi filter."""
        _, post = _build_video_args(
            copy_video=False,
            hw="vaapi",
            deinterlace=False,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
            is_hdr=True,
        )
        vf = post[post.index("-vf") + 1]
        assert "tonemap_vaapi" in vf


# =============================================================================
# Audio Args Tests
# =============================================================================


class TestBuildAudioArgs:
    """Tests for _build_audio_args."""

    def test_copy_audio(self):
        """Test copy_audio returns copy args."""
        assert _build_audio_args(copy_audio=True, audio=_media_info()) == ["-c:a", "copy"]
        assert _build_audio_args(copy_audio=True, audio=None) == ["-c:a", "copy"]

    @pytest.mark.parametrize(
        "sample_rate,expected",
        [
            (44_100, "44100"),
            (48_000, "48000"),
            (96_000, "48000"),
            (0, "48000"),
        ],
    )
    def test_sample_rates(self, sample_rate: int, expected: str):
        """Test sample rate handling."""
        args = _build_audio_args(copy_audio=False, audio=_media_info(audio_sample_rate=sample_rate))
        assert args[args.index("-ar") + 1] == expected

    @pytest.mark.parametrize("channels", [3, 6, 8])
    def test_surround_encodes_5_1(self, channels: int):
        """Surround sources stay surround instead of being downmixed to stereo."""
        args = _build_audio_args(copy_audio=False, audio=_media_info(audio_channels=channels))
        assert args[args.index("-af") + 1] == "aformat=channel_layouts=5.1"
        assert args[args.index("-b:a") + 1] == "384k"
        assert "-ac" not in args

    @pytest.mark.parametrize("channels", [1, 2])
    def test_stereo_stays_stereo(self, channels: int):
        args = _build_audio_args(copy_audio=False, audio=_media_info(audio_channels=channels))
        assert args[args.index("-ac") + 1] == "2"
        assert args[args.index("-b:a") + 1] == "192k"
        assert "-af" not in args

    def test_unknown_layout_keeps_surround(self):
        """Unprobed audio lets ffmpeg keep 5.1 or fall back to stereo at runtime."""
        assert _build_audio_args(copy_audio=False, audio=None) == _UNKNOWN_LAYOUT_AAC
        unknown_channels = _build_audio_args(
            copy_audio=False,
            audio=_media_info(audio_channels=0, audio_sample_rate=44_100),
        )
        assert unknown_channels[unknown_channels.index("-af") + 1] == (
            "aformat=channel_layouts=stereo|5.1"
        )
        assert unknown_channels[unknown_channels.index("-ar") + 1] == "44100"


# =============================================================================
# HLS Command Tests
# =============================================================================


_RECONNECT = [
    "-reconnect",
    "1",
    "-reconnect_streamed",
    "1",
    "-reconnect_on_network_error",
    "1",
    "-reconnect_on_http_error",
    "4xx,5xx",
    "-reconnect_delay_max",
    "30",
]
_UNKNOWN_LAYOUT_AAC = [
    "-c:a",
    "aac",
    "-af",
    "aformat=channel_layouts=stereo|5.1",
    "-ar",
    "48000",
    "-b:a",
    "192k",
    "-profile:a",
    "aac_low",
]


def _software_video(*, deinterlace: bool, height: int = 1080, crf: str = "20") -> list[str]:
    vf = f"{'yadif=0:-1:1,' if deinterlace else ''}scale=-2:min(ih\\,{height}),format=yuv420p"
    return [
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        crf,
        "-bf",
        "3",
        "-g",
        "60",
    ]


class TestHlsAssemblyContract:
    @pytest.mark.parametrize("is_vod", [False, True])
    @pytest.mark.parametrize("has_media", [False, True])
    @pytest.mark.parametrize("sr", [False, True])
    @pytest.mark.parametrize("user_agent", [None, "", "custom-agent"])
    def test_exact_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        is_vod: bool,
        has_media: bool,
        sr: bool,
        user_agent: str | None,
    ):
        _save_dvr(1)
        engine = _enable_sr(monkeypatch, tmp_path) if sr else None
        media = _media_info(height=720, is_hls=True) if has_media else None
        cmd = build_hls_ffmpeg_cmd(
            "https://source/stream",
            "nvenc+software" if sr else "software",
            "/output",
            is_vod=is_vod,
            media_info=media,
            user_agent=user_agent,
            subtitles=[
                SubtitleStream(index=4, lang="eng", name="English"),
                SubtitleStream(index=7, lang="fra", name="French"),
            ],
            sr_segment_duration=2.5,
        )
        deinterlace = not has_media and not is_vod
        video_pre: list[str] = []
        if engine is not None:
            video_pre = [
                "-init_hw_device",
                "cuda=cu",
                "-filter_hw_device",
                "cu",
                "-filter_threads",
                "4",
            ]
            sr_filter = (
                "scale=1280:720:force_original_aspect_ratio=decrease,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=rgb24,hwupload,"
                f"dnn_processing=dnn_backend=tensorrt:model={engine},scale_cuda=w=-2:h=1080"
            )
            video_post = [
                "-vf",
                f"{'yadif=0:-1:1,' if deinterlace else ''}{sr_filter}",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p5",
                *(["-rc", "constqp", "-qp", "20"] if is_vod else []),
                "-bf",
                "0",
                "-spatial-aq",
                "1",
                "-temporal-aq",
                "1",
                "-g",
                "60",
            ]
        elif has_media:
            video_post = ["-c:v", "copy"]
        else:
            video_post = _software_video(deinterlace=deinterlace)
        expected = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-noautorotate", *video_pre]
        if not has_media:
            expected += [
                "-probesize",
                "50000" if is_vod else "5000000",
                "-analyzeduration",
                "500000" if is_vod else "5000000",
            ]
        expected += ["-fflags", "+discardcorrupt+genpts", "-err_detect", "ignore_err", *_RECONNECT]
        if user_agent is not None:
            expected += ["-user_agent", user_agent]
        if has_media:
            expected += ["-f", "hls", "-extension_picky", "0"]
        expected += ["-i", "https://source/stream"]
        list_size = "0" if is_vod else ("24" if sr else "20")
        for number, index in [(0, 4), (1, 7)]:
            expected += [
                "-map",
                f"0:{index}",
                "-c:s",
                "webvtt",
                "-flush_packets",
                "1",
                f"/output/sub{number}.vtt",
                "-map",
                f"0:{index}",
                "-c:s",
                "webvtt",
                "-f",
                "segment",
                "-segment_time",
                "2" if sr else "3",
                "-segment_list",
                f"/output/sub{number}.m3u8",
                "-segment_list_type",
                "m3u8",
                "-segment_list_size",
                list_size,
                "-segment_format",
                "webvtt",
            ]
            if not is_vod:
                expected += ["-segment_list_flags", "+live"]
            expected += [f"/output/sub{number}_%06d.vtt"]
        expected += [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            *video_post,
            *(["-c:a", "copy"] if has_media else _UNKNOWN_LAYOUT_AAC),
            "-max_delay",
            "5000000",
            "-f",
            "hls",
            "-hls_time",
            "2" if sr else "3",
            "-hls_list_size",
            list_size,
            "-hls_segment_filename",
            "/output/seg%03d.ts",
        ]
        expected += (
            [
                "-hls_init_time",
                "2",
                "-hls_flags",
                "independent_segments",
                "-hls_playlist_type",
                "event",
            ]
            if is_vod
            else ["-hls_init_time", "1", "-hls_flags", "delete_segments"]
        )
        if sr and not is_vod:
            expected += [
                "-rc",
                "vbr",
                "-b:v",
                "6000000",
                "-maxrate",
                "8000000",
                "-bufsize",
                "8000000",
            ]
        expected += ["/output/stream.m3u8"]
        assert cmd == expected

    def test_default_forwarding(self):
        cmd = build_hls_ffmpeg_cmd("input", "software", "/output")
        assert _between(cmd, "0:a:0", "-max_delay") == [
            *_software_video(deinterlace=True),
            *_UNKNOWN_LAYOUT_AAC,
        ]
        assert "-user_agent" not in cmd
        assert "-c:s" not in cmd

    def test_default_transcodes_dolby_audio(self):
        dolby = _media_info(audio_codec="eac3", audio_channels=6)
        default = build_hls_ffmpeg_cmd("input", "software", "/output", media_info=dolby)
        passthrough = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/output",
            media_info=dolby,
            audio_passthrough=True,
        )
        assert default[default.index("-c:a") + 1] == "aac"
        assert passthrough[passthrough.index("-c:a") + 1] == "copy"

    @pytest.mark.parametrize("probed_as", ["media_info", "source"])
    def test_explicit_forwarding(self, probed_as: str):
        """Probed facts drive video and audio alike, however the caller holds them."""
        media = _media_info(
            height=2160,
            is_hdr=True,
            audio_sample_rate=44_100,
            audio_channels=6,
            audio_profile="Main",
        )
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/output",
            media_info=media if probed_as == "media_info" else None,
            source=media if probed_as == "source" else None,
            max_resolution="720p",
            quality="low",
        )
        tonemap = (
            "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
        )
        video = _software_video(deinterlace=False, height=720, crf="32")
        video[1] = tonemap + video[1]
        audio_args = [
            "-c:a",
            "aac",
            "-af",
            "aformat=channel_layouts=5.1",
            "-ar",
            "44100",
            "-b:a",
            "384k",
            "-profile:a",
            "aac_low",
        ]
        assert _between(cmd, "0:a:0", "-max_delay") == [*video, *audio_args]

    @pytest.mark.parametrize("hw", ["nvenc+software", "vaapi", "qsv", "amf+software", "software"])
    @pytest.mark.parametrize(
        "codec",
        ["h264", "hevc", "mpeg2video", "vp8", "vp9", "vc1", "av1", "unknown", ""],
    )
    def test_hardware_pipeline(self, fake_run: FakeRun, hw: HwAccel, codec: str):
        fake_run.program("nvidia-smi", Program(stdout="NVIDIA GeForce GTX 960, 5.2\n"))
        cmd = build_hls_ffmpeg_cmd("input", hw, "/output", source=_media_info(video_codec=codec))
        supported = {
            "nvenc+software": {"h264"},
            "vaapi": {"h264", "hevc", "mpeg2video", "vp8", "vp9", "vc1", "av1"},
            "qsv": {"h264", "hevc", "mpeg2video", "vp9", "vc1", "av1"},
            "amf+software": set[str](),
            "software": set[str](),
        }
        assert ("-hwaccel" in cmd) is (codec in supported[hw])
        probed_gpu = [c[0] for c in fake_run.calls] == ["nvidia-smi"]
        assert probed_gpu is (hw == "nvenc+software" and bool(codec))

    @pytest.mark.parametrize("hw", ["nvenc+software", "vaapi", "qsv", "amf+software", "software"])
    def test_copied_video_skips_hardware_pipeline(self, fake_run: FakeRun, hw: HwAccel):
        fake_run.program("nvidia-smi", Program(stdout="NVIDIA GeForce RTX 3080, 8.6\n"))
        cmd = build_hls_ffmpeg_cmd("input", hw, "/output", media_info=_media_info())
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-hwaccel" not in cmd
        assert fake_run.calls == []

    def test_media_info_overrides_source(self):
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "qsv",
            "/output",
            media_info=_media_info(video_codec="vp8"),
            source=_media_info(video_codec="h264"),
        )
        assert "-hwaccel" not in cmd
        assert cmd[cmd.index("-c:v") + 1] == "h264_qsv"

    @pytest.mark.parametrize("is_vod", [False, True])
    @pytest.mark.parametrize("fallback", [None, False, True])
    @pytest.mark.parametrize("height", [None, 0, 1])
    @pytest.mark.parametrize("interlaced", [False, True])
    @pytest.mark.parametrize("probed_as", ["media_info", "source"])
    def test_deinterlace_selection(
        self,
        is_vod: bool,
        fallback: bool | None,
        height: int | None,
        interlaced: bool,
        probed_as: str,
    ):
        media = (
            None
            if height is None
            else _media_info(height=height, interlaced=interlaced, pix_fmt="yuv420p10le")
        )
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/output",
            is_vod=is_vod,
            media_info=media if probed_as == "media_info" else None,
            source=media if probed_as == "source" else None,
            deinterlace_fallback=fallback,
        )
        expected = interlaced if height == 1 else (not is_vod if fallback is None else fallback)
        assert cmd[cmd.index("-vf") + 1].startswith("yadif=0:-1:1,") is expected

    @pytest.mark.parametrize(
        ("codec", "height", "live_deinterlaced"),
        [
            ("mpeg2video", 1080, True),
            ("h264", 480, True),
            ("mpeg2video", 576, True),
            ("mpeg2video", 720, False),
            ("hevc", 1080, False),
        ],
    )
    def test_a_live_sample_of_a_field_capable_raster_keeps_the_deinterlacer(
        self,
        codec: str,
        height: int,
        live_deinterlaced: bool,
    ):
        """A probed file is what it is; a live sample is only the moment it was taken."""
        progressive = _media_info(video_codec=codec, height=height, pix_fmt="yuv420p10le")
        live = build_hls_ffmpeg_cmd("input", "software", "/output", source=progressive)
        vod = build_hls_ffmpeg_cmd("input", "software", "/output", media_info=progressive)
        assert live[live.index("-vf") + 1].startswith("yadif=0:-1:1,") is live_deinterlaced
        assert not vod[vod.index("-vf") + 1].startswith("yadif")

    def test_missing_codec_does_not_probe_gpu(self, fake_run: FakeRun):
        cmd = build_hls_ffmpeg_cmd("input", "nvenc+software", "/output")
        assert fake_run.calls == []
        assert "-hwaccel" not in cmd


class TestBuildHlsFfmpegCmd:
    """Tests for build_hls_ffmpeg_cmd."""

    @pytest.mark.parametrize(
        "hw",
        ["nvenc+vaapi", "nvenc+software", "amf+vaapi", "amf+software", "qsv", "vaapi", "software"],
    )
    @pytest.mark.parametrize("is_vod", [True, False])
    def test_command_structure(self, hw: HwAccel, is_vod: bool):
        """Test command has correct structure for all hw/vod combinations."""
        cmd = build_hls_ffmpeg_cmd(
            "http://test/stream",
            hw,
            "/tmp/output",
            is_vod=is_vod,
        )

        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert "-map" in cmd
        assert "-c:v" in cmd
        assert "-c:a" in cmd
        assert "-f" in cmd
        assert "hls" in cmd

        i_idx = cmd.index("-i")
        if "-hwaccel" in cmd:
            hwaccel_idx = cmd.index("-hwaccel")
            assert hwaccel_idx < i_idx, "hwaccel must come before -i"

        if "-vf" in cmd:
            vf_idx = cmd.index("-vf")
            assert vf_idx > i_idx, "-vf must come after -i"

    def test_vod_hls_flags(self):
        """Test VOD has correct HLS flags."""
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", is_vod=True)
        assert "-hls_playlist_type" in cmd
        assert "event" in cmd
        assert "-hls_list_size" in cmd
        assert cmd[cmd.index("-hls_list_size") + 1] == "0"

    @pytest.mark.parametrize(("dvr_mins", "list_size"), [(0, "10"), (5, "100"), (120, "2400")])
    def test_live_hls_flags(self, dvr_mins: int, list_size: str):
        """Test live has correct HLS flags for the configured DVR window."""
        _save_dvr(dvr_mins)
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", is_vod=False)
        assert "delete_segments" in cmd
        assert "-hls_list_size" in cmd
        assert cmd[cmd.index("-hls_list_size") + 1] == list_size

    def test_copy_video_with_compatible_media(self):
        """Test copy_video is used for compatible VOD media."""
        media = _media_info(video_codec="h264", pix_fmt="yuv420p", height=1080)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,
            max_resolution="1080p",
        )
        assert "-c:v" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-hwaccel" not in cmd

    def test_no_copy_for_10bit(self):
        """Test 10-bit content is transcoded, not copied."""
        media = _media_info(video_codec="h264", pix_fmt="yuv420p10le", height=1080)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,
        )
        assert cmd[cmd.index("-c:v") + 1] != "copy"
        assert "-hwaccel" in cmd

    def test_no_copy_when_scaling_needed(self):
        """Test scaling requirement prevents copy."""
        media = _media_info(video_codec="h264", pix_fmt="yuv420p", height=2160)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,
            max_resolution="1080p",
        )
        assert cmd[cmd.index("-c:v") + 1] != "copy"

    def test_can_remux_live_compatible_source(self):
        """Test a browser-compatible progressive H.264/AAC source can remux."""
        assert can_remux_live(_media_info(), "1080p")

    def test_can_remux_live_rejects_unplayable_sources(self):
        """Test remux is rejected for missing info or incompatible sources."""
        assert not can_remux_live(None, "1080p")
        assert not can_remux_live(_media_info(video_codec="hevc"), "1080p")
        assert not can_remux_live(_media_info(interlaced=True), "1080p")
        assert not can_remux_live(_media_info(height=2160), "1080p")
        assert not can_remux_live(_media_info(audio_codec="ac3"), "1080p")
        assert not can_remux_live(_media_info(audio_profile="HE-AAC"), "1080p")

    def test_can_remux_live_surround_audio(self):
        """5.1 AAC copies for every client; Dolby only for passthrough clients."""
        assert can_remux_live(_media_info(audio_channels=6), "1080p")
        dolby = _media_info(audio_codec="eac3", audio_channels=6)
        assert not can_remux_live(dolby, "1080p")
        assert can_remux_live(dolby, "1080p", audio_passthrough=True)
        for profile in ("Main", "", "he-aac"):
            aac = _media_info(audio_channels=6, audio_profile=profile)
            assert not can_remux_live(aac, "1080p")

    @pytest.mark.parametrize(
        "codec,passthrough,expected",
        [("eac3", True, ["-c:a", "copy"]), ("ac3", True, ["-c:a", "copy"]), ("eac3", False, None)],
    )
    def test_source_audio_without_media_info(
        self,
        codec: str,
        passthrough: bool,
        expected: list[str] | None,
    ):
        """Audio facts alone decide the audio path; video is still transcoded."""
        audio = MediaInfo(
            video_codec="",
            audio_codec=codec,
            pix_fmt="",
            audio_channels=6,
            audio_sample_rate=48_000,
        )
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "software",
            "/tmp",
            source=audio,
            audio_passthrough=passthrough,
        )
        assert cmd[cmd.index("-c:v") + 1] != "copy"
        index = cmd.index("-c:a")
        if expected is not None:
            assert cmd[index : index + 2] == expected
        else:
            assert cmd[index + 1] == "aac"
            assert cmd[cmd.index("-af") + 1] == "aformat=channel_layouts=5.1"

    def test_user_agent(self):
        """Test user agent is included when provided."""
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "software",
            "/tmp",
            user_agent="TestAgent/1.0",
        )
        assert "-user_agent" in cmd
        assert "TestAgent/1.0" in cmd

    def test_probe_args_without_media_info(self):
        """Test probe args are added when no media_info."""
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", media_info=None)
        assert "-probesize" in cmd
        assert "-analyzeduration" in cmd

    def test_no_probe_args_with_media_info(self):
        """Test probe args are skipped when media_info provided."""
        media = _media_info()
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", media_info=media)
        assert "-probesize" not in cmd

    def test_subtitle_extraction(self):
        """Test subtitle streams are extracted."""
        subs = [
            SubtitleStream(index=2, lang="eng", name="English"),
            SubtitleStream(index=3, lang="spa", name="Spanish"),
        ]
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp/out", subtitles=subs)
        assert "-map" in cmd
        assert "0:2" in cmd
        assert "0:3" in cmd
        assert "/tmp/out/sub0.vtt" in cmd
        assert "/tmp/out/sub1.vtt" in cmd
        assert "/tmp/out/sub0.m3u8" in cmd
        assert "/tmp/out/sub1.m3u8" in cmd
        assert "/tmp/out/sub0_%06d.vtt" in cmd
        assert "/tmp/out/sub1_%06d.vtt" in cmd
        assert cmd.count("-segment_format") == 2

    @pytest.mark.parametrize(("dvr_mins", "list_size"), [(0, "10"), (5, "100"), (120, "2400")])
    def test_live_subtitle_renditions_use_bounded_playlists(self, dvr_mins: int, list_size: str):
        subs = [SubtitleStream(index=2, lang="eng", name="English")]
        _save_dvr(dvr_mins)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "software",
            "/tmp/out",
            is_vod=False,
            subtitles=subs,
        )
        segment_output = cmd[
            cmd.index("/tmp/out/sub0.vtt") + 1 : cmd.index("/tmp/out/sub0_%06d.vtt")
        ]
        assert segment_output[segment_output.index("-segment_list_size") + 1] == list_size
        assert segment_output[segment_output.index("-segment_list_flags") + 1] == "+live"

    def test_vod_subtitle_renditions_keep_the_full_playlist(self):
        subs = [SubtitleStream(index=2, lang="eng", name="English")]
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "software",
            "/tmp/out",
            is_vod=True,
            subtitles=subs,
        )
        segment_output = cmd[
            cmd.index("/tmp/out/sub0.vtt") + 1 : cmd.index("/tmp/out/sub0_%06d.vtt")
        ]
        assert segment_output[segment_output.index("-segment_list_size") + 1] == "0"
        assert "-segment_list_flags" not in segment_output


@pytest.mark.parametrize("url", ["http://test/live.ts", "https://test/live.m3u8"])
def test_http_reconnect_options_shared_by_live_and_vod(url: str):
    for is_vod in (False, True):
        cmd = build_hls_ffmpeg_cmd(url, "software", "/tmp", is_vod=is_vod)
        expected = http_reconnect_args(url)
        start = cmd.index("-reconnect")
        assert cmd[start : start + len(expected)] == expected
        assert start + len(expected) <= cmd.index("-i")
        assert "-rw_timeout" not in cmd
    assert http_reconnect_args(url, max_delay=5)[-1] == "5"


@pytest.mark.parametrize("url", ["/tmp/input.m3u8", "file:/tmp/input.ts", "udp://test:1234"])
def test_non_http_inputs_do_not_need_command_repairs(url: str):
    assert http_reconnect_args(url) == []
    cmd = build_hls_ffmpeg_cmd(url, "software", "/tmp")
    assert not any(arg.startswith("-reconnect") for arg in cmd)


@pytest.mark.parametrize("is_vod", [False, True])
def test_live_ai_upscale_uses_shared_bitrate_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    is_vod: bool,
):
    _enable_sr(monkeypatch, tmp_path, model="nomos")
    cmd = build_hls_ffmpeg_cmd(
        "https://provider.example/live.m3u8",
        "nvenc+software",
        "/tmp/output",
        is_vod=is_vod,
        max_resolution="4k",
    )
    assert "dnn_processing=" in cmd[cmd.index("-vf") + 1]
    if is_vod:
        assert cmd[cmd.index("-rc") + 1] == "constqp"
        assert "-maxrate" not in cmd
    else:
        assert cmd[cmd.index("-rc") + 1] == "vbr"
        assert cmd[cmd.index("-b:v") + 1] == "16000000"
        assert cmd[cmd.index("-maxrate") + 1] == "20000000"
        original = cmd.copy()
        limit_live_video_bitrate(cmd, "4k")
        assert cmd == original
        assert cmd.count("-maxrate") == 1


# =============================================================================
# Aspect Ratio Tests
# =============================================================================


class TestAspectRatioHandling:
    """Tests for various aspect ratio content."""

    @pytest.mark.parametrize(
        "input_height,max_res,should_scale",
        [
            (1080, "1080p", False),
            (1080, "720p", True),
            (720, "1080p", False),
            (2160, "1080p", True),
            (1600, "1080p", True),
            (1600, "4k", False),
        ],
    )
    def test_scaling_decisions(self, input_height: int, max_res: str, should_scale: bool):
        """Test correct scaling decisions for various input heights."""
        media = _media_info(height=input_height, pix_fmt="yuv420p10le")
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,
            max_resolution=max_res,
        )
        vf = cmd[cmd.index("-vf") + 1]
        max_h = {"1080p": 1080, "720p": 720, "4k": 2160}[max_res]
        # Comma is escaped in FFmpeg filter expressions.
        height_expr = f"min(ih\\,{max_h})"
        assert height_expr in vf, f"Expected {height_expr} in {vf}"


# =============================================================================
# GPU Detection Tests
# =============================================================================


class TestGpuDetection:
    """Tests for GPU/NVDEC detection."""

    def test_nvidia_gpu_detected(self, fake_run: FakeRun):
        """Test NVIDIA GPU detection parses compute capability."""
        fake_run.program("nvidia-smi", Program(stdout="NVIDIA GeForce RTX 3080, 8.6\n"))
        assert _get_gpu_nvdec_codecs() == {"h264", "hevc", "av1"}
        assert fake_run.calls == [
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        ]

    def test_no_nvidia_gpu(self, fake_run: FakeRun):
        """Test handling when no NVIDIA GPU present."""
        fake_run.program(
            "nvidia-smi",
            Program(stdout="NVIDIA GeForce RTX 3080, 8.6\n", returncode=1),
        )
        assert _get_gpu_nvdec_codecs() == set()

    def test_older_nvidia_gpu(self, fake_run: FakeRun):
        """Test older GPU with limited NVDEC support."""
        fake_run.program("nvidia-smi", Program(stdout="NVIDIA GeForce GTX 960, 5.2\n"))
        assert _get_gpu_nvdec_codecs() == {"h264"}


# =============================================================================
# User Agent Tests
# =============================================================================


def _save_user_agent(preset: str, custom: str = "") -> None:
    cache.save_server_settings(
        {**cache.load_server_settings(), "user_agent_preset": preset, "user_agent_custom": custom},
    )


class TestUserAgent:
    """Tests for user agent handling."""

    def test_default_user_agent(self):
        """Test default preset returns None."""
        _save_user_agent("default")
        assert get_user_agent() is None

    def test_vlc_user_agent(self):
        """Test VLC preset."""
        _save_user_agent("vlc")
        ua = get_user_agent()
        assert ua is not None
        assert "VLC" in ua

    def test_chrome_user_agent(self):
        """Test Chrome preset."""
        _save_user_agent("chrome")
        ua = get_user_agent()
        assert ua is not None
        assert "Chrome" in ua

    def test_custom_user_agent(self):
        """Test custom user agent."""
        _save_user_agent("custom", "MyAgent/1.0")
        assert get_user_agent() == "MyAgent/1.0"

    def test_custom_empty_returns_none(self):
        """Test empty custom user agent returns None."""
        _save_user_agent("custom")
        assert get_user_agent() is None


# =============================================================================
# Transcode Directory Tests
# =============================================================================


def _save_transcode_dir(path: str) -> None:
    cache.save_server_settings({**cache.load_server_settings(), "transcode_dir": path})


class TestTranscodeDir:
    """Tests for transcode directory handling."""

    def test_default_transcode_dir(self):
        """Test default uses system temp."""
        assert get_transcode_dir() == Path(tempfile.gettempdir())

    def test_custom_transcode_dir(self, tmp_path: Path):
        """Test custom directory is used and created."""
        custom_dir = tmp_path / "custom_transcode"
        _save_transcode_dir(str(custom_dir))
        assert get_transcode_dir() == custom_dir
        assert custom_dir.exists()


# =============================================================================
# HLS List Size Tests
# =============================================================================


class TestHlsListSize:
    """Tests for HLS list size calculation."""

    @pytest.mark.parametrize(
        ("dvr", "duration", "expected"),
        [(0, 1.0, 30), (0, 4.0, 7), (1, 4.0, 15), (2, 7.0, 17), (-1, 4.0, 7)],
    )
    def test_duration_arithmetic(self, dvr: int, duration: float, expected: int):
        _save_dvr(dvr)
        assert get_live_hls_list_size(duration) == expected

    @pytest.mark.parametrize(("dvr", "expected"), [(0, 5), (1, 15)])
    def test_custom_live_buffer(self, dvr: int, expected: int):
        _save_dvr(dvr)
        assert get_live_hls_list_size(4.0, live_buffer_secs=23.0) == expected

    def test_default_list_size(self):
        """Test DVR disabled uses 10 segments."""
        _save_dvr(0)
        assert get_live_hls_list_size() == 10

    def test_dvr_enabled_list_size(self):
        """Test DVR enabled calculates segments from minutes."""
        _save_dvr(5)
        # 5 min = 300 sec / 3 sec per segment = 100 segments.
        assert get_live_hls_list_size() == 100

    def test_dvr_list_size_follows_long_windows(self):
        _save_dvr(360)
        assert get_live_hls_list_size() == 7200

    def test_server_default_dvr_window(self):
        assert get_live_hls_list_size() == 2400


# =============================================================================
# Probe Media Tests
# =============================================================================


class TestProbeMedia:
    """Tests for media probing."""

    def test_probe_success(self, fake_run: FakeRun):
        """Test successful probe parses media info."""
        _ffprobe(
            fake_run,
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "pix_fmt": "yuv420p",
                        "height": 1080,
                        "field_order": "progressive",
                    },
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "channels": 2,
                        "sample_rate": "48000",
                    },
                ],
                "format": {"duration": "3600.0"},
            },
        )

        media_info, subs = probe_media("http://test/video.mp4")

        assert media_info is not None
        assert media_info.video_codec == "h264"
        assert media_info.audio_codec == "aac"
        assert media_info.height == 1080
        assert media_info.duration == 3600.0
        assert not media_info.interlaced
        assert subs == []
        assert len(fake_run.calls) == 1
        assert fake_run.calls[0][-1] == "http://test/video.mp4"
        assert fake_run.options == [
            {"check": False, "capture_output": True, "text": True, "timeout": 30},
        ]

    def test_probe_interlaced_detection(self, fake_run: FakeRun):
        """Test interlaced content detection."""
        _ffprobe(
            fake_run,
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "mpeg2video",
                        "pix_fmt": "yuv420p",
                        "height": 1080,
                        "field_order": "tt",  # Top field first = interlaced.
                    },
                ],
                "format": {},
            },
        )

        media_info, _ = probe_media("http://test/interlaced.ts")

        assert media_info is not None
        assert media_info.interlaced is True

    @pytest.mark.parametrize(
        "pix_fmt,expected",
        [
            ("yuv420p10le", True),  # 10-bit little endian.
            ("yuv420p10be", True),  # 10-bit big endian.
            ("yuv422p10le", True),  # 10-bit 4:2:2.
            ("p010le", True),  # CUDA/VAAPI 10-bit format.
            ("yuv420p", False),  # 8-bit.
            ("yuv410p", False),  # 4:1:0 chroma, NOT 10-bit (was a false positive)
            ("nv12", False),  # 8-bit NV12.
        ],
    )
    def test_probe_10bit_detection(self, fake_run: FakeRun, pix_fmt: str, expected: bool):
        """Test 10-bit content detection from pix_fmt."""
        _ffprobe(
            fake_run,
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "hevc",
                        "pix_fmt": pix_fmt,
                        "height": 2160,
                    },
                ],
                "format": {},
            },
        )

        media_info, _ = probe_media(f"http://test/{pix_fmt}.mkv")

        assert media_info is not None
        assert media_info.is_10bit is expected, f"pix_fmt={pix_fmt} should be is_10bit={expected}"

    @pytest.mark.parametrize(
        "color_transfer,expected",
        [
            ("smpte2084", True),  # PQ (HDR10, HDR10+, Dolby Vision)
            ("arib-std-b67", True),  # HLG
            ("bt709", False),  # SDR
            ("bt2020-10", False),  # Wide gamut but not HDR transfer.
            ("", False),  # Unknown/missing.
        ],
    )
    def test_probe_hdr_detection(self, fake_run: FakeRun, color_transfer: str, expected: bool):
        """Test HDR content detection from color_transfer."""
        _ffprobe(
            fake_run,
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "hevc",
                        "pix_fmt": "yuv420p10le",
                        "height": 2160,
                        "color_transfer": color_transfer,
                    },
                ],
                "format": {},
            },
        )

        media_info, _ = probe_media(f"http://test/{color_transfer or 'unknown'}.mkv")

        assert media_info is not None
        assert media_info.is_hdr is expected, (
            f"color_transfer={color_transfer} should be is_hdr={expected}"
        )

    def test_probe_failure(self, fake_run: FakeRun):
        """Test probe failure returns None."""
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}}, returncode=1)

        media_info, subs = probe_media("http://test/bad.mp4")

        assert media_info is None
        assert subs == []
        assert len(fake_run.calls) == 2
        assert ffmpeg_command._probe_cache == {}

    def test_probe_cache_hit(self, fake_run: FakeRun):
        """Test probe cache returns cached result."""
        cached_info = MediaInfo(
            video_codec="h264",
            audio_codec="aac",
            pix_fmt="yuv420p",
        )
        ffmpeg_command._probe_cache["http://cached"] = (
            time.time(),
            cached_info,
            list[SubtitleStream](),
        )

        media_info, _ = probe_media("http://cached")

        assert fake_run.calls == []
        assert media_info == cached_info

    def test_probe_extracts_subtitles(self, fake_run: FakeRun):
        """Test subtitle stream extraction."""
        _ffprobe(
            fake_run,
            {
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p"},
                    {
                        "codec_type": "subtitle",
                        "codec_name": "subrip",
                        "index": 2,
                        "tags": {"language": "eng", "title": "English"},
                    },
                    {
                        "codec_type": "subtitle",
                        "codec_name": "ass",
                        "index": 3,
                        "tags": {"language": "jpn"},
                    },
                ],
                "format": {},
            },
        )

        _, subs = probe_media("http://test/subs.mkv")

        assert subs == [
            SubtitleStream(index=2, lang="eng", name="English"),
            SubtitleStream(index=3, lang="jpn", name="Japanese"),
        ]


# =============================================================================
# Probe Cache Management Tests
# =============================================================================


def _series_cache_text() -> str:
    return (cache.CACHE_DIR / "series_probe_cache.json").read_text()


class TestProbeCacheManagement:
    """Tests for probe cache management functions."""

    def test_clear_all_probe_cache(self):
        """Test clearing all probe caches."""
        ffmpeg_command._probe_cache["url1"] = (time.time(), None, list[SubtitleStream]())
        ffmpeg_command._probe_cache["url2"] = (time.time(), None, list[SubtitleStream]())
        ffmpeg_command._series_probe_cache[123] = {
            "episodes": {1: (time.time(), None, list[SubtitleStream]())},
        }

        count = clear_all_probe_cache()

        assert count == 3
        assert len(ffmpeg_command._probe_cache) == 0
        assert len(ffmpeg_command._series_probe_cache) == 0
        assert json.loads(_series_cache_text()) == {}

    def test_invalidate_series_probe_cache_entire_series(self):
        """Test invalidating entire series cache."""
        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "episodes": {
                1: (0, None, list[SubtitleStream]()),
                2: (0, None, list[SubtitleStream]()),
            },
        }

        invalidate_series_probe_cache(123)

        assert 123 not in ffmpeg_command._series_probe_cache
        assert json.loads(_series_cache_text()) == {}

    def test_invalidate_series_probe_cache_single_episode(self):
        """Test invalidating single episode cache."""
        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "episodes": {
                1: (0, _media_info(), list[SubtitleStream]()),
                2: (0, _media_info(), list[SubtitleStream]()),
            },
        }

        invalidate_series_probe_cache(123, episode_id=1)

        assert 123 in ffmpeg_command._series_probe_cache
        assert 1 not in ffmpeg_command._series_probe_cache[123]["episodes"]
        assert 2 in ffmpeg_command._series_probe_cache[123]["episodes"]
        assert list(json.loads(_series_cache_text())["123"]["episodes"]) == ["2"]

    def test_invalidate_unknown_episode_keeps_series(self):
        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "episodes": {2: (0, _media_info(), list[SubtitleStream]())},
        }

        invalidate_series_probe_cache(123, episode_id=1)

        assert list(ffmpeg_command._series_probe_cache[123]["episodes"]) == [2]

    def test_clear_series_mru(self):
        """Test clearing series MRU."""
        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "mru": 5,
            "episodes": {1: (0, None, list[SubtitleStream]())},
        }

        clear_series_mru(123)

        assert "mru" not in ffmpeg_command._series_probe_cache[123]
        assert "episodes" in ffmpeg_command._series_probe_cache[123]
        assert json.loads(_series_cache_text()) == {
            "123": {"name": "Test", "mru": None, "episodes": {}},
        }

    def test_clear_series_mru_without_mru_still_saves(self):
        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "episodes": dict[int, ffmpeg_command._ProbeEntry](),
        }

        clear_series_mru(123)

        assert json.loads(_series_cache_text()) == {
            "123": {"name": "Test", "mru": None, "episodes": {}},
        }

    def test_restore_probe_cache_entry(self):
        """Test restoring probe cache entry."""
        media_info = MediaInfo(video_codec="h264", audio_codec="aac", pix_fmt="yuv420p")
        subs = [SubtitleStream(index=2, lang="eng", name="English")]

        restore_probe_cache_entry("http://test", media_info, subs, series_id=123, episode_id=5)

        assert "http://test" in ffmpeg_command._probe_cache
        assert 123 in ffmpeg_command._series_probe_cache
        assert 5 in ffmpeg_command._series_probe_cache[123]["episodes"]

    def test_get_series_probe_cache_stats(self):
        """Test getting cache stats for UI."""
        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test Series",
            "mru": 2,
            "episodes": {
                1: (time.time(), _media_info(), list[SubtitleStream]()),
                2: (time.time(), _media_info(), list[SubtitleStream]()),
            },
        }

        stats = get_series_probe_cache_stats()

        assert len(stats) == 1
        assert stats[0]["series_id"] == 123
        assert stats[0]["name"] == "Test Series"
        assert stats[0]["episode_count"] == 2


class TestResolveHlsMasterPlaylist:
    """Tests for resolve_hls_master_playlist function."""

    def test_non_m3u8_url_returns_unchanged(self, monkeypatch: pytest.MonkeyPatch):
        """Non-m3u8 URLs should be returned unchanged."""
        web = FakeWeb()
        install_web(monkeypatch, web, ffmpeg_command)
        url = "http://example.com/video.mp4"
        assert resolve_hls_master_playlist(url) == url
        assert web.fetches == []

    def test_master_playlist_selects_highest_bandwidth(self, monkeypatch: pytest.MonkeyPatch):
        """Master playlist should resolve to highest bandwidth variant."""
        master_playlist = b"""#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=500000,RESOLUTION=640x360
low.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1920x1080
high.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1500000,RESOLUTION=1280x720
mid.m3u8
"""
        url = "http://example.com/master.m3u8"
        web = FakeWeb({url: FakeResponse(master_playlist)})
        install_web(monkeypatch, web, ffmpeg_command)

        assert resolve_hls_master_playlist(url) == "http://example.com/high.m3u8"
        assert [(f.url, f.timeout, f.headers) for f in web.fetches] == [
            (url, 10, {"User-agent": "TiviMate/4.7.0"}),
        ]

    def test_media_playlist_returns_unchanged(self, monkeypatch: pytest.MonkeyPatch):
        """Media playlist (not master) should return original URL."""
        media_playlist = b"""#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXTINF:10.0,
segment0.ts
#EXTINF:10.0,
segment1.ts
"""
        url = "http://example.com/stream.m3u8"
        install_web(monkeypatch, FakeWeb({url: FakeResponse(media_playlist)}), ffmpeg_command)

        assert resolve_hls_master_playlist(url) == url

    def test_fetch_error_returns_original_url(self, monkeypatch: pytest.MonkeyPatch):
        """On fetch error, should return original URL."""
        url = "http://example.com/master.m3u8?token=1"
        web = FakeWeb({url: Exception("Network error")})
        install_web(monkeypatch, web, ffmpeg_command)

        assert resolve_hls_master_playlist(url) == url
        assert web.urls == [url]

    def test_relative_url_resolved_correctly(self, monkeypatch: pytest.MonkeyPatch):
        """Relative variant URLs should be resolved against base URL."""
        master_playlist = b"""#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=3000000
../streams/1080p/index.m3u8
"""
        url = "http://example.com/hls/master.m3u8"
        install_web(monkeypatch, FakeWeb({url: FakeResponse(master_playlist)}), ffmpeg_command)

        assert resolve_hls_master_playlist(url) == "http://example.com/streams/1080p/index.m3u8"

    def test_redirected_playlists_resolve_to_where_they_were_served(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Providers redirect each playlist request (~1 s); ffmpeg should not repeat the hop."""
        media = "http://example.com/live/1.m3u8"
        master = "http://example.com/live/2.m3u8"
        web = FakeWeb(
            {
                media: FakeResponse(b"#EXTM3U\n#EXTINF:10,\ns.ts\n", url="http://edge/a/1.m3u8"),
                master: FakeResponse(
                    b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nhd/index.m3u8\n",
                    url="http://edge/b/2.m3u8",
                ),
            },
        )
        install_web(monkeypatch, web, ffmpeg_command)

        assert resolve_hls_master_playlist(media) == "http://edge/a/1.m3u8"
        assert resolve_hls_master_playlist(master) == "http://edge/b/hd/index.m3u8"

    def test_default_user_agent_sends_no_header(self, monkeypatch: pytest.MonkeyPatch):
        _save_user_agent("default")
        url = "http://example.com/stream.m3u8"
        web = FakeWeb({url: FakeResponse(b"#EXTM3U\n")})
        install_web(monkeypatch, web, ffmpeg_command)

        assert resolve_hls_master_playlist(url) == url
        assert web.fetches[0].headers == {}

    @pytest.mark.parametrize(
        ("playlist", "variant", "count", "bandwidth"),
        [
            # A missing BANDWIDTH ranks as 0; ties keep playlist order.
            (b"BANDWIDTH=0\nzero.m3u8\n#EXT-X-STREAM-INF:RESOLUTION=1x1\nnone.m3u8", "zero", 2, 0),
            (b"BANDWIDTH=7\na.m3u8\n#EXT-X-STREAM-INF:BANDWIDTH=7\nb.m3u8", "a", 2, 7),
            (b"BANDWIDTH=abc\nonly.m3u8", "only", 1, 0),
            (b"BANDWIDTH=900, RESOLUTION=1x1\nhd.m3u8", "hd", 1, 900),
            (
                b"BANDWIDTH=500\nlow.m3u8\n#EXT-X-STREAM-INF:BANDWIDTH=3000000\nhigh.m3u8",
                "high",
                2,
                3_000_000,
            ),
        ],
    )
    def test_variant_selection_is_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        playlist: bytes,
        variant: str,
        count: int,
        bandwidth: int,
    ):
        caplog.set_level("INFO", logger="netv.ffmpeg_command")
        base = f"http://example.com/{'p' * 90}/"
        url = base + "master.m3u8"
        body = b"#EXTM3U\n#EXT-X-STREAM-INF:" + playlist + b"\n"
        install_web(monkeypatch, FakeWeb({url: FakeResponse(body)}), ffmpeg_command)

        best = f"{base}{variant}.m3u8"
        assert resolve_hls_master_playlist(url) == best
        assert _command_log(caplog) == [
            (
                "INFO",
                f"HLS master playlist resolved: {count} variants, selected {bandwidth} bps: {best[:80]}",
            ),
        ]

    def test_invalid_utf8_is_replaced(self, monkeypatch: pytest.MonkeyPatch):
        url = "http://example.com/master.m3u8"
        body = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nvid\xff.m3u8\n"
        install_web(monkeypatch, FakeWeb({url: FakeResponse(body)}), ffmpeg_command)

        assert resolve_hls_master_playlist(url) == "http://example.com/vid\ufffd.m3u8"

    def test_master_without_variants_logs_served_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        url = "http://example.com/master.m3u8"
        served = f"http://edge.example.com/{'p' * 90}/master.m3u8"
        body = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\n#comment\n"
        install_web(monkeypatch, FakeWeb({url: FakeResponse(body, url=served)}), ffmpeg_command)

        assert resolve_hls_master_playlist(url) == served
        assert _command_log(caplog) == [
            ("WARNING", f"HLS master playlist has no variants: {served[:80]}"),
        ]

    def test_fetch_error_is_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        url = f"http://example.com/{'p' * 90}/master.m3u8"
        install_web(monkeypatch, FakeWeb({url: Exception("Network error")}), ffmpeg_command)

        assert resolve_hls_master_playlist(url) == url
        assert _command_log(caplog) == [
            ("WARNING", f"Failed to resolve HLS master playlist {url[:80]}: Network error"),
        ]


def _command_log(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    return [
        (r.levelname, r.getMessage()) for r in caplog.records if r.name == "netv.ffmpeg_command"
    ]


@pytest.mark.parametrize("hw", ["software", "nvenc+software", "amf+software", "qsv", "vaapi"])
def test_bandwidth_saver_bypasses_ai_and_scales_down(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hw: HwAccel,
):
    _enable_sr(monkeypatch, tmp_path, model="nomos")
    media_info = MediaInfo(
        video_codec="h264",
        audio_codec="aac",
        pix_fmt="yuv420p",
        audio_channels=2,
        audio_sample_rate=48_000,
        audio_profile="LC",
        height=2160,
    )
    cmd = build_hls_ffmpeg_cmd(
        "http://example.com/live.ts",
        hw,
        "/tmp/hls",
        media_info=media_info,
        max_resolution="720p",
        quality="low",
        allow_upscale=False,
    )
    assert "720" in " ".join(cmd)
    assert "dnn_processing" not in " ".join(cmd)
    assert cmd[cmd.index("-c:v") + 1] != "copy"


@pytest.mark.parametrize("allow_upscale", [False, True])
def test_allow_upscale_gates_ai_for_low_resolution_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    allow_upscale: bool,
):
    _enable_sr(monkeypatch, tmp_path)
    cmd = build_hls_ffmpeg_cmd(
        "http://example.com/live.ts",
        "nvenc+software",
        "/tmp/hls",
        media_info=_media_info(height=480, pix_fmt="yuv420p10le"),
        max_resolution="720p",
        allow_upscale=allow_upscale,
    )
    assert ("dnn_processing" in " ".join(cmd)) is allow_upscale


@pytest.mark.parametrize(("height", "engine_height"), [(None, 1080), (480, 480)])
def test_ai_upscale_engine_follows_source_height(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    height: int | None,
    engine_height: int,
):
    """Probed height picks the nearest engine; an unprobed source gets the largest."""
    _enable_sr(monkeypatch, tmp_path).unlink()
    for h in (480, 1080):
        (tmp_path / f"2x-nomosuni-compact_{h}p_fp16.engine").write_bytes(b"engine")
    cmd = build_hls_ffmpeg_cmd(
        "http://example.com/live.ts",
        "nvenc+software",
        "/tmp/hls",
        media_info=None if height is None else _media_info(height=height),
        max_resolution="4k",
    )
    engine = tmp_path / f"2x-nomosuni-compact_{engine_height}p_fp16.engine"
    assert f"model={engine}," in cmd[cmd.index("-vf") + 1]


@pytest.mark.parametrize("height", [720, 1080, 2160])
def test_archive_policy_copies_compatible_source_without_ai_upscaling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    height: int,
):
    _enable_sr(monkeypatch, tmp_path, model="upscale-model")
    media_info = MediaInfo(
        video_codec="h264",
        audio_codec="aac",
        pix_fmt="yuv420p",
        audio_channels=2,
        audio_sample_rate=48_000,
        audio_profile="LC",
        height=height,
    )
    cmd = build_hls_ffmpeg_cmd(
        "https://upstream.test/timeshift/user/pass/60/2026-09-28:10-00/1.ts",
        "nvenc+software",
        "/tmp/hls",
        is_vod=True,
        media_info=media_info,
        max_resolution="4k",
        allow_upscale=False,
    )
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-vf" not in cmd


def test_probe_segment_parses_video_and_audio(fake_run: FakeRun):
    _ffprobe(
        fake_run,
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "H264",
                    "pix_fmt": "yuv420p",
                    "height": 1080,
                    "field_order": "tt",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "AAC",
                    "channels": 6,
                    "sample_rate": "48000",
                    "profile": "LC",
                },
            ],
            "format": {"format_name": "mpegts", "bit_rate": "5000000"},
        },
    )
    info = probe_segment("input_0.ts")
    assert info == MediaInfo(
        video_codec="h264",
        audio_codec="aac",
        pix_fmt="yuv420p",
        audio_channels=6,
        audio_sample_rate=48_000,
        audio_profile="LC",
        height=1080,
        video_bitrate=5_000_000,
        interlaced=True,
    )
    assert fake_run.calls == [
        ["ffprobe", "-v", "error", "-of", "json", "-show_streams", "-show_format", "input_0.ts"],
    ]
    assert fake_run.options == [
        {"capture_output": True, "text": True, "timeout": 5, "check": False},
    ]


@pytest.mark.parametrize(
    "outcome",
    [
        Program(error=OSError("unavailable")),
        Program(error=subprocess.TimeoutExpired("ffprobe", 5)),
        Program(stdout="not json"),
        Program(stdout='{"streams": []}'),
        Program(stdout='{"streams": [{"codec_type": "audio", "codec_name": "aac"}]}'),
    ],
)
def test_probe_segment_without_readable_video_returns_none(fake_run: FakeRun, outcome: Program):
    fake_run.program("ffprobe", outcome)
    assert probe_segment("input_0.ts") is None


@pytest.mark.parametrize(
    "side_data_type, expected",
    [
        ("ATSC A53 Part 4 Closed Captions", True),
        ("H.264 User Data Unregistered SEI message", False),
    ],
)
def test_probe_closed_captions_reads_video_side_data(
    fake_run: FakeRun,
    side_data_type: str,
    expected: bool,
):
    _ffprobe(fake_run, {"frames": [{"side_data_list": [{"side_data_type": side_data_type}]}]})
    assert probe_closed_captions("/tmp/input.ts") is expected

    command = fake_run.calls[0]
    assert command[0] == "ffprobe"
    assert command[-1] == "/tmp/input.ts"
    assert command[command.index("-read_intervals") + 1] == "%+2"
    assert command[command.index("-show_entries") + 1] == "frame=side_data_list"
    assert fake_run.options == [
        {"capture_output": True, "text": True, "timeout": 5, "check": False},
    ]


# =============================================================================
# Module state, caches, persistence, and probing edge cases
# =============================================================================


@pytest.fixture
def probe_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep the on-disk probe cache in tmp_path and send no User-Agent."""
    _save_user_agent("default")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


_VIDEO = {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "height": 720}


def _server_settings() -> ServerSettings:
    return {
        "transcode_mode": "auto",
        "transcode_hw": "software",
        "vod_transcode_cache_mins": 60,
        "live_dvr_mins": 120,
        "transcode_dir": "",
        "probe_live": True,
        "probe_movies": True,
        "probe_series": False,
        "sources": [],
        "user_agent_preset": "tivimate",
        "user_agent_custom": "",
    }


def test_init_sets_settings_and_engine_dir(monkeypatch: pytest.MonkeyPatch, probe_state: Path):
    # cache.initialize resolves a path still at its legacy default, so this proves it ran.
    monkeypatch.setattr(cache, "LOGOS_DIR", cache.APP_DIR / ".cache" / "logos")
    (probe_state / "series_probe_cache.json").write_text(
        json.dumps({"4": {"name": "Show", "episodes": {"1": {"video_codec": "h264"}}}}),
    )
    settings = _server_settings()
    settings["sr_model"] = "x"

    ffmpeg_command.init(lambda: settings, sr_engine_dir="/engines")
    assert probe_state / "logos" == cache.LOGOS_DIR
    assert ffmpeg_command._sr_engine_dir == "/engines"
    assert ffmpeg_command.get_settings() is settings
    assert list(ffmpeg_command._series_probe_cache[4]["episodes"]) == [1]


def test_init_without_engine_dir_disables_upscale(
    monkeypatch: pytest.MonkeyPatch,
    probe_state: Path,
):
    monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", "/stale")
    ffmpeg_command.init(_server_settings)
    assert ffmpeg_command._sr_engine_dir is None
    assert ffmpeg_command.get_settings() == _server_settings()


def test_transcode_dir_falls_back_when_inaccessible(probe_state: Path):
    blocker = probe_state / "file"
    blocker.write_text("")
    _save_transcode_dir(str(blocker / "x"))
    assert get_transcode_dir() == Path(tempfile.gettempdir())


def test_series_stats_skip_empty_and_unprobed_series(probe_state: Path):
    ffmpeg_command._series_probe_cache.update(
        {
            1: {"name": "Empty", "episodes": {}},
            2: {"name": "Unprobed", "episodes": {1: (5, None, [])}},
            3: {
                "episodes": {
                    2: (2, _media_info(), [SubtitleStream(index=1, lang="eng", name="E")]),
                    1: (1, None, []),
                },
            },
        },
    )
    stats = get_series_probe_cache_stats()
    assert [s["series_id"] for s in stats] == [3]
    assert stats[0]["episode_count"] == 2
    assert stats[0]["subtitle_count"] == 1
    assert stats[0]["episodes"] == [{"episode_id": 2, "duration": 0.0, "subtitle_count": 1}]


def test_cache_mutators_ignore_unknown_series(probe_state: Path):
    invalidate_series_probe_cache(999)
    invalidate_series_probe_cache(999, episode_id=1)
    clear_series_mru(999)
    assert not (probe_state / "series_probe_cache.json").exists()
    assert ffmpeg_command._series_probe_cache == {}


def test_master_playlist_without_variant_urls_returns_original(
    monkeypatch: pytest.MonkeyPatch,
    probe_state: Path,
):
    _save_user_agent("custom", "UA/1")
    url = "http://example.com/master.m3u8"
    body = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=abc\n#comment\n"
    web = FakeWeb({url: FakeResponse(body)})
    install_web(monkeypatch, web, ffmpeg_command)
    assert resolve_hls_master_playlist(url) == url
    assert web.fetches[0].headers == {"User-agent": "UA/1"}


class TestProbeMediaSeriesCache:
    """Series/episode cache paths in probe_media."""

    def test_exact_episode_hit_updates_mru_and_saves(self, fake_run: FakeRun, probe_state: Path):
        info = _media_info()
        ffmpeg_command._series_probe_cache[7] = {
            "name": "S",
            "mru": 1,
            "episodes": {2: (time.time(), info, list[SubtitleStream]())},
        }
        assert probe_media("http://x", series_id=7, episode_id=2) == (info, [])
        saved_file = probe_state / "series_probe_cache.json"
        assert json.loads(saved_file.read_text())["7"]["mru"] == 2
        saved_file.unlink()
        # Same MRU now: hit without another save.
        assert probe_media("http://x", series_id=7, episode_id=2) == (info, [])
        assert not saved_file.exists()
        assert fake_run.calls == []

    def test_mru_fallback_hit(self, fake_run: FakeRun, probe_state: Path):
        info = _media_info()
        ffmpeg_command._series_probe_cache[7] = {
            "mru": 1,
            "episodes": {1: (time.time(), info, list[SubtitleStream]())},
        }
        assert probe_media("http://x", series_id=7, episode_id=99) == (info, [])
        assert fake_run.calls == []

    @pytest.mark.parametrize("episode_id", [1, 99])
    def test_expired_series_entries_fall_through_to_probe(
        self,
        fake_run: FakeRun,
        probe_state: Path,
        episode_id: int,
    ):
        ffmpeg_command._series_probe_cache[7] = {
            "mru": 1,
            "episodes": {1: (0, _media_info(video_codec="old"), list[SubtitleStream]())},
        }
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}})
        info, _ = probe_media("http://x", series_id=7, episode_id=episode_id, series_name="N")
        assert info is not None and info.video_codec == "h264"
        series = ffmpeg_command._series_probe_cache[7]
        assert "name" in series
        assert series["name"] == "N"
        assert series["mru"] == episode_id
        assert series["episodes"][episode_id][1] is info

    def test_new_series_is_cached_and_persisted(self, fake_run: FakeRun, probe_state: Path):
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}})
        info, _ = probe_media("http://x", series_id=8, series_name="New")
        series = ffmpeg_command._series_probe_cache[8]
        assert "name" in series
        assert series["name"] == "New"
        assert "mru" in series
        assert series["mru"] == 0
        assert series["episodes"][0][1] is info
        saved = json.loads((probe_state / "series_probe_cache.json").read_text())
        assert saved["8"]["episodes"]["0"]["video_codec"] == "h264"

    def test_existing_named_series_keeps_name(self, fake_run: FakeRun, probe_state: Path):
        ffmpeg_command._series_probe_cache[8] = {
            "name": "Kept",
            "episodes": dict[int, ffmpeg_command._ProbeEntry](),
        }
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}})
        probe_media("http://x", series_id=8, episode_id=3, series_name="Other")
        assert ffmpeg_command._series_probe_cache[8]["name"] == "Kept"
        assert 3 in ffmpeg_command._series_probe_cache[8]["episodes"]


class TestProbeMediaEdgeCases:
    def test_subprocess_exceptions_yield_none(self, fake_run: FakeRun, probe_state: Path):
        fake_run.program("ffprobe", Program(error=OSError("boom")))
        assert probe_media("http://x") == (None, [])
        assert len(fake_run.calls) == 2
        assert fake_run.calls[0][-1] == "http://x"
        assert "hls" not in fake_run.calls[0]
        assert fake_run.calls[1][-5:] == ["-f", "hls", "-extension_picky", "0", "http://x"]

    def test_user_agent_and_format_bitrate_fallback(self, fake_run: FakeRun, probe_state: Path):
        _save_user_agent("custom", "UA/1")
        fmt = {"bit_rate": "1234", "format_name": "hls"}
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": fmt})
        info, _ = probe_media("http://x")
        assert info is not None
        assert info.video_bitrate == 1234
        assert info.is_hls
        cmd = fake_run.calls[0]
        assert cmd[cmd.index("-user_agent") + 1] == "UA/1"

    def test_invalid_format_bitrate_ignored(self, fake_run: FakeRun, probe_state: Path):
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {"bit_rate": "n/a"}})
        info, _ = probe_media("http://x")
        assert info is not None and info.video_bitrate == 0
        assert "-user_agent" not in fake_run.calls[0]

    def test_audio_only_returns_none(self, fake_run: FakeRun, probe_state: Path):
        audio = {"codec_type": "audio", "codec_name": "aac", "channels": 2}
        _ffprobe(fake_run, {"streams": [audio], "format": {}})
        assert probe_media("http://x") == (None, [])

    def test_zero_height_not_cached(self, fake_run: FakeRun, probe_state: Path):
        _ffprobe(fake_run, {"streams": [{**_VIDEO, "height": 0}], "format": {}})
        info, _ = probe_media("http://x", series_id=1)
        assert info is not None and info.height == 0
        assert ffmpeg_command._probe_cache == {}
        assert ffmpeg_command._series_probe_cache == {}


def test_uses_audio_passthrough():
    assert ffmpeg_command.uses_audio_passthrough(_media_info(audio_codec="eac3"), True)
    assert not ffmpeg_command.uses_audio_passthrough(_media_info(audio_codec="eac3"), False)
    assert not ffmpeg_command.uses_audio_passthrough(_media_info(audio_codec="aac"), True)
    assert not ffmpeg_command.uses_audio_passthrough(None, True)


def test_probe_closed_captions_reports_ffprobe_failure(
    fake_run: FakeRun,
    caplog: pytest.LogCaptureFixture,
):
    fake_run.program("ffprobe", Program(returncode=1, stderr="  bad input \n"))
    assert probe_closed_captions("/tmp/dir/a.ts") is False
    assert "a.ts: bad input" in caplog.text

    fake_run.program("ffprobe", Program(returncode=2))
    assert probe_closed_captions("/tmp/dir/a.ts") is False
    assert "ffprobe exited 2" in caplog.text


@pytest.mark.parametrize(
    "outcome",
    [
        Program(error=subprocess.TimeoutExpired("ffprobe", 5)),
        Program(stdout="not json"),
    ],
)
def test_probe_closed_captions_handles_errors(
    fake_run: FakeRun,
    outcome: Program,
    caplog: pytest.LogCaptureFixture,
):
    fake_run.program("ffprobe", outcome)
    assert probe_closed_captions("/tmp/a.ts") is False
    assert "Could not inspect embedded captions in a.ts" in caplog.text


@pytest.mark.parametrize("stdout", ['["not", "a", "dict"]', '{"frames": "not a list"}'])
def test_probe_closed_captions_rejects_malformed_shapes(fake_run: FakeRun, stdout: str):
    """A probe of the wrong shape means no captions, not a crash."""
    fake_run.program("ffprobe", Program(stdout=stdout))
    assert probe_closed_captions("/tmp/a.ts") is False


def test_probe_media_retries_hls_when_a_field_has_the_wrong_type(
    fake_run: FakeRun,
    probe_state: Path,
):
    """A mistyped field fails that probe attempt; the HLS retry still succeeds."""
    _ffprobe(fake_run, {"streams": [{**_VIDEO, "codec_name": 7}], "format": {}})
    _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}}, when="hls")
    info, _ = probe_media("http://x/mistyped")
    assert info is not None
    assert info.video_codec == "h264"
    assert info.is_hls
    assert len(fake_run.calls) == 2
    assert fake_run.calls[1][-5:] == [
        "-f",
        "hls",
        "-extension_picky",
        "0",
        "http://x/mistyped",
    ]


def test_hls_media_info_adds_hls_demuxer_options():
    cmd = build_hls_ffmpeg_cmd(
        "http://x/live.m3u8",
        "software",
        "/out",
        media_info=_media_info(is_hls=True),
    )
    i = cmd.index("-i")
    assert cmd[i - 4 : i] == ["-f", "hls", "-extension_picky", "0"]


class TestSrEngineSelection:
    def test_no_engine_dir(self, probe_state: Path):
        assert _find_sr_engine(720) is None
        assert _build_sr_filter(720, 1080) == ""

    def test_missing_engine_dir(self, monkeypatch: pytest.MonkeyPatch, probe_state: Path):
        monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", str(probe_state / "missing"))
        assert _find_sr_engine(720) is None

    def test_no_valid_engines(self, monkeypatch: pytest.MonkeyPatch, probe_state: Path):
        engine_dir = probe_state / "engines"
        engine_dir.mkdir()
        _enable_sr(monkeypatch, engine_dir).unlink()
        (engine_dir / "2x-nomosuni-compact_bigp_fp16.engine").write_bytes(b"")
        assert _find_sr_engine(720) is None
        assert _build_sr_filter(720, 1080) == ""

    def test_failed_probe_uses_largest_engine(
        self,
        monkeypatch: pytest.MonkeyPatch,
        probe_state: Path,
    ):
        for height in (480, 1080):
            (probe_state / f"2x-nomosuni-compact_{height}p_fp16.engine").write_bytes(b"")
        monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", str(probe_state))
        result = _find_sr_engine(0)
        assert result is not None
        assert result[1:] == (1080, 1920, 2)

    def test_source_larger_than_engines_uses_largest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        probe_state: Path,
    ):
        (probe_state / "2x-nomosuni-compact_480p_fp16.engine").write_bytes(b"")
        monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", str(probe_state))
        result = _find_sr_engine(1080)
        assert result is not None and result[1] == 480

    def test_filter_disabled_without_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        probe_state: Path,
    ):
        (probe_state / "2x-nomosuni-compact_720p_fp16.engine").write_bytes(b"")
        monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", str(probe_state))
        assert _build_sr_filter(480, 1080) == ""

    def test_filter_skipped_when_source_meets_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
        probe_state: Path,
    ):
        engine_dir = probe_state / "engines"
        engine_dir.mkdir()
        _enable_sr(monkeypatch, engine_dir).unlink()
        engine = engine_dir / "2x-nomosuni-compact_1080p_fp16.engine"
        engine.write_bytes(b"")
        assert _build_sr_filter(1080, 1080) == ""
        no_target = _build_sr_filter(720, 0)
        assert no_target.endswith(f"dnn_processing=dnn_backend=tensorrt:model={engine}")
        assert "scale_cuda" not in no_target


class TestHardwareProbes:
    def test_nvidia_smi_unparseable_output(self, fake_run: FakeRun):
        fake_run.program("nvidia-smi", Program(stdout="GPU only\n"))
        assert _get_gpu_nvdec_codecs() == set()

    def test_nvidia_smi_missing(self, fake_run: FakeRun):
        assert _get_gpu_nvdec_codecs() == set()
        # Cached: no second probe.
        fake_run.program("nvidia-smi", Program(stdout="NVIDIA GeForce RTX 3080, 8.6\n"))
        assert _get_gpu_nvdec_codecs() == set()
        assert len(fake_run.calls) == 1

    @pytest.mark.parametrize("available", [True, False])
    def test_libplacebo_probe(self, fake_run: FakeRun, available: bool):
        _script_libplacebo(fake_run, available=available)
        assert ffmpeg_command._has_libplacebo_filter() is available
        assert ffmpeg_command._has_libplacebo_filter() is available
        assert fake_run.calls == [["ffmpeg", "-filters"]]

    def test_libplacebo_probe_failure(self, fake_run: FakeRun):
        fake_run.program("ffmpeg", Program(error=OSError("no ffmpeg")))
        assert ffmpeg_command._has_libplacebo_filter() is False


class TestSeriesProbeCachePersistence:
    def test_load_missing_file_is_noop(self, probe_state: Path):
        ffmpeg_command._load_series_probe_cache()
        assert ffmpeg_command._series_probe_cache == {}

    def test_load_corrupt_file_is_logged(self, probe_state: Path, caplog: pytest.LogCaptureFixture):
        (probe_state / "series_probe_cache.json").write_text("{not json")
        ffmpeg_command._load_series_probe_cache()
        assert ffmpeg_command._series_probe_cache == {}
        assert "Failed to load series probe cache" in caplog.text

    def test_round_trip_and_merge(self, probe_state: Path):
        info = MediaInfo(
            video_codec="hevc",
            audio_codec="eac3",
            pix_fmt="yuv420p10le",
            audio_channels=6,
            audio_sample_rate=48_000,
            subtitle_codecs=["subrip"],
            duration=42.5,
            height=2160,
            video_bitrate=9_000,
            interlaced=True,
            is_10bit=True,
            is_hdr=True,
        )
        subs = [SubtitleStream(index=3, lang="eng", name="English")]
        ffmpeg_command._series_probe_cache.update(
            {
                5: {"name": "Show", "mru": 2, "episodes": {2: (123.0, info, subs)}},
                6: {"episodes": {1: (1.0, None, [])}},
            },
        )
        ffmpeg_command._save_series_probe_cache()
        saved = json.loads((probe_state / "series_probe_cache.json").read_text())
        assert saved["6"] == {"name": "", "mru": None, "episodes": {}}
        assert saved["5"]["episodes"]["2"]["subtitles"] == [
            {"index": 3, "lang": "eng", "name": "English"},
        ]

        # Add entries the loader must skip or default.
        saved["5"]["episodes"]["4"] = {"audio_codec": "aac"}
        saved["5"]["episodes"]["5"] = {"video_codec": "h264", "subtitles": [{"index": 1}]}
        (probe_state / "series_probe_cache.json").write_text(json.dumps(saved))

        existing = _media_info(video_codec="mpeg2video")
        ffmpeg_command._series_probe_cache.clear()
        ffmpeg_command._series_probe_cache[5] = {
            "episodes": {2: (999.0, existing, list[SubtitleStream]())},
        }
        ffmpeg_command._load_series_probe_cache()

        show = ffmpeg_command._series_probe_cache[5]
        assert "name" in show
        assert show["name"] == "Show"
        assert "mru" in show
        assert show["mru"] == 2
        assert show["episodes"][2] == (999.0, existing, list[SubtitleStream]())
        assert 4 not in show["episodes"]
        time5, info5, subs5 = show["episodes"][5]
        assert info5 is not None
        assert (time5, info5.video_codec, info5.height) == (0, "h264", 0)
        assert subs5 == [SubtitleStream(index=1, lang="und", name="")]
        assert ffmpeg_command._series_probe_cache[6] == {"name": "", "mru": None, "episodes": {}}

        ffmpeg_command._series_probe_cache.clear()
        ffmpeg_command._load_series_probe_cache()
        _, loaded, loaded_subs = ffmpeg_command._series_probe_cache[5]["episodes"][2]
        assert loaded == info
        assert loaded_subs == subs

    def test_save_failure_is_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        probe_state: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        blocker = probe_state / "file"
        blocker.write_text("")
        monkeypatch.setattr(ffmpeg_command.cache, "CACHE_DIR", blocker / "sub")
        ffmpeg_command._save_series_probe_cache()
        assert "Failed to save series probe cache" in caplog.text


class TestVideoArgsRemainingBranches:
    def _args(
        self,
        hw: str,
        *,
        hw_pipeline: bool = False,
        deinterlace: bool = False,
        hdr: bool = False,
        res: str = "1080p",
    ) -> tuple[list[str], list[str]]:
        return _build_video_args(
            copy_video=False,
            hw=hw,
            deinterlace=deinterlace,
            use_hw_pipeline=hw_pipeline,
            max_resolution=res,
            quality="high",
            is_hdr=hdr,
        )

    @classmethod
    def _vf(cls, post: list[str]) -> str:
        return post[post.index("-vf") + 1]

    def test_missing_vaapi_device_falls_back_to_software(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cache, "VAAPI_DEVICE", None)
        pre, post = self._args("vaapi", hw_pipeline=True)
        assert pre == []
        assert post[post.index("-c:v") + 1] == "libx264"

        pre, post = self._args("nvenc+vaapi")
        assert pre == []
        assert (
            self._vf(post) == "format=nv12,hwupload_cuda,scale_cuda=-2:min(ih\\,1080):format=nv12"
        )

    def test_nvenc_software_hdr_zscale_without_libplacebo(self, fake_run: FakeRun):
        _script_libplacebo(fake_run, available=False)
        _, post = self._args("nvenc+software", hdr=True)
        assert self._vf(post).startswith("zscale=t=linear")
        assert self._vf(post).endswith("hwupload_cuda,scale_cuda=-2:min(ih\\,1080):format=nv12")

    def test_amf_software_hdr(self):
        _, post = self._args("amf+software", hdr=True, deinterlace=True)
        vf = self._vf(post)
        assert vf.startswith("yadif=0:-1:1,zscale=t=linear")
        assert vf.endswith("format=nv12,scale=-2:min(ih\\,1080),format=nv12")

    def test_vaapi_baseline_only(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(cache.AVAILABLE_ENCODERS, "vaapi_baseline_only", True)
        _, post = self._args("vaapi", hw_pipeline=True)
        assert post[post.index("-profile:v") + 1] == "constrained_baseline"
        assert "-bf" not in post

    @pytest.mark.parametrize(
        ("hw_pipeline", "deinterlace", "hdr", "expected"),
        [
            (True, True, True, "vpp_qsv=deinterlace=2:tonemap=1:format=nv12,scale_qsv"),
            (True, False, True, "vpp_qsv=tonemap=1:format=nv12,scale_qsv"),
            (False, True, True, "hwupload=extra_hw_frames=64,vpp_qsv=deinterlace=2:tonemap=1"),
            (False, False, True, "hwupload=extra_hw_frames=64,vpp_qsv=tonemap=1:format=nv12,"),
        ],
    )
    def test_qsv_hdr(self, hw_pipeline: bool, deinterlace: bool, hdr: bool, expected: str):
        _, post = self._args("qsv", hw_pipeline=hw_pipeline, deinterlace=deinterlace, hdr=hdr)
        vf = self._vf(post)
        assert expected in vf
        assert vf.startswith("vpp_qsv" if hw_pipeline else "format=nv12,hwupload")

    def test_software_hdr_and_unbounded_resolution(self):
        _, post = self._args("software", hdr=True)
        assert self._vf(post).startswith("zscale=t=linear")
        _, post = self._args("software", res="original")
        assert self._vf(post) == "format=yuv420p"


def _media_info(
    *,
    video_codec: str = "h264",
    audio_codec: str = "aac",
    pix_fmt: str = "yuv420p",
    audio_channels: int = 2,
    audio_sample_rate: int = 48_000,
    audio_profile: str = "LC",
    height: int = 1080,
    interlaced: bool = False,
    is_10bit: bool = False,
    is_hdr: bool = False,
    is_hls: bool = False,
) -> MediaInfo:
    return MediaInfo(
        video_codec=video_codec,
        audio_codec=audio_codec,
        pix_fmt=pix_fmt,
        audio_channels=audio_channels,
        audio_sample_rate=audio_sample_rate,
        audio_profile=audio_profile,
        height=height,
        interlaced=interlaced,
        is_10bit=is_10bit,
        is_hdr=is_hdr,
        is_hls=is_hls,
    )


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
