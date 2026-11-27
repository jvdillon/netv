"""Tests for ffmpeg command building."""

from pathlib import Path

import json
import subprocess
import tempfile

import pytest

from netv import cache, proxy
from netv.cache import HwAccel, ServerSettings
from netv.ffmpeg import commands, probe, probe_test
from netv.ffmpeg.commands import (
    _build_audio_args,
    _build_sr_filter,
    _build_video_args,
    _find_sr_engine,
    _get_gpu_nvdec_codecs,
    build_hls_ffmpeg_cmd,
    can_remux_live,
    encoder_crop,
    get_live_hls_list_size,
    get_transcode_dir,
    http_reconnect_args,
    limit_live_video_bitrate,
)
from netv.ffmpeg.letterbox import DETECTION_FILE, CropBox, detection_args
from netv.ffmpeg.letterbox_test import SCOPE
from netv.ffmpeg.probe import MediaInfo, SubtitleStream
from netv.ffmpeg.probe_test import media_info, option, save_home_lineup
from netv.testing import FakeRun, Program, install_subprocess


# Fixtures resolve by module attribute, so the shared one is bound here.
probe_state = probe_test.probe_state


@pytest.fixture(autouse=True)
def command_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with empty probe caches, the real settings loader, and a VAAPI device."""
    monkeypatch.setattr(cache, "VAAPI_DEVICE", "/dev/dri/renderD128")
    monkeypatch.setattr(commands, "_load_settings", cache.load_server_settings)
    monkeypatch.setattr(commands, "_sr_engine_dir", None)
    monkeypatch.setattr(probe, "_probe_cache", {})
    monkeypatch.setattr(probe, "_series_probe_cache", {})
    monkeypatch.setattr(commands, "_gpu_nvdec_codecs", None)
    monkeypatch.setattr(commands, "_has_libplacebo", None)


@pytest.fixture(autouse=True)
def fake_run(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> FakeRun:
    """Serve ffprobe, ffmpeg, and nvidia-smi from scripts, except in real-ffmpeg tests."""
    run = FakeRun()
    if request.node.get_closest_marker("cli_ffmpeg") is None:
        install_subprocess(monkeypatch, commands, probe, run=run)
    return run


def _save_dvr(minutes: int) -> None:
    cache.save_server_settings({**cache.load_server_settings(), "live_dvr_mins": minutes})


def _enable_sr(monkeypatch: pytest.MonkeyPatch, engine_dir: Path, *, model: str = "on") -> Path:
    """Install one 720p AI Upscale engine and select a model; return the engine path."""
    engine = engine_dir / "2x-nomosuni-compact_720p_fp16.engine"
    engine.write_bytes(b"engine")
    monkeypatch.setattr(commands, "_sr_engine_dir", str(engine_dir))
    cache.save_server_settings({**cache.load_server_settings(), "sr_model": model})
    return engine


def _script_libplacebo(run: FakeRun, *, available: bool) -> None:
    run.program(
        "ffmpeg",
        Program(stdout="... libplacebo ..." if available else ""),
        when="-filters",
    )


def _between(cmd: list[str], start: str, end: str) -> list[str]:
    return cmd[cmd.index(start) + 1 : cmd.index(end)]


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
        monkeypatch.setattr(commands, "_sr_engine_dir", str(tmp_path))

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
            input_args=(),
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


class TestBuildAudioArgs:
    """Tests for _build_audio_args."""

    def test_copy_audio(self):
        """Test copy_audio returns copy args."""
        assert _build_audio_args(copy_audio=True, audio=media_info()) == ["-c:a", "copy"]
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
        args = _build_audio_args(copy_audio=False, audio=media_info(audio_sample_rate=sample_rate))
        assert args[args.index("-ar") + 1] == expected

    @pytest.mark.parametrize("channels", [3, 6, 8])
    def test_surround_encodes_5_1(self, channels: int):
        """Surround sources stay surround instead of being downmixed to stereo."""
        args = _build_audio_args(copy_audio=False, audio=media_info(audio_channels=channels))
        assert args[args.index("-af") + 1] == "aformat=channel_layouts=5.1"
        assert args[args.index("-b:a") + 1] == "384k"
        assert "-ac" not in args

    @pytest.mark.parametrize("channels", [1, 2])
    def test_stereo_stays_stereo(self, channels: int):
        args = _build_audio_args(copy_audio=False, audio=media_info(audio_channels=channels))
        assert args[args.index("-ac") + 1] == "2"
        assert args[args.index("-b:a") + 1] == "192k"
        assert "-af" not in args

    def test_unknown_layout_keeps_surround(self):
        """Unprobed audio lets ffmpeg keep 5.1 or fall back to stereo at runtime."""
        assert _build_audio_args(copy_audio=False, audio=None) == _UNKNOWN_LAYOUT_AAC
        unknown_channels = _build_audio_args(
            copy_audio=False,
            audio=media_info(audio_channels=0, audio_sample_rate=44_100),
        )
        assert unknown_channels[unknown_channels.index("-af") + 1] == (
            "aformat=channel_layouts=stereo|5.1"
        )
        assert unknown_channels[unknown_channels.index("-ar") + 1] == "44100"


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


def _software_video(
    *,
    deinterlace: bool,
    height: int = 1080,
    crf: str = "20",
    tonemap: str = "",
) -> list[str]:
    deint = "yadif=0:-1:1," if deinterlace else ""
    vf = f"fps=source_fps,{deint}{tonemap}scale=-2:min(ih\\,{height}),format=yuv420p"
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
        media = media_info(height=720, is_hls=True) if has_media else None
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
            input_args=proxy.ffmpeg_input_args("https://source/stream", ""),
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
                f"fps=source_fps,{'yadif=0:-1:1,' if deinterlace else ''}{sr_filter}",
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
        expected += [
            "-fflags",
            "+discardcorrupt+genpts",
            "-err_detect",
            "ignore_err",
            *proxy.ffmpeg_input_args("https://source/stream", ""),
            *_RECONNECT,
        ]
        if is_vod:
            expected += [
                "-readrate",
                "1",
                "-readrate_initial_burst",
                "120",
                "-readrate_catchup",
                "2",
            ]
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
        if engine is not None or not has_media:
            expected += detection_args("/output", "", is_10bit=False)
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
        cmd = build_hls_ffmpeg_cmd("input", "software", "/output", input_args=())
        assert _between(cmd, "0:a:0", "-max_delay") == [
            *_software_video(deinterlace=True),
            *_UNKNOWN_LAYOUT_AAC,
        ]
        assert "-user_agent" not in cmd
        assert "-c:s" not in cmd

    def test_default_transcodes_dolby_audio(self):
        dolby = media_info(audio_codec="eac3", audio_channels=6)
        default = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/output",
            media_info=dolby,
            input_args=(),
        )
        passthrough = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/output",
            media_info=dolby,
            audio_passthrough=True,
            input_args=(),
        )
        assert default[default.index("-c:a") + 1] == "aac"
        assert passthrough[passthrough.index("-c:a") + 1] == "copy"

    @pytest.mark.parametrize("probed_as", ["media_info", "source"])
    def test_explicit_forwarding(self, probed_as: str):
        """Probed facts drive video and audio alike, however the caller holds them."""
        media = media_info(
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
            input_args=(),
        )
        tonemap = (
            "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
        )
        video = _software_video(deinterlace=False, height=720, crf="32", tonemap=tonemap)
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
        cmd = build_hls_ffmpeg_cmd(
            "input",
            hw,
            "/output",
            source=media_info(video_codec=codec),
            input_args=(),
        )
        supported = {
            # Broadcast TV's MPEG-2 decodes on every NVDEC generation.
            "nvenc+software": {"h264", "mpeg2video"},
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
        cmd = build_hls_ffmpeg_cmd("input", hw, "/output", media_info=media_info(), input_args=())
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-hwaccel" not in cmd
        assert fake_run.calls == []

    def test_media_info_overrides_source(self):
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "qsv",
            "/output",
            media_info=media_info(video_codec="vp8"),
            source=media_info(video_codec="h264"),
            input_args=(),
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
            else media_info(height=height, interlaced=interlaced, pix_fmt="yuv420p10le")
        )
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/output",
            is_vod=is_vod,
            media_info=media if probed_as == "media_info" else None,
            source=media if probed_as == "source" else None,
            deinterlace_fallback=fallback,
            input_args=(),
        )
        expected = interlaced if height == 1 else (not is_vod if fallback is None else fallback)
        assert cmd[cmd.index("-vf") + 1].startswith("fps=source_fps,yadif=0:-1:1,") is expected

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
        progressive = media_info(video_codec=codec, height=height, pix_fmt="yuv420p10le")
        live = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/output",
            source=progressive,
            input_args=(),
        )
        vod = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/output",
            media_info=progressive,
            input_args=(),
        )
        live_vf = live[live.index("-vf") + 1]
        assert live_vf.startswith("fps=source_fps,yadif=0:-1:1,") is live_deinterlaced
        assert "yadif" not in vod[vod.index("-vf") + 1]

    def test_missing_codec_does_not_probe_gpu(self, fake_run: FakeRun):
        cmd = build_hls_ffmpeg_cmd("input", "nvenc+software", "/output", input_args=())
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
            input_args=(),
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
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", is_vod=True, input_args=())
        assert "-hls_playlist_type" in cmd
        assert "event" in cmd
        assert "-hls_list_size" in cmd
        assert cmd[cmd.index("-hls_list_size") + 1] == "0"

    @pytest.mark.parametrize(("dvr_mins", "list_size"), [(0, "10"), (5, "100"), (120, "2400")])
    def test_live_hls_flags(self, dvr_mins: int, list_size: str):
        """Test live has correct HLS flags for the configured DVR window."""
        _save_dvr(dvr_mins)
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", is_vod=False, input_args=())
        assert "delete_segments" in cmd
        assert "-hls_list_size" in cmd
        assert cmd[cmd.index("-hls_list_size") + 1] == list_size

    def test_copy_video_with_compatible_media(self):
        """Test copy_video is used for compatible VOD media."""
        media = media_info(video_codec="h264", pix_fmt="yuv420p", height=1080)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,
            max_resolution="1080p",
            input_args=(),
        )
        assert "-c:v" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-hwaccel" not in cmd

    def test_no_copy_for_10bit(self):
        """Test 10-bit content is transcoded, not copied."""
        media = media_info(video_codec="h264", pix_fmt="yuv420p10le", height=1080)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,
            input_args=(),
        )
        assert cmd[cmd.index("-c:v") + 1] != "copy"
        assert "-hwaccel" in cmd

    def test_no_copy_when_scaling_needed(self):
        """Test scaling requirement prevents copy."""
        media = media_info(video_codec="h264", pix_fmt="yuv420p", height=2160)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,
            max_resolution="1080p",
            input_args=(),
        )
        assert cmd[cmd.index("-c:v") + 1] != "copy"

    def test_can_remux_live_compatible_source(self):
        """Test a browser-compatible progressive H.264/AAC source can remux."""
        assert can_remux_live(media_info(), "1080p")

    def test_can_remux_live_defaults_to_a_1080p_ceiling(self):
        assert can_remux_live(media_info(height=1080))
        assert not can_remux_live(media_info(height=1440))

    def test_can_remux_live_rejects_unplayable_sources(self):
        """Test remux is rejected for missing info or incompatible sources."""
        assert not can_remux_live(None, "1080p")
        assert not can_remux_live(media_info(video_codec="hevc"), "1080p")
        assert not can_remux_live(media_info(interlaced=True), "1080p")
        assert not can_remux_live(media_info(height=2160), "1080p")
        assert not can_remux_live(media_info(audio_codec="ac3"), "1080p")
        assert not can_remux_live(media_info(audio_profile="HE-AAC"), "1080p")

    def test_can_remux_live_surround_audio(self):
        """5.1 AAC copies for every client; Dolby only for passthrough clients."""
        assert can_remux_live(media_info(audio_channels=6), "1080p")
        dolby = media_info(audio_codec="eac3", audio_channels=6)
        assert not can_remux_live(dolby, "1080p")
        assert can_remux_live(dolby, "1080p", audio_passthrough=True)
        for profile in ("Main", "", "he-aac"):
            aac = media_info(audio_channels=6, audio_profile=profile)
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
            input_args=(),
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
            input_args=(),
        )
        assert "-user_agent" in cmd
        assert "TestAgent/1.0" in cmd

    def test_probe_args_without_media_info(self):
        """Test probe args are added when no media_info."""
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "software",
            "/tmp",
            media_info=None,
            input_args=(),
        )
        assert "-probesize" in cmd
        assert "-analyzeduration" in cmd

    def test_no_probe_args_with_media_info(self):
        """Test probe args are skipped when media_info provided."""
        media = media_info()
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "software",
            "/tmp",
            media_info=media,
            input_args=(),
        )
        assert "-probesize" not in cmd

    @pytest.mark.parametrize(("audio_index", "mapped"), [(None, "0:a:0"), (3, "0:3")])
    def test_maps_the_chosen_audio_track(self, audio_index: int | None, mapped: str):
        media = MediaInfo(
            video_codec="h264",
            audio_codec="aac",
            pix_fmt="yuv420p",
            audio_index=audio_index,
        )
        cmd = build_hls_ffmpeg_cmd(
            "http://x/m.mkv",
            "software",
            "/o",
            True,
            media_info=media,
            input_args=(),
        )
        assert cmd[cmd.index("0:v:0") + 2] == mapped

    def test_subtitle_extraction(self):
        """Test subtitle streams are extracted."""
        subs = [
            SubtitleStream(index=2, lang="eng", name="English"),
            SubtitleStream(index=3, lang="spa", name="Spanish"),
        ]
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "software",
            "/tmp/out",
            subtitles=subs,
            input_args=(),
        )
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
            input_args=(),
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
            input_args=(),
        )
        segment_output = cmd[
            cmd.index("/tmp/out/sub0.vtt") + 1 : cmd.index("/tmp/out/sub0_%06d.vtt")
        ]
        assert segment_output[segment_output.index("-segment_list_size") + 1] == "0"
        assert "-segment_list_flags" not in segment_output


@pytest.mark.parametrize("url", ["http://test/live.ts", "https://test/live.m3u8"])
def test_http_reconnect_options_shared_by_live_and_vod(url: str):
    for is_vod in (False, True):
        cmd = build_hls_ffmpeg_cmd(url, "software", "/tmp", is_vod=is_vod, input_args=())
        expected = http_reconnect_args(url)
        start = cmd.index("-reconnect")
        assert cmd[start : start + len(expected)] == expected
        assert start + len(expected) <= cmd.index("-i")
        assert "-rw_timeout" not in cmd
    assert http_reconnect_args(url, max_delay=5)[-1] == "5"


@pytest.mark.parametrize("url", ["/tmp/input.m3u8", "file:/tmp/input.ts", "udp://test:1234"])
def test_non_http_inputs_do_not_need_command_repairs(url: str):
    assert http_reconnect_args(url) == []
    cmd = build_hls_ffmpeg_cmd(url, "software", "/tmp", input_args=())
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
        input_args=(),
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
        media = media_info(height=input_height, pix_fmt="yuv420p10le")
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,
            max_resolution=max_res,
            input_args=(),
        )
        vf = cmd[cmd.index("-vf") + 1]
        max_h = {"1080p": 1080, "720p": 720, "4k": 2160}[max_res]
        # Comma is escaped in FFmpeg filter expressions.
        height_expr = f"min(ih\\,{max_h})"
        assert height_expr in vf, f"Expected {height_expr} in {vf}"


class TestGpuDetection:
    """Tests for GPU/NVDEC detection."""

    @pytest.mark.parametrize(
        ("compute", "codecs"),
        [
            ("2.9", set[str]()),
            ("3.0", {"mpeg2video"}),
            ("5.0", {"mpeg2video", "h264"}),
            ("6.0", {"mpeg2video", "h264", "hevc"}),
            ("8.0", {"mpeg2video", "h264", "hevc", "av1"}),
        ],
    )
    def test_each_codec_decodes_from_its_first_generation(
        self,
        fake_run: FakeRun,
        caplog: pytest.LogCaptureFixture,
        compute: str,
        codecs: set[str],
    ):
        caplog.set_level("INFO", logger="netv.ffmpeg.commands")
        fake_run.program("nvidia-smi", Program(stdout=f"NVIDIA Card, X, {compute}\n"))
        assert _get_gpu_nvdec_codecs() == codecs
        # A bounded probe whose output is read as text.
        assert fake_run.options == [{"capture_output": True, "text": True, "timeout": 5}]
        named = ", ".join(sorted(codecs)) or "none"
        assert _command_log(caplog) == [
            ("INFO", f"GPU: NVIDIA Card, X (compute {float(compute):.1f}) NVDEC: {named}"),
        ]

    def test_a_failed_probe_says_why(self, fake_run: FakeRun, caplog: pytest.LogCaptureFixture):
        del fake_run
        caplog.set_level("DEBUG", logger="netv.ffmpeg.commands")
        assert _get_gpu_nvdec_codecs() == set()
        assert _command_log(caplog) == [
            ("DEBUG", "GPU probe failed: [Errno 2] No such file or directory: 'nvidia-smi'"),
        ]

    def test_nvidia_gpu_detected(self, fake_run: FakeRun):
        """Test NVIDIA GPU detection parses compute capability."""
        fake_run.program("nvidia-smi", Program(stdout="NVIDIA GeForce RTX 3080, 8.6\n"))
        assert _get_gpu_nvdec_codecs() == {"mpeg2video", "h264", "hevc", "av1"}
        assert fake_run.calls == [
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        ]

    def test_no_nvidia_gpu(self, fake_run: FakeRun, caplog: pytest.LogCaptureFixture):
        """Test handling when no NVIDIA GPU present."""
        caplog.set_level("INFO", logger="netv.ffmpeg.commands")
        fake_run.program(
            "nvidia-smi",
            Program(stdout="NVIDIA GeForce RTX 3080, 8.6\n", returncode=1),
        )
        assert _get_gpu_nvdec_codecs() == set()
        assert _command_log(caplog) == [("INFO", "No NVIDIA GPU detected")]

    def test_the_first_of_several_gpus_decides(self, fake_run: FakeRun):
        fake_run.program(
            "nvidia-smi",
            Program(stdout="NVIDIA GeForce RTX 3080, 8.6\nNVIDIA GeForce GTX 960, 5.2\n"),
        )
        assert _get_gpu_nvdec_codecs() == {"mpeg2video", "h264", "hevc", "av1"}

    def test_older_nvidia_gpu(self, fake_run: FakeRun):
        """Test older GPU with limited NVDEC support."""
        fake_run.program("nvidia-smi", Program(stdout="NVIDIA GeForce GTX 960, 5.2\n"))
        assert _get_gpu_nvdec_codecs() == {"mpeg2video", "h264"}
        # A Kepler card decodes broadcast MPEG-2 only.
        commands._gpu_nvdec_codecs = None
        fake_run.program("nvidia-smi", Program(stdout="NVIDIA GeForce GTX 780, 3.5\n"))
        assert _get_gpu_nvdec_codecs() == {"mpeg2video"}


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


def test_network_inputs_reach_the_provider_only_through_the_egress_proxy():
    save_home_lineup()
    tuner = "http://192.168.1.87:5004/auto/v5.1"
    for url, source_id in (("http://8.8.8.8/movie/u/p/1.mkv", ""), (tuner, "t")):
        expected = proxy.ffmpeg_input_args(url, source_id)
        cmd = build_hls_ffmpeg_cmd(url, "software", "/tmp", is_vod=True, input_args=expected)
        start = cmd.index("-protocol_whitelist")
        assert cmd[start : start + len(expected)] == expected
        assert start + len(expected) <= cmd.index("-i")
    assert option(cmd, "-http_proxy") == proxy.egress().url(lan=True)
    # The server's own files and pipes are read directly, with every protocol ffmpeg has.
    for local in ("/tmp/input.m3u8", "pipe:0"):
        assert proxy.ffmpeg_input_args(local, "t") == []


def _command_log(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    return [
        (r.levelname, r.getMessage()) for r in caplog.records if r.name == "netv.ffmpeg.commands"
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
        input_args=(),
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
        media_info=media_info(height=480, pix_fmt="yuv420p10le"),
        max_resolution="720p",
        allow_upscale=allow_upscale,
        input_args=(),
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
        media_info=None if height is None else media_info(height=height),
        max_resolution="4k",
        input_args=(),
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
        input_args=(),
    )
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-vf" not in cmd


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

    commands.init(lambda: settings, sr_engine_dir="/engines")
    assert probe_state / "logos" == cache.LOGOS_DIR
    assert commands._sr_engine_dir == "/engines"
    assert commands.get_settings() is settings
    assert list(probe._series_probe_cache[4]["episodes"]) == [1]


def test_init_without_engine_dir_disables_upscale(
    monkeypatch: pytest.MonkeyPatch,
    probe_state: Path,
):
    monkeypatch.setattr(commands, "_sr_engine_dir", "/stale")
    commands.init(_server_settings)
    assert commands._sr_engine_dir is None
    assert commands.get_settings() == _server_settings()


def test_transcode_dir_falls_back_when_inaccessible(probe_state: Path):
    blocker = probe_state / "file"
    blocker.write_text("")
    _save_transcode_dir(str(blocker / "x"))
    assert get_transcode_dir() == Path(tempfile.gettempdir())


def test_uses_audio_passthrough():
    assert commands.uses_audio_passthrough(media_info(audio_codec="eac3"), True)
    assert not commands.uses_audio_passthrough(media_info(audio_codec="eac3"), False)
    assert not commands.uses_audio_passthrough(media_info(audio_codec="aac"), True)
    assert not commands.uses_audio_passthrough(None, True)


def test_hls_media_info_adds_hls_demuxer_options():
    cmd = build_hls_ffmpeg_cmd(
        "http://x/live.m3u8",
        "software",
        "/out",
        media_info=media_info(is_hls=True),
        input_args=(),
    )
    i = cmd.index("-i")
    assert cmd[i - 4 : i] == ["-f", "hls", "-extension_picky", "0"]


class TestSrEngineSelection:
    def test_no_engine_dir(self, probe_state: Path):
        assert _find_sr_engine(720) is None
        assert _build_sr_filter(720, 1080) == ""

    def test_missing_engine_dir(self, monkeypatch: pytest.MonkeyPatch, probe_state: Path):
        monkeypatch.setattr(commands, "_sr_engine_dir", str(probe_state / "missing"))
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
        monkeypatch.setattr(commands, "_sr_engine_dir", str(probe_state))
        result = _find_sr_engine(0)
        assert result is not None
        assert result[1:] == (1080, 1920, 2)

    def test_source_larger_than_engines_uses_largest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        probe_state: Path,
    ):
        (probe_state / "2x-nomosuni-compact_480p_fp16.engine").write_bytes(b"")
        monkeypatch.setattr(commands, "_sr_engine_dir", str(probe_state))
        result = _find_sr_engine(1080)
        assert result is not None and result[1] == 480

    def test_filter_disabled_without_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        probe_state: Path,
    ):
        (probe_state / "2x-nomosuni-compact_720p_fp16.engine").write_bytes(b"")
        monkeypatch.setattr(commands, "_sr_engine_dir", str(probe_state))
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
        assert commands._has_libplacebo_filter() is available
        assert commands._has_libplacebo_filter() is available
        assert fake_run.calls == [["ffmpeg", "-filters"]]

    def test_libplacebo_probe_failure(self, fake_run: FakeRun):
        fake_run.program("ffmpeg", Program(error=OSError("no ffmpeg")))
        assert commands._has_libplacebo_filter() is False


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
        assert self._vf(post) == (
            "fps=source_fps,format=nv12,hwupload_cuda,scale_cuda=-2:min(ih\\,1080):format=nv12"
        )

    def test_nvenc_software_hdr_zscale_without_libplacebo(self, fake_run: FakeRun):
        _script_libplacebo(fake_run, available=False)
        _, post = self._args("nvenc+software", hdr=True)
        assert self._vf(post).startswith("fps=source_fps,zscale=t=linear")
        assert self._vf(post).endswith("hwupload_cuda,scale_cuda=-2:min(ih\\,1080):format=nv12")

    def test_amf_software_hdr(self):
        _, post = self._args("amf+software", hdr=True, deinterlace=True)
        vf = self._vf(post)
        assert vf.startswith("fps=source_fps,yadif=0:-1:1,zscale=t=linear")
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
        assert vf.startswith(
            "fps=source_fps," + ("vpp_qsv" if hw_pipeline else "format=nv12,hwupload"),
        )

    def test_software_hdr_and_unbounded_resolution(self):
        _, post = self._args("software", hdr=True)
        assert self._vf(post).startswith("fps=source_fps,zscale=t=linear")
        _, post = self._args("software", res="original")
        assert self._vf(post) == "fps=source_fps,format=yuv420p"

    @pytest.mark.parametrize(
        "hw",
        ["nvenc+software", "nvenc+vaapi", "vaapi", "qsv", "amf+software", "software"],
    )
    @pytest.mark.parametrize("hw_pipeline", [False, True])
    def test_every_encoder_paces_frames_before_its_filters(self, hw: HwAccel, hw_pipeline: bool):
        """Film with pulldown otherwise gets muxer-duplicated frames that garble CEA-608."""
        _, post = self._args(hw, hw_pipeline=hw_pipeline, deinterlace=True)
        assert self._vf(post).startswith("fps=source_fps,")


_VIDEO_ARGS_CASES = [
    pytest.param(
        "nvenc+software",
        {"hw_pipeline": True, "deinterlace": True, "hdr": False},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-hwaccel cuda -hwaccel_output_format cuda -extra_hw_frames 3",
        (
            "-vf fps=source_fps,yadif_cuda=0:-1:1,scale_cuda=-2:min(ih\\,"
            "1080):format=nv12 -c:v h264_nvenc -preset p4 -rc constqp -qp 20 "
            "-rc-lookahead 32 -bf 3 -spatial-aq 1 -temporal-aq 1 -g 60"
        ),
        id="cuda-deint",
    ),
    pytest.param(
        "nvenc+software",
        {"hw_pipeline": True, "deinterlace": False, "hdr": True},
        ("original", "medium"),
        {"libplacebo": True, "vaapi_baseline_only": False},
        "-hwaccel cuda -hwaccel_output_format cuda -extra_hw_frames 3",
        (
            "-vf fps=source_fps,hwdownload,format=p010le,"
            "libplacebo=tonemapping=hable:colorspace=bt709:color_primaries=bt709:co"
            "lor_trc=bt709,format=nv12,hwupload_cuda,scale_cuda=format=nv12 -c:v "
            "h264_nvenc -preset p2 -rc constqp -qp 28 -rc-lookahead 32 -bf 3 "
            "-spatial-aq 1 -temporal-aq 1 -g 60"
        ),
        id="cuda-hdr-placebo-orig",
    ),
    pytest.param(
        "nvenc+software",
        {"hw_pipeline": True, "deinterlace": False, "hdr": True},
        ("1080p", "low"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-hwaccel cuda -hwaccel_output_format cuda -extra_hw_frames 3",
        (
            "-vf fps=source_fps,hwdownload,format=p010le,zscale=t=linear:npl=100,"
            "format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,format=nv12,hwupload_cuda,"
            "scale_cuda=-2:min(ih\\,1080):format=nv12 -c:v h264_nvenc -preset p2 "
            "-rc constqp -qp 35 -rc-lookahead 32 -bf 3 -spatial-aq 1 -temporal-aq "
            "1 -g 60"
        ),
        id="cuda-hdr-zscale",
    ),
    pytest.param(
        "nvenc+software",
        {"hw_pipeline": False, "deinterlace": True, "hdr": True},
        ("1080p", "high"),
        {"libplacebo": True, "vaapi_baseline_only": False},
        "",
        (
            "-vf fps=source_fps,yadif=0:-1:1,"
            "libplacebo=tonemapping=hable:colorspace=bt709:color_primaries=bt709:co"
            "lor_trc=bt709,format=nv12,hwupload_cuda,scale_cuda=-2:min(ih\\,"
            "1080):format=nv12 -c:v h264_nvenc -preset p4 -rc constqp -qp 20 "
            "-rc-lookahead 32 -bf 3 -spatial-aq 1 -temporal-aq 1 -g 60"
        ),
        id="nvsw-deint-hdr-placebo",
    ),
    pytest.param(
        "nvenc+software",
        {"hw_pipeline": False, "deinterlace": False, "hdr": True},
        ("original", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "",
        (
            "-vf fps=source_fps,zscale=t=linear:npl=100,format=gbrpf32le,"
            "zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"
            "format=nv12,hwupload_cuda,scale_cuda=format=nv12 -c:v h264_nvenc "
            "-preset p2 -rc constqp -qp 20 -rc-lookahead 32 -bf 3 -spatial-aq 1 "
            "-temporal-aq 1 -g 60"
        ),
        id="nvsw-hdr-zscale-orig",
    ),
    pytest.param(
        "nvenc+software",
        {"hw_pipeline": False, "deinterlace": True, "hdr": False},
        ("original", "unknown"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "",
        (
            "-vf fps=source_fps,format=nv12,hwupload_cuda,yadif_cuda=0:-1:1,"
            "scale_cuda=format=nv12 -c:v h264_nvenc -preset p4 -rc constqp -qp 28 "
            "-rc-lookahead 32 -bf 3 -spatial-aq 1 -temporal-aq 1 -g 60"
        ),
        id="nvsw-deint-orig-default-q",
    ),
    pytest.param(
        "nvenc+software",
        {"hw_pipeline": False, "deinterlace": False, "hdr": False},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "",
        (
            "-vf fps=source_fps,format=nv12,hwupload_cuda,scale_cuda=-2:min(ih\\,"
            "1080):format=nv12 -c:v h264_nvenc -preset p2 -rc constqp -qp 20 "
            "-rc-lookahead 32 -bf 3 -spatial-aq 1 -temporal-aq 1 -g 60"
        ),
        id="nvsw-plain",
    ),
    pytest.param(
        "nvenc+vaapi",
        {"hw_pipeline": False, "deinterlace": True, "hdr": True},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        ("-hwaccel vaapi -hwaccel_output_format vaapi -hwaccel_device /dev/dri/renderD128"),
        (
            "-vf fps=source_fps,deinterlace_vaapi=auto=1,"
            "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709,"
            "scale_vaapi=w=-2:h=min(ih\\,1080):format=nv12,hwdownload,format=nv12,"
            "hwupload_cuda -c:v h264_nvenc -preset p4 -rc constqp -qp 20 "
            "-rc-lookahead 32 -bf 3 -spatial-aq 1 -temporal-aq 1 -g 60"
        ),
        id="nv-vaapi-deint-hdr",
    ),
    pytest.param(
        "nvenc+vaapi",
        {"hw_pipeline": False, "deinterlace": False, "hdr": False},
        ("original", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        ("-hwaccel vaapi -hwaccel_output_format vaapi -hwaccel_device /dev/dri/renderD128"),
        (
            "-vf fps=source_fps,scale_vaapi=format=nv12,hwdownload,format=nv12,"
            "hwupload_cuda -c:v h264_nvenc -preset p2 -rc constqp -qp 20 "
            "-rc-lookahead 32 -bf 3 -spatial-aq 1 -temporal-aq 1 -g 60"
        ),
        id="nv-vaapi-plain-orig",
    ),
    pytest.param(
        "amf+vaapi",
        {"hw_pipeline": False, "deinterlace": True, "hdr": True},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        ("-hwaccel vaapi -hwaccel_output_format vaapi -hwaccel_device /dev/dri/renderD128"),
        (
            "-vf fps=source_fps,deinterlace_vaapi=auto=1,"
            "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709,"
            "scale_vaapi=w=-2:h=min(ih\\,1080):format=nv12,hwdownload,format=nv12 "
            "-c:v h264_amf -rc cqp -qp_i 20 -qp_p 20 -quality balanced -g 60"
        ),
        id="amf-vaapi-deint-hdr",
    ),
    pytest.param(
        "amf+vaapi",
        {"hw_pipeline": False, "deinterlace": False, "hdr": False},
        ("original", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        ("-hwaccel vaapi -hwaccel_output_format vaapi -hwaccel_device /dev/dri/renderD128"),
        (
            "-vf fps=source_fps,scale_vaapi=format=nv12,hwdownload,format=nv12 "
            "-c:v h264_amf -rc cqp -qp_i 20 -qp_p 20 -quality balanced -g 60"
        ),
        id="amf-vaapi-plain-orig",
    ),
    pytest.param(
        "amf+software",
        {"hw_pipeline": False, "deinterlace": True, "hdr": False},
        ("original", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "",
        (
            "-vf fps=source_fps,yadif=0:-1:1,format=nv12 -c:v h264_amf -rc cqp "
            "-qp_i 20 -qp_p 20 -quality balanced -g 60"
        ),
        id="amf-sw-deint-orig",
    ),
    pytest.param(
        "amf+software",
        {"hw_pipeline": False, "deinterlace": False, "hdr": True},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "",
        (
            "-vf fps=source_fps,zscale=t=linear:npl=100,format=gbrpf32le,"
            "zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"
            "format=nv12,scale=-2:min(ih\\,1080),format=nv12 -c:v h264_amf -rc cqp "
            "-qp_i 20 -qp_p 20 -quality balanced -g 60"
        ),
        id="amf-sw-hdr",
    ),
    pytest.param(
        "amf+software",
        {"hw_pipeline": False, "deinterlace": False, "hdr": False},
        ("original", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "",
        (
            "-vf fps=source_fps,format=nv12 -c:v h264_amf -rc cqp -qp_i 20 -qp_p "
            "20 -quality balanced -g 60"
        ),
        id="amf-sw-plain-orig",
    ),
    pytest.param(
        "vaapi",
        {"hw_pipeline": True, "deinterlace": True, "hdr": True},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        (
            "-hwaccel vaapi -hwaccel_output_format vaapi -hwaccel_device "
            "/dev/dri/renderD128 -extra_hw_frames 3"
        ),
        (
            "-vf fps=source_fps,deinterlace_vaapi=auto=1,"
            "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709,"
            "scale_vaapi=w=-2:h=min(ih\\,1080):format=nv12 -c:v h264_vaapi -rc_mode "
            "CQP -qp 20 -bf 3 -g 60"
        ),
        id="vaapi-hw-deint-hdr",
    ),
    pytest.param(
        "vaapi",
        {"hw_pipeline": True, "deinterlace": False, "hdr": False},
        ("original", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        (
            "-hwaccel vaapi -hwaccel_output_format vaapi -hwaccel_device "
            "/dev/dri/renderD128 -extra_hw_frames 3"
        ),
        (
            "-vf fps=source_fps,scale_vaapi=format=nv12 -c:v h264_vaapi -rc_mode "
            "CQP -qp 20 -bf 3 -g 60"
        ),
        id="vaapi-hw-plain-orig",
    ),
    pytest.param(
        "vaapi",
        {"hw_pipeline": False, "deinterlace": True, "hdr": True},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-vaapi_device /dev/dri/renderD128",
        (
            "-vf fps=source_fps,format=nv12,hwupload,deinterlace_vaapi=auto=1,"
            "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709,"
            "scale_vaapi=w=-2:h=min(ih\\,1080):format=nv12 -c:v h264_vaapi -rc_mode "
            "CQP -qp 20 -bf 3 -g 60"
        ),
        id="vaapi-sw-deint-hdr",
    ),
    pytest.param(
        "vaapi",
        {"hw_pipeline": False, "deinterlace": False, "hdr": False},
        ("original", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-vaapi_device /dev/dri/renderD128",
        (
            "-vf fps=source_fps,format=nv12,hwupload,scale_vaapi=format=nv12 -c:v "
            "h264_vaapi -rc_mode CQP -qp 20 -bf 3 -g 60"
        ),
        id="vaapi-sw-plain-orig",
    ),
    pytest.param(
        "vaapi",
        {"hw_pipeline": False, "deinterlace": False, "hdr": False},
        ("1080p", "low"),
        {"libplacebo": False, "vaapi_baseline_only": True},
        "-vaapi_device /dev/dri/renderD128",
        (
            "-vf fps=source_fps,format=nv12,hwupload,scale_vaapi=w=-2:h=min(ih\\,"
            "1080):format=nv12 -c:v h264_vaapi -rc_mode CQP -qp 35 -profile:v "
            "constrained_baseline -g 60"
        ),
        id="vaapi-baseline",
    ),
    pytest.param(
        "qsv",
        {"hw_pipeline": True, "deinterlace": True, "hdr": True},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-hwaccel qsv -hwaccel_output_format qsv",
        (
            "-vf fps=source_fps,vpp_qsv=deinterlace=2:tonemap=1:format=nv12,"
            "scale_qsv=w=-2:h=min(ih\\,1080):format=nv12 -c:v h264_qsv "
            "-global_quality 20 -bf 3 -look_ahead 1 -look_ahead_depth 40 -g 60"
        ),
        id="qsv-hw-deint-hdr",
    ),
    pytest.param(
        "qsv",
        {"hw_pipeline": True, "deinterlace": True, "hdr": False},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-hwaccel qsv -hwaccel_output_format qsv",
        (
            "-vf fps=source_fps,vpp_qsv=deinterlace=2,scale_qsv=w=-2:h=min(ih\\,"
            "1080):format=nv12 -c:v h264_qsv -global_quality 20 -bf 3 -look_ahead "
            "1 -look_ahead_depth 40 -g 60"
        ),
        id="qsv-hw-deint",
    ),
    pytest.param(
        "qsv",
        {"hw_pipeline": True, "deinterlace": False, "hdr": True},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-hwaccel qsv -hwaccel_output_format qsv",
        (
            "-vf fps=source_fps,vpp_qsv=tonemap=1:format=nv12,"
            "scale_qsv=w=-2:h=min(ih\\,1080):format=nv12 -c:v h264_qsv "
            "-global_quality 20 -bf 3 -look_ahead 1 -look_ahead_depth 40 -g 60"
        ),
        id="qsv-hw-hdr",
    ),
    pytest.param(
        "qsv",
        {"hw_pipeline": True, "deinterlace": False, "hdr": False},
        ("original", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-hwaccel qsv -hwaccel_output_format qsv",
        (
            "-vf fps=source_fps,scale_qsv=format=nv12 -c:v h264_qsv "
            "-global_quality 20 -bf 3 -look_ahead 1 -look_ahead_depth 40 -g 60"
        ),
        id="qsv-hw-plain-orig",
    ),
    pytest.param(
        "qsv",
        {"hw_pipeline": False, "deinterlace": True, "hdr": True},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-init_hw_device qsv=hw -filter_hw_device hw",
        (
            "-vf fps=source_fps,format=nv12,hwupload=extra_hw_frames=64,"
            "vpp_qsv=deinterlace=2:tonemap=1:format=nv12,scale_qsv=w=-2:h=min(ih\\,"
            "1080):format=nv12 -c:v h264_qsv -global_quality 20 -bf 3 -look_ahead "
            "1 -look_ahead_depth 40 -g 60"
        ),
        id="qsv-sw-deint-hdr",
    ),
    pytest.param(
        "qsv",
        {"hw_pipeline": False, "deinterlace": True, "hdr": False},
        ("1080p", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-init_hw_device qsv=hw -filter_hw_device hw",
        (
            "-vf fps=source_fps,format=nv12,hwupload=extra_hw_frames=64,"
            "vpp_qsv=deinterlace=2,scale_qsv=w=-2:h=min(ih\\,1080):format=nv12 -c:v "
            "h264_qsv -global_quality 20 -bf 3 -look_ahead 1 -look_ahead_depth 40 "
            "-g 60"
        ),
        id="qsv-sw-deint",
    ),
    pytest.param(
        "qsv",
        {"hw_pipeline": False, "deinterlace": False, "hdr": True},
        ("original", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-init_hw_device qsv=hw -filter_hw_device hw",
        (
            "-vf fps=source_fps,format=nv12,hwupload=extra_hw_frames=64,"
            "vpp_qsv=tonemap=1:format=nv12,scale_qsv=format=nv12 -c:v h264_qsv "
            "-global_quality 20 -bf 3 -look_ahead 1 -look_ahead_depth 40 -g 60"
        ),
        id="qsv-sw-hdr-orig",
    ),
    pytest.param(
        "qsv",
        {"hw_pipeline": False, "deinterlace": False, "hdr": False},
        ("original", "high"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "-init_hw_device qsv=hw -filter_hw_device hw",
        (
            "-vf fps=source_fps,format=nv12,hwupload=extra_hw_frames=64,"
            "scale_qsv=format=nv12 -c:v h264_qsv -global_quality 20 -bf 3 "
            "-look_ahead 1 -look_ahead_depth 40 -g 60"
        ),
        id="qsv-sw-plain-orig",
    ),
    pytest.param(
        "software",
        {"hw_pipeline": False, "deinterlace": True, "hdr": True},
        ("1080p", "unknown"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "",
        (
            "-vf fps=source_fps,yadif=0:-1:1,zscale=t=linear:npl=100,"
            "format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,scale=-2:min(ih\\,1080),"
            "format=yuv420p -c:v libx264 -preset veryfast -crf 26 -bf 3 -g 60"
        ),
        id="x264-deint-hdr-default-q",
    ),
    pytest.param(
        "software",
        {"hw_pipeline": False, "deinterlace": False, "hdr": False},
        ("original", "medium"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "",
        ("-vf fps=source_fps,format=yuv420p -c:v libx264 -preset veryfast -crf 26 -bf 3 -g 60"),
        id="x264-plain-orig",
    ),
    pytest.param(
        "software",
        {"hw_pipeline": False, "deinterlace": False, "hdr": False},
        ("1080p", "low"),
        {"libplacebo": False, "vaapi_baseline_only": False},
        "",
        (
            "-vf fps=source_fps,scale=-2:min(ih\\,1080),format=yuv420p -c:v libx264 "
            "-preset veryfast -crf 32 -bf 3 -g 60"
        ),
        id="x264-low",
    ),
]


@pytest.mark.parametrize(("hw", "flags", "target", "env", "pre", "post"), _VIDEO_ARGS_CASES)
def test_video_args_per_encoder_branch(
    monkeypatch: pytest.MonkeyPatch,
    hw: HwAccel,
    flags: dict[str, bool],
    target: tuple[str, str],
    env: dict[str, bool],
    pre: str,
    post: str,
):
    monkeypatch.setattr(commands, "_has_libplacebo", env["libplacebo"])
    monkeypatch.setitem(cache.AVAILABLE_ENCODERS, "vaapi_baseline_only", env["vaapi_baseline_only"])
    resolution, quality = target
    actual_pre, actual_post = _build_video_args(
        copy_video=False,
        hw=hw,
        deinterlace=flags["deinterlace"],
        use_hw_pipeline=flags["hw_pipeline"],
        max_resolution=resolution,
        quality=quality,
        is_hdr=flags["hdr"],
    )
    assert (" ".join(actual_pre), " ".join(actual_post)) == (pre, post)


class TestVideoArgsUpscale:
    def _args(
        self,
        hw: str,
        *,
        resolution: str = "1080p",
        allow_upscale: bool = True,
    ) -> tuple[list[str], str]:
        pre, post = _build_video_args(
            copy_video=False,
            hw=hw,
            deinterlace=False,
            use_hw_pipeline=True,
            max_resolution=resolution,
            quality="high",
            allow_upscale=allow_upscale,
        )
        return pre, post[1]

    def test_unprobed_source_uses_the_largest_engine(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        engine = _enable_sr(monkeypatch, tmp_path)
        (tmp_path / "2x-nomosuni-compact_480p_fp16.engine").write_bytes(b"engine")
        pre, vf = self._args("nvenc+software")
        assert "-hwaccel" not in pre
        assert vf.endswith(f"model={engine},scale_cuda=w=-2:h=1080")

    def test_unbounded_resolution_keeps_the_engine_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        engine = _enable_sr(monkeypatch, tmp_path)
        _, vf = self._args("nvenc+software", resolution="original")
        assert vf.endswith(f"model={engine}")

    def test_amf_never_upscales(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        _enable_sr(monkeypatch, tmp_path)
        _, vf = self._args("amf+software")
        assert "dnn_processing" not in vf

    def test_disallowed_upscale_keeps_the_cuda_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        _enable_sr(monkeypatch, tmp_path)
        pre, vf = self._args("nvenc+software", allow_upscale=False)
        assert "-hwaccel" in pre
        assert "dnn_processing" not in vf


@pytest.mark.parametrize("hw", ["vaapi", "nvenc+vaapi", "amf+vaapi"])
def test_missing_vaapi_device_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    hw: str,
):
    monkeypatch.setattr(cache, "VAAPI_DEVICE", None)
    pre, _ = _build_video_args(
        copy_video=False,
        hw=hw,
        deinterlace=False,
        use_hw_pipeline=False,
        max_resolution="1080p",
        quality="high",
    )
    assert "-hwaccel" not in pre
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "VAAPI" in caplog.records[0].getMessage()


@pytest.mark.parametrize(
    ("url", "paced"),
    [
        ("http://host/movie.mkv", True),
        ("https://host/movie.mkv", True),
        ("/media/movie.mkv", False),
    ],
)
def test_vod_read_pacing_applies_to_network_sources(url: str, paced: bool):
    assert (
        "-readrate" in build_hls_ffmpeg_cmd(url, "software", "/output", True, input_args=())
    ) is paced


def _letterboxed(
    *,
    crop: CropBox | None = SCOPE,
    video_codec: str = "hevc",
    width: int = 1920,
    interlaced: bool = False,
    is_10bit: bool = False,
) -> MediaInfo:
    """A probed 1080p source with detected bars; as hevc, no client plays it copied."""
    return media_info(
        video_codec=video_codec,
        width=width,
        crop=crop,
        interlaced=interlaced,
        is_10bit=is_10bit,
    )


def _gpu(run: FakeRun) -> None:
    run.program("nvidia-smi", Program(stdout="NVIDIA GeForce GTX 960, 5.2\n"))


class TestLetterbox:
    def test_encoder_crop_needs_a_box_a_frame_size_and_a_reencode(self):
        assert encoder_crop(_letterboxed(), "1080p") == SCOPE
        assert encoder_crop(None, "1080p") is None
        assert encoder_crop(_letterboxed(crop=None), "1080p") is None
        assert encoder_crop(_letterboxed(crop=CropBox()), "1080p") is None
        assert encoder_crop(_letterboxed(width=0), "1080p") is None
        # A stream copy keeps every pixel.
        assert encoder_crop(_letterboxed(video_codec="h264"), "1080p") is None
        assert encoder_crop(_letterboxed(video_codec="h264"), "720p") == SCOPE

    @pytest.mark.parametrize("allow_upscale", [True, False])
    def test_encoder_crop_of_an_upscaled_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        allow_upscale: bool,
    ):
        _enable_sr(monkeypatch, tmp_path)
        crop = encoder_crop(_letterboxed(video_codec="h264"), "1080p", allow_upscale=allow_upscale)
        assert crop == (SCOPE if allow_upscale else None)

    def test_reencodes_detect_bars_in_an_output_before_the_main_one(self):
        media = _letterboxed(crop=None)
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/out",
            True,
            media_info=media,
            input_args=(),
        )
        detect = detection_args("/out", "", is_10bit=False)
        start = cmd.index("-filter:v") - 2
        assert cmd[start : start + len(detect) + 2] == [*detect, "-map", "0:v:0"]

    def test_stream_copies_detect_nothing(self):
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/out",
            True,
            media_info=media_info(),
            input_args=(),
        )
        assert "-filter:v" not in cmd

    @pytest.mark.parametrize(
        ("hw", "codec", "frames"),
        [
            ("software", "hevc", ""),
            ("nvenc+software", "h264", "cuda"),
            ("vaapi", "hevc", "vaapi"),
            ("qsv", "hevc", "qsv"),
        ],
    )
    def test_detection_reads_the_decoder_s_frames(
        self,
        fake_run: FakeRun,
        hw: HwAccel,
        codec: str,
        frames: str,
    ):
        _gpu(fake_run)
        media = _letterboxed(crop=None, video_codec=codec, interlaced=True, is_10bit=True)
        cmd = build_hls_ffmpeg_cmd("input", hw, "/out", True, media_info=media, input_args=())
        assert cmd[cmd.index("-filter:v") + 1] == detection_args("/out", frames, is_10bit=True)[3]

    def test_software_frames_crop_before_any_filter(self):
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/out",
            True,
            media_info=_letterboxed(),
            input_args=(),
        )
        assert cmd[cmd.index("-vf") + 1] == (
            "fps=source_fps,crop=1920:800:0:140,scale=-2:min(ih\\,1080),format=yuv420p"
        )

    @pytest.mark.parametrize(
        ("resolution", "size"),
        [("720p", "1728:720"), ("1080p", "1920:800"), ("original", "1920:800")],
    )
    def test_cuda_frames_crop_at_an_explicit_size(
        self,
        fake_run: FakeRun,
        resolution: str,
        size: str,
    ):
        _gpu(fake_run)
        media = _letterboxed(video_codec="h264", interlaced=True)
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "nvenc+software",
            "/out",
            True,
            media_info=media,
            max_resolution=resolution,
            input_args=(),
        )
        assert "-hwaccel" in cmd
        # scale_cuda sizes from the uncropped frame, so -2 would squash the picture.
        assert cmd[cmd.index("-vf") + 1] == (
            f"fps=source_fps,yadif_cuda=0:-1:1,crop=1920:800:0:140,scale_cuda={size}:format=nv12"
        )

    @pytest.mark.parametrize("hw", ["vaapi", "qsv", "nvenc+vaapi", "amf+vaapi"])
    def test_other_gpu_frames_decode_in_software_to_crop(self, hw: HwAccel):
        cmd = build_hls_ffmpeg_cmd(
            "input",
            hw,
            "/out",
            True,
            media_info=_letterboxed(),
            input_args=(),
        )
        assert "-hwaccel" not in cmd
        assert cmd[cmd.index("-vf") + 1].startswith("fps=source_fps,crop=1920:800:0:140,")
        assert cmd[cmd.index("-filter:v") + 1] == detection_args("/out", "", is_10bit=False)[3]

    def test_upscaling_crops_the_padding_it_adds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        engine = _enable_sr(monkeypatch, tmp_path)
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "nvenc+software",
            "/out",
            True,
            media_info=_letterboxed(),
            input_args=(),
        )
        vf = cmd[cmd.index("-vf") + 1]
        # The 800-line picture upscales; the 16:9 engine pads it to 1280x533 in 1280x720.
        assert vf.startswith("fps=source_fps,crop=1920:800:0:140,scale=1280:720:")
        assert vf.endswith(f"model={engine},crop=2560:1066:0:186,scale_cuda=w=2592:h=1080")

    def test_a_picture_already_past_the_target_is_not_upscaled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        _enable_sr(monkeypatch, tmp_path)
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "nvenc+software",
            "/out",
            True,
            media_info=_letterboxed(),
            max_resolution="720p",
            input_args=(),
        )
        # The picture's 800 lines, not its 140-line top bar, meet the 720p target.
        assert "dnn_processing" not in cmd[cmd.index("-vf") + 1]

    @pytest.mark.parametrize(("resolution", "cropped"), [("720p", True), ("1080p", False)])
    def test_a_copyable_source_is_cropped_only_when_re_encoded(
        self,
        resolution: str,
        cropped: bool,
    ):
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "software",
            "/out",
            True,
            media_info=_letterboxed(video_codec="h264"),
            max_resolution=resolution,
            input_args=(),
        )
        assert ("-vf" in cmd and "crop=" in cmd[cmd.index("-vf") + 1]) is cropped

    @pytest.mark.parametrize("allow_upscale", [True, False])
    def test_an_upscaled_source_is_re_encoded_and_cropped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        allow_upscale: bool,
    ):
        _enable_sr(monkeypatch, tmp_path)
        cmd = build_hls_ffmpeg_cmd(
            "input",
            "nvenc+software",
            "/out",
            True,
            media_info=_letterboxed(video_codec="h264"),
            allow_upscale=allow_upscale,
            input_args=(),
        )
        vf = cmd[cmd.index("-vf") + 1] if "-vf" in cmd else ""
        assert vf.startswith("fps=source_fps,crop=1920:800:0:140,") is allow_upscale


@pytest.mark.cli_ffmpeg
def test_letterboxed_source_encodes_without_its_bars(tmp_path: Path):
    source = tmp_path / "bars.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=1920x1080:r=10:d=4,drawbox=x=0:y=140:w=1920:h=800:c=white:t=fill",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "4",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    cmd = build_hls_ffmpeg_cmd(
        str(source),
        "software",
        str(tmp_path),
        True,
        media_info=_letterboxed(),
        input_args=(),
    )
    subprocess.run(cmd, check=True)
    size = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(tmp_path / "seg000.ts"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert size.stdout.split() == ["1920,800"]
    assert (tmp_path / DETECTION_FILE).exists()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
