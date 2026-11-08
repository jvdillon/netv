"""Tests for ffmpeg command generation and media probing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import json
import subprocess
import tempfile
import time

import pytest

from netv import ffmpeg_command
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
    probe_audio,
    probe_closed_captions,
    probe_media,
    probe_video_codec,
    resolve_hls_master_playlist,
    restore_probe_cache_entry,
)


@pytest.fixture(autouse=True)
def mock_vaapi_device():
    """Mock VAAPI_DEVICE for all tests to allow VAAPI tests on CI without hardware."""
    with patch("netv.cache.VAAPI_DEVICE", "/dev/dri/renderD128"):
        yield


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

    def test_nvenc_sr_uses_nomos(self, tmp_path: Path):
        engine = tmp_path / "2x-nomosuni-compact_720p_fp16.engine"
        engine.write_bytes(b"engine")
        retired_engine = tmp_path / "retired-model_720p_fp16.engine"
        retired_engine.write_bytes(b"engine")

        with (
            patch("netv.ffmpeg_command._sr_engine_dir", str(tmp_path)),
            patch(
                "netv.ffmpeg_command._load_settings",
                return_value={"sr_model": "retired-model"},
            ),
        ):
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
        tmp_path: Path,
        source_height: int,
        expected_height: int,
    ):
        for height in (480, 720, 1080):
            (tmp_path / f"2x-nomosuni-compact_{height}p_fp16.engine").write_bytes(b"engine")
        (tmp_path / "retired-model_576p_fp16.engine").write_bytes(b"engine")

        with patch("netv.ffmpeg_command._sr_engine_dir", str(tmp_path)):
            result = _find_sr_engine(source_height)

        assert result is not None
        engine_path, engine_height, _, scale = result
        assert engine_path.endswith(f"2x-nomosuni-compact_{expected_height}p_fp16.engine")
        assert engine_height == expected_height
        assert scale == 2

    def test_sr_maps_enabled_retired_setting_to_nomos(self, tmp_path: Path):
        engine = tmp_path / "2x-nomosuni-compact_1080p_fp16.engine"
        engine.write_bytes(b"engine")

        with (
            patch("netv.ffmpeg_command._sr_engine_dir", str(tmp_path)),
            patch(
                "netv.ffmpeg_command._load_settings",
                return_value={"sr_model": "retired-model"},
            ),
        ):
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
        assert "deinterlace_vaapi" in vf
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

    def test_sr_live_hls_uses_two_second_segments(self, tmp_path: Path):
        engine = tmp_path / "2x-nomosuni-compact_720p_fp16.engine"
        engine.write_bytes(b"engine")
        media_info = MediaInfo(
            video_codec="h264",
            audio_codec="aac",
            pix_fmt="yuv420p",
            audio_channels=2,
            audio_sample_rate=48_000,
            audio_profile="LC",
            height=720,
        )

        with (
            patch("netv.ffmpeg_command._sr_engine_dir", str(tmp_path)),
            patch(
                "netv.ffmpeg_command._load_settings",
                return_value={
                    "sr_model": "2x-nomosuni-compact",
                    "live_dvr_mins": 0,
                },
            ),
        ):
            command = build_hls_ffmpeg_cmd(
                "https://example.test/live.m3u8",
                "nvenc+software",
                str(tmp_path / "output"),
                media_info=media_info,
                max_resolution="4k",
                deinterlace_fallback=False,
            )

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

    @patch("netv.ffmpeg_command._has_libplacebo_filter", return_value=True)
    def test_nvenc_hdr_with_libplacebo(self, mock_placebo: MagicMock):
        """Test NVENC HDR uses libplacebo when available."""
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

    @patch("netv.ffmpeg_command._has_libplacebo_filter", return_value=False)
    def test_nvenc_hdr_zscale_fallback(self, mock_placebo: MagicMock):
        """Test NVENC HDR falls back to zscale when libplacebo unavailable."""
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

    @patch("netv.ffmpeg_command._has_libplacebo_filter", return_value=True)
    def test_nvenc_hdr_deinterlace_order(self, mock_placebo: MagicMock):
        """Test NVENC HDR hw decode deinterlaces BEFORE tonemap."""
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

    @patch("netv.ffmpeg_command._has_libplacebo_filter", return_value=True)
    def test_nvenc_sw_hdr_deinterlace_order(self, mock_placebo: MagicMock):
        """Test NVENC HDR sw decode uses CPU deinterlace before tonemap."""
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
        args = _build_audio_args(copy_audio=True, audio_sample_rate=48_000)
        assert args == ["-c:a", "copy"]

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
        args = _build_audio_args(copy_audio=False, audio_sample_rate=sample_rate)
        assert "-ar" in args
        assert expected in args

    @pytest.mark.parametrize("channels", [3, 6, 8])
    def test_surround_encodes_5_1(self, channels: int):
        """Surround sources stay surround instead of being downmixed to stereo."""
        args = _build_audio_args(
            copy_audio=False,
            audio_sample_rate=48_000,
            audio_channels=channels,
        )
        assert args[args.index("-af") + 1] == "aformat=channel_layouts=5.1"
        assert args[args.index("-b:a") + 1] == "384k"
        assert "-ac" not in args

    @pytest.mark.parametrize("channels", [1, 2])
    def test_stereo_stays_stereo(self, channels: int):
        args = _build_audio_args(
            copy_audio=False,
            audio_sample_rate=48_000,
            audio_channels=channels,
        )
        assert args[args.index("-ac") + 1] == "2"
        assert args[args.index("-b:a") + 1] == "192k"
        assert "-af" not in args

    def test_unknown_layout_keeps_surround(self):
        """Unprobed audio lets ffmpeg keep 5.1 or fall back to stereo at runtime."""
        args = _build_audio_args(copy_audio=False, audio_sample_rate=0)
        assert args[args.index("-af") + 1] == "aformat=channel_layouts=stereo|5.1"
        assert "-ac" not in args


# =============================================================================
# HLS Command Tests
# =============================================================================


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
        settings = _server_settings()
        settings["live_dvr_mins"] = dvr_mins
        with patch("netv.ffmpeg_command._load_settings", return_value=settings):
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
    def test_audio_info_without_media_info(
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
            audio_info=audio,
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
        settings = _server_settings()
        settings["live_dvr_mins"] = dvr_mins
        with patch("netv.ffmpeg_command._load_settings", return_value=settings):
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
def test_live_ai_upscale_uses_shared_bitrate_policy(is_vod: bool):
    with (
        patch("netv.ffmpeg_command._load_settings", return_value={"sr_model": "nomos"}),
        patch("netv.ffmpeg_command._sr_engine_dir", "/models"),
        patch("netv.ffmpeg_command._build_sr_filter", return_value="dnn_processing=model=test"),
    ):
        cmd = build_hls_ffmpeg_cmd(
            "https://provider.example/live.m3u8",
            "nvenc+software",
            "/tmp/output",
            is_vod=is_vod,
            max_resolution="4k",
        )
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

    def test_nvidia_gpu_detected(self):
        """Test NVIDIA GPU detection parses compute capability."""

        ffmpeg_command._gpu_nvdec_codecs = None  # Reset cache.

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 3080, 8.6\n"

        with patch("subprocess.run", return_value=mock_result):
            codecs = _get_gpu_nvdec_codecs()
            assert "h264" in codecs
            assert "hevc" in codecs
            assert "av1" in codecs

    def test_no_nvidia_gpu(self):
        """Test handling when no NVIDIA GPU present."""

        ffmpeg_command._gpu_nvdec_codecs = None

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            codecs = _get_gpu_nvdec_codecs()
            assert codecs == set()

    def test_older_nvidia_gpu(self):
        """Test older GPU with limited NVDEC support."""

        ffmpeg_command._gpu_nvdec_codecs = None

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce GTX 960, 5.2\n"

        with patch("subprocess.run", return_value=mock_result):
            codecs = _get_gpu_nvdec_codecs()
            assert "h264" in codecs
            assert "hevc" not in codecs
            assert "av1" not in codecs


# =============================================================================
# User Agent Tests
# =============================================================================


class TestUserAgent:
    """Tests for user agent handling."""

    def test_default_user_agent(self):
        """Test default preset returns None."""
        with patch(
            "netv.cache.load_server_settings",
            return_value={"user_agent_preset": "default"},
        ):
            assert get_user_agent() is None

    def test_vlc_user_agent(self):
        """Test VLC preset."""
        with patch("netv.cache.load_server_settings", return_value={"user_agent_preset": "vlc"}):
            ua = get_user_agent()
            assert ua is not None
            assert "VLC" in ua

    def test_chrome_user_agent(self):
        """Test Chrome preset."""
        with patch("netv.cache.load_server_settings", return_value={"user_agent_preset": "chrome"}):
            ua = get_user_agent()
            assert ua is not None
            assert "Chrome" in ua

    def test_custom_user_agent(self):
        """Test custom user agent."""
        with patch(
            "netv.cache.load_server_settings",
            return_value={"user_agent_preset": "custom", "user_agent_custom": "MyAgent/1.0"},
        ):
            assert get_user_agent() == "MyAgent/1.0"

    def test_custom_empty_returns_none(self):
        """Test empty custom user agent returns None."""
        with patch(
            "netv.cache.load_server_settings",
            return_value={"user_agent_preset": "custom", "user_agent_custom": ""},
        ):
            assert get_user_agent() is None


# =============================================================================
# Transcode Directory Tests
# =============================================================================


class TestTranscodeDir:
    """Tests for transcode directory handling."""

    def test_default_transcode_dir(self):
        """Test default uses system temp."""
        with patch("netv.ffmpeg_command._load_settings", return_value={}):
            path = get_transcode_dir()
            assert path == Path(tempfile.gettempdir())

    def test_custom_transcode_dir(self, tmp_path: Path):
        """Test custom directory is used and created."""
        custom_dir = tmp_path / "custom_transcode"
        with patch(
            "netv.ffmpeg_command._load_settings",
            return_value={"transcode_dir": str(custom_dir)},
        ):
            path = get_transcode_dir()
            assert path == custom_dir
            assert custom_dir.exists()


# =============================================================================
# HLS List Size Tests
# =============================================================================


class TestHlsListSize:
    """Tests for HLS list size calculation."""

    def test_default_list_size(self):
        """Test default (DVR disabled) uses 10 segments."""
        with patch("netv.ffmpeg_command._load_settings", return_value={}):
            assert get_live_hls_list_size() == 10

    def test_dvr_enabled_list_size(self):
        """Test DVR enabled calculates segments from minutes."""
        with patch("netv.ffmpeg_command._load_settings", return_value={"live_dvr_mins": 5}):
            # 5 min = 300 sec / 3 sec per segment = 100 segments.
            assert get_live_hls_list_size() == 100

    def test_dvr_list_size_follows_long_windows(self):
        with patch("netv.ffmpeg_command._load_settings", return_value={"live_dvr_mins": 360}):
            assert get_live_hls_list_size() == 7200


# =============================================================================
# Probe Media Tests
# =============================================================================


class TestProbeMedia:
    """Tests for media probing."""

    def test_probe_success(self):
        """Test successful probe parses media info."""

        ffmpeg_command._probe_cache.clear()

        probe_output: dict[str, object] = {
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
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(probe_output)

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("netv.ffmpeg_command._load_settings", return_value={}),
        ):
            media_info, subs = probe_media("http://test/video.mp4")

        assert media_info is not None
        assert media_info.video_codec == "h264"
        assert media_info.audio_codec == "aac"
        assert media_info.height == 1080
        assert media_info.duration == 3600.0
        assert not media_info.interlaced

    def test_probe_interlaced_detection(self):
        """Test interlaced content detection."""

        ffmpeg_command._probe_cache.clear()

        probe_output: dict[str, object] = {
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
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(probe_output)

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("netv.ffmpeg_command._load_settings", return_value={}),
        ):
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
    def test_probe_10bit_detection(self, pix_fmt: str, expected: bool):
        """Test 10-bit content detection from pix_fmt."""

        ffmpeg_command._probe_cache.clear()

        probe_output: dict[str, object] = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "pix_fmt": pix_fmt,
                    "height": 2160,
                },
            ],
            "format": {},
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(probe_output)

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("netv.ffmpeg_command._load_settings", return_value={}),
        ):
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
    def test_probe_hdr_detection(self, color_transfer: str, expected: bool):
        """Test HDR content detection from color_transfer."""

        ffmpeg_command._probe_cache.clear()

        probe_output: dict[str, object] = {
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
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(probe_output)

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("netv.ffmpeg_command._load_settings", return_value={}),
        ):
            media_info, _ = probe_media(f"http://test/{color_transfer or 'unknown'}.mkv")

        assert media_info is not None
        assert media_info.is_hdr is expected, (
            f"color_transfer={color_transfer} should be is_hdr={expected}"
        )

    def test_probe_failure(self):
        """Test probe failure returns None."""

        ffmpeg_command._probe_cache.clear()

        mock_result = MagicMock()
        mock_result.returncode = 1

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("netv.ffmpeg_command._load_settings", return_value={}),
        ):
            media_info, subs = probe_media("http://test/bad.mp4")

        assert media_info is None
        assert subs == []

    def test_probe_cache_hit(self):
        """Test probe cache returns cached result."""

        ffmpeg_command._probe_cache.clear()

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

        with (
            patch("subprocess.run") as mock_run,
            patch("netv.ffmpeg_command._load_settings", return_value={}),
        ):
            media_info, _ = probe_media("http://cached")

        mock_run.assert_not_called()
        assert media_info == cached_info

    def test_probe_extracts_subtitles(self):
        """Test subtitle stream extraction."""

        ffmpeg_command._probe_cache.clear()

        probe_output: dict[str, object] = {
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
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(probe_output)

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("netv.ffmpeg_command._load_settings", return_value={}),
        ):
            _, subs = probe_media("http://test/subs.mkv")

        assert len(subs) == 2
        assert subs[0].index == 2
        assert subs[0].lang == "eng"
        assert subs[0].name == "English"
        assert subs[1].index == 3
        assert subs[1].lang == "jpn"


# =============================================================================
# Probe Cache Management Tests
# =============================================================================


class TestProbeCacheManagement:
    """Tests for probe cache management functions."""

    def test_clear_all_probe_cache(self):
        """Test clearing all probe caches."""

        # Clear first to ensure known state.
        ffmpeg_command._probe_cache.clear()
        ffmpeg_command._series_probe_cache.clear()

        ffmpeg_command._probe_cache["url1"] = (time.time(), None, list[SubtitleStream]())
        ffmpeg_command._probe_cache["url2"] = (time.time(), None, list[SubtitleStream]())
        ffmpeg_command._series_probe_cache[123] = {
            "episodes": {1: (time.time(), None, list[SubtitleStream]())},
        }

        with patch("netv.ffmpeg_command._save_series_probe_cache"):
            count = clear_all_probe_cache()

        assert count == 3
        assert len(ffmpeg_command._probe_cache) == 0
        assert len(ffmpeg_command._series_probe_cache) == 0

    def test_invalidate_series_probe_cache_entire_series(self):
        """Test invalidating entire series cache."""

        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "episodes": {
                1: (0, None, list[SubtitleStream]()),
                2: (0, None, list[SubtitleStream]()),
            },
        }

        with patch("netv.ffmpeg_command._save_series_probe_cache"):
            invalidate_series_probe_cache(123)

        assert 123 not in ffmpeg_command._series_probe_cache

    def test_invalidate_series_probe_cache_single_episode(self):
        """Test invalidating single episode cache."""

        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "episodes": {
                1: (0, None, list[SubtitleStream]()),
                2: (0, None, list[SubtitleStream]()),
            },
        }

        with patch("netv.ffmpeg_command._save_series_probe_cache"):
            invalidate_series_probe_cache(123, episode_id=1)

        assert 123 in ffmpeg_command._series_probe_cache
        assert 1 not in ffmpeg_command._series_probe_cache[123]["episodes"]
        assert 2 in ffmpeg_command._series_probe_cache[123]["episodes"]

    def test_clear_series_mru(self):
        """Test clearing series MRU."""

        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "mru": 5,
            "episodes": {1: (0, None, list[SubtitleStream]())},
        }

        with patch("netv.ffmpeg_command._save_series_probe_cache"):
            clear_series_mru(123)

        assert "mru" not in ffmpeg_command._series_probe_cache[123]
        assert "episodes" in ffmpeg_command._series_probe_cache[123]

    def test_restore_probe_cache_entry(self):
        """Test restoring probe cache entry."""

        ffmpeg_command._probe_cache.clear()
        ffmpeg_command._series_probe_cache.clear()

        media_info = MediaInfo(video_codec="h264", audio_codec="aac", pix_fmt="yuv420p")
        subs = [SubtitleStream(index=2, lang="eng", name="English")]

        restore_probe_cache_entry("http://test", media_info, subs, series_id=123, episode_id=5)

        assert "http://test" in ffmpeg_command._probe_cache
        assert 123 in ffmpeg_command._series_probe_cache
        assert 5 in ffmpeg_command._series_probe_cache[123]["episodes"]

    def test_get_series_probe_cache_stats(self):
        """Test getting cache stats for UI."""

        ffmpeg_command._series_probe_cache.clear()
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

    def test_non_m3u8_url_returns_unchanged(self):
        """Non-m3u8 URLs should be returned unchanged."""

        url = "http://example.com/video.mp4"
        assert resolve_hls_master_playlist(url) == url

    def test_master_playlist_selects_highest_bandwidth(self):
        """Master playlist should resolve to highest bandwidth variant."""

        master_playlist = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=500000,RESOLUTION=640x360
low.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1500000,RESOLUTION=1280x720
mid.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1920x1080
high.m3u8
"""
        mock_response = MagicMock()
        mock_response.read.return_value = master_playlist.encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = resolve_hls_master_playlist("http://example.com/master.m3u8")

        assert result == "http://example.com/high.m3u8"

    def test_media_playlist_returns_unchanged(self):
        """Media playlist (not master) should return original URL."""

        media_playlist = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXTINF:10.0,
segment0.ts
#EXTINF:10.0,
segment1.ts
"""
        mock_response = MagicMock()
        mock_response.read.return_value = media_playlist.encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = resolve_hls_master_playlist("http://example.com/stream.m3u8")

        assert result == "http://example.com/stream.m3u8"

    def test_fetch_error_returns_original_url(self):
        """On fetch error, should return original URL."""

        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = resolve_hls_master_playlist("http://example.com/master.m3u8")

        assert result == "http://example.com/master.m3u8"

    def test_relative_url_resolved_correctly(self):
        """Relative variant URLs should be resolved against base URL."""

        master_playlist = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=3000000
../streams/1080p/index.m3u8
"""
        mock_response = MagicMock()
        mock_response.read.return_value = master_playlist.encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = resolve_hls_master_playlist("http://example.com/hls/master.m3u8")

        assert result == "http://example.com/streams/1080p/index.m3u8"


@pytest.mark.parametrize("hw", ["software", "nvenc+software", "amf+software", "qsv", "vaapi"])
def test_bandwidth_saver_bypasses_ai_and_scales_down(hw: HwAccel):
    media_info = MediaInfo(
        video_codec="h264",
        audio_codec="aac",
        pix_fmt="yuv420p",
        audio_channels=2,
        audio_sample_rate=48_000,
        audio_profile="LC",
        height=2160,
    )
    with (
        patch("netv.ffmpeg_command._load_settings", return_value={"sr_model": "nomos"}),
        patch("netv.ffmpeg_command._sr_engine_dir", "/models"),
        patch("netv.ffmpeg_command._build_sr_filter") as sr_filter,
    ):
        cmd = build_hls_ffmpeg_cmd(
            "http://example.com/live.ts",
            hw,
            "/tmp/hls",
            media_info=media_info,
            max_resolution="720p",
            quality="low",
            allow_upscale=False,
        )
    sr_filter.assert_not_called()
    assert "720" in " ".join(cmd)
    assert "dnn_processing" not in " ".join(cmd)
    assert cmd[cmd.index("-c:v") + 1] != "copy"


@pytest.mark.parametrize("height", [720, 1080, 2160])
def test_archive_policy_copies_compatible_source_without_ai_upscaling(height: int):
    media_info = MediaInfo(
        video_codec="h264",
        audio_codec="aac",
        pix_fmt="yuv420p",
        audio_channels=2,
        audio_sample_rate=48_000,
        audio_profile="LC",
        height=height,
    )
    with (
        patch("netv.ffmpeg_command._load_settings", return_value={"sr_model": "upscale-model"}),
        patch("netv.ffmpeg_command._sr_engine_dir", "/models"),
        patch("netv.ffmpeg_command._build_sr_filter") as sr_filter,
    ):
        cmd = build_hls_ffmpeg_cmd(
            "https://upstream.test/timeshift/user/pass/60/2026-09-28:10-00/1.ts",
            "nvenc+software",
            "/tmp/hls",
            is_vod=True,
            media_info=media_info,
            max_resolution="4k",
            allow_upscale=False,
        )
    sr_filter.assert_not_called()
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-vf" not in cmd


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")
def test_probe_audio_reads_local_surround_segment(tmp_path: Path):
    """probe_audio reports the layout that decides 5.1 handling."""

    path = tmp_path / "input.ts"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=48000",
            "-af",
            "pan=5.1|c0=c0|c1=c0|c2=c0|c3=c0|c4=c0|c5=c0",
            "-t",
            "1",
            "-c:a",
            "ac3",
            str(path),
        ],
        check=True,
    )
    info = probe_audio(str(path))
    assert info is not None
    assert (info.audio_codec, info.audio_channels, info.audio_sample_rate) == ("ac3", 6, 48_000)
    assert probe_audio(str(tmp_path / "missing.ts")) is None


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")
def test_probe_video_codec_reads_local_segment(tmp_path: Path):

    path = tmp_path / "input.ts"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64",
            "-t",
            "1",
            "-c:v",
            "mpeg2video",
            str(path),
        ],
        check=True,
    )
    assert probe_video_codec(str(path)) == "mpeg2video"
    assert probe_video_codec(str(tmp_path / "missing.ts")) == ""


@pytest.mark.parametrize(
    "side_data_type, expected",
    [
        ("ATSC A53 Part 4 Closed Captions", True),
        ("H.264 User Data Unregistered SEI message", False),
    ],
)
def test_probe_closed_captions_reads_video_side_data(side_data_type: str, expected: bool):
    result = MagicMock(
        returncode=0,
        stdout=json.dumps(
            {
                "frames": [{"side_data_list": [{"side_data_type": side_data_type}]}],
            },
        ),
    )
    with patch("netv.ffmpeg_command.subprocess.run", return_value=result) as run:
        assert probe_closed_captions("/tmp/input.ts") is expected

    command = run.call_args.args[0]
    assert command[command.index("-read_intervals") + 1] == "%+2"
    assert command[command.index("-show_entries") + 1] == "frame=side_data_list"


# =============================================================================
# Module state, caches, persistence, and probing edge cases
# =============================================================================


@pytest.fixture
def probe_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Isolate probe caches, settings, and the on-disk cache directory."""
    monkeypatch.setattr(ffmpeg_command, "_probe_cache", {})
    monkeypatch.setattr(ffmpeg_command, "_series_probe_cache", {})
    monkeypatch.setattr(ffmpeg_command, "_load_settings", dict)
    monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", None)
    monkeypatch.setattr(ffmpeg_command, "get_user_agent", lambda: None)
    monkeypatch.setattr(ffmpeg_command.cache, "CACHE_DIR", tmp_path)
    return tmp_path


_VIDEO = {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "height": 720}


def _server_settings() -> ServerSettings:
    return {
        "transcode_mode": "auto",
        "transcode_hw": "software",
        "vod_transcode_cache_mins": 60,
        "live_transcode_cache_secs": 0,
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
    initialized: list[int] = []
    monkeypatch.setattr(ffmpeg_command.cache, "initialize", lambda: initialized.append(1))
    settings = _server_settings()
    settings["sr_model"] = "x"

    ffmpeg_command.init(lambda: settings, sr_engine_dir="/engines")
    assert initialized == [1]
    assert ffmpeg_command._sr_engine_dir == "/engines"
    assert ffmpeg_command.get_settings() is settings


def test_init_without_engine_dir_disables_upscale(
    monkeypatch: pytest.MonkeyPatch,
    probe_state: Path,
):
    monkeypatch.setattr(ffmpeg_command.cache, "initialize", lambda: None)
    monkeypatch.setattr(ffmpeg_command, "_sr_engine_dir", "/stale")
    ffmpeg_command.init(_server_settings)
    assert ffmpeg_command._sr_engine_dir is None
    assert ffmpeg_command.get_settings() == _server_settings()


def test_transcode_dir_falls_back_when_inaccessible(probe_state: Path):
    blocker = probe_state / "file"
    blocker.write_text("")
    with patch(
        "netv.ffmpeg_command._load_settings",
        return_value={"transcode_dir": str(blocker / "x")},
    ):
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
    with patch("netv.ffmpeg_command._save_series_probe_cache") as save:
        invalidate_series_probe_cache(999)
        invalidate_series_probe_cache(999, episode_id=1)
        clear_series_mru(999)
    save.assert_not_called()
    assert ffmpeg_command._series_probe_cache == {}


def test_master_playlist_without_variant_urls_returns_original(probe_state: Path):
    response = MagicMock()
    response.read.return_value = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=abc\n#comment\n"
    response.__enter__.return_value = response
    url = "http://example.com/master.m3u8"
    with (
        patch("netv.ffmpeg_command.get_user_agent", return_value="UA/1"),
        patch("netv.ffmpeg_command.request.urlopen", return_value=response) as urlopen,
    ):
        assert resolve_hls_master_playlist(url) == url
    assert urlopen.call_args.args[0].get_header("User-agent") == "UA/1"


class TestProbeMediaSeriesCache:
    """Series/episode cache paths in probe_media."""

    def test_exact_episode_hit_updates_mru_and_saves(self, probe_state: Path):
        info = _media_info()
        ffmpeg_command._series_probe_cache[7] = {
            "name": "S",
            "mru": 1,
            "episodes": {2: (time.time(), info, list[SubtitleStream]())},
        }
        with patch("subprocess.run") as run:
            assert probe_media("http://x", series_id=7, episode_id=2) == (info, [])
            # Same MRU now: hit without another save.
            assert probe_media("http://x", series_id=7, episode_id=2) == (info, [])
        run.assert_not_called()
        saved = json.loads((probe_state / "series_probe_cache.json").read_text())
        assert saved["7"]["mru"] == 2

    def test_mru_fallback_hit(self, probe_state: Path):
        info = _media_info()
        ffmpeg_command._series_probe_cache[7] = {
            "mru": 1,
            "episodes": {1: (time.time(), info, list[SubtitleStream]())},
        }
        with patch("subprocess.run") as run:
            assert probe_media("http://x", series_id=7, episode_id=99) == (info, [])
        run.assert_not_called()

    @pytest.mark.parametrize("episode_id", [1, 99])
    def test_expired_series_entries_fall_through_to_probe(self, probe_state: Path, episode_id: int):
        ffmpeg_command._series_probe_cache[7] = {
            "mru": 1,
            "episodes": {1: (0, _media_info(video_codec="old"), list[SubtitleStream]())},
        }
        with patch(
            "subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout=json.dumps({"streams": [_VIDEO], "format": {}}),
            ),
        ):
            info, _ = probe_media("http://x", series_id=7, episode_id=episode_id, series_name="N")
        assert info is not None and info.video_codec == "h264"
        series = ffmpeg_command._series_probe_cache[7]
        assert "name" in series
        assert series["name"] == "N"
        assert series["mru"] == episode_id
        assert series["episodes"][episode_id][1] is info

    def test_new_series_is_cached_and_persisted(self, probe_state: Path):
        with patch(
            "subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout=json.dumps({"streams": [_VIDEO], "format": {}}),
            ),
        ):
            info, _ = probe_media("http://x", series_id=8, series_name="New")
        series = ffmpeg_command._series_probe_cache[8]
        assert "name" in series
        assert series["name"] == "New"
        assert "mru" in series
        assert series["mru"] == 0
        assert series["episodes"][0][1] is info
        saved = json.loads((probe_state / "series_probe_cache.json").read_text())
        assert saved["8"]["episodes"]["0"]["video_codec"] == "h264"

    def test_existing_named_series_keeps_name(self, probe_state: Path):
        ffmpeg_command._series_probe_cache[8] = {
            "name": "Kept",
            "episodes": dict[int, ffmpeg_command._ProbeEntry](),
        }
        with patch(
            "subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout=json.dumps({"streams": [_VIDEO], "format": {}}),
            ),
        ):
            probe_media("http://x", series_id=8, episode_id=3, series_name="Other")
        assert ffmpeg_command._series_probe_cache[8]["name"] == "Kept"
        assert 3 in ffmpeg_command._series_probe_cache[8]["episodes"]


class TestProbeMediaEdgeCases:
    def test_subprocess_exceptions_yield_none(self, probe_state: Path):
        with patch("subprocess.run", side_effect=OSError("boom")) as run:
            assert probe_media("http://x") == (None, [])
        assert run.call_count == 2
        assert run.call_args.args[0][-5:] == ["-f", "hls", "-extension_picky", "0", "http://x"]

    def test_user_agent_and_format_bitrate_fallback(self, probe_state: Path):
        fmt = {"bit_rate": "1234", "format_name": "hls"}
        with (
            patch("netv.ffmpeg_command.get_user_agent", return_value="UA/1"),
            patch(
                "subprocess.run",
                return_value=MagicMock(
                    returncode=0,
                    stdout=json.dumps({"streams": [_VIDEO], "format": fmt}),
                ),
            ) as run,
        ):
            info, _ = probe_media("http://x")
        assert info is not None
        assert info.video_bitrate == 1234
        assert info.is_hls
        cmd = run.call_args.args[0]
        assert cmd[cmd.index("-user_agent") + 1] == "UA/1"

    def test_invalid_format_bitrate_ignored(self, probe_state: Path):
        with patch(
            "subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout=json.dumps({"streams": [_VIDEO], "format": {"bit_rate": "n/a"}}),
            ),
        ):
            info, _ = probe_media("http://x")
        assert info is not None and info.video_bitrate == 0

    def test_audio_only_returns_none(self, probe_state: Path):
        audio = {"codec_type": "audio", "codec_name": "aac", "channels": 2}
        with patch(
            "subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout=json.dumps({"streams": [audio], "format": {}}),
            ),
        ):
            assert probe_media("http://x") == (None, [])

    def test_zero_height_not_cached(self, probe_state: Path):
        with patch(
            "subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout=json.dumps({"streams": [{**_VIDEO, "height": 0}], "format": {}}),
            ),
        ):
            info, _ = probe_media("http://x", series_id=1)
        assert info is not None and info.height == 0
        assert ffmpeg_command._probe_cache == {}
        assert ffmpeg_command._series_probe_cache == {}


def test_uses_audio_passthrough():
    assert ffmpeg_command.uses_audio_passthrough(_media_info(audio_codec="eac3"), True)
    assert not ffmpeg_command.uses_audio_passthrough(_media_info(audio_codec="eac3"), False)
    assert not ffmpeg_command.uses_audio_passthrough(_media_info(audio_codec="aac"), True)
    assert not ffmpeg_command.uses_audio_passthrough(None, True)


def test_probe_closed_captions_reports_ffprobe_failure(caplog: pytest.LogCaptureFixture):
    result = MagicMock(returncode=1, stderr="  bad input \n", stdout="")
    with patch("netv.ffmpeg_command.subprocess.run", return_value=result):
        assert probe_closed_captions("/tmp/dir/a.ts") is False
    assert "a.ts: bad input" in caplog.text

    result = MagicMock(returncode=2, stderr="", stdout="")
    with patch("netv.ffmpeg_command.subprocess.run", return_value=result):
        assert probe_closed_captions("/tmp/dir/a.ts") is False
    assert "ffprobe exited 2" in caplog.text


@pytest.mark.parametrize(
    "run",
    [
        MagicMock(side_effect=subprocess.TimeoutExpired("ffprobe", 5)),
        MagicMock(return_value=MagicMock(returncode=0, stdout="not json")),
    ],
)
def test_probe_closed_captions_handles_errors(run: MagicMock, caplog: pytest.LogCaptureFixture):
    with patch("netv.ffmpeg_command.subprocess.run", new=run):
        assert probe_closed_captions("/tmp/a.ts") is False
    assert "Could not inspect embedded captions in a.ts" in caplog.text


@pytest.mark.parametrize("stdout", ['["not", "a", "dict"]', '{"frames": "not a list"}'])
def test_probe_closed_captions_rejects_malformed_shapes(stdout: str):
    """A probe of the wrong shape means no captions, not a crash."""
    result = MagicMock(returncode=0, stdout=stdout)
    with patch("netv.ffmpeg_command.subprocess.run", return_value=result):
        assert probe_closed_captions("/tmp/a.ts") is False


def test_probe_media_retries_hls_when_a_field_has_the_wrong_type(probe_state: Path):
    """A mistyped field fails that probe attempt; the HLS retry still succeeds."""
    bad = json.dumps({"streams": [{**_VIDEO, "codec_name": 7}], "format": {}})
    good = json.dumps({"streams": [_VIDEO], "format": {}})
    attempts = [MagicMock(returncode=0, stdout=bad), MagicMock(returncode=0, stdout=good)]
    with patch("subprocess.run", side_effect=attempts) as run:
        info, _ = probe_media("http://x/mistyped")
    assert info is not None
    assert info.video_codec == "h264"
    assert run.call_count == 2
    assert run.call_args_list[1].args[0][-5:] == [
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

    def test_missing_engine_dir(self, probe_state: Path):
        with patch("netv.ffmpeg_command._sr_engine_dir", str(probe_state / "missing")):
            assert _find_sr_engine(720) is None

    def test_no_valid_engines(self, probe_state: Path):
        (probe_state / "2x-nomosuni-compact_bigp_fp16.engine").write_bytes(b"")
        with (
            patch("netv.ffmpeg_command._sr_engine_dir", str(probe_state)),
            patch("netv.ffmpeg_command._load_settings", return_value={"sr_model": "on"}),
        ):
            assert _find_sr_engine(720) is None
            assert _build_sr_filter(720, 1080) == ""

    def test_failed_probe_uses_largest_engine(self, probe_state: Path):
        for height in (480, 1080):
            (probe_state / f"2x-nomosuni-compact_{height}p_fp16.engine").write_bytes(b"")
        with patch("netv.ffmpeg_command._sr_engine_dir", str(probe_state)):
            result = _find_sr_engine(0)
        assert result is not None
        assert result[1:] == (1080, 1920, 2)

    def test_source_larger_than_engines_uses_largest(self, probe_state: Path):
        (probe_state / "2x-nomosuni-compact_480p_fp16.engine").write_bytes(b"")
        with patch("netv.ffmpeg_command._sr_engine_dir", str(probe_state)):
            result = _find_sr_engine(1080)
        assert result is not None and result[1] == 480

    def test_filter_disabled_without_model(self, probe_state: Path):
        (probe_state / "2x-nomosuni-compact_720p_fp16.engine").write_bytes(b"")
        with patch("netv.ffmpeg_command._sr_engine_dir", str(probe_state)):
            assert _build_sr_filter(480, 1080) == ""

    def test_filter_skipped_when_source_meets_target(self, probe_state: Path):
        (probe_state / "2x-nomosuni-compact_1080p_fp16.engine").write_bytes(b"")
        with (
            patch("netv.ffmpeg_command._sr_engine_dir", str(probe_state)),
            patch("netv.ffmpeg_command._load_settings", return_value={"sr_model": "on"}),
        ):
            assert _build_sr_filter(1080, 1080) == ""
            no_target = _build_sr_filter(720, 0)
        assert no_target.endswith(
            "dnn_processing=dnn_backend=tensorrt:model="
            + str(
                probe_state / "2x-nomosuni-compact_1080p_fp16.engine",
            ),
        )
        assert "scale_cuda" not in no_target


class TestHardwareProbes:
    def test_nvidia_smi_unparseable_output(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ffmpeg_command, "_gpu_nvdec_codecs", None)
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="GPU only\n")):
            assert _get_gpu_nvdec_codecs() == set()

    def test_nvidia_smi_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ffmpeg_command, "_gpu_nvdec_codecs", None)
        with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi")):
            assert _get_gpu_nvdec_codecs() == set()
        # Cached: no second probe.
        with patch("subprocess.run") as run:
            assert _get_gpu_nvdec_codecs() == set()
        run.assert_not_called()

    @pytest.mark.parametrize(("stdout", "expected"), [("... libplacebo ...", True), ("", False)])
    def test_libplacebo_probe(self, monkeypatch: pytest.MonkeyPatch, stdout: str, expected: bool):
        monkeypatch.setattr(ffmpeg_command, "_has_libplacebo", None)
        with patch("subprocess.run", return_value=MagicMock(stdout=stdout)) as run:
            assert ffmpeg_command._has_libplacebo_filter() is expected
            assert ffmpeg_command._has_libplacebo_filter() is expected
        assert run.call_count == 1
        assert run.call_args.args[0] == ["ffmpeg", "-filters"]

    def test_libplacebo_probe_failure(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ffmpeg_command, "_has_libplacebo", None)
        with patch("subprocess.run", side_effect=OSError("no ffmpeg")):
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

    def test_missing_vaapi_device_falls_back_to_software(self):
        with patch("netv.cache.VAAPI_DEVICE", None):
            pre, post = self._args("vaapi", hw_pipeline=True)
            assert pre == []
            assert post[post.index("-c:v") + 1] == "libx264"

            pre, post = self._args("nvenc+vaapi")
            assert pre == []
            assert (
                self._vf(post)
                == "format=nv12,hwupload_cuda,scale_cuda=-2:min(ih\\,1080):format=nv12"
            )

    def test_nvenc_software_hdr_zscale_without_libplacebo(self):
        with patch("netv.ffmpeg_command._has_libplacebo_filter", return_value=False):
            _, post = self._args("nvenc+software", hdr=True)
        assert self._vf(post).startswith("zscale=t=linear")
        assert self._vf(post).endswith("hwupload_cuda,scale_cuda=-2:min(ih\\,1080):format=nv12")

    def test_amf_software_hdr(self):
        _, post = self._args("amf+software", hdr=True, deinterlace=True)
        vf = self._vf(post)
        assert vf.startswith("yadif=0,zscale=t=linear")
        assert vf.endswith("format=nv12,scale=-2:min(ih\\,1080),format=nv12")

    def test_vaapi_baseline_only(self):
        with patch.dict("netv.cache.AVAILABLE_ENCODERS", {"vaapi_baseline_only": True}):
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
