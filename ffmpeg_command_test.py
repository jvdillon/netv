"""Tests for ffmpeg command generation and media probing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import json
import tempfile

import pytest

from ffmpeg_command import (
    _MAX_RES_HEIGHT,
    HwAccel,
    MediaInfo,
    SubtitleStream,
    _build_audio_args,
    _build_video_args,
    _get_gpu_nvdec_codecs,
    build_hls_ffmpeg_cmd,
    clear_all_probe_cache,
    clear_series_mru,
    get_live_hls_list_size,
    get_series_probe_cache_stats,
    get_transcode_dir,
    get_user_agent,
    invalidate_series_probe_cache,
    probe_media,
    restore_probe_cache_entry,
)


@pytest.fixture(autouse=True)
def mock_vaapi_device():
    """Mock VAAPI_DEVICE for all tests to allow VAAPI tests on CI without hardware."""
    with patch("ffmpeg_command.VAAPI_DEVICE", "/dev/dri/renderD128"):
        yield


class FakeMediaInfo:
    """Fake media info for testing."""

    def __init__(
        self,
        video_codec: str = "h264",
        audio_codec: str = "aac",
        pix_fmt: str = "yuv420p",
        audio_channels: int = 2,
        audio_sample_rate: int = 48000,
        audio_profile: str = "LC",
        height: int = 1080,
        interlaced: bool = False,
        is_10bit: bool = False,
        is_hdr: bool = False,
        is_hls: bool = False,
    ):
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.pix_fmt = pix_fmt
        self.audio_channels = audio_channels
        self.audio_sample_rate = audio_sample_rate
        self.audio_profile = audio_profile
        self.height = height
        self.interlaced = interlaced
        self.is_10bit = is_10bit
        self.is_hdr = is_hdr
        self.is_hls = is_hls


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
        # Upload to GPU, then deinterlace (mode=0 for original framerate) and scale on GPU
        assert "hwupload_cuda" in vf
        assert "yadif_cuda=0" in vf
        assert "scale_cuda" in vf

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
        "quality,expected_qp", [("high", "20"), ("medium", "28"), ("low", "35")]
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

    def test_invalid_hw_raises(self):
        """Test invalid hardware raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized hardware"):
            _build_video_args(
                copy_video=False,
                hw="invalid",  # type: ignore
                deinterlace=False,
                use_hw_pipeline=False,
                max_resolution="1080p",
                quality="high",
            )

    @patch("ffmpeg_command._has_libplacebo_filter", return_value=True)
    def test_nvenc_hdr_with_libplacebo(self, mock_placebo):
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
        # Should download from CUDA, process, re-upload
        assert "hwdownload" in vf
        assert "hwupload_cuda" in vf

    @patch("ffmpeg_command._has_libplacebo_filter", return_value=False)
    def test_nvenc_hdr_zscale_fallback(self, mock_placebo):
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

    @patch("ffmpeg_command._has_libplacebo_filter", return_value=True)
    def test_nvenc_hdr_deinterlace_order(self, mock_placebo):
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
        # Deinterlace should come before tonemap in hw decode path
        deint_pos = vf.find("yadif_cuda")
        tonemap_pos = vf.find("libplacebo")
        assert deint_pos < tonemap_pos, f"deinterlace should come before tonemap: {vf}"

    @patch("ffmpeg_command._has_libplacebo_filter", return_value=True)
    def test_nvenc_sw_hdr_deinterlace_order(self, mock_placebo):
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
        # SW decode HDR should use CPU yadif before tonemap
        assert "yadif=0" in vf  # CPU deinterlace, not yadif_cuda
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
        args = _build_audio_args(copy_audio=True, audio_sample_rate=48000)
        assert args == ["-c:a", "copy"]

    @pytest.mark.parametrize(
        "sample_rate,expected",
        [
            (44100, "44100"),
            (48000, "48000"),
            (96000, "48000"),
            (0, "48000"),
        ],
    )
    def test_sample_rates(self, sample_rate: int, expected: str):
        """Test sample rate handling."""
        args = _build_audio_args(copy_audio=False, audio_sample_rate=sample_rate)
        assert "-ar" in args
        assert expected in args


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

    def test_live_hls_flags(self):
        """Test live has correct HLS flags."""
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", is_vod=False)
        assert "delete_segments" in cmd
        assert "-hls_list_size" in cmd
        assert cmd[cmd.index("-hls_list_size") + 1] == "10"

    def test_copy_video_with_compatible_media(self):
        """Test copy_video is used for compatible VOD media."""
        media = FakeMediaInfo(video_codec="h264", pix_fmt="yuv420p", height=1080)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,  # type: ignore
            max_resolution="1080p",
        )
        assert "-c:v" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-hwaccel" not in cmd

    def test_no_copy_for_10bit(self):
        """Test 10-bit content is transcoded, not copied."""
        media = FakeMediaInfo(video_codec="h264", pix_fmt="yuv420p10le", height=1080)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,  # type: ignore
        )
        assert cmd[cmd.index("-c:v") + 1] != "copy"
        assert "-hwaccel" in cmd

    def test_no_copy_when_scaling_needed(self):
        """Test scaling requirement prevents copy."""
        media = FakeMediaInfo(video_codec="h264", pix_fmt="yuv420p", height=2160)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,  # type: ignore
            max_resolution="1080p",
        )
        assert cmd[cmd.index("-c:v") + 1] != "copy"

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
        media = FakeMediaInfo()
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", media_info=media)  # type: ignore
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
        media = FakeMediaInfo(height=input_height, pix_fmt="yuv420p10le")
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,  # type: ignore
            max_resolution=max_res,
        )
        vf = cmd[cmd.index("-vf") + 1]
        max_h = _MAX_RES_HEIGHT.get(max_res, 9999)
        # Comma is escaped in FFmpeg filter expressions
        height_expr = f"min(ih\\,{max_h})"
        assert height_expr in vf, f"Expected {height_expr} in {vf}"


# =============================================================================
# GPU Detection Tests
# =============================================================================


class TestGpuDetection:
    """Tests for GPU/NVDEC detection."""

    def test_nvidia_gpu_detected(self):
        """Test NVIDIA GPU detection parses compute capability."""
        import ffmpeg_command

        ffmpeg_command._gpu_nvdec_codecs = None  # Reset cache

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
        import ffmpeg_command

        ffmpeg_command._gpu_nvdec_codecs = None

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            codecs = _get_gpu_nvdec_codecs()
            assert codecs == set()

    def test_older_nvidia_gpu(self):
        """Test older GPU with limited NVDEC support."""
        import ffmpeg_command

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
        with patch("cache.load_server_settings", return_value={"user_agent_preset": "default"}):
            assert get_user_agent() is None

    def test_vlc_user_agent(self):
        """Test VLC preset."""
        with patch("cache.load_server_settings", return_value={"user_agent_preset": "vlc"}):
            ua = get_user_agent()
            assert ua is not None
            assert "VLC" in ua

    def test_chrome_user_agent(self):
        """Test Chrome preset."""
        with patch("cache.load_server_settings", return_value={"user_agent_preset": "chrome"}):
            ua = get_user_agent()
            assert ua is not None
            assert "Chrome" in ua

    def test_custom_user_agent(self):
        """Test custom user agent."""
        with patch(
            "cache.load_server_settings",
            return_value={"user_agent_preset": "custom", "user_agent_custom": "MyAgent/1.0"},
        ):
            assert get_user_agent() == "MyAgent/1.0"

    def test_custom_empty_returns_none(self):
        """Test empty custom user agent returns None."""
        with patch(
            "cache.load_server_settings",
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
        with patch("ffmpeg_command._load_settings", return_value={}):
            path = get_transcode_dir()
            assert path == Path(tempfile.gettempdir())

    def test_custom_transcode_dir(self, tmp_path):
        """Test custom directory is used and created."""
        custom_dir = tmp_path / "custom_transcode"
        with patch(
            "ffmpeg_command._load_settings", return_value={"transcode_dir": str(custom_dir)}
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
        with patch("ffmpeg_command._load_settings", return_value={}):
            assert get_live_hls_list_size() == 10

    def test_dvr_enabled_list_size(self):
        """Test DVR enabled calculates segments from minutes."""
        with patch("ffmpeg_command._load_settings", return_value={"live_dvr_mins": 5}):
            # 5 min = 300 sec / 3 sec per segment = 100 segments
            assert get_live_hls_list_size() == 100


# =============================================================================
# Probe Media Tests
# =============================================================================


class TestProbeMedia:
    """Tests for media probing."""

    def test_probe_success(self):
        """Test successful probe parses media info."""
        import ffmpeg_command

        ffmpeg_command._probe_cache.clear()

        probe_output = {
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
            patch("ffmpeg_command._load_settings", return_value={}),
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
        import ffmpeg_command

        ffmpeg_command._probe_cache.clear()

        probe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "mpeg2video",
                    "pix_fmt": "yuv420p",
                    "height": 1080,
                    "field_order": "tt",  # Top field first = interlaced
                },
            ],
            "format": {},
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(probe_output)

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("ffmpeg_command._load_settings", return_value={}),
        ):
            media_info, _ = probe_media("http://test/interlaced.ts")

        assert media_info is not None
        assert media_info.interlaced is True

    @pytest.mark.parametrize(
        "pix_fmt,expected",
        [
            ("yuv420p10le", True),  # 10-bit little endian
            ("yuv420p10be", True),  # 10-bit big endian
            ("yuv422p10le", True),  # 10-bit 4:2:2
            ("p010le", True),  # CUDA/VAAPI 10-bit format
            ("yuv420p", False),  # 8-bit
            ("yuv410p", False),  # 4:1:0 chroma, NOT 10-bit (was a false positive)
            ("nv12", False),  # 8-bit NV12
        ],
    )
    def test_probe_10bit_detection(self, pix_fmt: str, expected: bool):
        """Test 10-bit content detection from pix_fmt."""
        import ffmpeg_command

        ffmpeg_command._probe_cache.clear()

        probe_output = {
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
            patch("ffmpeg_command._load_settings", return_value={}),
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
            ("bt2020-10", False),  # Wide gamut but not HDR transfer
            ("", False),  # Unknown/missing
        ],
    )
    def test_probe_hdr_detection(self, color_transfer: str, expected: bool):
        """Test HDR content detection from color_transfer."""
        import ffmpeg_command

        ffmpeg_command._probe_cache.clear()

        probe_output = {
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
            patch("ffmpeg_command._load_settings", return_value={}),
        ):
            media_info, _ = probe_media(f"http://test/{color_transfer or 'unknown'}.mkv")

        assert media_info is not None
        assert media_info.is_hdr is expected, (
            f"color_transfer={color_transfer} should be is_hdr={expected}"
        )

    def test_probe_failure(self):
        """Test probe failure returns None."""
        import ffmpeg_command

        ffmpeg_command._probe_cache.clear()

        mock_result = MagicMock()
        mock_result.returncode = 1

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("ffmpeg_command._load_settings", return_value={}),
        ):
            media_info, subs = probe_media("http://test/bad.mp4")

        assert media_info is None
        assert subs == []

    def test_probe_cache_hit(self):
        """Test probe cache returns cached result."""
        import time

        import ffmpeg_command

        ffmpeg_command._probe_cache.clear()

        cached_info = MediaInfo(
            video_codec="h264",
            audio_codec="aac",
            pix_fmt="yuv420p",
        )
        ffmpeg_command._probe_cache["http://cached"] = (time.time(), cached_info, [])

        with (
            patch("subprocess.run") as mock_run,
            patch("ffmpeg_command._load_settings", return_value={}),
        ):
            media_info, _ = probe_media("http://cached")

        mock_run.assert_not_called()
        assert media_info == cached_info

    def test_probe_extracts_subtitles(self):
        """Test subtitle stream extraction."""
        import ffmpeg_command

        ffmpeg_command._probe_cache.clear()

        probe_output = {
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
            patch("ffmpeg_command._load_settings", return_value={}),
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
        import time

        import ffmpeg_command

        # Clear first to ensure known state
        ffmpeg_command._probe_cache.clear()
        ffmpeg_command._series_probe_cache.clear()

        ffmpeg_command._probe_cache["url1"] = (time.time(), None, [])
        ffmpeg_command._probe_cache["url2"] = (time.time(), None, [])
        ffmpeg_command._series_probe_cache[123] = {"episodes": {1: (time.time(), None, [])}}

        with patch("ffmpeg_command._save_series_probe_cache"):
            count = clear_all_probe_cache()

        assert count == 3
        assert len(ffmpeg_command._probe_cache) == 0
        assert len(ffmpeg_command._series_probe_cache) == 0

    def test_invalidate_series_probe_cache_entire_series(self):
        """Test invalidating entire series cache."""
        import ffmpeg_command

        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "episodes": {1: (0, None, []), 2: (0, None, [])},
        }

        with patch("ffmpeg_command._save_series_probe_cache"):
            invalidate_series_probe_cache(123)

        assert 123 not in ffmpeg_command._series_probe_cache

    def test_invalidate_series_probe_cache_single_episode(self):
        """Test invalidating single episode cache."""
        import ffmpeg_command

        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "episodes": {1: (0, None, []), 2: (0, None, [])},
        }

        with patch("ffmpeg_command._save_series_probe_cache"):
            invalidate_series_probe_cache(123, episode_id=1)

        assert 123 in ffmpeg_command._series_probe_cache
        assert 1 not in ffmpeg_command._series_probe_cache[123]["episodes"]
        assert 2 in ffmpeg_command._series_probe_cache[123]["episodes"]

    def test_clear_series_mru(self):
        """Test clearing series MRU."""
        import ffmpeg_command

        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test",
            "mru": 5,
            "episodes": {1: (0, None, [])},
        }

        with patch("ffmpeg_command._save_series_probe_cache"):
            clear_series_mru(123)

        assert "mru" not in ffmpeg_command._series_probe_cache[123]
        assert "episodes" in ffmpeg_command._series_probe_cache[123]

    def test_restore_probe_cache_entry(self):
        """Test restoring probe cache entry."""
        import ffmpeg_command

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
        import time

        import ffmpeg_command

        ffmpeg_command._series_probe_cache.clear()
        ffmpeg_command._series_probe_cache[123] = {
            "name": "Test Series",
            "mru": 2,
            "episodes": {
                1: (time.time(), MediaInfo("h264", "aac", "yuv420p"), []),
                2: (time.time(), MediaInfo("h264", "aac", "yuv420p"), []),
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
        from ffmpeg_command import resolve_hls_master_playlist

        url = "http://example.com/video.mp4"
        assert resolve_hls_master_playlist(url) == url

    def test_master_playlist_selects_highest_bandwidth(self):
        """Master playlist should resolve to highest bandwidth variant."""
        from unittest.mock import MagicMock, patch

        from ffmpeg_command import resolve_hls_master_playlist

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
        from unittest.mock import MagicMock, patch

        from ffmpeg_command import resolve_hls_master_playlist

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
        from unittest.mock import patch

        from ffmpeg_command import resolve_hls_master_playlist

        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = resolve_hls_master_playlist("http://example.com/master.m3u8")

        assert result == "http://example.com/master.m3u8"

    def test_relative_url_resolved_correctly(self):
        """Relative variant URLs should be resolved against base URL."""
        from unittest.mock import MagicMock, patch

        from ffmpeg_command import resolve_hls_master_playlist

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


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)
