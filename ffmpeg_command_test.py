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


class FakeMediaInfo:
    """Fake media info for testing."""

    def __init__(
        self,
        video_codec: str = "h264",
        audio_codec: str = "aac",
        pix_fmt: str = "yuv420p",
        audio_channels: int = 2,
        audio_sample_rate: int = 48000,
        height: int = 1080,
        interlaced: bool = False,
    ):
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.pix_fmt = pix_fmt
        self.audio_channels = audio_channels
        self.audio_sample_rate = audio_sample_rate
        self.height = height
        self.interlaced = interlaced


# =============================================================================
# Video Args Tests
# =============================================================================


class TestBuildVideoArgs:
    """Tests for _build_video_args."""

    @pytest.mark.parametrize("hw", ["nvidia", "intel", "vaapi", "software"])
    @pytest.mark.parametrize("deinterlace", [True, False])
    @pytest.mark.parametrize("max_resolution", ["1080p", "720p", "4k"])
    def test_all_hw_combinations(self, hw: HwAccel, deinterlace: bool, max_resolution: str):
        """Test all hardware/deinterlace/resolution combinations produce valid args."""
        pre, post = _build_video_args(
            copy_video=False,
            hw=hw,
            deinterlace=deinterlace,
            use_hw_pipeline=(hw != "software"),
            max_resolution=max_resolution,
            quality="high",
        )

        if hw == "nvidia":
            assert pre == [] or "-hwaccel" in pre
        elif hw == "intel":
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

    @pytest.mark.parametrize("hw", ["nvidia", "intel", "vaapi", "software"])
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

    def test_nvidia_hw_pipeline_filters(self):
        """Test NVIDIA with hw pipeline uses CUDA filters."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="nvidia",
            deinterlace=True,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
        )
        assert "-hwaccel" in pre
        vf = post[post.index("-vf") + 1]
        assert "yadif_cuda" in vf
        assert "scale_cuda" in vf

    def test_nvidia_sw_fallback_filters(self):
        """Test NVIDIA without hw pipeline uses software filters."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="nvidia",
            deinterlace=True,
            use_hw_pipeline=False,
            max_resolution="1080p",
            quality="high",
        )
        assert pre == []
        vf = post[post.index("-vf") + 1]
        assert "yadif=1" in vf
        assert "cuda" not in vf

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

    def test_intel_filters(self):
        """Test Intel uses QSV filters."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="intel",
            deinterlace=True,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
        )
        vf = post[post.index("-vf") + 1]
        assert "vpp_qsv" in vf
        assert "scale_qsv" in vf

    def test_software_filters(self):
        """Test software uses yadif and scale."""
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
        assert "yadif=1" in vf

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

    @pytest.mark.parametrize("hw", ["nvidia", "intel", "vaapi", "software"])
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
        with patch("ffmpeg_command._load_settings", return_value={"user_agent_preset": "default"}):
            assert get_user_agent() is None

    def test_vlc_user_agent(self):
        """Test VLC preset."""
        with patch("ffmpeg_command._load_settings", return_value={"user_agent_preset": "vlc"}):
            ua = get_user_agent()
            assert ua is not None
            assert "VLC" in ua

    def test_chrome_user_agent(self):
        """Test Chrome preset."""
        with patch("ffmpeg_command._load_settings", return_value={"user_agent_preset": "chrome"}):
            ua = get_user_agent()
            assert ua is not None
            assert "Chrome" in ua

    def test_custom_user_agent(self):
        """Test custom user agent."""
        with patch(
            "ffmpeg_command._load_settings",
            return_value={"user_agent_preset": "custom", "user_agent_custom": "MyAgent/1.0"},
        ):
            assert get_user_agent() == "MyAgent/1.0"

    def test_custom_empty_returns_none(self):
        """Test empty custom user agent returns None."""
        with patch(
            "ffmpeg_command._load_settings",
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


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)
