"""Tests for media probing and the probe caches."""

from dataclasses import replace
from pathlib import Path

import json
import subprocess
import time

import pytest

from netv import cache, proxy, testing, util
from netv.cache import get_user_agent
from netv.ffmpeg import probe
from netv.ffmpeg.letterbox import CropBox
from netv.ffmpeg.letterbox_test import SCOPE
from netv.ffmpeg.probe import (
    AudioTrack,
    MediaInfo,
    SubtitleStream,
    audio_tracks_from_cache,
    audio_tracks_to_cache,
    clear_all_probe_cache,
    clear_series_mru,
    get_series_probe_cache_stats,
    get_url_probe_cache_stats,
    has_cached_probe,
    invalidate_series_probe_cache,
    invalidate_url_probe_cache,
    probe_closed_captions,
    probe_media,
    probe_segment,
    remember_crop,
    resolve_hls_master_playlist,
    restore_probe_cache_entry,
    with_audio_language,
)
from netv.testing import FakeResponse, FakeRun, FakeWeb, Program, install_subprocess, install_web


@pytest.fixture(autouse=True)
def probe_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with empty probe caches."""
    monkeypatch.setattr(probe, "_probe_cache", {})
    monkeypatch.setattr(probe, "_series_probe_cache", {})


@pytest.fixture(autouse=True)
def fake_run(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> FakeRun:
    """Serve ffprobe from scripts, except in real-ffmpeg tests."""
    run = FakeRun()
    if request.node.get_closest_marker("cli_ffmpeg") is None:
        install_subprocess(monkeypatch, probe, run=run)
    return run


def _ffprobe(
    run: FakeRun,
    payload: object,
    *,
    returncode: int = 0,
    when: str | None = None,
) -> None:
    run.program("ffprobe", Program(stdout=json.dumps(payload), returncode=returncode), when=when)


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


class TestProbeMedia:
    """Tests for media probing."""

    def test_probes_reach_the_provider_only_through_the_egress_proxy(
        self,
        fake_run: FakeRun,
        monkeypatch: pytest.MonkeyPatch,
    ):
        save_home_lineup()
        monkeypatch.setenv("no_proxy", "192.168.1.87")
        _ffprobe(fake_run, {"streams": [], "format": {}})
        tuner = "http://192.168.1.87:5004/auto/v5.1"
        probe_media(tuner, source_id="t")
        for command, options in zip(fake_run.calls, fake_run.options, strict=True):
            assert command[-1] == tuner
            assert option(command, "-http_proxy") == proxy.egress().url(lan=True)
            assert option(command, "-protocol_whitelist") == proxy.ffmpeg_input_args(tuner, "t")[1]
            assert options["env"] == proxy.ffmpeg_env()

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

        media_info, subs = probe_media("http://test/video.mp4", source_id="")

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
            {
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": 30,
                "env": proxy.ffmpeg_env(),
            },
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

        media_info, _ = probe_media("http://test/interlaced.ts", source_id="")

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

        media_info, _ = probe_media(f"http://test/{pix_fmt}.mkv", source_id="")

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

        media_info, _ = probe_media(f"http://test/{color_transfer or 'unknown'}.mkv", source_id="")

        assert media_info is not None
        assert media_info.is_hdr is expected, (
            f"color_transfer={color_transfer} should be is_hdr={expected}"
        )

    def test_probe_failure(self, fake_run: FakeRun):
        """Test probe failure returns None."""
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}}, returncode=1)

        media_info, subs = probe_media("http://test/bad.mp4", source_id="")

        assert media_info is None
        assert subs == []
        assert len(fake_run.calls) == 2
        assert probe._probe_cache == {}

    def test_probe_cache_hit(self, fake_run: FakeRun):
        """Test probe cache returns cached result."""
        cached_info = MediaInfo(
            video_codec="h264",
            audio_codec="aac",
            pix_fmt="yuv420p",
        )
        probe._probe_cache["http://cached"] = (
            time.time(),
            cached_info,
            list[SubtitleStream](),
        )

        media_info, _ = probe_media("http://cached", source_id="")

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

        _, subs = probe_media("http://test/subs.mkv", source_id="")

        assert subs == [
            SubtitleStream(index=2, lang="eng", name="English"),
            SubtitleStream(index=3, lang="jpn", name="Japanese"),
        ]


def _series_cache_text() -> str:
    return (cache.CACHE_DIR / "series_probe_cache.json").read_text()


class TestProbeCacheManagement:
    """Tests for probe cache management functions."""

    def test_clear_all_probe_cache(self):
        """Test clearing all probe caches."""
        probe._probe_cache["url1"] = (time.time(), None, list[SubtitleStream]())
        probe._probe_cache["url2"] = (time.time(), None, list[SubtitleStream]())
        probe._series_probe_cache[123] = {
            "episodes": {1: (time.time(), None, list[SubtitleStream]())},
        }

        count = clear_all_probe_cache()

        assert count == 3
        assert len(probe._probe_cache) == 0
        assert len(probe._series_probe_cache) == 0
        assert json.loads(_series_cache_text()) == {}

    def test_invalidate_series_probe_cache_entire_series(self):
        """Test invalidating entire series cache."""
        probe._series_probe_cache[123] = {
            "name": "Test",
            "episodes": {
                1: (0, None, list[SubtitleStream]()),
                2: (0, None, list[SubtitleStream]()),
            },
        }

        invalidate_series_probe_cache(123)

        assert 123 not in probe._series_probe_cache
        assert json.loads(_series_cache_text()) == {}

    def test_invalidate_series_probe_cache_single_episode(self):
        """Test invalidating single episode cache."""
        probe._series_probe_cache[123] = {
            "name": "Test",
            "episodes": {
                1: (0, media_info(), list[SubtitleStream]()),
                2: (0, media_info(), list[SubtitleStream]()),
            },
        }

        invalidate_series_probe_cache(123, episode_id=1)

        assert 123 in probe._series_probe_cache
        assert 1 not in probe._series_probe_cache[123]["episodes"]
        assert 2 in probe._series_probe_cache[123]["episodes"]
        assert list(json.loads(_series_cache_text())["123"]["episodes"]) == ["2"]

    def test_invalidate_unknown_episode_keeps_series(self):
        probe._series_probe_cache[123] = {
            "name": "Test",
            "episodes": {2: (0, media_info(), list[SubtitleStream]())},
        }

        invalidate_series_probe_cache(123, episode_id=1)

        assert list(probe._series_probe_cache[123]["episodes"]) == [2]

    def test_clear_series_mru(self):
        """Test clearing series MRU."""
        probe._series_probe_cache[123] = {
            "name": "Test",
            "mru": 5,
            "episodes": {1: (0, None, list[SubtitleStream]())},
        }

        clear_series_mru(123)

        assert "mru" not in probe._series_probe_cache[123]
        assert "episodes" in probe._series_probe_cache[123]
        assert json.loads(_series_cache_text()) == {
            "123": {"name": "Test", "mru": None, "episodes": {}},
        }

    def test_clear_series_mru_without_mru_still_saves(self):
        probe._series_probe_cache[123] = {
            "name": "Test",
            "episodes": dict[int, probe._ProbeEntry](),
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

        assert "http://test" in probe._probe_cache
        assert 123 in probe._series_probe_cache
        assert 5 in probe._series_probe_cache[123]["episodes"]

    def test_get_series_probe_cache_stats(self):
        """Test getting cache stats for UI."""
        probe._series_probe_cache[123] = {
            "name": "Test Series",
            "mru": 2,
            "episodes": {
                1: (time.time(), media_info(), list[SubtitleStream]()),
                2: (time.time(), media_info(), list[SubtitleStream]()),
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
        install_web(monkeypatch, web, util)
        url = "http://example.com/video.mp4"
        assert resolve_hls_master_playlist(url, source_id="") == url
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
        install_web(monkeypatch, web, util)

        assert resolve_hls_master_playlist(url, source_id="") == "http://example.com/high.m3u8"
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
        install_web(monkeypatch, FakeWeb({url: FakeResponse(media_playlist)}), util)

        assert resolve_hls_master_playlist(url, source_id="") == url

    def test_fetch_error_returns_original_url(self, monkeypatch: pytest.MonkeyPatch):
        """On fetch error, should return original URL."""
        url = "http://example.com/master.m3u8?token=1"
        web = FakeWeb({url: Exception("Network error")})
        install_web(monkeypatch, web, util)

        assert resolve_hls_master_playlist(url, source_id="") == url
        assert web.urls == [url]

    def test_relative_url_resolved_correctly(self, monkeypatch: pytest.MonkeyPatch):
        """Relative variant URLs should be resolved against base URL."""
        master_playlist = b"""#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=3000000
../streams/1080p/index.m3u8
"""
        url = "http://example.com/hls/master.m3u8"
        install_web(monkeypatch, FakeWeb({url: FakeResponse(master_playlist)}), util)

        assert (
            resolve_hls_master_playlist(url, source_id="")
            == "http://example.com/streams/1080p/index.m3u8"
        )

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
        install_web(monkeypatch, web, util)

        assert resolve_hls_master_playlist(media, source_id="") == "http://edge/a/1.m3u8"
        assert resolve_hls_master_playlist(master, source_id="") == "http://edge/b/hd/index.m3u8"

    def test_a_provider_s_master_may_not_point_ffmpeg_inward(self, monkeypatch: pytest.MonkeyPatch):
        url = "http://8.8.8.8/live/master.m3u8"
        body = (
            b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=9000000\nhttp://127.0.0.1:8000/settings\n"
            b"#EXT-X-STREAM-INF:BANDWIDTH=1\nlow.m3u8\n"
        )
        web = FakeWeb({url: FakeResponse(body)})
        install_web(monkeypatch, web, util)
        # The best variant names this server, so the best one on the internet plays.
        assert resolve_hls_master_playlist(url, source_id="") == "http://8.8.8.8/live/low.m3u8"
        # And the master's own redirects may not turn inward either.
        [fetch] = web.fetches
        assert testing.dials_only_the_internet(fetch)

    def test_default_user_agent_sends_the_server_s_default(self, monkeypatch: pytest.MonkeyPatch):
        _save_user_agent("default")
        url = "http://example.com/stream.m3u8"
        web = FakeWeb({url: FakeResponse(b"#EXTM3U\n")})
        install_web(monkeypatch, web, util)

        assert resolve_hls_master_playlist(url, source_id="") == url
        # As every other server fetch does: providers block Python's own.
        assert web.fetches[0].headers == {"User-agent": "VLC/3.0.20 LibVLC/3.0.20"}

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
        caplog.set_level("INFO", logger="netv.ffmpeg.probe")
        base = f"http://example.com/{'p' * 90}/"
        url = base + "master.m3u8"
        body = b"#EXTM3U\n#EXT-X-STREAM-INF:" + playlist + b"\n"
        install_web(monkeypatch, FakeWeb({url: FakeResponse(body)}), util)

        best = f"{base}{variant}.m3u8"
        assert resolve_hls_master_playlist(url, source_id="") == best
        assert _probe_log(caplog) == [
            (
                "INFO",
                f"HLS master playlist resolved: {count} variants, selected {bandwidth} bps: {best[:80]}",
            ),
        ]

    def test_invalid_utf8_is_replaced(self, monkeypatch: pytest.MonkeyPatch):
        url = "http://example.com/master.m3u8"
        body = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nvid\xff.m3u8\n"
        install_web(monkeypatch, FakeWeb({url: FakeResponse(body)}), util)

        assert resolve_hls_master_playlist(url, source_id="") == "http://example.com/vid\ufffd.m3u8"

    def test_master_without_variants_logs_served_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        url = "http://example.com/master.m3u8"
        served = f"http://edge.example.com/{'p' * 90}/master.m3u8"
        body = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\n#comment\n"
        install_web(monkeypatch, FakeWeb({url: FakeResponse(body, url=served)}), util)

        assert resolve_hls_master_playlist(url, source_id="") == served
        assert _probe_log(caplog) == [
            ("WARNING", f"HLS master playlist has no variants: {served[:80]}"),
        ]

    def test_fetch_error_is_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        url = f"http://example.com/{'p' * 90}/master.m3u8"
        install_web(monkeypatch, FakeWeb({url: Exception("Network error")}), util)

        assert resolve_hls_master_playlist(url, source_id="") == url
        assert _probe_log(caplog) == [
            ("WARNING", f"Failed to resolve HLS master playlist {url[:80]}: Network error"),
        ]


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
        audio_tracks=(
            AudioTrack(
                index=None,
                lang="und",
                codec="aac",
                channels=6,
                sample_rate=48_000,
                profile="LC",
            ),
        ),
    )
    assert fake_run.calls == [
        ["ffprobe", "-v", "error", "-of", "json", "-show_streams", "-show_format", "input_0.ts"],
    ]
    assert fake_run.options == [
        {"capture_output": True, "text": True, "timeout": 5, "check": False},
    ]


def _multi_audio_probe() -> dict[str, object]:
    return {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p"},
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "eac3",
                "channels": 6,
                "tags": {"language": "chi"},
                "disposition": {"default": 1},
            },
            {"index": 2, "codec_type": "audio", "codec_name": "ac3", "tags": {"language": "cze"}},
            {
                "index": 3,
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "tags": {"language": "eng"},
            },
        ],
        "format": {"duration": "60"},
    }


def test_audio_language_selects_matching_track(fake_run: FakeRun):
    _ffprobe(fake_run, _multi_audio_probe())
    info = probe_segment("x.mkv")
    assert info is not None
    assert (info.audio_codec, info.audio_index) == ("eac3", None)
    english = with_audio_language(info, "en-US")
    assert english is not None
    assert (english.audio_codec, english.audio_channels, english.audio_index) == ("aac", 2, 3)
    assert with_audio_language(info, "cs") == replace(
        info,
        audio_codec="ac3",
        audio_channels=0,
        audio_index=2,
    )
    assert with_audio_language(info, "ja") == info
    assert with_audio_language(info, "") == info
    assert audio_tracks_from_cache(audio_tracks_to_cache(info.audio_tracks)) == info.audio_tracks


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


@pytest.fixture
def probe_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep the on-disk probe cache in tmp_path and send no User-Agent."""
    _save_user_agent("default")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


_VIDEO = {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "height": 720}


def test_series_stats_skip_empty_and_unprobed_series(probe_state: Path):
    probe._series_probe_cache.update(
        {
            1: {"name": "Empty", "episodes": {}},
            2: {"name": "Unprobed", "episodes": {1: (5, None, [])}},
            3: {
                "episodes": {
                    2: (2, media_info(), [SubtitleStream(index=1, lang="eng", name="E")]),
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
    assert probe._series_probe_cache == {}


def test_master_playlist_without_variant_urls_returns_original(
    monkeypatch: pytest.MonkeyPatch,
    probe_state: Path,
):
    _save_user_agent("custom", "UA/1")
    url = "http://example.com/master.m3u8"
    body = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=abc\n#comment\n"
    web = FakeWeb({url: FakeResponse(body)})
    install_web(monkeypatch, web, util)
    assert resolve_hls_master_playlist(url, source_id="") == url
    assert web.fetches[0].headers == {"User-agent": "UA/1"}


class TestProbeMediaSeriesCache:
    """Series/episode cache paths in probe_media."""

    def test_exact_episode_hit_keeps_mru(self, fake_run: FakeRun, probe_state: Path):
        info = media_info()
        probe._series_probe_cache[7] = {
            "name": "S",
            "mru": 1,
            "episodes": {2: (time.time(), info, list[SubtitleStream]())},
        }
        assert probe_media("http://x", series_id=7, episode_id=2, source_id="") == (info, [])
        assert probe._series_probe_cache[7].get("mru") == 1
        assert not (probe_state / "series_probe_cache.json").exists()
        assert fake_run.calls == []

    def test_uncached_probe_records_the_episode_and_keeps_mru(
        self,
        fake_run: FakeRun,
        probe_state: Path,
    ):
        typical = media_info(video_codec="typical")
        probe._series_probe_cache[7] = {
            "mru": 1,
            "episodes": {1: (time.time(), typical, list[SubtitleStream]())},
        }
        probe._probe_cache["http://x"] = (0, typical, list[SubtitleStream]())
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}})
        info, _ = probe_media("http://x", series_id=7, episode_id=2, source_id="", use_cache=False)
        assert info is not None and info.video_codec == "h264"
        series = probe._series_probe_cache[7]
        assert series["episodes"][2][1] is info
        assert series.get("mru") == 1
        # Episodes without their own probe still share the typical one.
        assert probe_media("http://y", series_id=7, episode_id=3, source_id="")[0] is typical

    def test_uncached_probe_replaces_a_url_probe(self, fake_run: FakeRun, probe_state: Path):
        old = media_info(video_codec="old")
        probe._probe_cache["http://x"] = (0, old, list[SubtitleStream]())
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}})
        info, _ = probe_media("http://x", source_id="", use_cache=False)
        assert info is not None and info.video_codec == "h264"
        assert probe_media("http://x", source_id="")[0] is info

    def test_mru_fallback_hit(self, fake_run: FakeRun, probe_state: Path):
        info = media_info()
        probe._series_probe_cache[7] = {
            "mru": 1,
            "episodes": {1: (time.time(), info, list[SubtitleStream]())},
        }
        assert probe_media("http://x", series_id=7, episode_id=99, source_id="") == (info, [])
        assert fake_run.calls == []

    @pytest.mark.parametrize("episode_id", [1, 99])
    def test_old_series_entries_still_hit(
        self,
        fake_run: FakeRun,
        probe_state: Path,
        episode_id: int,
    ):
        old = media_info(video_codec="old")
        probe._series_probe_cache[7] = {
            "mru": 1,
            "episodes": {1: (0, old, list[SubtitleStream]())},
        }
        probe._probe_cache["http://y"] = (0, old, list[SubtitleStream]())
        assert probe_media("http://x", series_id=7, episode_id=episode_id, source_id="") == (
            old,
            [],
        )
        assert probe_media("http://y", source_id="") == (old, [])
        assert fake_run.calls == []

    def test_a_series_without_cached_episodes_is_probed(
        self,
        fake_run: FakeRun,
        probe_state: Path,
    ):
        probe._series_probe_cache[7] = {
            "mru": 3,
            "episodes": dict[int, probe._ProbeEntry](),
        }
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}})
        info, _ = probe_media("http://x", series_id=7, episode_id=2, series_name="N", source_id="")
        assert info is not None and info.video_codec == "h264"
        series = probe._series_probe_cache[7]
        assert "name" in series
        assert series["name"] == "N"
        assert series["mru"] == 2
        assert series["episodes"][2][1] is info

    def test_new_series_is_cached_and_persisted(self, fake_run: FakeRun, probe_state: Path):
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}})
        info, _ = probe_media("http://x", series_id=8, series_name="New", source_id="")
        series = probe._series_probe_cache[8]
        assert "name" in series
        assert series["name"] == "New"
        assert "mru" in series
        assert series["mru"] == 0
        assert series["episodes"][0][1] is info
        saved = json.loads((probe_state / "series_probe_cache.json").read_text())
        assert saved["8"]["episodes"]["0"]["video_codec"] == "h264"

    def test_existing_named_series_keeps_name(self, fake_run: FakeRun, probe_state: Path):
        probe._series_probe_cache[8] = {
            "name": "Kept",
            "episodes": dict[int, probe._ProbeEntry](),
        }
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {}})
        probe_media("http://x", series_id=8, episode_id=3, series_name="Other", source_id="")
        assert probe._series_probe_cache[8]["name"] == "Kept"
        assert 3 in probe._series_probe_cache[8]["episodes"]


class TestProbeMediaEdgeCases:
    def test_subprocess_exceptions_yield_none(self, fake_run: FakeRun, probe_state: Path):
        fake_run.program("ffprobe", Program(error=OSError("boom")))
        assert probe_media("http://x", source_id="") == (None, [])
        assert len(fake_run.calls) == 2
        assert fake_run.calls[0][-1] == "http://x"
        assert "hls" not in fake_run.calls[0]
        assert fake_run.calls[1][-5:] == ["-f", "hls", "-extension_picky", "0", "http://x"]

    def test_user_agent_and_format_bitrate_fallback(self, fake_run: FakeRun, probe_state: Path):
        _save_user_agent("custom", "UA/1")
        fmt = {"bit_rate": "1234", "format_name": "hls"}
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": fmt})
        info, _ = probe_media("http://x", source_id="")
        assert info is not None
        assert info.video_bitrate == 1234
        assert info.is_hls
        cmd = fake_run.calls[0]
        assert cmd[cmd.index("-user_agent") + 1] == "UA/1"

    def test_invalid_format_bitrate_ignored(self, fake_run: FakeRun, probe_state: Path):
        _ffprobe(fake_run, {"streams": [_VIDEO], "format": {"bit_rate": "n/a"}})
        info, _ = probe_media("http://x", source_id="")
        assert info is not None and info.video_bitrate == 0
        assert "-user_agent" not in fake_run.calls[0]

    def test_audio_only_returns_none(self, fake_run: FakeRun, probe_state: Path):
        audio = {"codec_type": "audio", "codec_name": "aac", "channels": 2}
        _ffprobe(fake_run, {"streams": [audio], "format": {}})
        assert probe_media("http://x", source_id="") == (None, [])

    def test_zero_height_not_cached(self, fake_run: FakeRun, probe_state: Path):
        _ffprobe(fake_run, {"streams": [{**_VIDEO, "height": 0}], "format": {}})
        info, _ = probe_media("http://x", series_id=1, source_id="")
        assert info is not None and info.height == 0
        assert probe._probe_cache == {}
        assert probe._series_probe_cache == {}


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
    info, _ = probe_media("http://x/mistyped", source_id="")
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


class TestUrlProbeCachePersistence:
    def test_probes_survive_a_restart(self, probe_state: Path):
        track = AudioTrack(index=3, lang="eng", codec="ac3", channels=6, default=True)
        info = MediaInfo(
            video_codec="h264",
            audio_codec="ac3",
            pix_fmt="yuv420p",
            height=1080,
            duration=90.0,
            is_hls=True,
            audio_index=3,
            audio_tracks=(track,),
        )
        probe.remember_probe("http://h/movie/u/p/1.mkv", info)
        probe.remember_probe("http://h/movie/u/p/2.mkv", None)
        saved = json.loads((probe_state / "probe_cache.json").read_text())
        assert list(saved) == ["http://h/movie/u/p/1.mkv"]
        assert saved["http://h/movie/u/p/1.mkv"]["time"] > 0
        assert saved["http://h/movie/u/p/1.mkv"]["is_hls"] is True
        saved["http://h/movie/u/p/3.mkv"] = {"audio_codec": "aac"}
        (probe_state / "probe_cache.json").write_text(json.dumps(saved))

        probe._probe_cache.clear()
        probe.load_caches()
        assert set(probe._probe_cache) == {"http://h/movie/u/p/1.mkv"}
        assert probe_media("http://h/movie/u/p/1.mkv", source_id="") == (info, [])
        assert has_cached_probe("http://h/movie/u/p/1.mkv")
        assert not has_cached_probe("http://h/movie/u/p/2.mkv")

    def test_load_missing_or_corrupt_file(
        self,
        probe_state: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        probe._load_url_probe_cache()
        assert probe._probe_cache == {}
        (probe_state / "probe_cache.json").write_text("{not json")
        probe._load_url_probe_cache()
        assert probe._probe_cache == {}
        assert "Failed to load probe cache" in caplog.text

    def test_save_failure_is_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        blocker = tmp_path / "file"
        blocker.write_text("")
        monkeypatch.setattr(cache, "CACHE_DIR", blocker / "sub")
        probe.remember_probe("http://h/x", media_info())
        assert "Failed to save probe cache" in caplog.text

    def test_invalidation_rewrites_the_file(self, probe_state: Path):
        info = media_info()
        probe.remember_probe("http://a", info)
        probe.remember_probe("http://b", info)
        assert invalidate_url_probe_cache(["http://a", "http://missing"]) == 1
        assert list(json.loads((probe_state / "probe_cache.json").read_text())) == ["http://b"]


def test_url_probe_stats_list_recent_probes_first():
    track = AudioTrack(index=1, lang="eng", codec="ac3")
    old = MediaInfo(video_codec="h264", audio_codec="ac3", pix_fmt="yuv420p", audio_tracks=(track,))
    new = MediaInfo(video_codec="hevc", audio_codec="aac", pix_fmt="yuv420p", height=2160)
    subs = [SubtitleStream(index=2, lang="eng", name="English")]
    probe._probe_cache.update(
        {
            "http://old": (1.0, old, subs),
            "http://new": (2.0, new, []),
            "http://none": (3.0, None, []),
        },
    )
    assert get_url_probe_cache_stats() == [
        {
            "url": "http://new",
            "probed_at": 2.0,
            "video_codec": "hevc",
            "audio_codec": "aac",
            "audio_langs": [],
            "subtitle_count": 0,
            "duration": 0.0,
            "height": 2160,
        },
        {
            "url": "http://old",
            "probed_at": 1.0,
            "video_codec": "h264",
            "audio_codec": "ac3",
            "audio_langs": ["eng"],
            "subtitle_count": 1,
            "duration": 0.0,
            "height": 0,
        },
    ]


class TestSeriesProbeCachePersistence:
    def test_load_missing_file_is_noop(self, probe_state: Path):
        probe._load_series_probe_cache()
        assert probe._series_probe_cache == {}

    def test_load_corrupt_file_is_logged(self, probe_state: Path, caplog: pytest.LogCaptureFixture):
        (probe_state / "series_probe_cache.json").write_text("{not json")
        probe._load_series_probe_cache()
        assert probe._series_probe_cache == {}
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
        probe._series_probe_cache.update(
            {
                5: {"name": "Show", "mru": 2, "episodes": {2: (123.0, info, subs)}},
                6: {"episodes": {1: (1.0, None, [])}},
            },
        )
        probe._save_series_probe_cache()
        saved = json.loads((probe_state / "series_probe_cache.json").read_text())
        assert saved["6"] == {"name": "", "mru": None, "episodes": {}}
        assert saved["5"]["episodes"]["2"]["subtitles"] == [
            {"index": 3, "lang": "eng", "name": "English"},
        ]

        # Add entries the loader must skip or default.
        saved["5"]["episodes"]["4"] = {"audio_codec": "aac"}
        saved["5"]["episodes"]["5"] = {"video_codec": "h264", "subtitles": [{"index": 1}]}
        (probe_state / "series_probe_cache.json").write_text(json.dumps(saved))

        existing = media_info(video_codec="mpeg2video")
        probe._series_probe_cache.clear()
        probe._series_probe_cache[5] = {
            "episodes": {2: (999.0, existing, list[SubtitleStream]())},
        }
        probe._load_series_probe_cache()

        show = probe._series_probe_cache[5]
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
        assert probe._series_probe_cache[6] == {"name": "", "mru": None, "episodes": {}}

        probe._series_probe_cache.clear()
        probe._load_series_probe_cache()
        _, loaded, loaded_subs = probe._series_probe_cache[5]["episodes"][2]
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
        monkeypatch.setattr(cache, "CACHE_DIR", blocker / "sub")
        probe._save_series_probe_cache()
        assert "Failed to save series probe cache" in caplog.text


def media_info(
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
    width: int = 0,
    crop: CropBox | None = None,
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
        width=width,
        crop=crop,
    )


def test_audio_tracks_round_trip_through_the_cache():
    tracks = (
        AudioTrack(
            index=2,
            lang="spa",
            codec="aac",
            channels=6,
            sample_rate=44_100,
            profile="HE-AAC",
            default=True,
        ),
    )
    assert audio_tracks_from_cache(audio_tracks_to_cache(tracks)) == tracks


class TestCropCache:
    def test_probe_reads_the_frame_width(self, fake_run: FakeRun, probe_state: Path):
        _ffprobe(fake_run, {"streams": [{**_VIDEO, "width": 1280}]})
        info, _ = probe_media("http://m", source_id="")
        assert info is not None
        assert info.width == 1280

    def test_detections_accumulate_and_persist(self, probe_state: Path):
        probe.remember_probe("http://m", media_info(width=1920))
        # A dark scene and a bright one each reach one edge of the picture.
        remember_crop("http://m", CropBox(top=0.13, height=0.6))
        remember_crop("http://m", CropBox(top=0.2, height=0.67))
        remember_crop("http://unprobed", SCOPE)
        probe._probe_cache["http://no-video"] = (1.0, None, list[SubtitleStream]())
        remember_crop("http://no-video", SCOPE)
        assert probe._probe_cache["http://no-video"] == (1.0, None, [])
        probe._probe_cache.clear()
        probe._load_url_probe_cache()
        info, _ = probe_media("http://m", source_id="")
        assert info is not None
        assert (info.width, info.crop) == (1920, SCOPE)
        assert not has_cached_probe("http://unprobed")

    def test_an_episode_s_bars_stay_with_that_episode(self, fake_run: FakeRun, probe_state: Path):
        del fake_run
        episode = media_info(width=1920)
        probe._series_probe_cache[5] = {
            "name": "S",
            "mru": 1,
            "episodes": {1: (1.0, episode, list[SubtitleStream]())},
        }
        probe._probe_cache["http://e1"] = (1.0, episode, list[SubtitleStream]())
        remember_crop("http://e1", SCOPE, series_id=5, episode_id=1)
        own, _ = probe_media("http://e1", 5, 1, source_id="")
        # Another episode starts from this one's probe, but its bars are its own.
        other, _ = probe_media("http://e2", 5, 2, source_id="")
        assert own is not None
        assert other is not None
        assert (own.crop, other.crop) == (SCOPE, None)
        probe._series_probe_cache.clear()
        probe._load_series_probe_cache()
        _, loaded, _ = probe._series_probe_cache[5]["episodes"][1]
        assert loaded is not None
        assert loaded.crop == SCOPE


def _probe_log(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    return [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "netv.ffmpeg.probe"]


def option(cmd: list[str], flag: str) -> str:
    """The value following ``flag`` in ``cmd``."""
    return cmd[cmd.index(flag) + 1]


def save_home_lineup() -> None:
    """Configure source "t", a tuner on the LAN."""
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "sources": [
                {
                    "id": "t",
                    "name": "Tuner",
                    "type": "m3u",
                    "url": "http://192.168.1.24/lineup.m3u",
                },
            ],
        },
    )


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
