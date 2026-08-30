"""Tests for cache.py."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import subprocess

import pytest

import cache


@pytest.fixture
def cache_module(tmp_path: Path):
    """Import cache module with temp directories."""

    # Patch paths to temp locations
    original_server_settings = cache.SERVER_SETTINGS_FILE
    original_users_dir = cache.USERS_DIR
    original_cache_dir = cache.CACHE_DIR
    cache.SERVER_SETTINGS_FILE = tmp_path / "server_settings.json"
    cache.USERS_DIR = tmp_path / "users"
    cache.USERS_DIR.mkdir(exist_ok=True)
    cache.CACHE_DIR = tmp_path / "cache"
    cache.CACHE_DIR.mkdir(exist_ok=True)

    # Clear memory cache
    cache._cache.clear()

    yield cache

    cache.SERVER_SETTINGS_FILE = original_server_settings
    cache.USERS_DIR = original_users_dir
    cache.CACHE_DIR = original_cache_dir
    cache._cache.clear()


class TestFileCache:
    def test_save_and_load_file_cache(self, cache_module):
        cache_module.save_file_cache("test", {"key": "value"})
        result = cache_module.load_file_cache("test")
        assert result is not None
        data, ts = result
        assert data == {"key": "value"}
        assert ts > 0

    def test_load_nonexistent_cache(self, cache_module):
        assert cache_module.load_file_cache("nonexistent") is None

    def test_load_corrupted_cache(self, cache_module):
        path = cache_module.CACHE_DIR / "corrupted.json"
        path.write_text("not valid json")
        assert cache_module.load_file_cache("corrupted") is None


class TestMemoryCache:
    def test_get_cache_returns_reference(self, cache_module):
        cache = cache_module.get_cache()
        cache["test"] = 123
        assert cache_module.get_cache()["test"] == 123

    def test_clear_all_caches_preserves_epg(self, cache_module):
        cache = cache_module.get_cache()
        cache["epg"] = {"data": "epg"}
        cache["live"] = {"data": "live"}
        cache_module.clear_all_caches()
        assert "epg" in cache
        assert "live" not in cache


class TestCachedInfo:
    def test_get_cached_info_calls_fetch(self, cache_module):
        fetch_fn = mock.Mock(return_value={"result": 42})
        result = cache_module.get_cached_info("test_key", fetch_fn)
        assert result == {"result": 42}
        fetch_fn.assert_called_once()

    def test_get_cached_info_uses_memory_cache(self, cache_module):
        fetch_fn = mock.Mock(return_value={"result": 1})
        cache_module.get_cached_info("key1", fetch_fn)
        cache_module.get_cached_info("key1", fetch_fn)
        # Only called once - second call uses memory cache
        fetch_fn.assert_called_once()

    def test_get_cached_info_force_bypasses_memory(self, cache_module):
        fetch_fn = mock.Mock(return_value={"result": 1})
        cache_module.get_cached_info("key2", fetch_fn)
        cache_module.get_cached_info("key2", fetch_fn, force=True)
        assert fetch_fn.call_count == 2


class TestSettings:
    def test_load_settings_defaults(self, cache_module):
        settings = cache_module.load_server_settings()
        assert settings["sources"] == []
        assert settings["transcode_mode"] == "auto"
        assert settings["transcode_hw"] in (
            "nvenc+vaapi",
            "nvenc+software",
            "amf+vaapi",
            "amf+software",
            "qsv",
            "vaapi",
            "software",
        )
        assert settings["probe_movies"] is True

    def test_save_and_load_settings(self, cache_module):
        settings = {"sources": [{"id": "s1", "name": "Test"}], "custom": True}
        cache_module.save_server_settings(settings)
        loaded = cache_module.load_server_settings()
        assert loaded["sources"] == [{"id": "s1", "name": "Test"}]
        assert loaded["custom"] is True


class TestUserSettings:
    def test_load_user_settings_defaults(self, cache_module):
        settings = cache_module.load_user_settings("testuser")
        assert settings["guide_filter"] == []
        assert settings["captions_enabled"] is True
        assert settings["watch_history"] == {}

    def test_save_and_load_user_settings(self, cache_module):
        settings = {"guide_filter": ["cat1", "cat2"], "captions_enabled": False}
        cache_module.save_user_settings("testuser", settings)
        loaded = cache_module.load_user_settings("testuser")
        assert loaded["guide_filter"] == ["cat1", "cat2"]
        assert loaded["captions_enabled"] is False

    def test_watch_position_save_and_get(self, cache_module):
        cache_module.save_watch_position("user1", "http://video.url", 120.5, 3600.0)
        entry = cache_module.get_watch_position("user1", "http://video.url")
        assert entry is not None
        assert entry["position"] == 120.5
        assert entry["duration"] == 3600.0

    def test_watch_position_resets_at_95_percent(self, cache_module):
        # Save at 96% watched
        cache_module.save_watch_position("user1", "http://video.url", 960.0, 1000.0)
        entry = cache_module.get_watch_position("user1", "http://video.url")
        assert entry is None  # Should be reset


class TestSource:
    def test_source_dataclass(self, cache_module):
        source = cache_module.Source(
            id="test",
            name="Test Source",
            type="xtream",
            url="http://example.com",
        )
        assert source.id == "test"
        assert source.username == ""
        assert source.epg_timeout == 120
        assert source.epg_enabled is True

    def test_get_sources_empty(self, cache_module):
        sources = cache_module.get_sources()
        assert sources == []

    def test_get_sources_from_settings(self, cache_module):
        settings = {
            "sources": [
                {
                    "id": "s1",
                    "name": "Source 1",
                    "type": "m3u",
                    "url": "http://example.com/playlist.m3u",
                }
            ]
        }
        cache_module.save_server_settings(settings)
        sources = cache_module.get_sources()
        assert len(sources) == 1
        assert sources[0].id == "s1"
        assert sources[0].type == "m3u"


class TestUpdateSourceEpgUrl:
    def test_update_source_epg_url(self, cache_module):
        settings = {"sources": [{"id": "s1", "name": "S1", "type": "m3u", "url": "http://x"}]}
        cache_module.save_server_settings(settings)
        cache_module.update_source_epg_url("s1", "http://epg.example.com")
        loaded = cache_module.load_server_settings()
        assert loaded["sources"][0]["epg_url"] == "http://epg.example.com"

    def test_update_source_epg_url_not_overwrite(self, cache_module):
        settings = {
            "sources": [
                {
                    "id": "s1",
                    "name": "S1",
                    "type": "m3u",
                    "url": "http://x",
                    "epg_url": "http://existing",
                }
            ]
        }
        cache_module.save_server_settings(settings)
        cache_module.update_source_epg_url("s1", "http://new")
        loaded = cache_module.load_server_settings()
        assert loaded["sources"][0]["epg_url"] == "http://existing"

    def test_update_source_epg_url_empty_noop(self, cache_module):
        settings = {"sources": [{"id": "s1", "name": "S1", "type": "m3u", "url": "http://x"}]}
        cache_module.save_server_settings(settings)
        cache_module.update_source_epg_url("s1", "")
        loaded = cache_module.load_server_settings()
        assert "epg_url" not in loaded["sources"][0]

    def test_update_source_epg_url_does_not_log_credentials(self, cache_module, caplog):
        caplog.set_level("INFO", logger=cache_module.log.name)
        settings = {
            "sources": [{"id": "s1", "name": "S1", "type": "xtream", "url": "http://x"}]
        }
        cache_module.save_server_settings(settings)

        cache_module.update_source_epg_url(
            "s1",
            "https://provider.example/xmltv.php?username=private-user&password=private-secret",
        )

        assert "private-user" not in caplog.text
        assert "private-secret" not in caplog.text
        assert "provider.example" in caplog.text


class TestEncoderDetection:
    """Tests for encoder detection functions."""

    def test_test_encoder_success(self):
        """Test _test_encoder returns (True, '') on successful command."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            ok, err = cache._test_encoder(["echo", "test"])
            assert ok is True
            assert err == ""
            mock_run.assert_called_once()

    def test_test_encoder_failure(self):
        """Test _test_encoder returns (False, error) on non-zero return code."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stderr=b"encoder not found")
            ok, err = cache._test_encoder(["false"])
            assert ok is False
            assert "encoder not found" in err

    def test_test_encoder_timeout(self):
        """Test _test_encoder returns (False, 'timeout') on timeout."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=5)
            ok, err = cache._test_encoder(["sleep", "100"], timeout=5)
            assert ok is False
            assert err == "timeout"

    def test_test_encoder_exception(self):
        """Test _test_encoder returns (False, error) on exception."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffmpeg not found")
            ok, err = cache._test_encoder(["nonexistent_command"])
            assert ok is False
            assert err == "ffmpeg not found"

    def test_detect_encoders_all_available(self):
        """Test detect_encoders when all hardware is available."""
        with (
            mock.patch.object(cache, "_test_encoder", return_value=(True, "")),
            mock.patch.object(cache, "VAAPI_DEVICE", "/dev/dri/renderD128"),
            mock.patch.object(cache, "LIBVA_DRIVER", "i965"),
            mock.patch.object(cache, "DRI_PATH", "/usr/lib/x86_64-linux-gnu/dri"),
        ):
            result = cache.detect_encoders()
            assert result == {
                "nvenc": True,
                "amf": True,
                "qsv": True,
                "vaapi": True,
                "vaapi_baseline_only": False,
            }

    def test_detect_encoders_none_available(self):
        """Test detect_encoders when no hardware is available."""
        with mock.patch.object(cache, "_test_encoder", return_value=(False, "not found")):
            result = cache.detect_encoders()
            assert result == {
                "nvenc": False,
                "amf": False,
                "qsv": False,
                "vaapi": False,
            }

    def test_detect_encoders_partial(self):
        """Test detect_encoders with mixed hardware availability."""

        def mock_test(cmd, timeout=5, env=None):
            # Return True only for VAAPI
            if "h264_vaapi" in cmd:
                return True, ""
            return False, "not available"

        with (
            mock.patch.object(cache, "_test_encoder", side_effect=mock_test),
            mock.patch.object(cache, "VAAPI_DEVICE", "/dev/dri/renderD128"),
            mock.patch.object(cache, "LIBVA_DRIVER", "i965"),
            mock.patch.object(cache, "DRI_PATH", "/usr/lib/x86_64-linux-gnu/dri"),
        ):
            result = cache.detect_encoders()
            assert result["nvenc"] is False
            assert result["amf"] is False
            assert result["qsv"] is False
            assert result["vaapi"] is True

    def test_detect_encoders_nvenc_only(self):
        """Test detect_encoders when only NVENC is available."""

        def mock_test(cmd, timeout=5, env=None):
            if "h264_nvenc" in cmd:
                return True, ""
            return False, "not available"

        with mock.patch.object(cache, "_test_encoder", side_effect=mock_test):
            result = cache.detect_encoders()
            assert result["nvenc"] is True
            assert result["amf"] is False
            assert result["qsv"] is False
            assert result["vaapi"] is False

    def test_detect_encoders_vaapi_command_structure(self):
        """Test detect_encoders passes correct VAAPI command structure when GPU detected."""
        captured_cmds = []
        captured_envs = []

        def capture_cmd(cmd, timeout=5, env=None):
            captured_cmds.append(cmd)
            captured_envs.append(env)
            return False, "test"

        # Mock auto-detected VAAPI device
        with (
            mock.patch.object(cache, "_test_encoder", side_effect=capture_cmd),
            mock.patch.object(cache, "VAAPI_DEVICE", "/dev/dri/renderD128"),
            mock.patch.object(cache, "LIBVA_DRIVER", "i965"),
            mock.patch.object(cache, "DRI_PATH", "/usr/lib/x86_64-linux-gnu/dri"),
        ):
            cache.detect_encoders()

        # Find VAAPI commands (now 2: High profile first, then baseline fallback)
        vaapi_cmds = [c for c in captured_cmds if "h264_vaapi" in c]
        assert len(vaapi_cmds) == 2
        # First command: High profile (default, no -profile:v)
        assert "-init_hw_device" in vaapi_cmds[0]
        assert "hwupload" in " ".join(vaapi_cmds[0])
        assert "constrained_baseline" not in " ".join(vaapi_cmds[0])
        # Second command: constrained_baseline fallback
        assert "constrained_baseline" in " ".join(vaapi_cmds[1])

    def test_detect_encoders_qsv_command_structure(self):
        """Test detect_encoders passes correct QSV command structure."""
        captured_cmds = []

        def capture_cmd(cmd, timeout=5, env=None):
            captured_cmds.append(cmd)
            return False, "test"

        with mock.patch.object(cache, "_test_encoder", side_effect=capture_cmd):
            cache.detect_encoders()

        # Find QSV command
        qsv_cmd = [c for c in captured_cmds if "h264_qsv" in c][0]
        assert "-hwaccel" in qsv_cmd
        assert "qsv" in qsv_cmd
        assert "-hwaccel_output_format" in qsv_cmd

    def test_refresh_encoders_updates_global(self):
        """Test refresh_encoders updates AVAILABLE_ENCODERS."""
        original = cache.AVAILABLE_ENCODERS.copy()

        with mock.patch.object(
            cache,
            "detect_encoders",
            return_value={"nvenc": True, "amf": True, "qsv": True, "vaapi": True},
        ):
            result = cache.refresh_encoders()
            assert cache.AVAILABLE_ENCODERS == {
                "nvenc": True,
                "amf": True,
                "qsv": True,
                "vaapi": True,
            }
            assert result == cache.AVAILABLE_ENCODERS

        # Restore original
        cache.AVAILABLE_ENCODERS = original

    def test_default_encoder_prefers_nvenc_with_vaapi(self):
        """Test _default_encoder prefers NVENC+VAAPI when both available."""
        original = cache.AVAILABLE_ENCODERS.copy()
        cache.AVAILABLE_ENCODERS = {
            "nvenc": True,
            "amf": True,
            "qsv": True,
            "vaapi": True,
        }
        try:
            assert cache._default_encoder() == "nvenc+vaapi"
        finally:
            cache.AVAILABLE_ENCODERS = original

    def test_default_encoder_nvenc_without_vaapi(self):
        """Test _default_encoder uses NVENC+software when VAAPI unavailable."""
        original = cache.AVAILABLE_ENCODERS.copy()
        cache.AVAILABLE_ENCODERS = {
            "nvenc": True,
            "amf": False,
            "qsv": False,
            "vaapi": False,
        }
        try:
            assert cache._default_encoder() == "nvenc+software"
        finally:
            cache.AVAILABLE_ENCODERS = original

    def test_default_encoder_falls_back_to_amf(self):
        """Test _default_encoder falls back to AMF when NVENC unavailable."""
        original = cache.AVAILABLE_ENCODERS.copy()
        cache.AVAILABLE_ENCODERS = {
            "nvenc": False,
            "amf": True,
            "qsv": True,
            "vaapi": True,
        }
        try:
            assert cache._default_encoder() == "amf+vaapi"
        finally:
            cache.AVAILABLE_ENCODERS = original

    def test_default_encoder_falls_back_to_qsv(self):
        """Test _default_encoder falls back to QSV when NVENC/AMF unavailable."""
        original = cache.AVAILABLE_ENCODERS.copy()
        cache.AVAILABLE_ENCODERS = {
            "nvenc": False,
            "amf": False,
            "qsv": True,
            "vaapi": True,
        }
        try:
            assert cache._default_encoder() == "qsv"
        finally:
            cache.AVAILABLE_ENCODERS = original

    def test_default_encoder_falls_back_to_vaapi(self):
        """Test _default_encoder falls back to VAAPI when NVENC/AMF/QSV unavailable."""
        original = cache.AVAILABLE_ENCODERS.copy()
        cache.AVAILABLE_ENCODERS = {
            "nvenc": False,
            "amf": False,
            "qsv": False,
            "vaapi": True,
        }
        try:
            assert cache._default_encoder() == "vaapi"
        finally:
            cache.AVAILABLE_ENCODERS = original

    def test_default_encoder_falls_back_to_software(self):
        """Test _default_encoder falls back to software as last resort."""
        original = cache.AVAILABLE_ENCODERS.copy()
        cache.AVAILABLE_ENCODERS = {
            "nvenc": False,
            "amf": False,
            "qsv": False,
            "vaapi": False,
        }
        try:
            assert cache._default_encoder() == "software"
        finally:
            cache.AVAILABLE_ENCODERS = original


class TestLogoCache:
    """Tests for logo caching functions."""

    def test_sanitize_name_removes_path_traversal(self):
        assert ".." not in cache._sanitize_name("../../../etc/passwd")
        assert "/" not in cache._sanitize_name("foo/bar")
        assert "\\" not in cache._sanitize_name("foo\\bar")

    def test_sanitize_name_keeps_safe_chars(self):
        assert cache._sanitize_name("my-source_123") == "my-source_123"
        assert cache._sanitize_name("Source Name") == "Source Name"

    def test_sanitize_name_truncates_long_names(self):
        long_name = "a" * 300
        result = cache._sanitize_name(long_name)
        assert len(result) == 224

    def test_sanitize_name_empty_returns_default(self):
        assert cache._sanitize_name("") == "default"
        assert cache._sanitize_name("!!!") == "default"

    def test_url_to_filename_extracts_name(self):
        result = cache._url_to_filename("http://example.com/logos/channel1.png")
        assert result.startswith("channel1_")
        assert len(result) == len("channel1_") + 8  # name + underscore + 8 char hash

    def test_url_to_filename_strips_extension(self):
        result = cache._url_to_filename("http://example.com/logo.png")
        assert not result.endswith(".png")
        assert result.startswith("logo_")

    def test_url_to_filename_hash_differs_by_url(self):
        r1 = cache._url_to_filename("http://example.com/a/logo.png")
        r2 = cache._url_to_filename("http://example.com/b/logo.png")
        # Same base name but different hashes
        assert r1.startswith("logo_")
        assert r2.startswith("logo_")
        assert r1 != r2

    def test_url_to_filename_fallback_to_hash(self):
        result = cache._url_to_filename("http://example.com/")
        assert len(result) == 8  # Just the hash

    def test_save_and_get_cached_logo(self, cache_module, tmp_path):
        cache_module.LOGOS_DIR = tmp_path / "logos"
        cache_module.LOGOS_DIR.mkdir()

        # Save a logo
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake PNG
        path = cache_module.save_logo(
            "TestSource", "http://example.com/logo.png", data, "image/png"
        )
        assert path.exists()
        assert path.suffix == ".png"
        assert path.read_bytes() == data

        # Get cached logo
        cached = cache_module.get_cached_logo("TestSource", "http://example.com/logo.png")
        assert cached == path

    def test_get_cached_logo_returns_none_when_missing(self, cache_module, tmp_path):
        cache_module.LOGOS_DIR = tmp_path / "logos"
        cache_module.LOGOS_DIR.mkdir()

        cached = cache_module.get_cached_logo("NoSource", "http://missing.com/logo.png")
        assert cached is None

    def test_get_cached_logo_expires(self, cache_module, tmp_path):
        import time

        cache_module.LOGOS_DIR = tmp_path / "logos"
        cache_module.LOGOS_DIR.mkdir()

        # Save a logo
        data = b"\x89PNG" + b"\x00" * 100
        path = cache_module.save_logo("TestSource", "http://example.com/old.png", data, "image/png")

        # Backdate the file
        old_time = time.time() - cache_module.LOGO_CACHE_TTL - 100
        import os

        os.utime(path, (old_time, old_time))

        # Should be expired
        cached = cache_module.get_cached_logo("TestSource", "http://example.com/old.png")
        assert cached is None
        assert not path.exists()  # Should be deleted

    def test_save_logo_content_type_mapping(self, cache_module, tmp_path):
        cache_module.LOGOS_DIR = tmp_path / "logos"
        cache_module.LOGOS_DIR.mkdir()

        data = b"test"
        assert cache_module.save_logo("s", "http://a.com/1", data, "image/jpeg").suffix == ".jpg"
        assert cache_module.save_logo("s", "http://a.com/2", data, "image/gif").suffix == ".gif"
        assert cache_module.save_logo("s", "http://a.com/3", data, "image/webp").suffix == ".webp"
        assert cache_module.save_logo("s", "http://a.com/4", data, "image/svg+xml").suffix == ".svg"
        assert cache_module.save_logo("s", "http://a.com/5", data, "unknown/type").suffix == ".png"


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)
