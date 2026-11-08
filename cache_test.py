"""Tests for cache.py."""

from __future__ import annotations

from concurrent import futures
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import json
import os
import subprocess
import time

import pytest

import cache


@pytest.fixture
def cache_module(tmp_path: Path, monkeypatch):
    """Import cache module with temp directories."""

    # Skip real ffmpeg hardware probing on first settings load.
    monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", {"nvenc": False})
    monkeypatch.setattr(cache, "LOGOS_DIR", tmp_path / "logos")
    # Patch paths to temp locations.
    original_server_settings = cache.SERVER_SETTINGS_FILE
    original_users_dir = cache.USERS_DIR
    original_cache_dir = cache.CACHE_DIR
    cache.SERVER_SETTINGS_FILE = tmp_path / "server_settings.json"
    cache.USERS_DIR = tmp_path / "users"
    cache.USERS_DIR.mkdir(exist_ok=True)
    cache.CACHE_DIR = tmp_path / "cache"
    cache.CACHE_DIR.mkdir(exist_ok=True)

    # Clear memory cache.
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
        # Only called once - second call uses memory cache.
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
        assert settings["live_dvr_mins"] == 120

    def test_save_and_load_settings(self, cache_module):
        settings = {"sources": [{"id": "s1", "name": "Test"}], "custom": True}
        cache_module.save_server_settings(settings)
        loaded = cache_module.load_server_settings()
        assert loaded["sources"] == [{"id": "s1", "name": "Test"}]
        assert loaded["custom"] is True

    def test_live_dvr_window_is_user_set(self, cache_module):
        cache_module.save_server_settings({"live_dvr_mins": 999})
        assert cache_module.load_server_settings()["live_dvr_mins"] == 999
        cache_module.save_server_settings({"live_dvr_mins": -5})
        assert cache_module.load_server_settings()["live_dvr_mins"] == 0

    @pytest.mark.parametrize("value", ["120", [], {}])
    def test_live_dvr_window_rejects_non_integer_settings(self, cache_module, value):
        cache_module.save_server_settings({"live_dvr_mins": value})
        with pytest.raises(TypeError, match="live_dvr_mins must be an integer"):
            cache_module.load_server_settings()


class TestUserSettings:
    def test_load_user_settings_defaults(self, cache_module):
        settings = cache_module.load_user_settings("testuser")
        assert settings["guide_filter"] == []
        assert settings["captions_enabled"] is True
        assert settings["watch_history"] == {}
        assert settings["sort_live_by_views"] is True

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
        # Save at 96% watched.
        cache_module.save_watch_position("user1", "http://video.url", 960.0, 1000.0)
        entry = cache_module.get_watch_position("user1", "http://video.url")
        assert entry is None  # Should be reset.

    def test_live_views_are_counted_by_source_and_stream(self, cache_module):
        assert cache_module.record_live_view("user1", "source-a", 7) == 1
        assert cache_module.record_live_view("user1", "source-a", 7) == 2
        assert cache_module.record_live_view("user1", "source-b", 7) == 1

        assert cache_module.load_live_view_counts("user1") == {
            "source-a:7": 2,
            "source-b:7": 1,
        }


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
                },
            ],
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
                },
            ],
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
        settings = {"sources": [{"id": "s1", "name": "S1", "type": "xtream", "url": "http://x"}]}
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
        with (
            mock.patch.object(cache, "_test_encoder", return_value=(False, "not found")),
            mock.patch.object(cache, "VAAPI_DEVICE", None),
        ):
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
            # Return True only for VAAPI.
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

        # Mock auto-detected VAAPI device.
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
        # Second command: constrained_baseline fallback.
        assert "constrained_baseline" in " ".join(vaapi_cmds[1])

    def test_detect_encoders_qsv_command_structure(self):
        """Test detect_encoders passes correct QSV command structure."""
        captured_cmds = []

        def capture_cmd(cmd, timeout=5, env=None):
            captured_cmds.append(cmd)
            return False, "test"

        with mock.patch.object(cache, "_test_encoder", side_effect=capture_cmd):
            cache.detect_encoders()

        # Find QSV command.
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

        # Restore original.
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


class TestInitialize:
    def test_initialize_paths_prefers_legacy_cache_dir(self, monkeypatch, tmp_path):
        (tmp_path / "cache").mkdir()
        monkeypatch.setattr(cache, "APP_DIR", tmp_path)
        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / ".cache")
        monkeypatch.setattr(cache, "SERVER_SETTINGS_FILE", tmp_path / ".cache/server_settings.json")
        monkeypatch.setattr(cache, "USERS_DIR", tmp_path / ".cache/users")
        monkeypatch.setattr(cache, "LOGOS_DIR", tmp_path / ".cache/logos")
        cache.initialize_paths()
        assert tmp_path / "cache" == cache.CACHE_DIR
        assert tmp_path / "cache/server_settings.json" == cache.SERVER_SETTINGS_FILE
        assert tmp_path / "cache/users" == cache.USERS_DIR
        assert tmp_path / "cache/logos" == cache.LOGOS_DIR

    def test_initialize_paths_puts_installed_state_in_user_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("NETV_DATA_DIR", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(cache, "APP_DIR", tmp_path)
        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / ".cache")
        monkeypatch.setattr(cache, "SERVER_SETTINGS_FILE", tmp_path / ".cache/server_settings.json")
        monkeypatch.setattr(cache, "USERS_DIR", tmp_path / ".cache/users")
        monkeypatch.setattr(cache, "LOGOS_DIR", tmp_path / ".cache/logos")
        cache.initialize_paths()
        state = tmp_path / "xdg" / "netv"
        assert state == cache.CACHE_DIR
        assert state / "server_settings.json" == cache.SERVER_SETTINGS_FILE
        assert state / "users" == cache.USERS_DIR
        assert state / "logos" == cache.LOGOS_DIR

    def test_initialize_paths_keeps_caller_overrides(self, monkeypatch, tmp_path):
        (tmp_path / "cache").mkdir()
        custom = tmp_path / "custom"
        monkeypatch.setattr(cache, "APP_DIR", tmp_path)
        monkeypatch.setattr(cache, "CACHE_DIR", custom)
        monkeypatch.setattr(cache, "SERVER_SETTINGS_FILE", custom / "settings.json")
        monkeypatch.setattr(cache, "USERS_DIR", custom / "people")
        monkeypatch.setattr(cache, "LOGOS_DIR", custom / "icons")
        cache.initialize_paths()
        assert custom == cache.CACHE_DIR
        assert custom / "settings.json" == cache.SERVER_SETTINGS_FILE
        assert custom / "people" == cache.USERS_DIR
        assert custom / "icons" == cache.LOGOS_DIR
        assert not custom.exists()

    def test_initialize_detects_once_into_shared_table(self, monkeypatch):
        table: dict[str, bool] = {}
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", table)
        monkeypatch.setattr(cache, "VAAPI_DEVICE", None)
        monkeypatch.setattr(cache, "LIBVA_DRIVER", None)
        monkeypatch.setattr(cache, "DRI_PATH", None)
        monkeypatch.setattr(cache, "_detect_vaapi_device", lambda: "/dev/dri/renderD128")
        monkeypatch.setattr(cache, "_detect_libva_driver", lambda: "iHD")
        monkeypatch.setattr(cache, "_detect_dri_path", lambda: "/usr/lib/dri")
        detect = mock.Mock(return_value={"nvenc": True})
        monkeypatch.setattr(cache, "detect_encoders", detect)
        cache.initialize()
        cache.initialize()
        detect.assert_called_once()
        assert table == {"nvenc": True}
        assert cache.VAAPI_DEVICE == "/dev/dri/renderD128"
        assert cache.LIBVA_DRIVER == "iHD"
        assert cache.DRI_PATH == "/usr/lib/dri"

    def test_initialize_keeps_supplied_hardware(self, monkeypatch):
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", {})
        monkeypatch.setattr(cache, "VAAPI_DEVICE", "/dev/custom")
        monkeypatch.setattr(cache, "LIBVA_DRIVER", "radeonsi")
        monkeypatch.setattr(cache, "DRI_PATH", "/opt/dri")
        monkeypatch.setattr(cache, "detect_encoders", lambda: {"vaapi": True})
        cache.initialize()
        assert (cache.VAAPI_DEVICE, cache.LIBVA_DRIVER, cache.DRI_PATH) == (
            "/dev/custom",
            "radeonsi",
            "/opt/dri",
        )

    def test_refresh_encoders_updates_shared_table_in_place(self, monkeypatch):
        table = {"nvenc": True}
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", table)
        monkeypatch.setattr(cache, "detect_encoders", lambda: {"qsv": True})
        assert cache.refresh_encoders() is table
        assert table == {"qsv": True}


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
        assert len(result) == len("channel1_") + 8  # `name` + underscore + 8 char hash.

    def test_url_to_filename_strips_extension(self):
        result = cache._url_to_filename("http://example.com/logo.png")
        assert not result.endswith(".png")
        assert result.startswith("logo_")

    def test_url_to_filename_hash_differs_by_url(self):
        r1 = cache._url_to_filename("http://example.com/a/logo.png")
        r2 = cache._url_to_filename("http://example.com/b/logo.png")
        # Same base name but different hashes.
        assert r1.startswith("logo_")
        assert r2.startswith("logo_")
        assert r1 != r2

    def test_url_to_filename_fallback_to_hash(self):
        result = cache._url_to_filename("http://example.com/")
        assert len(result) == 8  # Just the hash.

    def test_save_and_get_cached_logo(self, cache_module, tmp_path):
        cache_module.LOGOS_DIR = tmp_path / "logos"
        cache_module.LOGOS_DIR.mkdir()

        # Save a logo.
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake PNG.
        path = cache_module.save_logo(
            "TestSource",
            "http://example.com/logo.png",
            data,
            "image/png",
        )
        assert path.exists()
        assert path.suffix == ".png"
        assert path.read_bytes() == data

        # Get cached logo.
        cached = cache_module.get_cached_logo("TestSource", "http://example.com/logo.png")
        assert cached == path

    def test_get_cached_logo_returns_none_when_missing(self, cache_module, tmp_path):
        cache_module.LOGOS_DIR = tmp_path / "logos"
        cache_module.LOGOS_DIR.mkdir()

        cached = cache_module.get_cached_logo("NoSource", "http://missing.com/logo.png")
        assert cached is None

    def test_get_cached_logo_expires(self, cache_module, tmp_path):

        cache_module.LOGOS_DIR = tmp_path / "logos"
        cache_module.LOGOS_DIR.mkdir()

        # Save a logo.
        data = b"\x89PNG" + b"\x00" * 100
        path = cache_module.save_logo("TestSource", "http://example.com/old.png", data, "image/png")

        # Backdate the file.
        old_time = time.time() - 7 * 24 * 3600 - 100

        os.utime(path, (old_time, old_time))

        # Should be expired.
        cached = cache_module.get_cached_logo("TestSource", "http://example.com/old.png")
        assert cached is None
        assert not path.exists()  # Should be deleted.

    def test_save_logo_content_type_mapping(self, cache_module, tmp_path):
        cache_module.LOGOS_DIR = tmp_path / "logos"
        cache_module.LOGOS_DIR.mkdir()

        data = b"test"
        assert cache_module.save_logo("s", "http://a.com/1", data, "image/jpeg").suffix == ".jpg"
        assert cache_module.save_logo("s", "http://a.com/2", data, "image/gif").suffix == ".gif"
        assert cache_module.save_logo("s", "http://a.com/3", data, "image/webp").suffix == ".webp"
        assert cache_module.save_logo("s", "http://a.com/4", data, "image/svg+xml").suffix == ".svg"
        assert cache_module.save_logo("s", "http://a.com/5", data, "unknown/type").suffix == ".png"


class TestFileCacheExtras:
    def test_load_in_subprocess(self, cache_module, monkeypatch):
        monkeypatch.setattr(cache.futures, "ProcessPoolExecutor", futures.ThreadPoolExecutor)
        cache_module.save_file_cache("proc", [1, 2])
        data, ts = cache_module.load_file_cache("proc", use_process=True)
        assert data == [1, 2]
        assert ts > 0

    def test_parse_json_file(self, tmp_path):
        good = tmp_path / "good.json"
        good.write_text(json.dumps({"data": {"a": 1}}))
        assert cache._parse_json_file(str(good)) == ({"a": 1}, 0)
        bad = tmp_path / "bad.json"
        bad.write_text("{")
        assert cache._parse_json_file(str(bad)) is None
        assert cache._parse_json_file(str(tmp_path / "missing.json")) is None

    def test_clear_all_file_caches(self, cache_module):
        cache_module.save_file_cache("live_data", [1])
        cache_module.save_file_cache("vod_data", [2])
        cache_module.save_file_cache("other", [3])
        cache_module.get_cache()["live"] = 1
        assert cache_module.clear_all_file_caches() == 2
        assert cache_module.load_file_cache("live_data") is None
        assert cache_module.load_file_cache("other") is not None
        assert cache_module.get_cache() == {}

    def test_get_cache_lock(self):
        assert cache.get_cache_lock() is cache._cache_lock


class TestStaleWhileRevalidate:
    def _backdate(self, cache_module, key, data, age):
        path = cache_module.CACHE_DIR / f"{key}.json"
        path.write_text(json.dumps({"data": data, "timestamp": time.time() - age}))

    def test_memory_hit_with_stale_file_refreshes_in_background(self, cache_module):
        self._backdate(cache_module, "k", {"v": "file"}, 2 * 24 * 3600)
        cache_module.get_cache()["k"] = {"v": "mem"}
        fetch = mock.Mock()
        with mock.patch.object(cache.threading, "Thread") as thread:
            assert cache_module.get_cached_info("k", fetch) == {"v": "mem"}
        assert thread.call_args.kwargs["args"] == ("k", fetch)
        thread.return_value.start.assert_called_once()
        fetch.assert_not_called()

    def test_stale_file_served_and_refreshed(self, cache_module):
        self._backdate(cache_module, "k", {"v": "file"}, 2 * 24 * 3600)
        fetch = mock.Mock()
        with mock.patch.object(cache.threading, "Thread") as thread:
            assert cache_module.get_cached_info("k", fetch) == {"v": "file"}
        assert cache_module.get_cache()["k"] == {"v": "file"}
        thread.return_value.start.assert_called_once()

    def test_fresh_file_served_without_refresh(self, cache_module):
        self._backdate(cache_module, "k", {"v": "file"}, 60)
        with mock.patch.object(cache.threading, "Thread") as thread:
            assert cache_module.get_cached_info("k", mock.Mock()) == {"v": "file"}
        thread.assert_not_called()

    def test_force_refetches_despite_file(self, cache_module):
        self._backdate(cache_module, "k", {"v": "file"}, 60)
        cache_module.get_cache()["k"] = {"v": "mem"}
        assert cache_module.get_cached_info("k", lambda: {"v": "new"}, force=True) == {"v": "new"}
        assert cache_module.load_file_cache("k")[0] == {"v": "new"}

    def test_refresh_success_and_failure(self, cache_module, caplog):
        caplog.set_level("INFO", logger=cache.log.name)
        cache._refresh_cached_info("k", lambda: [9])
        assert cache_module.get_cache()["k"] == [9]
        assert cache_module.load_file_cache("k")[0] == [9]
        cache._refresh_cached_info("k", mock.Mock(side_effect=RuntimeError("boom")))
        assert cache_module.get_cache()["k"] == [9]
        assert "Background refresh failed for k: boom" in caplog.text


class TestServerSettingsExtras:
    @pytest.mark.parametrize(
        ("old", "encoders", "new"),
        [
            ("nvidia", {"vaapi": True}, "nvenc+vaapi"),
            ("nvidia", {"vaapi": False}, "nvenc+software"),
            ("intel", {"vaapi": False}, "qsv"),
            ("vaapi", {"vaapi": False}, "vaapi"),
        ],
    )
    def test_legacy_transcode_hw_migrated(self, cache_module, monkeypatch, old, encoders, new):
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", encoders)
        cache_module.save_server_settings({"transcode_hw": old})
        assert cache_module.load_server_settings()["transcode_hw"] == new

    @pytest.mark.parametrize(
        ("settings", "expected"),
        [
            ({"user_agent_preset": "default"}, None),
            ({"user_agent_preset": "custom", "user_agent_custom": "Mine/1"}, "Mine/1"),
            ({"user_agent_preset": "custom", "user_agent_custom": ""}, None),
            ({"user_agent_preset": "vlc"}, "VLC/3.0.20 LibVLC/3.0.20"),
            ({}, "TiviMate/4.7.0"),
            ({"user_agent_preset": "bogus"}, None),
        ],
    )
    def test_user_agent(self, cache_module, settings, expected):
        cache_module.save_server_settings(settings)
        assert cache_module.get_user_agent() == expected

    def test_hidden_source_ids(self, cache_module):
        base = {"type": "m3u", "url": "http://x"}
        cache_module.save_server_settings(
            {
                "sources": [
                    {"id": "all", "name": "A", **base},
                    {"id": "mine", "name": "B", "users": ["alice"], **base},
                    {"id": "theirs", "name": "C", "users": ["bob"], **base},
                ],
            },
        )
        assert cache_module.hidden_source_ids("alice", admin=True) == set()
        assert cache_module.hidden_source_ids("alice", admin=False) == {"theirs"}

    def test_update_source_epg_url_unknown_source(self, cache_module):
        cache_module.save_server_settings({"sources": [{"id": "s1", "name": "S1"}]})
        cache_module.update_source_epg_url("s2", "http://epg")
        assert "epg_url" not in cache_module.load_server_settings()["sources"][0]


class TestUserSettingsExtras:
    def test_missing_watch_position(self, cache_module):
        assert cache_module.get_watch_position("user1", "http://none") is None

    def test_watch_position_without_duration_is_kept(self, cache_module):
        cache_module.save_watch_position("user1", "http://live", 50.0, 0)
        assert cache_module.get_watch_position("user1", "http://live")["position"] == 50.0

    def test_watch_history_keeps_200_most_recent(self, cache_module):
        history = {f"u{i}": {"position": 1, "duration": 10, "updated": i} for i in range(200)}
        cache_module.save_user_settings("user1", {"watch_history": history})
        cache_module.save_watch_position("user1", "new", 1.0, 10.0)
        saved = cache_module.load_user_settings("user1")["watch_history"]
        assert len(saved) == 200
        assert "new" in saved
        assert "u0" not in saved
        assert "u1" in saved

    @pytest.mark.parametrize("username", ["", "a" * 65, "a..b", "a/b", "a\\b"])
    def test_invalid_usernames_rejected(self, cache_module, username):
        with pytest.raises(ValueError, match="Invalid username"):
            cache_module.load_user_settings(username)

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            {"counts": []},
            {"counts": {"a:1": True}},
            {"counts": {"a:1": "3"}},
            {"counts": {"a:1": -1}},
        ],
    )
    def test_invalid_live_view_counts_rejected(self, cache_module, payload):
        path = cache_module.USERS_DIR / "user1" / "live_views.json"
        path.parent.mkdir()
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="Invalid live view counts"):
            cache_module.load_live_view_counts("user1")


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Redirect cache's absolute system paths under tmp_path."""
    monkeypatch.setattr(
        cache,
        "pathlib",
        SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
    )
    return tmp_path


class TestGpuDetection:
    @pytest.mark.parametrize(
        ("stdout", "vendor"),
        [
            ("00:02.0 VGA compatible controller [0300]: Intel [8086:46a6]\n", "8086"),
            ("03:00.0 Display controller [0380]: AMD [1002:73ff]\n", "1002"),
        ],
    )
    def test_lspci_vendor(self, fake_root, stdout, vendor):
        with mock.patch.object(
            cache.subprocess,
            "run",
            return_value=mock.Mock(stdout=("00:1f.0 ISA bridge [8086:7a84]\n" + stdout)),
        ):
            assert cache._get_gpu_vendor() == vendor

    def test_sysfs_fallback(self, fake_root):
        drm = fake_root / "sys/class/drm"
        for name, vendor in [("card0", "0x10de"), ("card1", "0x1002")]:
            (drm / name / "device").mkdir(parents=True)
            (drm / name / "device" / "vendor").write_text(vendor + "\n")
        (drm / "card2").mkdir()  # No vendor file.
        (drm / "card1-HDMI-A-1" / "device").mkdir(parents=True)
        (drm / "card1-HDMI-A-1" / "device" / "vendor").write_text("0x8086\n")
        (drm / "renderD128").mkdir()
        with mock.patch.object(cache.subprocess, "run", side_effect=FileNotFoundError):
            assert cache._get_gpu_vendor() == "1002"

    def test_unknown_gpu(self, fake_root):
        (fake_root / "sys/class/drm/card0/device").mkdir(parents=True)
        (fake_root / "sys/class/drm/card0/device/vendor").write_text("0x10de\n")
        with mock.patch.object(
            cache.subprocess,
            "run",
            return_value=mock.Mock(stdout="01:00.0 3D controller [0302]: NVIDIA [10de:2684]\n"),
        ):
            assert cache._get_gpu_vendor() is None

    def test_no_sysfs(self, fake_root):
        with mock.patch.object(cache.subprocess, "run", return_value=mock.Mock(stdout="")):
            assert cache._get_gpu_vendor() is None

    def test_unreadable_sysfs(self, fake_root):
        (fake_root / "sys/class").mkdir(parents=True)
        (fake_root / "sys/class/drm").write_text("")  # iterdir() raises NotADirectoryError.
        with mock.patch.object(cache.subprocess, "run", return_value=mock.Mock(stdout="")):
            assert cache._get_gpu_vendor() is None

    def test_vaapi_device(self, fake_root):
        assert cache._detect_vaapi_device() is None
        (fake_root / "dev/dri").mkdir(parents=True)
        (fake_root / "dev/dri/renderD128").touch()
        assert cache._detect_vaapi_device() == str(fake_root / "dev/dri/renderD128")

    def test_dri_path(self, fake_root):
        assert cache._detect_dri_path() is None
        (fake_root / "usr/lib/dri").mkdir(parents=True)
        assert cache._detect_dri_path() == "/usr/lib/dri"
        (fake_root / "usr/lib64/dri").mkdir(parents=True)
        assert cache._detect_dri_path() == "/usr/lib64/dri"

    @pytest.mark.parametrize(
        ("vendor", "ihd", "driver"),
        [
            ("8086", True, "iHD"),
            ("8086", False, "i965"),
            ("1002", False, "radeonsi"),
            (None, False, None),
        ],
    )
    def test_libva_driver(self, fake_root, monkeypatch, vendor, ihd, driver):
        monkeypatch.setattr(cache, "_get_gpu_vendor", lambda: vendor)
        dri = fake_root / "usr/lib/dri"
        dri.mkdir(parents=True)
        if ihd:
            (dri / "iHD_drv_video.so").touch()
        assert cache._detect_libva_driver() == driver


class TestEncoderExtras:
    def test_vaapi_baseline_fallback(self, monkeypatch):
        def fake(cmd, timeout=5, env=None):
            return ("constrained_baseline" in cmd, "no high profile")

        monkeypatch.setattr(cache, "_test_encoder", fake)
        monkeypatch.setattr(cache, "VAAPI_DEVICE", "/dev/dri/renderD128")
        monkeypatch.setattr(cache, "LIBVA_DRIVER", "radeonsi")
        monkeypatch.setattr(cache, "DRI_PATH", "/usr/lib/dri")
        result = cache.detect_encoders()
        assert result["vaapi"] is True
        assert result["vaapi_baseline_only"] is True

    def test_test_encoder_passes_env_and_picks_error_line(self):
        stderr = b"[h264 @ 0x1] noisy\nNo device available\n"
        with mock.patch(
            "subprocess.run",
            return_value=mock.Mock(returncode=1, stderr=stderr),
        ) as run:
            assert cache._test_encoder(["ffmpeg"], env={"LIBVA_DRIVER_NAME": "iHD"}) == (
                False,
                "No device available",
            )
        assert run.call_args.kwargs["env"]["LIBVA_DRIVER_NAME"] == "iHD"

    def test_test_encoder_bracket_only_and_empty_stderr(self):
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stderr=b"[x] y")):
            assert cache._test_encoder(["ffmpeg"]) == (False, "[x] y")
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stderr=b"")):
            assert cache._test_encoder(["ffmpeg"]) == (False, "unknown error")

    def test_test_encoder_unexpected_error(self):
        with mock.patch("subprocess.run", side_effect=PermissionError("denied")):
            assert cache._test_encoder(["ffmpeg"]) == (False, "denied")


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
