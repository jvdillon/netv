"""Tests for cache.py."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import functools
import json
import os
import subprocess
import time

import pytest

from netv import cache


@pytest.fixture
def cache_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point cache at temp directories."""
    # Skip real ffmpeg hardware probing on first settings load.
    monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", {"nvenc": False})
    monkeypatch.setattr(cache, "LOGOS_DIR", tmp_path / "logos")
    original_server_settings = cache.SERVER_SETTINGS_FILE
    original_users_dir = cache.USERS_DIR
    original_cache_dir = cache.CACHE_DIR
    cache.SERVER_SETTINGS_FILE = tmp_path / "server_settings.json"
    cache.USERS_DIR = tmp_path / "users"
    cache.USERS_DIR.mkdir(exist_ok=True)
    cache.CACHE_DIR = tmp_path / "cache"
    cache.CACHE_DIR.mkdir(exist_ok=True)
    _reset_memory()

    yield

    cache.SERVER_SETTINGS_FILE = original_server_settings
    cache.USERS_DIR = original_users_dir
    cache.CACHE_DIR = original_cache_dir
    _reset_memory()


def _reset_memory() -> None:
    cache.clear_all_caches()
    cache.get_cache().pop("epg", None)


def _write_settings(raw: dict[str, object]) -> None:
    """Store a settings file as an older or hand-edited release would."""
    cache.SERVER_SETTINGS_FILE.write_text(json.dumps(raw))


def _source(
    source_id: str,
    *,
    users: list[str] | None = None,
    epg_url: str = "",
) -> cache.SourceConfig:
    source: cache.SourceConfig = {
        "id": source_id,
        "name": source_id,
        "type": "m3u",
        "url": "http://x",
    }
    if users is not None:
        source["users"] = users
    if epg_url:
        source["epg_url"] = epg_url
    return source


@pytest.mark.usefixtures("cache_module")
class TestFileCache:
    def test_save_and_load_file_cache(self):
        cache.save_file_cache("test", {"key": "value"})
        result = cache.load_file_cache("test")
        assert result is not None
        data, ts = result
        assert data == {"key": "value"}
        assert ts > 0

    def test_load_nonexistent_cache(self):
        assert cache.load_file_cache("nonexistent") is None

    def test_load_corrupted_cache(self):
        path = cache.CACHE_DIR / "corrupted.json"
        path.write_text("not valid json")
        assert cache.load_file_cache("corrupted") is None


@pytest.mark.usefixtures("cache_module")
class TestMemoryCache:
    def test_get_cache_returns_reference(self):
        cache.get_cache()["epg_error"] = "down"
        memory = cache.get_cache()
        assert "epg_error" in memory
        assert memory["epg_error"] == "down"

    def test_clear_all_caches_covers_every_key_but_epg(self):
        memory = cache.get_cache()
        memory.update(
            {
                "live_categories": [],
                "live_streams": [],
                "vod_categories": [],
                "vod_streams": [],
                "series_categories": [],
                "series": [],
                "epg_urls": [],
                "epg_error": "x",
                "epg": None,
            },
        )
        # A key added to MemoryCache must also be popped by clear_all_caches.
        assert set(memory) == cache.MemoryCache.__optional_keys__
        cache.clear_all_caches()
        assert cache.get_cache() is memory
        assert set(memory) == {"epg"}

    def test_clear_all_caches_preserves_epg(self):
        memory = cache.get_cache()
        memory["epg"] = {"data": "epg"}
        memory["live_streams"] = [{"data": "live"}]
        cache._info_cache["info"] = {"v": 1}
        cache.clear_all_caches()
        assert "epg" in memory
        assert "live_streams" not in memory
        assert cache._info_cache == {}


@pytest.mark.usefixtures("cache_module")
class TestCachedInfo:
    def test_get_cached_info_calls_fetch(self):
        fetch_fn = mock.Mock(return_value={"result": 42})
        result = cache.get_cached_info("test_key", fetch_fn)
        assert result == {"result": 42}
        fetch_fn.assert_called_once()

    def test_get_cached_info_uses_memory_cache(self):
        fetch_fn = mock.Mock(return_value={"result": 1})
        cache.get_cached_info("key1", fetch_fn)
        cache.get_cached_info("key1", fetch_fn)
        # Only called once - second call uses memory cache.
        fetch_fn.assert_called_once()

    def test_get_cached_info_force_bypasses_memory(self):
        fetch_fn = mock.Mock(return_value={"result": 1})
        cache.get_cached_info("key2", fetch_fn)
        cache.get_cached_info("key2", fetch_fn, force=True)
        assert fetch_fn.call_count == 2

    def test_non_object_file_cache_is_refetched(self):
        cache.save_file_cache("key3", [1, 2])
        assert cache.get_cached_info("key3", lambda: {"fresh": True}) == {"fresh": True}


@pytest.mark.usefixtures("cache_module")
class TestSettings:
    def test_load_settings_defaults(self):
        settings = cache.load_server_settings()
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

    def test_save_and_load_settings(self):
        settings = cache.load_server_settings()
        settings["sources"] = [_source("s1")]
        settings["probe_live"] = False
        cache.save_server_settings(settings)
        loaded = cache.load_server_settings()
        assert loaded["sources"] == [_source("s1")]
        assert loaded["probe_live"] is False

    def test_save_keeps_unknown_keys_and_removes_dropped_owned_ones(self):
        _write_settings({"future": {"nested": [1]}, "sr_model": "old"})
        settings = cache.load_server_settings()
        assert "sr_model" in settings
        assert settings["sr_model"] == "old"
        del settings["sr_model"]
        settings["probe_live"] = False
        cache.save_server_settings(settings)
        saved = json.loads(cache.SERVER_SETTINGS_FILE.read_text())
        assert saved["future"] == {"nested": [1]}
        assert "sr_model" not in saved
        assert saved["probe_live"] is False

    def test_save_keeps_unknown_source_fields_across_reorder_and_deletion(self):
        _write_settings(
            {
                "sources": [
                    {**_source("a"), "vendor": "kept-a"},
                    {**_source("b"), "vendor": "dropped-b"},
                    {**_source("c"), "vendor": "kept-c"},
                ],
            },
        )
        settings = cache.load_server_settings()
        sources = settings["sources"]
        settings["sources"] = [sources[2], sources[0]]
        settings["sources"][0]["name"] = "renamed"
        cache.save_server_settings(settings)
        saved = json.loads(cache.SERVER_SETTINGS_FILE.read_text())["sources"]
        assert [(s["id"], s["name"], s["vendor"]) for s in saved] == [
            ("c", "renamed", "kept-c"),
            ("a", "a", "kept-a"),
        ]

    def test_live_dvr_window_is_user_set(self):
        _write_settings({"live_dvr_mins": 999})
        assert cache.load_server_settings()["live_dvr_mins"] == 999
        _write_settings({"live_dvr_mins": -5})
        assert cache.load_server_settings()["live_dvr_mins"] == 0

    @pytest.mark.parametrize("value", ["120", [], {}])
    def test_live_dvr_window_rejects_non_integer_settings(self, value: object):
        _write_settings({"live_dvr_mins": value})
        with pytest.raises(TypeError, match="live_dvr_mins must be an integer"):
            cache.load_server_settings()


@pytest.mark.usefixtures("cache_module")
class TestUserSettings:
    def test_load_user_settings_defaults(self):
        settings = cache.load_user_settings("testuser")
        assert settings["guide_filter"] == []
        assert settings["captions_enabled"] is True
        assert settings["watch_history"] == {}
        assert settings["sort_live_by_views"] is True

    def test_save_and_load_user_settings(self):
        settings = cache.load_user_settings("testuser")
        settings["guide_filter"] = ["cat1", "cat2"]
        settings["captions_enabled"] = False
        settings["vod_filter"] = ["films"]
        cache.save_user_settings("testuser", settings)
        loaded = cache.load_user_settings("testuser")
        assert loaded["guide_filter"] == ["cat1", "cat2"]
        assert loaded["captions_enabled"] is False
        assert "vod_filter" in loaded
        assert loaded["vod_filter"] == ["films"]

    def test_save_keeps_unknown_user_keys(self):
        path = cache.USERS_DIR / "testuser" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"theme": "dark", "cc_lang": "en"}))
        settings = cache.load_user_settings("testuser")
        settings["cc_lang"] = "fr"
        cache.save_user_settings("testuser", settings)
        saved = json.loads(path.read_text())
        assert saved["theme"] == "dark"
        assert saved["cc_lang"] == "fr"

    def test_watch_history_eviction_drops_entries_and_keeps_unknown_fields(self):
        path = cache.USERS_DIR / "testuser" / "settings.json"
        path.parent.mkdir()
        history = {
            f"u{i}": {"position": 1, "duration": 10, "updated": i, "device": f"d{i}"}
            for i in range(200)
        }
        path.write_text(json.dumps({"watch_history": history}))
        cache.save_watch_position("testuser", "new", 1.0, 10.0)
        saved = json.loads(path.read_text())["watch_history"]
        assert "u0" not in saved
        assert saved["u1"]["device"] == "d1"
        assert "device" not in saved["new"]

    def test_watch_position_save_and_get(self):
        cache.save_watch_position("user1", "http://video.url", 120.5, 3600.0)
        entry = cache.get_watch_position("user1", "http://video.url")
        assert entry is not None
        assert entry["position"] == 120.5
        assert entry["duration"] == 3600.0

    def test_watch_position_resets_at_95_percent(self):
        # Save at 96% watched.
        cache.save_watch_position("user1", "http://video.url", 960.0, 1000.0)
        entry = cache.get_watch_position("user1", "http://video.url")
        assert entry is None  # Should be reset.

    def test_live_views_are_counted_by_source_and_stream(self):
        assert cache.record_live_view("user1", "source-a", 7) == 1
        assert cache.record_live_view("user1", "source-a", 7) == 2
        assert cache.record_live_view("user1", "source-b", 7) == 1

        assert cache.load_live_view_counts("user1") == {
            "source-a:7": 2,
            "source-b:7": 1,
        }


@pytest.mark.usefixtures("cache_module")
class TestSource:
    def test_source_dataclass(self):
        source = cache.Source(
            id="test",
            name="Test Source",
            type="xtream",
            url="http://example.com",
        )
        assert source.id == "test"
        assert source.username == ""
        assert source.epg_timeout == 120
        assert source.epg_enabled is True

    def test_get_sources_empty(self):
        sources = cache.get_sources()
        assert sources == []

    def test_get_sources_from_settings(self):
        _write_settings(
            {
                "sources": [
                    {
                        "id": "s1",
                        "name": "Source 1",
                        "type": "m3u",
                        "url": "http://example.com/playlist.m3u",
                    },
                ],
            },
        )
        sources = cache.get_sources()
        assert len(sources) == 1
        assert sources[0].id == "s1"
        assert sources[0].type == "m3u"


@pytest.mark.usefixtures("cache_module")
class TestUpdateSourceEpgUrl:
    def test_update_source_epg_url(self):
        _write_settings({"sources": [_source("s1")]})
        cache.update_source_epg_url("s1", "http://epg.example.com")
        source = cache.load_server_settings()["sources"][0]
        assert "epg_url" in source
        assert source["epg_url"] == "http://epg.example.com"

    def test_update_source_epg_url_not_overwrite(self):
        _write_settings({"sources": [_source("s1", epg_url="http://existing")]})
        cache.update_source_epg_url("s1", "http://new")
        source = cache.load_server_settings()["sources"][0]
        assert "epg_url" in source
        assert source["epg_url"] == "http://existing"

    def test_update_source_epg_url_empty_noop(self):
        _write_settings({"sources": [_source("s1")]})
        cache.update_source_epg_url("s1", "")
        loaded = cache.load_server_settings()
        assert "epg_url" not in loaded["sources"][0]

    def test_update_source_epg_url_does_not_log_credentials(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level("INFO", logger=cache.log.name)
        _write_settings({"sources": [_source("s1")]})

        cache.update_source_epg_url(
            "s1",
            "https://provider.example/xmltv.php?username=private-user&password=private-secret",
        )

        assert "private-user" not in caplog.text
        assert "private-secret" not in caplog.text
        assert "provider.example" in caplog.text


def _encoder_result(
    accept: str,
    captured: list[list[str]] | None = None,
) -> mock.Mock:
    """Fake ``_test_encoder`` accepting commands that mention ``accept``."""

    def run(
        cmd: list[str],
        timeout: int = 5,
        env: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        del timeout, env
        if captured is not None:
            captured.append(cmd)
        return (accept in cmd, "" if accept in cmd else "not available")

    return mock.Mock(side_effect=run)


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
        with (
            mock.patch.object(cache, "_test_encoder", _encoder_result("h264_vaapi")),
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
        with mock.patch.object(cache, "_test_encoder", _encoder_result("h264_nvenc")):
            result = cache.detect_encoders()
            assert result["nvenc"] is True
            assert result["amf"] is False
            assert result["qsv"] is False
            assert result["vaapi"] is False

    def test_detect_encoders_vaapi_command_structure(self):
        """Test detect_encoders passes correct VAAPI command structure when GPU detected."""
        captured_cmds: list[list[str]] = []
        with (
            mock.patch.object(cache, "_test_encoder", _encoder_result("never", captured_cmds)),
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
        captured_cmds: list[list[str]] = []
        with mock.patch.object(cache, "_test_encoder", _encoder_result("never", captured_cmds)):
            cache.detect_encoders()

        # Find QSV command.
        qsv_cmd = next(c for c in captured_cmds if "h264_qsv" in c)
        assert "-hwaccel" in qsv_cmd
        assert "qsv" in qsv_cmd
        assert "-hwaccel_output_format" in qsv_cmd

    def test_refresh_encoders_updates_global(self, monkeypatch: pytest.MonkeyPatch):
        """Test refresh_encoders updates AVAILABLE_ENCODERS."""
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", {})
        detected = {"nvenc": True, "amf": True, "qsv": True, "vaapi": True}
        with mock.patch.object(cache, "detect_encoders", return_value=detected):
            result = cache.refresh_encoders()
        assert detected == cache.AVAILABLE_ENCODERS
        assert result == cache.AVAILABLE_ENCODERS

    @pytest.mark.parametrize(
        ("available", "expected"),
        [
            ({"nvenc": True, "amf": True, "qsv": True, "vaapi": True}, "nvenc+vaapi"),
            ({"nvenc": True, "amf": False, "qsv": False, "vaapi": False}, "nvenc+software"),
            ({"nvenc": False, "amf": True, "qsv": True, "vaapi": True}, "amf+vaapi"),
            ({"nvenc": False, "amf": True, "qsv": False, "vaapi": False}, "amf+software"),
            ({"nvenc": False, "amf": False, "qsv": True, "vaapi": True}, "qsv"),
            ({"nvenc": False, "amf": False, "qsv": False, "vaapi": True}, "vaapi"),
            ({"nvenc": False, "amf": False, "qsv": False, "vaapi": False}, "software"),
        ],
    )
    def test_default_encoder_preference(
        self,
        monkeypatch: pytest.MonkeyPatch,
        available: dict[str, bool],
        expected: str,
    ):
        """Test _default_encoder picks nvenc > amf > qsv > vaapi > software."""
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", available)
        assert cache._default_encoder() == expected


def _point_paths_at(monkeypatch: pytest.MonkeyPatch, app_dir: Path, cache_dir: Path) -> None:
    monkeypatch.setattr(cache, "APP_DIR", app_dir)
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(cache, "SERVER_SETTINGS_FILE", cache_dir / "server_settings.json")
    monkeypatch.setattr(cache, "USERS_DIR", cache_dir / "users")
    monkeypatch.setattr(cache, "LOGOS_DIR", cache_dir / "logos")


class TestInitialize:
    def test_initialize_paths_moves_legacy_cache_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        monkeypatch.delenv("NETV_DATA_DIR", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        (tmp_path / "cache").mkdir()
        (tmp_path / "cache" / "server_settings.json").write_text("{}")
        _point_paths_at(monkeypatch, tmp_path, tmp_path / ".cache")
        cache.initialize_paths()
        state = tmp_path / "xdg" / "netv"
        assert state == cache.CACHE_DIR
        assert state / "server_settings.json" == cache.SERVER_SETTINGS_FILE
        assert (state / "server_settings.json").exists()
        assert not (tmp_path / "cache").exists()

    def test_initialize_paths_puts_installed_state_in_user_data_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        monkeypatch.delenv("NETV_DATA_DIR", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        _point_paths_at(monkeypatch, tmp_path, tmp_path / ".cache")
        cache.initialize_paths()
        state = tmp_path / "xdg" / "netv"
        assert state == cache.CACHE_DIR
        assert state / "server_settings.json" == cache.SERVER_SETTINGS_FILE
        assert state / "users" == cache.USERS_DIR
        assert state / "logos" == cache.LOGOS_DIR

    def test_initialize_paths_keeps_caller_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
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

    def test_initialize_detects_once_into_shared_table(self, monkeypatch: pytest.MonkeyPatch):
        table: dict[str, bool] = {}
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", table)
        monkeypatch.setattr(cache, "VAAPI_DEVICE", None)
        monkeypatch.setattr(cache, "LIBVA_DRIVER", None)
        monkeypatch.setattr(cache, "DRI_PATH", None)
        monkeypatch.setattr(
            cache,
            "_detect_vaapi_device",
            mock.Mock(return_value="/dev/dri/renderD128"),
        )
        monkeypatch.setattr(cache, "_detect_libva_driver", mock.Mock(return_value="iHD"))
        monkeypatch.setattr(cache, "_detect_dri_path", mock.Mock(return_value="/usr/lib/dri"))
        detect = mock.Mock(return_value={"nvenc": True})
        monkeypatch.setattr(cache, "detect_encoders", detect)
        cache.initialize()
        cache.initialize()
        detect.assert_called_once()
        assert table == {"nvenc": True}
        assert cache.VAAPI_DEVICE == "/dev/dri/renderD128"
        assert cache.LIBVA_DRIVER == "iHD"
        assert cache.DRI_PATH == "/usr/lib/dri"

    def test_initialize_keeps_supplied_hardware(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", {})
        monkeypatch.setattr(cache, "VAAPI_DEVICE", "/dev/custom")
        monkeypatch.setattr(cache, "LIBVA_DRIVER", "radeonsi")
        monkeypatch.setattr(cache, "DRI_PATH", "/opt/dri")
        monkeypatch.setattr(cache, "detect_encoders", mock.Mock(return_value={"vaapi": True}))
        cache.initialize()
        assert (cache.VAAPI_DEVICE, cache.LIBVA_DRIVER, cache.DRI_PATH) == (
            "/dev/custom",
            "radeonsi",
            "/opt/dri",
        )

    def test_refresh_encoders_updates_shared_table_in_place(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        table = {"nvenc": True}
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", table)
        monkeypatch.setattr(cache, "detect_encoders", mock.Mock(return_value={"qsv": True}))
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

    @pytest.mark.usefixtures("cache_module")
    def test_save_and_get_cached_logo(self, tmp_path: Path):
        cache.LOGOS_DIR = tmp_path / "logos"
        cache.LOGOS_DIR.mkdir()

        # Save a logo.
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake PNG.
        path = cache.save_logo(
            "TestSource",
            "http://example.com/logo.png",
            data,
            "image/png",
        )
        assert path.exists()
        assert path.suffix == ".png"
        assert path.read_bytes() == data

        # Get cached logo.
        cached = cache.get_cached_logo("TestSource", "http://example.com/logo.png")
        assert cached == path

    @pytest.mark.usefixtures("cache_module")
    def test_get_cached_logo_returns_none_when_missing(self, tmp_path: Path):
        cache.LOGOS_DIR = tmp_path / "logos"
        cache.LOGOS_DIR.mkdir()

        cached = cache.get_cached_logo("NoSource", "http://missing.com/logo.png")
        assert cached is None

    @pytest.mark.usefixtures("cache_module")
    def test_get_cached_logo_expires(self, tmp_path: Path):
        cache.LOGOS_DIR = tmp_path / "logos"
        cache.LOGOS_DIR.mkdir()

        # Save a logo.
        data = b"\x89PNG" + b"\x00" * 100
        path = cache.save_logo("TestSource", "http://example.com/old.png", data, "image/png")

        # Backdate the file.
        old_time = time.time() - 7 * 24 * 3600 - 100

        os.utime(path, (old_time, old_time))

        # Should be expired.
        cached = cache.get_cached_logo("TestSource", "http://example.com/old.png")
        assert cached is None
        assert not path.exists()  # Should be deleted.

    @pytest.mark.usefixtures("cache_module")
    def test_save_logo_content_type_mapping(self, tmp_path: Path):
        cache.LOGOS_DIR = tmp_path / "logos"
        cache.LOGOS_DIR.mkdir()

        data = b"test"
        assert cache.save_logo("s", "http://a.com/1", data, "image/jpeg").suffix == ".jpg"
        assert cache.save_logo("s", "http://a.com/2", data, "image/gif").suffix == ".gif"
        assert cache.save_logo("s", "http://a.com/3", data, "image/webp").suffix == ".webp"
        assert cache.save_logo("s", "http://a.com/4", data, "image/svg+xml").suffix == ".svg"
        assert cache.save_logo("s", "http://a.com/5", data, "unknown/type").suffix == ".png"


class TestFileCacheExtras:
    @pytest.mark.usefixtures("cache_module")
    def test_load_in_subprocess(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cache.futures, "ProcessPoolExecutor", futures.ThreadPoolExecutor)
        cache.save_file_cache("proc", [1, 2])
        loaded = cache.load_file_cache("proc", use_process=True)
        assert loaded is not None
        data, ts = loaded
        assert data == [1, 2]
        assert ts > 0

    def test_parse_json_file(self, tmp_path: Path):
        good = tmp_path / "good.json"
        good.write_text(json.dumps({"data": {"a": 1}}))
        assert cache._parse_json_file(str(good)) == ({"a": 1}, 0)
        bad = tmp_path / "bad.json"
        bad.write_text("{")
        assert cache._parse_json_file(str(bad)) is None
        assert cache._parse_json_file(str(tmp_path / "missing.json")) is None

    @pytest.mark.usefixtures("cache_module")
    def test_clear_all_file_caches(self):
        cache.save_file_cache("live_data", [1])
        cache.save_file_cache("vod_data", [2])
        cache.save_file_cache("other", [3])
        cache.get_cache()["live_streams"] = []
        assert cache.clear_all_file_caches() == 2
        assert cache.load_file_cache("live_data") is None
        assert cache.load_file_cache("other") is not None
        assert cache.get_cache() == {}

    def test_get_cache_lock(self):
        assert cache.get_cache_lock() is cache._cache_lock


def _backdate(key: str, data: object, age: float) -> None:
    path = cache.CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({"data": data, "timestamp": time.time() - age}))


@pytest.mark.usefixtures("cache_module")
class TestStaleWhileRevalidate:
    def test_memory_hit_with_stale_file_refreshes_in_background(self):
        _backdate("k", {"v": "file"}, 2 * 24 * 3600)
        cache._info_cache["k"] = {"v": "mem"}
        fetch = mock.Mock()
        with mock.patch.object(cache.threading, "Thread") as thread:
            assert cache.get_cached_info("k", fetch) == {"v": "mem"}
        assert thread.call_args.kwargs["args"] == ("k", fetch)
        thread.return_value.start.assert_called_once()
        fetch.assert_not_called()

    def test_stale_file_served_and_refreshed(self):
        _backdate("k", {"v": "file"}, 2 * 24 * 3600)
        fetch = mock.Mock()
        with mock.patch.object(cache.threading, "Thread") as thread:
            assert cache.get_cached_info("k", fetch) == {"v": "file"}
        assert cache._info_cache["k"] == {"v": "file"}
        thread.return_value.start.assert_called_once()

    def test_fresh_file_served_without_refresh(self):
        _backdate("k", {"v": "file"}, 60)
        with mock.patch.object(cache.threading, "Thread") as thread:
            assert cache.get_cached_info("k", mock.Mock()) == {"v": "file"}
        thread.assert_not_called()

    def test_force_refetches_despite_file(self):
        _backdate("k", {"v": "file"}, 60)
        cache._info_cache["k"] = {"v": "mem"}
        assert cache.get_cached_info("k", mock.Mock(return_value={"v": "new"}), force=True) == {
            "v": "new",
        }
        loaded = cache.load_file_cache("k")
        assert loaded is not None
        assert loaded[0] == {"v": "new"}

    def test_refresh_success_and_failure(self, caplog: pytest.LogCaptureFixture):
        caplog.set_level("INFO", logger=cache.log.name)
        cache._refresh_cached_info("k", mock.Mock(return_value={"v": 9}))
        assert cache._info_cache["k"] == {"v": 9}
        loaded = cache.load_file_cache("k")
        assert loaded is not None
        assert loaded[0] == {"v": 9}
        cache._refresh_cached_info("k", mock.Mock(side_effect=RuntimeError("boom")))
        assert cache._info_cache["k"] == {"v": 9}
        assert "Background refresh failed for k: boom" in caplog.text


@pytest.mark.usefixtures("cache_module")
class TestServerSettingsExtras:
    @pytest.mark.parametrize(
        ("old", "encoders", "new"),
        [
            ("nvidia", {"vaapi": True}, "nvenc+vaapi"),
            ("nvidia", {"vaapi": False}, "nvenc+software"),
            ("intel", {"vaapi": False}, "qsv"),
            ("vaapi", {"vaapi": False}, "vaapi"),
            ("bogus", {"vaapi": False}, "software"),
        ],
    )
    def test_legacy_transcode_hw_migrated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        old: str,
        encoders: dict[str, bool],
        new: str,
    ):
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", encoders)
        _write_settings({"transcode_hw": old})
        assert cache.load_server_settings()["transcode_hw"] == new

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
    def test_user_agent(self, settings: dict[str, object], expected: str | None):
        _write_settings(settings)
        assert cache.get_user_agent() == expected

    def test_hidden_source_ids(self):
        _write_settings(
            {
                "sources": [
                    _source("all"),
                    _source("mine", users=["alice"]),
                    _source("theirs", users=["bob"]),
                ],
            },
        )
        assert cache.hidden_source_ids("alice", admin=True) == set()
        assert cache.hidden_source_ids("alice", admin=False) == {"theirs"}

    def test_update_source_epg_url_unknown_source(self):
        _write_settings({"sources": [_source("s1")]})
        cache.update_source_epg_url("s2", "http://epg")
        assert "epg_url" not in cache.load_server_settings()["sources"][0]


@pytest.mark.usefixtures("cache_module")
class TestUserSettingsExtras:
    def test_missing_watch_position(self):
        assert cache.get_watch_position("user1", "http://none") is None

    def test_watch_position_without_duration_is_kept(self):
        cache.save_watch_position("user1", "http://live", 50.0, 0)
        entry = cache.get_watch_position("user1", "http://live")
        assert entry is not None
        assert entry["position"] == 50.0

    def test_watch_history_keeps_200_most_recent(self):
        history: dict[str, cache.WatchEntry] = {
            f"u{i}": {"position": 1, "duration": 10, "updated": i} for i in range(200)
        }
        settings = cache.load_user_settings("user1")
        settings["watch_history"] = history
        cache.save_user_settings("user1", settings)
        cache.save_watch_position("user1", "new", 1.0, 10.0)
        saved = cache.load_user_settings("user1")["watch_history"]
        assert len(saved) == 200
        assert "new" in saved
        assert "u0" not in saved
        assert "u1" in saved

    @pytest.mark.parametrize("username", ["", "a" * 65, "a..b", "a/b", "a\\b"])
    def test_invalid_usernames_rejected(self, username: str):
        with pytest.raises(ValueError, match="Invalid username"):
            cache.load_user_settings(username)

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
    def test_invalid_live_view_counts_rejected(self, payload: object):
        path = cache.USERS_DIR / "user1" / "live_views.json"
        path.parent.mkdir()
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="Invalid live view counts"):
            cache.load_live_view_counts("user1")


def _rooted(root: Path, path: str | Path) -> Path:
    return root / str(path).lstrip("/")


@pytest.fixture
def fake_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect cache's absolute system paths under tmp_path."""
    monkeypatch.setattr(
        cache,
        "pathlib",
        SimpleNamespace(Path=functools.partial(_rooted, tmp_path)),
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
    @pytest.mark.usefixtures("fake_root")
    def test_lspci_vendor(self, stdout: str, vendor: str):
        with mock.patch.object(
            cache.subprocess,
            "run",
            return_value=mock.Mock(stdout=("00:1f.0 ISA bridge [8086:7a84]\n" + stdout)),
        ):
            assert cache._get_gpu_vendor() == vendor

    def test_sysfs_fallback(self, fake_root: Path):
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

    def test_unknown_gpu(self, fake_root: Path):
        (fake_root / "sys/class/drm/card0/device").mkdir(parents=True)
        (fake_root / "sys/class/drm/card0/device/vendor").write_text("0x10de\n")
        with mock.patch.object(
            cache.subprocess,
            "run",
            return_value=mock.Mock(stdout="01:00.0 3D controller [0302]: NVIDIA [10de:2684]\n"),
        ):
            assert cache._get_gpu_vendor() is None

    @pytest.mark.usefixtures("fake_root")
    def test_no_sysfs(self):
        with mock.patch.object(cache.subprocess, "run", return_value=mock.Mock(stdout="")):
            assert cache._get_gpu_vendor() is None

    def test_unreadable_sysfs(self, fake_root: Path):
        (fake_root / "sys/class").mkdir(parents=True)
        (fake_root / "sys/class/drm").write_text("")  # iterdir() raises NotADirectoryError.
        with mock.patch.object(cache.subprocess, "run", return_value=mock.Mock(stdout="")):
            assert cache._get_gpu_vendor() is None

    def test_vaapi_device(self, fake_root: Path):
        assert cache._detect_vaapi_device() is None
        (fake_root / "dev/dri").mkdir(parents=True)
        (fake_root / "dev/dri/renderD128").touch()
        assert cache._detect_vaapi_device() == str(fake_root / "dev/dri/renderD128")

    def test_dri_path(self, fake_root: Path):
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
    def test_libva_driver(
        self,
        fake_root: Path,
        monkeypatch: pytest.MonkeyPatch,
        vendor: str | None,
        ihd: bool,
        driver: str | None,
    ):
        monkeypatch.setattr(cache, "_get_gpu_vendor", mock.Mock(return_value=vendor))
        dri = fake_root / "usr/lib/dri"
        dri.mkdir(parents=True)
        if ihd:
            (dri / "iHD_drv_video.so").touch()
        assert cache._detect_libva_driver() == driver


class TestEncoderExtras:
    def test_vaapi_baseline_fallback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cache, "_test_encoder", _encoder_result("constrained_baseline"))
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
