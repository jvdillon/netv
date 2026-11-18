"""Tests for cache.py."""

from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import functools
import json
import os
import subprocess
import time

import pytest

from netv import cache, testing
from netv.xtream import ProviderRecord


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


@dataclass(kw_only=True, slots=True)
class _Fetch:
    """A ``get_cached_info`` fetch callback that counts its calls."""

    data: ProviderRecord
    calls: int = 0

    def __call__(self) -> ProviderRecord:
        self.calls += 1
        return self.data


def _failing_fetch() -> ProviderRecord:
    raise RuntimeError("boom")


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> testing.FakeRun:
    run = testing.FakeRun()
    testing.install_subprocess(monkeypatch, cache, run=run)
    return run


@pytest.fixture
def threads(monkeypatch: pytest.MonkeyPatch) -> testing.ThreadRecorder:
    recorder = testing.ThreadRecorder(run_inline=False)
    testing.install_threads(monkeypatch, recorder, cache)
    return recorder


@pytest.fixture
def vaapi_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, "VAAPI_DEVICE", "/dev/dri/renderD128")
    monkeypatch.setattr(cache, "LIBVA_DRIVER", "i965")
    monkeypatch.setattr(cache, "DRI_PATH", "/usr/lib/x86_64-linux-gnu/dri")


@pytest.fixture
def no_vaapi_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, "VAAPI_DEVICE", None)


def _ffmpeg_accepts(run: testing.FakeRun, encoder: str) -> None:
    """Script ffmpeg to fail every probe except those naming ``encoder``."""
    run.program("ffmpeg", testing.Program(returncode=1, stderr=b"not available"))
    run.program("ffmpeg", testing.Program(), when=encoder)


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


class TestRelayHints:
    """What live sessions learned about upstreams, kept across restarts."""

    def test_hints_start_empty_then_persist(self):
        assert cache.relay_hints() == {"live_ts": {}, "recurring_gaps": {}}
        cache.remember_live_ts("x", True)
        cache.remember_live_ts("y", False)
        cache.remember_burst_gap("panel", 9.5, 5)
        cache.remember_burst_gap("panel", 4.0, 29.9)
        cache.remember_burst_gap("tuner", 0.2, 5)
        # A short session never lowers the reserve a host has needed.
        assert cache.relay_hints() == {
            "live_ts": {"x": True, "y": False},
            "recurring_gaps": {"panel": 9.5, "tuner": 0.2},
        }

    def test_hints_from_older_releases_keep_live_ts(self):
        (cache.CACHE_DIR / "relay_hints.json").write_text(
            '{"data": {"live_ts": {"x": true}, "burst_gaps": {"tuner": 1.08}}, "timestamp": 0}',
        )
        assert cache.relay_hints() == {"live_ts": {"x": True}, "recurring_gaps": {}}

    def test_a_settled_session_replaces_the_burst_gap(self):
        cache.remember_burst_gap("tuner", 1.08, 5)
        cache.remember_burst_gap("tuner", 0.08, 30)
        assert cache.relay_hints()["recurring_gaps"] == {"tuner": 0.08}

    def test_unreadable_hints_start_over(self):
        (cache.CACHE_DIR / "relay_hints.json").write_text(
            '{"data": {"live_ts": 3}, "timestamp": 0}',
        )
        assert cache.relay_hints() == {"live_ts": {}, "recurring_gaps": {}}


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


class TestCachedInfo:
    def test_get_cached_info_calls_fetch(self):
        fetch = _Fetch(data={"result": 42})
        result = cache.get_cached_info("test_key", fetch)
        assert result == {"result": 42}
        assert fetch.calls == 1

    def test_get_cached_info_uses_memory_cache(self):
        fetch = _Fetch(data={"result": 1})
        cache.get_cached_info("key1", fetch)
        cache.get_cached_info("key1", fetch)
        # Only called once - second call uses memory cache.
        assert fetch.calls == 1

    def test_get_cached_info_force_bypasses_memory(self):
        fetch = _Fetch(data={"result": 1})
        cache.get_cached_info("key2", fetch)
        cache.get_cached_info("key2", fetch, force=True)
        assert fetch.calls == 2

    def test_non_object_file_cache_is_refetched(self):
        cache.save_file_cache("key3", [1, 2])
        assert cache.get_cached_info("key3", lambda: {"fresh": True}) == {"fresh": True}


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


class TestEncoderDetection:
    """Tests for encoder detection functions."""

    def test_test_encoder_success(self, run: testing.FakeRun):
        """Test _test_encoder returns (True, '') on successful command."""
        run.program("echo", testing.Program())
        ok, err = cache._test_encoder(["echo", "test"])
        assert ok is True
        assert err == ""
        assert run.calls == [["echo", "test"]]
        assert run.options == [{"capture_output": True, "timeout": 5, "env": None}]

    def test_test_encoder_failure(self, run: testing.FakeRun):
        """Test _test_encoder returns (False, error) on non-zero return code."""
        run.program("false", testing.Program(returncode=1, stderr=b"encoder not found"))
        ok, err = cache._test_encoder(["false"])
        assert ok is False
        assert err == "encoder not found"

    def test_test_encoder_timeout(self, run: testing.FakeRun):
        """Test _test_encoder returns (False, 'timeout') on timeout."""
        run.program(
            "sleep",
            testing.Program(error=subprocess.TimeoutExpired(cmd=["test"], timeout=5)),
        )
        ok, err = cache._test_encoder(["sleep", "100"], timeout=5)
        assert ok is False
        assert err == "timeout"

    def test_test_encoder_exception(self, run: testing.FakeRun):
        """Test _test_encoder reports a missing binary."""
        ok, err = cache._test_encoder(["nonexistent_command"])
        assert ok is False
        assert err == "ffmpeg not found"
        assert run.calls == [["nonexistent_command"]]

    @pytest.mark.usefixtures("vaapi_gpu")
    def test_detect_encoders_all_available(self, run: testing.FakeRun):
        """Test detect_encoders when all hardware is available."""
        run.program("ffmpeg", testing.Program())
        assert cache.detect_encoders() == {
            "nvenc": True,
            "amf": True,
            "qsv": True,
            "vaapi": True,
            "vaapi_baseline_only": False,
        }

    @pytest.mark.usefixtures("no_vaapi_gpu")
    def test_detect_encoders_none_available(self, run: testing.FakeRun):
        """Test detect_encoders when no hardware is available."""
        run.program("ffmpeg", testing.Program(returncode=1, stderr=b"not found"))
        assert cache.detect_encoders() == {
            "nvenc": False,
            "amf": False,
            "qsv": False,
            "vaapi": False,
        }
        assert len(run.calls) == 3

    @pytest.mark.usefixtures("vaapi_gpu")
    def test_detect_encoders_partial(self, run: testing.FakeRun):
        """Test detect_encoders with mixed hardware availability."""
        _ffmpeg_accepts(run, "h264_vaapi")
        result = cache.detect_encoders()
        assert result["nvenc"] is False
        assert result["amf"] is False
        assert result["qsv"] is False
        assert result["vaapi"] is True
        assert result["vaapi_baseline_only"] is False

    @pytest.mark.usefixtures("no_vaapi_gpu")
    def test_detect_encoders_nvenc_only(self, run: testing.FakeRun):
        """Test detect_encoders when only NVENC is available."""
        _ffmpeg_accepts(run, "h264_nvenc")
        result = cache.detect_encoders()
        assert result["nvenc"] is True
        assert result["amf"] is False
        assert result["qsv"] is False
        assert result["vaapi"] is False

    @pytest.mark.usefixtures("vaapi_gpu")
    def test_detect_encoders_vaapi_command_structure(self, run: testing.FakeRun):
        """Test detect_encoders passes correct VAAPI command structure when GPU detected."""
        _ffmpeg_accepts(run, "never")
        cache.detect_encoders()

        # Two VAAPI probes: High profile first, then the baseline fallback.
        vaapi = [(c, o) for c, o in zip(run.calls, run.options, strict=True) if "h264_vaapi" in c]
        assert len(vaapi) == 2
        (high, high_options), (baseline, _) = vaapi
        assert "-init_hw_device" in high
        assert "vaapi=va:/dev/dri/renderD128" in high
        assert "hwupload" in " ".join(high)
        assert "constrained_baseline" not in " ".join(high)
        assert "constrained_baseline" in " ".join(baseline)
        env = high_options["env"]
        assert isinstance(env, dict)
        assert (env["LIBVA_DRIVER_NAME"], env["LIBVA_DRIVERS_PATH"]) == (
            "i965",
            "/usr/lib/x86_64-linux-gnu/dri",
        )

    @pytest.mark.usefixtures("no_vaapi_gpu")
    def test_detect_encoders_qsv_command_structure(self, run: testing.FakeRun):
        """Test detect_encoders passes correct QSV command structure."""
        _ffmpeg_accepts(run, "never")
        cache.detect_encoders()

        qsv_cmd = next(c for c in run.calls if "h264_qsv" in c)
        assert "-hwaccel" in qsv_cmd
        assert "qsv" in qsv_cmd
        assert "-hwaccel_output_format" in qsv_cmd

    @pytest.mark.usefixtures("no_vaapi_gpu")
    def test_refresh_encoders_updates_global(
        self,
        monkeypatch: pytest.MonkeyPatch,
        run: testing.FakeRun,
    ):
        """Test refresh_encoders updates AVAILABLE_ENCODERS."""
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", {})
        run.program("ffmpeg", testing.Program())
        result = cache.refresh_encoders()
        assert cache.AVAILABLE_ENCODERS == {"nvenc": True, "amf": True, "qsv": True, "vaapi": False}
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

    def test_initialize_detects_once_into_shared_table(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_root: Path,
        run: testing.FakeRun,
    ):
        table: dict[str, bool] = {}
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", table)
        monkeypatch.setattr(cache, "VAAPI_DEVICE", None)
        monkeypatch.setattr(cache, "LIBVA_DRIVER", None)
        monkeypatch.setattr(cache, "DRI_PATH", None)
        (fake_root / "dev/dri").mkdir(parents=True)
        (fake_root / "dev/dri/renderD128").touch()
        (fake_root / "usr/lib/dri").mkdir(parents=True)
        (fake_root / "usr/lib/dri/iHD_drv_video.so").touch()
        run.program("lspci", testing.Program(stdout="00:02.0 VGA [0300]: Intel [8086:46a6]\n"))
        _ffmpeg_accepts(run, "h264_nvenc")
        cache.initialize()
        probes = len(run.calls)
        cache.initialize()
        assert len(run.calls) == probes
        assert table == {
            "nvenc": True,
            "amf": False,
            "qsv": False,
            "vaapi": False,
            "vaapi_baseline_only": False,
        }
        assert str(fake_root / "dev/dri/renderD128") == cache.VAAPI_DEVICE
        assert cache.LIBVA_DRIVER == "iHD"
        assert cache.DRI_PATH == "/usr/lib/dri"

    def test_initialize_keeps_supplied_hardware(
        self,
        monkeypatch: pytest.MonkeyPatch,
        run: testing.FakeRun,
    ):
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", {})
        monkeypatch.setattr(cache, "VAAPI_DEVICE", "/dev/custom")
        monkeypatch.setattr(cache, "LIBVA_DRIVER", "radeonsi")
        monkeypatch.setattr(cache, "DRI_PATH", "/opt/dri")
        _ffmpeg_accepts(run, "h264_vaapi")
        cache.initialize()
        assert (cache.VAAPI_DEVICE, cache.LIBVA_DRIVER, cache.DRI_PATH) == (
            "/dev/custom",
            "radeonsi",
            "/opt/dri",
        )
        assert cache.AVAILABLE_ENCODERS["vaapi"] is True
        assert "vaapi=va:/dev/custom" in next(c for c in run.calls if "h264_vaapi" in c)
        assert "lspci" not in [c[0] for c in run.calls]

    @pytest.mark.usefixtures("no_vaapi_gpu")
    def test_refresh_encoders_updates_shared_table_in_place(
        self,
        monkeypatch: pytest.MonkeyPatch,
        run: testing.FakeRun,
    ):
        table = {"nvenc": True}
        monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", table)
        _ffmpeg_accepts(run, "h264_qsv")
        assert cache.refresh_encoders() is table
        assert table == {"nvenc": False, "amf": False, "qsv": True, "vaapi": False}


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

    def test_save_and_get_cached_logo(self):
        # Save a logo.
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake PNG.
        path = cache.save_logo(
            "TestSource",
            "http://example.com/logo.png",
            data,
            "image/png",
        )
        assert path.exists()
        assert path.parent == cache.LOGOS_DIR / "TestSource"
        assert path.suffix == ".png"
        assert path.read_bytes() == data

        # Get cached logo.
        cached = cache.get_cached_logo("TestSource", "http://example.com/logo.png")
        assert cached == path

    def test_get_cached_logo_returns_none_when_missing(self):
        cache.LOGOS_DIR.mkdir()
        cached = cache.get_cached_logo("NoSource", "http://missing.com/logo.png")
        assert cached is None
        cache.save_logo("NoSource", "http://other.com/logo.png", b"x", "image/png")
        assert cache.get_cached_logo("NoSource", "http://missing.com/logo.png") is None

    def test_get_cached_logo_expires(self):
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

    def test_save_logo_content_type_mapping(self):
        data = b"test"
        assert cache.save_logo("s", "http://a.com/1", data, "image/jpeg").suffix == ".jpg"
        assert cache.save_logo("s", "http://a.com/2", data, "image/gif").suffix == ".gif"
        assert cache.save_logo("s", "http://a.com/3", data, "image/webp").suffix == ".webp"
        assert cache.save_logo("s", "http://a.com/4", data, "image/svg+xml").suffix == ".svg"
        assert cache.save_logo("s", "http://a.com/5", data, "unknown/type").suffix == ".png"


class TestFileCacheExtras:
    def test_load_in_subprocess(self, monkeypatch: pytest.MonkeyPatch):
        # A real worker process costs ~100ms; a thread pool runs the same parse.
        monkeypatch.setattr(
            cache,
            "futures",
            testing.ModuleView(futures, {"ProcessPoolExecutor": futures.ThreadPoolExecutor}),
        )
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

    def test_clear_all_file_caches(self):
        cache.save_file_cache("live_data", [1])
        cache.save_file_cache("vod_data", [2])
        cache.save_file_cache("series_data", [4])
        cache.save_file_cache("other", [3])
        cache.get_cache()["live_streams"] = []
        assert cache.clear_all_file_caches() == 3
        assert cache.load_file_cache("live_data") is None
        assert cache.load_file_cache("vod_data") is None
        assert cache.load_file_cache("series_data") is None
        assert cache.load_file_cache("other") is not None
        assert cache.get_cache() == {}

    def test_get_cache_lock(self):
        assert cache.get_cache_lock() is cache._cache_lock


def _backdate(key: str, data: object, age: float) -> None:
    path = cache.CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({"data": data, "timestamp": time.time() - age}))


class TestStaleWhileRevalidate:
    def test_memory_hit_with_stale_file_refreshes_in_background(
        self,
        threads: testing.ThreadRecorder,
    ):
        _backdate("k", {"v": "file"}, 2 * 24 * 3600)
        cache._info_cache["k"] = {"v": "mem"}
        fetch = _Fetch(data={"v": "new"})
        assert cache.get_cached_info("k", fetch) == {"v": "mem"}
        assert [(t.args, t.daemon, t.started) for t in threads.threads] == [
            (("k", fetch), True, True),
        ]
        assert fetch.calls == 0
        threads.run_all()
        assert (fetch.calls, cache._info_cache["k"]) == (1, {"v": "new"})

    def test_stale_file_served_and_refreshed(self, threads: testing.ThreadRecorder):
        _backdate("k", {"v": "file"}, 2 * 24 * 3600)
        fetch = _Fetch(data={"v": "new"})
        assert cache.get_cached_info("k", fetch) == {"v": "file"}
        assert cache._info_cache["k"] == {"v": "file"}
        assert [(t.args, t.daemon, t.started) for t in threads.threads] == [
            (("k", fetch), True, True),
        ]
        threads.run_all()
        loaded = cache.load_file_cache("k")
        assert loaded is not None
        assert loaded[0] == {"v": "new"}

    def test_fresh_file_served_without_refresh(self, threads: testing.ThreadRecorder):
        _backdate("k", {"v": "file"}, 60)
        fetch = _Fetch(data={"v": "new"})
        assert cache.get_cached_info("k", fetch) == {"v": "file"}
        assert threads.threads == []
        assert fetch.calls == 0

    def test_force_refetches_despite_file(self):
        _backdate("k", {"v": "file"}, 60)
        cache._info_cache["k"] = {"v": "mem"}
        assert cache.get_cached_info("k", _Fetch(data={"v": "new"}), force=True) == {"v": "new"}
        loaded = cache.load_file_cache("k")
        assert loaded is not None
        assert loaded[0] == {"v": "new"}

    def test_refresh_success_and_failure(self, caplog: pytest.LogCaptureFixture):
        caplog.set_level("INFO", logger=cache.log.name)
        cache._refresh_cached_info("k", _Fetch(data={"v": 9}))
        assert cache._info_cache["k"] == {"v": 9}
        loaded = cache.load_file_cache("k")
        assert loaded is not None
        assert loaded[0] == {"v": 9}
        cache._refresh_cached_info("k", _failing_fetch)
        assert cache._info_cache["k"] == {"v": 9}
        assert "Background refresh failed for k: boom" in caplog.text


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


def _lspci(run: testing.FakeRun, stdout: str) -> None:
    run.program("lspci", testing.Program(stdout=stdout))


class TestGpuDetection:
    @pytest.mark.parametrize(
        ("stdout", "vendor"),
        [
            ("00:02.0 VGA compatible controller [0300]: Intel [8086:46a6]\n", "8086"),
            ("03:00.0 Display controller [0380]: AMD [1002:73ff]\n", "1002"),
        ],
    )
    @pytest.mark.usefixtures("fake_root")
    def test_lspci_vendor(self, run: testing.FakeRun, stdout: str, vendor: str):
        _lspci(run, "00:1f.0 ISA bridge [8086:7a84]\n" + stdout)
        assert cache._get_gpu_vendor() == vendor
        assert run.calls == [["lspci", "-nn"]]
        assert run.options == [{"capture_output": True, "text": True, "timeout": 5}]

    @pytest.mark.usefixtures("run")
    def test_sysfs_fallback(self, fake_root: Path):
        drm = fake_root / "sys/class/drm"
        for name, vendor in [("card0", "0x10de"), ("card1", "0x1002")]:
            (drm / name / "device").mkdir(parents=True)
            (drm / name / "device" / "vendor").write_text(vendor + "\n")
        (drm / "card2").mkdir()  # No vendor file.
        (drm / "card1-HDMI-A-1" / "device").mkdir(parents=True)
        (drm / "card1-HDMI-A-1" / "device" / "vendor").write_text("0x8086\n")
        (drm / "renderD128").mkdir()
        # Unscripted lspci raises FileNotFoundError, as in a container.
        assert cache._get_gpu_vendor() == "1002"

    def test_unknown_gpu(self, fake_root: Path, run: testing.FakeRun):
        (fake_root / "sys/class/drm/card0/device").mkdir(parents=True)
        (fake_root / "sys/class/drm/card0/device/vendor").write_text("0x10de\n")
        _lspci(run, "01:00.0 3D controller [0302]: NVIDIA [10de:2684]\n")
        assert cache._get_gpu_vendor() is None

    @pytest.mark.usefixtures("fake_root")
    def test_no_sysfs(self, run: testing.FakeRun):
        _lspci(run, "")
        assert cache._get_gpu_vendor() is None

    def test_unreadable_sysfs(self, fake_root: Path, run: testing.FakeRun):
        (fake_root / "sys/class").mkdir(parents=True)
        (fake_root / "sys/class/drm").write_text("")  # iterdir() raises NotADirectoryError.
        _lspci(run, "")
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
        ("lspci", "ihd", "driver"),
        [
            ("00:02.0 VGA [0300]: Intel [8086:46a6]\n", True, "iHD"),
            ("00:02.0 VGA [0300]: Intel [8086:46a6]\n", False, "i965"),
            ("03:00.0 VGA [0300]: AMD [1002:73ff]\n", False, "radeonsi"),
            ("", False, None),
        ],
    )
    def test_libva_driver(
        self,
        fake_root: Path,
        run: testing.FakeRun,
        lspci: str,
        ihd: bool,
        driver: str | None,
    ):
        _lspci(run, lspci)
        dri = fake_root / "usr/lib/dri"
        dri.mkdir(parents=True)
        if ihd:
            (dri / "iHD_drv_video.so").touch()
        assert cache._detect_libva_driver() == driver


class TestEncoderExtras:
    def test_vaapi_baseline_fallback(self, monkeypatch: pytest.MonkeyPatch, run: testing.FakeRun):
        _ffmpeg_accepts(run, "constrained_baseline")
        monkeypatch.setattr(cache, "VAAPI_DEVICE", "/dev/dri/renderD128")
        monkeypatch.setattr(cache, "LIBVA_DRIVER", "radeonsi")
        monkeypatch.setattr(cache, "DRI_PATH", "/usr/lib/dri")
        result = cache.detect_encoders()
        assert result["vaapi"] is True
        assert result["vaapi_baseline_only"] is True

    def test_test_encoder_passes_env_and_picks_error_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
        run: testing.FakeRun,
    ):
        monkeypatch.setenv("NETV_TEST_INHERITED", "yes")
        stderr = b"[h264 @ 0x1] noisy\nNo device available\n"
        run.program("ffmpeg", testing.Program(returncode=1, stderr=stderr))
        assert cache._test_encoder(["ffmpeg"], env={"LIBVA_DRIVER_NAME": "iHD"}) == (
            False,
            "No device available",
        )
        env = run.options[0]["env"]
        assert isinstance(env, dict)
        assert (env["LIBVA_DRIVER_NAME"], env["NETV_TEST_INHERITED"]) == ("iHD", "yes")

    def test_test_encoder_bracket_only_and_empty_stderr(self, run: testing.FakeRun):
        run.program("ffmpeg", testing.Program(returncode=1, stderr=b"[x] y"))
        assert cache._test_encoder(["ffmpeg"]) == (False, "[x] y")
        run.program("ffmpeg", testing.Program(returncode=1, stderr=b""))
        assert cache._test_encoder(["ffmpeg"]) == (False, "unknown error")

    def test_test_encoder_unexpected_error(self, run: testing.FakeRun):
        run.program("ffmpeg", testing.Program(error=PermissionError("denied")))
        assert cache._test_encoder(["ffmpeg"]) == (False, "denied")


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
