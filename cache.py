"""File cache, settings, sources management."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import json
import logging
import pathlib
import subprocess
import threading
import time


log = logging.getLogger(__name__)

APP_DIR = pathlib.Path(__file__).parent
CACHE_DIR = APP_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
SERVER_SETTINGS_FILE = CACHE_DIR / "server_settings.json"
USERS_DIR = CACHE_DIR / "users"
USERS_DIR.mkdir(exist_ok=True)

# Cache TTLs in seconds
LIVE_CACHE_TTL = 2 * 3600  # 2 hours
EPG_CACHE_TTL = 6 * 3600  # 6 hours
VOD_CACHE_TTL = 12 * 3600  # 12 hours
SERIES_CACHE_TTL = 12 * 3600  # 12 hours
INFO_CACHE_TTL = 7 * 24 * 3600  # 7 days max for series/movie info
INFO_CACHE_STALE = 24 * 3600  # Refresh in background after 24 hours

# In-memory cache
_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()


def _parse_json_file(path: str) -> tuple[Any, float] | None:
    """Parse JSON file - runs in separate process to avoid GIL blocking."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("data"), data.get("timestamp", 0)
    except Exception:
        return None


def load_file_cache(name: str, use_process: bool = False) -> tuple[Any, float] | None:
    """Load cached data from file. Returns (data, timestamp) or None.

    Args:
        name: Cache file name (without .json extension)
        use_process: If True, parse in separate process to avoid GIL blocking
    """
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        return None
    if use_process:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_parse_json_file, str(path))
            return future.result(timeout=60)
    try:
        data = json.loads(path.read_text())
        return data.get("data"), data.get("timestamp", 0)
    except Exception:
        return None


def save_file_cache(name: str, data: Any) -> None:
    """Save data to cache file with current timestamp."""
    path = CACHE_DIR / f"{name}.json"
    path.write_text(json.dumps({"data": data, "timestamp": time.time()}))


def clear_all_caches() -> None:
    """Clear memory cache except EPG (file cache preserved for restart)."""
    with _cache_lock:
        epg = _cache.get("epg")
        _cache.clear()
        if epg:
            _cache["epg"] = epg


def get_cache() -> dict[str, Any]:
    """Get reference to memory cache."""
    return _cache


def get_cache_lock() -> threading.Lock:
    """Get cache lock."""
    return _cache_lock


def get_cached_info(cache_key: str, fetch_fn: Callable[[], Any], force: bool = False) -> Any:
    """Get info from memory cache, file cache, or fetch. Stale-while-revalidate."""
    cached = load_file_cache(cache_key)
    cached_data, cached_ts = cached if cached else (None, 0)
    age = time.time() - cached_ts

    if force and cached_data:
        _cache.pop(cache_key, None)
        cached_data = None

    if cache_key in _cache and not force:
        if cached_ts and age > INFO_CACHE_STALE:

            def bg_refresh() -> None:
                try:
                    data = fetch_fn()
                    _cache[cache_key] = data
                    save_file_cache(cache_key, data)
                    log.info("Background refreshed %s", cache_key)
                except Exception as e:
                    log.warning("Background refresh failed for %s: %s", cache_key, e)

            threading.Thread(target=bg_refresh, daemon=True).start()
        return _cache[cache_key]

    if cached_data and age < INFO_CACHE_TTL:
        _cache[cache_key] = cached_data
        if age > INFO_CACHE_STALE:

            def bg_refresh() -> None:
                try:
                    data = fetch_fn()
                    _cache[cache_key] = data
                    save_file_cache(cache_key, data)
                    log.info("Background refreshed %s", cache_key)
                except Exception as e:
                    log.warning("Background refresh failed for %s: %s", cache_key, e)

            threading.Thread(target=bg_refresh, daemon=True).start()
        return cached_data

    data = fetch_fn()
    _cache[cache_key] = data
    save_file_cache(cache_key, data)
    return data


def _test_encoder(cmd: list[str], timeout: int = 5) -> tuple[bool, str]:
    """Test if an encoder works. Returns (success, error_message)."""
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode == 0:
            return True, ""
        stderr = result.stderr.decode(errors="replace").strip()
        # Extract the most relevant error line
        for line in stderr.split("\n"):
            if line and not line.startswith("["):
                return False, line
        return False, stderr if stderr else "unknown error"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "ffmpeg not found"
    except Exception as e:
        return False, str(e)


def detect_encoders() -> dict[str, bool]:
    """Detect available FFmpeg H.264 encoders by testing actual hardware."""
    log.info("Detecting hardware encoders...")
    encoders = {
        "nvidia": False,
        "intel": False,
        "vaapi": False,
        "software": False,
    }

    # Test input: 1 frame of 64x64 black
    test_input = ["-f", "lavfi", "-i", "color=black:s=64x64:d=0.04", "-frames:v", "1"]
    base_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    null_out = ["-f", "null", "-"]

    # NVIDIA: try nvenc directly
    ok, err = _test_encoder(base_cmd + test_input + ["-c:v", "h264_nvenc"] + null_out)
    encoders["nvidia"] = ok
    if ok:
        log.info("  NVIDIA (h264_nvenc): available")
    else:
        log.info("  NVIDIA (h264_nvenc): unavailable - %s", err)

    # Intel QSV: needs hwaccel init
    ok, err = _test_encoder(
        base_cmd
        + ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
        + test_input
        + ["-c:v", "h264_qsv"]
        + null_out
    )
    encoders["intel"] = ok
    if ok:
        log.info("  Intel (h264_qsv): available")
    else:
        log.info("  Intel (h264_qsv): unavailable - %s", err)

    # VA-API: needs device and hwupload
    ok, err = _test_encoder(
        base_cmd
        + ["-vaapi_device", "/dev/dri/renderD128"]
        + test_input
        + ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi"]
        + null_out
    )
    encoders["vaapi"] = ok
    if ok:
        log.info("  VA-API (h264_vaapi): available")
    else:
        log.info("  VA-API (h264_vaapi): unavailable - %s", err)

    # Software: libx264
    ok, err = _test_encoder(
        base_cmd + test_input + ["-c:v", "libx264", "-preset", "ultrafast"] + null_out
    )
    encoders["software"] = ok
    if ok:
        log.info("  Software (libx264): available")
    else:
        log.info("  Software (libx264): unavailable - %s", err)

    return encoders


AVAILABLE_ENCODERS = detect_encoders()


def refresh_encoders() -> dict[str, bool]:
    """Re-detect available encoders and update the cache."""
    global AVAILABLE_ENCODERS
    AVAILABLE_ENCODERS = detect_encoders()
    return AVAILABLE_ENCODERS


def _default_encoder() -> str:
    """Return first available encoder, preferring most specific."""
    for enc in ("nvidia", "intel", "vaapi", "software"):
        if AVAILABLE_ENCODERS.get(enc):
            return enc
    return "software"


@dataclass(slots=True)
class Source:
    id: str
    name: str
    type: str  # "xtream", "m3u", or "epg"
    url: str
    username: str = ""
    password: str = ""
    epg_timeout: int = 120  # seconds
    epg_schedule: list[str] = field(default_factory=list)  # ["03:00", "15:00"]
    epg_enabled: bool = True  # Whether to fetch EPG from this source
    epg_url: str = ""  # EPG URL (auto-detected from M3U/Xtream, or manual override)
    deinterlace_fallback: bool = True  # Deinterlace when probe is skipped (for OTA/HDHomeRun)


def load_server_settings() -> dict[str, Any]:
    """Load server-wide settings."""
    if SERVER_SETTINGS_FILE.exists():
        data: dict[str, Any] = json.loads(SERVER_SETTINGS_FILE.read_text())
    else:
        data = {}
    data.setdefault("transcode_mode", "auto")
    data.setdefault("transcode_hw", _default_encoder())
    data.setdefault("vod_transcode_cache_mins", 60)
    data.setdefault("probe_live", True)
    data.setdefault("probe_movies", True)
    data.setdefault("probe_series", False)
    data.setdefault("sources", [])
    data.setdefault("users", {})
    data.setdefault("user_agent_preset", "tivimate")
    data.setdefault("user_agent_custom", "")
    return data


def save_server_settings(settings: dict[str, Any]) -> None:
    """Save server-wide settings."""
    SERVER_SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


def _validate_username(username: str) -> None:
    """Validate username to prevent path traversal and length attacks."""
    if (
        not username
        or len(username) > 64
        or ".." in username
        or "/" in username
        or "\\" in username
    ):
        raise ValueError("Invalid username")


def load_user_settings(username: str) -> dict[str, Any]:
    """Load per-user settings."""
    _validate_username(username)
    user_file = USERS_DIR / username / "settings.json"
    if user_file.exists():
        data = json.loads(user_file.read_text())
    else:
        data = {}
    data.setdefault("guide_filter", [])
    data.setdefault("captions_enabled", True)
    data.setdefault("watch_history", {})
    data.setdefault("favorites", {"series": {}, "movies": {}})
    data.setdefault("cc_lang", "")
    data.setdefault("cc_style", {})
    data.setdefault("cast_host", "")
    return data


def save_user_settings(username: str, settings: dict[str, Any]) -> None:
    """Save per-user settings."""
    _validate_username(username)
    user_dir = USERS_DIR / username
    user_dir.mkdir(exist_ok=True)
    (user_dir / "settings.json").write_text(json.dumps(settings, indent=2))


def get_watch_position(username: str, stream_url: str) -> dict[str, Any] | None:
    """Get saved watch position for a stream. Returns None if not found or >=95% watched."""
    settings = load_user_settings(username)
    history = settings.get("watch_history", {})
    entry = history.get(stream_url)
    if not entry:
        return None
    # Reset if >=95% watched
    if entry.get("duration", 0) > 0:
        pct = entry.get("position", 0) / entry["duration"]
        if pct >= 0.95:
            return None
    return entry


def save_watch_position(username: str, stream_url: str, position: float, duration: float) -> None:
    """Save watch position for a stream."""
    settings = load_user_settings(username)
    history = settings.setdefault("watch_history", {})
    history[stream_url] = {
        "position": position,
        "duration": duration,
        "updated": time.time(),
    }
    # Keep only last 200 entries
    if len(history) > 200:
        sorted_entries = sorted(history.items(), key=lambda x: x[1].get("updated", 0), reverse=True)
        settings["watch_history"] = dict(sorted_entries[:200])
    save_user_settings(username, settings)


def get_sources() -> list[Source]:
    """Get list of configured sources."""
    settings = load_server_settings()
    return [Source(**s) for s in settings.get("sources", [])]


def update_source_epg_url(source_id: str, epg_url: str) -> None:
    """Update a source's epg_url in settings (only if currently empty)."""
    if not epg_url:
        return
    settings = load_server_settings()
    for s in settings.get("sources", []):
        if s["id"] == source_id and not s.get("epg_url"):
            s["epg_url"] = epg_url
            save_server_settings(settings)
            log.info("Saved EPG URL for source %s: %s", source_id, epg_url)
            break
