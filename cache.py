"""File cache, settings, sources management."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import hashlib
import json
import logging
import pathlib
import subprocess
import threading
import time
import urllib.parse


log = logging.getLogger(__name__)


# ===========================================================================
# VAAPI Auto-Detection
# ===========================================================================


def _get_gpu_vendor() -> str | None:
    """Detect GPU vendor ID via lspci or sysfs. Returns '8086' (Intel) or '1002' (AMD)."""
    # Try lspci first (works on bare metal)
    try:
        result = subprocess.run(["lspci", "-nn"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if "VGA" in line or "Display" in line or "3D" in line:
                if "[8086:" in line:
                    return "8086"
                if "[1002:" in line:
                    return "1002"
    except Exception:
        pass

    # Fallback: check sysfs (works in containers)
    drm_path = pathlib.Path("/sys/class/drm")
    if drm_path.exists():
        for card in drm_path.iterdir():
            if card.name.startswith("card") and card.name[4:].isdigit():
                vendor_file = card / "device" / "vendor"
                if vendor_file.exists():
                    vendor = vendor_file.read_text().strip().replace("0x", "")
                    if vendor in ("8086", "1002"):
                        return vendor
    return None


def _detect_vaapi_device() -> str | None:
    """Auto-detect the VAAPI render device. Returns '/dev/dri/renderD128' or None."""
    render = pathlib.Path("/dev/dri/renderD128")
    return str(render) if render.exists() else None


def _detect_libva_driver() -> str | None:
    """Auto-detect LIBVA driver name. Returns 'iHD', 'i965', 'radeonsi', or None."""
    vendor = _get_gpu_vendor()
    if vendor == "8086":
        # iHD for Intel Gen8+ (Broadwell 2014+), supports Xe driver
        # Fall back to i965 for older Intel GPUs
        dri_path = _detect_dri_path()
        if dri_path and pathlib.Path(f"{dri_path}/iHD_drv_video.so").exists():
            return "iHD"
        return "i965"
    if vendor == "1002":
        return "radeonsi"
    return None


def _detect_dri_path() -> str | None:
    """Auto-detect the system DRI drivers path.

    Returns path like '/usr/lib/x86_64-linux-gnu/dri' or None.
    """
    # Check common locations in order of preference
    candidates = [
        "/usr/lib/x86_64-linux-gnu/dri",  # Debian/Ubuntu
        "/usr/lib64/dri",  # Fedora/RHEL
        "/usr/lib/dri",  # Arch
    ]
    for path in candidates:
        if pathlib.Path(path).is_dir():
            return path
    return None


# Cached detection results (computed once at import)
VAAPI_DEVICE = _detect_vaapi_device()
LIBVA_DRIVER = _detect_libva_driver()
DRI_PATH = _detect_dri_path()

APP_DIR = pathlib.Path(__file__).parent
# Use old "cache" if it exists (backwards compat), otherwise ".cache"
_OLD_CACHE = APP_DIR / "cache"
CACHE_DIR = _OLD_CACHE if _OLD_CACHE.exists() else APP_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
SERVER_SETTINGS_FILE = CACHE_DIR / "server_settings.json"
USERS_DIR = CACHE_DIR / "users"
USERS_DIR.mkdir(exist_ok=True)
LOGOS_DIR = CACHE_DIR / "logos"
LOGOS_DIR.mkdir(exist_ok=True)

# Cache TTLs in seconds
LIVE_CACHE_TTL = 2 * 3600  # 2 hours
EPG_CACHE_TTL = 6 * 3600  # 6 hours
VOD_CACHE_TTL = 12 * 3600  # 12 hours
SERIES_CACHE_TTL = 12 * 3600  # 12 hours
INFO_CACHE_TTL = 7 * 24 * 3600  # 7 days max for series/movie info
INFO_CACHE_STALE = 24 * 3600  # Refresh in background after 24 hours
LOGO_CACHE_TTL = 7 * 24 * 3600  # 7 days for logos (server-side)
LOGO_BROWSER_TTL = 24 * 3600  # 1 day for browser cache (re-validates before server expires)
LOGO_MAX_SIZE = 1024 * 1024  # 1MB max logo size

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


def clear_all_file_caches() -> int:
    """Clear all data file caches (live, vod, series). Returns count deleted."""
    cache_files = ["live_data.json", "vod_data.json", "series_data.json"]
    deleted = 0
    for name in cache_files:
        path = CACHE_DIR / name
        if path.exists():
            path.unlink()
            deleted += 1
    # Also clear memory cache
    clear_all_caches()
    return deleted


def get_cache() -> dict[str, Any]:
    """Get reference to memory cache."""
    return _cache


def get_cache_lock() -> threading.Lock:
    """Get cache lock."""
    return _cache_lock


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a directory/file name."""
    # Remove path traversal and special chars
    name = name.replace("..", "").replace("/", "_").replace("\\", "_")
    name = "".join(c for c in name if c.isalnum() or c in "-_ ")
    return name[:224] or "default"


def _url_to_filename(url: str) -> str:
    """Derive a readable filename from URL with hash suffix to avoid collisions."""
    # Always include hash suffix to avoid collisions
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/")
    if path:
        # Get last path component
        name = path.split("/")[-1]
        # Strip extension, we'll add our own
        if "." in name:
            name = name.rsplit(".", 1)[0]
        name = _sanitize_name(name)
        if name and len(name) >= 2:
            return f"{name}_{url_hash}"
    return url_hash


def get_cached_logo(source_name: str, url: str) -> pathlib.Path | None:
    """Get cached logo path if valid and not expired. Returns None if not cached."""
    safe_source = _sanitize_name(source_name)
    filename = _url_to_filename(url)
    source_dir = LOGOS_DIR / safe_source
    if not source_dir.exists():
        return None
    # Look for file with any extension
    for ext in ("png", "jpg", "jpeg", "gif", "webp", "svg"):
        path = source_dir / f"{filename}.{ext}"
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < LOGO_CACHE_TTL:
                return path
            # Expired, delete it
            path.unlink(missing_ok=True)
    return None


def save_logo(source_name: str, url: str, data: bytes, content_type: str) -> pathlib.Path:
    """Save logo to cache. Returns the saved path."""
    safe_source = _sanitize_name(source_name)
    filename = _url_to_filename(url)
    source_dir = LOGOS_DIR / safe_source
    source_dir.mkdir(parents=True, exist_ok=True)
    # Determine extension from content-type
    ext_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/svg+xml": "svg",
    }
    ext = ext_map.get(content_type.split(";")[0].strip(), "png")
    path = source_dir / f"{filename}.{ext}"
    # Atomic write: write to temp file then rename
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)
    return path


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


def _test_encoder(cmd: list[str], timeout: int = 5, env: dict | None = None) -> tuple[bool, str]:
    """Test if an encoder works. Returns (success, error_message)."""
    try:
        run_env = None
        if env:
            import os

            run_env = os.environ.copy()
            run_env.update(env)
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, env=run_env)
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
        "nvenc": False,
        "amf": False,
        "qsv": False,
        "vaapi": False,
    }

    # Test input: 1 frame of 256x256 black (64x64 is below NVENC minimum on newer GPUs)
    test_input = ["-f", "lavfi", "-i", "color=black:s=256x256:d=0.04", "-frames:v", "1"]
    base_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    null_out = ["-f", "null", "-"]

    # NVENC: try nvenc directly
    ok, err = _test_encoder(base_cmd + test_input + ["-c:v", "h264_nvenc"] + null_out)
    encoders["nvenc"] = ok
    if ok:
        log.info("  NVENC (h264_nvenc): available")
    else:
        log.info("  NVENC (h264_nvenc): unavailable - %s", err)

    # AMF: try amf directly
    ok, err = _test_encoder(base_cmd + test_input + ["-c:v", "h264_amf"] + null_out)
    encoders["amf"] = ok
    if ok:
        log.info("  AMF (h264_amf): available")
    else:
        log.info("  AMF (h264_amf): unavailable - %s", err)

    # QSV: needs hwaccel init
    ok, err = _test_encoder(
        base_cmd
        + ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
        + test_input
        + ["-c:v", "h264_qsv"]
        + null_out
    )
    encoders["qsv"] = ok
    if ok:
        log.info("  QSV (h264_qsv): available")
    else:
        log.info("  QSV (h264_qsv): unavailable - %s", err)

    # VA-API: needs device, hwupload, and driver env vars for hybrid GPU systems
    vaapi_baseline_only = False
    if VAAPI_DEVICE and LIBVA_DRIVER and DRI_PATH:
        vaapi_env = {
            "LIBVA_DRIVER_NAME": LIBVA_DRIVER,
            "LIBVA_DRIVERS_PATH": DRI_PATH,
        }
        # Try high profile first, fall back to constrained_baseline for older GPUs
        ok, err = _test_encoder(
            base_cmd
            + ["-init_hw_device", f"vaapi=va:{VAAPI_DEVICE}"]
            + test_input
            + ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi"]
            + null_out,
            env=vaapi_env,
        )
        if not ok:
            # Some older AMD GPUs (GCN 1.0) only support baseline profile
            ok, err = _test_encoder(
                base_cmd
                + ["-init_hw_device", f"vaapi=va:{VAAPI_DEVICE}"]
                + test_input
                + [
                    "-vf",
                    "format=nv12,hwupload",
                    "-c:v",
                    "h264_vaapi",
                    "-profile:v",
                    "constrained_baseline",
                ]
                + null_out,
                env=vaapi_env,
            )
            if ok:
                vaapi_baseline_only = True
        encoders["vaapi"] = ok
        encoders["vaapi_baseline_only"] = vaapi_baseline_only
        if ok:
            profile_note = " (baseline only)" if vaapi_baseline_only else ""
            log.info(
                "  VAAPI (h264_vaapi): available%s (device=%s, driver=%s)",
                profile_note,
                VAAPI_DEVICE,
                LIBVA_DRIVER,
            )
        else:
            log.info("  VAAPI (h264_vaapi): unavailable - %s", err)
    else:
        log.info("  VAAPI (h264_vaapi): unavailable - no Intel/AMD GPU detected")

    return encoders


AVAILABLE_ENCODERS = detect_encoders()


def refresh_encoders() -> dict[str, bool]:
    """Re-detect available encoders and update the cache."""
    global AVAILABLE_ENCODERS
    AVAILABLE_ENCODERS = detect_encoders()
    return AVAILABLE_ENCODERS


def _default_encoder() -> str:
    """Return first available encoder option.

    Preference order: nvenc > amf > qsv > vaapi > software
    For nvenc/amf, prefer +vaapi fallback if VAAPI is available.
    """
    if AVAILABLE_ENCODERS.get("nvenc"):
        return "nvenc+vaapi" if AVAILABLE_ENCODERS.get("vaapi") else "nvenc+software"
    if AVAILABLE_ENCODERS.get("amf"):
        return "amf+vaapi" if AVAILABLE_ENCODERS.get("vaapi") else "amf+software"
    if AVAILABLE_ENCODERS.get("qsv"):
        return "qsv"
    if AVAILABLE_ENCODERS.get("vaapi"):
        return "vaapi"
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
    max_streams: int = 0  # Max concurrent streams from this source (0 = unlimited)


def load_server_settings() -> dict[str, Any]:
    """Load server-wide settings."""
    if SERVER_SETTINGS_FILE.exists():
        data: dict[str, Any] = json.loads(SERVER_SETTINGS_FILE.read_text())
    else:
        data = {}
    data.setdefault("transcode_mode", "auto")

    # Migrate old transcode_hw values to new format
    old_hw = data.get("transcode_hw", "")
    if old_hw == "nvidia":
        data["transcode_hw"] = (
            "nvenc+vaapi" if AVAILABLE_ENCODERS.get("vaapi") else "nvenc+software"
        )
    elif old_hw == "intel":
        data["transcode_hw"] = "qsv"
    # "vaapi" and "software" remain unchanged

    data.setdefault("transcode_hw", _default_encoder())
    data.setdefault("vod_transcode_cache_mins", 60)
    # 0 = no caching (dead sessions cleaned immediately)
    data.setdefault("live_transcode_cache_secs", 0)
    data.setdefault("live_dvr_mins", 0)  # 0 = disabled (default 30 sec buffer)
    data.setdefault("transcode_dir", "")  # Empty = system temp dir
    data.setdefault("probe_live", True)
    data.setdefault("probe_movies", True)
    data.setdefault("probe_series", False)
    data.setdefault("sources", [])
    data.setdefault("users", {})
    data.setdefault("user_agent_preset", "tivimate")
    data.setdefault("user_agent_custom", "")
    data.setdefault("m3u_export_enabled", False)
    data.setdefault("m3u_export_mode", "redirect")  # "redirect" or "proxy"
    return data


def save_server_settings(settings: dict[str, Any]) -> None:
    """Save server-wide settings."""
    SERVER_SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


USER_AGENT_PRESETS = {
    "vlc": "VLC/3.0.20 LibVLC/3.0.20",
    "chrome": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "tivimate": "TiviMate/4.7.0",
}


def get_user_agent() -> str | None:
    """Get the configured User-Agent, or None for the caller's own default."""
    settings = load_server_settings()
    preset = settings.get("user_agent_preset", "default")
    if preset == "default":
        return None
    if preset == "custom":
        return settings.get("user_agent_custom") or None
    return USER_AGENT_PRESETS.get(preset)


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
