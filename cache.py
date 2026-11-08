"""File cache, settings, sources management."""

from __future__ import annotations

from collections.abc import Callable
from concurrent import futures
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypeAlias, TypedDict
from urllib import parse

import hashlib
import json
import logging
import os
import pathlib
import subprocess
import threading
import time

from treekle import MutablePlainTree, ReadError, from_plain, json_loads

from util import atomic_write_json, data_dir, read_json_object, write_json_object
from xtream import ProviderRecord


log = logging.getLogger(__name__)

HwAccel: TypeAlias = Literal[
    "nvenc+vaapi",
    "nvenc+software",
    "amf+vaapi",
    "amf+software",
    "qsv",
    "vaapi",
    "software",
]


class SourceConfig(TypedDict):
    """A configured provider as stored in the server settings file."""

    id: str
    name: str
    type: str
    url: str
    username: NotRequired[str]
    password: NotRequired[str]
    epg_timeout: NotRequired[int]
    epg_schedule: NotRequired[list[str]]
    epg_enabled: NotRequired[bool]
    epg_url: NotRequired[str]
    deinterlace_fallback: NotRequired[bool]
    max_streams: NotRequired[int]
    users: NotRequired[list[str] | None]


class ServerSettings(TypedDict):
    """Server-wide settings; keys ``load_server_settings`` defaults are required."""

    transcode_mode: str
    transcode_hw: HwAccel
    vod_transcode_cache_mins: int
    live_transcode_cache_secs: int
    live_dvr_mins: int
    transcode_dir: str
    probe_live: bool
    probe_movies: bool
    probe_series: bool
    sources: list[SourceConfig]
    user_agent_preset: str
    user_agent_custom: str
    max_resolution: NotRequired[str]
    quality: NotRequired[str]
    sr_model: NotRequired[str]
    vod_order: NotRequired[list[str]]
    series_order: NotRequired[list[str]]
    # ``users`` and ``secret_key`` live in this file too; auth owns and writes them.


class WatchEntry(TypedDict):
    """Saved playback position of one stream."""

    position: float
    duration: float
    updated: float


class UserSettings(TypedDict):
    """Per-user settings; keys ``load_user_settings`` defaults are required."""

    guide_filter: list[str]
    captions_enabled: bool
    sort_live_by_views: bool
    watch_history: dict[str, WatchEntry]
    # The browser defines favorite entries; the server stores them verbatim.
    favorites: dict[str, dict[str, MutablePlainTree]]
    cc_lang: str
    cc_style: dict[str, MutablePlainTree]
    cast_host: str
    vod_filter: NotRequired[list[str]]
    series_filter: NotRequired[list[str]]
    guide_selected_cats: NotRequired[list[str] | None]  # None selects all channels.
    virtual_scroll: NotRequired[bool]


class MemoryCache(TypedDict, total=False):
    """Provider catalogs held in memory between file-cache refreshes.

    Clear it with ``clear_all_caches``: a TypedDict has no portable ``clear()``.
    """

    live_categories: list[ProviderRecord]
    live_streams: list[ProviderRecord]
    vod_categories: list[ProviderRecord]
    vod_streams: list[ProviderRecord]
    series_categories: list[ProviderRecord]
    series: list[ProviderRecord]
    epg_urls: list[tuple[str, int, str]]
    epg_error: str
    epg: MutablePlainTree


def clamp_live_dvr_mins(value: int) -> int:
    return max(value, 0)


# ===========================================================================
# VAAPI Auto-Detection
# ===========================================================================


VAAPI_DEVICE: str | None = None
LIBVA_DRIVER: str | None = None
DRI_PATH: str | None = None
_initialize_lock = threading.Lock()

APP_DIR = pathlib.Path(__file__).parent
CACHE_DIR = APP_DIR / ".cache"
SERVER_SETTINGS_FILE = CACHE_DIR / "server_settings.json"
USERS_DIR = CACHE_DIR / "users"
LOGOS_DIR = CACHE_DIR / "logos"

# In-memory caches.
_cache: MemoryCache = {}
_info_cache: dict[str, ProviderRecord] = {}
_cache_lock = threading.Lock()
_live_view_lock = threading.Lock()


def load_file_cache(name: str, use_process: bool = False) -> tuple[MutablePlainTree, float] | None:
    """Load cached data from file. Returns (data, timestamp) or None.

    Args:
        name: Cache file name (without .json extension)
        use_process: If True, parse in separate process to avoid GIL blocking.

    Returns:
        cached: Data and write timestamp, or None for missing or invalid data.
    """
    initialize_paths()
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        return None
    if use_process:
        with futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_parse_json_file, str(path))
            return future.result(timeout=60)
    return _parse_json_file(str(path))


def save_file_cache(name: str, data: object) -> None:
    """Save data to cache file with current timestamp."""
    initialize_paths()
    path = CACHE_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data": data, "timestamp": time.time()}))


def clear_all_caches() -> None:
    """Clear memory caches except EPG (file cache preserved for restart)."""
    # Pops keep the dict callers hold; cache_test checks they cover every key.
    with _cache_lock:
        _cache.pop("live_categories", None)
        _cache.pop("live_streams", None)
        _cache.pop("vod_categories", None)
        _cache.pop("vod_streams", None)
        _cache.pop("series_categories", None)
        _cache.pop("series", None)
        _cache.pop("epg_urls", None)
        _cache.pop("epg_error", None)
        _info_cache.clear()


def clear_all_file_caches() -> int:
    """Clear live, VOD, and series file caches and their memory entries.

    Returns:
        deleted: Number of removed cache files.
    """
    initialize_paths()
    cache_files = ["live_data.json", "vod_data.json", "series_data.json"]
    deleted = 0
    for name in cache_files:
        path = CACHE_DIR / name
        if path.exists():
            path.unlink()
            deleted += 1
    # Also clear memory cache.
    clear_all_caches()
    return deleted


def get_cache() -> MemoryCache:
    """Get reference to memory cache."""
    return _cache


def get_cache_lock() -> threading.Lock:
    """Get cache lock."""
    return _cache_lock


def get_cached_logo(
    source_name: str,
    url: str,
    *,
    ttl_seconds: float = 7 * 24 * 3600,
) -> pathlib.Path | None:
    """Get a cached logo if it has not expired.

    Args:
        source_name: Source-specific cache namespace.
        url: Original logo URL.
        ttl_seconds: Maximum server-side cache age.

    Returns:
        path: Cached logo path, or None when unavailable.
    """
    initialize_paths()
    safe_source = _sanitize_name(source_name)
    filename = _url_to_filename(url)
    source_dir = LOGOS_DIR / safe_source
    if not source_dir.exists():
        return None
    # Look for file with any extension.
    for ext in ("png", "jpg", "jpeg", "gif", "webp", "svg"):
        path = source_dir / f"{filename}.{ext}"
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < ttl_seconds:
                return path
            # Expired, delete it.
            path.unlink(missing_ok=True)
    return None


def save_logo(source_name: str, url: str, data: bytes, content_type: str) -> pathlib.Path:
    """Save logo bytes to the source's cache directory.

    Args:
        source_name: Source-specific cache namespace.
        url: Original logo URL.
        data: Image payload.
        content_type: HTTP media type used to choose the extension.

    Returns:
        path: Saved logo path.
    """
    initialize_paths()
    safe_source = _sanitize_name(source_name)
    filename = _url_to_filename(url)
    source_dir = LOGOS_DIR / safe_source
    source_dir.mkdir(parents=True, exist_ok=True)
    # Determine extension from content-type.
    ext_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/svg+xml": "svg",
    }
    ext = ext_map.get(content_type.split(";")[0].strip(), "png")
    path = source_dir / f"{filename}.{ext}"
    # Atomic write: write to temp file then rename.
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)
    return path


def get_cached_info(
    cache_key: str,
    fetch_fn: Callable[[], ProviderRecord],
    force: bool = False,
    *,
    ttl_seconds: float = 7 * 24 * 3600,
    stale_seconds: float = 24 * 3600,
) -> ProviderRecord:
    """Get information using stale-while-revalidate caching.

    Args:
        cache_key: File and memory cache identity.
        fetch_fn: Callback retrieving fresh information.
        force: Bypass cached information.
        ttl_seconds: Maximum file cache age.
        stale_seconds: Age at which background refresh starts.

    Returns:
        data: Cached or freshly fetched information.
    """
    cached = load_file_cache(cache_key)
    cached_data: ProviderRecord = {}
    cached_ts = 0.0
    if cached is not None:
        tree, cached_ts = cached
        if isinstance(tree, dict):
            cached_data = tree
    age = time.time() - cached_ts

    if force and cached_data:
        _info_cache.pop(cache_key, None)
        cached_data = {}

    if cache_key in _info_cache and not force:
        if cached_ts and age > stale_seconds:
            threading.Thread(
                target=_refresh_cached_info,
                args=(cache_key, fetch_fn),
                daemon=True,
            ).start()
        return _info_cache[cache_key]

    if cached_data and age < ttl_seconds:
        _info_cache[cache_key] = cached_data
        if age > stale_seconds:
            threading.Thread(
                target=_refresh_cached_info,
                args=(cache_key, fetch_fn),
                daemon=True,
            ).start()
        return cached_data

    data = fetch_fn()
    _info_cache[cache_key] = data
    save_file_cache(cache_key, data)
    return data


def detect_encoders() -> dict[str, bool]:
    """Detect available FFmpeg H.264 encoders by testing actual hardware.

    Returns:
        encoders: Availability flags and VAAPI baseline-only capability.
    """
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
        + null_out,
    )
    encoders["qsv"] = ok
    if ok:
        log.info("  QSV (h264_qsv): available")
    else:
        log.info("  QSV (h264_qsv): unavailable - %s", err)

    # VA-API: needs device, hwupload, and driver env vars for hybrid GPU systems.
    vaapi_baseline_only = False
    if VAAPI_DEVICE is not None and LIBVA_DRIVER is not None and DRI_PATH is not None:
        vaapi_env = {
            "LIBVA_DRIVER_NAME": LIBVA_DRIVER,
            "LIBVA_DRIVERS_PATH": DRI_PATH,
        }
        # Try high profile first, fall back to constrained_baseline for older GPUs.
        ok, err = _test_encoder(
            base_cmd
            + ["-init_hw_device", f"vaapi=va:{VAAPI_DEVICE}"]
            + test_input
            + ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi"]
            + null_out,
            env=vaapi_env,
        )
        if not ok:
            # Some older AMD GPUs (GCN 1.0) only support baseline profile.
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


AVAILABLE_ENCODERS: dict[str, bool] = {}


def initialize_paths() -> None:
    """Resolve legacy storage without replacing caller-supplied paths."""
    global CACHE_DIR, SERVER_SETTINGS_FILE, USERS_DIR, LOGOS_DIR
    default_dir = APP_DIR / ".cache"
    if default_dir == CACHE_DIR:
        CACHE_DIR = data_dir(APP_DIR)
    if default_dir / "server_settings.json" == SERVER_SETTINGS_FILE:
        SERVER_SETTINGS_FILE = CACHE_DIR / "server_settings.json"
    if default_dir / "users" == USERS_DIR:
        USERS_DIR = CACHE_DIR / "users"
    if default_dir / "logos" == LOGOS_DIR:
        LOGOS_DIR = CACHE_DIR / "logos"


def initialize() -> None:
    """Initialize storage and detect hardware once before serving requests."""
    global VAAPI_DEVICE, LIBVA_DRIVER, DRI_PATH
    initialize_paths()
    with _initialize_lock:
        # A populated encoder table means detection ran or a caller supplied it.
        if AVAILABLE_ENCODERS:
            return
        if VAAPI_DEVICE is None:
            VAAPI_DEVICE = _detect_vaapi_device()
        if LIBVA_DRIVER is None:
            LIBVA_DRIVER = _detect_libva_driver()
        if DRI_PATH is None:
            DRI_PATH = _detect_dri_path()
        AVAILABLE_ENCODERS.update(detect_encoders())


def refresh_encoders() -> dict[str, bool]:
    """Re-detect available encoders and update the cache."""
    encoders = detect_encoders()
    AVAILABLE_ENCODERS.clear()
    AVAILABLE_ENCODERS.update(encoders)
    return AVAILABLE_ENCODERS


def hw_accel(value: str) -> HwAccel:
    """Map a stored or submitted ``transcode_hw`` value to a supported option.

    Args:
        value: A current option, a legacy one (``nvidia``, ``intel``), or anything else.

    Returns:
        hw: The option itself; the migrated one for a legacy value; else the default.
    """
    match value:
        case "nvidia":
            return "nvenc+vaapi" if AVAILABLE_ENCODERS.get("vaapi", False) else "nvenc+software"
        case "intel":
            return "qsv"
        case (
            "nvenc+vaapi"
            | "nvenc+software"
            | "amf+vaapi"
            | "amf+software"
            | "qsv"
            | "vaapi"
            | "software"
        ) as hw:
            return hw
        case _:
            return _default_encoder()


@dataclass(kw_only=True, slots=True)
class Source:
    id: str
    name: str
    type: str  # "xtream", "m3u", or "epg"
    url: str
    username: str = ""
    password: str = ""
    epg_timeout: int = 120  # Seconds.
    epg_schedule: list[str] = field(default_factory=list)  # ["03:00", "15:00"].
    epg_enabled: bool = True  # Whether to fetch EPG from this source.
    epg_url: str = ""  # EPG URL (auto-detected from M3U/Xtream, or manual override)
    deinterlace_fallback: bool = True  # Deinterlace when probe is skipped (for OTA/HDHomeRun)
    max_streams: int = 0  # Max concurrent streams from this source (0 = unlimited)
    users: list[str] | None = None  # Non-admin users with access (None = everyone)


def load_server_settings() -> ServerSettings:
    """Load server-wide settings with defaults and legacy migrations.

    Returns:
        settings: Persisted settings supplemented by current defaults.
    """
    initialize()
    raw = read_json_object(SERVER_SETTINGS_FILE)
    live_dvr_mins = raw.get("live_dvr_mins", 120)
    if not isinstance(live_dvr_mins, int):
        raise TypeError("live_dvr_mins must be an integer")
    stored_hw = raw.get("transcode_hw")
    return from_plain(
        {
            "transcode_mode": "auto",
            "vod_transcode_cache_mins": 60,
            # 0 = no caching (dead sessions cleaned immediately)
            "live_transcode_cache_secs": 0,
            "transcode_dir": "",  # Empty = system temp dir.
            "probe_live": True,
            "probe_movies": True,
            "probe_series": False,
            "sources": [],
            "user_agent_preset": "tivimate",
            "user_agent_custom": "",
            **raw,
            "transcode_hw": hw_accel(stored_hw)
            if isinstance(stored_hw, str)
            else _default_encoder(),
            "live_dvr_mins": clamp_live_dvr_mins(live_dvr_mins),
        },
        ServerSettings,
    )


def save_server_settings(settings: ServerSettings) -> None:
    """Save server-wide settings, keeping keys this release does not know.

    Args:
        settings: Settings as ``load_server_settings`` returned them, edited.
    """
    initialize_paths()
    write_json_object(SERVER_SETTINGS_FILE, settings, ServerSettings)


def get_user_agent() -> str | None:
    """Get the configured User-Agent.

    Returns:
        user_agent: Configured header, or None for the caller's default.
    """
    settings = load_server_settings()
    preset = settings["user_agent_preset"]
    if preset == "default":
        return None
    if preset == "custom":
        custom = settings["user_agent_custom"]
        return custom if custom else None
    return {
        "vlc": "VLC/3.0.20 LibVLC/3.0.20",
        "chrome": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "tivimate": "TiviMate/4.7.0",
    }.get(preset)


def load_user_settings(username: str) -> UserSettings:
    """Load per-user settings with defaults.

    Args:
        username: Account whose settings to read.

    Returns:
        settings: User preferences and viewing history.
    """
    initialize_paths()
    _validate_username(username)
    return from_plain(
        {
            "guide_filter": [],
            "captions_enabled": True,
            "watch_history": {},
            "favorites": {"series": {}, "movies": {}},
            "cc_lang": "",
            "cc_style": {},
            "cast_host": "",
            "sort_live_by_views": True,
            **read_json_object(USERS_DIR / username / "settings.json"),
        },
        UserSettings,
    )


def save_user_settings(username: str, settings: UserSettings) -> None:
    """Save per-user settings, keeping keys this release does not know.

    Args:
        username: Account whose settings to write.
        settings: Settings as ``load_user_settings`` returned them, edited.
    """
    initialize_paths()
    _validate_username(username)
    write_json_object(USERS_DIR / username / "settings.json", settings, UserSettings)


def live_stream_key(source_id: str, stream_id: str | int) -> str:
    """Build the source-qualified key used for per-user live view counts."""
    return f"{source_id or ''}:{stream_id}"


def load_live_view_counts(username: str) -> dict[str, int]:
    """Load source-qualified live view counts for a user."""
    initialize_paths()
    _validate_username(username)
    with _live_view_lock:
        return _load_live_view_counts(username)


def record_live_view(username: str, source_id: str, stream_id: str | int) -> int:
    """Increment a user's view count for a live stream.

    Args:
        username: Account whose viewing history to update.
        source_id: Provider identity.
        stream_id: Provider-specific stream identity.

    Returns:
        count: Updated view count.
    """
    initialize_paths()
    _validate_username(username)
    key = live_stream_key(source_id, stream_id)
    with _live_view_lock:
        counts = _load_live_view_counts(username)
        counts[key] = counts.get(key, 0) + 1
        atomic_write_json(
            USERS_DIR / username / "live_views.json",
            {"counts": counts},
        )
        return counts[key]


def get_watch_position(username: str, stream_url: str) -> WatchEntry | None:
    """Get a saved watch position for an unfinished stream.

    Args:
        username: Account whose viewing history to read.
        stream_url: Stream identity.

    Returns:
        position: Saved position metadata, or None if missing or at least 95% watched.
    """
    settings = load_user_settings(username)
    entry = settings["watch_history"].get(stream_url)
    if entry is None:
        return None
    # Reset if >=95% watched.
    if entry["duration"] > 0 and entry["position"] / entry["duration"] >= 0.95:
        return None
    return entry


def save_watch_position(username: str, stream_url: str, position: float, duration: float) -> None:
    """Save a watch position, retaining the 200 most recent entries.

    Args:
        username: Account whose viewing history to update.
        stream_url: Stream identity.
        position: Playback position in seconds.
        duration: Stream duration in seconds.
    """
    settings = load_user_settings(username)
    history = settings["watch_history"]
    history[stream_url] = {
        "position": position,
        "duration": duration,
        "updated": time.time(),
    }
    # Keep only last 200 entries.
    if len(history) > 200:
        sorted_entries = sorted(history.items(), key=_watch_updated, reverse=True)
        settings["watch_history"] = dict(sorted_entries[:200])
    save_user_settings(username, settings)


def get_sources() -> list[Source]:
    """Get list of configured sources."""
    settings = load_server_settings()
    return [_source(s) for s in settings["sources"]]


def hidden_source_ids(username: str, admin: bool) -> set[str]:
    """Sources the user may not see. Admins see every source."""
    if admin:
        return set()
    return {s.id for s in get_sources() if s.users is not None and username not in s.users}


def update_source_epg_url(source_id: str, epg_url: str) -> None:
    """Save a detected guide URL when the source has no configured URL.

    Args:
        source_id: Configured source identity.
        epg_url: Detected XMLTV URL; empty values are ignored.
    """
    if not epg_url:
        return
    settings = load_server_settings()
    for s in settings["sources"]:
        if s["id"] == source_id and not s.get("epg_url", ""):
            s["epg_url"] = epg_url
            save_server_settings(settings)
            host = parse.urlparse(epg_url).hostname or "unknown host"
            log.info("Saved EPG URL for source %s (%s)", source_id, host)
            break


class _CacheFile(TypedDict):
    data: MutablePlainTree
    timestamp: NotRequired[float]


class _LiveViews(TypedDict):
    counts: dict[str, int]


def _source(config: SourceConfig) -> Source:
    """Build a ``Source``; a TypedDict may carry extra keys, so it is not splatted."""
    return Source(
        id=config["id"],
        name=config["name"],
        type=config["type"],
        url=config["url"],
        username=config.get("username", ""),
        password=config.get("password", ""),
        epg_timeout=config.get("epg_timeout", 120),
        epg_schedule=config.get("epg_schedule", []),
        epg_enabled=config.get("epg_enabled", True),
        epg_url=config.get("epg_url", ""),
        deinterlace_fallback=config.get("deinterlace_fallback", True),
        max_streams=config.get("max_streams", 0),
        users=config.get("users"),
    )


def _refresh_cached_info(cache_key: str, fetch_fn: Callable[[], ProviderRecord]) -> None:
    try:
        data = fetch_fn()
        _info_cache[cache_key] = data
        save_file_cache(cache_key, data)
        log.info("Background refreshed %s", cache_key)
    except Exception as error:
        log.warning("Background refresh failed for %s: %s", cache_key, error)


def _parse_json_file(path: str) -> tuple[MutablePlainTree, float] | None:
    """Parse a cache file; runs in a separate process when asked, to avoid the GIL."""
    try:
        cached = json_loads(pathlib.Path(path).read_text(), _CacheFile)
    except (OSError, ValueError):
        return None
    return cached["data"], cached.get("timestamp", 0)


def _watch_updated(item: tuple[str, WatchEntry]) -> float:
    return item[1]["updated"]


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a directory/file name."""
    # Remove path traversal and special chars.
    name = name.replace("..", "").replace("/", "_").replace("\\", "_")
    name = "".join(c for c in name if c.isalnum() or c in "-_ ")
    return name[:224] or "default"


def _url_to_filename(url: str) -> str:
    """Derive a readable filename from URL with hash suffix to avoid collisions."""
    # Always include hash suffix to avoid collisions.
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    parsed = parse.urlparse(url)
    path = parsed.path.rstrip("/")
    if path:
        # Get last path component.
        name = path.split("/")[-1]
        # Strip extension, we'll add our own.
        if "." in name:
            name = name.rsplit(".", 1)[0]
        name = _sanitize_name(name)
        if name and len(name) >= 2:
            return f"{name}_{url_hash}"
    return url_hash


# Preference order: nvenc > amf > qsv > vaapi > software For nvenc/amf, prefer +vaapi
# fallback if VAAPI is available.
def _default_encoder() -> HwAccel:
    """Return first available encoder option."""
    vaapi = AVAILABLE_ENCODERS.get("vaapi", False)
    if AVAILABLE_ENCODERS.get("nvenc", False):
        return "nvenc+vaapi" if vaapi else "nvenc+software"
    if AVAILABLE_ENCODERS.get("amf", False):
        return "amf+vaapi" if vaapi else "amf+software"
    if AVAILABLE_ENCODERS.get("qsv", False):
        return "qsv"
    if vaapi:
        return "vaapi"
    return "software"


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


def _load_live_view_counts(username: str) -> dict[str, int]:
    path = USERS_DIR / username / "live_views.json"
    if not path.exists():
        return {}
    try:
        counts = json_loads(path.read_text(), _LiveViews)["counts"]
    except ReadError as error:
        raise ValueError("Invalid live view counts") from error
    if any(isinstance(value, bool) or value < 0 for value in counts.values()):
        raise ValueError("Invalid live view counts")
    return counts


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
        try:
            for card in drm_path.iterdir():
                if card.name.startswith("card") and card.name[4:].isdigit():
                    vendor_file = card / "device" / "vendor"
                    if vendor_file.exists():
                        vendor = vendor_file.read_text().strip().replace("0x", "")
                        if vendor in ("8086", "1002"):
                            return vendor
        except OSError:
            pass
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
        # Fall back to i965 for older Intel GPUs.
        dri_path = _detect_dri_path()
        if dri_path is not None and pathlib.Path(f"{dri_path}/iHD_drv_video.so").exists():
            return "iHD"
        return "i965"
    if vendor == "1002":
        return "radeonsi"
    return None


# Returns path like '/usr/lib/x86_64-linux-gnu/dri' or None.
def _detect_dri_path() -> str | None:
    """Auto-detect the system DRI drivers path."""
    # Check common locations in order of preference.
    candidates = [
        "/usr/lib/x86_64-linux-gnu/dri",  # Debian/Ubuntu.
        "/usr/lib64/dri",  # Fedora/RHEL.
        "/usr/lib/dri",  # Arch.
    ]
    for path in candidates:
        if pathlib.Path(path).is_dir():
            return path
    return None


def _test_encoder(
    cmd: list[str],
    timeout: int = 5,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Test if an encoder works. Returns (success, error_message)."""
    try:
        run_env = None
        if env is not None:
            run_env = os.environ.copy()
            run_env.update(env)
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, env=run_env)
        if result.returncode == 0:
            return True, ""
        stderr = result.stderr.decode(errors="replace").strip()
        # Extract the most relevant error line.
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
