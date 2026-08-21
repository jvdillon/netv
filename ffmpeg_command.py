"""FFmpeg command building and media probing."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

import json
import logging
import pathlib
import re
import subprocess
import tempfile
import threading
import time

# Import VAAPI auto-detection results (avoid circular import by importing constants only)
from cache import AVAILABLE_ENCODERS, VAAPI_DEVICE, get_user_agent


log = logging.getLogger(__name__)

HwAccel = Literal[
    "nvenc+vaapi", "nvenc+software", "amf+vaapi", "amf+software", "qsv", "vaapi", "software"
]


def _parse_hw(hw: HwAccel) -> tuple[str, str]:
    """Parse hw into (encoder, fallback). e.g. 'nvenc+vaapi' -> ('nvenc', 'vaapi')"""
    if "+" in hw:
        encoder, fallback = hw.split("+", 1)
        return encoder, fallback
    return hw, "software"  # standalone options fallback to software


# Timing constants
_HLS_SEGMENT_DURATION_SEC = 3.0  # Short segments for faster startup/seeking
_PROBE_CACHE_TTL_SEC = 3_600
_SERIES_PROBE_CACHE_TTL_SEC = 7 * 24 * 3_600  # 7 days
_PROBE_TIMEOUT_SEC = 30

# Segment file naming
SEG_PREFIX = "seg"  # Segment files are named seg000.ts, seg001.ts, etc.
DEFAULT_LIVE_BUFFER_SECS = 30.0  # Default live buffer when DVR disabled

TEXT_SUBTITLE_CODECS = {
    "subrip",
    "ass",
    "ssa",
    "mov_text",
    "webvtt",
    "srt",
}

# NVDEC capabilities by minimum compute capability
# https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
_NVDEC_MIN_COMPUTE: dict[str, float] = {
    "h264": 5.0,  # Maxwell+
    "hevc": 6.0,  # Pascal+ (HEVC 10-bit requires Pascal; Maxwell GM206 is edge case we ignore)
    "av1": 8.0,  # Ampere+
}

# VAAPI/QSV: static conservative lists (unlike NVIDIA, no clean runtime probe available).
# Could parse `vainfo` output, but format varies by driver (i965 vs iHD vs radeonsi).
# These codecs are nearly universal on any GPU from the last decade.
_VAAPI_SAFE_CODECS = {"h264", "hevc", "mpeg2video", "vp8", "vp9", "vc1", "av1"}
_QSV_SAFE_CODECS = {"h264", "hevc", "mpeg2video", "vp9", "vc1", "av1"}

# Max resolution height by setting
_MAX_RES_HEIGHT: dict[str, int] = {
    "4k": 2160,
    "1080p": 1080,
    "720p": 720,
    "480p": 480,
}

# Quality presets -> QP/CRF values (lower = higher quality)
_QUALITY_QP: dict[str, int] = {"high": 20, "medium": 28, "low": 35}
_QUALITY_CRF: dict[str, int] = {"high": 20, "medium": 26, "low": 32}

# Module state
_probe_lock = threading.Lock()
_probe_cache: dict[str, tuple[float, MediaInfo | None, list[SubtitleStream]]] = {}
_series_probe_cache: dict[int, dict[str, Any]] = {}
_gpu_nvdec_codecs: set[str] | None = None  # None = not probed yet
_has_libplacebo: bool | None = None  # None = not probed yet
_load_settings: Callable[[], dict[str, Any]] = dict

# Super-resolution configuration (set by init())
# Directory containing TensorRT engines: {model}_{height}p_fp16.engine
_sr_engine_dir: str = ""

# Use old "cache" if it exists (backwards compat), otherwise ".cache"
_OLD_CACHE = pathlib.Path(__file__).parent / "cache"
_CACHE_DIR = _OLD_CACHE if _OLD_CACHE.exists() else pathlib.Path(__file__).parent / ".cache"
_SERIES_PROBE_CACHE_FILE = _CACHE_DIR / "series_probe_cache.json"

_LANG_NAMES = {
    "eng": "English",
    "spa": "Spanish",
    "fre": "French",
    "ger": "German",
    "por": "Portuguese",
    "ita": "Italian",
    "jpn": "Japanese",
    "kor": "Korean",
    "chi": "Chinese",
    "ara": "Arabic",
    "rus": "Russian",
    "und": "Unknown",
}


@dataclass(slots=True)
class SubtitleStream:
    index: int
    lang: str
    name: str


@dataclass(slots=True)
class MediaInfo:
    video_codec: str
    audio_codec: str
    pix_fmt: str
    audio_channels: int = 0
    audio_sample_rate: int = 0
    audio_profile: str = ""  # e.g. "LC", "HE-AAC", "HE-AACv2"
    subtitle_codecs: list[str] | None = None
    duration: float = 0.0
    height: int = 0
    video_bitrate: int = 0  # bits per second, 0 if unknown
    interlaced: bool = False  # True if field_order indicates interlaced
    is_10bit: bool = False  # True if pix_fmt indicates 10-bit color
    is_hdr: bool = False  # True if color transfer indicates HDR
    is_hls: bool = False  # True if format is HLS (for input options)


def init(
    load_settings: Callable[[], dict[str, Any]],
    sr_engine_dir: str = "",
) -> None:
    """Initialize module with settings loader and optional AI Upscale config."""
    global _load_settings, _sr_engine_dir
    _load_settings = load_settings
    _sr_engine_dir = sr_engine_dir
    _load_series_probe_cache()
    if _sr_engine_dir:
        log.info("AI Upscale enabled: engine_dir=%s", _sr_engine_dir)


def get_settings() -> dict[str, Any]:
    """Get current settings."""
    return _load_settings()


def get_ffmpeg_env() -> dict[str, str] | None:
    """Get environment for ffmpeg subprocess. Returns None (ffmpeg has libtorch via rpath)."""
    # ffmpeg is built with -Wl,-rpath pointing to libtorch, so no LD_LIBRARY_PATH needed
    return None


def _find_sr_engine(model_name: str, source_height: int) -> tuple[str, int, int, int] | None:
    """Find the best matching SR engine file for the given model and resolution.

    Returns (engine_path, input_height, input_width, scale_factor) or None if not found.
    """
    import pathlib

    engine_dir = pathlib.Path(_sr_engine_dir)
    if not engine_dir.exists():
        return None

    # Find all engines for this model
    # Engine naming: {model}_{height}p_fp16.engine
    engines: list[tuple[int, pathlib.Path]] = []
    for engine in engine_dir.glob(f"{model_name}_*p_fp16.engine"):
        # Extract height from filename
        name = engine.stem  # e.g., "2x-liveaction-span_1080p_fp16"
        parts = name.rsplit("_", 2)
        if len(parts) >= 3:
            height_str = parts[1].rstrip("p")
            if height_str.isdigit():
                engines.append((int(height_str), engine))

    if not engines:
        return None

    # Determine scale factor from model name prefix (e.g., "2x-", "4x-")
    scale_match = re.match(r"^(\d+)x-", model_name)
    if scale_match:
        scale = int(scale_match.group(1))
    elif model_name == "realesrgan":
        # Legacy model name - was 4x
        scale = 4
    else:
        log.error(
            "SR: cannot determine scale from model name: %s (expected Nx- prefix)", model_name
        )
        return None

    # Sort by height ascending
    engines.sort(key=lambda x: x[0])

    # Select appropriate engine based on source height
    if source_height <= 0:
        # Probe failed - use highest resolution engine
        engine_height, engine_path = engines[-1]
        log.warning("SR: probe failed, using %dp engine", engine_height)
    else:
        # Find engine closest to but >= source height, or use largest if source is bigger
        engine_height, engine_path = engines[-1]  # default to largest
        for h, p in engines:
            if h >= source_height:
                engine_height, engine_path = h, p
                break

    # Calculate width assuming 16:9 aspect ratio, rounded to multiple of 8
    engine_width = ((engine_height * 16 // 9) + 7) // 8 * 8

    return str(engine_path), engine_height, engine_width, scale


def _build_sr_filter(source_height: int, target_height: int) -> str:
    """Build AI Upscale filter string if needed. Returns empty string if disabled.

    SR is controlled by sr_model setting - if a model is selected, SR is applied
    when source height < target height.
    """
    if not _sr_engine_dir:
        return ""

    # Get selected model from settings
    settings = _load_settings()
    model_name = settings.get("sr_model", "")
    if not model_name:
        return ""  # SR disabled (Off selected)

    # Find engine for this model and resolution
    engine_info = _find_sr_engine(model_name, source_height)
    if not engine_info:
        log.warning("SR: no engine found for model=%s, source=%dp", model_name, source_height)
        return ""

    engine_path, engine_height, engine_width, scale = engine_info

    # Apply SR when source resolution is below target (upscaling scenario)
    if target_height and source_height >= target_height:
        log.info(
            "SR: skipping %s - source %dp >= target %dp", model_name, source_height, target_height
        )
        return ""

    log.info(
        "SR: applying %s (%dx) to %dp -> %dp",
        model_name,
        scale,
        source_height,
        target_height or (source_height * scale),
    )

    # Build SR filter chain:
    # 1. Scale to engine's expected input size (preserving aspect with padding if needed)
    # 2. Convert to RGB (model expects 3-channel RGB input)
    # 3. hwupload to GPU (critical for performance - keeps data on GPU)
    # 4. Apply SR via TensorRT dnn_processing (outputs Nx resolution on GPU)
    # 5. Scale down on GPU to target resolution
    sr_filter = (
        f"scale={engine_width}:{engine_height}:force_original_aspect_ratio=decrease,"
        f"pad={engine_width}:{engine_height}:(ow-iw)/2:(oh-ih)/2,"
        f"format=rgb24,"
        f"hwupload,"
        f"dnn_processing=dnn_backend=tensorrt:model={engine_path}"
    )
    if target_height:
        # After dnn_processing, data is on GPU - use scale_cuda with explicit params
        sr_filter += f",scale_cuda=w=-2:h={target_height}"
    return sr_filter


def get_hls_segment_duration() -> float:
    """Get HLS segment duration in seconds."""
    return _HLS_SEGMENT_DURATION_SEC


# ===========================================================================
# GPU Detection
# ===========================================================================


def _get_gpu_nvdec_codecs() -> set[str]:
    """Get supported NVDEC codecs, probing GPU on first call."""
    global _gpu_nvdec_codecs
    if _gpu_nvdec_codecs is not None:
        return _gpu_nvdec_codecs
    _gpu_nvdec_codecs = set()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            log.info("No NVIDIA GPU detected")
            return _gpu_nvdec_codecs
        # Parse "NVIDIA GeForce GTX TITAN X, 5.2"
        line = result.stdout.strip().split("\n")[0]
        parts = line.rsplit(",", 1)
        if len(parts) != 2:
            return _gpu_nvdec_codecs
        gpu_name = parts[0].strip()
        compute_cap = float(parts[1].strip())
        _gpu_nvdec_codecs = {
            codec for codec, min_cap in _NVDEC_MIN_COMPUTE.items() if compute_cap >= min_cap
        }
        log.info(
            "GPU: %s (compute %.1f) NVDEC: %s",
            gpu_name,
            compute_cap,
            _gpu_nvdec_codecs or "none",
        )
    except Exception as e:
        log.debug("GPU probe failed: %s", e)
    return _gpu_nvdec_codecs


def _has_libplacebo_filter() -> bool:
    """Check if FFmpeg has libplacebo filter available (for GPU HDR tone mapping)."""
    global _has_libplacebo
    if _has_libplacebo is not None:
        return _has_libplacebo
    _has_libplacebo = False
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        _has_libplacebo = "libplacebo" in result.stdout
        log.info("libplacebo filter available: %s", _has_libplacebo)
    except Exception as e:
        log.debug("libplacebo probe failed: %s", e)
    return _has_libplacebo


# ===========================================================================
# Transcode Directory
# ===========================================================================


def get_transcode_dir() -> pathlib.Path:
    """Get the transcode output directory. Falls back to system temp if not set or inaccessible."""
    custom_dir = _load_settings().get("transcode_dir", "")
    if custom_dir:
        path = pathlib.Path(custom_dir)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except (PermissionError, OSError) as e:
            log.warning("Transcode dir %s inaccessible (%s), using temp dir", custom_dir, e)
    return pathlib.Path(tempfile.gettempdir())


# ===========================================================================
# Series Probe Cache Persistence
# ===========================================================================


def _load_series_probe_cache() -> None:
    """Load series probe cache from disk."""
    if not _SERIES_PROBE_CACHE_FILE.exists():
        return
    try:
        data = json.loads(_SERIES_PROBE_CACHE_FILE.read_text())
        count = 0
        with _probe_lock:
            for sid_str, series_data in data.items():
                sid = int(sid_str)
                if sid not in _series_probe_cache:
                    _series_probe_cache[sid] = {
                        "name": series_data.get("name", ""),
                        "mru": series_data.get("mru"),
                        "episodes": {},
                    }
                else:
                    _series_probe_cache[sid].setdefault("name", series_data.get("name", ""))
                    _series_probe_cache[sid].setdefault("mru", series_data.get("mru"))
                    _series_probe_cache[sid].setdefault("episodes", {})
                for eid_str, entry in series_data.get("episodes", {}).items():
                    eid = int(eid_str)
                    if eid in _series_probe_cache[sid]["episodes"]:
                        continue
                    # Use .get() for all fields to handle corrupt/incomplete cache
                    video_codec = entry.get("video_codec", "")
                    if not video_codec:
                        continue  # Skip entries without video codec
                    media_info = MediaInfo(
                        video_codec=video_codec,
                        audio_codec=entry.get("audio_codec", ""),
                        pix_fmt=entry.get("pix_fmt", ""),
                        audio_channels=entry.get("audio_channels", 0),
                        audio_sample_rate=entry.get("audio_sample_rate", 0),
                        subtitle_codecs=entry.get("subtitle_codecs"),
                        duration=entry.get("duration", 0),
                        height=entry.get("height", 0),
                        video_bitrate=entry.get("video_bitrate", 0),
                        interlaced=entry.get("interlaced", False),
                        is_10bit=entry.get("is_10bit", False),
                        is_hdr=entry.get("is_hdr", False),
                        is_hls=entry.get("is_hls", False),
                    )
                    subs = [
                        SubtitleStream(s["index"], s.get("lang", "und"), s.get("name", ""))
                        for s in entry.get("subtitles", [])
                    ]
                    _series_probe_cache[sid]["episodes"][eid] = (
                        entry.get("time", 0),
                        media_info,
                        subs,
                    )
                    count += 1
        log.info("Loaded %d series probe cache entries", count)
    except Exception as e:
        log.warning("Failed to load series probe cache: %s", e)


def _save_series_probe_cache() -> None:
    """Save series probe cache to disk."""
    with _probe_lock:
        data: dict[str, dict[str, Any]] = {}
        for sid, series_data in _series_probe_cache.items():
            episodes = series_data.get("episodes", {})
            data[str(sid)] = {
                "name": series_data.get("name", ""),
                "mru": series_data.get("mru"),
                "episodes": {},
            }
            for eid, (cache_time, media_info, subs) in episodes.items():
                if media_info is None:
                    continue
                data[str(sid)]["episodes"][str(eid)] = {
                    "time": cache_time,
                    "video_codec": media_info.video_codec,
                    "audio_codec": media_info.audio_codec,
                    "pix_fmt": media_info.pix_fmt,
                    "audio_channels": media_info.audio_channels,
                    "audio_sample_rate": media_info.audio_sample_rate,
                    "subtitle_codecs": media_info.subtitle_codecs,
                    "duration": media_info.duration,
                    "height": media_info.height,
                    "video_bitrate": media_info.video_bitrate,
                    "interlaced": media_info.interlaced,
                    "is_10bit": media_info.is_10bit,
                    "is_hdr": media_info.is_hdr,
                    "subtitles": [{"index": s.index, "lang": s.lang, "name": s.name} for s in subs],
                }
    try:
        _SERIES_PROBE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SERIES_PROBE_CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.warning("Failed to save series probe cache: %s", e)


# ===========================================================================
# Probe Cache Management
# ===========================================================================


def get_series_probe_cache_stats() -> list[dict[str, Any]]:
    """Get stats about cached series probes for settings UI."""
    with _probe_lock:
        log.info(
            "get_series_probe_cache_stats: cache has %d series: %s",
            len(_series_probe_cache),
            list(_series_probe_cache.keys()),
        )
        result = []
        for series_id, series_data in _series_probe_cache.items():
            episodes = series_data.get("episodes", {})
            if not episodes:
                continue
            # Get most recent entry for display info
            most_recent = max(episodes.values(), key=lambda x: x[0])
            _, media_info, subs = most_recent
            if media_info is None:
                continue
            # Build episode list
            episode_list = []
            for eid, (_, emedia, esubs) in episodes.items():
                if emedia:
                    episode_list.append(
                        {
                            "episode_id": eid,
                            "duration": emedia.duration,
                            "subtitle_count": len(esubs),
                        }
                    )
            result.append(
                {
                    "series_id": series_id,
                    "name": series_data.get("name", ""),
                    "mru": series_data.get("mru"),
                    "episode_count": len(episodes),
                    "video_codec": media_info.video_codec,
                    "audio_codec": media_info.audio_codec,
                    "subtitle_count": len(subs),
                    "episodes": sorted(episode_list, key=lambda x: x["episode_id"]),
                }
            )
        return sorted(result, key=lambda x: x.get("name") or str(x["series_id"]))


def clear_all_probe_cache() -> int:
    """Clear all probe caches. Returns count of entries cleared."""
    with _probe_lock:
        url_count = len(_probe_cache)
        series_count = sum(len(s.get("episodes", {})) for s in _series_probe_cache.values())
        _probe_cache.clear()
        _series_probe_cache.clear()
    _save_series_probe_cache()
    log.info("Cleared probe cache: %d URL entries, %d series entries", url_count, series_count)
    return url_count + series_count


def invalidate_series_probe_cache(series_id: int, episode_id: int | None = None) -> None:
    """Invalidate cached probe for series/episode.

    If episode_id is None, clears entire series. Otherwise clears just that episode.
    """
    with _probe_lock:
        if series_id not in _series_probe_cache:
            return
        if episode_id is None:
            del _series_probe_cache[series_id]
            log.info("Cleared probe cache for series=%d", series_id)
        else:
            series_data = _series_probe_cache[series_id]
            episodes = series_data.get("episodes", {})
            if episode_id in episodes:
                del episodes[episode_id]
                log.info(
                    "Cleared probe cache for series=%d episode=%d",
                    series_id,
                    episode_id,
                )
    _save_series_probe_cache()


def clear_series_mru(series_id: int) -> None:
    """Clear only the MRU for a series, keeping episode cache intact."""
    with _probe_lock:
        if series_id not in _series_probe_cache:
            return
        if "mru" in _series_probe_cache[series_id]:
            del _series_probe_cache[series_id]["mru"]
            log.info("Cleared MRU for series=%d", series_id)
    _save_series_probe_cache()


def restore_probe_cache_entry(
    url: str,
    media_info: MediaInfo,
    subs: list[SubtitleStream],
    series_id: int | None = None,
    episode_id: int | None = None,
) -> None:
    """Restore a probe cache entry (used during session recovery)."""
    now = time.time()
    with _probe_lock:
        if url not in _probe_cache:
            _probe_cache[url] = (now, media_info, subs)
        if series_id is not None:
            if series_id not in _series_probe_cache:
                _series_probe_cache[series_id] = {"name": "", "episodes": {}}
            _series_probe_cache[series_id].setdefault("episodes", {})
            eid = episode_id or 0
            if eid not in _series_probe_cache[series_id]["episodes"]:
                _series_probe_cache[series_id]["episodes"][eid] = (now, media_info, subs)


# ===========================================================================
# Media Probing
# ===========================================================================


def _lang_display_name(code: str) -> str:
    return _LANG_NAMES.get(code, code.upper())


def resolve_hls_master_playlist(url: str) -> str:
    """Resolve HLS master playlist to highest bandwidth variant URL.

    If the URL points to an HLS master playlist (contains #EXT-X-STREAM-INF),
    this fetches and parses it to find the variant with the highest bandwidth.
    Returns the resolved variant URL, or the original URL if not a master playlist
    or on any error.
    """
    from urllib.parse import urljoin

    import urllib.request

    if not url.endswith(".m3u8") and ".m3u8?" not in url:
        return url  # Not an m3u8, return as-is

    try:
        req = urllib.request.Request(url)
        user_agent = get_user_agent()
        if user_agent:
            req.add_header("User-Agent", user_agent)
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode("utf-8", errors="replace")

        # Check if this is a master playlist (has #EXT-X-STREAM-INF)
        if "#EXT-X-STREAM-INF" not in content:
            return url  # Not a master playlist

        # Parse variants: each #EXT-X-STREAM-INF is followed by a URL line
        lines = content.strip().split("\n")
        variants: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            if line.startswith("#EXT-X-STREAM-INF:"):
                # Extract BANDWIDTH from the tag
                bandwidth = 0
                for attr in line.split(":")[1].split(","):
                    if attr.startswith("BANDWIDTH="):
                        with suppress(ValueError):
                            bandwidth = int(attr.split("=")[1])
                        break
                # Next non-comment line is the variant URL
                for j in range(i + 1, len(lines)):
                    variant_line = lines[j].strip()
                    if variant_line and not variant_line.startswith("#"):
                        # Resolve relative URL
                        variant_url = urljoin(url, variant_line)
                        variants.append((bandwidth, variant_url))
                        break

        if not variants:
            log.warning("HLS master playlist has no variants: %s", url[:80])
            return url

        # Select highest bandwidth variant
        variants.sort(key=lambda x: x[0], reverse=True)
        best_bandwidth, best_url = variants[0]
        log.info(
            "HLS master playlist resolved: %d variants, selected %d bps: %s",
            len(variants),
            best_bandwidth,
            best_url[:80],
        )
        return best_url

    except Exception as e:
        log.warning("Failed to resolve HLS master playlist %s: %s", url[:80], e)
        return url


def probe_media(
    url: str,
    series_id: int | None = None,
    episode_id: int | None = None,
    series_name: str = "",
) -> tuple[MediaInfo | None, list[SubtitleStream]]:
    """Probe media, returns (media_info, subtitles)."""
    # Check series/episode cache first
    cache_hit_result: tuple[MediaInfo, list[SubtitleStream]] | None = None
    save_mru = False
    if series_id is not None:
        with _probe_lock:
            series_data = _series_probe_cache.get(series_id)
            if series_data:
                episodes = series_data.get("episodes", {})
                mru_eid = series_data.get("mru")
                # Try exact episode first
                if episode_id is not None and episode_id in episodes:
                    cache_time, media_info, subtitles = episodes[episode_id]
                    if time.time() - cache_time < _SERIES_PROBE_CACHE_TTL_SEC:
                        # Update MRU to this episode
                        if series_data.get("mru") != episode_id:
                            series_data["mru"] = episode_id
                            save_mru = True
                        log.info(
                            "Probe cache hit for series=%d episode=%d",
                            series_id,
                            episode_id,
                        )
                        cache_hit_result = (media_info, subtitles)
                # Fall back to MRU if set
                elif mru_eid is not None and mru_eid in episodes:
                    cache_time, media_info, subtitles = episodes[mru_eid]
                    if time.time() - cache_time < _SERIES_PROBE_CACHE_TTL_SEC:
                        log.info(
                            "Probe cache hit for series=%d (fallback from mru=%d)",
                            series_id,
                            mru_eid,
                        )
                        cache_hit_result = (media_info, subtitles)
        # Save MRU update outside the lock to avoid deadlock
        if save_mru:
            _save_series_probe_cache()
        if cache_hit_result:
            return cache_hit_result

    # Check URL cache (for movies, or series cache miss)
    with _probe_lock:
        cached = _probe_cache.get(url)
        if cached:
            cache_time, media_info, subtitles = cached
            if time.time() - cache_time < _PROBE_CACHE_TTL_SEC:
                log.info("Probe cache hit for %s", url[:50])
                return media_info, subtitles
    log.info(
        "Probe cache miss for %s (series=%s, episode=%s)",
        url[:50],
        series_id,
        episode_id,
    )

    # Build base probe command
    # MPEG-TS streams (HDHomeRun, live TV) need ~1MB to reach first keyframe
    # which contains the sequence header with dimensions. GOP at 15Mbps = ~1-2MB.
    base_cmd = [
        "ffprobe",
        "-probesize",
        "1000000",  # Had to increase for HDHomerun; was 50000.
        "-analyzeduration",
        "1500000",  # Had to increase for HDHomerun; was 500000.
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
    ]
    user_agent = get_user_agent()
    if user_agent:
        base_cmd.extend(["-user_agent", user_agent])

    # Try probe without forcing HLS first, retry with HLS options if it fails
    is_hls = False
    data = None
    for force_hls in (False, True):
        try:
            cmd = base_cmd.copy()
            if force_hls:
                cmd.extend(["-f", "hls", "-extension_picky", "0"])
            cmd.append(url)
            log.info("Probing%s: %s", " (HLS mode)" if force_hls else "", " ".join(cmd))
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_SEC,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Check detected format or if we forced HLS
                format_name = data.get("format", {}).get("format_name", "").lower()
                is_hls = force_hls or "hls" in format_name
                break
        except Exception as e:
            log.warning("Probe failed%s: %s", " (HLS mode)" if force_hls else "", e)
            continue

    if data is None:
        return None, []

    video_codec = audio_codec = pix_fmt = audio_profile = ""
    audio_channels = audio_sample_rate = 0
    subtitle_codecs: list[str] = []
    subtitles: list[SubtitleStream] = []

    height = 0
    video_bitrate = 0
    interlaced = False
    is_10bit = False
    is_hdr = False
    for stream in data.get("streams", []):
        codec = stream.get("codec_name", "").lower()
        codec_type = stream.get("codec_type", "")
        if codec_type == "video" and not video_codec:
            video_codec = codec
            pix_fmt = stream.get("pix_fmt", "")
            height = stream.get("height", 0) or 0
            # Detect interlacing from field_order (tt, bb, tb, bt = interlaced)
            field_order = stream.get("field_order", "").lower()
            interlaced = field_order in ("tt", "bb", "tb", "bt")
            # Detect 10-bit from pix_fmt (e.g. yuv420p10le, p010le)
            # Check for "p10" or "10le/10be" to avoid false positive on yuv410p
            is_10bit = "p10" in pix_fmt or "10le" in pix_fmt or "10be" in pix_fmt
            # Detect HDR from color_transfer (PQ = smpte2084, HLG = arib-std-b67)
            color_transfer = stream.get("color_transfer", "").lower()
            is_hdr = color_transfer in ("smpte2084", "arib-std-b67")
            # Try to get bitrate from stream, fall back to format
            with suppress(ValueError, TypeError):
                video_bitrate = int(stream.get("bit_rate", 0) or 0)
        elif codec_type == "audio" and not audio_codec:
            audio_codec = codec
            audio_channels = stream.get("channels", 0)
            audio_sample_rate = int(stream.get("sample_rate", 0) or 0)
            audio_profile = stream.get("profile", "")
        elif codec_type == "subtitle":
            subtitle_codecs.append(codec)
            if codec in TEXT_SUBTITLE_CODECS:
                idx = stream.get("index")
                if idx is not None:
                    tags = stream.get("tags", {})
                    lang = tags.get("language", "und").lower()
                    name = tags.get("name") or tags.get("title") or _lang_display_name(lang)
                    subtitles.append(
                        SubtitleStream(
                            index=idx,
                            lang=lang,
                            name=name,
                        )
                    )

    duration = 0.0
    fmt = data.get("format", {})
    if fmt.get("duration"):
        with suppress(ValueError, TypeError):
            duration = float(fmt["duration"])
    # Fall back to format bitrate if stream bitrate unavailable (common for MKV)
    if not video_bitrate and fmt.get("bit_rate"):
        with suppress(ValueError, TypeError):
            video_bitrate = int(fmt["bit_rate"])

    if not video_codec:
        return None, []

    media_info = MediaInfo(
        video_codec=video_codec,
        audio_codec=audio_codec,
        pix_fmt=pix_fmt,
        audio_channels=audio_channels,
        audio_sample_rate=audio_sample_rate,
        audio_profile=audio_profile,
        subtitle_codecs=subtitle_codecs or None,
        duration=duration,
        height=height,
        video_bitrate=video_bitrate,
        interlaced=interlaced,
        is_10bit=is_10bit,
        is_hdr=is_hdr,
        is_hls=is_hls,
    )
    # Only cache if we got valid video info (height > 0)
    if height <= 0:
        log.warning("Probe returned invalid height=%d, not caching: %s", height, url[:80])
        return media_info, subtitles
    with _probe_lock:
        _probe_cache[url] = (time.time(), media_info, subtitles)
        # Cache by series_id/episode_id if provided
        if series_id is not None:
            if series_id not in _series_probe_cache:
                _series_probe_cache[series_id] = {"name": series_name, "episodes": {}}
            elif not _series_probe_cache[series_id].get("name") and series_name:
                _series_probe_cache[series_id]["name"] = series_name
            eid = episode_id if episode_id is not None else 0
            _series_probe_cache[series_id].setdefault("episodes", {})[eid] = (
                time.time(),
                media_info,
                subtitles,
            )
            # Set MRU to this episode
            old_mru = _series_probe_cache[series_id].get("mru")
            _series_probe_cache[series_id]["mru"] = eid
            log.info(
                "Probe cached: series=%s episode=%s, mru changed from %s to %s",
                series_id,
                eid,
                old_mru,
                eid,
            )
    if series_id is not None:
        _save_series_probe_cache()
    return media_info, subtitles


# ===========================================================================
# FFmpeg Command Building
# ===========================================================================


def _build_video_args(
    *,
    copy_video: bool,
    hw: HwAccel,
    deinterlace: bool,
    use_hw_pipeline: bool,
    max_resolution: str,
    quality: str,
    is_hdr: bool = False,
    source_height: int = 0,
) -> tuple[list[str], list[str]]:
    """Build video args. Returns (pre_input_args, post_input_args)."""
    if copy_video:
        return [], ["-c:v", "copy"]

    # Parse hw into encoder and fallback
    enc_type, fallback = _parse_hw(hw)

    max_h = _MAX_RES_HEIGHT.get(max_resolution)

    # Check if SR should be applied (discrete GPUs only)
    sr_filter = ""
    sr_model = _load_settings().get("sr_model", "")
    if sr_model and enc_type in ("nvenc", "amf") and _sr_engine_dir:
        sr_filter = _build_sr_filter(source_height, max_h or 0)
        # SR requires CPU frames, so disable hw pipeline when SR active
        if sr_filter:
            use_hw_pipeline = False

    # Fall back gracefully if VAAPI is needed but no device was detected
    needs_vaapi = enc_type == "vaapi" or fallback == "vaapi"
    if needs_vaapi and not VAAPI_DEVICE:
        if enc_type == "vaapi":
            # Pure VAAPI encoder requested but not available - fall back to software
            log.warning("VAAPI unavailable (no Intel/AMD GPU), falling back to software encoding")
            enc_type = "software"
        else:
            # VAAPI fallback requested but not available - use software decode instead
            log.warning("VAAPI fallback unavailable (no Intel/AMD GPU), using software decode")
        fallback = "software"

    # Height expr for scale filter (scale down only, -2 keeps width divisible by 2)
    h = f"min(ih\\,{max_h})" if max_h else None
    qp = _QUALITY_QP.get(quality, 28)

    if enc_type == "nvenc":
        if use_hw_pipeline:
            # CUDA decode path
            pre = [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-extra_hw_frames",
                "3",
            ]
            scale = f"scale_cuda=-2:{h}:format=nv12" if h else "scale_cuda=format=nv12"
            deint = "yadif_cuda=0," if deinterlace else ""  # mode=0 keeps original framerate
            # HDR tone mapping: prefer libplacebo (Vulkan GPU), fall back to CPU zscale+tonemap
            if is_hdr:
                if _has_libplacebo_filter():
                    tonemap = "hwdownload,format=p010le,libplacebo=tonemapping=hable:colorspace=bt709:color_primaries=bt709:color_trc=bt709,format=nv12,hwupload_cuda,"
                else:
                    tonemap = "hwdownload,format=p010le,zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=nv12,hwupload_cuda,"
            else:
                tonemap = ""
            vf = f"{deint}{tonemap}{scale}"
        elif fallback == "vaapi":
            # VAAPI decode + VAAPI filters + hwdownload + hwupload_cuda for NVENC
            pre = [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-hwaccel_device",
                VAAPI_DEVICE,
            ]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            deint = "deinterlace_vaapi," if deinterlace else ""
            vf = f"{deint}{tonemap}{scale},hwdownload,format=nv12,hwupload_cuda"
        else:
            # Software decode, upload to GPU for scaling/encoding
            pre = []
            scale = f"scale_cuda=-2:{h}:format=nv12" if h else "scale_cuda=format=nv12"
            # HDR tone mapping: prefer libplacebo (Vulkan GPU), fall back to CPU zscale+tonemap
            # Deinterlace before tonemap (CPU yadif) for consistency with hw decode path
            if sr_filter:
                # SR path: CPU decode -> deinterlace -> SR (GPU) -> encode
                # SR filter ends with scale_cuda, outputs CUDA frames ready for nvenc
                # Need init_hw_device for TensorRT dnn_processing to use GPU
                pre = ["-init_hw_device", "cuda=cu", "-filter_hw_device", "cu"]
                deint = "yadif=0," if deinterlace else ""
                vf = f"{deint}{sr_filter}"
            elif is_hdr:
                deint = "yadif=0," if deinterlace else ""  # CPU deinterlace before tonemap
                if _has_libplacebo_filter():
                    tonemap = "libplacebo=tonemapping=hable:colorspace=bt709:color_primaries=bt709:color_trc=bt709,format=nv12,hwupload_cuda,"
                else:
                    tonemap = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=nv12,hwupload_cuda,"
                vf = f"{deint}{tonemap}{scale}"
            else:
                deint = "yadif_cuda=0," if deinterlace else ""  # GPU deinterlace after upload
                tonemap = "format=nv12,hwupload_cuda,"
                vf = f"{tonemap}{deint}{scale}"
        preset = "p4" if deinterlace or sr_filter else "p2"
        encoder = "h264_nvenc"
        # Lookahead for better quality, B-frames for compression, AQ for adaptive quantization
        enc_opts = [
            "-preset",
            preset,
            "-rc",
            "constqp",
            "-qp",
            str(qp),
            "-rc-lookahead",
            "32",
            "-bf",
            "3",
            "-spatial-aq",
            "1",
            "-temporal-aq",
            "1",
        ]

    elif enc_type == "amf":
        # AMF has no hardware decode - always uses fallback for decode/filter
        if fallback == "vaapi":
            # VAAPI decode + VAAPI filters + hwdownload for AMF encode
            pre = [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-hwaccel_device",
                VAAPI_DEVICE,
            ]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            deint = "deinterlace_vaapi," if deinterlace else ""
            vf = f"{deint}{tonemap}{scale},hwdownload,format=nv12"
        else:
            # Software decode + software filters
            pre = []
            if is_hdr:
                tonemap = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=nv12,"
            else:
                tonemap = ""
            deint = "yadif=0," if deinterlace else ""
            scale = f"scale=-2:{h}" if h else ""
            vf = f"{deint}{tonemap}{scale},format=nv12".strip(",").replace(",,", ",")
        encoder = "h264_amf"
        enc_opts = [
            "-rc",
            "cqp",
            "-qp_i",
            str(qp),
            "-qp_p",
            str(qp),
            "-quality",
            "balanced",
        ]

    elif enc_type == "vaapi":
        if use_hw_pipeline:
            pre = [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-hwaccel_device",
                VAAPI_DEVICE,
                "-extra_hw_frames",
                "3",
            ]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            # HDR tone mapping on VAAPI
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            vf = f"deinterlace_vaapi,{tonemap}{scale}" if deinterlace else f"{tonemap}{scale}"
        else:
            # Software decode, upload to GPU for scaling/encoding
            pre = ["-vaapi_device", VAAPI_DEVICE]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            deint = "deinterlace_vaapi," if deinterlace else ""
            vf = f"format=nv12,hwupload,{deint}{tonemap}{scale}"
        encoder = "h264_vaapi"
        # Use baseline profile for older GPUs (e.g., AMD GCN 1.0) that don't support High profile
        if AVAILABLE_ENCODERS.get("vaapi_baseline_only"):
            # Baseline doesn't support B-frames
            enc_opts = ["-rc_mode", "CQP", "-qp", str(qp), "-profile:v", "constrained_baseline"]
        else:
            enc_opts = ["-rc_mode", "CQP", "-qp", str(qp), "-bf", "3"]

    elif enc_type == "qsv":
        if use_hw_pipeline:
            pre = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
            scale = f"scale_qsv=w=-2:h={h}:format=nv12" if h else "scale_qsv=format=nv12"
            # Combine deinterlace and tonemap into single vpp_qsv call when possible
            if deinterlace and is_hdr:
                vf = f"vpp_qsv=deinterlace=2:tonemap=1:format=nv12,{scale}"
            elif deinterlace:
                vf = f"vpp_qsv=deinterlace=2,{scale}"
            elif is_hdr:
                vf = f"vpp_qsv=tonemap=1:format=nv12,{scale}"
            else:
                vf = scale
        else:
            # Software decode, upload to GPU for scaling/encoding
            pre = ["-init_hw_device", "qsv=hw", "-filter_hw_device", "hw"]
            scale = f"scale_qsv=w=-2:h={h}:format=nv12" if h else "scale_qsv=format=nv12"
            # Combine deinterlace and tonemap into single vpp_qsv call when possible
            if deinterlace and is_hdr:
                vf = f"format=nv12,hwupload=extra_hw_frames=64,vpp_qsv=deinterlace=2:tonemap=1:format=nv12,{scale}"
            elif deinterlace:
                vf = f"format=nv12,hwupload=extra_hw_frames=64,vpp_qsv=deinterlace=2,{scale}"
            elif is_hdr:
                vf = (
                    f"format=nv12,hwupload=extra_hw_frames=64,vpp_qsv=tonemap=1:format=nv12,{scale}"
                )
            else:
                vf = f"format=nv12,hwupload=extra_hw_frames=64,{scale}"
        encoder = "h264_qsv"
        enc_opts = [
            "-global_quality",
            str(qp),
            "-bf",
            "3",
            "-look_ahead",
            "1",
            "-look_ahead_depth",
            "40",
        ]

    elif enc_type == "software":
        pre = []
        # HDR tone mapping on CPU
        if is_hdr:
            tonemap = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
        else:
            tonemap = ""
        deint = "yadif=0," if deinterlace else ""  # mode=0 keeps original framerate
        if h:
            vf = f"{deint}{tonemap}scale=-2:{h},format=yuv420p"
        else:
            vf = f"{deint}{tonemap}format=yuv420p".rstrip(",")
        crf = _QUALITY_CRF.get(quality, 26)
        encoder = "libx264"
        enc_opts = ["-preset", "veryfast", "-crf", str(crf), "-bf", "3"]

    else:
        raise ValueError(f"Unrecognized hardware encoder: '{enc_type}'.")

    post = ["-vf", vf, "-c:v", encoder, *enc_opts, "-g", "60"]
    return pre, post


def _build_audio_args(*, copy_audio: bool, audio_sample_rate: int) -> list[str]:
    """Build audio args."""
    if copy_audio:
        return ["-c:a", "copy"]
    rate = str(audio_sample_rate) if audio_sample_rate in (44100, 48000) else "48000"
    return ["-c:a", "aac", "-ac", "2", "-ar", rate, "-b:a", "192k", "-profile:a", "aac_low"]


def get_live_hls_list_size() -> int:
    """Get hls_list_size for live streams based on DVR setting."""
    dvr_mins = _load_settings().get("live_dvr_mins", 0)
    if dvr_mins <= 0:
        # Default buffer when DVR disabled
        return int(DEFAULT_LIVE_BUFFER_SECS / _HLS_SEGMENT_DURATION_SEC)
    # DVR enabled: calculate segments from minutes
    return int(dvr_mins * 60 / _HLS_SEGMENT_DURATION_SEC)


def build_hls_ffmpeg_cmd(
    input_url: str,
    hw: HwAccel,
    output_dir: str,
    is_vod: bool = False,
    subtitles: list[SubtitleStream] | None = None,
    media_info: MediaInfo | None = None,
    max_resolution: str = "1080p",
    quality: str = "high",
    user_agent: str | None = None,
    deinterlace_fallback: bool | None = None,
) -> list[str]:
    """Build ffmpeg command for HLS transcoding."""
    # Check if we can copy streams directly (compatible codecs, no processing needed)
    max_h = _MAX_RES_HEIGHT.get(max_resolution, 9999)
    needs_scale = media_info and media_info.height > max_h

    # SR requires re-encode (can't copy video when SR is active)
    sr_active = bool(_sr_engine_dir and _load_settings().get("sr_model", ""))

    copy_video = bool(
        media_info
        and media_info.video_codec == "h264"
        and media_info.pix_fmt == "yuv420p"
        and not needs_scale
        and not sr_active
        and not media_info.interlaced  # Can't copy if deinterlacing needed
    )
    copy_audio = bool(
        media_info
        and media_info.audio_codec == "aac"
        and media_info.audio_channels <= 2
        and media_info.audio_sample_rate in (44100, 48000)
        # HE-AAC has browser compatibility issues - only copy LC-AAC
        and "HE" not in media_info.audio_profile
    )

    # Full hardware pipeline if GPU supports the codec
    # Parse hw to get encoder type
    enc_type, _ = _parse_hw(hw)
    codec = media_info.video_codec if media_info else ""
    use_hw_pipeline = bool(
        not copy_video
        and media_info
        and (
            (enc_type == "nvenc" and codec in _get_gpu_nvdec_codecs())
            or (enc_type == "vaapi" and codec in _VAAPI_SAFE_CODECS)
            or (enc_type == "qsv" and codec in _QSV_SAFE_CODECS)
            # AMF never has hw decode pipeline - always False
        )
    )

    # Deinterlace: use probe result if available, else use fallback setting
    # (fallback defaults to True for live, False for VOD when not explicitly set)
    fallback = deinterlace_fallback if deinterlace_fallback is not None else (not is_vod)
    # If probe failed (height=0), don't trust interlaced flag - use fallback
    probe_valid = media_info is not None and media_info.height > 0
    deinterlace = media_info.interlaced if probe_valid and media_info else fallback

    # Build component arg lists
    video_pre, video_post = _build_video_args(
        copy_video=copy_video,
        hw=hw,
        deinterlace=deinterlace,
        use_hw_pipeline=use_hw_pipeline,
        max_resolution=max_resolution,
        quality=quality,
        is_hdr=media_info.is_hdr if media_info else False,
        source_height=media_info.height if media_info else 0,
    )
    audio_args = _build_audio_args(
        copy_audio=copy_audio,
        audio_sample_rate=media_info.audio_sample_rate if media_info else 0,
    )

    # Base args
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-noautorotate",
    ]

    # Hwaccel args (before -i)
    cmd.extend(video_pre)

    # Probe args (only when no media_info, since we already probed)
    if media_info is None:
        probe_size = "50000" if is_vod else "5000000"
        analyze_dur = "500000" if is_vod else "5000000"
        cmd.extend(["-probesize", probe_size, "-analyzeduration", analyze_dur])

    # Input args
    cmd.extend(
        [
            "-fflags",
            "+discardcorrupt+genpts",
            "-err_detect",
            "ignore_err",
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_on_network_error",
            "1",
            "-reconnect_on_http_error",
            "4xx,5xx",
            "-reconnect_delay_max",
            "30",
        ]
    )
    if user_agent:
        cmd.extend(["-user_agent", user_agent])
    # Use HLS demuxer options if probe detected HLS format
    if media_info and media_info.is_hls:
        cmd.extend(["-f", "hls", "-extension_picky", "0"])
    cmd.extend(["-i", input_url])

    # Subtitle extraction
    for i, sub in enumerate(subtitles or []):
        cmd.extend(
            [
                "-map",
                f"0:{sub.index}",
                "-c:s",
                "webvtt",
                "-flush_packets",
                "1",
                f"{output_dir}/sub{i}.vtt",
            ]
        )

    # Stream mapping + video + audio
    cmd.extend(["-map", "0:v:0", "-map", "0:a:0"])
    cmd.extend(video_post)
    cmd.extend(audio_args)

    # HLS output args
    cmd.extend(
        [
            "-max_delay",
            "5000000",
            "-f",
            "hls",
            "-hls_time",
            str(int(_HLS_SEGMENT_DURATION_SEC)),
            "-hls_list_size",
            "0" if is_vod else str(get_live_hls_list_size()),
            "-hls_segment_filename",
            f"{output_dir}/{SEG_PREFIX}%03d.ts",
        ]
    )
    if is_vod:
        cmd.extend(
            [
                "-hls_init_time",
                "2",
                "-hls_flags",
                "independent_segments",
                "-hls_playlist_type",
                "event",
            ]
        )
    else:
        cmd.extend(["-hls_flags", "delete_segments"])

    cmd.append(f"{output_dir}/stream.m3u8")
    return cmd
