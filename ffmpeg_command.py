"""FFmpeg command building and media probing."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

import json
import logging
import pathlib
import subprocess
import tempfile
import threading
import time


log = logging.getLogger(__name__)

HwAccel = Literal["nvidia", "intel", "vaapi", "software"]

# Timing constants
_HLS_SEGMENT_DURATION_SEC = 3.0  # Short segments for faster startup/seeking
_PROBE_CACHE_TTL_SEC = 3_600
_SERIES_PROBE_CACHE_TTL_SEC = float("inf")  # Never expire
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

# User-Agent presets
_USER_AGENT_PRESETS = {
    "vlc": "VLC/3.0.20 LibVLC/3.0.20",
    "chrome": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "tivimate": "TiviMate/4.7.0",
}

# NVDEC capabilities by minimum compute capability
# https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
_NVDEC_MIN_COMPUTE: dict[str, float] = {
    "h264": 5.0,  # Maxwell+
    "hevc": 6.0,  # Pascal+ (HEVC 10-bit requires Pascal; Maxwell GM206 is edge case we ignore)
    "av1": 8.0,  # Ampere+
}

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
_load_settings: Callable[[], dict[str, Any]] = dict

_CACHE_DIR = pathlib.Path(__file__).parent / "cache"
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
    subtitle_codecs: list[str] | None = None
    duration: float = 0.0
    height: int = 0
    video_bitrate: int = 0  # bits per second, 0 if unknown
    interlaced: bool = False  # True if field_order indicates interlaced


def init(load_settings: Callable[[], dict[str, Any]]) -> None:
    """Initialize module with settings loader."""
    global _load_settings
    _load_settings = load_settings
    _load_series_probe_cache()


def get_settings() -> dict[str, Any]:
    """Get current settings."""
    return _load_settings()


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


# ===========================================================================
# User-Agent
# ===========================================================================


def get_user_agent() -> str | None:
    """Get user-agent string from settings, or None to use FFmpeg default."""
    settings = _load_settings()
    preset = settings.get("user_agent_preset", "default")
    if preset == "default":
        return None
    if preset == "custom":
        return settings.get("user_agent_custom") or None
    return _USER_AGENT_PRESETS.get(preset)


# ===========================================================================
# Transcode Directory
# ===========================================================================


def get_transcode_dir() -> pathlib.Path:
    """Get the transcode output directory. Falls back to system temp if not set."""
    custom_dir = _load_settings().get("transcode_dir", "")
    if custom_dir:
        path = pathlib.Path(custom_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
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
                    media_info = MediaInfo(
                        video_codec=entry["video_codec"],
                        audio_codec=entry["audio_codec"],
                        pix_fmt=entry["pix_fmt"],
                        audio_channels=entry.get("audio_channels", 0),
                        audio_sample_rate=entry.get("audio_sample_rate", 0),
                        subtitle_codecs=entry.get("subtitle_codecs"),
                        duration=entry.get("duration", 0),
                        height=entry.get("height", 0),
                        video_bitrate=entry.get("video_bitrate", 0),
                        interlaced=entry.get("interlaced", False),
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
                    "subtitles": [{"index": s.index, "lang": s.lang, "name": s.name} for s in subs],
                }
    try:
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

    try:
        cmd = [
            "ffprobe",
            "-probesize",
            "50000",
            "-analyzeduration",
            "500000",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ]
        user_agent = get_user_agent()
        if user_agent:
            cmd.extend(["-user_agent", user_agent])
        cmd.append(url)
        log.info("Probing: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            return None, []
        data = json.loads(result.stdout)
    except Exception as e:
        log.warning("Failed to probe media: %s", e)
        return None, []

    video_codec = audio_codec = pix_fmt = ""
    audio_channels = audio_sample_rate = 0
    subtitle_codecs: list[str] = []
    subtitles: list[SubtitleStream] = []

    height = 0
    video_bitrate = 0
    interlaced = False
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
            # Try to get bitrate from stream, fall back to format
            with suppress(ValueError, TypeError):
                video_bitrate = int(stream.get("bit_rate", 0) or 0)
        elif codec_type == "audio" and not audio_codec:
            audio_codec = codec
            audio_channels = stream.get("channels", 0)
            audio_sample_rate = int(stream.get("sample_rate", 0) or 0)
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
        subtitle_codecs=subtitle_codecs or None,
        duration=duration,
        height=height,
        video_bitrate=video_bitrate,
        interlaced=interlaced,
    )
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
) -> tuple[list[str], list[str]]:
    """Build video args. Returns (pre_input_args, post_input_args)."""
    if copy_video:
        return [], ["-c:v", "copy"]

    # Height expr for scale filter (scale down only, -2 keeps width divisible by 2)
    max_h = _MAX_RES_HEIGHT.get(max_resolution)
    h = f"'min(ih,{max_h})'" if max_h else None
    qp = _QUALITY_QP.get(quality, 28)

    if hw == "nvidia":
        if use_hw_pipeline:
            pre = [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-extra_hw_frames",
                "3",
            ]
            scale = f"scale_cuda=-2:{h}:format=nv12" if h else "scale_cuda=format=nv12"
            vf = f"yadif_cuda=1,{scale}" if deinterlace else scale
        else:
            pre = []
            if deinterlace:
                vf = f"yadif=1,scale=-2:{h}" if h else "yadif=1"
            else:
                vf = f"scale=-2:{h},format=nv12" if h else "format=nv12"
        preset = "p4" if deinterlace else "p2"
        encoder = "h264_nvenc"
        enc_opts = ["-preset", preset, "-rc", "constqp", "-qp", str(qp)]

    elif hw == "vaapi":
        pre = [
            "-hwaccel",
            "vaapi",
            "-hwaccel_output_format",
            "vaapi",
            "-hwaccel_device",
            "/dev/dri/renderD128",
        ]
        scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
        vf = f"deinterlace_vaapi,{scale}" if deinterlace else scale
        encoder = "h264_vaapi"
        enc_opts = ["-rc_mode", "CQP", "-qp", str(qp)]

    elif hw == "intel":
        pre = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
        scale = f"scale_qsv=w=-2:h={h}:format=nv12" if h else "scale_qsv=format=nv12"
        vf = f"vpp_qsv=deinterlace=2,{scale}" if deinterlace else scale
        encoder = "h264_qsv"
        enc_opts = ["-preset", "medium", "-global_quality", str(qp)]

    elif hw == "software":
        pre = []
        if deinterlace:
            vf = f"yadif=1,scale=-2:{h}" if h else "yadif=1"
        else:
            vf = f"scale=-2:{h},format=yuv420p" if h else "format=yuv420p"
        crf = _QUALITY_CRF.get(quality, 26)
        encoder = "libx264"
        enc_opts = ["-preset", "veryfast", "-crf", str(crf)]

    else:
        raise ValueError(f"Unrecognized hardware: '{hw}'.")

    post = ["-vf", vf, "-c:v", encoder, *enc_opts, "-g", "60"]
    return pre, post


def _build_audio_args(*, copy_audio: bool, audio_sample_rate: int) -> list[str]:
    """Build audio args."""
    if copy_audio:
        return ["-c:a", "copy"]
    rate = str(audio_sample_rate) if audio_sample_rate in (44100, 48000) else "48000"
    return ["-c:a", "aac", "-ac", "2", "-ar", rate, "-b:a", "192k"]


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
    # Check if we can copy streams directly (VOD with compatible codecs)
    max_h = _MAX_RES_HEIGHT.get(max_resolution, 9999)
    needs_scale = media_info and media_info.height > max_h
    copy_video = bool(
        is_vod
        and media_info
        and media_info.video_codec == "h264"
        and media_info.pix_fmt == "yuv420p"
        and not needs_scale
    )
    copy_audio = bool(
        is_vod
        and media_info
        and media_info.audio_codec == "aac"
        and media_info.audio_channels <= 2
        and media_info.audio_sample_rate in (44100, 48000)
    )

    # Full hardware pipeline if GPU supports the codec
    use_hw_pipeline = bool(
        not copy_video
        and hw in ("nvidia", "intel", "vaapi")
        and (hw != "nvidia" or (media_info and media_info.video_codec in _get_gpu_nvdec_codecs()))
    )

    # Deinterlace: use probe result if available, else use fallback setting
    # (fallback defaults to True for live, False for VOD when not explicitly set)
    fallback = deinterlace_fallback if deinterlace_fallback is not None else (not is_vod)
    deinterlace = media_info.interlaced if media_info else fallback

    # Build component arg lists
    video_pre, video_post = _build_video_args(
        copy_video=copy_video,
        hw=hw,
        deinterlace=deinterlace,
        use_hw_pipeline=use_hw_pipeline,
        max_resolution=max_resolution,
        quality=quality,
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

    # Probe args (when no media_info)
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
