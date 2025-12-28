"""HLS transcoding with ffmpeg and session management."""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException


log = logging.getLogger(__name__)

TEXT_SUBTITLE_CODECS = {
    "subrip",
    "ass",
    "ssa",
    "mov_text",
    "webvtt",
    "srt",
}

# Timing constants (seconds)
_POLL_INTERVAL_SEC = 0.2
_HLS_SEGMENT_DURATION_SEC = 3.0  # Short segments for faster startup/seeking
_PROBE_CACHE_TTL_SEC = 3_600
_SERIES_PROBE_CACHE_TTL_SEC = float("inf")  # Never expire
_PROBE_TIMEOUT_SEC = 30
_QUICK_FAILURE_THRESHOLD_SEC = 10.0

# Wait timeouts (seconds) - converted to iterations via _POLL_INTERVAL_SEC
_PLAYLIST_WAIT_TIMEOUT_SEC = 30.0
_PLAYLIST_WAIT_SEEK_TIMEOUT_SEC = 40.0
_REUSE_ACTIVE_WAIT_TIMEOUT_SEC = 15.0
_RESUME_WAIT_TIMEOUT_SEC = 10.0
_RESUME_SEGMENT_WAIT_TIMEOUT_SEC = 5.0

# Size thresholds (bytes)
_MIN_SEGMENT_SIZE_BYTES = 1_000

_transcode_sessions: dict[str, dict[str, Any]] = {}
_vod_url_to_session: dict[str, str] = {}
_transcode_lock = threading.Lock()
_probe_lock = threading.Lock()
_background_tasks: set[asyncio.Task[None]] = set()
_load_settings: Callable[[], dict[str, Any]] = dict
_probe_cache: dict[str, tuple[float, MediaInfo | None, list[SubtitleStream]]] = {}

# NVDEC capabilities by minimum compute capability
# https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
_NVDEC_MIN_COMPUTE: dict[str, float] = {
    "h264": 5.0,  # Maxwell+
    "hevc": 6.0,  # Pascal+ (HEVC 10-bit requires Pascal; Maxwell GM206 is edge case we ignore)
    "av1": 8.0,  # Ampere+
}
_gpu_nvdec_codecs: set[str] | None = None  # None = not probed yet
# Series probe cache: {series_id: {"name": str, "episodes": {episode_id: (time, media_info, subtitles)}}}
_series_probe_cache: dict[int, dict[str, Any]] = {}
_CACHE_DIR = pathlib.Path(__file__).parent / "cache"
_SERIES_PROBE_CACHE_FILE = _CACHE_DIR / "series_probe_cache.json"


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
            "GPU: %s (compute %.1f) NVDEC: %s", gpu_name, compute_cap, _gpu_nvdec_codecs or "none"
        )
    except Exception as e:
        log.debug("GPU probe failed: %s", e)
    return _gpu_nvdec_codecs


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
                    "subtitles": [{"index": s.index, "lang": s.lang, "name": s.name} for s in subs],
                }
    try:
        _SERIES_PROBE_CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.warning("Failed to save series probe cache: %s", e)


def init(load_settings: Callable[[], dict[str, Any]]) -> None:
    global _load_settings
    _load_settings = load_settings
    _load_series_probe_cache()


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


class _DeadProcess:
    """Placeholder for dead/recovered processes."""

    returncode = -1

    def kill(self) -> None:
        pass


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


def _lang_display_name(code: str) -> str:
    return _LANG_NAMES.get(code, code.upper())


def get_series_probe_cache_stats() -> list[dict[str, Any]]:
    """Get stats about cached series probes for settings UI."""
    with _probe_lock:
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
                log.info("Cleared probe cache for series=%d episode=%d", series_id, episode_id)
    _save_series_probe_cache()


def is_browser_compatible(
    url: str,
    series_id: int | None = None,
    episode_id: int | None = None,
) -> bool | None:
    """Check probe cache for browser-compatible codecs (h264+aac).

    Returns True if compatible, False if not, None if not in cache.
    """
    with _probe_lock:
        media_info = None
        # Check series cache first
        if series_id is not None:
            series_data = _series_probe_cache.get(series_id)
            if series_data:
                episodes = series_data.get("episodes", {})
                if episode_id is not None and episode_id in episodes:
                    cache_time, media_info, _ = episodes[episode_id]
                    if time.time() - cache_time >= _SERIES_PROBE_CACHE_TTL_SEC:
                        media_info = None
                elif mru_eid := series_data.get("mru"):
                    if mru_eid in episodes:
                        cache_time, media_info, _ = episodes[mru_eid]
                        if time.time() - cache_time >= _SERIES_PROBE_CACHE_TTL_SEC:
                            media_info = None
        # Check URL cache
        if media_info is None:
            cached = _probe_cache.get(url)
            if cached:
                cache_time, media_info, _ = cached
                if time.time() - cache_time >= _PROBE_CACHE_TTL_SEC:
                    media_info = None
        if media_info is None:
            return None
        return media_info.video_codec == "h264" and media_info.audio_codec == "aac"


def probe_media(
    url: str,
    series_id: int | None = None,
    episode_id: int | None = None,
    series_name: str = "",
) -> tuple[MediaInfo | None, list[SubtitleStream]]:
    """Probe media, returns (media_info, subtitles)."""
    # Check series/episode cache first
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
                        series_data["mru"] = episode_id
                        log.info("Probe cache hit for series=%d episode=%d", series_id, episode_id)
                        return media_info, subtitles
                # Fall back to MRU if set
                if mru_eid is not None and mru_eid in episodes:
                    cache_time, media_info, subtitles = episodes[mru_eid]
                    if time.time() - cache_time < _SERIES_PROBE_CACHE_TTL_SEC:
                        log.info(
                            "Probe cache hit for series=%d (fallback from mru=%d)",
                            series_id,
                            mru_eid,
                        )
                        return media_info, subtitles

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
            url,
        ]
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
    for stream in data.get("streams", []):
        codec = stream.get("codec_name", "").lower()
        codec_type = stream.get("codec_type", "")
        if codec_type == "video" and not video_codec:
            video_codec = codec
            pix_fmt = stream.get("pix_fmt", "")
            height = stream.get("height", 0) or 0
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
            _series_probe_cache[series_id]["mru"] = eid
    if series_id is not None:
        _save_series_probe_cache()
    return media_info, subtitles


def _get_thread_count(copy_video: bool, is_vod: bool, hw: str) -> str:
    if copy_video:
        return "2"
    if is_vod:
        return "3" if hw == "nvidia" else "2"
    return "4"


_MAX_RES_HEIGHT: dict[str, int] = {
    "4k": 2160,
    "1080p": 1080,
    "720p": 720,
    "480p": 480,
}

_DEFAULT_BITRATE_MBPS: dict[str, float] = {
    "4k": 20,
    "1080p": 6,
    "720p": 3,
    "480p": 1.5,
}


def _build_video_args(
    cmd: list[str],
    copy_video: bool,
    hw: str,
    is_vod: bool,
    use_full_cuda: bool,
    max_resolution: str,
    max_bitrate: str,
) -> None:
    if copy_video:
        cmd.extend(["-c:v", "copy"])
        return

    # Build scale filter if max_resolution is set (scale down only, preserve aspect)
    max_h = _MAX_RES_HEIGHT.get(max_resolution)
    # scale='min(iw,ih*16/9)':min(ih,MAX_H) - but simpler: scale=-2:MIN(ih,MAX_H)
    # Using -2 keeps width divisible by 2, and min() only scales down
    scale_expr = f"'min(ih,{max_h})'" if max_h else None

    if hw == "nvidia":
        if use_full_cuda:
            if is_vod:
                if scale_expr:
                    cmd.extend(
                        [
                            "-vf",
                            f"scale_cuda=-2:{scale_expr}:format=yuv420p",
                        ]
                    )
                else:
                    cmd.extend(["-vf", "scale_cuda=format=yuv420p"])
            else:
                if scale_expr:
                    cmd.extend(
                        [
                            "-vf",
                            f"yadif_cuda=1,scale_cuda=-2:{scale_expr}:format=yuv420p",
                        ]
                    )
                else:
                    cmd.extend(
                        [
                            "-vf",
                            "yadif_cuda=1,scale_cuda=format=yuv420p",
                        ]
                    )
        elif is_vod:
            if scale_expr:
                cmd.extend(
                    [
                        "-vf",
                        f"scale=-2:{scale_expr},format=yuv420p",
                    ]
                )
            else:
                cmd.extend(
                    [
                        "-vf",
                        "format=yuv420p",
                    ]
                )
        else:
            if scale_expr:
                cmd.extend(
                    [
                        "-vf",
                        f"yadif=1,scale=-2:{scale_expr}",
                    ]
                )
            else:
                cmd.extend(["-vf", "yadif=1"])
        preset = "p2" if is_vod else "p4"
        cmd.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                preset,
                "-b:v",
                max_bitrate,
                "-g",
                "60",
            ]
        )
    elif hw == "vaapi":
        if is_vod:
            if scale_expr:
                vf = f"scale=-2:{scale_expr},format=nv12,hwupload"
            else:
                vf = "format=nv12,hwupload"
        else:
            if scale_expr:
                vf = f"yadif=1,scale=-2:{scale_expr},format=nv12,hwupload"
            else:
                vf = "yadif=1,format=nv12,hwupload"
        cmd.extend(
            [
                "-vf",
                vf,
                "-vaapi_device",
                "/dev/dri/renderD128",
                "-c:v",
                "h264_vaapi",
                "-b:v",
                max_bitrate,
                "-g",
                "60",
            ]
        )
    elif hw == "qsv":
        if is_vod:
            if scale_expr:
                vf = f"scale=-2:{scale_expr},format=yuv420p"
            else:
                vf = "format=yuv420p"
        else:
            if scale_expr:
                vf = f"yadif=1,scale=-2:{scale_expr}"
            else:
                vf = "yadif=1"
        cmd.extend(
            [
                "-vf",
                vf,
                "-c:v",
                "h264_qsv",
                "-preset",
                "medium",
                "-b:v",
                max_bitrate,
                "-g",
                "60",
            ]
        )
    else:
        if is_vod:
            if scale_expr:
                vf = f"scale=-2:{scale_expr},format=yuv420p"
            else:
                vf = "format=yuv420p"
        else:
            if scale_expr:
                vf = f"yadif=1,scale=-2:{scale_expr}"
            else:
                vf = "yadif=1"
        cmd.extend(
            [
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-b:v",
                max_bitrate,
                "-g",
                "60",
            ]
        )


def _build_audio_args(
    cmd: list[str],
    copy_audio: bool,
    media_info: MediaInfo | None,
) -> None:
    if copy_audio:
        cmd.extend(["-c:a", "copy"])
        return
    sample_rate = "48000"
    if media_info and media_info.audio_sample_rate in (44100, 48000):
        sample_rate = str(media_info.audio_sample_rate)
    cmd.extend(
        [
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-ar",
            sample_rate,
            "-b:a",
            "192k",
        ]
    )


def build_hls_ffmpeg_cmd(
    input_url: str,
    hw: str,
    output_dir: str,
    is_vod: bool = False,
    subtitles: list[SubtitleStream] | None = None,
    media_info: MediaInfo | None = None,
    max_resolution: str = "1080p",
    max_bitrate_mbps: float = 0,
) -> list[str]:
    # Check if we need to scale down
    max_h = _MAX_RES_HEIGHT.get(max_resolution, 9999)
    needs_scale = media_info and media_info.height > max_h
    # Check if source bitrate exceeds max (0 = unlimited)
    max_bps = int(max_bitrate_mbps * 1_000_000) if max_bitrate_mbps > 0 else 0
    source_bps = media_info.video_bitrate if media_info else 0
    needs_bitrate_limit = max_bps > 0 and source_bps > max_bps

    copy_video = bool(
        is_vod
        and media_info
        and media_info.video_codec == "h264"
        and media_info.pix_fmt == "yuv420p"
        and not needs_scale  # Can't copy if we need to scale down
        and not needs_bitrate_limit  # Can't copy if source exceeds max bitrate
    )
    copy_audio = bool(
        is_vod
        and media_info
        and media_info.audio_codec == "aac"
        and media_info.audio_channels <= 2
        and media_info.audio_sample_rate in (44100, 48000)
    )
    # Full CUDA pipeline (hwaccel decode + GPU filter + nvenc) if GPU supports the codec
    use_full_cuda = bool(
        hw == "nvidia"
        and not copy_video
        and media_info
        and media_info.video_codec in _get_gpu_nvdec_codecs()
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-noautorotate",
        "-threads",
        _get_thread_count(copy_video, is_vod, hw),
    ]

    if use_full_cuda:
        cmd.extend(
            [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-extra_hw_frames",
                "3",
            ]
        )

    if media_info is None:
        cmd.extend(
            [
                "-probesize",
                "50000" if is_vod else "5000000",
                "-analyzeduration",
                "500000" if is_vod else "5000000",
            ]
        )
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
            "-reconnect_delay_max",
            "2",
            "-i",
            input_url,
        ]
    )

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

    cmd.extend(["-map", "0:v:0", "-map", "0:a:0"])
    # Use provided bitrate, or default based on resolution (format as "XM" for ffmpeg)
    effective_mbps = (
        max_bitrate_mbps if max_bitrate_mbps > 0 else _DEFAULT_BITRATE_MBPS.get(max_resolution, 6)
    )
    bitrate_str = f"{effective_mbps}M"
    _build_video_args(
        cmd,
        copy_video,
        hw,
        is_vod,
        use_full_cuda,
        max_resolution,
        bitrate_str,
    )
    _build_audio_args(cmd, copy_audio, media_info)

    cmd.extend(
        [
            "-max_delay",
            "5000000",
            "-f",
            "hls",
            "-hls_time",
            str(int(_HLS_SEGMENT_DURATION_SEC)),
            "-hls_list_size",
            "0" if is_vod else "10",
            "-hls_segment_filename",
            f"{output_dir}/seg%03d.ts",
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


def get_vod_cache_timeout() -> int:
    return _load_settings().get("vod_transcode_cache_mins", 60) * 60


def is_vod_session_valid(session: dict[str, Any]) -> bool:
    if not session.get("is_vod"):
        return False
    cache_timeout = get_vod_cache_timeout()
    if cache_timeout <= 0:
        return False
    age = time.time() - session.get("last_access", session["started"])
    return age < cache_timeout


def _kill_process(proc: Any) -> bool:
    """Kill process, return True if killed."""
    try:
        proc.kill()
        return True
    except (ProcessLookupError, OSError):
        return False


def stop_session(session_id: str, force: bool = False) -> None:
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if not session:
            return

        # Skip stop if session was accessed recently (race with page navigation)
        # The beacon from old page may arrive after new page starts using session
        if not force and time.time() - session.get("last_access", 0) < 5.0:
            log.info(
                "Ignoring stop for recently-accessed session %s",
                session_id,
            )
            return

        if _kill_process(session["process"]):
            log.info("Killed ffmpeg for session %s", session_id)

        if session.get("is_vod") and not force and get_vod_cache_timeout() > 0:
            session["last_access"] = time.time()
            log.info(
                "VOD session %s cached (ffmpeg stopped, segments kept)",
                session_id,
            )
            return

        _transcode_sessions.pop(session_id, None)
        url = session.get("url")
        if url:
            _vod_url_to_session.pop(url, None)
        dir_to_remove = session["dir"]

    shutil.rmtree(dir_to_remove, ignore_errors=True)
    log.info("Stopped transcode session %s", session_id)


def cleanup_expired_vod_sessions() -> None:
    with _transcode_lock:
        expired = [
            sid
            for sid, session in list(_transcode_sessions.items())
            if session.get("is_vod") and not is_vod_session_valid(session)
        ]
    for session_id in expired:
        stop_session(session_id, force=True)


def recover_vod_sessions() -> None:
    cache_timeout = get_vod_cache_timeout()
    now = time.time()
    for d in pathlib.Path("/tmp").glob("netv_transcode_*"):
        if not d.is_dir():
            continue
        info_file = d / "session.json"
        try:
            mtime = d.stat().st_mtime
        except OSError:
            shutil.rmtree(d, ignore_errors=True)
            log.info("Removed orphaned transcode dir %s", d)
            continue

        if now - mtime > cache_timeout or not info_file.exists():
            shutil.rmtree(d, ignore_errors=True)
            log.info("Removed expired/orphaned transcode dir %s", d)
            continue

        if not list(d.glob("seg*.ts")):
            shutil.rmtree(d, ignore_errors=True)
            log.info("Removed segmentless transcode dir %s", d)
            continue

        try:
            info = json.loads(info_file.read_text())
            if not (info.get("is_vod") and info.get("url")):
                continue
            session_id = info["session_id"]
            url = info["url"]
            new_seek = info.get("seek_offset", 0)

            with _transcode_lock:
                _transcode_sessions[session_id] = {
                    "dir": str(d),
                    "process": _DeadProcess(),
                    "started": info.get("started", mtime),
                    "url": url,
                    "is_vod": True,
                    "last_access": mtime,
                    "subtitles": info.get("subtitles") or info.get("subtitle_indices"),
                    "duration": info.get("duration", 0),
                    "seek_offset": new_seek,
                    "series_id": info.get("series_id"),
                    "episode_id": info.get("episode_id"),
                }
                # Prefer session with seek_offset or more recent mtime
                existing_id = _vod_url_to_session.get(url)
                if existing_id:
                    existing = _transcode_sessions.get(existing_id, {})
                    existing_seek = existing.get("seek_offset", 0)
                    existing_mtime = existing.get("last_access", 0)
                    if (new_seek > 0 and existing_seek == 0) or (
                        existing_seek == 0 and new_seek == 0 and mtime > existing_mtime
                    ):
                        _vod_url_to_session[url] = session_id
                else:
                    _vod_url_to_session[url] = session_id

            # Restore probe cache (outside transcode lock, uses probe lock)
            if p := info.get("probe"):
                media_info = MediaInfo(
                    video_codec=p.get("video_codec", ""),
                    audio_codec=p.get("audio_codec", ""),
                    pix_fmt=p.get("pix_fmt", ""),
                    audio_channels=p.get("audio_channels", 0),
                    audio_sample_rate=p.get("audio_sample_rate", 0),
                    subtitle_codecs=p.get("subtitle_codecs"),
                    duration=info.get("duration", 0),
                )
                subs = [
                    SubtitleStream(s["index"], s.get("lang", "und"), s.get("name", ""))
                    for s in (info.get("subtitles") or [])
                    if isinstance(s, dict) and "index" in s
                ]
                with _probe_lock:
                    if url not in _probe_cache:
                        _probe_cache[url] = (now, media_info, subs)
                    # Restore series cache if series_id present
                    if sid := info.get("series_id"):
                        if sid not in _series_probe_cache:
                            _series_probe_cache[sid] = {"name": "", "episodes": {}}
                        _series_probe_cache[sid].setdefault("episodes", {})
                        eid = info.get("episode_id") or 0
                        if eid not in _series_probe_cache[sid]["episodes"]:
                            _series_probe_cache[sid]["episodes"][eid] = (now, media_info, subs)
            log.info("Recovered VOD session %s for %s", session_id, url[:50])
        except Exception as e:
            log.warning("Failed to recover session from %s: %s", d, e)


async def _monitor_ffmpeg_stderr(
    process: asyncio.subprocess.Process,
    session_id: str,
    stderr_lines: list[str] | None = None,
) -> None:
    assert process.stderr is not None
    while True:
        line = await process.stderr.readline()
        if not line:
            break
        text = line.decode().rstrip()
        if stderr_lines is not None:
            stderr_lines.append(text)
        # Only log actual fatal errors as WARNING, not decoder warnings
        is_fatal = "fatal" in text.lower() or "aborting" in text.lower()
        level = logging.WARNING if is_fatal else logging.DEBUG
        log.log(level, "ffmpeg:%s %s", session_id, text)


async def _monitor_resume_ffmpeg(
    process: asyncio.subprocess.Process,
    session_id: str,
    url: str,
) -> None:
    start_time = time.time()
    await _monitor_ffmpeg_stderr(process, session_id)
    await process.wait()
    if process.returncode != 0:
        log.warning(
            "Resume ffmpeg exited with code %s for session %s",
            process.returncode,
            session_id,
        )
        if time.time() - start_time < _QUICK_FAILURE_THRESHOLD_SEC:
            log.info("Resume failed quickly, invalidating session %s", session_id)
            with _transcode_lock:
                _vod_url_to_session.pop(url, None)
                _transcode_sessions.pop(session_id, None)


async def _monitor_seek_ffmpeg(
    process: asyncio.subprocess.Process,
    session_id: str,
) -> None:
    await _monitor_ffmpeg_stderr(process, session_id)
    await process.wait()
    if process.returncode != 0:
        log.warning(
            "Seek ffmpeg exited with code %s for session %s",
            process.returncode,
            session_id,
        )


def _spawn_background_task(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _build_subtitle_tracks(
    session_id: str,
    sub_info: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not sub_info or not isinstance(sub_info[0], dict):
        return []
    return [
        {
            "url": f"/subs/{session_id}/sub{i}.vtt",
            "lang": s["lang"],
            "label": s["name"],
            "default": i == 0,
        }
        for i, s in enumerate(sub_info)
    ]


async def _wait_for_playlist(
    playlist_path: pathlib.Path,
    process: asyncio.subprocess.Process,
    min_segments: int = 1,
    timeout_sec: float = _PLAYLIST_WAIT_TIMEOUT_SEC,
) -> bool:
    """Wait for playlist with min_segments, checking process health."""
    output_dir = playlist_path.parent
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.returncode is not None:
            return False
        if playlist_path.exists():
            content = playlist_path.read_text()
            seg_count = content.count("#EXTINF")
            if seg_count >= min_segments:
                seg_files = list(output_dir.glob("seg*.ts"))
                if len(seg_files) >= min_segments:
                    first_seg = min(seg_files, key=lambda f: f.name)
                    if (
                        first_seg.stat().st_size > _MIN_SEGMENT_SIZE_BYTES
                        and process.returncode is None
                    ):
                        return True
        await asyncio.sleep(_POLL_INTERVAL_SEC)
    return False


def _calc_hls_duration(playlist_path: pathlib.Path, segment_count: int) -> float:
    """Calculate HLS duration from playlist or estimate from segment count."""
    if playlist_path.exists():
        durations = re.findall(
            r"#EXTINF:([\d.]+)",
            playlist_path.read_text(),
        )
        if durations:
            return sum(float(d) for d in durations)
    return segment_count * _HLS_SEGMENT_DURATION_SEC


@dataclass(slots=True)
class _SessionSnapshot:
    """Immutable snapshot of session state for lock-free access."""

    output_dir: str
    process: Any
    seek_offset: float
    subtitles: list[dict[str, Any]]
    duration: float


def _build_session_response(
    session_id: str,
    snap: _SessionSnapshot,
    playlist_path: pathlib.Path,
) -> dict[str, Any]:
    """Build response dict for existing session, recalculating duration."""
    segments = list(playlist_path.parent.glob("seg*.ts"))
    return {
        "session_id": session_id,
        "playlist": f"/transcode/{session_id}/stream.m3u8",
        "subtitles": _build_subtitle_tracks(session_id, snap.subtitles),
        "duration": snap.duration,
        "seek_offset": snap.seek_offset,
        "transcoded_duration": _calc_hls_duration(playlist_path, len(segments)),
    }


def _get_session_snapshot(session_id: str) -> _SessionSnapshot | None:
    """Get atomic snapshot of session state under lock."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if not session:
            return None
        session["last_access"] = time.time()
        return _SessionSnapshot(
            output_dir=session["dir"],
            process=session["process"],
            seek_offset=session.get("seek_offset", 0),
            subtitles=session.get("subtitles") or [],
            duration=session.get("duration", 0),
        )


def _update_session_process(session_id: str, process: Any) -> bool:
    """Atomically update session process. Returns False if session gone."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if not session:
            return False
        session["process"] = process
        return True


async def _handle_existing_vod_session(
    existing_id: str,
    url: str,
    hw: str,
    do_probe: bool,
    max_resolution: str = "1080p",
    max_bitrate_mbps: float = 0,
) -> dict[str, Any] | None:
    """Handle existing VOD session: reuse active, return cached, or append.

    Returns None to trigger fresh start if session is invalid.
    """
    snap = _get_session_snapshot(existing_id)
    if not snap:
        return None

    playlist_path = pathlib.Path(snap.output_dir) / "stream.m3u8"
    segments = sorted(pathlib.Path(snap.output_dir).glob("seg*.ts"))

    # Case 1: Active session - reuse it
    if snap.process.returncode is None:
        log.info("Reusing active session %s", existing_id)
        await _wait_for_playlist(
            playlist_path,
            snap.process,
            min_segments=1,
            timeout_sec=_REUSE_ACTIVE_WAIT_TIMEOUT_SEC,
        )
        return _build_session_response(existing_id, snap, playlist_path)

    # Case 2: Dead session with no segments - invalid
    if not segments:
        stop_session(existing_id, force=True)
        with _transcode_lock:
            _vod_url_to_session.pop(url, None)
        return None

    # Case 3: Dead session with seek_offset - return cached content
    if snap.seek_offset > 0:
        log.info(
            "Returning cached session %s (seek_offset=%d)",
            existing_id,
            snap.seek_offset,
        )
        return _build_session_response(existing_id, snap, playlist_path)

    # Case 4: Dead session, no seek_offset - append new content
    # Note: CC only works for existing content (0 to hls_duration)
    hls_duration = _calc_hls_duration(playlist_path, len(segments))
    log.info("Resuming session %s from %.1fs", existing_id, hls_duration)

    media_info = probe_media(url)[0] if do_probe else None
    cmd = build_hls_ffmpeg_cmd(
        url, hw, snap.output_dir, True, None, media_info, max_resolution, max_bitrate_mbps
    )

    i_idx = cmd.index("-i")
    cmd.insert(i_idx, str(hls_duration))
    cmd.insert(i_idx, "-ss")
    try:
        hls_flags_idx = cmd.index("-hls_flags")
        cmd[hls_flags_idx + 1] += "+append_list"
    except ValueError:
        cmd.extend(["-hls_flags", "append_list"])
    cmd.extend(["-start_number", str(len(segments))])

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if not _update_session_process(existing_id, process):
        _kill_process(process)
        return None

    _spawn_background_task(_monitor_resume_ffmpeg(process, existing_id, url))
    log.info("Started resume ffmpeg pid=%s for %s", process.pid, existing_id)

    deadline = time.monotonic() + _RESUME_SEGMENT_WAIT_TIMEOUT_SEC
    next_seg = f"seg{len(segments):03d}.ts"
    while time.monotonic() < deadline:
        if process.returncode is not None:
            log.warning("Resume ffmpeg died immediately for %s", existing_id)
            return None
        if (pathlib.Path(snap.output_dir) / next_seg).exists():
            break
        await asyncio.sleep(_POLL_INTERVAL_SEC)

    await _wait_for_playlist(
        playlist_path,
        process,
        min_segments=1,
        timeout_sec=_RESUME_WAIT_TIMEOUT_SEC,
    )
    return _build_session_response(existing_id, snap, playlist_path)


def _get_existing_vod_session(
    url: str,
) -> tuple[str | None, bool, float]:
    """Get existing VOD session info atomically. Returns (session_id, is_valid, seek_offset)."""
    with _transcode_lock:
        existing_id = _vod_url_to_session.get(url)
        if not existing_id:
            return None, False, 0.0
        session = _transcode_sessions.get(existing_id)
        if not session:
            return None, False, 0.0
        return (
            existing_id,
            is_vod_session_valid(session),
            session.get("seek_offset", 0),
        )


async def _do_start_transcode(
    url: str,
    content_type: str,
    series_id: int | None,
    episode_id: int | None,
    old_seek_offset: float,
    series_name: str = "",
) -> dict[str, Any]:
    """Core transcode logic. Raises HTTPException on failure."""
    settings = _load_settings()
    hw = settings.get("transcode_hw", "software")
    max_resolution = settings.get("max_resolution", "1080p")
    max_bitrate_mbps = float(settings.get("max_bitrate_mbps", 0) or 0)
    is_vod = content_type in ("movie", "series")
    probe_key = {"movie": "probe_movies", "series": "probe_series"}
    do_probe = is_vod and settings.get(probe_key.get(content_type, ""), False)

    session_id = str(uuid.uuid4())
    output_dir = tempfile.mkdtemp(prefix=f"netv_transcode_{session_id}_")
    playlist_path = pathlib.Path(output_dir) / "stream.m3u8"

    media_info: MediaInfo | None = None
    subtitles: list[SubtitleStream] = []
    if do_probe:
        media_info, subtitles = await asyncio.to_thread(
            probe_media, url, series_id, episode_id, series_name
        )
        if media_info:
            subs_str = (
                ",".join(media_info.subtitle_codecs) if media_info.subtitle_codecs else "none"
            )
            if subtitles:
                subs_str += f" [extract:{','.join(s.lang for s in subtitles)}]"
            bitrate_str = (
                f"{media_info.video_bitrate / 1_000_000:.1f}Mbps"
                if media_info.video_bitrate
                else "?"
            )
            log.info(
                "Probe: video=%s/%s/%dp/%s audio=%s/%dch/%dHz duration=%.0fs subs=%s",
                media_info.video_codec,
                media_info.pix_fmt,
                media_info.height,
                bitrate_str,
                media_info.audio_codec,
                media_info.audio_channels,
                media_info.audio_sample_rate,
                media_info.duration,
                subs_str,
            )

    cmd = build_hls_ffmpeg_cmd(
        url, hw, output_dir, is_vod, subtitles, media_info, max_resolution, max_bitrate_mbps
    )
    if old_seek_offset > 0:
        i_idx = cmd.index("-i")
        cmd.insert(i_idx, str(old_seek_offset))
        cmd.insert(i_idx, "-ss")
        log.info("Applying seek_offset=%.1f from previous session", old_seek_offset)

    log.info(
        "Starting transcode session %s (vod=%s): %s",
        session_id,
        is_vod,
        " ".join(cmd),
    )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stderr_lines: list[str] = []
    _spawn_background_task(_monitor_ffmpeg_stderr(process, session_id, stderr_lines))

    sub_info = [
        {
            "index": s.index,
            "lang": s.lang,
            "name": s.name,
        }
        for s in subtitles
    ]
    total_duration = media_info.duration if media_info else 0.0

    with _transcode_lock:
        _transcode_sessions[session_id] = {
            "dir": output_dir,
            "process": process,
            "started": time.time(),
            "url": url,
            "is_vod": is_vod,
            "last_access": time.time(),
            "subtitles": sub_info,
            "duration": total_duration,
            "seek_offset": old_seek_offset,
            "series_id": series_id,
            "episode_id": episode_id,
        }
        if is_vod:
            _vod_url_to_session[url] = session_id

    if is_vod:
        session_info: dict[str, Any] = {
            "session_id": session_id,
            "url": url,
            "is_vod": True,
            "started": time.time(),
            "subtitles": sub_info,
            "duration": total_duration,
            "seek_offset": old_seek_offset,
            "series_id": series_id,
            "episode_id": episode_id,
        }
        if media_info:
            session_info["probe"] = {
                "video_codec": media_info.video_codec,
                "audio_codec": media_info.audio_codec,
                "pix_fmt": media_info.pix_fmt,
                "audio_channels": media_info.audio_channels,
                "audio_sample_rate": media_info.audio_sample_rate,
                "subtitle_codecs": media_info.subtitle_codecs,
            }
        (pathlib.Path(output_dir) / "session.json").write_text(json.dumps(session_info))

    timeout = _PLAYLIST_WAIT_SEEK_TIMEOUT_SEC if old_seek_offset > 0 else _PLAYLIST_WAIT_TIMEOUT_SEC
    if not await _wait_for_playlist(
        playlist_path,
        process,
        min_segments=2,
        timeout_sec=timeout,
    ):
        await asyncio.sleep(_POLL_INTERVAL_SEC)
        error_msg = "\n".join(stderr_lines[-10:]) if stderr_lines else "unknown"
        log.error(
            "ffmpeg:%s failed (exit %d): %s",
            session_id,
            process.returncode or -1,
            error_msg,
        )
        stop_session(session_id)
        raise HTTPException(500, "Transcode failed - check server logs for details")

    return {
        "session_id": session_id,
        "playlist": f"/transcode/{session_id}/stream.m3u8",
        "subtitles": _build_subtitle_tracks(session_id, sub_info),
        "duration": total_duration,
        "seek_offset": old_seek_offset,
    }


async def _start_transcode(
    url: str,
    content_type: str = "live",
    series_id: int | None = None,
    episode_id: int | None = None,
    series_name: str = "",
) -> dict[str, Any]:
    is_vod = content_type in ("movie", "series")
    existing_id, is_valid, old_seek_offset = (
        _get_existing_vod_session(url) if is_vod else (None, False, 0.0)
    )

    if existing_id:
        log.info(
            "Found VOD session %s, valid=%s",
            existing_id,
            is_valid,
        )
        if is_valid:
            settings = _load_settings()
            hw = settings.get("transcode_hw", "software")
            max_resolution = settings.get("max_resolution", "1080p")
            max_bitrate_mbps = float(settings.get("max_bitrate_mbps", 0) or 0)
            probe_key = {"movie": "probe_movies", "series": "probe_series"}
            do_probe = settings.get(probe_key.get(content_type, ""), False)
            result = await _handle_existing_vod_session(
                existing_id,
                url,
                hw,
                do_probe,
                max_resolution,
                max_bitrate_mbps,
            )
            if result:
                return result
            # Invalid session (no segments) - old_seek_offset already captured

        with _transcode_lock:
            _vod_url_to_session.pop(url, None)
        stop_session(existing_id, force=True)

    # Try transcode, retry once if series cache was stale
    try:
        return await _do_start_transcode(
            url, content_type, series_id, episode_id, old_seek_offset, series_name
        )
    except HTTPException:
        if series_id is not None:
            # Clear MRU and this episode's cache (if any), then retry with fresh probe
            log.info("Transcode failed, clearing MRU and retrying")
            invalidate_series_probe_cache(series_id, episode_id)
            return await _do_start_transcode(
                url, content_type, series_id, episode_id, old_seek_offset, series_name
            )
        raise


def get_session(session_id: str) -> dict[str, Any] | None:
    """Get a copy of session dict (safe to use outside lock)."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        return dict(session) if session else None


def get_session_progress(session_id: str) -> dict[str, Any] | None:
    session = get_session(session_id)
    if not session:
        return None
    playlist_path = pathlib.Path(session["dir"]) / "stream.m3u8"
    if not playlist_path.exists():
        return {"segment_count": 0, "duration": 0.0}
    durations = re.findall(r"#EXTINF:([\d.]+)", playlist_path.read_text())
    return {
        "segment_count": len(durations),
        "duration": sum(float(d) for d in durations),
    }


@dataclass(slots=True)
class _SeekSessionInfo:
    """Snapshot of session info needed for seek."""

    url: str
    output_dir: str
    process: Any
    subtitles: list[dict[str, Any]]
    series_id: int | None
    episode_id: int | None


def _get_seek_session_info(session_id: str) -> _SeekSessionInfo | None:
    """Get session info for seek atomically. Returns None if not VOD."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if not session or not session.get("is_vod"):
            return None
        return _SeekSessionInfo(
            url=session["url"],
            output_dir=session["dir"],
            process=session["process"],
            subtitles=session.get("subtitles") or [],
            series_id=session.get("series_id"),
            episode_id=session.get("episode_id"),
        )


def _update_seek_session(
    session_id: str,
    url: str,
    process: Any,
    seek_time: float,
) -> bool:
    """Update session after seek. Returns False if session gone."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if not session:
            return False
        session["process"] = process
        session["seek_offset"] = seek_time
        if url:
            _vod_url_to_session[url] = session_id
        return True


async def seek_transcode(session_id: str, seek_time: float) -> dict[str, Any]:
    info = _get_seek_session_info(session_id)
    if not info:
        raise HTTPException(404, "Session not found or not VOD")

    settings = _load_settings()
    hw = settings.get("transcode_hw", "software")
    max_resolution = settings.get("max_resolution", "1080p")
    max_bitrate_mbps = float(settings.get("max_bitrate_mbps", 0) or 0)
    segment_num = int(seek_time / _HLS_SEGMENT_DURATION_SEC)

    # Kill existing process
    if _kill_process(info.process):
        log.info("Killed ffmpeg for seek in session %s", session_id)

    # Clear old files
    output_path = pathlib.Path(info.output_dir)
    playlist_file = output_path / "stream.m3u8"
    playlist_file.unlink(missing_ok=True)
    for seg_file in output_path.glob("seg*.ts"):
        seg_file.unlink(missing_ok=True)
    for vtt_file in output_path.glob("sub*.vtt"):
        vtt_file.unlink(missing_ok=True)

    # Use probe_series if series_id, else probe_movies
    probe_setting = "probe_series" if info.series_id else "probe_movies"
    do_probe = settings.get(probe_setting, False)
    media_info = probe_media(info.url, info.series_id, info.episode_id)[0] if do_probe else None

    subtitles: list[SubtitleStream] = []
    for s in info.subtitles:
        if isinstance(s, dict) and "index" in s:
            subtitles.append(
                SubtitleStream(
                    index=s["index"],
                    lang=s.get("lang", "und"),
                    name=s.get("name", "Unknown"),
                )
            )

    cmd = build_hls_ffmpeg_cmd(
        info.url,
        hw,
        info.output_dir,
        True,
        subtitles or None,
        media_info,
        max_resolution,
        max_bitrate_mbps,
    )
    i_idx = cmd.index("-i")
    cmd.insert(i_idx, str(seek_time))
    cmd.insert(i_idx, "-ss")
    # Shift output timestamps so subtitles start at 0 after seek
    # -output_ts_offset must go before -f (output format)
    f_idx = cmd.index("-f")
    cmd.insert(f_idx, str(-seek_time))
    cmd.insert(f_idx, "-output_ts_offset")
    cmd.extend(["-start_number", "0"])

    log.info(
        "Seek transcode %s to %.1fs (seg %d): %s",
        session_id,
        seek_time,
        segment_num,
        " ".join(cmd),
    )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if not _update_seek_session(
        session_id,
        info.url,
        process,
        seek_time,
    ):
        _kill_process(process)
        raise HTTPException(
            404,
            "Session disappeared during seek",
        )

    # Persist seek_offset
    session_json = output_path / "session.json"
    if session_json.exists():
        try:
            data = json.loads(session_json.read_text())
            data["seek_offset"] = seek_time
            session_json.write_text(json.dumps(data))
        except Exception as e:
            log.warning("Failed to update session.json for %s: %s", session_id, e)

    _spawn_background_task(_monitor_seek_ffmpeg(process, session_id))

    if not await _wait_for_playlist(
        playlist_file,
        process,
        min_segments=2,
        timeout_sec=_PLAYLIST_WAIT_TIMEOUT_SEC,
    ):
        raise HTTPException(
            500,
            "Seek transcode timed out waiting for playlist",
        )

    log.info("Seek ready: %s", playlist_file)

    return {
        "ok": True,
        "segment": segment_num,
        "time": seek_time,
    }


def clear_url_session(url: str) -> str | None:
    with _transcode_lock:
        return _vod_url_to_session.pop(url, None)
