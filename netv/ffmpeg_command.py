"""FFmpeg command building and media probing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Final, NotRequired, TypedDict
from urllib.parse import urljoin

import functools
import json
import logging
import pathlib
import subprocess
import tempfile
import threading
import time

from treekle import json_loads

from netv import cache, proxy
from netv.cache import HwAccel, ServerSettings, get_user_agent
from netv.util import atomic_write_json, safe_urlopen


log = logging.getLogger(__name__)


# Segment names are a persistence-format contract shared by command and session readers.
SEG_PREFIX: Final = "seg"
# Dolby audio that Apple players decode natively; other clients need AAC.
PASSTHROUGH_AUDIO_CODECS = frozenset({"ac3", "eac3"})

# Module state.
_probe_lock = threading.Lock()
_probe_cache: dict[str, _ProbeEntry] = {}
_series_probe_cache: dict[int, _SeriesProbe] = {}
_gpu_nvdec_codecs: set[str] | None = None  # None = not probed yet.
_has_libplacebo: bool | None = None  # None = not probed yet.
# The loader main injects; init() replaces it (the gateway layers upscale checks on top).
_load_settings: Callable[[], ServerSettings] = cache.load_server_settings

# Directory containing TensorRT engines; None until init() enables AI Upscale.
_sr_engine_dir: str | None = None


@dataclass(kw_only=True, slots=True, frozen=True)
class SubtitleStream:
    index: int
    lang: str
    name: str


@dataclass(kw_only=True, slots=True, frozen=True)
class AudioTrack:
    index: int | None  # Input stream index; None when ffprobe omits it.
    lang: str  # Lowercase ISO 639-2 tag as muxed; "und" when untagged.
    codec: str
    channels: int = 0
    sample_rate: int = 0
    profile: str = ""
    default: bool = False


@dataclass(kw_only=True, slots=True, frozen=True)
class MediaInfo:
    video_codec: str
    audio_codec: str
    pix_fmt: str
    audio_channels: int = 0
    audio_sample_rate: int = 0
    audio_profile: str = ""  # e.g. "LC", "HE-AAC", "HE-AACv2"
    audio_index: int | None = None  # Input stream index of a non-first chosen audio track.
    audio_tracks: tuple[AudioTrack, ...] = ()
    subtitle_codecs: list[str] | None = None
    duration: float = 0.0
    height: int = 0
    video_bitrate: int = 0  # Bits per second, 0 if unknown.
    interlaced: bool = False  # True if field_order indicates interlaced.
    is_10bit: bool = False  # True if pix_fmt indicates 10-bit color.
    is_hdr: bool = False  # True if color transfer indicates HDR.
    is_hls: bool = False  # True if format is HLS (for input options)


_ProbeEntry = tuple[float, MediaInfo | None, list[SubtitleStream]]


class CachedAudioTrack(TypedDict):
    index: int | None
    lang: str
    codec: str
    channels: int
    sample_rate: int
    profile: str
    default: bool


class _CachedSubtitle(TypedDict):
    index: int
    lang: NotRequired[str]
    name: NotRequired[str]


class _CachedEpisode(TypedDict, total=False):
    time: float
    video_codec: str
    audio_codec: str
    pix_fmt: str
    audio_channels: int
    audio_sample_rate: int
    audio_index: int | None
    audio_tracks: list[CachedAudioTrack]
    subtitle_codecs: list[str] | None
    duration: float
    height: int
    video_bitrate: int
    interlaced: bool
    is_10bit: bool
    is_hdr: bool
    is_hls: bool
    subtitles: list[_CachedSubtitle]


class _ProbeTags(TypedDict, total=False):
    language: str
    name: str
    title: str


class _ProbeDisposition(TypedDict, total=False):
    default: int


# Numeric fields accept every shape ffprobe versions emit, which the conversions tolerate.
_ProbeNumber = str | int | float | None


class _ProbeStream(TypedDict, total=False):
    index: int | None
    codec_name: str
    codec_type: str
    pix_fmt: str
    height: int | None
    field_order: str
    color_transfer: str
    bit_rate: _ProbeNumber
    channels: _ProbeNumber
    sample_rate: _ProbeNumber
    profile: str
    tags: _ProbeTags
    disposition: _ProbeDisposition


class _ProbeFormat(TypedDict, total=False):
    format_name: str
    duration: _ProbeNumber
    bit_rate: _ProbeNumber


class EpisodeProbeStats(TypedDict):
    episode_id: int
    duration: float
    subtitle_count: int


class UrlProbeStats(TypedDict):
    url: str  # Carries provider credentials: never send it to a client.
    probed_at: float
    video_codec: str
    audio_codec: str
    audio_langs: list[str]
    subtitle_count: int
    duration: float
    height: int


class SeriesProbeStats(TypedDict):
    series_id: int
    name: str
    mru: int | None
    episode_count: int
    video_codec: str
    audio_codec: str
    subtitle_count: int
    episodes: list[EpisodeProbeStats]


def init(
    load_settings: Callable[[], ServerSettings],
    sr_engine_dir: str = "",
) -> None:
    """Initialize module with settings loader and optional AI Upscale config.

    Args:
        load_settings: Callable returning current server settings.
        sr_engine_dir: TensorRT engine directory; empty disables AI Upscale.
    """
    global _load_settings, _sr_engine_dir
    cache.initialize()
    _load_settings = load_settings
    _sr_engine_dir = sr_engine_dir or None
    _load_series_probe_cache()
    _load_url_probe_cache()
    if _sr_engine_dir is not None:
        log.info("AI Upscale enabled: engine_dir=%s", _sr_engine_dir)


def get_settings() -> ServerSettings:
    """Get current settings."""
    return _load_settings()


def get_hls_segment_duration() -> float:
    """Get HLS segment duration in seconds."""
    # Short segments start and seek faster.
    return 3.0


# ===========================================================================
# GPU Detection
# ===========================================================================


# ===========================================================================
# Transcode Directory
# ===========================================================================


def get_transcode_dir() -> pathlib.Path:
    """Get the transcode output directory, falling back to system temp.

    Returns:
        directory: Configured accessible directory, or the system temp directory.
    """
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


# ===========================================================================
# Probe Cache Management
# ===========================================================================


def get_url_probe_cache_stats() -> list[UrlProbeStats]:
    """Probes cached by URL (movies and live channels), most recent first.

    Returns:
        stats: One summary per cached URL with probe results.
    """
    with _probe_lock:
        entries = sorted(_probe_cache.items(), key=lambda item: item[1][0], reverse=True)
    return [
        {
            "url": url,
            "probed_at": probed_at,
            "video_codec": media_info.video_codec,
            "audio_codec": media_info.audio_codec,
            "audio_langs": [t.lang for t in media_info.audio_tracks],
            "subtitle_count": len(subs),
            "duration": media_info.duration,
            "height": media_info.height,
        }
        for url, (probed_at, media_info, subs) in entries
        if media_info is not None
    ]


def invalidate_url_probe_cache(urls: list[str]) -> int:
    """Forget the cached probes of ``urls``.

    Args:
        urls: Probed URLs.

    Returns:
        removed: Number of cached URLs dropped.
    """
    with _probe_lock:
        removed = sum(_probe_cache.pop(url, None) is not None for url in urls)
    _save_url_probe_cache()
    return removed


def get_series_probe_cache_stats() -> list[SeriesProbeStats]:
    """Get stats about cached series probes for settings UI.

    Returns:
        stats: One summary per series with cached episode probe data.
    """
    with _probe_lock:
        log.info(
            "get_series_probe_cache_stats: cache has %d series: %s",
            len(_series_probe_cache),
            list(_series_probe_cache.keys()),
        )
        result: list[SeriesProbeStats] = []
        for series_id, series_data in _series_probe_cache.items():
            episodes = series_data["episodes"]
            if not episodes:
                continue
            # Get most recent entry for display info.
            most_recent = max(episodes.values(), key=lambda x: x[0])
            _, media_info, subs = most_recent
            if media_info is None:
                continue
            # Build episode list.
            episode_list: list[EpisodeProbeStats] = []
            for eid, (_, emedia, esubs) in episodes.items():
                if emedia:
                    episode_list.append(
                        {
                            "episode_id": eid,
                            "duration": emedia.duration,
                            "subtitle_count": len(esubs),
                        },
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
                },
            )
        return sorted(result, key=lambda x: x["name"] or str(x["series_id"]))


def clear_all_probe_cache() -> int:
    """Clear all probe caches.

    Returns:
        cleared: Number of URL and series episode entries removed.
    """
    with _probe_lock:
        url_count = len(_probe_cache)
        series_count = sum(len(s.get("episodes", {})) for s in _series_probe_cache.values())
        _probe_cache.clear()
        _series_probe_cache.clear()
    _save_series_probe_cache()
    _save_url_probe_cache()
    log.info("Cleared probe cache: %d URL entries, %d series entries", url_count, series_count)
    return url_count + series_count


def invalidate_series_probe_cache(series_id: int, episode_id: int | None = None) -> None:
    """Invalidate cached probe for series/episode.

    Args:
        series_id: Series whose cache entries are invalidated.
        episode_id: Episode to clear; None clears the entire series.
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
    """Clear only the MRU for a series, keeping episode cache intact.

    Args:
        series_id: Series whose most-recently-used episode is cleared.
    """
    with _probe_lock:
        if series_id not in _series_probe_cache:
            return
        if "mru" in _series_probe_cache[series_id]:
            del _series_probe_cache[series_id]["mru"]
            log.info("Cleared MRU for series=%d", series_id)
    _save_series_probe_cache()


def has_cached_probe(url: str, series_id: int | None = None) -> bool:
    """Whether ``probe_media`` would answer from cache rather than open ``url``.

    Args:
        url: Media URL.
        series_id: Series whose episodes share probes.

    Returns:
        cached: True when a URL or series probe is stored.
    """
    with _probe_lock:
        return url in _probe_cache or bool(
            series_id is not None and _series_probe_cache.get(series_id, {}).get("episodes"),
        )


def remember_probe(url: str, media_info: MediaInfo | None) -> None:
    """Record ``media_info`` as ``url``'s latest probe, replacing any older one.

    Args:
        url: Probed stream URL, such as a live channel's.
        media_info: Probe result; None records nothing.
    """
    if media_info is None:
        return
    with _probe_lock:
        _probe_cache[url] = (time.time(), media_info, [])
    _save_url_probe_cache()


def restore_probe_cache_entry(
    url: str,
    media_info: MediaInfo,
    subs: list[SubtitleStream],
    series_id: int | None = None,
    episode_id: int | None = None,
) -> None:
    """Restore a probe cache entry during session recovery.

    Args:
        url: Probed media URL.
        media_info: Recovered media facts.
        subs: Recovered text subtitle streams.
        series_id: Series cache key, when the media is an episode.
        episode_id: Episode cache key; None stores episode 0.
    """
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


def resolve_hls_master_playlist(url: str, source_id: str) -> str:
    """Resolve HLS master playlist to highest bandwidth variant URL.

    Args:
        url: Media URL, possibly an HLS master playlist.
        source_id: The configured source that listed it; "" for none.

    Returns:
        resolved_url: Highest-bandwidth variant the source's rule lets ffmpeg reach,
            or url when none is resolvable.
    """

    if not url.endswith(".m3u8") and ".m3u8?" not in url:
        return url  # Not an m3u8, return as-is.

    lan = proxy.home_stream(url, source_id)
    try:
        with safe_urlopen(
            url,
            10,
            get_user_agent(),
            allow_address=functools.partial(proxy.allowed_address, lan=lan),
        ) as response:
            content = response.read().decode(errors="replace")
            # Where the playlist was served after redirects; reusing it saves a hop.
            url = response.url

        # Check if this is a master playlist (has #EXT-X-STREAM-INF)
        if "#EXT-X-STREAM-INF" not in content:
            return url  # Not a master playlist.

        # Parse variants: each #EXT-X-STREAM-INF is followed by a URL line.
        lines = content.strip().split("\n")
        variants: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            if line.startswith("#EXT-X-STREAM-INF:"):
                # Extract BANDWIDTH from the tag.
                bandwidth = 0
                for attr in line.split(":")[1].split(","):
                    if attr.startswith("BANDWIDTH="):
                        with suppress(ValueError):
                            bandwidth = int(attr.split("=")[1])
                        break
                # Next non-comment line is the variant URL.
                for j in range(i + 1, len(lines)):
                    variant_line = lines[j].strip()
                    if variant_line and not variant_line.startswith("#"):
                        # Resolve relative URL.
                        variant_url = urljoin(url, variant_line)
                        if proxy.allowed_upstream(variant_url, lan=lan):
                            variants.append((bandwidth, variant_url))
                        break

        if not variants:
            log.warning("HLS master playlist has no variants: %s", url[:80])
            return url

        # Select highest bandwidth variant.
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
    *,
    source_id: str,
    probe_timeout_sec: float = 30,
    use_cache: bool = True,
) -> tuple[MediaInfo | None, list[SubtitleStream]]:
    """Probe media facts and text subtitle streams, using cached results.

    Cached probes never expire: a start that fails on one re-probes and replaces it.
    An episode without its own probe shares its series' MRU episode's probe.

    Args:
        url: Media URL to probe.
        series_id: Series cache key, when the media is an episode.
        episode_id: Episode cache key.
        series_name: Display name stored with new series entries.
        source_id: The configured source that listed ``url``; "" for none.
        probe_timeout_sec: Deadline for each ffprobe attempt.
        use_cache: Answer from a cached probe when one applies.

    Returns:
        media_info: Probed media facts, or None when no video stream was found.
        subtitles: Extractable text subtitle streams.
    """
    if use_cache and (cached := _cached_probe(url, series_id, episode_id)) is not None:
        return cached
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
    if user_agent is not None:
        base_cmd.extend(["-user_agent", user_agent])
    base_cmd.extend(proxy.ffmpeg_input_args(url, source_id))

    # Try probe without forcing HLS first, retry with HLS options if it fails.
    is_hls = False
    data: _ProbeOutput | None = None
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
                timeout=probe_timeout_sec,
                env=proxy.ffmpeg_env(),
            )
            if result.returncode == 0:
                data = json_loads(result.stdout, _ProbeOutput)
                # Check detected format or if we forced HLS.
                format_name = data.get("format", {}).get("format_name", "").lower()
                is_hls = force_hls or "hls" in format_name
                break
        except Exception as e:
            log.warning("Probe failed%s: %s", " (HLS mode)" if force_hls else "", e)
            continue

    if data is None:
        return None, []
    media_info, subtitles = _media_info_from_probe(data, is_hls=is_hls)
    if media_info is None:
        return None, []
    # Only cache if we got valid video info (height > 0)
    if media_info.height <= 0:
        log.warning(
            "Probe returned invalid height=%d, not caching: %s",
            media_info.height,
            url[:80],
        )
        return media_info, subtitles
    with _probe_lock:
        _probe_cache[url] = (time.time(), media_info, subtitles)
        # Cache by series_id/episode_id if provided.
        if series_id is not None:
            if series_id not in _series_probe_cache:
                _series_probe_cache[series_id] = {"name": series_name, "episodes": {}}
            elif not _series_probe_cache[series_id].get("name", "") and series_name:
                _series_probe_cache[series_id]["name"] = series_name
            series = _series_probe_cache[series_id]
            eid = episode_id if episode_id is not None else 0
            episodes = series.setdefault("episodes", {})
            episodes[eid] = (time.time(), media_info, subtitles)
            # An episode probed because the shared probe misfit it must not become
            # the shared probe, or every typical sibling would then fail once too.
            if series.get("mru") not in episodes:
                series["mru"] = eid
            log.info(
                "Probe cached: series=%s episode=%s mru=%s",
                series_id,
                eid,
                series.get("mru"),
            )
    if series_id is not None:
        _save_series_probe_cache()
    _save_url_probe_cache()
    return media_info, subtitles


def probe_segment(path: str) -> MediaInfo | None:
    """Probe the video and audio facts of a local media segment.

    Args:
        path: Local segment file.

    Returns:
        media_info: Source facts, or None when the segment holds no readable video.
    """
    cmd = ["ffprobe", "-v", "error", "-of", "json", "-show_streams", "-show_format", path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        data = json_loads(result.stdout, _ProbeOutput)
    except (OSError, subprocess.SubprocessError, ValueError):  # ReadError is a ValueError.
        return None
    return _media_info_from_probe(data, is_hls=False)[0]


# ===========================================================================
# FFmpeg Command Building
# ===========================================================================


def http_reconnect_args(url: str, *, max_delay: int = 30) -> list[str]:
    """Build reconnect options for HTTP inputs only.

    Args:
        url: FFmpeg input URL or path.
        max_delay: Maximum reconnect delay in seconds.

    Returns:
        args: Reconnect options, empty for local or non-HTTP inputs.
    """
    if not url.startswith(("http://", "https://")):
        return []
    return [
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_on_http_error",
        "4xx,5xx",
        "-reconnect_delay_max",
        str(max_delay),
    ]


def live_video_bitrates(resolution: str) -> tuple[int, int]:
    """Return target and maximum video bitrate shared by live renditions.

    Args:
        resolution: Rendition resolution preset; unknown presets use 720p rates.

    Returns:
        target: Average video bitrate in bits per second.
        maximum: Peak video bitrate and buffer size in bits per second.
    """
    return {
        "480p": (2_000_000, 3_000_000),
        "720p": (4_000_000, 6_000_000),
        "1080p": (6_000_000, 8_000_000),
        "4k": (16_000_000, 20_000_000),
    }.get(resolution, (4_000_000, 6_000_000))


def limit_live_video_bitrate(cmd: list[str], resolution: str) -> None:
    """Replace quality-only rate control with bounded live bitrates, in place.

    Args:
        cmd: FFmpeg command whose final element is the output path.
        resolution: Rendition resolution preset selecting the bitrate bounds.
    """
    target, maximum = live_video_bitrates(resolution)
    encoder = cmd[cmd.index("-c:v") + 1]
    for flag in (
        "-rc",
        "-rc_mode",
        "-qp",
        "-qp_i",
        "-qp_p",
        "-global_quality",
        "-crf",
        "-b:v",
        "-maxrate",
        "-bufsize",
    ):
        if flag in cmd:
            index = cmd.index(flag)
            del cmd[index : index + 2]
    rate_mode = {
        "h264_nvenc": ["-rc", "vbr"],
        "h264_amf": ["-rc", "vbr_peak"],
        "h264_vaapi": ["-rc_mode", "VBR"],
    }.get(encoder, [])
    cmd[-1:-1] = rate_mode + [
        "-b:v",
        str(target),
        "-maxrate",
        str(maximum),
        "-bufsize",
        str(maximum),
    ]


def get_live_hls_list_size(
    segment_duration: float = 3.0,
    *,
    live_buffer_secs: float = 30.0,
) -> int:
    """Get hls_list_size for live streams based on DVR setting.

    Args:
        segment_duration: HLS segment length in seconds.
        live_buffer_secs: Retained live window when DVR is disabled.

    Returns:
        list_size: Number of segments retained in the live playlist.
    """
    dvr_mins = _load_settings()["live_dvr_mins"]
    if dvr_mins <= 0:
        return int(live_buffer_secs / segment_duration)
    # DVR enabled: calculate segments from minutes.
    return int(dvr_mins * 60 / segment_duration)


def uses_audio_passthrough(media_info: MediaInfo | None, passthrough: bool) -> bool:
    """Will the output carry Dolby audio that only passthrough-capable clients can play?"""
    return bool(passthrough and media_info and media_info.audio_codec in PASSTHROUGH_AUDIO_CODECS)


def can_remux_live(
    media_info: MediaInfo | None,
    max_resolution: str = "1080p",
    *,
    audio_passthrough: bool = False,
) -> bool:
    """Can a live source be retained via a stream-copy HLS remux?"""
    return _can_copy_video(media_info, max_resolution) and _can_copy_audio(
        media_info,
        passthrough=audio_passthrough,
    )


def probe_closed_captions(path: str) -> bool:
    """Detect embedded closed-caption data in a local video segment.

    Args:
        path: Local video segment.

    Returns:
        has_captions: Whether the first frames carry closed-caption side data.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        "%+2",
        "-show_frames",
        "-show_entries",
        "frame=side_data_list",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip()[-500:] or f"ffprobe exited {result.returncode}"
            log.warning(
                "Could not inspect embedded captions in %s: %s",
                pathlib.Path(path).name,
                detail,
            )
            return False
        probe = json_loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        log.warning(
            "Could not inspect embedded captions in %s: %s",
            pathlib.Path(path).name,
            error,
        )
        return False
    # Read as a plain tree: frames without side data, or of another shape, are skipped.
    frames = probe.get("frames") if isinstance(probe, dict) else None
    return any(
        "closed caption" in str(side_data.get("side_data_type", "")).lower()
        for frame in (frames if isinstance(frames, list) else [])
        if isinstance(frame, dict)
        if isinstance(side_list := frame.get("side_data_list"), list)
        for side_data in side_list
        if isinstance(side_data, dict)
    )


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
    allow_upscale: bool = True,
    source: MediaInfo | None = None,
    audio_passthrough: bool = False,
    *,
    input_args: Sequence[str],
    sr_segment_duration: float = 2.0,
    vod_read_burst_sec: int = 120,
) -> list[str]:
    """Build ffmpeg command for HLS transcoding.

    Args:
        input_url: Source URL or local path.
        hw: Encoder and decode-fallback selection.
        output_dir: Directory receiving playlists and segments.
        is_vod: Whether to keep a complete event playlist.
        subtitles: Text subtitle streams to extract.
        media_info: Probed source facts; None adds probe options to the command.
        max_resolution: Maximum output resolution preset.
        quality: Quality preset for constant-quality encoders.
        user_agent: HTTP User-Agent for the input.
        deinterlace_fallback: Deinterlace when probe data is unavailable.
        allow_upscale: Whether AI Upscale may apply.
        source: Facts probed from a sample of an input the command still probes and
            re-encodes (a live relay); used in place of ``media_info`` for every
            decision but stream copy.
        audio_passthrough: Copy Dolby audio for clients that decode it natively.
        input_args: Options that keep the input's connections behind the egress
            proxy (``proxy.ffmpeg_input_args``); none for a pipe.
        sr_segment_duration: HLS segment length when AI Upscale applies.
        vod_read_burst_sec: Seconds of a network VOD input read at full speed
            before reading holds to real time.

    Returns:
        cmd: FFmpeg arguments.
    """
    # Every decision below reads this one record, so no caller can pass some probed
    # facts and drop others. Only stream copy needs media_info: a ``source`` is re-encoded.
    probed = media_info or source
    copy_video = _can_copy_video(media_info, max_resolution, allow_upscale=allow_upscale)
    copy_audio = _can_copy_audio(probed, passthrough=audio_passthrough)

    # Full hardware pipeline if GPU supports the codec
    # Parse hw to get encoder type.
    enc_type, _ = _parse_hw(hw)
    codec = probed.video_codec if probed else ""
    use_hw_pipeline = bool(
        not copy_video
        and codec
        and (
            (enc_type == "nvenc" and codec in _get_gpu_nvdec_codecs())
            # VAAPI/QSV lack a portable runtime probe (vainfo output varies by
            # driver), so use codecs nearly universal on GPUs from the last decade.
            or (
                enc_type == "vaapi"
                and codec in {"h264", "hevc", "mpeg2video", "vp8", "vp9", "vc1", "av1"}
            )
            or (enc_type == "qsv" and codec in {"h264", "hevc", "mpeg2video", "vp9", "vc1", "av1"})
            # AMF never has hw decode pipeline - always False.
        ),
    )

    # Deinterlace: use probe result if available, else use fallback setting
    # (fallback defaults to True for live, False for VOD when not explicitly set)
    fallback = deinterlace_fallback if deinterlace_fallback is not None else (not is_vod)
    # If probe failed (height=0), don't trust interlaced flag - use fallback.
    # A live sample shows only the moment it was taken: a channel airing a progressive
    # segment switches to 60i shows later, so a raster that can carry fields keeps the
    # deinterlacer, which passes frames marked progressive through untouched.
    deinterlace = (
        (probed.interlaced or (probed is source and _can_carry_fields(probed)))
        if probed and probed.height > 0
        else fallback
    )

    # Build component arg lists.
    video_pre, video_post = _build_video_args(
        copy_video=copy_video,
        hw=hw,
        deinterlace=deinterlace,
        use_hw_pipeline=use_hw_pipeline,
        max_resolution=max_resolution,
        quality=quality,
        is_hdr=probed.is_hdr if probed else False,
        source_height=probed.height if probed else 0,
        allow_upscale=allow_upscale,
    )
    audio_args = _build_audio_args(copy_audio=copy_audio, audio=probed)
    sr_applied = any("dnn_processing=" in arg for arg in video_post)
    segment_duration = sr_segment_duration if sr_applied else get_hls_segment_duration()

    # Base args.
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

    # Input args.
    cmd.extend(
        [
            "-fflags",
            "+discardcorrupt+genpts",
            "-err_detect",
            "ignore_err",
        ],
    )
    cmd.extend(input_args)
    cmd.extend(http_reconnect_args(input_url))
    if is_vod and input_url.startswith(("http://", "https://")):
        # Copying video outruns playback many times over and fills the transcode dir
        # (often a RAM tmpfs) with a whole movie; stay a burst ahead of real time instead.
        cmd.extend(
            [
                "-readrate",
                "1",
                "-readrate_initial_burst",
                str(vod_read_burst_sec),
                "-readrate_catchup",
                "2",
            ],
        )
    if user_agent is not None:
        cmd.extend(["-user_agent", user_agent])
    # Use HLS demuxer options if probe detected HLS format.
    if media_info and media_info.is_hls:
        cmd.extend(["-f", "hls", "-extension_picky", "0"])
    cmd.extend(["-i", input_url])

    # Keep the growing VTT files for web playback and package the same tracks
    # as HLS subtitle renditions for native players.
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
                "-map",
                f"0:{sub.index}",
                "-c:s",
                "webvtt",
                "-f",
                "segment",
                "-segment_time",
                str(int(segment_duration)),
                "-segment_list",
                f"{output_dir}/sub{i}.m3u8",
                "-segment_list_type",
                "m3u8",
                "-segment_list_size",
                "0" if is_vod else str(get_live_hls_list_size(segment_duration)),
                "-segment_format",
                "webvtt",
            ],
        )
        if not is_vod:
            cmd.extend(["-segment_list_flags", "+live"])
        cmd.append(f"{output_dir}/sub{i}_%06d.vtt")

    # Stream mapping + video + audio.
    audio_index = media_info.audio_index if media_info else None
    cmd.extend(["-map", "0:v:0", "-map", "0:a:0" if audio_index is None else f"0:{audio_index}"])
    cmd.extend(video_post)
    cmd.extend(audio_args)

    # HLS output args.
    cmd.extend(
        [
            "-max_delay",
            "5000000",
            "-f",
            "hls",
            "-hls_time",
            str(int(segment_duration)),
            "-hls_list_size",
            "0" if is_vod else str(get_live_hls_list_size(segment_duration)),
            "-hls_segment_filename",
            f"{output_dir}/{SEG_PREFIX}%03d.ts",
        ],
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
            ],
        )
    else:
        # Startup waits for two segments; short first ones (cut at the next
        # keyframe) let live playback begin without waiting out two full ones.
        cmd.extend(["-hls_init_time", "1", "-hls_flags", "delete_segments"])

    cmd.append(f"{output_dir}/stream.m3u8")
    if sr_applied and not is_vod:
        limit_live_video_bitrate(cmd, max_resolution)
    return cmd


class _ProbeOutput(TypedDict, total=False):
    """The subset of ffprobe's JSON this module reads; other keys are dropped."""

    streams: list[_ProbeStream]
    format: _ProbeFormat


# ISO 639-1 codes (what settings and browsers use) to the 639-2 B/T tags containers carry.
_ISO639_2: Final = {
    "ar": ("ara",),
    "cs": ("cze", "ces"),
    "da": ("dan",),
    "de": ("ger", "deu"),
    "el": ("gre", "ell"),
    "en": ("eng",),
    "es": ("spa",),
    "fi": ("fin",),
    "fr": ("fre", "fra"),
    "he": ("heb",),
    "hi": ("hin",),
    "hu": ("hun",),
    "it": ("ita",),
    "ja": ("jpn",),
    "ko": ("kor",),
    "nl": ("dut", "nld"),
    "no": ("nor", "nob", "nno"),
    "pl": ("pol",),
    "pt": ("por",),
    "ro": ("rum", "ron"),
    "ru": ("rus",),
    "sk": ("slo", "slk"),
    "sv": ("swe",),
    "tr": ("tur",),
    "uk": ("ukr",),
    "zh": ("chi", "zho"),
}


def with_audio_language(media_info: MediaInfo | None, lang: str) -> MediaInfo | None:
    """``media_info`` set to play its first audio track in ``lang``, when it has one.

    Args:
        media_info: Probed source, or None when probing is off.
        lang: ISO 639-1 or 639-2 code, optionally with a region (``en-US``); empty
            keeps the probed choice (the track the file marks default).

    Returns:
        media_info: The input with its audio fields describing the chosen track.
    """
    code = lang.split("-")[0].strip().lower()
    if media_info is None or not code:
        return media_info
    tags = _ISO639_2.get(code, (code,))
    track = next(
        (t for t in media_info.audio_tracks if t.lang in tags and t.index is not None),
        None,
    )
    return media_info if track is None else with_audio_track(media_info, track)


def with_audio_track(media_info: MediaInfo, track: AudioTrack | None) -> MediaInfo:
    """Point ``media_info``'s audio fields, and its ``-map``, at ``track``.

    Args:
        media_info: Probed source listing its audio tracks.
        track: One of its tracks; None keeps ``media_info`` as is.

    Returns:
        media_info: The input describing ``track``.
    """
    if track is None:
        return media_info
    first = media_info.audio_tracks[0] if media_info.audio_tracks else None
    return replace(
        media_info,
        audio_codec=track.codec,
        audio_channels=track.channels,
        audio_sample_rate=track.sample_rate,
        audio_profile=track.profile,
        # The first track keeps the 0:a:0 map, which also holds across HLS variants.
        audio_index=None if track == first else track.index,
    )


def audio_tracks_to_cache(tracks: tuple[AudioTrack, ...]) -> list[CachedAudioTrack]:
    """Audio tracks as stored in probe caches and session files.

    Args:
        tracks: Probed audio tracks.

    Returns:
        cached: One JSON-ready record per track.
    """
    return [
        {
            "index": t.index,
            "lang": t.lang,
            "codec": t.codec,
            "channels": t.channels,
            "sample_rate": t.sample_rate,
            "profile": t.profile,
            "default": t.default,
        }
        for t in tracks
    ]


def audio_tracks_from_cache(tracks: list[CachedAudioTrack]) -> tuple[AudioTrack, ...]:
    """Audio tracks read back from ``audio_tracks_to_cache``.

    Args:
        tracks: Cached records.

    Returns:
        tracks: The probed tracks.
    """
    return tuple(
        AudioTrack(
            index=t["index"],
            lang=t["lang"],
            codec=t["codec"],
            channels=t["channels"],
            sample_rate=t["sample_rate"],
            profile=t["profile"],
            default=t["default"],
        )
        for t in tracks
    )


def _parse_hw(hw: str) -> tuple[str, str]:
    """Parse hw into (encoder, fallback). e.g. 'nvenc+vaapi' -> ('nvenc', 'vaapi')"""
    if "+" in hw:
        encoder, fallback = hw.split("+", 1)
        return encoder, fallback
    return hw, "software"  # Standalone options fallback to software.


# Returns (engine_path, input_height, input_width, scale_factor) or None if not found.
def _find_sr_engine(
    source_height: int,
    *,
    model_name: str = "2x-nomosuni-compact",
) -> tuple[str, int, int, int] | None:
    """Find the best matching engine file for the given resolution."""
    if _sr_engine_dir is None:
        return None
    engine_dir = pathlib.Path(_sr_engine_dir)
    if not engine_dir.exists():
        return None

    # Engine naming: {model_name}_{height}p_fp16.engine.
    engines: list[tuple[int, pathlib.Path]] = []
    for engine in engine_dir.glob(f"{model_name}_*p_fp16.engine"):
        # Extract height from filename.
        name = engine.stem
        parts = name.rsplit("_", 2)
        if len(parts) >= 3:
            height_str = parts[1].rstrip("p")
            if height_str.isdigit():
                engines.append((int(height_str), engine))

    if not engines:
        return None

    # Sort by height ascending.
    engines.sort(key=lambda x: x[0])

    # Select appropriate engine based on source height.
    if source_height <= 0:
        # Probe failed - use highest resolution engine.
        engine_height, engine_path = engines[-1]
        log.warning("SR: probe failed, using %dp engine", engine_height)
    else:
        # Find engine closest to but >= source height, or use largest if source is bigger.
        engine_height, engine_path = engines[-1]  # Default to largest.
        for h, p in engines:
            if h >= source_height:
                engine_height, engine_path = h, p
                break

    # Calculate width assuming 16:9 aspect ratio, rounded to multiple of 8.
    engine_width = ((engine_height * 16 // 9) + 7) // 8 * 8

    return str(engine_path), engine_height, engine_width, 2


# SR is controlled by sr_model setting - if a model is selected, SR is applied when
# source height < target height.
def _build_sr_filter(
    source_height: int,
    target_height: int,
    *,
    model_name: str = "2x-nomosuni-compact",
) -> str:
    """Build AI Upscale filter string if needed. Returns empty string if disabled."""
    if _sr_engine_dir is None:
        return ""

    if not _load_settings().get("sr_model", ""):
        return ""  # SR disabled (Off selected)

    engine_info = _find_sr_engine(source_height, model_name=model_name)
    if not engine_info:
        log.warning("SR: no engine found for model=%s, source=%dp", model_name, source_height)
        return ""

    engine_path, engine_height, engine_width, scale = engine_info

    # Apply SR when source resolution is below target (upscaling scenario)
    if target_height and source_height >= target_height:
        log.info(
            "SR: skipping %s - source %dp >= target %dp",
            model_name,
            source_height,
            target_height,
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
    # 5. Scale down on GPU to target resolution.
    sr_filter = (
        f"scale={engine_width}:{engine_height}:force_original_aspect_ratio=decrease,"
        f"pad={engine_width}:{engine_height}:(ow-iw)/2:(oh-ih)/2,"
        f"format=rgb24,"
        f"hwupload,"
        f"dnn_processing=dnn_backend=tensorrt:model={engine_path}"
    )
    if target_height:
        # After dnn_processing, data is on GPU - use scale_cuda with explicit params.
        sr_filter += f",scale_cuda=w=-2:h={target_height}"
    return sr_filter


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
        # Minimum compute capability per NVDEC codec:
        # https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
        # HEVC 10-bit requires Pascal; the Maxwell GM206 edge case is ignored. Every
        # NVDEC (Kepler on) decodes MPEG-2, the codec of US broadcast TV.
        nvdec_min_compute = {"mpeg2video": 3.0, "h264": 5.0, "hevc": 6.0, "av1": 8.0}
        _gpu_nvdec_codecs = {
            codec for codec, min_cap in nvdec_min_compute.items() if compute_cap >= min_cap
        }
        log.info(
            "GPU: %s (compute %.1f) NVDEC: %s",
            gpu_name,
            compute_cap,
            ", ".join(sorted(_gpu_nvdec_codecs)) or "none",
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


def _probe_to_cache(
    cache_time: float,
    media_info: MediaInfo,
    subs: list[SubtitleStream],
) -> _CachedEpisode:
    """One probe as stored on disk."""
    return {
        "time": cache_time,
        "video_codec": media_info.video_codec,
        "audio_codec": media_info.audio_codec,
        "pix_fmt": media_info.pix_fmt,
        "audio_channels": media_info.audio_channels,
        "audio_sample_rate": media_info.audio_sample_rate,
        "audio_index": media_info.audio_index,
        "audio_tracks": audio_tracks_to_cache(media_info.audio_tracks),
        "subtitle_codecs": media_info.subtitle_codecs,
        "duration": media_info.duration,
        "height": media_info.height,
        "video_bitrate": media_info.video_bitrate,
        "interlaced": media_info.interlaced,
        "is_10bit": media_info.is_10bit,
        "is_hdr": media_info.is_hdr,
        "is_hls": media_info.is_hls,
        "subtitles": [{"index": s.index, "lang": s.lang, "name": s.name} for s in subs],
    }


def _probe_from_cache(entry: _CachedEpisode) -> _ProbeEntry | None:
    """One probe read back from disk; None for an entry without video facts."""
    # Use .get() for all fields to handle corrupt/incomplete cache.
    video_codec = entry.get("video_codec", "")
    if not video_codec:
        return None
    media_info = MediaInfo(
        video_codec=video_codec,
        audio_codec=entry.get("audio_codec", ""),
        pix_fmt=entry.get("pix_fmt", ""),
        audio_channels=entry.get("audio_channels", 0),
        audio_sample_rate=entry.get("audio_sample_rate", 0),
        audio_index=entry.get("audio_index"),
        audio_tracks=audio_tracks_from_cache(entry.get("audio_tracks", [])),
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
        SubtitleStream(
            index=subtitle["index"],
            lang=subtitle.get("lang", "und"),
            name=subtitle.get("name", ""),
        )
        for subtitle in entry.get("subtitles", [])
    ]
    return entry.get("time", 0), media_info, subs


def _load_url_probe_cache() -> None:
    """Load probes of movies and live channels, cached by URL, from disk."""
    path = cache.CACHE_DIR / "probe_cache.json"
    if not path.exists():
        return
    try:
        data = json_loads(path.read_text(), dict[str, _CachedEpisode])
    except Exception as e:
        log.warning("Failed to load probe cache: %s", e)
        return
    with _probe_lock:
        for url, entry in data.items():
            if url not in _probe_cache and (probe := _probe_from_cache(entry)) is not None:
                _probe_cache[url] = probe
    log.info("Loaded %d URL probe cache entries", len(data))


def _save_url_probe_cache() -> None:
    """Save probes cached by URL; the file holds provider credentials like the settings."""
    with _probe_lock:
        data = {
            url: _probe_to_cache(cache_time, media_info, subs)
            for url, (cache_time, media_info, subs) in _probe_cache.items()
            if media_info is not None
        }
    try:
        atomic_write_json(cache.CACHE_DIR / "probe_cache.json", data)
    except Exception as e:
        log.warning("Failed to save probe cache: %s", e)


def _load_series_probe_cache() -> None:
    """Load series probe cache from disk."""
    path = cache.CACHE_DIR / "series_probe_cache.json"
    if not path.exists():
        return
    try:
        data = json_loads(path.read_text(), dict[str, _CachedSeries])
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
                for eid_str, entry in series_data.get("episodes", {}).items():
                    eid = int(eid_str)
                    if eid in _series_probe_cache[sid]["episodes"]:
                        continue
                    if (probe := _probe_from_cache(entry)) is not None:
                        _series_probe_cache[sid]["episodes"][eid] = probe
                        count += 1
        log.info("Loaded %d series probe cache entries", count)
    except Exception as e:
        log.warning("Failed to load series probe cache: %s", e)


def _save_series_probe_cache() -> None:
    """Save series probe cache to disk."""
    with _probe_lock:
        data: dict[str, _CachedSeries] = {}
        for sid, series_data in _series_probe_cache.items():
            cached_episodes: dict[str, _CachedEpisode] = {}
            data[str(sid)] = {
                "name": series_data.get("name", ""),
                "mru": series_data.get("mru"),
                "episodes": cached_episodes,
            }
            for eid, (cache_time, media_info, subs) in series_data["episodes"].items():
                if media_info is not None:
                    cached_episodes[str(eid)] = _probe_to_cache(cache_time, media_info, subs)
    try:
        path = cache.CACHE_DIR / "series_probe_cache.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.warning("Failed to save series probe cache: %s", e)


def _probe_int(value: _ProbeNumber) -> int:
    """Convert an ffprobe numeric field, treating absent or empty values as zero."""
    return 0 if value is None or value == "" else int(value)


def _lang_display_name(code: str) -> str:
    return {
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
    }.get(code, code.upper())


def _build_video_args(
    *,
    copy_video: bool,
    hw: str,
    deinterlace: bool,
    use_hw_pipeline: bool,
    max_resolution: str,
    quality: str,
    is_hdr: bool = False,
    source_height: int = 0,
    allow_upscale: bool = True,
    sr_filter_threads: int = 4,
) -> tuple[list[str], list[str]]:
    """Build video args. Returns (pre_input_args, post_input_args)."""
    if copy_video:
        return [], ["-c:v", "copy"]

    # Parse hw into encoder and fallback.
    enc_type, fallback = _parse_hw(hw)

    max_h = _max_resolution_height(max_resolution)

    # AI Upscale runs on NVENC only; it needs CPU frames, so it turns off the hw pipeline.
    sr_filter = (
        _build_sr_filter(source_height, max_h or 0) if allow_upscale and enc_type == "nvenc" else ""
    )
    use_hw_pipeline = use_hw_pipeline and not sr_filter

    # Fall back gracefully if VAAPI is needed but no device was detected.
    vaapi_device = cache.VAAPI_DEVICE or ""
    vaapi_decode = fallback == "vaapi" and bool(vaapi_device)
    if not vaapi_device:
        if enc_type == "vaapi":
            log.warning(
                "VAAPI unavailable (no Intel/AMD GPU), falling back to software encoding",  # pragma: no mutate[diagnostic-text]
            )
            enc_type = "software"
        elif fallback == "vaapi":
            log.warning(
                "VAAPI fallback unavailable (no Intel/AMD GPU), using software decode",  # pragma: no mutate[diagnostic-text]
            )

    # Height expr for scale filter (scale down only, -2 keeps width divisible by 2)
    h = "" if max_h is None else f"min(ih\\,{max_h})"
    # Lower QP/CRF means higher quality.
    qp = {"high": 20, "low": 35}.get(quality, 28)
    pre: list[str]

    if enc_type == "nvenc":
        if use_hw_pipeline:
            # CUDA decode path.
            pre = [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-extra_hw_frames",
                "3",
            ]
            scale = f"scale_cuda=-2:{h}:format=nv12" if h else "scale_cuda=format=nv12"
            deint = (
                "yadif_cuda=0:-1:1," if deinterlace else ""
            )  # One frame per frame; frames marked progressive pass.
            # HDR tone mapping: prefer libplacebo (Vulkan GPU), fall back to CPU zscale+tonemap.
            if is_hdr:
                if _has_libplacebo_filter():
                    tonemap = "hwdownload,format=p010le,libplacebo=tonemapping=hable:colorspace=bt709:color_primaries=bt709:color_trc=bt709,format=nv12,hwupload_cuda,"
                else:
                    tonemap = "hwdownload,format=p010le,zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=nv12,hwupload_cuda,"
            else:
                tonemap = ""
            vf = f"{deint}{tonemap}{scale}"
        elif vaapi_decode:
            # VAAPI decode + VAAPI filters + hwdownload + hwupload_cuda for NVENC.
            pre = [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-hwaccel_device",
                vaapi_device,
            ]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            deint = "deinterlace_vaapi=auto=1," if deinterlace else ""
            vf = f"{deint}{tonemap}{scale},hwdownload,format=nv12,hwupload_cuda"
        else:
            # Software decode, upload to GPU for scaling/encoding.
            pre = []
            scale = f"scale_cuda=-2:{h}:format=nv12" if h else "scale_cuda=format=nv12"
            # HDR tone mapping: prefer libplacebo (Vulkan GPU), fall back to CPU zscale+tonemap
            # Deinterlace before tonemap (CPU yadif) for consistency with hw decode path.
            if sr_filter:
                # SR path: CPU decode -> deinterlace -> SR (GPU) -> encode
                # SR filter ends with scale_cuda, outputs CUDA frames ready for nvenc
                # Need init_hw_device for TensorRT dnn_processing to use GPU.
                pre = [
                    "-init_hw_device",
                    "cuda=cu",
                    "-filter_hw_device",
                    "cu",
                    "-filter_threads",
                    str(sr_filter_threads),
                ]
                deint = "yadif=0:-1:1," if deinterlace else ""
                vf = f"{deint}{sr_filter}"
            elif is_hdr:
                deint = "yadif=0:-1:1," if deinterlace else ""  # CPU deinterlace before tonemap.
                if _has_libplacebo_filter():
                    tonemap = "libplacebo=tonemapping=hable:colorspace=bt709:color_primaries=bt709:color_trc=bt709,format=nv12,hwupload_cuda,"
                else:
                    tonemap = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=nv12,hwupload_cuda,"
                vf = f"{deint}{tonemap}{scale}"
            else:
                deint = "yadif_cuda=0:-1:1," if deinterlace else ""  # GPU deinterlace after upload.
                tonemap = "format=nv12,hwupload_cuda,"
                vf = f"{tonemap}{deint}{scale}"
        preset = "p5" if sr_filter else ("p4" if deinterlace else "p2")
        encoder = "h264_nvenc"
        enc_opts = [
            "-preset",
            preset,
            "-rc",
            "constqp",
            "-qp",
            str(qp),
            "-bf",
            "0" if sr_filter else "3",
            "-spatial-aq",
            "1",
            "-temporal-aq",
            "1",
        ]
        if not sr_filter:
            enc_opts[6:6] = ["-rc-lookahead", "32"]

    elif enc_type == "amf":
        # AMF has no hardware decode - always uses fallback for decode/filter.
        if vaapi_decode:
            # VAAPI decode + VAAPI filters + hwdownload for AMF encode.
            pre = [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-hwaccel_device",
                vaapi_device,
            ]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            deint = "deinterlace_vaapi=auto=1," if deinterlace else ""
            vf = f"{deint}{tonemap}{scale},hwdownload,format=nv12"
        else:
            # Software decode + software filters.
            pre = []
            if is_hdr:
                tonemap = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=nv12,"
            else:
                tonemap = ""
            deint = "yadif=0:-1:1," if deinterlace else ""
            scale = f"scale=-2:{h}," if h else ""
            vf = f"{deint}{tonemap}{scale}format=nv12"
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
                vaapi_device,
                "-extra_hw_frames",
                "3",
            ]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            # HDR tone mapping on VAAPI.
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            vf = (
                f"deinterlace_vaapi=auto=1,{tonemap}{scale}" if deinterlace else f"{tonemap}{scale}"
            )
        else:
            # Software decode, upload to GPU for scaling/encoding.
            pre = ["-vaapi_device", vaapi_device]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            deint = "deinterlace_vaapi=auto=1," if deinterlace else ""
            vf = f"format=nv12,hwupload,{deint}{tonemap}{scale}"
        encoder = "h264_vaapi"
        # Use baseline profile for older GPUs (e.g., AMD GCN 1.0) that don't support High profile.
        if cache.AVAILABLE_ENCODERS.get("vaapi_baseline_only") is True:
            # Baseline doesn't support B-frames.
            enc_opts = ["-rc_mode", "CQP", "-qp", str(qp), "-profile:v", "constrained_baseline"]
        else:
            enc_opts = ["-rc_mode", "CQP", "-qp", str(qp), "-bf", "3"]

    elif enc_type == "qsv":
        if use_hw_pipeline:
            pre = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
            scale = f"scale_qsv=w=-2:h={h}:format=nv12" if h else "scale_qsv=format=nv12"
            # Combine deinterlace and tonemap into single vpp_qsv call when possible.
            if deinterlace and is_hdr:
                vf = f"vpp_qsv=deinterlace=2:tonemap=1:format=nv12,{scale}"
            elif deinterlace:
                vf = f"vpp_qsv=deinterlace=2,{scale}"
            elif is_hdr:
                vf = f"vpp_qsv=tonemap=1:format=nv12,{scale}"
            else:
                vf = scale
        else:
            # Software decode, upload to GPU for scaling/encoding.
            pre = ["-init_hw_device", "qsv=hw", "-filter_hw_device", "hw"]
            scale = f"scale_qsv=w=-2:h={h}:format=nv12" if h else "scale_qsv=format=nv12"
            # Combine deinterlace and tonemap into single vpp_qsv call when possible.
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
        # HDR tone mapping on CPU.
        if is_hdr:
            tonemap = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
        else:
            tonemap = ""
        deint = (
            "yadif=0:-1:1," if deinterlace else ""
        )  # One frame per frame; frames marked progressive pass.
        if h:
            vf = f"{deint}{tonemap}scale=-2:{h},format=yuv420p"
        else:
            vf = f"{deint}{tonemap}format=yuv420p"
        crf = {"high": 20, "low": 32}.get(quality, 26)
        encoder = "libx264"
        enc_opts = ["-preset", "veryfast", "-crf", str(crf), "-bf", "3"]

    else:
        raise ValueError(f"Unrecognized hardware encoder: '{enc_type}'.")

    # Film with pulldown decodes slower than the stream's rate; left to the muxer,
    # the missing frames are duplicated with their CEA-608 bytes, garbling captions.
    # The fps filter fills them and re-paces the caption data.
    post = ["-vf", f"fps=source_fps,{vf}", "-c:v", encoder, *enc_opts, "-g", "60"]
    return pre, post


def _build_audio_args(*, copy_audio: bool, audio: MediaInfo | None) -> list[str]:
    """Build audio args."""
    if copy_audio:
        return ["-c:a", "copy"]
    audio_sample_rate = audio.audio_sample_rate if audio else None
    audio_channels = audio.audio_channels if audio else 0
    rate = str(audio_sample_rate) if audio_sample_rate in (44_100, 48_000) else "48000"
    if audio_channels > 2:
        # Every client decodes 5.1 AAC; wider layouts fold down to 5.1.
        return [
            "-c:a",
            "aac",
            "-af",
            "aformat=channel_layouts=5.1",
            "-ar",
            rate,
            "-b:a",
            "384k",
            "-profile:a",
            "aac_low",
        ]
    if audio_channels == 0:
        # Unknown layout (not probed): keep surround as 5.1, everything else as stereo.
        return [
            "-c:a",
            "aac",
            "-af",
            "aformat=channel_layouts=stereo|5.1",
            "-ar",
            rate,
            "-b:a",
            "192k",
            "-profile:a",
            "aac_low",
        ]
    return ["-c:a", "aac", "-ac", "2", "-ar", rate, "-b:a", "192k", "-profile:a", "aac_low"]


def _can_carry_fields(media_info: MediaInfo) -> bool:
    """Whether a broadcast codec and raster can switch to interlaced pictures midstream."""
    return media_info.video_codec in {"mpeg2video", "h264"} and media_info.height in {
        480,
        576,
        1080,
    }


def _can_copy_video(
    media_info: MediaInfo | None,
    max_resolution: str,
    *,
    allow_upscale: bool = True,
) -> bool:
    """Can video be stream-copied (compatible codec, no processing needed)?"""
    max_h = _max_resolution_height(max_resolution) or 9999
    # SR requires re-encode (can't copy video when SR is active)
    sr_active = bool(
        allow_upscale and _sr_engine_dir is not None and _load_settings().get("sr_model", ""),
    )
    return bool(
        media_info
        and media_info.video_codec == "h264"
        and media_info.pix_fmt == "yuv420p"
        and media_info.height <= max_h
        and not sr_active
        and not media_info.interlaced,  # Can't copy if deinterlacing needed.
    )


def _can_copy_audio(media_info: MediaInfo | None, *, passthrough: bool = False) -> bool:
    """Can audio be stream-copied?"""
    if passthrough and media_info and media_info.audio_codec in PASSTHROUGH_AUDIO_CODECS:
        return True
    profile = media_info.audio_profile.upper() if media_info else ""
    return bool(
        media_info
        and media_info.audio_codec == "aac"
        and media_info.audio_sample_rate in (44_100, 48_000)
        # HE-AAC has browser compatibility issues - only copy LC-AAC.
        and "HE" not in profile
        # Surround is copied only when positively identified as LC.
        and (
            media_info.audio_channels <= 2 or (media_info.audio_channels <= 6 and profile == "LC")
        ),
    )


def _max_resolution_height(max_resolution: str) -> int | None:
    """Return the scale-down height for a resolution setting, or None for no limit."""
    return {"4k": 2160, "1080p": 1080, "720p": 720, "480p": 480}.get(
        max_resolution,
    )


class _SeriesProbe(TypedDict):
    name: NotRequired[str]
    mru: NotRequired[int | None]
    episodes: dict[int, _ProbeEntry]


class _CachedSeries(TypedDict, total=False):
    name: str
    mru: int | None
    episodes: dict[str, _CachedEpisode]


def _cached_probe(
    url: str,
    series_id: int | None,
    episode_id: int | None,
) -> tuple[MediaInfo | None, list[SubtitleStream]] | None:
    """The episode's, its series MRU's, or ``url``'s cached probe; None when uncached."""
    with _probe_lock:
        if series_id is not None and (series := _series_probe_cache.get(series_id)):
            episodes = series.get("episodes", {})
            hit = episode_id if episode_id in episodes else series.get("mru")
            if hit is not None and hit in episodes:
                log.info(
                    "Probe cache hit: series=%d episode=%s from=%d",
                    series_id,
                    episode_id,
                    hit,
                )
                _, media_info, subtitles = episodes[hit]
                return media_info, subtitles
        cached = _probe_cache.get(url)
    if cached is None:
        return None
    log.info("Probe cache hit for %s", url[:50])
    return cached[1], cached[2]


def _media_info_from_probe(
    data: _ProbeOutput,
    *,
    is_hls: bool,
) -> tuple[MediaInfo | None, list[SubtitleStream]]:
    """Parse ffprobe stream and format output into source facts and text subtitles."""
    video_codec = pix_fmt = ""
    subtitle_codecs: list[str] = []
    subtitles: list[SubtitleStream] = []
    audio_tracks: list[AudioTrack] = []

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
            stream_height = stream.get("height")
            height = 0 if stream_height is None else stream_height
            # Detect interlacing from field_order (tt, bb, tb, bt = interlaced)
            field_order = stream.get("field_order", "").lower()
            interlaced = field_order in ("tt", "bb", "tb", "bt")
            # Detect 10-bit from pix_fmt (e.g. yuv420p10le, p010le)
            # Check for "p10" or "10le/10be" to avoid false positive on yuv410p.
            is_10bit = "p10" in pix_fmt or "10le" in pix_fmt or "10be" in pix_fmt
            # Detect HDR from color_transfer (PQ = smpte2084, HLG = arib-std-b67)
            color_transfer = stream.get("color_transfer", "").lower()
            is_hdr = color_transfer in ("smpte2084", "arib-std-b67")
            # Try to get bitrate from stream, fall back to format.
            with suppress(ValueError):
                video_bitrate = _probe_int(stream.get("bit_rate"))
        elif codec_type == "audio":
            audio_tracks.append(
                AudioTrack(
                    index=stream.get("index"),
                    lang=stream.get("tags", {}).get("language", "und").lower(),
                    codec=codec,
                    channels=_probe_int(stream.get("channels")),
                    sample_rate=_probe_int(stream.get("sample_rate")),
                    profile=stream.get("profile", ""),
                    default=stream.get("disposition", {}).get("default") == 1,
                ),
            )
        elif codec_type == "subtitle":
            subtitle_codecs.append(codec)
            if codec in {"subrip", "ass", "ssa", "mov_text", "webvtt", "srt"}:
                idx = stream.get("index")
                if idx is not None:
                    tags = stream.get("tags", {})
                    lang = tags.get("language", "und").lower()
                    name = tags.get("name", "") or tags.get("title", "") or _lang_display_name(lang)
                    subtitles.append(
                        SubtitleStream(
                            index=idx,
                            lang=lang,
                            name=name,
                        ),
                    )

    duration = 0.0
    fmt = data.get("format", {})
    format_duration = fmt.get("duration")
    if format_duration is not None and format_duration != "":
        with suppress(ValueError):
            duration = float(format_duration)
    # Fall back to format bitrate if stream bitrate unavailable (common for MKV)
    if not video_bitrate:
        with suppress(ValueError):
            video_bitrate = _probe_int(fmt.get("bit_rate"))

    if not video_codec:
        return None, []

    media_info = MediaInfo(
        video_codec=video_codec,
        audio_codec="",
        pix_fmt=pix_fmt,
        subtitle_codecs=subtitle_codecs or None,
        duration=duration,
        height=height,
        video_bitrate=video_bitrate,
        interlaced=interlaced,
        is_10bit=is_10bit,
        is_hdr=is_hdr,
        is_hls=is_hls,
        audio_tracks=tuple(audio_tracks),
    )
    default = next((t for t in audio_tracks if t.default), None)
    return with_audio_track(
        media_info,
        default or (audio_tracks[0] if audio_tracks else None),
    ), subtitles
