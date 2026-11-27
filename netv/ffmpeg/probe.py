"""Media probing: what a source holds, and the caches that remember it."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Final, NotRequired, TypedDict
from urllib.parse import urljoin

import functools
import json
import logging
import pathlib
import subprocess
import threading
import time

from treekle import json_loads

from netv import cache, proxy
from netv.cache import get_user_agent
from netv.ffmpeg.letterbox import CachedCrop, CropBox
from netv.util import atomic_write_json, safe_urlopen


log = logging.getLogger(__name__)

# Module state.
_probe_lock = threading.Lock()

_probe_cache: dict[str, _ProbeEntry] = {}

_series_probe_cache: dict[int, _SeriesProbe] = {}


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
    width: int = 0
    crop: CropBox | None = None  # The picture's area, as transcodes of it detected.


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
    width: int
    crop: CachedCrop
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
    width: int | None
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


def remember_crop(
    url: str,
    crop: CropBox,
    series_id: int | None = None,
    episode_id: int | None = None,
) -> None:
    """Widen ``url``'s cached picture area by ``crop``, so later transcodes cut its bars.

    Only a probed source records one: the crop needs its frame size.

    Args:
        url: Probed media URL.
        crop: Picture area a transcode of it detected.
        series_id: Series cache key, when the media is an episode.
        episode_id: Episode cache key.
    """
    with _probe_lock:
        if url in _probe_cache:
            _probe_cache[url] = _widened(_probe_cache[url], crop)
        if series_id is not None and (series := _series_probe_cache.get(series_id)) is not None:
            episodes = series["episodes"]
            eid = episode_id if episode_id is not None else 0
            if eid in episodes:
                episodes[eid] = _widened(episodes[eid], crop)
    _save_url_probe_cache()
    if series_id is not None:
        _save_series_probe_cache()


def load_caches() -> None:
    """Load the probe caches persisted by an earlier run."""
    _load_series_probe_cache()
    _load_url_probe_cache()


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


def _probe_to_cache(
    cache_time: float,
    media_info: MediaInfo,
    subs: list[SubtitleStream],
) -> _CachedEpisode:
    """One probe as stored on disk."""
    entry: _CachedEpisode = {
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
        "width": media_info.width,
        "subtitles": [{"index": s.index, "lang": s.lang, "name": s.name} for s in subs],
    }
    if media_info.crop is not None:
        entry["crop"] = media_info.crop.to_cache()
    return entry


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
        width=entry.get("width", 0),
        crop=CropBox.from_cache(entry["crop"]) if "crop" in entry else None,
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
                # Another episode's codecs carry over; its bars may not.
                if hit != episode_id and media_info is not None and media_info.crop is not None:
                    media_info = replace(media_info, crop=None)
                return media_info, subtitles
        cached = _probe_cache.get(url)
    if cached is None:
        return None
    log.info("Probe cache hit for %s", url[:50])
    return cached[1], cached[2]


def _widened(entry: _ProbeEntry, crop: CropBox) -> _ProbeEntry:
    """A cached probe whose picture area also holds ``crop``."""
    cache_time, media_info, subtitles = entry
    if media_info is None:
        return entry
    seen = crop if media_info.crop is None else media_info.crop.union(crop)
    return cache_time, replace(media_info, crop=seen), subtitles


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

    width = height = 0
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
            stream_width = stream.get("width")
            width = 0 if stream_width is None else stream_width
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
        width=width,
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
