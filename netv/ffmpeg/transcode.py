"""Transcode sessions: starting, seeking, recovering, and stopping each ffmpeg."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    Protocol,
    Required,
    TypeAlias,
    TypedDict,
    TypeGuard,
    cast,
)
from urllib import parse

import asyncio
import contextlib
import functools
import io
import json
import logging
import math
import pathlib
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid

from treekle import json_loads

from netv import proxy
from netv.cache import (
    HwAccel,
    ServerSettings,
    get_user_agent,
    load_user_settings,
    relay_hints,
    remember_burst_gap,
)
from netv.ffmpeg.commands import (
    SEG_PREFIX,
    build_hls_ffmpeg_cmd,
    can_remux_live,
    encoder_crop,
    get_hls_segment_duration,
    get_live_hls_list_size,
    get_settings,
    get_transcode_dir,
    live_video_bitrates,
    uses_audio_passthrough,
)
from netv.ffmpeg.fast_start import (
    closed_caption_master_playlist,
    dated_playlist,
    encoder_command,
    first_video_pts,
    is_hls_url,
    keyframe_replay,
    last_media_sequence,
    live_buffer_durations,
    master_playlist,
    playlist_end_pts,
    ready_bitrate,
    recorder_command,
    remux_command,
    segment_pts,
    source_command,
    splice_playlists,
    splice_point,
    surround_audio_bitrate,
)
from netv.ffmpeg.letterbox import CachedCrop, CropBox, detected_crop
from netv.ffmpeg.probe import (
    CachedAudioTrack,
    MediaInfo,
    SubtitleStream,
    audio_tracks_from_cache,
    audio_tracks_to_cache,
    has_cached_probe,
    probe_closed_captions,
    probe_media,
    probe_segment,
    remember_crop,
    remember_probe,
    resolve_hls_master_playlist,
    restore_probe_cache_entry,
    with_audio_language,
)
from netv.m3u import live_ts_offered
from netv.playback_policy import LiveRecoveryPolicy, PlaybackHealth, PlaybackPolicy
from netv.relay import Readable, Relay, Subscription
from netv.util import safe_urlopen


if TYPE_CHECKING:
    import fastapi
else:
    from wrapt import lazy_import

    # A module proxy, not lazy_import("fastapi", "HTTPException"): `except` and
    # pytest.raises need the real class, which attribute access returns.
    fastapi = lazy_import("fastapi")


log = logging.getLogger(__name__)


class ManagedProcess(Protocol):
    """An ffmpeg process handle, or a placeholder for one that has exited."""

    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int | None: ...


VodType: TypeAlias = Literal["movie", "series"]
Rendition: TypeAlias = Literal["low", "up"]
ContentType: TypeAlias = Literal["live"] | VodType


class SubtitleInfo(TypedDict):
    lang: str
    name: str


class SubtitleTrack(TypedDict):
    url: str
    lang: str
    label: str
    default: bool


class StoredSubtitle(TypedDict):
    """A subtitle as sessions and session.json keep it; older files may omit lang or name."""

    index: int
    lang: NotRequired[str]
    name: NotRequired[str]


class TranscodeSession(TypedDict, total=False):
    """Mutable state of one transcode session; `get_session` returns a shallow copy."""

    dir: Required[str]
    process: Required[ManagedProcess | None]
    started: Required[float]
    url: Required[str]
    last_access: float
    is_vod: bool
    is_archive: bool
    request_url: str
    username: str
    guests: frozenset[str]  # Other users a shared VOD session was handed to.
    source_id: str
    # Recovered pre-dict sessions stored bare stream indices.
    subtitles: list[StoredSubtitle | int]
    duration: float
    seek_offset: float
    evicted_below: int  # VOD segments numbered below this were deleted after playback.
    series_id: int | None
    episode_id: int | None
    audio_passthrough: bool
    audio_lang: str  # Preferred audio language the session was started for.
    bandwidth_saver: bool
    recovery_bitrate: float
    caption_timestamp_origin: float
    fast_start: bool
    playlist_generation: int
    playback_buffer_seconds: float
    closed_captions: bool
    origin_pts: float | None
    origin_time: float
    segment_dates: dict[str, float]
    audio_info: MediaInfo | None
    watchdog_stale_after: float
    watchdog_task: asyncio.Task[None]
    watchdog_recovering: str
    pipeline_restart_log_at: float
    # The crop every ffmpeg of a VOD session cuts, so its segments share a frame size.
    crop: CropBox


@dataclass(frozen=True, kw_only=True, slots=True)
class _FastLivePlan:
    """What a fast-start session reads and how each of its workers publishes it."""

    source_url: str
    # The configured source that listed it, whose network its relay may reach.
    source_id: str
    user_agent: str | None
    hw: HwAccel
    resolution: str
    quality: str
    deinterlace: bool
    bandwidth_saver: bool
    # Whether the client can play Dolby audio, not whether this source has it.
    audio_passthrough: bool
    dvr: bool

    def relay(self) -> Relay:
        """A relay opening the session's one upstream connection."""
        # Wall-clock stats date the source: the first byte's time is its origin.
        return Relay(
            functools.partial(_open_source, self.source_url, self.user_agent, self.source_id),
            clock=time.time,
        )

    def worker(
        self,
        directory: str,
        source: MediaInfo | None,
        *,
        restarting: bool,
        rendition: Rendition = "low",
        grid_offset: float = 0.0,
    ) -> tuple[list[str], str]:
        """Choose the worker for ``source``: a stream copy when every client plays it.

        Args:
            directory: Session directory.
            source: Probed source facts, or None when unknown.
            restarting: Append to the rendition an earlier worker published.
            rendition: "low" starts at once; "up" applies AI Upscale where available.
            grid_offset: Source time of the worker's first frame, modulo one second.

        Returns:
            cmd: FFmpeg arguments reading the relay from stdin.
            role: "remux", "encoder", or "upscale", for logs.
        """
        upscale = rendition == "up"
        # A DVR session needs only server-side retention, so a source every client
        # plays is copied rather than re-encoded.
        if (
            not upscale
            and not self.bandwidth_saver
            and self.dvr
            and can_remux_live(source, self.resolution, audio_passthrough=self.audio_passthrough)
        ):
            return remux_command(directory, restarting=restarting), "remux"
        cmd = encoder_command(
            directory,
            self.hw,
            self.resolution,
            self.quality,
            self.deinterlace,
            allow_upscale=upscale,
            source=source,
            audio_passthrough=self.audio_passthrough,
            restarting=restarting,
            rendition=rendition,
            grid_offset=grid_offset,
        )
        return cmd, "upscale" if upscale else "encoder"

    def upscales(self, directory: str, source: MediaInfo | None) -> bool:
        """Whether an AI Upscale worker would apply to ``source`` (a TensorRT engine fits)."""
        if self.bandwidth_saver:
            return False
        cmd, _ = self.worker(directory, source, restarting=False, rendition="up")
        return any("dnn_processing" in arg for arg in cmd)


class FastLiveSession(TranscodeSession):
    """A fast-start session: one relay reading upstream, feeding one worker."""

    # Player page loads sharing the session -> when each last reported. A reloaded
    # page joins before its old page's stop arrives, so stops count viewers.
    viewers: dict[str, float]
    viewer_joined: asyncio.Event  # Set when a page becomes a viewer.
    live_recovery: LiveRecoveryPolicy
    playback_policy: PlaybackPolicy
    plan: _FastLivePlan
    relay: Relay
    # Set once the source is probed and a worker launches.
    subscription: NotRequired[Subscription]
    master_audio_bitrate: NotRequired[int]
    # The rendition low.m3u8 serves: "up" once the AI Upscale worker caught up.
    rendition: NotRequired[Rendition]
    # The plain playlist as last served, which the upscaled one continues.
    splice_low: NotRequired[str]
    # The upscaled media sequence number that continues it, once known.
    splice_point: NotRequired[int]


class TranscodeResponse(TypedDict):
    session_id: str
    playlist: str
    subtitles: list[SubtitleTrack]
    duration: float
    seek_offset: float
    caption_playlist: NotRequired[str]
    master_playlist: NotRequired[str]
    playback_buffer_seconds: NotRequired[float]
    transcoded_duration: NotRequired[float]
    # This page load's id in a shared live session; sent back with health and stop.
    viewer_id: NotRequired[str]


class HealthResponse(TypedDict):
    bandwidth_saver: bool
    playlist: NotRequired[str]
    caption_playlist: NotRequired[str]


class SeekResponse(TypedDict, total=False):
    session_id: str
    playlist: str
    ok: bool
    segment: int
    time: float


class SessionProgress(TypedDict):
    segment_count: int
    duration: float


class _SessionProbe(TypedDict, total=False):
    audio_tracks: list[CachedAudioTrack]
    video_codec: str
    audio_codec: str
    pix_fmt: str
    audio_channels: int
    audio_sample_rate: int
    subtitle_codecs: list[str] | None
    height: int
    video_bitrate: int
    interlaced: bool


# Module state.
_transcode_sessions: dict[str, TranscodeSession] = {}
_url_to_session: dict[str, str] = {}  # URL -> session_id (all content types)
_transcode_lock = threading.Lock()
_background_tasks: set[asyncio.Task[None]] = set()
_startup_guards: dict[str, _StartupGuard] = {}  # URL -> in-flight lookup-or-start.


# ===========================================================================
# Cache Timeout Helpers
# ===========================================================================


def get_vod_cache_timeout() -> int:
    """Get VOD session cache timeout in seconds."""
    return get_settings().get("vod_transcode_cache_mins", 60) * 60


# ===========================================================================
# Session Validity
# ===========================================================================


def is_session_valid(
    session: TranscodeSession,
    *,
    heartbeat_timeout_sec: float = 30.0,
) -> bool:
    """Check heartbeat freshness and process or cache availability.

    Args:
        session: Session state to inspect.
        heartbeat_timeout_sec: Maximum time without a client heartbeat.

    Returns:
        valid: Whether the session remains available for playback.
    """
    last_access = session.get("last_access", session["started"])
    time_since_heartbeat = time.time() - last_access

    if time_since_heartbeat > heartbeat_timeout_sec:
        return False

    # Keep a recently-accessed live session addressable while its watchdog
    # replaces a dead primary process.
    if session.get("fast_start", False) and _is_fast_live(session):
        return True

    # Active process with recent heartbeat = valid.
    if _is_process_alive(session.get("process")):
        return True

    # Dead process: only VOD keeps a cached session.
    cache_timeout = get_vod_cache_timeout()
    if cache_timeout <= 0:
        return False  # No caching of dead sessions.
    return time_since_heartbeat < cache_timeout


# ===========================================================================
# Session Start/Stop
# ===========================================================================


async def stop_session(
    session_id: str,
    force: bool = False,
    *,
    viewer: str = "",
    heartbeat_timeout_sec: float = 30.0,
) -> None:
    """Stop a transcode session once its processes have exited.

    Args:
        session_id: Session to stop.
        force: Bypass recent-access protection and remove cached segments.
        viewer: The leaving page's ``viewer_id``. A live session another viewer
            reported to within ``heartbeat_timeout_sec`` keeps playing; without
            one, it stops after a grace (``stop_when_unwatched``).
        heartbeat_timeout_sec: How long a silent viewer still counts.
    """
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None:
            return
        if viewer and _is_fast_live(session):
            viewers = session["viewers"]
            viewers.pop(viewer, None)
            cutoff = time.time() - heartbeat_timeout_sec
            for silent in [v for v, heard in viewers.items() if heard < cutoff]:
                del viewers[silent]
            if viewers:
                log.info("Viewer left session %s; %d remain", session_id, len(viewers))
            else:
                log.info("Last viewer left session %s", session_id)
                _spawn_background_task(stop_when_unwatched(session_id))
            return

        # Archive seeks open another upstream URL, and a fast-start session
        # cannot resume once its pipeline stops, so neither keeps a cached slot.
        force = force or session.get("fast_start") or _is_archive_url(session["url"])
        # Skip stop if session was accessed recently (race with seeking/resume,
        # or multiple users watching same stream)
        last_access = session.get("last_access")
        if not force and last_access is not None and time.time() - last_access < 5.0:
            log.info("Ignoring stop for recently-accessed session %s", session_id)
            return

        processes = _retire_session(session)
        # Only VOD reaches here unforced; cache it if a timeout is set.
        cached = not force and bool(get_vod_cache_timeout())
        if cached:
            session["last_access"] = time.time()
        else:
            del _transcode_sessions[session_id]
            if _url_to_session.get(session["url"]) == session_id:
                del _url_to_session[session["url"]]

    await _terminate_processes(processes)
    if session.get("is_vod", False) and not _is_archive_url(session["url"]):
        await asyncio.to_thread(
            _remember_detected_crop,
            session["url"],
            session["dir"],
            session.get("series_id"),
            session.get("episode_id"),
        )
    if _is_fast_live(session):
        subscription = session.get("subscription")
        if subscription is not None:
            subscription.cancel()
            await subscription.wait_closed()
        # A recording reading this relay keeps it open until the recording ends.
        await session["relay"].release()
        # Later sessions from this host buffer past the bursts this one saw.
        host = parse.urlsplit(session["plan"].source_url).netloc
        stats = session["relay"].stats
        await asyncio.to_thread(remember_burst_gap, host, stats.burst_gap_sec, stats.delivery_sec)
    if cached:
        log.info("Session %s cached (ffmpeg stopped, segments kept)", session_id)
        return
    await asyncio.to_thread(shutil.rmtree, session["dir"], ignore_errors=True)
    log.info("Stopped transcode session %s", session_id)


@dataclass(frozen=True, kw_only=True, slots=True)
class LiveRecording:
    """A recording of a live stream, controllable from any thread."""

    process: asyncio.subprocess.Process
    relay: Relay
    loop: asyncio.AbstractEventLoop

    def finished(self) -> bool:
        return self.process.returncode is not None

    def stop(self) -> None:
        """Stop recording, keeping what was recorded; call from outside the event loop."""
        asyncio.run_coroutine_threadsafe(_terminate_process(self.process), self.loop).result()


_recordings: set[LiveRecording] = set()


async def record_live(
    username: str,
    url: str,
    source_id: str,
    directory: str,
    seconds: float,
) -> LiveRecording:
    """Record a live stream, taking over the user's session of it if one is playing.

    The recording reads the session's relay, so no second upstream connection
    opens; the viewer's session stops and the relay closes when the recording ends.

    Args:
        username: User whose stream is recorded.
        url: Live URL as the client requested it.
        source_id: Source the URL belongs to.
        directory: Recording directory receiving stream.m3u8.
        seconds: Recording length.

    Returns:
        recording: The running recording.
    """
    with _transcode_lock:
        viewing = [
            (sid, s)
            for sid, s in _transcode_sessions.items()
            if _is_fast_live(s) and s.get("username") == username and s["url"] == url
        ]
    if viewing:
        stream = viewing[0][1]["relay"]
    else:
        source_url = await asyncio.to_thread(_relay_url, url, source_id)
        stream = Relay(
            functools.partial(_open_source, source_url, get_user_agent(), source_id),
            clock=time.time,
        )
        await stream.start()
    process: asyncio.subprocess.Process | None = None
    try:
        sample = await _wait_for_keyframe(stream)
        received = stream.stats.bytes_received
        path = pathlib.Path(directory) / "probe.ts"
        await asyncio.to_thread(path.write_bytes, sample)
        source = await asyncio.to_thread(probe_segment, str(path))
        remember_probe(url, source)
        settings = get_settings()
        cmd = recorder_command(
            directory,
            settings["transcode_hw"],
            settings.get("max_resolution", "1080p"),
            settings.get("quality", "high"),
            seconds,
            source,
        )
        process = await _launch_ffmpeg(cmd, stdin=asyncio.subprocess.PIPE)
        if process.stdin is None:
            raise ValueError("Expected process.stdin is not None.")
        subscription = stream.subscribe(process.stdin, replay=_since(stream, sample, received))
    except BaseException:
        await _terminate_process(process)
        if not viewing:
            await stream.close()
        raise
    recording = LiveRecording(process=process, relay=stream, loop=asyncio.get_running_loop())
    _recordings.add(recording)
    _spawn_background_task(_finish_recording(recording, subscription, pathlib.Path(directory)))
    for session_id, _ in viewing:
        await stop_session(session_id, force=True)
    await stream.release()
    log.info(
        "Recording %s for %.0fs from %s",
        directory,
        seconds,
        "its session" if viewing else "upstream",
    )
    return recording


async def cleanup_expired_sessions() -> None:
    """Clean up all expired sessions (VOD and live)."""
    with _transcode_lock:
        expired = [
            sid
            for sid, session in list(_transcode_sessions.items())
            if not is_session_valid(session)
        ]
    for session_id in expired:
        await stop_session(session_id, force=True)


async def shutdown() -> None:
    """Stop every ffmpeg process and upstream connection for a clean shutdown."""
    with _transcode_lock:
        sessions = list(_transcode_sessions.values())
        processes = [process for session in sessions for process in _retire_session(session)]
        _transcode_sessions.clear()
    recordings = list(_recordings)
    _recordings.clear()
    processes += [recording.process for recording in recordings]
    await _terminate_processes(processes)
    relays = {id(r.relay): r.relay for r in recordings}
    relays.update({id(s["relay"]): s["relay"] for s in sessions if _is_fast_live(s)})
    await asyncio.gather(*(relay.close() for relay in relays.values()))


# ===========================================================================
# Stream Limits
# ===========================================================================


def get_user_sessions(username: str) -> list[tuple[str, TranscodeSession]]:
    """Get all active sessions for a user, sorted by start time (oldest first)."""
    with _transcode_lock:
        sessions = [
            (sid, s) for sid, s in _transcode_sessions.items() if s.get("username") == username
        ]
    return sorted(sessions, key=lambda x: x[1].get("started", 0))


def get_source_sessions(source_id: str) -> list[tuple[str, TranscodeSession]]:
    """Get all active sessions for a source, sorted by start time (oldest first)."""
    with _transcode_lock:
        sessions = [
            (sid, s) for sid, s in _transcode_sessions.items() if s.get("source_id") == source_id
        ]
    return sorted(sessions, key=lambda x: x[1].get("started", 0))


async def enforce_stream_limits(
    username: str,
    source_id: str | None,
    user_max: int,
    source_max: int,
) -> str | None:
    """Enforce stream limits, stopping oldest sessions if needed.

    Args:
        username: User starting a stream.
        source_id: Upstream source, or None when unknown.
        user_max: Per-user stream limit; zero disables it.
        source_max: Per-source stream limit; zero disables it.

    Returns:
        error: Capacity message when the user cannot reclaim a source slot, else None.
    """
    # Check source limit first (hard limit - can only reclaim own slots)
    if source_id is not None and source_id and source_max > 0:
        source_sessions = get_source_sessions(source_id)
        if len(source_sessions) >= source_max:
            user_source_sessions = [
                (sid, s) for sid, s in source_sessions if s.get("username") == username
            ]
            if user_source_sessions:
                oldest_sid, _ = user_source_sessions[0]
                log.info(
                    "Source %s at limit (%d), stopping user %s's oldest session %s",
                    source_id,
                    source_max,
                    username,
                    oldest_sid,
                )
                await stop_session(oldest_sid, force=True)
            else:
                return f"Source at capacity ({source_max} streams)"

    # Check user limit (soft limit - auto-rotate oldest)
    if user_max > 0:
        user_sessions = get_user_sessions(username)
        if len(user_sessions) >= user_max:
            oldest_sid, _ = user_sessions[0]
            log.info(
                "User %s at limit (%d), stopping oldest session %s",
                username,
                user_max,
                oldest_sid,
            )
            await stop_session(oldest_sid, force=True)

    return None


# ===========================================================================
# Session Recovery (Startup)
# ===========================================================================


def cleanup_and_recover_sessions() -> None:
    """Clean up orphaned transcode dirs and recover valid VOD sessions.

    Called on startup to:
    1. Remove all orphaned dirs (no session.json - leftover live sessions)
    2. Remove expired VOD dirs (older than cache timeout)
    3. Recover valid VOD sessions for resume
    """
    cache_timeout = get_vod_cache_timeout()
    now = time.time()
    removed = recovered = 0

    for d in get_transcode_dir().glob("netv_transcode_*"):
        try:
            entry = d.stat()
        except OSError:  # Dangling symlink, or removed since the glob.
            continue
        if not stat.S_ISDIR(entry.st_mode):
            continue
        mtime = entry.st_mtime

        info_file = d / "session.json"

        # No session.json = orphaned (live session or failed VOD)
        if not info_file.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
            continue

        # Expired VOD session.
        if now - mtime > cache_timeout:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
            continue

        # No segments = nothing to recover.
        if not list(d.glob(f"{SEG_PREFIX}*.ts")):
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
            continue

        # Try to recover VOD session.
        try:
            info = json_loads(info_file.read_text(), _SessionInfo)
            url = info.get("url", "")
            if not (info.get("is_vod", False) and url) or _is_archive_url(url):
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
                continue

            session_id = info["session_id"]
            new_seek = info.get("seek_offset", 0)
            subtitles: list[StoredSubtitle | int] = info.get("subtitles", []) or [
                *info.get("subtitle_indices", []),
            ]

            with _transcode_lock:
                _transcode_sessions[session_id] = {
                    "dir": str(d),
                    "process": _DeadProcess(),
                    "started": info.get("started", mtime),
                    "url": url,
                    "is_vod": True,
                    "last_access": now,  # Use current time, not mtime, to avoid immediate expiration.
                    "subtitles": subtitles,
                    "duration": info.get("duration", 0),
                    "seek_offset": new_seek,
                    "series_id": info.get("series_id"),
                    "episode_id": info.get("episode_id"),
                    "username": info.get("username", ""),
                    "source_id": info.get("source_id", ""),
                    "audio_passthrough": info.get("audio_passthrough", False),
                    "audio_lang": info.get("audio_lang", ""),
                }
                if "crop" in info:
                    _transcode_sessions[session_id]["crop"] = CropBox.from_cache(info["crop"])
                # Prefer session with seek_offset or more recent mtime.
                existing_id = _url_to_session.get(url)
                existing = None if existing_id is None else _transcode_sessions.get(existing_id)
                if existing_id is None:
                    _url_to_session[url] = session_id
                else:
                    existing_seek = 0.0 if existing is None else existing.get("seek_offset", 0)
                    existing_mtime = 0.0 if existing is None else existing.get("last_access", 0)
                    if (new_seek > 0 and existing_seek == 0) or (
                        existing_seek == 0 and new_seek == 0 and mtime > existing_mtime
                    ):
                        _url_to_session[url] = session_id

            # Restore probe cache.
            if (p := info.get("probe")) is not None and "audio_tracks" in p:
                media_info = MediaInfo(
                    video_codec=p.get("video_codec", ""),
                    audio_codec=p.get("audio_codec", ""),
                    pix_fmt=p.get("pix_fmt", ""),
                    audio_channels=p.get("audio_channels", 0),
                    audio_sample_rate=p.get("audio_sample_rate", 0),
                    subtitle_codecs=p.get("subtitle_codecs"),
                    duration=info.get("duration", 0),
                    height=p.get("height", 0),
                    video_bitrate=p.get("video_bitrate", 0),
                    interlaced=p.get("interlaced", False),
                    audio_tracks=audio_tracks_from_cache(p.get("audio_tracks", [])),
                )
                subs = [
                    SubtitleStream(
                        index=subtitle["index"],
                        lang=subtitle.get("lang", "und"),
                        name=subtitle.get("name", ""),
                    )
                    for subtitle in subtitles
                    if not isinstance(subtitle, int)
                ]
                restore_probe_cache_entry(
                    url,
                    media_info,
                    subs,
                    info.get("series_id"),
                    info.get("episode_id"),
                )
            recovered += 1
            log.debug("Recovered VOD session %s for %s", session_id, url[:50])
        except Exception as e:
            log.warning("Failed to recover session from %s: %s", d, e)
            shutil.rmtree(d, ignore_errors=True)
            removed += 1

    if removed or recovered:
        log.info(
            "Startup cleanup: removed %d orphaned dirs, recovered %d VOD sessions",
            removed,
            recovered,
        )


# ===========================================================================
# FFmpeg Monitoring
# ===========================================================================


# ===========================================================================
# Playlist Helpers
# ===========================================================================


def get_caption_timestamp_origin(session_id: str) -> float | None:
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None:
            return None
        existing = session.get("caption_timestamp_origin")
        if existing is not None:
            return float(existing)
        output_dir = session["dir"]

    origin = _caption_timestamp_origin(output_dir)
    if origin is None:
        return None
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None:
            return None
        return float(session.setdefault("caption_timestamp_origin", origin))


def dated_fast_playlist(session_id: str, content: str) -> str:
    """Add program dates to a fast-start playlist, dating each segment once.

    Args:
        session_id: Fast-start session whose clock dates the segments.
        content: Text of the session's media playlist.

    Returns:
        playlist: Playlist with program dates, or ``content`` when the clock is unknown.
    """
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None:
            return content
        origin_pts = session.get("origin_pts")
        if origin_pts is None:
            return content
        directory = session["dir"]
        if "origin_time" not in session:
            raise KeyError("origin_time")
        origin_time = session["origin_time"]
        dates = session.get("segment_dates", {})
    content, dates = dated_playlist(directory, content, origin_pts, origin_time, dates)
    with _transcode_lock:
        session["segment_dates"] = dates
    return content


def add_webvtt_timestamp_map(content: str, origin_pts: float | None) -> str:
    if origin_pts is None or "X-TIMESTAMP-MAP=" in content:
        return content
    line_ending = "\r\n" if content.startswith("WEBVTT\r\n") else "\n"
    header = f"WEBVTT{line_ending}"
    if not content.startswith(header):
        return content
    timestamp = round(origin_pts * 90_000) % (1 << 33)
    mapping = f"X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:{timestamp}{line_ending}"
    return content.replace(header, header + mapping, 1)


# ===========================================================================
# Session Snapshots
# ===========================================================================


@dataclass(kw_only=True, slots=True, frozen=True)
class _SessionSnapshot:
    """Immutable snapshot of session state for lock-free access."""

    output_dir: str
    process: ManagedProcess | None
    seek_offset: float
    subtitles: list[StoredSubtitle | int]
    duration: float
    audio_passthrough: bool = False
    username: str = ""
    source_id: str = ""
    crop: CropBox | None = None


# ===========================================================================
# Existing Session Handling
# ===========================================================================


# ===========================================================================
# Core Transcode Logic
# ===========================================================================


async def start_transcode(
    url: str,
    content_type: ContentType = "live",
    series_id: int | None = None,
    episode_id: int | None = None,
    series_name: str = "",
    deinterlace_fallback: bool = True,
    username: str = "",
    source_id: str = "",
    user_max_streams: int = 0,
    source_max_streams: int = 0,
    bandwidth_saver: bool = False,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    audio_passthrough: bool = False,
    min_segments: int = 2,
) -> TranscodeResponse:
    """Start or reuse a transcode session; every live stream is read through a relay.

    Args:
        url: Requested stream URL.
        content_type: "live", "movie", or "series".
        series_id: Series probe cache key.
        episode_id: Episode probe cache key.
        series_name: Series display name for the probe cache.
        deinterlace_fallback: Deinterlace when probe data is unavailable.
        username: Session owner.
        source_id: Upstream source used for stream limits.
        user_max_streams: Per-user stream limit; zero disables it.
        source_max_streams: Per-source stream limit; zero disables it.
        bandwidth_saver: Start live playback at reduced quality.
        is_disconnected: Reports whether the requesting client has gone away.
        audio_passthrough: The client decodes AC-3/E-AC-3, so Dolby may be copied.
        min_segments: Live segments a new session publishes before it is returned;
            one for a client that waits for its own buffer, two for any player.

    Returns:
        response: Session identifier, playlist URLs, and playback metadata.
    """
    if bandwidth_saver and content_type != "live":
        raise fastapi.HTTPException(400, "Adaptive playback is only supported for live streams")
    # Until a start registers its session, a concurrent request for the URL
    # would open a second upstream connection or reuse an unfinished startup.
    async with _url_startup(url):
        return await _start_or_reuse(
            url,
            content_type,
            series_id,
            episode_id,
            series_name,
            deinterlace_fallback,
            username,
            source_id,
            user_max_streams,
            source_max_streams,
            bandwidth_saver,
            is_disconnected,
            audio_passthrough,
            min_segments,
        )


# A reload's old page sends its stop as it unloads, usually before the new page asks
# for the stream: stopping at once would make the new page start it from scratch.
async def stop_when_unwatched(session_id: str, *, grace_sec: float = 10.0) -> None:
    """Stop a live session that still has no viewer ``grace_sec`` from now.

    Args:
        session_id: Live session whose last viewer left, or that no page joined yet.
        grace_sec: How long a page has to join before the stream stops.
    """
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None or not _is_fast_live(session) or session["viewers"]:
            return
        joined = session["viewer_joined"]
        joined.clear()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(joined.wait(), grace_sec)
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None or not _is_fast_live(session) or session["viewers"]:
            return
    log.info("No viewer joined session %s within %.0fs", session_id, grace_sec)
    await stop_session(session_id)


def prestart_live(
    url: str,
    *,
    username: str,
    source_id: str,
    deinterlace_fallback: bool,
    user_max_streams: int,
    source_max_streams: int,
) -> None:
    """Start the live session a play page is about to request, without a viewer.

    The tuner, probe, and encoder start while the page's scripts load; the page's
    ``/transcode/start`` then joins the startup in progress and becomes its viewer.
    A start that fails is left for the page's own request to report.

    Args:
        url: Live stream URL the page will play.
        username: Viewer the page belongs to.
        source_id: Source used for access checks and stream limits.
        deinterlace_fallback: Deinterlace when probe data is unavailable.
        user_max_streams: Per-user stream limit; zero disables it.
        source_max_streams: Per-source stream limit; zero disables it.
    """
    _spawn_background_task(
        _prestart(
            url,
            username,
            source_id,
            deinterlace_fallback,
            user_max_streams,
            source_max_streams,
        ),
    )


# The channel each viewer's guide focus rests on, started before they press Enter.
_pretuned: dict[str, str] = {}


def pretune_live(
    url: str,
    *,
    username: str,
    source_id: str,
    deinterlace_fallback: bool,
    user_max_streams: int,
    source_max_streams: int,
) -> None:
    """Start the channel a viewer rests on in the guide, releasing their previous one.

    Like ``prestart_live``, the session has no viewer until a page joins it, so it stops
    ``stop_when_unwatched``'s grace after starting if none does. A viewer holds one
    pretune at a time.

    Args:
        url: Live stream URL the guide's focus rests on.
        username: Viewer browsing the guide.
        source_id: Source used for access checks and stream limits.
        deinterlace_fallback: Deinterlace when probe data is unavailable.
        user_max_streams: Per-user stream limit; zero disables it.
        source_max_streams: Per-source stream limit; zero disables it.
    """
    previous = _pretuned.get(username)
    _pretuned[username] = url
    if previous is not None and previous != url:
        _spawn_background_task(_stop_if_unwatched(previous))
    prestart_live(
        url,
        username=username,
        source_id=source_id,
        deinterlace_fallback=deinterlace_fallback,
        user_max_streams=user_max_streams,
        source_max_streams=source_max_streams,
    )


async def release_pretune(username: str) -> None:
    """Stop ``username``'s pretuned channel now, unless a page has joined it."""
    url = _pretuned.pop(username, None)
    if url is not None:
        await _stop_if_unwatched(url)


# ===========================================================================
# Session Query/Update
# ===========================================================================


def get_session(session_id: str) -> TranscodeSession | None:
    """Get a shallow copy of the session state (safe to read outside the lock)."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        return None if session is None else session.copy()


def user_session(session_id: str, username: str) -> TranscodeSession | None:
    """A copy of the session ``username`` started or was handed, else None."""
    session = get_session(session_id)
    if session is None or username not in {session.get("username"), *session.get("guests", ())}:
        return None
    return session


def touch_session(session_id: str) -> bool:
    """Record a session heartbeat.

    Args:
        session_id: Session to refresh.

    Returns:
        exists: Whether the session was found.
    """
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is not None:
            session["last_access"] = time.time()
            return True
        return False


def note_segment_served(
    session_id: str,
    filename: str,
    *,
    keep_behind_sec: float = 1_200,
) -> int:
    """Delete VOD segments far behind the one a player just fetched.

    A long movie transcoded whole would not fit a RAM-backed transcode dir. A seek
    back to a deleted segment restarts ffmpeg there, as any seek past the transcode does.

    Args:
        session_id: Session the segment belongs to.
        filename: Served file name.
        keep_behind_sec: Seconds of played segments kept for seeking back.

    Returns:
        deleted: Number of segment files removed.
    """
    stem = filename.removesuffix(".ts")
    if stem == filename or not stem.startswith(SEG_PREFIX):
        return 0
    try:
        served = int(stem[len(SEG_PREFIX) :])
    except ValueError:
        return 0
    keep = math.ceil(keep_behind_sec / get_hls_segment_duration())
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if (
            session is None
            or session.get("is_vod") is not True
            or session.get("fast_start") is True
        ):
            return 0
        evicted_below = session.get("evicted_below", 0)
        # Playback moved back (a seek restarted ffmpeg earlier): evict from there again.
        if served < evicted_below:
            evicted_below = max(0, served - keep)
        cutoff = served - keep
        session["evicted_below"] = max(evicted_below, cutoff)
        directory = pathlib.Path(session["dir"])
    deleted = 0
    for number in range(evicted_below, cutoff):
        try:
            (directory / f"{SEG_PREFIX}{number:03d}.ts").unlink()
        except FileNotFoundError:
            continue
        deleted += 1
    return deleted


def note_viewer(session_id: str, viewer: str) -> None:
    """Record that a player page still watches a live session.

    Args:
        session_id: Live session the page reported on.
        viewer: The page's ``viewer_id``; empty for clients without one.
    """
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        # Only a page still counted is refreshed: a report in flight when its page
        # left must not bring it back and hold the stream.
        if session is not None and _is_fast_live(session) and viewer in session["viewers"]:
            session["viewers"][viewer] = time.time()


def get_session_progress(session_id: str, username: str) -> SessionProgress | None:
    """Get transcode progress for a session, recording a heartbeat.

    Args:
        session_id: Session to inspect.
        username: Viewer asking; another user's session reads as absent.

    Returns:
        progress: Segment count and duration of the selected playlist, or None.
    """
    session = user_session(session_id, username)
    if session is None:
        return None
    touch_session(session_id)
    playlist_path = pathlib.Path(session["dir"]) / playlist_file(session)
    if not playlist_path.exists():
        return {"segment_count": 0, "duration": 0.0}
    durations = re.findall(r"#EXTINF:([\d.]+)", playlist_path.read_text())
    return {
        "segment_count": len(durations),
        "duration": sum(float(d) for d in durations),
    }


def crop_view(session_id: str, username: str) -> CropBox | None:
    """The part of a session's output frame that is picture, for players to show alone.

    Args:
        session_id: Session to inspect.
        username: Viewer asking; another user's session is not found.

    Returns:
        crop: Picture area as fractions of the output frame, past any crop the encoder
            applied; None until detection has sampled five seconds.

    Raises:
        HTTPException: 404 when the session is not the viewer's.
    """
    session = user_session(session_id, username)
    if session is None:
        raise fastapi.HTTPException(404, "Session not found")
    # A tight early box shows a little less picture until it widens; the encoder
    # cache, which cuts pixels for good, waits the full minute.
    seen = detected_crop(session["dir"], min_samples=5)
    if seen is None:
        return None
    return seen.within(session.get("crop", CropBox()))


def clear_url_session(url: str) -> str | None:
    """Clear URL-to-session mapping."""
    with _transcode_lock:
        return _url_to_session.pop(url, None)


# ===========================================================================
# Seek
# ===========================================================================


@dataclass(kw_only=True, slots=True, frozen=True)
class _SeekSessionInfo:
    """Snapshot of session info needed for seek."""

    url: str
    output_dir: str
    process: ManagedProcess | None
    subtitles: list[StoredSubtitle | int]
    series_id: int | None
    episode_id: int | None
    audio_passthrough: bool = False
    username: str = ""
    source_id: str = ""
    crop: CropBox | None = None


async def seek_transcode(
    session_id: str,
    seek_time: float,
    *,
    min_segment_size_bytes: int = 1_000,
    playlist_wait_timeout_sec: float = 30.0,
) -> SeekResponse:
    """Seek to a specific time in a VOD session.

    Args:
        session_id: Existing VOD session identifier.
        seek_time: Requested position in seconds.
        min_segment_size_bytes: Minimum complete segment size for smart seek.
        playlist_wait_timeout_sec: Deadline for the restarted playlist.

    Returns:
        response: Cached playlist location or new seek position.
    """
    info = _get_seek_session_info(session_id)
    if info is None:
        raise fastapi.HTTPException(404, "Session not found or not VOD")
    if _is_archive_url(info.url):
        raise fastapi.HTTPException(400, "Seek an archive by opening it at the desired start time")

    settings = get_settings()
    hw = settings["transcode_hw"]
    max_resolution = settings.get("max_resolution", "1080p")
    quality = settings.get("quality", "high")
    seg_duration = get_hls_segment_duration()
    segment_num = int(seek_time / seg_duration)

    output_path = pathlib.Path(info.output_dir)
    target_segment = output_path / f"{SEG_PREFIX}{segment_num:03d}.ts"

    # Smart seek: if target segment exists, no need to restart ffmpeg. A dead ffmpeg
    # (crashed, or killed by a full disk) would leave the stream ending where it stopped.
    if (
        _is_process_alive(info.process)
        and target_segment.exists()
        and target_segment.stat().st_size > min_segment_size_bytes
    ):
        log.info(
            "Smart seek: segment %d exists for time %.1fs, skipping ffmpeg restart",
            segment_num,
            seek_time,
        )
        with _transcode_lock:
            session = _transcode_sessions.get(session_id)
            if session is not None:
                session["seek_offset"] = seek_time
        _regenerate_playlist(output_path, segment_num)
        return {"session_id": session_id, "playlist": f"/transcode/{session_id}/stream.m3u8"}

    await _terminate_process(info.process)
    # The next ffmpeg truncates the detections; keep what this one found.
    await asyncio.to_thread(
        _remember_detected_crop,
        info.url,
        info.output_dir,
        info.series_id,
        info.episode_id,
    )

    # Clear playlist but keep segments (for backward seeks later)
    playlist_file = output_path / "stream.m3u8"
    playlist_file.unlink(missing_ok=True)
    # Only clear segments AFTER target (we might seek back to earlier ones)
    for seg_file in output_path.glob(f"{SEG_PREFIX}*.ts"):
        try:
            seg_num = int(seg_file.stem[len(SEG_PREFIX) :])
            if seg_num >= segment_num:
                seg_file.unlink(missing_ok=True)
        except ValueError:
            pass
    for subtitle_file in (*output_path.glob("sub*.vtt"), *output_path.glob("sub*.m3u8")):
        subtitle_file.unlink(missing_ok=True)

    # Resolve HLS master playlist to highest bandwidth variant.
    url = await asyncio.to_thread(resolve_hls_master_playlist, info.url, info.source_id)

    # Use probe_series if series_id, else probe_movies.
    if info.series_id is not None and info.series_id != 0:
        do_probe = settings["probe_series"]
    else:
        do_probe = settings["probe_movies"]
    if do_probe:
        media_info = _user_audio(
            (
                await asyncio.to_thread(
                    functools.partial(
                        probe_media,
                        url,
                        info.series_id,
                        info.episode_id,
                        source_id=info.source_id,
                    ),
                )
            )[0],
            info.username,
        )
        media_info = _with_crop(media_info, info.crop)
    else:
        media_info = None

    subtitles: list[SubtitleStream] = []
    for s in info.subtitles:
        if not isinstance(s, int):
            subtitles.append(
                SubtitleStream(
                    index=s["index"],
                    lang=s.get("lang", "und"),
                    name=s.get("name", "Unknown"),
                ),
            )

    try:
        _write_empty_caption_playlists(info.output_dir, len(subtitles))
    except OSError as error:
        raise fastapi.HTTPException(500, "Could not prepare captions after seeking") from error

    cmd = build_hls_ffmpeg_cmd(
        url,
        hw,
        info.output_dir,
        True,
        subtitles or None,
        media_info,
        max_resolution,
        quality,
        get_user_agent(),
        None,
        audio_passthrough=info.audio_passthrough,
        input_args=proxy.ffmpeg_input_args(url, info.source_id),
    )
    i_idx = cmd.index("-i")
    cmd.insert(i_idx, str(seek_time))
    cmd.insert(i_idx, "-ss")
    # Keep the main HLS timeline aligned with the requested input position.
    main_output_idx = cmd.index("-max_delay")
    cmd.insert(main_output_idx, str(-seek_time))
    cmd.insert(main_output_idx, "-output_ts_offset")
    # Before the playlist path: ffmpeg ignores options after its last output.
    cmd[-1:-1] = ["-start_number", str(segment_num)]

    log.info(
        "Seek transcode %s to %.1fs (seg %d): %s",
        session_id,
        seek_time,
        segment_num,
        " ".join(cmd),
    )

    process = await _launch_ffmpeg(cmd)

    if not _update_session_process(session_id, process, url=info.url, seek_time=seek_time):
        await _terminate_process(process)
        raise fastapi.HTTPException(404, "Session disappeared during seek")

    # Persist seek_offset.
    session_json = output_path / "session.json"
    if session_json.exists():
        try:
            # A plain tree, not _SessionInfo, so keys this version does not declare survive.
            data = json_loads(session_json.read_text())
            assert isinstance(data, dict)
            data["seek_offset"] = seek_time
            session_json.write_text(json.dumps(data))
        except Exception as e:
            log.warning("Failed to update session.json for %s: %s", session_id, e)

    _spawn_background_task(_monitor_seek_ffmpeg(process, session_id))

    if not await _wait_for_playlist(
        playlist_file,
        process,
        min_segments=2,
        timeout_sec=playlist_wait_timeout_sec,
    ):
        raise fastapi.HTTPException(500, "Seek transcode timed out waiting for playlist")

    if subtitles:
        get_caption_timestamp_origin(session_id)
    log.info("Seek ready: %s", playlist_file)

    return {
        "ok": True,
        "segment": segment_num,
        "time": seek_time,
    }


def report_playback_health(
    session_id: str,
    username: str,
    health: PlaybackHealth,
) -> HealthResponse:
    """Evaluate playback pressure for the current live stream.

    Args:
        session_id: Live session reporting health.
        username: Reporting user; must own the session.
        health: Client buffer and throughput sample.

    Returns:
        decision: Bandwidth-saver state and, for fast start, the session playlists.
    """
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None or session.get("username") != username:
            raise fastapi.HTTPException(404, "Session not found")
        if not _is_fast_live(session):
            raise fastapi.HTTPException(400, "Playback feedback is only supported for live streams")
        session["last_access"] = time.time()
        now = time.monotonic()
        policy = session["playback_policy"]
        bandwidth_saver = session.get("bandwidth_saver", False)
        # Remember actual full-quality requirements, since the configured
        # bitrate may only be an estimate.
        if not policy.bandwidth_saver and not bandwidth_saver:
            session["recovery_bitrate"] = max(
                session.get("recovery_bitrate", 0),
                health.required_bitrate,
            )
        # A full-quality encoder's output measures what recovery must sustain;
        # a saver session's 720p output would understate it.
        if not bandwidth_saver and _is_process_alive(session.get("process")):
            session["recovery_bitrate"] = max(
                session.get("recovery_bitrate", 0),
                ready_bitrate(session["dir"], playlist_file(session)),
            )
        # The single encoder has no second rendition to wait for.
        policy = policy.observe(health, now, session.get("recovery_bitrate", 0))
        session["playback_policy"] = policy
        saver = policy.bandwidth_saver
        result: HealthResponse = {"bandwidth_saver": saver}
        # A saver change needs a differently configured encoder. Without a
        # playlist, clients retune to a new session instead of staying here.
        if saver == bandwidth_saver:
            generation = session.get("playlist_generation", 0)
            result["playlist"] = _fast_playlist_url(session_id, "low.m3u8", generation)
            if session.get("closed_captions", False):
                result["caption_playlist"] = _fast_playlist_url(
                    session_id,
                    "low-captions.m3u8",
                    generation,
                )
        return result


# The read timeout turns a silent upstream into a relay end the watchdog acts on;
# panels relaying HLS deliver TS in segment-sized bursts up to ~10 s apart.
def _open_source(
    url: str,
    user_agent: str | None,
    source_id: str,
    timeout_sec: int = 30,
) -> Readable:
    """Open a session's one upstream connection: MPEG-TS directly, HLS through a remuxer."""
    if is_hls_url(url):
        # Ffmpeg reads the first-listed variant of a master playlist; pick the best one.
        # Resolved per connection: redirect targets carry tokens that expire.
        process = subprocess.Popen(
            source_command(resolve_hls_master_playlist(url, source_id), user_agent, source_id),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            env=proxy.ffmpeg_env(),
        )
        return _ProcessOutput(process=process, stdout=cast(io.BufferedReader, process.stdout))
    # The provider's redirects may not point the stream at this server or the LAN.
    return safe_urlopen(
        url,
        timeout_sec,
        user_agent,
        allow_address=proxy.address_rule(url, source_id),
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class _ProcessOutput:
    """A remuxer's stdout read as an upstream body; closing it stops the remuxer."""

    process: subprocess.Popen[bytes]
    stdout: io.BufferedReader

    def read1(self, amt: int, /) -> bytes:
        return self.stdout.read1(amt)

    def close(self) -> None:
        self.process.kill()
        self.process.wait()
        self.stdout.close()


def playlist_file(session: TranscodeSession) -> str:
    """The playlist file a session's clients read: stream.m3u8, or its live rendition's."""
    if not _is_fast_live(session):
        return "stream.m3u8"
    return f"{session.get('rendition', 'low')}.m3u8"


async def wait_for_segment(
    session_id: str,
    msn: int,
    *,
    timeout_sec: float = 3.0,
    poll_sec: float = 0.02,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Return once the session's live playlist lists segment ``msn``, or at the timeout.

    Args:
        session_id: Fast-start session.
        msn: Media sequence number a blocking playlist reload asked for.
        timeout_sec: Longest wait; the client then gets the playlist as it is.
        poll_sec: How often the playlist is checked.
        sleep: Waits between checks.
    """
    for _ in range(round(timeout_sec / poll_sec)):
        content = live_playlist(session_id)
        if content is None or last_media_sequence(content) >= msn:
            return
        await sleep(poll_sec)


def live_playlist(session_id: str) -> str | None:
    """The live playlist a session's clients read, continuing across a rendition switch.

    Args:
        session_id: Fast-start session.

    Returns:
        content: Playlist text, or None for an unknown session or an unwritten playlist.
    """
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None or not _is_fast_live(session):
            return None
        directory, name = session["dir"], playlist_file(session)
        low, point = session.get("splice_low"), session.get("splice_point")
    try:
        content = (pathlib.Path(directory) / name).read_text()
    except OSError:
        return None
    if low is None:
        return content
    if point is None:
        point = splice_point(directory, low, content)
        if point is not None:
            with _transcode_lock:
                session["splice_point"] = point
    return splice_playlists(low, content, point, list_size=get_live_hls_list_size(1.0))


def _require_current(session_id: str, session: TranscodeSession) -> None:
    """Abort a startup step once its session was stopped or replaced."""
    if _transcode_sessions.get(session_id) is not session:
        raise KeyError(session_id)


def _is_fast_live(session: TranscodeSession) -> TypeGuard[FastLiveSession]:
    return "live_recovery" in session


def _is_archive_url(url: str) -> bool:
    return "/timeshift/" in parse.urlsplit(url).path


class _DeadProcess:
    """Placeholder for dead/recovered processes."""

    returncode = -1

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass

    async def wait(self) -> int:
        return self.returncode


def _is_process_alive(proc: object) -> bool:
    """Check if process is still running; anything without a return code is not."""
    return getattr(proc, "returncode", 0) is None


async def _launch_ffmpeg(
    cmd: list[str],
    *,
    stdin: int = asyncio.subprocess.DEVNULL,
) -> asyncio.subprocess.Process:
    """Launch ffmpeg; a caller cancelled mid-launch still terminates the process."""
    launch = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *cmd,
            stdin=stdin,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=proxy.ffmpeg_env(),
        ),
    )
    try:
        return await asyncio.shield(launch)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await _terminate_process(await launch)
        raise


async def _terminate_processes(processes: Sequence[ManagedProcess | None]) -> None:
    await asyncio.gather(*(_terminate_process(process) for process in processes))


async def _terminate_process(process: ManagedProcess | None) -> None:
    """Stop a process with SIGTERM so ffmpeg can flush, then SIGKILL after 5 s."""
    if process is None or not _is_process_alive(process):
        return
    try:
        process.terminate()
    except (ProcessLookupError, OSError):
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    # asyncio.TimeoutError is distinct from the builtin before Python 3.11.
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()
        with contextlib.suppress(Exception):
            await process.wait()
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()
        with contextlib.suppress(Exception):
            await asyncio.shield(process.wait())
        raise


async def _monitor_ffmpeg_stderr(
    process: asyncio.subprocess.Process,
    session_id: str,
    stderr_lines: list[str] | None = None,
    done: asyncio.Event | None = None,
) -> None:
    if process.stderr is None:
        raise ValueError(
            "Expected process.stderr is not None.",  # pragma: no mutate[diagnostic-text]
        )
    while True:
        line = await process.stderr.readline()
        if not line:
            break
        text = line.decode().rstrip()
        if stderr_lines is not None:
            stderr_lines.append(text)
        is_fatal = "fatal" in text.lower() or "aborting" in text.lower()
        level = logging.WARNING if is_fatal else logging.DEBUG
        log.log(level, "ffmpeg:%s %s", session_id, text)
    if done is not None:
        done.set()


async def _monitor_resume_ffmpeg(
    process: asyncio.subprocess.Process,
    session_id: str,
    url: str,
    *,
    quick_failure_threshold_sec: float = 10.0,
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
        if time.time() - start_time < quick_failure_threshold_sec:
            log.info("Resume failed quickly, invalidating session %s", session_id)
            with _transcode_lock:
                _url_to_session.pop(url, None)
                session = _transcode_sessions.pop(session_id, None)
            # Clean up output directory.
            if session is not None:
                shutil.rmtree(session["dir"], ignore_errors=True)


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


def _spawn_background_task(coro: Coroutine[object, object, None]) -> asyncio.Task[None]:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _wait_for_playlist(
    playlist_path: pathlib.Path,
    process: ManagedProcess,
    min_segments: int = 1,
    timeout_sec: float = 30.0,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    *,
    poll_interval_sec: float = 0.2,
    min_segment_size_bytes: int = 1_000,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> bool:
    """Wait for playlist with min_segments, checking process health."""
    output_dir = playlist_path.parent
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if is_disconnected and await is_disconnected():
            raise fastapi.HTTPException(499, "Playback request disconnected")
        if process.returncode is not None:
            return False
        if playlist_path.exists():
            content = playlist_path.read_text()
            seg_count = content.count("#EXTINF")
            if seg_count >= min_segments:
                seg_files = list(output_dir.glob(f"{SEG_PREFIX}*.ts"))
                if len(seg_files) >= min_segments:
                    first_seg = min(seg_files, key=lambda f: f.name)
                    if first_seg.stat().st_size > min_segment_size_bytes and _is_process_alive(
                        process,
                    ):
                        return True
        await sleep(poll_interval_sec)
    return False


def _calc_hls_duration(playlist_path: pathlib.Path, segment_count: int) -> float:
    """Calculate HLS duration from playlist or estimate from segment count."""
    if playlist_path.exists():
        durations = re.findall(r"#EXTINF:([\d.]+)", playlist_path.read_text())
        if durations:
            return sum(float(d) for d in durations)
    return segment_count * get_hls_segment_duration()


def _build_subtitle_tracks(
    session_id: str,
    sub_info: Sequence[StoredSubtitle | int] | None,
) -> list[SubtitleTrack]:
    tracks: list[SubtitleTrack] = []
    for index, subtitle in enumerate(() if sub_info is None else sub_info):
        if isinstance(subtitle, int):
            return []
        lang = subtitle.get("lang")
        name = subtitle.get("name")
        assert isinstance(lang, str) and isinstance(name, str)
        tracks.append(
            {
                "url": f"/subs/{session_id}/sub{index}.vtt",
                "lang": lang,
                "label": name,
                "default": index == 0,
            },
        )
    return tracks


def _caption_master_playlist(
    sub_info: Sequence[SubtitleInfo],
    max_resolution: str,
) -> str:
    _, maximum_bitrate = live_video_bitrates(max_resolution)
    lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
    used_names: dict[str, int] = {}
    for i, subtitle in enumerate(sub_info):
        base_name = _caption_attribute(subtitle["name"], f"Captions {i + 1}")
        occurrence = used_names.get(base_name, 0)
        used_names[base_name] = occurrence + 1
        name = base_name if occurrence == 0 else f"{base_name} {occurrence + 1}"
        language = _caption_attribute(subtitle["lang"], "und")
        lines.append(
            '#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="captions",'
            f'NAME="{name}",LANGUAGE="{language}",DEFAULT=NO,AUTOSELECT=YES,'
            f'FORCED=NO,URI="sub{i}.m3u8"',
        )
    bandwidth = int(maximum_bitrate * 1.1) + 640_000
    lines += [
        f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},SUBTITLES="captions"',
        "stream.m3u8",
    ]
    return "\n".join(lines) + "\n"


def _caption_attribute(value: str, fallback: str) -> str:
    cleaned = " ".join((value or fallback).split())
    return cleaned.replace('"', "'")


def _write_caption_master(
    output_dir: str,
    sub_info: Sequence[SubtitleInfo],
    max_resolution: str,
) -> pathlib.Path | None:
    if not sub_info:
        return None
    path = pathlib.Path(output_dir) / "captions.m3u8"
    temporary = path.with_name(".captions.m3u8.tmp")
    temporary.write_text(_caption_master_playlist(sub_info, max_resolution))
    temporary.replace(path)
    _write_empty_caption_playlists(output_dir, len(sub_info))
    return path


def _write_empty_caption_playlists(output_dir: str, count: int) -> None:
    target_duration = max(1, int(get_hls_segment_duration() + 0.999))
    content = "\n".join(
        [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "",
        ],
    )
    for index in range(count):
        (pathlib.Path(output_dir) / f"sub{index}.m3u8").write_text(content)


def _caption_playlist_url(
    session_id: str,
    output_dir: str,
    sub_info: Sequence[StoredSubtitle | int],
) -> str | None:
    if sub_info and (pathlib.Path(output_dir) / "captions.m3u8").exists():
        return f"/transcode/{session_id}/captions.m3u8"
    return None


def _caption_timestamp_origin(output_dir: str) -> float | None:
    playlist_path = pathlib.Path(output_dir) / "stream.m3u8"
    try:
        lines = playlist_path.read_text().splitlines()
    except OSError:
        return None

    elapsed = 0.0
    duration = 0.0
    for line in lines:
        value = line.strip()
        if value.startswith("#EXTINF:"):
            with contextlib.suppress(ValueError):
                duration = float(value.removeprefix("#EXTINF:").split(",", 1)[0])
            continue
        if not value or value.startswith("#") or pathlib.Path(value).name != value:
            continue
        pts = segment_pts(pathlib.Path(output_dir) / value)
        if pts is not None:
            return pts - elapsed
        elapsed += duration
        duration = 0.0
    return None


def _playlist_complete(path: pathlib.Path) -> bool:
    """Whether ffmpeg finished ``path``, so its segments run to the end of the source."""
    try:
        return "#EXT-X-ENDLIST" in path.read_text()
    except OSError:
        return False


def _regenerate_playlist(
    output_dir: pathlib.Path,
    start_segment: int,
    *,
    min_segment_size_bytes: int = 1_000,
) -> None:
    """Regenerate HLS playlist starting from a specific segment (for smart seek)."""
    playlist_path = output_dir / "stream.m3u8"
    seg_duration = get_hls_segment_duration()

    # Find all existing segments from start_segment onwards.
    segments = []
    for seg_file in sorted(output_dir.glob(f"{SEG_PREFIX}*.ts")):
        try:
            seg_num = int(seg_file.stem[len(SEG_PREFIX) :])
            if seg_num >= start_segment and seg_file.stat().st_size > min_segment_size_bytes:
                segments.append((seg_num, seg_file.name))
        except ValueError:
            pass

    if not segments:
        return

    # Build playlist.
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{int(seg_duration) + 1}",
        f"#EXT-X-MEDIA-SEQUENCE:{start_segment}",
        "#EXT-X-PLAYLIST-TYPE:EVENT",
    ]

    for _, seg_name in segments:
        lines.append(f"#EXTINF:{seg_duration:.6f},")
        lines.append(seg_name)

    playlist_path.write_text("\n".join(lines) + "\n")
    log.debug("Regenerated playlist with %d segments starting at %d", len(segments), start_segment)


def _get_session_snapshot(session_id: str) -> _SessionSnapshot | None:
    """Get atomic snapshot of session state under lock."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None:
            return None
        session["last_access"] = time.time()
        return _SessionSnapshot(
            output_dir=session["dir"],
            process=session["process"],
            seek_offset=session.get("seek_offset", 0),
            subtitles=session.get("subtitles", []),
            duration=session.get("duration", 0),
            audio_passthrough=session.get("audio_passthrough", False),
            username=session.get("username", ""),
            source_id=session.get("source_id", ""),
            crop=session.get("crop"),
        )


def _update_session_process(
    session_id: str,
    process: ManagedProcess,
    *,
    seek_time: float | None = None,
    url: str = "",
) -> bool:
    """Atomically update session process. Returns False if session gone."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None:
            return False
        session["process"] = process
        if seek_time is not None:
            session["seek_offset"] = seek_time
            session.pop("caption_timestamp_origin", None)
        if url:
            _url_to_session[url] = session_id
        return True


def _build_session_response(
    session_id: str,
    snap: _SessionSnapshot,
    playlist_path: pathlib.Path,
) -> TranscodeResponse:
    """Build response dict for existing session, recalculating duration."""
    segments = list(playlist_path.parent.glob(f"{SEG_PREFIX}*.ts"))
    response: TranscodeResponse = {
        "session_id": session_id,
        "playlist": f"/transcode/{session_id}/stream.m3u8",
        "subtitles": _build_subtitle_tracks(session_id, snap.subtitles),
        "duration": snap.duration,
        "seek_offset": snap.seek_offset,
        "transcoded_duration": _calc_hls_duration(playlist_path, len(segments)),
    }
    caption_playlist = _caption_playlist_url(session_id, snap.output_dir, snap.subtitles)
    if caption_playlist is not None:
        response["caption_playlist"] = caption_playlist
    return response


def _fast_playlist_url(session_id: str, name: str, generation: int) -> str:
    url = f"/transcode/{session_id}/{name}"
    return f"{url}?generation={generation}" if generation > 0 else url


def _adaptive_session_response(
    session_id: str,
    *,
    default_playback_buffer_seconds: float = 12.0,
) -> TranscodeResponse:
    """The response handing a live session to one more player page, its new viewer."""
    viewer_id = ""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None:
            generation = 0
            closed_captions = False
            playback_buffer_seconds = default_playback_buffer_seconds
        else:
            generation = session.get("playlist_generation", 0)
            closed_captions = session.get("closed_captions", False)
            playback_buffer_seconds = session.get(
                "playback_buffer_seconds",
                default_playback_buffer_seconds,
            )
            if _is_fast_live(session):
                viewer_id = uuid.uuid4().hex
                session["viewers"][viewer_id] = time.time()
                session["viewer_joined"].set()
    response: TranscodeResponse = {
        "session_id": session_id,
        "playlist": _fast_playlist_url(session_id, "low.m3u8", generation),
        "master_playlist": f"/transcode/{session_id}/master.m3u8",
        "playback_buffer_seconds": playback_buffer_seconds,
        "subtitles": [],
        "duration": 0,
        "seek_offset": 0,
    }
    if closed_captions:
        response["caption_playlist"] = _fast_playlist_url(
            session_id,
            "low-captions.m3u8",
            generation,
        )
    if viewer_id:
        response["viewer_id"] = viewer_id
    return response


@dataclass(frozen=True, kw_only=True, slots=True)
class _ExistingSession:
    """A URL's current session, read atomically."""

    session_id: str
    session: TranscodeSession  # Shallow copy.
    is_valid: bool


def _get_existing_session(url: str) -> _ExistingSession | None:
    """The session ``url`` maps to, or None when there is none."""
    with _transcode_lock:
        session_id = _url_to_session.get(url)
        if session_id is None:
            return None
        session = _transcode_sessions.get(session_id)
        if session is None:
            return None
        return _ExistingSession(
            session_id=session_id,
            session=session.copy(),
            is_valid=is_session_valid(session),
        )


# Returns None to trigger fresh start if session is invalid.
async def _handle_existing_vod_session(
    existing_id: str,
    url: str,
    hw: HwAccel,
    do_probe: bool,
    max_resolution: str = "1080p",
    quality: str = "high",
    *,
    reuse_wait_timeout_sec: float = 15.0,
    resume_wait_timeout_sec: float = 10.0,
    resume_segment_wait_timeout_sec: float = 5.0,
    poll_interval_sec: float = 0.2,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> TranscodeResponse | None:
    """Handle existing VOD session: reuse active, return cached, or append."""
    snap = _get_session_snapshot(existing_id)
    if snap is None:
        return None

    playlist_path = pathlib.Path(snap.output_dir) / "stream.m3u8"
    segments = sorted(pathlib.Path(snap.output_dir).glob(f"{SEG_PREFIX}*.ts"))

    # Case 1: Active session - reuse it.
    if snap.process is not None and snap.process.returncode is None:
        log.info("Reusing active session %s", existing_id)
        await _wait_for_playlist(
            playlist_path,
            snap.process,
            min_segments=1,
            timeout_sec=reuse_wait_timeout_sec,
            sleep=sleep,
        )
        return _build_session_response(existing_id, snap, playlist_path)

    # Case 2: Dead session with no segments, or one that died partway after a seek
    # (only a finished playlist covers the rest of the source) - invalid.
    if (
        not segments
        or _is_archive_url(url)
        or (snap.seek_offset > 0 and not _playlist_complete(playlist_path))
    ):
        await stop_session(existing_id, force=True)
        with _transcode_lock:
            _url_to_session.pop(url, None)
        return None

    # Case 3: Finished session with seek_offset - return cached content.
    if snap.seek_offset > 0:
        log.info(
            "Returning cached session %s (seek_offset=%.1f)",
            existing_id,
            snap.seek_offset,
        )
        return _build_session_response(existing_id, snap, playlist_path)

    # Case 4: Dead session, no seek_offset - append new content.
    hls_duration = _calc_hls_duration(playlist_path, len(segments))
    log.info("Resuming session %s from %.1fs", existing_id, hls_duration)

    # Resolve HLS master playlist to highest bandwidth variant.
    url = await asyncio.to_thread(resolve_hls_master_playlist, url, snap.source_id)

    media_info = (
        _with_crop(
            _user_audio(
                (
                    await asyncio.to_thread(
                        functools.partial(probe_media, url, source_id=snap.source_id),
                    )
                )[0],
                snap.username,
            ),
            snap.crop,
        )
        if do_probe
        else None
    )
    cmd = build_hls_ffmpeg_cmd(
        url,
        hw,
        snap.output_dir,
        True,
        None,
        media_info,
        max_resolution,
        quality,
        get_user_agent(),
        None,
        audio_passthrough=snap.audio_passthrough,
        input_args=proxy.ffmpeg_input_args(url, snap.source_id),
    )

    i_idx = cmd.index("-i")
    cmd.insert(i_idx, str(hls_duration))
    cmd.insert(i_idx, "-ss")
    cmd[cmd.index("-hls_flags") + 1] += "+append_list"
    # Before the playlist path: ffmpeg ignores options after its last output.
    cmd[-1:-1] = ["-start_number", str(len(segments))]

    process = await _launch_ffmpeg(cmd)
    if not _update_session_process(existing_id, process):
        await _terminate_process(process)
        return None

    _spawn_background_task(_monitor_resume_ffmpeg(process, existing_id, url))
    log.info("Started resume ffmpeg pid=%s for %s", process.pid, existing_id)

    deadline = time.monotonic() + resume_segment_wait_timeout_sec
    next_seg = f"{SEG_PREFIX}{len(segments):03d}.ts"
    while time.monotonic() < deadline:
        if process.returncode is not None:
            log.warning("Resume ffmpeg died immediately for %s", existing_id)
            return None
        if (pathlib.Path(snap.output_dir) / next_seg).exists():
            break
        await sleep(poll_interval_sec)

    await _wait_for_playlist(
        playlist_path,
        process,
        min_segments=1,
        timeout_sec=resume_wait_timeout_sec,
        sleep=sleep,
    )
    return _build_session_response(existing_id, snap, playlist_path)


async def _try_reuse_session(
    existing_id: str,
    url: str,
    content_type: VodType,
) -> TranscodeResponse | None:
    """Try to reuse an existing valid VOD session. Returns response or None if can't reuse."""
    settings = get_settings()
    return await _handle_existing_vod_session(
        existing_id,
        url,
        settings["transcode_hw"],
        _probe_enabled(settings, content_type),
        settings.get("max_resolution", "1080p"),
        settings.get("quality", "high"),
    )


def _probe_enabled(settings: ServerSettings, content_type: VodType) -> bool:
    """Whether settings ask for probing ``content_type``."""
    return settings["probe_movies"] if content_type == "movie" else settings["probe_series"]


async def _do_start_transcode(
    url: str,
    content_type: VodType,
    series_id: int | None,
    episode_id: int | None,
    old_seek_offset: float,
    series_name: str = "",
    deinterlace_fallback: bool = True,
    username: str = "",
    source_id: str = "",
    audio_passthrough: bool = False,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    *,
    playlist_wait_timeout_sec: float = 30.0,
    seek_wait_timeout_sec: float = 40.0,
    use_probe_cache: bool = True,
) -> TranscodeResponse:
    """Start a VOD transcode (movies, series, catchup). Raises fastapi.HTTPException on failure."""
    is_archive = _is_archive_url(url)
    request_url = url
    # Resolve HLS master playlist to highest bandwidth variant.
    url = await asyncio.to_thread(resolve_hls_master_playlist, url, source_id)

    settings = get_settings()
    hw = settings["transcode_hw"]
    max_resolution = settings.get("max_resolution", "1080p")
    quality = settings.get("quality", "high")
    do_probe = _probe_enabled(settings, content_type)

    media_info: MediaInfo | None = None
    subtitles: list[SubtitleStream] = []
    if do_probe:
        media_info, subtitles = await asyncio.to_thread(
            functools.partial(
                probe_media,
                url,
                series_id,
                episode_id,
                series_name,
                source_id=source_id,
                use_cache=use_probe_cache,
            ),
        )
        media_info = _user_audio(media_info, username)
        if media_info is not None:
            codecs = media_info.subtitle_codecs
            # Probing stores None, never [], for a source without subtitles.
            subs_str = ",".join(codecs) if codecs is not None else "none"
            if subtitles:
                subs_str += f" [extract:{','.join(s.lang for s in subtitles)}]"
            bitrate_str = (
                f"{media_info.video_bitrate / 1_000_000:.1f}Mbps"
                if media_info.video_bitrate
                else "?"
            )
            log.info(
                "Probe: video=%s/%s/%dp/%s%s audio=%s/%dch/%dHz duration=%.0fs subs=%s",
                media_info.video_codec,
                media_info.pix_fmt,
                media_info.height,
                bitrate_str,
                "/interlaced" if media_info.interlaced else "",
                media_info.audio_codec,
                media_info.audio_channels,
                media_info.audio_sample_rate,
                media_info.duration,
                subs_str,
            )

    if is_archive and is_disconnected and await is_disconnected():
        raise fastapi.HTTPException(499, "Playback request disconnected")

    session_id = str(uuid.uuid4())
    output_dir = tempfile.mkdtemp(
        prefix=f"netv_transcode_{session_id}_",
        dir=get_transcode_dir(),
    )
    playlist_path = pathlib.Path(output_dir) / "stream.m3u8"
    cmd = build_hls_ffmpeg_cmd(
        url,
        hw,
        output_dir,
        True,
        subtitles,
        media_info,
        max_resolution,
        quality,
        get_user_agent(),
        deinterlace_fallback,
        allow_upscale=not is_archive,
        audio_passthrough=audio_passthrough,
        input_args=proxy.ffmpeg_input_args(url, source_id),
    )
    passthrough_used = uses_audio_passthrough(media_info, audio_passthrough)
    crop = encoder_crop(media_info, max_resolution, allow_upscale=not is_archive)
    if old_seek_offset > 0:
        i_idx = cmd.index("-i")
        cmd.insert(i_idx, str(old_seek_offset))
        cmd.insert(i_idx, "-ss")
        log.info("Applying seek_offset=%.1f from previous session", old_seek_offset)

    sub_info: list[SubtitleInfo] = [
        {"lang": subtitle.lang, "name": subtitle.name} for subtitle in subtitles
    ]
    stored_subtitles: list[StoredSubtitle | int] = [
        {"index": subtitle.index, "lang": subtitle.lang, "name": subtitle.name}
        for subtitle in subtitles
    ]
    total_duration = 0.0 if media_info is None else media_info.duration
    log.info("Starting transcode session %s: %s", session_id, " ".join(cmd))
    try:
        caption_master = _write_caption_master(output_dir, sub_info, max_resolution)
        process = await _launch_ffmpeg(cmd)
    except OSError as error:
        shutil.rmtree(output_dir)
        raise fastapi.HTTPException(500, "Could not start the transcoder") from error
    except asyncio.CancelledError:
        shutil.rmtree(output_dir)
        raise

    stderr_lines: list[str] = []
    stderr_done = asyncio.Event()
    _spawn_background_task(_monitor_ffmpeg_stderr(process, session_id, stderr_lines, stderr_done))

    with _transcode_lock:
        _transcode_sessions[session_id] = {
            "dir": output_dir,
            "process": process,
            "started": time.time(),
            "url": url,
            "request_url": request_url,
            "is_vod": True,
            "is_archive": is_archive,
            "last_access": time.time(),
            "subtitles": stored_subtitles,
            "duration": total_duration,
            "seek_offset": old_seek_offset,
            "series_id": series_id,
            "episode_id": episode_id,
            "username": username,
            "source_id": source_id,
            "audio_passthrough": passthrough_used,
            "audio_lang": _audio_lang(username),
        }
        if crop is not None:
            _transcode_sessions[session_id]["crop"] = crop
        _url_to_session[url] = session_id

    if not is_archive:
        session_info: _SessionInfo = {
            "session_id": session_id,
            "url": url,
            "is_vod": True,
            "started": time.time(),
            "subtitles": stored_subtitles,
            "duration": total_duration,
            "seek_offset": old_seek_offset,
            "series_id": series_id,
            "episode_id": episode_id,
            "username": username,
            "source_id": source_id,
            "audio_passthrough": passthrough_used,
            "audio_lang": _audio_lang(username),
        }
        if media_info is not None:
            session_info["probe"] = {
                "video_codec": media_info.video_codec,
                "audio_codec": media_info.audio_codec,
                "pix_fmt": media_info.pix_fmt,
                "audio_channels": media_info.audio_channels,
                "audio_sample_rate": media_info.audio_sample_rate,
                "subtitle_codecs": media_info.subtitle_codecs,
                "height": media_info.height,
                "video_bitrate": media_info.video_bitrate,
                "interlaced": media_info.interlaced,
                "audio_tracks": audio_tracks_to_cache(media_info.audio_tracks),
            }
        if crop is not None:
            session_info["crop"] = crop.to_cache()
        (pathlib.Path(output_dir) / "session.json").write_text(json.dumps(session_info))

    timeout = seek_wait_timeout_sec if old_seek_offset > 0 else playlist_wait_timeout_sec
    try:
        ready = await _wait_for_playlist(
            playlist_path,
            process,
            min_segments=2,
            timeout_sec=timeout,
            is_disconnected=is_disconnected if is_archive else None,
        )
    except (asyncio.CancelledError, fastapi.HTTPException):
        await stop_session(session_id, force=True)
        raise
    if not ready:
        # The exit code and the last stderr lines name the failure; both arrive as ffmpeg exits.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=1.0)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stderr_done.wait(), timeout=1.0)
        error_msg = "\n".join(stderr_lines[-10:]) if stderr_lines else "unknown"
        log.error(
            "ffmpeg:%s failed (exit %d): %s",
            session_id,
            -1 if process.returncode in (None, 0) else process.returncode,
            error_msg,
        )
        await stop_session(session_id, force=True)
        raise fastapi.HTTPException(500, "Transcode failed - check server logs for details")

    if sub_info:
        get_caption_timestamp_origin(session_id)

    response: TranscodeResponse = {
        "session_id": session_id,
        "playlist": f"/transcode/{session_id}/stream.m3u8",
        "subtitles": _build_subtitle_tracks(session_id, stored_subtitles),
        "duration": total_duration,
        "seek_offset": old_seek_offset,
    }
    if caption_master is not None:
        response["caption_playlist"] = f"/transcode/{session_id}/captions.m3u8"
    return response


def _get_seek_session_info(session_id: str) -> _SeekSessionInfo | None:
    """Get session info for seek atomically. Returns None if not VOD."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is None or not session.get("is_vod", False):
            return None
        return _SeekSessionInfo(
            url=session["url"],
            output_dir=session["dir"],
            process=session["process"],
            subtitles=session.get("subtitles", []),
            series_id=session.get("series_id"),
            episode_id=session.get("episode_id"),
            audio_passthrough=session.get("audio_passthrough", False),
            username=session.get("username", ""),
            source_id=session.get("source_id", ""),
            crop=session.get("crop"),
        )


def _playlist_marker(path: pathlib.Path) -> tuple[int, int, int] | None:
    try:
        info = path.stat()
        return info.st_ino, info.st_mtime_ns, info.st_size
    except OSError:
        return None


def _playlist_segments(path: pathlib.Path) -> set[str]:
    try:
        return {
            filename
            for line in path.read_text().splitlines()
            if (filename := line.strip())
            and not filename.startswith("#")
            and pathlib.Path(filename).name == filename
        }
    except OSError:
        return set()


def _prepare_encoder_restart(command: list[str], previous_segments: set[str]) -> list[str]:
    if previous_segments or "-hls_start_number_source" in command:
        return command
    command = list(command)
    command[-1:-1] = ["-hls_start_number_source", "epoch_us"]
    return command


async def _wait_for_fresh_fast_playlist(
    directory: str,
    name: str,
    process: ManagedProcess,
    previous_marker: tuple[int, int, int] | None,
    *,
    previous_segments: set[str] | None = None,
    minimum_segments: int = 2,
    minimum_seconds: float = 0,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.05,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> bool:
    path = pathlib.Path(directory) / name
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.returncode is not None:
            return False
        if is_disconnected is not None and await is_disconnected():
            raise fastapi.HTTPException(499, "Playback request disconnected")
        marker = _playlist_marker(path)
        if (
            marker is not None
            and marker != previous_marker
            and (
                previous_segments is None
                or len(_playlist_segments(path) - previous_segments) >= minimum_segments
            )
            and ready_bitrate(directory, name, minimum_segments=minimum_segments)
            and sum(map(float, re.findall(r"#EXTINF:([\d.]+)", path.read_text())))
            >= minimum_seconds
        ):
            return True
        await sleep(poll_interval_sec)
    return False


async def _monitor_fast_process(
    process: asyncio.subprocess.Process,
    role: str,
    session_id: str,
) -> None:
    if process.stderr is None:
        raise ValueError("Expected process.stderr is not None.")
    recent: list[str] = []
    while line := await process.stderr.readline():
        recent = (recent + [line.decode(errors="replace").rstrip()])[-10:]
    await process.wait()
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        current = session is not None and process is session.get("process")
    if current:
        log.warning(
            "Fast-start %s exited for %s (code %s): %s",
            role,
            session_id,
            process.returncode,
            " | ".join(recent) or "no stderr",
        )


def _publish_fast_master(session: FastLiveSession) -> None:
    directory = pathlib.Path(session["dir"])
    resolution = session["plan"].resolution
    # Startup publishes only after the worker launch recorded it.
    audio_bitrate = session.get("master_audio_bitrate")
    assert isinstance(audio_bitrate, int)
    _publish(directory / "master.m3u8", master_playlist(resolution, audio_bitrate=audio_bitrate))
    if session.get("closed_captions") is True:
        _publish(
            directory / "low-captions.m3u8",
            closed_caption_master_playlist(resolution, audio_bitrate=audio_bitrate),
        )


def _publish(path: pathlib.Path, content: str) -> None:
    """Replace ``path`` atomically, so a client never reads a partial playlist."""
    temporary = path.with_name(f".{path.name}.tmp")  # pragma: no mutate
    temporary.write_text(content)
    temporary.replace(path)


async def _restart_fast_pipeline(
    session_id: str,
    session: FastLiveSession,
    stalled_stage: str,
) -> None:
    """Replace the session's worker; a stalled input also reconnects the relay."""
    recovery = session["live_recovery"]
    reconnect = stalled_stage == "input"
    replacement: ManagedProcess | None = None
    success = False
    try:
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is not session:
                return
            old_process = session.get("process")
            old_subscription = session.get("subscription")
            session["process"] = _DeadProcess()
            session["watchdog_recovering"] = stalled_stage
        if old_subscription is not None:
            old_subscription.cancel()
        await _terminate_process(old_process)
        if reconnect:
            await session["relay"].close()
            stream = session["plan"].relay()
            with _transcode_lock:
                _require_current(session_id, session)
                session["relay"] = stream
            await stream.start()

        low_playlist = pathlib.Path(session["dir"]) / playlist_file(session)
        low_marker = _playlist_marker(low_playlist)
        low_segments = _playlist_segments(low_playlist)
        replacement = await _launch_worker(
            session_id,
            session,
            fresh_relay=reconnect,
            previous_segments=low_segments,
        )
        if not await _wait_for_fresh_fast_playlist(
            session["dir"],
            low_playlist.name,
            replacement,
            low_marker,
            previous_segments=low_segments,
        ):
            raise TimeoutError("low output did not become ready")

        with _transcode_lock:
            _require_current(session_id, session)
            session["playlist_generation"] = session.get("playlist_generation", 0) + 1
            recovery = recovery.reset_pipeline()
            session["live_recovery"] = recovery
        success = True
        log.warning(
            "Live watchdog %s recovered pipeline generation %d after %s stalled",
            session_id,
            session.get("playlist_generation", 0),
            stalled_stage,
        )
    except KeyError:
        log.info("Live watchdog %s stopped recovering: the session ended", session_id)
    except OSError:
        log.exception(
            "Live watchdog %s failed to recover the pipeline after %s stalled",
            session_id,
            stalled_stage,
        )
    finally:
        if not success:
            with _transcode_lock:
                if (
                    _transcode_sessions.get(session_id) is session
                    and session.get("process") is replacement
                ):
                    session["process"] = _DeadProcess()
            await _terminate_process(replacement)
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is session:
                session["live_recovery"] = recovery.finish_restart("pipeline")
                session.pop("watchdog_recovering", None)


# A fresh relay may start on a new source clock, so its segments need a new origin;
# segments already listed keep the dates the old origin gave them.
async def _restart_session_clock(
    session_id: str,
    session: TranscodeSession,
    origin_pts: float | None,
    origin_time: float,
) -> None:
    """Date the old worker's segments, then move the clock to the new relay."""
    with contextlib.suppress(OSError):
        content = (pathlib.Path(session["dir"]) / playlist_file(session)).read_text()
        await asyncio.to_thread(dated_fast_playlist, session_id, content)
    with _transcode_lock:
        _require_current(session_id, session)
        session.update(origin_pts=origin_pts, origin_time=origin_time)


def _log_watchdog_delay(
    session_id: str,
    session: TranscodeSession,
    action: Literal["pipeline"],
    now: float,
) -> None:
    if now - session.get("pipeline_restart_log_at", float("-inf")) >= 30:
        log.warning(
            "Live watchdog %s is delaying %s recovery because its restart budget is cooling down",
            session_id,
            action,
        )
        session["pipeline_restart_log_at"] = now


async def _check_fast_live_session(session_id: str, session: TranscodeSession) -> None:
    if not _is_fast_live(session):
        raise KeyError("live_recovery")
    recovery = session["live_recovery"]
    now = time.monotonic()

    stream = session["relay"]
    input_alive = not stream.finished
    stale_after = session.get("watchdog_stale_after", 10)
    last_byte_at = stream.stats.last_byte_at
    input_ready = last_byte_at is not None and time.time() - last_byte_at <= stale_after
    low_alive = _is_process_alive(session.get("process"))
    playlist = playlist_file(session)
    low_ready = bool(low_alive and ready_bitrate(session["dir"], playlist, max_age=stale_after))

    recovery, input_stalled = recovery.stalled("input", input_alive, input_ready, now)
    recovery, low_stalled = recovery.stalled("low", low_alive, low_ready, now)
    session["live_recovery"] = recovery
    if not (input_stalled or low_stalled):
        return
    stage = "input" if input_stalled else "low output"
    recovery, permitted = recovery.permit_restart("pipeline", now)
    session["live_recovery"] = recovery
    if permitted:
        await _restart_fast_pipeline(session_id, session, stage)
    elif "pipeline" in recovery.pending:
        return
    elif recovery.exhausted("pipeline", now):
        # Clients keep a session alive by polling it; without a pipeline it
        # would be reused until the window allows another restart.
        log.error(
            "Live watchdog %s stopped the session: %s stalled and its restart budget is spent",
            session_id,
            stage,
        )
        await stop_session(session_id, force=True)
    else:
        _log_watchdog_delay(session_id, session, "pipeline", now)


async def _watch_fast_live(
    session_id: str,
    *,
    watchdog_interval_sec: float = 2.0,
    heartbeat_timeout_sec: float = 30.0,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> None:
    while True:
        await sleep(watchdog_interval_sec)
        with _transcode_lock:
            session = _transcode_sessions.get(session_id)
            if session is None:
                return
            recently_accessed = time.time() - session.get("last_access", 0) <= heartbeat_timeout_sec
        if not recently_accessed:
            continue
        try:
            await _check_fast_live_session(session_id, session)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Live watchdog %s failed while checking pipeline health", session_id)


def _retire_session(session: TranscodeSession) -> list[ManagedProcess | None]:
    """Cancel the session's watchdog and list the processes to terminate."""
    task = session.get("watchdog_task")
    # The watchdog stops a session it cannot restart; it must finish that stop.
    if task is not None and task is not asyncio.current_task():
        task.cancel()
    return [session.get("process")]


async def _start_or_reuse(
    url: str,
    content_type: ContentType,
    series_id: int | None,
    episode_id: int | None,
    series_name: str,
    deinterlace_fallback: bool,
    username: str,
    source_id: str,
    user_max_streams: int,
    source_max_streams: int,
    bandwidth_saver: bool,
    is_disconnected: Callable[[], Awaitable[bool]] | None,
    audio_passthrough: bool,
    min_segments: int,
) -> TranscodeResponse:
    """Reuse the URL's session or start one; callers hold the URL's startup guard."""
    # A quality change replaces the current stream; release its slot first.
    existing = _get_existing_session(url)
    # An archive request names its own start, so it never resumes another position.
    old_seek_offset = (
        existing.session.get("seek_offset", 0)
        if existing is not None and not _is_archive_url(url)
        else 0.0
    )
    if existing is not None:
        owner = existing.session.get("username")
        if _is_archive_url(url) and owner != username:
            raise fastapi.HTTPException(409, "This archive is playing on another device.")
        if bandwidth_saver:
            if owner != username:
                raise fastapi.HTTPException(404, "Session not found")
            await stop_session(existing.session_id, force=True)
            existing = None
        elif not audio_passthrough and existing.session.get("audio_passthrough") is True:
            # This client cannot decode the session's Dolby audio.
            if owner != username:
                raise fastapi.HTTPException(409, "This stream is playing on another device.")
            await stop_session(existing.session_id, force=True)
            existing = None

    if existing is not None and existing.is_valid:
        if existing.session.get("fast_start") is True:
            if existing.session.get("username") != username:
                raise fastapi.HTTPException(404, "Session not found")
            touch_session(existing.session_id)
            return _adaptive_session_response(existing.session_id)
        if (
            content_type != "live"
            and not _is_archive_url(url)
            and existing.session.get("username") == username
            and existing.session.get("audio_lang") != _audio_lang(username)
        ):
            # Its segments carry another audio track; restart at the same position.
            log.info("Restarting session %s for audio language", existing.session_id)
            await stop_session(existing.session_id, force=True)
            existing = None
        elif content_type != "live":
            log.info("Found valid existing VOD session %s", existing.session_id)
            result = await _try_reuse_session(existing.session_id, url, content_type)
            if result:
                _admit(existing.session_id, username)
                return result

    if existing is not None:
        log.info("Cleaning up invalid session %s", existing.session_id)
        await stop_session(existing.session_id, force=True)

    # Limits count upstream connections, so only a new session is checked; a
    # reused one already holds its slot.
    if username:
        error = await enforce_stream_limits(
            username,
            source_id,
            user_max_streams,
            source_max_streams,
        )
        if error is not None:
            raise fastapi.HTTPException(status_code=429, detail=error)

    if content_type == "live":
        return await _start_fast_live(
            url,
            username,
            source_id,
            deinterlace_fallback,
            bandwidth_saver=bandwidth_saver,
            is_disconnected=is_disconnected,
            audio_passthrough=audio_passthrough,
            min_segments=min_segments,
        )

    # A cached probe makes the start fast; if the source changed since, or the episode
    # differs from the series probe it shared, ffmpeg rejects the stale facts, so
    # re-probe and retry once.
    probe_was_cached = has_cached_probe(url, series_id)
    try:
        return await _do_start_transcode(
            url,
            content_type,
            series_id,
            episode_id,
            old_seek_offset,
            series_name,
            deinterlace_fallback,
            username,
            source_id,
            audio_passthrough=audio_passthrough,
            is_disconnected=is_disconnected,
        )
    except fastapi.HTTPException:
        if not probe_was_cached:
            raise
        log.info("Transcode failed on a cached probe, re-probing and retrying")
        return await _do_start_transcode(
            url,
            content_type,
            series_id,
            episode_id,
            old_seek_offset,
            series_name,
            deinterlace_fallback,
            username,
            source_id,
            audio_passthrough=audio_passthrough,
            is_disconnected=is_disconnected,
            use_probe_cache=False,  # pragma: no mutate[truthiness]
        )


@dataclass(kw_only=True, slots=True, frozen=True)
class _StartupGuard:
    """Serializes one URL's lookup-or-start; removed when no request holds it."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@contextlib.asynccontextmanager
async def _url_startup(url: str) -> AsyncGenerator[None]:
    guard = _startup_guards.setdefault(url, _StartupGuard())
    _startup_guards[url] = replace(guard, users=guard.users + 1)
    try:
        async with guard.lock:
            yield
    finally:
        guard = _startup_guards[url]
        if guard.users == 1:
            del _startup_guards[url]
        else:
            _startup_guards[url] = replace(guard, users=guard.users - 1)


class _SessionInfo(TypedDict, total=False):
    """The session.json a VOD session persists for recovery after restart."""

    session_id: Required[str]
    url: str
    is_vod: bool
    started: float
    subtitles: list[StoredSubtitle | int]
    subtitle_indices: list[int]
    duration: float
    seek_offset: float
    series_id: int | None
    episode_id: int | None
    username: str
    source_id: str
    audio_passthrough: bool
    audio_lang: str
    probe: _SessionProbe
    crop: CachedCrop


async def _finish_recording(
    recording: LiveRecording,
    subscription: Subscription,
    directory: pathlib.Path,
) -> None:
    """Keep the recorder's stderr in ffmpeg.log; once it exits, stop feeding it."""
    process = recording.process
    if process.stderr is None:
        raise ValueError("Expected process.stderr is not None.")
    with (directory / "ffmpeg.log").open("ab") as log_file:
        while line := await process.stderr.readline():
            log_file.write(line)
    await process.wait()
    subscription.cancel()
    _recordings.discard(recording)


async def _start_fast_live(
    url: str,
    username: str,
    source_id: str,
    deinterlace: bool,
    *,
    bandwidth_saver: bool = False,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    audio_passthrough: bool = False,
    min_segments: int = 2,
    watchdog_stale_floor_sec: float = 45.0,
) -> TranscodeResponse:
    """One upstream reader (the relay), one worker, one session/stream slot."""
    settings = get_settings()
    session_id = str(uuid.uuid4())
    directory = tempfile.mkdtemp(prefix=f"netv_transcode_{session_id}_", dir=get_transcode_dir())
    started = time.time()
    try:
        plan = _FastLivePlan(
            source_url=await asyncio.to_thread(_relay_url, url, source_id),
            source_id=source_id,
            user_agent=get_user_agent(),
            hw=settings["transcode_hw"],
            resolution=_live_resolution(settings, bandwidth_saver=bandwidth_saver),
            quality="low" if bandwidth_saver else settings.get("quality", "high"),
            deinterlace=deinterlace,
            bandwidth_saver=bandwidth_saver,
            audio_passthrough=audio_passthrough,
            dvr=settings["live_dvr_mins"] > 0,
        )
        session: FastLiveSession = {
            "dir": directory,
            "process": None,
            "started": started,
            "last_access": started,
            "url": url,
            "is_vod": False,
            "username": username,
            "source_id": source_id,
            "bandwidth_saver": bandwidth_saver,
            "playback_policy": PlaybackPolicy(bandwidth_saver=bandwidth_saver),
            "live_recovery": LiveRecoveryPolicy(),
            # Full-quality requirement a saver session must sustain to recover.
            "recovery_bitrate": live_video_bitrates(settings.get("max_resolution", "1080p"))[1]
            * 1.1,
            "fast_start": True,
            "viewers": {},
            "viewer_joined": asyncio.Event(),
            "playlist_generation": 0,
            "playback_buffer_seconds": 12.0,
            "closed_captions": False,
            "plan": plan,
            "relay": plan.relay(),
            # Conservative until the probe settles it, so AAC-only clients never
            # reuse a session that may start copying Dolby.
            "audio_passthrough": audio_passthrough,
            "subtitles": [],
            "duration": 0,
            "seek_offset": 0,
        }
        with _transcode_lock:
            _transcode_sessions[session_id] = session
            _url_to_session[url] = session_id
        await session["relay"].start()
        process = await _launch_worker(
            session_id,
            session,
            fresh_relay=True,
            previous_segments=None,
        )
        first_byte_at = session["relay"].stats.first_byte_at or started
        # A source that went silent between bursts before needs output that bridges it.
        hints = await asyncio.to_thread(relay_hints)
        burst_gap = hints["recurring_gaps"].get(parse.urlsplit(plan.source_url).netloc, 0.0)
        log.info(
            "Fast-start session %s: first byte after %.2fs, worker launched after %.2fs",
            session_id,
            first_byte_at - started,
            time.time() - started,
        )
        if not await _wait_for_fresh_fast_playlist(
            directory,
            "low.m3u8",
            process,
            None,
            minimum_segments=min_segments,
            minimum_seconds=burst_gap,
            is_disconnected=is_disconnected,
        ):
            log.error(
                "Fast-start session %s failed waiting for low.m3u8 after %.2fs "
                "(exit=%s, upstream=%s)",
                session_id,
                time.time() - started,
                process.returncode,
                session["relay"].error or "open",
            )
            raise fastapi.HTTPException(500, "Fast-start stream failed to become ready")
        content = (pathlib.Path(directory) / "low.m3u8").read_text()
        # Clients hold two of the longest waits for new output: a segment, or a burst gap.
        reserve, playback_buffer = live_buffer_durations(
            [burst_gap, *(float(value) for value in re.findall(r"#EXTINF:([\d.]+)", content))],
        )
        with _transcode_lock:
            _require_current(session_id, session)
            session["watchdog_stale_after"] = max(watchdog_stale_floor_sec, reserve + 2)
            session["playback_buffer_seconds"] = playback_buffer
            session["watchdog_task"] = _spawn_background_task(_watch_fast_live(session_id))
        log.info(
            "Fast-start session %s ready at %s after %.2fs with %.1fs playback buffer",
            session_id,
            plan.resolution,
            time.time() - started,
            playback_buffer,
        )
        _publish_fast_master(session)
        return _adaptive_session_response(session_id)
    except BaseException:
        await stop_session(session_id, force=True)
        await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
        raise


# Nothing before a keyframe decodes, so starting there costs no picture and makes the
# probe and the worker's own stream analysis nearly instant. A fresh relay also starts
# the session clock.
async def _launch_worker(
    session_id: str,
    session: FastLiveSession,
    *,
    fresh_relay: bool,
    previous_segments: set[str] | None,
) -> asyncio.subprocess.Process:
    """Launch the worker that suits the source, fed from the relay's last keyframe."""
    stream = session["relay"]
    plan = session["plan"]
    started = time.time()
    sample = await _wait_for_keyframe(stream)
    received = stream.stats.bytes_received
    path = pathlib.Path(session["dir"]) / "probe.ts"
    await asyncio.to_thread(path.write_bytes, sample)
    # Captions only shape the master playlist, published once the worker is ready.
    captions = asyncio.ensure_future(asyncio.to_thread(probe_closed_captions, str(path)))
    source = await asyncio.to_thread(probe_segment, str(path))
    remember_probe(session["url"], source)
    if fresh_relay:
        history = stream.history()
        origin_time = stream.stats.first_byte_at or time.time()
        await _restart_session_clock(session_id, session, first_video_pts(history), origin_time)
    cmd, role = plan.worker(
        session["dir"],
        source,
        restarting=previous_segments is not None,
        rendition=session.get("rendition", "low"),
        grid_offset=(first_video_pts(sample) or 0.0) % 1,
    )
    if previous_segments is not None:
        cmd = _prepare_encoder_restart(cmd, previous_segments)
    try:
        process = await _launch_ffmpeg(cmd, stdin=asyncio.subprocess.PIPE)
    except BaseException:
        captions.cancel()
        raise
    try:
        with _transcode_lock:
            _require_current(session_id, session)
            if process.stdin is None:
                raise ValueError("Expected process.stdin is not None.")
            session["process"] = process
            session["subscription"] = stream.subscribe(
                process.stdin,
                replay=_since(stream, sample, received),
            )
            session["audio_info"] = source
            session["audio_passthrough"] = uses_audio_passthrough(source, plan.audio_passthrough)
            session["master_audio_bitrate"] = surround_audio_bitrate(source, plan.audio_passthrough)
        _spawn_background_task(_monitor_fast_process(process, role, session_id))
        if previous_segments is None and plan.upscales(session["dir"], source):
            _spawn_background_task(_upgrade_to_upscale(session_id, session, source))
        closed_captions = await captions
        with _transcode_lock:
            _require_current(session_id, session)
            session["closed_captions"] = closed_captions
    except BaseException:
        captions.cancel()
        await _terminate_process(process)
        raise
    log.info(
        "Fast-start session %s probed from its keyframe in %.2fs and launched its %s%s",
        session_id,
        time.time() - started,
        role,
        " (embedded captions)" if closed_captions else "",
    )
    return process


async def _wait_for_keyframe(stream: Relay, *, max_bytes: int = 4 << 20) -> bytes:
    """Wait until the relay holds a video keyframe and return the stream from it."""
    while True:
        history = stream.history()
        if (replay := keyframe_replay(history)) is not None:
            return replay
        if stream.finished or len(history) >= max_bytes:
            return history
        await stream.wait_for_bytes(len(history) + 1)


# Panels serve every live channel at both ``.m3u8`` and ``.ts``. The TS stream has no
# playlist or redirect round trips and needs no remuxer; it measured 2.5-4.3 s to
# playable output against 5.8-10.3 s through HLS.
def _relay_url(url: str, source_id: str) -> str:
    """The URL the relay reads: an Xtream live stream as one continuous MPEG-TS stream."""
    if url.endswith(".m3u8") and live_ts_offered(source_id):
        return url.removesuffix(".m3u8") + ".ts"
    return url


def _live_resolution(settings: ServerSettings, *, bandwidth_saver: bool) -> str:
    """The resolution a live worker encodes: the configured maximum, or at most 720p for saver."""
    maximum = settings.get("max_resolution", "1080p")
    if bandwidth_saver and maximum != "480p":
        return "720p"
    return maximum


# A worker's keyframe grid is computed from its replay's first frame, so it must get
# that replay, not a newer keyframe that arrived while it launched.
def _since(stream: Relay, replay: bytes, received: int) -> bytes:
    """``replay`` plus the bytes the relay received after it was taken."""
    arrived = stream.stats.bytes_received - received
    return replay + stream.history()[-arrived:] if arrived else replay


# TensorRT takes seconds to load, so playback starts on the plain worker. The upscaled
# rendition replaces it at the next playlist generation.
async def _upgrade_to_upscale(
    session_id: str,
    session: FastLiveSession,
    source: MediaInfo | None,
) -> None:
    """Run the AI Upscale worker beside the fast one; serve it once it has caught up."""
    stream = session["relay"]
    history = stream.history()
    replay, received = keyframe_replay(history) or history, stream.stats.bytes_received
    cmd, role = session["plan"].worker(
        session["dir"],
        source,
        restarting=False,
        rendition="up",
        grid_offset=(first_video_pts(replay) or 0.0) % 1,
    )
    process = await _launch_ffmpeg(cmd, stdin=asyncio.subprocess.PIPE)
    directory = session["dir"]
    subscription: Subscription | None = None
    switched = False
    try:
        if process.stdin is None:
            raise ValueError("Expected process.stdin is not None.")
        subscription = stream.subscribe(process.stdin, replay=_since(stream, replay, received))
        _spawn_background_task(_monitor_fast_process(process, role, session_id))
        if await _wait_for_fresh_fast_playlist(
            directory,
            "up.m3u8",
            process,
            None,
        ) and await _caught_up(directory, process):
            # Snapshot the plain playlist before stopping its worker: a stopped worker
            # flushes a short last segment, and published sequence numbers stay fixed.
            low_path = pathlib.Path(directory) / "low.m3u8"
            with _transcode_lock:
                _require_current(session_id, session)
                fast_process, fast_subscription = session["process"], session.get("subscription")
                session.update(
                    process=process,
                    subscription=subscription,
                    rendition="up",
                    splice_low=low_path.read_text(),
                    playlist_generation=session.get("playlist_generation", 0) + 1,
                )
            switched = True
            if fast_subscription is not None:
                fast_subscription.cancel()
            await _terminate_process(fast_process)
            log.info("Fast-start session %s switched to its AI Upscale rendition", session_id)
        else:
            log.warning(
                "Fast-start session %s keeps its plain rendition: AI Upscale failed",
                session_id,
            )
    except KeyError:
        log.info("Fast-start session %s ended before AI Upscale caught up", session_id)
    finally:
        if not switched:
            if subscription is not None:
                subscription.cancel()
            await _terminate_process(process)


async def _caught_up(
    directory: str,
    process: ManagedProcess,
    *,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.2,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> bool:
    """Wait until the upscaled rendition reaches where the plain one has got to."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and process.returncode is None:
        texts = [
            await asyncio.to_thread((pathlib.Path(directory) / name).read_text)
            for name in ("low.m3u8", "up.m3u8")
        ]
        low_end, up_end = (playlist_end_pts(directory, text) for text in texts)
        if low_end is not None and up_end is not None and up_end >= low_end:
            return True
        await sleep(poll_interval_sec)
    return False


async def _prestart(
    url: str,
    username: str,
    source_id: str,
    deinterlace_fallback: bool,
    user_max_streams: int,
    source_max_streams: int,
) -> None:
    try:
        response = await start_transcode(
            url,
            "live",
            deinterlace_fallback=deinterlace_fallback,
            username=username,
            source_id=source_id,
            user_max_streams=user_max_streams,
            source_max_streams=source_max_streams,
            min_segments=1,
        )
    except fastapi.HTTPException as error:
        log.info("Live prestart did not start: %s", error.detail)
        return
    # Not a viewer itself: a page that never asks for the stream lets it stop.
    if "viewer_id" in response:
        await stop_session(response["session_id"], viewer=response["viewer_id"])


def _audio_lang(username: str) -> str:
    """The user's preferred audio language; English when unset."""
    return load_user_settings(username).get("audio_lang", "en") if username else ""


def _user_audio(media_info: MediaInfo | None, username: str) -> MediaInfo | None:
    """``media_info`` playing the audio track in the user's preferred language."""
    return with_audio_language(media_info, _audio_lang(username))


def _with_crop(media_info: MediaInfo | None, crop: CropBox | None) -> MediaInfo | None:
    """``media_info`` cutting ``crop``, the crop its session's earlier segments used."""
    return None if media_info is None else replace(media_info, crop=crop)


def _remember_detected_crop(
    url: str,
    directory: str,
    series_id: int | None,
    episode_id: int | None,
) -> None:
    """Cache the picture area a VOD transcode detected, for its later transcodes."""
    crop = detected_crop(directory)
    if crop is not None:
        remember_crop(url, crop, series_id, episode_id)


def _admit(session_id: str, username: str) -> None:
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session is not None:
            session["guests"] = session.get("guests", frozenset()) | {username}


async def _stop_if_unwatched(url: str) -> None:
    existing = _get_existing_session(url)
    if existing is not None and _is_fast_live(existing.session) and not existing.session["viewers"]:
        await stop_session(existing.session_id)
