"""FFmpeg session lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncio
import contextlib
import json
import logging
import pathlib
import re
import shutil
import tempfile
import threading
import time
import uuid

from fastapi import HTTPException

from ffmpeg_command import (
    SEG_PREFIX,
    HwAccel,
    MediaInfo,
    SubtitleStream,
    build_hls_ffmpeg_cmd,
    get_hls_segment_duration,
    get_settings,
    get_transcode_dir,
    get_user_agent,
    invalidate_series_probe_cache,
    probe_media,
    restore_probe_cache_entry,
)


log = logging.getLogger(__name__)

# Timing constants
_POLL_INTERVAL_SEC = 0.2
_QUICK_FAILURE_THRESHOLD_SEC = 10.0
_HEARTBEAT_TIMEOUT_SEC = 30.0  # 30 sec without progress poll = dead

# Wait timeouts (seconds)
_PLAYLIST_WAIT_TIMEOUT_SEC = 30.0
_PLAYLIST_WAIT_SEEK_TIMEOUT_SEC = 40.0
_REUSE_ACTIVE_WAIT_TIMEOUT_SEC = 15.0
_RESUME_WAIT_TIMEOUT_SEC = 10.0
_RESUME_SEGMENT_WAIT_TIMEOUT_SEC = 5.0

# Size thresholds
_MIN_SEGMENT_SIZE_BYTES = 1_000

# Module state
_transcode_sessions: dict[str, dict[str, Any]] = {}
_url_to_session: dict[str, str] = {}  # URL -> session_id (all content types)
_transcode_lock = threading.Lock()
_background_tasks: set[asyncio.Task[None]] = set()


class _DeadProcess:
    """Placeholder for dead/recovered processes."""

    returncode = -1

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


# ===========================================================================
# Cache Timeout Helpers
# ===========================================================================


def get_vod_cache_timeout() -> int:
    """Get VOD session cache timeout in seconds."""
    return get_settings().get("vod_transcode_cache_mins", 60) * 60


def get_live_cache_timeout() -> int:
    """Get live session cache timeout in seconds."""
    return get_settings().get("live_transcode_cache_secs", 0)


# ===========================================================================
# Session Validity
# ===========================================================================


def _is_process_alive(proc: Any) -> bool:
    """Check if process is still running."""
    if proc is None:
        return False
    if isinstance(proc, _DeadProcess):
        return False
    if hasattr(proc, "returncode"):
        return proc.returncode is None
    return False


def is_session_valid(session: dict[str, Any]) -> bool:
    """Check if session is still valid (not expired).

    A session is valid if:
    - Has received a heartbeat (progress poll) within timeout, AND
    - Process is still running, OR process is dead but within cache timeout
    """
    last_access = session.get("last_access", session["started"])
    time_since_heartbeat = time.time() - last_access

    # No heartbeat in 30 sec = dead regardless of process state
    if time_since_heartbeat > _HEARTBEAT_TIMEOUT_SEC:
        return False

    # Active process with recent heartbeat = valid
    if _is_process_alive(session.get("process")):
        return True

    # Dead process: check cache timeout
    is_vod = session.get("is_vod", False)
    cache_timeout = get_vod_cache_timeout() if is_vod else get_live_cache_timeout()
    if cache_timeout <= 0:
        return False  # No caching of dead sessions
    return time_since_heartbeat < cache_timeout


def _kill_process(proc: Any) -> bool:
    """Kill process gracefully (SIGTERM then SIGKILL), return True if killed."""
    try:
        # Try graceful termination first (lets ffmpeg flush buffers)
        proc.terminate()
        # Give it a moment to exit cleanly
        for _ in range(10):  # 100ms total
            if proc.returncode is not None:
                return True
            time.sleep(0.01)
        # Force kill if still running
        proc.kill()
        return True
    except (ProcessLookupError, OSError):
        return False


# ===========================================================================
# Session Start/Stop
# ===========================================================================


def stop_session(session_id: str, force: bool = False) -> None:
    """Stop a transcode session."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if not session:
            return

        # Skip stop if VOD session was accessed recently (race with seeking/resume)
        # Live streams kill immediately - no resume needed
        if (
            not force
            and session.get("is_vod")
            and time.time() - session.get("last_access", 0) < 5.0
        ):
            log.info("Ignoring stop for recently-accessed VOD session %s", session_id)
            return

        if _kill_process(session["process"]):
            log.info("Killed ffmpeg for session %s", session_id)

        # Cache session if timeout > 0
        is_vod = session.get("is_vod", False)
        cache_timeout = get_vod_cache_timeout() if is_vod else get_live_cache_timeout()
        if not force and cache_timeout > 0:
            session["last_access"] = time.time()
            log.info(
                "Session %s cached (vod=%s, ffmpeg stopped, segments kept)",
                session_id,
                is_vod,
            )
            return

        _transcode_sessions.pop(session_id, None)
        url = session.get("url")
        if url:
            _url_to_session.pop(url, None)
        dir_to_remove = session["dir"]

    shutil.rmtree(dir_to_remove, ignore_errors=True)
    log.info("Stopped transcode session %s", session_id)


def cleanup_expired_sessions() -> None:
    """Clean up all expired sessions (VOD and live)."""
    with _transcode_lock:
        expired = [
            sid
            for sid, session in list(_transcode_sessions.items())
            if not is_session_valid(session)
        ]
    for session_id in expired:
        stop_session(session_id, force=True)


def shutdown() -> None:
    """Kill all running ffmpeg processes for clean shutdown."""
    with _transcode_lock:
        for session_id, session in list(_transcode_sessions.items()):
            proc = session.get("process")
            if proc and _kill_process(proc):
                log.info("Shutdown: killed ffmpeg for session %s", session_id)
        _transcode_sessions.clear()


# ===========================================================================
# Stream Limits
# ===========================================================================


def get_user_sessions(username: str) -> list[tuple[str, dict[str, Any]]]:
    """Get all active sessions for a user, sorted by start time (oldest first)."""
    with _transcode_lock:
        sessions = [
            (sid, s) for sid, s in _transcode_sessions.items() if s.get("username") == username
        ]
    return sorted(sessions, key=lambda x: x[1].get("started", 0))


def get_source_sessions(source_id: str) -> list[tuple[str, dict[str, Any]]]:
    """Get all active sessions for a source, sorted by start time (oldest first)."""
    with _transcode_lock:
        sessions = [
            (sid, s) for sid, s in _transcode_sessions.items() if s.get("source_id") == source_id
        ]
    return sorted(sessions, key=lambda x: x[1].get("started", 0))


def enforce_stream_limits(
    username: str,
    source_id: str | None,
    user_max: int,
    source_max: int,
) -> str | None:
    """Enforce stream limits, stopping oldest sessions if needed.

    Returns error message if source is at capacity and user can't reclaim,
    or None if limits are satisfied.
    """
    # Check source limit first (hard limit - can only reclaim own slots)
    if source_id and source_max > 0:
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
                stop_session(oldest_sid, force=True)
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
            stop_session(oldest_sid, force=True)

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
        if not d.is_dir():
            continue

        info_file = d / "session.json"
        try:
            mtime = d.stat().st_mtime
        except OSError:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
            continue

        # No session.json = orphaned (live session or failed VOD)
        if not info_file.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
            continue

        # Expired VOD session
        if now - mtime > cache_timeout:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
            continue

        # No segments = nothing to recover
        if not list(d.glob(f"{SEG_PREFIX}*.ts")):
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
            continue

        # Try to recover VOD session
        try:
            info = json.loads(info_file.read_text())
            if not (info.get("is_vod") and info.get("url")):
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
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
                    "last_access": now,  # Use current time, not mtime, to avoid immediate expiration
                    "subtitles": info.get("subtitles") or info.get("subtitle_indices"),
                    "duration": info.get("duration", 0),
                    "seek_offset": new_seek,
                    "series_id": info.get("series_id"),
                    "episode_id": info.get("episode_id"),
                    "username": info.get("username", ""),
                    "source_id": info.get("source_id", ""),
                }
                # Prefer session with seek_offset or more recent mtime
                existing_id = _url_to_session.get(url)
                if existing_id:
                    existing = _transcode_sessions.get(existing_id, {})
                    existing_seek = existing.get("seek_offset", 0)
                    existing_mtime = existing.get("last_access", 0)
                    if (new_seek > 0 and existing_seek == 0) or (
                        existing_seek == 0 and new_seek == 0 and mtime > existing_mtime
                    ):
                        _url_to_session[url] = session_id
                else:
                    _url_to_session[url] = session_id

            # Restore probe cache
            if p := info.get("probe"):
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
                )
                subs = [
                    SubtitleStream(s["index"], s.get("lang", "und"), s.get("name", ""))
                    for s in (info.get("subtitles") or [])
                    if isinstance(s, dict) and "index" in s
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
                _url_to_session.pop(url, None)
                session = _transcode_sessions.pop(session_id, None)
            # Clean up output directory
            if session:
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


def _spawn_background_task(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ===========================================================================
# Playlist Helpers
# ===========================================================================


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
                seg_files = list(output_dir.glob(f"{SEG_PREFIX}*.ts"))
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
        durations = re.findall(r"#EXTINF:([\d.]+)", playlist_path.read_text())
        if durations:
            return sum(float(d) for d in durations)
    return segment_count * get_hls_segment_duration()


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


def _regenerate_playlist(output_dir: pathlib.Path, start_segment: int) -> None:
    """Regenerate HLS playlist starting from a specific segment (for smart seek)."""
    playlist_path = output_dir / "stream.m3u8"
    seg_duration = get_hls_segment_duration()

    # Find all existing segments from start_segment onwards
    segments = []
    for seg_file in sorted(output_dir.glob(f"{SEG_PREFIX}*.ts")):
        try:
            seg_num = int(seg_file.stem[len(SEG_PREFIX) :])
            if seg_num >= start_segment and seg_file.stat().st_size > _MIN_SEGMENT_SIZE_BYTES:
                segments.append((seg_num, seg_file.name))
        except ValueError:
            pass

    if not segments:
        return

    # Build playlist
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


# ===========================================================================
# Session Snapshots
# ===========================================================================


@dataclass(slots=True)
class _SessionSnapshot:
    """Immutable snapshot of session state for lock-free access."""

    output_dir: str
    process: Any
    seek_offset: float
    subtitles: list[dict[str, Any]]
    duration: float


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


def _build_session_response(
    session_id: str,
    snap: _SessionSnapshot,
    playlist_path: pathlib.Path,
) -> dict[str, Any]:
    """Build response dict for existing session, recalculating duration."""
    segments = list(playlist_path.parent.glob(f"{SEG_PREFIX}*.ts"))
    return {
        "session_id": session_id,
        "playlist": f"/transcode/{session_id}/stream.m3u8",
        "subtitles": _build_subtitle_tracks(session_id, snap.subtitles),
        "duration": snap.duration,
        "seek_offset": snap.seek_offset,
        "transcoded_duration": _calc_hls_duration(playlist_path, len(segments)),
    }


# ===========================================================================
# Existing Session Handling
# ===========================================================================


def _get_existing_session(url: str) -> tuple[str | None, bool, float]:
    """Get existing session info atomically. Returns (session_id, is_valid, seek_offset)."""
    with _transcode_lock:
        existing_id = _url_to_session.get(url)
        if not existing_id:
            return None, False, 0.0
        session = _transcode_sessions.get(existing_id)
        if not session:
            return None, False, 0.0
        return (
            existing_id,
            is_session_valid(session),
            session.get("seek_offset", 0),
        )


async def _handle_existing_vod_session(
    existing_id: str,
    url: str,
    hw: HwAccel,
    do_probe: bool,
    max_resolution: str = "1080p",
    quality: str = "high",
) -> dict[str, Any] | None:
    """Handle existing VOD session: reuse active, return cached, or append.

    Returns None to trigger fresh start if session is invalid.
    """
    snap = _get_session_snapshot(existing_id)
    if not snap:
        return None

    playlist_path = pathlib.Path(snap.output_dir) / "stream.m3u8"
    segments = sorted(pathlib.Path(snap.output_dir).glob(f"{SEG_PREFIX}*.ts"))

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
            _url_to_session.pop(url, None)
        return None

    # Case 3: Dead session with seek_offset - return cached content
    if snap.seek_offset > 0:
        log.info(
            "Returning cached session %s (seek_offset=%.1f)",
            existing_id,
            snap.seek_offset,
        )
        return _build_session_response(existing_id, snap, playlist_path)

    # Case 4: Dead session, no seek_offset - append new content
    hls_duration = _calc_hls_duration(playlist_path, len(segments))
    log.info("Resuming session %s from %.1fs", existing_id, hls_duration)

    media_info = (await asyncio.to_thread(probe_media, url))[0] if do_probe else None
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
    next_seg = f"{SEG_PREFIX}{len(segments):03d}.ts"
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


async def _try_reuse_session(
    existing_id: str,
    url: str,
    is_vod: bool,
    content_type: str,
) -> dict[str, Any] | None:
    """Try to reuse an existing valid session. Returns response or None if can't reuse."""
    if is_vod:
        settings = get_settings()
        return await _handle_existing_vod_session(
            existing_id,
            url,
            settings.get("transcode_hw", "software"),
            settings.get(
                {"movie": "probe_movies", "series": "probe_series"}.get(content_type, ""), False
            ),
            settings.get("max_resolution", "1080p"),
            settings.get("quality", "high"),
        )

    # Live: return existing session if snapshot available
    snap = _get_session_snapshot(existing_id)
    if not snap:
        return None
    playlist_path = pathlib.Path(snap.output_dir) / "stream.m3u8"
    return _build_session_response(existing_id, snap, playlist_path)


def _cleanup_invalid_session(url: str, session_id: str) -> None:
    """Clean up an invalid/expired session."""
    with _transcode_lock:
        _url_to_session.pop(url, None)
    stop_session(session_id, force=True)


# ===========================================================================
# Core Transcode Logic
# ===========================================================================


async def _do_start_transcode(
    url: str,
    content_type: str,
    series_id: int | None,
    episode_id: int | None,
    old_seek_offset: float,
    series_name: str = "",
    deinterlace_fallback: bool = True,
    username: str = "",
    source_id: str = "",
) -> dict[str, Any]:
    """Core transcode logic. Raises HTTPException on failure."""
    settings = get_settings()
    hw = settings.get("transcode_hw", "software")
    max_resolution = settings.get("max_resolution", "1080p")
    quality = settings.get("quality", "high")
    is_vod = content_type in ("movie", "series")
    probe_key = {"movie": "probe_movies", "series": "probe_series", "live": "probe_live"}
    do_probe = settings.get(probe_key.get(content_type, ""), False)

    session_id = str(uuid.uuid4())
    output_dir = tempfile.mkdtemp(
        prefix=f"netv_transcode_{session_id}_",
        dir=get_transcode_dir(),
    )
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

    cmd = build_hls_ffmpeg_cmd(
        url,
        hw,
        output_dir,
        is_vod,
        subtitles,
        media_info,
        max_resolution,
        quality,
        get_user_agent(),
        deinterlace_fallback,
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

    sub_info = [{"index": s.index, "lang": s.lang, "name": s.name} for s in subtitles]
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
            "username": username,
            "source_id": source_id,
        }
        _url_to_session[url] = session_id

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
            "username": username,
            "source_id": source_id,
        }
        if media_info:
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
            }
        (pathlib.Path(output_dir) / "session.json").write_text(json.dumps(session_info))

    timeout = _PLAYLIST_WAIT_SEEK_TIMEOUT_SEC if old_seek_offset > 0 else _PLAYLIST_WAIT_TIMEOUT_SEC
    if not await _wait_for_playlist(
        playlist_path,
        process,
        min_segments=2,
        timeout_sec=timeout,
    ):
        # Wait for process to fully exit and stderr to be captured
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=1.0)
        # Give stderr monitor time to process final output
        await asyncio.sleep(0.1)
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


async def start_transcode(
    url: str,
    content_type: str = "live",
    series_id: int | None = None,
    episode_id: int | None = None,
    series_name: str = "",
    deinterlace_fallback: bool = True,
    username: str = "",
    source_id: str = "",
    user_max_streams: int = 0,
    source_max_streams: int = 0,
) -> dict[str, Any]:
    """Start or reuse a transcode session."""
    # Enforce stream limits
    if username:
        error = enforce_stream_limits(username, source_id, user_max_streams, source_max_streams)
        if error:
            raise HTTPException(status_code=429, detail=error)

    is_vod = content_type in ("movie", "series")
    existing_id, is_valid, old_seek_offset = _get_existing_session(url)

    # Try to reuse existing valid session
    if existing_id and is_valid:
        log.info("Found valid existing session %s (vod=%s)", existing_id, is_vod)
        result = await _try_reuse_session(existing_id, url, is_vod, content_type)
        if result:
            return result

    # Clean up any existing invalid session
    if existing_id:
        log.info("Cleaning up invalid session %s", existing_id)
        _cleanup_invalid_session(url, existing_id)

    # Start fresh transcode (with retry for series probe cache staleness)
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
        )
    except HTTPException:
        if series_id is None:
            raise
        log.info("Transcode failed, clearing probe cache and retrying")
        invalidate_series_probe_cache(series_id, episode_id)
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
        )


# ===========================================================================
# Session Query/Update
# ===========================================================================


def get_session(session_id: str) -> dict[str, Any] | None:
    """Get a copy of session dict (safe to use outside lock)."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        return dict(session) if session else None


def touch_session(session_id: str) -> bool:
    """Update session last_access timestamp (heartbeat). Returns True if session exists."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if session:
            session["last_access"] = time.time()
            return True
        return False


def get_session_progress(session_id: str) -> dict[str, Any] | None:
    """Get transcode progress for a session."""
    touch_session(session_id)

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


def clear_url_session(url: str) -> str | None:
    """Clear URL-to-session mapping."""
    with _transcode_lock:
        return _url_to_session.pop(url, None)


# ===========================================================================
# Seek
# ===========================================================================


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
            _url_to_session[url] = session_id
        return True


async def seek_transcode(session_id: str, seek_time: float) -> dict[str, Any]:
    """Seek to a specific time in a VOD session."""
    info = _get_seek_session_info(session_id)
    if not info:
        raise HTTPException(404, "Session not found or not VOD")

    settings = get_settings()
    hw = settings.get("transcode_hw", "software")
    max_resolution = settings.get("max_resolution", "1080p")
    quality = settings.get("quality", "high")
    seg_duration = get_hls_segment_duration()
    segment_num = int(seek_time / seg_duration)

    output_path = pathlib.Path(info.output_dir)
    target_segment = output_path / f"{SEG_PREFIX}{segment_num:03d}.ts"

    # Smart seek: if target segment exists, no need to restart ffmpeg
    if target_segment.exists() and target_segment.stat().st_size > _MIN_SEGMENT_SIZE_BYTES:
        log.info(
            "Smart seek: segment %d exists for time %.1fs, skipping ffmpeg restart",
            segment_num,
            seek_time,
        )
        with _transcode_lock:
            session = _transcode_sessions.get(session_id)
            if session:
                session["seek_offset"] = seek_time
        _regenerate_playlist(output_path, segment_num)
        return {"session_id": session_id, "playlist": f"/transcode/{session_id}/stream.m3u8"}

    # Kill existing process
    if _kill_process(info.process):
        log.info("Killed ffmpeg for seek in session %s", session_id)

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
    for vtt_file in output_path.glob("sub*.vtt"):
        vtt_file.unlink(missing_ok=True)

    # Use probe_series if series_id, else probe_movies
    probe_setting = "probe_series" if info.series_id else "probe_movies"
    do_probe = settings.get(probe_setting, False)
    if do_probe:
        media_info = (
            await asyncio.to_thread(
                probe_media,
                info.url,
                info.series_id,
                info.episode_id,
            )
        )[0]
    else:
        media_info = None

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
        quality,
        get_user_agent(),
    )
    i_idx = cmd.index("-i")
    cmd.insert(i_idx, str(seek_time))
    cmd.insert(i_idx, "-ss")
    # Shift output timestamps so subtitles start at 0 after seek
    f_idx = cmd.index("-f")
    cmd.insert(f_idx, str(-seek_time))
    cmd.insert(f_idx, "-output_ts_offset")
    cmd.extend(["-start_number", str(segment_num)])

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

    if not _update_seek_session(session_id, info.url, process, seek_time):
        _kill_process(process)
        raise HTTPException(404, "Session disappeared during seek")

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
        raise HTTPException(500, "Seek transcode timed out waiting for playlist")

    log.info("Seek ready: %s", playlist_file)

    return {
        "ok": True,
        "segment": segment_num,
        "time": seek_time,
    }
