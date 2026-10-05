"""FFmpeg session lifecycle management."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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
import urllib.parse
import uuid

from fastapi import HTTPException

from fast_start import (
    aligned,
    encoder_command,
    ingest_command,
    master_playlist,
    playlist_duration,
    ready_bitrate,
    segment_pts,
    surround_audio_bitrate,
)
from ffmpeg_command import (
    SEG_PREFIX,
    HwAccel,
    MediaInfo,
    SubtitleStream,
    build_hls_ffmpeg_cmd,
    can_remux_live,
    get_hls_segment_duration,
    get_settings,
    get_transcode_dir,
    get_user_agent,
    invalidate_series_probe_cache,
    live_video_bitrates,
    probe_audio,
    probe_media,
    resolve_hls_master_playlist,
    restore_probe_cache_entry,
    uses_audio_passthrough,
)
from playback_policy import LiveRecoveryPolicy, PlaybackHealth, PlaybackPolicy, UpgradePolicy
from util import redact_url_credentials


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
_LIVE_WATCHDOG_INTERVAL_SEC = 2.0
# Publication gaps are not failures; dead processes bypass this age check.
_LIVE_WATCHDOG_STALE_FLOOR_SEC = 45.0

# Size thresholds
_MIN_SEGMENT_SIZE_BYTES = 1_000

# Module state
_transcode_sessions: dict[str, dict[str, Any]] = {}
_url_to_session: dict[str, str] = {}  # URL -> session_id (all content types)
_transcode_lock = threading.Lock()
_background_tasks: set[asyncio.Task[None]] = set()


def _is_archive_url(url: str) -> bool:
    return "/timeshift/" in urllib.parse.urlsplit(url).path


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
    return getattr(proc, "returncode", 0) is None


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

    # Keep a recently-accessed live session addressable while its watchdog
    # replaces a dead primary process.
    if session.get("fast_start") and session.get("live_recovery"):
        return True

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

        # Archive seeks open another upstream URL, so never retain their slot.
        force = force or _is_archive_url(session.get("url", ""))
        # Skip stop if session was accessed recently (race with seeking/resume,
        # or multiple users watching same stream)
        if not force and time.time() - session.get("last_access", 0) < 5.0:
            log.info("Ignoring stop for recently-accessed session %s", session_id)
            return

        for task_name in ("watchdog_task", "high_recovery_task"):
            task = session.get(task_name)
            if task is not None and not task.done():
                task.cancel()

        if _kill_process(session["process"]):
            log.info("Killed ffmpeg for session %s", session_id)
        for proc in session.get("extra_processes", []):
            _kill_process(proc)

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
        if url and _url_to_session.get(url) == session_id:
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
            for task_name in ("watchdog_task", "high_recovery_task"):
                task = session.get(task_name)
                if task is not None and not task.done():
                    task.cancel()
            for extra in session.get("extra_processes", []):
                _kill_process(extra)
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
            if not (info.get("is_vod") and info.get("url")) or _is_archive_url(info["url"]):
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
                    "audio_passthrough": info.get("audio_passthrough", False),
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


async def _launch_ffmpeg(cmd: list[str]) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )


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


def _spawn_background_task(coro: Any) -> asyncio.Task[None]:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


# ===========================================================================
# Playlist Helpers
# ===========================================================================


async def _wait_for_playlist(
    playlist_path: pathlib.Path,
    process: asyncio.subprocess.Process,
    min_segments: int = 1,
    timeout_sec: float = _PLAYLIST_WAIT_TIMEOUT_SEC,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> bool:
    """Wait for playlist with min_segments, checking process health."""
    output_dir = playlist_path.parent
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if is_disconnected and await is_disconnected():
            raise HTTPException(499, "Playback request disconnected")
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


def _caption_master_playlist(
    sub_info: list[dict[str, Any]],
    max_resolution: str,
) -> str:
    def attribute(value: Any, fallback: str) -> str:
        cleaned = " ".join(str(value or fallback).split())
        return cleaned.replace('"', "'")

    _, maximum_bitrate = live_video_bitrates(max_resolution)
    lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
    used_names: dict[str, int] = {}
    for i, subtitle in enumerate(sub_info):
        base_name = attribute(subtitle.get("name"), f"Captions {i + 1}")
        occurrence = used_names.get(base_name, 0)
        used_names[base_name] = occurrence + 1
        name = base_name if occurrence == 0 else f"{base_name} {occurrence + 1}"
        language = attribute(subtitle.get("lang"), "und")
        lines.append(
            '#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="captions",'
            f'NAME="{name}",LANGUAGE="{language}",DEFAULT=NO,AUTOSELECT=YES,'
            f'FORCED=NO,URI="sub{i}.m3u8"'
        )
    bandwidth = int(maximum_bitrate * 1.1) + 640_000
    lines += [
        f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},SUBTITLES="captions"',
        "stream.m3u8",
    ]
    return "\n".join(lines) + "\n"


def _write_caption_master(
    output_dir: str,
    sub_info: list[dict[str, Any]],
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
        ]
    )
    for index in range(count):
        (pathlib.Path(output_dir) / f"sub{index}.m3u8").write_text(content)


def _caption_playlist_url(
    session_id: str,
    output_dir: str,
    sub_info: list[dict[str, Any]],
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


def get_caption_timestamp_origin(session_id: str) -> float | None:
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if not session:
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
        if not session:
            return None
        session.setdefault("caption_timestamp_origin", origin)
        return float(session["caption_timestamp_origin"])


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
    audio_passthrough: bool = False


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
            audio_passthrough=session.get("audio_passthrough", False),
        )


def _update_session_process(
    session_id: str,
    process: Any,
    *,
    seek_time: float | None = None,
    url: str = "",
) -> bool:
    """Atomically update session process. Returns False if session gone."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if not session:
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
) -> dict[str, Any]:
    """Build response dict for existing session, recalculating duration."""
    segments = list(playlist_path.parent.glob(f"{SEG_PREFIX}*.ts"))
    response = {
        "session_id": session_id,
        "playlist": f"/transcode/{session_id}/stream.m3u8",
        "subtitles": _build_subtitle_tracks(session_id, snap.subtitles),
        "duration": snap.duration,
        "seek_offset": snap.seek_offset,
        "transcoded_duration": _calc_hls_duration(playlist_path, len(segments)),
    }
    caption_playlist = _caption_playlist_url(session_id, snap.output_dir, snap.subtitles)
    if caption_playlist:
        response["caption_playlist"] = caption_playlist
    return response


# ===========================================================================
# Existing Session Handling
# ===========================================================================


def _fast_playlist_url(session_id: str, name: str, generation: int) -> str:
    url = f"/transcode/{session_id}/{name}"
    return f"{url}?generation={generation}" if generation > 0 else url


def _adaptive_session_response(session_id: str) -> dict[str, Any]:
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        generation = int(session.get("playlist_generation", 0)) if session else 0
    return {
        "session_id": session_id,
        "playlist": _fast_playlist_url(session_id, "low.m3u8", generation),
        "master_playlist": f"/transcode/{session_id}/master.m3u8",
        "subtitles": [],
        "duration": 0,
        "seek_offset": 0,
    }


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
    if not segments or _is_archive_url(url):
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

    # Resolve HLS master playlist to highest bandwidth variant
    url = await asyncio.to_thread(resolve_hls_master_playlist, url)

    media_info = (
        (await asyncio.to_thread(probe_media, url, None, None, ""))[0] if do_probe else None
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

    process = await _launch_ffmpeg(cmd)
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
    bandwidth_saver: bool = False,
    audio_passthrough: bool = False,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> dict[str, Any]:
    """Core transcode logic. Raises HTTPException on failure."""
    is_archive = _is_archive_url(url)
    # Resolve HLS master playlist to highest bandwidth variant
    url = await asyncio.to_thread(resolve_hls_master_playlist, url)

    settings = get_settings()
    hw = settings.get("transcode_hw", "software")
    max_resolution = settings.get("max_resolution", "1080p")
    quality = settings.get("quality", "high")
    if bandwidth_saver:
        max_resolution = "480p" if max_resolution == "480p" else "720p"
        quality = "low"
    is_vod = content_type in ("movie", "series")
    probe_key = {"movie": "probe_movies", "series": "probe_series", "live": "probe_live"}
    do_probe = settings.get(probe_key.get(content_type, ""), False)

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

    if is_archive and is_disconnected and await is_disconnected():
        raise HTTPException(499, "Playback request disconnected")

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
        is_vod,
        subtitles,
        media_info,
        max_resolution,
        quality,
        get_user_agent(),
        deinterlace_fallback,
        allow_upscale=not bandwidth_saver and not is_archive,
        audio_passthrough=audio_passthrough,
    )
    passthrough_used = uses_audio_passthrough(media_info, audio_passthrough)
    if old_seek_offset > 0:
        i_idx = cmd.index("-i")
        cmd.insert(i_idx, str(old_seek_offset))
        cmd.insert(i_idx, "-ss")
        log.info("Applying seek_offset=%.1f from previous session", old_seek_offset)

    sub_info = [{"index": s.index, "lang": s.lang, "name": s.name} for s in subtitles]
    total_duration = media_info.duration if media_info else 0.0
    try:
        caption_master = _write_caption_master(output_dir, sub_info, max_resolution)
    except OSError as error:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(500, "Could not prepare captions") from error

    log.info(
        "Starting transcode session %s (vod=%s): %s",
        session_id,
        is_vod,
        " ".join(redact_url_credentials(arg) for arg in cmd),
    )

    process = await _launch_ffmpeg(cmd)

    stderr_lines: list[str] = []
    _spawn_background_task(_monitor_ffmpeg_stderr(process, session_id, stderr_lines))

    with _transcode_lock:
        _transcode_sessions[session_id] = {
            "dir": output_dir,
            "process": process,
            "started": time.time(),
            "url": url,
            "is_vod": is_vod,
            "is_archive": is_archive,
            "last_access": time.time(),
            "subtitles": sub_info,
            "duration": total_duration,
            "seek_offset": old_seek_offset,
            "series_id": series_id,
            "episode_id": episode_id,
            "username": username,
            "source_id": source_id,
            "bandwidth_saver": bandwidth_saver,
            "audio_passthrough": passthrough_used,
            "playback_policy": PlaybackPolicy(bandwidth_saver=bandwidth_saver),
            "recovery_bitrate": live_video_bitrates(settings.get("max_resolution", "1080p"))[1]
            * 1.1,
        }
        _url_to_session[url] = session_id

    if is_vod and not is_archive:
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
            "audio_passthrough": passthrough_used,
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
    try:
        ready = await _wait_for_playlist(
            playlist_path,
            process,
            min_segments=2,
            timeout_sec=timeout,
            is_disconnected=is_disconnected if is_archive else None,
        )
    except (asyncio.CancelledError, HTTPException):
        stop_session(session_id, force=True)
        raise
    if not ready:
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
        stop_session(session_id, force=True)
        raise HTTPException(500, "Transcode failed - check server logs for details")

    if sub_info:
        get_caption_timestamp_origin(session_id)

    response = {
        "session_id": session_id,
        "playlist": f"/transcode/{session_id}/stream.m3u8",
        "subtitles": _build_subtitle_tracks(session_id, sub_info),
        "duration": total_duration,
        "seek_offset": old_seek_offset,
    }
    if caption_master:
        response["caption_playlist"] = f"/transcode/{session_id}/captions.m3u8"
    return response


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
    bandwidth_saver: bool = False,
    fast_start: bool = False,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    audio_passthrough: bool = False,
) -> dict[str, Any]:
    """Start or reuse a transcode session.

    audio_passthrough: the client decodes AC-3/E-AC-3, so Dolby audio may be copied.
    """
    if bandwidth_saver and content_type != "live":
        raise HTTPException(400, "Adaptive playback is only supported for live streams")
    # Enforce stream limits
    if username:
        error = enforce_stream_limits(username, source_id, user_max_streams, source_max_streams)
        if error:
            raise HTTPException(status_code=429, detail=error)

    # A quality change replaces the current stream; release its slot first.
    existing_id, is_valid, old_seek_offset = _get_existing_session(url)
    if _is_archive_url(url):
        old_seek_offset = 0.0
        existing = get_session(existing_id) if existing_id else None
        if existing and existing.get("username") != username:
            raise HTTPException(409, "This archive is playing on another device.")
    if bandwidth_saver and existing_id:
        session = get_session(existing_id)
        if session and session.get("username") != username:
            raise HTTPException(404, "Session not found")
        stop_session(existing_id, force=True)
        existing_id, is_valid, old_seek_offset = None, False, 0.0
    if existing_id and not audio_passthrough:
        session = get_session(existing_id)
        if session and session.get("audio_passthrough"):
            # This client cannot decode the session's Dolby audio.
            if session.get("username") != username:
                raise HTTPException(409, "This stream is playing on another device.")
            stop_session(existing_id, force=True)
            existing_id, is_valid = None, False

    is_vod = content_type in ("movie", "series")

    # Try to reuse existing valid session
    if existing_id and is_valid:
        existing = get_session(existing_id)
        if existing and existing.get("fast_start"):
            if existing.get("username") != username:
                raise HTTPException(404, "Session not found")
            touch_session(existing_id)
            return _adaptive_session_response(existing_id)
        log.info("Found valid existing session %s (vod=%s)", existing_id, is_vod)
        result = await _try_reuse_session(existing_id, url, is_vod, content_type)
        if result:
            return result

    # Clean up any existing invalid session
    if existing_id:
        log.info("Cleaning up invalid session %s", existing_id)
        _cleanup_invalid_session(url, existing_id)

    if (
        fast_start
        and content_type == "live"
        and get_settings().get("max_resolution", "1080p") in ("1080p", "1440p", "4k")
    ):
        # A DVR request only needs server-side retention. When the source can
        # be stream-copied, serve one remux session instead of the fast-start
        # encoder pair; otherwise keep the adaptive transcode pipeline.
        settings = get_settings()
        remux = False
        if (
            not bandwidth_saver
            and settings.get("live_dvr_mins", 0) > 0
            and settings.get("probe_live", True)
        ):
            resolved = await asyncio.to_thread(resolve_hls_master_playlist, url)
            media_info = (await asyncio.to_thread(probe_media, resolved))[0]
            remux = can_remux_live(
                media_info,
                settings.get("max_resolution", "1080p"),
                audio_passthrough=audio_passthrough,
            )
        if remux:
            log.info(
                "DVR live session uses a stream-copy remux instead of fast-start: %s",
                redact_url_credentials(url),
            )
        else:
            return await _start_fast_live(
                url,
                username,
                source_id,
                deinterlace_fallback,
                bandwidth_saver=bandwidth_saver,
                is_disconnected=is_disconnected,
                audio_passthrough=audio_passthrough,
            )

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
            bandwidth_saver,
            audio_passthrough=audio_passthrough,
            is_disconnected=is_disconnected,
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
            bandwidth_saver,
            audio_passthrough=audio_passthrough,
            is_disconnected=is_disconnected,
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
    name = "stream.m3u8"
    if session.get("fast_start"):
        name = "high.m3u8" if session.get("high_selected") else "low.m3u8"
    playlist_path = pathlib.Path(session["dir"]) / name
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
    audio_passthrough: bool = False


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
            audio_passthrough=session.get("audio_passthrough", False),
        )


async def seek_transcode(session_id: str, seek_time: float) -> dict[str, Any]:
    """Seek to a specific time in a VOD session."""
    info = _get_seek_session_info(session_id)
    if not info:
        raise HTTPException(404, "Session not found or not VOD")
    if _is_archive_url(info.url):
        raise HTTPException(400, "Seek an archive by opening it at the desired start time")

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
    for subtitle_file in (*output_path.glob("sub*.vtt"), *output_path.glob("sub*.m3u8")):
        subtitle_file.unlink(missing_ok=True)

    # Resolve HLS master playlist to highest bandwidth variant
    url = await asyncio.to_thread(resolve_hls_master_playlist, info.url)

    # Use probe_series if series_id, else probe_movies
    probe_setting = "probe_series" if info.series_id else "probe_movies"
    do_probe = settings.get(probe_setting, False)
    if do_probe:
        media_info = (
            await asyncio.to_thread(
                probe_media,
                url,
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

    try:
        _write_empty_caption_playlists(info.output_dir, len(subtitles))
    except OSError as error:
        raise HTTPException(500, "Could not prepare captions after seeking") from error

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
    )
    i_idx = cmd.index("-i")
    cmd.insert(i_idx, str(seek_time))
    cmd.insert(i_idx, "-ss")
    # Keep the main HLS timeline aligned with the requested input position.
    main_output_idx = cmd.index("-max_delay")
    cmd.insert(main_output_idx, str(-seek_time))
    cmd.insert(main_output_idx, "-output_ts_offset")
    cmd.extend(["-start_number", str(segment_num)])

    log.info(
        "Seek transcode %s to %.1fs (seg %d): %s",
        session_id,
        seek_time,
        segment_num,
        " ".join(cmd),
    )

    process = await _launch_ffmpeg(cmd)

    if not _update_session_process(session_id, process, url=info.url, seek_time=seek_time):
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

    if subtitles:
        get_caption_timestamp_origin(session_id)
    log.info("Seek ready: %s", playlist_file)

    return {
        "ok": True,
        "segment": segment_num,
        "time": seek_time,
    }


def _playlist_marker(path: pathlib.Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_mtime_ns, stat.st_size
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
    process: Any,
    previous_marker: tuple[int, int, int] | None,
    *,
    previous_segments: set[str] | None = None,
    minimum_segments: int = 2,
    minimum_duration: float = 0,
    timeout_sec: float = _PLAYLIST_WAIT_TIMEOUT_SEC,
) -> bool:
    path = pathlib.Path(directory) / name
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.returncode is not None:
            return False
        marker = _playlist_marker(path)
        if (
            marker is not None
            and marker != previous_marker
            and (
                previous_segments is None
                or len(_playlist_segments(path) - previous_segments) >= minimum_segments
            )
            and ready_bitrate(directory, name, minimum_segments=minimum_segments)
            and playlist_duration(directory, name) >= minimum_duration
        ):
            return True
        await asyncio.sleep(_POLL_INTERVAL_SEC)
    return False


async def _terminate_fast_process(process: Any) -> None:
    if not _is_process_alive(process):
        return
    try:
        process.terminate()
    except (ProcessLookupError, OSError):
        return
    wait = getattr(process, "wait", None)
    if wait is None:
        _kill_process(process)
        return
    try:
        await asyncio.wait_for(wait(), timeout=5)
    except TimeoutError:
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


async def _launch_recovery_ffmpeg(command: list[str]) -> asyncio.subprocess.Process:
    launch = asyncio.create_task(_launch_ffmpeg(command))
    try:
        return await asyncio.shield(launch)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            process = await launch
            await _terminate_fast_process(process)
        raise


async def _monitor_fast_process(
    process: asyncio.subprocess.Process,
    role: str,
    session_id: str,
) -> None:
    assert process.stderr is not None
    recent: list[str] = []
    while line := await process.stderr.readline():
        message = re.sub(
            r"https?://[^\s\]]+",
            lambda match: redact_url_credentials(match.group()),
            line.decode(errors="replace").rstrip(),
        )
        recent = (recent + [message])[-10:]
    await process.wait()
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        current = bool(
            session
            and (
                process is session.get("process")
                or process is session.get("ingest_process")
                or process is session.get("high_process")
            )
        )
    if current:
        log.warning(
            "Fast-start %s exited for %s (code %s): %s",
            role,
            session_id,
            process.returncode,
            " | ".join(recent) or "no stderr",
        )


def _publish_fast_master(session: dict[str, Any], *, include_high: bool) -> None:
    path = pathlib.Path(session["dir"]) / "master.m3u8"
    temporary = path.with_name(".master.m3u8.tmp")
    temporary.write_text(
        master_playlist(
            session["master_resolution"],
            include_high=include_high,
            audio_bitrate=session["master_audio_bitrate"],
        )
    )
    temporary.replace(path)


async def _cancel_high_recovery(session_id: str, session: dict[str, Any]) -> None:
    with _transcode_lock:
        if _transcode_sessions.get(session_id) is not session:
            return
        task = session.pop("high_recovery_task", None)
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _restart_high_encoder(session_id: str, session: dict[str, Any]) -> None:
    recovery: LiveRecoveryPolicy = session["live_recovery"]
    replacement: asyncio.subprocess.Process | None = None
    success = False
    try:
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is not session:
                return
            old_process = session.get("high_process")
            session["high_process"] = None
            session["high_selected"] = False
            session["high_recovering"] = True
            session["upgrade_policy"] = UpgradePolicy()
            session["extra_processes"] = [
                process
                for process in session.get("extra_processes", [])
                if process is not old_process
            ]
            command = list(session["high_restart_command"])
        _publish_fast_master(session, include_high=False)
        await _terminate_fast_process(old_process)
        playlist_path = pathlib.Path(session["dir"]) / "high.m3u8"
        previous_marker = _playlist_marker(playlist_path)
        previous_segments = _playlist_segments(playlist_path)
        command = _prepare_encoder_restart(command, previous_segments)
        replacement = await _launch_recovery_ffmpeg(command)
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is not session:
                raise asyncio.CancelledError
            session["high_process"] = replacement
            session["extra_processes"] = [
                *session.get("extra_processes", []),
                replacement,
            ]
        _spawn_background_task(_monitor_fast_process(replacement, "high-quality", session_id))
        if not await _wait_for_fresh_fast_playlist(
            session["dir"],
            "high.m3u8",
            replacement,
            previous_marker,
            previous_segments=previous_segments,
        ):
            raise TimeoutError("high-quality output did not become ready")
        _publish_fast_master(session, include_high=True)
        recovery.mark_stage_started("high", time.monotonic())
        success = True
        log.info("Live watchdog %s restarted high-quality output", session_id)
    except asyncio.CancelledError:
        raise
    except (OSError, TimeoutError):
        log.exception("Live watchdog %s could not restart high-quality output", session_id)
    finally:
        if replacement is not None and not success:
            with _transcode_lock:
                if _transcode_sessions.get(session_id) is session:
                    if session.get("high_process") is replacement:
                        session["high_process"] = None
                    session["extra_processes"] = [
                        process
                        for process in session.get("extra_processes", [])
                        if process is not replacement
                    ]
            await _terminate_fast_process(replacement)
        recovery.finish_restart("high")
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is session:
                session["high_recovering"] = False
                if session.get("high_recovery_task") is asyncio.current_task():
                    session.pop("high_recovery_task", None)


async def _restart_fast_pipeline(
    session_id: str,
    session: dict[str, Any],
    stalled_stage: str,
) -> None:
    recovery: LiveRecoveryPolicy = session["live_recovery"]
    replacements: list[asyncio.subprocess.Process] = []
    success = False
    await _cancel_high_recovery(session_id, session)
    try:
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is not session:
                return
            old_processes: list[Any] = []
            for process in (
                session.get("process"),
                session.get("ingest_process"),
                session.get("high_process"),
                *session.get("extra_processes", []),
            ):
                if process is not None and all(process is not old for old in old_processes):
                    old_processes.append(process)
            session["process"] = _DeadProcess()
            session["ingest_process"] = None
            session["high_process"] = None
            session["extra_processes"] = []
            session["high_selected"] = False
            session["upgrade_policy"] = UpgradePolicy()
            session["watchdog_recovering"] = stalled_stage
            ingest_command_line = list(session["ingest_restart_command"])
            low_command_line = list(session["low_restart_command"])
        _publish_fast_master(session, include_high=False)
        await asyncio.gather(*(_terminate_fast_process(process) for process in old_processes))

        directory = session["dir"]
        input_playlist = pathlib.Path(directory) / "input.m3u8"
        input_playlist.unlink(missing_ok=True)
        for input_segment in pathlib.Path(directory).glob("input_*.ts*"):
            input_segment.unlink(missing_ok=True)
        low_playlist = pathlib.Path(directory) / "low.m3u8"
        low_marker = _playlist_marker(low_playlist)
        low_segments = _playlist_segments(low_playlist)
        low_command_line = _prepare_encoder_restart(low_command_line, low_segments)

        ingest = await _launch_recovery_ffmpeg(ingest_command_line)
        replacements.append(ingest)
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is not session:
                raise asyncio.CancelledError
            session["ingest_process"] = ingest
            session["extra_processes"] = [ingest]
        _spawn_background_task(_monitor_fast_process(ingest, "ingest", session_id))
        if not await _wait_for_fresh_fast_playlist(
            directory,
            "input.m3u8",
            ingest,
            None,
            previous_segments=set(),
            minimum_segments=1,
        ):
            raise TimeoutError("ingest output did not become ready")

        low = await _launch_recovery_ffmpeg(low_command_line)
        replacements.append(low)
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is not session:
                raise asyncio.CancelledError
            session["process"] = low
        _spawn_background_task(_monitor_fast_process(low, "720p", session_id))
        if not await _wait_for_fresh_fast_playlist(
            directory,
            "low.m3u8",
            low,
            low_marker,
            previous_segments=low_segments,
        ):
            raise TimeoutError("low output did not become ready")

        with _transcode_lock:
            if _transcode_sessions.get(session_id) is not session:
                raise asyncio.CancelledError
            session["playlist_generation"] = (
                int(session.get("playlist_generation", 0)) + 1
            )
        recovery.reset_pipeline(time.monotonic())
        success = True
        log.warning(
            "Live watchdog %s recovered pipeline generation %d after %s stalled",
            session_id,
            session["playlist_generation"],
            stalled_stage,
        )
    except asyncio.CancelledError:
        raise
    except (OSError, TimeoutError):
        log.exception(
            "Live watchdog %s failed to recover the pipeline after %s stalled",
            session_id,
            stalled_stage,
        )
    finally:
        if not success:
            with _transcode_lock:
                if _transcode_sessions.get(session_id) is session:
                    if any(session.get("process") is replacement for replacement in replacements):
                        session["process"] = _DeadProcess()
                    if any(
                        session.get("ingest_process") is replacement for replacement in replacements
                    ):
                        session["ingest_process"] = None
                    session["extra_processes"] = [
                        process
                        for process in session.get("extra_processes", [])
                        if all(process is not replacement for replacement in replacements)
                    ]
            for process in replacements:
                await _terminate_fast_process(process)
        recovery.finish_restart("pipeline")
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is session:
                session.pop("watchdog_recovering", None)


def _log_watchdog_delay(
    session_id: str,
    session: dict[str, Any],
    action: str,
    now: float,
) -> None:
    key = f"{action}_restart_log_at"
    if now - session.get(key, float("-inf")) >= 30:
        log.warning(
            "Live watchdog %s is delaying %s recovery because its restart budget is cooling down",
            session_id,
            action,
        )
        session[key] = now


async def _check_fast_live_session(session_id: str, session: dict[str, Any]) -> None:
    recovery: LiveRecoveryPolicy = session["live_recovery"]
    directory = session["dir"]
    now = time.monotonic()

    ingest_alive = _is_process_alive(session.get("ingest_process"))
    low_alive = _is_process_alive(session.get("process"))
    high_alive = _is_process_alive(session.get("high_process"))
    stale_after = session.get("watchdog_stale_after", 10)
    input_ready = bool(
        ingest_alive
        and ready_bitrate(
            directory,
            "input.m3u8",
            minimum_segments=1,
            max_age=stale_after,
        )
    )
    low_ready = bool(low_alive and ready_bitrate(directory, "low.m3u8", max_age=stale_after))
    high_ready = bool(high_alive and ready_bitrate(directory, "high.m3u8", max_age=stale_after))

    input_stalled = recovery.stalled("input", ingest_alive, input_ready, now)
    low_stalled = recovery.stalled("low", low_alive, low_ready, now)
    if input_stalled or low_stalled:
        stage = "input" if input_stalled else "low output"
        if recovery.permit_restart("pipeline", now):
            await _restart_fast_pipeline(session_id, session, stage)
        elif "pipeline" not in recovery.pending:
            _log_watchdog_delay(session_id, session, "pipeline", now)
        return

    high_stalled = recovery.stalled("high", high_alive, high_ready, now)
    if not high_stalled:
        return
    if recovery.permit_restart("high", now):
        task = _spawn_background_task(_restart_high_encoder(session_id, session))
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is session:
                session["high_recovery_task"] = task
            else:
                task.cancel()
    elif "high" not in recovery.pending:
        _log_watchdog_delay(session_id, session, "high-quality", now)


async def _watch_fast_live(session_id: str) -> None:
    while True:
        await asyncio.sleep(_LIVE_WATCHDOG_INTERVAL_SEC)
        with _transcode_lock:
            session = _transcode_sessions.get(session_id)
            if session is None:
                return
            recently_accessed = (
                time.time() - session.get("last_access", 0) <= _HEARTBEAT_TIMEOUT_SEC
            )
        if not recently_accessed:
            continue
        try:
            await _check_fast_live_session(session_id, session)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Live watchdog %s failed while checking pipeline health", session_id)


def report_playback_health(
    session_id: str, username: str, health: PlaybackHealth
) -> dict[str, Any]:
    """Evaluate playback pressure for the current live stream."""
    with _transcode_lock:
        session = _transcode_sessions.get(session_id)
        if not session or session.get("username") != username:
            raise HTTPException(404, "Session not found")
        if session.get("is_vod"):
            raise HTTPException(400, "Playback feedback is only supported for live streams")
        session["last_access"] = time.time()
        now = time.monotonic()
        policy = session["playback_policy"]
        # Remember actual high-rendition requirements for legacy sessions, whose
        # configured bitrate may only be an estimate.
        if not policy.bandwidth_saver and not session.get("bandwidth_saver"):
            session["recovery_bitrate"] = max(
                session.get("recovery_bitrate", 0), health.required_bitrate
            )
        high_alive = _is_process_alive(session.get("high_process"))
        bitrate = ready_bitrate(session["dir"], "high.m3u8") if high_alive else 0
        caught_up = bitrate > 0 and aligned(session["dir"]) if session.get("fast_start") else False
        if session.get("fast_start") and bitrate > 0:
            session["recovery_bitrate"] = max(session.get("recovery_bitrate", 0), bitrate)
        saver = policy.observe(
            health,
            now,
            session.get("recovery_bitrate", 0),
            recovery_ready=caught_up if session.get("fast_start") else True,
        )
        result: dict[str, Any] = {"bandwidth_saver": saver}
        if session.get("fast_start"):
            upgrade = session.setdefault("upgrade_policy", UpgradePolicy())
            was_high = session.get("high_selected", False)
            if upgrade.observe(health, bitrate, caught_up, now):
                session["high_selected"] = True
            # A temporary gap in segment publication is not a quality downgrade.
            # Existing playback health handles sustained stalls after promotion.
            if saver or not high_alive:
                session["high_selected"] = False
            # Keep the local encoder warm so recovery uses the same ingest and
            # timeline. A dead high encoder still leaves low playback available.
            name = "high.m3u8" if session.get("high_selected") else "low.m3u8"
            if now - session.get("upgrade_log_at", float("-inf")) >= 10 or was_high != session.get(
                "high_selected", False
            ):
                reason = "bandwidth saver recovering" if saver else upgrade.reason
                log.info(
                    "Playback quality %s: selected=%s reason=%s buffer=%.1fs waiting=%s "
                    "observed=%.2fMbps high=%.2fMbps aligned=%s samples=%d",
                    session_id,
                    name,
                    reason,
                    health.buffer_seconds,
                    health.waiting,
                    health.observed_bitrate / 1e6,
                    bitrate / 1e6,
                    caught_up,
                    len(upgrade.samples),
                )
                session["upgrade_log_at"] = now
            result["playlist"] = _fast_playlist_url(
                session_id,
                name,
                int(session.get("playlist_generation", 0)),
            )
        return result


async def _start_fast_live(
    url: str,
    username: str,
    source_id: str,
    deinterlace: bool,
    *,
    bandwidth_saver: bool = False,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    audio_passthrough: bool = False,
) -> dict[str, Any]:
    """One upstream reader, two independent encoders, one session/stream slot."""
    settings = get_settings()
    session_id = str(uuid.uuid4())
    directory = tempfile.mkdtemp(prefix=f"netv_transcode_{session_id}_", dir=get_transcode_dir())
    processes: list[asyncio.subprocess.Process] = []
    startup_started = time.monotonic()

    async def launch(cmd: list[str], role: str) -> asyncio.subprocess.Process:
        proc = await _launch_ffmpeg(cmd)
        processes.append(proc)
        _spawn_background_task(_monitor_fast_process(proc, role, session_id))
        log.info("Fast-start %s launched for session %s", role, session_id)
        return proc

    async def wait_ready(
        name: str,
        proc: asyncio.subprocess.Process,
        minimum_duration: float = 0,
        *,
        minimum_segments: int = 2,
    ) -> None:
        deadline = time.monotonic() + _PLAYLIST_WAIT_TIMEOUT_SEC
        while time.monotonic() < deadline and proc.returncode is None:
            if is_disconnected and await is_disconnected():
                log.info("Adaptive startup disconnected for session %s", session_id)
                raise HTTPException(499, "Playback request disconnected")
            if (
                ready_bitrate(directory, name, minimum_segments=minimum_segments)
                and playlist_duration(directory, name) >= minimum_duration
            ):
                return
            await asyncio.sleep(_POLL_INTERVAL_SEC)
        log.error(
            "Fast-start session %s failed waiting for %s after %.2fs (exit=%s)",
            session_id,
            name,
            time.monotonic() - startup_started,
            proc.returncode,
        )
        raise HTTPException(500, "Fast-start stream failed to become ready")

    try:
        user_agent = get_user_agent()
        ingest = await launch(ingest_command(url, directory, user_agent), "ingest")
        with _transcode_lock:
            _transcode_sessions[session_id] = {
                "dir": directory,
                "process": ingest,
                "ingest_process": ingest,
                "extra_processes": [],
                "started": time.time(),
                "last_access": time.time(),
                "url": url,
                "is_vod": False,
                "username": username,
                "source_id": source_id,
                "bandwidth_saver": bandwidth_saver,
                "playback_policy": PlaybackPolicy(bandwidth_saver=bandwidth_saver),
                "live_recovery": LiveRecoveryPolicy(high_started=time.monotonic()),
                "fast_start": True,
                "playlist_generation": 0,
                "ingest_restart_command": ingest_command(
                    url,
                    directory,
                    user_agent,
                    restarting=True,
                ),
                # Conservative until the audio probe settles it, so AAC-only
                # clients never reuse a session that may start copying Dolby.
                "audio_passthrough": audio_passthrough,
                "subtitles": [],
                "duration": 0,
                "seek_offset": 0,
            }
            _url_to_session[url] = session_id
        # Warm the low encoder on the first complete source segment, not a full
        # playback reserve. It still has its own buffer gate.
        await wait_ready("input.m3u8", ingest, minimum_segments=1)
        log.info(
            "Fast-start session %s input ready after %.2fs",
            session_id,
            time.monotonic() - startup_started,
        )
        first_input = min(pathlib.Path(directory).glob("input_*.ts"))
        origin_pts = segment_pts(first_input)
        # The local segment reveals the audio layout without re-reading upstream.
        audio = await asyncio.to_thread(probe_audio, str(first_input))
        use_audio_passthrough = uses_audio_passthrough(audio, audio_passthrough)
        with _transcode_lock:
            _transcode_sessions[session_id].update(
                origin_pts=origin_pts,
                origin_time=time.time(),
                audio_info=audio,
                audio_passthrough=use_audio_passthrough,
            )
        hw = settings.get("transcode_hw", "software")
        low_restart_command = encoder_command(
            directory,
            hw,
            "720p",
            "low",
            deinterlace,
            False,
            audio=audio,
            audio_passthrough=audio_passthrough,
            restarting=True,
        )
        low = await launch(
            encoder_command(
                directory,
                hw,
                "720p",
                "low",
                deinterlace,
                False,
                audio=audio,
                audio_passthrough=audio_passthrough,
            ),
            "720p",
        )
        with _transcode_lock:
            _transcode_sessions[session_id].update(
                process=low,
                extra_processes=[ingest],
                low_restart_command=low_restart_command,
            )
        # A remuxed source is released in source-keyframe-sized bursts. A couple
        # of tiny output segments cannot bridge the next upstream delivery gap.
        input_text = (pathlib.Path(directory) / "input.m3u8").read_text()
        input_durations = [float(value) for value in re.findall(r"#EXTINF:([\d.]+)", input_text)]
        startup_buffer = max(8.0, 2 * max(input_durations, default=4.0))
        with _transcode_lock:
            _transcode_sessions[session_id]["watchdog_stale_after"] = max(
                _LIVE_WATCHDOG_STALE_FLOOR_SEC,
                startup_buffer + 2,
            )
        await wait_ready("low.m3u8", low, startup_buffer)
        log.info(
            "Fast-start session %s ready at 720p after %.2fs with %.1fs startup reserve",
            session_id,
            time.monotonic() - startup_started,
            startup_buffer,
        )
        # Keep high-quality initialization off the GPU until the startup
        # rendition has built its reserve. Do not wait for high-quality output.
        high = None
        max_resolution = settings.get("max_resolution", "1080p")
        quality = settings.get("quality", "high")
        high_restart_command = encoder_command(
            directory,
            hw,
            max_resolution,
            quality,
            deinterlace,
            True,
            audio=audio,
            audio_passthrough=audio_passthrough,
            restarting=True,
        )
        with _transcode_lock:
            _transcode_sessions[session_id].update(
                high_restart_command=high_restart_command,
                master_resolution=max_resolution,
                master_audio_bitrate=surround_audio_bitrate(audio, audio_passthrough),
            )
        try:
            high = await launch(
                encoder_command(
                    directory,
                    hw,
                    max_resolution,
                    quality,
                    deinterlace,
                    True,
                    audio=audio,
                    audio_passthrough=audio_passthrough,
                ),
                "high-quality",
            )
            with _transcode_lock:
                session = _transcode_sessions[session_id]
                session.update(high_process=high, extra_processes=[ingest, high])
                session["live_recovery"].mark_stage_started("high", time.monotonic())
        except OSError:
            log.exception("High-quality encoder unavailable; continuing at 720p")
        with _transcode_lock:
            session = _transcode_sessions[session_id]
        _publish_fast_master(session, include_high=high is not None)
        watchdog_task = _spawn_background_task(_watch_fast_live(session_id))
        with _transcode_lock:
            if _transcode_sessions.get(session_id) is session:
                session["watchdog_task"] = watchdog_task
            elif watchdog_task is not None:
                watchdog_task.cancel()
        return _adaptive_session_response(session_id)
    except BaseException:
        for proc in processes:
            _kill_process(proc)
        stop_session(session_id, force=True)
        shutil.rmtree(directory, ignore_errors=True)
        raise
