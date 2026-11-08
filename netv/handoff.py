"""Handoff: pick up a live channel on another device where it was left.

While a live channel plays, the client reports the broadcast moment on screen.
When playback stops, explicitly or because those reports lapse, that moment
becomes the user's handoff. Channels with an upstream archive resume from it
as catchup. Others keep recording on the server until the program ends and
resume from the recording. Starting anything else cancels the handoff.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import json
import logging
import pathlib
import shutil
import subprocess
import threading
import time
import uuid

from netv import ffmpeg_command
from netv.util import atomic_write_json


log = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class HandoffPolicy:
    """Timing limits for retaining and recording abandoned playback."""

    min_watch_secs: float = 120.0
    watch_timeout_secs: float = 45.0
    keep_secs: float = 24 * 3600.0
    max_recording_secs: float = 3 * 3600.0
    program_end_margin_secs: float = 30.0


@dataclass(kw_only=True, slots=True, frozen=True)
class Channel:
    """A live channel and the program airing at the handoff position."""

    stream_id: str
    name: str
    url: str
    source_id: str = ""
    catchup_days: int = 0
    program_title: str = ""
    program_start: float = 0.0
    program_end: float = 0.0


@dataclass(kw_only=True, slots=True, frozen=True)
class _Watch:
    stream_id: str
    position: float
    first_seen: float
    last_seen: float
    handoff_id: str = ""


# Resolves a stream ID and broadcast moment to its channel, or None.
Resolver = Callable[[str, float], Channel | None]
# Releases a user's live transcode of a stream URL before it is recorded.
Releaser = Callable[[str, str], None]

_lock = threading.Lock()
_users_dir: pathlib.Path | None = None
_records: dict[str, dict[str, Any]] = {}
_watches: dict[str, _Watch] = {}
_recorders: dict[str, subprocess.Popen[bytes]] = {}
# Bumped on every cancel so a recorder launched concurrently is discarded.
_generations: dict[str, int] = {}


# ===========================================================================
# Persistence
# ===========================================================================


def load(users_dir: pathlib.Path) -> None:
    """Load persisted handoffs. Recorders did not survive the restart.

    Args:
        users_dir: Directory containing each user's persisted handoff.
    """
    global _users_dir
    with _lock:
        _users_dir = users_dir
        _records.clear()
        for path in users_dir.glob("*/handoff.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(record, dict) and record.get("id"):
                record["recording_live"] = False
                _records[path.parent.name] = record


# ===========================================================================
# Recording
# ===========================================================================


def recording_dir(recording_id: str) -> pathlib.Path:
    return ffmpeg_command.get_transcode_dir() / f"netv_handoff_{recording_id}"


def recording_file(recording_id: str, filename: str) -> pathlib.Path | None:
    """Find a recording's playlist or segment.

    Args:
        recording_id: Identifier of a retained recording.
        filename: Single filename inside the recording directory.

    Returns:
        path: Existing file in a known recording, or None.
    """
    if not recording_id.isalnum() or pathlib.Path(filename).name != filename:
        return None
    with _lock:
        known = any(r.get("recording_id") == recording_id for r in _records.values())
    path = recording_dir(recording_id) / filename
    return path if known and path.is_file() else None


def cleanup_orphans() -> None:
    """Delete recordings no handoff refers to."""
    with _lock:
        known = {r.get("recording_id") for r in _records.values()}
    for directory in ffmpeg_command.get_transcode_dir().glob("netv_handoff_*"):
        if directory.name.removeprefix("netv_handoff_") not in known:
            shutil.rmtree(directory, ignore_errors=True)


def shutdown() -> None:
    """Stop recorders, keeping what they recorded."""
    with _lock:
        processes = list(_recorders.values())
        _recorders.clear()
    for process in processes:
        if process.poll() is None:
            process.terminate()


# ===========================================================================
# Watching
# ===========================================================================


def report_watch(
    username: str,
    stream_id: str,
    position: float,
    handoff_id: str = "",
    now: float | None = None,
) -> None:
    """Note the broadcast moment on screen while a live channel plays.

    Args:
        username: Owner of the watch.
        stream_id: Live channel being watched.
        position: Broadcast timestamp on screen.
        handoff_id: Existing handoff being resumed, if any.
        now: Current timestamp; defaults to the wall clock.
    """
    now = time.time() if now is None else now
    with _lock:
        watch = _watches.get(username)
        if watch and watch.stream_id == stream_id and watch.handoff_id == handoff_id:
            _watches[username] = replace(watch, position=position, last_seen=now)
        else:
            _watches[username] = _Watch(
                stream_id=stream_id,
                position=position,
                first_seen=now,
                last_seen=now,
                handoff_id=handoff_id,
            )


def stop_watch(
    username: str,
    stream_id: str,
    position: float,
    resolve: Resolver,
    release: Releaser,
    handoff_id: str = "",
    now: float | None = None,
    *,
    policy: HandoffPolicy | None = None,
) -> dict[str, Any] | None:
    """Playback stopped on a device: hand off where it was.

    Args:
        username: Owner of the watch.
        stream_id: Live channel being stopped.
        position: Broadcast timestamp on screen.
        resolve: Resolve the channel and its program.
        release: Release the user's upstream connection before recording.
        handoff_id: Existing handoff being resumed, if any.
        now: Current timestamp; defaults to the wall clock.
        policy: Retention and recording timing limits; defaults to HandoffPolicy().

    Returns:
        record: Created or updated handoff, or None when nothing can resume.
    """
    policy = policy or HandoffPolicy()
    now = time.time() if now is None else now
    with _lock:
        watch = _watches.get(username)
        if watch is None or watch.stream_id != stream_id or watch.handoff_id != handoff_id:
            return None
        del _watches[username]
    watch = replace(watch, position=position, last_seen=now)
    return _hand_off(username, watch, resolve, release, now, policy=policy)


# ===========================================================================
# Handoffs
# ===========================================================================


def current(username: str, now: float | None = None) -> dict[str, Any] | None:
    """Find the handoff to offer the user.

    Args:
        username: Owner of the handoff.
        now: Current timestamp; defaults to the wall clock.

    Returns:
        record: Unclaimed, unexpired handoff copy, or None.
    """
    now = time.time() if now is None else now
    with _lock:
        record = _records.get(username)
        if record is None or record.get("resumed") or record.get("expires", 0) <= now:
            return None
        return dict(record)


def claim(username: str, handoff_id: str) -> dict[str, Any] | None:
    """Resume a handoff: it stops being offered but its recording stays.

    Args:
        username: Owner of the handoff.
        handoff_id: Identifier of the offered handoff.

    Returns:
        record: Claimed handoff copy, or None when the identifier is stale.
    """
    with _lock:
        record = _records.get(username)
        if record is None or record["id"] != handoff_id:
            return None
        if not record.get("resumed"):
            record["resumed"] = True
            _save(username)
        return dict(record)


def cancel(username: str, keep: str = "") -> int:
    """Drop the user's handoff and its recording, unless it is ``keep``.

    Args:
        username: Owner of the handoff.
        keep: Handoff identifier to retain instead of cancelling.

    Returns:
        generation: User's cancel generation for rejecting stale recorder launches.
    """
    with _lock:
        record = _records.get(username)
        if record is not None and keep and record["id"] == keep:
            return _generations.get(username, 0)
        generation = _generations.get(username, 0) + 1
        _generations[username] = generation
        if record is not None:
            del _records[username]
            _save(username)
    if record is not None:
        log.info("Handoff for %s cancelled", username)
        if record.get("recording_id"):
            _stop_recorder(record["recording_id"])
    return generation


def sweep(
    resolve: Resolver,
    release: Releaser,
    now: float | None = None,
    *,
    policy: HandoffPolicy | None = None,
) -> None:
    """Hand off lapsed watches, finish recordings and drop expired handoffs.

    Args:
        resolve: Resolve a watched channel and its program.
        release: Release the upstream connection before recording.
        now: Current timestamp; defaults to the wall clock.
        policy: Watch expiry, retention and recording timing limits; defaults to
            HandoffPolicy().
    """
    policy = policy or HandoffPolicy()
    now = time.time() if now is None else now
    with _lock:
        lapsed = [
            (username, watch)
            for username, watch in _watches.items()
            if now - watch.last_seen > policy.watch_timeout_secs
        ]
        for username, _ in lapsed:
            del _watches[username]
        expired = [u for u, r in _records.items() if r.get("expires", 0) <= now]
        failed = []
        for username, record in _records.items():
            process = _recorders.get(record.get("recording_id", ""))
            if process is not None and process.poll() is not None:
                del _recorders[record["recording_id"]]
                record["recording_live"] = False
                _save(username)
                playlist = recording_dir(record["recording_id"]) / "stream.m3u8"
                if not playlist.exists():
                    failed.append(username)
    for username in expired + failed:
        cancel(username)
    for username, watch in lapsed:
        # The last report is where the device left off.
        _hand_off(username, watch, resolve, release, now, policy=policy)


def _save(username: str) -> None:
    if _users_dir is None:
        return
    path = _users_dir / username / "handoff.json"
    record = _records.get(username)
    try:
        if record is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write_json(path, record)
    except OSError as e:
        log.warning("Could not save handoff for %s: %s", username, e)


def _recorder_command(url: str, directory: pathlib.Path, seconds: float) -> list[str]:
    settings = ffmpeg_command.get_settings()
    url = ffmpeg_command.resolve_hls_master_playlist(url)
    media_info = ffmpeg_command.probe_media(url)[0] if settings.get("probe_live", True) else None
    # The recording is kept like a VOD transcode: every segment stays and
    # the playlist grows as an event, so any client can play it directly.
    cmd = ffmpeg_command.build_hls_ffmpeg_cmd(
        url,
        settings.get("transcode_hw", "software"),
        str(directory),
        is_vod=True,
        media_info=media_info,
        max_resolution=settings.get("max_resolution", "1080p"),
        quality=settings.get("quality", "high"),
        user_agent=ffmpeg_command.get_user_agent(),
        allow_upscale=False,
    )
    # Stop reading the input when the program ends; ffmpeg then closes the playlist.
    i = cmd.index("-i")
    cmd[i:i] = ["-t", str(int(seconds))]
    return cmd


def _start_recorder(url: str, recording_id: str, seconds: float) -> subprocess.Popen[bytes]:
    directory = recording_dir(recording_id)
    directory.mkdir(parents=True, exist_ok=True)
    cmd = _recorder_command(url, directory, seconds)
    log.info("Recording handoff %s for %.0fs", recording_id, seconds)
    with (directory / "ffmpeg.log").open("wb") as stderr:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )


def _stop_recorder(recording_id: str) -> None:
    """Stop and delete a recording. Call without holding the lock."""
    with _lock:
        process = _recorders.pop(recording_id, None)
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    shutil.rmtree(recording_dir(recording_id), ignore_errors=True)


def _hand_off(
    username: str,
    watch: _Watch,
    resolve: Resolver,
    release: Releaser,
    now: float,
    *,
    policy: HandoffPolicy,
) -> dict[str, Any] | None:
    position = min(watch.position, now)
    with _lock:
        existing = _records.get(username)
        resuming = bool(existing and watch.handoff_id and existing["id"] == watch.handoff_id)
        # A recording has nothing after its program; an archive rolls on into the next.
        finished = bool(
            existing
            and existing["mode"] == "recording"
            and existing["program_end"]
            and position >= existing["program_end"] - policy.program_end_margin_secs,
        )
        if resuming and existing and existing["mode"] == "recording" and not finished:
            # Stopped again within a recording: keep it and move the bookmark.
            existing.update(position=position, resumed=False, updated=now)
            _save(username)
            return dict(existing)
    if resuming and finished:
        cancel(username)
        return None
    if not resuming and watch.last_seen - watch.first_seen < policy.min_watch_secs:
        return None

    channel = resolve(watch.stream_id, position)
    if channel is None:
        cancel(username)
        return None
    if channel.program_end and position >= channel.program_end - policy.program_end_margin_secs:
        cancel(username)
        return None

    generation = cancel(username)
    record: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "stream_id": channel.stream_id,
        "channel_name": channel.name,
        "program_title": channel.program_title,
        "program_start": channel.program_start,
        "program_end": channel.program_end,
        "position": position,
        "resumed": False,
        "created": now,
        "updated": now,
    }
    if channel.catchup_days > 0:
        record.update(mode="catchup", expires=now + policy.keep_secs)
    else:
        end = channel.program_end if channel.program_end > now else 0.0
        seconds = min(end - now if end else policy.max_recording_secs, policy.max_recording_secs)
        recording_id = uuid.uuid4().hex
        release(username, channel.url)
        try:
            process = _start_recorder(channel.url, recording_id, seconds)
        except Exception as e:
            log.warning("Could not record handoff for %s: %s", username, e)
            shutil.rmtree(recording_dir(recording_id), ignore_errors=True)
            return None
        record.update(
            mode="recording",
            recording_id=recording_id,
            recording_start=now,
            recording_live=True,
            expires=now + seconds + policy.keep_secs,
        )
        with _lock:
            if _generations.get(username, 0) == generation:
                _recorders[recording_id] = process
                process = None
        if process is not None:
            # Something else started while the recorder was launching.
            process.terminate()
            shutil.rmtree(recording_dir(recording_id), ignore_errors=True)
            return None
    with _lock:
        if _generations.get(username, 0) != generation:
            return None
        _records[username] = record
        _save(username)
    log.info("Handoff for %s: %s at %.0f (%s)", username, channel.name, position, record["mode"])
    return dict(record)
