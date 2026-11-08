"""Catchup: play past programs from an upstream archive.

Xtream sources flag streams that keep an archive with ``tv_archive`` and give
its length in days as ``tv_archive_duration``. An archived program is fetched
from a timeshift URL whose start time is in the upstream server's time zone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone, tzinfo
from urllib import parse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import logging
import math
import threading
import time

from treekle import MutablePlainTree


log = logging.getLogger(__name__)


_tz_lock = threading.Lock()
_tz_cache: dict[str, tuple[float, tzinfo]] = {}


def archive_days(stream: Mapping[str, MutablePlainTree], *, max_days: int = 14) -> int:
    """Return the archive duration, or zero when catchup is unavailable.

    Args:
        stream: Upstream stream metadata.
        max_days: Maximum supported archive duration.

    Returns:
        days: Nonnegative archive duration in days.
    """
    if stream.get("source_type") != "xtream" or str(stream.get("tv_archive")) != "1":
        return 0
    raw = stream.get("tv_archive_duration")
    if not isinstance(raw, (int, float, str)):
        return 0
    try:
        days = int(raw) if raw else 0
    except ValueError:
        return 0
    return max(0, min(days, max_days))


def can_play(
    stream: Mapping[str, MutablePlainTree],
    start: datetime,
    now: datetime | None = None,
) -> bool:
    """Check whether a program starting at ``start`` is still in the stream's archive.

    Args:
        stream: Upstream stream metadata.
        start: Program start instant.
        now: Reference instant; defaults to the current UTC time.

    Returns:
        playable: Whether the archive still holds the program.
    """
    days = archive_days(stream)
    if not days:
        return False
    if now is None:
        now = datetime.now(timezone.utc)
    return now - timedelta(days=days) <= start < now


def history(streams: Sequence[Mapping[str, MutablePlainTree]]) -> timedelta:
    """How far back EPG listings must be kept to cover every archive."""
    days = max((archive_days(s) for s in streams), default=0)
    return max(timedelta(hours=24), timedelta(days=days))


def server_timezone(
    source_id: str,
    fetch_server_info: Callable[[], Mapping[str, MutablePlainTree]],
    *,
    ttl_seconds: float = 6 * 3600,
) -> tzinfo:
    """Return the cached upstream time zone, falling back to UTC.

    Args:
        source_id: Cache identity for the upstream source.
        fetch_server_info: Callback returning upstream server metadata.
        ttl_seconds: Time zone cache lifetime.

    Returns:
        zone: Upstream time zone, or UTC when unavailable.
    """
    now = time.monotonic()
    with _tz_lock:
        cached = _tz_cache.get(source_id)
        if cached is not None and now - cached[0] < ttl_seconds:
            return cached[1]
    tz: tzinfo = timezone.utc
    try:
        info = fetch_server_info().get("server_info")
        name = info.get("timezone") if isinstance(info, dict) else None
        if isinstance(name, str) and name:
            tz = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as e:
        log.warning("Unknown upstream time zone for %s: %s", source_id, e)
    except Exception as e:
        log.warning("Could not read upstream time zone for %s: %s", source_id, e)
        return tz  # don't cache a transient failure.
    with _tz_lock:
        _tz_cache[source_id] = (now, tz)
    return tz


def duration_minutes(start: datetime, stop: datetime, *, max_minutes: int = 12 * 60) -> int:
    """Return the archive request length in whole minutes."""
    minutes = math.ceil((stop - start).total_seconds() / 60)
    return max(1, min(minutes, max_minutes))


def timeshift_url(
    stream: Mapping[str, MutablePlainTree],
    start: datetime,
    minutes: int,
    tz: tzinfo,
    ext: str = "ts",
) -> str:
    """Build a timeshift URL for the requested archive interval.

    Args:
        stream: Upstream source credentials and stream identity.
        start: Archive start instant.
        minutes: Requested duration in minutes.
        tz: Upstream server time zone.
        ext: Stream container extension.

    Returns:
        url: Credential-encoded upstream timeshift URL.
    """
    base = str(stream["source_url"]).rstrip("/")
    user = parse.quote(str(stream["source_username"]), safe="")
    pwd = parse.quote(str(stream["source_password"]), safe="")
    stream_id = str(stream["stream_id"])
    stream_id = stream_id.split("_")[-1] if "_" in stream_id else stream_id
    local_start = start.astimezone(tz).strftime("%Y-%m-%d:%H-%M")
    return f"{base}/timeshift/{user}/{pwd}/{minutes}/{local_start}/{stream_id}.{ext}"
