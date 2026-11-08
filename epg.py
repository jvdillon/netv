"""EPG storage and XMLTV parsing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import TypedDict, cast
from weakref import finalize
from xml.etree.ElementTree import Element

import contextlib
import gzip
import logging
import re
import sqlite3
import threading
import time

import defusedxml.ElementTree as ET  # Safe XML parsing.

from util import safe_urlopen


log = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================


@dataclass(kw_only=True, slots=True)
class Program:
    channel_id: str
    title: str
    start: datetime
    stop: datetime
    desc: str = ""
    source_id: str = ""


class ChannelMatch(TypedDict):
    """An EPG channel found by ``search_channels``."""

    id: str
    name: str
    has_listings: bool


# =============================================================================
# SQLite Storage
# =============================================================================


class _ThreadState(threading.local):
    """Per-thread SQLite connection and the finalizer that closes it."""

    conn: sqlite3.Connection | None = None
    closer: finalize[[], Thread] | None = None


_DB_PATH: Path | None = None
_local: _ThreadState | None = None
_closers: list[finalize[[], Thread]] = []


def init(cache_dir: Path) -> None:
    """Initialize EPG database, closing connections to any earlier one.

    Args:
        cache_dir: Directory containing the persistent guide database.
    """
    global _DB_PATH, _local
    close()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _DB_PATH = cache_dir / "epg.db"
    if _local is None:
        _local = _ThreadState()
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS channels (
            id TEXT PRIMARY KEY,
            name TEXT,
            source_id TEXT
        );
        CREATE TABLE IF NOT EXISTS icons (
            channel_id TEXT PRIMARY KEY,
            url TEXT
        );
        CREATE TABLE IF NOT EXISTS programs (
            id INTEGER PRIMARY KEY,
            channel_id TEXT,
            title TEXT,
            start_ts REAL,
            stop_ts REAL,
            desc TEXT,
            source_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_programs_channel_time
            ON programs(channel_id, start_ts, stop_ts);
        CREATE INDEX IF NOT EXISTS idx_programs_time
            ON programs(start_ts);
    """)
    conn.commit()


def close() -> None:
    """Close every thread's connection and forget the database path."""
    global _DB_PATH
    _DB_PATH = None
    while _closers:
        _closers.pop()()


def clear() -> None:
    """Clear all EPG data."""
    conn = _get_conn()
    conn.executescript("DELETE FROM programs; DELETE FROM channels; DELETE FROM icons;")
    conn.commit()


def clear_source(source_id: str) -> None:
    """Clear EPG data for a specific source."""
    conn = _get_conn()
    conn.execute("DELETE FROM programs WHERE source_id = ?", (source_id,))
    conn.execute("DELETE FROM channels WHERE source_id = ?", (source_id,))
    conn.commit()


def insert_channel(channel_id: str, name: str, source_id: str) -> None:
    """Insert or update a channel."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO channels (id, name, source_id) VALUES (?, ?, ?)",
        (channel_id, name, source_id),
    )


def insert_icon(channel_id: str, url: str) -> None:
    """Insert or update a channel icon."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO icons (channel_id, url) VALUES (?, ?)",
        (channel_id, url),
    )


def insert_programs(programs: list[tuple[str, str, float, float, str, str]]) -> None:
    """Bulk insert programs. Each tuple: (channel_id, title, start_ts, stop_ts, desc, source_id)."""
    conn = _get_conn()
    conn.executemany(
        "INSERT INTO programs (channel_id, title, start_ts, stop_ts, desc, source_id) VALUES (?, ?, ?, ?, ?, ?)",
        programs,
    )


def commit() -> None:
    """Commit current transaction."""
    _get_conn().commit()


def get_icon(channel_id: str) -> str:
    """Get icon URL for a channel."""
    conn = _get_conn()
    row = conn.execute("SELECT url FROM icons WHERE channel_id = ?", (channel_id,)).fetchone()
    # Typeshed types sqlite rows as Any; insert_icon stores text.
    return cast(str, row["url"]) if row else ""


def get_programs_in_range(
    channel_id: str,
    start: datetime,
    end: datetime,
    preferred_source_id: str = "",
) -> list[Program]:
    """Get programs for a channel within a time range.

    Args:
        channel_id: XMLTV channel identifier.
        start: Beginning of the requested interval.
        end: End of the requested interval.
        preferred_source_id: Source preferred when listings overlap.

    Returns:
        programs: Matching programs ordered by start time.
    """
    conn = _get_conn()
    start_ts = start.timestamp()
    end_ts = end.timestamp()

    rows = conn.execute(
        """
        SELECT channel_id, title, start_ts, stop_ts, desc, source_id
        FROM programs
        WHERE channel_id = ? AND stop_ts > ? AND start_ts < ?
        ORDER BY start_ts
        """,
        (channel_id, start_ts, end_ts),
    ).fetchall()

    programs = [
        Program(
            channel_id=row["channel_id"],
            title=row["title"],
            start=datetime.fromtimestamp(row["start_ts"], tz=timezone.utc),
            stop=datetime.fromtimestamp(row["stop_ts"], tz=timezone.utc),
            desc=row["desc"] or "",
            source_id=row["source_id"] or "",
        )
        for row in rows
    ]

    if not preferred_source_id or len(programs) <= 1:
        return programs

    # Deduplicate overlapping programs, preferring the preferred source.
    result: list[Program] = []
    for p in programs:
        dominated = False
        for i, existing in enumerate(result):
            if p.start < existing.stop and p.stop > existing.start:
                if p.source_id == preferred_source_id and existing.source_id != preferred_source_id:
                    result[i] = p
                dominated = True
                break
        if not dominated:
            result.append(p)
    return sorted(result, key=lambda p: p.start)


def get_programs_batch(
    channel_ids: list[str],
    start: datetime,
    end: datetime,
    preferred_sources: dict[str, str] | None = None,
    *,
    chunk_size: int = 500,
) -> dict[str, list[Program]]:
    """Get programs for multiple channels in a single query.

    Args:
        channel_ids: List of EPG channel IDs to query
        start: Start of time window
        end: End of time window
        preferred_sources: Optional dict mapping channel_id -> preferred source_id
            for deduplication. If provided, overlapping programs from the preferred
            source will be kept over programs from other sources.
        chunk_size: Channel IDs per query, below SQLite's parameter limit.

    Returns:
        programs: Matching programs grouped by channel identifier.
    """
    if not channel_ids:
        return {}
    conn = _get_conn()
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    result: dict[str, list[Program]] = {ch: [] for ch in channel_ids}

    # Process in chunks to avoid huge IN clauses.
    for i in range(0, len(channel_ids), chunk_size):
        chunk = channel_ids[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""
            SELECT channel_id, title, start_ts, stop_ts, desc, source_id
            FROM programs
            WHERE channel_id IN ({placeholders}) AND stop_ts > ? AND start_ts < ?
            ORDER BY channel_id, start_ts
            """,
            [*chunk, start_ts, end_ts],
        ).fetchall()
        for row in rows:
            result[row["channel_id"]].append(
                Program(
                    channel_id=row["channel_id"],
                    title=row["title"],
                    start=datetime.fromtimestamp(row["start_ts"], tz=timezone.utc),
                    stop=datetime.fromtimestamp(row["stop_ts"], tz=timezone.utc),
                    desc=row["desc"] or "",
                    source_id=row["source_id"] or "",
                ),
            )

    # Deduplicate overlapping programs if preferred_sources provided.
    if preferred_sources is not None:
        for ch_id in result:
            if ch_id in preferred_sources and result[ch_id]:
                result[ch_id] = _dedupe_programs(result[ch_id], preferred_sources[ch_id])

    channels_with_programs = sum(1 for progs in result.values() if progs)
    log.debug(
        "EPG batch query: requested %d channel IDs, found programs for %d",
        len(channel_ids),
        channels_with_programs,
    )
    return result


def get_icons_batch(channel_ids: list[str], *, chunk_size: int = 500) -> dict[str, str]:
    """Get icons for multiple channels in batched queries.

    Args:
        channel_ids: XMLTV channel identifiers.
        chunk_size: Channel IDs per query, below SQLite's parameter limit.

    Returns:
        icons: Icon URLs indexed by channel identifier.
    """
    if not channel_ids:
        return {}
    conn = _get_conn()
    result: dict[str, str] = {}
    for i in range(0, len(channel_ids), chunk_size):
        chunk = channel_ids[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT channel_id, url FROM icons WHERE channel_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            result[row["channel_id"]] = row["url"]
    return result


def has_programs() -> bool:
    """Check if there are any programs in the database."""
    conn = _get_conn()
    row = conn.execute("SELECT 1 FROM programs LIMIT 1").fetchone()
    return row is not None


def has_programs_after(when: datetime) -> bool:
    """Check whether programs extend beyond an instant.

    Args:
        when: Exclusive lower bound on program end times.

    Returns:
        available: Whether a program ends after the instant.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM programs WHERE stop_ts > ? LIMIT 1",
        (when.timestamp(),),
    ).fetchone()
    return row is not None


def search_channels(terms: list[str], limit: int = 20) -> list[ChannelMatch]:
    """Find EPG channels whose id or name contains every term.

    Channels with upcoming listings come first.

    Args:
        terms: Case-insensitive substrings required in channel IDs or names.
        limit: Maximum number of results.

    Returns:
        channels: Matching IDs, names, and upcoming-listing flags.
    """
    terms = [t for t in terms if t]
    if not terms:
        return []
    where = " AND ".join("(c.id LIKE ? OR c.name LIKE ?)" for _ in terms)
    params: list[str] = []
    for t in terms:
        like = f"%{t}%"
        params += [like, like]
    now = datetime.now(timezone.utc).timestamp()
    conn = _get_conn()
    rows = conn.execute(
        f"""
        SELECT c.id, c.name, EXISTS(
            SELECT 1 FROM programs p WHERE p.channel_id = c.id AND p.stop_ts > ?
        ) AS has_listings
        FROM channels c WHERE {where}
        ORDER BY has_listings DESC, length(c.id)
        LIMIT ?
        """,
        [now, *params, limit],
    ).fetchall()
    return [
        {"id": r["id"], "name": r["name"], "has_listings": bool(r["has_listings"])} for r in rows
    ]


def get_program_count() -> int:
    """Get total program count."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
    return cast(int, row[0]) if row else 0


def get_channel_count() -> int:
    """Get total channel count."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) FROM channels").fetchone()
    return cast(int, row[0]) if row else 0


def prune_old_programs(before: datetime) -> int:
    """Delete programs that ended before the given time. Returns count deleted."""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM programs WHERE stop_ts < ?", (before.timestamp(),))
    conn.commit()
    return cursor.rowcount


# =============================================================================
# XMLTV Parsing
# =============================================================================


def fetch_epg(
    epg_url: str,
    cache_dir: Path,
    timeout: int = 120,
    source_id: str = "",
    user_agent: str | None = None,
) -> int:
    """Fetch and parse XMLTV EPG data directly into sqlite.

    Args:
        epg_url: URL of the XMLTV EPG feed
        cache_dir: Directory for debug files if parsing fails
        timeout: Request timeout in seconds
        source_id: Source identifier for multi-source support
        user_agent: User-Agent header to send. If None, uses default.

    Returns:
        program_count: Number of programs inserted.
    """
    with safe_urlopen(epg_url, timeout=timeout, user_agent=user_agent) as resp:
        content = resp.read()
        with contextlib.suppress(Exception):
            content = gzip.decompress(content)
        xml_str = content.decode("utf-8")

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        debug_file = cache_dir / f"epg_debug_{int(time.time())}.xml"
        debug_file.write_text(xml_str)
        log.warning("EPG parse failed (%s), attempting sanitization...", e)
        try:
            sanitized = _sanitize_epg_xml(xml_str)
            root = ET.fromstring(sanitized)
            log.info("Sanitized EPG parsed successfully")
        except ET.ParseError as e2:
            log.error("Sanitized EPG also failed: %s", e2)
            raise

    # Parse channels directly into sqlite.
    channel_ids: set[str] = set()
    for ch in root.findall("channel"):
        ch_id = ch.get("id", "")
        channel_ids.add(ch_id)
        name = _element_text(ch.find("display-name"), default=ch_id)
        insert_channel(ch_id, name, source_id)
        icon_el = ch.find("icon")
        if icon_el is not None:
            insert_icon(ch_id, icon_el.get("src", ""))

    # Parse programs in batches.
    batch: list[tuple[str, str, float, float, str, str]] = []
    batch_size = 10_000
    program_count = 0
    program_channel_ids: set[str] = set()

    for prog in root.findall("programme"):
        ch_id = prog.get("channel", "")
        program_channel_ids.add(ch_id)
        start_str = prog.get("start", "")
        stop_str = prog.get("stop", "")

        title = _element_text(prog.find("title"), default="Unknown")
        desc = _element_text(prog.find("desc"), default="")

        try:
            start = _parse_epg_time(start_str)
            stop = _parse_epg_time(stop_str)
        except Exception:
            continue

        batch.append((ch_id, title, start.timestamp(), stop.timestamp(), desc, source_id))
        program_count += 1

        if len(batch) >= batch_size:
            insert_programs(batch)
            batch.clear()

    if batch:
        insert_programs(batch)

    commit()
    log.debug(
        "EPG parsed: %d channels, %d unique program channel IDs, %d programs",
        len(channel_ids),
        len(program_channel_ids),
        program_count,
    )
    return program_count


def _get_conn() -> sqlite3.Connection:
    """Get thread-local database connection."""
    if _local is None or _DB_PATH is None:
        raise RuntimeError("EPG database not initialized")
    if _local.conn is None or _local.closer is None or not _local.closer.alive:
        # A thread's locals vanish when it exits, dropping the connection
        # unclosed. The finalizer closes it once the thread object is gone, or
        # at exit; it may run on another thread, hence check_same_thread=False.
        conn = sqlite3.connect(_DB_PATH, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        closer = finalize(threading.current_thread(), conn.close)
        _closers.append(closer)
        _local.conn = conn
        _local.closer = closer
    return _local.conn


def _dedupe_programs(programs: list[Program], preferred_source_id: str) -> list[Program]:
    """Deduplicate overlapping programs, preferring the preferred source."""
    if not preferred_source_id or len(programs) <= 1:
        return programs
    result: list[Program] = []
    for p in programs:
        dominated = False
        for i, existing in enumerate(result):
            # Check for overlap.
            if p.start < existing.stop and p.stop > existing.start:
                # Prefer the preferred source.
                if p.source_id == preferred_source_id and existing.source_id != preferred_source_id:
                    result[i] = p
                dominated = True
                break
        if not dominated:
            result.append(p)
    return sorted(result, key=lambda p: p.start)


def _parse_epg_time(s: str) -> datetime:
    """Parse XMLTV time format: 20241130120000 +0000 or 20241130120000+0530."""
    s = s.replace(" ", "")
    if len(s) >= 14:
        dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S")
        if len(s) > 14:
            tz_str = s[14:]
            sign = -1 if tz_str[0] == "-" else 1
            tz_hours = int(tz_str[1:3]) if len(tz_str) >= 3 else 0
            tz_mins = int(tz_str[3:5]) if len(tz_str) >= 5 else 0
            offset = timedelta(hours=tz_hours, minutes=tz_mins)
            dt = dt.replace(tzinfo=timezone(sign * offset))
        return dt
    return datetime.now(timezone.utc)


def _element_text(element: Element | None, *, default: str) -> str:
    """Return an XML element's text, or ``default`` when it is absent or empty."""
    text = None if element is None else element.text
    return default if text is None or text == "" else text


def _sanitize_epg_xml(xml_str: str) -> str:
    """Try to fix corrupted EPG XML by extracting valid elements."""
    channels = [
        m.group(0) for m in re.finditer(r"<channel\s+[^>]*>.*?</channel>", xml_str, re.DOTALL)
    ]
    programmes = [
        m.group(0)
        for m in re.finditer(
            r'<programme\s+start="[^"<>]+"\s+stop="[^"<>]+"\s+channel="[^"<>]+"[^>]*>.*?</programme>',
            xml_str,
            re.DOTALL,
        )
    ]
    log.info("Sanitized EPG: extracted %d channels, %d programmes", len(channels), len(programmes))
    return '<?xml version="1.0"?>\n<tv>\n' + "\n".join(channels) + "\n".join(programmes) + "\n</tv>"
