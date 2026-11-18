"""EPG storage and XMLTV parsing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import Final, Protocol, TypedDict, cast
from weakref import finalize
from xml.etree.ElementTree import Element

import logging
import os
import re
import sqlite3
import threading
import time
import zlib

import defusedxml.ElementTree as ET  # Safe XML parsing.

from netv.util import safe_urlopen


log = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================


@dataclass(kw_only=True, slots=True, frozen=True)
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


def insert_programs(programs: Iterable[tuple[str, str, float, float, str, str]]) -> int:
    """Bulk insert programs, streaming ``programs`` without materializing them.

    Args:
        programs: ``(channel_id, title, start_ts, stop_ts, desc, source_id)`` rows.

    Returns:
        count: Rows inserted.
    """
    conn = _get_conn()
    return conn.executemany(
        "INSERT INTO programs (channel_id, title, start_ts, stop_ts, desc, source_id) VALUES (?, ?, ?, ?, ?, ?)",  # pragma: no mutate[case-insensitive]
        programs,
    ).rowcount


def clear_source_from(source_id: str, channel_ids: Iterable[str], start_ts: float) -> None:
    """Delete a source's programs on ``channel_ids`` starting at or after ``start_ts``.

    The caller commits.

    Args:
        source_id: Source whose listings a fresh feed replaces.
        channel_ids: Channels the fresh feed lists programs for.
        start_ts: Start of the fresh feed's earliest program.
    """
    _get_conn().executemany(
        "DELETE FROM programs WHERE source_id = ? AND channel_id = ? AND start_ts >= ?",  # pragma: no mutate[case-insensitive]
        ((source_id, channel_id, start_ts) for channel_id in channel_ids),
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


_GZIP_MAGIC: Final = b"\x1f\x8b"


class EPGTooLargeError(ValueError):
    """An EPG feed exceeded the size limit, compressed or not."""


def max_epg_bytes(*, default: int = 1 << 30) -> int:
    """The EPG size limit: ``NETV_EPG_MAX_BYTES`` if set to a positive integer.

    Args:
        default: Limit on the decompressed feed size when the variable is unset or invalid.

    Returns:
        max_bytes: Largest EPG body to read, in bytes.
    """
    configured = os.environ.get("NETV_EPG_MAX_BYTES", "").strip()
    try:
        value = int(configured)
    except ValueError:
        return default
    return value if value > 0 else default


class _Readable(Protocol):
    def read(self, amt: int = ..., /) -> bytes: ...


def read_epg_body(resp: _Readable, max_bytes: int, *, chunk_size: int = 1 << 20) -> bytes:
    """Read a whole EPG response, gunzipping it if it is gzip, up to ``max_bytes``.

    Args:
        resp: Response to read from.
        max_bytes: Largest (decompressed) body allowed.
        chunk_size: Bytes requested from ``resp`` per read.

    Returns:
        body: The (decompressed) EPG body.

    Raises:
        EPGTooLargeError: The (decompressed) body is larger than ``max_bytes``.
        zlib.error: The body starts like gzip but is not valid gzip.
        EOFError: The gzip stream ends before its last member is complete.
    """
    return b"".join(iter_epg_body(resp, max_bytes, chunk_size=chunk_size))


def iter_epg_body(resp: _Readable, max_bytes: int, *, chunk_size: int = 1 << 20) -> Iterator[bytes]:
    """Yield an EPG response in chunks, gunzipping it if it is gzip, up to ``max_bytes``.

    Decompression is streamed and bounded, so a small compressed bomb fails
    with ``EPGTooLargeError`` instead of being expanded in memory. No chunk is
    larger than ``chunk_size``, compressed or not, so a parser reading the
    stream holds only a chunk of the feed at a time.

    Concatenated gzip members are read as one stream, as ``gzip.decompress``
    does; anything after the last member that is not gzip is ignored.

    Args:
        resp: Response to read from.
        max_bytes: Largest (decompressed) body allowed.
        chunk_size: Bytes requested from ``resp`` per read, and the most yielded at once.

    Yields:
        chunk: The next piece of the (decompressed) EPG body.

    Raises:
        EPGTooLargeError: The (decompressed) body is larger than ``max_bytes``.
        zlib.error: The body starts like gzip but is not valid gzip.
        EOFError: The gzip stream ends before its last member is complete.
    """
    size = 0
    data = resp.read(chunk_size)
    if data[:2] != _GZIP_MAGIC:
        while data:
            size += len(data)
            _checked(size, max_bytes)
            yield data
            data = resp.read(chunk_size)
        return

    gunzip: zlib._Decompress | None = None  # None between gzip members.
    while True:
        if gunzip is None:
            if len(data) < len(_GZIP_MAGIC):
                more = resp.read(chunk_size)
                if more:
                    data += more
                    continue
                return
            if data[:2] != _GZIP_MAGIC:
                return
            gunzip = zlib.decompressobj(16 + zlib.MAX_WBITS)
        if data:
            # max_length bounds each step's output to a chunk and to the budget. It
            # is at least 1, because _checked raises once size exceeds max_bytes.
            chunk = gunzip.decompress(data, min(chunk_size, max_bytes + 1 - size))
            size += len(chunk)
            _checked(size, max_bytes)
            if chunk:
                yield chunk
            if gunzip.eof:
                data, gunzip = gunzip.unused_data, None
                continue
            # Input the chunk bound left undecompressed is decompressed before reading on.
            data = gunzip.unconsumed_tail
            if data:
                continue
        data = resp.read(chunk_size)
        if not data:
            raise EOFError("EPG gzip stream ended before the end of the gzip member")


class _ChunkReader:
    """A file-like ``read`` over byte chunks, so ``iterparse`` can parse a stream.

    Like a raw stream, a read may return fewer bytes than asked for.
    """

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self._chunk = b""
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        """Return up to ``size`` bytes of the current chunk, or all of it.

        Args:
            size: Most bytes to return; negative means the rest of the chunk.

        Returns:
            data: The next bytes, or ``b""`` once the chunks run out.
        """
        while self._pos >= len(self._chunk):
            chunk = next(self._chunks, None)
            if chunk is None:
                return b""
            self._chunk, self._pos = chunk, 0
        end = len(self._chunk) if size < 0 else self._pos + size
        out = self._chunk[self._pos : end]
        self._pos += len(out)
        return out


ProgramRow = tuple[str, str, float, float, str, str]


@dataclass(kw_only=True, slots=True)
class _ParsedEpg:
    """What a feed adds to the database, gathered before any of it is written."""

    channels: list[tuple[str, str, str | None]] = field(default_factory=list)  # `id`, name, icon.
    programs: list[ProgramRow] = field(default_factory=list)
    program_channel_ids: set[str] = field(default_factory=set)

    def add(self, element: Element, source_id: str) -> None:
        """Record one top-level XMLTV element; anything but a channel or programme is ignored.

        Args:
            element: A child of the feed's root element.
            source_id: Source the feed belongs to.
        """
        if element.tag == "channel":
            ch_id = element.get("id", "")
            name = _element_text(element.find("display-name"), default=ch_id)
            icon_el = element.find("icon")
            icon = None if icon_el is None else icon_el.get("src", "")
            self.channels.append((ch_id, name, icon))
        elif element.tag == "programme":
            row = _program_row(element, source_id, self.program_channel_ids)
            if row is not None:
                self.programs.append(row)


def fetch_epg(
    epg_url: str,
    cache_dir: Path,
    timeout: int = 120,
    source_id: str = "",
    user_agent: str | None = None,
    *,
    chunk_size: int = 1 << 20,
) -> int:
    """Fetch and parse XMLTV EPG data directly into sqlite.

    The feed is parsed as it streams in, and rows are written once it has all
    parsed.

    Args:
        epg_url: URL of the XMLTV EPG feed
        cache_dir: Directory for debug files if parsing fails
        timeout: Request timeout in seconds
        source_id: Source identifier for multi-source support
        user_agent: User-Agent header to send. If None, uses default.
        chunk_size: Bytes read from the response, and parsed, at a time.

    Returns:
        program_count: Number of programs inserted.
    """
    # Nothing is written until the whole feed has parsed: feeds load in parallel,
    # and a write transaction held through a download would lock out the others.
    try:
        with safe_urlopen(epg_url, timeout=timeout, user_agent=user_agent) as resp:
            chunks = iter_epg_body(resp, max_epg_bytes(), chunk_size=chunk_size)
            parsed = _parse_epg_stream(_ChunkReader(chunks), source_id)
    except ET.ParseError as e:
        log.warning("EPG parse failed (%s), attempting sanitization...", e)
        parsed = _parse_sanitized_epg(epg_url, cache_dir, timeout, source_id, user_agent)

    channel_ids: set[str] = set()
    for ch_id, name, icon in parsed.channels:
        channel_ids.add(ch_id)
        insert_channel(ch_id, name, source_id)
        if icon is not None:
            insert_icon(ch_id, icon)
    # Replace this feed's channels from its first programme onward, so a refetch does not
    # duplicate them. Earlier listings stay for catch-up. Other channels are kept, since a
    # source may load several guides. Feeds without a source all share "", so skip them.
    if source_id and parsed.programs:
        start_ts = min(row[2] for row in parsed.programs)
        clear_source_from(source_id, parsed.program_channel_ids, start_ts)
    program_count = insert_programs(parsed.programs)
    commit()
    log.debug(
        "EPG parsed: %d channels, %d unique program channel IDs, %d programs",
        len(channel_ids),
        len(parsed.program_channel_ids),
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


# A programme without both times is skipped: placing it at "now" would show it as airing
# on every refresh.
def _program_row(prog: Element, source_id: str, channel_ids: set[str]) -> ProgramRow | None:
    """Return the insert row for a well-formed ``programme``, adding its channel to ``channel_ids``."""
    ch_id = prog.get("channel", "")
    channel_ids.add(ch_id)
    start_str = prog.get("start")
    stop_str = prog.get("stop")
    if start_str is None or stop_str is None:
        return None
    try:
        start = _parse_epg_time(start_str)
        stop = _parse_epg_time(stop_str)
    except ValueError:
        return None
    title = _element_text(prog.find("title"), default="Unknown")
    desc = _element_text(prog.find("desc"), default="")
    return ch_id, title, start.timestamp(), stop.timestamp(), desc, source_id


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


def _checked(size: int, max_bytes: int) -> None:
    if size > max_bytes:
        raise EPGTooLargeError(
            f"EPG feed is larger than {max_bytes} bytes (set NETV_EPG_MAX_BYTES to raise the limit)",
        )


# Only the rows are kept, never the document or its tree, so memory follows the size of
# the guide data rather than a multiple of the feed.
def _parse_epg_stream(source: _ChunkReader, source_id: str) -> _ParsedEpg:
    """Parse XMLTV as it arrives, dropping each element once it is recorded."""
    parsed = _ParsedEpg()
    root: Element | None = None
    depth = 0
    events = cast("Iterator[tuple[str, Element]]", ET.iterparse(source, events=("start", "end")))
    for event, element in events:
        if event == "start":
            depth += 1
            if root is None:
                root = element
            continue
        depth -= 1
        if depth == 1 and root is not None:
            parsed.add(element, source_id)
            # The finished element is root's only child, so clearing root frees it.
            root.clear()
    return parsed


# The streamed copy is gone by the time parsing fails, and the repair works on the full
# text, so this rare path holds the feed in memory as before.
def _parse_sanitized_epg(
    epg_url: str,
    cache_dir: Path,
    timeout: int,
    source_id: str,
    user_agent: str | None,
) -> _ParsedEpg:
    """Fetch the feed again, whole, and parse what ``_sanitize_epg_xml`` can salvage."""
    with safe_urlopen(epg_url, timeout=timeout, user_agent=user_agent) as resp:
        xml_str = read_epg_body(resp, max_epg_bytes()).decode()
    debug_file = cache_dir / f"epg_debug_{int(time.time())}.xml"
    debug_file.write_text(xml_str)
    try:
        root = ET.fromstring(_sanitize_epg_xml(xml_str))
        log.info("Sanitized EPG parsed successfully")
    except ET.ParseError as e:
        log.error("Sanitized EPG also failed: %s", e)
        raise
    parsed = _ParsedEpg()
    for element in root:
        parsed.add(element, source_id)
    return parsed
