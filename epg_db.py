"""SQLite-backed EPG storage for memory efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import logging
import sqlite3
import threading

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Program:
    channel_id: str
    title: str
    start: datetime
    stop: datetime
    desc: str = ""
    source_id: str = ""


_DB_PATH: Path | None = None
_local = threading.local()


def init(cache_dir: Path) -> None:
    """Initialize EPG database."""
    global _DB_PATH
    _DB_PATH = cache_dir / "epg.db"
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


def _get_conn() -> sqlite3.Connection:
    """Get thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        if _DB_PATH is None:
            raise RuntimeError("EPG database not initialized")
        _local.conn = sqlite3.connect(_DB_PATH, timeout=30.0)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


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
    return row["url"] if row else ""


def get_programs_in_range(
    channel_id: str,
    start: datetime,
    end: datetime,
    preferred_source_id: str = "",
) -> list[Program]:
    """Get programs for a channel within a time range."""
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
            start=datetime.fromtimestamp(row["start_ts"], tz=UTC),
            stop=datetime.fromtimestamp(row["stop_ts"], tz=UTC),
            desc=row["desc"] or "",
            source_id=row["source_id"] or "",
        )
        for row in rows
    ]

    if not preferred_source_id or len(programs) <= 1:
        return programs

    # Deduplicate overlapping programs, preferring the preferred source
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


_MAX_IN_CLAUSE = 500  # SQLite limit is 999, stay well below


def get_programs_batch(
    channel_ids: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, list[Program]]:
    """Get programs for multiple channels in a single query."""
    if not channel_ids:
        return {}
    conn = _get_conn()
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    result: dict[str, list[Program]] = {ch: [] for ch in channel_ids}

    # Process in chunks to avoid huge IN clauses
    for i in range(0, len(channel_ids), _MAX_IN_CLAUSE):
        chunk = channel_ids[i : i + _MAX_IN_CLAUSE]
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
                    start=datetime.fromtimestamp(row["start_ts"], tz=UTC),
                    stop=datetime.fromtimestamp(row["stop_ts"], tz=UTC),
                    desc=row["desc"] or "",
                    source_id=row["source_id"] or "",
                )
            )
    channels_with_programs = sum(1 for progs in result.values() if progs)
    log.debug(
        "EPG batch query: requested %d channel IDs, found programs for %d",
        len(channel_ids),
        channels_with_programs,
    )
    return result


def get_icons_batch(channel_ids: list[str]) -> dict[str, str]:
    """Get icons for multiple channels in a single query."""
    if not channel_ids:
        return {}
    conn = _get_conn()
    result: dict[str, str] = {}
    for i in range(0, len(channel_ids), _MAX_IN_CLAUSE):
        chunk = channel_ids[i : i + _MAX_IN_CLAUSE]
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


def get_program_count() -> int:
    """Get total program count."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) FROM programs").fetchone()
    return row[0] if row else 0


def get_channel_count() -> int:
    """Get total channel count."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) FROM channels").fetchone()
    return row[0] if row else 0


def prune_old_programs(before: datetime) -> int:
    """Delete programs that ended before the given time. Returns count deleted."""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM programs WHERE stop_ts < ?", (before.timestamp(),))
    conn.commit()
    return cursor.rowcount
