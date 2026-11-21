"""neTV's SQLite database, one connection per thread.

The guide and the IMDb index keep their tables here; each module creates its own
schema once ``open_database`` has chosen the file.
"""

from __future__ import annotations

from pathlib import Path
from threading import Thread
from weakref import finalize

import sqlite3
import threading


class _ThreadState(threading.local):
    """Per-thread SQLite connection and the finalizer that closes it."""

    conn: sqlite3.Connection | None = None
    closer: finalize[[], Thread] | None = None


_path: Path | None = None
_local: _ThreadState | None = None
_closers: list[finalize[[], Thread]] = []


def open_database(path: Path) -> None:
    """Use the database at ``path``, closing connections to any earlier one.

    Args:
        path: SQLite file, created with its directory when missing.
    """
    global _path, _local
    close()
    path.parent.mkdir(parents=True, exist_ok=True)
    _path = path
    if _local is None:
        _local = _ThreadState()


def close() -> None:
    """Close every thread's connection and forget the database path."""
    global _path
    _path = None
    while _closers:
        _closers.pop()()


def connection() -> sqlite3.Connection:
    """Return this thread's connection to the open database.

    Returns:
        conn: Connection opened by this thread, rows readable by column name.
    """
    if _local is None or _path is None:
        raise RuntimeError("Database not initialized")
    if _local.conn is None or _local.closer is None or not _local.closer.alive:
        # A thread's locals vanish when it exits, dropping the connection
        # unclosed. The finalizer closes it once the thread object is gone, or
        # at exit; it may run on another thread, hence check_same_thread=False.
        conn = sqlite3.connect(_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        closer = finalize(threading.current_thread(), conn.close)
        _closers.append(closer)
        _local.conn = conn
        _local.closer = closer
    return _local.conn
