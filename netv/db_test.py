"""Tests for db.py - neTV's SQLite database and its per-thread connections."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import gc
import sqlite3
import threading
import warnings

import pytest

from netv import db


@pytest.fixture
def opened(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "sub" / "netv.db"
    db.open_database(path)
    yield path
    db.close()


def test_open_creates_the_directory_and_rows_read_by_name(opened: Path):
    row = db.connection().execute("SELECT 1 AS one").fetchone()
    assert row["one"] == 1
    assert opened.exists()
    assert db.connection() is db.connection()
    assert db.connection().execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_connection_of_exited_thread_is_closed(opened: Path):
    del opened
    worker = threading.Thread(target=db.connection)
    worker.start()
    worker.join()
    del worker
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        gc.collect()
    assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []


def test_close_closes_connections_of_every_thread(opened: Path):
    del opened
    conns: list[sqlite3.Connection] = []
    worker = threading.Thread(target=lambda: conns.append(db.connection()))
    worker.start()
    worker.join()
    conns.append(db.connection())
    db.close()
    for conn in conns:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="Database not initialized"):
        db.connection()


def test_reopen_reconnects_to_the_new_file(opened: Path, tmp_path: Path):
    db.connection().execute("CREATE TABLE t (x)")
    db.open_database(tmp_path / "other.db")
    tables = db.connection().execute("SELECT name FROM sqlite_master").fetchall()
    assert tables == []
    assert opened.exists()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
