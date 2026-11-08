"""Tests for debug_remap."""

import pathlib
import runpy
import sqlite3
import sys
import time

import pytest

from tools import debug_remap


def _make_cache(root: pathlib.Path) -> None:
    (root / "cache").mkdir()
    conn = sqlite3.connect(root / "cache" / "epg.db")
    conn.execute("CREATE TABLE channels (id TEXT, name TEXT, source_id TEXT)")
    conn.execute("CREATE TABLE programs (channel_id TEXT, stop_ts REAL)")
    future = time.time() + 3600
    for i in range(10):
        cid = f"Syfy{i}.us"
        conn.execute("INSERT INTO channels VALUES (?, ?, NULL)", (cid, f"US - SYFY {i}"))
        conn.execute("INSERT INTO programs VALUES (?, ?)", (cid, future))
    conn.execute("INSERT INTO channels VALUES ('Nbc.us', 'NBC', NULL)")
    conn.execute("INSERT INTO programs VALUES ('Nbc.us', ?)", (future,))
    conn.commit()
    conn.close()


def test_main_prints_variants_and_top_eight(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_cache(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["debug_remap.py", "US| SYFY HD", "Syfy.us"])
    assert debug_remap.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        "VARIANT {'country': 'US', 'tokens': ['SYFY'], 'from_name': True}",
        "VARIANT {'country': 'US', 'tokens': ['SYFY'], 'from_name': False}",
    ]
    scores = lines[2:]
    assert len(scores) == 8
    assert all("Syfy" in line for line in scores)
    assert "from_name=True" in scores[0]
    assert "nc='US' ic='US' shift=False" in scores[0]


def test_main_without_current_id(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_cache(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["debug_remap.py", "NBC"])
    assert debug_remap.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "VARIANT {'country': '', 'tokens': ['NBC'], 'from_name': True}"
    assert len(lines) == 2
    assert "Nbc.us" in lines[1]


def test_script_entry_point(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_cache(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["debug_remap.py", "ZZZ"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(debug_remap.__file__, run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("VARIANT")


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
