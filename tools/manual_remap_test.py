"""Tests for manual_remap."""

import json
import pathlib
import runpy
import sqlite3
import sys
import time

import pytest

from tools import manual_remap

import cache


@pytest.fixture(autouse=True)
def _restore_cache_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # _add_arguments calls cache.initialize_paths(), which rebinds these globals.
    for name in ("CACHE_DIR", "SERVER_SETTINGS_FILE", "USERS_DIR", "LOGOS_DIR"):
        monkeypatch.setattr(cache, name, getattr(cache, name))


def _setup(data_dir: pathlib.Path) -> pathlib.Path:
    data_dir.mkdir(exist_ok=True)
    conn = sqlite3.connect(data_dir / "epg.db")
    conn.execute("CREATE TABLE channels (id TEXT, name TEXT)")
    conn.execute("CREATE TABLE programs (channel_id TEXT, stop_ts REAL)")
    conn.execute("INSERT INTO channels VALUES ('466831', 'KBCW')")
    conn.execute("INSERT INTO programs VALUES ('466831', ?)", (time.time() + 3600,))
    conn.execute("INSERT INTO channels VALUES ('464798', 'Cheddar')")
    conn.execute("INSERT INTO programs VALUES ('464798', 1.0)")
    conn.commit()
    conn.close()
    data = {
        "playlists": [
            {
                "channels": [
                    {"name": "US| CW SAN FRANCISCO HD"},
                    {"name": "US| CHEDDAR NEWS HD", "epg_channel_id": "old"},
                    {"name": "OTHER", "epg_channel_id": "keep"},
                ],
            },
            {},
        ],
    }
    (data_dir / "playlists.json").write_text(json.dumps(data))
    return data_dir


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["manual_remap.py", *argv])
    return manual_remap.main()


def test_dry_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = _setup(tmp_path / "c")
    before = (data_dir / "playlists.json").read_text()
    assert _run(monkeypatch, "--cache-dir", str(data_dir)) == 0
    out = capsys.readouterr().out
    assert "SET US| CW SAN FRANCISCO HD: (none) -> KBCW [466831] (1 upcoming)" in out
    assert "SKIP (no data): US| CHEDDAR NEWS HD -> [464798]" in out
    assert "1 channels to update" in out
    assert "Dry-run only; re-run with --apply." in out
    assert (data_dir / "playlists.json").read_text() == before


def test_defaults_to_netv_data_dir(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", _setup(tmp_path / "data"))
    assert _run(monkeypatch) == 0
    assert "1 channels to update" in capsys.readouterr().out


def test_apply(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = _setup(tmp_path / "c")
    assert _run(monkeypatch, "--cache-dir", str(data_dir), "--apply") == 0
    out = capsys.readouterr().out
    assert "Backup:" in out
    assert "Applied." in out
    chans = json.loads((data_dir / "playlists.json").read_text())["playlists"][0]["channels"]
    assert [c.get("epg_channel_id") for c in chans] == ["466831", "old", "keep"]
    backups = list(data_dir.glob("playlists.json.bak-*"))
    assert len(backups) == 1
    assert "466831" not in backups[0].read_text()
    assert not (data_dir / "playlists.json.tmp").exists()


def test_apply_with_nothing_to_change(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = _setup(tmp_path / "c")
    (data_dir / "playlists.json").write_text("{}")
    assert _run(monkeypatch, "--cache-dir", str(data_dir), "--apply") == 0
    out = capsys.readouterr().out
    assert "0 channels to update" in out
    assert "Applied." not in out
    assert "Dry-run" not in out
    assert not list(data_dir.glob("*.bak-*"))


def test_script_entry_point(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = _setup(tmp_path / "c")
    monkeypatch.setattr(sys, "argv", ["manual_remap.py", "--cache-dir", str(data_dir)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(manual_remap.__file__, run_name="__main__")
    assert exc.value.code == 0
    assert "1 channels to update" in capsys.readouterr().out


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
