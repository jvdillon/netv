"""Tests for manual_remap."""

import json
import pathlib
import runpy
import sqlite3
import sys
import time

import pytest

from tools import manual_remap


def _setup(cache: pathlib.Path) -> None:
    conn = sqlite3.connect(cache / "epg.db")
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
    (cache / "playlists.json").write_text(json.dumps(data))


@pytest.fixture
def cache(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    real = pathlib.Path

    class _Path:
        def __new__(cls, p: str) -> pathlib.Path:
            assert p == "/home/ankur/netv/cache"
            return real(tmp_path)

    monkeypatch.setattr(manual_remap.pathlib, "Path", _Path)
    _setup(tmp_path)
    return tmp_path


def test_dry_run(
    cache: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = (cache / "playlists.json").read_text()
    monkeypatch.setattr(sys, "argv", ["manual_remap.py"])
    assert manual_remap.main() == 0
    out = capsys.readouterr().out
    assert "SET US| CW SAN FRANCISCO HD: (none) -> KBCW [466831] (1 upcoming)" in out
    assert "SKIP (no data): US| CHEDDAR NEWS HD -> [464798]" in out
    assert "1 channels to update" in out
    assert "Dry-run only; re-run with --apply." in out
    assert (cache / "playlists.json").read_text() == before


def test_apply(
    cache: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["manual_remap.py", "--apply"])
    assert manual_remap.main() == 0
    out = capsys.readouterr().out
    assert "Backup:" in out
    assert "Applied." in out
    chans = json.loads((cache / "playlists.json").read_text())["playlists"][0]["channels"]
    assert [c.get("epg_channel_id") for c in chans] == ["466831", "old", "keep"]
    backups = list(cache.glob("playlists.json.bak-*"))
    assert len(backups) == 1
    assert "466831" not in backups[0].read_text()


def test_apply_with_nothing_to_change(
    cache: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (cache / "playlists.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["manual_remap.py", "--apply"])
    assert manual_remap.main() == 0
    out = capsys.readouterr().out
    assert "0 channels to update" in out
    assert "Applied." not in out
    assert "Dry-run" not in out
    assert not list(cache.glob("*.bak-*"))


def test_script_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[object] = []

    def fake_connect(path: object) -> None:
        opened.append(path)
        raise sqlite3.OperationalError("no db")

    monkeypatch.setattr(sqlite3, "connect", fake_connect)
    monkeypatch.setattr(sys, "argv", ["manual_remap.py"])
    with pytest.raises(sqlite3.OperationalError, match="no db"):
        runpy.run_path(manual_remap.__file__, run_name="__main__")
    assert opened == [pathlib.Path("/home/ankur/netv/cache/epg.db")]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
