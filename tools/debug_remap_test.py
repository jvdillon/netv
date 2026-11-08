"""Tests for debug_remap."""

import json
import pathlib
import runpy
import sqlite3
import sys
import time

import pytest

from tools import debug_remap

import cache


@pytest.fixture(autouse=True)
def _restore_cache_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # _add_arguments calls cache.initialize_paths(), which rebinds these globals.
    for name in ("CACHE_DIR", "SERVER_SETTINGS_FILE", "USERS_DIR", "LOGOS_DIR"):
        monkeypatch.setattr(cache, name, getattr(cache, name))


def _make_cache(root: pathlib.Path, rows: list[tuple[str, str, str | None]]) -> pathlib.Path:
    root.mkdir(exist_ok=True)
    conn = sqlite3.connect(root / "epg.db")
    conn.execute("CREATE TABLE channels (id TEXT, name TEXT, source_id TEXT)")
    conn.execute("CREATE TABLE programs (channel_id TEXT, stop_ts REAL)")
    future = time.time() + 3600
    for cid, name, src in rows:
        conn.execute("INSERT INTO channels VALUES (?, ?, ?)", (cid, name, src))
        conn.execute("INSERT INTO programs VALUES (?, ?)", (cid, future))
    conn.commit()
    conn.close()
    return root


def _syfy_rows() -> list[tuple[str, str, str | None]]:
    rows: list[tuple[str, str, str | None]] = [
        (f"Syfy{i}.us", f"US - SYFY {i}", None) for i in range(10)
    ]
    return [*rows, ("Nbc.us", "NBC", None)]


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["debug_remap.py", *argv])
    return debug_remap.main()


def test_main_prints_variants_and_top_eight(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = _make_cache(tmp_path / "c", _syfy_rows())
    assert _run(monkeypatch, "--cache-dir", str(data_dir), "US| SYFY HD", "Syfy.us") == 0
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


def test_main_defaults_to_netv_data_dir(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", _make_cache(tmp_path / "data", _syfy_rows()))
    assert _run(monkeypatch, "NBC") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "VARIANT {'country': '', 'tokens': ['NBC'], 'from_name': True}"
    assert len(lines) == 2
    assert "Nbc.us" in lines[1]


def test_main_scores_with_source_countries_like_remap_epg(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = _make_cache(tmp_path / "c", [("gb1", "SYFY", "gb"), ("us1", "SYFY", "us")])
    sources = [
        {"id": "gb", "type": "epg", "url": "http://x/epg_GB.xml"},
        {"id": "us", "type": "epg", "url": "http://x/epg_US.xml"},
    ]
    (data_dir / "server_settings.json").write_text(json.dumps({"sources": sources}))
    assert _run(monkeypatch, "--cache-dir", str(data_dir), "US| SYFY") == 0
    scores = capsys.readouterr().out.splitlines()[1:]
    assert scores[0].split()[:2] == ["170", "from_name=True"]
    assert " us1 " in scores[0]
    assert scores[1].split()[0] == "125"
    assert " gb1 " in scores[1]


def test_script_entry_point(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = _make_cache(tmp_path / "c", _syfy_rows())
    monkeypatch.setattr(sys, "argv", ["debug_remap.py", "--cache-dir", str(data_dir), "ZZZ"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(debug_remap.__file__, run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("VARIANT")


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
