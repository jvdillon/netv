"""Tests for remap_epg."""

import json
import pathlib
import runpy
import sqlite3
import sys
import time

import pytest

from netv import cache
from tools import remap_epg


@pytest.fixture(autouse=True)
def _restore_cache_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # _add_arguments calls cache.initialize_paths(), which rebinds these globals.
    for name in ("CACHE_DIR", "SERVER_SETTINGS_FILE", "USERS_DIR", "LOGOS_DIR"):
        monkeypatch.setattr(cache, name, getattr(cache, name))


def test_parse_name_coerces_untyped_values() -> None:
    assert remap_epg.parse_name(None) == ("", [], False)
    assert remap_epg.parse_name(5) == ("", ["5"], False)


def test_parse_id_coerces_untyped_values() -> None:
    assert remap_epg.parse_id(None) == ("", [], False)
    assert remap_epg.parse_id(4647) == ("", ["4647"], False)


def test_query_variants_accepts_untyped_name() -> None:
    assert remap_epg.query_variants(None, "") == [
        {"country": "", "tokens": [], "from_name": True},
    ]
    assert remap_epg.query_variants(4, "") == [
        {"country": "", "tokens": ["4"], "from_name": True},
    ]


def test_parse_name_tag_forms() -> None:
    assert remap_epg.parse_name("US| FOX SPORTS 1 HD") == ("US", ["FOX", "SPORTS", "1"], False)
    assert remap_epg.parse_name("[FR] TLC HD") == ("FR", ["TLC"], False)
    assert remap_epg.parse_name("UK - SKY SHOWCASE") == ("UK", ["SKY", "SHOWCASE"], False)
    assert remap_epg.parse_name("NOR - TV2") == ("NO", ["TV2"], False)
    assert remap_epg.parse_name("BRA | GLOBO") == ("LAT", ["GLOBO"], False)
    assert remap_epg.parse_name("ZZ| GEOTV") == ("", ["GEO"], False)


def test_parse_name_local_callsign_drops_numbers() -> None:
    assert remap_epg.parse_name("NBC 11 (KNTV) SAN JOSE") == ("", ["NBC", "SAN", "JOSE"], True)


def test_parse_id_forms() -> None:
    assert remap_epg.parse_id("NBCSportsBayArea.us") == (
        "US",
        ["NBC", "SPORTS", "BAY", "AREA"],
        False,
    )
    assert remap_epg.parse_id("US-NFL Network") == ("US", ["NFL", "NETWORK"], False)
    assert remap_epg.parse_id("loc.KQED9") == ("", ["KQED"], True)
    # ".zz" is not a country, so it stays in the tokens.
    assert remap_epg.parse_id("FooBar.zz") == ("", ["FOO", "BAR", "ZZ"], False)


def test_parse_id_keeps_non_country_prefix() -> None:
    assert remap_epg.parse_id("BBC.One.uk") == ("UK", ["BBC", "1"], False)
    assert remap_epg.parse_id("Disc.Turbo") == ("", ["DISC", "TURBO"], False)
    assert remap_epg.parse_id("Sky-Atlantic") == ("", ["SKY", "ATLANTIC"], False)


def test_tokenize_and_split() -> None:
    assert remap_epg._tokenize("Crime+Inv") == ["CRIME", "INV"]
    assert remap_epg._tokenize("Sky One +1") == ["SKY", "1", "PLUS", "1"]
    assert remap_epg._tokenize("HISTORY2 5ACTION FS1 E4") == [
        "HISTORY",
        "2",
        "5",
        "ACTION",
        "FS1",
        "E4",
    ]
    assert remap_epg._tokenize("NBC 11", local=True) == ["NBC"]


def test_country_adj_branches() -> None:
    avail = frozenset({"UK"})
    assert remap_epg._country_adj("", "US", "", avail) == 0
    assert remap_epg._country_adj("US", "US", "", avail) == 20
    assert remap_epg._country_adj("US", "UK", "", avail) == -25
    assert remap_epg._country_adj("UK", "", "UK", avail) == 20
    assert remap_epg._country_adj("UK", "", "US", avail) == -25
    assert remap_epg._country_adj("PK", "", "UK", avail) == 0
    assert remap_epg._country_adj("PK", "", "", avail) == 0


def test_base_score_branches() -> None:
    assert remap_epg._base_score([], ["A"]) == 0.0
    assert remap_epg._base_score(["A"], []) == 0.0
    assert remap_epg._base_score(["A"], ["B"]) == 0.0
    assert remap_epg._base_score(["A", "B"], ["A", "B", "EAST"]) == 150.0
    assert remap_epg._base_score(["A", "B"], ["A", "C"]) == 47.0


def test_expand_abbrevs_and_helpers() -> None:
    assert remap_epg._expand_abbrevs(["DISC", "X", "CN"]) == [
        "DISCOVERY",
        "X",
        "CARTOON",
        "NETWORK",
    ]
    assert remap_epg._is_timeshift(["PLUS", "1"])
    assert not remap_epg._is_timeshift(["PLUS"])
    assert remap_epg._is_west({"name_tokens": ["WEST"], "id_tokens": []})
    assert not remap_epg._is_west({"name_tokens": ["A"], "id_tokens": ["B"]})


def test_equivalent() -> None:
    a: remap_epg.ChannelTokens = {"name_tokens": ["SYFY", "EAST"], "id_tokens": []}
    b: remap_epg.ChannelTokens = {"name_tokens": [], "id_tokens": ["SYFY", "WEST"]}
    assert remap_epg._equivalent(a, b, [])
    c: remap_epg.ChannelTokens = {"name_tokens": ["FOX"], "id_tokens": []}
    d2: remap_epg.ChannelTokens = {"name_tokens": ["FOX", "SPORTS"], "id_tokens": []}
    variants: list[remap_epg.QueryTokens] = [{"tokens": ["FOX"]}, {"tokens": ["FOX", "SPORTS"]}]
    assert remap_epg._equivalent(c, d2, variants)
    assert not remap_epg._equivalent(c, {"name_tokens": ["NBC"], "id_tokens": []}, variants)


def test_query_variants_expansions() -> None:
    v = remap_epg.query_variants("US| ID", "Investigation.us")
    assert v == [
        {"country": "US", "tokens": ["ID"], "from_name": True},
        {"country": "US", "tokens": ["INVESTIGATION"], "from_name": False},
        {"country": "US", "tokens": ["INVESTIGATION", "DISCOVERY"], "from_name": True},
    ]
    v = remap_epg.query_variants("PBS (KQED) BAY AREA", "HD")
    assert v == [
        {"country": "", "tokens": ["PBS", "BAY", "AREA"], "from_name": True},
        {"country": "", "tokens": ["KQED"], "from_name": True},
        {"country": "", "tokens": ["PBS", "SAN", "FRANCISCO"], "from_name": True},
    ]
    # A callsign that tokenizes to nothing adds no variant.
    assert remap_epg.query_variants("FOO (USA)", "") == [
        {"country": "", "tokens": ["FOO"], "from_name": True},
    ]


def _cand(cid: str, name: str) -> remap_epg.Candidate:
    nc, nt, local = remap_epg.parse_name(name)
    ic, it, _ = remap_epg.parse_id(cid)
    return remap_epg.Candidate(
        id=cid,
        name=name,
        upcoming=5,
        name_country=nc,
        src_country="",
        name_tokens=remap_epg._expand_abbrevs(nt),
        id_country=ic,
        id_tokens=remap_epg._expand_abbrevs(it),
        timeshift=remap_epg._is_timeshift(nt) or remap_epg._is_timeshift(it),
        local=local,
    )


def test_candidate_scores_and_pick_match() -> None:
    syfy = _cand("Syfy.us", "US - SYFY")
    syfy_west = _cand("SyfyWest.us", "US - SYFY WEST")
    shifted = _cand("Syfy1.us", "SYFY +1")
    other = _cand("Nbc.us", "NBC")
    cands = [syfy_west, syfy, shifted, other]
    variants = remap_epg.query_variants("US| SYFY HD", "")
    scored = remap_epg.candidate_scores(variants, cands)
    assert [c["id"] for _, c, _, _ in scored] == ["Syfy.us", "SyfyWest.us"]
    best, top = remap_epg.pick_match(variants, cands, 140, 10)
    assert best is syfy  # Equivalent to the west feed despite a tiny margin.
    assert len(top) == 2
    best, top = remap_epg.pick_match(variants, cands, 500, 10)
    assert best is None
    assert top[0][1] is syfy
    assert remap_epg.pick_match(remap_epg.query_variants("ZZZ", ""), cands, 0, 0) == (None, [])
    best, _ = remap_epg.pick_match(variants, [syfy], 140, 10)
    assert best is syfy


def _make_db(path: pathlib.Path, rows: list[tuple[str, str, str | None, float]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE channels (id TEXT, name TEXT, source_id TEXT)")
    conn.execute("CREATE TABLE programs (channel_id TEXT, stop_ts REAL)")
    for cid, name, src, stop in rows:
        conn.execute("INSERT INTO channels VALUES (?, ?, ?)", (cid, name, src))
        conn.execute("INSERT INTO programs VALUES (?, ?)", (cid, stop))
    conn.commit()
    conn.close()


def _settings(data_dir: pathlib.Path) -> None:
    (data_dir / "server_settings.json").write_text(
        json.dumps(
            {
                "sources": [
                    {"id": "gb", "type": "epg", "url": "http://x/epg_GB.xml"},
                    {"id": "noc", "type": "epg", "url": "http://x/guide.xml"},
                    {"id": "m3u", "type": "m3u", "url": "http://x/epg_US.xml"},
                ],
            },
        ),
    )


def test_source_countries(tmp_path: pathlib.Path) -> None:
    assert remap_epg.source_countries(tmp_path) == {}
    _settings(tmp_path)
    assert remap_epg.source_countries(tmp_path) == {"gb": "UK"}


def test_load_candidates_and_upcoming(tmp_path: pathlib.Path) -> None:
    future = time.time() + 3600
    db = tmp_path / "epg.db"
    _make_db(db, [("a", "SYFY", "gb", future), ("b", "OLD", None, 1.0)])
    _settings(tmp_path)
    cands = remap_epg.load_candidates(db, tmp_path)
    assert [(c["id"], c["src_country"], c["name_tokens"]) for c in cands] == [("a", "UK", ["SYFY"])]
    assert remap_epg.load_candidates(db)[0]["src_country"] == ""
    assert remap_epg.channels_with_upcoming(db) == {"a"}


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["remap_epg.py", *argv])
    return remap_epg.main()


def test_main_missing_files(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="playlists.json not found"):
        _run_main(monkeypatch, "--cache-dir", str(tmp_path))
    (tmp_path / "playlists.json").write_text("{}")
    with pytest.raises(SystemExit, match="epg.db not found"):
        _run_main(monkeypatch, "--cache-dir", str(tmp_path))


def _setup_cache(data_dir: pathlib.Path) -> dict[str, object]:
    future = time.time() + 3600
    _make_db(
        data_dir / "epg.db",
        [
            ("Syfy.us", "US - SYFY", None, future),
            ("Ok.us", "OK CHANNEL", None, future),
            ("Fox1.us", "FOX ONE", None, future),
            ("Fox2.us", "FOX TWO", None, future),
        ],
    )
    data: dict[str, object] = {
        "extra": "kept",
        "playlists": [
            {
                "id": "p1",
                "channels": [
                    {"name": "US| SYFY HD", "epg_channel_id": "broken"},
                    {"name": "OK", "epg_channel_id": "Ok.us"},
                    {"name": "US| FOX", "epg_channel_id": ""},
                    {"name": "ZZZZ", "epg_channel_id": "nope"},
                ],
            },
            {"name": "Empty", "channels": [{"name": "NOTHING"}]},
        ],
    }
    (data_dir / "playlists.json").write_text(json.dumps(data))
    return data


def test_main_dry_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = _setup_cache(tmp_path)
    assert _run_main(monkeypatch, "--cache-dir", str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "EPG channels with upcoming listings: 4" in out
    assert "=== p1 (4 channels)" in out
    assert "=== Empty (1 channels)" in out
    assert "REMAP US| SYFY HD: broken -> US - SYFY [Syfy.us]" in out
    assert "MISS  US| FOX: (none) -- best: FOX ONE [Fox1.us]" in out
    assert "MISS  ZZZZ: nope -- no candidate" in out
    assert "Summary: 1 remapped, 3 unmatched, 1 already OK" in out
    assert "Dry-run only" in out
    assert json.loads((tmp_path / "playlists.json").read_text()) == data


def test_main_apply_writes_backup(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _setup_cache(tmp_path)
    assert _run_main(monkeypatch, "--cache-dir", str(tmp_path), "--apply", "--skip-unmapped") == 0
    out = capsys.readouterr().out
    assert "Summary: 1 remapped, 1 unmatched, 1 already OK" in out
    assert "Updated 1 playlist(s)" in out
    written = json.loads((tmp_path / "playlists.json").read_text())
    assert written["extra"] == "kept"
    assert written["playlists"][0]["channels"][0]["epg_channel_id"] == "Syfy.us"
    backups = list(tmp_path.glob("playlists.json.bak-*"))
    assert len(backups) == 1
    assert (
        json.loads(backups[0].read_text())["playlists"][0]["channels"][0]["epg_channel_id"]
        == "broken"
    )
    assert not (tmp_path / "playlists.json.tmp").exists()


def test_main_apply_nothing_to_change(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_db(tmp_path / "epg.db", [])
    (tmp_path / "playlists.json").write_text(json.dumps({"playlists": []}))
    assert _run_main(monkeypatch, "--cache-dir", str(tmp_path), "--apply") == 0
    assert "Nothing to change." in capsys.readouterr().out
    assert not list(tmp_path.glob("*.bak-*"))


def test_main_defaults_to_netv_data_dir(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _setup_cache(tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    assert _run_main(monkeypatch) == 0
    assert "Summary: 1 remapped, 3 unmatched, 1 already OK" in capsys.readouterr().out


def test_script_entry_point(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_db(tmp_path / "epg.db", [])
    (tmp_path / "playlists.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["remap_epg.py", "--cache-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(remap_epg.__file__, run_name="__main__")
    assert exc.value.code == 0


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
