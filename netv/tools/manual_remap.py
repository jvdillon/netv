#!/bin/sh
# ruff: noqa: EXE003, D300, D205, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P -m netv.tools.manual_remap "$@"

Verify manual remap targets have upcoming data, then apply them.

Usage:
    python -m netv.tools.manual_remap --cache-dir ~/netv/cache [--apply]
'''
# fmt: on

from __future__ import annotations

from typing import Protocol, cast

import argparse
import json
import pathlib
import shutil
import sqlite3
import time

from netv import cache


def main() -> int:
    """Preview or apply the hand-picked guide-ID replacements.

    Returns:
        exit_code: Process exit status.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    # Renamed channels and abbreviated EPG names need hand-picked targets.
    targets = {
        "US| CW SAN FRANCISCO HD": "466831",
        "US| CHEDDAR NEWS HD": "464798",
        "US| DISCOVERY SCIENCE HD": "464805",
        "UK| SKY SHOWCASE FHD": "524289",
        "UK| SKY SPORTS PREMIER LEAGUE FHD": "12453",
        "UK| SKY SPORTS + FHD": "381881",
        "UK| TNT BOX OFFICE FHD": "486781",
        "PK| MTA-MUSLIM TV": "12112",
        "PK| ARY QTV": "12256",
        "IN| AKAAL": "12510",
    }
    cache = flags.cache_dir
    conn = sqlite3.connect(cache / "epg.db")
    conn.row_factory = sqlite3.Row
    now = time.time()
    playlists_path = cache / "playlists.json"
    data = json.loads(playlists_path.read_text())
    changed = 0
    try:
        for playlist in data.get("playlists", []):
            for channel in playlist.get("channels", []):
                target = targets.get(channel.get("name", ""))
                if target is None:
                    continue
                row = conn.execute(
                    "SELECT c.name, COUNT(*) AS up FROM channels c "
                    "JOIN programs p ON p.channel_id = c.id "
                    "WHERE c.id = ? AND p.stop_ts > ? GROUP BY c.id",
                    (target, now),
                ).fetchone()
                if not row or not row["up"]:
                    print(f"SKIP (no data): {channel['name']} -> [{target}]")
                    continue
                print(
                    f"SET {channel['name']}: {channel.get('epg_channel_id') or '(none)'} "
                    f"-> {row['name']} [{target}] ({row['up']} upcoming)",
                )
                channel["epg_channel_id"] = target
                changed += 1
    finally:
        conn.close()
    print(f"\n{changed} channels to update")
    if flags.apply and changed:
        backup = cache / f"playlists.json.bak-{int(time.time())}"
        shutil.copy2(playlists_path, backup)
        print(f"Backup: {backup}")
        tmp = playlists_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(playlists_path)
        print("Applied.")
    elif not flags.apply:
        print("Dry-run only; re-run with --apply.")
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    cache_dir: pathlib.Path
    apply: bool


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    cache.initialize_paths()
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=cache.CACHE_DIR,
        help="netv data directory containing epg.db and playlists.json (default: %(default)s)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to playlists.json (default: dry-run)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
