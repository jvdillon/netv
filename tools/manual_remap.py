#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"

Verify manual remap targets have upcoming data, then apply them.

Usage: manual_remap.py [--apply]
'''
# fmt: on

from __future__ import annotations

import json
import pathlib
import shutil
import sqlite3
import sys
import time


def main() -> int:
    """Preview or apply the hand-picked guide-ID replacements.

    Returns:
        exit_code: Process exit status.
    """
    cache = pathlib.Path("/home/ankur/netv/cache")
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
    conn = sqlite3.connect(cache / "epg.db")
    conn.row_factory = sqlite3.Row
    now = time.time()
    apply = "--apply" in sys.argv
    playlists_path = cache / "playlists.json"
    data = json.loads(playlists_path.read_text())
    changed = 0
    try:
        for playlist in data.get("playlists", []):
            for channel in playlist.get("channels", []):
                target = targets.get(channel.get("name", ""))
                if not target:
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
    if apply and changed:
        backup = cache / f"playlists.json.bak-{int(time.time())}"
        shutil.copy2(playlists_path, backup)
        print(f"Backup: {backup}")
        tmp = playlists_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(playlists_path)
        print("Applied.")
    elif not apply:
        print("Dry-run only; re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
