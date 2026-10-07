#!/usr/bin/env python3
"""Verify manual remap targets have upcoming data, then apply them.

Usage: manual_remap.py [--apply]
"""
import json
import pathlib
import shutil
import sqlite3
import sys
import time


CACHE = pathlib.Path("/home/ankur/netv/cache")
# Hand-picked targets for channels the matcher can't resolve: renamed
# channels (Sky One -> Sky Showcase, KBCW -> KPYX, Cheddar News -> Cheddar)
# and heavily abbreviated EPG names (SkySp PL HD).
TARGETS = {
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

conn = sqlite3.connect(CACHE / "epg.db")
conn.row_factory = sqlite3.Row
now = time.time()
apply = "--apply" in sys.argv

playlists_path = CACHE / "playlists.json"
data = json.loads(playlists_path.read_text())
changed = 0
for pl in data.get("playlists", []):
    for ch in pl.get("channels", []):
        target = TARGETS.get(ch.get("name", ""))
        if not target:
            continue
        r = conn.execute(
            "SELECT c.name, COUNT(*) AS up FROM channels c "
            "JOIN programs p ON p.channel_id = c.id "
            "WHERE c.id = ? AND p.stop_ts > ? GROUP BY c.id",
            (target, now),
        ).fetchone()
        if not r or not r["up"]:
            print(f"SKIP (no data): {ch['name']} -> [{target}]")
            continue
        print(f"SET {ch['name']}: {ch.get('epg_channel_id') or '(none)'} -> {r['name']} [{target}] ({r['up']} upcoming)")
        ch["epg_channel_id"] = target
        changed += 1

print(f"\n{changed} channels to update")
if apply and changed:
    backup = CACHE / f"playlists.json.bak-{int(time.time())}"
    shutil.copy2(playlists_path, backup)
    print(f"Backup: {backup}")
    tmp = playlists_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(playlists_path)
    print("Applied.")
elif not apply:
    print("Dry-run only; re-run with --apply.")
