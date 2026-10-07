#!/usr/bin/env python3
"""Debug remap scoring for one channel. Usage: debug_remap.py "NAME" "CURRENT_ID"."""
import pathlib
import sys


sys.path.insert(0, "tools")
import remap_epg as R


name = sys.argv[1]
cur = sys.argv[2] if len(sys.argv) > 2 else ""
variants = R.query_variants(name, cur)
for v in variants:
    print("VARIANT", v)
cands = R.load_candidates(pathlib.Path("cache/epg.db"))
scored = R._candidate_scores(variants, cands)
for s, c, fn in scored[:8]:
    print(
        f"{s:6.0f} from_name={fn} {c['id']:38s} "
        f"name_tokens={c['name_tokens']} id_tokens={c['id_tokens']} "
        f"nc={c['name_country']!r} ic={c['id_country']!r} shift={c['timeshift']}"
    )
