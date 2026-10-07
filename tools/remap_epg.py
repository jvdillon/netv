#!/usr/bin/env python3
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""remap_epg.py -- Remap playlist channel guide IDs to the EPG database.

Playlist channels carry an epg_channel_id (from the provider's tvg-id or a
manual assignment). When the EPG feed uses a different channel-ID scheme,
those channels show no guide data. This tool finds playlist channels whose
epg_channel_id has no upcoming listings in the EPG database and remaps them
to the best name-matched EPG channel that has upcoming listings.

Usage:
    # Preview proposed remaps (dry-run, default):
    python tools/remap_epg.py --cache-dir ~/netv/cache

    # Write changes (backs up playlists.json first):
    python tools/remap_epg.py --cache-dir ~/netv/cache --apply

    # Only remap channels whose current ID is broken; leave empty ones alone:
    python tools/remap_epg.py --cache-dir ~/netv/cache --skip-unmapped
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sqlite3
import time
import unicodedata


# =============================================================================
# Name normalization
# =============================================================================

_KNOWN_COUNTRIES = {
    "US", "UK", "IE", "AU", "NZ", "CA", "IN", "PK", "AF", "ZA", "EU", "FR",
    "DE", "ES", "IT", "NL", "BE", "PT", "CH", "AT", "SE", "NO", "DK", "FI",
    "IS", "PL", "CZ", "SK", "HU", "RO", "BG", "GR", "TR", "RU", "UA", "IL",
    "AE", "SA", "QA", "KW", "BH", "OM", "JO", "LB", "EG", "MA", "DZ", "TN",
    "NG", "KE", "TZ", "UG", "ZM", "ZW", "GH", "ET", "SG", "MY", "ID", "TH",
    "VN", "PH", "JP", "KR", "CN", "TW", "HK", "MO", "LK", "BD", "NP", "MM",
    "KH", "MN", "KZ", "UZ", "AZ", "GE", "AM", "BY", "MD", "RS", "HR", "SI",
    "BA", "MK", "ME", "AL", "XK", "LT", "LV", "EE", "MX", "LAT", "BR",
}
# 3-letter codes seen in EPG name prefixes ("NOR - ...") and misc aliases.
_COUNTRY_ALIASES = {
    "USA": "US", "GB": "UK", "GBR": "UK", "GREAT": "UK", "ENGLAND": "UK",
    "NOR": "NO", "DEN": "DK", "ESP": "ES", "ITA": "IT", "DEU": "DE",
    "FRA": "FR", "NLD": "NL", "BEL": "BE", "CHE": "CH", "AUT": "AT",
    "SWE": "SE", "FIN": "FI", "POL": "PL", "CZE": "CZ", "TUR": "TR",
    "RUS": "RU", "UKR": "UA", "ISR": "IL", "ARE": "AE", "KSA": "SA",
    "QAT": "QA", "KWT": "KW", "BHR": "BH", "OMN": "OM", "JOR": "JO",
    "LBN": "LB", "EGY": "EG", "MAR": "MA", "DZA": "DZ", "TUN": "TN",
    "NGA": "NG", "KEN": "KE", "TZA": "TZ", "UGA": "UG", "ZMB": "ZM",
    "ZWE": "ZW", "GHA": "GH", "ETH": "ET", "SGP": "SG", "MYS": "MY",
    "IDN": "ID", "THA": "TH", "VNM": "VN", "PHL": "PH", "JPN": "JP",
    "KOR": "KR", "CHN": "CN", "TWN": "TW", "HKG": "HK", "MAC": "MO",
    "LKA": "LK", "BGD": "BD", "NPL": "NP", "MMR": "MM", "KHM": "KH",
    "MNG": "MN", "KAZ": "KZ", "UZB": "UZ", "AZE": "AZ", "GEO": "GE",
    "ARM": "AM", "BLR": "BY", "MDA": "MD", "SRB": "RS", "HRV": "HR",
    "SVN": "SI", "BIH": "BA", "MKD": "MK", "MNE": "ME", "ALB": "AL",
    "LTU": "LT", "LVA": "LV", "EST": "EE", "MEX": "MX", "BRA": "LAT",
    "ARG": "LAT", "CHL": "LAT", "COL": "LAT", "PER": "LAT", "VEN": "LAT",
    "URY": "LAT", "PRY": "LAT", "BOL": "LAT", "ECU": "LAT", "CRI": "LAT",
    "PAN": "LAT", "GTM": "LAT", "HND": "LAT", "SLV": "LAT", "NIC": "LAT",
    "DOM": "LAT", "PRI": "LAT", "PR": "LAT", "CL": "LAT", "PE": "LAT",
    "VE": "LAT", "UY": "LAT", "PY": "LAT", "BO": "LAT", "EC": "LAT",
    "CR": "LAT", "PA": "LAT", "GT": "LAT", "HN": "LAT", "SV": "LAT",
    "NI": "LAT", "DO": "LAT", "CO": "LAT",
}

_NOISE = {
    "HD", "FHD", "UHD", "SD", "4K", "8K", "16K", "RAW", "60FPS", "HEVC",
    "CITY", "A3", "US", "USA", "TV", "PRIME", "BK", "HDTV", "EVENT", "ONLY",
    "THE", "AND", "U", "EU", "LOC", "DT", "DT2", "STREAM",
} | (_KNOWN_COUNTRIES - {"ID", "CN"})  # fmt: skip
# Feed tags that don't distinguish channels. EAST is the default feed; when
# both feeds exist EAST wins the tie, but a lone WEST feed still matches.
_SOFT = {"EAST", "EASTERN", "WESTERN", "WEST", "FEED", "FS1", "FS2"}
_WORD2NUM = {
    "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5",
    "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9", "TEN": "10",
}
# Extra query variants for common abbreviations.
_ALIASES = {
    "ID": ["INVESTIGATION", "DISCOVERY"],
    "FS1": ["FOX", "SPORTS", "1"],
    "FS2": ["FOX", "SPORTS", "2"],
    "ESPNEWS": ["ESPN", "NEWS"],
    "ESPNU": ["ESPN", "U"],
    "CBSSN": ["CBS", "SPORTS", "NETWORK"],
    "BTN": ["BIG", "TEN", "NETWORK"],
}
# Multi-word substitutions applied to query token lists.
_PHRASE_ALIASES = {
    ("BAY", "AREA"): ("SAN", "FRANCISCO"),
    ("SKY", "CINEMA"): ("SKY",),  # GB feed drops "Cinema": "Sky Family HD"
}
# Abbreviations expanded in EPG candidate names ("ComedyCent+1", "Disc.Turbo").
_CAND_ABBREV = {
    "CENT": ["CENTRAL"],
    "DISC": ["DISCOVERY"],
    "CN": ["CARTOON", "NETWORK"],
    "INV": ["INVESTIGATION"],
}

_PLAYLIST_PREFIX_RE = re.compile(r"^\s*([A-Z0-9&/ ]{1,12}?)\s*\|\s*")
_EPG_PREFIX_RE = re.compile(r"^\s*([A-Z0-9&/]{1,4})\s*-\s")
_BRACKET_PREFIX_RE = re.compile(r"^\s*\[([A-Z0-9]{1,4})\]\s*")
_PARENS_RE = re.compile(r"\([^)]*\)")
_CALLSIGN_RE = re.compile(r"\([A-Z]{2,6}(-DT)?\)")
_ID_LEADING_TAG_RE = re.compile(r"^([A-Za-z]{2,4})[.\-]\s*")
_ID_TLD_RE = re.compile(r"\.([A-Za-z]{2})$")


def _country_of(raw: str) -> str:
    raw = raw.upper()
    if raw in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[raw]
    if raw in _KNOWN_COUNTRIES:
        return raw
    return ""


def _split_token(t: str) -> list[str]:
    """Split glued tokens: "5ACTION" -> 5|ACTION, "HISTORY2" -> HISTORY|2,
    "GEOTV" -> GEO|TV. Short glued forms (FS1, E4, S4C, ITV) stay whole."""
    if len(t) >= 5 and t.endswith("TV"):
        t = t[:-2] + " TV"
    # Always split digit->letter ("5ACTION", "NOW90s" -> NOW|90S).
    t = re.sub(r"(?<=\d)(?=[A-Z])", " ", t)
    # Split letter->digit only for longer tokens ("HISTORY2", "ESPN2"),
    # keeping FS1 / E4 / S4C intact.
    if len(t) >= 4:
        t = re.sub(r"(?<=[A-Z])(?=\d)", " ", t)
    return t.split()


def _tokenize(text: str, local: bool = False) -> list[str]:
    text = str(text or "")
    # CamelCase glue: "TalkingPictures" -> Talking Pictures.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = text.upper()
    # "+1" marks a timeshift channel; a bare "+" is just a separator
    # ("Crime+Inv", "SkySp+HD").
    text = re.sub(r"\+1\b", " PLUS 1 ", text)
    text = text.replace("+", " ").replace("&", " AND ")
    toks = re.findall(r"[A-Z0-9]+", text)
    toks = [p for t in toks for p in _split_token(t)]
    toks = [_WORD2NUM.get(t, t) for t in toks]
    toks = [t for t in toks if t not in _NOISE]
    if local:
        toks = [t for t in toks if not t.isdigit()]
    return toks


def parse_name(name: str) -> tuple[str, list[str], bool]:
    """Split a leading country/category tag off a channel name.

    Playlist names look like "US| FOX SPORTS 1 HD"; EPG display names look
    like "UK - SKY SHOWCASE" or "[FR] TLC HD". Returns (country, tokens,
    is_local). Local channels (OTA affiliates with callsigns) drop their
    channel numbers so "NBC 11 SAN JOSE" can match "NBC SAN FRANCISCO".
    """
    text = unicodedata.normalize("NFKD", str(name or ""))
    local = bool(_CALLSIGN_RE.search(text.upper()))
    text = _PARENS_RE.sub(" ", text)
    upper = text.upper()
    m = _PLAYLIST_PREFIX_RE.match(upper)
    if m:
        raw, rest = m.group(1).strip(), text[m.end():]
    else:
        m2 = _BRACKET_PREFIX_RE.match(upper) or _EPG_PREFIX_RE.match(upper)
        raw = m2.group(1) if m2 else ""
        rest = text[m2.end():] if m2 else text
    return _country_of(raw), _tokenize(rest, local=local), local


def parse_id(epg_id: str) -> tuple[str, list[str], bool]:
    """Tokenize an EPG channel id like "NBCSportsBayArea.us" or "US-NFL Network"."""
    s = unicodedata.normalize("NFKD", str(epg_id or ""))
    local = s.startswith("loc.") or bool(_CALLSIGN_RE.search(s.upper()))
    country = ""
    m = _ID_TLD_RE.search(s)
    if m:
        country = _country_of(m.group(1))
        if country:
            s = s[: m.start()]
    m = _ID_LEADING_TAG_RE.match(s)
    if m:
        country = country or _country_of(m.group(1))
        s = s[m.end():]
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    s = _PARENS_RE.sub(" ", s)
    s = re.sub(r"[._\-/|]", " ", s)
    toks = _tokenize(s, local=local)
    return country, toks, local


# =============================================================================
# Scoring
# =============================================================================


def _country_adj(
    q_country: str,
    c_text_country: str,
    c_src_country: str,
    available: frozenset[str],
) -> int:
    """Country score adjustment.

    A country embedded in the candidate's name/id ("FR - SYFY",
    "DiscoveryScience.nl") always signals the channel's market: mismatch is
    penalized. A country inferred only from the source feed (epg.pw names
    carry no tag) is weaker: mismatch is penalized only when the query's own
    country has a feed too -- otherwise the channel is diaspora content
    (e.g. Pakistani channels carried by the GB feed) and any feed is fine.
    """
    if not q_country:
        return 0
    if c_text_country:
        return 20 if q_country == c_text_country else -25
    if c_src_country:
        if q_country == c_src_country:
            return 20
        return -25 if q_country in available else 0
    return 0


def _base_score(query: list[str], cand: list[str]) -> float:
    if not query or not cand:
        return 0.0
    cand_set = set(cand)
    matched = sum(1 for t in query if t in cand_set)
    if matched == 0:
        return 0.0
    coverage = matched / len(query)
    extras = len({t for t in cand_set if t not in _SOFT} - set(query))
    score = coverage * 100 - extras * 3
    stripped_c = [t for t in cand if t not in _SOFT]
    stripped_q = [t for t in query if t not in _SOFT]
    if stripped_c == stripped_q:
        score += 50
    return score


def _is_timeshift(tokens: list[str]) -> bool:
    return "PLUS" in tokens and "1" in tokens


# =============================================================================
# EPG database
# =============================================================================


def _source_countries(cache_dir: pathlib.Path) -> dict[str, str]:
    """Country per source id, for EPG-only sources named like epg_GB.xml.

    epg.pw channel names carry no country tag, so the feed a channel came
    from is the best signal: a UK query must not match the US feed's Syfy.
    """
    settings_file = cache_dir / "server_settings.json"
    result: dict[str, str] = {}
    try:
        settings = json.loads(settings_file.read_text())
    except (OSError, ValueError):
        return result
    for s in settings.get("sources", []):
        if s.get("type") != "epg":
            continue
        m = re.search(r"epg_([A-Za-z]{2})\.", str(s.get("url", "")))
        if m:
            result[s["id"]] = _country_of(m.group(1))
    return result


def _expand_abbrevs(tokens: list[str]) -> list[str]:
    """Expand abbreviations in candidate names ("ComedyCent+1")."""
    return [p for t in tokens for p in _CAND_ABBREV.get(t, [t])]


def load_candidates(db_path: pathlib.Path, cache_dir: pathlib.Path | None = None) -> list[dict]:
    """EPG channels that have upcoming listings."""
    source_countries = _source_countries(cache_dir) if cache_dir else {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    now = time.time()
    rows = conn.execute(
        """
        SELECT c.id, c.name, c.source_id, COUNT(*) AS upcoming
        FROM channels c JOIN programs p ON p.channel_id = c.id
        WHERE p.stop_ts > ?
        GROUP BY c.id
        """,
        (now,),
    ).fetchall()
    conn.close()
    candidates = []
    for r in rows:
        name_country, name_tokens, local = parse_name(r["name"])
        id_country, id_tokens, _ = parse_id(r["id"])
        source_country = source_countries.get(r["source_id"] or "", "")
        candidates.append(
            {
                "id": r["id"],
                "name": r["name"],
                "upcoming": r["upcoming"],
                "name_country": name_country,
                "src_country": source_country,
                "name_tokens": _expand_abbrevs(name_tokens),
                "id_country": id_country,
                "id_tokens": _expand_abbrevs(id_tokens),
                "timeshift": _is_timeshift(name_tokens) or _is_timeshift(id_tokens),
                "local": local,
            }
        )
    return candidates


def channels_with_upcoming(db_path: pathlib.Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    now = time.time()
    rows = conn.execute(
        "SELECT DISTINCT channel_id FROM programs WHERE stop_ts > ?", (now,)
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


# =============================================================================
# Matching
# =============================================================================


def query_variants(name: str, current_id: str) -> list[dict]:
    """Query variants from the channel name and its current guide id."""
    country, tokens, _ = parse_name(name)
    variants = [{"country": country, "tokens": tokens, "from_name": True}]
    if current_id:
        c2, t2, _ = parse_id(current_id)
        if t2:
            variants.append({"country": c2, "tokens": t2, "from_name": False})
    # A callsign in parentheses ("PBS (KQED) San Francisco") is its own
    # variant so it can match callsign-only EPG names like "KQED-DT".
    m = re.search(r"\(([A-Z]{2,6}(?:-DT\d?)?)\)", str(name or "").upper())
    if m:
        toks = _tokenize(m.group(1))
        if toks:
            variants.append({"country": country, "tokens": toks, "from_name": True})
    # Abbreviation and phrase aliases expand the name variant.
    base = variants[0]
    for i, t in enumerate(base["tokens"]):
        if t in _ALIASES:
            expanded = base["tokens"][:i] + _ALIASES[t] + base["tokens"][i + 1:]
            variants.append({**base, "tokens": expanded, "from_name": True})
    for i in range(len(base["tokens"]) - 1):
        pair = tuple(base["tokens"][i : i + 2])
        if pair in _PHRASE_ALIASES:
            sub = list(_PHRASE_ALIASES[pair])
            expanded = base["tokens"][:i] + sub + base["tokens"][i + 2 :]
            variants.append({**base, "tokens": expanded, "from_name": True})
    return variants


def _stripped(tokens: list[str]) -> frozenset:
    return frozenset(t for t in tokens if t not in _SOFT)


def _is_west(c: dict) -> bool:
    return bool({"WEST", "WESTERN"} & (set(c["name_tokens"]) | set(c["id_tokens"])))


def _candidate_scores(
    variants: list[dict],
    candidates: list[dict],
    available: frozenset[str] = frozenset(),
) -> list[tuple[float, dict, bool, int]]:
    """Score every candidate: (score, candidate, matched_name_variant,
    corroborations). Corroborations count how many query-variant/source
    pairs independently scored >= 100, to break ties."""
    query_shift = any(_is_timeshift(v["tokens"]) for v in variants if v["from_name"])
    scored = []
    for c in candidates:
        if query_shift != c["timeshift"]:
            continue
        best = 0.0
        from_name = False
        corroboration = 0
        for v in variants:
            # Name source falls back to the id's text country ("8K | DISCOVERY
            # SCIENCE" + id "DiscoveryScience.nl"); both then fall back to the
            # source feed's country, which gets the diaspora exemption.
            for c_text, c_tokens in (
                (c["name_country"] or c["id_country"], c["name_tokens"]),
                (c["id_country"], c["id_tokens"]),
            ):
                s = _base_score(v["tokens"], c_tokens)
                if s > 0:
                    s += _country_adj(v["country"], c_text, c["src_country"], available)
                if s >= 100:
                    corroboration += 1
                if s > best:
                    best, from_name = s, v["from_name"]
        if best > 0:
            scored.append((best, c, from_name, corroboration))
    scored.sort(
        key=lambda item: (
            -item[0],
            -item[3],
            not item[2],  # prefer name-variant matches on ties
            _is_west(item[1]),
            -item[1]["upcoming"],
        )
    )
    return scored


def _equivalent(best: dict, second: dict, variants: list[dict]) -> bool:
    """Two candidates are interchangeable when they differ only by feed tag,
    or when each exactly matches a different query variant of the channel."""
    if _stripped(best["name_tokens"] or best["id_tokens"]) == _stripped(
        second["name_tokens"] or second["id_tokens"]
    ):
        return True
    stripped_queries = {_stripped(v["tokens"]) for v in variants}
    return (
        _stripped(best["name_tokens"] or best["id_tokens"]) in stripped_queries
        and _stripped(second["name_tokens"] or second["id_tokens"]) in stripped_queries
    )


def pick_match(
    variants: list[dict],
    candidates: list[dict],
    min_score: float,
    margin: float,
    available: frozenset[str] = frozenset(),
) -> tuple[dict | None, list[tuple[float, dict]]]:
    """Pick the best candidate, or None when nothing is confident."""
    scored = _candidate_scores(variants, candidates, available)
    if not scored:
        return None, []
    best_score, best = scored[0][0], scored[0][1]
    second = scored[1][1] if len(scored) > 1 else None
    ok = best_score >= min_score and (
        second is None
        or best_score - scored[1][0] >= margin
        or _equivalent(best, second, variants)
    )
    if not ok:
        return None, [(s, c) for s, c, _, _ in scored[:3]]
    return best, [(s, c) for s, c, _, _ in scored[:3]]


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap playlist channel guide IDs to the EPG database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent / "cache",
        help="netv cache directory containing epg.db and playlists.json",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to playlists.json (default: dry-run)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=140.0,
        help="Minimum score to accept a match (default: 140)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=10.0,
        help="Required score lead over the runner-up candidate (default: 10)",
    )
    parser.add_argument(
        "--skip-unmapped",
        action="store_true",
        help="Only remap channels with a broken ID; leave empty ones alone",
    )
    args = parser.parse_args()

    playlists_path = args.cache_dir / "playlists.json"
    db_path = args.cache_dir / "epg.db"
    if not playlists_path.exists():
        raise SystemExit(f"playlists.json not found: {playlists_path}")
    if not db_path.exists():
        raise SystemExit(f"epg.db not found: {db_path}")

    data = json.loads(playlists_path.read_text())
    all_playlists = data.get("playlists", [])
    candidates = load_candidates(db_path, args.cache_dir)
    available = frozenset(_source_countries(args.cache_dir).values())
    has_upcoming = channels_with_upcoming(db_path)
    print(f"EPG channels with upcoming listings: {len(candidates)}")

    remapped = 0
    unmatched = 0
    already_ok = 0
    changed_playlists: list[dict] = []

    for playlist in all_playlists:
        channels = playlist.get("channels", [])
        changed = False
        print(f"\n=== {playlist.get('name', playlist.get('id'))} ({len(channels)} channels)")
        for ch in channels:
            name = ch.get("name", "")
            current = (ch.get("epg_channel_id") or "").strip()
            if current and current in has_upcoming:
                already_ok += 1
                continue
            if not current and args.skip_unmapped:
                continue

            variants = query_variants(name, current)
            best, top = pick_match(variants, candidates, args.min_score, args.margin, available)
            if best:
                old = current or "(none)"
                print(
                    f"  REMAP {name}: {old} -> {best['name']} [{best['id']}] "
                    f"(score {top[0][0]:.0f}, {best['upcoming']} upcoming)"
                )
                ch["epg_channel_id"] = best["id"]
                remapped += 1
                changed = True
            else:
                unmatched += 1
                if top:
                    hints = ", ".join(f"{c['name']} [{c['id']}] ({s:.0f})" for s, c in top)
                    print(f"  MISS  {name}: {current or '(none)'} -- best: {hints}")
                else:
                    print(f"  MISS  {name}: {current or '(none)'} -- no candidate")
        if changed:
            changed_playlists.append(playlist)

    print(f"\nSummary: {remapped} remapped, {unmatched} unmatched, {already_ok} already OK")

    if not args.apply:
        print("Dry-run only; re-run with --apply to write changes.")
        return

    if not changed_playlists:
        print("Nothing to change.")
        return

    backup = args.cache_dir / f"playlists.json.bak-{int(time.time())}"
    shutil.copy2(playlists_path, backup)
    print(f"Backup: {backup}")

    tmp = playlists_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, playlists_path)
    print(f"Updated {len(changed_playlists)} playlist(s) in {playlists_path}")


if __name__ == "__main__":
    main()
