#!/bin/sh
# ruff: noqa: EXE003, D300, D205, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"

Remap playlist channel guide IDs to the EPG database.

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
'''
# fmt: on

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict, cast

import argparse
import json
import os
import pathlib
import re
import shutil
import sqlite3
import time
import unicodedata

from treekle import json_loads

from netv import cache


def parse_name(name: object) -> tuple[str, list[str], bool]:
    """Split a leading country/category tag off a channel name.

    Playlist names look like "US| FOX SPORTS 1 HD"; EPG display names look
    like "UK - SKY SHOWCASE" or "[FR] TLC HD". Local channels (OTA affiliates
    with callsigns) drop their channel numbers so "NBC 11 SAN JOSE" can match
    "NBC SAN FRANCISCO".

    Args:
        name: Playlist or guide display name; untyped JSON/SQLite values
            (None, numbers) are coerced to text.

    Returns:
        country: Normalized country tag, or an empty string.
        tokens: Significant channel-name tokens.
        is_local: Whether a callsign identifies a local affiliate.
    """
    text = unicodedata.normalize("NFKD", str(name or ""))
    local = bool(_callsign_re().search(text.upper()))
    text = _parens_re().sub(" ", text)
    upper = text.upper()
    m = re.match(r"^\s*([A-Z0-9&/ ]{1,12}?)\s*\|\s*", upper)
    if m:
        raw, rest = m.group(1).strip(), text[m.end() :]
    else:
        m2 = re.match(r"^\s*\[([A-Z0-9]{1,4})\]\s*", upper) or re.match(
            r"^\s*([A-Z0-9&/]{1,4})\s*-\s",
            upper,
        )
        raw = m2.group(1) if m2 else ""
        rest = text[m2.end() :] if m2 else text
    return _country(raw), _tokenize(rest, local=local), local


def parse_id(epg_id: object) -> tuple[str, list[str], bool]:
    """Tokenize an EPG channel ID.

    Args:
        epg_id: Provider ID such as NBCSportsBayArea.us or US-NFL Network; an
            untyped SQLite value (None, number) is coerced to text.

    Returns:
        country: Normalized country tag, or an empty string.
        tokens: Significant channel-ID tokens.
        is_local: Whether the ID identifies a local affiliate.
    """
    s = unicodedata.normalize("NFKD", str(epg_id or ""))
    local = s.startswith("loc.") or bool(_callsign_re().search(s.upper()))
    country = ""
    m = re.search(r"\.([A-Za-z]{2})$", s)
    if m:
        country = _country(m.group(1))
        if country:
            s = s[: m.start()]
    # Strip only a country prefix: "BBC.One.uk" keeps its "BBC".
    m = re.match(r"^([A-Za-z]{2,4})[.\-]\s*", s)
    if m and _country(m.group(1)):
        country = country or _country(m.group(1))
        s = s[m.end() :]
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    s = _parens_re().sub(" ", s)
    s = re.sub(r"[._\-/|]", " ", s)
    toks = _tokenize(s, local=local)
    return country, toks, local


def load_candidates(
    db_path: pathlib.Path,
    cache_dir: pathlib.Path | None = None,
) -> list[Candidate]:
    """Load EPG channels with upcoming listings.

    Args:
        db_path: SQLite guide database.
        cache_dir: Directory containing source-country settings, when available.

    Returns:
        candidates: Channel metadata and normalized matching tokens.
    """
    countries = source_countries(cache_dir) if cache_dir else {}
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
    candidates: list[Candidate] = []
    for r in rows:
        name = cast(str | None, r["name"])
        channel_id = cast(str, r["id"])
        name_country, name_tokens, local = parse_name(name)
        id_country, id_tokens, _ = parse_id(channel_id)
        source_country = countries.get(cast(str | None, r["source_id"]) or "", "")
        candidates.append(
            Candidate(
                id=channel_id,
                name=name,
                upcoming=cast(int, r["upcoming"]),
                name_country=name_country,
                src_country=source_country,
                name_tokens=_expand_abbrevs(name_tokens),
                id_country=id_country,
                id_tokens=_expand_abbrevs(id_tokens),
                timeshift=_is_timeshift(name_tokens) or _is_timeshift(id_tokens),
                local=local,
            ),
        )
    return candidates


def channels_with_upcoming(db_path: pathlib.Path) -> set[str]:
    """Find guide channels that have programs ending in the future.

    Args:
        db_path: SQLite guide database.

    Returns:
        channel_ids: Guide channel IDs with upcoming listings.
    """
    conn = sqlite3.connect(db_path)
    now = time.time()
    rows = conn.execute(
        "SELECT DISTINCT channel_id FROM programs WHERE stop_ts > ?",
        (now,),
    ).fetchall()
    conn.close()
    return {cast(str, r[0]) for r in rows}


# epg.pw channel names carry no country tag, so the feed a channel came from is the best
# signal: a UK query must not match the US feed's Syfy.
def source_countries(cache_dir: pathlib.Path) -> dict[str, str]:
    """Map EPG-only sources named like epg_GB.xml to their country.

    Args:
        cache_dir: Directory containing server_settings.json.

    Returns:
        countries: Normalized country per source id; empty when unreadable.
    """
    settings_file = cache_dir / "server_settings.json"
    result: dict[str, str] = {}
    try:
        settings = json_loads(settings_file.read_text(), _ServerSettings)
    except (OSError, ValueError):
        return result
    for s in settings.get("sources", []):
        if s.get("type") != "epg":
            continue
        m = re.search(r"epg_([A-Za-z]{2})\.", s.get("url", ""))
        if m:
            result[s["id"]] = _country(m.group(1))
    return result


def query_variants(name: object, current_id: str) -> list[Variant]:
    """Build query variants from the channel name and current guide ID.

    Args:
        name: Playlist channel display name, as read from untyped JSON.
        current_id: Existing provider guide ID, if assigned.

    Returns:
        variants: Country tags, matching tokens, and their name/ID provenance.
    """
    country, tokens, _ = parse_name(name)
    variants = [Variant(country=country, tokens=tokens, from_name=True)]
    if current_id:
        c2, t2, _ = parse_id(current_id)
        if t2:
            variants.append(Variant(country=c2, tokens=t2, from_name=False))
    # A callsign in parentheses ("PBS (KQED) San Francisco") is its own
    # variant so it can match callsign-only EPG names like "KQED-DT".
    m = re.search(r"\(([A-Z]{2,6}(?:-DT\d?)?)\)", str(name or "").upper())
    if m:
        toks = _tokenize(m.group(1))
        if toks:
            variants.append(Variant(country=country, tokens=toks, from_name=True))
    # Abbreviation and phrase aliases expand the name variant.
    aliases = {
        "ID": ["INVESTIGATION", "DISCOVERY"],
        "FS1": ["FOX", "SPORTS", "1"],
        "FS2": ["FOX", "SPORTS", "2"],
        "ESPNEWS": ["ESPN", "NEWS"],
        "ESPNU": ["ESPN", "U"],
        "CBSSN": ["CBS", "SPORTS", "NETWORK"],
        "BTN": ["BIG", "TEN", "NETWORK"],
    }
    phrase_aliases = {
        ("BAY", "AREA"): ("SAN", "FRANCISCO"),
        ("SKY", "CINEMA"): ("SKY",),  # GB feed drops "Cinema": "Sky Family HD"
    }
    base = variants[0]
    for i, t in enumerate(base["tokens"]):
        if t in aliases:
            expanded = base["tokens"][:i] + aliases[t] + base["tokens"][i + 1 :]
            variants.append(Variant(country=base["country"], tokens=expanded, from_name=True))
    for i in range(len(base["tokens"]) - 1):
        pair = (base["tokens"][i], base["tokens"][i + 1])
        if pair in phrase_aliases:
            sub = list(phrase_aliases[pair])
            expanded = base["tokens"][:i] + sub + base["tokens"][i + 2 :]
            variants.append(Variant(country=base["country"], tokens=expanded, from_name=True))
    return variants


def pick_match(
    variants: Sequence[Variant],
    candidates: Sequence[Candidate],
    min_score: float,
    margin: float,
    available: frozenset[str] = frozenset(),
) -> tuple[Candidate | None, list[tuple[float, Candidate]]]:
    """Pick a confident guide match and retain the leading suggestions.

    Args:
        variants: Query tokens and country tags.
        candidates: Guide channels with upcoming listings.
        min_score: Minimum score required for acceptance.
        margin: Required lead over a non-equivalent runner-up.
        available: Countries represented by source feeds.

    Returns:
        best: Accepted candidate, or None when confidence is insufficient.
        top: Up to three leading score/candidate pairs.
    """
    scored = candidate_scores(variants, candidates, available)
    if not scored:
        return None, []
    best_score, best = scored[0][0], scored[0][1]
    second = scored[1][1] if len(scored) > 1 else None
    ok = best_score >= min_score and (
        second is None or best_score - scored[1][0] >= margin or _equivalent(best, second, variants)
    )
    if not ok:
        return None, [(s, c) for s, c, _, _ in scored[:3]]
    return best, [(s, c) for s, c, _, _ in scored[:3]]


def candidate_scores(
    variants: Sequence[Variant],
    candidates: Sequence[Candidate],
    available: frozenset[str] = frozenset(),
) -> list[tuple[float, Candidate, bool, int]]:
    """Score and rank every candidate against the query variants.

    Corroborations count how many query-variant/source pairs independently
    scored >= 100; they break ties between equal scores.

    Args:
        variants: Query tokens and country tags.
        candidates: Guide channels with upcoming listings.
        available: Countries represented by source feeds.

    Returns:
        scored: Best-first tuples of score, candidate, whether a name variant
            produced the score, and corroboration count.
    """
    query_shift = any(_is_timeshift(v["tokens"]) for v in variants if v["from_name"])
    scored: list[tuple[float, Candidate, bool, int]] = []
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
            not item[2],  # Prefer name-variant matches on ties.
            _is_west(item[1]),
            -item[1]["upcoming"],
        ),
    )
    return scored


def main() -> int:
    """Preview or apply guide-ID remaps.

    Returns:
        exit_code: Process exit status.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())

    playlists_path = flags.cache_dir / "playlists.json"
    db_path = flags.cache_dir / "epg.db"
    if not playlists_path.exists():
        raise SystemExit(f"playlists.json not found: {playlists_path}")
    if not db_path.exists():
        raise SystemExit(f"epg.db not found: {db_path}")

    # Read untyped: the file is rewritten whole, and a typed read would drop
    # every key this tool does not declare.
    data = json.loads(playlists_path.read_text())
    all_playlists = data.get("playlists", [])
    candidates = load_candidates(db_path, flags.cache_dir)
    available = frozenset(source_countries(flags.cache_dir).values())
    has_upcoming = channels_with_upcoming(db_path)
    print(f"EPG channels with upcoming listings: {len(candidates)}")

    remapped = 0
    unmatched = 0
    already_ok = 0
    changed_playlists = 0

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
            if not current and flags.skip_unmapped:
                continue

            variants = query_variants(name, current)
            best, top = pick_match(variants, candidates, flags.min_score, flags.margin, available)
            if best is not None:
                old = current or "(none)"
                print(
                    f"  REMAP {name}: {old} -> {best['name']} [{best['id']}] "
                    f"(score {top[0][0]:.0f}, {best['upcoming']} upcoming)",
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
            changed_playlists += 1

    print(f"\nSummary: {remapped} remapped, {unmatched} unmatched, {already_ok} already OK")

    if not flags.apply:
        print("Dry-run only; re-run with --apply to write changes.")
        return 0

    if not changed_playlists:
        print("Nothing to change.")
        return 0

    backup = flags.cache_dir / f"playlists.json.bak-{int(time.time())}"
    shutil.copy2(playlists_path, backup)
    print(f"Backup: {backup}")

    tmp = playlists_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, playlists_path)
    print(f"Updated {changed_playlists} playlist(s) in {playlists_path}")
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    cache_dir: pathlib.Path
    apply: bool
    min_score: float
    margin: float
    skip_unmapped: bool


class QueryTokens(TypedDict):
    """The tokens one query variant matches with."""

    tokens: list[str]


class Variant(QueryTokens):
    """One query spelling of a playlist channel."""

    country: str
    from_name: bool


class ChannelTokens(TypedDict):
    """The tokens a guide channel is matched by."""

    name_tokens: list[str]
    id_tokens: list[str]


class Candidate(ChannelTokens):
    """A guide channel with upcoming listings, normalized for matching."""

    id: str
    name: str | None
    upcoming: int
    name_country: str
    src_country: str
    id_country: str
    timeshift: bool
    local: bool


class _Source(TypedDict, total=False):
    id: str
    type: str
    url: str


class _ServerSettings(TypedDict, total=False):
    sources: list[_Source]


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


def _callsign_re() -> re.Pattern[str]:
    return re.compile(r"\([A-Z]{2,6}(-DT)?\)")


def _parens_re() -> re.Pattern[str]:
    return re.compile(r"\([^)]*\)")


def _country(raw: str) -> str:
    """Normalize a country tag; 3-letter codes come from EPG prefixes ("NOR - ...")."""
    raw = raw.upper()
    latin_america = [
        "BRA",
        "ARG",
        "CHL",
        "COL",
        "PER",
        "VEN",
        "URY",
        "PRY",
        "BOL",
        "ECU",
        "CRI",
        "PAN",
        "GTM",
        "HND",
        "SLV",
        "NIC",
        "DOM",
        "PRI",
        "PR",
        "CL",
        "PE",
        "VE",
        "UY",
        "PY",
        "BO",
        "EC",
        "CR",
        "PA",
        "GT",
        "HN",
        "SV",
        "NI",
        "DO",
        "CO",
    ]
    pairs = [
        "USA:US",
        "GB:UK",
        "GBR:UK",
        "GREAT:UK",
        "ENGLAND:UK",
        "NOR:NO",
        "DEN:DK",
        "ESP:ES",
        "ITA:IT",
        "DEU:DE",
        "FRA:FR",
        "NLD:NL",
        "BEL:BE",
        "CHE:CH",
        "AUT:AT",
        "SWE:SE",
        "FIN:FI",
        "POL:PL",
        "CZE:CZ",
        "TUR:TR",
        "RUS:RU",
        "UKR:UA",
        "ISR:IL",
        "ARE:AE",
        "KSA:SA",
        "QAT:QA",
        "KWT:KW",
        "BHR:BH",
        "OMN:OM",
        "JOR:JO",
        "LBN:LB",
        "EGY:EG",
        "MAR:MA",
        "DZA:DZ",
        "TUN:TN",
        "NGA:NG",
        "KEN:KE",
        "TZA:TZ",
        "UGA:UG",
        "ZMB:ZM",
        "ZWE:ZW",
        "GHA:GH",
        "ETH:ET",
        "SGP:SG",
        "MYS:MY",
        "IDN:ID",
        "THA:TH",  # codespell:ignore tha -- ISO 3166 alpha-3 code for Thailand.
        "VNM:VN",
        "PHL:PH",
        "JPN:JP",
        "KOR:KR",
        "CHN:CN",
        "TWN:TW",
        "HKG:HK",
        "MAC:MO",
        "LKA:LK",
        "BGD:BD",
        "NPL:NP",
        "MMR:MM",
        "KHM:KH",
        "MNG:MN",
        "KAZ:KZ",
        "UZB:UZ",
        "AZE:AZ",
        "GEO:GE",
        "ARM:AM",
        "BLR:BY",
        "MDA:MD",
        "SRB:RS",
        "HRV:HR",
        "SVN:SI",
        "BIH:BA",
        "MKD:MK",
        "MNE:ME",
        "ALB:AL",
        "LTU:LT",
        "LVA:LV",
        "EST:EE",
        "MEX:MX",
    ]
    aliases = dict.fromkeys(latin_america, "LAT")
    for pair in pairs:
        alias, country = pair.split(":")
        aliases[alias] = country
    if raw in aliases:
        return aliases[raw]
    return raw if raw in _known_countries() else ""


def _known_countries() -> frozenset[str]:
    return frozenset(
        [
            "US",
            "UK",
            "IE",
            "AU",
            "NZ",
            "CA",
            "IN",
            "PK",
            "AF",
            "ZA",
            "EU",
            "FR",
            "DE",
            "ES",
            "IT",
            "NL",
            "BE",
            "PT",
            "CH",
            "AT",
            "SE",
            "NO",
            "DK",
            "FI",
            "IS",
            "PL",
            "CZ",
            "SK",
            "HU",
            "RO",
            "BG",
            "GR",
            "TR",
            "RU",
            "UA",
            "IL",
            "AE",
            "SA",
            "QA",
            "KW",
            "BH",
            "OM",
            "JO",
            "LB",
            "EG",
            "MA",
            "DZ",
            "TN",
            "NG",
            "KE",
            "TZ",
            "UG",
            "ZM",
            "ZW",
            "GH",
            "ET",
            "SG",
            "MY",
            "ID",
            "TH",
            "VN",
            "PH",
            "JP",
            "KR",
            "CN",
            "TW",
            "HK",
            "MO",
            "LK",
            "BD",
            "NP",
            "MM",
            "KH",
            "MN",
            "KZ",
            "UZ",
            "AZ",
            "GE",
            "AM",
            "BY",
            "MD",
            "RS",
            "HR",
            "SI",
            "BA",
            "MK",
            "ME",
            "AL",
            "XK",
            "LT",
            "LV",
            "EE",
            "MX",
            "LAT",
            "BR",
        ],
    )


# EAST is the default feed; when both feeds exist EAST wins the tie, but a lone WEST
# feed still matches.
def _soft_tags() -> frozenset[str]:
    """Feed tags that don't distinguish channels."""
    return frozenset({"EAST", "EASTERN", "WESTERN", "WEST", "FEED", "FS1", "FS2"})


# Short glued forms (FS1, E4, S4C, ITV) stay whole.
def _split_token(t: str) -> list[str]:
    """Split glued tokens: "5ACTION" -> 5|ACTION, "HISTORY2" -> HISTORY|2, "GEOTV" -> GEO|TV."""
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
    # CamelCase glue: "TalkingPictures" -> Talking Pictures.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = text.upper()
    # "+1" marks a timeshift channel; a bare "+" is just a separator
    # ("Crime+Inv", "SkySp+HD").
    text = re.sub(r"\+1\b", " PLUS 1 ", text)
    text = text.replace("+", " ").replace("&", " AND ")
    words = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"]
    word_numbers = {word: str(number) for number, word in enumerate(words, start=1)}
    noise = frozenset(
        [
            "HD",
            "FHD",
            "UHD",
            "SD",
            "4K",
            "8K",
            "16K",
            "RAW",
            "60FPS",
            "HEVC",
            "CITY",
            "A3",
            "US",
            "USA",
            "TV",
            "PRIME",
            "BK",
            "HDTV",
            "EVENT",
            "ONLY",
            "THE",
            "AND",
            "U",
            "EU",
            "LOC",
            "DT",
            "DT2",
            "STREAM",
        ],
    ) | (_known_countries() - {"ID", "CN"})
    toks = re.findall(r"[A-Z0-9]+", text)
    toks = [p for t in toks for p in _split_token(t)]
    toks = [word_numbers.get(t, t) for t in toks]
    toks = [t for t in toks if t not in noise]
    if local:
        toks = [t for t in toks if not t.isdigit()]
    return toks


# A country embedded in the candidate's name/id ("FR - SYFY", "DiscoveryScience.nl")
# always signals the channel's market: mismatch is penalized. A country inferred only
# from the source feed (epg.pw names carry no tag) is weaker: mismatch is penalized only
# when the query's own country has a feed too -- otherwise the channel is diaspora
# content (e.g. Pakistani channels carried by the GB feed) and any feed is fine.
def _country_adj(
    q_country: str,
    c_text_country: str,
    c_src_country: str,
    available: frozenset[str],
) -> int:
    """Country score adjustment."""
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
    soft = _soft_tags()
    coverage = matched / len(query)
    extras = len({t for t in cand_set if t not in soft} - set(query))
    score = coverage * 100 - extras * 3
    stripped_c = [t for t in cand if t not in soft]
    stripped_q = [t for t in query if t not in soft]
    if stripped_c == stripped_q:
        score += 50
    return score


def _is_timeshift(tokens: list[str]) -> bool:
    return "PLUS" in tokens and "1" in tokens


def _expand_abbrevs(tokens: list[str]) -> list[str]:
    """Expand abbreviations in candidate names ("ComedyCent+1", "Disc.Turbo")."""
    abbreviations = {
        "CENT": ["CENTRAL"],
        "DISC": ["DISCOVERY"],
        "CN": ["CARTOON", "NETWORK"],
        "INV": ["INVESTIGATION"],
    }
    return [p for t in tokens for p in abbreviations.get(t, [t])]


def _is_west(c: ChannelTokens) -> bool:
    return bool({"WEST", "WESTERN"} & (set(c["name_tokens"]) | set(c["id_tokens"])))


def _equivalent(
    best: ChannelTokens,
    second: ChannelTokens,
    variants: Sequence[QueryTokens],
) -> bool:
    """Whether two candidates differ only by feed tag or each match a query variant."""
    soft = _soft_tags()
    best_tokens = frozenset(t for t in (best["name_tokens"] or best["id_tokens"]) if t not in soft)
    second_tokens = frozenset(
        t for t in (second["name_tokens"] or second["id_tokens"]) if t not in soft
    )
    if best_tokens == second_tokens:
        return True
    stripped_queries = {frozenset(t for t in v["tokens"] if t not in soft) for v in variants}
    return best_tokens in stripped_queries and second_tokens in stripped_queries


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
