"""Shared Live TV playlists.

Admins curate named, ordered channel lists in the web UI. Playlists surface in
the guide as virtual categories (``pl:<id>``) so every client, including the
Apple apps, can show them without editing support.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Final, NotRequired, TypedDict

import pathlib
import re
import threading
import unicodedata
import uuid

from treekle import MutablePlainTree, ReadError, from_plain

from util import read_json_object, write_json_object
from xtream import ProviderRecord, category_ids, text_field

import cache


class PlaylistChannel(TypedDict, total=False):
    """A saved playlist entry, identifying a stream as it was when added.

    Entries saved by older releases may lack fields; a missing one reads as empty.
    """

    stream_id: str
    name: str
    source_id: str
    epg_channel_id: str


class Playlist(TypedDict):
    """A named, ordered channel list."""

    id: str
    name: str
    channels: NotRequired[list[PlaylistChannel]]


class StreamSummary(TypedDict):
    """The stream fields the playlist editor shows."""

    stream_id: str
    name: str
    source_id: str
    epg_channel_id: str
    icon: str
    category: str


class ScoredSummary(StreamSummary):
    """A stream summary ranked against a pasted channel name."""

    score: float


class NameMatch(TypedDict):
    """The candidate streams for one pasted channel name."""

    query: str
    candidates: list[ScoredSummary]


PREFIX: Final = "pl:"

_lock = threading.Lock()


def is_playlist_id(category_id: object) -> bool:
    return str(category_id).startswith(PREFIX)


def stream_key(stream: Mapping[str, MutablePlainTree]) -> str:
    """Stream ids are only unique within a source, so qualify them."""
    return f"{text_field(stream, 'source_id')}:{text_field(stream, 'stream_id')}"


def load() -> list[Playlist]:
    try:
        raw = read_json_object(_path()).get("playlists", [])
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    playlists: list[Playlist] = []
    for entry in raw:
        try:
            playlist = from_plain(entry, Playlist)
        except ReadError:
            continue
        if playlist["id"]:
            playlists.append(playlist)
    return playlists


def save(playlists: list[Playlist]) -> None:
    write_json_object(_path(), {"playlists": playlists}, _PlaylistFile)


def get(playlist_id: str) -> Playlist | None:
    return next((p for p in load() if p["id"] == playlist_id), None)


def create(
    name: MutablePlainTree,
    channels: MutablePlainTree = None,
    *,
    max_playlists: int = 100,
) -> Playlist:
    playlist: Playlist = {
        "id": uuid.uuid4().hex[:12],
        "name": _clean_name(name),
        "channels": [] if channels is None else _clean_channels(channels),
    }
    with _lock:
        playlists = load()
        if len(playlists) >= max_playlists:
            raise ValueError(f"At most {max_playlists} playlists are allowed")
        playlists.append(playlist)
        save(playlists)
        return playlist


def update(
    playlist_id: str,
    name: MutablePlainTree = None,
    channels: MutablePlainTree = None,
) -> Playlist | None:
    with _lock:
        playlists = load()
        playlist = next((p for p in playlists if p["id"] == playlist_id), None)
        if playlist is None:
            return None
        if name is not None:
            playlist["name"] = _clean_name(name)
        if channels is not None:
            playlist["channels"] = _clean_channels(channels)
        save(playlists)
        return playlist


def delete(playlist_id: str) -> bool:
    with _lock:
        playlists = load()
        remaining = [p for p in playlists if p["id"] != playlist_id]
        if len(remaining) == len(playlists):
            return False
        save(remaining)
        return True


def reorder(ids: list[str]) -> list[Playlist]:
    with _lock:
        playlists = load()
        position = {pid: i for i, pid in enumerate(ids)}
        playlists.sort(key=lambda p: position.get(p["id"], len(position)))
        save(playlists)
        return playlists


# =============================================================================
# Stream resolution
# =============================================================================

_index_lock = threading.Lock()
_index: (
    tuple[
        list[ProviderRecord],
        dict[tuple[str, str], ProviderRecord],
        dict[str, ProviderRecord],
        dict[tuple[str, str], ProviderRecord],
        dict[tuple[str, str], ProviderRecord],
    ]
    | None
) = None


def resolve_channel(
    entry: PlaylistChannel,
    streams: list[ProviderRecord],
) -> ProviderRecord | None:
    """Find the current stream for a saved entry.

    Stream IDs can change (or be reused) when a provider refreshes, so an id
    match must agree on the name; otherwise try the name, then the EPG id, within
    the same source, and only then the bare id match.

    Args:
        entry: Saved playlist channel metadata.
        streams: Current provider streams.

    Returns:
        stream: Best matching stream, or None.
    """
    by_id, by_bare_id, by_name, by_epg = _stream_index(streams)
    source = entry.get("source_id", "")
    name = entry.get("name", "")
    stream_id = entry.get("stream_id", "")
    stream = by_id.get((source, stream_id)) if source else by_bare_id.get(stream_id)
    if stream is not None and (not name or stream.get("name") == name):
        return stream
    if name and (source, name) in by_name:
        return by_name[(source, name)]
    epg_id = entry.get("epg_channel_id", "")
    if epg_id and (source, epg_id) in by_epg:
        return by_epg[(source, epg_id)]
    return stream


def with_entry_epg(entry: PlaylistChannel, stream: ProviderRecord) -> ProviderRecord:
    """Apply a curated guide identity without mutating the provider stream.

    Args:
        entry: Saved playlist channel metadata.
        stream: Resolved provider stream.

    Returns:
        stream: Stream metadata with any curated guide identity applied.
    """
    epg_id = entry.get("epg_channel_id", "")
    if epg_id:
        return {**stream, "epg_channel_id": epg_id}
    return stream


def resolve(playlist: Playlist, streams: list[ProviderRecord]) -> list[ProviderRecord]:
    """Resolve available streams in playlist order without duplicates.

    Args:
        playlist: Saved playlist with ordered channel entries.
        streams: Current provider streams.

    Returns:
        resolved: Available streams with curated guide identities.
    """
    resolved: list[ProviderRecord] = []
    seen: set[str] = set()
    for entry in playlist.get("channels", []):
        stream = resolve_channel(entry, streams)
        if stream is None:
            continue
        key = stream_key(stream)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(with_entry_epg(entry, stream))
    return resolved


# =============================================================================
# Search and bulk matching
# =============================================================================


def normalize(name: object) -> list[str]:
    noise = {
        "HD",
        "FHD",
        "UHD",
        "SD",
        "4K",
        "8K",
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
    }
    text = unicodedata.normalize("NFKD", str(name or "")).upper()
    text = re.sub(r"^\s*[A-Z0-9+&/ ]{1,12}\|\s*", "", text)
    text = text.replace("&", " AND ").replace("+", " PLUS ")
    tokens = re.findall(r"[A-Z0-9]+", text)
    return [t for t in tokens if t not in noise]


def summarize(stream: ProviderRecord, category_names: Mapping[str, str]) -> StreamSummary:
    category = next(
        (category_names[c] for c in category_ids(stream) if c in category_names),
        "",
    )
    return {
        "stream_id": text_field(stream, "stream_id"),
        "name": text_field(stream, "name"),
        "source_id": text_field(stream, "source_id"),
        "epg_channel_id": text_field(stream, "epg_channel_id"),
        "icon": text_field(stream, "stream_icon"),
        "category": category,
    }


def search(
    streams: list[ProviderRecord],
    category_names: Mapping[str, str],
    query: str = "",
    category_id: str = "",
    category_prefix: str = "",
    limit: int = 200,
) -> list[StreamSummary]:
    query_tokens = normalize(query) if query else []
    raw = query.strip().upper()
    results: list[StreamSummary] = []
    for s in _category_filter(streams, category_names, category_prefix):
        if category_id and category_id not in category_ids(s):
            continue
        if query_tokens:
            tokens = set(normalize(s.get("name", "")))
            if (
                not all(t in tokens for t in query_tokens)
                and raw not in text_field(s, "name").upper()
            ):
                continue
        results.append(summarize(s, category_names))
        if len(results) >= limit:
            break
    return results


def match_names(
    lines: list[str],
    streams: list[ProviderRecord],
    category_names: Mapping[str, str],
    category_prefix: str = "",
    candidates_per_line: int = 5,
) -> list[NameMatch]:
    """Fuzzy-match pasted channel names to streams.

    Args:
        lines: Channel names to match.
        streams: Available provider streams.
        category_names: Category names indexed by identifier.
        category_prefix: Restrict matching to categories with this prefix.
        candidates_per_line: Maximum distinct candidate names per input line.

    Returns:
        matches: Queries with ranked candidate stream metadata and scores.
    """
    aliases = {
        "FS1": "FOX SPORTS 1",
        "FS2": "FOX SPORTS 2",
        "MSNBC": "MS NOW",
        "ESPNU": "ESPN U",
        "ESPNEWS": "ESPN NEWS",
        "ESPN2": "ESPN 2",
        "ID": "INVESTIGATION DISCOVERY",
        "NICK": "NICKELODEON",
        "BTN": "BIG TEN NETWORK",
        "CBSSN": "CBS SPORTS NETWORK",
    }
    pool = [
        (s, normalize(s.get("name", "")))
        for s in _category_filter(streams, category_names, category_prefix)
    ]
    results: list[NameMatch] = []
    for line in lines:
        query = line.strip()
        if not query:
            continue
        variants = [normalize(query)]
        alias = aliases.get(" ".join(variants[0]))
        if alias is not None:
            variants.append(normalize(alias))
        scored: list[tuple[float, ProviderRecord]] = []
        for s, tokens in pool:
            score = max(_score(v, tokens) for v in variants)
            if score >= 50:
                scored.append((score, s))
        scored.sort(key=lambda item: -item[0])
        seen_names: set[str] = set()
        candidates: list[ScoredSummary] = []
        for score, s in scored:
            name = text_field(s, "name")
            if name in seen_names:
                continue
            seen_names.add(name)
            summary = summarize(s, category_names)
            candidates.append(
                {
                    "stream_id": summary["stream_id"],
                    "name": summary["name"],
                    "source_id": summary["source_id"],
                    "epg_channel_id": summary["epg_channel_id"],
                    "icon": summary["icon"],
                    "category": summary["category"],
                    "score": round(score, 1),
                },
            )
            if len(candidates) >= candidates_per_line:
                break
        results.append({"query": query, "candidates": candidates})
    return results


class _PlaylistFile(TypedDict):
    playlists: list[Playlist]


def _path() -> pathlib.Path:
    cache.initialize_paths()
    return cache.CACHE_DIR / "playlists.json"


def _clean_name(name: MutablePlainTree, *, max_length: int = 80) -> str:
    cleaned = ("" if name is None else str(name)).strip()[:max_length]
    if not cleaned:
        raise ValueError("Playlist name is required")
    return cleaned


def _clean_channel(entry: MutablePlainTree) -> PlaylistChannel | None:
    if not isinstance(entry, dict):
        return None
    stream_id = text_field(entry, "stream_id").strip()
    name = text_field(entry, "name").strip()
    if not stream_id and not name:
        return None
    return {
        "stream_id": stream_id,
        "name": name,
        "source_id": text_field(entry, "source_id").strip(),
        "epg_channel_id": text_field(entry, "epg_channel_id").strip(),
    }


def _clean_channels(
    channels: MutablePlainTree,
    *,
    max_channels: int = 2000,
) -> list[PlaylistChannel]:
    if not isinstance(channels, list):
        raise ValueError("channels must be a list")
    if len(channels) > max_channels:
        raise ValueError(f"A playlist can hold at most {max_channels} channels")
    cleaned: list[PlaylistChannel] = []
    seen: set[str] = set()
    for entry in channels:
        channel = _clean_channel(entry)
        if channel is None:
            continue
        source_id = channel.get("source_id", "")
        stream_id = channel.get("stream_id", "")
        key = (
            f"{source_id}:{stream_id}"
            if stream_id
            else f"name:{source_id}:{channel.get('name', '')}"
        )
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(channel)
    return cleaned


# Cached per stream list; the list itself is kept so its identity can't be reused.
def _stream_index(
    streams: list[ProviderRecord],
) -> tuple[
    dict[tuple[str, str], ProviderRecord],
    dict[str, ProviderRecord],
    dict[tuple[str, str], ProviderRecord],
    dict[tuple[str, str], ProviderRecord],
]:
    """Index streams by (source, id), bare id, (source, name) and (source, epg id)."""
    global _index
    with _index_lock:
        if _index is not None and _index[0] is streams:
            return _index[1], _index[2], _index[3], _index[4]
        by_id: dict[tuple[str, str], ProviderRecord] = {}
        by_bare_id: dict[str, ProviderRecord] = {}
        by_name: dict[tuple[str, str], ProviderRecord] = {}
        by_epg: dict[tuple[str, str], ProviderRecord] = {}
        for s in streams:
            source = text_field(s, "source_id")
            stream_id = text_field(s, "stream_id")
            by_id.setdefault((source, stream_id), s)
            by_bare_id.setdefault(stream_id, s)
            by_name.setdefault((source, text_field(s, "name")), s)
            epg_id = text_field(s, "epg_channel_id")
            if epg_id:
                by_epg.setdefault((source, epg_id), s)
        _index = (streams, by_id, by_bare_id, by_name, by_epg)
        return by_id, by_bare_id, by_name, by_epg


def _is_separator(name: str) -> bool:
    return "#####" in name or not re.search(r"[A-Za-z0-9]", name)


def _category_filter(
    streams: list[ProviderRecord],
    category_names: Mapping[str, str],
    category_prefix: str,
) -> Iterator[ProviderRecord]:
    prefix = category_prefix.strip().upper()
    for s in streams:
        if _is_separator(text_field(s, "name")):
            continue
        if prefix:
            names = [category_names.get(c, "") for c in category_ids(s)]
            if not any(n.upper().startswith(prefix) for n in names):
                continue
        yield s


def _score(query_tokens: list[str], stream_tokens: list[str]) -> float:
    if not query_tokens or not stream_tokens:
        return 0.0
    stream_set = set(stream_tokens)
    matched = sum(1 for t in query_tokens if t in stream_set)
    if matched == 0:
        return 0.0
    coverage = matched / len(query_tokens)
    extra = len(stream_set - set(query_tokens))
    score = coverage * 100 - extra * 3
    if stream_tokens == query_tokens:
        score += 50
    elif stream_tokens[: len(query_tokens)] == query_tokens:
        score += 20
    return score
