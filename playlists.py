"""Shared Live TV playlists.

Admins curate named, ordered channel lists in the web UI. Playlists surface in
the guide as virtual categories (``pl:<id>``) so every client, including the
Apple apps, can show them without editing support.
"""

from __future__ import annotations

from typing import Any, Final

import json
import re
import threading
import unicodedata
import uuid

from util import atomic_write_json

import cache


PREFIX: Final = "pl:"

_lock = threading.Lock()


def is_playlist_id(category_id: object) -> bool:
    return str(category_id).startswith(PREFIX)


def stream_key(stream: dict) -> str:
    """Stream ids are only unique within a source, so qualify them."""
    return f"{stream.get('source_id') or ''}:{stream.get('stream_id', '')}"


def load() -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    playlists = data.get("playlists", []) if isinstance(data, dict) else []
    return [p for p in playlists if isinstance(p, dict) and p.get("id")]


def save(playlists: list[dict[str, Any]]) -> None:
    atomic_write_json(_path(), {"playlists": playlists})


def get(playlist_id: str) -> dict[str, Any] | None:
    return next((p for p in load() if p["id"] == playlist_id), None)


def create(name: str, channels: Any = None, *, max_playlists: int = 100) -> dict[str, Any]:
    playlist = {
        "id": uuid.uuid4().hex[:12],
        "name": _clean_name(name),
        "channels": _clean_channels(channels) if channels is not None else [],
    }
    with _lock:
        playlists = load()
        if len(playlists) >= max_playlists:
            raise ValueError(f"At most {max_playlists} playlists are allowed")
        playlists.append(playlist)
        save(playlists)
        return playlist


def update(playlist_id: str, name: Any = None, channels: Any = None) -> dict[str, Any] | None:
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


def reorder(ids: list[str]) -> list[dict[str, Any]]:
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
        list[dict],
        dict[tuple[str, str], dict],
        dict[str, dict],
        dict[tuple[str, str], dict],
        dict[tuple[str, str], dict],
    ]
    | None
) = None


def resolve_channel(entry: dict, streams: list[dict]) -> dict | None:
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


def with_entry_epg(entry: dict, stream: dict) -> dict:
    """Apply a curated guide identity without mutating the provider stream.

    Args:
        entry: Saved playlist channel metadata.
        stream: Resolved provider stream.

    Returns:
        stream: Stream metadata with any curated guide identity applied.
    """
    epg_id = entry.get("epg_channel_id") or ""
    if epg_id:
        return {**stream, "epg_channel_id": epg_id}
    return stream


def resolve(playlist: dict, streams: list[dict]) -> list[dict]:
    """Resolve available streams in playlist order without duplicates.

    Args:
        playlist: Saved playlist with ordered channel entries.
        streams: Current provider streams.

    Returns:
        resolved: Available streams with curated guide identities.
    """
    resolved: list[dict] = []
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


def summarize(stream: dict, category_names: dict[str, str]) -> dict[str, Any]:
    category_ids = [str(c) for c in (stream.get("category_ids") or [])]
    category = next((category_names[c] for c in category_ids if c in category_names), "")
    return {
        "stream_id": str(stream.get("stream_id", "")),
        "name": stream.get("name", ""),
        "source_id": str(stream.get("source_id") or ""),
        "epg_channel_id": stream.get("epg_channel_id") or "",
        "icon": stream.get("stream_icon") or "",
        "category": category,
    }


def search(
    streams: list[dict],
    category_names: dict[str, str],
    query: str = "",
    category_id: str = "",
    category_prefix: str = "",
    limit: int = 200,
) -> list[dict]:
    query_tokens = normalize(query) if query else []
    raw = query.strip().upper()
    results = []
    for s in _category_filter(streams, category_names, category_prefix):
        if category_id and category_id not in {str(c) for c in (s.get("category_ids") or [])}:
            continue
        if query_tokens:
            tokens = set(normalize(s.get("name", "")))
            if (
                not all(t in tokens for t in query_tokens)
                and raw not in str(s.get("name", "")).upper()
            ):
                continue
        results.append(summarize(s, category_names))
        if len(results) >= limit:
            break
    return results


def match_names(
    lines: list[str],
    streams: list[dict],
    category_names: dict[str, str],
    category_prefix: str = "",
    candidates_per_line: int = 5,
) -> list[dict[str, Any]]:
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
    results = []
    for line in lines:
        query = line.strip()
        if not query:
            continue
        variants = [normalize(query)]
        alias = aliases.get(" ".join(variants[0]))
        if alias:
            variants.append(normalize(alias))
        scored = []
        for s, tokens in pool:
            score = max(_score(v, tokens) for v in variants)
            if score >= 50:
                scored.append((score, s))
        scored.sort(key=lambda item: -item[0])
        seen_names: set[str] = set()
        candidates = []
        for score, s in scored:
            name = str(s.get("name", ""))
            if name in seen_names:
                continue
            seen_names.add(name)
            candidates.append({**summarize(s, category_names), "score": round(score, 1)})
            if len(candidates) >= candidates_per_line:
                break
        results.append({"query": query, "candidates": candidates})
    return results


def _path():
    cache.initialize_paths()
    return cache.CACHE_DIR / "playlists.json"


def _clean_name(name: Any, *, max_length: int = 80) -> str:
    name = str(name or "").strip()[:max_length]
    if not name:
        raise ValueError("Playlist name is required")
    return name


def _clean_channel(entry: Any) -> dict[str, str] | None:
    if not isinstance(entry, dict):
        return None
    stream_id = str(entry.get("stream_id", "")).strip()
    name = str(entry.get("name", "")).strip()
    if not stream_id and not name:
        return None
    return {
        "stream_id": stream_id,
        "name": name,
        "source_id": str(entry.get("source_id") or "").strip(),
        "epg_channel_id": str(entry.get("epg_channel_id") or "").strip(),
    }


def _clean_channels(channels: Any, *, max_channels: int = 2000) -> list[dict[str, str]]:
    if not isinstance(channels, list):
        raise ValueError("channels must be a list")
    if len(channels) > max_channels:
        raise ValueError(f"A playlist can hold at most {max_channels} channels")
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in channels:
        channel = _clean_channel(entry)
        if not channel:
            continue
        key = (
            f"{channel['source_id']}:{channel['stream_id']}"
            if channel["stream_id"]
            else f"name:{channel['source_id']}:{channel['name']}"
        )
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(channel)
    return cleaned


# Cached per stream list; the list itself is kept so its identity can't be reused.
def _stream_index(streams: list[dict]):
    """Index streams by (source, id), bare id, (source, name) and (source, epg id)."""
    global _index
    with _index_lock:
        if _index is not None and _index[0] is streams:
            return _index[1:]
        by_id: dict[tuple[str, str], dict] = {}
        by_bare_id: dict[str, dict] = {}
        by_name: dict[tuple[str, str], dict] = {}
        by_epg: dict[tuple[str, str], dict] = {}
        for s in streams:
            source = str(s.get("source_id") or "")
            stream_id = str(s.get("stream_id", ""))
            by_id.setdefault((source, stream_id), s)
            by_bare_id.setdefault(stream_id, s)
            by_name.setdefault((source, str(s.get("name", ""))), s)
            epg_id = s.get("epg_channel_id") or ""
            if epg_id:
                by_epg.setdefault((source, epg_id), s)
        _index = (streams, by_id, by_bare_id, by_name, by_epg)
        return _index[1:]


def _is_separator(name: str) -> bool:
    return "#####" in name or not re.search(r"[A-Za-z0-9]", name)


def _category_filter(streams: list[dict], category_names: dict[str, str], category_prefix: str):
    prefix = category_prefix.strip().upper()
    for s in streams:
        name = str(s.get("name", ""))
        if _is_separator(name):
            continue
        if prefix:
            names = [category_names.get(str(c), "") for c in (s.get("category_ids") or [])]
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
