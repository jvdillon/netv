"""M3U parsing, live/VOD/series data loading."""

from __future__ import annotations

from typing import Any

import logging
import re
import threading
import time

from cache import (
    LIVE_CACHE_TTL,
    SERIES_CACHE_TTL,
    VOD_CACHE_TTL,
    get_cache,
    get_cache_lock,
    get_sources,
    load_file_cache,
    save_file_cache,
    update_source_epg_url,
)
from util import safe_urlopen
from xtream import XtreamClient


log = logging.getLogger(__name__)

_refresh_in_progress: set[str] = set()
_fetch_locks: dict[str, threading.Lock] = {
    "live": threading.Lock(),
    "vod": threading.Lock(),
    "series": threading.Lock(),
    "epg": threading.Lock(),
}


def parse_m3u(content: str, source_id: str) -> tuple[list[dict], list[dict], str]:
    """Parse M3U content, return (categories, streams, epg_url)."""
    categories: dict[str, dict] = {}
    streams: list[dict] = []
    stream_id_counter = 0
    epg_url = ""

    lines = content.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#EXTM3U"):
            match = re.search(r'(?:url-tvg|x-tvg-url)="([^"]*)"', line)
            if match:
                epg_url = match.group(1)
        elif line.startswith("#EXTINF:"):
            attrs: dict[str, str] = {}
            match = re.search(r"#EXTINF:[^,]*,(.*)", line)
            name = match.group(1).strip() if match else "Unknown"

            for attr_match in re.finditer(r'(\w+[-\w]*)="([^"]*)"', line):
                attrs[attr_match.group(1)] = attr_match.group(2)

            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("#")):
                i += 1
            url = lines[i].strip() if i < len(lines) else ""

            group = attrs.get("group-title", "Uncategorized")
            if group not in categories:
                cat_slug = re.sub(r"[^a-zA-Z0-9]+", "_", group).strip("_").lower()
                cat_id = f"{source_id}_{cat_slug}"
                categories[group] = {
                    "category_id": cat_id,
                    "category_name": group,
                    "parent_id": 0,
                    "source_id": source_id,
                }

            stream_id_counter += 1
            streams.append(
                {
                    "stream_id": f"{source_id}_{stream_id_counter}",
                    "name": name,
                    "stream_icon": attrs.get("tvg-logo", ""),
                    "epg_channel_id": attrs.get("tvg-id", ""),
                    "category_ids": [categories[group]["category_id"]],
                    "direct_url": url,
                    "source_id": source_id,
                }
            )
        i += 1

    streams_with_epg = sum(1 for s in streams if s.get("epg_channel_id"))
    log.debug(
        "M3U parsed: %d streams (%d with tvg-id, %d without), %d categories",
        len(streams),
        streams_with_epg,
        len(streams) - streams_with_epg,
        len(categories),
    )
    return list(categories.values()), streams, epg_url


def fetch_m3u(url: str, source_id: str, timeout: int = 30) -> tuple[list[dict], list[dict], str]:
    """Fetch and parse M3U from URL, return (categories, streams, epg_url)."""
    with safe_urlopen(url, timeout=timeout) as resp:
        content = resp.read().decode("utf-8")
    return parse_m3u(content, source_id)


def _fetch_all_live_data() -> tuple[list[dict], list[dict], list[tuple[str, int, str]]]:
    """Fetch live categories/streams from all sources."""
    all_categories: list[dict] = []
    all_streams: list[dict] = []
    epg_urls: list[tuple[str, int, str]] = []

    for source in get_sources():
        try:
            if source.type == "xtream":
                client = XtreamClient(source.url, source.username, source.password)
                cats = client.get_live_categories()
                streams = client.get_live_streams()
                for c in cats:
                    c["source_id"] = source.id
                    c["category_id"] = f"{source.id}_{c['category_id']}"
                for s in streams:
                    s["source_id"] = source.id
                    s["source_type"] = "xtream"
                    s["source_url"] = source.url
                    s["source_username"] = source.username
                    s["source_password"] = source.password
                    orig_cats = s.get("category_ids") or [s.get("category_id")]
                    s["category_ids"] = [f"{source.id}_{c}" for c in orig_cats if c]
                all_categories.extend(cats)
                all_streams.extend(streams)
                if source.epg_enabled:
                    epg_urls.append((client.epg_url, source.epg_timeout, source.id))
            elif source.type == "m3u":
                cats, streams, epg_url = fetch_m3u(source.url, source.id)
                all_categories.extend(cats)
                all_streams.extend(streams)
                if epg_url and source.epg_enabled:
                    epg_urls.append((epg_url, source.epg_timeout, source.id))
            elif source.type == "epg":
                if source.epg_enabled:
                    epg_urls.append((source.url, source.epg_timeout, source.id))
        except Exception as e:
            log.error("Error loading source %s: %s", source.name, e)

    return all_categories, all_streams, epg_urls


def fetch_source_live_data(source: Any) -> tuple[list[dict], list[dict], str | None, int]:
    """Fetch live data for a single source. Returns (cats, streams, epg_url, epg_timeout)."""
    cats: list[dict] = []
    streams: list[dict] = []
    epg_url: str | None = None

    if source.type == "xtream":
        client = XtreamClient(source.url, source.username, source.password)
        cats = client.get_live_categories()
        streams = client.get_live_streams()
        for c in cats:
            c["source_id"] = source.id
            c["category_id"] = f"{source.id}_{c['category_id']}"
        for s in streams:
            s["source_id"] = source.id
            s["source_type"] = "xtream"
            s["source_url"] = source.url
            s["source_username"] = source.username
            s["source_password"] = source.password
            orig_cats = s.get("category_ids") or [s.get("category_id")]
            s["category_ids"] = [f"{source.id}_{c}" for c in orig_cats if c]
        detected_epg = client.epg_url
        update_source_epg_url(source.id, detected_epg)
        epg_url = detected_epg if source.epg_enabled else None
    elif source.type == "m3u":
        cats, streams, detected_epg = fetch_m3u(source.url, source.id)
        update_source_epg_url(source.id, detected_epg)
        epg_url = detected_epg if source.epg_enabled else None
    elif source.type == "epg":
        epg_url = source.url

    return cats, streams, epg_url, source.epg_timeout


def fetch_source_vod_data(source: Any) -> tuple[list[dict], list[dict]]:
    """Fetch VOD data for a single Xtream source."""
    if source.type != "xtream":
        return [], []
    client = XtreamClient(source.url, source.username, source.password)
    return client.get_vod_categories(), client.get_vod_streams()


def parse_epg_urls(raw: list) -> list[tuple[str, int, str]]:
    """Convert JSON list back to tuples (JSON stores tuples as lists)."""
    return [(u[0], u[1], u[2]) for u in raw if isinstance(u, (list, tuple)) and len(u) >= 3]


def load_all_live_data() -> tuple[list[dict], list[dict], list[tuple[str, int, str]]]:
    """Load live data with file cache and stale-while-revalidate."""
    _cache = get_cache()
    _cache_lock = get_cache_lock()
    cached = load_file_cache("live_data")
    now = time.time()

    if cached:
        data, ts = cached
        cats, streams = data["cats"], data["streams"]
        epg_urls = parse_epg_urls(data.get("epg_urls", []))
        age = now - ts

        if age > LIVE_CACHE_TTL and "live" not in _refresh_in_progress:
            _refresh_in_progress.add("live")

            def refresh() -> None:
                try:
                    log.info("Refreshing live data in background")
                    new_cats, new_streams, new_epg_urls = _fetch_all_live_data()
                    save_file_cache(
                        "live_data",
                        {"cats": new_cats, "streams": new_streams, "epg_urls": new_epg_urls},
                    )
                    with _cache_lock:
                        _cache.pop("live_categories", None)
                        _cache.pop("live_streams", None)
                        _cache["epg_urls"] = new_epg_urls
                    log.info("Live data refreshed")
                finally:
                    _refresh_in_progress.discard("live")

            threading.Thread(target=refresh, daemon=True).start()

        return cats, streams, epg_urls

    with _fetch_locks["live"]:
        cached = load_file_cache("live_data")
        if cached:
            data, _ = cached
            return data["cats"], data["streams"], parse_epg_urls(data.get("epg_urls", []))
        log.info("No live cache, fetching")
        cats, streams, epg_urls = _fetch_all_live_data()
        save_file_cache("live_data", {"cats": cats, "streams": streams, "epg_urls": epg_urls})
        return cats, streams, epg_urls


def _fetch_vod_data() -> tuple[list[dict], list[dict]]:
    """Fetch VOD categories and streams from first Xtream source."""
    source_id, xtream = get_first_xtream_source_and_client()
    if not xtream:
        return [], []
    cats = xtream.get_vod_categories()
    streams = xtream.get_vod_streams()
    # Add source_id for access control
    for c in cats:
        c["source_id"] = source_id
    for s in streams:
        s["source_id"] = source_id
    return cats, streams


def load_vod_data() -> tuple[list[dict], list[dict]]:
    """Load VOD data with file cache and stale-while-revalidate."""
    _cache = get_cache()
    _cache_lock = get_cache_lock()
    cached = load_file_cache("vod_data")
    now = time.time()

    if cached:
        data, ts = cached
        cats, streams = data["cats"], data["streams"]
        age = now - ts

        if age > VOD_CACHE_TTL and "vod" not in _refresh_in_progress:
            _refresh_in_progress.add("vod")

            def refresh() -> None:
                try:
                    log.info("Refreshing VOD data in background")
                    new_cats, new_streams = _fetch_vod_data()
                    save_file_cache("vod_data", {"cats": new_cats, "streams": new_streams})
                    with _cache_lock:
                        _cache.pop("vod_categories", None)
                        _cache.pop("vod_streams", None)
                    log.info("VOD data refreshed")
                finally:
                    _refresh_in_progress.discard("vod")

            threading.Thread(target=refresh, daemon=True).start()

        return cats, streams

    with _fetch_locks["vod"]:
        cached = load_file_cache("vod_data")
        if cached:
            data, _ = cached
            return data["cats"], data["streams"]
        log.info("No VOD cache, fetching")
        cats, streams = _fetch_vod_data()
        if cats or streams:
            save_file_cache("vod_data", {"cats": cats, "streams": streams})
        return cats, streams


def _fetch_series_data() -> tuple[list[dict], list[dict]]:
    """Fetch series categories and list from first Xtream source."""
    source_id, xtream = get_first_xtream_source_and_client()
    if not xtream:
        return [], []
    cats = xtream.get_series_categories()
    series = xtream.get_series()
    # Add source_id for access control
    for c in cats:
        c["source_id"] = source_id
    for s in series:
        s["source_id"] = source_id
    return cats, series


def load_series_data() -> tuple[list[dict], list[dict]]:
    """Load series data with file cache and stale-while-revalidate."""
    _cache = get_cache()
    _cache_lock = get_cache_lock()
    cached = load_file_cache("series_data")
    now = time.time()

    if cached:
        data, ts = cached
        cats, series = data["cats"], data["series"]
        age = now - ts

        if age > SERIES_CACHE_TTL and "series" not in _refresh_in_progress:
            _refresh_in_progress.add("series")

            def refresh() -> None:
                try:
                    log.info("Refreshing series data in background")
                    new_cats, new_series = _fetch_series_data()
                    save_file_cache("series_data", {"cats": new_cats, "series": new_series})
                    with _cache_lock:
                        _cache.pop("series_categories", None)
                        _cache.pop("series", None)
                    log.info("Series data refreshed")
                finally:
                    _refresh_in_progress.discard("series")

            threading.Thread(target=refresh, daemon=True).start()

        return cats, series

    with _fetch_locks["series"]:
        cached = load_file_cache("series_data")
        if cached:
            data, _ = cached
            return data["cats"], data["series"]
        log.info("No series cache, fetching")
        cats, series = _fetch_series_data()
        if cats or series:
            save_file_cache("series_data", {"cats": cats, "series": series})
        return cats, series


def get_first_xtream_client() -> XtreamClient | None:
    """Get the first available Xtream client (for VOD/series)."""
    for source in get_sources():
        if source.type == "xtream":
            return XtreamClient(source.url, source.username, source.password)
    return None


def get_first_xtream_source_and_client() -> tuple[str, XtreamClient] | tuple[None, None]:
    """Get the first available Xtream source ID and client."""
    for source in get_sources():
        if source.type == "xtream":
            return source.id, XtreamClient(source.url, source.username, source.password)
    return None, None


def get_fetch_lock(name: str) -> threading.Lock:
    """Get fetch lock by name."""
    return _fetch_locks[name]


def get_refresh_in_progress() -> set[str]:
    """Get refresh in progress set."""
    return _refresh_in_progress
