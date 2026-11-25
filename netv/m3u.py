"""M3U parsing, live/VOD/series data loading."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NotRequired, TypedDict
from urllib import parse

import collections
import logging
import re
import threading
import time

from treekle import MutablePlainTree, ReadError, from_plain

from netv.cache import (
    Source,
    adopt_stable_live_ids,
    get_cache,
    get_cache_lock,
    get_sources,
    get_user_agent,
    load_file_cache,
    relay_hints,
    remember_live_ts,
    save_file_cache,
    update_source_epg_url,
)
from netv.util import safe_urlopen
from netv.xtream import ProviderRecord, XtreamClient, text_field


class LiveData(TypedDict):
    """Live categories, streams, and guide URLs of every source."""

    cats: list[ProviderRecord]
    streams: list[ProviderRecord]
    epg_urls: list[tuple[str, int, str]]


class VodData(TypedDict):
    """VOD categories and streams of every Xtream source."""

    cats: list[ProviderRecord]
    streams: list[ProviderRecord]


class SeriesData(TypedDict):
    """Series categories and listings of every Xtream source."""

    cats: list[ProviderRecord]
    series: list[ProviderRecord]


log = logging.getLogger(__name__)

_refresh_in_progress: set[str] = set()
_fetch_locks: dict[str, threading.Lock] = {
    "live": threading.Lock(),
    "vod": threading.Lock(),
    "series": threading.Lock(),
}


def parse_m3u(
    content: str,
    source_id: str,
) -> tuple[list[ProviderRecord], list[ProviderRecord], str]:
    """Parse a source's M3U playlist.

    Args:
        content: M3U text.
        source_id: Namespace for category and stream identities.

    Returns:
        categories: Source-qualified playlist categories.
        streams: Stream metadata and direct playback URLs.
        epg_url: Advertised guide URL, or an empty string.
    """
    categories: dict[str, ProviderRecord] = {}
    streams: list[ProviderRecord] = []
    id_counts: collections.Counter[str] = collections.Counter()
    streams_with_epg = 0
    epg_url = ""

    lines = content.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#EXTM3U"):
            match = re.search(r'(?:url-tvg|x-tvg-url)="([^"]*)"', line)
            if match:
                # Typeshed types a group as ``str | Any``; this one always matched.
                epg_url = str(match.group(1))
        elif line.startswith("#EXTINF:"):
            attrs: dict[str, str] = {}
            match = re.search(r"#EXTINF:[^,]*,(.*)", line)
            name = match.group(1).strip() if match else "Unknown"

            for attr_match in re.finditer(r'(\w+[-\w]*)="([^"]*)"', line):
                attrs[attr_match.group(1)] = attr_match.group(2)

            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("#")):
                if lines[i].strip().startswith("#EXTINF:"):
                    break
                i += 1
            url = ""
            if i < len(lines) and not lines[i].strip().startswith("#EXTINF:"):
                url = lines[i].strip()
            else:
                # URL-less entry: leave the next #EXTINF for the outer loop.
                i -= 1

            group = attrs.get("group-title", "Uncategorized")
            cat_slug = re.sub(r"[^a-zA-Z0-9]+", "_", group).strip("_").lower()
            cat_id = f"{source_id}_{cat_slug}"
            if group not in categories:
                categories[group] = {
                    "category_id": cat_id,
                    "category_name": group,
                    "parent_id": 0,
                    "source_id": source_id,
                }

            # The URL's last path segment survives a lineup reorder, a tuner's new
            # address and a provider password change; a position survives none.
            key = parse.urlsplit(url).path.rpartition("/")[2] or name
            id_counts[key] += 1
            if id_counts[key] > 1:
                key = f"{key}~{id_counts[key]}"
            epg_channel_id = attrs.get("tvg-id", "")
            if epg_channel_id:
                streams_with_epg += 1
            streams.append(
                {
                    "stream_id": f"{source_id}:{key}",
                    "name": name,
                    "stream_icon": attrs.get("tvg-logo", ""),
                    "epg_channel_id": epg_channel_id,
                    "category_ids": [cat_id],
                    "direct_url": url,
                    "source_id": source_id,
                },
            )
        i += 1

    log.debug(
        "M3U parsed: %d streams (%d with tvg-id, %d without), %d categories",  # pragma: no mutate[diagnostic-text]
        len(streams),
        streams_with_epg,
        len(streams) - streams_with_epg,
        len(categories),
    )
    return list(categories.values()), streams, epg_url


def fetch_m3u(
    url: str,
    source_id: str,
    timeout: int = 30,
) -> tuple[list[ProviderRecord], list[ProviderRecord], str]:
    """Fetch and parse M3U from URL, return (categories, streams, epg_url)."""
    with safe_urlopen(url, timeout=timeout, user_agent=get_user_agent()) as resp:
        content = resp.read().decode("utf-8")  # pragma: no mutate[case-insensitive]
    categories, streams, epg_url = parse_m3u(content, source_id)
    adopt_stable_live_ids(source_id, [text_field(s, "stream_id") for s in streams])
    return categories, streams, epg_url


def fetch_source_live_data(
    source: Source,
    persist_epg_url: bool = True,
) -> tuple[list[ProviderRecord], list[ProviderRecord], str | None, int]:
    """Fetch live data for a single source.

    Args:
        source: Configured Xtream, M3U, or guide-only source.
        persist_epg_url: Save an automatically detected guide URL.

    Returns:
        cats: Source-qualified categories.
        streams: Source-qualified stream metadata.
        epg_url: Enabled guide URL, or None.
        epg_timeout: Source's guide request timeout in seconds.
    """
    cats: list[ProviderRecord] = []
    streams: list[ProviderRecord] = []
    epg_url: str | None = None

    if source.type == "xtream":
        client = _make_client(source)
        cats = client.get_live_categories()
        streams = client.get_live_streams()
        _learn_live_ts(source.id, client)
        for c in cats:
            c["source_id"] = source.id
            c["category_id"] = f"{source.id}_{c['category_id']}"
        for s in streams:
            s["source_id"] = source.id
            s["source_type"] = "xtream"
            s["source_url"] = source.url
            s["source_username"] = source.username
            s["source_password"] = source.password
            ids = s.get("category_ids")
            orig_cats = ids if isinstance(ids, list) and ids else [s.get("category_id")]
            s["category_ids"] = [
                f"{source.id}_{c}" for c in orig_cats if c is not None and c not in ("", 0)
            ]
        detected_epg = client.epg_url
        if persist_epg_url:
            update_source_epg_url(source.id, detected_epg)
        epg_url = detected_epg if source.epg_enabled else None
    elif source.type == "m3u":
        cats, streams, detected_epg = fetch_m3u(source.url, source.id)
        if persist_epg_url:
            update_source_epg_url(source.id, detected_epg)
        epg_url = detected_epg if source.epg_enabled else None
    elif source.type == "epg":
        epg_url = source.url

    return cats, streams, epg_url, source.epg_timeout


def fetch_source_vod_data(source: Source) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
    """Fetch VOD data for a single Xtream source.

    Args:
        source: Provider configuration.

    Returns:
        cats: VOD categories, empty for non-Xtream sources.
        streams: VOD streams, empty for non-Xtream sources.
    """
    if source.type != "xtream":
        return [], []
    client = _make_client(source)
    cats = client.get_vod_categories()
    streams = client.get_vod_streams()
    # Tag with source_id for playback.
    for c in cats:
        c["source_id"] = source.id
    for s in streams:
        s["source_id"] = source.id
    return cats, streams


def parse_epg_urls(raw: Sequence[object]) -> list[tuple[str, int, str]]:
    """Convert JSON lists back to guide URL tuples, skipping malformed ones.

    Args:
        raw: Stored entries, each ``[url, timeout, source_id]``.

    Returns:
        epg_urls: The well-formed entries as ``(url, timeout, source_id)``.
    """
    epg_urls: list[tuple[str, int, str]] = []
    for entry in raw:
        match entry:
            case [str() as url, int() as timeout, str() as source_id, *_]:
                epg_urls.append((url, timeout, source_id))
            case _:
                pass
    return epg_urls


def cached_live_data() -> tuple[LiveData, float] | None:
    """Read the live data file cache.

    Returns:
        cached: Data and write time, or None when absent or unreadable.
    """
    cached = load_file_cache("live_data")
    if cached is None:
        return None
    tree, timestamp = cached
    try:
        data = from_plain(tree, _LiveFile)
    except ReadError:
        return None
    return {
        "cats": data["cats"],
        "streams": data["streams"],
        "epg_urls": parse_epg_urls(data.get("epg_urls", [])),
    }, timestamp


def cached_vod_data() -> tuple[VodData, float] | None:
    """Read the VOD data file cache.

    Returns:
        cached: Data and write time, or None when absent or unreadable.
    """
    cached = load_file_cache("vod_data")
    if cached is None:
        return None
    tree, timestamp = cached
    try:
        return from_plain(tree, VodData), timestamp
    except ReadError:
        return None


def cached_series_data() -> tuple[SeriesData, float] | None:
    """Read the series data file cache.

    Returns:
        cached: Data and write time, or None when absent or unreadable.
    """
    cached = load_file_cache("series_data")
    if cached is None:
        return None
    tree, timestamp = cached
    try:
        return from_plain(tree, SeriesData), timestamp
    except ReadError:
        return None


def load_all_live_data(
    *,
    ttl_seconds: float = 2 * 3600,
) -> tuple[list[ProviderRecord], list[ProviderRecord], list[tuple[str, int, str]]]:
    """Load live data with file cache and stale-while-revalidate.

    Args:
        ttl_seconds: Cache age triggering background refresh.

    Returns:
        cats: Live categories across sources.
        streams: Live streams across sources.
        epg_urls: Guide URLs with request timeouts and source IDs.
    """
    _cache = get_cache()
    _cache_lock = get_cache_lock()
    cached = cached_live_data()
    now = time.time()

    if cached is not None:
        data, ts = cached
        age = now - ts

        if age > ttl_seconds and "live" not in _refresh_in_progress:
            _refresh_in_progress.add("live")

            def refresh() -> None:
                try:
                    log.info("Refreshing live data in background")
                    new_cats, new_streams, new_epg_urls, failed = _fetch_all_live_data()
                    if failed and not (new_cats or new_streams):
                        log.warning("Live refresh fetched nothing; keeping cached channels")
                        return
                    epg_urls = new_epg_urls + [e for e in data["epg_urls"] if e[2] in failed]
                    save_file_cache(
                        "live_data",
                        {
                            "cats": new_cats + _kept_records(data["cats"], failed),
                            "streams": new_streams + _kept_records(data["streams"], failed),
                            "epg_urls": epg_urls,
                        },
                    )
                    with _cache_lock:
                        _cache.pop("live_categories", None)
                        _cache.pop("live_streams", None)
                        _cache["epg_urls"] = epg_urls
                    log.info("Live data refreshed")
                finally:
                    _refresh_in_progress.discard("live")

            threading.Thread(target=refresh, daemon=True).start()

        return data["cats"], data["streams"], data["epg_urls"]

    with _fetch_locks["live"]:
        cached = cached_live_data()
        if cached is not None:
            data, _ = cached
            return data["cats"], data["streams"], data["epg_urls"]
        log.info("No live cache, fetching")
        cats, streams, epg_urls, _ = _fetch_all_live_data()
        if cats or streams:
            save_file_cache("live_data", {"cats": cats, "streams": streams, "epg_urls": epg_urls})
        return cats, streams, epg_urls


def load_vod_data(
    *,
    ttl_seconds: float = 12 * 3600,
) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
    """Load VOD data with file cache and stale-while-revalidate.

    Args:
        ttl_seconds: Cache age triggering background refresh.

    Returns:
        cats: VOD categories across sources.
        streams: VOD streams across sources.
    """
    _cache = get_cache()
    _cache_lock = get_cache_lock()
    cached = cached_vod_data()
    now = time.time()

    if cached is not None:
        data, ts = cached
        age = now - ts

        if age > ttl_seconds and "vod" not in _refresh_in_progress:
            _refresh_in_progress.add("vod")

            def refresh() -> None:
                try:
                    log.info("Refreshing VOD data in background")
                    new_cats, new_streams, failed = _fetch_vod_data()
                    if failed and not (new_cats or new_streams):
                        log.warning("VOD refresh fetched nothing; keeping cached movies")
                        return
                    save_file_cache(
                        "vod_data",
                        {
                            "cats": new_cats + _kept_records(data["cats"], failed),
                            "streams": new_streams + _kept_records(data["streams"], failed),
                        },
                    )
                    with _cache_lock:
                        _cache.pop("vod_categories", None)
                        _cache.pop("vod_streams", None)
                    log.info("VOD data refreshed")
                finally:
                    _refresh_in_progress.discard("vod")

            threading.Thread(target=refresh, daemon=True).start()

        return data["cats"], data["streams"]

    with _fetch_locks["vod"]:
        cached = cached_vod_data()
        if cached is not None:
            data, _ = cached
            return data["cats"], data["streams"]
        log.info("No VOD cache, fetching")
        cats, streams, _ = _fetch_vod_data()
        if cats or streams:
            save_file_cache("vod_data", {"cats": cats, "streams": streams})
        return cats, streams


def load_series_data(
    *,
    ttl_seconds: float = 12 * 3600,
) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
    """Load series data with file cache and stale-while-revalidate.

    Args:
        ttl_seconds: Cache age triggering background refresh.

    Returns:
        cats: Series categories across sources.
        series: Series metadata across sources.
    """
    _cache = get_cache()
    _cache_lock = get_cache_lock()
    cached = cached_series_data()
    now = time.time()

    if cached is not None:
        data, ts = cached
        age = now - ts

        if age > ttl_seconds and "series" not in _refresh_in_progress:
            _refresh_in_progress.add("series")

            def refresh() -> None:
                try:
                    log.info("Refreshing series data in background")
                    new_cats, new_series, failed = _fetch_series_data()
                    if failed and not (new_cats or new_series):
                        log.warning("Series refresh fetched nothing; keeping cached series")
                        return
                    save_file_cache(
                        "series_data",
                        {
                            "cats": new_cats + _kept_records(data["cats"], failed),
                            "series": new_series + _kept_records(data["series"], failed),
                        },
                    )
                    with _cache_lock:
                        _cache.pop("series_categories", None)
                        _cache.pop("series", None)
                    log.info("Series data refreshed")
                finally:
                    _refresh_in_progress.discard("series")

            threading.Thread(target=refresh, daemon=True).start()

        return data["cats"], data["series"]

    with _fetch_locks["series"]:
        cached = cached_series_data()
        if cached is not None:
            data, _ = cached
            return data["cats"], data["series"]
        log.info("No series cache, fetching")
        cats, series, _ = _fetch_series_data()
        if cats or series:
            save_file_cache("series_data", {"cats": cats, "series": series})
        return cats, series


def get_first_xtream_client() -> XtreamClient | None:
    """Get the first available Xtream client (for VOD/series)."""
    for source in get_sources():
        if source.type == "xtream":
            return _make_client(source)
    return None


def live_ts_offered(source_id: str) -> bool:
    """Whether an Xtream source's panel serves live channels as continuous MPEG-TS.

    Learned at every live refresh; a source not refreshed since is asked now, once.

    Args:
        source_id: Configured source.

    Returns:
        offered: True when the panel lists ``ts`` among its output formats.
    """
    known = relay_hints()["live_ts"].get(source_id)
    if known is not None:
        return known
    client = get_xtream_client_by_source(source_id)
    return client is not None and _learn_live_ts(source_id, client)


def get_xtream_client_by_source(source_id: str) -> XtreamClient | None:
    """Get Xtream client for a specific source ID."""
    for source in get_sources():
        if source.id == source_id and source.type == "xtream":
            return _make_client(source)
    return None


def get_refresh_in_progress() -> set[str]:
    """Get refresh in progress set."""
    return _refresh_in_progress


class _LiveFile(TypedDict):
    cats: list[ProviderRecord]
    streams: list[ProviderRecord]
    epg_urls: NotRequired[list[MutablePlainTree]]


def _make_client(source: Source) -> XtreamClient:
    """Build an Xtream client using the configured User-Agent."""
    return XtreamClient(
        base_url=source.url,
        username=source.username,
        password=source.password,
        user_agent=get_user_agent(),
    )


def _kept_records(records: list[ProviderRecord], source_ids: set[str]) -> list[ProviderRecord]:
    """The ``records`` from ``source_ids``: what a refresh keeps of sources that failed."""
    return [r for r in records if r.get("source_id") in source_ids]


def _fetch_all_live_data() -> tuple[
    list[ProviderRecord],
    list[ProviderRecord],
    list[tuple[str, int, str]],
    set[str],
]:
    """Fetch live categories/streams from all sources; also name the sources that failed."""
    all_categories: list[ProviderRecord] = []
    all_streams: list[ProviderRecord] = []
    epg_urls: list[tuple[str, int, str]] = []
    failed: set[str] = set()

    for source in get_sources():
        try:
            cats, streams, epg_url, timeout = fetch_source_live_data(source, persist_epg_url=False)
            all_categories.extend(cats)
            all_streams.extend(streams)
            if epg_url is not None and epg_url and source.epg_enabled:
                epg_urls.append((epg_url, timeout, source.id))
        except Exception as e:
            failed.add(source.id)
            log.error("Error loading source %s: %s", source.name, e)

    return all_categories, all_streams, epg_urls, failed


def _fetch_vod_data() -> tuple[list[ProviderRecord], list[ProviderRecord], set[str]]:
    """Fetch VOD categories and streams from all Xtream sources; name those that failed."""
    all_cats: list[ProviderRecord] = []
    all_streams: list[ProviderRecord] = []
    failed: set[str] = set()
    for source in get_sources():
        if source.type != "xtream":
            continue
        try:
            cats, streams = fetch_source_vod_data(source)
            all_cats.extend(cats)
            all_streams.extend(streams)
        except Exception as e:
            failed.add(source.id)
            log.warning("Failed to fetch VOD from source %s: %s", source.id, e)
    return all_cats, all_streams, failed


def _fetch_series_data() -> tuple[list[ProviderRecord], list[ProviderRecord], set[str]]:
    """Fetch series categories and list from all Xtream sources; name those that failed."""
    all_cats: list[ProviderRecord] = []
    all_series: list[ProviderRecord] = []
    failed: set[str] = set()
    for source in get_sources():
        if source.type != "xtream":
            continue
        try:
            client = _make_client(source)
            cats = client.get_series_categories()
            series = client.get_series()
            # Tag with source_id for playback and access control.
            for c in cats:
                c["source_id"] = source.id
            for s in series:
                s["source_id"] = source.id
            all_cats.extend(cats)
            all_series.extend(series)
        except Exception as e:
            failed.add(source.id)
            log.warning("Failed to fetch series from source %s: %s", source.id, e)
    return all_cats, all_series, failed


def _learn_live_ts(source_id: str, client: XtreamClient) -> bool:
    """Ask the panel for its output formats and remember whether TS is among them."""
    try:
        user_info = client.get_server_info().get("user_info")
    except (OSError, ValueError):
        log.warning("Xtream source %s did not report its output formats", source_id)
        return False
    formats = user_info.get("allowed_output_formats") if isinstance(user_info, dict) else None
    offered = isinstance(formats, list) and "ts" in formats
    remember_live_ts(source_id, offered)
    return offered
