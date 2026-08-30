"""Movie and series catalogs for the native-player gateway."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

import json
import logging
import threading
import time

from gateway_catalog import RegistryError, StreamIdRegistry
from m3u import load_series_data, load_vod_data

import cache


log = logging.getLogger(__name__)
MediaKind = Literal["movie", "series"]
_PRIVATE_FIELDS = {
    "direct_source",
    "direct_url",
    "source_id",
    "source_password",
    "source_url",
    "source_username",
}


@dataclass(frozen=True, slots=True)
class GatewayMediaItem:
    """A public movie or series entry and its private upstream identity."""

    local_id: int
    source_id: str
    upstream_id: str
    category_ids: tuple[str, ...]
    payload: bytes

    @property
    def public(self) -> dict[str, Any]:
        value = json.loads(self.payload)
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True, slots=True)
class MediaCatalogSnapshot:
    categories: list[dict[str, Any]]
    category_source_ids: dict[str, str]
    items: list[GatewayMediaItem]
    items_by_id: dict[int, GatewayMediaItem]
    category_id_map: dict[tuple[str, str], str] = field(default_factory=dict)


def _public_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_value(item)
            for key, item in value.items()
            if key not in _PRIVATE_FIELDS
        }
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    return deepcopy(value)


def _public_copy(value: dict[str, Any]) -> dict[str, Any]:
    sanitized = _public_value(value)
    return sanitized if isinstance(sanitized, dict) else {}


class GatewayMediaCatalog:
    """Load one type of on-demand catalog only when a client requests it."""

    def __init__(
        self,
        kind: MediaKind,
        registry: StreamIdRegistry,
        ttl_seconds: int | None = None,
    ) -> None:
        self.kind = kind
        self._registry = registry
        self._ttl_seconds = ttl_seconds or (
            cache.VOD_CACHE_TTL if kind == "movie" else cache.SERIES_CACHE_TTL
        )
        self._lock = threading.Lock()
        self._snapshot: MediaCatalogSnapshot | None = None
        self._loaded_at = 0.0
        self._cache_mtime_ns = 0

    @property
    def id_field(self) -> str:
        return "stream_id" if self.kind == "movie" else "series_id"

    def _current_cache_mtime_ns(self) -> int:
        cache_name = "vod_data.json" if self.kind == "movie" else "series_data.json"
        try:
            return (cache.CACHE_DIR / cache_name).stat().st_mtime_ns
        except OSError:
            return 0

    def get(self, force: bool = False) -> MediaCatalogSnapshot:
        cache_mtime_ns = self._current_cache_mtime_ns()
        if (
            not force
            and self._snapshot is not None
            and cache_mtime_ns == self._cache_mtime_ns
            and time.monotonic() - self._loaded_at < self._ttl_seconds
        ):
            return self._snapshot
        with self._lock:
            cache_mtime_ns = self._current_cache_mtime_ns()
            if (
                not force
                and self._snapshot is not None
                and cache_mtime_ns == self._cache_mtime_ns
                and time.monotonic() - self._loaded_at < self._ttl_seconds
            ):
                return self._snapshot
            try:
                cache_mtime_before = self._current_cache_mtime_ns()
                categories, items = (
                    load_vod_data() if self.kind == "movie" else load_series_data()
                )
                candidate = self._map(categories, items)
            except (KeyError, OSError, RegistryError, TypeError, ValueError) as exc:
                if self._snapshot is None:
                    raise
                log.warning(
                    "Gateway %s catalog refresh failed; retaining previous snapshot (%s)",
                    self.kind,
                    type(exc).__name__,
                )
                return self._snapshot
            self._snapshot = candidate
            self._loaded_at = time.monotonic()
            self._cache_mtime_ns = cache_mtime_before
            return candidate

    def _map(
        self,
        categories: list[dict[str, Any]],
        items: list[dict[str, Any]],
    ) -> MediaCatalogSnapshot:
        category_keys = {
            (str(category.get("source_id") or ""), str(category.get("category_id") or "")): (
                f"{category.get('source_id')}:{self.kind}-category:"
                f"{category.get('category_id')}"
            )
            for category in categories
            if category.get("source_id") and category.get("category_id") not in (None, "")
        }
        uncategorized_sources: set[str] = set()
        for item in items:
            source_id = str(item.get("source_id") or "")
            source_categories = [
                str(value)
                for value in item.get("category_ids")
                or [item.get("category_id")]
                if value not in (None, "")
            ]
            if source_id and not any(
                (source_id, category_id) in category_keys
                for category_id in source_categories
            ):
                uncategorized_sources.add(source_id)
        for source_id in uncategorized_sources:
            category_keys[(source_id, "__uncategorized__")] = (
                f"{source_id}:{self.kind}-category:__uncategorized__"
            )
        category_ids = self._registry.get_or_create_many(list(category_keys.values()))
        category_id_map = {
            identity: str(category_ids[key])
            for identity, key in category_keys.items()
        }
        public_categories: list[dict[str, Any]] = []
        category_source_ids: dict[str, str] = {}
        for category in categories:
            identity = (
                str(category.get("source_id") or ""),
                str(category.get("category_id") or ""),
            )
            public_id = category_id_map.get(identity)
            if public_id is None:
                continue
            public_categories.append(
                {
                    "category_id": public_id,
                    "category_name": str(category.get("category_name") or "Uncategorized"),
                    "parent_id": 0,
                }
            )
            category_source_ids[public_id] = identity[0]
        for source_id in sorted(uncategorized_sources):
            public_id = category_id_map[(source_id, "__uncategorized__")]
            public_categories.append(
                {
                    "category_id": public_id,
                    "category_name": "Uncategorized",
                    "parent_id": 0,
                }
            )
            category_source_ids[public_id] = source_id

        public_items: list[GatewayMediaItem] = []
        for start in range(0, len(items), 5000):
            entries: list[tuple[dict[str, Any], str, str, str]] = []
            for item in items[start : start + 5000]:
                source_id = str(item.get("source_id") or "")
                upstream_id = str(item.get(self.id_field) or "")
                if not source_id or not upstream_id:
                    continue
                key = f"{source_id}:{self.kind}:{upstream_id}"
                entries.append((item, source_id, upstream_id, key))
            local_ids = self._registry.get_or_create_many(
                [key for _, _, _, key in entries]
            )
            for item, source_id, upstream_id, key in entries:
                public = _public_copy(item)
                local_id = local_ids[key]
                public[self.id_field] = local_id
                source_categories = [
                    str(value)
                    for value in item.get("category_ids")
                    or [item.get("category_id")]
                    if value not in (None, "")
                ]
                mapped_categories = [
                    category_id_map[(source_id, category_id)]
                    for category_id in source_categories
                    if (source_id, category_id) in category_id_map
                ]
                if not mapped_categories:
                    uncategorized_id = category_id_map.get(
                        (source_id, "__uncategorized__")
                    )
                    if uncategorized_id:
                        mapped_categories = [uncategorized_id]
                public["category_ids"] = mapped_categories
                public["category_id"] = (
                    mapped_categories[0] if mapped_categories else ""
                )
                public_items.append(
                    GatewayMediaItem(
                        local_id=local_id,
                        source_id=source_id,
                        upstream_id=upstream_id,
                        category_ids=tuple(mapped_categories),
                        payload=json.dumps(
                            public,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode(),
                    )
                )

        return MediaCatalogSnapshot(
            categories=public_categories,
            category_source_ids=category_source_ids,
            category_id_map=category_id_map,
            items=public_items,
            items_by_id={item.local_id: item for item in public_items},
        )

    def remap_info(
        self,
        item: GatewayMediaItem,
        raw_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Remove private fields and replace provider IDs in a detail response."""
        result = _public_copy(raw_info)
        snapshot = self._snapshot
        category_id_map = snapshot.category_id_map if snapshot is not None else {}

        def remap_categories(value: dict[str, Any]) -> None:
            upstream_categories = [
                str(category_id)
                for category_id in value.get("category_ids")
                or [value.get("category_id")]
                if category_id not in (None, "")
            ]
            mapped = [
                category_id_map[(item.source_id, category_id)]
                for category_id in upstream_categories
                if (item.source_id, category_id) in category_id_map
            ]
            if not mapped:
                uncategorized_id = category_id_map.get(
                    (item.source_id, "__uncategorized__")
                )
                if uncategorized_id:
                    mapped = [uncategorized_id]
            value["category_ids"] = mapped
            value["category_id"] = mapped[0] if mapped else ""

        if self.kind == "movie":
            movie_data = result.get("movie_data")
            if isinstance(movie_data, dict):
                movie_data["stream_id"] = item.local_id
                remap_categories(movie_data)
            info = result.get("info")
            if isinstance(info, dict):
                remap_categories(info)
            return result

        info = result.get("info")
        if isinstance(info, dict):
            info["series_id"] = item.local_id
            remap_categories(info)
        episodes = result.get("episodes")
        if not isinstance(episodes, dict):
            return result

        episode_entries: list[tuple[dict[str, Any], str]] = []
        for season_episodes in episodes.values():
            if not isinstance(season_episodes, list):
                continue
            for episode in season_episodes:
                if not isinstance(episode, dict):
                    continue
                upstream_id = str(episode.get("id") or "")
                if upstream_id:
                    episode_entries.append(
                        (
                            episode,
                            f"{item.source_id}:series-episode:{upstream_id}",
                        )
                    )
        local_ids = self._registry.get_or_create_many(
            [key for _, key in episode_entries]
        )
        for episode, key in episode_entries:
            episode["id"] = local_ids[key]
        return result

    def resolve_registered_id(
        self,
        local_id: int,
        item_kind: str | None = None,
    ) -> tuple[str, str] | None:
        """Resolve a persisted local ID without loading the full catalog."""
        kind = item_kind or self.kind
        key = self._registry.key_for_id(local_id)
        marker = f":{kind}:"
        if key is None or marker not in key:
            return None
        source_id, upstream_id = key.rsplit(marker, 1)
        if not source_id or not upstream_id:
            return None
        return source_id, upstream_id
