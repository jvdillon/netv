"""Native-player catalog built from neTV's configured IPTV sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import hashlib
import json
import logging
import pathlib
import sqlite3
import threading
import time
import urllib.error
import urllib.parse

from m3u import fetch_source_live_data

import cache


log = logging.getLogger(__name__)
_FAILED_LOAD_RETRY_SECONDS = 60


class RegistryError(RuntimeError):
    """Raised when stable gateway IDs cannot be read or updated."""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class GatewayStream:
    """A public stream entry and its private upstream mapping."""

    local_id: int
    source_id: str
    source_type: str
    upstream_id: str
    direct_url: str
    access_group_ids: tuple[str, ...]
    public: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    categories: list[dict[str, Any]]
    category_access_ids: dict[str, str]
    streams: list[GatewayStream]
    streams_by_id: dict[int, GatewayStream]


class StreamIdRegistry:
    """Persist stable integer IDs in a transactional SQLite registry."""

    def __init__(
        self,
        path: pathlib.Path | None = None,
        legacy_path: pathlib.Path | None = None,
    ) -> None:
        self._path = path
        self._legacy_path = legacy_path
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def path(self) -> pathlib.Path:
        return self._path or cache.CACHE_DIR / "gateway_stream_ids.db"

    @property
    def legacy_path(self) -> pathlib.Path:
        if self._legacy_path is not None:
            return self._legacy_path
        if self._path is None:
            return cache.CACHE_DIR / "gateway_stream_ids.json"
        return self.path.with_suffix(".json")

    @property
    def migrated_legacy_path(self) -> pathlib.Path:
        return self.legacy_path.with_name(f"{self.legacy_path.name}.migrated")

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        if self._initialized:
            return
        should_mark_legacy = False
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ids (
                        id INTEGER PRIMARY KEY,
                        key TEXT NOT NULL UNIQUE
                    )
                    """
                )
                connection.execute("BEGIN IMMEDIATE")
                has_ids_row = connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM ids LIMIT 1)"
                ).fetchone()
                has_ids = bool(has_ids_row and has_ids_row[0])
                if not has_ids and self.legacy_path.exists():
                    raw = json.loads(self.legacy_path.read_text())
                    if not isinstance(raw, dict) or not isinstance(raw.get("ids"), dict):
                        raise ValueError(
                            "registry must contain an object-valued 'ids' field"
                        )
                    ids = raw["ids"]
                    entries = [
                        (int(local_id), str(key))
                        for key, local_id in ids.items()
                        if isinstance(local_id, int)
                        and not isinstance(local_id, bool)
                        and local_id > 0
                    ]
                    if (
                        len(entries) != len(ids)
                        or len({local_id for local_id, _ in entries}) != len(entries)
                    ):
                        raise ValueError("registry contains invalid or duplicate IDs")
                    connection.executemany(
                        "INSERT INTO ids (id, key) VALUES (?, ?)",
                        entries,
                    )
                    log.info(
                        "Migrated %d stable gateway IDs to SQLite",
                        len(entries),
                    )
                    should_mark_legacy = True
                elif not has_ids and self.migrated_legacy_path.exists():
                    raise ValueError(
                        "SQLite registry is empty after legacy migration"
                    )
                elif has_ids and self.legacy_path.exists():
                    should_mark_legacy = True
        except (OSError, sqlite3.Error, ValueError, TypeError, json.JSONDecodeError) as exc:
            log.error("Invalid gateway stream ID registry: %s", exc)
            raise RegistryError("Invalid gateway stream ID registry") from exc
        if should_mark_legacy:
            try:
                self.legacy_path.replace(self.migrated_legacy_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                log.warning("Could not mark legacy gateway ID registry migrated: %s", exc)
        self._initialized = True

    def get_or_create(self, key: str) -> int:
        return self.get_or_create_many([key])[key]

    def get_or_create_many(self, keys: list[str]) -> dict[str, int]:
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            return {}
        with self._lock:
            self._initialize()
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connect()
                connection.execute("BEGIN IMMEDIATE")
                existing: dict[str, int] = {}
                for index in range(0, len(unique_keys), 400):
                    chunk = unique_keys[index : index + 400]
                    placeholders = ",".join("?" for _ in chunk)
                    existing.update(
                        {
                            key: local_id
                            for key, local_id in connection.execute(
                                f"SELECT key, id FROM ids WHERE key IN ({placeholders})",
                                chunk,
                            )
                        }
                    )
                missing = [key for key in unique_keys if key not in existing]
                if missing:
                    next_id_row = connection.execute(
                        "SELECT COALESCE(MAX(id), 0) + 1 FROM ids"
                    ).fetchone()
                    if next_id_row is None:
                        raise RegistryError("Unable to allocate gateway ID")
                    next_id = int(next_id_row[0])
                    new_entries = [
                        (next_id + offset, key)
                        for offset, key in enumerate(missing)
                    ]
                    connection.executemany(
                        "INSERT INTO ids (id, key) VALUES (?, ?)",
                        new_entries,
                    )
                    existing.update(
                        {key: local_id for local_id, key in new_entries}
                    )
                connection.commit()
                return {key: existing[key] for key in keys}
            except sqlite3.Error as exc:
                if connection is not None:
                    connection.rollback()
                raise RegistryError("Unable to update gateway ID registry") from exc
            finally:
                if connection is not None:
                    connection.close()

    def key_for_id(self, local_id: int) -> str | None:
        with self._lock:
            self._initialize()
            try:
                with self._connect() as connection:
                    row = connection.execute(
                        "SELECT key FROM ids WHERE id = ?",
                        (local_id,),
                    ).fetchone()
            except sqlite3.Error as exc:
                raise RegistryError("Unable to read gateway ID registry") from exc
            return str(row[0]) if row is not None else None


class GatewayCatalog:
    """Load and cache a sanitized live catalog for native players."""

    def __init__(
        self,
        registry: StreamIdRegistry | None = None,
        ttl_seconds: int = cache.LIVE_CACHE_TTL,
    ) -> None:
        self._registry = registry or StreamIdRegistry()
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._snapshot: CatalogSnapshot | None = None
        self._loaded_at = 0.0

    def get(self, force: bool = False) -> CatalogSnapshot:
        if (
            not force
            and self._snapshot is not None
            and time.monotonic() - self._loaded_at < self._ttl_seconds
        ):
            return self._snapshot
        with self._lock:
            if (
                not force
                and self._snapshot is not None
                and time.monotonic() - self._loaded_at < self._ttl_seconds
            ):
                return self._snapshot
            candidate, had_errors = self._load()
            if had_errors and self._snapshot is not None:
                log.warning("Gateway catalog refresh failed; retaining the previous snapshot")
                self._loaded_at = (
                    time.monotonic()
                    - self._ttl_seconds
                    + min(_FAILED_LOAD_RETRY_SECONDS, self._ttl_seconds)
                )
                return self._snapshot
            self._snapshot = candidate
            self._loaded_at = (
                time.monotonic()
                if not had_errors
                else time.monotonic()
                - self._ttl_seconds
                + min(_FAILED_LOAD_RETRY_SECONDS, self._ttl_seconds)
            )
            return self._snapshot

    def _load(self) -> tuple[CatalogSnapshot, bool]:
        categories: list[dict[str, Any]] = []
        category_access_ids: dict[str, str] = {}
        streams: list[GatewayStream] = []
        had_errors = False

        for source in cache.get_sources():
            if source.type not in ("xtream", "m3u"):
                continue
            try:
                source_categories, source_streams, _, _ = fetch_source_live_data(
                    source,
                    persist_epg_url=False,
                )
            except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as exc:
                had_errors = True
                log.error(
                    "Gateway failed to load source %s (%s)",
                    source.name,
                    type(exc).__name__,
                )
                continue

            category_values = {
                str(category.get("category_id"))
                for category in source_categories
                if category.get("category_id") not in (None, "")
            }
            category_values.update(
                str(category_id)
                for stream in source_streams
                for category_id in stream.get("category_ids") or []
                if category_id not in (None, "")
            )
            uncategorized_id = f"{source.id}_gateway_uncategorized"
            needs_uncategorized = any(
                not (stream.get("category_ids") or []) for stream in source_streams
            )
            if needs_uncategorized:
                category_values.add(uncategorized_id)
            category_keys = {
                category_id: f"{source.id}:category:{category_id}"
                for category_id in category_values
            }
            category_local_ids = self._registry.get_or_create_many(list(category_keys.values()))
            category_id_map = {
                category_id: str(category_local_ids[key])
                for category_id, key in category_keys.items()
            }
            category_access_ids.update(
                {
                    public_id: source_category_id
                    for source_category_id, public_id in category_id_map.items()
                    if source_category_id != uncategorized_id
                }
            )
            categories.extend(self._public_categories(source_categories, category_id_map))
            if needs_uncategorized:
                categories.append(
                    {
                        "category_id": category_id_map[uncategorized_id],
                        "category_name": "Uncategorized",
                        "parent_id": 0,
                    }
                )
            registry_entries: list[tuple[dict[str, Any], str]] = []
            key_occurrences: dict[str, int] = {}
            for stream in source_streams:
                if stream.get("stream_id") in (None, ""):
                    continue
                base_key = self._stream_registry_key(source.id, source.type, stream)
                occurrence = key_occurrences.get(base_key, 0)
                key_occurrences[base_key] = occurrence + 1
                key = (
                    base_key
                    if occurrence == 0
                    else f"{base_key}:duplicate:{occurrence + 1}"
                )
                registry_entries.append((stream, key))
            registry_keys = [key for _, key in registry_entries]
            local_ids = self._registry.get_or_create_many(registry_keys)
            for stream, key in registry_entries:
                local_id = local_ids[key]
                mapped = self._map_stream(
                    source.id,
                    source.type,
                    stream,
                    local_id,
                    category_id_map,
                    uncategorized_id,
                    category_id_map.get(uncategorized_id, ""),
                )
                if mapped is not None:
                    streams.append(mapped)

        return (
            CatalogSnapshot(
                categories=categories,
                category_access_ids=category_access_ids,
                streams=streams,
                streams_by_id={stream.local_id: stream for stream in streams},
            ),
            had_errors,
        )

    @staticmethod
    def _stream_registry_key(
        source_id: str,
        source_type: str,
        stream: dict[str, Any],
    ) -> str:
        upstream_id = str(stream.get("stream_id", ""))
        if source_type != "m3u":
            return f"{source_id}:live:{upstream_id}"
        direct_url = str(stream.get("direct_url") or "")
        epg_channel_id = str(stream.get("epg_channel_id") or "")
        name = str(stream.get("name") or "")
        parsed_url = urllib.parse.urlparse(direct_url)
        stable_url = urllib.parse.urlunparse(
            (parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", "", "")
        )
        identity = f"{epg_channel_id}\0{name}\0{stable_url}"
        if not epg_channel_id and not name and not stable_url:
            identity = upstream_id
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return f"{source_id}:live-url:{digest}"

    @staticmethod
    def _public_categories(
        categories: list[dict[str, Any]],
        category_id_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        return [
            {
                "category_id": category_id_map[str(category["category_id"])],
                "category_name": str(category.get("category_name", "Uncategorized")),
                "parent_id": 0,
            }
            for category in categories
            if str(category.get("category_id", "")) in category_id_map
        ]

    def _map_stream(
        self,
        source_id: str,
        source_type: str,
        stream: dict[str, Any],
        local_id: int,
        category_id_map: dict[str, str],
        uncategorized_source_id: str,
        uncategorized_public_id: str,
    ) -> GatewayStream | None:
        upstream_id = str(stream.get("stream_id", ""))
        if not upstream_id:
            return None
        source_category_ids = [str(value) for value in stream.get("category_ids") or []]
        category_ids = [
            category_id_map[value] for value in source_category_ids if value in category_id_map
        ]
        source_category_id = str(stream.get("category_id", ""))
        access_group_ids = tuple(source_category_ids)
        fallback_source_id = (
            source_category_id
            if source_category_id in category_id_map
            else uncategorized_source_id
        )
        category_id = (
            category_ids[0]
            if category_ids
            else category_id_map.get(fallback_source_id, uncategorized_public_id)
        )
        if not category_ids and category_id:
            category_ids = [category_id]
            access_group_ids = (
                (fallback_source_id,)
                if fallback_source_id != uncategorized_source_id
                else ()
            )
        access_group_ids = tuple(value for value in access_group_ids if value)
        public = {
            "num": stream.get("num", local_id),
            "name": str(stream.get("name", "Unknown")),
            "stream_type": "live",
            "stream_id": local_id,
            "stream_icon": str(stream.get("stream_icon") or ""),
            "epg_channel_id": str(stream.get("epg_channel_id") or ""),
            "added": str(stream.get("added") or ""),
            "category_id": category_id,
            "category_ids": category_ids or ([category_id] if category_id else []),
            "custom_sid": str(stream.get("custom_sid") or ""),
            "tv_archive": _safe_int(stream.get("tv_archive")),
            "tv_archive_duration": _safe_int(stream.get("tv_archive_duration")),
        }
        return GatewayStream(
            local_id=local_id,
            source_id=source_id,
            source_type=source_type,
            upstream_id=upstream_id,
            direct_url=str(stream.get("direct_url") or ""),
            access_group_ids=access_group_ids,
            public=public,
        )
