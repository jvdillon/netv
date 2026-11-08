"""Xtream Codes API client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias
from urllib import parse

from treekle import MutablePlainTree, json_loads

from netv.util import safe_urlopen


# Provider objects keep every upstream key: providers extend the API freely and
# the app forwards records it does not fully understand.
ProviderRecord: TypeAlias = dict[str, MutablePlainTree]


def text_field(record: Mapping[str, MutablePlainTree], key: str) -> str:
    """Return a provider field as text; missing and null are empty."""
    value = record.get(key)
    return "" if value is None else str(value)


def list_field(record: Mapping[str, MutablePlainTree], key: str) -> list[MutablePlainTree]:
    """Return a provider field that should be a list, or an empty list."""
    value = record.get(key)
    return value if isinstance(value, list) else []


def record_field(record: Mapping[str, MutablePlainTree], key: str) -> ProviderRecord:
    """Return a provider field that should be an object, or an empty one."""
    value = record.get(key)
    return value if isinstance(value, dict) else {}


def category_ids(record: Mapping[str, MutablePlainTree]) -> list[str]:
    """Return a stream's category ids as text."""
    return [str(c) for c in list_field(record, "category_ids")]


@dataclass(kw_only=True, slots=True, frozen=True)
class XtreamClient:
    """Client for Xtream Codes API.

    Handles authentication and API calls to Xtream-compatible IPTV providers.
    """

    base_url: str
    username: str
    password: str
    # Some providers reject API requests with an unexpected User-Agent (HTTP 403).
    user_agent: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def api_url(self) -> str:
        params = parse.urlencode(self._base_params)
        return f"{self.base_url}/player_api.php?{params}"

    def get_server_info(self, timeout: int = 15) -> ProviderRecord:
        """Returns user_info and server_info; check user_info['auth'] == 1."""
        return json_loads(self._api(timeout=timeout), dict[str, MutablePlainTree])

    def get_live_categories(self) -> list[ProviderRecord]:
        return json_loads(self._api("get_live_categories"), list[ProviderRecord])

    def get_live_streams(self, category_id: int | None = None) -> list[ProviderRecord]:
        return json_loads(
            self._api("get_live_streams", category_id=_category_filter(category_id)),
            list[ProviderRecord],
        )

    def get_vod_categories(self) -> list[ProviderRecord]:
        return json_loads(self._api("get_vod_categories"), list[ProviderRecord])

    def get_vod_streams(self, category_id: int | None = None) -> list[ProviderRecord]:
        return json_loads(
            self._api("get_vod_streams", category_id=_category_filter(category_id)),
            list[ProviderRecord],
        )

    def get_series_categories(self) -> list[ProviderRecord]:
        return json_loads(self._api("get_series_categories"), list[ProviderRecord])

    def get_series(self, category_id: int | None = None) -> list[ProviderRecord]:
        return json_loads(
            self._api("get_series", category_id=_category_filter(category_id)),
            list[ProviderRecord],
        )

    def get_series_info(self, series_id: int) -> ProviderRecord:
        return json_loads(
            self._api("get_series_info", series_id=series_id),
            dict[str, MutablePlainTree],
        )

    def get_vod_info(self, vod_id: int) -> ProviderRecord:
        return json_loads(self._api("get_vod_info", vod_id=vod_id), dict[str, MutablePlainTree])

    def get_short_epg(self, stream_id: int, limit: int = 10) -> ProviderRecord:
        """Returns epg_listings for stream; some providers ignore limit."""
        return json_loads(
            self._api("get_short_epg", stream_id=stream_id, limit=limit),
            dict[str, MutablePlainTree],
        )

    def build_stream_url(self, stream_type: str, stream_id: int, ext: str = "") -> str:
        # URL-encode username/password to handle special chars like # in passwords.
        user = parse.quote(self.username, safe="")
        pwd = parse.quote(self.password, safe="")
        base = f"{self.base_url}/{stream_type}/{user}/{pwd}/{stream_id}"
        return f"{base}.{ext}" if ext else base

    def build_timeshift_url(
        self,
        stream_id: int,
        duration: int,
        start: str,
        ext: str = "ts",
    ) -> str:
        """For streams with tv_archive=1. start format: YYYY-MM-DD:HH-MM."""
        # URL-encode username/password to handle special chars like # in passwords.
        user = parse.quote(self.username, safe="")
        pwd = parse.quote(self.password, safe="")
        return f"{self.base_url}/timeshift/{user}/{pwd}/{duration}/{start}/{stream_id}.{ext}"

    @property
    def epg_url(self) -> str:
        params = parse.urlencode(self._base_params)
        return f"{self.base_url}/xmltv.php?{params}"

    @property
    def _base_params(self) -> dict[str, str]:
        return {"username": self.username, "password": self.password}

    def _fetch(self, url: str, timeout: int = 30) -> str:
        with safe_urlopen(url, timeout=timeout, user_agent=self.user_agent) as resp:
            return resp.read().decode("utf-8")

    def _api(
        self,
        action: str = "",
        timeout: int = 30,
        **params: str | int | None,
    ) -> str:
        """Return the raw JSON text of one API call; ``None`` params are omitted."""
        query = dict(self._base_params)
        if action:
            query["action"] = action
        query.update({k: str(v) for k, v in params.items() if v is not None})
        url = f"{self.base_url}/player_api.php?{parse.urlencode(query)}"
        return self._fetch(url, timeout=timeout)


def _category_filter(category_id: int | None) -> int | None:
    """Return the id to send; ``None`` and ``0`` both mean every category."""
    return None if category_id is None or category_id == 0 else category_id
