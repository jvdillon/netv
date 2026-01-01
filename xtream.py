"""Xtream Codes API client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json
import urllib.parse

from util import safe_urlopen


@dataclass(slots=True)
class XtreamClient:
    """Client for Xtream Codes API.

    Handles authentication and API calls to Xtream-compatible IPTV providers.
    """

    base_url: str
    username: str
    password: str

    def __post_init__(self) -> None:
        # Normalize URL: strip trailing slashes
        self.base_url = self.base_url.rstrip("/")

    @property
    def _base_params(self) -> dict[str, str]:
        return {"username": self.username, "password": self.password}

    @property
    def api_url(self) -> str:
        params = urllib.parse.urlencode(self._base_params)
        return f"{self.base_url}/player_api.php?{params}"

    def _fetch(self, url: str, timeout: int = 30) -> str:
        with safe_urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")

    def _api(self, action: str | None = None, timeout: int = 30, **params: Any) -> Any:
        query = dict(self._base_params)
        if action:
            query["action"] = action
        query.update(params)
        url = f"{self.base_url}/player_api.php?{urllib.parse.urlencode(query)}"
        return json.loads(self._fetch(url, timeout=timeout))

    def get_server_info(self, timeout: int = 15) -> dict[str, Any]:
        """Returns user_info and server_info; check user_info['auth'] == 1."""
        return self._api(timeout=timeout)

    def get_live_categories(self) -> list[dict[str, Any]]:
        return self._api("get_live_categories")

    def get_live_streams(self, category_id: int | None = None) -> list[dict[str, Any]]:
        if category_id:
            return self._api("get_live_streams", category_id=category_id)
        return self._api("get_live_streams")

    def get_vod_categories(self) -> list[dict[str, Any]]:
        return self._api("get_vod_categories")

    def get_vod_streams(self, category_id: int | None = None) -> list[dict[str, Any]]:
        if category_id:
            return self._api("get_vod_streams", category_id=category_id)
        return self._api("get_vod_streams")

    def get_series_categories(self) -> list[dict[str, Any]]:
        return self._api("get_series_categories")

    def get_series(self, category_id: int | None = None) -> list[dict[str, Any]]:
        if category_id:
            return self._api("get_series", category_id=category_id)
        return self._api("get_series")

    def get_series_info(self, series_id: int) -> dict[str, Any]:
        return self._api("get_series_info", series_id=series_id)

    def get_vod_info(self, vod_id: int) -> dict[str, Any]:
        return self._api("get_vod_info", vod_id=vod_id)

    def get_short_epg(self, stream_id: int, limit: int = 10) -> dict[str, Any]:
        """Returns epg_listings for stream; some providers ignore limit."""
        return self._api("get_short_epg", stream_id=stream_id, limit=limit)

    def build_stream_url(self, stream_type: str, stream_id: int, ext: str = "") -> str:
        base = f"{self.base_url}/{stream_type}/{self.username}/{self.password}/{stream_id}"
        return f"{base}.{ext}" if ext else base

    def build_timeshift_url(
        self,
        stream_id: int,
        duration: int,
        start: str,
        ext: str = "ts",
    ) -> str:
        """For streams with tv_archive=1. start format: YYYY-MM-DD:HH-MM."""
        return (
            f"{self.base_url}/timeshift/{self.username}/{self.password}/"
            f"{duration}/{start}/{stream_id}.{ext}"
        )

    @property
    def epg_url(self) -> str:
        params = urllib.parse.urlencode(self._base_params)
        return f"{self.base_url}/xmltv.php?{params}"
