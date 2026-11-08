"""Tests for xtream.py - Xtream Codes API client."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import json

from treekle import ReadError

import pytest

from xtream import (
    ProviderRecord,
    XtreamClient,
    category_ids,
    list_field,
    record_field,
    text_field,
)


class TestXtreamClient:
    """Tests for XtreamClient."""

    def test_api_url_property(self):
        client = XtreamClient(base_url="http://example.com", username="user", password="pass")
        assert client.api_url == "http://example.com/player_api.php?username=user&password=pass"

    def test_epg_url_property(self):
        client = XtreamClient(base_url="http://example.com", username="user", password="pass")
        assert client.epg_url == "http://example.com/xmltv.php?username=user&password=pass"

    def test_url_normalization_strips_trailing_slash(self):
        client = XtreamClient(base_url="http://example.com/", username="user", password="pass")
        assert client.base_url == "http://example.com"
        # No double slashes after the domain.
        assert "example.com/player_api" in client.api_url

    def test_url_normalization_strips_multiple_trailing_slashes(self):
        client = XtreamClient(base_url="http://example.com///", username="user", password="pass")
        assert client.base_url == "http://example.com"

    def test_special_chars_in_credentials_are_encoded(self):
        client = XtreamClient(
            base_url="http://example.com",
            username="user@test",
            password="p&ss=word",
        )
        # Check that special chars are URL-encoded in api_url.
        assert "user%40test" in client.api_url
        assert "p%26ss%3Dword" in client.api_url
        # Same for epg_url.
        assert "user%40test" in client.epg_url
        assert "p%26ss%3Dword" in client.epg_url

    def test_build_stream_url_live_no_ext(self):
        client = XtreamClient(base_url="http://example.com", username="user", password="pass")
        url = client.build_stream_url("live", 123)
        assert url == "http://example.com/live/user/pass/123"

    def test_build_stream_url_live_with_ext(self):
        client = XtreamClient(base_url="http://example.com", username="user", password="pass")
        url = client.build_stream_url("live", 123, "m3u8")
        assert url == "http://example.com/live/user/pass/123.m3u8"

    def test_build_stream_url_movie(self):
        client = XtreamClient(base_url="http://example.com", username="user", password="pass")
        url = client.build_stream_url("movie", 456, "mkv")
        assert url == "http://example.com/movie/user/pass/456.mkv"

    def test_build_stream_url_series(self):
        client = XtreamClient(base_url="http://example.com", username="user", password="pass")
        url = client.build_stream_url("series", 789, "mp4")
        assert url == "http://example.com/series/user/pass/789.mp4"

    def test_build_timeshift_url(self):
        client = XtreamClient(base_url="http://example.com", username="user", password="pass")
        url = client.build_timeshift_url(123, 60, "2024-01-15:14-30")
        assert url == "http://example.com/timeshift/user/pass/60/2024-01-15:14-30/123.ts"

    def test_build_timeshift_url_custom_ext(self):
        client = XtreamClient(base_url="http://example.com", username="user", password="pass")
        url = client.build_timeshift_url(123, 30, "2024-01-15:10-00", ext="m3u8")
        assert url == "http://example.com/timeshift/user/pass/30/2024-01-15:10-00/123.m3u8"


class TestXtreamClientApi:
    """Tests for XtreamClient API methods with mocked network."""

    @pytest.fixture
    def client(self) -> XtreamClient:
        return XtreamClient(base_url="http://example.com", username="user", password="pass")

    @pytest.fixture
    def mock_urlopen(self) -> Iterator[MagicMock]:
        mock = MagicMock()
        with patch("xtream.safe_urlopen", mock):
            yield mock

    def _setup_response(self, mock_urlopen: MagicMock, data: object) -> None:
        """Helper to setup mock response."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

    def test_get_live_categories(self, client: XtreamClient, mock_urlopen: MagicMock):
        categories = [{"category_id": "1", "category_name": "News"}]
        self._setup_response(mock_urlopen, categories)

        result = client.get_live_categories()

        assert result == categories
        mock_urlopen.assert_called_once()
        url = mock_urlopen.call_args[0][0]
        assert "action=get_live_categories" in url

    def test_get_live_streams(self, client: XtreamClient, mock_urlopen: MagicMock):
        streams = [{"stream_id": 1, "name": "CNN"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_live_streams()

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "action=get_live_streams" in url

    def test_get_live_streams_with_category(self, client: XtreamClient, mock_urlopen: MagicMock):
        streams = [{"stream_id": 1, "name": "CNN"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_live_streams(category_id=5)

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "action=get_live_streams" in url
        assert "category_id=5" in url

    def test_get_vod_categories(self, client: XtreamClient, mock_urlopen: MagicMock):
        categories = [{"category_id": "10", "category_name": "Movies"}]
        self._setup_response(mock_urlopen, categories)

        result = client.get_vod_categories()

        assert result == categories
        url = mock_urlopen.call_args[0][0]
        assert "action=get_vod_categories" in url

    def test_get_vod_streams(self, client: XtreamClient, mock_urlopen: MagicMock):
        streams = [{"stream_id": 100, "name": "Movie 1"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_vod_streams()

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "action=get_vod_streams" in url

    def test_get_vod_streams_with_category(self, client: XtreamClient, mock_urlopen: MagicMock):
        streams = [{"stream_id": 100, "name": "Movie 1"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_vod_streams(category_id=10)

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "category_id=10" in url

    def test_get_series_categories(self, client: XtreamClient, mock_urlopen: MagicMock):
        categories = [{"category_id": "20", "category_name": "Drama"}]
        self._setup_response(mock_urlopen, categories)

        result = client.get_series_categories()

        assert result == categories
        url = mock_urlopen.call_args[0][0]
        assert "action=get_series_categories" in url

    def test_get_series(self, client: XtreamClient, mock_urlopen: MagicMock):
        series = [{"series_id": 200, "name": "Show 1"}]
        self._setup_response(mock_urlopen, series)

        result = client.get_series()

        assert result == series
        url = mock_urlopen.call_args[0][0]
        assert "action=get_series" in url

    def test_get_series_with_category(self, client: XtreamClient, mock_urlopen: MagicMock):
        series = [{"series_id": 200, "name": "Show 1"}]
        self._setup_response(mock_urlopen, series)

        result = client.get_series(category_id=20)

        assert result == series
        url = mock_urlopen.call_args[0][0]
        assert "category_id=20" in url

    def test_get_series_info(self, client: XtreamClient, mock_urlopen: MagicMock):
        info: ProviderRecord = {"info": {"name": "Show 1"}, "episodes": {"1": []}}
        self._setup_response(mock_urlopen, info)

        result = client.get_series_info(series_id=200)

        assert result == info
        url = mock_urlopen.call_args[0][0]
        assert "action=get_series_info" in url
        assert "series_id=200" in url

    def test_get_vod_info(self, client: XtreamClient, mock_urlopen: MagicMock):
        info: ProviderRecord = {"info": {"name": "Movie 1", "plot": "A story"}}
        self._setup_response(mock_urlopen, info)

        result = client.get_vod_info(vod_id=100)

        assert result == info
        url = mock_urlopen.call_args[0][0]
        assert "action=get_vod_info" in url
        assert "vod_id=100" in url

    def test_get_server_info(self, client: XtreamClient, mock_urlopen: MagicMock):
        server_info = {
            "user_info": {
                "auth": 1,
                "username": "user",
                "status": "Active",
                "exp_date": "1735689600",
                "max_connections": "2",
            },
            "server_info": {
                "url": "example.com",
                "port": "80",
                "https_port": "443",
                "server_protocol": "http",
            },
        }
        self._setup_response(mock_urlopen, server_info)

        result = client.get_server_info()

        assert result == server_info
        assert record_field(result, "user_info")["auth"] == 1
        url = mock_urlopen.call_args[0][0]
        # get_server_info calls API with no action.
        assert "action" not in url

    def test_get_server_info_auth_failed(self, client: XtreamClient, mock_urlopen: MagicMock):
        server_info = {"user_info": {"auth": 0}}
        self._setup_response(mock_urlopen, server_info)

        result = client.get_server_info()

        assert record_field(result, "user_info")["auth"] == 0

    def test_get_server_info_uses_shorter_timeout(
        self,
        client: XtreamClient,
        mock_urlopen: MagicMock,
    ):
        self._setup_response(mock_urlopen, {"user_info": {"auth": 1}})

        client.get_server_info()

        # get_server_info uses 15s timeout by default (vs 30s for other calls)
        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 15

    def test_custom_timeout_passed_through(self, client: XtreamClient, mock_urlopen: MagicMock):
        self._setup_response(mock_urlopen, [])

        client.get_live_categories()

        # Default timeout is 30s.
        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 30

    def test_api_encodes_special_chars_in_params(
        self,
        client: XtreamClient,
        mock_urlopen: MagicMock,
    ):
        self._setup_response(mock_urlopen, {})

        client._api("test_action", foo="bar&baz", key="left=right")

        url = mock_urlopen.call_args[0][0]
        assert "foo=bar%26baz" in url
        assert "key=left%3Dright" in url

    def test_get_short_epg(self, client: XtreamClient, mock_urlopen: MagicMock):
        epg_data = {
            "epg_listings": [
                {"title": "Show 1", "start": "2024-01-15 14:00:00"},
                {"title": "Show 2", "start": "2024-01-15 15:00:00"},
            ],
        }
        self._setup_response(mock_urlopen, epg_data)

        result = client.get_short_epg(stream_id=123)

        assert result == epg_data
        url = mock_urlopen.call_args[0][0]
        assert "action=get_short_epg" in url
        assert "stream_id=123" in url
        assert "limit=10" in url  # Default limit.

    def test_get_short_epg_custom_limit(self, client: XtreamClient, mock_urlopen: MagicMock):
        self._setup_response(mock_urlopen, {"epg_listings": []})

        client.get_short_epg(stream_id=456, limit=5)

        url = mock_urlopen.call_args[0][0]
        assert "stream_id=456" in url
        assert "limit=5" in url

    def test_zero_category_means_all(self, client: XtreamClient, mock_urlopen: MagicMock):
        self._setup_response(mock_urlopen, [])

        client.get_vod_streams(category_id=0)

        assert "category_id" not in mock_urlopen.call_args[0][0]

    def test_provider_extensions_are_kept(self, client: XtreamClient, mock_urlopen: MagicMock):
        streams = [{"stream_id": 1, "vendor": {"extra": [True, None, 1.5]}}]
        self._setup_response(mock_urlopen, streams)

        assert client.get_live_streams() == streams

    def test_wrong_shape_raises(self, client: XtreamClient, mock_urlopen: MagicMock):
        self._setup_response(mock_urlopen, {"user_info": {"auth": 0}})

        with pytest.raises(ReadError):
            client.get_live_streams()


class TestFieldReaders:
    RECORD: ProviderRecord = {
        "name": "CNN",
        "num": 7,
        "missing": None,
        "category_ids": [1, "2"],
        "info": {"plot": "p"},
    }

    def test_text_field(self):
        assert text_field(self.RECORD, "name") == "CNN"
        assert text_field(self.RECORD, "num") == "7"
        assert text_field(self.RECORD, "missing") == ""
        assert text_field(self.RECORD, "absent") == ""

    def test_list_and_record_fields(self):
        assert list_field(self.RECORD, "category_ids") == [1, "2"]
        assert list_field(self.RECORD, "name") == []
        assert record_field(self.RECORD, "info") == {"plot": "p"}
        assert record_field(self.RECORD, "name") == {}

    def test_category_ids(self):
        assert category_ids(self.RECORD) == ["1", "2"]
        assert category_ids({}) == []


class TestXtreamClientUserAgent:
    """The configured User-Agent must be sent on API calls (issue #57).

    Some providers return HTTP 403 for channel list / VOD / series requests
    that omit the same User-Agent used for stream playback.
    """

    @pytest.fixture
    def mock_urlopen(self) -> Iterator[MagicMock]:
        resp = MagicMock()
        resp.read.return_value = b"[]"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock = MagicMock(return_value=resp)
        with patch("xtream.safe_urlopen", mock):
            yield mock

    def test_defaults_to_none(self):
        assert (
            XtreamClient(base_url="http://example.com", username="user", password="pass").user_agent
            is None
        )

    def test_user_agent_forwarded_to_urlopen(self, mock_urlopen: MagicMock):
        client = XtreamClient(
            base_url="http://example.com",
            username="user",
            password="pass",
            user_agent="TiviMate/4.7.0",
        )
        client.get_live_streams()
        assert mock_urlopen.call_args.kwargs["user_agent"] == "TiviMate/4.7.0"

    def test_none_user_agent_forwarded_so_default_applies(self, mock_urlopen: MagicMock):
        XtreamClient(
            base_url="http://example.com",
            username="user",
            password="pass",
        ).get_live_streams()
        assert mock_urlopen.call_args.kwargs["user_agent"] is None


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
