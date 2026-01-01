"""Tests for xtream.py - Xtream Codes API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import json

import pytest

from xtream import XtreamClient


class TestXtreamClient:
    """Tests for XtreamClient."""

    def test_api_url_property(self):
        client = XtreamClient("http://example.com", "user", "pass")
        assert client.api_url == "http://example.com/player_api.php?username=user&password=pass"

    def test_epg_url_property(self):
        client = XtreamClient("http://example.com", "user", "pass")
        assert client.epg_url == "http://example.com/xmltv.php?username=user&password=pass"

    def test_url_normalization_strips_trailing_slash(self):
        client = XtreamClient("http://example.com/", "user", "pass")
        assert client.base_url == "http://example.com"
        # No double slashes after the domain
        assert "example.com/player_api" in client.api_url

    def test_url_normalization_strips_multiple_trailing_slashes(self):
        client = XtreamClient("http://example.com///", "user", "pass")
        assert client.base_url == "http://example.com"

    def test_special_chars_in_credentials_are_encoded(self):
        client = XtreamClient("http://example.com", "user@test", "p&ss=word")
        # Check that special chars are URL-encoded in api_url
        assert "user%40test" in client.api_url
        assert "p%26ss%3Dword" in client.api_url
        # Same for epg_url
        assert "user%40test" in client.epg_url
        assert "p%26ss%3Dword" in client.epg_url

    def test_build_stream_url_live_no_ext(self):
        client = XtreamClient("http://example.com", "user", "pass")
        url = client.build_stream_url("live", 123)
        assert url == "http://example.com/live/user/pass/123"

    def test_build_stream_url_live_with_ext(self):
        client = XtreamClient("http://example.com", "user", "pass")
        url = client.build_stream_url("live", 123, "m3u8")
        assert url == "http://example.com/live/user/pass/123.m3u8"

    def test_build_stream_url_movie(self):
        client = XtreamClient("http://example.com", "user", "pass")
        url = client.build_stream_url("movie", 456, "mkv")
        assert url == "http://example.com/movie/user/pass/456.mkv"

    def test_build_stream_url_series(self):
        client = XtreamClient("http://example.com", "user", "pass")
        url = client.build_stream_url("series", 789, "mp4")
        assert url == "http://example.com/series/user/pass/789.mp4"

    def test_build_timeshift_url(self):
        client = XtreamClient("http://example.com", "user", "pass")
        url = client.build_timeshift_url(123, 60, "2024-01-15:14-30")
        assert url == "http://example.com/timeshift/user/pass/60/2024-01-15:14-30/123.ts"

    def test_build_timeshift_url_custom_ext(self):
        client = XtreamClient("http://example.com", "user", "pass")
        url = client.build_timeshift_url(123, 30, "2024-01-15:10-00", ext="m3u8")
        assert url == "http://example.com/timeshift/user/pass/30/2024-01-15:10-00/123.m3u8"


class TestXtreamClientApi:
    """Tests for XtreamClient API methods with mocked network."""

    @pytest.fixture
    def client(self):
        return XtreamClient("http://example.com", "user", "pass")

    @pytest.fixture
    def mock_urlopen(self):
        with patch("xtream.safe_urlopen") as mock:
            yield mock

    def _setup_response(self, mock_urlopen, data):
        """Helper to setup mock response."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

    def test_get_live_categories(self, client, mock_urlopen):
        categories = [{"category_id": "1", "category_name": "News"}]
        self._setup_response(mock_urlopen, categories)

        result = client.get_live_categories()

        assert result == categories
        mock_urlopen.assert_called_once()
        url = mock_urlopen.call_args[0][0]
        assert "action=get_live_categories" in url

    def test_get_live_streams(self, client, mock_urlopen):
        streams = [{"stream_id": 1, "name": "CNN"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_live_streams()

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "action=get_live_streams" in url

    def test_get_live_streams_with_category(self, client, mock_urlopen):
        streams = [{"stream_id": 1, "name": "CNN"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_live_streams(category_id=5)

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "action=get_live_streams" in url
        assert "category_id=5" in url

    def test_get_vod_categories(self, client, mock_urlopen):
        categories = [{"category_id": "10", "category_name": "Movies"}]
        self._setup_response(mock_urlopen, categories)

        result = client.get_vod_categories()

        assert result == categories
        url = mock_urlopen.call_args[0][0]
        assert "action=get_vod_categories" in url

    def test_get_vod_streams(self, client, mock_urlopen):
        streams = [{"stream_id": 100, "name": "Movie 1"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_vod_streams()

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "action=get_vod_streams" in url

    def test_get_vod_streams_with_category(self, client, mock_urlopen):
        streams = [{"stream_id": 100, "name": "Movie 1"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_vod_streams(category_id=10)

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "category_id=10" in url

    def test_get_series_categories(self, client, mock_urlopen):
        categories = [{"category_id": "20", "category_name": "Drama"}]
        self._setup_response(mock_urlopen, categories)

        result = client.get_series_categories()

        assert result == categories
        url = mock_urlopen.call_args[0][0]
        assert "action=get_series_categories" in url

    def test_get_series(self, client, mock_urlopen):
        series = [{"series_id": 200, "name": "Show 1"}]
        self._setup_response(mock_urlopen, series)

        result = client.get_series()

        assert result == series
        url = mock_urlopen.call_args[0][0]
        assert "action=get_series" in url

    def test_get_series_with_category(self, client, mock_urlopen):
        series = [{"series_id": 200, "name": "Show 1"}]
        self._setup_response(mock_urlopen, series)

        result = client.get_series(category_id=20)

        assert result == series
        url = mock_urlopen.call_args[0][0]
        assert "category_id=20" in url

    def test_get_series_info(self, client, mock_urlopen):
        info = {"info": {"name": "Show 1"}, "episodes": {"1": []}}
        self._setup_response(mock_urlopen, info)

        result = client.get_series_info(series_id=200)

        assert result == info
        url = mock_urlopen.call_args[0][0]
        assert "action=get_series_info" in url
        assert "series_id=200" in url

    def test_get_vod_info(self, client, mock_urlopen):
        info = {"info": {"name": "Movie 1", "plot": "A story"}}
        self._setup_response(mock_urlopen, info)

        result = client.get_vod_info(vod_id=100)

        assert result == info
        url = mock_urlopen.call_args[0][0]
        assert "action=get_vod_info" in url
        assert "vod_id=100" in url

    def test_get_server_info(self, client, mock_urlopen):
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
        assert result["user_info"]["auth"] == 1
        url = mock_urlopen.call_args[0][0]
        # get_server_info calls API with no action
        assert "action" not in url

    def test_get_server_info_auth_failed(self, client, mock_urlopen):
        server_info = {"user_info": {"auth": 0}}
        self._setup_response(mock_urlopen, server_info)

        result = client.get_server_info()

        assert result["user_info"]["auth"] == 0

    def test_get_server_info_uses_shorter_timeout(self, client, mock_urlopen):
        self._setup_response(mock_urlopen, {"user_info": {"auth": 1}})

        client.get_server_info()

        # get_server_info uses 15s timeout by default (vs 30s for other calls)
        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 15

    def test_custom_timeout_passed_through(self, client, mock_urlopen):
        self._setup_response(mock_urlopen, [])

        client.get_live_categories()

        # Default timeout is 30s
        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 30

    def test_api_encodes_special_chars_in_params(self, client, mock_urlopen):
        self._setup_response(mock_urlopen, {})

        client._api("test_action", foo="bar&baz", key="val=ue")

        url = mock_urlopen.call_args[0][0]
        assert "foo=bar%26baz" in url
        assert "key=val%3Due" in url

    def test_get_short_epg(self, client, mock_urlopen):
        epg_data = {
            "epg_listings": [
                {"title": "Show 1", "start": "2024-01-15 14:00:00"},
                {"title": "Show 2", "start": "2024-01-15 15:00:00"},
            ]
        }
        self._setup_response(mock_urlopen, epg_data)

        result = client.get_short_epg(stream_id=123)

        assert result == epg_data
        url = mock_urlopen.call_args[0][0]
        assert "action=get_short_epg" in url
        assert "stream_id=123" in url
        assert "limit=10" in url  # default limit

    def test_get_short_epg_custom_limit(self, client, mock_urlopen):
        self._setup_response(mock_urlopen, {"epg_listings": []})

        client.get_short_epg(stream_id=456, limit=5)

        url = mock_urlopen.call_args[0][0]
        assert "stream_id=456" in url
        assert "limit=5" in url


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)
