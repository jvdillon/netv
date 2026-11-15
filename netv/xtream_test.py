"""Tests for xtream.py - Xtream Codes API client."""

from __future__ import annotations

import json

from treekle import ReadError

import pytest

from netv import testing, util
from netv.xtream import (
    ProviderRecord,
    XtreamClient,
    category_ids,
    credentials_path,
    list_field,
    live_stream_url,
    record_field,
    text_field,
)


API = "http://example.com/player_api.php?username=user&password=pass"


def test_credentials_path_keeps_each_credential_one_segment():
    assert credentials_path("a/b c", "p#/w%") == "a%2Fb%20c/p%23%2Fw%25"


@pytest.mark.parametrize(
    ("stream_id", "source_url"),
    [("src1_42", "http://up.test/"), ("42", "http://up.test")],
)
def test_live_stream_url_uses_upstream_id_and_encoded_credentials(
    stream_id: str,
    source_url: str,
):
    stream: ProviderRecord = {
        "stream_id": stream_id,
        "source_url": source_url,
        "source_username": "u/1",
        "source_password": "p#",
    }
    assert live_stream_url(stream) == "http://up.test/live/u%2F1/p%23/42.m3u8"


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

    def test_build_stream_url_encodes_slashes_in_credentials(self):
        client = XtreamClient(base_url="http://example.com", username="a/b", password="p#/w")
        assert client.build_stream_url("live", 1) == "http://example.com/live/a%2Fb/p%23%2Fw/1"
        assert (
            client.build_timeshift_url(1, 30, "2024-01-15:10-00")
            == "http://example.com/timeshift/a%2Fb/p%23%2Fw/30/2024-01-15:10-00/1.ts"
        )

    def test_build_timeshift_url(self):
        client = XtreamClient(base_url="http://example.com", username="user", password="pass")
        url = client.build_timeshift_url(123, 60, "2024-01-15:14-30")
        assert url == "http://example.com/timeshift/user/pass/60/2024-01-15:14-30/123.ts"

    def test_build_timeshift_url_custom_ext(self):
        client = XtreamClient(base_url="http://example.com", username="user", password="pass")
        url = client.build_timeshift_url(123, 30, "2024-01-15:10-00", ext="m3u8")
        assert url == "http://example.com/timeshift/user/pass/30/2024-01-15:10-00/123.m3u8"


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> testing.FakeWeb:
    web = testing.FakeWeb()
    testing.install_web(monkeypatch, web, util)
    return web


def _serve(web: testing.FakeWeb, url: str, data: object) -> None:
    body = json.dumps(data).encode("utf-8")
    web.routes[url] = lambda: testing.FakeResponse(body)


class TestXtreamClientApi:
    """Tests for XtreamClient API methods against a fake provider."""

    @pytest.fixture
    def client(self) -> XtreamClient:
        return XtreamClient(base_url="http://example.com", username="user", password="pass")

    def test_get_live_categories(self, client: XtreamClient, web: testing.FakeWeb):
        categories = [{"category_id": "1", "category_name": "News"}]
        _serve(web, f"{API}&action=get_live_categories", categories)

        assert client.get_live_categories() == categories
        assert web.urls == [f"{API}&action=get_live_categories"]

    def test_get_live_streams(self, client: XtreamClient, web: testing.FakeWeb):
        streams = [{"stream_id": 1, "name": "CNN"}]
        _serve(web, f"{API}&action=get_live_streams", streams)

        assert client.get_live_streams() == streams

    def test_get_live_streams_with_category(self, client: XtreamClient, web: testing.FakeWeb):
        streams = [{"stream_id": 1, "name": "CNN"}]
        _serve(web, f"{API}&action=get_live_streams&category_id=5", streams)

        assert client.get_live_streams(category_id=5) == streams

    def test_get_vod_categories(self, client: XtreamClient, web: testing.FakeWeb):
        categories = [{"category_id": "10", "category_name": "Movies"}]
        _serve(web, f"{API}&action=get_vod_categories", categories)

        assert client.get_vod_categories() == categories

    def test_get_vod_streams(self, client: XtreamClient, web: testing.FakeWeb):
        streams = [{"stream_id": 100, "name": "Movie 1"}]
        _serve(web, f"{API}&action=get_vod_streams", streams)

        assert client.get_vod_streams() == streams

    def test_get_vod_streams_with_category(self, client: XtreamClient, web: testing.FakeWeb):
        streams = [{"stream_id": 100, "name": "Movie 1"}]
        _serve(web, f"{API}&action=get_vod_streams&category_id=10", streams)

        assert client.get_vod_streams(category_id=10) == streams

    def test_get_series_categories(self, client: XtreamClient, web: testing.FakeWeb):
        categories = [{"category_id": "20", "category_name": "Drama"}]
        _serve(web, f"{API}&action=get_series_categories", categories)

        assert client.get_series_categories() == categories

    def test_get_series(self, client: XtreamClient, web: testing.FakeWeb):
        series = [{"series_id": 200, "name": "Show 1"}]
        _serve(web, f"{API}&action=get_series", series)

        assert client.get_series() == series

    def test_get_series_with_category(self, client: XtreamClient, web: testing.FakeWeb):
        series = [{"series_id": 200, "name": "Show 1"}]
        _serve(web, f"{API}&action=get_series&category_id=20", series)

        assert client.get_series(category_id=20) == series

    def test_get_series_info(self, client: XtreamClient, web: testing.FakeWeb):
        info: ProviderRecord = {"info": {"name": "Show 1"}, "episodes": {"1": []}}
        _serve(web, f"{API}&action=get_series_info&series_id=200", info)

        assert client.get_series_info(series_id=200) == info

    def test_get_vod_info(self, client: XtreamClient, web: testing.FakeWeb):
        info: ProviderRecord = {"info": {"name": "Movie 1", "plot": "A story"}}
        _serve(web, f"{API}&action=get_vod_info&vod_id=100", info)

        assert client.get_vod_info(vod_id=100) == info

    def test_get_server_info(self, client: XtreamClient, web: testing.FakeWeb):
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
        # get_server_info calls the API with no action.
        _serve(web, API, server_info)

        result = client.get_server_info()

        assert result == server_info
        assert record_field(result, "user_info")["auth"] == 1

    def test_get_server_info_auth_failed(self, client: XtreamClient, web: testing.FakeWeb):
        _serve(web, API, {"user_info": {"auth": 0}})

        result = client.get_server_info()

        assert record_field(result, "user_info")["auth"] == 0

    def test_get_server_info_uses_shorter_timeout(
        self,
        client: XtreamClient,
        web: testing.FakeWeb,
    ):
        _serve(web, API, {"user_info": {"auth": 1}})

        client.get_server_info()
        client.get_server_info(timeout=4)

        # get_server_info uses 15s timeout by default (vs 30s for other calls).
        assert [f.timeout for f in web.fetches] == [15, 4]

    def test_custom_timeout_passed_through(self, client: XtreamClient, web: testing.FakeWeb):
        _serve(web, f"{API}&action=get_live_categories", [])

        client.get_live_categories()

        # Default timeout is 30s.
        assert web.fetches[0].timeout == 30

    def test_api_encodes_special_chars_in_params(
        self,
        client: XtreamClient,
        web: testing.FakeWeb,
    ):
        url = f"{API}&action=test_action&foo=bar%26baz&key=left%3Dright"
        _serve(web, url, {})

        assert client._api("test_action", foo="bar&baz", key="left=right") == "{}"
        assert web.urls == [url]

    def test_get_short_epg(self, client: XtreamClient, web: testing.FakeWeb):
        epg_data = {
            "epg_listings": [
                {"title": "Show 1", "start": "2024-01-15 14:00:00"},
                {"title": "Show 2", "start": "2024-01-15 15:00:00"},
            ],
        }
        # Default limit is 10.
        _serve(web, f"{API}&action=get_short_epg&stream_id=123&limit=10", epg_data)

        assert client.get_short_epg(stream_id=123) == epg_data

    def test_get_short_epg_custom_limit(self, client: XtreamClient, web: testing.FakeWeb):
        _serve(web, f"{API}&action=get_short_epg&stream_id=456&limit=5", {"epg_listings": []})

        assert client.get_short_epg(stream_id=456, limit=5) == {"epg_listings": []}

    def test_zero_category_means_all(self, client: XtreamClient, web: testing.FakeWeb):
        _serve(web, f"{API}&action=get_vod_streams", [])

        assert client.get_vod_streams(category_id=0) == []
        assert web.urls == [f"{API}&action=get_vod_streams"]

    def test_provider_extensions_are_kept(self, client: XtreamClient, web: testing.FakeWeb):
        streams = [{"stream_id": 1, "vendor": {"extra": [True, None, 1.5]}}]
        _serve(web, f"{API}&action=get_live_streams", streams)

        assert client.get_live_streams() == streams

    def test_wrong_shape_raises(self, client: XtreamClient, web: testing.FakeWeb):
        _serve(web, f"{API}&action=get_live_streams", {"user_info": {"auth": 0}})

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

    def test_defaults_to_none(self):
        assert (
            XtreamClient(base_url="http://example.com", username="user", password="pass").user_agent
            is None
        )

    def test_user_agent_sent_with_request(self, web: testing.FakeWeb):
        _serve(web, f"{API}&action=get_live_streams", [])
        client = XtreamClient(
            base_url="http://example.com",
            username="user",
            password="pass",
            user_agent="TiviMate/4.7.0",
        )
        client.get_live_streams()
        assert web.fetches[0].headers == {"User-agent": "TiviMate/4.7.0"}

    def test_no_user_agent_sends_default(self, web: testing.FakeWeb):
        _serve(web, f"{API}&action=get_live_streams", [])
        XtreamClient(
            base_url="http://example.com",
            username="user",
            password="pass",
        ).get_live_streams()
        assert web.fetches[0].headers == {"User-agent": "VLC/3.0.20 LibVLC/3.0.20"}


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
