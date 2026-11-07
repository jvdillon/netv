"""Tests for m3u.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import json

import pytest

import cache
import m3u


@pytest.fixture
def m3u_module(tmp_path: Path):
    """Import m3u module with mocked cache."""

    cache.SERVER_SETTINGS_FILE = tmp_path / "server_settings.json"
    cache.USERS_DIR = tmp_path / "users"
    cache.USERS_DIR.mkdir(exist_ok=True)
    cache.CACHE_DIR = tmp_path / "cache"
    cache.CACHE_DIR.mkdir(exist_ok=True)
    cache.get_cache().clear()

    yield m3u

    cache.get_cache().clear()


class TestParseM3u:
    def test_parse_basic_m3u(self, m3u_module):
        content = """#EXTM3U
#EXTINF:-1 tvg-id="ch1" tvg-logo="http://logo.png" group-title="News",Channel One
http://stream.example.com/ch1.m3u8
#EXTINF:-1 tvg-id="ch2" group-title="Sports",Channel Two
http://stream.example.com/ch2.m3u8
"""
        cats, streams, _ = m3u_module.parse_m3u(content, "src1")

        assert len(cats) == 2
        assert any(c["category_name"] == "News" for c in cats)
        assert any(c["category_name"] == "Sports" for c in cats)

        assert len(streams) == 2
        assert streams[0]["name"] == "Channel One"
        assert streams[0]["epg_channel_id"] == "ch1"
        assert streams[0]["stream_icon"] == "http://logo.png"
        assert streams[0]["direct_url"] == "http://stream.example.com/ch1.m3u8"
        assert streams[0]["source_id"] == "src1"

    def test_parse_m3u_with_epg_url(self, m3u_module):
        content = """#EXTM3U url-tvg="http://epg.example.com/guide.xml"
#EXTINF:-1,Test Channel
http://test.stream
"""
        _, _, epg_url = m3u_module.parse_m3u(content, "src1")
        assert epg_url == "http://epg.example.com/guide.xml"

    def test_parse_m3u_x_tvg_url(self, m3u_module):
        content = """#EXTM3U x-tvg-url="http://alt.epg.com/guide.xml"
#EXTINF:-1,Test Channel
http://test.stream
"""
        _, _, epg_url = m3u_module.parse_m3u(content, "src1")
        assert epg_url == "http://alt.epg.com/guide.xml"

    def test_parse_m3u_uncategorized(self, m3u_module):
        content = """#EXTM3U
#EXTINF:-1,No Group Channel
http://stream.example.com/nogroupch.m3u8
"""
        cats, streams, _ = m3u_module.parse_m3u(content, "src1")
        assert len(cats) == 1
        assert cats[0]["category_name"] == "Uncategorized"
        assert streams[0]["category_ids"][0].endswith("_uncategorized")

    def test_parse_m3u_category_ids_prefixed(self, m3u_module):
        content = """#EXTM3U
#EXTINF:-1 group-title="Movies",Test
http://test
"""
        cats, streams, _ = m3u_module.parse_m3u(content, "mysource")
        assert cats[0]["category_id"].startswith("mysource_")
        assert streams[0]["category_ids"][0].startswith("mysource_")

    def test_parse_m3u_empty(self, m3u_module):
        cats, streams, epg_url = m3u_module.parse_m3u("", "src1")
        assert cats == []
        assert streams == []
        assert epg_url == ""


class TestParseEpgUrls:
    def test_parse_tuple_list(self, m3u_module):
        raw = [["http://epg1.com", 120, "src1"], ["http://epg2.com", 60, "src2"]]
        result = m3u_module.parse_epg_urls(raw)
        assert len(result) == 2
        assert result[0] == ("http://epg1.com", 120, "src1")
        assert result[1] == ("http://epg2.com", 60, "src2")

    def test_parse_tuple_passthrough(self, m3u_module):
        raw = [("http://epg.com", 100, "s1")]
        result = m3u_module.parse_epg_urls(raw)
        assert result[0] == ("http://epg.com", 100, "s1")

    def test_parse_empty(self, m3u_module):
        assert m3u_module.parse_epg_urls([]) == []

    def test_parse_skips_malformed(self, m3u_module):
        raw = [["http://epg.com", 90], "plain_string", ["http://valid.com", 60, "src"]]
        result = m3u_module.parse_epg_urls(raw)
        assert len(result) == 1
        assert result[0] == ("http://valid.com", 60, "src")


class TestRefreshState:
    def test_get_refresh_in_progress(self, m3u_module):
        rip = m3u_module.get_refresh_in_progress()
        assert isinstance(rip, set)


class TestMakeClientUserAgent:
    """m3u fetches must use the configured User-Agent (issue #57)."""

    def test_make_client_uses_persisted_user_agent(self, m3u_module):
        source = SimpleNamespace(url="http://example.com", username="u", password="p")
        cache.SERVER_SETTINGS_FILE.write_text(
            json.dumps({"user_agent_preset": "custom", "user_agent_custom": "MyAgent/9.9"}),
        )
        assert m3u_module._make_client(source).user_agent == "MyAgent/9.9"

    def test_fetch_m3u_sends_user_agent(self, m3u_module):
        cache.SERVER_SETTINGS_FILE.write_text(json.dumps({"user_agent_preset": "vlc"}))
        resp = MagicMock()
        resp.read.return_value = b"#EXTM3U\n"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch.object(m3u_module, "safe_urlopen", return_value=resp) as mock_open:
            m3u_module.fetch_m3u("http://example.com/list.m3u", "src1")
        assert mock_open.call_args.kwargs["user_agent"] == "VLC/3.0.20 LibVLC/3.0.20"


class TestFetchSourceLiveData:
    def test_can_skip_epg_url_persistence(self, m3u_module):
        source = SimpleNamespace(
            id="source-a",
            name="Provider",
            type="xtream",
            url="https://provider.example",
            username="remote-user",
            password="remote-password",
            epg_enabled=True,
            epg_timeout=120,
        )
        client = MagicMock()
        client.get_live_categories.return_value = []
        client.get_live_streams.return_value = []
        client.epg_url = "https://provider.example/xmltv.php?credentials=private"

        with (
            patch.object(m3u_module, "_make_client", return_value=client),
            patch.object(m3u_module, "update_source_epg_url") as update_epg_url,
        ):
            m3u_module.fetch_source_live_data(source, persist_epg_url=False)

        update_epg_url.assert_not_called()


class TestAggregateSourceData:
    @pytest.mark.parametrize("epg_enabled", [True, False])
    def test_live_sources_share_normalization_without_persisting_epg(self, m3u_module, epg_enabled):
        sources = [
            cache.Source(
                id=kind,
                name=kind,
                type=kind,
                url=f"https://{kind}.example",
                epg_enabled=epg_enabled,
                epg_timeout=90 + index,
            )
            for index, kind in enumerate(("xtream", "m3u", "epg"))
        ]
        client = MagicMock()
        client.get_live_categories.return_value = [{"category_id": "news"}]
        client.get_live_streams.return_value = [
            {"stream_id": 1, "category_id": "news"},
            {"stream_id": 2, "category_ids": ["news", "sport"]},
        ]
        client.epg_url = "https://xtream.example/guide.xml"
        m3u_cats = [{"category_id": "m3u_news", "source_id": "m3u"}]
        m3u_streams = [{"stream_id": "m3u_1", "direct_url": "https://m3u.example/live"}]
        with (
            patch.object(m3u_module, "get_sources", return_value=sources),
            patch.object(m3u_module, "_make_client", return_value=client),
            patch.object(
                m3u_module,
                "fetch_m3u",
                return_value=(m3u_cats, m3u_streams, "https://m3u.example/guide.xml"),
            ),
            patch.object(m3u_module, "update_source_epg_url") as persist,
        ):
            cats, streams, epg_urls = m3u_module._fetch_all_live_data()

        assert cats == [{"category_id": "xtream_news", "source_id": "xtream"}, *m3u_cats]
        assert [s["stream_id"] for s in streams] == [1, 2, "m3u_1"]
        assert streams[0]["category_ids"] == ["xtream_news"]
        assert streams[1]["category_ids"] == ["xtream_news", "xtream_sport"]
        assert all(s["source_id"] == "xtream" for s in streams[:2])
        assert all(s["source_url"] == sources[0].url for s in streams[:2])
        assert streams[-1] == m3u_streams[0]
        assert epg_urls == (
            [
                ("https://xtream.example/guide.xml", 90, "xtream"),
                ("https://m3u.example/guide.xml", 91, "m3u"),
                ("https://epg.example", 92, "epg"),
            ]
            if epg_enabled
            else []
        )
        persist.assert_not_called()

    def test_live_source_failure_does_not_drop_other_sources(self, m3u_module, caplog):
        broken = cache.Source(
            id="broken",
            name="Offline source",
            type="xtream",
            url="https://offline.example",
        )
        working = cache.Source(
            id="working",
            name="EPG source",
            type="epg",
            url="https://epg.example",
        )
        with (
            patch.object(m3u_module, "get_sources", return_value=[broken, working]),
            patch.object(m3u_module, "_make_client", side_effect=OSError("offline")),
        ):
            assert m3u_module._fetch_all_live_data() == (
                [],
                [],
                [(working.url, working.epg_timeout, working.id)],
            )
        assert "Error loading source Offline source" in caplog.text

    def test_vod_sources_use_shared_loader_and_keep_order(self, m3u_module):
        sources = [
            cache.Source(id="a", name="First", type="xtream", url="https://a.example"),
            cache.Source(
                id="playlist",
                name="Playlist",
                type="m3u",
                url="https://playlist.example",
            ),
            cache.Source(id="b", name="Second", type="xtream", url="https://b.example"),
        ]
        clients = {source.id: MagicMock() for source in sources if source.type == "xtream"}
        for client in clients.values():
            client.get_vod_categories.return_value = [{"category_id": "films"}]
            client.get_vod_streams.return_value = [{"stream_id": 1}]
        with (
            patch.object(m3u_module, "get_sources", return_value=sources),
            patch.object(m3u_module, "_make_client", side_effect=lambda s: clients[s.id]),
            patch.object(
                m3u_module,
                "fetch_source_vod_data",
                wraps=m3u_module.fetch_source_vod_data,
            ) as fetch,
        ):
            cats, streams = m3u_module._fetch_vod_data()

        assert [call.args[0] for call in fetch.call_args_list] == [sources[0], sources[2]]
        assert cats == [
            {"category_id": "films", "source_id": "a"},
            {"category_id": "films", "source_id": "b"},
        ]
        assert streams == [{"stream_id": 1, "source_id": "a"}, {"stream_id": 1, "source_id": "b"}]

    def test_vod_source_failure_does_not_drop_other_sources(self, m3u_module, caplog):
        sources = [
            cache.Source(id="broken", name="Offline", type="xtream", url="https://offline.example"),
            cache.Source(id="working", name="Online", type="xtream", url="https://online.example"),
        ]
        client = MagicMock()
        client.get_vod_categories.return_value = [{"category_id": "films"}]
        client.get_vod_streams.return_value = [{"stream_id": 1}]
        with (
            patch.object(m3u_module, "get_sources", return_value=sources),
            patch.object(m3u_module, "_make_client", side_effect=[OSError("offline"), client]),
        ):
            cats, streams = m3u_module._fetch_vod_data()
        assert cats == [{"category_id": "films", "source_id": "working"}]
        assert streams == [{"stream_id": 1, "source_id": "working"}]
        assert "Failed to fetch VOD from source broken" in caplog.text


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
