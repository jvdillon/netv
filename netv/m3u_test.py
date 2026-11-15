"""Tests for m3u.py."""

from __future__ import annotations

from urllib import parse

import json
import logging
import time

import pytest

from netv import cache, m3u, testing, util
from netv.xtream import ProviderRecord, category_ids, text_field


@pytest.fixture(autouse=True)
def no_refresh_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m3u, "_refresh_in_progress", set())


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> testing.FakeWeb:
    web = testing.FakeWeb()
    testing.install_web(monkeypatch, web, util)
    return web


@pytest.fixture
def threads(monkeypatch: pytest.MonkeyPatch) -> testing.ThreadRecorder:
    recorder = testing.ThreadRecorder()
    testing.install_threads(monkeypatch, recorder, m3u)
    return recorder


def _configure(*sources: cache.SourceConfig) -> None:
    cache.save_server_settings({**cache.load_server_settings(), "sources": list(sources)})


def _stored_sources() -> list[cache.SourceConfig]:
    return cache.load_server_settings()["sources"]


def _api(base_url: str, action: str, *, username: str = "", password: str = "") -> str:
    query = parse.urlencode({"username": username, "password": password, "action": action})
    return f"{base_url}/player_api.php?{query}"


def _serve(web: testing.FakeWeb, url: str, body: object) -> None:
    payload = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    web.routes[url] = lambda: testing.FakeResponse(payload)


def _write_cache(name: str, data: object, timestamp: float) -> None:
    (cache.CACHE_DIR / f"{name}.json").write_text(
        json.dumps({"data": data, "timestamp": timestamp}),
    )


def _read_cache(name: str) -> object:
    return json.loads((cache.CACHE_DIR / f"{name}.json").read_text())["data"]


_PLAYLIST = '#EXTM3U url-tvg="http://n"\n#EXTINF:-1 group-title="N",New\nhttp://n/1\n'
_PLAYLIST_URL = "https://s.example/list.m3u"


def _serve_playlist(
    web: testing.FakeWeb,
) -> tuple[
    list[ProviderRecord],
    list[ProviderRecord],
    list[tuple[str, int, str]],
]:
    """Configure one M3U source ``s`` and return the live data it yields."""
    _configure({"id": "s", "name": "s", "type": "m3u", "url": _PLAYLIST_URL, "epg_timeout": 9})
    _serve(web, _PLAYLIST_URL, _PLAYLIST)
    cats, streams, epg_url = m3u.parse_m3u(_PLAYLIST, "s")
    return cats, streams, [(epg_url, 9, "s")]


def _serve_xtream(
    web: testing.FakeWeb,
    source_id: str,
    **actions: object,
) -> cache.SourceConfig:
    """Return an Xtream source config whose provider answers ``actions``."""
    url = f"https://{source_id}.example"
    for action, body in actions.items():
        _serve(web, _api(url, action, username="u", password="p"), body)
    return {
        "id": source_id,
        "name": source_id,
        "type": "xtream",
        "url": url,
        "username": "u",
        "password": "p",
    }


class TestParseM3u:
    def test_parse_basic_m3u(self):
        content = """#EXTM3U
#EXTINF:-1 tvg-id="ch1" tvg-logo="http://logo.png" group-title="News",Channel One
http://stream.example.com/ch1.m3u8
#EXTINF:-1 tvg-id="ch2" group-title="Sports",Channel Two
http://stream.example.com/ch2.m3u8
"""
        cats, streams, _ = m3u.parse_m3u(content, "src1")

        assert len(cats) == 2
        assert any(c["category_name"] == "News" for c in cats)
        assert any(c["category_name"] == "Sports" for c in cats)

        assert len(streams) == 2
        assert streams[0]["name"] == "Channel One"
        assert streams[0]["epg_channel_id"] == "ch1"
        assert streams[0]["stream_icon"] == "http://logo.png"
        assert streams[0]["direct_url"] == "http://stream.example.com/ch1.m3u8"
        assert streams[0]["source_id"] == "src1"

    def test_parse_m3u_with_epg_url(self):
        content = """#EXTM3U url-tvg="http://epg.example.com/guide.xml"
#EXTINF:-1,Test Channel
http://test.stream
"""
        _, _, epg_url = m3u.parse_m3u(content, "src1")
        assert epg_url == "http://epg.example.com/guide.xml"

    def test_parse_m3u_x_tvg_url(self):
        content = """#EXTM3U x-tvg-url="http://alt.epg.com/guide.xml"
#EXTINF:-1,Test Channel
http://test.stream
"""
        _, _, epg_url = m3u.parse_m3u(content, "src1")
        assert epg_url == "http://alt.epg.com/guide.xml"

    def test_parse_m3u_uncategorized(self):
        content = """#EXTM3U
#EXTINF:-1,No Group Channel
http://stream.example.com/nogroupch.m3u8
"""
        cats, streams, _ = m3u.parse_m3u(content, "src1")
        assert len(cats) == 1
        assert cats[0]["category_name"] == "Uncategorized"
        assert category_ids(streams[0])[0].endswith("_uncategorized")

    def test_parse_m3u_category_ids_prefixed(self):
        content = """#EXTM3U
#EXTINF:-1 group-title="Movies",Test
http://test
"""
        cats, streams, _ = m3u.parse_m3u(content, "mysource")
        assert text_field(cats[0], "category_id").startswith("mysource_")
        assert category_ids(streams[0])[0].startswith("mysource_")

    def test_parse_m3u_empty(self):
        cats, streams, epg_url = m3u.parse_m3u("", "src1")
        assert cats == []
        assert streams == []
        assert epg_url == ""


class TestParseEpgUrls:
    def test_parse_tuple_list(self):
        raw = [["http://epg1.com", 120, "src1"], ["http://epg2.com", 60, "src2"]]
        result = m3u.parse_epg_urls(raw)
        assert len(result) == 2
        assert result[0] == ("http://epg1.com", 120, "src1")
        assert result[1] == ("http://epg2.com", 60, "src2")

    def test_parse_tuple_passthrough(self):
        raw = [("http://epg.com", 100, "s1")]
        result = m3u.parse_epg_urls(raw)
        assert result[0] == ("http://epg.com", 100, "s1")

    def test_parse_empty(self):
        assert m3u.parse_epg_urls([]) == []

    def test_parse_skips_malformed(self):
        raw = [["http://epg.com", 90], "plain_string", ["http://valid.com", 60, "src"]]
        result = m3u.parse_epg_urls(raw)
        assert len(result) == 1
        assert result[0] == ("http://valid.com", 60, "src")


class TestRefreshState:
    def test_get_refresh_in_progress(self):
        assert m3u.get_refresh_in_progress() is m3u._refresh_in_progress


class TestMakeClientUserAgent:
    """m3u fetches must use the configured User-Agent (issue #57)."""

    def test_make_client_uses_persisted_user_agent(self):
        source = cache.Source(
            id="s",
            name="s",
            type="xtream",
            url="http://example.com",
            username="u",
            password="p",
        )
        cache.save_server_settings(
            {
                **cache.load_server_settings(),
                "user_agent_preset": "custom",
                "user_agent_custom": "MyAgent/9.9",
            },
        )
        assert m3u._make_client(source).user_agent == "MyAgent/9.9"

    def test_fetch_m3u_sends_user_agent(self, web: testing.FakeWeb):
        cache.save_server_settings({**cache.load_server_settings(), "user_agent_preset": "vlc"})
        _serve(web, "http://example.com/list.m3u", "#EXTM3U\n")
        assert m3u.fetch_m3u("http://example.com/list.m3u", "src1") == ([], [], "")
        assert [(f.url, f.timeout, f.headers) for f in web.fetches] == [
            ("http://example.com/list.m3u", 30, {"User-agent": "VLC/3.0.20 LibVLC/3.0.20"}),
        ]


class TestFetchSourceLiveData:
    def test_can_skip_epg_url_persistence(self, web: testing.FakeWeb):
        config = _serve_xtream(web, "a", get_live_categories=[], get_live_streams=[])
        _configure(config)

        result = m3u.fetch_source_live_data(m3u.get_sources()[0], persist_epg_url=False)

        assert result == ([], [], "https://a.example/xmltv.php?username=u&password=p", 120)
        assert "epg_url" not in _stored_sources()[0]


class TestAggregateSourceData:
    @pytest.mark.parametrize("epg_enabled", [True, False])
    def test_live_sources_share_normalization_without_persisting_epg(
        self,
        web: testing.FakeWeb,
        epg_enabled: bool,
    ):
        sources: list[cache.SourceConfig] = [
            {
                "id": kind,
                "name": kind,
                "type": kind,
                "url": f"https://{kind}.example",
                "epg_enabled": epg_enabled,
                "epg_timeout": 90 + index,
            }
            for index, kind in enumerate(("xtream", "m3u", "epg"))
        ]
        _configure(*sources)
        _serve(
            web,
            _api("https://xtream.example", "get_live_categories"),
            [{"category_id": "news"}],
        )
        _serve(
            web,
            _api("https://xtream.example", "get_live_streams"),
            [
                {"stream_id": 1, "category_id": "news"},
                {"stream_id": 2, "category_ids": ["news", "sport"]},
            ],
        )
        playlist = '#EXTM3U url-tvg="https://m3u.example/guide.xml"\n#EXTINF:-1,One\nhttps://m3u.example/live\n'
        _serve(web, "https://m3u.example", playlist)
        m3u_cats, m3u_streams, _ = m3u.parse_m3u(playlist, "m3u")

        cats, streams, epg_urls = m3u._fetch_all_live_data()

        assert cats == [{"category_id": "xtream_news", "source_id": "xtream"}, *m3u_cats]
        assert [s["stream_id"] for s in streams] == [1, 2, "m3u_1"]
        assert streams[0]["category_ids"] == ["xtream_news"]
        assert streams[1]["category_ids"] == ["xtream_news", "xtream_sport"]
        assert all(s["source_id"] == "xtream" for s in streams[:2])
        assert all(s["source_url"] == "https://xtream.example" for s in streams[:2])
        assert streams[-1] == m3u_streams[0]
        assert epg_urls == (
            [
                ("https://xtream.example/xmltv.php?username=&password=", 90, "xtream"),
                ("https://m3u.example/guide.xml", 91, "m3u"),
                ("https://epg.example", 92, "epg"),
            ]
            if epg_enabled
            else []
        )
        assert all("epg_url" not in s for s in _stored_sources())

    def test_live_source_failure_does_not_drop_other_sources(
        self,
        web: testing.FakeWeb,
        caplog: pytest.LogCaptureFixture,
    ):
        _configure(
            {
                "id": "broken",
                "name": "Offline source",
                "type": "xtream",
                "url": "https://offline.example",
            },
            {"id": "working", "name": "EPG source", "type": "epg", "url": "https://epg.example"},
        )
        assert m3u._fetch_all_live_data() == ([], [], [("https://epg.example", 120, "working")])
        assert web.urls == [_api("https://offline.example", "get_live_categories")]
        assert "Error loading source Offline source" in caplog.text

    def test_vod_sources_use_shared_loader_and_keep_order(self, web: testing.FakeWeb):
        vod = {
            "get_vod_categories": [{"category_id": "films"}],
            "get_vod_streams": [{"stream_id": 1}],
        }
        _configure(
            _serve_xtream(web, "a", **vod),
            {
                "id": "playlist",
                "name": "Playlist",
                "type": "m3u",
                "url": "https://playlist.example",
            },
            _serve_xtream(web, "b", **vod),
        )

        cats, streams = m3u._fetch_vod_data()

        assert web.urls == [
            _api("https://a.example", "get_vod_categories", username="u", password="p"),
            _api("https://a.example", "get_vod_streams", username="u", password="p"),
            _api("https://b.example", "get_vod_categories", username="u", password="p"),
            _api("https://b.example", "get_vod_streams", username="u", password="p"),
        ]
        assert cats == [
            {"category_id": "films", "source_id": "a"},
            {"category_id": "films", "source_id": "b"},
        ]
        assert streams == [{"stream_id": 1, "source_id": "a"}, {"stream_id": 1, "source_id": "b"}]

    def test_vod_source_failure_does_not_drop_other_sources(
        self,
        web: testing.FakeWeb,
        caplog: pytest.LogCaptureFixture,
    ):
        _configure(
            _serve_xtream(web, "broken"),
            _serve_xtream(
                web,
                "working",
                get_vod_categories=[{"category_id": "films"}],
                get_vod_streams=[{"stream_id": 1}],
            ),
        )
        cats, streams = m3u._fetch_vod_data()
        assert cats == [{"category_id": "films", "source_id": "working"}]
        assert streams == [{"stream_id": 1, "source_id": "working"}]
        assert "Failed to fetch VOD from source broken" in caplog.text


class TestParseM3uEdgeCases:
    def test_skips_blank_and_directive_lines_before_url(self):
        content = "#EXTM3U\n#EXTINF:-1,Alpha\n\n#EXTVLCOPT:http-user-agent=x\nhttp://a\n"
        _, streams, _ = m3u.parse_m3u(content, "s")
        assert [(s["name"], s["direct_url"]) for s in streams] == [("Alpha", "http://a")]

    def test_trailing_extinf_without_url(self):
        _, streams, _ = m3u.parse_m3u("#EXTM3U\n#EXTINF:-1,Last\n", "s")
        assert streams[0]["direct_url"] == ""

    def test_extinf_without_url_does_not_steal_next_entry(self):
        content = "#EXTM3U\n#EXTINF:-1,A\n#EXTINF:-1,B\nhttp://b\n"
        _, streams, _ = m3u.parse_m3u(content, "s")
        assert [(s["name"], s["direct_url"]) for s in streams] == [("A", ""), ("B", "http://b")]

    def test_extinf_without_comma_is_unknown(self):
        _, streams, _ = m3u.parse_m3u('#EXTINF:-1 tvg-id="x"\nhttp://x\n', "s")
        assert streams[0]["name"] == "Unknown"
        assert streams[0]["epg_channel_id"] == "x"

    def test_category_slug(self):
        cats, streams, _ = m3u.parse_m3u(
            '#EXTINF:-1 group-title="US | News & Talk!",A\nhttp://a\n'
            '#EXTINF:-1 group-title="US | News & Talk!",B\nhttp://b\n',
            "s",
        )
        assert [c["category_id"] for c in cats] == ["s_us_news_talk"]
        assert [s["stream_id"] for s in streams] == ["s_1", "s_2"]


class TestFetchSourceLiveDataPersistence:
    def test_xtream_persists_epg_url_and_respects_disabled_epg(self, web: testing.FakeWeb):
        config = _serve_xtream(
            web,
            "x",
            get_live_categories=[{"category_id": "1"}],
            get_live_streams=[{"stream_id": 9, "category_id": None}],
        )
        _configure({**config, "epg_enabled": False, "epg_timeout": 33})
        cats, streams, epg_url, timeout = m3u.fetch_source_live_data(m3u.get_sources()[0])
        assert (
            _stored_sources()[0].get("epg_url")
            == "https://x.example/xmltv.php?username=u&password=p"
        )
        assert cats == [{"category_id": "x_1", "source_id": "x"}]
        assert streams[0]["category_ids"] == []
        assert streams[0]["source_username"] == "u"
        assert streams[0]["source_password"] == "p"
        assert streams[0]["source_type"] == "xtream"
        assert (epg_url, timeout) == (None, 33)

    def test_m3u_persists_detected_epg_url(self, web: testing.FakeWeb):
        _configure({"id": "m", "name": "m", "type": "m3u", "url": "https://m.example/list.m3u"})
        _serve(web, "https://m.example/list.m3u", '#EXTM3U url-tvg="https://m/epg"\n')
        result = m3u.fetch_source_live_data(m3u.get_sources()[0])
        assert web.urls == ["https://m.example/list.m3u"]
        assert _stored_sources()[0].get("epg_url") == "https://m/epg"
        assert result == ([], [], "https://m/epg", 120)

    def test_unknown_type_yields_nothing(self):
        source = cache.Source(id="z", name="z", type="other", url="https://z")
        assert m3u.fetch_source_live_data(source) == ([], [], None, 120)

    def test_vod_ignores_non_xtream(self):
        source = cache.Source(id="m", name="m", type="m3u", url="https://m")
        assert m3u.fetch_source_vod_data(source) == ([], [])


_LIVE: m3u.LiveData = {
    "cats": [{"category_id": "c"}],
    "streams": [{"stream_id": 1}],
    "epg_urls": [],
}


class TestLoadAllLiveData:
    def test_fresh_cache_returns_without_refresh(
        self,
        web: testing.FakeWeb,
        threads: testing.ThreadRecorder,
    ):
        _serve_playlist(web)
        _write_cache("live_data", {**_LIVE, "epg_urls": [["http://e", 5, "s"]]}, time.time())
        result = m3u.load_all_live_data()
        assert result == (_LIVE["cats"], _LIVE["streams"], [("http://e", 5, "s")])
        assert web.fetches == []
        assert threads.threads == []

    def test_cache_exactly_ttl_old_is_still_fresh(
        self,
        threads: testing.ThreadRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _write_cache("live_data", _LIVE, 900.0)
        testing.install_clock(monkeypatch, testing.FakeClock(now=1000.0), m3u)
        assert m3u.load_all_live_data(ttl_seconds=100) == (_LIVE["cats"], _LIVE["streams"], [])
        assert threads.threads == []

    @pytest.mark.parametrize("memory_has_live_lists", [True, False])
    def test_stale_cache_refreshes_in_background(
        self,
        web: testing.FakeWeb,
        threads: testing.ThreadRecorder,
        caplog: pytest.LogCaptureFixture,
        memory_has_live_lists: bool,
    ):
        new = _serve_playlist(web)
        _write_cache("live_data", _LIVE, time.time() - 100)
        mem = cache.get_cache()
        mem["epg_error"] = "kept"
        if memory_has_live_lists:
            mem.update({"live_categories": [], "live_streams": []})
        caplog.set_level(logging.INFO, logger="netv.m3u")
        result = m3u.load_all_live_data(ttl_seconds=10)
        assert result == (_LIVE["cats"], _LIVE["streams"], [])
        assert len(threads.threads) == 1 and threads.threads[0].daemon
        assert threads.errors == []
        assert _read_cache("live_data") == {
            "cats": new[0],
            "streams": new[1],
            "epg_urls": [["http://n", 9, "s"]],
        }
        assert mem == {"epg_error": "kept", "epg_urls": new[2]}
        assert m3u.get_refresh_in_progress() == set()
        assert caplog.messages == ["Refreshing live data in background", "Live data refreshed"]

    def test_failed_refresh_clears_in_progress_flag(self, threads: testing.ThreadRecorder):
        _write_cache("live_data", _LIVE, 0)
        cache.SERVER_SETTINGS_FILE.write_text(json.dumps({"live_dvr_mins": "corrupt"}))
        m3u.load_all_live_data()
        assert [type(e) for e in threads.errors] == [TypeError]
        assert m3u.get_refresh_in_progress() == set()
        assert _read_cache("live_data") == _LIVE

    def test_stale_cache_skips_refresh_already_running(self, threads: testing.ThreadRecorder):
        _write_cache("live_data", _LIVE, 0)
        m3u.get_refresh_in_progress().add("live")
        m3u.load_all_live_data()
        assert threads.threads == []

    def test_no_cache_fetches_and_saves(
        self,
        web: testing.FakeWeb,
        caplog: pytest.LogCaptureFixture,
    ):
        new = _serve_playlist(web)
        caplog.set_level(logging.INFO, logger="netv.m3u")
        assert m3u.load_all_live_data() == new
        assert caplog.messages == ["No live cache, fetching"]
        assert _read_cache("live_data") == {
            "cats": new[0],
            "streams": new[1],
            "epg_urls": [["http://n", 9, "s"]],
        }

    def test_cache_filled_while_waiting_for_lock(
        self,
        web: testing.FakeWeb,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _serve_playlist(web)
        filled = {**_LIVE, "epg_urls": [["http://e", 5, "s"]]}
        lock = testing.ContendedLock(lambda: cache.save_file_cache("live_data", filled))
        monkeypatch.setattr(m3u, "_fetch_locks", {**m3u._fetch_locks, "live": lock})
        result = m3u.load_all_live_data()
        assert result == (_LIVE["cats"], _LIVE["streams"], [("http://e", 5, "s")])
        assert lock.acquired == 1
        assert web.fetches == []


class TestCachedReaders:
    @pytest.mark.parametrize("name", ["live_data", "vod_data", "series_data"])
    def test_unreadable_cache_is_absent(self, name: str):
        _write_cache(name, {"cats": "not a list"}, time.time())
        readers = {
            "live_data": m3u.cached_live_data,
            "vod_data": m3u.cached_vod_data,
            "series_data": m3u.cached_series_data,
        }
        assert readers[name]() is None

    def test_missing_epg_urls_default_to_none(self):
        _write_cache("live_data", {"cats": [], "streams": []}, 5)
        assert m3u.cached_live_data() == ({"cats": [], "streams": [], "epg_urls": []}, 5)

    def test_cached_streams_keep_unknown_provider_keys(self):
        stream = {"stream_id": 1, "provider_extension": {"x": [True]}}
        _write_cache("live_data", {"cats": [], "streams": [stream]}, 5)
        cached = m3u.cached_live_data()
        assert cached is not None
        assert cached[0]["streams"] == [stream]


_VOD: m3u.VodData = {"cats": [{"category_id": "c"}], "streams": [{"stream_id": 1}]}


def _serve_vod(web: testing.FakeWeb) -> None:
    _configure(_serve_xtream(web, "a", get_vod_categories=[{"n": 1}], get_vod_streams=[]))


class TestLoadVodData:
    def test_fresh_cache(self, threads: testing.ThreadRecorder):
        _write_cache("vod_data", _VOD, time.time())
        assert m3u.load_vod_data() == (_VOD["cats"], _VOD["streams"])
        assert threads.threads == []

    def test_stale_cache_refreshes(self, web: testing.FakeWeb, threads: testing.ThreadRecorder):
        _serve_vod(web)
        _write_cache("vod_data", _VOD, 0)
        mem = cache.get_cache()
        mem.update({"vod_categories": [], "vod_streams": [], "epg_error": "kept"})
        assert m3u.load_vod_data() == (_VOD["cats"], _VOD["streams"])
        assert _read_cache("vod_data") == {"cats": [{"n": 1, "source_id": "a"}], "streams": []}
        assert mem == {"epg_error": "kept"}
        assert m3u.get_refresh_in_progress() == set()
        assert len(threads.threads) == 1 and threads.errors == []

    def test_stale_cache_skips_refresh_already_running(self, threads: testing.ThreadRecorder):
        _write_cache("vod_data", _VOD, 0)
        m3u.get_refresh_in_progress().add("vod")
        m3u.load_vod_data()
        assert threads.threads == []

    def test_no_cache_fetches_and_saves(self, web: testing.FakeWeb):
        _serve_vod(web)
        assert m3u.load_vod_data() == ([{"n": 1, "source_id": "a"}], [])
        assert _read_cache("vod_data") == {"cats": [{"n": 1, "source_id": "a"}], "streams": []}

    def test_empty_fetch_is_not_cached(self):
        assert m3u.load_vod_data() == ([], [])
        assert not (cache.CACHE_DIR / "vod_data.json").exists()

    def test_cache_filled_while_waiting_for_lock(
        self,
        web: testing.FakeWeb,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _serve_vod(web)
        lock = testing.ContendedLock(lambda: cache.save_file_cache("vod_data", _VOD))
        monkeypatch.setattr(m3u, "_fetch_locks", {**m3u._fetch_locks, "vod": lock})
        assert m3u.load_vod_data() == (_VOD["cats"], _VOD["streams"])
        assert lock.acquired == 1
        assert web.fetches == []


_SERIES: m3u.SeriesData = {"cats": [{"category_id": "c"}], "series": [{"series_id": 1}]}


def _serve_series(web: testing.FakeWeb) -> None:
    _configure(_serve_xtream(web, "a", get_series_categories=[], get_series=[{"n": 1}]))


class TestLoadSeriesData:
    def test_fresh_cache(self, threads: testing.ThreadRecorder):
        _write_cache("series_data", _SERIES, time.time())
        assert m3u.load_series_data() == (_SERIES["cats"], _SERIES["series"])
        assert threads.threads == []

    def test_stale_cache_refreshes(self, web: testing.FakeWeb, threads: testing.ThreadRecorder):
        _serve_series(web)
        _write_cache("series_data", _SERIES, 0)
        mem = cache.get_cache()
        mem.update({"series_categories": [], "series": [], "epg_error": "kept"})
        assert m3u.load_series_data() == (_SERIES["cats"], _SERIES["series"])
        assert _read_cache("series_data") == {"cats": [], "series": [{"n": 1, "source_id": "a"}]}
        assert mem == {"epg_error": "kept"}
        assert m3u.get_refresh_in_progress() == set()
        assert len(threads.threads) == 1 and threads.errors == []

    def test_stale_cache_skips_refresh_already_running(self, threads: testing.ThreadRecorder):
        _write_cache("series_data", _SERIES, 0)
        m3u.get_refresh_in_progress().add("series")
        m3u.load_series_data()
        assert threads.threads == []

    def test_no_cache_fetches_and_saves(self, web: testing.FakeWeb):
        _serve_series(web)
        assert m3u.load_series_data() == ([], [{"n": 1, "source_id": "a"}])
        assert _read_cache("series_data") == {"cats": [], "series": [{"n": 1, "source_id": "a"}]}

    def test_empty_fetch_is_not_cached(self):
        assert m3u.load_series_data() == ([], [])
        assert not (cache.CACHE_DIR / "series_data.json").exists()

    def test_cache_filled_while_waiting_for_lock(
        self,
        web: testing.FakeWeb,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _serve_series(web)
        lock = testing.ContendedLock(lambda: cache.save_file_cache("series_data", _SERIES))
        monkeypatch.setattr(m3u, "_fetch_locks", {**m3u._fetch_locks, "series": lock})
        assert m3u.load_series_data() == (_SERIES["cats"], _SERIES["series"])
        assert lock.acquired == 1
        assert web.fetches == []


_SOURCES: list[cache.SourceConfig] = [
    {"id": "m", "name": "m", "type": "m3u", "url": "https://m"},
    {"id": "a", "name": "a", "type": "xtream", "url": "https://a.example/", "username": "ua"},
    {"id": "b", "name": "b", "type": "xtream", "url": "https://b.example", "username": "ub"},
]


class TestXtreamClientLookup:
    def test_first_xtream_client(self):
        _configure(*_SOURCES)
        client = m3u.get_first_xtream_client()
        assert client is not None
        assert (client.base_url, client.username) == ("https://a.example", "ua")

    def test_first_xtream_client_none(self):
        _configure(*_SOURCES[:1])
        assert m3u.get_first_xtream_client() is None

    def test_client_by_source(self):
        _configure(*_SOURCES)
        client = m3u.get_xtream_client_by_source("b")
        assert client is not None and client.username == "ub"
        assert m3u.get_xtream_client_by_source("m") is None
        assert m3u.get_xtream_client_by_source("missing") is None


class TestFetchSeriesData:
    def test_tags_sources_skips_non_xtream_and_survives_failures(
        self,
        web: testing.FakeWeb,
        caplog: pytest.LogCaptureFixture,
    ):
        _configure(
            _serve_xtream(web, "broken"),
            {"id": "m", "name": "m", "type": "m3u", "url": "https://m"},
            _serve_xtream(
                web,
                "ok",
                get_series_categories=[{"category_id": "1"}],
                get_series=[{"series_id": 5}],
            ),
        )
        cats, series = m3u._fetch_series_data()
        assert cats == [{"category_id": "1", "source_id": "ok"}]
        assert series == [{"series_id": 5, "source_id": "ok"}]
        assert "https://m" not in web.urls
        assert "Failed to fetch series from source broken" in caplog.text


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
