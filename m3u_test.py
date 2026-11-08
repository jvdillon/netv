"""Tests for m3u.py."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import json
import time

import pytest

from xtream import ProviderRecord, category_ids, text_field

import cache
import m3u


@pytest.fixture
def m3u_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point m3u's cache at temp directories."""
    monkeypatch.setattr(cache, "SERVER_SETTINGS_FILE", tmp_path / "server_settings.json")
    monkeypatch.setattr(cache, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    # Skip hardware encoder detection in cache.initialize().
    monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", {"software": True})
    cache.USERS_DIR.mkdir(exist_ok=True)
    cache.CACHE_DIR.mkdir(exist_ok=True)
    _reset_memory()
    m3u.get_refresh_in_progress().clear()

    yield

    _reset_memory()
    m3u.get_refresh_in_progress().clear()


def _reset_memory() -> None:
    cache.clear_all_caches()
    cache.get_cache().pop("epg", None)


class _SyncThread:
    """Thread stand-in that runs its target inline on start()."""

    started: list[_SyncThread] = []

    def __init__(self, target: Callable[[], object], daemon: bool = False) -> None:
        self.target = target
        self.daemon = daemon
        self.error: Exception | None = None

    def start(self) -> None:
        _SyncThread.started.append(self)
        try:
            self.target()
        except Exception as e:  # A real thread would swallow it too.
            self.error = e


@pytest.fixture
def sync_thread() -> Iterator[list[_SyncThread]]:
    _SyncThread.started = []
    with patch.object(m3u.threading, "Thread", _SyncThread):
        yield _SyncThread.started


def _write_cache(name: str, data: object, timestamp: float) -> None:
    (cache.CACHE_DIR / f"{name}.json").write_text(
        json.dumps({"data": data, "timestamp": timestamp}),
    )


def _read_cache(name: str) -> object:
    return json.loads((cache.CACHE_DIR / f"{name}.json").read_text())["data"]


def _xtream(source_id: str, *, epg_enabled: bool = True, epg_timeout: int = 120) -> cache.Source:
    return cache.Source(
        id=source_id,
        name=source_id,
        type="xtream",
        url=f"https://{source_id}.example",
        username="u",
        password="p",
        epg_enabled=epg_enabled,
        epg_timeout=epg_timeout,
    )


@pytest.mark.usefixtures("m3u_module")
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


@pytest.mark.usefixtures("m3u_module")
class TestRefreshState:
    def test_get_refresh_in_progress(self):
        m3u.get_refresh_in_progress()


@pytest.mark.usefixtures("m3u_module")
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
        cache.SERVER_SETTINGS_FILE.write_text(
            json.dumps({"user_agent_preset": "custom", "user_agent_custom": "MyAgent/9.9"}),
        )
        assert m3u._make_client(source).user_agent == "MyAgent/9.9"

    def test_fetch_m3u_sends_user_agent(self):
        cache.SERVER_SETTINGS_FILE.write_text(json.dumps({"user_agent_preset": "vlc"}))
        resp = MagicMock()
        resp.read.return_value = b"#EXTM3U\n"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch.object(m3u, "safe_urlopen", return_value=resp) as mock_open:
            m3u.fetch_m3u("http://example.com/list.m3u", "src1")
        assert mock_open.call_args.kwargs["user_agent"] == "VLC/3.0.20 LibVLC/3.0.20"


@pytest.mark.usefixtures("m3u_module")
class TestFetchSourceLiveData:
    def test_can_skip_epg_url_persistence(self):
        source = cache.Source(
            id="source-a",
            name="Provider",
            type="xtream",
            url="https://provider.example",
            username="remote-user",
            password="remote-password",
        )
        client = MagicMock()
        no_records: list[ProviderRecord] = []
        client.get_live_categories.return_value = no_records
        client.get_live_streams.return_value = no_records
        client.epg_url = "https://provider.example/xmltv.php?credentials=private"

        with (
            patch.object(m3u, "_make_client", return_value=client),
            patch.object(m3u, "update_source_epg_url") as update_epg_url,
        ):
            m3u.fetch_source_live_data(source, persist_epg_url=False)

        update_epg_url.assert_not_called()


@pytest.mark.usefixtures("m3u_module")
class TestAggregateSourceData:
    @pytest.mark.parametrize("epg_enabled", [True, False])
    def test_live_sources_share_normalization_without_persisting_epg(self, epg_enabled: bool):
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
        m3u_cats: list[ProviderRecord] = [{"category_id": "m3u_news", "source_id": "m3u"}]
        m3u_streams: list[ProviderRecord] = [
            {"stream_id": "m3u_1", "direct_url": "https://m3u.example/live"},
        ]
        with (
            patch.object(m3u, "get_sources", return_value=sources),
            patch.object(m3u, "_make_client", return_value=client),
            patch.object(
                m3u,
                "fetch_m3u",
                return_value=(m3u_cats, m3u_streams, "https://m3u.example/guide.xml"),
            ),
            patch.object(m3u, "update_source_epg_url") as persist,
        ):
            cats, streams, epg_urls = m3u._fetch_all_live_data()

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

    def test_live_source_failure_does_not_drop_other_sources(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
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
            patch.object(m3u, "get_sources", return_value=[broken, working]),
            patch.object(m3u, "_make_client", side_effect=OSError("offline")),
        ):
            assert m3u._fetch_all_live_data() == (
                [],
                [],
                [(working.url, working.epg_timeout, working.id)],
            )
        assert "Error loading source Offline source" in caplog.text

    def test_vod_sources_use_shared_loader_and_keep_order(self):
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

        def source_client(source: cache.Source) -> MagicMock:
            return clients[source.id]

        with (
            patch.object(m3u, "get_sources", return_value=sources),
            patch.object(m3u, "_make_client", side_effect=source_client),
            patch.object(
                m3u,
                "fetch_source_vod_data",
                wraps=m3u.fetch_source_vod_data,
            ) as fetch,
        ):
            cats, streams = m3u._fetch_vod_data()

        assert [call.args[0] for call in fetch.call_args_list] == [sources[0], sources[2]]
        assert cats == [
            {"category_id": "films", "source_id": "a"},
            {"category_id": "films", "source_id": "b"},
        ]
        assert streams == [{"stream_id": 1, "source_id": "a"}, {"stream_id": 1, "source_id": "b"}]

    def test_vod_source_failure_does_not_drop_other_sources(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        sources = [
            cache.Source(id="broken", name="Offline", type="xtream", url="https://offline.example"),
            cache.Source(id="working", name="Online", type="xtream", url="https://online.example"),
        ]
        client = MagicMock()
        client.get_vod_categories.return_value = [{"category_id": "films"}]
        client.get_vod_streams.return_value = [{"stream_id": 1}]
        with (
            patch.object(m3u, "get_sources", return_value=sources),
            patch.object(m3u, "_make_client", side_effect=[OSError("offline"), client]),
        ):
            cats, streams = m3u._fetch_vod_data()
        assert cats == [{"category_id": "films", "source_id": "working"}]
        assert streams == [{"stream_id": 1, "source_id": "working"}]
        assert "Failed to fetch VOD from source broken" in caplog.text


@pytest.mark.usefixtures("m3u_module")
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


@pytest.mark.usefixtures("m3u_module")
class TestFetchSourceLiveDataPersistence:
    def test_xtream_persists_epg_url_and_respects_disabled_epg(self):
        source = _xtream("x", epg_enabled=False, epg_timeout=33)
        client = MagicMock()
        client.get_live_categories.return_value = [{"category_id": "1"}]
        client.get_live_streams.return_value = [{"stream_id": 9, "category_id": None}]
        client.epg_url = "https://x.example/xmltv.php"
        with (
            patch.object(m3u, "_make_client", return_value=client),
            patch.object(m3u, "update_source_epg_url") as persist,
        ):
            cats, streams, epg_url, timeout = m3u.fetch_source_live_data(source)
        persist.assert_called_once_with("x", "https://x.example/xmltv.php")
        assert cats == [{"category_id": "x_1", "source_id": "x"}]
        assert streams[0]["category_ids"] == []
        assert streams[0]["source_username"] == "u"
        assert streams[0]["source_password"] == "p"
        assert streams[0]["source_type"] == "xtream"
        assert (epg_url, timeout) == (None, 33)

    def test_m3u_persists_detected_epg_url(self):
        source = cache.Source(id="m", name="m", type="m3u", url="https://m.example/list.m3u")
        with (
            patch.object(m3u, "fetch_m3u", return_value=([], [], "https://m/epg")) as fm,
            patch.object(m3u, "update_source_epg_url") as persist,
        ):
            result = m3u.fetch_source_live_data(source)
        fm.assert_called_once_with("https://m.example/list.m3u", "m")
        persist.assert_called_once_with("m", "https://m/epg")
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


@pytest.mark.usefixtures("m3u_module")
class TestLoadAllLiveData:
    def test_fresh_cache_returns_without_refresh(self, sync_thread: list[_SyncThread]):
        _write_cache("live_data", {**_LIVE, "epg_urls": [["http://e", 5, "s"]]}, time.time())
        with patch.object(m3u, "_fetch_all_live_data") as fetch:
            result = m3u.load_all_live_data()
        assert result == (_LIVE["cats"], _LIVE["streams"], [("http://e", 5, "s")])
        fetch.assert_not_called()
        assert sync_thread == []

    def test_stale_cache_refreshes_in_background(self, sync_thread: list[_SyncThread]):
        _write_cache("live_data", _LIVE, time.time() - 100)
        mem = cache.get_cache()
        mem.update({"live_categories": [], "live_streams": [], "epg_error": "kept"})
        new: tuple[list[ProviderRecord], list[ProviderRecord], list[tuple[str, int, str]]] = (
            [{"category_id": "n"}],
            [{"stream_id": 2}],
            [("http://n", 9, "s")],
        )
        with patch.object(m3u, "_fetch_all_live_data", return_value=new):
            result = m3u.load_all_live_data(ttl_seconds=10)
        assert result == (_LIVE["cats"], _LIVE["streams"], [])
        assert len(sync_thread) == 1 and sync_thread[0].daemon
        assert _read_cache("live_data") == {
            "cats": new[0],
            "streams": new[1],
            "epg_urls": [["http://n", 9, "s"]],
        }
        assert mem == {"epg_error": "kept", "epg_urls": new[2]}
        assert m3u.get_refresh_in_progress() == set()

    def test_failed_refresh_clears_in_progress_flag(self, sync_thread: list[_SyncThread]):
        _write_cache("live_data", _LIVE, 0)
        with patch.object(m3u, "_fetch_all_live_data", side_effect=RuntimeError("boom")):
            m3u.load_all_live_data()
        assert isinstance(sync_thread[0].error, RuntimeError)
        assert m3u.get_refresh_in_progress() == set()
        assert _read_cache("live_data") == _LIVE

    def test_stale_cache_skips_refresh_already_running(self, sync_thread: list[_SyncThread]):
        _write_cache("live_data", _LIVE, 0)
        m3u.get_refresh_in_progress().add("live")
        m3u.load_all_live_data()
        assert sync_thread == []

    def test_no_cache_fetches_and_saves(self):
        new: tuple[list[ProviderRecord], list[ProviderRecord], list[tuple[str, int, str]]] = (
            [{"category_id": "n"}],
            [],
            [("http://n", 9, "s")],
        )
        with patch.object(m3u, "_fetch_all_live_data", return_value=new):
            assert m3u.load_all_live_data() == new
        assert _read_cache("live_data") == {
            "cats": [{"category_id": "n"}],
            "streams": [],
            "epg_urls": [["http://n", 9, "s"]],
        }

    def test_cache_filled_while_waiting_for_lock(self):
        filled = ({**_LIVE, "epg_urls": [["http://e", 5, "s"]]}, time.time())
        with (
            patch.object(m3u, "load_file_cache", side_effect=[None, filled]),
            patch.object(m3u, "_fetch_all_live_data") as fetch,
        ):
            result = m3u.load_all_live_data()
        assert result == (_LIVE["cats"], _LIVE["streams"], [("http://e", 5, "s")])
        fetch.assert_not_called()


@pytest.mark.usefixtures("m3u_module")
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


@pytest.mark.usefixtures("m3u_module")
class TestLoadVodData:
    def test_fresh_cache(self, sync_thread: list[_SyncThread]):
        _write_cache("vod_data", _VOD, time.time())
        assert m3u.load_vod_data() == (_VOD["cats"], _VOD["streams"])
        assert sync_thread == []

    def test_stale_cache_refreshes(self, sync_thread: list[_SyncThread]):
        _write_cache("vod_data", _VOD, 0)
        mem = cache.get_cache()
        mem.update({"vod_categories": [], "vod_streams": [], "epg_error": "kept"})
        with patch.object(m3u, "_fetch_vod_data", return_value=([{"n": 1}], [])):
            assert m3u.load_vod_data() == (_VOD["cats"], _VOD["streams"])
        assert _read_cache("vod_data") == {"cats": [{"n": 1}], "streams": []}
        assert mem == {"epg_error": "kept"}
        assert m3u.get_refresh_in_progress() == set()
        assert len(sync_thread) == 1

    def test_stale_cache_skips_refresh_already_running(self, sync_thread: list[_SyncThread]):
        _write_cache("vod_data", _VOD, 0)
        m3u.get_refresh_in_progress().add("vod")
        m3u.load_vod_data()
        assert sync_thread == []

    def test_no_cache_fetches_and_saves(self):
        with patch.object(m3u, "_fetch_vod_data", return_value=([{"n": 1}], [])):
            assert m3u.load_vod_data() == ([{"n": 1}], [])
        assert _read_cache("vod_data") == {"cats": [{"n": 1}], "streams": []}

    def test_empty_fetch_is_not_cached(self):
        with patch.object(m3u, "_fetch_vod_data", return_value=([], [])):
            assert m3u.load_vod_data() == ([], [])
        assert not (cache.CACHE_DIR / "vod_data.json").exists()

    def test_cache_filled_while_waiting_for_lock(self):
        with patch.object(m3u, "load_file_cache", side_effect=[None, (_VOD, 0)]):
            assert m3u.load_vod_data() == (_VOD["cats"], _VOD["streams"])


_SERIES: m3u.SeriesData = {"cats": [{"category_id": "c"}], "series": [{"series_id": 1}]}


@pytest.mark.usefixtures("m3u_module")
class TestLoadSeriesData:
    def test_fresh_cache(self, sync_thread: list[_SyncThread]):
        _write_cache("series_data", _SERIES, time.time())
        assert m3u.load_series_data() == (_SERIES["cats"], _SERIES["series"])
        assert sync_thread == []

    def test_stale_cache_refreshes(self, sync_thread: list[_SyncThread]):
        _write_cache("series_data", _SERIES, 0)
        mem = cache.get_cache()
        mem.update({"series_categories": [], "series": [], "epg_error": "kept"})
        with patch.object(m3u, "_fetch_series_data", return_value=([], [{"n": 1}])):
            assert m3u.load_series_data() == (_SERIES["cats"], _SERIES["series"])
        assert _read_cache("series_data") == {"cats": [], "series": [{"n": 1}]}
        assert mem == {"epg_error": "kept"}
        assert m3u.get_refresh_in_progress() == set()
        assert len(sync_thread) == 1

    def test_stale_cache_skips_refresh_already_running(self, sync_thread: list[_SyncThread]):
        _write_cache("series_data", _SERIES, 0)
        m3u.get_refresh_in_progress().add("series")
        m3u.load_series_data()
        assert sync_thread == []

    def test_no_cache_fetches_and_saves(self):
        with patch.object(m3u, "_fetch_series_data", return_value=([], [{"n": 1}])):
            assert m3u.load_series_data() == ([], [{"n": 1}])
        assert _read_cache("series_data") == {"cats": [], "series": [{"n": 1}]}

    def test_empty_fetch_is_not_cached(self):
        with patch.object(m3u, "_fetch_series_data", return_value=([], [])):
            assert m3u.load_series_data() == ([], [])
        assert not (cache.CACHE_DIR / "series_data.json").exists()

    def test_cache_filled_while_waiting_for_lock(self):
        with patch.object(m3u, "load_file_cache", side_effect=[None, (_SERIES, 0)]):
            assert m3u.load_series_data() == (_SERIES["cats"], _SERIES["series"])


_SOURCES = [
    cache.Source(id="m", name="m", type="m3u", url="https://m"),
    cache.Source(id="a", name="a", type="xtream", url="https://a.example/", username="ua"),
    cache.Source(id="b", name="b", type="xtream", url="https://b.example", username="ub"),
]


@pytest.mark.usefixtures("m3u_module")
class TestXtreamClientLookup:
    def test_first_xtream_client(self):
        with patch.object(m3u, "get_sources", return_value=_SOURCES):
            client = m3u.get_first_xtream_client()
        assert client is not None
        assert (client.base_url, client.username) == ("https://a.example", "ua")

    def test_first_xtream_client_none(self):
        with patch.object(m3u, "get_sources", return_value=_SOURCES[:1]):
            assert m3u.get_first_xtream_client() is None

    def test_client_by_source(self):
        with patch.object(m3u, "get_sources", return_value=_SOURCES):
            client = m3u.get_xtream_client_by_source("b")
            assert client is not None and client.username == "ub"
            assert m3u.get_xtream_client_by_source("m") is None
            assert m3u.get_xtream_client_by_source("missing") is None


@pytest.mark.usefixtures("m3u_module")
class TestFetchSeriesData:
    def test_tags_sources_skips_non_xtream_and_survives_failures(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        sources = [
            _xtream("broken"),
            cache.Source(id="m", name="m", type="m3u", url="https://m"),
            _xtream("ok"),
        ]
        client = MagicMock()
        client.get_series_categories.return_value = [{"category_id": "1"}]
        client.get_series.return_value = [{"series_id": 5}]
        with (
            patch.object(m3u, "get_sources", return_value=sources),
            patch.object(m3u, "_make_client", side_effect=[OSError("down"), client]),
        ):
            cats, series = m3u._fetch_series_data()
        assert cats == [{"category_id": "1", "source_id": "ok"}]
        assert series == [{"series_id": 5, "source_id": "ok"}]
        assert "Failed to fetch series from source broken" in caplog.text


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
