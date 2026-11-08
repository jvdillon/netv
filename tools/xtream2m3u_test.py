"""Tests for the Xtream-to-M3U fetch and playlist writer."""

from __future__ import annotations

from email import message
from typing import Any
from urllib import error, parse, request

import gzip
import json
import pathlib
import pickle
import runpy
import sys
import time

import pytest

from tools import xtream2m3u


SCRIPT = pathlib.Path(xtream2m3u.__file__).resolve()

AUTH = {
    "user_info": {
        "status": "Active",
        "max_connections": "2",
        "username": "u",
        "password": "p",
    },
    "server_info": {
        "server_protocol": "http",
        "url": "host.example",
        "https_port": "443",
        "xui": True,
        "version": "1.5",
    },
}
LIVE_CATEGORIES = [
    {"category_id": "1", "category_name": "News"},
    {"category_id": "2", "category_name": "Locals"},
]
LIVE = [
    {
        "stream_type": "live",
        "stream_id": 10,
        "name": "CNN",
        "stream_icon": "logo",
        "category_ids": [1],
        "epg_channel_id": "cnn.us",
        "num": 1,
        "tv_archive": 0,
    },
    {
        "stream_type": "radio_streams",
        "stream_id": 11,
        "name": "",
        "title": "KQED Radio",
        "stream_icon": "r",
        "category_ids": [2],
        "epg_channel_id": None,
        "num": 2,
        "tv_archive": 1,
    },
    {
        "stream_type": "live",
        "stream_id": 12,
        "name": "Other Local",
        "stream_icon": "x",
        "category_ids": [2],
        "epg_channel_id": None,
        "num": 3,
        "tv_archive": 0,
    },
]
VOD_CATEGORIES = [{"category_id": "5", "category_name": "Movies"}]
VOD = [
    {
        "stream_type": "movie",
        "stream_id": 20,
        "name": "Film",
        "stream_icon": "f",
        "category_ids": [5],
        "num": 1,
        "container_extension": "mkv",
        "year": "1999",
        "rating": "7.5",
    },
]
SERIES_CATEGORIES = [{"category_id": "7", "category_name": "Drama"}]
SERIES = [
    {
        "series_id": 30,
        "name": "Show",
        "cover": "c",
        "category_ids": [7],
        "num": 1,
        "last_modified": "100",
        "year": "",
        "rating": None,
    },
]
SERIES_INFO = {
    "info": {"last_modified": "100"},
    "episodes": {"1": [{"id": "301", "title": "Show S01E01", "container_extension": "mp4"}]},
}
CONFIG = {
    "url": "https://prov.example/",
    "username": "u",
    "password": "p",
    "live_filter": {"2": "Locals", "1": "News"},
    "locals_group": "Locals",
    "locals_filter": ["KQED"],
}
API_URL = "https://prov.example/player_api.php?username=u&password=p"
EPG_URL = "https://prov.example/xmltv.php?username=u&password=p"

EXPECTED_LIVE = f"""\
#EXTM3U url-tvg="{EPG_URL}" x-tvg-url="{EPG_URL}"
#EXT-X-SESSION-DATA:DATA-ID="com.xui.1.5"
#EXTINF:-1 tvg-id="" tvg-name="KQED Radio" tvg-logo="r" group-title="TV | Locals",KQED Radio
https://host.example:443/u/p/11
#EXTINF:-1 tvg-id="cnn.us" tvg-name="CNN" tvg-logo="logo" group-title="TV | News",CNN
https://host.example:443/u/p/10
"""
EXPECTED_VOD = """\
#EXTM3U
#EXT-X-SESSION-DATA:DATA-ID="com.xui.1.5"
#EXTINF:-1 tvg-name="Film" tvg-logo="f" group-title="VOD | Movies",Film
https://host.example:443/movie/u/p/20.mkv
"""
EXPECTED_SERIES = """\
#EXTM3U
#EXT-X-SESSION-DATA:DATA-ID="com.xui.1.5"
#EXTINF:-1 tvg-name="Show S01E01" tvg-logo="c" group-title="Series | Drama",Show S01E01
https://host.example:443/series/u/p/301.mp4
"""


def fake_api(url: str, timeout: int = 5) -> str:
    del timeout
    action = parse.parse_qs(parse.urlparse(url).query).get("action", [None])[0]
    payload = {
        None: AUTH,
        "get_live_streams": LIVE,
        "get_live_categories": LIVE_CATEGORIES,
        "get_vod_streams": VOD,
        "get_vod_categories": VOD_CATEGORIES,
        "get_series": SERIES,
        "get_series_categories": SERIES_CATEGORIES,
        "get_series_info": SERIES_INFO,
    }[action]
    return json.dumps(payload)


@pytest.fixture
def workdir(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_main_fetches_and_writes_playlists(
    workdir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (workdir / "xtream.json").write_text(json.dumps(CONFIG))
    urls: list[str] = []

    def fetch(url: str, timeout: int = 5) -> str:
        urls.append(url)
        return fake_api(url, timeout)

    monkeypatch.setattr(xtream2m3u, "fetch_text", fetch)
    monkeypatch.setattr(sys, "argv", ["xtream2m3u", "--dir", str(workdir)])
    assert xtream2m3u.main() == 0
    assert (workdir / "live.m3u").read_text() == EXPECTED_LIVE
    assert (workdir / "vod.m3u").read_text() == EXPECTED_VOD
    assert (workdir / "series.m3u").read_text() == EXPECTED_SERIES
    assert urls[0] == API_URL
    assert (workdir / "get_vod_categories.json").read_text() == json.dumps(VOD_CATEGORIES)
    assert urls[-1] == API_URL + "&action=get_series_info&series_id=30"
    with gzip.open(workdir / "series_info.pickle.gz", "rb") as f:
        assert pickle.load(f) == {30: SERIES_INFO}

    for name in ("live.m3u", "vod.m3u", "series.m3u"):
        (workdir / name).unlink()
    monkeypatch.setattr(xtream2m3u, "fetch_text", None)
    monkeypatch.setattr(sys, "argv", ["xtream2m3u", "--dir", str(workdir), "--cached-only"])
    assert xtream2m3u.main() == 0
    assert (workdir / "series.m3u").read_text() == EXPECTED_SERIES


def test_script_entrypoint_runs_main(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "xtream.json").write_text(json.dumps(CONFIG))
    monkeypatch.setattr(
        request,
        "urlopen",
        lambda url, timeout: FakeResponse(fake_api(url).encode()),
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 0
    assert (tmp_path / "live.m3u").read_text() == EXPECTED_LIVE


def test_main_defaults_to_cwd_not_script_dir(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_config = SCRIPT.parent / "xtream.json"
    before = script_config.stat().st_mtime_ns if script_config.exists() else None
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["xtream2m3u"])
    with pytest.raises(SystemExit, match="Created xtream.json - edit"):
        xtream2m3u.main()
    assert [p.name for p in tmp_path.iterdir()] == ["xtream.json"]
    assert (script_config.stat().st_mtime_ns if script_config.exists() else None) == before


def test_fetch_text_success_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="Unsupported URL scheme: ftp"):
        xtream2m3u.fetch_text("ftp://x")

    def urlopen(url: str, timeout: int) -> Any:
        assert timeout == 7
        if url.endswith("ok"):
            return FakeResponse("héllo".encode())
        if url.endswith("429"):
            raise error.HTTPError(url, 429, "slow", message.Message(), None)
        if url.endswith("500"):
            raise error.HTTPError(url, 500, "bad", message.Message(), None)
        if url.endswith("dns"):
            raise error.URLError("no host")
        raise TimeoutError("timed out")

    monkeypatch.setattr(request, "urlopen", urlopen)
    assert xtream2m3u.fetch_text("https://x/ok", timeout=7) == "héllo"
    with pytest.raises(xtream2m3u.RetryableError, match="http error 429"):
        xtream2m3u.fetch_text("https://x/429", timeout=7)
    with pytest.raises(ValueError, match="http error 500"):
        xtream2m3u.fetch_text("https://x/500", timeout=7)
    with pytest.raises(xtream2m3u.RetryableError, match="timeout no host"):
        xtream2m3u.fetch_text("https://x/dns", timeout=7)
    with pytest.raises(xtream2m3u.RetryableError, match="timeout timed out"):
        xtream2m3u.fetch_text("http://x/slow", timeout=7)


def _series(last_modified: int) -> Any:
    return {"last_modified": last_modified}


def test_fetch_series_info_refreshes_cache(
    workdir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = workdir / "series_info.pickle.gz"
    fresh = {"info": {"last_modified": "50"}}
    with gzip.open(cache, "wb") as f:
        pickle.dump({1: fresh, 2: {"info": {}}, 3: {"info": {"last_modified": "9"}}}, f)
    calls: list[int] = []

    def fetch(url: str, timeout: int) -> str:
        assert timeout == 60
        id_ = int(url.rsplit("=", 1)[1])
        calls.append(id_)
        if id_ == 2:
            return json.dumps({"info": {"last_modified": "10"}})
        if id_ == 4:
            raise xtream2m3u.RetryableError("try again")
        return "not json"

    monkeypatch.setattr(xtream2m3u, "fetch_text", fetch)
    series = {1: _series(50), 2: _series(10), 4: _series(1), 5: _series(1)}
    result = xtream2m3u.fetch_series_info(series, "https://api", workdir)
    assert result == {1: fresh, 2: {"info": {"last_modified": "10"}}, 4: None, 5: {}}
    assert sorted(calls) == [2, 4, 4, 4, 5]
    with gzip.open(cache, "rb") as f:
        assert pickle.load(f) == result
    assert not cache.with_suffix(".tmp").exists()


def test_fetch_series_info_cache_miss_cached_only(workdir: pathlib.Path) -> None:
    result = xtream2m3u.fetch_series_info(
        {7: _series(1)},
        "https://api",
        workdir,
        cached_only=True,
    )
    assert result == {7: None}
    assert not (workdir / "series_info.pickle.gz").exists()


def test_fetch_series_info_unchanged_skips_write(workdir: pathlib.Path) -> None:
    cache = workdir / "series_info.pickle.gz"
    with gzip.open(cache, "wb") as f:
        pickle.dump({1: {"info": {"last_modified": "5"}}}, f)
    before = cache.stat().st_mtime_ns
    result = xtream2m3u.fetch_series_info({1: _series(5)}, "https://api", workdir)
    assert result == {1: {"info": {"last_modified": "5"}}}
    assert cache.stat().st_mtime_ns == before


def test_fetch_series_info_keyboard_interrupt_cancels(
    workdir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def task(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(xtream2m3u, "_task", task)
    with pytest.raises(KeyboardInterrupt):
        xtream2m3u.fetch_series_info({1: _series(1)}, "https://api", workdir)
    assert "Cancelling..." in capsys.readouterr().out
    assert not (workdir / "series_info.pickle.gz").exists()


def _video(name: str, groups: list[str], cats: list[int]) -> Any:
    return {"tvg-name": name, "group-title": groups, "category_ids": cats}


def test_filter_live_orders_and_filters_locals() -> None:
    live = {
        1: _video("A", ["News"], [1]),
        2: _video("KQED", ["Locals"], [2]),
        3: _video("Other", ["Locals"], [2]),
        4: _video("Skip", ["Sports"], [3]),
    }
    result = xtream2m3u.filter_live(live, {2: "Locals", 1: "News"}, "Locals", {"KQED"})
    assert list(result) == [2, 1]
    assert xtream2m3u.filter_live(live, {}, "", set()) is live
    with pytest.raises(ValueError, match="Expected single group-title"):
        xtream2m3u.filter_live({1: _video("A", ["x", "y"], [1])}, {1: "x"}, "", set())


def test_process_handles_every_stream_type(capsys: pytest.CaptureFixture[str]) -> None:
    elements, categories = xtream2m3u.process(
        [*LIVE, {"stream_type": "unknown", "name": "?"}],
        LIVE_CATEGORIES,
        ["https:/", "h:1", "u", "p"],
    )
    assert categories == {1: "News", 2: "Locals"}
    assert list(elements) == [10, 11, 12]
    assert elements[11]["tvg-name"] == "KQED Radio"
    assert elements[11]["tvg-id"] == ""
    assert elements[10]["url"] == "https://h:1/u/p/10"
    assert "Unrecognized stream_type='unknown'" in capsys.readouterr().out

    series, _ = xtream2m3u.process(SERIES, SERIES_CATEGORIES, ["s"])
    assert series[30] == {
        "tvg-name": "Show",
        "tvg-logo": "c",
        "group-title": ["Drama"],
        "tvg-id": None,
        "url": None,
        "category_ids": [7],
        "year": None,
        "rating": None,
        "num": 1,
        "last_modified": 100,
    }
    vod, _ = xtream2m3u.process(VOD, VOD_CATEGORIES, ["m"])
    assert vod[20]["url"] == "m/20.mkv"
    assert (vod[20]["year"], vod[20]["rating"]) == (1999, 7.5)

    with pytest.raises(ValueError, match="Invalid tv_archive"):
        xtream2m3u.process([{**LIVE[0], "tv_archive": 2}], LIVE_CATEGORIES, [])
    with pytest.raises(ValueError, match="Duplicate id 10"):
        xtream2m3u.process([LIVE[0], LIVE[0]], LIVE_CATEGORIES, [])


def test_process_iptv_url_validates_auth() -> None:
    assert xtream2m3u.process_iptv_url(AUTH) == ["https:/", "host.example:443", "u", "p"]

    def auth(user: dict[str, Any], server: dict[str, Any]) -> dict[str, Any]:
        return {
            "user_info": {**AUTH["user_info"], **user},
            "server_info": {**AUTH["server_info"], **server},
        }

    with pytest.raises(ValueError, match="status='Banned'"):
        xtream2m3u.process_iptv_url(auth({"status": "Banned"}, {}))
    with pytest.raises(ValueError, match="max_connections=0"):
        xtream2m3u.process_iptv_url(auth({"max_connections": "0"}, {}))
    with pytest.raises(ValueError, match="server_protocol='rtmp'"):
        xtream2m3u.process_iptv_url(auth({}, {"server_protocol": "rtmp"}))


def test_load_helpers_check_types(workdir: pathlib.Path) -> None:
    (workdir / "d.json").write_text('{"a": 1}')
    (workdir / "l.json").write_text("[1]")
    assert xtream2m3u.load_dict(workdir / "d.json") == {"a": 1}
    assert xtream2m3u.load_list(workdir / "l.json") == [1]
    with pytest.raises(TypeError, match=r"Expected dict from .*l\.json"):
        xtream2m3u.load_dict(workdir / "l.json")
    with pytest.raises(TypeError, match=r"Expected list from .*d\.json"):
        xtream2m3u.load_list(workdir / "d.json")


def test_load_config_creates_template(workdir: pathlib.Path) -> None:
    with pytest.raises(SystemExit, match="edit with your credentials"):
        xtream2m3u._load_config(workdir)
    template = json.loads((workdir / "xtream.json").read_text())
    assert xtream2m3u._load_config(workdir) == template
    assert template["username"] == "your_username"
    assert xtream2m3u._get_filters(template) == ({}, "", set())
    assert xtream2m3u._get_urls(template)[1] == (
        "https://your-provider.com/get.php?username=your_username&password=your_password"
    )


def test_sliding_rate_limiter_waits_when_full() -> None:
    limiter = xtream2m3u.SlidingRateLimiter(max_calls=1, per_seconds=0.05)
    t0 = time.perf_counter()
    limiter.acquire()
    limiter.acquire()
    assert time.perf_counter() - t0 >= 0.05
    assert len(limiter.requests) == 1


def test_chunking_rate_limiter_resets() -> None:
    limiter = xtream2m3u.ChunkingRateLimiter(max_calls=1, per_seconds=0.05)
    t0 = time.perf_counter()
    limiter.acquire()
    limiter.acquire()
    assert time.perf_counter() - t0 >= 0.04
    assert limiter.calls == 1
    limiter.per_seconds = 0
    limiter.acquire()
    assert limiter.calls == 1


def test_print_progress_bar(capsys: pytest.CaptureFixture[str]) -> None:
    xtream2m3u.print_progress_bar(1, 0)
    assert capsys.readouterr().out == ""
    xtream2m3u.print_progress_bar(1, 2, prefix="P", suffix="S", length=4)
    assert capsys.readouterr().out == "\rP |##--| 50.0% S\r".replace("#", "█")
    xtream2m3u.print_progress_bar(2, 2, length=2)
    assert capsys.readouterr().out == "\r |██| 100.0% \r\n"


def test_write_m3u_series_skips_bad_info(
    workdir: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    series: dict[int, Any] = {
        1: {"tvg-logo": "a", "group-title": ["G"]},
        2: {"tvg-logo": "b", "group-title": ["G"]},
        3: {"tvg-logo": "c", "group-title": ["G"]},
        4: {"tvg-logo": "d", "group-title": ["H"]},
    }
    info: dict[int, Any] = {
        2: {"info": {}},
        3: ["episodes"],
        4: {"episodes": [{"season": {"id": 9, "title": "E", "container_extension": "ts"}}]},
    }
    auth = {"server_info": {}}
    xtream2m3u.write_m3u_series(series, info, auth, ["s"], workdir)
    assert "Series 3:" in capsys.readouterr().out
    assert (workdir / "series.m3u").read_text() == (
        '#EXTM3U\n#EXTINF:-1 tvg-name="E" tvg-logo="d" group-title="Series | H",E\ns/9.ts\n'
    )
    xtream2m3u.write_m3u_vod({}, auth, workdir)
    assert (workdir / "vod.m3u").read_text() == "#EXTM3U\n"
    xtream2m3u.write_m3u_live({}, auth, "e", workdir)
    assert (workdir / "live.m3u").read_text() == '#EXTM3U url-tvg="e" x-tvg-url="e"\n'


def test_descend_yields_nested_episodes() -> None:
    tree = {"a": [{"id": 1}, {"b": {"id": 2, "x": {"id": 3}}}], "c": 5}
    assert list(xtream2m3u._descend(tree)) == [{"id": 1}, {"id": 2, "x": {"id": 3}}]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
