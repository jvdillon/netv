"""Tests for util.py."""

from __future__ import annotations

from collections.abc import Callable
from http.client import HTTPMessage
from typing import TypedDict, cast
from urllib import error, request

import io
import ipaddress
import json
import logging
import pathlib
import socket
import ssl

from treekle import MutablePlainTree

import pytest

from netv import testing, util


class TestDataDir:
    @pytest.fixture(autouse=True)
    def _isolated_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
        monkeypatch.delenv("NETV_DATA_DIR", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    def test_env_overrides_everything(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ):
        (tmp_path / "cache").mkdir()
        monkeypatch.setenv("NETV_DATA_DIR", "~/netv-state")
        assert util.data_dir(tmp_path) == pathlib.Path("~/netv-state").expanduser()

    def test_defaults_to_user_data_dir(self, tmp_path: pathlib.Path):
        (tmp_path / "pyproject.toml").touch()
        assert util.data_dir(tmp_path / "netv") == tmp_path / "xdg" / "netv"
        assert not (tmp_path / "xdg").exists()

    def test_without_xdg_uses_local_share(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ):
        monkeypatch.delenv("XDG_DATA_HOME")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setattr(util.userdirs.sys, "platform", "linux")
        assert util.data_dir(tmp_path) == tmp_path / "home" / ".local" / "share" / "netv"

    @pytest.mark.parametrize("legacy", ["cache", ".cache"])
    def test_moves_legacy_checkout_state(self, tmp_path: pathlib.Path, legacy: str):
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / legacy).mkdir()
        (tmp_path / legacy / "server_settings.json").write_text("{}")
        assert util.data_dir(tmp_path / "netv") == tmp_path / "xdg" / "netv"
        assert (tmp_path / "xdg" / "netv" / "server_settings.json").read_text() == "{}"
        assert not (tmp_path / legacy).exists()

    def test_move_creates_missing_parents(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "a" / "b"))
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache" / "server_settings.json").write_text("{}")
        assert util.data_dir(tmp_path) == tmp_path / "a" / "b" / "netv"
        assert (tmp_path / "a" / "b" / "netv" / "server_settings.json").exists()

    def test_move_into_existing_base_dir(self, tmp_path: pathlib.Path):
        (tmp_path / "xdg").mkdir()
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache" / "server_settings.json").write_text("{}")
        assert util.data_dir(tmp_path) == tmp_path / "xdg" / "netv"
        assert (tmp_path / "xdg" / "netv" / "server_settings.json").exists()

    def test_empty_env_falls_back_to_user_data_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ):
        monkeypatch.setenv("NETV_DATA_DIR", "")
        assert util.data_dir(tmp_path) == tmp_path / "xdg" / "netv"

    def test_moves_legacy_cache_before_dot_cache(self, tmp_path: pathlib.Path):
        for legacy in ("cache", ".cache"):
            (tmp_path / legacy).mkdir()
            (tmp_path / legacy / "server_settings.json").write_text(legacy)
        util.data_dir(tmp_path)
        assert (tmp_path / "xdg" / "netv" / "server_settings.json").read_text() == "cache"
        assert (tmp_path / ".cache").exists()

    def test_keeps_existing_user_data_dir(self, tmp_path: pathlib.Path):
        (tmp_path / "xdg" / "netv").mkdir(parents=True)
        (tmp_path / "xdg" / "netv" / "server_settings.json").write_text("new")
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache" / "server_settings.json").write_text("{}")
        assert util.data_dir(tmp_path) == tmp_path / "xdg" / "netv"
        assert (tmp_path / "xdg" / "netv" / "server_settings.json").read_text() == "new"
        assert (tmp_path / ".cache" / "server_settings.json").exists()

    def test_moves_legacy_state_beside_tool_dirs(self, tmp_path: pathlib.Path):
        (tmp_path / "xdg" / "netv" / "zap2xml").mkdir(parents=True)
        (tmp_path / ".cache" / "users").mkdir(parents=True)
        (tmp_path / ".cache" / "server_settings.json").write_text("{}")
        assert util.data_dir(tmp_path) == tmp_path / "xdg" / "netv"
        assert sorted(p.name for p in (tmp_path / "xdg" / "netv").iterdir()) == [
            "server_settings.json",
            "users",
            "zap2xml",
        ]
        assert not (tmp_path / ".cache").exists()

    def test_ignores_legacy_dir_without_settings(self, tmp_path: pathlib.Path):
        (tmp_path / ".cache" / "tools").mkdir(parents=True)
        assert util.data_dir(tmp_path) == tmp_path / "xdg" / "netv"
        assert (tmp_path / ".cache" / "tools").exists()
        assert not (tmp_path / "xdg").exists()


def test_atomic_write_json_replaces_file_and_preserves_permissions(tmp_path: pathlib.Path):
    path = tmp_path / "settings.json"
    path.write_text('{"old": true}')
    path.chmod(0o640)

    util.atomic_write_json(path, {"new": True})

    assert json.loads(path.read_text()) == {"new": True}
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_json_removes_temp_file_on_failure(tmp_path: pathlib.Path):
    path = tmp_path / "settings.json"
    with pytest.raises(TypeError):
        util.atomic_write_json(path, {"bad": object()})
    assert list(tmp_path.iterdir()) == []


class TestJsonObjects:
    def test_read_missing_file_is_empty(self, tmp_path: pathlib.Path):
        assert util.read_json_object(tmp_path / "missing.json") == {}

    def test_write_replaces_declared_keys_and_keeps_others(self, tmp_path: pathlib.Path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"n": 1, "tags": ["a"], "foreign": {"x": [True]}}))
        util.write_json_object(path, {"n": 3}, _Doc)
        assert json.loads(path.read_text()) == {"foreign": {"x": [True]}, "n": 3}

    def test_write_over_unreadable_file(self, tmp_path: pathlib.Path):
        path = tmp_path / "settings.json"
        path.write_text("[not an object]")
        util.write_json_object(path, {"n": 3}, _Doc)
        assert json.loads(path.read_text()) == {"n": 3}


class _Item(TypedDict):
    id: str
    v: int


class _Doc(TypedDict, total=False):
    n: int
    tags: list[str]
    items: list[_Item]
    table: dict[str, _Item]


class TestMergeUnknown:
    def test_items_pair_by_id_across_reorder_and_deletion(self):
        stored: MutablePlainTree = {
            "items": [{"id": "a", "v": 1, "x": "a"}, {"id": "b", "v": 2, "x": "b"}],
        }
        value: MutablePlainTree = {"items": [{"id": "b", "v": 3}, {"id": "c", "v": 4}]}
        assert util.merge_unknown(stored, value, _Doc) == {
            "items": [{"id": "b", "v": 3, "x": "b"}, {"id": "c", "v": 4}],
        }

    def test_items_without_id_take_no_stored_keys(self):
        stored: MutablePlainTree = {"items": [{"v": 1, "x": "orphan"}]}
        value: MutablePlainTree = {"items": [{"v": 2}]}
        assert util.merge_unknown(stored, value, _Doc) == {"items": [{"v": 2}]}

    def test_dict_entries_pair_by_key_and_deleted_keys_go(self):
        stored: MutablePlainTree = {
            "table": {"a": {"id": "a", "v": 1, "x": 1}, "b": {"id": "b", "v": 2, "x": 2}},
        }
        value: MutablePlainTree = {"table": {"a": {"id": "a", "v": 5}}}
        assert util.merge_unknown(stored, value, _Doc) == {
            "table": {"a": {"id": "a", "v": 5, "x": 1}},
        }

    def test_plain_lists_and_scalars_come_from_value(self):
        stored: MutablePlainTree = {"tags": ["old", "older"], "n": 1, "extra": True}
        value: MutablePlainTree = {"tags": ["new"], "n": 2}
        assert util.merge_unknown(stored, value, _Doc) == {"extra": True, "tags": ["new"], "n": 2}

    def test_mismatched_stored_shapes_are_ignored(self):
        value: MutablePlainTree = {"items": [{"id": "a", "v": 1}], "table": {}}
        stored: MutablePlainTree = {"items": {"a": 1}, "table": []}
        assert util.merge_unknown(stored, value, _Doc) == value
        assert util.merge_unknown([1], value, _Doc) == value
        assert util.merge_unknown({"x": 1}, [1, 2], list[int]) == [1, 2]
        assert util.merge_unknown({"a": 1}, {"b": 2}, None) == {"b": 2}
        assert util.merge_unknown([0], [1], None) == [1]


def _redirect(handler: util._SafeRedirectHandler, newurl: str) -> request.Request | None:
    return handler.redirect_request(
        request.Request("http://original.com"),
        io.BytesIO(),
        302,
        "Found",
        HTTPMessage(),
        newurl,
    )


class TestSafeRedirectHandler:
    @pytest.mark.parametrize("newurl", ["http://redirect.com/path", "https://secure.com/path"])
    def test_handler_allows_http_and_https(self, newurl: str):
        result = _redirect(util._SafeRedirectHandler(), newurl)
        assert result is not None
        assert result.full_url == newurl

    @pytest.mark.parametrize(
        "newurl",
        [
            "file:///etc/passwd",
            "data:text/html,<script>alert(1)</script>",
            "javascript:alert(1)",
        ],
    )
    def test_handler_rejects_other_schemes(self, newurl: str):
        with pytest.raises(error.URLError, match="Unsafe redirect scheme"):
            _redirect(util._SafeRedirectHandler(), newurl)


def _answers(*addresses: str) -> Callable[..., list[tuple[object, ...]]]:
    """A resolver giving each address in turn, as DNS rebinding answers change."""
    remaining = list(addresses)

    def getaddrinfo(
        host: str,
        port: int,
        *args: object,
        **kwargs: object,
    ) -> list[tuple[object, ...]]:
        del host, args, kwargs
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (remaining.pop(0), port))]

    return getaddrinfo


class _Dialed(Exception):
    """Stops a fetch at its dial, recording where it went."""


def _record_dials(
    monkeypatch: pytest.MonkeyPatch,
    answers: Callable[..., object],
) -> list[tuple[str, int]]:
    dialed: list[tuple[str, int]] = []

    def create_connection(address: tuple[str, int], *args: object, **kwargs: object) -> object:
        del args, kwargs
        dialed.append(address)
        raise _Dialed

    monkeypatch.setattr(socket, "getaddrinfo", answers)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    return dialed


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_a_vetted_fetch_dials_the_address_it_vetted(monkeypatch: pytest.MonkeyPatch, scheme: str):
    # The first answer is public and vetted; a rebinding second answer, 127.0.0.1,
    # is never asked for: the dial goes to the address that was checked.
    dialed = _record_dials(monkeypatch, _answers("93.184.216.34", "127.0.0.1"))
    with pytest.raises(_Dialed):
        util.safe_urlopen(f"{scheme}://rebind.example/x", allow_address=lambda ip: ip.is_global)
    assert dialed == [("93.184.216.34", 80 if scheme == "http" else 443)]


def test_a_vetted_fetch_refuses_an_address_the_caller_does_not_allow(
    monkeypatch: pytest.MonkeyPatch,
):
    dialed = _record_dials(monkeypatch, _answers("127.0.0.1"))
    with pytest.raises(error.URLError) as refused:
        util.safe_urlopen("http://internal.example:8000/x", allow_address=lambda ip: ip.is_global)
    assert isinstance(refused.value.reason, util.RefusedAddress)
    assert dialed == []


def test_a_vetted_fetch_judges_an_ipv4_mapped_answer_as_ipv4(monkeypatch: pytest.MonkeyPatch):
    dialed = _record_dials(monkeypatch, _answers("::ffff:93.184.216.34"))
    with pytest.raises(_Dialed):
        util.safe_urlopen(
            "http://mapped.example/x",
            allow_address=lambda ip: isinstance(ip, ipaddress.IPv4Address),
        )
    assert dialed == [("::ffff:93.184.216.34", 80)]


def test_a_vetted_https_fetch_still_checks_the_name_s_certificate(monkeypatch: pytest.MonkeyPatch):
    wrapped: list[str | None] = []
    # A real TCP socket: the connection sets TCP options on it.
    listener = socket.create_server(("127.0.0.1", 0))
    sock = socket.create_connection(listener.getsockname())

    class Context:
        check_hostname = True

        def wrap_socket(self, raw: object, *, server_hostname: str | None) -> object:
            assert raw is sock
            wrapped.append(server_hostname)
            return raw

    def create_connection(address: tuple[str, int], *args: object, **kwargs: object) -> object:
        del address, args, kwargs
        return sock

    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))
    monkeypatch.setattr(socket, "create_connection", create_connection)
    connection = util._PinnedHTTPSConnection(
        "secure.example",
        allow_address=lambda ip: True,
        timeout=5,
        context=cast(ssl.SSLContext, Context()),
    )
    connection.connect()
    # SNI and certificate checks use the name, not the address dialed.
    assert wrapped == ["secure.example"]
    sock.close()
    listener.close()


def test_safe_urlopen_pins_connections_only_when_asked(monkeypatch: pytest.MonkeyPatch):
    web = testing.FakeWeb({"http://x/1": testing.FakeResponse(b"ok")})
    testing.install_web(monkeypatch, web, util)
    util.safe_urlopen("http://x/1").close()
    util.safe_urlopen("http://x/1", allow_address=lambda ip: True).close()
    plain, pinned = (tuple(type(h) for h in fetch.handlers) for fetch in web.fetches)
    assert plain == (util._SafeRedirectHandler,)
    # Redirects go back through the same handlers, so every hop is vetted too.
    assert pinned == (util._SafeRedirectHandler, util._PinnedHTTPHandler, util._PinnedHTTPSHandler)


def test_redact_url_credentials():
    assert (
        util.redact_url_credentials(
            "https://provider.example/live/remote-user/remote-pass/42.m3u8",
        )
        == "https://provider.example/live/***/***/42.m3u8"
    )
    assert (
        util.redact_url_credentials(
            "https://remote-user:remote-pass@provider.example/stream"
            "?username=remote-user&password=remote-pass&channel=42",
        )
        == "https://***:***@provider.example/stream"
        "?username=%2A%2A%2A&password=%2A%2A%2A&channel=42"
    )


def test_redact_url_credentials_masks_timeshift_path():
    assert (
        util.redact_url_credentials(
            "http://provider.example/timeshift/remote-user/remote-pass/120/2026-11-12:20-00/42.ts",
        )
        == "http://provider.example/timeshift/***/***/120/2026-11-12:20-00/42.ts"
    )


@pytest.mark.parametrize(
    "value",
    ["http://[::1", "rtmp://user:pass@host/live/a/b/c", "not a url"],
)
def test_redact_url_credentials_passes_through_invalid_and_non_http(value: str):
    assert util.redact_url_credentials(value) == value


def test_redact_url_credentials_keeps_ipv6_and_port():
    assert (
        util.redact_url_credentials("http://u:p@[::1]:8080/movie/u/p/1.mkv?token=x")
        == "http://***:***@[::1]:8080/movie/***/***/1.mkv?token=%2A%2A%2A"
    )


def test_redact_url_credentials_drops_zero_port():
    assert util.redact_url_credentials("http://host:0/x") == "http://host/x"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "http://provider.example/xc/live/remote-user/remote-pass/42.ts",
            "http://provider.example/xc/live/***/***/42.ts",
        ),
        (
            "http://provider.example/live/remote-user/remote-pa",
            "http://provider.example/live/***/***",
        ),
        ("http://provider.example/live/remote-us", "http://provider.example/live/***"),
    ],
)
def test_redact_url_credentials_masks_prefixed_and_truncated_paths(value: str, expected: str):
    assert util.redact_url_credentials(value) == expected


@pytest.fixture
def _restore_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("_restore_root_logger")
def test_configure_logging_redacts_messages_and_tracebacks(capsys: pytest.CaptureFixture[str]):
    util.configure_logging(logging.INFO)
    log = logging.getLogger("netv.test")
    log.info("ffmpeg:s http://provider.example/live/remote-user/remote-pass/1.ts: 403")
    log.debug("hidden http://provider.example/live/remote-user/remote-pass/2.ts")
    try:
        raise OSError(
            "Error opening input 'http://provider.example/movie/remote-user/remote-pass/3.mkv'",
        )
    except OSError:
        log.exception("Probe failed")

    err = capsys.readouterr().err
    assert " | INFO | ffmpeg:s http://provider.example/live/***/***/1.ts: 403\n" in err
    assert "hidden" not in err
    assert "'http://provider.example/movie/***/***/3.mkv'" in err
    assert "remote-user" not in err
    assert "remote-pass" not in err


def test_safe_urlopen_rejects_unsafe_scheme():
    with pytest.raises(error.URLError, match="Unsafe URL scheme: file"):
        util.safe_urlopen("file:///etc/passwd")


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        (None, "VLC/3.0.20 LibVLC/3.0.20"),
        ("", "VLC/3.0.20 LibVLC/3.0.20"),
        ("Custom/1.0", "Custom/1.0"),
    ],
)
def test_safe_urlopen_uses_safe_opener(
    monkeypatch: pytest.MonkeyPatch,
    user_agent: str | None,
    expected: str,
):
    response = testing.FakeResponse(b"ok")
    web = testing.FakeWeb({"https://example.com/x": response})
    testing.install_web(monkeypatch, web, util)
    assert util.safe_urlopen("https://example.com/x", 7, user_agent) is response
    [fetch] = web.fetches
    [handler] = fetch.handlers
    assert isinstance(handler, util._SafeRedirectHandler)
    assert (fetch.url, fetch.timeout, fetch.headers) == (
        "https://example.com/x",
        7,
        {"User-agent": expected},
    )


def test_safe_urlopen_sends_extra_headers(monkeypatch: pytest.MonkeyPatch):
    web = testing.FakeWeb({"https://example.com/x": testing.FakeResponse(b"ok")})
    testing.install_web(monkeypatch, web, util)
    util.safe_urlopen("https://example.com/x", headers={"Authorization": "Bearer t"})
    [fetch] = web.fetches
    assert fetch.headers == {"User-agent": "VLC/3.0.20 LibVLC/3.0.20", "Authorization": "Bearer t"}


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
