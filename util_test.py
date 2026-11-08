"""Tests for util.py."""

from __future__ import annotations

from http.client import HTTPMessage
from typing import TypedDict
from unittest.mock import MagicMock
from urllib import error, request

import io
import json
import pathlib

from treekle import MutablePlainTree

import pytest

import util


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

    def test_legacy_cache_wins_over_dot_cache(self, tmp_path: pathlib.Path):
        (tmp_path / "cache").mkdir()
        (tmp_path / ".cache").mkdir()
        assert util.data_dir(tmp_path) == tmp_path / "cache"

    def test_existing_dot_cache_is_kept(self, tmp_path: pathlib.Path):
        (tmp_path / ".cache").mkdir()
        assert util.data_dir(tmp_path) == tmp_path / ".cache"

    def test_fresh_source_checkout_uses_dot_cache(self, tmp_path: pathlib.Path):
        (tmp_path / "pyproject.toml").touch()
        assert util.data_dir(tmp_path) == tmp_path / ".cache"

    def test_installed_package_uses_user_data_dir(self, tmp_path: pathlib.Path):
        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()
        assert util.data_dir(site_packages) == tmp_path / "xdg" / "netv"

    def test_installed_package_without_xdg_uses_local_share(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ):
        monkeypatch.delenv("XDG_DATA_HOME")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert util.data_dir(tmp_path) == tmp_path / "home" / ".local" / "share" / "netv"


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
    opener = MagicMock()
    opener.open.return_value = "response"
    build_opener = MagicMock(return_value=opener)
    monkeypatch.setattr(util.request, "build_opener", build_opener)
    assert util.safe_urlopen("https://example.com/x", 7, user_agent) == "response"
    [handler] = build_opener.call_args.args
    assert isinstance(handler, util._SafeRedirectHandler)
    req = opener.open.call_args.args[0]
    assert isinstance(req, request.Request)
    assert opener.open.call_args.kwargs == {"timeout": 7}
    assert req.full_url == "https://example.com/x"
    assert req.get_header("User-agent") == expected


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
