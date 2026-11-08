"""Tests for util.py."""

from __future__ import annotations

from typing import Any
from urllib import error

import json
import pathlib

import pytest

import util


def test_atomic_write_json_replaces_file_and_preserves_permissions(tmp_path: pathlib.Path):
    path = tmp_path / "settings.json"
    path.write_text('{"old": true}')
    path.chmod(0o640)

    util.atomic_write_json(path, {"new": True})

    assert json.loads(path.read_text()) == {"new": True}
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [path]


class TestSafeRedirectHandler:
    def test_handler_allows_http(self):
        handler = util._SafeRedirectHandler()
        req = _fake_request("http://original.com")
        result = handler.redirect_request(
            req,
            fp=None,
            code=302,
            msg="Found",
            headers={},
            newurl="http://redirect.com/path",
        )
        assert result is not None

    def test_handler_allows_https(self):
        handler = util._SafeRedirectHandler()
        req = _fake_request("https://original.com")
        result = handler.redirect_request(
            req,
            fp=None,
            code=302,
            msg="Found",
            headers={},
            newurl="https://secure.com/path",
        )
        assert result is not None

    def test_handler_rejects_file_scheme(self):
        handler = util._SafeRedirectHandler()
        req = _fake_request("http://original.com")
        with pytest.raises(error.URLError, match="Unsafe redirect scheme"):
            handler.redirect_request(
                req,
                fp=None,
                code=302,
                msg="Found",
                headers={},
                newurl="file:///etc/passwd",
            )

    def test_handler_rejects_data_scheme(self):
        handler = util._SafeRedirectHandler()
        req = _fake_request("http://original.com")
        with pytest.raises(error.URLError, match="Unsafe redirect scheme"):
            handler.redirect_request(
                req,
                fp=None,
                code=302,
                msg="Found",
                headers={},
                newurl="data:text/html,<script>alert(1)</script>",
            )

    def test_handler_rejects_javascript_scheme(self):
        handler = util._SafeRedirectHandler()
        req = _fake_request("http://original.com")
        with pytest.raises(error.URLError, match="Unsafe redirect scheme"):
            handler.redirect_request(
                req,
                fp=None,
                code=302,
                msg="Found",
                headers={},
                newurl="javascript:alert(1)",
            )


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


def test_safe_urlopen_rejects_unsafe_scheme():
    with pytest.raises(error.URLError, match="Unsafe URL scheme: file"):
        util.safe_urlopen("file:///etc/passwd")


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [(None, "VLC/3.0.20 LibVLC/3.0.20"), ("Custom/1.0", "Custom/1.0")],
)
def test_safe_urlopen_uses_safe_opener(
    monkeypatch: pytest.MonkeyPatch,
    user_agent: str | None,
    expected: str,
):
    calls: list[tuple[Any, ...]] = []

    class _Opener:
        def open(self, req: Any, timeout: int) -> str:
            calls.append((req, timeout))
            return "response"

    def _build_opener(*handlers: Any) -> _Opener:
        assert [type(h) for h in handlers] == [util._SafeRedirectHandler]
        return _Opener()

    monkeypatch.setattr(util.request, "build_opener", _build_opener)
    assert util.safe_urlopen("https://example.com/x", 7, user_agent) == "response"
    [(req, timeout)] = calls
    assert timeout == 7
    assert req.full_url == "https://example.com/x"
    assert req.get_header("User-agent") == expected


def _fake_request(url: str) -> Any:
    """Create a minimal request object for testing."""

    class _Req:
        full_url = url
        headers: dict[str, str] = {}
        data = None
        origin_req_host = "original.com"

        def get_method(self) -> str:
            return "GET"

    return _Req()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
