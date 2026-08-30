"""Tests for util.py."""

from __future__ import annotations

from typing import Any

import json
import pathlib
import urllib.error

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
        with pytest.raises(urllib.error.URLError, match="Unsafe redirect scheme"):
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
        with pytest.raises(urllib.error.URLError, match="Unsafe redirect scheme"):
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
        with pytest.raises(urllib.error.URLError, match="Unsafe redirect scheme"):
            handler.redirect_request(
                req,
                fp=None,
                code=302,
                msg="Found",
                headers={},
                newurl="javascript:alert(1)",
            )


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)
