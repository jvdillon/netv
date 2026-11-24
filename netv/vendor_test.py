"""Tests for the vendored JavaScript in static/vendor."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import base64
import hashlib
import re

from fastapi.testclient import TestClient

from netv import main


_CWD: Final[Path] = Path(__file__).resolve().parent

VENDOR_DIR = _CWD / "static" / "vendor"
BASE_TEMPLATE = _CWD / "templates" / "base.html"


def _recorded_hashes() -> dict[str, str]:
    """Maps file name to the SHA-384 recorded in VENDOR.md."""
    rows = re.findall(
        r"^\| `([^`]+\.js)` \|.*?`(sha384-[^`]+)`",
        (VENDOR_DIR / "VENDOR.md").read_text(),
        re.MULTILINE,
    )
    return dict(rows)


def test_vendor_md_lists_htmx_and_hls() -> None:
    names = _recorded_hashes()
    assert any(n.startswith("htmx-") for n in names)
    assert any(n.startswith("hls-") for n in names)


def test_vendored_files_match_recorded_hashes() -> None:
    recorded = _recorded_hashes()
    on_disk = {p.name for p in VENDOR_DIR.glob("*.js")}
    assert on_disk == set(recorded)
    for name, expected in recorded.items():
        digest = hashlib.sha384((VENDOR_DIR / name).read_bytes()).digest()
        assert "sha384-" + base64.b64encode(digest).decode() == expected, name


def test_base_template_loads_vendored_scripts_only() -> None:
    html = BASE_TEMPLATE.read_text()
    assert "unpkg.com" not in html
    assert "jsdelivr" not in html
    srcs = re.findall(r'<script src="([^"]+)"', html)
    for name in _recorded_hashes():
        assert f"/static/vendor/{name}" in srcs


def test_vendored_files_are_served() -> None:
    client = TestClient(main.app)
    for name in _recorded_hashes():
        response = client.get(f"/static/vendor/{name}")
        assert response.status_code == 200, name
        assert response.content == (VENDOR_DIR / name).read_bytes()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
