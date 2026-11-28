"""Tests for the prebuilt Tailwind stylesheet in static/css."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import re

from fastapi.testclient import TestClient

from netv import main


_CWD: Final[Path] = Path(__file__).resolve().parent

NETV_DIR = _CWD
TEMPLATES_DIR = NETV_DIR / "templates"
STYLESHEET = NETV_DIR / "static" / "css" / "tailwind.css"
LINK = '<link rel="stylesheet" href="/static/css/tailwind.css">'
# Templates that are full pages. The rest are partials rendered inside them.
PAGES = ("base.html", "login.html", "setup.html")


def test_no_template_uses_the_play_cdn() -> None:
    for path in TEMPLATES_DIR.glob("*.html"):
        assert "cdn.tailwindcss.com" not in path.read_text(), path.name


def test_pages_link_the_stylesheet_once() -> None:
    for name in PAGES:
        assert (TEMPLATES_DIR / name).read_text().count(LINK) == 1, name


def test_stylesheet_comes_after_inline_styles() -> None:
    # The Play CDN appended its rules at the end of <head>. Keep that order so a
    # utility like .hidden still beats a same-specificity rule in an inline <style>.
    html = (TEMPLATES_DIR / "base.html").read_text()
    head = html[: html.index("</head>")]
    assert head.rindex("</style>") < head.index(LINK)
    assert head.index("{% block head_extra %}") < head.index(LINK)


def test_stylesheet_is_a_minified_tailwind_build() -> None:
    css = STYLESHEET.read_text()
    assert len(css) > 10_000
    # Preflight (the base reset) and a utility from the default theme.
    assert "box-sizing:border-box" in css
    assert ".hidden{display:none}" in css
    # Minified: no banner and no indentation.
    assert "\n  " not in css


def test_stylesheet_has_rules_for_template_utilities() -> None:
    css = STYLESHEET.read_text()
    html = (TEMPLATES_DIR / "base.html").read_text()
    for token in re.findall(r'<body class="([^"]+)"', html)[0].split():
        assert f".{token}" in css, token


def test_stylesheet_is_served() -> None:
    response = TestClient(main.app).get("/static/css/tailwind.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert response.content == STYLESHEET.read_bytes()


def test_standalone_pages_render_the_link() -> None:
    for name in ("login.html", "setup.html"):
        html = main.TEMPLATES.get_template(name).render()
        assert LINK in html, name
        assert "cdn.tailwindcss.com" not in html, name


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
