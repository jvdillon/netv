"""Suite-wide test isolation for neTV."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from netv import cache


@pytest.fixture(autouse=True)
def no_encoder_detection() -> Iterator[None]:
    """Supply a software-only encoder table so ``cache.initialize`` never probes hardware.

    Detection runs ffmpeg once per process (~3s) and makes results depend on the
    host's GPU. The table is filled in place because ``main`` imports it by name.
    """
    table = cache.AVAILABLE_ENCODERS
    saved = dict(table)
    table.clear()
    table.update(nvenc=False, amf=False, qsv=False, vaapi=False)
    yield
    table.clear()
    table.update(saved)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Roll ``cli_ffmpeg`` up to ``slow``, as the monorepo's resource plugin does for ``cli_*``."""
    for item in items:
        if item.get_closest_marker("cli_ffmpeg") is not None:
            item.add_marker(pytest.mark.slow)
