"""Suite-wide test isolation for neTV."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import shutil

import pytest

from netv import auth, cache, epg, testing, util


# Each invocation claims its own basetemp, so concurrent runs never delete each other's.
pytest_plugins: Final = ("codedoctor.lib.testing.basetemp",)


@pytest.fixture(scope="session")
def empty_guide(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one empty guide database per worker: building one costs ~70 ms of fsyncs."""
    path = tmp_path_factory.mktemp("guide")
    epg.init(path)
    epg.close()
    return path / "epg.db"


@pytest.fixture
def guide_db(empty_guide: Path, tmp_path: Path) -> Iterator[None]:
    """Open a copy of the empty guide database; copying skips the build's fsyncs."""
    shutil.copyfile(empty_guide, tmp_path / "epg.db")
    epg.init(tmp_path)
    yield
    epg.close()


@pytest.fixture(autouse=True)
def isolated_state(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep settings, users, and file caches in a per-test directory, memory caches empty.

    Password hashing also runs at minimum cost, and settings writes skip fsync.

    Without it a test that forgets to redirect storage reads and writes the
    developer's real neTV data directory.
    """
    state = tmp_path_factory.mktemp("state")
    (state / "users").mkdir()
    for module in (cache, auth):
        monkeypatch.setattr(module, "CACHE_DIR", state)
        monkeypatch.setattr(module, "SERVER_SETTINGS_FILE", state / "server_settings.json")
        monkeypatch.setattr(module, "USERS_DIR", state / "users")
    monkeypatch.setattr(cache, "LOGOS_DIR", state / "logos")
    testing.install_cheap_kdf(monkeypatch, auth)
    testing.install_no_fsync(monkeypatch, util)
    cache.clear_all_caches()
    cache.get_cache().pop("epg", None)
    yield
    cache.clear_all_caches()
    cache.get_cache().pop("epg", None)


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
