"""Tests for userdirs.py."""

from __future__ import annotations

from collections.abc import Callable

import pathlib

import pytest

from netv import userdirs


@pytest.fixture(autouse=True)
def _home(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
        "LOCALAPPDATA",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("func", "linux", "darwin"),
    [
        (userdirs.data_dir, ".local/share", "Library/Application Support"),
        (userdirs.config_dir, ".config", "Library/Application Support"),
        (userdirs.cache_dir, ".cache", "Library/Caches"),
        (userdirs.state_dir, ".local/state", "Library/Application Support"),
    ],
)
def test_platform_defaults(
    tmp_path: pathlib.Path,
    func: Callable[..., pathlib.Path],
    linux: str,
    darwin: str,
):
    assert func(platform="linux") == tmp_path / linux
    assert func(platform="darwin") == tmp_path / darwin
    assert func(platform="win32") == tmp_path / "AppData" / "Local"


@pytest.mark.parametrize(
    ("func", "var"),
    [
        (userdirs.data_dir, "XDG_DATA_HOME"),
        (userdirs.config_dir, "XDG_CONFIG_HOME"),
        (userdirs.cache_dir, "XDG_CACHE_HOME"),
        (userdirs.state_dir, "XDG_STATE_HOME"),
    ],
)
def test_xdg_variable_wins_on_posix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    func: Callable[..., pathlib.Path],
    var: str,
):
    monkeypatch.setenv(var, str(tmp_path / "x"))
    assert func(platform="linux") == tmp_path / "x"
    assert func(platform="darwin") == tmp_path / "x"
    assert func(platform="win32") == tmp_path / "AppData" / "Local"


def test_empty_xdg_variable_reads_as_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setenv("XDG_DATA_HOME", "")
    assert userdirs.data_dir(platform="linux") == tmp_path / ".local" / "share"


def test_windows_reads_localappdata(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    for func in (userdirs.data_dir, userdirs.config_dir, userdirs.cache_dir, userdirs.state_dir):
        assert func(platform="win32") == tmp_path / "lad"


def test_platform_defaults_to_sys_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(userdirs.sys, "platform", "darwin")
    assert userdirs.data_dir() == tmp_path / "Library" / "Application Support"


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
