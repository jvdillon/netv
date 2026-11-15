"""Tests for the package entry point."""

import logging
import runpy
import sys

import pytest

from netv import main, testing


@pytest.fixture(autouse=True)
def root_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the root-logger setup ``main`` performs, so later tests log as before."""
    monkeypatch.setattr(logging.root, "handlers", list(logging.root.handlers))
    monkeypatch.setattr(logging.root, "level", logging.root.level)


def test_package_entry_point_returns_server_status(monkeypatch: pytest.MonkeyPatch):
    server = testing.ServerRecorder()
    testing.install_server(monkeypatch, server, main)
    monkeypatch.setattr(sys, "argv", ["netv", "--port", "8123"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("netv", run_name="__main__")
    assert caught.value.code == 0
    [run] = server.runs
    assert run.app is main.app
    assert run.options["port"] == 8123


def test_package_entry_point_exits_with_startup_error(monkeypatch: pytest.MonkeyPatch):
    server = testing.ServerRecorder()
    testing.install_server(monkeypatch, server, main)
    monkeypatch.setattr(sys, "argv", ["netv", "--https", "no-such-domain.invalid"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("netv", run_name="__main__")
    assert caught.value.code == (
        "Cert not found: /etc/letsencrypt/live/no-such-domain.invalid/fullchain.pem"
    )
    assert server.runs == []


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
