"""Tests for testing.py."""

from __future__ import annotations

import sys

import pytest

from netv import testing


def test_run_tests_invokes_pytest_and_exits_with_its_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def _main(args: list[str]) -> int:
        calls.append(args)
        return 3

    monkeypatch.setattr(testing.pytest, "main", _main)
    monkeypatch.setattr(sys, "argv", ["prog", "-k", "foo"])
    with pytest.raises(SystemExit) as exc:
        testing.run_tests("some_test.py")
    assert exc.value.code == 3
    assert calls == [
        [
            "some_test.py",
            "-v",
            "-s",
            "-W",
            "ignore::pytest.PytestAssertRewriteWarning",
            "-k",
            "foo",
        ],
    ]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
