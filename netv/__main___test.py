"""Tests for the package entry point."""

from unittest.mock import patch

import runpy

import pytest


@pytest.mark.parametrize("exit_code", [0, 3])
def test_package_entry_point_returns_server_status(exit_code: int):
    with (
        patch("netv.main.main", return_value=exit_code) as main,
        pytest.raises(SystemExit) as caught,
    ):
        runpy.run_module("netv", run_name="__main__")
    main.assert_called_once_with()
    assert caught.value.code == exit_code


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
