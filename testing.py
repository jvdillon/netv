"""Test utilities."""

import sys

import pytest


def run_tests(test_file: str) -> None:
    """Run pytest on a test file with standard flags.

    Args:
        test_file: Test module to pass to pytest.

    Usage:
        if __name__ == "__main__":
            from testing import run_tests
            run_tests(__file__)
    """

    sys.exit(
        pytest.main(
            [
                test_file,
                "-v",
                "-s",
                "-W",
                "ignore::pytest.PytestAssertRewriteWarning",
                *sys.argv[1:],
            ],
        ),
    )
