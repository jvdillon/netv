"""Per-user filesystem locations following OS conventions.

Copied from ``loop/lib/userdirs.py``. These helpers satisfy the XDG Base
Directory Specification: https://specifications.freedesktop.org/basedir/latest/

Each function returns a BASE directory. The caller joins its own namespace::

    data_dir() / "netv"

Each reads one environment variable and falls back to a platform default
when it is unset *or empty*, per spec:

==========  =================  ==============  =============================
Function    Env var (POSIX)    Linux/BSD       macOS
==========  =================  ==============  =============================
data_dir    XDG_DATA_HOME      ~/.local/share  ~/Library/Application Support
config_dir  XDG_CONFIG_HOME    ~/.config       ~/Library/Application Support
cache_dir   XDG_CACHE_HOME     ~/.cache        ~/Library/Caches
state_dir   XDG_STATE_HOME     ~/.local/state  ~/Library/Application Support
==========  =================  ==============  =============================

On Windows all four read ``LOCALAPPDATA`` (default ``~/AppData/Local``) and
ignore the XDG variables.

Two deliberate deviations from the specification:

* A relative ``XDG_*_HOME`` is honored, where the spec says to ignore it, so
  tests and sandboxes that point these at relative scratch dirs stay inside
  them.
* The ``XDG_*`` variables are honored on macOS too, ahead of
  ``Library/Application Support``: an operator who exports one means it on
  every platform.
"""

from __future__ import annotations

from pathlib import Path

import os
import sys


__all__ = ["cache_dir", "config_dir", "data_dir", "state_dir"]


def data_dir(*, platform: str | None = None) -> Path:
    """Resolve the per-user data base directory.

    Holds durable user data: the content worth backing up and carrying to
    another machine.

    Args:
        platform: ``sys.platform`` string. Override for testing; ``None``
            reads ``sys.platform`` at call time.

    Returns:
        path: The data base directory. Join your own namespace segment;
            nothing is created.
    """
    platform = platform or sys.platform
    if platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    if xdg_data_home := _env("XDG_DATA_HOME"):
        return Path(xdg_data_home)
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".local" / "share"


def config_dir(*, platform: str | None = None) -> Path:
    """Resolve the per-user config base directory.

    Args:
        platform: ``sys.platform`` string. Override for testing; ``None``
            reads ``sys.platform`` at call time.

    Returns:
        path: The config base directory. Join your own namespace segment;
            nothing is created.
    """
    platform = platform or sys.platform
    if platform == "win32":
        return data_dir(platform=platform)
    if xdg_config_home := _env("XDG_CONFIG_HOME"):
        return Path(xdg_config_home)
    if platform == "darwin":
        return data_dir(platform=platform)
    return Path.home() / ".config"


def cache_dir(*, platform: str | None = None) -> Path:
    """Resolve the per-user cache base directory.

    Every file here must be reconstructible: deleting the tree may cost time
    but must not lose anything.

    Args:
        platform: ``sys.platform`` string. Override for testing; ``None``
            reads ``sys.platform`` at call time.

    Returns:
        path: The cache base directory. Join your own namespace segment;
            nothing is created.
    """
    platform = platform or sys.platform
    if platform == "win32":
        return data_dir(platform=platform)
    if xdg_cache_home := _env("XDG_CACHE_HOME"):
        return Path(xdg_cache_home)
    if platform == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path.home() / ".cache"


def state_dir(*, platform: str | None = None) -> Path:
    """Resolve the per-user state base directory.

    Holds state that should survive restarts but is not worth carrying to
    another machine: logs, history, where the application left off.

    Args:
        platform: ``sys.platform`` string. Override for testing; ``None``
            reads ``sys.platform`` at call time.

    Returns:
        path: The state base directory. Join your own namespace segment;
            nothing is created.
    """
    platform = platform or sys.platform
    if platform == "win32":
        return data_dir(platform=platform)
    if xdg_state_home := _env("XDG_STATE_HOME"):
        return Path(xdg_state_home)
    if platform == "darwin":
        return data_dir(platform=platform)
    return Path.home() / ".local" / "state"


def _env(name: str) -> str:
    """Return ``name``'s value, or empty when unset: both read as unset, per spec."""
    return os.environ.get(name, "")
