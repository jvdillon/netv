"""Shared utilities."""

from __future__ import annotations

from typing import Any
from urllib import error, parse, request

import json
import os
import pathlib
import tempfile

from codedoctor.lib import userdirs


def redact_url_credentials(value: str) -> str:
    """Redact credentials from a URL before logging it.

    Args:
        value: URL that may contain provider credentials.

    Returns:
        redacted_url: URL with user information and credential fields masked.
    """
    try:
        parsed = parse.urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme not in ("http", "https"):
        return value

    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username is not None:
        netloc = f"***:***@{netloc}"

    path_parts = parsed.path.split("/")
    if len(path_parts) >= 5 and path_parts[1] in ("live", "movie", "series"):
        path_parts[2:4] = ["***", "***"]

    query = parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_query = parse.urlencode(
        [
            (key, "***" if key.lower() in ("username", "password", "token") else value)
            for key, value in query
        ],
    )
    return parse.urlunsplit(
        (
            parsed.scheme,
            netloc,
            "/".join(path_parts),
            redacted_query,
            parsed.fragment,
        ),
    )


def data_dir(app_dir: pathlib.Path) -> pathlib.Path:
    """Return where neTV keeps settings, users, and caches.

    Args:
        app_dir: Directory holding the neTV modules.

    Returns:
        directory: ``$NETV_DATA_DIR``; else the legacy ``cache`` (Docker
            volume) or ``.cache`` of a source checkout; else, for an installed
            package (pip, uv tool, uvx), the per-user data directory (e.g.
            ``~/.local/share/netv``), so state survives reinstalls.
    """
    configured = os.environ.get("NETV_DATA_DIR")
    if configured:
        return pathlib.Path(configured).expanduser()
    if (app_dir / "cache").exists():
        return app_dir / "cache"
    if (app_dir / ".cache").exists() or (app_dir / "pyproject.toml").exists():
        return app_dir / ".cache"
    return userdirs.data_dir() / "netv"


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    """Atomically replace a JSON file while preserving existing permissions.

    Args:
        path: Destination file.
        value: JSON-serializable content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temp:
            json.dump(value, temp, indent=2)
            temp.flush()
            if path.exists():
                os.fchmod(temp.fileno(), path.stat().st_mode & 0o777)
            os.fsync(temp.fileno())
            temp_path = pathlib.Path(temp.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def safe_urlopen(url: str, timeout: int = 30, user_agent: str | None = None) -> Any:
    """Open URL with safe redirect handling.

    Args:
        url: URL to open
        timeout: Request timeout in seconds
        user_agent: User-Agent header to send. If None, uses a default VLC User-Agent
            to avoid being blocked by providers that reject Python's default.

    Returns:
        response: Open HTTP response; the caller must close it.
    """
    parsed = parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise error.URLError(f"Unsafe URL scheme: {parsed.scheme}")
    ua = user_agent if user_agent else "VLC/3.0.20 LibVLC/3.0.20"
    req = request.Request(url, headers={"User-Agent": ua})
    opener = request.build_opener(_SafeRedirectHandler())
    return opener.open(req, timeout=timeout)


class _SafeRedirectHandler(request.HTTPRedirectHandler):
    """Redirect handler that only allows http/https schemes."""

    def redirect_request(
        self,
        req: request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> request.Request | None:
        parsed = parse.urlparse(newurl)
        if parsed.scheme not in ("http", "https"):
            raise error.URLError(f"Unsafe redirect scheme: {parsed.scheme}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)
