"""Shared utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from http import client
from http.client import HTTPMessage, HTTPResponse
from typing import IO, cast, get_args, get_origin, get_type_hints, is_typeddict, override
from urllib import error, parse, request

import functools
import ipaddress
import json
import logging
import os
import pathlib
import re
import shutil
import socket
import ssl
import tempfile

from treekle import MutablePlainTree, json_loads, mutable, to_plain

from netv import userdirs


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
    if parsed.port is not None and parsed.port != 0:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username is not None:
        netloc = f"***:***@{netloc}"

    path_parts = parsed.path.split("/")
    kind = next(
        (
            i
            for i, part in enumerate(path_parts)
            if part in ("live", "movie", "series", "timeshift")
        ),
        len(path_parts),
    )
    # Callers truncate URLs before logging, so a lone username must still be masked.
    path_parts[kind + 1 : kind + 3] = ["***"] * len(path_parts[kind + 1 : kind + 3])

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


def configure_logging(level: int) -> None:
    """Send every log record to stderr with provider credentials in URLs masked.

    Redacting the formatted line, not each call's arguments, also covers ffmpeg
    stderr, exception text, and third-party loggers.

    Args:
        level: Root logger level.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        _RedactingFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"),
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)


def data_dir(app_dir: pathlib.Path) -> pathlib.Path:
    """Return where neTV keeps settings, users, and caches.

    State lives outside the source tree, so ``git clean -X`` or a reinstall
    cannot delete it. A ``cache``/``.cache`` left in a checkout by an older
    neTV is moved into the per-user directory the first time it is resolved.

    Args:
        app_dir: Directory holding the neTV modules.

    Returns:
        directory: ``$NETV_DATA_DIR`` when set; else the per-user data
            directory (``~/.local/share/netv`` on Linux,
            ``~/Library/Application Support/netv`` on macOS).
    """
    configured = os.environ.get("NETV_DATA_DIR") or ""
    if configured:
        return pathlib.Path(configured).expanduser()
    target = userdirs.data_dir() / "netv"
    # The tools keep their own subdirectories here, so a directory without
    # server settings still lacks the state a checkout may hold.
    if not (target / "server_settings.json").exists():
        root = app_dir.parent if (app_dir.parent / "pyproject.toml").is_file() else app_dir
        for legacy in (root / "cache", root / ".cache"):
            if (legacy / "server_settings.json").is_file():
                target.mkdir(parents=True, exist_ok=True)
                for child in legacy.iterdir():
                    shutil.move(child, target / child.name)
                legacy.rmdir()
                break
    return target


def atomic_write_json(path: pathlib.Path, value: object) -> None:
    """Atomically replace a JSON file while preserving existing permissions.

    Args:
        path: Destination file.
        value: JSON-serializable content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp_path = pathlib.Path(name)
    try:
        with os.fdopen(fd, "w") as temp:
            json.dump(value, temp, indent=2)
            temp.flush()
            if path.exists():
                os.fchmod(temp.fileno(), path.stat().st_mode & 0o777)
            os.fsync(temp.fileno())
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def read_json_object(path: pathlib.Path) -> dict[str, MutablePlainTree]:
    """Return the JSON object stored at ``path``, or an empty one if it is absent.

    Args:
        path: JSON file holding one object.

    Returns:
        tree: Every key in the file, values as plain JSON.
    """
    if not path.exists():
        return {}
    return json_loads(path.read_text(), dict[str, MutablePlainTree])


def write_json_object(path: pathlib.Path, value: Mapping[str, object], schema: object) -> None:
    """Replace the JSON object at ``path`` with ``value``, keeping keys ``schema`` lacks.

    Args:
        path: JSON file holding one object.
        value: New content, shaped by ``schema``.
        schema: The ``TypedDict`` describing ``value``; see ``merge_unknown``.
    """
    try:
        stored: MutablePlainTree = read_json_object(path)
    except ValueError:
        # An unreadable file was always overwritten whole; nothing in it survives.
        stored = {}
    atomic_write_json(path, merge_unknown(stored, mutable(to_plain(value)), schema))


def merge_unknown(
    stored: MutablePlainTree,
    value: MutablePlainTree,
    schema: object,
) -> MutablePlainTree:
    """Return ``value`` plus the parts of ``stored`` that ``schema`` does not declare.

    A typed view drops keys it does not declare -- written by another module or a
    newer release. Writing that view back would delete them; this puts them back.

    - In a ``TypedDict``, undeclared stored keys are kept; declared keys come from
      ``value``, so a declared key ``value`` lacks is removed.
    - In ``dict[str, X]``, entries pair up by key; keys ``value`` lacks are removed.
    - In ``list[X]`` where ``X`` declares an ``id``, items pair up by it; other
      lists are taken from ``value`` whole.

    Args:
        stored: What the file holds now.
        value: The new content.
        schema: The type annotation describing ``value``.

    Returns:
        merged: ``value``, with undeclared stored keys restored at every level.
    """
    if isinstance(value, dict):
        stored_dict = stored if isinstance(stored, dict) else {}
        if is_typeddict(schema):
            fields = cast(dict[str, object], get_type_hints(schema))
            merged = {k: v for k, v in stored_dict.items() if k not in fields}
            for key, item in value.items():
                merged[key] = merge_unknown(stored_dict.get(key), item, fields.get(key))
            return merged
        if get_origin(schema) is dict:
            item_schema = cast(object, get_args(schema)[1])
            return {k: merge_unknown(stored_dict.get(k), v, item_schema) for k, v in value.items()}
        return value
    if isinstance(value, list) and get_origin(schema) is list:
        item_schema = cast(object, get_args(schema)[0])
        # Only a stable id pairs list items; pairing by position would hand a deleted
        # item's unknown keys to whatever took its place.
        if is_typeddict(item_schema) and "id" in get_type_hints(item_schema):
            stored_list = stored if isinstance(stored, list) else []
            by_id = {key: item for item in stored_list if (key := _item_id(item))}
            return [merge_unknown(by_id.get(_item_id(item)), item, item_schema) for item in value]
    return value


def safe_urlopen(
    url: str,
    timeout: int = 30,
    user_agent: str | None = None,
    headers: Mapping[str, str] | None = None,
    *,
    allow_address: Callable[[IPAddress], bool] | None = None,
) -> HTTPResponse:
    """Open URL with safe redirect handling.

    Args:
        url: URL to open
        timeout: Request timeout in seconds
        user_agent: User-Agent header to send. If None, uses a default VLC User-Agent
            to avoid being blocked by providers that reject Python's default.
        headers: Other headers to send, such as an API's Authorization.
        allow_address: When given, each connection, redirects' included, resolves
            its host once, refuses it unless every address passes, and dials the
            address it checked, so a second DNS answer cannot redirect it.

    Returns:
        response: Open HTTP response; the caller must close it.
    """
    parsed = parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise error.URLError(f"Unsafe URL scheme: {parsed.scheme}")
    ua = "VLC/3.0.20 LibVLC/3.0.20" if user_agent is None or user_agent == "" else user_agent
    req = request.Request(url, headers={"User-Agent": ua, **(headers or {})})
    handlers: list[request.BaseHandler] = [_SafeRedirectHandler()]
    if allow_address is not None:
        handlers += [_PinnedHTTPHandler(allow_address), _PinnedHTTPSHandler(allow_address)]
    opener = request.build_opener(*handlers)
    # Typeshed types ``open`` as Any; the scheme check above leaves only HTTP(S).
    return cast(HTTPResponse, opener.open(req, timeout=timeout))


IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class RefusedAddress(OSError):
    """A host resolved to an address the caller does not allow."""


def pinned_connection(
    address: tuple[str, int],
    timeout: float | None = None,
    source_address: tuple[str, int] | None = None,
    *,
    allow_address: Callable[[IPAddress], bool],
) -> socket.socket:
    """``socket.create_connection`` to the one address of ``address``'s host it vetted.

    Args:
        address: Host name or address, and port.
        timeout: Socket timeout; None blocks.
        source_address: Local address to bind, or None for any.
        allow_address: Whether a resolved address may be dialed.

    Returns:
        connection: Connected to the first address, once every address passed.

    Raises:
        RefusedAddress: The host resolves to an address ``allow_address`` refuses.
    """
    host, port = address
    found = [
        str(sockaddr[0]).split("%")[0]
        for *_, sockaddr in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ]
    for answer in found:
        ip = ipaddress.ip_address(answer)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if not allow_address(ip):
            raise RefusedAddress(f"Refused address for {host}")
    return socket.create_connection((found[0], port), timeout, source_address)


class _PinnedHTTPConnection(client.HTTPConnection):
    """An HTTP connection that dials only the address it vetted."""

    def __init__(
        self,
        host: str,
        /,
        *,
        allow_address: Callable[[IPAddress], bool],
        port: int | None = None,
        timeout: float = 30.0,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 8192,
    ) -> None:
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            source_address=source_address,
            blocksize=blocksize,
        )
        self._create_connection = functools.partial(pinned_connection, allow_address=allow_address)


class _PinnedHTTPSConnection(client.HTTPSConnection):
    """An HTTPS connection that dials only the address it vetted.

    It wraps the socket for the host's name, so SNI and certificate checks use the
    name, not the address.
    """

    def __init__(
        self,
        host: str,
        /,
        *,
        allow_address: Callable[[IPAddress], bool],
        port: int | None = None,
        timeout: float = 30.0,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 8192,
        context: ssl.SSLContext | None = None,
    ) -> None:
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            source_address=source_address,
            blocksize=blocksize,
            context=context,
        )
        self._create_connection = functools.partial(pinned_connection, allow_address=allow_address)


class _PinnedHTTPHandler(request.HTTPHandler):
    def __init__(self, allow_address: Callable[[IPAddress], bool]) -> None:
        super().__init__()
        self._connection = functools.partial(_PinnedHTTPConnection, allow_address=allow_address)

    @override
    def http_open(self, req: request.Request) -> HTTPResponse:
        return self.do_open(self._connection, req)


class _PinnedHTTPSHandler(request.HTTPSHandler):
    def __init__(self, allow_address: Callable[[IPAddress], bool]) -> None:
        super().__init__()
        self._connection = functools.partial(_PinnedHTTPSConnection, allow_address=allow_address)

    @override
    def https_open(self, req: request.Request) -> HTTPResponse:
        return self.do_open(self._connection, req)


class _SafeRedirectHandler(request.HTTPRedirectHandler):
    """Redirect handler that only allows http/https schemes."""

    @override
    def redirect_request(
        self,
        req: request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> request.Request | None:
        parsed = parse.urlparse(newurl)
        if parsed.scheme not in ("http", "https"):
            raise error.URLError(f"Unsafe redirect scheme: {parsed.scheme}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _RedactingFormatter(logging.Formatter):
    """Formatter that masks credentials in every http(s) URL of the output."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        return re.sub(
            r"https?://[^\s'\"<>\]]+",
            lambda match: redact_url_credentials(match.group()),
            super().format(record),
        )


def _item_id(item: MutablePlainTree) -> str:
    """Return a list item's string ``id``, or "" when it has none."""
    ident = item.get("id") if isinstance(item, dict) else None
    return ident if isinstance(ident, str) else ""
