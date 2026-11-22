"""Pass an upstream HTTP body to a client, so the client never learns its URL."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from http.client import HTTPResponse
from typing import Final, override
from urllib import parse

import functools
import ipaddress
import os
import re
import select
import socket
import socketserver
import threading

from fastapi.responses import StreamingResponse

from netv import cache, util
from netv.relay import Feed, Relay


async def relay_response(
    upstream: HTTPResponse,
    *,
    on_close: Callable[[], object] = lambda: None,
) -> StreamingResponse:
    """A response streaming ``upstream``'s body with its status and media headers.

    Args:
        upstream: An open upstream response; the returned response closes it.
        on_close: Called once the body ends or the client leaves.

    Returns:
        response: Its status (206 for a range), a media type, and the headers a
            client needs to play, seek, and size the body; nothing else upstream's.
    """
    content_type = upstream.headers.get("Content-Type") or "video/mp2t"
    # Only media: a provider's HTML or SVG served from this origin would run as it.
    if not content_type.lower().startswith(
        ("video/", "audio/", "application/vnd.apple.mpegurl", "application/x-mpegurl"),
    ):
        content_type = "application/octet-stream"
    # Response header names are case-insensitive; Starlette sends them lowercase.
    headers = {
        "cache-control": "no-store",  # pragma: no mutate
        "x-content-type-options": "nosniff",
        "content-security-policy": "sandbox",
    }
    for name in ("Content-Length", "Content-Encoding", "Content-Range", "Accept-Ranges"):
        value = upstream.headers.get(name)
        if value is not None and value:
            headers[name.lower()] = value
    # The client pulls at its own pace; one that falls far behind is dropped
    # rather than stalling the upstream read. Its only subscriber joins before
    # the first byte, so the relay keeps no history (memory, not behavior).
    stream, feed = Relay(lambda: upstream, history_sec=0), Feed()  # pragma: no mutate
    stream.subscribe(feed)
    await stream.start()
    return StreamingResponse(
        _relay_body(stream, feed, on_close),
        status_code=upstream.status,
        media_type=content_type,
        headers=headers,
    )


def rewrite_playlist(
    text: str,
    base: str,
    link: Callable[[str], str],
    *,
    max_lines: int = 20_000,
) -> str:
    """An HLS playlist with each URI, resolved against ``base``, replaced by ``link``'s.

    Args:
        text: The playlist as the upstream served it.
        base: The URL it was served from, after redirects.
        link: Maps an absolute upstream URL to the one the client should fetch.
        max_lines: Most lines a real playlist has; each URI grows several-fold.

    Returns:
        playlist: The same tags, with every segment, variant, and key URI replaced.

    Raises:
        ValueError: The playlist is too long to be real.
    """
    given = text.splitlines()
    if len(given) > max_lines:
        raise ValueError(f"Playlist too long: {len(given)} lines")
    lines = []
    for line in given:
        entry = line.strip()
        if entry and not entry.startswith("#"):
            line = link(parse.urljoin(base, entry))
        elif entry.startswith("#"):
            line = re.sub(
                r'URI="([^"]*)"',
                lambda m: f'URI="{link(parse.urljoin(base, m.group(1)))}"',
                line,
            )
        lines.append(line)
    return "\n".join(lines) + "\n"


def home_stream(url: str, source_id: str) -> bool:
    """Whether ``url`` is a home-network stream listed by a home-network source.

    An admin put a source on the home network, so what it lists there (a tuner
    on another host than its lineup) is the household's own. A provider's list
    naming an internal address must not make that address one the server fetches.

    Args:
        url: An upstream URL.
        source_id: The configured source that listed it; "" for none.

    Returns:
        home: True when that source and ``url`` are both off the internet.
    """
    source = next((s for s in cache.get_sources() if s.id == source_id), None)
    return source is not None and is_lan_url(source.url) and is_lan_url(url)


def address_rule(
    url: str,
    source_id: str,
) -> Callable[[ipaddress.IPv4Address | ipaddress.IPv6Address], bool]:
    """The addresses a fetch starting at ``url``, and its redirects, may dial.

    Args:
        url: A provider or source URL.
        source_id: The configured source that listed it; "" for none.

    Returns:
        allowed: For ``safe_urlopen``'s ``allow_address``.
    """
    return functools.partial(allowed_address, lan=home_stream(url, source_id))


def is_lan_url(url: str) -> bool:
    """Whether ``url``'s host is on this network (private, loopback, link-local).

    Args:
        url: An upstream URL the server itself was configured with.

    Returns:
        lan: True when any address it resolves to is not on the internet.
    """
    return any(not ip.is_global for ip in _addresses(url))


def allowed_upstream(url: str, *, lan: bool) -> bool:
    """Whether a relay may fetch ``url``, given where its stream started.

    Playlists and redirects come from the provider, so they could point the
    server at itself or the home network and have the answer relayed. A stream
    that started on the internet may only reach the internet; one that started
    on the home network (a tuner) is the household's own.

    Args:
        url: A URL to fetch: the stream's, a playlist entry, or a redirect.
        lan: Whether the stream started on the home network.

    Returns:
        allowed: False for other schemes, and for internal addresses unless ``lan``.
    """
    if parse.urlparse(url).scheme not in ("http", "https"):
        return False
    return all(allowed_address(ip, lan=lan) for ip in _addresses(url))


def allowed_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, *, lan: bool) -> bool:
    """Whether a relay may dial ``ip``: any address for a home stream, else the internet only.

    Args:
        ip: An address a host resolved to.
        lan: Whether the stream started on the home network.

    Returns:
        allowed: False for internal addresses unless ``lan``.
    """
    return lan or ip.is_global


_IDLE_SEC: Final = 120.0
_MAX_LINE: Final = 8192


class Egress:
    """Loopback HTTP proxies ffmpeg reaches providers through, one per address rule.

    FFmpeg follows redirects, HLS variants, segments, and keys itself. Through these,
    each hop's connection is vetted and pinned like the server's own fetches: one
    proxy dials only the internet, the other (for home streams) anywhere.
    """

    def __init__(self) -> None:
        self._servers = {lan: _EgressServer(lan=lan) for lan in (False, True)}
        for server in self._servers.values():
            threading.Thread(
                target=functools.partial(server.serve_forever, poll_interval=0.05),
                name="egress",
                daemon=True,
            ).start()

    def url(self, *, lan: bool) -> str:
        """The proxy URL for ffmpeg's ``-http_proxy``.

        Args:
            lan: Whether the stream may reach the home network.

        Returns:
            url: ``http://127.0.0.1:<port>``.
        """
        host, port = self._servers[lan].server_address[:2]
        return f"http://{host!s}:{port}"

    def close(self) -> None:
        """Stop both proxies."""
        for server in self._servers.values():
            server.shutdown()
            server.server_close()


@functools.cache
def egress() -> Egress:
    """This process's egress proxies, started on first use."""
    return Egress()


def ffmpeg_input_args(url: str, source_id: str) -> list[str]:
    """FFmpeg input options that keep every connection for ``url`` behind the egress proxy.

    Args:
        url: An ffmpeg input: a provider URL, or a file or pipe of the server's own.
        source_id: The configured source that listed it; "" for none.

    Returns:
        args: For HTTP(S), a protocol whitelist and the proxy for the source's rule;
            none for the server's own files and pipes; for any other scheme, a
            whitelist it is not on, so ffmpeg cannot dial around the proxy.
    """
    if url.startswith(("/", "pipe:")):
        return []
    if parse.urlparse(url).scheme not in ("http", "https"):
        return ["-protocol_whitelist", "http,https"]
    return [
        "-protocol_whitelist",
        # Measured: HTTPS through the proxy needs tls, tcp and httpproxy; AES-128 HLS crypto.
        "http,https,tls,tcp,httpproxy,crypto",
        "-http_proxy",
        egress().url(lan=home_stream(url, source_id)),
    ]


def ffmpeg_env() -> dict[str, str]:
    """The environment for ffmpeg: this one, less ``no_proxy``, which would bypass the proxy."""
    return {k: v for k, v in os.environ.items() if k.lower() != "no_proxy"}


class RelayLimits:
    """How many relayed streams run, per viewer and in all.

    Each holds a provider connection and a thread, so a ref holder cannot open them
    without end.

    Args:
        max_per_owner: Streams one viewer's refs may relay at once.
        max_total: Streams relayed at once in all.
    """

    def __init__(self, *, max_per_owner: int = 4, max_total: int = 16) -> None:
        self.max_per_owner = max_per_owner
        self.max_total = max_total
        self.active: dict[str, int] = {}
        self._lock = threading.Lock()

    def acquire(self, owner: str) -> bool:
        """Take a slot for ``owner``.

        Args:
            owner: Viewer whose ref is relayed.

        Returns:
            taken: False, taking none, when either limit is reached.
        """
        with self._lock:
            if (
                self.active.get(owner, 0) >= self.max_per_owner
                or sum(self.active.values()) >= self.max_total
            ):
                return False
            self.active[owner] = self.active.get(owner, 0) + 1
            return True

    def release(self, owner: str) -> None:
        """Return a slot ``acquire`` took.

        Args:
            owner: Viewer the slot was taken for.
        """
        with self._lock:
            left = self.active.get(owner, 0) - 1
            if left > 0:
                self.active[owner] = left
            else:
                self.active.pop(owner, None)


async def _relay_body(
    stream: Relay,
    feed: Feed,
    on_close: Callable[[], object],
) -> AsyncGenerator[bytes]:
    """Yield the relay's bytes; a client that leaves closes the upstream."""
    try:
        async for chunk in feed:
            yield chunk
    finally:
        await stream.close()
        on_close()


def _addresses(url: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """The addresses ``url``'s host resolves to; none when it does not resolve."""
    host = parse.urlparse(url).hostname or ""
    try:
        found = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return []
    addresses = []
    for *_, sockaddr in found:
        ip = ipaddress.ip_address(str(sockaddr[0]).split("%")[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        addresses.append(ip)
    return addresses


class _EgressHandler(socketserver.StreamRequestHandler):
    # Unbuffered: bytes after the head belong to the tunnel, not to a hidden buffer.
    rbufsize = 0

    def __init__(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: tuple[str, int],
        server: socketserver.BaseServer,
        *,
        allow: Callable[[ipaddress.IPv4Address | ipaddress.IPv6Address], bool],
    ) -> None:
        self.allow = allow
        super().__init__(request, client_address, server)

    @override
    def handle(self) -> None:
        try:
            method, target, version = self.rfile.readline(_MAX_LINE).decode("latin-1").split()
        except ValueError:
            self._answer(400, "Bad Request")
            return
        head: list[str] = []
        while (line := self.rfile.readline(_MAX_LINE).decode("latin-1").rstrip("\r\n")) and len(
            head,
        ) < 100:
            head.append(line)
        tunnel = method == "CONNECT"
        url = parse.urlsplit(f"//{target}" if tunnel else target)
        try:
            port = url.port or (0 if tunnel else 80)
        except ValueError:
            port = 0
        if url.hostname is None or not port or (not tunnel and url.scheme != "http"):
            self._answer(400, "Bad Request")
            return
        try:
            upstream = util.pinned_connection(
                (url.hostname, port),
                _IDLE_SEC,
                allow_address=self.allow,
            )
        except util.RefusedAddress:
            self._answer(403, "Forbidden")
            return
        except OSError:
            self._answer(502, "Bad Gateway")
            return
        with upstream:
            if tunnel:
                self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            else:
                path = url.path or "/"
                kept = [h for h in head if not h.lower().startswith("proxy-")]
                upstream.sendall(
                    "".join(
                        f"{h}\r\n"
                        for h in (
                            f"{method} {path}{'?' if url.query else ''}{url.query} {version}",
                            *kept,
                            "",
                        )
                    ).encode("latin-1"),
                )
            _splice(self.connection, upstream)

    def _answer(self, status: int, reason: str) -> None:
        self.wfile.write(f"HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\n\r\n".encode())


def _splice(client: socket.socket, upstream: socket.socket) -> None:
    """Copy bytes both ways until either side closes or both sit idle."""
    peers = {client: upstream, upstream: client}
    try:
        while True:
            ready, _, _ = select.select(list(peers), (), (), _IDLE_SEC)
            if not ready:
                return
            for sock in ready:
                data = sock.recv(65_536)
                if not data:
                    return
                peers[sock].sendall(data)
    except OSError:
        return


class _EgressServer(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, *, lan: bool) -> None:
        super().__init__(
            ("127.0.0.1", 0),
            functools.partial(_EgressHandler, allow=functools.partial(allowed_address, lan=lan)),
        )
