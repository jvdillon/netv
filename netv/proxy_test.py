"""Tests for proxy.py: relayed upstream bodies and rewritten playlists."""

from __future__ import annotations

from collections.abc import Iterator
from http.client import HTTPResponse
from typing import cast
from urllib import parse

import asyncio
import contextlib
import ipaddress
import os
import socket
import struct
import threading

import pytest

from netv import cache, proxy, testing


def _relay(upstream: testing.FakeResponse) -> tuple[int, dict[str, str], str, list[object]]:
    async def run() -> tuple[int, dict[str, str], str, list[object]]:
        response = await proxy.relay_response(cast(HTTPResponse, upstream))
        body: list[object] = [chunk async for chunk in response.body_iterator]
        return response.status_code, dict(response.headers), str(response.media_type), body

    return asyncio.run(run())


def test_relay_releases_its_slot_when_the_stream_ends():
    limits = proxy.RelayLimits()
    assert limits.acquire("ann")
    upstream = testing.FakeResponse(b"x")

    async def run() -> None:
        response = await proxy.relay_response(
            cast(HTTPResponse, upstream),
            on_close=lambda: limits.release("ann"),
        )
        assert [chunk async for chunk in response.body_iterator] == [b"x"]

    asyncio.run(run())
    assert limits.active == {}


def test_relay_passes_a_range_with_its_headers():
    upstream = testing.FakeResponse(
        b"bytes",
        status=206,
        headers={
            "Content-Type": "video/x-matroska",
            "Content-Length": "5",
            "Content-Range": "bytes 10-14/100",
            "Accept-Ranges": "bytes",
            "Content-Encoding": "identity",
            "Set-Cookie": "upstream=secret",
        },
    )
    status, headers, media_type, body = _relay(upstream)
    assert (status, media_type, body) == (206, "video/x-matroska", [b"bytes"])
    assert headers["content-length"] == "5"
    assert headers["content-range"] == "bytes 10-14/100"
    assert headers["accept-ranges"] == "bytes"
    assert headers["content-encoding"] == "identity"
    assert headers["cache-control"] == "no-store"
    # Nothing else of the upstream's reaches the client.
    assert "set-cookie" not in headers
    assert upstream.closed


@pytest.mark.parametrize("hostile", ["text/html", "image/svg+xml", "application/javascript"])
def test_relay_never_serves_a_page_on_this_origin(hostile: str):
    # A provider's HTML or SVG opened from a relayed link would run as this site.
    _, headers, media_type, _ = _relay(
        testing.FakeResponse(b"<script>", headers={"Content-Type": hostile}),
    )
    assert media_type == "application/octet-stream"
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["content-security-policy"] == "sandbox"


@pytest.mark.parametrize(
    "media",
    ["video/mp4", "audio/aac", "application/vnd.apple.mpegurl", "application/octet-stream"],
)
def test_relay_keeps_media_types(media: str):
    assert _relay(testing.FakeResponse(b"x", headers={"Content-Type": media}))[2] == media


def test_relay_omits_empty_headers_and_defaults_to_transport_stream():
    status, headers, media_type, _ = _relay(
        testing.FakeResponse(b"x", headers={"Content-Range": "", "Accept-Ranges": ""}),
    )
    assert (status, media_type) == (200, "video/mp2t")
    assert "content-range" not in headers and "accept-ranges" not in headers


def test_rewrite_playlist_refuses_one_too_long_to_be_real():
    text = "#EXTM3U\n" + "s.ts\n" * 6
    assert proxy.rewrite_playlist(text, "http://e/", str, max_lines=7).count("http://e/s.ts") == 6
    with pytest.raises(ValueError, match="too long"):
        proxy.rewrite_playlist(text + "s.ts\n", "http://e/", str, max_lines=7)


@pytest.mark.parametrize(
    ("url", "lan", "allowed"),
    [
        ("http://8.8.8.8/seg.ts", False, True),
        # A provider's playlist must not reach into this server or the home network.
        ("http://127.0.0.1:8000/settings", False, False),
        ("http://[::1]/x", False, False),
        ("http://169.254.169.254/latest/meta-data/", False, False),
        ("http://10.0.0.5/admin", False, False),
        ("http://192.168.1.1/", False, False),
        ("http://100.64.0.1/", False, False),
        ("http://[::ffff:127.0.0.1]/", False, False),
        ("http://0.0.0.0/", False, False),
        # A stream on the home network, such as a tuner, may use it.
        ("http://192.168.1.87:5004/auto/v5.1", True, True),
        # A name that does not resolve cannot be fetched either way.
        ("http://no-such-host.invalid/x", False, True),
        ("file:///etc/passwd", True, False),
    ],
)
def test_upstream_addresses_a_relay_may_fetch(url: str, lan: bool, allowed: bool):
    assert proxy.allowed_upstream(url, lan=lan) is allowed


@pytest.mark.parametrize(
    ("address", "lan", "allowed"),
    [
        ("8.8.8.8", False, True),
        ("127.0.0.1", False, False),
        ("10.0.0.5", False, False),
        ("10.0.0.5", True, True),
    ],
)
def test_addresses_a_relay_may_dial(address: str, lan: bool, allowed: bool):
    assert proxy.allowed_address(ipaddress.ip_address(address), lan=lan) is allowed


def test_a_home_source_s_streams_are_home_streams_and_a_provider_s_are_not():
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "sources": [
                {
                    "id": "t",
                    "name": "Tuner",
                    "type": "m3u",
                    "url": "http://192.168.1.24/lineup.m3u",
                },
                {"id": "p", "name": "Provider", "type": "m3u", "url": "http://8.8.8.8/list.m3u"},
            ],
        },
    )
    # A lineup on the home network may list a tuner on another home host.
    tuner = "http://192.168.1.87:5004/auto/v5.1"
    assert proxy.home_stream(tuner, "t")
    # A provider's list naming an internal address does not make it a home stream.
    assert not proxy.home_stream(tuner, "p")
    assert not proxy.home_stream("http://127.0.0.1:8000/settings", "p")
    assert not proxy.home_stream(tuner, "")
    assert not proxy.home_stream(tuner, "gone")
    # Nor does a home source make an internet stream reach the home network.
    assert not proxy.home_stream("http://8.8.8.8/live/1.ts", "t")
    rule = proxy.address_rule("http://8.8.8.8/live/1.ts", "p")
    assert rule(ipaddress.ip_address("8.8.4.4")) and not rule(ipaddress.ip_address("127.0.0.1"))
    assert proxy.address_rule(tuner, "t")(ipaddress.ip_address("192.168.1.87"))
    assert not proxy.address_rule(tuner, "p")(ipaddress.ip_address("192.168.1.87"))


def test_a_stream_is_on_the_home_network_when_its_host_is():
    assert proxy.is_lan_url("http://192.168.1.87:5004/auto/v5.1")
    assert proxy.is_lan_url("http://localhost:8080/x")
    assert not proxy.is_lan_url("http://8.8.8.8/live.m3u8")
    assert not proxy.is_lan_url("http://no-such-host.invalid/x")


def test_relay_limits_default_to_a_few_per_viewer_and_more_in_all():
    limits = proxy.RelayLimits()
    assert [limits.acquire("ann") for _ in range(5)] == [True] * 4 + [False]
    assert [limits.acquire(f"v{i}") for i in range(13)] == [True] * 12 + [False]


def test_relay_limits_count_streams_per_viewer_and_in_all():
    limits = proxy.RelayLimits(max_per_owner=2, max_total=3)
    assert limits.acquire("ann") and limits.acquire("ann")
    # A viewer's third stream waits for one of the first two to end.
    assert not limits.acquire("ann")
    assert limits.acquire("bob")
    assert not limits.acquire("cat")
    limits.release("ann")
    assert limits.acquire("cat")
    limits.release("cat")
    assert limits.active == {"ann": 1, "bob": 1}
    limits.release("ann")
    limits.release("bob")
    assert limits.active == {}


def test_rewrite_playlist_links_every_uri_and_keeps_every_tag():
    text = (
        "#EXTM3U\n"
        '#EXT-X-KEY:METHOD=AES-128,URI="key.bin"\n'
        '#EXT-X-MEDIA:TYPE=AUDIO,URI="audio/en.m3u8",NAME="en"\n'
        "#EXTINF:6,\n"
        "seg1.ts\n"
        "\n"
        "  http://cdn/a/seg2.ts  \n"
        "#EXT-X-ENDLIST"
    )
    rewritten = proxy.rewrite_playlist(text, "http://edge/x/list.m3u8", lambda u: f"<{u}>")
    assert rewritten == (
        "#EXTM3U\n"
        '#EXT-X-KEY:METHOD=AES-128,URI="<http://edge/x/key.bin>"\n'
        '#EXT-X-MEDIA:TYPE=AUDIO,URI="<http://edge/x/audio/en.m3u8>",NAME="en"\n'
        "#EXTINF:6,\n"
        "<http://edge/x/seg1.ts>\n"
        "\n"
        "<http://cdn/a/seg2.ts>\n"
        "#EXT-X-ENDLIST\n"
    )


class _Origin:
    """A loopback server that reads each connection's request head and answers ``reply``."""

    def __init__(self, reply: bytes) -> None:
        self.reply = reply
        self.received: list[bytes] = []
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port = int(self.listener.getsockname()[1])
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self.listener.accept()
            except OSError:
                return
            with connection:
                head = b""
                while b"\r\n\r\n" not in head and (chunk := connection.recv(4096)):
                    head += chunk
                self.received.append(head)
                # A client that resets its tunnel leaves no one to read the reply.
                with contextlib.suppress(OSError):
                    connection.sendall(self.reply)


@pytest.fixture
def egress() -> Iterator[proxy.Egress]:
    proxies = proxy.Egress()
    yield proxies
    proxies.close()


def _ask(egress: proxy.Egress, *, lan: bool, request: bytes, then: bytes = b"") -> bytes:
    """What the egress proxy answers ``request``, and ``then`` sent after its first reply."""
    url = parse.urlsplit(egress.url(lan=lan))
    with socket.create_connection((str(url.hostname), url.port or 0), timeout=5) as client:
        client.sendall(request)
        answer = client.recv(65_536)
        if then:
            client.sendall(then)
        while chunk := client.recv(65_536):
            answer += chunk
    return answer


def test_egress_listens_on_loopback_only_and_neither_reaches_the_other(egress: proxy.Egress):
    # Neither takes a token: on a public interface the home one would open the LAN.
    home, away = (parse.urlsplit(egress.url(lan=lan)) for lan in (True, False))
    assert home.hostname == away.hostname == "127.0.0.1"
    request = f"CONNECT 127.0.0.1:{home.port} HTTP/1.1\r\nHost: x\r\n\r\n".encode()
    assert _ask(egress, lan=False, request=request).startswith(b"HTTP/1.1 403 ")
    # Nor does the relay serve them to an internet stream.
    assert not proxy.allowed_upstream(egress.url(lan=True), lan=False)


def test_home_egress_relays_a_request_and_passes_its_redirect_through(egress: proxy.Egress):
    redirect = b"HTTP/1.1 302 Found\r\nLocation: http://8.8.8.8/x\r\nContent-Length: 0\r\n\r\n"
    origin = _Origin(redirect)
    request = (
        f"GET http://127.0.0.1:{origin.port}/live/a.m3u8?t=1 HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{origin.port}\r\nProxy-Connection: keep-alive\r\nRange: bytes=5-\r\n\r\n"
    ).encode()
    # FFmpeg follows the redirect itself, through the proxy again, so it is vetted too.
    assert _ask(egress, lan=True, request=request) == redirect
    [head] = origin.received
    assert head.startswith(b"GET /live/a.m3u8?t=1 HTTP/1.1\r\n")
    assert b"Range: bytes=5-\r\n" in head and b"Proxy-" not in head


def test_home_egress_tunnels_a_connect(egress: proxy.Egress):
    origin = _Origin(b"hello")
    connect = f"CONNECT 127.0.0.1:{origin.port} HTTP/1.1\r\nHost: x\r\n\r\n".encode()
    answer = _ask(egress, lan=True, request=connect, then=b"TLS\r\n\r\n")
    assert answer == b"HTTP/1.1 200 Connection established\r\n\r\nhello"
    assert origin.received == [b"TLS\r\n\r\n"]


@pytest.mark.parametrize(
    "request_line",
    [
        "GET http://127.0.0.1:{port}/settings HTTP/1.1",
        "CONNECT 127.0.0.1:{port} HTTP/1.1",
        "CONNECT 10.0.0.5:443 HTTP/1.1",
        "CONNECT [::1]:{port} HTTP/1.1",
    ],
)
def test_internet_egress_refuses_this_server_and_the_home_network(
    egress: proxy.Egress,
    request_line: str,
):
    origin = _Origin(b"secret")
    request = f"{request_line.format(port=origin.port)}\r\nHost: x\r\n\r\n".encode()
    assert _ask(egress, lan=False, request=request).startswith(b"HTTP/1.1 403 ")
    assert origin.received == []


@pytest.mark.parametrize(
    "request_line",
    [
        "GET /relative HTTP/1.1",
        "GET ftp://8.8.8.8/x HTTP/1.1",
        "CONNECT nohost HTTP/1.1",
        "CONNECT host:99999 HTTP/1.1",
        "junk",
    ],
)
def test_egress_refuses_what_it_cannot_route(egress: proxy.Egress, request_line: str):
    request = f"{request_line}\r\nHost: x\r\n\r\n".encode()
    assert _ask(egress, lan=True, request=request).startswith(b"HTTP/1.1 400 ")


def test_egress_answers_bad_gateway_when_the_host_does_not_answer(egress: proxy.Egress):
    closed = socket.create_server(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    request = f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\nHost: x\r\n\r\n".encode()
    assert _ask(egress, lan=True, request=request).startswith(b"HTTP/1.1 502 ")


def test_egress_closes_a_tunnel_that_sits_idle(
    egress: proxy.Egress,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(proxy, "_IDLE_SEC", 0.2)
    silent = socket.create_server(("127.0.0.1", 0))
    request = f"CONNECT 127.0.0.1:{silent.getsockname()[1]} HTTP/1.1\r\nHost: x\r\n\r\n".encode()
    # Nothing either way: the proxy hangs up rather than hold the connection forever.
    assert _ask(egress, lan=True, request=request) == b"HTTP/1.1 200 Connection established\r\n\r\n"
    silent.close()


def test_egress_survives_a_client_that_resets_its_tunnel(egress: proxy.Egress):
    origin = _Origin(b"x" * 1_000_000)
    url = parse.urlsplit(egress.url(lan=True))
    client = socket.create_connection((str(url.hostname), url.port or 0), timeout=5)
    client.sendall(f"CONNECT 127.0.0.1:{origin.port} HTTP/1.1\r\n\r\n".encode())
    client.recv(64)
    client.sendall(b"GO\r\n\r\n")
    # Close with a reset while the origin's reply is still flowing.
    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    client.close()
    after = _Origin(b"ok")
    request = f"CONNECT 127.0.0.1:{after.port} HTTP/1.1\r\n\r\n".encode()
    assert _ask(egress, lan=True, request=request, then=b"\r\n\r\n").endswith(b"ok")


def test_the_process_starts_one_egress_on_first_use():
    proxy.egress.cache_clear()
    try:
        first = proxy.egress()
        assert proxy.egress() is first
        assert parse.urlsplit(first.url(lan=False)).hostname == "127.0.0.1"
        first.close()
    finally:
        # The next caller starts a fresh one.
        proxy.egress.cache_clear()


def test_ffmpeg_reaches_providers_only_through_the_egress_proxy(monkeypatch: pytest.MonkeyPatch):
    cache.save_server_settings(
        {
            **cache.load_server_settings(),
            "sources": [
                {
                    "id": "t",
                    "name": "Tuner",
                    "type": "m3u",
                    "url": "http://192.168.1.24/lineup.m3u",
                },
            ],
        },
    )
    allowed = "http,https,tls,tcp,httpproxy,crypto"
    tuner = "http://192.168.1.87:5004/auto/v5.1"
    home = proxy.egress().url(lan=True)
    away = proxy.egress().url(lan=False)
    assert home != away
    assert proxy.ffmpeg_input_args(tuner, "t") == [
        "-protocol_whitelist",
        allowed,
        "-http_proxy",
        home,
    ]
    assert proxy.ffmpeg_input_args(tuner, "") == [
        "-protocol_whitelist",
        allowed,
        "-http_proxy",
        away,
    ]
    assert proxy.ffmpeg_input_args("https://p.example/1.mkv", "t")[-1] == away
    # The server's own files and pipes are read directly.
    assert proxy.ffmpeg_input_args("/tmp/rec/stream.ts", "") == []
    assert proxy.ffmpeg_input_args("pipe:0", "") == []
    # Any other scheme (tcp, file, rtmp) is one ffmpeg may not open.
    assert proxy.ffmpeg_input_args("tcp://10.0.0.5:22", "") == ["-protocol_whitelist", "http,https"]
    # A no_proxy variable would send matching hosts around the proxy.
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "10.0.0.0/8")
    env = proxy.ffmpeg_env()
    assert "no_proxy" not in env and "NO_PROXY" not in env
    assert env["PATH"] == os.environ["PATH"]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
