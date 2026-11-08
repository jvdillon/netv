"""LAN Chromecast control and ownership of receiver-side playback sessions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from typing import Literal, NotRequired, TypedDict
from urllib.parse import urlsplit
from uuid import UUID

import logging
import threading
import time

from fastapi import HTTPException
from pychromecast import Chromecast, get_chromecast_from_host
from pychromecast.discovery import CastBrowser, SimpleCastListener
from pychromecast.error import PyChromecastError
from zeroconf import Zeroconf

from cache import save_watch_position

import ffmpeg_session


log = logging.getLogger(__name__)
_LAN_NETWORKS = tuple(
    ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)


def lan_host(host: str) -> str:
    try:
        address = ip_address(host)
    except ValueError as exc:
        raise HTTPException(
            400,
            "Enter the Chromecast's LAN IP address, not a hostname or URL.",
        ) from exc
    if not any(address in network for network in _LAN_NETWORKS):
        raise HTTPException(400, "The Chromecast address must be a private LAN IP address.")
    return str(address)


def media_origin(address: str) -> str:
    """Validate the public-facing origin without trusting a container's own address.

    Args:
        address: Server origin reachable by the receiver.

    Returns:
        origin: Validated HTTP or HTTPS origin without a trailing slash.
    """
    try:
        parsed = urlsplit(address)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(400, "Invalid neTV server address.") from exc
    if (
        parsed.scheme not in ("http", "https")
        or hostname is None
        or hostname == ""
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or (port is not None and port == 0)
    ):
        raise HTTPException(400, "Use a full neTV address, such as http://192.168.1.10:8000.")
    try:
        address_ip = ip_address(hostname)
    except ValueError:
        address_ip = None
    if (
        hostname in ("localhost", "host.docker.internal")
        or hostname.endswith(".localhost")
        or (
            address_ip
            and (address_ip.is_loopback or address_ip.is_unspecified or address_ip.is_multicast)
        )
    ):
        raise HTTPException(
            400,
            "Set the neTV server address to an address reachable by the TV, not localhost.",
        )
    return f"{parsed.scheme}://{parsed.netloc}"


class CastStatus(TypedDict):
    """Cast state reported to the browser for one user."""

    active: bool
    connecting: NotRequired[bool]
    host: NotRequired[str]
    name: NotRequired[str]
    session_id: NotRequired[str]
    state: NotRequired[str]
    current_time: NotRequired[float | None]
    volume: NotRequired[float]


@dataclass(kw_only=True, slots=True)
class CastSession:
    username: str
    host: str
    session_id: str
    url: str
    source_url: str
    name: str
    cast: Chromecast | None = None
    starting: bool = True


class CastManager:
    def __init__(self) -> None:
        self._sessions: dict[str, CastSession] = {}
        self._lock = threading.RLock()
        self._operations = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    def discover(self) -> list[dict[str, str]]:
        with Zeroconf() as zconf:
            browser = CastBrowser(SimpleCastListener(lambda *_: None), zconf)
            browser.start_discovery()
            try:
                self._stop.wait(5)
                devices = list(browser.devices.values())
            finally:
                browser.stop_discovery()
        result = {}
        for device in devices:
            try:
                host = lan_host(device.host)
            except HTTPException:
                continue  # Ignore non-LAN interfaces advertised by mDNS.
            result[host] = {"host": host, "name": device.friendly_name or host}
        return sorted(result.values(), key=lambda device: device["name"].lower())

    def owns_session(self, session_id: str) -> bool:
        with self._lock:
            return any(s.session_id == session_id for s in self._sessions.values())

    def owns_url(self, url: str) -> bool:
        with self._lock:
            return any(s.source_url == url for s in self._sessions.values())

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._stop.clear()
                self._worker = threading.Thread(
                    target=self._monitor,
                    daemon=True,
                    name="cast-monitor",
                )
                self._worker.start()

    def start(
        self,
        username: str,
        host: str,
        session_id: str,
        origin: str,
        title: str,
        current_time: float,
    ) -> CastStatus:
        host = lan_host(host)
        origin = media_origin(origin)
        with self._operations:
            session = ffmpeg_session.get_session(session_id)
            if session is None or session.get("username") != username:
                raise HTTPException(
                    404,
                    "Playback session not found. Start playback and try again.",
                )
            with self._lock:
                if host in self._sessions or any(
                    s.username == username for s in self._sessions.values()
                ):
                    raise HTTPException(409, "Stop the existing cast before starting another.")
                filename = "master.m3u8" if session.get("fast_start", False) else "stream.m3u8"
                record = CastSession(
                    username=username,
                    host=host,
                    session_id=session_id,
                    url=f"{origin}/transcode/{session_id}/{filename}",
                    source_url=session["url"],
                    name=host,
                )
                self._sessions[host] = record
            self._ensure_worker()
            ffmpeg_session.touch_session(session_id)
            try:
                cast = get_chromecast_from_host(
                    (host, 8009, UUID(int=0), None, host),
                    tries=1,
                    timeout=5,
                    retry_wait=1,
                )
                record.cast = cast
                cast.wait(timeout=10)
                record.name = cast.cast_info.friendly_name or host
                cast.media_controller.app_must_match = True
                cast.media_controller.play_media(
                    record.url,
                    "application/x-mpegurl",
                    title=title,
                    stream_type="BUFFERED" if session.get("is_vod", False) else "LIVE",
                    current_time=current_time if session.get("is_vod", False) else None,
                    media_info={"hlsSegmentFormat": "TS", "hlsVideoSegmentFormat": "MPEG2_TS"},
                )
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    status = cast.media_controller.status
                    if status.content_id == record.url and status.player_state in (
                        "PLAYING",
                        "PAUSED",
                        "BUFFERING",
                    ):
                        record.starting = False
                        return self.status(username)
                    if status.content_id == record.url and status.idle_reason == "ERROR":
                        break
                    self._stop.wait(0.1)
                raise HTTPException(
                    502,
                    "The TV could not load the stream. Check its access to the neTV server address "
                    "and, for HTTPS, certificate trust.",
                )
            except (PyChromecastError, OSError) as exc:
                log.warning("Chromecast connection failed for %s: %s", host, type(exc).__name__)
                raise HTTPException(
                    502,
                    "Could not connect to the Chromecast. Check its IP and LAN access.",
                ) from exc
            finally:
                if record.starting:
                    self._forget(record)

    def status(self, username: str) -> CastStatus:
        with self._lock:
            record = next((s for s in self._sessions.values() if s.username == username), None)
            if not record:
                return {"active": False}
            if record.starting:
                return {"active": False, "connecting": True}
            status = record.cast.media_controller.status if record.cast else None
            return {
                "active": True,
                "host": record.host,
                "name": record.name,
                "session_id": record.session_id,
                "state": status.player_state if status else "UNKNOWN",
                "current_time": status.adjusted_current_time if status else 0,
                "volume": record.cast.status.volume_level
                if record.cast and record.cast.status
                else 1,
            }

    def command(
        self,
        username: str,
        action: Literal["play", "pause", "stop", "volume"],
        volume: float | None = None,
        *,
        release: Callable[[str], None],
    ) -> CastStatus:
        """Control the user's cast.

        Args:
            username: Owner of the cast.
            action: Receiver command.
            volume: Level between 0 and 1 for ``volume``.
            release: Stops a transcode session by ID; called on ``stop``.

        Returns:
            status: Cast status after the command.
        """
        with self._operations:
            with self._lock:
                record = next((s for s in self._sessions.values() if s.username == username), None)
            if not record or not record.cast:
                raise HTTPException(404, "No active cast for this user.")
            cast = record.cast
            try:
                # Never stop or control another app that has taken over the receiver.
                if cast.media_controller.status.content_id != record.url:
                    self._forget(record)
                    raise HTTPException(409, "The TV is no longer playing this neTV stream.")
                if action == "stop":
                    self._save_position(record)
                    cast.media_controller.stop()
                    self._forget(record)
                    release(record.session_id)
                elif action == "play":
                    cast.media_controller.play()
                elif action == "pause":
                    cast.media_controller.pause()
                else:
                    if volume is None:
                        raise HTTPException(400, "A volume between 0 and 1 is required.")
                    cast.set_volume(volume)
            except (PyChromecastError, OSError) as exc:
                log.warning(
                    "Chromecast %s failed for %s: %s",
                    action,
                    record.host,
                    type(exc).__name__,
                )
                raise HTTPException(
                    502,
                    "The Chromecast did not respond. Check the TV and network.",
                ) from exc
            return self.status(username)

    def _forget(self, record: CastSession) -> None:
        with self._lock:
            if self._sessions.get(record.host) is record:
                del self._sessions[record.host]
        if record.cast:
            record.cast.disconnect(timeout=0)

    def _monitor(self) -> None:
        while not self._stop.wait(5):
            self.maintain_sessions()

    def _save_position(self, record: CastSession) -> None:
        session = ffmpeg_session.get_session(record.session_id)
        if session is None or not session.get("is_vod", False) or not record.cast:
            return
        status = record.cast.media_controller.status
        if status.content_id != record.url:
            return
        position = (status.adjusted_current_time or 0) + session.get("seek_offset", 0)
        duration = session.get("duration", 0)
        if duration > 0:
            position = min(position, duration)
        if position < 5:
            return
        try:
            save_watch_position(record.username, record.source_url, position, duration)
        except OSError:
            log.exception("Could not save Chromecast watch position")

    def maintain_sessions(self) -> None:
        with self._lock:
            records = list(self._sessions.values())
        for record in records:
            if record.starting:
                ffmpeg_session.touch_session(record.session_id)
                continue
            cast = record.cast
            if (
                cast
                and cast.socket_client.is_connected
                and cast.media_controller.status.content_id == record.url
                and cast.media_controller.status.player_state in ("PLAYING", "PAUSED", "BUFFERING")
            ):
                ffmpeg_session.touch_session(record.session_id)
                self._save_position(record)
            else:
                log.info("Cast ended or disconnected: %s", record.host)
                self._forget(record)
                # Normal session expiry releases the encoder after the last media request.

    def close(self) -> None:
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=6)
        with self._lock:
            records = list(self._sessions.values())
        for record in records:
            self._forget(record)


manager = CastManager()
