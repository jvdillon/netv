"""M3U playlist / XMLTV export for external IPTV player apps (TiviMate, Smarters, VLC).

Exposes the live channels a neTV user can access as an Xtream-style playlist:

    {base_url}/get.php?username=U&password=P     -> M3U playlist
    {base_url}/xmltv.php?username=U&password=P   -> XMLTV guide
    {base_url}/live/U/P/{stream_id}.ts           -> channel playback

Playback has two modes (server setting ``m3u_export_mode``):

- ``redirect``: 302 to the upstream URL. Zero server bandwidth, but the
  provider URL (including provider credentials for Xtream sources) is
  visible to the client.
- ``proxy``: stream bytes through this server. Hides the provider and
  enforces per-user/per-source stream limits, at the cost of upstream +
  downstream bandwidth per viewer. HLS (.m3u8) direct URLs are always
  redirected since proxying would require rewriting segment URLs.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from xml.sax.saxutils import (
    escape as xml_escape,
    quoteattr as xml_quoteattr,
)

import hashlib
import hmac
import logging
import threading
import time
import urllib.parse

from cache import get_sources
from ffmpeg_command import get_user_agent
from util import safe_urlopen

import auth
import epg


log = logging.getLogger(__name__)

# Verified-credential cache: PBKDF2 verification is ~100ms on small ARM
# boards, too slow to run on every channel zap. username -> (sha256, expiry).
_CRED_CACHE: dict[str, tuple[str, float]] = {}
_CRED_CACHE_TTL = 300
_cred_lock = threading.Lock()

# Active proxied streams: (username, source_id) -> count. Tracked separately
# from web transcode sessions (ffmpeg_session), so limits apply per system.
_active_streams: dict[tuple[str, str], int] = {}
_streams_lock = threading.Lock()

XMLTV_WINDOW_HOURS = 48
_PROXY_CHUNK_SIZE = 64 * 1024
_STRIP_EXTS = (".ts", ".m3u8")


def verify_credentials(username: str, password: str) -> bool:
    """Verify username/password, caching successes to avoid repeated PBKDF2."""
    digest = hashlib.sha256(password.encode()).hexdigest()
    now = time.time()
    with _cred_lock:
        cached = _CRED_CACHE.get(username)
        if cached and cached[1] > now and hmac.compare_digest(cached[0], digest):
            return True
    if not auth.verify_password(username, password):
        return False
    with _cred_lock:
        _CRED_CACHE[username] = (digest, now + _CRED_CACHE_TTL)
        # Bounded: one entry per user, plus purge anything expired
        for u in [u for u, (_, exp) in _CRED_CACHE.items() if exp <= now]:
            del _CRED_CACHE[u]
    return True


def clear_credential_cache(username: str | None = None) -> None:
    """Drop cached credentials (call after password change/user delete)."""
    with _cred_lock:
        if username is None:
            _CRED_CACHE.clear()
        else:
            _CRED_CACHE.pop(username, None)


def stream_allowed(stream: dict, unavailable_groups: set[str]) -> bool:
    """Same rule as the guide: blocked if ANY of its categories is unavailable."""
    cat_ids = stream.get("category_ids") or []
    return not any(f"cat:{c}" in unavailable_groups for c in cat_ids)


def allowed_live_streams(streams: list[dict], username: str) -> list[dict]:
    """Filter live streams down to what this user may access."""
    limits = auth.get_user_limits(username)
    unavailable = set(limits.get("unavailable_groups", []))
    if not unavailable:
        return streams
    return [s for s in streams if stream_allowed(s, unavailable)]


def upstream_url(stream: dict) -> str:
    """Resolve a live stream dict to its upstream URL (MPEG-TS for Xtream)."""
    if stream.get("direct_url"):
        return stream["direct_url"]
    if stream.get("source_type") == "xtream":
        user = urllib.parse.quote(stream["source_username"], safe="")
        pwd = urllib.parse.quote(stream["source_password"], safe="")
        stream_id = str(stream["stream_id"])
        orig_id = stream_id.split("_")[-1] if "_" in stream_id else stream_id
        return f"{stream['source_url']}/live/{user}/{pwd}/{orig_id}.ts"
    return ""


def strip_stream_ext(stream_spec: str) -> str:
    """Strip a trailing .ts/.m3u8 that players append to playback URLs."""
    for ext in _STRIP_EXTS:
        if stream_spec.endswith(ext):
            return stream_spec[: -len(ext)]
    return stream_spec


def build_playlist(
    categories: list[dict],
    streams: list[dict],
    base_url: str,
    username: str,
    password: str,
) -> str:
    """Build an m3u_plus playlist pointing back at this server's /live/ URLs."""
    cat_names = {str(c.get("category_id")): c.get("category_name", "") for c in categories}
    user = urllib.parse.quote(username, safe="")
    pwd = urllib.parse.quote(password, safe="")
    lines = ["#EXTM3U"]
    for s in streams:
        name = s.get("name", "Unknown")
        tvg_id = s.get("epg_channel_id") or ""
        logo = s.get("stream_icon") or ""
        group = next(
            (cat_names[str(c)] for c in (s.get("category_ids") or []) if str(c) in cat_names),
            "",
        )
        attrs = f'tvg-id="{tvg_id}" tvg-name="{name}" tvg-logo="{logo}" group-title="{group}"'
        lines.append(f"#EXTINF:-1 {attrs},{name}")
        lines.append(f"{base_url}/live/{user}/{pwd}/{urllib.parse.quote(str(s['stream_id']))}.ts")
    return "\n".join(lines) + "\n"


def build_xmltv(streams: list[dict], hours: int = XMLTV_WINDOW_HOURS) -> str:
    """Build an XMLTV guide for the given streams from the local EPG database."""
    epg_ids = sorted({s.get("epg_channel_id") or "" for s in streams} - {""})
    now = datetime.now(UTC)
    programs_map = epg.get_programs_batch(epg_ids, now - timedelta(hours=2), now + timedelta(hours=hours))
    icons_map = epg.get_icons_batch(epg_ids)

    out = ['<?xml version="1.0" encoding="utf-8"?>', '<tv generator-info-name="neTV">']
    seen: set[str] = set()
    for s in streams:
        epg_id = s.get("epg_channel_id") or ""
        if not epg_id or epg_id in seen:
            continue
        seen.add(epg_id)
        out.append(f"  <channel id={xml_quoteattr(epg_id)}>")
        out.append(f"    <display-name>{xml_escape(s.get('name', ''))}</display-name>")
        icon = s.get("stream_icon") or icons_map.get(epg_id, "")
        if icon:
            out.append(f"    <icon src={xml_quoteattr(icon)}/>")
        out.append("  </channel>")
    for epg_id in epg_ids:
        for p in programs_map.get(epg_id, []):
            start = p.start.strftime("%Y%m%d%H%M%S %z")
            stop = p.stop.strftime("%Y%m%d%H%M%S %z")
            out.append(
                f'  <programme start="{start}" stop="{stop}" channel={xml_quoteattr(epg_id)}>'
            )
            out.append(f"    <title>{xml_escape(p.title)}</title>")
            if p.desc:
                out.append(f"    <desc>{xml_escape(p.desc)}</desc>")
            out.append("  </programme>")
    out.append("</tv>")
    return "\n".join(out) + "\n"


def _source_max_streams(source_id: str) -> int:
    source = next((s for s in get_sources() if s.id == source_id), None)
    return source.max_streams if source else 0


def try_acquire_stream(username: str, source_id: str) -> bool:
    """Reserve a proxied stream slot. Returns False if a limit is hit (0 = unlimited)."""
    user_max = auth.get_user_limits(username).get("max_streams_per_source", {}).get(source_id, 0)
    source_max = _source_max_streams(source_id)
    with _streams_lock:
        user_count = _active_streams.get((username, source_id), 0)
        source_count = sum(n for (_, sid), n in _active_streams.items() if sid == source_id)
        if user_max and user_count >= user_max:
            log.info("Proxy stream denied: user=%s source=%s at user limit", username, source_id)
            return False
        if source_max and source_count >= source_max:
            log.info("Proxy stream denied: user=%s source=%s at source limit", username, source_id)
            return False
        _active_streams[(username, source_id)] = user_count + 1
    return True


def release_stream(username: str, source_id: str) -> None:
    """Release a slot reserved by try_acquire_stream."""
    with _streams_lock:
        key = (username, source_id)
        count = _active_streams.get(key, 0)
        if count <= 1:
            _active_streams.pop(key, None)
        else:
            _active_streams[key] = count - 1


def iter_proxy_stream(url: str, username: str, source_id: str, timeout: int = 15) -> Iterator[bytes]:
    """Yield upstream bytes; caller must have acquired a slot, released here."""
    try:
        with safe_urlopen(url, timeout=timeout, user_agent=get_user_agent()) as resp:
            while chunk := resp.read(_PROXY_CHUNK_SIZE):
                yield chunk
    except Exception as e:
        log.warning("Proxy stream ended: user=%s source=%s: %s", username, source_id, e)
    finally:
        release_stream(username, source_id)
