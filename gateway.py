#!/usr/bin/env python3
"""Xtream-compatible native-player gateway for neTV."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import argparse
import asyncio
import contextlib
import hashlib
import logging
import os
import threading
import time
import urllib.error
import urllib.parse

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from gateway_catalog import CatalogSnapshot, GatewayCatalog, GatewayStream
from gateway_epg import EpgUnavailableError, build_filtered_xmltv
from m3u import get_xtream_client_by_source
from util import safe_urlopen

import auth
import cache
import ffmpeg_command
import ffmpeg_session


log = logging.getLogger(__name__)
catalog = GatewayCatalog()
_UPSCALE_ENGINE_DIR = Path(
    os.environ.get("SR_ENGINE_DIR", Path.home() / "ffmpeg_build/models")
)
_PASSTHROUGH_ACTIONS = {
    "get_vod_categories",
    "get_vod_streams",
    "get_series_categories",
    "get_series",
}
_AUTH_CACHE_SECONDS = 30
_LOGIN_WINDOW_SECONDS = 300
_MAX_LOGIN_FAILURES = 10
_MAX_GLOBAL_LOGIN_FAILURES = 100
_MAX_AUTH_CACHE_ENTRIES = 1024
_MAX_FAILURE_CLIENTS = 1024


def _available_upscale_models() -> list[str]:
    if not _UPSCALE_ENGINE_DIR.exists():
        return []
    pattern = f"{ffmpeg_command.SR_MODEL_NAME}_*p_fp16.engine"
    return (
        [ffmpeg_command.SR_MODEL_NAME]
        if next(_UPSCALE_ENGINE_DIR.glob(pattern), None)
        else []
    )


def _upscale_settings() -> dict[str, Any]:
    settings = cache.load_server_settings()
    available_models = _available_upscale_models()
    configured_model = (
        ffmpeg_command.SR_MODEL_NAME
        if str(settings.get("sr_model") or "").strip()
        else ""
    )
    if configured_model and configured_model not in available_models:
        raise RuntimeError(
            f"Configured AI Upscale model {configured_model!r} is not installed; "
            f"available models: {', '.join(available_models) or 'none'}"
        )
    upscale_settings = {
        **settings,
        "transcode_hw": str(settings.get("transcode_hw") or "nvenc+software"),
        "max_resolution": str(settings.get("max_resolution") or "1080p"),
        "quality": str(settings.get("quality") or "high"),
        "sr_model": configured_model,
    }
    if upscale_settings["max_resolution"] not in ("4k", "1080p", "720p", "480p"):
        raise RuntimeError(
            "Configured max_resolution must be one of: 4k, 1080p, 720p, 480p"
        )
    if upscale_settings["quality"] not in ("high", "medium", "low"):
        raise RuntimeError(
            "Configured quality must be one of: high, medium, low"
        )
    return upscale_settings


def _upscale_enabled() -> bool:
    return bool(_upscale_settings()["sr_model"])


async def _cleanup_upscale_sessions() -> None:
    while True:
        await asyncio.sleep(60)
        await asyncio.to_thread(ffmpeg_session.cleanup_expired_sessions)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    ffmpeg_command.init(
        _upscale_settings,
        sr_engine_dir=str(_UPSCALE_ENGINE_DIR),
    )
    ffmpeg_session.cleanup_and_recover_sessions()
    settings = _upscale_settings()
    if settings["sr_model"]:
        log.info(
            "Native upscaling enabled: model=%s target=%s hw=%s",
            settings["sr_model"],
            settings["max_resolution"],
            settings["transcode_hw"],
        )
    else:
        log.info("Native upscaling disabled; gateway will use passthrough")
    cleanup_task = asyncio.create_task(_cleanup_upscale_sessions())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task
        ffmpeg_session.shutdown()


app = FastAPI(
    title="neTV Native Player Gateway",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


class GatewayAuthenticator:
    """Rate-limit login failures and cache successful password checks briefly."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}
        self._global_failures: list[float] = []
        self._success_cache: dict[str, float] = {}

    @staticmethod
    def _fingerprint(username: str, password: str) -> str:
        return hashlib.sha256(f"{username}\0{password}".encode()).hexdigest()

    async def verify(self, request: Request, username: str, password: str) -> bool:
        client_ip = request.client.host if request.client else "unknown"
        fingerprint = self._fingerprint(username, password)
        now = time.monotonic()
        with self._lock:
            expires_at = self._success_cache.get(fingerprint, 0)
            if expires_at > now:
                return True
            self._success_cache.pop(fingerprint, None)

            cutoff = now - _LOGIN_WINDOW_SECONDS
            self._global_failures = [
                timestamp
                for timestamp in self._global_failures
                if timestamp > cutoff
            ]
            self._failures = {
                key: [timestamp for timestamp in timestamps if timestamp > cutoff]
                for key, timestamps in self._failures.items()
                if any(timestamp > cutoff for timestamp in timestamps)
            }
            failures = self._failures.get(client_ip, [])
            if (
                len(failures) >= _MAX_LOGIN_FAILURES
                or len(self._global_failures) >= _MAX_GLOBAL_LOGIN_FAILURES
            ):
                raise HTTPException(429, "Too many login attempts, try again later")

        valid = await asyncio.to_thread(auth.verify_password, username, password)
        now = time.monotonic()
        with self._lock:
            if valid:
                self._failures.pop(client_ip, None)
                if len(self._success_cache) >= _MAX_AUTH_CACHE_ENTRIES:
                    self._success_cache = {
                        key: expiry
                        for key, expiry in self._success_cache.items()
                        if expiry > now
                    }
                    if len(self._success_cache) >= _MAX_AUTH_CACHE_ENTRIES:
                        oldest = min(
                            self._success_cache.items(),
                            key=lambda item: item[1],
                        )[0]
                        del self._success_cache[oldest]
                self._success_cache[fingerprint] = now + _AUTH_CACHE_SECONDS
            else:
                if (
                    client_ip not in self._failures
                    and len(self._failures) >= _MAX_FAILURE_CLIENTS
                ):
                    oldest_client = min(
                        self._failures,
                        key=lambda key: self._failures[key][-1],
                    )
                    del self._failures[oldest_client]
                self._failures.setdefault(client_ip, []).append(now)
                self._global_failures.append(now)
        return valid


authenticator = GatewayAuthenticator()


def _public_base_url(request: Request) -> str:
    configured = os.environ.get("NETV_GATEWAY_PUBLIC_URL", "").strip()
    return (configured or str(request.base_url)).rstrip("/")


def _server_info(request: Request, username: str, password: str) -> dict[str, Any]:
    url = urllib.parse.urlparse(_public_base_url(request))
    now = int(time.time())
    return {
        "user_info": {
            "username": username,
            "password": password,
            "message": "neTV native-player gateway",
            "auth": 1,
            "status": "Active",
            "exp_date": None,
            "is_trial": "0",
            "active_cons": "0",
            "created_at": "0",
            "max_connections": "1",
            "allowed_output_formats": ["m3u8"] if _upscale_enabled() else ["ts"],
        },
        "server_info": {
            "url": url.hostname or "localhost",
            "port": str(url.port or (443 if url.scheme == "https" else 80)),
            "https_port": str(url.port or 443) if url.scheme == "https" else "",
            "server_protocol": url.scheme,
            "rtmp_port": "0",
            "timezone": "UTC",
            "timestamp_now": now,
            "time_now": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now)),
        },
    }


def _hidden_sources(username: str) -> set[str]:
    return cache.hidden_source_ids(username, auth.is_admin(username))


def _allowed_category_ids(username: str, snapshot: CatalogSnapshot) -> set[str]:
    unavailable = set(auth.get_user_limits(username).get("unavailable_groups", []))
    category_restricted = any(value.startswith("cat:") for value in unavailable)
    hidden = _hidden_sources(username)
    visible = (
        {
            c
            for stream in _visible_streams(username, snapshot)
            for c in stream.public["category_ids"]
        }
        if hidden
        else None
    )
    allowed: set[str] = set()
    for category in snapshot.categories:
        public_id = str(category["category_id"])
        access_id = snapshot.category_access_ids.get(public_id)
        if access_id is None:
            if not category_restricted:
                allowed.add(public_id)
        elif f"cat:{access_id}" not in unavailable:
            allowed.add(public_id)
    return allowed if visible is None else allowed & visible


def _visible_streams(
    username: str,
    snapshot: CatalogSnapshot,
    category_id: str | None = None,
) -> list[GatewayStream]:
    unavailable = set(auth.get_user_limits(username).get("unavailable_groups", []))
    category_restricted = any(value.startswith("cat:") for value in unavailable)
    hidden = _hidden_sources(username)
    streams = [
        stream
        for stream in snapshot.streams
        if (
            stream.source_id not in hidden
            and (stream.access_group_ids or not category_restricted)
            and not any(
                f"cat:{category}" in unavailable for category in stream.access_group_ids
            )
        )
    ]
    if category_id is not None:
        streams = [
            stream for stream in streams if category_id in stream.public["category_ids"]
        ]
    settings = cache.load_user_settings(username)
    if settings.get("sort_live_by_views", True):
        view_counts = cache.load_live_view_counts(username)
        if view_counts:
            if category_id is None:
                category_order: dict[str, int] = {}
                for stream in streams:
                    primary_category = str(stream.public.get("category_id", ""))
                    category_order.setdefault(primary_category, len(category_order))
                streams.sort(
                    key=lambda stream: (
                        category_order[str(stream.public.get("category_id", ""))],
                        -view_counts.get(
                            cache.live_stream_key(stream.source_id, stream.upstream_id),
                            0,
                        ),
                    )
                )
            else:
                streams.sort(
                    key=lambda stream: -view_counts.get(
                        cache.live_stream_key(stream.source_id, stream.upstream_id),
                        0,
                    )
                )
    return streams


async def _get_catalog() -> CatalogSnapshot:
    return await asyncio.to_thread(catalog.get)


def _auth_failure() -> JSONResponse:
    return JSONResponse(
        {
            "user_info": {
                "auth": 0,
                "status": "Disabled",
                "message": "Invalid username or password",
            }
        }
    )


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    if _upscale_enabled():
        return {"status": "ok", "mode": "upscale"}
    return {"status": "ok"}


@app.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    settings = _upscale_settings()
    upscale = bool(settings["sr_model"])
    return {
        "upscale": upscale,
        "models": _available_upscale_models(),
        "selected_model": settings["sr_model"],
        "max_resolution": settings["max_resolution"],
        "quality": settings["quality"],
        "output_format": "hls" if upscale else "mpegts",
    }


@app.get("/player_api.php")
async def player_api(
    request: Request,
    username: str = Query(""),
    password: str = Query(""),
    action: str | None = None,
    category_id: str | None = None,
    stream_id: int | None = None,
    limit: int = 10,
) -> Any:
    authenticated = await authenticator.verify(request, username, password)
    log.info(
        "Xtream request authenticated=%s action=%s",
        authenticated,
        action or "server_info",
    )
    if not authenticated:
        return _auth_failure()
    if action is None:
        return _server_info(request, username, password)

    snapshot = await _get_catalog()
    allowed_categories = _allowed_category_ids(username, snapshot)
    if action == "get_live_categories":
        return [
            category
            for category in snapshot.categories
            if str(category["category_id"]) in allowed_categories
        ]
    if action == "get_live_streams":
        return [
            stream.public
            for stream in _visible_streams(username, snapshot, category_id)
        ]
    if action == "get_short_epg":
        if stream_id is None:
            raise HTTPException(400, "stream_id is required")
        stream = snapshot.streams_by_id.get(stream_id)
        if stream is None or stream not in _visible_streams(username, snapshot):
            raise HTTPException(404, "Stream not found")
        client = get_xtream_client_by_source(stream.source_id)
        if client is None:
            return {"epg_listings": []}
        try:
            return await asyncio.to_thread(
                client.get_short_epg,
                int(stream.upstream_id),
                max(1, min(limit, 100)),
            )
        except (OSError, TypeError, ValueError, urllib.error.URLError) as exc:
            log.warning("Gateway EPG lookup failed (%s)", type(exc).__name__)
            raise HTTPException(502, "Unable to load upstream EPG data") from exc
    if action in _PASSTHROUGH_ACTIONS:
        return []
    raise HTTPException(400, f"Unsupported action: {action}")


def _m3u_value(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").replace('"', "'")


@app.get("/get.php")
async def playlist(
    request: Request,
    username: str = Query(""),
    password: str = Query(""),
    output: str = "ts",
    type: str = "m3u_plus",
) -> Response:
    if not await authenticator.verify(request, username, password):
        return Response("Invalid username or password\n", status_code=401, media_type="text/plain")
    if type not in ("m3u", "m3u_plus"):
        raise HTTPException(400, "Unsupported playlist type")
    allowed_outputs = ("", "ts", "mpegts", "m3u8", "hls")
    if output not in allowed_outputs:
        raise HTTPException(400, "Unsupported playlist output")
    base_url = _public_base_url(request)
    encoded_user = urllib.parse.quote(username, safe="")
    encoded_password = urllib.parse.quote(password, safe="")
    epg_query = urllib.parse.urlencode({"username": username, "password": password})
    lines = [f'#EXTM3U url-tvg="{base_url}/xmltv.php?{epg_query}"']
    snapshot = await _get_catalog()
    categories = {
        str(category["category_id"]): str(category["category_name"])
        for category in snapshot.categories
    }
    extension = "m3u8" if _upscale_enabled() else "ts"
    for stream in _visible_streams(username, snapshot):
        info = stream.public
        group = categories.get(str(info["category_id"]), "Uncategorized")
        lines.append(
            '#EXTINF:-1 tvg-id="{epg}" tvg-name="{name}" tvg-logo="{logo}" '
            'group-title="{group}",{name}'.format(
                epg=_m3u_value(info["epg_channel_id"]),
                name=_m3u_value(info["name"]),
                logo=_m3u_value(info["stream_icon"]),
                group=_m3u_value(group),
            )
        )
        lines.append(
            f"{base_url}/live/{encoded_user}/{encoded_password}/{stream.local_id}.{extension}"
        )
    return Response("\n".join(lines) + "\n", media_type="audio/x-mpegurl")


def _resolve_upstream(stream: GatewayStream, extension: str) -> str:
    if stream.source_type == "m3u":
        if not stream.direct_url:
            raise HTTPException(502, "Source stream URL is missing")
        return stream.direct_url
    client = get_xtream_client_by_source(stream.source_id)
    if client is None:
        raise HTTPException(404, "Source is no longer configured")
    return client.build_stream_url("live", int(stream.upstream_id), extension)


def _iter_upstream(upstream: Any) -> Iterator[bytes]:
    with contextlib.closing(upstream):
        while chunk := upstream.read(64 * 1024):
            yield chunk


async def _proxy_url(url: str) -> StreamingResponse:
    try:
        upstream = await asyncio.to_thread(
            safe_urlopen,
            url,
            30,
            cache.get_user_agent(),
        )
    except (OSError, urllib.error.URLError) as exc:
        log.warning("Gateway upstream connection failed (%s)", type(exc).__name__)
        raise HTTPException(502, "Unable to connect to upstream stream") from exc
    content_type = upstream.headers.get("Content-Type") or "video/mp2t"
    headers = {"Cache-Control": "no-store"}
    content_length = upstream.headers.get("Content-Length")
    if content_length:
        headers["Content-Length"] = content_length
    content_encoding = upstream.headers.get("Content-Encoding")
    if content_encoding:
        headers["Content-Encoding"] = content_encoding
    return StreamingResponse(_iter_upstream(upstream), media_type=content_type, headers=headers)


@app.get("/live/{stream_path:path}", name="live_stream")
async def live_stream(request: Request, stream_path: str) -> Response:
    try:
        username, remainder = stream_path.split("/", 1)
        password, filename = remainder.rsplit("/", 1)
        stream_id_text, extension = filename.rsplit(".", 1)
        stream_id = int(stream_id_text)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, "Invalid live stream path") from exc
    if not await authenticator.verify(request, username, password):
        raise HTTPException(401, "Invalid username or password")
    if extension not in ("ts", "m3u8"):
        raise HTTPException(400, "Unsupported stream format")
    snapshot = await _get_catalog()
    stream = snapshot.streams_by_id.get(stream_id)
    if stream is None or stream not in _visible_streams(username, snapshot):
        raise HTTPException(404, "Stream not found")
    await asyncio.to_thread(
        cache.record_live_view,
        username,
        stream.source_id,
        stream.upstream_id,
    )
    if _upscale_enabled():
        settings = cache.load_server_settings()
        source = next(
            (
                source
                for source in settings.get("sources", [])
                if source.get("id") == stream.source_id
            ),
            {},
        )
        user_limits = auth.get_user_limits(username)
        result = await ffmpeg_session.start_transcode(
            _resolve_upstream(stream, "ts"),
            content_type="live",
            deinterlace_fallback=bool(source.get("deinterlace_fallback", True)),
            username=username,
            source_id=stream.source_id,
            user_max_streams=user_limits.get("max_streams_per_source", {}).get(
                stream.source_id, 0
            ),
            source_max_streams=int(source.get("max_streams", 0) or 0),
        )
        return RedirectResponse(
            f"/upscale/{result['session_id']}/stream.m3u8",
            status_code=302,
        )
    return await _proxy_url(_resolve_upstream(stream, extension))


@app.get("/upscale/{session_id}/{filename}")
async def upscale_file(session_id: str, filename: str) -> Response:
    safe_filename = Path(filename).name
    if safe_filename != filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    if not ffmpeg_session.touch_session(session_id):
        raise HTTPException(404, "Upscale session not found")
    session = ffmpeg_session.get_session(session_id)
    if not session:
        raise HTTPException(404, "Upscale session not found")
    file_path = Path(session["dir"]) / safe_filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    if safe_filename.endswith(".m3u8"):
        return Response(
            file_path.read_text(),
            media_type="application/vnd.apple.mpegurl",
            headers=headers,
        )
    if safe_filename.endswith(".ts"):
        return FileResponse(file_path, media_type="video/mp2t", headers=headers)
    raise HTTPException(400, "Unsupported upscale file")


@app.get("/xmltv.php")
async def xmltv(
    request: Request,
    username: str = Query(""),
    password: str = Query(""),
) -> Response:
    if not await authenticator.verify(request, username, password):
        raise HTTPException(401, "Invalid username or password")
    snapshot = await _get_catalog()
    streams = _visible_streams(username, snapshot)
    try:
        content = await asyncio.to_thread(
            build_filtered_xmltv,
            cache.CACHE_DIR / "epg.db",
            streams,
        )
    except EpgUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(
        content,
        media_type="application/xml",
        headers={"Cache-Control": "private, max-age=300"},
    )


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="neTV native-player gateway")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("NETV_GATEWAY_PORT", "8100")),
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else getattr(
        logging,
        os.environ.get("LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        access_log=False,
        log_level="debug" if args.debug else "info",
        log_config=None,
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get(
            "NETV_GATEWAY_TRUSTED_PROXIES",
            "127.0.0.1",
        ),
    )
