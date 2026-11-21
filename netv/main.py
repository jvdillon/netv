#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P -m netv.main "$@"
neTV web app: live TV, EPG guide, VOD, and Chromecast for your IPTV sources.

Settings and caches live in $NETV_DATA_DIR when set; otherwise in
~/.local/share/netv (Linux) or ~/Library/Application Support/netv (macOS).

Examples:
  netv                              # HTTP on port 8000
  netv --https                      # HTTPS with auto-detected Let's Encrypt certs
  netv --cert c.pem --key k.pem     # HTTPS with custom certs
'''
# fmt: on

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping
from concurrent import futures
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    override,
)
from urllib import error, parse
from xml.sax.saxutils import escape as xml_escape

import argparse
import asyncio
import contextlib
import functools
import logging
import math
import os
import pathlib
import re
import signal
import subprocess
import threading
import time

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from starlette.types import Scope
from treekle import MutablePlainTree, ReadError, from_plain, json_loads

from netv import (
    auth,
    cache,
    catchup,
    chromecast,
    epg,
    ffmpeg_command,
    ffmpeg_session,
    handoff,
    imdb,
    playlists,
)
from netv.auth import TokenClaims, create_token, verify_password, verify_token
from netv.cache import (
    AVAILABLE_ENCODERS,
    Source,
    UserSettings,
    clamp_live_dvr_mins,
    clear_all_caches,
    clear_all_file_caches,
    get_cache,
    get_cache_lock,
    get_cached_info,
    get_cached_logo,
    get_sources,
    get_watch_position,
    hidden_source_ids,
    load_live_view_counts,
    load_server_settings,
    load_user_settings,
    record_live_view,
    refresh_encoders,
    save_file_cache,
    save_logo,
    save_server_settings,
    save_user_settings,
    save_watch_position,
    update_source_epg_url,
)
from netv.epg import fetch_epg
from netv.fast_start import add_live_start_offset, blocking_master, with_blocking_reload
from netv.m3u import (
    cached_live_data,
    cached_series_data,
    cached_vod_data,
    fetch_m3u,
    fetch_source_live_data,
    fetch_source_vod_data,
    get_first_xtream_client,
    get_refresh_in_progress,
    get_xtream_client_by_source,
    load_all_live_data,
    load_series_data,
    load_vod_data,
)
from netv.playback_policy import PlaybackHealth
from netv.util import configure_logging, safe_urlopen
from netv.xtream import (
    ProviderRecord,
    XtreamClient,
    category_ids,
    list_field,
    live_stream_url,
    record_field,
    text_field,
)


if TYPE_CHECKING:
    import uvicorn
else:
    from wrapt import lazy_import

    uvicorn = lazy_import("uvicorn")


log = logging.getLogger(__name__)

# SSE subscribers for EPG ready notifications (limit to prevent DoS)
_epg_subscribers: set[asyncio.Queue[str]] = set()
_shutdown_event: asyncio.Event | None = None  # Set during shutdown to close SSE.

# Login rate limiting: track failed attempts per IP.
_login_attempts: dict[str, list[float]] = {}


# =============================================================================
# App Setup
# =============================================================================

APP_DIR = pathlib.Path(__file__).parent


def _create_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=APP_DIR / "templates")
    templates.env.auto_reload = True
    templates.env.filters["logo_url"] = _logo_url_filter
    templates.env.globals["get_content_access"] = _get_content_access
    templates.env.globals["get_content_access_from_request"] = _get_content_access_from_request
    return templates


def sr_engine_dir() -> pathlib.Path:
    """Return the TensorRT engine directory, honoring ``SR_ENGINE_DIR``."""
    # The FFmpeg/TensorRT build writes engines to this vendor location, not Netv state.
    return pathlib.Path(os.environ.get("SR_ENGINE_DIR", "~/ffmpeg_build/models")).expanduser()


def get_sr_models(model_name: str = "2x-nomosuni-compact") -> list[str]:
    """Return the model when at least one matching TensorRT engine is installed.

    Args:
        model_name: Super-resolution model whose engines are searched.

    Returns:
        models: The model name when an engine exists, otherwise empty.
    """
    engine_dir = sr_engine_dir()
    if not engine_dir.exists():
        return []
    pattern = f"{model_name}_*p_fp16.engine"
    return [model_name] if next(engine_dir.glob(pattern), None) else []


def is_sr_available() -> bool:
    """Check if AI Upscale is available (at least one TensorRT engine exists)."""
    return len(get_sr_models()) > 0


_epg_fetch_lock = threading.Lock()
_EPG_MIN_LOOKAHEAD = timedelta(hours=6)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Clean up orphaned transcodes and preload data on startup.

    Args:
        app: Application whose startup and shutdown are managed.

    Yields:
        ready: Control while background workers serve the application.
    """
    cache.initialize()
    # Initialize EPG database.
    epg.init(cache.CACHE_DIR)

    # Initialize transcoding module with settings callback.
    ffmpeg_command.init(
        load_server_settings,
        sr_engine_dir=str(sr_engine_dir()) if is_sr_available() else "",
    )

    _kill_orphaned_ffmpeg()
    # Clean up orphaned dirs and recover valid VOD sessions.
    ffmpeg_session.cleanup_and_recover_sessions()
    handoff.load(cache.USERS_DIR)
    handoff.cleanup_orphans()

    def load_vod():
        vod_cats, vod_streams = load_vod_data()
        with get_cache_lock():
            get_cache()["vod_categories"] = vod_cats
            get_cache()["vod_streams"] = vod_streams
        log.info("VOD data loaded")

    def load_series():
        series_cats, series_list = load_series_data()
        with get_cache_lock():
            get_cache()["series_categories"] = series_cats
            get_cache()["series"] = series_list
        log.info("Series data loaded")

    # Start all preloads in parallel (EPG waits for live data internally)
    def load_all():
        _preload_live()
        # Pruning needs the live streams to know how far back archives reach.
        _prune_epg()
        # EPG needs epg_urls from live data, so run after.
        _preload_epg()

    threading.Thread(target=load_all, daemon=True).start()
    threading.Thread(target=load_vod, daemon=True).start()
    threading.Thread(target=load_series, daemon=True).start()
    log.info("Preload started: live+EPG, VOD, series loading in parallel")

    # Periodic cleanup of expired sessions (VOD and live)
    cleanup_stop = threading.Event()
    loop = asyncio.get_running_loop()

    def cleanup_loop():
        while not cleanup_stop.wait(60):  # Check every minute.
            _run_on_loop(loop, ffmpeg_session.cleanup_expired_sessions())

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

    # Hand off channels whose devices stopped reporting, and expire handoffs.
    def handoff_loop():
        while not cleanup_stop.wait(15):
            try:
                handoff.sweep(_handoff_channel, _handoff_recorder(loop))
            except Exception:
                log.exception("Handoff sweep failed")

    threading.Thread(target=handoff_loop, daemon=True).start()

    def imdb_loop():
        while True:
            movies, series = get_cache().get("vod_streams"), get_cache().get("series")
            # A build from half the catalog would stay fresh, missing the rest, for a week.
            loaded = movies is not None and series is not None
            index = cache.CACHE_DIR / "imdb.json"
            if (
                movies is not None
                and series is not None
                and imdb.is_stale(index, max_age_sec=7 * 86_400)
            ):
                try:
                    imdb.build(index, [*movies, *series])
                except Exception:
                    log.exception("IMDb index build failed")
            # Each minute until the catalog loads, then hourly; fresh indexes are skipped.
            if cleanup_stop.wait(3600 if loaded else 60):
                return

    threading.Thread(target=imdb_loop, daemon=True).start()

    # EPG scheduler.
    scheduler_stop = threading.Event()
    _last_triggered: dict[str, str] = {}  # source_id -> last triggered date and time.

    def scheduler_loop():
        while not scheduler_stop.wait(30):  # Check every 30 seconds.
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            # Include the date so a daily time fires again the next day.
            slot = now.strftime("%Y-%m-%d %H:%M")
            for source in get_sources():
                if current_time in source.epg_schedule:
                    key = f"{source.id}_epg"
                    # Only trigger once per scheduled time.
                    if (
                        _last_triggered.get(source.id) != slot
                        and key not in get_refresh_in_progress()
                    ):
                        log.info("Scheduled EPG refresh for %s at %s", source.name, current_time)
                        _last_triggered[source.id] = slot
                        get_refresh_in_progress().add(key)

                        def do_refresh(src: Source = source, k: str = key):
                            try:
                                epg_url = None
                                if src.type == "xtream":
                                    client = XtreamClient(
                                        base_url=src.url,
                                        username=src.username,
                                        password=src.password,
                                    )
                                    epg_url = client.epg_url
                                elif src.type == "m3u":
                                    _, _, epg_url = fetch_m3u(src.url, src.id)
                                elif src.type == "epg":
                                    epg_url = src.url
                                if epg_url is not None and epg_url:
                                    _fetch_all_epg([(epg_url, src.epg_timeout, src.id)])
                                    log.info("Scheduled EPG refresh complete for %s", src.name)
                            except Exception as e:
                                log.error("Scheduled EPG refresh failed for %s: %s", src.name, e)
                            finally:
                                get_refresh_in_progress().discard(k)

                        threading.Thread(target=do_refresh, daemon=True).start()

    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()

    yield

    # Shutdown - signal SSE connections to close.
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    _shutdown_event.set()
    cleanup_stop.set()
    scheduler_stop.set()
    await asyncio.to_thread(chromecast.manager.close)
    await ffmpeg_session.shutdown()


def _run_on_loop(
    loop: asyncio.AbstractEventLoop,
    coroutine: Coroutine[object, object, None],
) -> None:
    """Run a session coroutine from a worker thread on the loop that owns its processes."""
    asyncio.run_coroutine_threadsafe(coroutine, loop).result()


def _handoff_recorder(loop: asyncio.AbstractEventLoop) -> handoff.Recorder:
    """Record handoffs on the loop that owns live sessions and their relays."""

    def record(
        username: str,
        channel: handoff.Channel,
        directory: pathlib.Path,
        seconds: float,
    ) -> handoff.Recording:
        recording = ffmpeg_session.record_live(
            username,
            channel.url,
            channel.source_id,
            str(directory),
            seconds,
        )
        return asyncio.run_coroutine_threadsafe(recording, loop).result()

    return record


def _kill_orphaned_ffmpeg() -> None:
    """Kill transcoders left behind by a previous server process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ffmpeg.*iptv_transcode"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return
    for pid in result.stdout.strip().split("\n"):
        if not pid:
            continue
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            continue
        except Exception:
            return
        log.info("Killed orphaned ffmpeg pid %s", pid)


def _preload_live() -> None:
    """Load live channels into the memory cache."""
    get_refresh_in_progress().add("guide_load")
    try:
        log.info("Preloading live data")
        cats, streams, epg_urls = load_all_live_data()
        with get_cache_lock():
            get_cache()["live_categories"] = cats
            get_cache()["live_streams"] = streams
            get_cache()["epg_urls"] = epg_urls
        log.info("Live data loaded")
    finally:
        get_refresh_in_progress().discard("guide_load")


def _preload_epg() -> None:
    """Load guide listings and notify waiting SSE subscribers."""
    try:
        epg_urls = get_cache().get("epg_urls", [])
        if epg_urls:
            load_all_epg(epg_urls)
            log.info("EPG data loaded: %d programs", epg.get_program_count())
            for q in list(_epg_subscribers):
                with contextlib.suppress(Exception):
                    q.put_nowait("epg_ready")
    except Exception as e:
        log.error("EPG load error: %s", e)


class _RevalidatedStaticFiles(StaticFiles):
    """Static files browsers must revalidate, so an upgrade never runs stale scripts.

    Unversioned URLs otherwise get heuristic freshness of days; revalidation is a 304.
    """

    @override
    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _create_app() -> FastAPI:
    application = FastAPI(title="neTV", lifespan=lifespan)
    application.mount(
        "/static",
        _RevalidatedStaticFiles(directory=APP_DIR / "static"),
        name="static",
    )
    return application


app = _create_app()


class AuthRequired(Exception):
    """Raised when authentication is required."""


@app.exception_handler(AuthRequired)
async def auth_required_handler(request: Request, exc: AuthRequired):
    del exc
    return RedirectResponse("/login", status_code=303)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Render HTTP errors in the format requested by the client.

    Args:
        request: Request whose accepted content types select the response format.
        exc: HTTP error to render.
    """
    # Only handle HTML requests, let API requests get JSON.
    accept = request.headers.get("accept", "")
    if "text/html" not in accept:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    return TEMPLATES.TemplateResponse(
        request,
        "error.html",
        {"title": f"Error {exc.status_code}", "message": exc.detail},
        status_code=exc.status_code,
    )


def get_current_user(request: Request) -> TokenClaims | None:
    token = request.cookies.get("token")
    if token is None or token == "":
        return None
    return verify_token(token)


def require_auth(request: Request) -> TokenClaims:
    user = get_current_user(request)
    if user is None:
        raise AuthRequired
    return user


def require_admin(request: Request) -> TokenClaims:
    user = require_auth(request)
    username = user["sub"]
    if not auth.is_admin(username):
        raise HTTPException(403, "Admin access required")
    return user


# =============================================================================
# Auth Routes
# =============================================================================


@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    """Initial setup page - create first admin user."""
    if not auth.is_setup_required():
        return RedirectResponse("/login", status_code=303)
    return TEMPLATES.TemplateResponse(request, "setup.html", {"error": None})


@app.post("/setup")
async def setup_create_user(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    confirm: Annotated[str, Form()],
):
    """Create the initial admin user.

    Args:
        request: Setup form request.
        username: New administrator name.
        password: New account password.
        confirm: Repeated password for confirmation.
    """
    if not auth.is_setup_required():
        return RedirectResponse("/login", status_code=303)
    # Validate.
    if len(username) < 3:
        return TEMPLATES.TemplateResponse(
            request,
            "setup.html",
            {"error": "Username must be at least 3 characters"},
        )
    if len(password) < 8:
        return TEMPLATES.TemplateResponse(
            request,
            "setup.html",
            {"error": "Password must be at least 8 characters"},
        )
    if password != confirm:
        return TEMPLATES.TemplateResponse(
            request,
            "setup.html",
            {"error": "Passwords do not match"},
        )
    auth.create_user(username, password)
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str | None = None):
    """Render login or redirect to setup when no users exist.

    Args:
        request: Request carrying the remembered username cookie.
        error: Login error indicator to display.
    """
    if auth.is_setup_required():
        return RedirectResponse("/setup", status_code=303)
    last_user = request.cookies.get("last_user", "")
    return TEMPLATES.TemplateResponse(
        request,
        "login.html",
        {"error": error, "last_user": last_user},
    )


@app.post("/login")
async def login(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    """Authenticate user and create session.

    Args:
        request: Login request and client address.
        username: Account name.
        password: Submitted account password.
    """
    ip = request.client.host if request.client else "unknown"
    _check_rate_limit(ip)
    if not verify_password(username, password):
        _login_attempts.setdefault(ip, []).append(time.time())
        return RedirectResponse("/login?error=invalid", status_code=303)
    token = create_token({"sub": username})
    response = RedirectResponse("/", status_code=303)
    is_secure = (
        request.url.scheme == "https"
        or "https" in request.headers.get("x-forwarded-proto", "").lower()
        or "https" in request.headers.get("x-forwarded-scheme", "").lower()
    )
    response.set_cookie(
        "token",
        token,
        httponly=True,
        samesite="strict",
        max_age=86_400 * 7,
        secure=is_secure,
    )
    response.set_cookie("last_user", username, max_age=86_400 * 365, secure=is_secure)
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("token")
    return response


# =============================================================================
# Main Pages
# =============================================================================


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: Annotated[TokenClaims, Depends(require_auth)]):
    del user
    return RedirectResponse("/guide", status_code=303)


def load_all_epg(epg_urls: list[tuple[str, int, str]]) -> None:
    """Load EPG into sqlite database if it is empty or has run out of listings.

    Args:
        epg_urls: List of (url, timeout, source_id) tuples
    """
    if _epg_is_fresh():
        log.info("EPG database has %d programs", epg.get_program_count())
        return

    with _epg_fetch_lock:
        if _epg_is_fresh():
            return
        log.info("EPG data missing or stale, fetching")
        try:
            _fetch_all_epg(epg_urls)
        except Exception as e:
            log.error("EPG fetch failed: %s", e)
            get_cache()["epg_error"] = str(e)


@app.get("/events/epg")
async def epg_events(user: Annotated[TokenClaims, Depends(require_auth)]):
    """Stream a server-sent event when the EPG is ready.

    Args:
        user: Authenticated viewer claims.
    """
    del user
    # Bounds open SSE connections so clients cannot exhaust server resources.
    if len(_epg_subscribers) >= 100:
        raise HTTPException(503, "Too many subscribers")
    queue: asyncio.Queue[str] = asyncio.Queue()
    _epg_subscribers.add(queue)
    return StreamingResponse(_epg_event_stream(queue), media_type="text/event-stream")


@app.get("/guide", response_class=HTMLResponse)
async def guide_page(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
    offset: int = 0,  # Hours offset from now.
    cats: str = "",  # comma-separated category IDs.
):
    username = user["sub"]
    # Check if cats was explicitly in URL (even if empty)
    cats_in_url = "cats" in request.query_params

    # If no channel data in memory, try file cache first (async to avoid blocking)
    if (
        "live_categories" not in get_cache() or "live_streams" not in get_cache()
    ) and not await asyncio.to_thread(_store_cached_live_data):
        # No cache at all - start background load and show loading page.
        _start_guide_background_load()
        return TEMPLATES.TemplateResponse(
            request,
            "guide.html",
            {
                "grid_data": [],
                "selected_cats": [],
                "cats_param": cats,
                "time_markers": [],
                "offset": offset,
                "window_start": "",
                "loading_message": "Loading channel data...",
                "channel_count": 0,
                "loading": True,
            },
        )

    unavailable_groups = _unavailable_groups(username)
    categories = [
        c for c in get_cache().get("live_categories", []) if _live_allowed(c, unavailable_groups)
    ]
    # EPG is optional - check sqlite db for data.
    epg_loading = not epg.has_programs()

    # Sidebar: "All Channels" (shared playlists, then the settings filter),
    # then each playlist and category on its own - same as the Apple apps.
    user_settings = load_user_settings(username)
    saved_filter_list = user_settings.get("guide_filter", [])
    cat_by_id = {str(c["category_id"]): c for c in categories}
    playlist_cats = _playlist_categories()
    default_cats = _default_guide_cats(saved_filter_list, playlist_cats)
    all_streams, _, _ = _get_guide_streams(",".join(default_cats), username)
    counts: dict[str, int] = {}
    for s in all_streams:
        for c in {str(c) for c in list_field(s, "category_ids")}:
            counts[c] = counts.get(c, 0) + 1
    sidebar_playlists = [
        {
            "id": c["category_id"],
            "name": c["category_name"],
            "count": counts.get(c["category_id"], 0),
        }
        for c in playlist_cats
    ]
    sidebar_categories = [
        {"id": cid, "name": cat_by_id[cid]["category_name"], "count": counts.get(cid, 0)}
        for cid in saved_filter_list
        if cid in cat_by_id
    ]
    names: dict[str, MutablePlainTree] = {cid: c["category_name"] for cid, c in cat_by_id.items()}
    names.update({c["category_id"]: c["category_name"] for c in playlist_cats})

    # Selection: URL (?cats=<id>, empty = all) > saved selection > all.
    saved_view = user_settings.get("guide_selected_cats")
    if cats_in_url:
        requested = cats
    else:
        requested = saved_view[0] if saved_view is not None and len(saved_view) == 1 else ""
    selected_id = requested if requested and requested in names else None
    view = None if selected_id is None else [selected_id]
    if saved_view != view and (cats_in_url or (saved_view is not None and len(saved_view) > 0)):
        user_settings["guide_selected_cats"] = view
        save_user_settings(username, user_settings)
    if selected_id is not None:
        effective_cats = selected_id
        streams, ordered_cats, selected_cats = _get_guide_streams(selected_id, username)
    else:
        effective_cats = ",".join(default_cats)
        streams, ordered_cats, selected_cats = all_streams, default_cats, set(default_cats)
    selected_name = "All Channels" if selected_id is None else names[selected_id]
    total_count = len(streams)
    guide_sections = _guide_sections(streams, ordered_cats) if selected_id is None else []

    # Time window: 3 hours starting from offset.
    now = datetime.now(timezone.utc)
    window_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=offset)
    window_end = window_start + timedelta(hours=3)

    # Virtual scrolling: only render first batch, JS fetches rest on scroll
    # When disabled, render all rows server-side.
    virtual_scroll_enabled = user_settings.get("virtual_scroll", True)
    initial_batch_size = 50 if virtual_scroll_enabled else total_count
    grid_data = _build_guide_rows(streams, 0, initial_batch_size, window_start, window_end)

    # Time markers (every 30 min) - convert to local time for display.
    time_markers: list[dict[str, str | float]] = []
    for i in range(7):  # 0, 30, 60, 90, 120, 150, 180 minutes.
        t = window_start + timedelta(minutes=i * 30)
        t_local = t.astimezone()  # Convert to local timezone.
        time_markers.append(
            {
                "label": t_local.strftime("%H:%M"),
                "timestamp": t.timestamp(),
                "left_pct": (i * 30 / 180) * 100,
            },
        )

    # Mobile time markers (2 hour window instead of 3)
    time_markers_mobile: list[dict[str, str | float]] = []
    for i in range(5):  # 0, 30, 60, 90, 120 minutes.
        t = window_start + timedelta(minutes=i * 30)
        t_local = t.astimezone()
        time_markers_mobile.append(
            {
                "label": t_local.strftime("%H:%M"),
                "timestamp": t.timestamp(),
                "left_pct": (i * 30 / 120) * 100,
            },
        )

    return TEMPLATES.TemplateResponse(
        request,
        "guide.html",
        {
            "categories": categories,
            "selected_cats": selected_cats,
            "sidebar_playlists": sidebar_playlists,
            "sidebar_categories": sidebar_categories,
            "all_count": len(all_streams),
            "selected_id": selected_id,
            "selected_name": selected_name,
            "cats_param": "" if selected_id is None else selected_id,
            "effective_cats": effective_cats,  # What's actually being used.
            "grid_data": grid_data,
            "guide_sections": guide_sections,
            "section_by_start": {s["start"]: s for s in guide_sections},
            "time_markers": time_markers,
            "time_markers_mobile": time_markers_mobile,
            "offset": offset,
            "window_start": window_start.strftime("%Y-%m-%d %H:%M"),
            "epg_error": get_cache().get("epg_error"),
            "epg_loading": epg_loading,
            "channel_count": len(grid_data),
            "total_count": total_count,  # For virtual scrolling.
            "virtual_scroll": virtual_scroll_enabled,
            "loading": False,
            "content_access": _get_content_access(username),
        },
    )


@app.get("/api/guide/rows")
async def guide_rows_api(
    user: Annotated[TokenClaims, Depends(require_auth)],
    start: int = Query(default=0, ge=0, description="Starting row index"),
    count: int = Query(default=130, ge=1, le=500, description="Number of rows to fetch"),
    offset: int = Query(default=0, ge=-168, le=168, description="Hours offset from now"),
    cats: str = "",
    exact: bool = Query(default=False, description="Use cats as-is, without adding playlists"),
    direct: bool = Query(
        default=False,
        description="Include provider URLs for direct-play clients",
    ),
):
    """Return guide rows for virtual scrolling.

    Args:
        user: Authenticated viewer claims.
        start: First requested row index.
        count: Maximum number of rows.
        offset: Guide offset in hours.
        cats: Comma-separated category identifiers.
        exact: Use categories without prepending shared playlists.
        direct: Include permission-filtered upstream live URLs.
    """
    username = user.get("sub", "")

    # Use saved filter if no cats provided. Clients that send category ids
    # (like the Apple apps) also get the shared playlists unless exact is set.
    requested = [c.strip() for c in cats.split(",") if c.strip()]
    if not requested:
        user_settings = load_user_settings(username)
        requested = user_settings.get("guide_filter", [])
    if not exact and not any(playlists.is_playlist_id(c) for c in requested):
        requested = _default_guide_cats(requested)
    cats = ",".join(requested)

    # Ensure data is loaded.
    if "live_streams" not in get_cache():
        await asyncio.to_thread(_store_cached_live_data)

    streams, ordered_cat_ids, _ = _get_guide_streams(cats, username)
    total_count = len(streams)

    if total_count == 0:
        empty: GuideRowsResponse = {
            "rows": [],
            "categories": [],
            "total": 0,
            "start": start,
            "direct_playback": direct,
        }
        return JSONResponse(empty, headers={"Cache-Control": "no-store"})

    category_names: dict[str, MutablePlainTree] = {
        str(category["category_id"]): category.get("category_name")
        for category in get_cache().get("live_categories", [])
    }
    category_names.update({c["category_id"]: c["category_name"] for c in _playlist_categories()})
    populated_category_ids = {
        str(category_id) for stream in streams for category_id in list_field(stream, "category_ids")
    }
    guide_categories: list[dict[str, MutablePlainTree]] = [
        {
            "category_id": category_id,
            "category_name": category_names[category_id],
            **({"kind": "playlist"} if playlists.is_playlist_id(category_id) else {}),
        }
        for category_id in ordered_cat_ids
        if category_id in category_names and category_id in populated_category_ids
    ]

    # Time window.
    now = datetime.now(timezone.utc)
    window_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=offset)
    window_end = window_start + timedelta(hours=3)

    rows = _build_guide_rows(streams, start, count, window_start, window_end)
    if direct:
        # Only export URLs from the already permission-filtered guide page.
        # Nothing is proxied or transcoded; the client connects to the provider.
        for row, stream in zip(rows, streams[start : start + count], strict=False):
            url = text_field(stream, "direct_url")
            if not url and stream.get("source_type") == "xtream":
                url = live_stream_url(stream)
            row["channel"]["direct_url"] = url
            # This contract exports live URLs only, not archive playback URLs.
            row["channel"]["catchup_days"] = 0

    page: GuideRowsResponse = {
        "rows": rows,
        "categories": guide_categories,
        "total": total_count,
        "start": start,
        "window_start_timestamp": window_start.timestamp(),
        "direct_playback": direct,
    }
    return JSONResponse(page, headers={"Cache-Control": "no-store"})


@dataclass(kw_only=True, slots=True, frozen=True)
class _Catalog:
    """One browsable catalog: where it is cached and how it renders."""

    group: Literal["movies", "series"]
    newest_field: str
    template: str
    cards_template: str
    cards_url: str


_MOVIES = _Catalog(
    group="movies",
    newest_field="added",
    template="vod.html",
    cards_template="_movie_cards.html",
    cards_url="/api/vod/cards",
)
_SERIES = _Catalog(
    group="series",
    newest_field="last_modified",
    template="series.html",
    cards_template="_series_cards.html",
    cards_url="/api/series/cards",
)


class _BrowseRow(TypedDict):
    category_id: str
    name: str
    total: int
    records: list[ProviderRecord] | None


@app.get("/vod", response_class=HTMLResponse)
async def vod_page(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
    category: int | None = None,
    sort: str | None = None,
):
    # Load from file cache if not in memory (async to avoid blocking)
    if (
        "vod_categories" not in get_cache() or "vod_streams" not in get_cache()
    ) and not await asyncio.to_thread(_store_cached_vod_data):
        # No cache - start background load and show loading page.
        _start_vod_background_load()
        return TEMPLATES.TemplateResponse(
            request,
            "vod.html",
            {"loading": True, "favorites": _page_favorites(load_user_settings(user["sub"]))},
        )
    return _browse_page(request, user.get("sub", ""), _MOVIES, category, sort)


@app.get("/api/vod/cards", response_class=HTMLResponse)
async def vod_cards(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
    category: int,
    sort: str | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=120)] = 120,
):
    if "vod_streams" not in get_cache():
        await asyncio.to_thread(_store_cached_vod_data)
    return _browse_cards(request, user.get("sub", ""), _MOVIES, category, sort, offset, limit)


@app.get("/series", response_class=HTMLResponse)
async def series_page(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
    category: int | None = None,
    sort: str | None = None,
):
    # Load from file cache if not in memory (async to avoid blocking)
    if (
        "series_categories" not in get_cache() or "series" not in get_cache()
    ) and not await asyncio.to_thread(_store_cached_series_data):
        # No cache - start background load and show loading page.
        _start_series_background_load()
        return TEMPLATES.TemplateResponse(
            request,
            "series.html",
            {"loading": True, "favorites": _page_favorites(load_user_settings(user["sub"]))},
        )
    return _browse_page(request, user.get("sub", ""), _SERIES, category, sort)


@app.get("/api/series/cards", response_class=HTMLResponse)
async def series_cards(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
    category: int,
    sort: str | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=120)] = 120,
):
    if "series" not in get_cache():
        await asyncio.to_thread(_store_cached_series_data)
    return _browse_cards(request, user.get("sub", ""), _SERIES, category, sort, offset, limit)


@app.get("/series/{series_id}", response_class=HTMLResponse)
async def series_detail_page(
    request: Request,
    series_id: int,
    user: Annotated[TokenClaims, Depends(require_auth)],
    refresh: bool = False,
):
    username = user["sub"]

    # Check access for this specific series and get source_id.
    source_id = ""
    cached_series = next(
        (s for s in get_cache().get("series", []) if str(s.get("series_id")) == str(series_id)),
        None,
    )
    if cached_series is not None:
        source_id = text_field(cached_series, "source_id")
        unavailable_groups = _unavailable_groups(username)
        if f"series:{source_id}" in unavailable_groups:
            raise HTTPException(403, "Access to this series is restricted")

    # Use the series' source, fall back to first Xtream source.
    xtream = get_xtream_client_by_source(source_id) if source_id else None
    if not xtream:
        xtream = get_first_xtream_client()
    if not xtream:
        raise HTTPException(404, "No Xtream source configured")
    cache_key = f"series_info_{source_id}_{series_id}" if source_id else f"series_info_{series_id}"
    try:
        series_data = await asyncio.to_thread(
            get_cached_info,
            cache_key,
            lambda: xtream.get_series_info(series_id),
            refresh,
        )
    except (error.URLError, TimeoutError) as e:
        return TEMPLATES.TemplateResponse(
            request,
            "error.html",
            {
                "title": "Provider Error",
                "message": f"Failed to load series info: {e}",
            },
            status_code=502,
        )
    if refresh:
        log.info("Force refreshed series info %s", series_id)
    # Extract year from releaseDate if not present.
    info = record_field(series_data, "info")
    release_date = info.get("releaseDate")
    if not _present(info.get("year")) and isinstance(release_date, str) and release_date:
        info["year"] = release_date[:4]
    # Strip redundant series title and episode numbers from episode titles.
    for season_eps in record_field(series_data, "episodes").values():
        if isinstance(season_eps, list):
            for ep in season_eps:
                if isinstance(ep, dict):
                    _clean_episode(ep)

    user_settings = load_user_settings(username)
    return TEMPLATES.TemplateResponse(
        request,
        "series_detail.html",
        {
            "series": series_data,
            "series_id": series_id,
            "favorites": _page_favorites(user_settings),
        },
    )


@app.get("/movie/{stream_id}", response_class=HTMLResponse)
async def movie_detail_page(
    request: Request,
    stream_id: int,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    username = user["sub"]

    # Load from file cache if not in memory.
    if "vod_streams" not in get_cache():
        vod_cats, vod_streams = load_vod_data()
        get_cache()["vod_categories"] = vod_cats
        get_cache()["vod_streams"] = vod_streams

    vod_streams = get_cache().get("vod_streams", [])
    movie = next((m for m in vod_streams if m.get("stream_id") == stream_id), None)
    if movie is not None:
        source_id = text_field(movie, "source_id")
        if f"movies:{source_id}" in _unavailable_groups(username):
            raise HTTPException(403, "Access to this movie is restricted")
        movie = await _with_vod_details(movie, source_id, stream_id)

    user_settings = load_user_settings(username)
    return TEMPLATES.TemplateResponse(
        request,
        "movie_detail.html",
        {
            "movie": movie,
            "favorites": _page_favorites(user_settings),
        },
    )


class CurrentProgram(TypedDict):
    """EPG program airing now on a channel."""

    title: str
    desc: str
    start: float
    end: float


class GuideProgram(TypedDict):
    """One program cell in the guide grid or archive list."""

    title: str
    desc: str
    start: str
    end: str
    start_timestamp: float
    end_timestamp: float
    left_pct: float
    width_pct: float
    catchup: bool
    unavailable: NotRequired[bool]


class CatchupListing(TypedDict):
    """Archive length and the archived programs of one channel."""

    days: int
    programs: list[GuideProgram]


class GuideChannel(TypedDict):
    """Channel header of one guide row."""

    stream_id: MutablePlainTree
    name: MutablePlainTree
    icon: MutablePlainTree
    icon_proxy: str
    epg_id: str
    category_ids: list[str]
    catchup_days: int
    direct_url: NotRequired[str]


class GuideRow(TypedDict):
    """One channel and its programs for the desktop and mobile windows."""

    channel: GuideChannel
    programs: list[GuideProgram]
    programs_mobile: list[GuideProgram]
    index: int


class GuideRowsResponse(TypedDict):
    """One page of guide rows for virtual scrolling and native clients."""

    rows: list[GuideRow]
    categories: list[dict[str, MutablePlainTree]]
    total: int
    start: int
    direct_playback: bool
    window_start_timestamp: NotRequired[float]


class PlaylistChannelView(playlists.PlaylistChannel, total=False):
    """A playlist entry resolved against current streams."""

    icon: str
    category: str
    available: bool
    guide_assignable: bool


class PlaylistPayload(TypedDict):
    """A shared playlist with its channels resolved against current streams."""

    id: str
    category_id: str
    name: str
    channels: list[PlaylistChannelView]


class GuideSection(TypedDict):
    """Guide row index where a selected category's channels begin."""

    start: int
    name: str


class PlaylistCategory(TypedDict):
    """Shared playlist presented as a guide category."""

    category_id: str
    category_name: str
    kind: str


@dataclass(kw_only=True, slots=True, frozen=True)
class PlayerInfo:
    """Info needed to render the player page."""

    url: str = ""
    is_m3u: bool = False
    channel_name: str = ""
    program_title: str = ""
    program_desc: str = ""
    program_start: float = 0.0  # Unix timestamp of current program start (EPG)
    program_end: float = 0.0  # Unix timestamp of current program end (EPG)
    deinterlace_fallback: bool = True  # Used when probe is skipped.
    source_id: str = ""  # Source ID for stream limit tracking.
    category_ids: list[MutablePlainTree] | None = None  # Live stream categories (access check)
    catchup_days: int = 0  # Days of upstream archive for live streams (0 = none)
    catchup_start: float = 0.0  # Unix time the archived stream starts at.
    catchup_seek: float = 0.0  # Seconds into the archived stream to start playing.
    handoff_recording: bool = False  # Plays a handoff recording kept on this server.


@app.get("/play/{stream_type}/{stream_id:path}", response_class=HTMLResponse)
async def player_page(
    request: Request,
    stream_type: str,
    stream_id: str,
    user: Annotated[TokenClaims, Depends(require_auth)],
    ext: str = "",
    series_id: int | None = None,
    start: float | None = Query(default=None, description="Catchup: unix start of a past program"),
    handoff_id: str = Query(default="", alias="handoff", description="Resume this handoff"),
):
    """Render player page for live/movie/series stream.

    Args:
        request: Player request carrying connection information.
        stream_type: Live, movie or series content type.
        stream_id: Source stream identifier.
        user: Authenticated viewer claims.
        ext: Requested container extension.
        series_id: Parent series identifier.
        start: Archive broadcast timestamp to resume.
        handoff_id: Handoff identifier to claim.
    """
    username = user["sub"]
    next_episode_url = None
    if stream_type == "live" and handoff_id:
        info = await asyncio.to_thread(
            _get_handoff_player_info,
            username,
            stream_id,
            handoff_id,
            start,
        )
    elif stream_type == "live" and start is not None:
        info = await asyncio.to_thread(_get_catchup_player_info, stream_id, start)
    elif stream_type == "live":
        info = await asyncio.to_thread(_get_live_player_info, stream_id)
    elif stream_type == "movie":
        info = await asyncio.to_thread(_get_movie_player_info, stream_id, ext)
    elif stream_type == "series":
        info, next_episode_url = await asyncio.to_thread(
            _get_series_player_info,
            stream_id,
            series_id,
            ext,
        )
    else:
        raise HTTPException(404, "Invalid stream type")

    if not info.url:
        raise HTTPException(404, "Stream not found")

    # Playing anything other than the handoff ends it and its recording. A stale
    # handoff link has already failed above, leaving a newer handoff in place.
    await asyncio.to_thread(handoff.cancel, username, handoff_id if stream_type == "live" else "")

    # Check user's group access.
    unavailable_groups = _unavailable_groups(username)
    log.info(
        "Access check: user=%s type=%s source_id=%s unavailable=%s",
        username,
        stream_type,
        info.source_id,
        unavailable_groups,
    )
    if unavailable_groups:
        if stream_type == "live":
            # Live streams: blocked if the source or any category is unavailable.
            live: ProviderRecord = {"source_id": info.source_id, "category_ids": info.category_ids}
            if not _live_allowed(live, unavailable_groups):
                raise HTTPException(403, "Access to this channel is restricted")
        elif stream_type == "movie" and info.source_id:
            if f"movies:{info.source_id}" in unavailable_groups:
                raise HTTPException(403, "Access to movies is restricted")
        elif (
            stream_type == "series"
            and info.source_id
            and f"series:{info.source_id}" in unavailable_groups
        ):
            raise HTTPException(403, "Access to series is restricted")

    log.info("Play %s/%s: %s", stream_type, stream_id, info.url)

    if stream_type == "live":
        await asyncio.to_thread(record_live_view, username, info.source_id, stream_id)

    is_catchup = bool(info.catchup_start)
    server_settings = load_server_settings()
    user_settings = load_user_settings(username)
    transcode_mode = server_settings["transcode_mode"]
    is_https = (
        request.url.scheme == "https"
        or "https" in request.headers.get("x-forwarded-proto", "").lower()
        or "https" in request.headers.get("x-forwarded-scheme", "").lower()
    )
    if info.handoff_recording:
        # Recordings are already packaged for every client.
        transcode_mode = "never"
    elif transcode_mode == "auto":
        needs_transcode = info.is_m3u or is_catchup or ext in ("mkv", "mp4", "avi", "wmv", "flv")
        mixed_content = is_https and info.url.startswith("http://")
        # Pause/rewind requires a server-side buffer; route live through a
        # local HLS session (stream copy when codecs allow) when DVR is on.
        needs_dvr = stream_type == "live" and server_settings["live_dvr_mins"] > 0
        if needs_transcode or mixed_content or needs_dvr:
            transcode_mode = "always"

    # Get saved watch position for VOD (per-user)
    resume_position = 0.0
    if stream_type in ("movie", "series"):
        watch_entry = get_watch_position(username, info.url)
        if watch_entry is not None:
            resume_position = watch_entry["position"]

    # The page starts a live stream it must transcode as soon as its scripts load;
    # starting it now overlaps the tuner's first keyframe with the page load.
    if (
        stream_type == "live"
        and not is_catchup
        and transcode_mode == "always"
        and not chromecast.manager.owns_url(info.url)
    ):
        user_max_streams, source_max_streams = _stream_limits(username, info.source_id)
        ffmpeg_session.prestart_live(
            info.url,
            username=username,
            source_id=info.source_id,
            deinterlace_fallback=info.deinterlace_fallback,
            user_max_streams=user_max_streams,
            source_max_streams=source_max_streams,
        )

    # For series, stream_id is episode_id.
    episode_id = int(stream_id) if stream_type == "series" and stream_id.isdigit() else None
    # Extract series name from channel_name (format: "Series Name (Year)" or just "Series Name")
    series_name = ""
    if stream_type == "series" and info.channel_name:
        # Strip year suffix like " (2020)"
        series_name = re.sub(r"\s*\(\d{4}\)$", "", info.channel_name)

    return TEMPLATES.TemplateResponse(
        request,
        "player.html",
        {
            "raw_url": info.url,
            "raw_url_token": auth.stream_url_token(username, info.url),
            "stream_id": stream_id,
            "transcode_mode": transcode_mode,
            "live_dvr_mins": server_settings["live_dvr_mins"],
            "stream_type": stream_type,
            "channel_name": info.channel_name,
            "program_title": info.program_title,
            "program_desc": info.program_desc,
            "program_start": info.program_start,
            "program_end": info.program_end,
            "captions_enabled": user_settings.get("captions_enabled", False),
            "resume_position": resume_position,
            "series_id": series_id,
            "episode_id": episode_id,
            "series_name": series_name,
            "cc_lang": user_settings.get("cc_lang", ""),
            "audio_lang": user_settings.get("audio_lang", "en"),
            "cc_style": user_settings.get("cc_style", {}),
            "cast_host": user_settings.get("cast_host", ""),
            "next_episode_url": next_episode_url,
            "deinterlace_fallback": info.deinterlace_fallback,
            "source_id": info.source_id,
            "content_access": _get_content_access(username),
            "catchup_days": info.catchup_days,
            "catchup_start": info.catchup_start,
            "catchup_seek": info.catchup_seek,
            "handoff_id": handoff_id if stream_type == "live" else "",
            "handoff_recording": info.handoff_recording,
        },
    )


@app.get("/api/live/program/{stream_id:path}")
async def live_program_api(
    stream_id: str,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    """Return the current EPG program for a live channel.

    Args:
        stream_id: Source stream identifier.
        user: Authenticated viewer claims used for access checks.
    """

    def lookup() -> CurrentProgram | None:
        _ensure_live_cache()
        stream = _find_live_stream(stream_id)
        if stream is None or not _live_stream_filter(user["sub"])(stream):
            raise HTTPException(404, "Stream not found")
        return _get_current_program(_live_epg_id(stream))

    program = await asyncio.to_thread(lookup)
    no_program: CurrentProgram = {"title": "", "desc": "", "start": 0.0, "end": 0.0}
    return JSONResponse(
        no_program if program is None else program,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/live/catchup/{stream_id:path}")
async def live_catchup_api(
    stream_id: str,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    """Return archived programs newest first, including the airing program.

    Args:
        stream_id: Source stream identifier.
        user: Authenticated viewer claims used for access checks.
    """

    def lookup() -> CatchupListing:
        _ensure_live_cache()
        stream = _find_live_stream(stream_id)
        if stream is None or not _live_stream_filter(user["sub"])(stream):
            raise HTTPException(404, "Stream not found")
        days = catchup.archive_days(stream)
        epg_id = _live_epg_id(stream)
        if not days or not epg_id:
            return {"days": days, "programs": []}
        now = datetime.now(timezone.utc)
        programs = epg.get_programs_in_range(
            epg_id,
            now - timedelta(days=days),
            now,
            text_field(stream, "source_id"),
        )
        return {
            "days": days,
            "programs": [
                {
                    "title": p.title,
                    "desc": p.desc,
                    "start": p.start.astimezone().strftime("%H:%M"),
                    "end": p.stop.astimezone().strftime("%H:%M"),
                    "start_timestamp": p.start.timestamp(),
                    "end_timestamp": p.stop.timestamp(),
                    "left_pct": 0,
                    "width_pct": 0,
                    "catchup": True,
                }
                for p in reversed(programs)
                if catchup.can_play(stream, p.start, now)
            ],
        }

    return JSONResponse(await asyncio.to_thread(lookup), headers={"Cache-Control": "no-store"})


@app.get("/search", response_class=HTMLResponse)
async def search_page(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
    q: str = "",
    regex: bool = False,
    live: bool = False,
    vod: bool = False,
    series: bool = False,
    limit: int = 100,
):
    results: dict[str, list[ProviderRecord]] = {"live": [], "vod": [], "series": []}

    # Default all on if none specified.
    if not live and not vod and not series:
        live = vod = series = True

    if q:
        # Limit regex length to prevent ReDoS.
        if regex and len(q) > 100:
            raise HTTPException(400, "Regex pattern too long")
        match_fn = _name_matcher(q, regex=regex)

        # Load live data (run in thread to avoid blocking)
        if live:
            if "live_streams" not in get_cache():
                cats, streams, epg_urls = await asyncio.to_thread(load_all_live_data)
                with get_cache_lock():
                    get_cache()["live_categories"] = cats
                    get_cache()["live_streams"] = streams
                    get_cache()["epg_urls"] = epg_urls
            results["live"] = _matching_names(
                get_cache().get("live_streams", []),
                match_fn,
                limit,
            )

        # Load VOD data (run in thread to avoid blocking)
        if vod:
            if "vod_streams" not in get_cache():
                vod_cats, vod_streams = await asyncio.to_thread(load_vod_data)
                with get_cache_lock():
                    get_cache()["vod_categories"] = vod_cats
                    get_cache()["vod_streams"] = vod_streams
            results["vod"] = _matching_names(get_cache().get("vod_streams", []), match_fn, limit)

        # Load series data (run in thread to avoid blocking)
        if series:
            if "series" not in get_cache():
                series_cats, series_list = await asyncio.to_thread(load_series_data)
                with get_cache_lock():
                    get_cache()["series_categories"] = series_cats
                    get_cache()["series"] = series_list
            results["series"] = _matching_names(get_cache().get("series", []), match_fn, limit)

    username = user["sub"]
    user_settings = load_user_settings(username)

    # Filter results based on user access.
    unavailable_groups = _unavailable_groups(username)

    # Filter live results by category access.
    results["live"] = [s for s in results["live"] if _live_allowed(s, unavailable_groups)]
    # Filter movie results by source access.
    results["vod"] = [
        s for s in results["vod"] if f"movies:{s.get('source_id', '')}" not in unavailable_groups
    ]
    # Filter series results by source access.
    results["series"] = [
        s for s in results["series"] if f"series:{s.get('source_id', '')}" not in unavailable_groups
    ]

    # Apply user's category filters from settings.
    guide_filter = user_settings.get("guide_filter", [])
    if guide_filter:
        guide_filter_set = set(guide_filter)
        results["live"] = [
            s
            for s in results["live"]
            if any(str(cat_id) in guide_filter_set for cat_id in list_field(s, "category_ids"))
        ]

    vod_filter = user_settings.get("vod_filter", [])
    if vod_filter:
        vod_filter_set = set(vod_filter)
        results["vod"] = [s for s in results["vod"] if str(s.get("category_id")) in vod_filter_set]

    series_filter = user_settings.get("series_filter", [])
    if series_filter:
        series_filter_set = set(series_filter)
        results["series"] = [
            s for s in results["series"] if str(s.get("category_id")) in series_filter_set
        ]

    content_access = _get_content_access(username)
    return TEMPLATES.TemplateResponse(
        request,
        "search.html",
        {
            "query": q,
            "results": results,
            "regex": regex,
            "search_live": live,
            "search_vod": vod and content_access["movies"],
            "search_series": series and content_access["series"],
            "limit": limit,
            "favorites": _page_favorites(user_settings),
            "content_access": content_access,
        },
    )


@app.get("/stream/{stream_type}/{stream_id}")
async def stream_redirect(
    stream_type: str,
    stream_id: int,
    user: Annotated[TokenClaims, Depends(require_auth)],
    ext: str = "",
):
    xtream = get_first_xtream_client()
    if not xtream:
        raise HTTPException(404, "No Xtream source configured")
    first_xtream = next((s for s in get_sources() if s.type == "xtream"), None)
    if first_xtream and first_xtream.id in _hidden_sources(user.get("sub", "")):
        raise HTTPException(403, "Access to this source is restricted")
    url = xtream.build_stream_url(stream_type, stream_id, ext)
    return RedirectResponse(url, status_code=302)


@app.get("/playlist.xspf")
async def playlist_xspf(
    user: Annotated[TokenClaims, Depends(require_auth)],
    url: str,
):
    del user
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<playlist xmlns="http://xspf.org/ns/0/" version="1">
  <trackList><track><location>{xml_escape(url)}</location></track></trackList>
</playlist>"""
    return Response(
        content=content,
        media_type="application/xspf+xml",
        headers={"Content-Disposition": "attachment; filename=stream.xspf"},
    )


# =============================================================================
# Transcoding routes (logic in ffmpeg_session.py)
# =============================================================================


@app.get("/transcode/start")
async def transcode_start(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
    url: str,
    url_token: str = "",
    content_type: ffmpeg_session.ContentType = "live",
    series_id: int | None = None,
    episode_id: int | None = None,
    series_name: str = "",
    deinterlace_fallback: str = "1",  # "1" or "0"
    source_id: str = "",
    bandwidth_saver: bool = False,
    audio_passthrough: bool = False,  # `client` decodes AC-3/E-AC-3.
    min_segments: Annotated[int, Query(ge=1, le=2)] = 2,
):
    """Start a transcode session and return its identifier.

    Args:
        request: Request used to detect disconnected playback clients.
        user: Authenticated viewer claims.
        url: Upstream media URL.
        url_token: ``stream_url_token`` for ``url``, from the player page.
        content_type: Live, movie or series content type.
        series_id: Parent series identifier.
        episode_id: Episode identifier for probe caching.
        series_name: Series display name.
        deinterlace_fallback: Whether deinterlacing is enabled when probing is skipped.
        source_id: Source used for access checks and stream limits.
        bandwidth_saver: Request the lower-bandwidth rendition.
        audio_passthrough: Whether the client decodes Dolby audio.
        min_segments: Live segments to wait for: 1 for a client that holds its own
            buffer before playing (the web player), else 2.
    """
    username = user.get("sub", "")
    # The server opens this URL, so it must be one a player page issued to this
    # user: otherwise any account could point ffmpeg at internal hosts or files.
    if not auth.verify_stream_url_token(username, url, url_token):
        raise HTTPException(403, "This stream link is not valid. Reload the player page.")
    if chromecast.manager.owns_url(url):
        raise HTTPException(
            409,
            "This stream is playing on a TV. Stop casting before starting it here.",
        )
    deinterlace_fb = deinterlace_fallback == "1"

    hidden = _hidden_sources(username)
    if hidden and (
        source_id in hidden
        or any(
            s.id in hidden and s.type == "xtream" and _is_xtream_source_url(url, s)
            for s in get_sources()
        )
    ):
        raise HTTPException(403, "Access to this source is restricted")

    user_max_streams, source_max_streams = _stream_limits(username, source_id)
    result = await ffmpeg_session.start_transcode(
        url,
        content_type,
        series_id,
        episode_id,
        series_name,
        deinterlace_fb,
        username,
        source_id,
        user_max_streams,
        source_max_streams,
        bandwidth_saver=bandwidth_saver,
        is_disconnected=request.is_disconnected,
        audio_passthrough=audio_passthrough,
        min_segments=min_segments,
    )
    if content_type == "live" and await request.is_disconnected():
        # A reload disconnects the old page's request: end only its viewing.
        await ffmpeg_session.stop_session(
            result["session_id"],
            force=True,
            viewer=result.get("viewer_id", ""),
        )
        raise HTTPException(499, "Playback request disconnected")
    return result


@app.post("/transcode/{session_id}/health")
async def transcode_health(
    session_id: str,
    health: PlaybackHealth,
    user: Annotated[TokenClaims, Depends(require_auth)],
    viewer: str = "",
):
    decision = ffmpeg_session.report_playback_health(session_id, user.get("sub", ""), health)
    ffmpeg_session.note_viewer(session_id, viewer)
    return decision


@app.get("/transcode/seek/{session_id}")
async def transcode_seek(
    session_id: str,
    time: float,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    """Seek VOD transcode to a new position."""
    del user
    if chromecast.manager.owns_session(session_id):
        raise HTTPException(409, "Stop casting before changing the transcode position.")
    return await ffmpeg_session.seek_transcode(session_id, time)


@app.get("/transcode/progress/{session_id}")
async def transcode_progress(
    session_id: str,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    """Get transcode progress (segment count, duration)."""
    del user
    progress = ffmpeg_session.get_session_progress(session_id)
    if progress is None:
        raise HTTPException(404, "Session not found")
    return progress


@app.get("/transcode/{session_id}/{filename}")
async def transcode_file(
    request: Request,
    session_id: str,
    filename: str,
    blocking: bool = False,
    hls_msn: Annotated[int | None, Query(alias="_HLS_msn")] = None,
):
    """Serve HLS playlists or segments using unguessable session identifiers.

    Args:
        request: Request used for receiver diagnostics.
        session_id: Active playback session identifier.
        filename: Playlist, subtitle or segment filename.
        blocking: Serve a live session's playlists for blocking reload (the web
            player); other clients poll the plain ones.
        hls_msn: Segment a blocking reload waits for before the playlist is sent.
    """
    # Prevent path traversal.
    safe_filename = pathlib.Path(filename).name
    if safe_filename != filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")

    session = ffmpeg_session.get_session(session_id)
    if session is None:
        log.debug("[CAST] 404 session not found: %s", session_id)
        raise HTTPException(404, "Transcode session not found")

    file_path = pathlib.Path(session["dir"]) / safe_filename
    if not file_path.exists():
        log.debug("[CAST] 404 file not found: %s", file_path)
        raise HTTPException(404, "File not found")

    ffmpeg_session.touch_session(session_id)
    ffmpeg_session.note_segment_served(session_id, safe_filename)

    # Log Chromecast requests.
    ua = request.headers.get("user-agent", "")
    if "CrKey" in ua or "Chromecast" in ua.lower() or "cast" in ua.lower():
        log.debug("[CAST] Chromecast request: %s UA=%s", filename, ua[:80])

    cors = {"Access-Control-Allow-Origin": "*"}
    if filename.endswith(".m3u8"):
        fast_start = session.get("fast_start", False)
        if fast_start and filename == "low.m3u8" and blocking and hls_msn is not None:
            await ffmpeg_session.wait_for_segment(session_id, hls_msn)
        content = file_path.read_text()
        if fast_start and filename == "low.m3u8":
            content = add_live_start_offset(
                ffmpeg_session.dated_fast_playlist(
                    session_id,
                    ffmpeg_session.live_playlist(session_id) or content,
                ),
                float(session.get("playback_buffer_seconds", 12.0)),
            )
            if blocking:
                content = with_blocking_reload(content)
        elif fast_start and blocking:
            content = blocking_master(content)
        return Response(
            content=content,
            media_type="application/vnd.apple.mpegurl",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", **cors},
        )
    if filename.endswith(".vtt"):
        content = file_path.read_text()
        if re.fullmatch(r"sub\d+_\d+\.vtt", filename):
            content = ffmpeg_session.add_webvtt_timestamp_map(
                content,
                ffmpeg_session.get_caption_timestamp_origin(session_id),
            )
        return Response(content=content, media_type="text/vtt", headers=cors)
    return FileResponse(file_path, media_type="video/mp2t", headers=cors)


@app.get("/subs/{session_id}/{filename}")
async def subtitle_file(session_id: str, filename: str):
    """Serve VTT subtitle files using unguessable session identifiers.

    Args:
        session_id: Active playback session identifier.
        filename: Subtitle filename within the session directory.
    """
    # Prevent path traversal.
    safe_filename = pathlib.Path(filename).name
    if safe_filename != filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    if not safe_filename.endswith(".vtt"):
        raise HTTPException(400, "Only VTT files allowed")
    session = ffmpeg_session.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    file_path = pathlib.Path(session["dir"]) / safe_filename
    # A file without cues yet is served empty at once: the player polls every 0.5 s.
    try:
        content = file_path.read_text() if file_path.exists() else "WEBVTT\n\n"
    except (UnicodeDecodeError, OSError):
        # File may be partially written or corrupted.
        content = "WEBVTT\n\n"
    return Response(
        content=content,
        media_type="text/vtt",
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.delete("/transcode/{session_id}")
async def transcode_stop(
    session_id: str,
    user: Annotated[TokenClaims, Depends(require_auth)],
    force: bool = False,
    viewer: str = "",
):
    """Stop a transcode session while retaining reusable VOD caches.

    Args:
        session_id: Playback session identifier.
        user: Authenticated viewer claims used to verify forced-stop ownership.
        force: Remove the session and its cache immediately.
        viewer: The leaving page's ``viewer_id``; a live session other pages
            still watch keeps playing.
    """
    if chromecast.manager.owns_session(session_id):
        return {"status": "casting"}
    session = ffmpeg_session.get_session(session_id)
    force = force or (session is not None and session.get("is_archive", False))
    if force and session is not None and session.get("username") != user["sub"]:
        raise HTTPException(404, "Session not found")
    await ffmpeg_session.stop_session(session_id, force=force, viewer=viewer)
    return {"status": "stopped"}


@app.post("/transcode/{session_id}/stop")
async def transcode_stop_post(
    session_id: str,
    user: Annotated[TokenClaims, Depends(require_auth)],
    force: bool = False,
    viewer: str = "",
):
    """Stop a transcode session (POST for sendBeacon, VOD cached)."""
    return await transcode_stop(session_id, user, force=force, viewer=viewer)


@app.delete("/transcode-clear")
async def transcode_clear(
    url: str,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    """Force-delete any cached transcode session for a URL.

    Args:
        url: Upstream URL identifying the cached session.
        user: Authenticated viewer claims.
    """
    del user
    if chromecast.manager.owns_url(url):
        raise HTTPException(409, "Stop casting before restarting this transcode.")
    session_id = ffmpeg_session.clear_url_session(url)
    if session_id is not None:
        await ffmpeg_session.stop_session(session_id, force=True)
        log.info("Force-cleared transcode session %s for URL", session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: Annotated[TokenClaims, Depends(require_auth)]):
    username = user.get("sub", "")
    is_admin = auth.is_admin(username)
    server_settings = load_server_settings()
    sr_model_name = "2x-nomosuni-compact"
    sr_models = get_sr_models(sr_model_name)
    sr_model = (
        sr_model_name if server_settings.get("sr_model", "") and sr_model_name in sr_models else ""
    )
    user_settings = load_user_settings(username)
    # Load categories (from file cache or trigger background load)
    if "live_categories" not in get_cache() and not await asyncio.to_thread(
        _store_cached_live_data,
    ):
        # No cache - start background load.
        _start_guide_background_load()

    # Load VOD categories if not cached.
    if "vod_categories" not in get_cache():
        await asyncio.to_thread(_store_cached_vod_data)

    # Load series categories if not cached.
    if "series_categories" not in get_cache():
        await asyncio.to_thread(_store_cached_series_data)

    # Build source_id -> source_name mapping.
    source_names = {s["id"]: s["name"] for s in server_settings["sources"]}

    # Filter categories based on user's unavailable groups.
    unavailable_groups = _unavailable_groups(username)
    all_live_cats = get_cache().get("live_categories", [])
    live_categories = [cat for cat in all_live_cats if _live_allowed(cat, unavailable_groups)]
    vod_categories = [
        c
        for c in get_cache().get("vod_categories", [])
        if f"movies:{c.get('source_id', '')}" not in unavailable_groups
    ]
    series_categories = [
        c
        for c in get_cache().get("series_categories", [])
        if f"series:{c.get('source_id', '')}" not in unavailable_groups
    ]

    return TEMPLATES.TemplateResponse(
        request,
        "settings.html",
        {
            # Server settings.
            "sources": server_settings["sources"],
            "transcode_mode": server_settings["transcode_mode"],
            "transcode_hw": server_settings["transcode_hw"],
            "max_resolution": server_settings.get("max_resolution", "1080p"),
            "quality": server_settings.get("quality", "high"),
            "vod_transcode_cache_mins": server_settings["vod_transcode_cache_mins"],
            "live_dvr_mins": server_settings["live_dvr_mins"],
            "transcode_dir": server_settings["transcode_dir"],
            "probe_live": server_settings["probe_live"],
            "probe_movies": server_settings["probe_movies"],
            "probe_series": server_settings["probe_series"],
            "user_agent_preset": server_settings["user_agent_preset"],
            "user_agent_custom": server_settings["user_agent_custom"],
            "available_encoders": AVAILABLE_ENCODERS,
            "sr_available": is_sr_available(),
            "sr_models": sr_models,
            "sr_model": sr_model,
            "sr_model_name": sr_model_name,
            "all_users": auth.get_users_with_admin(),
            "all_groups": _build_all_groups(),
            "current_user": username,
            "is_admin": is_admin,
            # User settings.
            "captions_enabled": user_settings.get("captions_enabled", False),
            "virtual_scroll": user_settings.get("virtual_scroll", True),
            "sort_live_by_views": user_settings.get("sort_live_by_views", True),
            "live_categories": live_categories,
            "vod_categories": vod_categories,
            "series_categories": series_categories,
            "source_names": source_names,
            "selected_cats": user_settings.get("guide_filter", []),
            "selected_vod_cats": user_settings.get("vod_filter", []),
            "selected_series_cats": user_settings.get("series_filter", []),
            "cc_lang": user_settings.get("cc_lang", ""),
            "audio_lang": user_settings.get("audio_lang", "en"),
            "cc_style": user_settings.get("cc_style", {}),
            "cast_host": user_settings.get("cast_host", ""),
            "content_access": _get_content_access(username),
        },
    )


@app.post("/settings/guide-filter")
async def settings_guide_filter(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    return await _save_category_filter(request, user, "guide_filter")


@app.post("/settings/vod-filter")
async def settings_vod_filter(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    return await _save_category_filter(request, user, "vod_filter")


@app.post("/settings/series-filter")
async def settings_series_filter(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    return await _save_category_filter(request, user, "series_filter")


@app.post("/settings/add")
async def settings_add_source(
    user: Annotated[TokenClaims, Depends(require_admin)],
    name: Annotated[str, Form()],
    source_type: Annotated[str, Form()],
    url: Annotated[str, Form()],
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    epg_timeout: Annotated[int, Form()] = 120,
    epg_schedule: Annotated[str, Form()] = "",
    epg_enabled: Annotated[str, Form()] = "",  # Checkbox: "on" if checked.
    deinterlace_fallback: Annotated[str, Form()] = "",  # Checkbox: "on" if checked.
    max_streams: Annotated[int, Form()] = 0,
    access: Annotated[str, Form()] = "all",  # "all" or "selected"
    users: Annotated[list[str] | None, Form()] = None,
):
    # Validate inputs.
    del user
    if not name or not name.strip():
        raise HTTPException(400, "Name is required")
    if source_type not in ("xtream", "m3u", "epg"):
        raise HTTPException(400, "Invalid source type")
    parsed_url = parse.urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise HTTPException(400, "URL must use http or https")
    if len(name) > 200:
        raise HTTPException(400, "Name too long")

    # Parse schedule times.
    schedule_list: list[str] = []
    for t in epg_schedule.split(","):
        t = t.strip()
        if t and re.match(r"^\d{1,2}:\d{2}$", t):
            schedule_list.append(t.zfill(5))

    source_users = _source_users(access, users)
    settings = load_server_settings()
    sources = settings["sources"]
    source_id = f"src_{int(time.time())}_{len(sources)}"
    sources.append(
        {
            "id": source_id,
            "name": name,
            "type": source_type,
            "url": url.rstrip("/"),
            "username": username,
            "password": password,
            "epg_timeout": max(1, min(3600, epg_timeout)),
            "epg_schedule": schedule_list,
            "epg_enabled": epg_enabled == "on" or source_type == "epg",
            "deinterlace_fallback": deinterlace_fallback == "on",
            "max_streams": max(0, max_streams),
            "users": source_users,
        },
    )
    settings["sources"] = sources
    save_server_settings(settings)
    clear_all_caches()
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/edit/{source_id}")
async def settings_edit_source(
    source_id: str,
    user: Annotated[TokenClaims, Depends(require_admin)],
    name: Annotated[str, Form()],
    source_type: Annotated[str, Form()],
    url: Annotated[str, Form()],
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    epg_timeout: Annotated[int, Form()] = 120,
    epg_schedule: Annotated[str, Form()] = "",
    epg_enabled: Annotated[str, Form()] = "",  # Checkbox: "on" if checked.
    epg_url: Annotated[str, Form()] = "",
    deinterlace_fallback: Annotated[str, Form()] = "",  # Checkbox: "on" if checked.
    max_streams: Annotated[int, Form()] = 0,
    access: Annotated[str, Form()] = "all",  # "all" or "selected"
    users: Annotated[list[str] | None, Form()] = None,
):
    # Validate inputs.
    del user
    if not name or not name.strip():
        raise HTTPException(400, "Name is required")
    if source_type not in ("xtream", "m3u", "epg"):
        raise HTTPException(400, "Invalid source type")
    parsed_url = parse.urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise HTTPException(400, "URL must use http or https")
    if len(name) > 200:
        raise HTTPException(400, "Name too long")

    # Parse schedule times (comma-separated HH:MM)
    schedule_list: list[str] = []
    for t in epg_schedule.split(","):
        t = t.strip()
        if t and re.match(r"^\d{1,2}:\d{2}$", t):
            schedule_list.append(t.zfill(5))  # Normalize to HH:MM.

    source_users = _source_users(access, users)
    settings = load_server_settings()
    for s in settings["sources"]:
        if s["id"] == source_id:
            s["name"] = name
            s["type"] = source_type
            s["url"] = url.rstrip("/")
            s["username"] = username
            s["password"] = password
            s["epg_timeout"] = max(1, min(3600, epg_timeout))
            s["epg_schedule"] = schedule_list
            s["epg_enabled"] = epg_enabled == "on" or source_type == "epg"
            s["epg_url"] = epg_url.strip()
            s["deinterlace_fallback"] = deinterlace_fallback == "on"
            s["max_streams"] = max(0, max_streams)
            s["users"] = source_users
            break
    save_server_settings(settings)
    clear_all_caches()
    return {"ok": True}


@app.post("/settings/delete/{source_id}")
async def settings_delete_source(
    source_id: str,
    user: Annotated[TokenClaims, Depends(require_admin)],
):
    del user
    settings = load_server_settings()
    settings["sources"] = [s for s in settings["sources"] if s["id"] != source_id]
    save_server_settings(settings)
    # Clear all caches including EPG data for this source.
    epg.clear_source(source_id)
    clear_all_file_caches()
    return RedirectResponse("/settings", status_code=303)


@app.get("/guide/refresh")
async def guide_refresh(user: Annotated[TokenClaims, Depends(require_auth)]):
    """Refresh guide data in background while serving cached results.

    Args:
        user: Authenticated viewer claims.
    """

    del user

    def refresh_live():
        try:
            log.info("Live refresh: fetching channels")
            cats, streams, epg_urls = load_all_live_data()
            with get_cache_lock():
                get_cache()["live_categories"] = cats
                get_cache()["live_streams"] = streams
                get_cache()["epg_urls"] = epg_urls
            save_file_cache("live_data", {"cats": cats, "streams": streams, "epg_urls": epg_urls})
            log.info("Live refresh: complete (%d categories, %d streams)", len(cats), len(streams))
        except Exception as e:
            log.error("Live refresh failed: %s", e)
        finally:
            get_refresh_in_progress().discard("live_refresh")

    def refresh_epg():
        try:
            epg_urls = get_cache().get("epg_urls", [])
            if epg_urls:
                log.info("EPG refresh: fetching %d sources", len(epg_urls))
                epg.clear()
                count = _fetch_all_epg(epg_urls)
                with get_cache_lock():
                    get_cache().pop("epg_error", None)
                log.info("EPG refresh: complete (%d programs)", count)
            else:
                log.warning("EPG refresh: no EPG URLs available")
        except Exception as e:
            log.error("EPG refresh failed: %s", e)
            with get_cache_lock():
                get_cache()["epg_error"] = str(e)
        finally:
            get_refresh_in_progress().discard("epg_refresh")

    # Set flags before starting threads to avoid race with status polling.
    if "live_refresh" not in get_refresh_in_progress():
        get_refresh_in_progress().add("live_refresh")
        threading.Thread(target=refresh_live, daemon=True).start()
    if "epg_refresh" not in get_refresh_in_progress():
        get_refresh_in_progress().add("epg_refresh")
        threading.Thread(target=refresh_epg, daemon=True).start()
    return RedirectResponse("/guide?refreshing=1", status_code=303)


@app.get("/guide/refresh-status")
async def guide_refresh_status(user: Annotated[TokenClaims, Depends(require_auth)]):
    """Return refresh status for polling."""
    del user
    return {
        "live": "live_refresh" in get_refresh_in_progress(),
        "epg": "epg_refresh" in get_refresh_in_progress(),
    }


@app.post("/settings/refresh/{source_id}/{refresh_type}")
async def settings_refresh_source(
    source_id: str,
    refresh_type: str,
    user: Annotated[TokenClaims, Depends(require_admin)],
):
    """Refresh a specific data type for a single source.

    Args:
        source_id: Configured source identifier.
        refresh_type: Live, EPG, VOD or M3U data to refresh.
        user: Authenticated administrator claims.
    """
    del user
    sources = get_sources()
    source = next((s for s in sources if s.id == source_id), None)
    if not source:
        return {"error": "Source not found"}

    key = f"{source_id}_{refresh_type}"
    if key in get_refresh_in_progress():
        return {"status": "already_running"}

    get_refresh_in_progress().add(key)

    def do_refresh():
        try:
            if refresh_type == "live":
                log.info("Refreshing live data for source: %s", source.name)
                cats, streams, epg_url, timeout = fetch_source_live_data(source)
                # Update cache by replacing this source's data.
                with get_cache_lock():
                    existing_cats = [
                        c
                        for c in get_cache().get("live_categories", [])
                        if c.get("source_id") != source_id
                    ]
                    existing_streams = [
                        s
                        for s in get_cache().get("live_streams", [])
                        if s.get("source_id") != source_id
                    ]
                    existing_epg = [e for e in get_cache().get("epg_urls", []) if e[2] != source_id]
                    new_cats = existing_cats + cats
                    new_streams = existing_streams + streams
                    new_epg = existing_epg + (
                        [] if epg_url is None or epg_url == "" else [(epg_url, timeout, source_id)]
                    )
                    get_cache()["live_categories"] = new_cats
                    get_cache()["live_streams"] = new_streams
                    get_cache()["epg_urls"] = new_epg
                # Save to file cache.
                save_file_cache(
                    "live_data",
                    {"cats": new_cats, "streams": new_streams, "epg_urls": new_epg},
                )
                log.info(
                    "Live refresh complete for %s: %d cats, %d streams",
                    source.name,
                    len(cats),
                    len(streams),
                )

            elif refresh_type == "epg":
                log.info(
                    "Refreshing EPG for source: %s (timeout=%ds)",
                    source.name,
                    source.epg_timeout,
                )
                epg_url = source.epg_url or (source.url if source.type == "epg" else "")
                if epg_url:
                    epg.clear_source(source_id)
                    count = _fetch_all_epg([(epg_url, source.epg_timeout, source_id)])
                    log.info("EPG refresh complete for %s: %d programs", source.name, count)
                else:
                    log.warning("No EPG URL for source: %s", source.name)

            elif refresh_type == "vod" and source.type == "xtream":
                log.info("Refreshing VOD for source: %s", source.name)
                new_cats, new_streams = fetch_source_vod_data(source)
                # Merge with existing data from other sources.
                existing_cats, existing_streams = load_vod_data()
                # Remove old data from this source, keep others.
                merged_cats = [c for c in existing_cats if c.get("source_id") != source_id]
                merged_streams = [s for s in existing_streams if s.get("source_id") != source_id]
                # Add new data from this source.
                merged_cats.extend(new_cats)
                merged_streams.extend(new_streams)
                with get_cache_lock():
                    get_cache().pop("vod_categories", None)
                    get_cache().pop("vod_streams", None)
                for f in cache.CACHE_DIR.glob("vod_data*.json"):
                    f.unlink(missing_ok=True)
                save_file_cache("vod_data", {"cats": merged_cats, "streams": merged_streams})
                log.info(
                    "VOD refresh complete for %s: %d cats, %d streams (total: %d)",
                    source.name,
                    len(new_cats),
                    len(new_streams),
                    len(merged_streams),
                )

            elif refresh_type == "m3u" and source.type == "m3u":
                log.info("Refreshing M3U playlist for source: %s", source.name)
                cats, streams, detected_epg_url = fetch_m3u(source.url, source.id)
                update_source_epg_url(source_id, detected_epg_url)
                with get_cache_lock():
                    existing_cats = [
                        c
                        for c in get_cache().get("live_categories", [])
                        if c.get("source_id") != source_id
                    ]
                    existing_streams = [
                        s
                        for s in get_cache().get("live_streams", [])
                        if s.get("source_id") != source_id
                    ]
                    new_cats = existing_cats + cats
                    new_streams = existing_streams + streams
                    get_cache()["live_categories"] = new_cats
                    get_cache()["live_streams"] = new_streams
                    epg_urls = get_cache().get("epg_urls", [])
                save_file_cache(
                    "live_data",
                    {
                        "cats": new_cats,
                        "streams": new_streams,
                        "epg_urls": epg_urls,
                    },
                )
                log.info(
                    "M3U refresh complete for %s: %d cats, %d streams",
                    source.name,
                    len(cats),
                    len(streams),
                )

        except Exception as e:
            log.error("Source refresh failed (%s/%s): %s", source.name, refresh_type, e)
        finally:
            get_refresh_in_progress().discard(key)

    threading.Thread(target=do_refresh, daemon=True).start()
    return {"status": "started", "key": key}


@app.get("/settings/refresh-status")
async def settings_refresh_status(user: Annotated[TokenClaims, Depends(require_auth)]):
    """Return per-source refresh status.

    Args:
        user: Authenticated viewer claims.
    """
    del user
    statuses: dict[str, dict[str, bool]] = {}
    for key in list(get_refresh_in_progress()):
        if "_" in key:
            # Format: source_id_type (e.g., "ota_epg" or "src_123_epg")
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                source_id, rtype = parts
                statuses.setdefault(source_id, {})[rtype] = True
    # Report global guide_load as affecting all sources.
    if "guide_load" in get_refresh_in_progress():
        statuses["_global"] = {"live": True, "epg": True}
    return statuses


@app.post("/settings/captions")
async def settings_captions(
    user: Annotated[TokenClaims, Depends(require_auth)],
    enabled: Annotated[str, Form()] = "",
):
    username = user.get("sub", "")
    user_settings = load_user_settings(username)
    user_settings["captions_enabled"] = enabled == "on"
    save_user_settings(username, user_settings)
    return {"ok": True}


class CastStart(BaseModel):
    host: str = Field(min_length=1, max_length=45)
    session_id: str = Field(min_length=1, max_length=128)
    server_url: str = Field(default="", max_length=512)
    title: str = Field(default="neTV", max_length=256)
    current_time: float = Field(default=0, ge=0, le=86_400_000, allow_inf_nan=False)


class CastCommand(BaseModel):
    action: Literal["play", "pause", "stop", "volume"]
    volume: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


def require_cast_auth(request: Request) -> TokenClaims:
    user = get_current_user(request)
    if user is None:
        raise HTTPException(401, "Sign in to control Chromecast devices.")
    return user


@app.get("/api/cast/devices")
def cast_devices(user: Annotated[TokenClaims, Depends(require_cast_auth)]):
    del user
    try:
        return {"devices": chromecast.manager.discover()}
    except (OSError, chromecast.PyChromecastError) as exc:
        log.warning("Chromecast discovery failed: %s", type(exc).__name__)
        raise HTTPException(
            503,
            "Discovery failed. Check LAN access or enter the TV's IP manually.",
        ) from exc


@app.get("/api/cast/status")
def cast_status(user: Annotated[TokenClaims, Depends(require_cast_auth)]):
    return chromecast.manager.status(user.get("sub", ""))


@app.post("/api/cast/start")
def cast_start(
    data: CastStart,
    request: Request,
    user: Annotated[TokenClaims, Depends(require_cast_auth)],
):
    username = user.get("sub", "")
    settings = load_user_settings(username)
    address = data.server_url.strip() or settings.get("cast_host", "") or str(request.base_url)
    if "://" not in address:
        address = f"{request.url.scheme}://{address}"
    origin = chromecast.media_origin(address)
    result = chromecast.manager.start(
        username,
        data.host,
        data.session_id,
        origin,
        data.title,
        data.current_time,
    )
    if data.server_url:
        settings = load_user_settings(username)
        settings["cast_host"] = origin
        save_user_settings(username, settings)
    return result


@app.post("/api/cast/control")
async def cast_control(data: CastCommand, user: Annotated[TokenClaims, Depends(require_cast_auth)]):
    loop = asyncio.get_running_loop()
    return await asyncio.to_thread(
        chromecast.manager.command,
        user.get("sub", ""),
        data.action,
        data.volume,
        release=lambda session_id: _run_on_loop(
            loop,
            ffmpeg_session.stop_session(session_id, force=True),
        ),
    )


@app.get("/api/user-prefs")
async def get_user_prefs(user: Annotated[TokenClaims, Depends(require_auth)]):
    """Get user preferences.

    Args:
        user: Authenticated viewer claims identifying the settings owner.
    """
    username = user.get("sub", "")
    settings = load_user_settings(username)
    return {
        "favorites": _page_favorites(settings),
        "cc_lang": settings.get("cc_lang", ""),
        "audio_lang": settings.get("audio_lang", "en"),
        "cc_style": settings.get("cc_style", {}),
        "cast_host": settings.get("cast_host", ""),
        "virtual_scroll": settings.get("virtual_scroll", True),
        "sort_live_by_views": settings.get("sort_live_by_views", True),
    }


@app.post("/api/favorites")
async def set_favorite(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    """Add or remove one favorite, leaving the others as stored.

    Pages send one change rather than their whole list, so a page loaded before another
    change (another tab, or one restored by Back) cannot overwrite it.

    Args:
        request: Request carrying the ``_FavoriteUpdate`` body.
        user: Authenticated viewer claims identifying the favorites owner.
    """
    body = await request.body()
    if len(body) > 64 * 1024:
        raise HTTPException(400, "Request too large")
    try:
        update = from_plain(json_loads(body), _FavoriteUpdate)
    except ValueError as exc:  # ReadError is a ValueError.
        raise HTTPException(400, f"Invalid favorite: {exc}") from exc
    username = user["sub"]
    settings = load_user_settings(username)
    entries = settings["favorites"].setdefault(update["kind"], {})
    entry = update["entry"]
    if entry is None:
        entries.pop(update["id"], None)
    else:
        entries[update["id"]] = _stored_entry(entry)
    save_user_settings(username, settings)
    return {"favorites": _page_favorites(settings)}


@app.post("/api/user-prefs")
async def save_user_prefs(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    """Save a partial update to user preferences.

    Args:
        request: Request carrying the preference update body.
        user: Authenticated viewer claims identifying the settings owner.
    """
    username = user["sub"]
    body = await request.body()
    if len(body) > 64 * 1024:  # 64KB limit.
        raise HTTPException(400, "Request too large")
    data = json_loads(body)
    updates = data if isinstance(data, dict) else {}
    if "sort_live_by_views" in updates and not isinstance(updates["sort_live_by_views"], bool):
        raise HTTPException(400, "sort_live_by_views must be a boolean")
    try:
        prefs = from_plain(updates, _UserPrefsUpdate)
    except ReadError as exc:
        raise HTTPException(400, f"Invalid preferences: {exc}") from exc
    settings = load_user_settings(username)
    if "cc_lang" in prefs:
        settings["cc_lang"] = prefs["cc_lang"]
    if "audio_lang" in prefs:
        settings["audio_lang"] = prefs["audio_lang"]
    if "cc_style" in prefs:
        settings["cc_style"] = prefs["cc_style"]
    if "cast_host" in prefs:
        settings["cast_host"] = prefs["cast_host"]
    if "virtual_scroll" in prefs:
        settings["virtual_scroll"] = prefs["virtual_scroll"]
    if "sort_live_by_views" in prefs:
        settings["sort_live_by_views"] = prefs["sort_live_by_views"]
    if "guide_selected_cats" in prefs:
        settings["guide_selected_cats"] = prefs["guide_selected_cats"]
    save_user_settings(username, settings)
    return {"ok": True}


# =============================================================================
# Shared playlists (curated in the web UI; clients read them via the guide)
# =============================================================================


@app.get("/playlists", response_class=HTMLResponse)
async def playlists_page(request: Request, user: Annotated[TokenClaims, Depends(require_admin)]):
    del user
    return TEMPLATES.TemplateResponse(request, "playlists.html", {})


@app.get("/api/playlists")
async def list_playlists(user: Annotated[TokenClaims, Depends(require_auth)]):
    viewer = _payload_viewer(user)
    streams, names = await _live_streams_and_category_names()
    result: list[dict[str, str | int]] = []
    for p in playlists.load():
        payload = _playlist_payload(p, streams, names, viewer)
        if viewer is not None and not payload["channels"]:
            continue
        result.append(
            {
                "id": p["id"],
                "category_id": payload["category_id"],
                "name": p["name"],
                "channel_count": len(payload["channels"]),
            },
        )
    return {"playlists": result}


@app.get("/api/playlists/search")
async def search_playlist_channels(
    user: Annotated[TokenClaims, Depends(require_admin)],
    q: str = "",
    category_id: str = "",
    category_prefix: str = "",
    limit: int = Query(default=200, ge=1, le=1000),
):
    del user
    streams, names = await _live_streams_and_category_names()
    results = await asyncio.to_thread(
        playlists.search,
        streams,
        names,
        q,
        category_id,
        category_prefix,
        limit,
    )
    categories = [{"category_id": cid, "category_name": name} for cid, name in names.items()]
    return {"results": results, "categories": categories}


@app.get("/api/playlists/epg-channels")
async def search_playlist_epg_channels(
    user: Annotated[TokenClaims, Depends(require_admin)],
    q: str = "",
    limit: int = Query(default=20, ge=1, le=100),
):
    """Guide channels an admin can assign to a playlist entry that has no listings."""
    del user
    terms = playlists.normalize(q) or [q.strip()]
    results = await asyncio.to_thread(epg.search_channels, terms, limit)
    return {"results": results}


@app.post("/api/playlists/match")
async def match_playlist_channels(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_admin)],
):
    del user
    data = await _json_body(request)
    lines = [str(line) for line in list_field(data, "lines") if str(line).strip()][:1000]
    streams, names = await _live_streams_and_category_names()
    matches = await asyncio.to_thread(
        playlists.match_names,
        lines,
        streams,
        names,
        str(data.get("category_prefix", "")),
    )
    return {"matches": matches}


@app.post("/api/playlists/order")
async def reorder_playlists(request: Request, user: Annotated[TokenClaims, Depends(require_admin)]):
    del user
    data = await _json_body(request)
    ids = [str(i) for i in list_field(data, "ids")]
    return {"playlists": [p["id"] for p in playlists.reorder(ids)]}


@app.post("/api/playlists")
async def create_playlist(request: Request, user: Annotated[TokenClaims, Depends(require_admin)]):
    del user
    data = await _json_body(request)
    try:
        playlist = playlists.create(data.get("name", ""), data.get("channels"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    streams, names = await _live_streams_and_category_names()
    return _playlist_payload(playlist, streams, names)


@app.get("/api/playlists/{playlist_id}")
async def get_playlist(playlist_id: str, user: Annotated[TokenClaims, Depends(require_auth)]):
    playlist = playlists.get(playlist_id)
    if playlist is None:
        raise HTTPException(404, "Playlist not found")
    streams, names = await _live_streams_and_category_names()
    return _playlist_payload(playlist, streams, names, _payload_viewer(user))


@app.put("/api/playlists/{playlist_id}")
async def update_playlist(
    playlist_id: str,
    request: Request,
    user: Annotated[TokenClaims, Depends(require_admin)],
):
    del user
    data = await _json_body(request)
    try:
        playlist = playlists.update(playlist_id, data.get("name"), data.get("channels"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if playlist is None:
        raise HTTPException(404, "Playlist not found")
    streams, names = await _live_streams_and_category_names()
    return _playlist_payload(playlist, streams, names)


@app.delete("/api/playlists/{playlist_id}")
async def delete_playlist(playlist_id: str, user: Annotated[TokenClaims, Depends(require_admin)]):
    del user
    if not playlists.delete(playlist_id):
        raise HTTPException(404, "Playlist not found")
    return {"ok": True}


@app.get("/api/logo")
async def get_logo(
    url: str,
    user: Annotated[TokenClaims, Depends(require_auth)],
    source: str = "default",
    sig: str = "",
):
    """Proxy and cache external logos to avoid mixed-content issues.

    Args:
        url: External image URL.
        user: Authenticated viewer claims.
        source: Source name partitioning cached images.
        sig: Server signature from ``_logo_url_filter``; unsigned URLs are refused.
    """
    del user
    if not url:
        raise HTTPException(400, "Missing url parameter")
    # Only URLs the server put on a page are proxied: without this, any viewer
    # could make the server fetch internal addresses and read image replies.
    if not auth.verify_logo_url_signature(url, sig):
        raise HTTPException(403, "Invalid logo URL signature")
    # Check cache first.
    cached = get_cached_logo(source, url)
    if cached:
        return FileResponse(cached, headers={"Cache-Control": f"max-age={24 * 3600}"})
    # Validate URL scheme.
    parsed = parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "Invalid URL scheme")
    # Fetch the logo (in thread to avoid blocking)
    try:
        data, content_type = await asyncio.to_thread(_fetch_logo, url)
        path = save_logo(source, url, data, content_type)
        return FileResponse(path, headers={"Cache-Control": f"max-age={24 * 3600}"})
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except error.URLError as e:
        log.debug("Logo fetch failed for %s: %s", url, e)
        raise HTTPException(502, "Failed to fetch logo") from None
    except Exception as e:
        log.debug("Logo fetch error for %s: %s", url, e)
        raise HTTPException(500, "Logo fetch error") from None


@app.post("/settings/transcode")
async def settings_transcode(
    user: Annotated[TokenClaims, Depends(require_admin)],
    transcode_mode: Annotated[str, Form()],
    transcode_hw: Annotated[str, Form()],
    max_resolution: Annotated[cache.MaxResolution, Form()] = "1080p",
    quality: Annotated[str, Form()] = "high",
    vod_transcode_cache_mins: Annotated[int, Form()] = 60,
    live_dvr_mins: Annotated[int, Form()] = 120,
    transcode_dir: Annotated[str, Form()] = "",
    probe_live: Annotated[str | None, Form()] = None,
    probe_movies: Annotated[str | None, Form()] = None,
    probe_series: Annotated[str | None, Form()] = None,
    sr_model: Annotated[str, Form()] = "",
):
    del user
    settings = load_server_settings()
    settings["transcode_mode"] = transcode_mode
    # Validate sr_model against available models.
    available_models = get_sr_models()
    settings["sr_model"] = sr_model if sr_model in available_models else ""
    settings["transcode_hw"] = cache.hw_accel(transcode_hw)
    settings["max_resolution"] = max_resolution
    settings["quality"] = quality if quality in ("high", "medium", "low") else "high"
    settings["vod_transcode_cache_mins"] = max(0, vod_transcode_cache_mins)
    settings["live_dvr_mins"] = clamp_live_dvr_mins(live_dvr_mins)
    settings["transcode_dir"] = transcode_dir  # Empty uses the system temp dir.
    settings["probe_live"] = probe_live == "on"
    settings["probe_movies"] = probe_movies == "on"
    settings["probe_series"] = probe_series == "on"
    save_server_settings(settings)
    return {"ok": True}


@app.post("/settings/refresh-encoders")
async def settings_refresh_encoders(
    user: Annotated[TokenClaims, Depends(require_admin)],
):
    """Re-detect available hardware encoders."""
    del user
    encoders = refresh_encoders()
    return {"ok": True, "encoders": encoders}


@app.post("/settings/user-agent")
async def settings_user_agent(
    user: Annotated[TokenClaims, Depends(require_admin)],
    preset: Annotated[str, Form()],
    custom: Annotated[str, Form()] = "",
):
    del user
    valid_presets = {"default", "vlc", "chrome", "tivimate", "custom"}
    if preset not in valid_presets:
        preset = "default"
    settings = load_server_settings()
    settings["user_agent_preset"] = preset
    settings["user_agent_custom"] = custom
    save_server_settings(settings)
    return {"ok": True}


@app.get("/settings/probe-cache")
async def get_probe_cache(
    user: Annotated[TokenClaims, Depends(require_auth)],
    response: Response,
):
    """Get probe cache stats for settings UI.

    Args:
        user: Authenticated viewer claims.
        response: Response receiving no-cache headers.
    """
    del user
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    probes = await asyncio.to_thread(_labeled_url_probes)
    stats = ffmpeg_command.get_series_probe_cache_stats()
    xtream = get_first_xtream_client()
    if xtream:
        stats = await asyncio.to_thread(_enrich_probe_cache_stats, stats, xtream)
    return {"probes": [probe for _, probe in probes], "series": stats}


@app.post("/settings/probe-cache/clear")
async def clear_probe_cache(user: Annotated[TokenClaims, Depends(require_admin)]):
    """Clear all probe caches."""
    del user
    count = ffmpeg_command.clear_all_probe_cache()
    return {"ok": True, "cleared": count}


@app.post("/settings/probe-cache/clear/{series_id}")
async def clear_series_probe_cache(
    series_id: int,
    user: Annotated[TokenClaims, Depends(require_admin)],
    episode_id: int | None = None,
):
    """Clear probe cache for a specific series or episode."""
    del user
    ffmpeg_command.invalidate_series_probe_cache(series_id, episode_id)
    return {"ok": True}


@app.post("/settings/probe-cache/clear-url")
async def clear_url_probe_cache(
    user: Annotated[TokenClaims, Depends(require_admin)],
    kind: Literal["movie", "live"],
    stream_id: str,
):
    """Clear the cached probe of one movie or live channel.

    Args:
        user: Authenticated admin claims.
        kind: Whether ``stream_id`` names a movie or a live channel.
        stream_id: The provider's stream id.
    """
    del user
    urls = [
        url
        for url, probe in await asyncio.to_thread(_labeled_url_probes)
        if (probe["kind"], probe["stream_id"]) == (kind, stream_id)
    ]
    return {"ok": True, "cleared": ffmpeg_command.invalidate_url_probe_cache(urls)}


class _UrlProbe(TypedDict):
    kind: Literal["movie", "live", "other"]
    stream_id: str
    name: str
    probed_at: float
    video_codec: str
    audio_codec: str
    audio_langs: list[str]
    subtitle_count: int
    duration: float
    height: int


@app.post("/settings/probe-cache/clear-mru/{series_id}")
async def clear_series_mru(
    series_id: int,
    user: Annotated[TokenClaims, Depends(require_admin)],
):
    """Clear only the MRU for a series, keeping episode cache intact."""
    del user
    ffmpeg_command.clear_series_mru(series_id)
    return {"ok": True}


@app.post("/settings/data-cache/clear")
async def clear_data_cache(user: Annotated[TokenClaims, Depends(require_admin)]):
    """Clear all data file caches (live, VOD, series) and memory cache."""
    del user
    count = clear_all_file_caches()
    return {"ok": True, "cleared": count}


@app.get("/api/settings")
async def get_settings_api(user: Annotated[TokenClaims, Depends(require_auth)]):
    del user
    return _api_settings(load_server_settings())


@app.post("/api/settings")
async def update_settings_api(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_admin)],
):
    del user
    data = json_loads(await request.body())
    try:
        update = from_plain(data if isinstance(data, dict) else {}, _ApiSettingsUpdate)
    except ReadError as exc:
        raise HTTPException(400, f"Invalid settings: {exc}") from exc
    settings = load_server_settings()
    if "transcode_mode" in update:
        settings["transcode_mode"] = update["transcode_mode"]
    if "transcode_hw" in update:
        settings["transcode_hw"] = cache.hw_accel(update["transcode_hw"])
    if "vod_transcode_cache_mins" in update:
        settings["vod_transcode_cache_mins"] = update["vod_transcode_cache_mins"]
    if "probe_live" in update:
        settings["probe_live"] = update["probe_live"]
    if "probe_movies" in update:
        settings["probe_movies"] = update["probe_movies"]
    if "probe_series" in update:
        settings["probe_series"] = update["probe_series"]
    if "vod_order" in update:
        settings["vod_order"] = update["vod_order"]
    if "series_order" in update:
        settings["series_order"] = update["series_order"]
    save_server_settings(settings)
    return {"status": "ok"}


@app.post("/api/watch-position")
async def save_watch_position_api(
    request: Request,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    """Save a user's watch position for a stream.

    Args:
        request: Request carrying URL, position and duration.
        user: Authenticated viewer claims identifying the watch owner.
    """
    username = user["sub"]
    data = json_loads(await request.body())
    fields = data if isinstance(data, dict) else {}
    url = text_field(fields, "url")
    position = _float_field(fields, "position")
    duration = _float_field(fields, "duration")
    if url and position >= 0:
        save_watch_position(username, url, position, duration)
    return {"status": "ok"}


@app.get("/api/watch-position")
async def get_watch_position_api(
    user: Annotated[TokenClaims, Depends(require_auth)],
    url: str,
):
    """Get watch position for a stream (per-user)."""
    username = user.get("sub", "")
    entry = get_watch_position(username, url)
    if entry is not None:
        return {"position": entry["position"], "duration": entry["duration"]}
    return {"position": 0, "duration": 0}


class HandoffReport(BaseModel):
    """The broadcast moment on screen while a live channel plays."""

    stream_id: str = Field(min_length=1, max_length=200)
    position: float = Field(gt=0)
    handoff_id: str = Field(default="", max_length=64)


@app.post("/api/handoff/watch")
async def handoff_watch_api(
    report: HandoffReport,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    """Note where a live channel is playing for resuming elsewhere.

    Args:
        report: Channel and broadcast timestamp on screen.
        user: Authenticated viewer claims identifying the watch owner.
    """
    handoff.report_watch(
        user.get("sub", ""),
        report.stream_id,
        min(report.position, time.time()),
        report.handoff_id,
    )
    return {"status": "ok"}


@app.post("/api/handoff/stop")
async def handoff_stop_api(
    report: HandoffReport,
    user: Annotated[TokenClaims, Depends(require_auth)],
):
    """Offer stopped playback to the user's other devices.

    Args:
        report: Channel and broadcast timestamp at playback stop.
        user: Authenticated viewer claims identifying the watch owner.
    """
    username = user.get("sub", "")
    if not await asyncio.to_thread(_handoff_stream_allowed, username, report.stream_id):
        raise HTTPException(404, "Stream not found")
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(
        handoff.stop_watch,
        username,
        report.stream_id,
        report.position,
        _handoff_channel,
        _handoff_recorder(loop),
        report.handoff_id,
    )
    return {"status": "ok"}


@app.get("/api/handoff")
async def handoff_api(user: Annotated[TokenClaims, Depends(require_auth)]):
    """Return the channel left playing on another device, if any.

    Args:
        user: Authenticated viewer claims identifying the handoff owner.
    """
    record = handoff.current(user["sub"])
    if record is None:
        return JSONResponse({"handoff": None}, headers={"Cache-Control": "no-store"})
    keys = (
        "id",
        "stream_id",
        "channel_name",
        "program_title",
        "program_start",
        "program_end",
        "position",
        "mode",
    )
    return JSONResponse(
        {"handoff": {key: record[key] for key in keys}},
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/api/handoff")
async def handoff_dismiss_api(user: Annotated[TokenClaims, Depends(require_auth)]):
    """Dismiss the handoff, ending its recording."""
    await asyncio.to_thread(handoff.cancel, user.get("sub", ""))
    return {"status": "ok"}


@app.get("/handoff/{recording_id}/{filename}")
async def handoff_file(recording_id: str, filename: str):
    """Serve a handoff recording using its unguessable identifier.

    Args:
        recording_id: Retained recording identifier.
        filename: Recording playlist or segment filename.
    """
    path = handoff.recording_file(recording_id, filename)
    if path is None:
        raise HTTPException(404, "File not found")
    if filename.endswith(".m3u8"):
        return Response(
            content=path.read_text(),
            media_type="application/vnd.apple.mpegurl",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    if not filename.endswith(".ts"):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp2t")


# User management endpoints.
@app.post("/settings/users/delete/{username}")
async def settings_delete_user(
    username: str,
    user: Annotated[TokenClaims, Depends(require_auth)],
    password: Annotated[str, Form()] = "",
):
    """Delete an account after checking its owner's or administrator's access.

    Args:
        username: Account to delete.
        user: Authenticated caller claims.
        password: Current password required for self-deletion.
    """
    current_user = user.get("sub", "")
    if username == current_user:
        if not password or not auth.verify_password(username, password):
            raise HTTPException(400, "Password required to delete your own account")
        auth.delete_user(username)
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie("token")
        return response
    # Deleting other users requires admin.
    if not auth.is_admin(current_user):
        raise HTTPException(403, "Admin access required")
    if not auth.delete_user(username):
        raise HTTPException(404, "User not found")
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/users/add")
async def settings_add_user(
    user: Annotated[TokenClaims, Depends(require_admin)],
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    admin: Annotated[str, Form()] = "",
    max_streams_per_source: Annotated[str | None, Form()] = None,
    unavailable_groups: Annotated[str | None, Form()] = None,
):
    """Add a new user.

    Args:
        user: Authenticated administrator claims.
        username: New account name.
        password: New account password.
        admin: Form checkbox granting administrator access.
        max_streams_per_source: Optional encoded source stream limits.
        unavailable_groups: Optional encoded restricted group identifiers.
    """
    del user
    username = username.strip()
    if not username or len(username) < 2:
        raise HTTPException(400, "Username must be at least 2 characters")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if username in auth.get_all_usernames():
        raise HTTPException(400, "User already exists")
    auth.create_user(username, password, admin=admin == "on")
    # Apply limits if provided; malformed limits are ignored.
    parsed_max_streams: dict[str, int] | None = None
    if max_streams_per_source is not None and max_streams_per_source:
        with contextlib.suppress(ValueError):
            parsed_max_streams = json_loads(max_streams_per_source, dict[str, int])
    parsed_unavailable: list[str] | None = None
    if unavailable_groups is not None and unavailable_groups:
        with contextlib.suppress(ValueError):
            parsed_unavailable = json_loads(unavailable_groups, list[str])
    if (parsed_max_streams is not None and len(parsed_max_streams) > 0) or (
        parsed_unavailable is not None and len(parsed_unavailable) > 0
    ):
        auth.set_user_limits(username, parsed_max_streams, parsed_unavailable)
    return {"status": "ok"}


@app.post("/settings/users/password")
async def settings_change_own_password(
    user: Annotated[TokenClaims, Depends(require_auth)],
    current_password: Annotated[str, Form()],
    new_password: Annotated[str, Form()],
):
    """Change the caller's password after verifying the current password.

    Args:
        user: Authenticated account claims.
        current_password: Current account password.
        new_password: Replacement password.
    """
    if len(new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if not auth.change_password(
        user.get("sub", ""),
        new_password,
        current_password=current_password,
    ):
        raise HTTPException(400, "Current password is incorrect")
    return {"status": "ok"}


@app.post("/settings/users/password/{target_user}")
async def settings_change_password(
    target_user: str,
    user: Annotated[TokenClaims, Depends(require_auth)],
    new_password: Annotated[str, Form()],
):
    """Change a password for the account owner or an administrator.

    Args:
        target_user: Account whose password is replaced.
        user: Authenticated caller claims.
        new_password: Replacement password.
    """
    current_user = user.get("sub", "")
    if target_user != current_user and not auth.is_admin(current_user):
        raise HTTPException(403, "Admin access required")
    if len(new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if not auth.change_password(target_user, new_password):
        raise HTTPException(404, "User not found")
    return {"status": "ok"}


@app.post("/settings/users/admin/{target_user}")
async def settings_set_admin(
    target_user: str,
    user: Annotated[TokenClaims, Depends(require_admin)],
    admin: Annotated[str, Form()] = "",
):
    """Set admin status for a user."""
    del user
    if not auth.set_admin(target_user, admin == "on"):
        raise HTTPException(404, "User not found")
    return {"status": "ok"}


@app.post("/settings/users/limits/{target_user}")
async def settings_set_user_limits(
    target_user: str,
    user: Annotated[TokenClaims, Depends(require_admin)],
    max_streams_per_source: Annotated[str | None, Form()] = None,  # JSON object string.
    unavailable_groups: Annotated[str | None, Form()] = None,  # JSON array string.
):
    """Set stream limits and group restrictions for a user.

    Args:
        target_user: Account receiving the limits.
        user: Authenticated administrator claims.
        max_streams_per_source: Optional encoded source stream limits.
        unavailable_groups: Optional encoded restricted group identifiers.
    """
    del user
    parsed_max_streams: dict[str, int] | None = None
    if max_streams_per_source is not None:
        try:
            parsed_max_streams = json_loads(max_streams_per_source, dict[str, int])
        except ReadError as err:
            raise HTTPException(400, "max_streams_per_source must be a JSON object") from err
        except ValueError as err:
            raise HTTPException(400, "Invalid JSON for max_streams_per_source") from err

    parsed_unavailable: list[str] | None = None
    if unavailable_groups is not None:
        try:
            parsed_unavailable = json_loads(unavailable_groups, list[str])
        except ReadError as err:
            raise HTTPException(400, "unavailable_groups must be a JSON array") from err
        except ValueError as err:
            raise HTTPException(400, "Invalid JSON for unavailable_groups") from err

    if not auth.set_user_limits(target_user, parsed_max_streams, parsed_unavailable):
        raise HTTPException(404, "User not found")
    return {"status": "ok"}


def _logo_url_filter(url: str) -> str:
    """Wrap external logo URLs through the signed /api/logo proxy."""
    if not url or url.startswith("/") or url.startswith("data:"):
        return url  # Already local or data URL.
    # Use hostname as source for organization.
    parsed = parse.urlparse(url)
    source = parsed.netloc.split(":")[0] if parsed.netloc else "external"
    sig = auth.logo_url_signature(url)
    return f"/api/logo?source={parse.quote(source)}&url={parse.quote(url)}&sig={sig}"


Favorites = dict[str, dict[str, MutablePlainTree]]


def _hidden_sources(username: str) -> set[str]:
    """Sources not assigned to this user (admins see all)."""
    return hidden_source_ids(username, auth.is_admin(username))


def _unavailable_groups(username: str) -> set[str]:
    """User's blocked groups, plus every group of sources not assigned to them."""
    groups = set(auth.get_user_limits(username).get("unavailable_groups", []))
    for source_id in _hidden_sources(username):
        groups.update((f"source:{source_id}", f"movies:{source_id}", f"series:{source_id}"))
    return groups


# Returns dict with 'movies' and 'series' booleans. If no xtream sources exist, access
# is granted (nothing to restrict).
def _get_content_access(username: str) -> dict[str, bool]:
    """Check if user has access to movies/series from any source."""
    unavailable_groups = _unavailable_groups(username)

    has_movies = False
    has_series = False
    has_xtream_source = False

    for source in get_sources():
        if source.type == "xtream":
            has_xtream_source = True
            if f"movies:{source.id}" not in unavailable_groups:
                has_movies = True
            if f"series:{source.id}" not in unavailable_groups:
                has_series = True

    # If no xtream sources, allow access (nothing to restrict)
    if not has_xtream_source:
        return {"movies": True, "series": True}

    return {"movies": has_movies, "series": has_series}


def _get_content_access_from_request(request: Request) -> dict[str, bool]:
    """Get content access from request (for use in base template)."""
    token = request.cookies.get("token")
    if token is None or token == "":
        return {"movies": True, "series": True}  # Not logged in, show all.
    payload = verify_token(token)
    if payload is None:
        return {"movies": True, "series": True}
    username = payload["sub"]
    if not username:
        return {"movies": True, "series": True}
    return _get_content_access(username)


# Built last so its filter and globals are defined; routes read it only per request.
TEMPLATES = _create_templates()


def main() -> int:
    """Serve the web app.

    Returns:
        status: Process exit code.
    """

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())

    # LOG_LEVEL env var takes precedence, then --debug flag, then default INFO.
    log_level_env = os.environ.get("LOG_LEVEL", "").upper()
    if log_level_env in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        log_level = getattr(logging, log_level_env)
    else:
        log_level = logging.DEBUG if flags.debug else logging.INFO
    configure_logging(log_level)

    ssl_certfile = None
    ssl_keyfile = None
    if flags.cert is not None and flags.key is not None and flags.cert and flags.key:
        ssl_certfile = flags.cert
        ssl_keyfile = flags.key
    elif flags.https is not None:
        live_dir = pathlib.Path("/etc/letsencrypt/live")
        if flags.https:
            domain = flags.https
        else:
            # Auto-detect first domain.
            domains = [
                d.name for d in live_dir.iterdir() if d.is_dir() and (d / "fullchain.pem").exists()
            ]
            if not domains:
                raise SystemExit("No Let's Encrypt certs found in /etc/letsencrypt/live/")
            domain = domains[0]
        cert = live_dir / domain / "fullchain.pem"
        key = live_dir / domain / "privkey.pem"
        if not cert.exists():
            raise SystemExit(f"Cert not found: {cert}")
        log.info("Using Let's Encrypt certs for %s", domain)
        ssl_certfile = str(cert)
        ssl_keyfile = str(key)

    uv_log = "debug" if flags.debug else "info"
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=flags.port,
        access_log=flags.debug,
        log_level=uv_log,
        log_config=None,  # Preserve our basicConfig.
        timeout_graceful_shutdown=5,
        proxy_headers=True,
        forwarded_allow_ips="*",
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    port: int
    debug: bool
    https: str | None
    cert: str | None
    key: str | None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--https",
        nargs="?",
        const="",
        metavar="DOMAIN",
        help="Enable HTTPS (auto-detect domain, or specify one)",
    )
    parser.add_argument("--cert", help="SSL certificate file (e.g., fullchain.pem)")
    parser.add_argument("--key", help="SSL private key file (e.g., privkey.pem)")


def _api_settings(
    settings: Mapping[str, object],
    keys: tuple[str, ...] = (
        "transcode_mode",
        "transcode_hw",
        "vod_transcode_cache_mins",
        "probe_live",
        "probe_movies",
        "probe_series",
        "vod_order",
        "series_order",
    ),
) -> dict[str, object]:
    """Select settings the API may expose."""
    # Users, secrets and sources must never be exposed or edited through this API.
    return {key: settings[key] for key in keys if key in settings}


async def _save_category_filter(
    request: Request,
    user: TokenClaims,
    setting: Literal["guide_filter", "vod_filter", "series_filter"],
    *,
    max_categories: int = 10_000,
) -> dict[str, str]:
    """Validate and store a user's category filter list under ``setting``."""
    username = user["sub"]
    data = json_loads(await request.body())
    raw_cats = data.get("cats", []) if isinstance(data, dict) else None
    if not isinstance(raw_cats, list) or len(raw_cats) > max_categories:
        raise HTTPException(400, "Invalid filter list")
    cats = [c for c in raw_cats if isinstance(c, str)]
    if len(cats) != len(raw_cats):
        raise HTTPException(400, "Invalid filter list")
    user_settings = load_user_settings(username)
    if setting == "guide_filter":
        user_settings["guide_filter"] = cats
    elif setting == "vod_filter":
        user_settings["vod_filter"] = cats
    else:
        user_settings["series_filter"] = cats
    save_user_settings(username, user_settings)
    return {"status": "ok"}


def _name_matcher(q: str, *, regex: bool) -> Callable[[str], bool]:
    """Build a case-insensitive name predicate for search."""
    if not regex:
        q_lower = q.lower()
        return lambda name: q_lower in name.lower()
    try:
        pattern = re.compile(q, re.IGNORECASE)
    except re.error:
        return lambda name: False
    return lambda name: _regex_matches(pattern, name)


def _regex_matches(pattern: re.Pattern[str], name: str) -> bool:
    """Search only the first 1000 characters to bound regex cost."""
    try:
        return pattern.search(name[:1000]) is not None
    except Exception:
        return False


async def _epg_event_stream(queue: asyncio.Queue[str]) -> AsyncGenerator[str, None]:
    """Yield one EPG readiness event, stopping early on shutdown."""
    try:
        if epg.has_programs():
            yield "data: epg_ready\n\n"
            return
        while not (_shutdown_event and _shutdown_event.is_set()):
            event = await _next_epg_event(queue)
            if event is not None:
                yield f"data: {event}\n\n"
                return
    finally:
        _epg_subscribers.discard(queue)


async def _next_epg_event(queue: asyncio.Queue[str]) -> str | None:
    """Wait one second for an EPG event."""
    try:
        return await asyncio.wait_for(queue.get(), timeout=1)
    # Python 3.10 raises asyncio.TimeoutError, which only aliases TimeoutError on 3.11+.
    except asyncio.TimeoutError:
        return None


def _safe_float(value: MutablePlainTree, default: float = 0.0) -> float:
    """Safely convert value to float, returning default on failure."""
    if not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _float_field(record: Mapping[str, MutablePlainTree], key: str) -> float:
    """Read a numeric request field, defaulting a missing one to zero."""
    value = record.get(key, 0)
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"{key} must be a number")


def _int_value(value: MutablePlainTree) -> int:
    """Integer form of a provider value; unset or non-scalar values are zero."""
    return int(value) if isinstance(value, (int, float, str)) and value else 0


def _present(value: MutablePlainTree) -> bool:
    """Whether a provider value is set: neither null nor empty, zero or false."""
    return value is not None and bool(value)


def _first_present(record: Mapping[str, MutablePlainTree], *keys: str) -> MutablePlainTree:
    """The first set value among ``keys``, else an empty string."""
    return next((record[key] for key in keys if _present(record.get(key))), "")


def _store_cached_live_data() -> bool:
    """Fill the memory cache from the live file cache; report whether one existed."""
    cached = cached_live_data()
    if cached is None:
        return False
    data, _ = cached
    with get_cache_lock():
        store = get_cache()
        store["live_categories"] = data["cats"]
        store["live_streams"] = data["streams"]
        store["epg_urls"] = data["epg_urls"]
    return True


def _store_cached_vod_data() -> bool:
    """Fill the memory cache from the VOD file cache; report whether one existed."""
    cached = cached_vod_data()
    if cached is None:
        return False
    data, _ = cached
    with get_cache_lock():
        store = get_cache()
        store["vod_categories"] = data["cats"]
        store["vod_streams"] = data["streams"]
    return True


def _store_cached_series_data() -> bool:
    """Fill the memory cache from the series file cache; report whether one existed."""
    cached = cached_series_data()
    if cached is None:
        return False
    data, _ = cached
    with get_cache_lock():
        store = get_cache()
        store["series_categories"] = data["cats"]
        store["series"] = data["series"]
    return True


# Movies and series add IMDb's credits to the provider's. A ``limit`` of 0 keeps all.
def _matching_names(
    records: list[ProviderRecord],
    match_fn: Callable[[str], bool],
    limit: int,
) -> list[ProviderRecord]:
    """Records whose name or credited people match, sorted by name, at most ``limit``."""
    index = cache.CACHE_DIR / "imdb.json"
    matched = sorted(
        (r for r in records if any(map(match_fn, _search_texts(r, index)))),
        key=lambda r: text_field(r, "name").lower(),
    )
    return matched[:limit] if limit else matched


def _search_texts(record: ProviderRecord, index: pathlib.Path) -> list[str]:
    """The record's name, cast, and director, plus everyone IMDb credits on it."""
    texts = [text_field(record, f) for f in ("name", "cast", "director")]
    if "container_extension" in record or "series_id" in record:
        found = imdb.info(index, record)
        if found:
            texts += imdb.people(found)
    return texts


def _clean_episode(ep: ProviderRecord) -> None:
    """Strip redundant series prefixes from an episode title and expose its plot."""
    title = ep.get("title")
    if isinstance(title, str) and title:
        # Remove patterns like "Series Name - S01E01 - Episode Title"; keep the title.
        title = re.sub(r"^S\d+E\d+\s*-\s*", "", title)
        if " - " in title:
            title = title.split(" - ")[-1]
        ep["title"] = title.strip()
    info = ep.get("info")
    if isinstance(info, str) and info:
        try:
            parsed = json_loads(info)
        except ValueError:
            return
        if isinstance(parsed, dict):
            ep["description"] = _first_present(parsed, "plot", "description")
    elif isinstance(info, dict) and info:
        ep["description"] = _first_present(info, "plot", "description")


async def _with_vod_details(
    movie: ProviderRecord,
    source_id: str,
    stream_id: int,
) -> ProviderRecord:
    """A copy of ``movie`` merged with its provider details, when they load."""
    xtream = get_xtream_client_by_source(source_id) if source_id else None
    if xtream is None:
        xtream = get_first_xtream_client()
    if xtream is None:
        return movie
    client = xtream
    cache_key = f"vod_info_{source_id}_{stream_id}" if source_id else f"vod_info_{stream_id}"
    try:
        vod_info = await asyncio.to_thread(
            get_cached_info,
            cache_key,
            lambda: client.get_vod_info(stream_id),
        )
    except (error.URLError, TimeoutError):
        return movie
    info = record_field(vod_info, "info")
    if not info:
        return movie
    release = info.get("releasedate")
    return {
        **movie,
        "plot": _first_present(info, "plot", "description"),
        "director": info.get("director", ""),
        "cast": _first_present(info, "cast", "actors"),
        "genre": info.get("genre", ""),
        "rating": info.get("rating", ""),
        "year": release[:4] if isinstance(release, str) and release else "",
        "duration": info.get("duration", ""),
        "cover_big": _first_present(info, "cover_big", "movie_image"),
        "youtube_trailer": info.get("youtube_trailer", ""),
    }


def _check_rate_limit(ip: str, *, window_secs: float = 300, max_attempts: int = 10) -> None:
    """Check login rate limit. Raises HTTPException if exceeded."""
    now = time.time()
    attempts = _login_attempts.get(ip, [])
    # Clean old attempts for this IP.
    attempts = [t for t in attempts if now - t < window_secs]
    if attempts:
        _login_attempts[ip] = attempts
    elif ip in _login_attempts:
        del _login_attempts[ip]
    # Periodically clean stale IPs (when dict is large)
    if len(_login_attempts) > 1000:
        stale = [k for k, v in _login_attempts.items() if not v or now - max(v) > window_secs]
        for k in stale[:100]:
            del _login_attempts[k]
    if len(attempts) >= max_attempts:
        raise HTTPException(429, "Too many login attempts, try again later")


def _fetch_all_epg(epg_urls: list[tuple[str, int, str]]) -> int:
    """Fetch EPG from all URLs into sqlite (in parallel). Returns total program count."""
    user_agent = cache.get_user_agent()

    def fetch_one(url_timeout_source: tuple[str, int, str]) -> tuple[str, int]:
        url, timeout, source_id = url_timeout_source
        try:
            log.info("Fetching EPG (timeout=%ds): %s", timeout, url[:80])
            count = fetch_epg(
                url,
                cache.CACHE_DIR,
                timeout=timeout,
                source_id=source_id,
                user_agent=user_agent,
            )
            log.info("EPG done: %d programs from %s", count, url[:50])
            return url, count
        except Exception as e:
            log.error("EPG failed: %s - %s", url[:50], e)
            return url, 0

    total = 0
    max_workers = min(len(epg_urls) or 1, 8)  # Cap at 8 concurrent fetches.
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        pending = [ex.submit(fetch_one, u) for u in epg_urls]
        for future in futures.as_completed(pending):
            _, count = future.result()
            total += count

    _prune_epg()

    log.info("EPG fetch complete: %d programs total", total)
    return total


def _prune_epg() -> None:
    """Drop listings older than the longest catchup archive (at least 24h)."""
    cutoff = datetime.now(timezone.utc) - catchup.history(get_cache().get("live_streams", []))
    pruned = epg.prune_old_programs(cutoff)
    if pruned:
        log.info("Pruned %d expired EPG programs", pruned)


def _epg_is_fresh() -> bool:
    """True if the EPG still has listings a few hours ahead."""
    return epg.has_programs_after(datetime.now(timezone.utc) + _EPG_MIN_LOOKAHEAD)


def _start_guide_background_load() -> None:
    """Start background loading of guide data if not already in progress."""
    if "guide_load" in get_refresh_in_progress():
        return
    get_refresh_in_progress().add("guide_load")

    def load():
        try:
            log.info("Loading guide data in background")
            cats, streams, epg_urls = load_all_live_data()
            with get_cache_lock():
                get_cache()["live_categories"] = cats
                get_cache()["live_streams"] = streams
                get_cache()["epg_urls"] = epg_urls
            try:
                _fetch_all_epg(epg_urls)
            except Exception as e:
                with get_cache_lock():
                    get_cache()["epg_error"] = str(e)
            log.info("Guide data loaded")
        finally:
            get_refresh_in_progress().discard("guide_load")

    threading.Thread(target=load, daemon=True).start()


def _get_guide_streams(
    cats: str,
    username: str,
) -> tuple[list[ProviderRecord], list[str], set[str]]:
    """Get filtered and sorted streams for guide."""
    all_streams = get_cache().get("live_streams", [])

    # Parse selected category IDs (ordered list)
    ordered_cats: list[str] = []
    if cats:
        ordered_cats = [c.strip() for c in cats.split(",") if c.strip()]
    selected_cats = set(ordered_cats)

    if not selected_cats:
        return [], ordered_cats, selected_cats

    stream_allowed = _live_stream_filter(username)
    cat_order = {c: i for i, c in enumerate(ordered_cats)}
    user_settings = load_user_settings(username)
    view_counts = (
        load_live_view_counts(username) if user_settings.get("sort_live_by_views", True) else {}
    )

    def stream_sort_key(s: ProviderRecord) -> int:
        return _guide_section_rank(s, cat_order, len(ordered_cats))

    def view_count(s: ProviderRecord) -> int:
        return view_counts.get(playlists.stream_key(s), 0)

    playlist_ids = [c for c in ordered_cats if playlists.is_playlist_id(c)]
    category_cats = selected_cats.difference(playlist_ids)
    streams = [
        s
        for s in all_streams
        if any(c in category_cats for c in category_ids(s)) and stream_allowed(s)
    ]
    streams.sort(key=lambda s: (stream_sort_key(s), -view_count(s)))
    if not playlist_ids:
        return streams, ordered_cats, selected_cats

    # Playlist channels appear once, at their earliest selected position. Their
    # rows carry the playlist ids so clients can group them like categories.
    playlist_by_id = {f"{playlists.PREFIX}{p['id']}": p for p in playlists.load()}
    placement: dict[str, tuple[int, int]] = {}
    memberships: dict[str, list[str]] = {}
    stream_by_key: dict[str, ProviderRecord] = {}
    for playlist_id in playlist_ids:
        playlist = playlist_by_id.get(playlist_id)
        if playlist is None:
            continue
        for position, s in enumerate(playlists.resolve(playlist, all_streams)):
            if not stream_allowed(s):
                continue
            key = playlists.stream_key(s)
            memberships.setdefault(key, []).append(playlist_id)
            stream_by_key[key] = s
            placement.setdefault(key, (cat_order[playlist_id], position))
    for position, s in enumerate(streams):
        key = playlists.stream_key(s)
        rank = (stream_sort_key(s), position)
        stream_by_key.setdefault(key, s)
        if key not in placement or rank[0] < placement[key][0]:
            placement[key] = rank

    merged: list[ProviderRecord] = []
    for key in sorted(
        placement,
        key=lambda key: (
            placement[key][0],
            -view_counts.get(key, 0),
            placement[key][1],
        ),
    ):
        s = stream_by_key[key]
        if key in memberships:
            s = {**s, "category_ids": [*list_field(s, "category_ids"), *memberships[key]]}
        merged.append(s)
    return merged, ordered_cats, selected_cats


def _guide_section_rank(s: ProviderRecord, cat_order: dict[str, int], default: int) -> int:
    """Position of the earliest selected category (or playlist) a stream belongs to."""
    return min(
        (cat_order[c] for c in category_ids(s) if c in cat_order),
        default=default,
    )


def _guide_sections(streams: list[ProviderRecord], ordered_cats: list[str]) -> list[GuideSection]:
    """Headings for the web guide: the row index where each selected category starts."""
    cat_order = {c: i for i, c in enumerate(ordered_cats)}
    names = {
        str(c["category_id"]): text_field(c, "category_name")
        for c in get_cache().get("live_categories", [])
    }
    names.update({c["category_id"]: f"★ {c['category_name']}" for c in _playlist_categories()})
    sections: list[GuideSection] = []
    current = None
    for index, s in enumerate(streams):
        rank = _guide_section_rank(s, cat_order, -1)
        if rank < 0 or rank == current:
            continue
        current = rank
        name = names.get(ordered_cats[rank], "")
        if name:
            sections.append({"start": index, "name": name})
    return sections


def _live_allowed(s: ProviderRecord, unavailable_groups: set[str]) -> bool:
    """Whether a live stream or category dict passes the user's restrictions."""
    if f"source:{s.get('source_id', '')}" in unavailable_groups:
        return False
    cat_ids = list_field(s, "category_ids") or ([s["category_id"]] if "category_id" in s else [])
    return not any(f"cat:{c}" in unavailable_groups for c in cat_ids)


def _live_stream_filter(username: str) -> Callable[[ProviderRecord], bool]:
    """Predicate for live streams the user may see (source and category restrictions)."""
    unavailable_groups = _unavailable_groups(username)

    def stream_allowed(s: ProviderRecord) -> bool:
        return _live_allowed(s, unavailable_groups)

    return stream_allowed


def _playlist_categories() -> list[PlaylistCategory]:
    """Non-empty shared playlists as guide categories, in admin-defined order."""
    return [
        {
            "category_id": f"{playlists.PREFIX}{p['id']}",
            "category_name": p["name"],
            "kind": "playlist",
        }
        for p in playlists.load()
        if p.get("channels", [])
    ]


def _default_guide_cats(
    saved_filter: list[str],
    playlist_cats: list[PlaylistCategory] | None = None,
) -> list[str]:
    if playlist_cats is None:
        playlist_cats = _playlist_categories()
    return [c["category_id"] for c in playlist_cats] + [
        c for c in saved_filter if not playlists.is_playlist_id(c)
    ]


def _build_guide_rows(
    streams: list[ProviderRecord],
    start_idx: int,
    count: int,
    window_start: datetime,
    window_end: datetime,
) -> list[GuideRow]:
    """Build guide grid rows for a range of streams."""
    end_idx = min(start_idx + count, len(streams))
    slice_streams = streams[start_idx:end_idx]

    if not slice_streams:
        return []

    # Collect EPG IDs for batch query.
    epg_ids = [text_field(s, "epg_channel_id") for s in slice_streams]
    epg_ids_set = [e for e in epg_ids if e]

    # Batch fetch icons and programs.
    icons_map = epg.get_icons_batch(epg_ids_set) if epg_ids_set else {}

    # Build preferred_sources for EPG matching.
    preferred_sources = {
        epg_id: text_field(s, "source_id")
        for s, epg_id in zip(slice_streams, epg_ids, strict=False)
        if epg_id and text_field(s, "source_id")
    }
    programs_map = (
        epg.get_programs_batch(epg_ids_set, window_start, window_end, preferred_sources)
        if epg_ids_set
        else {}
    )

    # Build rows.
    window_end_mobile = window_start + timedelta(hours=2)
    now = datetime.now(timezone.utc)
    grid_data: list[GuideRow] = []

    for idx, (s, epg_id) in enumerate(zip(slice_streams, epg_ids, strict=False), start=start_idx):
        stream_icon = s.get("stream_icon", "")
        icon = stream_icon if _present(stream_icon) else icons_map.get(epg_id, "")
        ch: GuideChannel = {
            "stream_id": s["stream_id"],
            "name": s["name"],
            "icon": icon,
            # Signed /api/logo URL for the web guide; `icon` stays raw for apps.
            "icon_proxy": _logo_url_filter(icon) if isinstance(icon, str) else "",
            "epg_id": epg_id,
            "category_ids": category_ids(s),
            "catchup_days": catchup.archive_days(s),
        }
        row: GuideRow = {"channel": ch, "programs": [], "programs_mobile": [], "index": idx}

        for p in programs_map.get(epg_id, []):
            # Ended programs still in the archive can be watched from the start.
            can_catch_up = p.stop <= now and catchup.can_play(s, p.start, now)
            p_start = max(p.start, window_start)
            p_end = min(p.stop, window_end)
            start_mins = (p_start - window_start).total_seconds() / 60
            duration_mins = (p_end - p_start).total_seconds() / 60
            left_pct = (start_mins / 180) * 100
            width_pct = (duration_mins / 180) * 100
            row["programs"].append(
                {
                    "title": p.title,
                    "desc": p.desc,
                    "start": p.start.strftime("%H:%M"),
                    "end": p.stop.strftime("%H:%M"),
                    "start_timestamp": p.start.timestamp(),
                    "end_timestamp": p.stop.timestamp(),
                    "left_pct": left_pct,
                    "width_pct": width_pct,
                    "catchup": can_catch_up,
                    "unavailable": p.stop <= now and not can_catch_up,
                },
            )
            # Mobile: 2-hour window.
            if p.start < window_end_mobile:
                p_end_m = min(p.stop, window_end_mobile)
                duration_mins_m = (p_end_m - p_start).total_seconds() / 60
                left_pct_m = (start_mins / 120) * 100
                width_pct_m = (duration_mins_m / 120) * 100
                row["programs_mobile"].append(
                    {
                        "title": p.title,
                        "desc": p.desc,
                        "start": p.start.strftime("%H:%M"),
                        "end": p.stop.strftime("%H:%M"),
                        "left_pct": left_pct_m,
                        "width_pct": width_pct_m,
                        "start_timestamp": p.start.timestamp(),
                        "end_timestamp": p.stop.timestamp(),
                        "catchup": can_catch_up,
                        "unavailable": p.stop <= now and not can_catch_up,
                    },
                )

        grid_data.append(row)

    return grid_data


def _start_vod_background_load() -> None:
    """Start background loading of VOD data if not already in progress."""
    if "vod_load" in get_refresh_in_progress():
        return
    get_refresh_in_progress().add("vod_load")

    def load():
        try:
            log.info("Loading VOD data in background")
            vod_cats, vod_streams = load_vod_data()
            with get_cache_lock():
                get_cache()["vod_categories"] = vod_cats
                get_cache()["vod_streams"] = vod_streams
            log.info("VOD data loaded")
        finally:
            get_refresh_in_progress().discard("vod_load")

    threading.Thread(target=load, daemon=True).start()


def _start_series_background_load() -> None:
    """Start background loading of series data if not already in progress."""
    if "series_load" in get_refresh_in_progress():
        return
    get_refresh_in_progress().add("series_load")

    def load():
        try:
            log.info("Loading series data in background")
            series_cats, series_list = load_series_data()
            with get_cache_lock():
                get_cache()["series_categories"] = series_cats
                get_cache()["series"] = series_list
            log.info("Series data loaded")
        finally:
            get_refresh_in_progress().discard("series_load")

    threading.Thread(target=load, daemon=True).start()


def _get_episode_desc(ep: ProviderRecord) -> str:
    """Extract description from episode info (handles str or dict)."""
    info = ep.get("info")
    if isinstance(info, str):
        try:
            info = json_loads(info)
        except ValueError:
            info = None
    if isinstance(info, dict):
        return str(_first_present(info, "plot", "description"))
    return str(_first_present(ep, "description", "plot"))


def _get_current_program(epg_id: str) -> CurrentProgram | None:
    """Current EPG program for a channel, or None when the guide has no entry."""
    if not epg_id:
        return None
    now = datetime.now(timezone.utc)
    programs = epg.get_programs_in_range(epg_id, now, now + timedelta(minutes=1))
    if not programs:
        return None
    program = programs[0]
    return {
        "title": program.title,
        "desc": program.desc,
        "start": program.start.timestamp(),
        "end": program.stop.timestamp(),
    }


def _find_live_stream(stream_id: str) -> ProviderRecord | None:
    return next(
        (s for s in get_cache().get("live_streams", []) if str(s.get("stream_id")) == stream_id),
        None,
    )


def _live_epg_id(stream: ProviderRecord) -> str:
    """A stream's EPG id, falling back to one assigned to it in a shared playlist."""
    epg_id = text_field(stream, "epg_channel_id")
    if epg_id:
        return epg_id
    key = playlists.stream_key(stream)
    streams = get_cache().get("live_streams", [])
    for playlist in playlists.load():
        for resolved in playlists.resolve(playlist, streams):
            resolved_epg_id = text_field(resolved, "epg_channel_id")
            if playlists.stream_key(resolved) == key and resolved_epg_id:
                return resolved_epg_id
    return ""


def _get_live_player_info(stream_id: str) -> PlayerInfo:
    """Get player info for live stream."""
    _ensure_live_cache()
    stream = _find_live_stream(stream_id)
    if stream is None:
        return PlayerInfo()

    info = PlayerInfo(
        channel_name=text_field(stream, "name"),
        catchup_days=catchup.archive_days(stream),
    )

    direct_url = text_field(stream, "direct_url")
    if direct_url:
        info = replace(info, url=direct_url, is_m3u=True)
    elif stream.get("source_type") == "xtream":
        info = replace(info, url=live_stream_url(stream))

    # Look up source settings.
    source_id = text_field(stream, "source_id")
    stream_category_ids = stream.get("category_ids")
    info = replace(
        info,
        source_id=source_id,
        category_ids=stream_category_ids if isinstance(stream_category_ids, list) else None,
    )
    if source_id:
        sources = load_server_settings()["sources"]
        source = next((s for s in sources if s.get("id") == source_id), None)
        if source is not None:
            info = replace(info, deinterlace_fallback=source.get("deinterlace_fallback", True))

    # Look up current program from EPG.
    program = _get_current_program(_live_epg_id(stream))
    if program is not None:
        info = replace(
            info,
            program_title=program["title"],
            program_desc=program["desc"],
            program_start=program["start"],
            program_end=program["end"],
        )
    return info


def _find_archived_program(stream: ProviderRecord, start: datetime) -> epg.Program | None:
    """The EPG program airing at ``start`` on a stream's channel."""
    epg_id = _live_epg_id(stream)
    if not epg_id:
        return None
    programs = epg.get_programs_in_range(
        epg_id,
        start,
        start + timedelta(seconds=1),
        text_field(stream, "source_id"),
    )
    return next((p for p in programs if p.start <= start < p.stop), None)


def _get_catchup_player_info(stream_id: str, start_ts: float) -> PlayerInfo:
    """Player info for an archived program, played from its start."""
    info = _get_live_player_info(stream_id)
    stream = _find_live_stream(stream_id) if info.url else None
    if stream is None:
        return PlayerInfo()
    try:
        start = datetime.fromtimestamp(start_ts, timezone.utc)
    except (OverflowError, OSError, ValueError):
        start = None
    if start is None or not catchup.can_play(stream, start):
        raise HTTPException(404, "This program is not available to watch from the start")

    program = _find_archived_program(stream, start)
    stop = program.stop if program else start + timedelta(hours=1)
    # Archives start on whole minutes; the player skips the remaining seconds.
    begin = start.replace(second=0, microsecond=0)
    source_id = text_field(stream, "source_id")
    client = get_xtream_client_by_source(source_id)
    tz = catchup.server_timezone(source_id, client.get_server_info) if client else timezone.utc
    return replace(
        info,
        url=catchup.timeshift_url(stream, begin, catchup.duration_minutes(begin, stop), tz),
        is_m3u=False,
        catchup_start=begin.timestamp(),
        catchup_seek=(start - begin).total_seconds(),
        program_title=program.title if program else "",
        program_desc=program.desc if program else "",
        program_start=program.start.timestamp() if program else begin.timestamp(),
        program_end=stop.timestamp(),
    )


def _get_movie_player_info(stream_id: str, ext: str) -> PlayerInfo:
    """Get player info for movie."""
    # Find movie in cache to get its source_id.
    cached_movie = next(
        (m for m in get_cache().get("vod_streams", []) if str(m.get("stream_id")) == stream_id),
        None,
    )

    # Get client for the movie's source (fall back to first if not found)
    source_id = "" if cached_movie is None else text_field(cached_movie, "source_id")
    xtream = get_xtream_client_by_source(source_id) if source_id else None
    if xtream is None:
        xtream = get_first_xtream_client()
    if xtream is None:
        return PlayerInfo()
    client = xtream

    ext = ext or "mkv"
    info = PlayerInfo(
        url=client.build_stream_url("movie", int(stream_id), ext),
        source_id=source_id,
    )

    cache_key = f"vod_info_{source_id}_{stream_id}"
    try:
        movie = get_cached_info(cache_key, lambda: client.get_vod_info(int(stream_id)))
    except (error.URLError, TimeoutError):
        return info
    m = record_field(movie, "info")
    if m:
        name = m.get("name", "")
        year_value = m.get("year")
        year = str(year_value if _present(year_value) else m.get("releasedate", ""))[:4]
        info = replace(
            info,
            channel_name=f"{name} ({year})" if year else str(name),
            program_desc=str(_first_present(m, "plot", "description")),
        )
    return info


def _get_series_player_info(
    stream_id: str,
    series_id: int | None,
    ext: str,
) -> tuple[PlayerInfo, str | None]:
    """Get player info for series episode. Returns (info, next_episode_url)."""
    # Find series in cache to get its source_id.
    source_id = ""
    if series_id is not None and series_id != 0:
        cached_series = next(
            (s for s in get_cache().get("series", []) if str(s.get("series_id")) == str(series_id)),
            None,
        )
        if cached_series is not None:
            source_id = text_field(cached_series, "source_id")

    # Get client for the series' source (fall back to first if not found)
    xtream = get_xtream_client_by_source(source_id) if source_id else None
    if xtream is None:
        xtream = get_first_xtream_client()
    if xtream is None:
        return PlayerInfo(), None
    client = xtream

    ext = ext or "mkv"
    info = PlayerInfo(
        url=client.build_stream_url("series", int(stream_id), ext),
        source_id=source_id,
    )

    if series_id is None or series_id == 0:
        return info, None
    requested_series = series_id

    cache_key = f"series_info_{source_id}_{series_id}"
    try:
        series = get_cached_info(cache_key, lambda: client.get_series_info(requested_series))
    except (error.URLError, TimeoutError) as e:
        log.warning("Failed to fetch series info %s: %s", series_id, e)
        return info, None
    if not series:
        return info, None

    series_info = record_field(series, "info")
    if series_info:
        name = series_info.get("name", "")
        year = series_info.get("year")
        info = replace(info, channel_name=f"{name} ({year})" if _present(year) else str(name))

    # Build flat list of all episodes in order (season, episode)
    all_episodes: list[tuple[int, ProviderRecord]] = []
    seasons = record_field(series, "episodes")
    for season_num, eps in sorted(seasons.items(), key=lambda x: int(x[0])):
        season_eps = [ep for ep in eps if isinstance(ep, dict)] if isinstance(eps, list) else []
        for ep in sorted(season_eps, key=lambda e: _int_value(e.get("episode_num"))):
            all_episodes.append((int(season_num), ep))

    # Find current episode and next.
    next_episode_url = None
    for i, (season_num, ep) in enumerate(all_episodes):
        if str(ep.get("id")) == stream_id:
            title = re.sub(r"^S\d+E\d+\s*-\s*", "", text_field(ep, "title"))
            if " - " in title:
                title = title.split(" - ")[-1]
            episode_num = _int_value(ep.get("episode_num"))
            info = replace(
                info,
                program_title=f"S{season_num:02d}E{episode_num:02d} -- {title.strip()}",
                program_desc=_get_episode_desc(ep),
            )
            # Get next episode URL.
            if i + 1 < len(all_episodes):
                _, next_ep = all_episodes[i + 1]
                next_ext = text_field(next_ep, "container_extension") or ext
                next_episode_url = (
                    f"/play/series/{next_ep['id']}?series_id={series_id}&ext={next_ext}"
                )
            break
    return info, next_episode_url


def _ensure_live_cache() -> None:
    """Ensure live streams and EPG are loaded."""
    if "live_streams" not in get_cache():
        cats, streams, epg_urls = load_all_live_data()
        with get_cache_lock():
            get_cache()["live_categories"] = cats
            get_cache()["live_streams"] = streams
            get_cache()["epg_urls"] = epg_urls
    if not epg.has_programs():
        with contextlib.suppress(Exception):
            load_all_epg(get_cache().get("epg_urls", []))


def _handoff_channel(
    stream_id: str,
    position: float,
    *,
    archive_lag_secs: float = 60.0,
) -> handoff.Channel | None:
    """A live channel and the program airing on it at ``position``."""
    info = _get_live_player_info(stream_id)
    stream = _find_live_stream(stream_id) if info.url else None
    if stream is None:
        return None
    channel = handoff.Channel(
        stream_id=stream_id,
        name=info.channel_name,
        url=info.url,
        source_id=info.source_id,
        catchup_days=info.catchup_days,
        program_title=info.program_title,
        program_start=info.program_start,
        program_end=info.program_end,
    )
    if math.isnan(position) or position < info.program_start or position >= info.program_end:
        program = _find_archived_program(stream, datetime.fromtimestamp(position, timezone.utc))
        channel = replace(
            channel,
            program_title=program.title if program else "",
            program_start=program.start.timestamp() if program else 0.0,
            program_end=program.stop.timestamp() if program else 0.0,
        )
    # Catchup only reaches back as far as the archive holds.
    if channel.catchup_days and not catchup.can_play(
        stream,
        datetime.fromtimestamp(position - archive_lag_secs, timezone.utc),
    ):
        channel = replace(channel, catchup_days=0)
    return channel


def _get_handoff_player_info(
    username: str,
    stream_id: str,
    handoff_id: str,
    start: float | None,
    *,
    archive_lag_secs: float = 60.0,
) -> PlayerInfo:
    """Player info resuming a handoff, from where it was left or from ``start``."""
    record = handoff.claim(username, handoff_id)
    if record is None or record["stream_id"] != stream_id:
        raise HTTPException(404, "This program is no longer available to resume")
    position = record["position"] if start is None else start
    if record["mode"] == "catchup":
        # Archives trail the broadcast by up to a minute.
        return _get_catchup_player_info(
            stream_id,
            min(position, time.time() - archive_lag_secs),
        )
    info = _get_live_player_info(stream_id)
    if not info.url:
        return info
    recording_start = record["recording_start"]
    return replace(
        info,
        url=f"/handoff/{record['recording_id']}/stream.m3u8",
        program_title=record["program_title"] or info.program_title,
        program_start=record["program_start"],
        program_end=record["program_end"],
        catchup_start=recording_start,
        catchup_seek=max(0.0, position - recording_start),
        handoff_recording=True,
    )


def _is_xtream_source_url(url: str, source: Source) -> bool:
    """Whether a playback URL points at this Xtream account."""
    base = source.url.rstrip("/")
    if not base or not url.startswith(base + "/"):
        return False
    parts = url[len(base) + 1 :].split("/")
    return not source.username or source.username in parts


# Groups are: - Live TV categories: cat:{category_id} displayed as "{source.name}:
# {category.name}" - Movies: movies:{source_id} displayed as "{source.name}: Movies" -
# Series: series:{source_id} displayed as "{source.name}: Series"
def _build_all_groups() -> list[dict[str, str]]:
    """Build list of all available groups for user restrictions."""
    groups = []
    sources_by_id = {s.id: s for s in get_sources()}

    # Live TV categories.
    for cat in get_cache().get("live_categories", []):
        source = sources_by_id.get(text_field(cat, "source_id"))
        if source:
            groups.append(
                {
                    "id": f"cat:{cat['category_id']}",
                    "name": f"{source.name}: {cat['category_name']}",
                    "type": "live",
                },
            )

    # Movies and Series for Xtream sources.
    for source in get_sources():
        if source.type == "xtream":
            groups.append(
                {
                    "id": f"movies:{source.id}",
                    "name": f"{source.name}: Movies",
                    "type": "movies",
                },
            )
            groups.append(
                {
                    "id": f"series:{source.id}",
                    "name": f"{source.name}: Series",
                    "type": "series",
                },
            )

    return groups


def _source_users(access: str, users: list[str] | None) -> list[str] | None:
    """Parse source assignment: None shares with everyone, a list limits to those users."""
    if access == "all":
        return None
    if access != "selected":
        raise HTTPException(400, "Invalid access option")
    known = set(auth.get_all_usernames())
    return sorted({u for u in users or [] if u in known})


async def _live_streams_and_category_names() -> tuple[list[ProviderRecord], dict[str, str]]:
    if "live_streams" not in get_cache():
        await asyncio.to_thread(_store_cached_live_data)
    streams = get_cache().get("live_streams", [])
    names = {
        str(c["category_id"]): text_field(c, "category_name")
        for c in get_cache().get("live_categories", [])
    }
    return streams, names


# Admins (username=None) see every entry, including unavailable ones; other users only
# see channels they are allowed to watch.
def _playlist_payload(
    playlist: playlists.Playlist,
    streams: list[ProviderRecord],
    category_names: dict[str, str],
    username: str | None = None,
) -> PlaylistPayload:
    """Playlist with resolved channels."""
    stream_allowed = _live_stream_filter(username) if username is not None else None
    channels: list[PlaylistChannelView] = []
    for entry in playlist.get("channels", []):
        stream = playlists.resolve_channel(entry, streams)
        if stream_allowed is not None and (stream is None or not stream_allowed(stream)):
            continue
        if stream is None:
            channels.append(_unavailable_channel(entry))
            continue
        summary = playlists.summarize(playlists.with_entry_epg(entry, stream), category_names)
        channels.append(
            {
                "stream_id": summary["stream_id"],
                "name": summary["name"],
                "source_id": summary["source_id"],
                "epg_channel_id": summary["epg_channel_id"],
                "icon": summary["icon"],
                "category": summary["category"],
                "available": True,
                # Only streams without their own EPG id take the entry's guide channel.
                "guide_assignable": not _present(stream.get("epg_channel_id")),
            },
        )
    return {
        "id": playlist["id"],
        "category_id": f"{playlists.PREFIX}{playlist['id']}",
        "name": playlist["name"],
        "channels": channels,
    }


def _unavailable_channel(entry: playlists.PlaylistChannel) -> PlaylistChannelView:
    """A saved entry whose stream is gone, keeping only the fields it was saved with."""
    view: PlaylistChannelView = {}
    if "stream_id" in entry:
        view["stream_id"] = entry["stream_id"]
    if "name" in entry:
        view["name"] = entry["name"]
    if "source_id" in entry:
        view["source_id"] = entry["source_id"]
    if "epg_channel_id" in entry:
        view["epg_channel_id"] = entry["epg_channel_id"]
    view["available"] = False
    return view


async def _json_body(request: Request, limit: int = 1024 * 1024) -> ProviderRecord:
    body = await request.body()
    if len(body) > limit:
        raise HTTPException(400, "Request too large")
    try:
        data = json_loads(body or b"{}")
    except ValueError as e:
        raise HTTPException(400, "Invalid JSON") from e
    if not isinstance(data, dict):
        raise HTTPException(400, "Expected a JSON object")
    return data


def _payload_viewer(user: TokenClaims) -> str | None:
    """None for admins (full view), else the username to apply restrictions for."""
    username = user["sub"]
    return None if auth.is_admin(username) else username


def _fetch_logo(url: str, timeout: int = 3, *, max_size: int = 1024 * 1024) -> tuple[bytes, str]:
    """Fetch logo synchronously. Returns (data, content_type)."""

    with safe_urlopen(url, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError("URL is not an image")
        data = resp.read(max_size)
        if len(data) >= max_size:
            raise ValueError("Image too large")
    return data, content_type


def _enrich_probe_cache_stats(
    stats: list[ffmpeg_command.SeriesProbeStats],
    xtream: XtreamClient,
) -> list[dict[str, object]]:
    """Copy probe cache stats, adding series and episode names (blocking)."""
    enriched: list[dict[str, object]] = []
    for entry in stats:
        name = entry["name"]
        episodes: list[dict[str, object]] = [{**ep} for ep in entry["episodes"]]
        series: ProviderRecord = {}
        if not name or episodes:
            cache_key = f"series_info_{entry['series_id']}"
            with contextlib.suppress(Exception):
                series = get_cached_info(
                    cache_key,
                    functools.partial(xtream.get_series_info, entry["series_id"]),
                )
        if series:
            info = record_field(series, "info")
            if not name and info:
                name = text_field(info, "name")
            ep_map = _episode_labels(series)
            for ep in episodes:
                ep_id = ep.get("episode_id")
                if isinstance(ep_id, int) and ep_id in ep_map:
                    ep["name"] = ep_map[ep_id]
        enriched.append({**entry, "name": name, "episodes": episodes})
    return enriched


def _episode_labels(series: ProviderRecord) -> dict[int, str]:
    """Map each episode id to an ``SxxEyy Title`` label."""
    labels: dict[int, str] = {}
    for season_num, eps in record_field(series, "episodes").items():
        for ep in eps if isinstance(eps, list) else []:
            if not isinstance(ep, dict) or not _present(ep.get("id")):
                continue
            title = re.sub(r"^S\d+E\d+\s*-\s*", "", text_field(ep, "title"))
            if " - " in title:
                title = title.split(" - ")[-1]
            episode_num = _int_value(ep.get("episode_num", 0))
            labels[_int_value(ep.get("id"))] = (
                f"S{int(season_num):02d}E{episode_num:02d} {title.strip()}"
            )
    return labels


def _handoff_stream_allowed(username: str, stream_id: str) -> bool:
    _ensure_live_cache()
    stream = _find_live_stream(stream_id)
    return stream is not None and _live_stream_filter(username)(stream)


class _ApiSettingsUpdate(TypedDict, total=False):
    """Server settings an administrator may change through ``/api/settings``."""

    transcode_mode: str
    transcode_hw: str
    vod_transcode_cache_mins: int
    probe_live: bool
    probe_movies: bool
    probe_series: bool
    vod_order: list[str]
    series_order: list[str]


class _UserPrefsUpdate(TypedDict, total=False):
    """Preferences a client may change through ``/api/user-prefs``."""

    cc_lang: str
    audio_lang: str
    cc_style: dict[str, MutablePlainTree]
    cast_host: str
    virtual_scroll: bool
    sort_live_by_views: bool
    guide_selected_cats: list[str] | None


def _raw_logo_url(url: str) -> str:
    """Return the external URL inside a ``/api/logo`` link, or ``url`` unchanged."""
    if not url.startswith("/api/logo?"):
        return url
    return parse.parse_qs(parse.urlsplit(url).query).get("url", [url])[0]


def _with_cover(entry: MutablePlainTree, cover: str) -> MutablePlainTree:
    """Copy a favorite entry with its cover replaced; other shapes pass through."""
    if not isinstance(entry, dict):
        return entry
    return {**entry, "cover": cover}


# Pages post back the covers they were given, which are signed ``/api/logo`` links.
# Storing the raw URL keeps saved favorites valid if ``secret_key`` changes.
def _stored_entry(entry: dict[str, MutablePlainTree]) -> MutablePlainTree:
    """One saved favorite: a text cover is unwrapped to the provider URL."""
    cover = entry.get("cover")
    return _with_cover(entry, _raw_logo_url(cover)) if isinstance(cover, str) else entry


# Covers come from the provider catalog, looked up by the favorite's id. The stored
# cover is never signed: viewers write favorites through ``/api/user-prefs``, so signing
# it would let any viewer proxy any URL. A favorite missing from the loaded catalog
# shows no cover.
def _page_favorites(settings: UserSettings) -> Favorites:
    """The viewer's favorites, with covers signed for ``/api/logo``."""
    favorites = settings.get("favorites", {"series": {}, "movies": {}})
    memory = get_cache()
    # Favorite kind -> (provider catalog, id field, cover field).
    catalogs: dict[str, tuple[list[ProviderRecord], str, str]] = {
        "movies": (memory.get("vod_streams", []), "stream_id", "stream_icon"),
        "series": (memory.get("series", []), "series_id", "cover"),
    }
    page: Favorites = {}
    for kind, entries in favorites.items():
        covers: dict[str, str] = {}
        if kind in catalogs and entries:
            items, id_field, cover_field = catalogs[kind]
            for item in items:
                item_id = text_field(item, id_field)
                if item_id in entries:
                    covers[item_id] = _logo_url_filter(text_field(item, cover_field))
        page[kind] = {
            fid: _with_cover(entry, covers.get(fid, "")) for fid, entry in entries.items()
        }
    return page


# Zero is unlimited.
def _stream_limits(username: str, source_id: str) -> tuple[int, int]:
    """The user's and the source's stream limits for ``source_id``."""
    if not source_id:
        return 0, 0
    user_max_streams = auth.get_user_limits(username).get("max_streams_per_source", {})
    source = next(
        (s for s in load_server_settings()["sources"] if s.get("id") == source_id),
        None,
    )
    source_max = 0 if source is None else source.get("max_streams", 0)
    return user_max_streams.get(source_id, 0), source_max


def _catalog_records(
    catalog: _Catalog,
    username: str,
    settings: UserSettings,
) -> tuple[list[ProviderRecord], list[ProviderRecord]]:
    """Categories and items the user may browse, after group access and their filter."""
    store = get_cache()
    if catalog.group == "movies":
        cats, items = store.get("vod_categories", []), store.get("vod_streams", [])
        wanted = settings.get("vod_filter", [])
    else:
        cats, items = store.get("series_categories", []), store.get("series", [])
        wanted = settings.get("series_filter", [])
    unavailable = _unavailable_groups(username)
    return (
        [c for c in cats if _browsable(c, catalog.group, unavailable, wanted)],
        [i for i in items if _browsable(i, catalog.group, unavailable, wanted)],
    )


def _browsable(
    record: ProviderRecord,
    group: str,
    unavailable: set[str],
    wanted: list[str],
) -> bool:
    """Whether the user's group access and category filter admit ``record``."""
    if f"{group}:{text_field(record, 'source_id')}" in unavailable:
        return False
    return not wanted or str(record.get("category_id")) in wanted


def _sorted_records(
    records: list[ProviderRecord],
    sort: str | None,
    newest_field: str,
) -> list[ProviderRecord]:
    """``records`` in browse order; an unknown sort keeps provider order."""
    if sort == "alpha":
        return sorted(records, key=lambda r: text_field(r, "name").lower())
    if sort == "rating":
        return sorted(records, key=lambda r: _safe_float(r.get("rating")), reverse=True)
    if sort == "newest":
        return sorted(records, key=lambda r: _int_value(r.get(newest_field)), reverse=True)
    return records


def _in_category(records: list[ProviderRecord], category: int) -> list[ProviderRecord]:
    return [r for r in records if str(r.get("category_id")) == str(category)]


# Only the first ``eager_rows`` carry their items.
#
# Later rows load on scroll, so a large catalog never renders in one page.
def _browse_rows(
    cats: list[ProviderRecord],
    items: list[ProviderRecord],
    sort: str | None,
    newest_field: str,
    *,
    row_size: int = 20,
    eager_rows: int = 4,
) -> list[_BrowseRow]:
    """One row per non-empty category."""
    by_category: dict[str, list[ProviderRecord]] = {}
    for item in items:
        by_category.setdefault(str(item.get("category_id")), []).append(item)
    rows: list[_BrowseRow] = []
    for cat in cats:
        category_id = str(cat.get("category_id"))
        members = by_category.pop(category_id, [])
        if members:
            row: _BrowseRow = {
                "category_id": category_id,
                "name": text_field(cat, "category_name"),
                "total": len(members),
                "records": None,
            }
            if len(rows) < eager_rows:
                row["records"] = _sorted_records(members, sort, newest_field)[:row_size]
            rows.append(row)
    # Items whose category is unknown cannot be fetched by category, so they render now.
    orphans = [item for members in by_category.values() for item in members]
    if orphans:
        rows.append(
            {
                "category_id": "",
                "name": "Other",
                "total": len(orphans),
                "records": _sorted_records(orphans, sort, newest_field)[:row_size],
            },
        )
    return rows


def _browse_page(
    request: Request,
    username: str,
    catalog: _Catalog,
    category: int | None,
    sort: str | None,
    *,
    row_size: int = 20,
    page_size: int = 120,
) -> HTMLResponse:
    """The catalog page: favorites, category rows, or one category's first page."""
    user_settings = load_user_settings(username)
    content_access = _get_content_access(username)
    if not content_access[catalog.group]:
        raise HTTPException(403, f"Access to {catalog.group} is restricted")
    cats, items = _catalog_records(catalog, username, user_settings)
    context: dict[str, object] = {
        "categories": cats,
        "current_category": category,
        "current_sort": sort,
        "favorites": _page_favorites(user_settings),
        "content_access": content_access,
        "cards_url": catalog.cards_url,
        "row_size": row_size,
        "page_size": page_size,
        "rows": None,
        "items": [],
        "total": 0,
    }
    if sort is not None and category is not None:
        members = _sorted_records(_in_category(items, category), sort, catalog.newest_field)
        context.update(items=members[:page_size], total=len(members))
    elif sort is not None:
        context["rows"] = _browse_rows(cats, items, sort, catalog.newest_field, row_size=row_size)
    return TEMPLATES.TemplateResponse(request, catalog.template, context)


def _browse_cards(
    request: Request,
    username: str,
    catalog: _Catalog,
    category: int,
    sort: str | None,
    offset: int,
    limit: int,
) -> HTMLResponse:
    """A slice of one category's cards, for rows and pages that load on scroll."""
    if not _get_content_access(username)[catalog.group]:
        raise HTTPException(403, f"Access to {catalog.group} is restricted")
    _, items = _catalog_records(catalog, username, load_user_settings(username))
    members = _sorted_records(_in_category(items, category), sort, catalog.newest_field)
    return TEMPLATES.TemplateResponse(
        request,
        catalog.cards_template,
        {"items": members[offset : offset + limit]},
        headers={"Cache-Control": "no-store"},
    )


# URLs embed provider credentials, so callers send only the labels to clients.
def _labeled_url_probes() -> list[tuple[str, _UrlProbe]]:
    """URL-keyed probes named by the movie or channel they belong to."""
    store = get_cache()
    movies = {str(m.get("stream_id")): text_field(m, "name") for m in store.get("vod_streams", [])}
    channels = store.get("live_streams", [])
    by_url = {text_field(c, "direct_url"): c for c in channels if c.get("direct_url") is not None}
    by_id = {str(c.get("stream_id")): c for c in channels}
    labeled: list[tuple[str, _UrlProbe]] = []
    for stats in ffmpeg_command.get_url_probe_cache_stats():
        url = stats["url"]
        parts = parse.urlparse(url).path.split("/")
        stem = parts[-1].split(".")[0]
        kind: Literal["movie", "live", "other"] = "other"
        stream_id = name = ""
        if (channel := by_url.get(url)) is not None:
            kind, stream_id, name = (
                "live",
                str(channel.get("stream_id")),
                text_field(channel, "name"),
            )
        elif len(parts) >= 5 and parts[-4] in ("movie", "live"):
            kind = "movie" if parts[-4] == "movie" else "live"
            stream_id = stem
            name = (
                movies.get(stem, "") if kind == "movie" else text_field(by_id.get(stem, {}), "name")
            )
        labeled.append(
            (
                url,
                {
                    "kind": kind,
                    "stream_id": stream_id,
                    "name": name,
                    "probed_at": stats["probed_at"],
                    "video_codec": stats["video_codec"],
                    "audio_codec": stats["audio_codec"],
                    "audio_langs": stats["audio_langs"],
                    "subtitle_count": stats["subtitle_count"],
                    "duration": stats["duration"],
                    "height": stats["height"],
                },
            ),
        )
    return labeled


class _FavoriteUpdate(TypedDict):
    kind: Literal["movies", "series"]
    id: str
    entry: dict[str, MutablePlainTree] | None  # None removes the favorite.


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
