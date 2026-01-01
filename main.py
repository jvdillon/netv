#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi", "uvicorn[standard]", "jinja2", "python-multipart", "cryptography", "defusedxml"]
# ///
"""IPTV Web App.

Usage:
    ./main.py [--port PORT] [--https] [--cert FILE --key FILE]

Options:
    --port PORT     Port to listen on (default: 8000)
    --https         Enable HTTPS using Let's Encrypt certs (auto-detect domain)
    --cert FILE     SSL certificate file (overrides --https)
    --key FILE      SSL private key file (overrides --https)

Examples:
    ./main.py                              # HTTP on port 8000
    ./main.py --https                      # HTTPS with auto-detected Let's Encrypt certs
    ./main.py --cert c.pem --key k.pem     # HTTPS with custom certs
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from xml.sax.saxutils import escape as xml_escape

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
import pathlib
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse

from auth import create_token, verify_password, verify_token
from cache import (
    AVAILABLE_ENCODERS,
    CACHE_DIR,
    Source,
    clear_all_caches,
    clear_all_file_caches,
    get_cache,
    get_cache_lock,
    get_cached_info,
    get_sources,
    get_watch_position,
    load_file_cache,
    load_server_settings,
    load_user_settings,
    refresh_encoders,
    save_file_cache,
    save_server_settings,
    save_user_settings,
    save_watch_position,
    update_source_epg_url,
)
from epg import fetch_epg
from m3u import (
    fetch_m3u,
    fetch_source_live_data,
    fetch_source_vod_data,
    get_first_xtream_client,
    get_refresh_in_progress,
    load_all_live_data,
    load_series_data,
    load_vod_data,
    parse_epg_urls,
)
from xtream import XtreamClient

import auth
import epg_db
import ffmpeg_command
import ffmpeg_session


log = logging.getLogger()

# SSE subscribers for EPG ready notifications (limit to prevent DoS)
_epg_subscribers: set[asyncio.Queue[str]] = set()
_shutdown_event: asyncio.Event | None = None  # Set during shutdown to close SSE
_MAX_SSE_SUBSCRIBERS = 100

# Login rate limiting: track failed attempts per IP
_login_attempts: dict[str, list[float]] = {}
_LOGIN_WINDOW = 300  # 5 minutes
_LOGIN_MAX_ATTEMPTS = 10


# =============================================================================
# App Setup
# =============================================================================

APP_DIR = pathlib.Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=APP_DIR / "templates")
TEMPLATES.env.auto_reload = True

# Thread locks for fetch operations
_fetch_locks: dict[str, threading.Lock] = {
    "live": threading.Lock(),
    "vod": threading.Lock(),
    "series": threading.Lock(),
    "epg": threading.Lock(),
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Clean up orphaned transcodes and preload data on startup."""
    # Initialize EPG database
    epg_db.init(CACHE_DIR)

    # Initialize transcoding module with settings callback
    ffmpeg_command.init(load_server_settings)

    # Kill orphaned ffmpeg processes
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ffmpeg.*iptv_transcode"],
            check=False,
            capture_output=True,
            text=True,
        )
        for pid in result.stdout.strip().split("\n"):
            if pid:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    log.info("Killed orphaned ffmpeg pid %s", pid)
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass
    # Clean up orphaned dirs and recover valid VOD sessions
    ffmpeg_session.cleanup_and_recover_sessions()

    # Preload all data in background threads (parallel)
    def load_live():
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

    def load_epg_data():
        try:
            epg_urls = get_cache().get("epg_urls", [])
            if epg_urls:
                load_all_epg(epg_urls)
                log.info("EPG data loaded: %d programs", epg_db.get_program_count())
                # Notify SSE subscribers
                for q in list(_epg_subscribers):
                    with contextlib.suppress(Exception):
                        q.put_nowait("epg_ready")
        except Exception as e:
            log.error("EPG load error: %s", e)

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
        load_live()
        # EPG needs epg_urls from live data, so run after
        load_epg_data()

    threading.Thread(target=load_all, daemon=True).start()
    threading.Thread(target=load_vod, daemon=True).start()
    threading.Thread(target=load_series, daemon=True).start()
    log.info("Preload started: live+EPG, VOD, series loading in parallel")

    # Periodic cleanup of expired sessions (VOD and live)
    cleanup_stop = threading.Event()

    def cleanup_loop():
        while not cleanup_stop.wait(60):  # Check every minute
            ffmpeg_session.cleanup_expired_sessions()

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

    # EPG scheduler
    scheduler_stop = threading.Event()
    _last_triggered: dict[str, str] = {}  # source_id -> last triggered time

    def scheduler_loop():
        while not scheduler_stop.wait(30):  # Check every 30 seconds
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            for source in get_sources():
                if current_time in source.epg_schedule:
                    key = f"{source.id}_epg"
                    # Only trigger once per scheduled time
                    if (
                        _last_triggered.get(source.id) != current_time
                        and key not in get_refresh_in_progress()
                    ):
                        log.info("Scheduled EPG refresh for %s at %s", source.name, current_time)
                        _last_triggered[source.id] = current_time
                        get_refresh_in_progress().add(key)

                        def do_refresh(src: Source = source, k: str = key):
                            try:
                                epg_url = None
                                if src.type == "xtream":
                                    client = XtreamClient(src.url, src.username, src.password)
                                    epg_url = client.epg_url
                                elif src.type == "m3u":
                                    _, _, epg_url = fetch_m3u(src.url, src.id)
                                elif src.type == "epg":
                                    epg_url = src.url
                                if epg_url:
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

    # Shutdown - signal SSE connections to close
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    _shutdown_event.set()
    cleanup_stop.set()
    scheduler_stop.set()
    ffmpeg_session.shutdown()


app = FastAPI(title="neTV", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


class AuthRequired(Exception):
    """Raised when authentication is required."""


@app.exception_handler(AuthRequired)
async def auth_required_handler(request: Request, _exc: AuthRequired):
    return RedirectResponse("/login", status_code=303)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Show nice HTML error pages for HTTP errors."""
    # Only handle HTML requests, let API requests get JSON
    accept = request.headers.get("accept", "")
    if "text/html" not in accept:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    return TEMPLATES.TemplateResponse(
        request,
        "error.html",
        {"title": f"Error {exc.status_code}", "message": exc.detail},
        status_code=exc.status_code,
    )


def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get("token")
    if not token:
        return None
    return verify_token(token)


def require_auth(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise AuthRequired
    return user


def require_admin(request: Request) -> dict:
    user = require_auth(request)
    username = user.get("sub", "")
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
    """Create the initial admin user."""
    if not auth.is_setup_required():
        return RedirectResponse("/login", status_code=303)
    # Validate
    if len(username) < 3:
        return TEMPLATES.TemplateResponse(
            request, "setup.html", {"error": "Username must be at least 3 characters"}
        )
    if len(password) < 8:
        return TEMPLATES.TemplateResponse(
            request, "setup.html", {"error": "Password must be at least 8 characters"}
        )
    if password != confirm:
        return TEMPLATES.TemplateResponse(
            request, "setup.html", {"error": "Passwords do not match"}
        )
    auth.create_user(username, password)
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str | None = None):
    """Login page - redirects to setup if no users exist."""
    if auth.is_setup_required():
        return RedirectResponse("/setup", status_code=303)
    last_user = request.cookies.get("last_user", "")
    return TEMPLATES.TemplateResponse(
        request, "login.html", {"error": error, "last_user": last_user}
    )


def _check_rate_limit(ip: str) -> None:
    """Check login rate limit. Raises HTTPException if exceeded."""
    now = time.time()
    attempts = _login_attempts.get(ip, [])
    # Clean old attempts for this IP
    attempts = [t for t in attempts if now - t < _LOGIN_WINDOW]
    if attempts:
        _login_attempts[ip] = attempts
    elif ip in _login_attempts:
        del _login_attempts[ip]
    # Periodically clean stale IPs (when dict is large)
    if len(_login_attempts) > 1000:
        stale = [k for k, v in _login_attempts.items() if not v or now - max(v) > _LOGIN_WINDOW]
        for k in stale[:100]:
            del _login_attempts[k]
    if len(attempts) >= _LOGIN_MAX_ATTEMPTS:
        raise HTTPException(429, "Too many login attempts, try again later")


@app.post("/login")
async def login(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    """Authenticate user and create session."""
    ip = request.client.host if request.client else "unknown"
    _check_rate_limit(ip)
    if not verify_password(username, password):
        _login_attempts.setdefault(ip, []).append(time.time())
        return RedirectResponse("/login?error=invalid", status_code=303)
    token = create_token({"sub": username})
    response = RedirectResponse("/", status_code=303)
    is_secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(
        "token", token, httponly=True, samesite="strict", max_age=86400 * 7, secure=is_secure
    )
    response.set_cookie("last_user", username, max_age=86400 * 365, secure=is_secure)
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
async def index(request: Request, _user: Annotated[dict, Depends(require_auth)]):
    return RedirectResponse("/guide", status_code=303)


def _fetch_all_epg(epg_urls: list[tuple[str, int, str]]) -> int:
    """Fetch EPG from all URLs into sqlite (in parallel). Returns total program count."""

    def fetch_one(url_timeout_source: tuple[str, int, str]) -> tuple[str, int]:
        url, timeout, source_id = url_timeout_source
        try:
            log.info("Fetching EPG (timeout=%ds): %s", timeout, url[:80])
            count = fetch_epg(url, CACHE_DIR, timeout=timeout, source_id=source_id)
            log.info("EPG done: %d programs from %s", count, url[:50])
            return url, count
        except Exception as e:
            log.error("EPG failed: %s - %s", url[:50], e)
            return url, 0

    total = 0
    max_workers = min(len(epg_urls) or 1, 8)  # Cap at 8 concurrent fetches
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_one, u) for u in epg_urls]
        for future in concurrent.futures.as_completed(futures):
            _, count = future.result()
            total += count
    log.info("EPG fetch complete: %d programs total", total)
    return total


def load_all_epg(epg_urls: list[tuple[str, int, str]]) -> None:
    """Load EPG into sqlite database if empty.

    Args:
        epg_urls: List of (url, timeout, source_id) tuples
    """
    if epg_db.has_programs():
        log.info("EPG database has %d programs", epg_db.get_program_count())
        return

    # No data - fetch synchronously
    with _fetch_locks["epg"]:
        if epg_db.has_programs():
            return
        log.info("No EPG data, fetching")
        try:
            _fetch_all_epg(epg_urls)
        except Exception as e:
            log.error("EPG fetch failed: %s", e)
            get_cache()["epg_error"] = str(e)


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


@app.get("/events/epg")
async def epg_events(_user: Annotated[dict, Depends(require_auth)]):
    """SSE endpoint - notifies when EPG is ready."""
    if len(_epg_subscribers) >= _MAX_SSE_SUBSCRIBERS:
        raise HTTPException(503, "Too many subscribers")
    queue: asyncio.Queue[str] = asyncio.Queue()
    _epg_subscribers.add(queue)

    async def event_stream():
        try:
            # If EPG already loaded, send immediately
            if epg_db.has_programs():
                yield "data: epg_ready\n\n"
                return
            # Wait for EPG ready event or shutdown
            while True:
                if _shutdown_event and _shutdown_event.is_set():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1)
                    yield f"data: {event}\n\n"
                    return
                except TimeoutError:
                    continue
        except TimeoutError:
            yield "data: timeout\n\n"
        finally:
            _epg_subscribers.discard(queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/guide", response_class=HTMLResponse)
async def guide_page(
    request: Request,
    user: Annotated[dict, Depends(require_auth)],
    offset: int = 0,  # hours offset from now
    cats: str = "",  # comma-separated category IDs
):
    # If no cats param, redirect to saved filter (per-user)
    username = user.get("sub", "")
    if not cats:
        user_settings = load_user_settings(username)
        saved = user_settings.get("guide_filter", [])
        if saved:
            return RedirectResponse(
                f"/guide?offset={offset}&cats={','.join(saved)}", status_code=303
            )

    # If no channel data in memory, try file cache first (async to avoid blocking)
    if "live_categories" not in get_cache() or "live_streams" not in get_cache():
        cached = await asyncio.to_thread(load_file_cache, "live_data")
        if cached:
            data, _ = cached
            with get_cache_lock():
                get_cache()["live_categories"] = data["cats"]
                get_cache()["live_streams"] = data["streams"]
                get_cache()["epg_urls"] = parse_epg_urls(data.get("epg_urls", []))
        else:
            # No cache at all - start background load and show loading page
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

    categories = get_cache()["live_categories"]
    all_streams = get_cache()["live_streams"]
    # EPG is optional - check sqlite db for data
    epg_loading = not epg_db.has_programs()

    # Parse selected category IDs (ordered list)
    ordered_cats: list[str] = []
    if cats:
        ordered_cats = [c.strip() for c in cats.split(",") if c.strip()]
    selected_cats = set(ordered_cats)

    # Get user's unavailable groups for filtering
    user_limits = auth.get_user_limits(username)
    unavailable_groups = set(user_limits.get("unavailable_groups", []))

    # Filter and sort streams by category order
    if selected_cats:
        cat_order = {c: i for i, c in enumerate(ordered_cats)}

        def stream_sort_key(s: dict) -> int:
            for c in s.get("category_ids") or []:
                if str(c) in cat_order:
                    return cat_order[str(c)]
            return len(ordered_cats)

        def stream_allowed(s: dict) -> bool:
            """Check if stream is allowed (not in any unavailable group)."""
            cat_ids = s.get("category_ids") or []
            # Stream is blocked if ANY of its categories are unavailable
            return not any(f"cat:{c}" in unavailable_groups for c in cat_ids)

        streams = [
            s
            for s in all_streams
            if any(str(c) in selected_cats for c in (s.get("category_ids") or []))
            and stream_allowed(s)
        ]
        streams.sort(key=stream_sort_key)
    else:
        streams = []

    # Build channel list with EPG IDs
    # Collect EPG IDs for batch query
    epg_ids = [s.get("epg_channel_id") or "" for s in streams]
    epg_ids_set = [e for e in epg_ids if e]

    # Batch fetch icons and programs
    icons_map = epg_db.get_icons_batch(epg_ids_set) if epg_ids_set else {}

    # Time window: 3 hours starting from offset
    now = datetime.now(UTC)
    window_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=offset)
    window_end = window_start + timedelta(hours=3)

    programs_map = (
        epg_db.get_programs_batch(epg_ids_set, window_start, window_end) if epg_ids_set else {}
    )

    # Log streams with EPG IDs that returned no programs (potential misconfiguration)
    for s, epg_id in zip(streams, epg_ids, strict=False):
        if epg_id and not programs_map.get(epg_id):
            log.warning(
                "Stream '%s' has epg_channel_id='%s' but no EPG programs found",
                s["name"],
                epg_id,
            )

    # Build channel list and grid data
    window_end_mobile = window_start + timedelta(hours=2)
    grid_data = []
    for s, epg_id in zip(streams, epg_ids, strict=False):
        icon = s.get("stream_icon", "") or icons_map.get(epg_id, "")
        ch = {
            "stream_id": s["stream_id"],
            "name": s["name"],
            "icon": icon,
            "epg_id": epg_id,
        }
        row = {"channel": ch, "programs": [], "programs_mobile": []}
        for p in programs_map.get(epg_id, []):
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
                    "left_pct": left_pct,
                    "width_pct": width_pct,
                }
            )
            # Mobile: 2-hour window
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
                    }
                )
        grid_data.append(row)

    # Time markers (every 30 min) - convert to local time for display
    time_markers = []
    for i in range(7):  # 0, 30, 60, 90, 120, 150, 180 minutes
        t = window_start + timedelta(minutes=i * 30)
        t_local = t.astimezone()  # Convert to local timezone
        time_markers.append(
            {
                "label": t_local.strftime("%H:%M"),
                "left_pct": (i * 30 / 180) * 100,
            }
        )

    # Mobile time markers (2 hour window instead of 3)
    time_markers_mobile = []
    for i in range(5):  # 0, 30, 60, 90, 120 minutes
        t = window_start + timedelta(minutes=i * 30)
        t_local = t.astimezone()
        time_markers_mobile.append(
            {
                "label": t_local.strftime("%H:%M"),
                "left_pct": (i * 30 / 120) * 100,
            }
        )

    return TEMPLATES.TemplateResponse(
        request,
        "guide.html",
        {
            "categories": categories,
            "selected_cats": selected_cats,
            "cats_param": cats,
            "grid_data": grid_data,
            "time_markers": time_markers,
            "time_markers_mobile": time_markers_mobile,
            "offset": offset,
            "window_start": window_start.strftime("%Y-%m-%d %H:%M"),
            "epg_error": get_cache().get("epg_error"),
            "epg_loading": epg_loading,
            "channel_count": len(grid_data),
            "loading": False,
            "content_access": _get_content_access(username),
        },
    )


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


@app.get("/vod", response_class=HTMLResponse)
async def vod_page(
    request: Request,
    user: Annotated[dict, Depends(require_auth)],
    category: int | None = None,
    sort: str | None = None,
):
    # Load from file cache if not in memory (async to avoid blocking)
    if "vod_categories" not in get_cache() or "vod_streams" not in get_cache():
        cached = await asyncio.to_thread(load_file_cache, "vod_data")
        if cached:
            data, _ = cached
            get_cache()["vod_categories"] = data["cats"]
            get_cache()["vod_streams"] = data["streams"]
        else:
            # No cache - start background load and show loading page
            _start_vod_background_load()
            username = user.get("sub", "")
            user_settings = load_user_settings(username)
            return TEMPLATES.TemplateResponse(
                request,
                "vod.html",
                {
                    "categories": [],
                    "streams": [],
                    "current_category": category,
                    "current_sort": sort,
                    "loading": True,
                    "favorites": user_settings.get("favorites", {"series": {}, "movies": {}}),
                },
            )

    username = user.get("sub", "")
    user_settings = load_user_settings(username)

    # Check if user has access to any movies
    content_access = _get_content_access(username)
    if not content_access["movies"]:
        raise HTTPException(403, "Access to movies is restricted")

    # Get user's unavailable groups for filtering
    user_limits = auth.get_user_limits(username)
    unavailable_groups = set(user_limits.get("unavailable_groups", []))

    # Filter by group access (movies:{source_id})
    def movie_allowed(s: dict) -> bool:
        source_id = s.get("source_id", "")
        return f"movies:{source_id}" not in unavailable_groups

    streams = [s for s in get_cache()["vod_streams"] if movie_allowed(s)]
    categories = [
        c
        for c in get_cache()["vod_categories"]
        if f"movies:{c.get('source_id', '')}" not in unavailable_groups
    ]

    # Filter by category if specified
    if category:
        streams = [s for s in streams if str(s.get("category_id")) == str(category)]

    # Sort
    if sort == "alpha":
        streams.sort(key=lambda s: (s.get("name") or "").lower())
    elif sort == "rating":
        streams.sort(key=lambda s: float(s.get("rating") or 0), reverse=True)
    elif sort == "newest":
        streams.sort(key=lambda s: int(s.get("added") or 0), reverse=True)

    return TEMPLATES.TemplateResponse(
        request,
        "vod.html",
        {
            "categories": categories,
            "streams": streams,
            "current_category": category,
            "current_sort": sort,
            "favorites": user_settings.get("favorites", {"series": {}, "movies": {}}),
            "content_access": _get_content_access(username),
        },
    )


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


@app.get("/series", response_class=HTMLResponse)
async def series_page(
    request: Request,
    user: Annotated[dict, Depends(require_auth)],
    category: int | None = None,
    sort: str | None = None,
):
    # Load from file cache if not in memory (async to avoid blocking)
    if "series_categories" not in get_cache() or "series" not in get_cache():
        cached = await asyncio.to_thread(load_file_cache, "series_data")
        if cached:
            data, _ = cached
            get_cache()["series_categories"] = data["cats"]
            get_cache()["series"] = data["series"]
        else:
            # No cache - start background load and show loading page
            _start_series_background_load()
            username = user.get("sub", "")
            user_settings = load_user_settings(username)
            return TEMPLATES.TemplateResponse(
                request,
                "series.html",
                {
                    "categories": [],
                    "series": [],
                    "current_category": category,
                    "current_sort": sort,
                    "loading": True,
                    "favorites": user_settings.get("favorites", {"series": {}, "movies": {}}),
                },
            )

    username = user.get("sub", "")
    user_settings = load_user_settings(username)

    # Check if user has access to any series
    content_access = _get_content_access(username)
    if not content_access["series"]:
        raise HTTPException(403, "Access to series is restricted")

    # Get user's unavailable groups for filtering
    user_limits = auth.get_user_limits(username)
    unavailable_groups = set(user_limits.get("unavailable_groups", []))

    # Filter by group access (series:{source_id})
    def series_allowed(s: dict) -> bool:
        source_id = s.get("source_id", "")
        return f"series:{source_id}" not in unavailable_groups

    series = [s for s in get_cache()["series"] if series_allowed(s)]
    categories = [
        c
        for c in get_cache()["series_categories"]
        if f"series:{c.get('source_id', '')}" not in unavailable_groups
    ]

    # Filter by category if specified
    if category:
        series = [s for s in series if str(s.get("category_id")) == str(category)]

    # Sort
    if sort == "alpha":
        series.sort(key=lambda s: (s.get("name") or "").lower())
    elif sort == "rating":
        series.sort(key=lambda s: float(s.get("rating") or 0), reverse=True)
    elif sort == "newest":
        series.sort(key=lambda s: int(s.get("last_modified") or 0), reverse=True)

    return TEMPLATES.TemplateResponse(
        request,
        "series.html",
        {
            "categories": categories,
            "series": series,
            "current_category": category,
            "current_sort": sort,
            "favorites": user_settings.get("favorites", {"series": {}, "movies": {}}),
            "content_access": _get_content_access(username),
        },
    )


@app.get("/series/{series_id}", response_class=HTMLResponse)
async def series_detail_page(
    request: Request,
    series_id: int,
    user: Annotated[dict, Depends(require_auth)],
    refresh: bool = False,
):
    username = user.get("sub", "")

    # Check access for this specific series
    if "series" in get_cache():
        cached_series = next(
            (s for s in get_cache()["series"] if str(s.get("series_id")) == str(series_id)),
            None,
        )
        if cached_series:
            source_id = cached_series.get("source_id", "")
            user_limits = auth.get_user_limits(username)
            unavailable_groups = set(user_limits.get("unavailable_groups", []))
            if f"series:{source_id}" in unavailable_groups:
                raise HTTPException(403, "Access to this series is restricted")

    xtream = get_first_xtream_client()
    if not xtream:
        raise HTTPException(404, "No Xtream source configured")
    cache_key = f"series_info_{series_id}"
    try:
        series_data = await asyncio.to_thread(
            get_cached_info, cache_key, lambda: xtream.get_series_info(series_id), refresh
        )
    except (urllib.error.URLError, TimeoutError) as e:
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
    # Extract year from releaseDate if not present
    if series_data.get("info"):
        info = series_data["info"]
        if not info.get("year") and info.get("releaseDate"):
            info["year"] = info["releaseDate"][:4]
    # Strip redundant series title and episode numbers from episode titles
    if series_data.get("episodes"):
        for season_eps in series_data["episodes"].values():
            for ep in season_eps:
                if ep.get("title"):
                    # Remove patterns like "Series Name - S01E01 - Episode Title"
                    # Keep only the actual episode title
                    title = ep["title"]
                    # Remove S##E## - pattern
                    title = re.sub(r"^S\d+E\d+\s*-\s*", "", title)
                    # Remove any leading "SeriesName - " pattern
                    if " - " in title and len(title.split(" - ")) > 1:
                        parts = title.split(" - ")
                        # Take the last part which should be the actual episode title
                        title = parts[-1]
                    ep["title"] = title.strip()

                # Parse info field if it's JSON
                if ep.get("info"):
                    if isinstance(ep["info"], str):
                        try:
                            info_obj = json.loads(ep["info"])
                            # Extract plot/description from parsed JSON
                            if isinstance(info_obj, dict):
                                ep["description"] = (
                                    info_obj.get("plot") or info_obj.get("description") or ""
                                )
                        except (json.JSONDecodeError, TypeError):
                            pass
                    elif isinstance(ep["info"], dict):
                        # Already a dict
                        ep["description"] = (
                            ep["info"].get("plot") or ep["info"].get("description") or ""
                        )

    username = user.get("sub", "")
    user_settings = load_user_settings(username)
    return TEMPLATES.TemplateResponse(
        request,
        "series_detail.html",
        {
            "series": series_data,
            "series_id": series_id,
            "favorites": user_settings.get("favorites", {"series": {}, "movies": {}}),
        },
    )


@app.get("/movie/{stream_id}", response_class=HTMLResponse)
async def movie_detail_page(
    request: Request,
    stream_id: int,
    user: Annotated[dict, Depends(require_auth)],
):
    username = user.get("sub", "")

    # Load from file cache if not in memory
    if "vod_streams" not in get_cache():
        vod_cats, vod_streams = load_vod_data()
        get_cache()["vod_categories"] = vod_cats
        get_cache()["vod_streams"] = vod_streams

    vod_streams = get_cache().get("vod_streams", [])
    movie = next((m for m in vod_streams if m.get("stream_id") == stream_id), None)

    # Check access for this specific movie
    if movie:
        source_id = movie.get("source_id", "")
        user_limits = auth.get_user_limits(username)
        unavailable_groups = set(user_limits.get("unavailable_groups", []))
        if f"movies:{source_id}" in unavailable_groups:
            raise HTTPException(403, "Access to this movie is restricted")

    # Fetch detailed movie info
    if movie:
        xtream = get_first_xtream_client()
        if xtream:
            cache_key = f"vod_info_{stream_id}"
            try:
                vod_info = await asyncio.to_thread(
                    get_cached_info, cache_key, lambda: xtream.get_vod_info(stream_id)
                )
            except (urllib.error.URLError, TimeoutError):
                vod_info = {}
            if vod_info and vod_info.get("info"):
                info = vod_info["info"]
                # Merge detailed info into movie object
                movie = {**movie}  # Copy
                movie["plot"] = info.get("plot") or info.get("description", "")
                movie["director"] = info.get("director", "")
                movie["cast"] = info.get("cast") or info.get("actors", "")
                movie["genre"] = info.get("genre", "")
                movie["rating"] = info.get("rating", "")
                movie["year"] = info.get("releasedate", "")[:4] if info.get("releasedate") else ""
                movie["duration"] = info.get("duration", "")
                movie["cover_big"] = info.get("cover_big") or info.get("movie_image", "")
                movie["youtube_trailer"] = info.get("youtube_trailer", "")

    username = user.get("sub", "")
    user_settings = load_user_settings(username)
    return TEMPLATES.TemplateResponse(
        request,
        "movie_detail.html",
        {
            "movie": movie,
            "favorites": user_settings.get("favorites", {"series": {}, "movies": {}}),
        },
    )


@dataclass(slots=True)
class PlayerInfo:
    """Info needed to render the player page."""

    url: str = ""
    is_m3u: bool = False
    channel_name: str = ""
    program_title: str = ""
    program_desc: str = ""
    deinterlace_fallback: bool = True  # Used when probe is skipped
    source_id: str = ""  # Source ID for stream limit tracking
    category_ids: list[str] | None = None  # Category IDs for live streams (access check)


def _get_episode_desc(ep: dict) -> str:
    """Extract description from episode info (handles str or dict)."""
    info = ep.get("info")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except (json.JSONDecodeError, TypeError):
            info = None
    if isinstance(info, dict):
        return info.get("plot") or info.get("description") or ""
    return ep.get("description") or ep.get("plot") or ""


def _get_live_player_info(stream_id: str) -> PlayerInfo:
    """Get player info for live stream."""
    _ensure_live_cache()
    stream = next(
        (s for s in get_cache()["live_streams"] if str(s.get("stream_id")) == stream_id),
        None,
    )
    if not stream:
        return PlayerInfo()

    info = PlayerInfo(channel_name=stream.get("name", ""))

    if stream.get("direct_url"):
        info.url = stream["direct_url"]
        info.is_m3u = True
    elif stream.get("source_type") == "xtream":
        base, user, pwd = stream["source_url"], stream["source_username"], stream["source_password"]
        orig_id = stream_id.split("_")[-1] if "_" in stream_id else stream_id
        info.url = f"{base}/live/{user}/{pwd}/{orig_id}.m3u8"

    # Look up source settings
    source_id = stream.get("source_id", "")
    info.source_id = source_id
    info.category_ids = stream.get("category_ids")
    if source_id:
        sources = load_server_settings().get("sources", [])
        source = next((s for s in sources if s.get("id") == source_id), None)
        if source:
            info.deinterlace_fallback = source.get("deinterlace_fallback", True)

    # Look up current program from EPG
    epg_id = stream.get("epg_channel_id") or ""
    if epg_id:
        now = datetime.now(UTC)
        programs = epg_db.get_programs_in_range(epg_id, now, now + timedelta(minutes=1))
        if programs:
            info.program_title, info.program_desc = programs[0].title, programs[0].desc
    return info


def _get_movie_player_info(stream_id: str, ext: str) -> PlayerInfo:
    """Get player info for movie."""
    xtream = get_first_xtream_client()
    if not xtream:
        return PlayerInfo()

    ext = ext or "mkv"
    info = PlayerInfo(url=xtream.build_stream_url("movie", int(stream_id), ext))

    # Get source_id from cached movie data for access control
    if "vod_streams" in get_cache():
        cached_movie = next(
            (m for m in get_cache()["vod_streams"] if str(m.get("stream_id")) == str(stream_id)),
            None,
        )
        if cached_movie:
            info.source_id = cached_movie.get("source_id", "")

    cache_key = f"vod_info_{stream_id}"
    try:
        movie = get_cached_info(cache_key, lambda: xtream.get_vod_info(int(stream_id)))
    except (urllib.error.URLError, TimeoutError):
        return info
    if movie and movie.get("info"):
        m = movie["info"]
        name = m.get("name", "")
        year = str(m.get("year") or m.get("releasedate", ""))[:4]
        info.channel_name = f"{name} ({year})" if year else name
        info.program_desc = m.get("plot") or m.get("description") or ""
    return info


def _get_series_player_info(
    stream_id: str, series_id: int | None, ext: str
) -> tuple[PlayerInfo, str | None]:
    """Get player info for series episode. Returns (info, next_episode_url)."""
    xtream = get_first_xtream_client()
    if not xtream:
        return PlayerInfo(), None

    ext = ext or "mkv"
    info = PlayerInfo(url=xtream.build_stream_url("series", int(stream_id), ext))

    # Get source_id from cached series data for access control
    if series_id and "series" in get_cache():
        cached_series = next(
            (s for s in get_cache()["series"] if str(s.get("series_id")) == str(series_id)),
            None,
        )
        if cached_series:
            info.source_id = cached_series.get("source_id", "")
            log.info("Series %s found in cache, source_id=%s", series_id, info.source_id)
        else:
            log.warning(
                "Series %s not found in cache (cache has %d entries)",
                series_id,
                len(get_cache()["series"]),
            )
    else:
        log.warning(
            "Series lookup skipped: series_id=%s, 'series' in cache=%s",
            series_id,
            "series" in get_cache(),
        )

    if not series_id:
        return info, None

    cache_key = f"series_info_{series_id}"
    try:
        series = get_cached_info(cache_key, lambda: xtream.get_series_info(series_id))
    except (urllib.error.URLError, TimeoutError) as e:
        log.warning("Failed to fetch series info %s: %s", series_id, e)
        return info, None
    if not series:
        return info, None

    if series.get("info"):
        name = series["info"].get("name", "")
        year = series["info"].get("year", "")
        info.channel_name = f"{name} ({year})" if year else name

    # Build flat list of all episodes in order (season, episode)
    all_episodes: list[tuple[int, dict]] = []
    for season_num, eps in sorted((series.get("episodes") or {}).items(), key=lambda x: int(x[0])):
        for ep in sorted(eps, key=lambda e: int(e.get("episode_num", 0))):
            all_episodes.append((int(season_num), ep))

    # Find current episode and next
    next_episode_url = None
    for i, (season_num, ep) in enumerate(all_episodes):
        if str(ep.get("id")) == str(stream_id):
            title = re.sub(r"^S\d+E\d+\s*-\s*", "", ep.get("title", ""))
            if " - " in title:
                title = title.split(" - ")[-1]
            info.program_title = (
                f"S{int(season_num):02d}E{int(ep.get('episode_num', 0)):02d} — {title.strip()}"
            )
            info.program_desc = _get_episode_desc(ep)
            # Get next episode URL
            if i + 1 < len(all_episodes):
                _, next_ep = all_episodes[i + 1]
                next_ext = next_ep.get("container_extension") or ext
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
    if not epg_db.has_programs():
        with contextlib.suppress(Exception):
            load_all_epg(get_cache().get("epg_urls", []))


@app.get("/play/{stream_type}/{stream_id:path}", response_class=HTMLResponse)
async def player_page(
    request: Request,
    stream_type: str,
    stream_id: str,
    user: Annotated[dict, Depends(require_auth)],
    ext: str = "",
    series_id: int | None = None,
):
    """Render player page for live/movie/series stream."""
    username = user.get("sub", "")
    next_episode_url = None
    if stream_type == "live":
        info = await asyncio.to_thread(_get_live_player_info, stream_id)
    elif stream_type == "movie":
        info = await asyncio.to_thread(_get_movie_player_info, stream_id, ext)
    elif stream_type == "series":
        info, next_episode_url = await asyncio.to_thread(
            _get_series_player_info, stream_id, series_id, ext
        )
    else:
        raise HTTPException(404, "Invalid stream type")

    if not info.url:
        raise HTTPException(404, "Stream not found")

    # Check user's group access
    user_limits = auth.get_user_limits(username)
    unavailable_groups = set(user_limits.get("unavailable_groups", []))
    log.info(
        "Access check: user=%s type=%s source_id=%s unavailable=%s",
        username,
        stream_type,
        info.source_id,
        unavailable_groups,
    )
    if unavailable_groups:
        if stream_type == "live" and info.category_ids:
            # Live streams: blocked if any category is unavailable
            if any(f"cat:{cat_id}" in unavailable_groups for cat_id in info.category_ids):
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

    server_settings = load_server_settings()
    user_settings = load_user_settings(username)
    transcode_mode = server_settings.get("transcode_mode", "auto")
    needs_transcode = info.is_m3u or ext in ("mkv", "mp4", "avi", "wmv", "flv")
    if transcode_mode != "never" and needs_transcode:
        transcode_mode = "always"

    # Get saved watch position for VOD (per-user)
    resume_position = 0.0
    if stream_type in ("movie", "series"):
        watch_entry = get_watch_position(username, info.url)
        if watch_entry:
            resume_position = watch_entry.get("position", 0.0)

    # For series, stream_id is episode_id
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
            "transcode_mode": transcode_mode,
            "stream_type": stream_type,
            "channel_name": info.channel_name,
            "program_title": info.program_title,
            "program_desc": info.program_desc,
            "captions_enabled": user_settings.get("captions_enabled", False),
            "resume_position": resume_position,
            "series_id": series_id,
            "episode_id": episode_id,
            "series_name": series_name,
            "cc_lang": user_settings.get("cc_lang", ""),
            "cc_style": user_settings.get("cc_style", {}),
            "cast_host": user_settings.get("cast_host", ""),
            "next_episode_url": next_episode_url,
            "deinterlace_fallback": info.deinterlace_fallback,
            "source_id": info.source_id,
            "content_access": _get_content_access(username),
        },
    )


@app.get("/search", response_class=HTMLResponse)
async def search_page(
    request: Request,
    user: Annotated[dict, Depends(require_auth)],
    q: str = "",
    regex: bool = False,
    live: bool = False,
    vod: bool = False,
    series: bool = False,
):
    results: dict[str, list] = {"live": [], "vod": [], "series": []}

    # Default all on if none specified
    if not live and not vod and not series:
        live = vod = series = True

    if q:
        if regex:
            # Limit regex length to prevent ReDoS
            if len(q) > 100:
                raise HTTPException(400, "Regex pattern too long")
            try:
                pattern = re.compile(q, re.IGNORECASE)

                def match_fn(name: str) -> bool:
                    try:
                        # Timeout via match limit - search only first 1000 chars
                        return pattern.search(name[:1000]) is not None
                    except Exception:
                        return False
            except re.error:

                def match_fn(name: str) -> bool:
                    return False
        else:
            q_lower = q.lower()

            def match_fn(name: str) -> bool:
                return q_lower in name.lower()

        # Load live data (run in thread to avoid blocking)
        if live:
            if "live_streams" not in get_cache():
                cats, streams, epg_urls = await asyncio.to_thread(load_all_live_data)
                with get_cache_lock():
                    get_cache()["live_categories"] = cats
                    get_cache()["live_streams"] = streams
                    get_cache()["epg_urls"] = epg_urls
            results["live"] = [
                s for s in get_cache()["live_streams"] if match_fn(s.get("name") or "")
            ][:20]

        # Load VOD data (run in thread to avoid blocking)
        if vod:
            if "vod_streams" not in get_cache():
                vod_cats, vod_streams = await asyncio.to_thread(load_vod_data)
                with get_cache_lock():
                    get_cache()["vod_categories"] = vod_cats
                    get_cache()["vod_streams"] = vod_streams
            results["vod"] = [
                s for s in get_cache()["vod_streams"] if match_fn(s.get("name") or "")
            ][:20]

        # Load series data (run in thread to avoid blocking)
        if series:
            if "series" not in get_cache():
                series_cats, series_list = await asyncio.to_thread(load_series_data)
                with get_cache_lock():
                    get_cache()["series_categories"] = series_cats
                    get_cache()["series"] = series_list
            results["series"] = [s for s in get_cache()["series"] if match_fn(s.get("name") or "")][
                :20
            ]

    username = user.get("sub", "")
    user_settings = load_user_settings(username)

    # Filter results based on user access
    user_limits = auth.get_user_limits(username)
    unavailable_groups = set(user_limits.get("unavailable_groups", []))

    # Filter live results by category access
    results["live"] = [
        s
        for s in results["live"]
        if not any(
            f"cat:{cat_id}" in unavailable_groups for cat_id in (s.get("category_ids") or [])
        )
    ]
    # Filter movie results by source access
    results["vod"] = [
        s for s in results["vod"] if f"movies:{s.get('source_id', '')}" not in unavailable_groups
    ]
    # Filter series results by source access
    results["series"] = [
        s for s in results["series"] if f"series:{s.get('source_id', '')}" not in unavailable_groups
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
            "favorites": user_settings.get("favorites", {"series": {}, "movies": {}}),
            "content_access": content_access,
        },
    )


@app.get("/stream/{stream_type}/{stream_id}")
async def stream_redirect(
    stream_type: str,
    stream_id: int,
    _user: Annotated[dict, Depends(require_auth)],
    ext: str = "",
):
    xtream = get_first_xtream_client()
    if not xtream:
        raise HTTPException(404, "No Xtream source configured")
    url = xtream.build_stream_url(stream_type, stream_id, ext)
    return RedirectResponse(url, status_code=302)


@app.get("/playlist.xspf")
async def playlist_xspf(
    _user: Annotated[dict, Depends(require_auth)],
    url: str,
):
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
    user: Annotated[dict, Depends(require_auth)],
    url: str,
    content_type: str = "live",  # "movie", "series", or "live"
    series_id: int | None = None,
    episode_id: int | None = None,
    series_name: str = "",
    deinterlace_fallback: str = "1",  # "1" or "0"
    source_id: str = "",
):
    """Start a transcode session, return session ID."""
    deinterlace_fb = deinterlace_fallback == "1"
    username = user.get("sub", "")

    # Get user limits for this source
    user_limits = auth.get_user_limits(username)
    max_streams_per_source = user_limits.get("max_streams_per_source", {})
    user_max_streams = max_streams_per_source.get(source_id, 0) if source_id else 0

    # Get source max_streams (global limit for this source)
    source_max_streams = 0
    if source_id:
        settings = load_server_settings()
        sources = settings.get("sources", [])
        source = next((s for s in sources if s.get("id") == source_id), None)
        if source:
            source_max_streams = source.get("max_streams", 0)

    return await ffmpeg_session.start_transcode(
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
    )


@app.get("/transcode/seek/{session_id}")
async def transcode_seek(
    session_id: str,
    time: float,
    _user: Annotated[dict, Depends(require_auth)],
):
    """Seek VOD transcode to a new position."""
    return await ffmpeg_session.seek_transcode(session_id, time)


@app.get("/transcode/progress/{session_id}")
async def transcode_progress(
    session_id: str,
    _user: Annotated[dict, Depends(require_auth)],
):
    """Get transcode progress (segment count, duration)."""
    progress = ffmpeg_session.get_session_progress(session_id)
    if not progress:
        raise HTTPException(404, "Session not found")
    return progress


@app.get("/transcode/{session_id}/{filename}")
async def transcode_file(
    request: Request,
    session_id: str,
    filename: str,
):
    """Serve HLS playlist or segments (no auth - session IDs are unguessable)."""
    # Prevent path traversal
    safe_filename = pathlib.Path(filename).name
    if safe_filename != filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")

    session = ffmpeg_session.get_session(session_id)
    if not session:
        log.debug(f"[CAST] 404 session not found: {session_id}")
        raise HTTPException(404, "Transcode session not found")

    file_path = pathlib.Path(session["dir"]) / safe_filename
    if not file_path.exists():
        log.debug(f"[CAST] 404 file not found: {file_path}")
        raise HTTPException(404, "File not found")

    # Log Chromecast requests
    ua = request.headers.get("user-agent", "")
    if "CrKey" in ua or "Chromecast" in ua.lower() or "cast" in ua.lower():
        log.debug(f"[CAST] Chromecast request: {filename} UA={ua[:80]}")

    cors = {"Access-Control-Allow-Origin": "*"}
    if filename.endswith(".m3u8"):
        content = file_path.read_text()
        return Response(
            content=content,
            media_type="application/vnd.apple.mpegurl",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", **cors},
        )
    if filename.endswith(".vtt"):
        content = file_path.read_text()
        return Response(content=content, media_type="text/vtt", headers=cors)
    return FileResponse(file_path, media_type="video/mp2t", headers=cors)


@app.get("/subs/{session_id}/{filename}")
async def subtitle_file(session_id: str, filename: str):
    """Serve VTT subtitle files (no auth - session IDs are unguessable)."""
    # Prevent path traversal
    safe_filename = pathlib.Path(filename).name
    if safe_filename != filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    if not safe_filename.endswith(".vtt"):
        raise HTTPException(400, "Only VTT files allowed")
    session = ffmpeg_session.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    file_path = pathlib.Path(session["dir"]) / safe_filename
    # Wait briefly for file, return empty VTT if not ready (client will poll again)
    for _ in range(15):  # 3 seconds
        if file_path.exists() and file_path.stat().st_size > 20:
            break
        await asyncio.sleep(0.2)
    try:
        content = file_path.read_text() if file_path.exists() else "WEBVTT\n\n"
    except (UnicodeDecodeError, OSError):
        # File may be partially written or corrupted
        content = "WEBVTT\n\n"
    return Response(
        content=content,
        media_type="text/vtt",
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.delete("/transcode/{session_id}")
async def transcode_stop(
    session_id: str,
    _user: Annotated[dict, Depends(require_auth)],
):
    """Stop a transcode session (VOD sessions stay cached)."""
    ffmpeg_session.stop_session(session_id, force=False)
    return {"status": "stopped"}


@app.post("/transcode/{session_id}/stop")
async def transcode_stop_post(
    session_id: str,
    _user: Annotated[dict, Depends(require_auth)],
):
    """Stop a transcode session (POST for sendBeacon, VOD cached)."""
    ffmpeg_session.stop_session(session_id, force=False)
    return {"status": "stopped"}


@app.delete("/transcode-clear")
async def transcode_clear(
    url: str,
    _user: Annotated[dict, Depends(require_auth)],
):
    """Force-delete any cached transcode session for a URL."""
    session_id = ffmpeg_session.clear_url_session(url)
    if session_id:
        ffmpeg_session.stop_session(session_id, force=True)
        log.info("Force-cleared transcode session %s for URL", session_id)
    return {"status": "cleared", "session_id": session_id}


def _build_all_groups() -> list[dict[str, str]]:
    """Build list of all available groups for user restrictions.

    Groups are:
    - Live TV categories: cat:{category_id} displayed as "{source.name}: {category.name}"
    - Movies: movies:{source_id} displayed as "{source.name}: Movies"
    - Series: series:{source_id} displayed as "{source.name}: Series"
    """
    groups = []
    sources_by_id = {s.id: s for s in get_sources()}

    # Live TV categories
    for cat in get_cache().get("live_categories", []):
        source_id = cat.get("source_id", "")
        source = sources_by_id.get(source_id)
        if source:
            groups.append(
                {
                    "id": f"cat:{cat['category_id']}",
                    "name": f"{source.name}: {cat['category_name']}",
                    "type": "live",
                }
            )

    # Movies and Series for Xtream sources
    for source in get_sources():
        if source.type == "xtream":
            groups.append(
                {
                    "id": f"movies:{source.id}",
                    "name": f"{source.name}: Movies",
                    "type": "movies",
                }
            )
            groups.append(
                {
                    "id": f"series:{source.id}",
                    "name": f"{source.name}: Series",
                    "type": "series",
                }
            )

    return groups


def _get_content_access(username: str) -> dict[str, bool]:
    """Check if user has access to movies/series from any source.

    Returns dict with 'movies' and 'series' booleans.
    If no xtream sources exist, access is granted (nothing to restrict).
    """
    user_limits = auth.get_user_limits(username)
    unavailable_groups = set(user_limits.get("unavailable_groups", []))

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


# Register template global for content access check
TEMPLATES.env.globals["get_content_access"] = _get_content_access


def _get_content_access_from_request(request: Request) -> dict[str, bool]:
    """Get content access from request (for use in base template)."""
    token = request.cookies.get("token")
    if not token:
        return {"movies": True, "series": True}  # Not logged in, show all
    payload = verify_token(token)
    if not payload:
        return {"movies": True, "series": True}
    username = payload.get("sub", "")
    if not username:
        return {"movies": True, "series": True}
    return _get_content_access(username)


TEMPLATES.env.globals["get_content_access_from_request"] = _get_content_access_from_request


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: Annotated[dict, Depends(require_auth)]):
    username = user.get("sub", "")
    is_admin = auth.is_admin(username)
    server_settings = load_server_settings()
    user_settings = load_user_settings(username)
    # Load categories (from file cache or trigger background load)
    if "live_categories" not in get_cache():
        cached = await asyncio.to_thread(load_file_cache, "live_data")
        if cached:
            data, _ = cached
            with get_cache_lock():
                get_cache()["live_categories"] = data["cats"]
                get_cache()["live_streams"] = data["streams"]
                get_cache()["epg_urls"] = parse_epg_urls(data.get("epg_urls", []))
        else:
            # No cache - start background load
            _start_guide_background_load()

    # Filter live categories based on user's unavailable groups
    user_limits = auth.get_user_limits(username)
    unavailable_groups = set(user_limits.get("unavailable_groups", []))
    all_live_cats = get_cache().get("live_categories", [])
    live_categories = [
        cat for cat in all_live_cats if f"cat:{cat['category_id']}" not in unavailable_groups
    ]

    return TEMPLATES.TemplateResponse(
        request,
        "settings.html",
        {
            # Server settings
            "sources": server_settings.get("sources", []),
            "transcode_mode": server_settings.get("transcode_mode", "auto"),
            "transcode_hw": server_settings.get("transcode_hw", "nvidia"),
            "max_resolution": server_settings.get("max_resolution", "1080p"),
            "quality": server_settings.get("quality", "high"),
            "vod_transcode_cache_mins": server_settings.get("vod_transcode_cache_mins", 60),
            "live_transcode_cache_secs": server_settings.get("live_transcode_cache_secs", 60),
            "live_dvr_mins": server_settings.get("live_dvr_mins", 0),
            "transcode_dir": server_settings.get("transcode_dir", ""),
            "probe_live": server_settings.get("probe_live", True),
            "probe_movies": server_settings.get("probe_movies", True),
            "probe_series": server_settings.get("probe_series", False),
            "user_agent_preset": server_settings.get("user_agent_preset", "default"),
            "user_agent_custom": server_settings.get("user_agent_custom", ""),
            "available_encoders": AVAILABLE_ENCODERS,
            "all_users": auth.get_users_with_admin(),
            "all_groups": _build_all_groups(),
            "current_user": username,
            "is_admin": is_admin,
            # User settings
            "captions_enabled": user_settings.get("captions_enabled", False),
            "live_categories": live_categories,
            "selected_cats": user_settings.get("guide_filter", []),
            "cc_lang": user_settings.get("cc_lang", ""),
            "cc_style": user_settings.get("cc_style", {}),
            "cast_host": user_settings.get("cast_host", ""),
            "content_access": _get_content_access(username),
        },
    )


@app.post("/settings/guide-filter")
async def settings_guide_filter(
    request: Request,
    user: Annotated[dict, Depends(require_auth)],
):
    username = user.get("sub", "")
    data = await request.json()
    cats = data.get("cats", [])
    if not isinstance(cats, list) or len(cats) > 500:
        raise HTTPException(400, "Invalid filter list")
    user_settings = load_user_settings(username)
    user_settings["guide_filter"] = cats
    save_user_settings(username, user_settings)
    return {"status": "ok"}


@app.post("/settings/add")
async def settings_add_source(
    _user: Annotated[dict, Depends(require_admin)],
    name: Annotated[str, Form()],
    source_type: Annotated[str, Form()],
    url: Annotated[str, Form()],
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    epg_timeout: Annotated[int, Form()] = 120,
    epg_schedule: Annotated[str, Form()] = "",
    epg_enabled: Annotated[str, Form()] = "",  # Checkbox: "on" if checked
    deinterlace_fallback: Annotated[str, Form()] = "",  # Checkbox: "on" if checked
    max_streams: Annotated[int, Form()] = 0,
):
    # Validate inputs
    if not name or not name.strip():
        raise HTTPException(400, "Name is required")
    if source_type not in ("xtream", "m3u", "epg"):
        raise HTTPException(400, "Invalid source type")
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise HTTPException(400, "URL must use http or https")
    if len(name) > 200:
        raise HTTPException(400, "Name too long")

    # Parse schedule times
    schedule_list = []
    for t in epg_schedule.split(","):
        t = t.strip()
        if t and re.match(r"^\d{1,2}:\d{2}$", t):
            schedule_list.append(t.zfill(5))

    settings = load_server_settings()
    sources = settings.get("sources", [])
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
        }
    )
    settings["sources"] = sources
    save_server_settings(settings)
    clear_all_caches()
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/edit/{source_id}")
async def settings_edit_source(
    source_id: str,
    _user: Annotated[dict, Depends(require_admin)],
    name: Annotated[str, Form()],
    source_type: Annotated[str, Form()],
    url: Annotated[str, Form()],
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    epg_timeout: Annotated[int, Form()] = 120,
    epg_schedule: Annotated[str, Form()] = "",
    epg_enabled: Annotated[str, Form()] = "",  # Checkbox: "on" if checked
    epg_url: Annotated[str, Form()] = "",
    deinterlace_fallback: Annotated[str, Form()] = "",  # Checkbox: "on" if checked
    max_streams: Annotated[int, Form()] = 0,
):
    # Validate inputs
    if not name or not name.strip():
        raise HTTPException(400, "Name is required")
    if source_type not in ("xtream", "m3u", "epg"):
        raise HTTPException(400, "Invalid source type")
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise HTTPException(400, "URL must use http or https")
    if len(name) > 200:
        raise HTTPException(400, "Name too long")

    # Parse schedule times (comma-separated HH:MM)
    schedule_list = []
    for t in epg_schedule.split(","):
        t = t.strip()
        if t and re.match(r"^\d{1,2}:\d{2}$", t):
            schedule_list.append(t.zfill(5))  # Normalize to HH:MM

    settings = load_server_settings()
    for s in settings.get("sources", []):
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
            break
    save_server_settings(settings)
    clear_all_caches()
    return {"ok": True}


@app.post("/settings/delete/{source_id}")
async def settings_delete_source(
    source_id: str,
    _user: Annotated[dict, Depends(require_admin)],
):
    settings = load_server_settings()
    settings["sources"] = [s for s in settings.get("sources", []) if s["id"] != source_id]
    save_server_settings(settings)
    # Clear all caches including EPG data for this source
    epg_db.clear_source(source_id)
    clear_all_file_caches()
    return RedirectResponse("/settings", status_code=303)


@app.get("/guide/refresh")
async def guide_refresh(_user: Annotated[dict, Depends(require_auth)]):
    """Refresh guide data in background (stale-while-revalidate)."""

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
                epg_db.clear()
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

    # Set flags before starting threads to avoid race with status polling
    if "live_refresh" not in get_refresh_in_progress():
        get_refresh_in_progress().add("live_refresh")
        threading.Thread(target=refresh_live, daemon=True).start()
    if "epg_refresh" not in get_refresh_in_progress():
        get_refresh_in_progress().add("epg_refresh")
        threading.Thread(target=refresh_epg, daemon=True).start()
    return RedirectResponse("/guide?refreshing=1", status_code=303)


@app.get("/guide/refresh-status")
async def guide_refresh_status(_user: Annotated[dict, Depends(require_auth)]):
    """Return refresh status for polling."""
    return {
        "live": "live_refresh" in get_refresh_in_progress(),
        "epg": "epg_refresh" in get_refresh_in_progress(),
    }


@app.post("/settings/refresh/{source_id}/{refresh_type}")
async def settings_refresh_source(
    source_id: str,
    refresh_type: str,
    _user: Annotated[dict, Depends(require_admin)],
):
    """Refresh a specific data type for a single source."""
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
                # Update cache by replacing this source's data
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
                    new_epg = existing_epg + ([(epg_url, timeout, source_id)] if epg_url else [])
                    get_cache()["live_categories"] = new_cats
                    get_cache()["live_streams"] = new_streams
                    get_cache()["epg_urls"] = new_epg
                # Save to file cache
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
                    "Refreshing EPG for source: %s (timeout=%ds)", source.name, source.epg_timeout
                )
                epg_url = source.epg_url or (source.url if source.type == "epg" else "")
                if epg_url:
                    epg_db.clear_source(source_id)
                    count = _fetch_all_epg([(epg_url, source.epg_timeout, source_id)])
                    log.info("EPG refresh complete for %s: %d programs", source.name, count)
                else:
                    log.warning("No EPG URL for source: %s", source.name)

            elif refresh_type == "vod" and source.type == "xtream":
                log.info("Refreshing VOD for source: %s", source.name)
                cats, streams = fetch_source_vod_data(source)
                # For now, VOD is single-source, so just replace entirely
                with get_cache_lock():
                    get_cache().pop("vod_categories", None)
                    get_cache().pop("vod_streams", None)
                for f in CACHE_DIR.glob("vod_data*.json"):
                    f.unlink(missing_ok=True)
                save_file_cache("vod_data", {"cats": cats, "streams": streams})
                log.info(
                    "VOD refresh complete for %s: %d cats, %d streams",
                    source.name,
                    len(cats),
                    len(streams),
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
async def settings_refresh_status(_user: Annotated[dict, Depends(require_auth)]):
    """Return per-source refresh status."""
    statuses: dict[str, Any] = {}
    for key in list(get_refresh_in_progress()):
        if "_" in key:
            # Format: source_id_type (e.g., "ota_epg" or "src_123_epg")
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                source_id, rtype = parts
                statuses.setdefault(source_id, {})[rtype] = True
    # Report global guide_load as affecting all sources
    if "guide_load" in get_refresh_in_progress():
        statuses["_global"] = {"live": True, "epg": True}
    return statuses


@app.post("/settings/captions")
async def settings_captions(
    user: Annotated[dict, Depends(require_auth)],
    enabled: Annotated[str, Form()] = "",
):
    username = user.get("sub", "")
    user_settings = load_user_settings(username)
    user_settings["captions_enabled"] = enabled == "on"
    save_user_settings(username, user_settings)
    return {"ok": True}


@app.post("/api/cast-log")
async def cast_log_endpoint(request: Request):
    """Log cast events from client (debug mode only)."""
    if log.isEnabledFor(logging.DEBUG):
        body = await request.body()
        # Sanitize: limit length, single line, printable chars only
        msg = body.decode("utf-8", errors="replace")[:2048]
        msg = "".join(c if c.isprintable() and c != "\n" else "?" for c in msg)
        log.debug(f"[CAST] {msg}")
    return {"ok": True}


@app.get("/api/user-prefs")
async def get_user_prefs(user: Annotated[dict, Depends(require_auth)]):
    """Get user preferences (favorites, cc_lang, cc_style, cast_host)."""
    username = user.get("sub", "")
    settings = load_user_settings(username)
    return {
        "favorites": settings.get("favorites", {}),
        "cc_lang": settings.get("cc_lang", ""),
        "cc_style": settings.get("cc_style", {}),
        "cast_host": settings.get("cast_host", ""),
    }


@app.post("/api/user-prefs")
async def save_user_prefs(
    request: Request,
    user: Annotated[dict, Depends(require_auth)],
):
    """Save user preferences (partial update)."""
    username = user.get("sub", "")
    body = await request.body()
    if len(body) > 64 * 1024:  # 64KB limit
        raise HTTPException(400, "Request too large")
    data = json.loads(body)
    settings = load_user_settings(username)
    for key in ("favorites", "cc_lang", "cc_style", "cast_host"):
        if key in data:
            settings[key] = data[key]
    save_user_settings(username, settings)
    return {"ok": True}


@app.post("/settings/transcode")
async def settings_transcode(
    _user: Annotated[dict, Depends(require_admin)],
    transcode_mode: Annotated[str, Form()],
    transcode_hw: Annotated[str, Form()],
    max_resolution: Annotated[str, Form()] = "1080p",
    quality: Annotated[str, Form()] = "high",
    vod_transcode_cache_mins: Annotated[int, Form()] = 60,
    live_transcode_cache_secs: Annotated[int, Form()] = 0,
    live_dvr_mins: Annotated[int, Form()] = 0,
    transcode_dir: Annotated[str, Form()] = "",
    probe_live: Annotated[str | None, Form()] = None,
    probe_movies: Annotated[str | None, Form()] = None,
    probe_series: Annotated[str | None, Form()] = None,
):
    settings = load_server_settings()
    settings["transcode_mode"] = transcode_mode
    settings["transcode_hw"] = transcode_hw
    settings["max_resolution"] = max_resolution
    settings["quality"] = quality if quality in ("high", "medium", "low") else "high"
    settings["vod_transcode_cache_mins"] = max(0, vod_transcode_cache_mins)
    settings["live_transcode_cache_secs"] = max(0, live_transcode_cache_secs)
    settings["live_dvr_mins"] = max(0, live_dvr_mins)
    if transcode_dir:
        settings["transcode_dir"] = transcode_dir
    elif "transcode_dir" in settings:
        del settings["transcode_dir"]  # Use default
    settings["probe_live"] = probe_live == "on"
    settings["probe_movies"] = probe_movies == "on"
    settings["probe_series"] = probe_series == "on"
    save_server_settings(settings)
    return {"ok": True}


@app.post("/settings/refresh-encoders")
async def settings_refresh_encoders(
    _user: Annotated[dict, Depends(require_admin)],
):
    """Re-detect available hardware encoders."""
    encoders = refresh_encoders()
    return {"ok": True, "encoders": encoders}


@app.post("/settings/user-agent")
async def settings_user_agent(
    _user: Annotated[dict, Depends(require_admin)],
    preset: Annotated[str, Form()],
    custom: Annotated[str, Form()] = "",
):
    valid_presets = {"default", "vlc", "chrome", "tivimate", "custom"}
    if preset not in valid_presets:
        preset = "default"
    settings = load_server_settings()
    settings["user_agent_preset"] = preset
    settings["user_agent_custom"] = custom
    save_server_settings(settings)
    return {"ok": True}


def _enrich_probe_cache_stats(stats: list[dict], xtream: Any) -> list[dict]:
    """Enrich probe cache stats with series/episode names (blocking)."""
    for entry in stats:
        series: dict | None = None
        if not entry.get("name") or entry.get("episodes"):
            cache_key = f"series_info_{entry['series_id']}"
            with contextlib.suppress(Exception):
                series = get_cached_info(
                    cache_key, lambda sid=entry["series_id"]: xtream.get_series_info(sid)
                )
        if series:
            if not entry.get("name") and series.get("info"):
                entry["name"] = series["info"].get("name", "")
            ep_map: dict[int, str] = {}
            for season_num, eps in (series.get("episodes") or {}).items():
                for ep in eps:
                    eid = ep.get("id")
                    if eid:
                        ep_num = ep.get("episode_num", 0)
                        title = ep.get("title", "")
                        title = re.sub(r"^S\d+E\d+\s*-\s*", "", title)
                        if " - " in title:
                            title = title.split(" - ")[-1]
                        ep_map[int(eid)] = (
                            f"S{int(season_num):02d}E{int(ep_num):02d} {title.strip()}"
                        )
            for ep in entry.get("episodes", []):
                ep_id = ep.get("episode_id")
                if ep_id in ep_map:
                    ep["name"] = ep_map[ep_id]
    return stats


@app.get("/settings/probe-cache")
async def get_probe_cache(
    _user: Annotated[dict, Depends(require_auth)],
    response: Response,
):
    """Get probe cache stats for settings UI."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    stats = ffmpeg_command.get_series_probe_cache_stats()
    xtream = get_first_xtream_client()
    if not xtream:
        return {"series": stats}
    stats = await asyncio.to_thread(_enrich_probe_cache_stats, stats, xtream)
    return {"series": stats}


@app.post("/settings/probe-cache/clear")
async def clear_probe_cache(_user: Annotated[dict, Depends(require_admin)]):
    """Clear all probe caches."""
    count = ffmpeg_command.clear_all_probe_cache()
    return {"ok": True, "cleared": count}


@app.post("/settings/probe-cache/clear/{series_id}")
async def clear_series_probe_cache(
    series_id: int,
    _user: Annotated[dict, Depends(require_admin)],
    episode_id: int | None = None,
):
    """Clear probe cache for a specific series or episode."""
    ffmpeg_command.invalidate_series_probe_cache(series_id, episode_id)
    return {"ok": True}


@app.post("/settings/probe-cache/clear-mru/{series_id}")
async def clear_series_mru(
    series_id: int,
    _user: Annotated[dict, Depends(require_admin)],
):
    """Clear only the MRU for a series, keeping episode cache intact."""
    ffmpeg_command.clear_series_mru(series_id)
    return {"ok": True}


@app.post("/settings/data-cache/clear")
async def clear_data_cache(_user: Annotated[dict, Depends(require_admin)]):
    """Clear all data file caches (live, VOD, series) and memory cache."""
    count = clear_all_file_caches()
    return {"ok": True, "cleared": count}


@app.get("/api/settings")
async def get_settings_api(_user: Annotated[dict, Depends(require_auth)]):
    return load_server_settings()


@app.post("/api/settings")
async def update_settings_api(
    request: Request,
    _user: Annotated[dict, Depends(require_admin)],
):
    data = await request.json()
    # Whitelist allowed keys - never allow users/secret_key to be overwritten
    allowed_keys = {
        "transcode_mode",
        "transcode_hw",
        "vod_transcode_cache_mins",
        "probe_live",
        "probe_movies",
        "probe_series",
        "vod_order",
        "series_order",
    }
    settings = load_server_settings()
    for key in allowed_keys:
        if key in data:
            settings[key] = data[key]
    save_server_settings(settings)
    return {"status": "ok"}


@app.post("/api/watch-position")
async def save_watch_position_api(
    request: Request,
    user: Annotated[dict, Depends(require_auth)],
):
    """Save watch position for a stream (per-user)."""
    username = user.get("sub", "")
    data = await request.json()
    url = data.get("url", "")
    position = float(data.get("position", 0))
    duration = float(data.get("duration", 0))
    if url and position >= 0:
        save_watch_position(username, url, position, duration)
    return {"status": "ok"}


@app.get("/api/watch-position")
async def get_watch_position_api(
    user: Annotated[dict, Depends(require_auth)],
    url: str,
):
    """Get watch position for a stream (per-user)."""
    username = user.get("sub", "")
    entry = get_watch_position(username, url)
    if entry:
        return {"position": entry.get("position", 0), "duration": entry.get("duration", 0)}
    return {"position": 0, "duration": 0}


# User management endpoints
@app.post("/settings/users/delete/{username}")
async def settings_delete_user(
    username: str,
    user: Annotated[dict, Depends(require_auth)],
    password: Annotated[str, Form()] = "",
):
    """Delete a user. Self-deletion requires password. Other users require admin."""
    current_user = user.get("sub", "")
    if username == current_user:
        if not password or not auth.verify_password(username, password):
            raise HTTPException(400, "Password required to delete your own account")
        auth.delete_user(username)
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie("token")
        return response
    # Deleting other users requires admin
    if not auth.is_admin(current_user):
        raise HTTPException(403, "Admin access required")
    if not auth.delete_user(username):
        raise HTTPException(404, "User not found")
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/users/add")
async def settings_add_user(
    _user: Annotated[dict, Depends(require_admin)],
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    admin: Annotated[str, Form()] = "",
    max_streams_per_source: Annotated[str | None, Form()] = None,
    unavailable_groups: Annotated[str | None, Form()] = None,
):
    """Add a new user."""
    username = username.strip()
    if not username or len(username) < 2:
        raise HTTPException(400, "Username must be at least 2 characters")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if username in auth.get_all_usernames():
        raise HTTPException(400, "User already exists")
    auth.create_user(username, password, admin=admin == "on")
    # Apply limits if provided
    parsed_max_streams = None
    if max_streams_per_source:
        with contextlib.suppress(json.JSONDecodeError):
            parsed_max_streams = json.loads(max_streams_per_source)
    parsed_unavailable = None
    if unavailable_groups:
        with contextlib.suppress(json.JSONDecodeError):
            parsed_unavailable = json.loads(unavailable_groups)
    if parsed_max_streams or parsed_unavailable:
        auth.set_user_limits(username, parsed_max_streams, parsed_unavailable)
    return {"status": "ok"}


@app.post("/settings/users/password")
async def settings_change_own_password(
    user: Annotated[dict, Depends(require_auth)],
    current_password: Annotated[str, Form()],
    new_password: Annotated[str, Form()],
):
    """Change own password. Requires current password verification."""
    username = user.get("sub", "")
    if not auth.verify_password(username, current_password):
        raise HTTPException(400, "Current password is incorrect")
    if len(new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if not auth.change_password(username, new_password):
        raise HTTPException(404, "User not found")
    return {"status": "ok"}


@app.post("/settings/users/password/{target_user}")
async def settings_change_password(
    target_user: str,
    user: Annotated[dict, Depends(require_auth)],
    new_password: Annotated[str, Form()],
):
    """Change a user's password. Own password or admin required."""
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
    _user: Annotated[dict, Depends(require_admin)],
    admin: Annotated[str, Form()] = "",
):
    """Set admin status for a user."""
    if not auth.set_admin(target_user, admin == "on"):
        raise HTTPException(404, "User not found")
    return {"status": "ok"}


@app.post("/settings/users/limits/{target_user}")
async def settings_set_user_limits(
    target_user: str,
    _user: Annotated[dict, Depends(require_admin)],
    max_streams_per_source: Annotated[str | None, Form()] = None,  # JSON object string
    unavailable_groups: Annotated[str | None, Form()] = None,  # JSON array string
):
    """Set stream limits and group restrictions for a user."""
    parsed_max_streams = None
    if max_streams_per_source is not None:
        try:
            parsed_max_streams = json.loads(max_streams_per_source)
            if not isinstance(parsed_max_streams, dict):
                raise HTTPException(400, "max_streams_per_source must be a JSON object")
        except json.JSONDecodeError as err:
            raise HTTPException(400, "Invalid JSON for max_streams_per_source") from err

    parsed_unavailable = None
    if unavailable_groups is not None:
        try:
            parsed_unavailable = json.loads(unavailable_groups)
            if not isinstance(parsed_unavailable, list):
                raise HTTPException(400, "unavailable_groups must be a JSON array")
        except json.JSONDecodeError as err:
            raise HTTPException(400, "Invalid JSON for unavailable_groups") from err

    if not auth.set_user_limits(target_user, parsed_max_streams, parsed_unavailable):
        raise HTTPException(404, "User not found")
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse

    import uvicorn  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser(description="IPTV Web App")
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
    args = parser.parse_args()

    # LOG_LEVEL env var takes precedence, then --debug flag, then default INFO
    log_level_env = os.environ.get("LOG_LEVEL", "").upper()
    if log_level_env in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        log_level = getattr(logging, log_level_env)
    else:
        log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    ssl_args = {}
    if args.cert and args.key:
        ssl_args = {"ssl_certfile": args.cert, "ssl_keyfile": args.key}
    elif args.https is not None:
        live_dir = pathlib.Path("/etc/letsencrypt/live")
        if args.https:
            domain = args.https
        else:
            # Auto-detect first domain
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
        ssl_args = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}

    uv_log = "debug" if args.debug else "info"
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        access_log=args.debug,
        log_level=uv_log,
        log_config=None,  # preserve our basicConfig
        timeout_graceful_shutdown=5,
        **ssl_args,  # pyright: ignore[reportArgumentType]
    )
