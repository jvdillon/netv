"""Fakes for neTV's process, network, thread, task, and clock boundaries.

Tests install these where neTV leaves the process -- ffmpeg launches,
``subprocess.run``/``Popen``, HTTP fetches, background threads and tasks, and
time -- and keep every neTV function between those seams real.

Each ``install_*`` helper replaces one module's ``asyncio``/``subprocess``/
``request``/``threading``/``time`` global with a view of the real stdlib module
that overrides only the faked names, so the event loop and pytest keep the
real ones. Example::

    launcher = ProcessLauncher()
    install_launcher(monkeypatch, launcher, ffmpeg_session)
    ...
    assert launcher.commands[0][0] == "ffmpeg"
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from email.message import Message
from types import FunctionType, ModuleType, TracebackType
from typing import ClassVar, Self, TypeVar, override
from urllib import error, request

import asyncio
import hashlib
import importlib
import inspect
import json
import os
import queue
import subprocess
import threading
import time

from pychromecast.error import PyChromecastError

import pytest


_T = TypeVar("_T")


class FakeStream:
    """An asyncio ``StreamReader`` that yields scripted lines, then EOF."""

    def __init__(self, lines: Iterable[bytes] = ()) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""

    async def read(self, n: int = -1) -> bytes:
        del n
        data = b"".join(self._lines)
        self._lines.clear()
        return data


def probed_video_json() -> str:
    """What ffprobe reports for a sample holding a video frame size and no audio."""
    return '{"streams": [{"codec_type": "video", "codec_name": "mpeg2video", "height": 1080}]}'


def probe_capture(name: str) -> str:
    """What ``ffprobe -show_streams -show_format`` reported for a captured source.

    ``ota_*`` are samples from a live HDHomeRun tuner taken as the relay takes them
    (from the first keyframe); ``iptv_*`` are files made with ffmpeg. Only the keys
    netv reads are kept.

    Args:
        name: Capture name, such as ``"ota_1080i_mpeg2"``.

    Returns:
        report: The probe's JSON text.
    """
    captures = {
        "iptv_1080p_h264_aac": {
            "streams": [
                {
                    "index": 0,
                    "codec_name": "h264",
                    "profile": "Constrained Baseline",
                    "codec_type": "video",
                    "height": 1080,
                    "pix_fmt": "yuv420p",
                    "field_order": "progressive",
                },
                {
                    "index": 1,
                    "codec_name": "aac",
                    "profile": "LC",
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 2,
                    "bit_rate": "107599",
                },
            ],
            "format": {"format_name": "mpegts", "duration": "1.021333", "bit_rate": "15668307"},
        },
        "iptv_2160p_hevc_hdr": {
            "streams": [
                {
                    "index": 0,
                    "codec_name": "hevc",
                    "profile": "Main 10",
                    "codec_type": "video",
                    "height": 2160,
                    "pix_fmt": "yuv420p10le",
                    "color_transfer": "smpte2084",
                },
                {
                    "index": 1,
                    "codec_name": "eac3",
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 6,
                    "bit_rate": "448000",
                },
            ],
            "format": {"format_name": "mpegts", "duration": "1.005333", "bit_rate": "13233808"},
        },
        # KPIX (5.1) while a progressive segment aired; it carries 60i video too.
        "ota_1080_mpeg2_progressive_segment": {
            "streams": [
                {
                    "index": 0,
                    "codec_name": "mpeg2video",
                    "profile": "Main",
                    "codec_type": "video",
                    "height": 1080,
                    "pix_fmt": "yuv420p",
                    "field_order": "progressive",
                },
                {
                    "index": 1,
                    "codec_name": "ac3",
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 6,
                    "bit_rate": "384000",
                    "tags": {"language": "eng"},
                },
            ],
            "format": {"format_name": "mpegts", "duration": "1.637011", "bit_rate": "10332196"},
        },
        "ota_1080i_mpeg2": {
            "streams": [
                {
                    "index": 0,
                    "codec_name": "mpeg2video",
                    "profile": "Main",
                    "codec_type": "video",
                    "height": 1080,
                    "pix_fmt": "yuv420p",
                    "field_order": "tt",
                },
                {
                    "index": 1,
                    "codec_name": "ac3",
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 6,
                    "bit_rate": "384000",
                    "tags": {"language": "eng"},
                },
                {
                    "index": 2,
                    "codec_name": "ac3",
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 1,
                    "bit_rate": "96000",
                    "tags": {"language": "fre"},
                },
                {
                    "index": 3,
                    "codec_name": "ac3",
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 1,
                    "bit_rate": "96000",
                    "tags": {"language": "spa"},
                },
            ],
            "format": {"format_name": "mpegts", "duration": "0.666812", "bit_rate": "3780327"},
        },
        "ota_480i_mpeg2": {
            "streams": [
                {
                    "index": 0,
                    "codec_name": "mpeg2video",
                    "profile": "Main",
                    "codec_type": "video",
                    "height": 480,
                    "pix_fmt": "yuv420p",
                    "field_order": "tt",
                },
                {
                    "index": 1,
                    "codec_name": "ac3",
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 2,
                    "bit_rate": "128000",
                },
            ],
            "format": {"format_name": "mpegts", "duration": "1.594134", "bit_rate": "701933"},
        },
        "ota_720p_h264": {
            "streams": [
                {
                    "index": 0,
                    "codec_name": "h264",
                    "profile": "Main",
                    "codec_type": "video",
                    "height": 720,
                    "pix_fmt": "yuv420p",
                    "color_transfer": "bt709",
                    "field_order": "progressive",
                },
                {
                    "index": 1,
                    "codec_name": "ac3",
                    "codec_type": "audio",
                    "sample_rate": "0",
                    "channels": 0,
                    "tags": {"language": "eng"},
                },
            ],
            "format": {"format_name": "mpegts", "duration": "0.008333", "bit_rate": "8482899"},
        },
        "ota_720p_mpeg2": {
            "streams": [
                {
                    "index": 0,
                    "codec_name": "mpeg2video",
                    "profile": "Main",
                    "codec_type": "video",
                    "height": 720,
                    "pix_fmt": "yuv420p",
                    "field_order": "progressive",
                },
                {
                    "index": 1,
                    "codec_name": "ac3",
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 6,
                    "bit_rate": "384000",
                    "tags": {"language": "eng"},
                },
                {
                    "index": 2,
                    "codec_name": "ac3",
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 2,
                    "bit_rate": "192000",
                    "tags": {"language": "eng"},
                },
            ],
            "format": {"format_name": "mpegts", "duration": "1.633400", "bit_rate": "5382578"},
        },
    }
    return json.dumps(captures[name])


def pat_packet() -> bytes:
    """A PAT naming program 1 at PMT PID 0x30."""
    return b"\x47\x40\x00\x10\x00\x00\xb0\x0d\x00\x01\xc1\x00\x00\x00\x01\xe0\x30".ljust(
        188,
        b"\xff",
    )


def pmt_packet() -> bytes:
    """A PMT packet on PID 0x30."""
    return b"\x47\x40\x30\x10\x00\x02\xb0\x12\x00\x01\xc1\x00\x00".ljust(188, b"\xff")


def ts_source(seconds: float = 10, packets: int = 100) -> bytes:
    """MPEG-TS opening at a keyframe whose timestamp is ``seconds``, as a tuner sends it.

    Args:
        seconds: Presentation time of the first video frame.
        packets: Non-key video packets following the keyframe.

    Returns:
        stream: PAT, PMT, the keyframe packet, then the other packets.
    """
    keyframe = video_packet(seconds, keyframe=True)
    return pat_packet() + pmt_packet() + keyframe + video_packet(seconds) * packets


def video_packet(seconds: float, *, keyframe: bool = False) -> bytes:
    """A video PES start packet; a keyframe carries the random-access indicator.

    Args:
        seconds: Presentation time stamped on the packet.
        keyframe: Mark the packet as a random-access point.

    Returns:
        packet: One 188-byte transport packet.
    """
    pts = int(seconds * 90_000)
    encoded = bytes(
        [
            0x21 | ((pts >> 29) & 14),
            (pts >> 22) & 255,
            ((pts >> 14) & 254) | 1,
            (pts >> 7) & 255,
            ((pts << 1) & 254) | 1,
        ],
    )
    pes = b"\x00\x00\x01\xe0\x00\x00\x80\x80\x05" + encoded
    header = b"\x47\x41\x00\x30\x01\x40" if keyframe else b"\x47\x41\x00\x10"
    return (header + pes).ljust(188, b"\xff")


class FakeSink:
    """An asyncio ``StreamWriter`` (a worker's stdin) recording what it receives.

    Args:
        blocked: ``drain`` never completes, as for a worker that stopped reading.
        error: Raise this from ``write``, as for a worker that exited.
    """

    def __init__(self, *, blocked: bool = False, error: Exception | None = None) -> None:
        self.written = bytearray()
        self.closed = False
        self.blocked = blocked
        self.error = error
        self._changed = asyncio.Event()

    def write(self, data: bytes, /) -> None:
        if self.error is not None:
            raise self.error
        self.written += data
        self._changed.set()

    async def drain(self) -> None:
        if self.blocked:
            await asyncio.Event().wait()

    def close(self) -> None:
        self.closed = True

    async def wait_for(self, size: int) -> None:
        """Wait until at least ``size`` bytes were written."""
        while len(self.written) < size:
            self._changed.clear()
            await asyncio.wait_for(self._changed.wait(), timeout=1)


class FakeProcess:
    """An ``asyncio.subprocess.Process`` whose lifetime the test scripts.

    Args:
        exit_code: Code ``wait()`` reports when the process ends on its own;
            ``None`` keeps it running until signalled.
        stderr: Lines the process writes to stderr before EOF.
        ignores_terminate: Survive SIGTERM, so callers must escalate to SIGKILL.
        exited: Start already reaped with ``exit_code`` (0 when it is ``None``).
        reaped: Signals raise ``ProcessLookupError``, as for a vanished pid.
        deferred_exit: Deliver SIGTERM's exit on the next loop iteration, as
            asyncio's child watcher does, rather than immediately.
        pid: Process id to report.
    """

    def __init__(
        self,
        *,
        exit_code: int | None = 0,
        stderr: Iterable[bytes] = (),
        ignores_terminate: bool = False,
        exited: bool = False,
        reaped: bool = False,
        deferred_exit: bool = False,
        pid: int = 4242,
    ) -> None:
        self.pid = pid
        self.exit_code = exit_code
        self.stdin = FakeSink()
        self.stderr: FakeStream | None = FakeStream(stderr)
        self.signals: list[str] = []
        self.waits = 0
        self.ignores_terminate = ignores_terminate
        self.reaped = reaped
        self.deferred_exit = deferred_exit
        self.returncode: int | None = None
        self._final_code = 0
        self._exited = asyncio.Event()
        if exited:
            self._exit(0 if exit_code is None else exit_code)

    def terminate(self) -> None:
        self._signal("TERM")
        if self.ignores_terminate:
            return
        if self.deferred_exit:
            asyncio.get_running_loop().call_soon(self._exit, -15)
        else:
            self._exit(-15)

    def kill(self) -> None:
        self._signal("KILL")
        self._exit(-9)

    async def wait(self) -> int:
        self.waits += 1
        if self.returncode is None and self.exit_code is not None:
            self._exit(self.exit_code)
        await self._exited.wait()
        return self._final_code

    def _signal(self, name: str) -> None:
        if self.reaped:
            raise ProcessLookupError("No such process")
        self.signals.append(name)

    def _exit(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = self._final_code = code
            self._exited.set()


@dataclass(frozen=True, kw_only=True, slots=True)
class Launch:
    """One recorded ``create_subprocess_exec`` call."""

    command: list[str]
    options: dict[str, object]
    process: FakeProcess


class ProcessLauncher:
    """Stands in for ``asyncio.create_subprocess_exec``.

    Args:
        processes: Processes for the first launches; later launches get fresh
            ``FakeProcess`` objects that run until signalled.
        error: Raise this from every launch instead, as for a missing binary.
        on_launch: The program's effect, run (and awaited, if it returns an
            awaitable) with each command before the launch is recorded -- e.g.
            writing the playlist a real ffmpeg would produce. Raising fails
            the launch.
    """

    def __init__(
        self,
        processes: Iterable[FakeProcess] = (),
        *,
        error: OSError | None = None,
        on_launch: Callable[[list[str]], object] | None = None,
    ) -> None:
        self._queued = list(processes)
        self.error = error
        self.on_launch = on_launch
        self.launches: list[Launch] = []

    async def __call__(self, *command: str, **options: object) -> FakeProcess:
        if self.error is not None:
            raise self.error
        if self.on_launch is not None:
            effect = self.on_launch(list(command))
            if inspect.isawaitable(effect):
                await effect
        process = self._queued.pop(0) if self._queued else FakeProcess(exit_code=None)
        self.launches.append(Launch(command=list(command), options=options, process=process))
        return process

    @property
    def commands(self) -> list[list[str]]:
        return [launch.command for launch in self.launches]

    @property
    def processes(self) -> list[FakeProcess]:
        return [launch.process for launch in self.launches]


@dataclass(frozen=True, kw_only=True, slots=True)
class Program:
    """Scripted outcome of one external program under ``FakeRun``."""

    stdout: str | bytes = ""
    stderr: str | bytes = ""
    returncode: int = 0
    error: BaseException | None = None


class FakeRun:
    """Stands in for ``subprocess.run`` with scripted external programs.

    A program is chosen by executable name and, optionally, by an argument the
    command must contain; the most recently scripted match wins. Unscripted
    executables raise ``FileNotFoundError``, as an uninstalled binary would.
    """

    def __init__(self) -> None:
        self._programs: list[tuple[str, str | None, Program]] = []
        self.calls: list[list[str]] = []
        self.options: list[dict[str, object]] = []

    def program(self, executable: str, outcome: Program, *, when: str | None = None) -> None:
        """Script ``executable``, or only its runs whose arguments include ``when``."""
        self._programs.insert(0, (executable, when, outcome))

    def __call__(
        self,
        args: Sequence[str],
        **options: object,
    ) -> subprocess.CompletedProcess[str | bytes]:
        command = list(args)
        self.calls.append(command)
        self.options.append(options)
        executable = command[0].rsplit("/", 1)[-1]
        for name, when, outcome in self._programs:
            if name == executable and (when is None or when in command):
                return _completed(command, outcome, text=bool(options.get("text")))
        raise FileNotFoundError(2, "No such file or directory", command[0])


class FakePopen:
    """A ``subprocess.Popen`` that runs until signalled or ``finish`` is called."""

    def __init__(self, command: Sequence[str], **options: object) -> None:
        self.args = list(command)
        self.options = options
        self.stdout = FakeUpstream()
        self.returncode: int | None = None
        self.signals: list[str] = []
        self.ignores_terminate = False

    def finish(self, code: int = 0) -> None:
        if self.returncode is None:
            self.returncode = code

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.args, 0 if timeout is None else timeout)
        return self.returncode

    def terminate(self) -> None:
        self.signals.append("TERM")
        if not self.ignores_terminate:
            self.finish(-15)

    def kill(self) -> None:
        self.signals.append("KILL")
        self.finish(-9)


class PopenRecorder:
    """Stands in for ``subprocess.Popen``, keeping every process it starts.

    Args:
        error: Raise this from every launch instead, as for a missing binary.
    """

    def __init__(self, *, error: OSError | None = None) -> None:
        self.error = error
        self.started: list[FakePopen] = []

    def __call__(self, command: Sequence[str], **options: object) -> FakePopen:
        if self.error is not None:
            raise self.error
        process = FakePopen(command, **options)
        self.started.append(process)
        return process


class FakeResponse:
    """An ``http.client.HTTPResponse`` with a fixed status, headers, and body."""

    def __init__(
        self,
        body: bytes = b"",
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        url: str = "",
    ) -> None:
        self.status = status
        self.url = url
        self.headers = Message()
        for name, value in (headers or {}).items():
            self.headers[name] = value
        self._body = body
        self._offset = 0
        self.closed = False

    def read(self, amt: int | None = None) -> bytes:
        end = len(self._body) if amt is None or amt < 0 else self._offset + amt
        chunk = self._body[self._offset : end]
        self._offset += len(chunk)
        return chunk

    def read1(self, amt: int = -1, /) -> bytes:
        return self.read(amt)

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self.url

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class FakeUpstream(FakeResponse):
    """A live upstream body (an ``HTTPResponse``) that the test feeds chunk by chunk.

    ``read1`` blocks until the test calls ``feed``, ``end``, or ``fail``, as a
    socket read waits for the network; ``close`` unblocks a pending read.
    """

    def __init__(self, *, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(headers=headers)
        self._items: queue.Queue[bytes | Exception] = queue.Queue()
        self.requested: list[int] = []
        # Set once a reader is blocked in read1, as a socket read waits.
        self.reading = threading.Event()

    def feed(self, data: bytes) -> None:
        self._items.put(data)

    def end(self) -> None:
        self._items.put(b"")

    def fail(self, error: Exception) -> None:
        self._items.put(error)

    @override
    def read1(self, amt: int = -1, /) -> bytes:
        self.requested.append(amt)
        self.reading.set()
        item = self._items.get()
        if isinstance(item, Exception):
            raise item
        return item

    @override
    def close(self) -> None:
        super().close()
        self._items.put(b"")


def ended_upstream(data: bytes | None = None) -> FakeUpstream:
    """A provider connection that sends ``data`` (default ``ts_source()``), then ends."""
    upstream = FakeUpstream()
    upstream.feed(ts_source() if data is None else data)
    upstream.end()
    return upstream


class FakeUpstreams:
    """A ``FakeWeb`` route opening a fresh ``FakeUpstream`` per request, keeping each.

    Live upstreams block their readers until fed or closed; ``close`` them all
    before the event loop ends so no reader thread outlives the test.
    """

    def __init__(self) -> None:
        self.opened: list[FakeUpstream] = []

    def __call__(self) -> FakeUpstream:
        upstream = FakeUpstream()
        self.opened.append(upstream)
        return upstream

    def close(self) -> None:
        for upstream in self.opened:
            upstream.close()


Route = FakeResponse | Exception | Callable[[], FakeResponse]


@dataclass(frozen=True, kw_only=True, slots=True)
class Fetch:
    """One recorded HTTP request."""

    url: str
    timeout: float | None
    headers: dict[str, str]
    handlers: tuple[object, ...] = ()


class FakeWeb:
    """The internet as a table of URLs; install with ``install_web``.

    A route is a response, an exception to raise, or a factory for a fresh
    response per request. Unrouted URLs raise ``URLError``, as an unreachable
    host would.
    """

    def __init__(self, routes: Mapping[str, Route] | None = None) -> None:
        self.routes: dict[str, Route] = dict(routes or {})
        self.fetches: list[Fetch] = []

    def __call__(
        self,
        url: str | request.Request,
        data: bytes | None = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        """Answer like ``urllib.request.urlopen``."""
        del data
        return self.fetch(url, timeout)

    def fetch(
        self,
        url: str | request.Request,
        timeout: float | None,
        handlers: tuple[object, ...] = (),
    ) -> FakeResponse:
        """Answer one request, recording the opener handlers it passed through.

        Args:
            url: Requested URL or ``Request``.
            timeout: Request timeout in seconds.
            handlers: Handlers of the opener that sent the request.

        Returns:
            response: The routed response.
        """
        headers: dict[str, str] = {}
        if isinstance(url, request.Request):
            target = url.full_url
            headers.update(url.header_items())
        else:
            target = url
        self.fetches.append(Fetch(url=target, timeout=timeout, headers=headers, handlers=handlers))
        route = self.routes.get(target)
        if route is None:
            raise error.URLError(f"no route to {target}")
        if isinstance(route, Exception):
            raise route
        response = route if isinstance(route, FakeResponse) else route()
        response.url = response.url or target
        return response

    @property
    def urls(self) -> list[str]:
        return [fetch.url for fetch in self.fetches]

    def build_opener(self, *handlers: object) -> _FakeOpener:
        """Answer like ``urllib.request.build_opener``; the handlers are kept, not run."""
        return _FakeOpener(web=self, handlers=handlers)


@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeOpener:
    web: FakeWeb
    handlers: tuple[object, ...]

    def open(
        self,
        url: str | request.Request,
        data: bytes | None = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        del data
        return self.web.fetch(url, timeout, self.handlers)


class FakeThread:
    """A ``threading.Thread`` whose start a ``ThreadRecorder`` controls."""

    def __init__(
        self,
        recorder: ThreadRecorder,
        target: Callable[..., object] | None,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        *,
        daemon: bool | None,
        name: str | None,
    ) -> None:
        self._recorder = recorder
        self.target = target
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.daemon = bool(daemon)
        self.name = name or "fake-thread"
        self.started = False
        self.finished = False
        self.error: Exception | None = None
        self.joins: list[float | None] = []

    def start(self) -> None:
        self.started = True
        self._recorder.threads.append(self)
        if self._recorder.run_inline:
            self.run()

    def run(self) -> None:
        if self.finished:
            return
        self.finished = True
        try:
            if self.target is not None:
                self.target(*self.args, **self.kwargs)
        except Exception as e:  # A real thread reports it and dies.
            self.error = e

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)

    def is_alive(self) -> bool:
        return self.started and not self.finished


class ThreadRecorder:
    """Stands in for ``threading.Thread``; install with ``install_threads``.

    Args:
        run_inline: Run each thread's target as soon as it starts; otherwise
            hold it until ``run_all``.
    """

    def __init__(self, *, run_inline: bool = True) -> None:
        self.run_inline = run_inline
        self.threads: list[FakeThread] = []

    def make_thread(
        self,
        group: None = None,
        target: Callable[..., object] | None = None,
        name: str | None = None,
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] | None = None,
        *,
        daemon: bool | None = None,
    ) -> FakeThread:
        del group
        return FakeThread(self, target, args, kwargs or {}, daemon=daemon, name=name)

    def run_all(self) -> None:
        """Run every started thread that has not run, including ones they start."""
        while pending := [t for t in self.threads if not t.finished]:
            for thread in pending:
                thread.run()

    @property
    def errors(self) -> list[Exception]:
        return [t.error for t in self.threads if t.error is not None]


class SignalRecorder:
    """Stands in for ``os.kill``; install with ``install_signals``.

    Args:
        errors: Exception to raise when signalling each pid, e.g.
            ``ProcessLookupError`` for a pid that already exited.
    """

    def __init__(self, errors: Mapping[int, OSError] | None = None) -> None:
        self.errors = dict(errors or {})
        self.kills: list[tuple[int, int]] = []

    def __call__(self, pid: int, sig: int) -> None:
        self.kills.append((pid, sig))
        error_ = self.errors.get(pid)
        if error_ is not None:
            raise error_


class ContendedLock:
    """A ``threading.Lock`` another thread held: acquiring it first runs that thread's work.

    Reaches double-checked code paths ("someone filled the cache while we
    waited") deterministically.

    Args:
        on_acquire: Work the other thread finishes before the lock is granted.
    """

    def __init__(self, on_acquire: Callable[[], object]) -> None:
        self._on_acquire: Callable[[], object] | None = on_acquire
        self.acquired = 0

    def __enter__(self) -> bool:
        if self._on_acquire is not None:
            work, self._on_acquire = self._on_acquire, None
            work()
        self.acquired += 1
        return True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb


class FakeTask:
    """The ``asyncio.Task`` handle ``BackgroundTasks`` returns for a held coroutine."""

    def __init__(self, coroutine: Coroutine[object, object, object]) -> None:
        self.coroutine = coroutine
        self._cancelled = False
        self._done = False

    def cancel(self) -> bool:
        if self._done:
            return False
        self._cancelled = self._done = True
        self.coroutine.close()
        return True

    def cancelled(self) -> bool:
        return self._cancelled

    def done(self) -> bool:
        return self._done

    async def run(self) -> None:
        """Run the coroutine to completion, as the event loop would."""
        if not self._done:
            self._done = True
            await self.coroutine


class BackgroundTasks:
    """Collects coroutines a module would spawn as fire-and-forget tasks."""

    def __init__(self) -> None:
        self.tasks: list[FakeTask] = []

    def __call__(self, coroutine: Coroutine[object, object, object]) -> FakeTask:
        task = FakeTask(coroutine)
        self.tasks.append(task)
        return task

    @property
    def pending(self) -> list[Coroutine[object, object, object]]:
        return [task.coroutine for task in self.tasks if not task.done()]

    async def run_all(self) -> None:
        """Await every held coroutine, including ones they spawn."""
        while waiting := [task for task in self.tasks if not task.done()]:
            await waiting[0].run()

    def discard(self) -> None:
        """Cancel every held coroutine without running it."""
        for task in self.tasks:
            task.cancel()


class ClockDatetime(datetime):
    """``datetime`` whose ``now`` reads the ``clock`` of a ``FakeClock.datetime_class``."""

    clock: ClassVar[FakeClock]

    @classmethod
    @override
    def now(cls, tz: tzinfo | None = None) -> Self:
        return cls.fromtimestamp(cls.clock.now, tz)


class ClockEvent:
    """A ``threading.Event`` whose timed waits pass on a ``FakeClock``."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self._flag = False

    def set(self) -> None:
        self._flag = True

    def clear(self) -> None:
        self._flag = False

    def is_set(self) -> bool:
        return self._flag

    def wait(self, timeout: float | None = None) -> bool:
        if self._flag:
            return True
        deadline = self._clock.deadline
        if deadline is not None and (timeout is None or self._clock.now + timeout >= deadline):
            self._clock.sleep(max(deadline - self._clock.now, 0))
            self._flag = True
            return True
        if timeout is None:
            raise RuntimeError("waiting forever on an unset event: nothing else can set it")
        self._clock.sleep(timeout)
        return False


class FakeClock:
    """Wall and monotonic time that moves only when the test or a sleep moves it.

    Args:
        now: Starting POSIX time.
        deadline: When set, every ``ClockEvent`` reports set once a wait would
            reach this time -- as if shutdown were requested then -- so
            ``while not stop.wait(interval)`` loops end.
    """

    def __init__(self, now: float = 1_700_000_000.0, *, deadline: float | None = None) -> None:
        self.now = now
        self.deadline = deadline
        self.sleeps: list[float] = []
        self.events: list[ClockEvent] = []

    def make_event(self) -> ClockEvent:
        event = ClockEvent(self)
        self.events.append(event)
        return event

    def datetime_class(self) -> type[datetime]:
        """A ``datetime`` subclass whose ``now`` reads this clock."""
        return type("ClockDatetime", (ClockDatetime,), {"clock": self})

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)

    async def async_wait_for(self, awaitable: Awaitable[_T], timeout: float | None) -> _T:
        """Like ``asyncio.wait_for``: whatever is not done after one loop pass times out.

        Args:
            awaitable: Work to wait for.
            timeout: Seconds the clock advances when the work is still pending.

        Returns:
            result: The work's result.
        """
        task = asyncio.ensure_future(awaitable)
        await _real_asyncio_sleep(0)
        if task.done() or timeout is None:
            return await task
        task.cancel()
        self.sleep(timeout)
        raise TimeoutError

    async def async_sleep(self, seconds: float, result: object = None) -> object:
        self.sleeps.append(seconds)
        self.advance(seconds)
        await _real_asyncio_sleep(0)
        return result


_real_asyncio_sleep = asyncio.sleep


class ModuleView(ModuleType):
    """A module exposing a real module's attributes, except the overridden ones.

    Gives one neTV module a fake ``time``, ``asyncio``, or ``threading``
    without touching the shared stdlib module the event loop and pytest use.
    """

    def __init__(self, real: ModuleType, overrides: Mapping[str, object]) -> None:
        super().__init__(real.__name__, real.__doc__)
        self._real = real
        self._overrides = dict(overrides)
        self.__dict__.update(self._overrides)

    @override
    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def install_clock(monkeypatch: pytest.MonkeyPatch, clock: FakeClock, *modules: ModuleType) -> None:
    """Point each module's ``time``, ``asyncio.sleep``, and ``threading.Event`` at ``clock``.

    Events the module already created keep real time; construct objects that
    own events after installing.

    Args:
        monkeypatch: The test's monkeypatch.
        clock: Clock that every given module reads and sleeps on.
        modules: neTV modules whose ``time``/``asyncio``/``threading`` globals to replace.
    """
    for module in modules:
        if isinstance(getattr(module, "time", None), ModuleType):
            _override(
                monkeypatch,
                module,
                "time",
                time,
                time=clock.time,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
        if isinstance(getattr(module, "asyncio", None), ModuleType):
            _override(
                monkeypatch,
                module,
                "asyncio",
                asyncio,
                sleep=clock.async_sleep,
                wait_for=clock.async_wait_for,
            )
        if isinstance(getattr(module, "threading", None), ModuleType):
            _override(monkeypatch, module, "threading", threading, Event=clock.make_event)
        if getattr(module, "datetime", None) is datetime:
            monkeypatch.setattr(module, "datetime", clock.datetime_class())
        _route_injected_sleeps(monkeypatch, clock, module)


def install_signals(
    monkeypatch: pytest.MonkeyPatch,
    recorder: SignalRecorder,
    *modules: ModuleType,
) -> None:
    """Point each module's ``os.kill`` at ``recorder``.

    Args:
        monkeypatch: The test's monkeypatch.
        recorder: Recorder for every signal the modules send.
        modules: neTV modules whose ``os`` global to replace.
    """
    for module in modules:
        _override(monkeypatch, module, "os", os, kill=recorder)


def install_launcher(
    monkeypatch: pytest.MonkeyPatch,
    launcher: ProcessLauncher,
    module: ModuleType,
) -> None:
    """Point ``module.asyncio.create_subprocess_exec`` at ``launcher``.

    Args:
        monkeypatch: The test's monkeypatch.
        launcher: Launcher recording every process the module starts.
        module: neTV module whose ``asyncio`` global to replace.
    """
    _override(monkeypatch, module, "asyncio", asyncio, create_subprocess_exec=launcher)


def install_web(monkeypatch: pytest.MonkeyPatch, web: FakeWeb, *modules: ModuleType) -> None:
    """Route each module's ``urllib.request`` traffic to ``web``.

    Install on ``netv.util`` to serve everything ``safe_urlopen`` fetches, with
    its scheme checks and headers intact.

    Args:
        monkeypatch: The test's monkeypatch.
        web: Fake internet answering every request.
        modules: Modules whose ``request`` global (``urllib.request``) to replace.
    """
    for module in modules:
        _override(
            monkeypatch,
            module,
            "request",
            request,
            urlopen=web,
            build_opener=web.build_opener,
        )


def thread_pool(*, max_workers: int, mp_context: object) -> ThreadPoolExecutor:
    """A ``ProcessPoolExecutor`` stand-in whose workers share the test's fakes."""
    del mp_context
    return ThreadPoolExecutor(max_workers=max_workers)


def cheap_pbkdf2_hmac(
    hash_name: str,
    password: bytes,
    salt: bytes,
    iterations: int,
    dklen: int | None = None,
) -> bytes:
    """``hashlib.pbkdf2_hmac`` at one iteration: same shape and determinism, no cost."""
    del iterations
    return hashlib.pbkdf2_hmac(hash_name, password, salt, 1, dklen)


def install_cheap_kdf(monkeypatch: pytest.MonkeyPatch, *modules: ModuleType) -> None:
    """Point each module's ``hashlib.pbkdf2_hmac`` at ``cheap_pbkdf2_hmac``.

    Real password hashing costs ~50 ms; tests check hashing behavior, not cost.

    Args:
        monkeypatch: The test's monkeypatch.
        modules: Modules whose ``hashlib`` global to replace.
    """
    for module in modules:
        _override(monkeypatch, module, "hashlib", hashlib, pbkdf2_hmac=cheap_pbkdf2_hmac)


def install_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    run: FakeRun | None = None,
    popen: PopenRecorder | None = None,
) -> None:
    """Point ``module.subprocess.run``/``Popen`` at fakes.

    Args:
        monkeypatch: The test's monkeypatch.
        module: neTV module whose ``subprocess`` global to replace.
        run: Replacement for ``subprocess.run``; real when omitted.
        popen: Replacement for ``subprocess.Popen``; real when omitted.
    """
    overrides: dict[str, object] = {}
    if run is not None:
        overrides["run"] = run
    if popen is not None:
        overrides["Popen"] = popen
    _override(monkeypatch, module, "subprocess", subprocess, **overrides)


def install_threads(
    monkeypatch: pytest.MonkeyPatch,
    recorder: ThreadRecorder,
    *modules: ModuleType,
) -> None:
    """Point each module's ``threading.Thread`` at ``recorder``.

    Args:
        monkeypatch: The test's monkeypatch.
        recorder: Recorder that starts every thread the modules create.
        modules: neTV modules whose ``threading`` global to replace.
    """
    for module in modules:
        _override(monkeypatch, module, "threading", threading, Thread=recorder.make_thread)


@dataclass(frozen=True, kw_only=True, slots=True)
class ServerRun:
    """One recorded ``uvicorn.run`` call."""

    app: object
    options: dict[str, object]


class ServerRecorder:
    """Stands in for ``uvicorn.run``: records the app and options, serves nothing."""

    def __init__(self) -> None:
        self.runs: list[ServerRun] = []

    def __call__(self, app: object, **options: object) -> None:
        self.runs.append(ServerRun(app=app, options=options))


def install_server(
    monkeypatch: pytest.MonkeyPatch,
    recorder: ServerRecorder,
    *modules: ModuleType,
) -> None:
    """Route ``uvicorn.run`` to ``recorder``.

    With modules, replaces each one's ``uvicorn`` global (a lazy-import proxy)
    without importing uvicorn. Without modules, patches the real
    ``uvicorn.run``, which also covers modules re-executed through ``runpy``.

    Args:
        monkeypatch: The test's monkeypatch.
        recorder: Recorder for every server that would start.
        modules: neTV modules whose ``uvicorn`` global to replace.
    """
    for module in modules:
        monkeypatch.setattr(module, "uvicorn", ModuleView(ModuleType("uvicorn"), {"run": recorder}))
    if not modules:
        monkeypatch.setattr(importlib.import_module("uvicorn"), "run", recorder)


@dataclass(kw_only=True, slots=True)
class FakeMediaStatus:
    """A pychromecast ``MediaStatus``."""

    content_id: str | None = None
    player_state: str = "UNKNOWN"
    idle_reason: str | None = None
    adjusted_current_time: float | None = 0.0


@dataclass(frozen=True, kw_only=True, slots=True)
class CastLoad:
    """One recorded ``play_media`` call."""

    url: str
    content_type: str
    options: dict[str, object]


class FakeMediaController:
    """A pychromecast ``MediaController`` on a receiver with scripted behavior.

    Args:
        outcome: What loading media does: ``"plays"`` starts it, ``"fails"``
            ends it with an error, ``"hangs"`` never reports anything.
        error: Raise this from every command, as for a dropped connection.
        clock: Clock the receiver's start-up delay runs on.
        load_secs: Seconds media reports ``IDLE`` before it starts (needs ``clock``).
        started_state: State media reports once started.
    """

    def __init__(
        self,
        *,
        outcome: str = "plays",
        error: Exception | None = None,
        clock: FakeClock | None = None,
        load_secs: float = 0.0,
        started_state: str = "PLAYING",
    ) -> None:
        self._status = FakeMediaStatus()
        self._starting: tuple[FakeClock, float, FakeMediaStatus] | None = None
        self.app_must_match = False
        self.outcome = outcome
        self.error = error
        self.clock = clock
        self.load_secs = load_secs
        self.started_state = started_state
        self.loaded: list[CastLoad] = []
        self.commands: list[str] = []

    @property
    def status(self) -> FakeMediaStatus:
        if self._starting is not None and self._starting[0].now >= self._starting[1]:
            self._status, self._starting = self._starting[2], None
        return self._status

    @status.setter
    def status(self, value: FakeMediaStatus) -> None:
        self._status, self._starting = value, None

    def play_media(self, url: str, content_type: str, **options: object) -> None:
        self._command("load")
        self.loaded.append(CastLoad(url=url, content_type=content_type, options=options))
        if self.outcome == "plays":
            started = FakeMediaStatus(content_id=url, player_state=self.started_state)
            if self.clock is None or self.load_secs <= 0:
                self.status = started
            else:
                self.status = FakeMediaStatus(content_id=url, player_state="IDLE")
                self._starting = (self.clock, self.clock.now + self.load_secs, started)
        elif self.outcome == "fails":
            self.status = FakeMediaStatus(content_id=url, player_state="IDLE", idle_reason="ERROR")

    def play(self) -> None:
        self._command("play")
        self.status.player_state = "PLAYING"

    def pause(self) -> None:
        self._command("pause")
        self.status.player_state = "PAUSED"

    def stop(self) -> None:
        self._command("stop")
        self.status = FakeMediaStatus(player_state="IDLE")

    def _command(self, name: str) -> None:
        if self.error is not None:
            raise self.error
        self.commands.append(name)


@dataclass(kw_only=True, slots=True)
class FakeCastInfo:
    friendly_name: str | None


@dataclass(kw_only=True, slots=True)
class FakeReceiverStatus:
    volume_level: float = 1.0


@dataclass(kw_only=True, slots=True)
class FakeSocketClient:
    is_connected: bool = True


class FakeChromecast:
    """A pychromecast ``Chromecast`` receiver on the LAN.

    Args:
        friendly_name: Name the receiver reports once connected.
        outcome: How it reacts to loading media; see ``FakeMediaController``.
        error: Raise this from every media command.
        clock: Clock the start-up delay runs on.
        load_secs: Seconds loaded media takes to start.
        started_state: State media reports once started.
    """

    def __init__(
        self,
        *,
        friendly_name: str | None = "TV",
        outcome: str = "plays",
        error: Exception | None = None,
        clock: FakeClock | None = None,
        load_secs: float = 0.0,
        started_state: str = "PLAYING",
    ) -> None:
        self.cast_info = FakeCastInfo(friendly_name=friendly_name)
        self.media_controller = FakeMediaController(
            outcome=outcome,
            error=error,
            clock=clock,
            load_secs=load_secs,
            started_state=started_state,
        )
        self.status: FakeReceiverStatus | None = FakeReceiverStatus()
        self.socket_client = FakeSocketClient()
        self.disconnected = False

    def wait(self, timeout: float | None = None) -> None:
        del timeout

    def disconnect(self, timeout: float | None = None) -> None:
        del timeout
        self.disconnected = True

    def set_volume(self, volume: float) -> None:
        self.media_controller._command("volume")
        if self.status is not None:
            self.status.volume_level = volume


@dataclass(frozen=True, kw_only=True, slots=True)
class FakeCastDevice:
    """A receiver as mDNS discovery advertises it."""

    host: str
    friendly_name: str | None


class FakeCastBrowser:
    """A pychromecast ``CastBrowser`` that finds the network's advertised devices."""

    def __init__(self, network: CastNetwork, listener: object, zconf: object) -> None:
        del listener, zconf
        self._network = network
        self.devices: dict[str, FakeCastDevice] = {}

    def start_discovery(self) -> None:
        self._network.browsing = True
        self.devices = {str(i): device for i, device in enumerate(self._network.advertised)}

    def stop_discovery(self) -> None:
        self._network.browsing = False


class CastNetwork:
    """The LAN as pychromecast and zeroconf see it; install with ``install_chromecast``.

    Args:
        receivers: Receivers reachable by IP address.
        advertised: Devices mDNS discovery reports.
        discovery_error: Raise this when discovery starts, as for no multicast.
    """

    def __init__(
        self,
        receivers: Mapping[str, FakeChromecast] | None = None,
        *,
        advertised: Iterable[FakeCastDevice] = (),
        discovery_error: OSError | None = None,
    ) -> None:
        self.receivers: dict[str, FakeChromecast] = dict(receivers or {})
        self.advertised = list(advertised)
        self.discovery_error = discovery_error
        self.connects: list[str] = []
        self.browsing = False

    def get_chromecast_from_host(
        self,
        host: tuple[object, ...],
        **options: object,
    ) -> FakeChromecast:
        del options
        address = str(host[0])
        self.connects.append(address)
        if address not in self.receivers:
            raise PyChromecastError(f"no receiver at {address}")
        return self.receivers[address]

    def zeroconf(self) -> Self:
        if self.discovery_error is not None:
            raise self.discovery_error
        return self

    def cast_browser(self, listener: object, zconf: object) -> FakeCastBrowser:
        return FakeCastBrowser(self, listener, zconf)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb


def install_chromecast(
    monkeypatch: pytest.MonkeyPatch,
    network: CastNetwork,
    module: ModuleType,
) -> None:
    """Point ``module``'s pychromecast and zeroconf entry points at ``network``.

    Args:
        monkeypatch: The test's monkeypatch.
        network: Fake LAN to discover and connect on.
        module: Module importing ``get_chromecast_from_host``, ``CastBrowser``, ``Zeroconf``.
    """
    monkeypatch.setattr(module, "get_chromecast_from_host", network.get_chromecast_from_host)
    monkeypatch.setattr(module, "CastBrowser", network.cast_browser)
    monkeypatch.setattr(module, "Zeroconf", network.zeroconf)


@dataclass(kw_only=True, slots=True)
class FakeXtreamClient:
    """An ``XtreamClient`` whose provider answers are canned."""

    server_info: dict[str, object] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def get_server_info(self) -> dict[str, object]:
        self.calls.append("get_server_info")
        return self.server_info


def _completed(
    command: list[str],
    outcome: Program,
    *,
    text: bool,
) -> subprocess.CompletedProcess[str | bytes]:
    if outcome.error is not None:
        raise outcome.error
    return subprocess.CompletedProcess(
        command,
        outcome.returncode,
        _as_output(outcome.stdout, text=text),
        _as_output(outcome.stderr, text=text),
    )


def _as_output(value: str | bytes, *, text: bool) -> str | bytes:
    if text:
        return value.decode() if isinstance(value, bytes) else value
    return value.encode() if isinstance(value, str) else value


def _override(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    name: str,
    real: ModuleType,
    **overrides: object,
) -> None:
    """Replace ``module.<name>`` with a view of ``real``, keeping earlier overrides."""
    current = getattr(module, name)
    kept = current._overrides if isinstance(current, ModuleView) else {}
    monkeypatch.setattr(module, name, ModuleView(real, {**kept, **overrides}))


# A default binds when its function is defined, so replacing the module's ``asyncio`` or
# ``time`` afterwards does not reach it.
def _route_injected_sleeps(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    module: ModuleType,
) -> None:
    """Point every real-sleep default a module's functions take as ``sleep=`` at ``clock``."""
    fakes: dict[object, object] = {asyncio.sleep: clock.async_sleep, time.sleep: clock.sleep}
    for value in list(vars(module).values()):
        for function in _functions(value):
            defaults = function.__kwdefaults__
            if defaults is not None and defaults.get("sleep") in fakes:
                monkeypatch.setitem(defaults, "sleep", fakes[defaults["sleep"]])


def _functions(value: object) -> list[FunctionType]:
    """``value`` if it is a function; a class's own functions; else nothing."""
    if isinstance(value, FunctionType):
        return [value]
    if isinstance(value, type):
        return [member for member in vars(value).values() if isinstance(member, FunctionType)]
    return []
