"""Read an upstream byte stream once and fan it out to independent subscribers.

A ``Relay`` owns the only reader of an upstream body. It keeps a rolling window
of recent bytes, so a subscriber added later (a replacement or second worker)
starts from recent history instead of waiting for new data, and it queues
bytes per subscriber, so one slow subscriber cannot stall the others. Example::

    stream = Relay(lambda: safe_urlopen(url))
    await stream.start()
    await stream.wait_for_bytes(256 << 10)
    stream.subscribe(process.stdin, replay=stream.history(within_sec=3))
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol

import asyncio
import enum
import math
import threading
import time


class Readable(Protocol):
    """A blocking upstream body, such as an ``http.client.HTTPResponse``."""

    def read1(self, amt: int, /) -> bytes: ...
    def close(self) -> None: ...


class Sink(Protocol):
    """A subscriber's input, such as a subprocess's ``asyncio.StreamWriter``."""

    def write(self, data: bytes, /) -> None: ...
    async def drain(self) -> None: ...
    def close(self) -> None: ...


class SubscriptionEnd(enum.Enum):
    """Why a subscription stopped receiving bytes."""

    UPSTREAM_ENDED = "upstream ended"
    OVERFLOW = "overflow"
    SINK_FAILED = "sink failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, kw_only=True, slots=True)
class RelayStats:
    """Upstream delivery so far; times come from the relay's clock."""

    bytes_received: int
    first_byte_at: float | None
    last_byte_at: float | None
    # Longest silence between deliveries: how far ahead a client must buffer.
    max_gap_sec: float = 0.0


class Subscription:
    """One subscriber's queue and the task writing it to the subscriber's sink.

    Attributes:
        end: Why delivery stopped, or None while it continues.
    """

    def __init__(
        self,
        sink: Sink,
        *,
        max_stall_sec: float,
        clock: Callable[[], float],
        on_closed: Callable[[Subscription], None],
    ) -> None:
        self.end: SubscriptionEnd | None = None
        self._sink = sink
        self._max_stall_sec = max_stall_sec
        self._clock = clock
        self._on_closed = on_closed
        self._queue: deque[bytes] = deque()
        # When the sink last started refusing bytes, or None while it accepts them.
        self._stalled_since: float | None = None
        self._ending: SubscriptionEnd | None = None
        self._wake = asyncio.Event()
        self._closed = asyncio.Event()
        self._task = asyncio.create_task(self._pump())
        # A task cancelled before its first step never runs its body, so cleanup
        # in the coroutine's ``finally`` would be skipped; the callback always runs.
        self._task.add_done_callback(self._release)

    def cancel(self) -> None:
        """Stop delivery now and close the sink, dropping queued bytes."""
        self._stop(SubscriptionEnd.CANCELLED)

    async def wait_closed(self) -> None:
        """Wait until the sink is closed."""
        await self._closed.wait()

    def _push(self, chunk: bytes) -> None:
        if self.end is not None or self._ending is not None:
            return
        # A sink is behind only once it has refused bytes for a while: a worker
        # starting up reads nothing while a provider's opening burst arrives.
        stalled_since = self._stalled_since
        if stalled_since is not None and self._clock() - stalled_since > self._max_stall_sec:
            self._stop(SubscriptionEnd.OVERFLOW)
            return
        self._queue.append(chunk)
        self._wake.set()

    def _finish(self, end: SubscriptionEnd) -> None:
        """End once queued bytes are delivered."""
        self._ending = end
        self._wake.set()

    def _stop(self, end: SubscriptionEnd) -> None:
        if self.end is None:
            self.end = end
            self._task.cancel()

    async def _pump(self) -> None:
        try:
            while True:
                while self._queue:
                    self._sink.write(self._queue.popleft())
                    self._stalled_since = self._clock()
                    await self._sink.drain()
                    self._stalled_since = None
                if self._ending is not None:
                    self.end = self._ending
                    return
                self._wake.clear()
                await self._wake.wait()
        except OSError:
            self.end = SubscriptionEnd.SINK_FAILED

    def _release(self, task: asyncio.Task[None]) -> None:
        del task
        self._sink.close()
        self._closed.set()
        self._on_closed(self)


class Feed:
    """A subscriber its owner pulls from, such as an HTTP response body.

    ``drain`` waits until the owner took what was written, so an owner that stops
    pulling backs up its subscription until the relay drops it.
    """

    def __init__(self) -> None:
        self._chunks: deque[bytes] = deque()
        self._written = asyncio.Event()
        self._taken = asyncio.Event()
        self._closed = False

    def write(self, data: bytes, /) -> None:
        self._chunks.append(data)
        self._written.set()

    async def drain(self) -> None:
        while self._chunks:
            self._taken.clear()
            await self._taken.wait()

    def close(self) -> None:
        self._closed = True
        self._written.set()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            while self._chunks:
                chunk = self._chunks.popleft()
                self._taken.set()
                yield chunk
            if self._closed:
                return
            self._written.clear()
            await self._written.wait()


class Relay:
    """The single reader of one upstream body, with history and fan-out.

    Args:
        open_upstream: Opens the upstream body; called once, in a worker thread.
        history_sec: Seconds of recent bytes kept for late subscribers.
        max_stall_sec: Seconds a subscriber may refuse bytes before it is dropped.
        chunk_size: Bytes requested per upstream read.
        clock: Monotonic clock timing history and stats.
    """

    def __init__(
        self,
        open_upstream: Callable[[], Readable],
        *,
        history_sec: float = 30.0,
        max_stall_sec: float = 10.0,
        chunk_size: int = 64 << 10,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._open_upstream = open_upstream
        self._history_sec = history_sec
        self._max_stall_sec = max_stall_sec
        self._chunk_size = chunk_size
        self._clock = clock
        self._history: deque[tuple[float, bytes]] = deque()
        self._buffered_bytes = 0
        self._bytes_received = 0
        self._first_byte_at: float | None = None
        self._last_byte_at: float | None = None
        self._max_gap_sec = 0.0
        self._subscriptions: set[Subscription] = set()
        self._grew = asyncio.Event()
        self._finished = asyncio.Event()
        self._stopping = threading.Event()
        self._upstream: Readable | None = None
        self._error: BaseException | None = None
        self._reader: asyncio.Task[None] | None = None
        self._released = False
        self._closing: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start reading upstream in a worker thread."""
        self._reader = asyncio.create_task(self._run(asyncio.get_running_loop()))

    def history(self, *, within_sec: float = math.inf) -> bytes:
        """Return buffered bytes received within the last ``within_sec`` seconds."""
        cutoff = self._clock() - within_sec
        return b"".join(chunk for at, chunk in self._history if at >= cutoff)

    async def wait_for_bytes(self, size: int) -> None:
        """Wait until the history holds ``size`` bytes or upstream ended."""
        while self._buffered_bytes < size and not self._finished.is_set():
            self._grew.clear()
            await self._grew.wait()

    def subscribe(self, sink: Sink, *, replay: bytes = b"") -> Subscription:
        """Deliver ``replay`` (typically a slice of ``history``), then live bytes, to ``sink``.

        Call without awaiting after reading ``history``, so no byte falls between.

        Args:
            sink: Subscriber input, such as a worker's stdin.
            replay: Bytes delivered before live ones.

        Returns:
            subscription: Its delivery task; cancel it to stop and close ``sink``.
        """
        subscription = Subscription(
            sink,
            max_stall_sec=self._max_stall_sec,
            clock=self._clock,
            on_closed=self._unsubscribed,
        )
        if replay:
            subscription._push(replay)
        if self._finished.is_set():
            subscription._finish(SubscriptionEnd.UPSTREAM_ENDED)
        else:
            self._subscriptions.add(subscription)
        return subscription

    @property
    def stats(self) -> RelayStats:
        return RelayStats(
            bytes_received=self._bytes_received,
            first_byte_at=self._first_byte_at,
            last_byte_at=self._last_byte_at,
            max_gap_sec=self._max_gap_sec,
        )

    @property
    def finished(self) -> bool:
        """Whether upstream ended, failed, or the relay was closed."""
        return self._finished.is_set()

    @property
    def error(self) -> BaseException | None:
        """Why reading stopped, once it has; None after a clean end or ``close``."""
        return self._error

    async def wait_finished(self) -> None:
        """Wait until upstream ended, failed, or the relay was closed."""
        await self._finished.wait()

    async def release(self) -> None:
        """Close now when no one subscribes, else once the last subscription ends."""
        self._released = True
        if not self._subscriptions:
            await self.close()

    def _unsubscribed(self, subscription: Subscription) -> None:
        self._subscriptions.discard(subscription)
        if self._released and not self._subscriptions and self._closing is None:
            self._closing = asyncio.ensure_future(self.close())

    async def close(self) -> None:
        """Stop reading, close upstream, and cancel every subscription."""
        self._stopping.set()
        if self._upstream is not None:
            self._upstream.close()
        subscriptions = list(self._subscriptions)
        for subscription in subscriptions:
            subscription.cancel()
        for subscription in subscriptions:
            await subscription.wait_closed()
        if self._reader is not None:
            await self._reader
            await self._finished.wait()

    async def _run(self, loop: asyncio.AbstractEventLoop) -> None:
        await asyncio.to_thread(self._read_upstream, loop)

    # The end goes through the same callback queue as the chunks: completion of
    # ``to_thread`` can overtake ``call_soon_threadsafe`` callbacks.
    def _read_upstream(self, loop: asyncio.AbstractEventLoop) -> None:
        """Read in a worker thread, then report the end after the last chunk."""
        loop.call_soon_threadsafe(self._end, self._read_chunks(loop))

    def _end(self, error: BaseException | None) -> None:
        self._error = error
        self._finished.set()
        self._grew.set()
        for subscription in list(self._subscriptions):
            subscription._finish(SubscriptionEnd.UPSTREAM_ENDED)

    def _read_chunks(self, loop: asyncio.AbstractEventLoop) -> BaseException | None:
        """Read until end, failure, or ``close``."""
        try:
            self._upstream = self._open_upstream()
        except OSError as error:
            return error
        try:
            while not self._stopping.is_set():
                chunk = self._upstream.read1(self._chunk_size)
                if not chunk:
                    return None
                loop.call_soon_threadsafe(self._receive, chunk)
        # ``close`` closes the body under a pending read, which then fails with
        # either; a closed buffered file raises ValueError.
        except (OSError, ValueError) as error:
            return None if self._stopping.is_set() else error
        finally:
            self._upstream.close()
        return None

    def _receive(self, chunk: bytes) -> None:
        now = self._clock()
        self._history.append((now, chunk))
        self._buffered_bytes += len(chunk)
        self._bytes_received += len(chunk)
        if self._last_byte_at is not None:
            self._max_gap_sec = max(self._max_gap_sec, now - self._last_byte_at)
        if self._first_byte_at is None:
            self._first_byte_at = now
        self._last_byte_at = now
        while self._history and self._history[0][0] < now - self._history_sec:
            self._buffered_bytes -= len(self._history.popleft()[1])
        for subscription in list(self._subscriptions):
            subscription._push(chunk)
        self._grew.set()
