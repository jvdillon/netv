"""Tests for relay.py: one upstream reader fanned out to independent subscribers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import asyncio
import threading

import pytest
import pytest_asyncio

from netv import relay
from netv.testing import FakeClock, FakeSink, FakeUpstream, FakeUpstreams


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


# Async, so teardown runs before the event loop closes: a test that fails before
# closing its relay would otherwise leave the reader thread blocked in read1.
@pytest_asyncio.fixture
async def upstreams() -> AsyncIterator[FakeUpstreams]:
    opened = FakeUpstreams()
    yield opened
    opened.close()


@pytest.fixture
def upstream(upstreams: FakeUpstreams) -> FakeUpstream:
    return upstreams()


def _relay(
    upstream: FakeUpstream,
    clock: FakeClock,
    *,
    history_sec: float = 30.0,
    max_stall_sec: float = 10.0,
) -> relay.Relay:
    return relay.Relay(
        lambda: upstream,
        clock=clock.monotonic,
        history_sec=history_sec,
        max_stall_sec=max_stall_sec,
    )


@pytest.mark.asyncio
async def test_subscriber_gets_its_replay_then_live_bytes_in_order(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    stream = _relay(upstream, clock)
    await stream.start()
    upstream.feed(b"ab")
    await stream.wait_for_bytes(2)
    assert stream.history() == b"ab"

    replayed, live = FakeSink(), FakeSink()
    stream.subscribe(replayed, replay=stream.history())
    stream.subscribe(live)
    upstream.feed(b"cd")
    await replayed.wait_for(4)
    await live.wait_for(2)
    assert (bytes(replayed.written), bytes(live.written)) == (b"abcd", b"cd")
    await stream.close()


@pytest.mark.asyncio
async def test_history_window_selects_recent_bytes(clock: FakeClock, upstream: FakeUpstream):
    stream = _relay(upstream, clock)
    await stream.start()
    upstream.feed(b"old")
    await stream.wait_for_bytes(3)
    clock.advance(5)
    upstream.feed(b"new")
    await stream.wait_for_bytes(6)
    assert stream.history(within_sec=2) == b"new"
    assert stream.history() == b"oldnew"
    await stream.close()


@pytest.mark.asyncio
async def test_history_keeps_only_the_configured_window(clock: FakeClock, upstream: FakeUpstream):
    stream = _relay(upstream, clock, history_sec=10)
    await stream.start()
    sink = FakeSink()
    stream.subscribe(sink)
    upstream.feed(b"a")
    await sink.wait_for(1)
    clock.advance(11)
    upstream.feed(b"b")
    await sink.wait_for(2)
    assert stream.history() == b"b"
    # The sample size counts bytes still buffered, not bytes ever received: with
    # one byte buffered, the wait suspends on its first step instead of returning.
    pending = stream.wait_for_bytes(2)
    try:
        pending.send(None)
    except StopIteration:
        pytest.fail("wait_for_bytes(2) returned with one byte buffered")
    pending.close()
    upstream.end()
    await stream.wait_for_bytes(2)
    assert stream.finished
    stats = stream.stats
    assert (stats.bytes_received, stats.first_byte_at, stats.last_byte_at) == (
        2,
        clock.now - 11,
        clock.now,
    )
    await stream.close()


@pytest.mark.asyncio
async def test_stats_record_a_recurring_silence_between_deliveries(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    """Panels relaying HLS send TS in segment bursts; clients must buffer past the gaps."""
    stream = _relay(upstream, clock)
    await stream.start()
    assert stream.stats.burst_gap_sec == 0
    for delay, chunk in ((0, b"a"), (9, b"b"), (3, b"c"), (1, b"d"), (4, b"e"), (0.5, b"f")):
        clock.advance(delay)
        upstream.feed(chunk)
        await stream.wait_for_bytes(len(stream.history()) + 1)
        if chunk == b"d":
            # Two long silences are not yet a pattern.
            assert stream.stats.burst_gap_sec == 1
    # The third longest: a single 9 s stall does not set the reserve.
    assert stream.stats.burst_gap_sec == 3
    assert stream.stats.delivery_sec == 17.5
    await stream.close()


def test_stats_before_any_delivery_span_nothing():
    stats = relay.RelayStats(bytes_received=0, first_byte_at=None, last_byte_at=None)
    assert stats.delivery_sec == 0


@pytest.mark.asyncio
async def test_bytes_arriving_after_a_drop_are_not_queued(clock: FakeClock, upstream: FakeUpstream):
    """Chunks already in flight when a subscriber is dropped are discarded, not delivered."""
    stream = _relay(upstream, clock)
    await stream.start()
    upstream.feed(b"aaa")
    await stream.wait_for_bytes(3)
    stuck = FakeSink(blocked=True)
    subscription = stream.subscribe(stuck, replay=stream.history())
    await stuck.wait_for(3)
    clock.advance(11)
    # In one tick: a chunk that finds the sink stalled, then one for the dropped subscriber.
    subscription._push(b"bbbbb")
    subscription._push(b"c")
    assert subscription.end == relay.SubscriptionEnd.OVERFLOW
    await subscription.wait_closed()
    assert bytes(stuck.written) == b"aaa"
    await stream.close()


@pytest.mark.asyncio
async def test_defaults_keep_thirty_seconds_and_drop_after_ten_stalled(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    stream = relay.Relay(lambda: upstream, clock=clock.monotonic)
    assert stream.error is None
    await stream.start()
    upstream.feed(b"a")
    await stream.wait_for_bytes(1)
    # Upstream is read in 64 KiB pieces.
    assert upstream.requested[0] == 64 << 10
    stuck = FakeSink(blocked=True)
    subscription = stream.subscribe(stuck, replay=stream.history())
    await stuck.wait_for(1)
    clock.advance(10)
    subscription._push(b"b")
    assert subscription.end is None
    clock.advance(0.5)
    subscription._push(b"b")
    assert subscription.end == relay.SubscriptionEnd.OVERFLOW
    clock.advance(19.5)
    upstream.feed(b"c")
    await stream.wait_for_bytes(2)
    # Thirty seconds of history: "a" arrived exactly 30 s ago.
    assert stream.history() == b"ac"
    clock.advance(0.5)
    # A subscriber receives "d" only after the relay recorded it.
    witness = FakeSink()
    stream.subscribe(witness)
    upstream.feed(b"d")
    await witness.wait_for(1)
    assert stream.history() == b"cd"
    assert subscription.end == relay.SubscriptionEnd.OVERFLOW
    await stream.close()


@pytest.mark.asyncio
async def test_release_closes_at_once_without_subscribers(clock: FakeClock, upstream: FakeUpstream):
    stream = _relay(upstream, clock)
    await stream.start()
    await stream.release()
    assert stream.finished
    assert upstream.closed


@pytest.mark.asyncio
async def test_a_released_relay_stays_open_until_its_last_subscriber_ends(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    """A recording keeps the upstream open after the viewer leaves, then frees it."""
    stream = _relay(upstream, clock)
    await stream.start()
    viewer, recorder = FakeSink(), FakeSink()
    watching = stream.subscribe(viewer)
    recording = stream.subscribe(recorder)
    watching.cancel()
    await watching.wait_closed()
    await stream.release()
    assert not stream.finished
    upstream.feed(b"live")
    await recorder.wait_for(4)
    recording.cancel()
    await stream.wait_finished()
    assert upstream.closed
    assert bytes(viewer.written) == b""


@pytest.mark.asyncio
async def test_a_released_relay_survives_losing_one_of_several_subscribers(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    stream = _relay(upstream, clock)
    await stream.start()
    first, second = FakeSink(), FakeSink()
    leaving = stream.subscribe(first)
    staying = stream.subscribe(second)
    await stream.release()
    leaving.cancel()
    await leaving.wait_closed()
    # Live bytes reach the remaining subscriber before anything closes it.
    upstream.feed(b"live")
    delivered = asyncio.ensure_future(second.wait_for(4))
    closed = asyncio.ensure_future(staying.wait_closed())
    await asyncio.wait({delivered, closed}, return_when=asyncio.FIRST_COMPLETED)
    assert staying.end is None
    assert bytes(second.written) == b"live"
    closed.cancel()
    staying.cancel()
    await stream.wait_finished()
    assert upstream.closed


@pytest.mark.asyncio
async def test_closing_an_unstarted_relay_is_quiet():
    stream = relay.Relay(FakeUpstream, clock=FakeClock().monotonic)
    await stream.close()
    assert stream.error is None


@pytest.mark.asyncio
async def test_a_finishing_subscription_takes_no_new_bytes(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    stream = _relay(upstream, clock)
    await stream.start()
    sink = FakeSink()
    subscription = stream.subscribe(sink, replay=b"a")
    subscription._finish(relay.SubscriptionEnd.UPSTREAM_ENDED)
    subscription._push(b"late")
    await subscription.wait_closed()
    assert bytes(sink.written) == b"a"
    await stream.close()


@pytest.mark.asyncio
async def test_a_burst_while_the_sink_is_briefly_busy_is_kept(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    """A panel's opening burst (~25 MB) lands while the worker starts up and reads nothing."""
    stream = _relay(upstream, clock)
    await stream.start()
    upstream.feed(b"a")
    await stream.wait_for_bytes(1)
    starting = FakeSink(blocked=True)
    subscription = stream.subscribe(starting, replay=stream.history())
    await starting.wait_for(1)
    clock.advance(9)
    for _ in range(64):
        subscription._push(b"x" * (1 << 20))
    assert subscription.end is None
    await stream.close()


@pytest.mark.asyncio
async def test_slow_subscriber_is_dropped_without_stalling_others(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    stream = _relay(upstream, clock)
    await stream.start()
    upstream.feed(b"aaa")
    await stream.wait_for_bytes(3)
    stuck, healthy = FakeSink(blocked=True), FakeSink()
    stuck_subscription = stream.subscribe(stuck, replay=stream.history())
    stream.subscribe(healthy, replay=stream.history())
    # The stuck sink has refused bytes for longer than a worker takes to start.
    await stuck.wait_for(3)
    clock.advance(11)
    for chunk in (b"bbb", b"ccc"):
        upstream.feed(chunk)
    await healthy.wait_for(9)
    await stuck_subscription.wait_closed()
    assert stuck_subscription.end == relay.SubscriptionEnd.OVERFLOW
    assert stuck.closed
    assert bytes(healthy.written) == b"aaabbbccc"
    await stream.close()


@pytest.mark.asyncio
async def test_upstream_end_delivers_queued_bytes_then_closes_subscribers(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    stream = _relay(upstream, clock)
    await stream.start()
    sink = FakeSink()
    subscription = stream.subscribe(sink)
    upstream.feed(b"last")
    upstream.end()
    await subscription.wait_closed()
    assert (bytes(sink.written), sink.closed) == (b"last", True)
    assert subscription.end == relay.SubscriptionEnd.UPSTREAM_ENDED
    await stream.wait_finished()
    assert stream.error is None
    late = stream.subscribe(FakeSink())
    await late.wait_closed()
    assert late.end == relay.SubscriptionEnd.UPSTREAM_ENDED
    await stream.close()


@pytest.mark.asyncio
async def test_upstream_failure_is_reported(clock: FakeClock, upstream: FakeUpstream):
    stream = _relay(upstream, clock)
    await stream.start()
    upstream.fail(ConnectionResetError("reset by peer"))
    await stream.wait_finished()
    assert isinstance(stream.error, ConnectionResetError)
    await stream.wait_for_bytes(1)
    assert stream.history() == b""
    await stream.close()


@pytest.mark.asyncio
async def test_open_failure_is_reported():
    def refuse() -> FakeUpstream:
        raise OSError("connection refused")

    stream = relay.Relay(refuse, clock=FakeClock().monotonic)
    await stream.start()
    await stream.wait_finished()
    assert isinstance(stream.error, OSError)
    await stream.close()


@pytest.mark.asyncio
async def test_failing_sink_ends_only_its_subscription(clock: FakeClock, upstream: FakeUpstream):
    stream = _relay(upstream, clock)
    await stream.start()
    broken, healthy = FakeSink(error=BrokenPipeError()), FakeSink()
    broken_subscription = stream.subscribe(broken)
    stream.subscribe(healthy)
    upstream.feed(b"xy")
    await broken_subscription.wait_closed()
    await healthy.wait_for(2)
    assert broken_subscription.end == relay.SubscriptionEnd.SINK_FAILED
    assert broken.closed
    await stream.close()


@pytest.mark.asyncio
async def test_cancel_and_close_release_everything(clock: FakeClock, upstream: FakeUpstream):
    stream = _relay(upstream, clock)
    await stream.start()
    first, second = FakeSink(), FakeSink()
    cancelled = stream.subscribe(first)
    remaining = stream.subscribe(second)
    cancelled.cancel()
    await cancelled.wait_closed()
    assert cancelled.end == relay.SubscriptionEnd.CANCELLED
    await stream.close()
    await remaining.wait_closed()
    assert remaining.end == relay.SubscriptionEnd.CANCELLED
    assert (first.closed, second.closed, upstream.closed) == (True, True, True)
    await stream.close()


@pytest.mark.asyncio
async def test_wait_for_bytes_returns_as_soon_as_enough_arrived(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    stream = _relay(upstream, clock)
    await stream.start()
    waiter = asyncio.ensure_future(stream.wait_for_bytes(4))
    upstream.feed(b"ab")
    await stream.wait_for_bytes(2)
    assert not waiter.done()
    assert not stream.finished
    upstream.feed(b"cd")
    await waiter
    assert stream.history() == b"abcd"
    await stream.close()


@pytest.mark.asyncio
async def test_closing_under_a_pending_read_is_a_clean_stop(
    clock: FakeClock,
    upstreams: FakeUpstreams,
):
    """A buffered body closed under a pending read raises ValueError; that is not an error."""
    upstream = upstreams()
    stream = _relay(upstream, clock)
    await stream.start()
    upstream.fail(ValueError("read of closed file"))
    await stream.wait_finished()
    assert isinstance(stream.error, ValueError)

    upstream = upstreams()
    stream = _relay(upstream, clock)
    await stream.start()
    # Stop only once the reader is blocked in read1, as close() finds it.
    assert await asyncio.to_thread(upstream.reading.wait, 1)
    stream._stopping.set()
    upstream.fail(ValueError("read of closed file"))
    await stream.wait_finished()
    assert stream.error is None
    await stream.close()


@pytest.mark.asyncio
async def test_a_relay_reads_on_its_own_thread_not_the_shared_pool(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    # A server with few cores has a small default pool; a relay held its worker for
    # the stream's life, so a few streams stalled every other to_thread call.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    stream = _relay(upstream, clock)
    await stream.start()
    assert await asyncio.wait_for(asyncio.to_thread(upstream.reading.wait, 5), 5)
    await stream.close()


def test_a_relay_whose_loop_closes_stops_reading_and_closes_upstream(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    loop = asyncio.new_event_loop()
    stream = _relay(upstream, clock)
    loop.run_until_complete(stream.start())
    loop.run_until_complete(asyncio.to_thread(upstream.reading.wait, 5))
    loop.close()
    # Its thread finds no loop to deliver to: it stops, quietly, closing upstream.
    upstream.feed(b"late")
    for thread in threading.enumerate():
        if thread.name == "relay":
            thread.join(5)
    assert upstream.closed


@pytest.mark.asyncio
async def test_feed_yields_what_its_subscription_delivers(clock: FakeClock, upstream: FakeUpstream):
    stream = _relay(upstream, clock)
    feed = relay.Feed()
    stream.subscribe(feed)
    await stream.start()
    upstream.feed(b"ab")
    upstream.feed(b"cd")
    upstream.end()
    assert [chunk async for chunk in feed] == [b"ab", b"cd"]
    await stream.close()


@pytest.mark.asyncio
async def test_feed_holds_back_its_subscription_until_pulled(
    clock: FakeClock,
    upstream: FakeUpstream,
):
    stream = _relay(upstream, clock)
    await stream.start()
    upstream.feed(b"aaa")
    await stream.wait_for_bytes(3)
    feed = relay.Feed()
    subscription = stream.subscribe(feed, replay=stream.history())
    # The pump writes the replay, then waits for it to be taken.
    await feed._written.wait()
    clock.advance(11)
    # A witness holding both chunks means the relay pushed both to the feed too.
    witness = FakeSink()
    stream.subscribe(witness)
    for chunk in (b"bbb", b"ccc"):
        upstream.feed(chunk)
    await witness.wait_for(6)
    # Nothing pulls, so the first chunk is never taken and the rest overflow.
    assert subscription.end == relay.SubscriptionEnd.OVERFLOW
    await subscription.wait_closed()
    assert [chunk async for chunk in feed] == [b"aaa"]
    await stream.close()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
