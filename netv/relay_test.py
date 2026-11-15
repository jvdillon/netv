"""Tests for relay.py: one upstream reader fanned out to independent subscribers."""

from __future__ import annotations

import asyncio

import pytest

from netv import relay
from netv.testing import FakeClock, FakeSink, FakeUpstream


def _relay(
    upstream: FakeUpstream,
    clock: FakeClock,
    *,
    history_sec: float = 30.0,
    max_pending_bytes: int = 16 << 20,
) -> relay.Relay:
    return relay.Relay(
        lambda: upstream,
        clock=clock.monotonic,
        history_sec=history_sec,
        max_pending_bytes=max_pending_bytes,
    )


@pytest.mark.asyncio
async def test_subscriber_gets_its_replay_then_live_bytes_in_order():
    clock, upstream = FakeClock(), FakeUpstream()
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
async def test_history_window_selects_recent_bytes():
    clock, upstream = FakeClock(), FakeUpstream()
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
async def test_history_keeps_only_the_configured_window():
    clock, upstream = FakeClock(), FakeUpstream()
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
    # The sample size counts bytes still buffered, not bytes ever received.
    waiter = asyncio.ensure_future(stream.wait_for_bytes(2))
    await asyncio.sleep(0)
    assert not waiter.done()
    upstream.end()
    await waiter
    assert stream.finished
    stats = stream.stats
    assert (stats.bytes_received, stats.first_byte_at, stats.last_byte_at) == (
        2,
        clock.now - 11,
        clock.now,
    )
    await stream.close()


@pytest.mark.asyncio
async def test_stats_record_the_longest_silence_between_deliveries():
    """Panels relaying HLS send TS in segment bursts; clients must buffer past the gaps."""
    clock, upstream = FakeClock(), FakeUpstream()
    stream = _relay(upstream, clock)
    await stream.start()
    assert stream.stats.max_gap_sec == 0
    for delay, chunk in ((0, b"a"), (3, b"b"), (1, b"c")):
        clock.advance(delay)
        upstream.feed(chunk)
        await stream.wait_for_bytes(len(stream.history()) + 1)
    assert stream.stats.max_gap_sec == 3
    await stream.close()


@pytest.mark.asyncio
async def test_bytes_arriving_after_a_drop_are_not_queued():
    """Chunks already in flight when a subscriber is dropped are discarded, not delivered."""
    clock, upstream = FakeClock(), FakeUpstream()
    stream = _relay(upstream, clock, max_pending_bytes=4)
    await stream.start()
    upstream.feed(b"aaa")
    await stream.wait_for_bytes(3)
    stuck = FakeSink(blocked=True)
    subscription = stream.subscribe(stuck, replay=stream.history())
    await stuck.wait_for(3)
    # In one tick: a backlog that overflows, then a chunk for the dropped subscriber.
    subscription._push(b"bbbbb")
    subscription._push(b"c")
    assert subscription.end == relay.SubscriptionEnd.OVERFLOW
    await subscription.wait_closed()
    assert bytes(stuck.written) == b"aaa"
    await stream.close()


@pytest.mark.asyncio
async def test_slow_subscriber_is_dropped_without_stalling_others():
    clock, upstream = FakeClock(), FakeUpstream()
    stream = _relay(upstream, clock, max_pending_bytes=4)
    await stream.start()
    upstream.feed(b"aaa")
    await stream.wait_for_bytes(3)
    stuck, healthy = FakeSink(blocked=True), FakeSink()
    stuck_subscription = stream.subscribe(stuck, replay=stream.history())
    stream.subscribe(healthy, replay=stream.history())
    # The stuck sink is now refusing bytes; what arrives next is a backlog, not a burst.
    await stuck.wait_for(3)
    for chunk in (b"bbb", b"ccc"):
        upstream.feed(chunk)
    await healthy.wait_for(9)
    await stuck_subscription.wait_closed()
    assert stuck_subscription.end == relay.SubscriptionEnd.OVERFLOW
    assert stuck.closed
    assert bytes(healthy.written) == b"aaabbbccc"
    await stream.close()


@pytest.mark.asyncio
async def test_upstream_end_delivers_queued_bytes_then_closes_subscribers():
    clock, upstream = FakeClock(), FakeUpstream()
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
async def test_upstream_failure_is_reported():
    clock, upstream = FakeClock(), FakeUpstream()
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
async def test_failing_sink_ends_only_its_subscription():
    clock, upstream = FakeClock(), FakeUpstream()
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
async def test_cancel_and_close_release_everything():
    clock, upstream = FakeClock(), FakeUpstream()
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
async def test_wait_for_bytes_returns_as_soon_as_enough_arrived():
    clock, upstream = FakeClock(), FakeUpstream()
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
async def test_closing_under_a_pending_read_is_a_clean_stop():
    """A buffered body closed under a pending read raises ValueError; that is not an error."""
    clock, upstream = FakeClock(), FakeUpstream()
    stream = _relay(upstream, clock)
    await stream.start()
    upstream.fail(ValueError("read of closed file"))
    await stream.wait_finished()
    assert isinstance(stream.error, ValueError)

    upstream = FakeUpstream()
    stream = _relay(upstream, clock)
    await stream.start()
    stream._stopping.set()
    upstream.fail(ValueError("read of closed file"))
    await stream.wait_finished()
    assert stream.error is None
    await stream.close()


@pytest.mark.asyncio
async def test_feed_yields_what_its_subscription_delivers():
    clock, upstream = FakeClock(), FakeUpstream()
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
async def test_feed_holds_back_its_subscription_until_pulled():
    clock, upstream = FakeClock(), FakeUpstream()
    stream = _relay(upstream, clock, max_pending_bytes=4)
    await stream.start()
    upstream.feed(b"aaa")
    await stream.wait_for_bytes(3)
    feed = relay.Feed()
    subscription = stream.subscribe(feed, replay=stream.history())
    # The pump writes the replay and waits for it to be taken.
    await asyncio.sleep(0)
    for chunk in (b"bbb", b"ccc"):
        upstream.feed(chunk)
    # Nothing pulls, so the first chunk is never taken and the rest overflow.
    await subscription.wait_closed()
    assert subscription.end == relay.SubscriptionEnd.OVERFLOW
    assert [chunk async for chunk in feed] == [b"aaa"]
    await stream.close()


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
