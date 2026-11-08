"""Playback fallback should react to sustained pressure, not transient stalls."""

from pydantic import ValidationError

import pytest

from playback_policy import LiveRecoveryPolicy, PlaybackHealth, PlaybackPolicy


def health(
    *,
    buffer: float = 0,
    waiting: bool = True,
    observed: float = 0,
    required: float = 0,
) -> PlaybackHealth:
    return PlaybackHealth(
        buffer_seconds=buffer,
        waiting=waiting,
        observed_bitrate=observed,
        required_bitrate=required,
    )


def test_startup_grace_and_sustained_stall_then_recovery_cooldown():
    policy = PlaybackPolicy()
    for now in range(0, 24, 2):
        assert not policy.observe(health(), now)
    assert policy.observe(health(), 24)
    assert policy.observe(health(buffer=30, waiting=False), 26)


def test_low_throughput_with_dwindling_buffer_triggers_fallback():
    policy = PlaybackPolicy()
    slow = health(buffer=2, waiting=False, observed=1_000_000, required=4_000_000)
    for now in range(0, 24, 2):
        assert not policy.observe(slow, now)
    assert policy.observe(slow, 24)


@pytest.mark.parametrize(
    "sample",
    [
        health(buffer=20, observed=1_000_000, required=4_000_000),
        health(waiting=False),  # Paused, or throughput unavailable.
        health(waiting=False, observed=10_000_000, required=4_000_000),
    ],
)
def test_healthy_or_unknown_samples_do_not_trigger(sample: PlaybackHealth):
    policy = PlaybackPolicy()
    assert not any(policy.observe(sample, now) for now in range(0, 60, 2))


def test_short_stall_and_pause_reset_evidence():
    policy = PlaybackPolicy()
    for now in range(0, 22, 2):
        assert not policy.observe(health(), now)
    assert not policy.observe(health(waiting=False), 22)
    for now in range(24, 32, 2):
        assert not policy.observe(health(), now)
    assert policy.observe(health(), 32)


def test_background_gap_resets_evidence():
    policy = PlaybackPolicy()
    for now in range(0, 22, 2):
        assert not policy.observe(health(), now)
    assert not policy.observe(health(), 100)
    for now in range(102, 108, 2):
        assert not policy.observe(health(), now)
    assert policy.observe(health(), 108)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_metrics_rejected(value: float):
    with pytest.raises(ValidationError):
        health(buffer=value)
    with pytest.raises(ValidationError):
        health(observed=value)


def test_recovery_requires_cooldown_and_sustained_headroom():
    policy = PlaybackPolicy(bandwidth_saver=True)
    good = health(buffer=10, waiting=False, observed=30_000_000)
    for now in range(0, 60, 2):
        assert policy.observe(good, now, 20_000_000)
    assert not policy.observe(good, 60, 20_000_000)
    # Recovery has its own startup grace; a later sustained stall can fall back.
    for now in range(62, 84, 2):
        assert not policy.observe(health(), now, 20_000_000)
    assert policy.observe(health(), 84, 20_000_000)


@pytest.mark.parametrize(
    "bad",
    [
        health(buffer=10, waiting=False, observed=29_000_000),
        health(buffer=7, waiting=False, observed=40_000_000),
        health(buffer=10, waiting=True, observed=40_000_000),
        health(buffer=10, waiting=False),
    ],
)
def test_recovery_rejects_insufficient_or_unknown_capacity(bad: PlaybackHealth):
    policy = PlaybackPolicy(bandwidth_saver=True)
    for now in range(0, 180, 2):
        assert policy.observe(bad, now, 20_000_000)


def test_recovery_survives_bursty_downloads_but_expires_on_pause():
    policy = PlaybackPolicy(bandwidth_saver=True)
    for now in range(0, 40, 2):
        assert policy.observe(
            health(buffer=10, waiting=False, observed=40_000_000 if now % 4 == 0 else 0),
            now,
            20_000_000,
        )
    # No downloads for over ten seconds invalidates earlier evidence.
    for now in range(40, 54, 2):
        assert policy.observe(health(buffer=10, waiting=False), now, 20_000_000)
    good = health(buffer=10, waiting=False, observed=40_000_000)
    for now in range(54, 84, 2):
        assert policy.observe(good, now, 20_000_000)
    assert not policy.observe(good, 84, 20_000_000)


def test_recovery_resets_after_telemetry_gap():
    policy = PlaybackPolicy(bandwidth_saver=True)
    good = health(buffer=10, waiting=False, observed=40_000_000)
    for now in range(0, 40, 2):
        assert policy.observe(good, now, 20_000_000)
    for now in range(100, 130, 2):
        assert policy.observe(good, now, 20_000_000)
    assert not policy.observe(good, 130, 20_000_000)


def test_recovery_needs_known_target():
    policy = PlaybackPolicy(bandwidth_saver=True)
    for now in range(0, 100, 2):
        assert policy.observe(health(buffer=10, waiting=False, observed=100_000_000), now)


def test_live_recovery_confirms_stalls():
    policy = LiveRecoveryPolicy()
    assert not policy.stalled("input", True, False, 100)
    assert not policy.stalled("input", True, True, 101)
    assert not policy.stalled("input", True, False, 102)
    assert policy.stalled("input", True, False, 104)
    assert not policy.stalled("low", False, False, 101)
    assert policy.stalled("low", False, False, 103)


def test_live_restarts_have_cooldowns_and_sliding_budget():
    policy = LiveRecoveryPolicy()
    assert policy.permit_restart("pipeline", 0)
    assert not policy.permit_restart("pipeline", 100)
    policy.finish_restart("pipeline")
    assert not policy.permit_restart("pipeline", 4)
    assert policy.permit_restart("pipeline", 5)
    policy.finish_restart("pipeline")
    assert not policy.permit_restart("pipeline", 19)
    assert policy.permit_restart("pipeline", 20)
    policy.finish_restart("pipeline")
    assert not policy.permit_restart("pipeline", 299)
    assert policy.permit_restart("pipeline", 301)


def test_live_recovery_reports_a_spent_restart_budget_until_attempts_expire():
    policy = LiveRecoveryPolicy()
    assert not policy.exhausted("pipeline", 0)
    for now in (0, 5, 20):
        assert policy.permit_restart("pipeline", now)
        policy.finish_restart("pipeline")
    assert policy.exhausted("pipeline", 299)
    assert not policy.exhausted("pipeline", 300)


def test_live_recovery_reset_clears_failures():
    policy = LiveRecoveryPolicy()
    policy.failures = {"input": 1, "low": 2}
    policy.reset_pipeline()
    assert policy.failures == {}


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
