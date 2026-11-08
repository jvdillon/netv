"""Playback fallback should react to sustained pressure, not transient stalls."""

from pydantic import ValidationError

import pytest

from netv.playback_policy import LiveRecoveryPolicy, PlaybackHealth, PlaybackPolicy


def test_observation_returns_a_new_policy_without_changing_the_previous_state():
    policy = PlaybackPolicy()
    updated = policy.observe(PlaybackHealth(buffer_seconds=0, waiting=True), 0, 1000)
    assert updated == PlaybackPolicy(started=0, last_sample=0)
    assert policy == PlaybackPolicy()


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
        policy = policy.observe(health(), now)
        assert not policy.bandwidth_saver
    policy = policy.observe(health(), 24)
    assert policy.bandwidth_saver
    policy = policy.observe(health(buffer=30, waiting=False), 26)
    assert policy.bandwidth_saver


def test_low_throughput_with_dwindling_buffer_triggers_fallback():
    policy = PlaybackPolicy()
    slow = health(buffer=2, waiting=False, observed=1_000_000, required=4_000_000)
    for now in range(0, 24, 2):
        policy = policy.observe(slow, now)
        assert not policy.bandwidth_saver
    assert policy.observe(slow, 24).bandwidth_saver


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
    for now in range(0, 60, 2):
        policy = policy.observe(sample, now)
        assert not policy.bandwidth_saver


def test_short_stall_and_pause_reset_evidence():
    policy = PlaybackPolicy()
    for now in range(0, 22, 2):
        policy = policy.observe(health(), now)
        assert not policy.bandwidth_saver
    policy = policy.observe(health(waiting=False), 22)
    assert not policy.bandwidth_saver
    for now in range(24, 32, 2):
        policy = policy.observe(health(), now)
        assert not policy.bandwidth_saver
    assert policy.observe(health(), 32).bandwidth_saver


def test_background_gap_resets_evidence():
    policy = PlaybackPolicy()
    for now in range(0, 22, 2):
        policy = policy.observe(health(), now)
        assert not policy.bandwidth_saver
    policy = policy.observe(health(), 100)
    assert not policy.bandwidth_saver
    for now in range(102, 108, 2):
        policy = policy.observe(health(), now)
        assert not policy.bandwidth_saver
    assert policy.observe(health(), 108).bandwidth_saver


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
        policy = policy.observe(good, now, 20_000_000)
        assert policy.bandwidth_saver
    policy = policy.observe(good, 60, 20_000_000)
    assert not policy.bandwidth_saver
    # Recovery has its own startup grace; a later sustained stall can fall back.
    for now in range(62, 84, 2):
        policy = policy.observe(health(), now, 20_000_000)
        assert not policy.bandwidth_saver
    assert policy.observe(health(), 84, 20_000_000).bandwidth_saver


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
        policy = policy.observe(bad, now, 20_000_000)
        assert policy.bandwidth_saver


def test_recovery_survives_bursty_downloads_but_expires_on_pause():
    policy = PlaybackPolicy(bandwidth_saver=True)
    for now in range(0, 40, 2):
        policy = policy.observe(
            health(buffer=10, waiting=False, observed=40_000_000 if now % 4 == 0 else 0),
            now,
            20_000_000,
        )
        assert policy.bandwidth_saver
    # No downloads for over ten seconds invalidates earlier evidence.
    for now in range(40, 54, 2):
        policy = policy.observe(health(buffer=10, waiting=False), now, 20_000_000)
        assert policy.bandwidth_saver
    good = health(buffer=10, waiting=False, observed=40_000_000)
    for now in range(54, 84, 2):
        policy = policy.observe(good, now, 20_000_000)
        assert policy.bandwidth_saver
    assert not policy.observe(good, 84, 20_000_000).bandwidth_saver


def test_recovery_resets_after_telemetry_gap():
    policy = PlaybackPolicy(bandwidth_saver=True)
    good = health(buffer=10, waiting=False, observed=40_000_000)
    for now in range(0, 40, 2):
        policy = policy.observe(good, now, 20_000_000)
        assert policy.bandwidth_saver
    for now in range(100, 130, 2):
        policy = policy.observe(good, now, 20_000_000)
        assert policy.bandwidth_saver
    assert not policy.observe(good, 130, 20_000_000).bandwidth_saver


def test_recovery_needs_known_target():
    policy = PlaybackPolicy(bandwidth_saver=True)
    for now in range(0, 100, 2):
        policy = policy.observe(health(buffer=10, waiting=False, observed=100_000_000), now)
        assert policy.bandwidth_saver


def test_live_recovery_confirms_stalls():
    original = policy = LiveRecoveryPolicy()
    for stage, alive, ready, now, expected in [
        ("input", True, False, 100, False),
        ("input", True, True, 101, False),
        ("input", True, False, 102, False),
        ("input", True, False, 104, True),
        ("low", False, False, 101, False),
        ("low", False, False, 103, True),
    ]:
        policy, stalled = policy.stalled(stage, alive, ready, now)
        assert stalled is expected
    assert policy.failures == {"input": 102, "low": 101}
    assert original.failures == {}


def test_live_restarts_have_cooldowns_and_sliding_budget():
    original = policy = LiveRecoveryPolicy()
    policy, permitted = policy.permit_restart("pipeline", 0)
    assert permitted
    policy, permitted = policy.permit_restart("pipeline", 100)
    assert not permitted
    policy = policy.finish_restart("pipeline")
    for now, expected in [
        (4, False),
        (5, True),
        (19, False),
        (20, True),
        (299, False),
        (301, True),
    ]:
        policy, permitted = policy.permit_restart("pipeline", now)
        assert permitted is expected
        policy = policy.finish_restart("pipeline")
    assert policy.attempts == {"pipeline": (5, 20, 301)}
    assert original.attempts == {}
    assert original.pending == frozenset()


def test_live_recovery_reports_a_spent_restart_budget_until_attempts_expire():
    policy = LiveRecoveryPolicy()
    assert not policy.exhausted("pipeline", 0)
    for now in (0, 5, 20):
        policy, permitted = policy.permit_restart("pipeline", now)
        assert permitted
        policy = policy.finish_restart("pipeline")
    assert policy.exhausted("pipeline", 299)
    assert not policy.exhausted("pipeline", 300)
    assert policy.attempts == {"pipeline": (0, 5, 20)}


def test_live_recovery_reset_clears_failures():
    policy = LiveRecoveryPolicy(failures={"input": 1, "low": 2})
    updated = policy.reset_pipeline()
    assert updated.failures == {}
    assert policy.failures == {"input": 1, "low": 2}


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
