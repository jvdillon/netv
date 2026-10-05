"""Playback fallback should react to sustained pressure, not transient stalls."""

from pydantic import ValidationError

import pytest

from playback_policy import LiveRecoveryPolicy, PlaybackHealth, PlaybackPolicy, UpgradePolicy


def health(*, buffer=0, waiting=True, observed=0, required=0):
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
def test_healthy_or_unknown_samples_do_not_trigger(sample):
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
def test_invalid_metrics_rejected(value):
    with pytest.raises(ValidationError):
        health(buffer=value)
    with pytest.raises(ValidationError):
        health(observed=value)


def test_upgrade_survives_idle_download_polls_and_encoder_batches():
    policy = UpgradePolicy()
    # A 10-second source batch makes the encoder temporarily trail the low
    # rendition. Network samples still establish capacity during that time.
    for now in range(0, 8, 2):
        sample = health(buffer=8, waiting=False, observed=100_000_000 if now % 4 == 0 else 0)
        assert not policy.observe(sample, 60_000_000, ready=False, now=now)
    assert policy.observe(
        health(buffer=8, waiting=False, observed=100_000_000), 60_000_000, ready=True, now=8
    )


@pytest.mark.parametrize("observed", [0, 50_000_000])
def test_upgrade_does_not_invent_bandwidth(observed):
    policy = UpgradePolicy()
    for now in range(0, 30, 2):
        assert not policy.observe(
            health(buffer=12, waiting=False, observed=observed), 60_000_000, ready=True, now=now
        )


def test_upgrade_expires_old_samples_and_resets_on_stall():
    policy = UpgradePolicy()
    good = health(buffer=10, waiting=False, observed=100_000_000)
    for now in (0, 4, 8):
        policy.observe(good, 60_000_000, ready=False, now=now)
    assert not policy.observe(health(buffer=10, waiting=False), 60_000_000, ready=True, now=20)
    for now in (22, 26, 30):
        policy.observe(good, 60_000_000, ready=False, now=now)
    assert not policy.observe(health(), 60_000_000, ready=True, now=31)
    assert not policy.observe(good, 60_000_000, ready=True, now=32)


def test_upgrade_keeps_final_readiness_and_buffer_requirements():
    policy = UpgradePolicy()
    good = health(buffer=8, waiting=False, observed=100_000_000)
    for now in (0, 4, 8):
        assert not policy.observe(good, 60_000_000, ready=False, now=now)
    assert not policy.observe(health(buffer=4, waiting=False), 60_000_000, ready=True, now=9)
    assert policy.observe(health(buffer=8, waiting=False), 60_000_000, ready=True, now=10)


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
def test_recovery_rejects_insufficient_or_unknown_capacity(bad):
    policy = PlaybackPolicy(bandwidth_saver=True)
    for now in range(0, 180, 2):
        assert policy.observe(bad, now, 20_000_000)


def test_recovery_survives_bursty_downloads_but_expires_on_pause():
    policy = PlaybackPolicy(bandwidth_saver=True)
    for now in range(0, 60, 2):
        assert policy.observe(
            health(buffer=10, waiting=False, observed=40_000_000 if now % 4 == 0 else 0),
            now,
            20_000_000,
        )
    # No downloads for over ten seconds invalidates earlier evidence.
    for now in range(60, 74, 2):
        assert policy.observe(
            health(buffer=10, waiting=False), now, 20_000_000, recovery_ready=False
        )
    good = health(buffer=10, waiting=False, observed=40_000_000)
    for now in range(74, 104, 2):
        assert policy.observe(good, now, 20_000_000)
    assert not policy.observe(good, 104, 20_000_000)


def test_recovery_waits_for_encoder_readiness_and_resets_after_telemetry_gap():
    policy = PlaybackPolicy(bandwidth_saver=True)
    good = health(buffer=10, waiting=False, observed=40_000_000)
    for now in range(0, 70, 2):
        assert policy.observe(good, now, 20_000_000, recovery_ready=False)
    for now in range(100, 130, 2):
        assert policy.observe(good, now, 20_000_000)
    assert not policy.observe(good, 130, 20_000_000)


def test_recovery_needs_known_target():
    policy = PlaybackPolicy(bandwidth_saver=True)
    for now in range(0, 100, 2):
        assert policy.observe(health(buffer=10, waiting=False, observed=100_000_000), now)


def test_live_recovery_confirms_stalls_and_allows_high_warmup():
    policy = LiveRecoveryPolicy(high_started=0)
    assert not policy.stalled("input", True, False, 100)
    assert not policy.stalled("input", True, True, 101)
    assert not policy.stalled("input", True, False, 102)
    assert policy.stalled("input", True, False, 104)

    assert not policy.stalled("high", True, False, 10)
    assert not policy.stalled("high", True, False, 29)
    assert not policy.stalled("high", True, False, 30)
    assert policy.stalled("high", True, False, 32)


def test_dead_high_bypasses_warmup_but_still_requires_confirmation():
    policy = LiveRecoveryPolicy(high_started=100)
    assert not policy.stalled("high", False, False, 101)
    assert policy.stalled("high", False, False, 103)


def test_live_restarts_have_cooldowns_and_sliding_budget():
    policy = LiveRecoveryPolicy(high_started=0)
    assert policy.permit_restart("high", 0)
    assert not policy.permit_restart("high", 100)
    policy.finish_restart("high")
    assert not policy.permit_restart("high", 29)
    assert policy.permit_restart("high", 30)
    policy.finish_restart("high")
    assert policy.permit_restart("high", 60)
    policy.finish_restart("high")
    assert not policy.permit_restart("high", 299)
    assert policy.permit_restart("high", 301)

    assert policy.permit_restart("pipeline", 0)
    policy.finish_restart("pipeline")
    assert not policy.permit_restart("pipeline", 4)
    assert policy.permit_restart("pipeline", 5)
    policy.finish_restart("pipeline")
    assert not policy.permit_restart("pipeline", 19)
    assert policy.permit_restart("pipeline", 20)
