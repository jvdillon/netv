"""Conservative quality fallback and sustained recovery for live playback."""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class PlaybackHealth(BaseModel):
    buffer_seconds: float = Field(ge=0, le=86400, allow_inf_nan=False)
    waiting: bool
    # Zero means unavailable. The client supplies a recent download sample.
    observed_bitrate: float = Field(default=0, ge=0, le=1e12, allow_inf_nan=False)
    required_bitrate: float = Field(default=0, ge=0, le=1e12, allow_inf_nan=False)


@dataclass
class PlaybackPolicy:
    started: float | None = None
    last_sample: float | None = None
    unhealthy_since: float | None = None
    bandwidth_saver: bool = False
    saver_since: float | None = None
    recovery_since: float | None = None
    recovery_last_transfer: float | None = None
    recovery_samples: int = 0

    def observe(
        self,
        health: PlaybackHealth,
        now: float,
        recovery_bitrate: float = 0,
        *,
        recovery_ready: bool = True,
    ) -> bool:
        if self.bandwidth_saver:
            if self.saver_since is None:
                self.saver_since = now
            gap = self.last_sample is not None and now - self.last_sample > 10
            self.last_sample = now
            healthy = not health.waiting and health.buffer_seconds >= 8 and recovery_bitrate > 0
            if health.observed_bitrate > 0 and health.observed_bitrate < recovery_bitrate * 1.5:
                healthy = False
            if (
                gap
                or not healthy
                or (
                    self.recovery_last_transfer is not None
                    and now - self.recovery_last_transfer > 10
                )
            ):
                self.recovery_since = None
                self.recovery_last_transfer = None
                self.recovery_samples = 0
            if healthy and health.observed_bitrate > 0:
                if self.recovery_since is None:
                    self.recovery_since = now
                self.recovery_last_transfer = now
                self.recovery_samples += 1
            if (
                now - self.saver_since >= 60
                and self.recovery_since is not None
                and now - self.recovery_since >= 30
                and self.recovery_samples >= 3
                and recovery_ready
            ):
                self.bandwidth_saver = False
                self.saver_since = None
                self.recovery_since = None
                self.recovery_last_transfer = None
                self.recovery_samples = 0
                self.unhealthy_since = None
                self.started = now
                return False
            return True
        if self.started is None:
            self.started = now
        # A backgrounded/paused client must establish fresh evidence on return.
        if self.last_sample is not None and now - self.last_sample > 10:
            self.unhealthy_since = None
        self.last_sample = now
        slow_download = (
            health.required_bitrate > 0
            and 0 < health.observed_bitrate < health.required_bitrate * 1.2
        )
        unhealthy = health.buffer_seconds < 3 and (health.waiting or slow_download)
        # Allow initial tuning and buffering to settle before making a decision.
        if now - self.started < 15 or not unhealthy:
            self.unhealthy_since = None
        elif self.unhealthy_since is None:
            self.unhealthy_since = now
        elif now - self.unhealthy_since >= 8:
            self.bandwidth_saver = True
            self.saver_since = now
        return self.bandwidth_saver


@dataclass
class UpgradePolicy:
    """Keep recent capacity evidence across HLS download and encoder batching gaps."""

    samples: list[tuple[float, float]] = field(default_factory=list)
    reason: str = "waiting for throughput"

    def observe(
        self, health: PlaybackHealth, target_bitrate: float, ready: bool, now: float
    ) -> bool:
        self.samples = [(at, rate) for at, rate in self.samples if now - at <= 10]
        if health.waiting or health.buffer_seconds < 3:
            self.samples.clear()
            self.reason = "playback pressure"
            return False
        # Zero means no transfer measurement, not zero available bandwidth.
        if health.observed_bitrate > 0:
            if target_bitrate > 0 and health.observed_bitrate < target_bitrate * 1.5:
                self.samples.clear()
                self.reason = "insufficient bandwidth headroom"
                return False
            self.samples.append((now, health.observed_bitrate))
        if target_bitrate <= 0:
            self.reason = "4K segments unavailable"
        elif any(rate < target_bitrate * 1.5 for _, rate in self.samples):
            self.reason = "insufficient bandwidth headroom"
        elif len(self.samples) < 3 or self.samples[-1][0] - self.samples[0][0] < 6:
            self.reason = "waiting for throughput evidence"
        elif health.buffer_seconds < 6:
            self.reason = "building playback buffer"
        elif not ready:
            self.reason = "high-quality encoder catching up"
        else:
            self.reason = "ready"
            return True
        return False


_LIVE_STALL_CONFIRM_SECONDS = 2
_LIVE_HIGH_WARMUP_SECONDS = 30
_LIVE_RESTART_WINDOW_SECONDS = 300
_LIVE_RESTART_LIMIT = 3


@dataclass
class LiveRecoveryPolicy:
    """Confirm media stalls and bound stage restarts independently of playback pressure."""

    high_started: float
    failures: dict[str, float] = field(default_factory=dict)
    attempts: dict[str, list[float]] = field(default_factory=dict)
    pending: set[str] = field(default_factory=set)

    def stalled(
        self,
        stage: str,
        process_alive: bool,
        output_ready: bool,
        now: float,
    ) -> bool:
        if process_alive and output_ready:
            self.failures.pop(stage, None)
            return False
        if (
            stage == "high"
            and process_alive
            and now - self.high_started < _LIVE_HIGH_WARMUP_SECONDS
        ):
            self.failures.pop(stage, None)
            return False
        since = self.failures.setdefault(stage, now)
        return now - since >= _LIVE_STALL_CONFIRM_SECONDS

    def permit_restart(self, action: str, now: float) -> bool:
        if action in self.pending:
            return False
        attempts = [
            attempt
            for attempt in self.attempts.get(action, [])
            if now - attempt < _LIVE_RESTART_WINDOW_SECONDS
        ]
        self.attempts[action] = attempts
        if len(attempts) >= _LIVE_RESTART_LIMIT:
            return False
        if attempts:
            cooldown = 30 if action == "high" else (5 if len(attempts) == 1 else 15)
            if now - attempts[-1] < cooldown:
                return False
        attempts.append(now)
        self.pending.add(action)
        return True

    def finish_restart(self, action: str) -> None:
        self.pending.discard(action)

    def mark_stage_started(self, stage: str, now: float) -> None:
        self.failures.pop(stage, None)
        if stage == "high":
            self.high_started = now

    def reset_pipeline(self, now: float) -> None:
        self.failures.clear()
        self.high_started = now
