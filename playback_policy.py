"""Conservative quality fallback and sustained recovery for live playback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from pydantic import BaseModel, Field


class PlaybackHealth(BaseModel):
    buffer_seconds: float = Field(ge=0, le=86_400, allow_inf_nan=False)
    waiting: bool
    # Zero means unavailable. The client supplies a recent download sample.
    observed_bitrate: float = Field(default=0, ge=0, le=1e12, allow_inf_nan=False)
    required_bitrate: float = Field(default=0, ge=0, le=1e12, allow_inf_nan=False)


@dataclass(kw_only=True, slots=True, frozen=True)
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
    ) -> PlaybackPolicy:
        if self.bandwidth_saver:
            saver_since = now if self.saver_since is None else self.saver_since
            gap = self.last_sample is not None and now - self.last_sample > 10
            healthy = not health.waiting and health.buffer_seconds >= 8 and recovery_bitrate > 0
            if health.observed_bitrate > 0 and health.observed_bitrate < recovery_bitrate * 1.5:
                healthy = False
            recovery_since = self.recovery_since
            recovery_last_transfer = self.recovery_last_transfer
            recovery_samples = self.recovery_samples
            if (
                gap
                or not healthy
                or (recovery_last_transfer is not None and now - recovery_last_transfer > 10)
            ):
                recovery_since = None
                recovery_last_transfer = None
                recovery_samples = 0
            if healthy and health.observed_bitrate > 0:
                if recovery_since is None:
                    recovery_since = now
                recovery_last_transfer = now
                recovery_samples += 1
            if (
                now - saver_since >= 60
                and recovery_since is not None
                and now - recovery_since >= 30
                and recovery_samples >= 3
            ):
                return replace(
                    self,
                    bandwidth_saver=False,
                    saver_since=None,
                    recovery_since=None,
                    recovery_last_transfer=None,
                    recovery_samples=0,
                    unhealthy_since=None,
                    started=now,
                    last_sample=now,
                )
            return replace(
                self,
                saver_since=saver_since,
                last_sample=now,
                recovery_since=recovery_since,
                recovery_last_transfer=recovery_last_transfer,
                recovery_samples=recovery_samples,
            )
        started = now if self.started is None else self.started
        unhealthy_since = self.unhealthy_since
        # A backgrounded/paused client must establish fresh evidence on return.
        if self.last_sample is not None and now - self.last_sample > 10:
            unhealthy_since = None
        slow_download = (
            health.required_bitrate > 0
            and 0 < health.observed_bitrate < health.required_bitrate * 1.2
        )
        unhealthy = health.buffer_seconds < 3 and (health.waiting or slow_download)
        bandwidth_saver = self.bandwidth_saver
        saver_since = self.saver_since
        # Allow initial tuning and buffering to settle before making a decision.
        if now - started < 15 or not unhealthy:
            unhealthy_since = None
        elif unhealthy_since is None:
            unhealthy_since = now
        elif now - unhealthy_since >= 8:
            bandwidth_saver = True
            saver_since = now
        return replace(
            self,
            started=started,
            last_sample=now,
            unhealthy_since=unhealthy_since,
            bandwidth_saver=bandwidth_saver,
            saver_since=saver_since,
        )


@dataclass(kw_only=True, slots=True, frozen=True)
class LiveRecoveryPolicy:
    """Confirm media stalls and bound stage restarts independently of playback pressure."""

    stall_confirm_seconds: float = 2
    restart_window_seconds: float = 300
    restart_limit: int = 3
    failures: Mapping[str, float] = field(default_factory=lambda: MappingProxyType[str, float]({}))
    attempts: Mapping[str, tuple[float, ...]] = field(
        default_factory=lambda: MappingProxyType[str, tuple[float, ...]]({}),
    )
    pending: frozenset[str] = frozenset()

    def stalled(
        self,
        stage: str,
        process_alive: bool,
        output_ready: bool,
        now: float,
    ) -> tuple[LiveRecoveryPolicy, bool]:
        failures = dict(self.failures)
        if process_alive and output_ready:
            failures.pop(stage, None)
            return replace(self, failures=MappingProxyType(failures)), False
        since = failures.setdefault(stage, now)
        return replace(
            self,
            failures=MappingProxyType(failures),
        ), now - since >= self.stall_confirm_seconds

    def permit_restart(self, action: str, now: float) -> tuple[LiveRecoveryPolicy, bool]:
        if action in self.pending or self.exhausted(action, now):
            return self, False
        attempts = self._recent_attempts(action, now)
        if attempts:
            cooldown = 5 if len(attempts) == 1 else 15
            if now - attempts[-1] < cooldown:
                return self, False
        return replace(
            self,
            attempts=MappingProxyType({**self.attempts, action: (*attempts, now)}),
            pending=self.pending | {action},
        ), True

    def exhausted(self, action: str, now: float) -> bool:
        """Report whether ``action`` has spent its restart budget for the window.

        Args:
            action: Restart action, such as ``"pipeline"``.
            now: Monotonic time in seconds.

        Returns:
            exhausted: Whether the window holds ``restart_limit`` attempts.
        """
        return len(self._recent_attempts(action, now)) >= self.restart_limit

    def _recent_attempts(self, action: str, now: float) -> tuple[float, ...]:
        return tuple(
            attempt
            for attempt in self.attempts.get(action, ())
            if now - attempt < self.restart_window_seconds
        )

    def finish_restart(self, action: str) -> LiveRecoveryPolicy:
        return replace(self, pending=self.pending - {action})

    def reset_pipeline(self) -> LiveRecoveryPolicy:
        return replace(self, failures=MappingProxyType({}))
