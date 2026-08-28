"""Deterministic semantic-triggered disturbance schedules.

This module decides *when* an allowlisted controller should apply or restore a
network, capacity, provider, or node condition.  It intentionally performs no
host mutation and accepts no shell command.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from fable.common.base import FrozenFableModel
from fable.common.time import ensure_utc


class DisturbanceTrigger(StrEnum):
    BEFORE_DEMAND = "BEFORE_DEMAND"
    AFTER_PLAN_DISPATCH = "AFTER_PLAN_DISPATCH"
    DURING_PROVIDER_EXECUTION = "DURING_PROVIDER_EXECUTION"


class DisturbanceKind(StrEnum):
    NETWORK_PROFILE = "NETWORK_PROFILE"
    CAPACITY_PROFILE = "CAPACITY_PROFILE"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    NODE_UNAVAILABLE = "NODE_UNAVAILABLE"


class DisturbanceStep(FrozenFableModel):
    step_id: str = Field(min_length=1)
    trigger: DisturbanceTrigger
    kind: DisturbanceKind
    target_id: str = Field(min_length=1)
    condition_id: str = Field(min_length=1)
    delay_ms: int = Field(default=1000, ge=0)
    duration_ms: int = Field(default=20_000, ge=1, le=30_000)
    restore_condition_id: str = Field(default="N0", min_length=1)


class DisturbanceSchedule(FrozenFableModel):
    schema_version: Literal["fable.disturbance_schedule.v1"] = "fable.disturbance_schedule.v1"
    schedule_id: str = Field(min_length=1)
    steps: tuple[DisturbanceStep, ...]
    post_restore_observation_ms: int = Field(default=10_000, ge=0)

    @model_validator(mode="after")
    def _unique_steps(self):
        ids = [item.step_id for item in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("disturbance step IDs must be unique")
        return self


class ScheduledDisturbanceAction(FrozenFableModel):
    step_id: str
    action: str = Field(pattern=r"^(APPLY|RESTORE)$")
    kind: DisturbanceKind
    target_id: str
    condition_id: str
    due_at: datetime


class DisturbanceScheduleController:
    """Advance an immutable schedule from typed semantic lifecycle events."""

    def __init__(self, schedule: DisturbanceSchedule) -> None:
        self.schedule = schedule
        self._triggered: set[str] = set()
        self._emitted: set[tuple[str, str]] = set()
        self._actions: list[ScheduledDisturbanceAction] = []

    def observe(
        self,
        trigger: DisturbanceTrigger,
        *,
        observed_at: datetime,
    ) -> None:
        observed_at = ensure_utc(observed_at)
        for step in self.schedule.steps:
            if step.trigger != trigger or step.step_id in self._triggered:
                continue
            self._triggered.add(step.step_id)
            apply_at = observed_at + timedelta(milliseconds=step.delay_ms)
            self._actions.extend(
                (
                    ScheduledDisturbanceAction(
                        step_id=step.step_id,
                        action="APPLY",
                        kind=step.kind,
                        target_id=step.target_id,
                        condition_id=step.condition_id,
                        due_at=apply_at,
                    ),
                    ScheduledDisturbanceAction(
                        step_id=step.step_id,
                        action="RESTORE",
                        kind=step.kind,
                        target_id=step.target_id,
                        condition_id=step.restore_condition_id,
                        due_at=apply_at
                        + timedelta(milliseconds=step.duration_ms),
                    ),
                )
            )

    def due(self, *, now: datetime) -> tuple[ScheduledDisturbanceAction, ...]:
        now = ensure_utc(now)
        ready = tuple(
            sorted(
                (
                    item
                    for item in self._actions
                    if item.due_at <= now
                    and (item.step_id, item.action) not in self._emitted
                ),
                key=lambda item: (item.due_at, item.step_id, item.action),
            )
        )
        self._emitted.update((item.step_id, item.action) for item in ready)
        return ready

    @property
    def complete(self) -> bool:
        return len(self._emitted) == 2 * len(self.schedule.steps)
