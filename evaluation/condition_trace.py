"""Exogenous, monotonic-time operating-condition traces for RQ3a."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from fable.common.base import FrozenFableModel


class ConditionAction(StrEnum):
    APPLY_NETWORK_PROFILE = "APPLY_NETWORK_PROFILE"
    RESTORE_NETWORK_PROFILE = "RESTORE_NETWORK_PROFILE"
    APPLY_COMPUTE_CONTENTION = "APPLY_COMPUTE_CONTENTION"
    CLEAR_COMPUTE_CONTENTION = "CLEAR_COMPUTE_CONTENTION"
    FAIL_PROVIDER = "FAIL_PROVIDER"
    RESTORE_PROVIDER = "RESTORE_PROVIDER"
    FAIL_LINK = "FAIL_LINK"
    RESTORE_LINK = "RESTORE_LINK"


class ConditionAnchor(StrEnum):
    TRACE_START = "TRACE_START"
    ADMISSION = "ADMISSION"


class ConditionTransition(FrozenFableModel):
    transition_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    offset_s: float = Field(ge=0)
    action: ConditionAction
    target_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_.:-]+$"
    )
    profile_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_.:-]+$"
    )
    duration_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _required_fields(self):
        if self.action in {
            ConditionAction.APPLY_NETWORK_PROFILE,
            ConditionAction.APPLY_COMPUTE_CONTENTION,
        } and self.profile_id is None:
            raise ValueError(f"{self.action.value} requires profile_id")
        if self.action in {
            ConditionAction.FAIL_PROVIDER,
            ConditionAction.RESTORE_PROVIDER,
            ConditionAction.FAIL_LINK,
            ConditionAction.RESTORE_LINK,
        } and self.target_id is None:
            raise ValueError(f"{self.action.value} requires target_id")
        if self.action in {ConditionAction.FAIL_LINK, ConditionAction.RESTORE_LINK}:
            assert self.target_id is not None
            if not (
                self.target_id == "physical_link:rpi_to_jetson"
                or self.target_id.startswith("link:s_")
            ):
                raise ValueError("link transitions require a link target")
        return self


class ConditionTrace(FrozenFableModel):
    schema_version: Literal["fable.condition_trace.v1"] = "fable.condition_trace.v1"
    trace_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    initial_network_profile: str = Field(min_length=1)
    initial_compute_profile: str = Field(min_length=1)
    anchor: ConditionAnchor = ConditionAnchor.TRACE_START
    transitions: tuple[ConditionTransition, ...]
    duration_s: float = Field(gt=0)
    random_seed: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_trace(self):
        ids = [item.transition_id for item in self.transitions]
        if len(ids) != len(set(ids)):
            raise ValueError("condition transition IDs must be unique")
        offsets = [item.offset_s for item in self.transitions]
        if offsets != sorted(offsets):
            raise ValueError("condition transitions must be ordered by offset_s")
        for item in self.transitions:
            if item.offset_s > self.duration_s:
                raise ValueError(
                    f"transition {item.transition_id} is after trace duration"
                )
            if item.duration_s is not None and (
                item.offset_s + item.duration_s > self.duration_s
            ):
                raise ValueError(
                    f"transition {item.transition_id} duration exceeds trace"
                )
        return self


class DueConditionTransition(FrozenFableModel):
    transition: ConditionTransition
    requested_offset_s: float


class MonotonicConditionTraceController:
    """Release immutable transitions solely from elapsed experiment time."""

    def __init__(self, trace: ConditionTrace) -> None:
        self.trace = trace
        self.transitions = self._expanded_transitions(trace)
        self._emitted: set[str] = set()

    def due(self, *, elapsed_s: float) -> tuple[DueConditionTransition, ...]:
        if elapsed_s < 0:
            raise ValueError("elapsed_s cannot be negative")
        due = tuple(
            DueConditionTransition(
                transition=item,
                requested_offset_s=item.offset_s,
            )
            for item in self.transitions
            if item.offset_s <= elapsed_s and item.transition_id not in self._emitted
        )
        self._emitted.update(item.transition.transition_id for item in due)
        return due

    @property
    def complete(self) -> bool:
        return len(self._emitted) == len(self.transitions)

    @staticmethod
    def _expanded_transitions(
        trace: ConditionTrace,
    ) -> tuple[ConditionTransition, ...]:
        inverse = {
            ConditionAction.APPLY_NETWORK_PROFILE: ConditionAction.RESTORE_NETWORK_PROFILE,
            ConditionAction.APPLY_COMPUTE_CONTENTION: ConditionAction.CLEAR_COMPUTE_CONTENTION,
            ConditionAction.FAIL_PROVIDER: ConditionAction.RESTORE_PROVIDER,
            ConditionAction.FAIL_LINK: ConditionAction.RESTORE_LINK,
        }
        expanded = list(trace.transitions)
        for item in trace.transitions:
            if item.duration_s is None or item.action not in inverse:
                continue
            profile_id = item.profile_id
            if item.action == ConditionAction.APPLY_NETWORK_PROFILE:
                profile_id = trace.initial_network_profile
            elif item.action == ConditionAction.APPLY_COMPUTE_CONTENTION:
                profile_id = trace.initial_compute_profile
            expanded.append(
                ConditionTransition(
                    transition_id=f"{item.transition_id}:auto-restore",
                    offset_s=item.offset_s + item.duration_s,
                    action=inverse[item.action],
                    target_id=item.target_id,
                    profile_id=profile_id,
                )
            )
        return tuple(sorted(expanded, key=lambda item: (item.offset_s, item.transition_id)))


def midpoint_disturbance_schedule(duration_s: float) -> tuple[float, float]:
    """Disturb from 50% through 95% of the labeled replay duration."""

    if duration_s < 40:
        raise ValueError("midpoint disturbance requires a replay of at least 40 seconds")
    apply_at = duration_s / 2.0
    clear_at = duration_s * 0.95
    if clear_at <= apply_at:
        raise ValueError("replay is too short for midpoint disturbance and recovery")
    return round(apply_at, 3), round(clear_at, 3)


def classify_disturbance_exposure(
    *,
    demand_start_s: float,
    demand_end_s: float,
    disturbance_start_s: float,
    disturbance_end_s: float,
) -> str:
    """Classify a demand interval against one apply/recovery interval."""

    if demand_end_s < demand_start_s:
        raise ValueError("demand interval is reversed")
    if disturbance_end_s <= disturbance_start_s:
        raise ValueError("disturbance interval is empty or reversed")
    if demand_end_s < disturbance_start_s or demand_start_s > disturbance_end_s:
        return "NO_ACTIVE_DEMAND_EXPOSURE"
    if demand_start_s < disturbance_start_s <= demand_end_s:
        return "ACTIVE_DEMAND_CROSSES_DISTURBANCE_ONSET"
    if disturbance_start_s <= demand_start_s < disturbance_end_s:
        if demand_end_s >= disturbance_end_s:
            return "ACTIVE_DEMAND_CROSSES_RECOVERY"
        return "DEMAND_BEGINS_UNDER_DISTURBANCE"
    return "NO_ACTIVE_DEMAND_EXPOSURE"
