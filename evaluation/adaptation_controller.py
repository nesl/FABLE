"""Allowlisted execution boundary for evaluation disturbances.

The controller never accepts a command string.  A schedule supplies only typed
kind/target/condition/action values, all of which must appear in an immutable
policy.  Privileged execution, when explicitly enabled, is delegated to one
exact root-owned helper using an argument array and ``shell=False``.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from subprocess import CompletedProcess, run
from time import perf_counter_ns
from typing import Callable, Literal

from pydantic import Field, model_validator

from evaluation.disturbance_schedule import (
    DisturbanceKind,
    DisturbanceScheduleController,
    DisturbanceTrigger,
    ScheduledDisturbanceAction,
)
from evaluation.schemas import BaselineId, DisturbanceEvent
from fable.common.base import FrozenFableModel, JSONValue
from fable.common.time import ensure_utc


class AdaptationControllerMode(StrEnum):
    DRY_RUN = "DRY_RUN"
    PROFILED = "PROFILED"
    HOST_HELPER = "HOST_HELPER"


class AllowedDisturbanceTarget(FrozenFableModel):
    target_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    kinds: tuple[DisturbanceKind, ...]
    condition_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _require_allowlists(self):
        if not self.kinds or not self.condition_ids:
            raise ValueError("disturbance target requires kind and condition allowlists")
        return self


class AdaptationControlPolicy(FrozenFableModel):
    schema_version: Literal["fable.adaptation_control_policy.v1"] = "fable.adaptation_control_policy.v1"
    policy_id: str = Field(min_length=1)
    mode: AdaptationControllerMode = AdaptationControllerMode.DRY_RUN
    targets: tuple[AllowedDisturbanceTarget, ...]
    helper_path: Path | None = None
    helper_timeout_seconds: float = Field(default=10, gt=0, le=30)

    @model_validator(mode="after")
    def _validate_policy(self):
        ids = [item.target_id for item in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("disturbance target IDs must be unique")
        if self.mode == AdaptationControllerMode.HOST_HELPER:
            if self.helper_path is None or not self.helper_path.is_absolute():
                raise ValueError("HOST_HELPER requires one absolute helper path")
        elif self.helper_path is not None:
            raise ValueError("helper_path is valid only in HOST_HELPER mode")
        return self


class AdaptationActionResult(FrozenFableModel):
    schema_version: Literal["fable.adaptation_action_result.v1"] = "fable.adaptation_action_result.v1"
    step_id: str
    action: str
    kind: DisturbanceKind
    target_id: str
    condition_id: str
    condition_epoch: int = Field(ge=0)
    mode: AdaptationControllerMode
    status: str = Field(pattern=r"^(DRY_RUN|APPLIED|RESTORED|FAILED)$")
    validated: bool = False
    observed_at: datetime
    helper_argv: tuple[str, ...] = ()
    measurements: dict[str, JSONValue] = Field(default_factory=dict)
    reason: str = ""


ProfileApplier = Callable[
    [ScheduledDisturbanceAction, int],
    dict[str, JSONValue],
]


class CompositeProfileApplier:
    """Route profiled actions by their closed disturbance-kind enum."""

    def __init__(self, appliers: dict[DisturbanceKind, ProfileApplier]) -> None:
        self.appliers = dict(appliers)

    def __call__(
        self,
        action: ScheduledDisturbanceAction,
        condition_epoch: int,
    ) -> dict[str, JSONValue]:
        try:
            applier = self.appliers[action.kind]
        except KeyError as exc:
            raise ValueError(
                f"no profiled applier for {action.kind.value}"
            ) from exc
        return applier(action, condition_epoch)


class AdaptationController:
    """Validate and execute typed scheduled actions with monotonically advancing epochs."""

    def __init__(
        self,
        policy: AdaptationControlPolicy,
        *,
        profile_applier: ProfileApplier | None = None,
    ) -> None:
        self.policy = policy
        self.profile_applier = profile_applier
        self.condition_epoch = 0
        self._active_conditions: dict[str, str] = {}
        self._targets = {item.target_id: item for item in policy.targets}

    def execute(
        self,
        action: ScheduledDisturbanceAction,
        *,
        observed_at: datetime,
    ) -> AdaptationActionResult:
        observed_at = ensure_utc(observed_at)
        target = self._targets.get(action.target_id)
        if target is None:
            raise ValueError(f"disturbance target is not allowlisted: {action.target_id}")
        if action.kind not in target.kinds:
            raise ValueError(
                f"{action.kind.value} is not allowed for target {action.target_id}"
            )
        if action.condition_id not in target.condition_ids:
            raise ValueError(
                f"condition is not allowlisted for {action.target_id}: "
                f"{action.condition_id}"
            )

        prior = self._active_conditions.get(action.target_id)
        if prior != action.condition_id:
            self.condition_epoch += 1
        self._active_conditions[action.target_id] = action.condition_id

        if self.policy.mode == AdaptationControllerMode.DRY_RUN:
            return self._result(
                action,
                observed_at=observed_at,
                status="DRY_RUN",
                validated=False,
                reason="typed action validated; host mutation disabled",
            )
        if self.policy.mode == AdaptationControllerMode.PROFILED:
            measurements = (
                self.profile_applier(action, self.condition_epoch)
                if self.profile_applier is not None
                else {}
            )
            return self._result(
                action,
                observed_at=observed_at,
                status="APPLIED" if action.action == "APPLY" else "RESTORED",
                validated=True,
                measurements=measurements,
                reason="profiled evaluation state updated",
            )
        return self._execute_helper(action, observed_at=observed_at)

    def _execute_helper(
        self,
        action: ScheduledDisturbanceAction,
        *,
        observed_at: datetime,
    ) -> AdaptationActionResult:
        helper = self._validated_helper()
        argv = (
            str(helper),
            "--kind",
            action.kind.value,
            "--target",
            action.target_id,
            "--condition",
            action.condition_id,
            "--action",
            action.action,
            "--condition-epoch",
            str(self.condition_epoch),
        )
        completed: CompletedProcess[str] = run(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.policy.helper_timeout_seconds,
        )
        if completed.returncode != 0:
            return self._result(
                action,
                observed_at=observed_at,
                status="FAILED",
                validated=False,
                helper_argv=argv,
                reason=f"helper exited with status {completed.returncode}",
            )
        try:
            document = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("adaptation helper returned invalid JSON") from exc
        if document.get("validated") is not True:
            raise RuntimeError("adaptation helper did not validate the applied condition")
        measurements = document.get("measurements") or {}
        if not isinstance(measurements, dict):
            raise RuntimeError("adaptation helper measurements must be an object")
        return self._result(
            action,
            observed_at=observed_at,
            status="APPLIED" if action.action == "APPLY" else "RESTORED",
            validated=True,
            helper_argv=argv,
            measurements=measurements,
            reason=str(document.get("reason") or "host condition validated"),
        )

    def _validated_helper(self) -> Path:
        assert self.policy.helper_path is not None
        helper = self.policy.helper_path.resolve(strict=True)
        stat = helper.stat()
        if not helper.is_file():
            raise ValueError("adaptation helper is not a regular file")
        if stat.st_uid != 0:
            raise ValueError("adaptation helper must be owned by root")
        if stat.st_mode & 0o022:
            raise ValueError("adaptation helper cannot be group/other writable")
        if not stat.st_mode & 0o111:
            raise ValueError("adaptation helper is not executable")
        return helper

    def _result(
        self,
        action: ScheduledDisturbanceAction,
        *,
        observed_at: datetime,
        status: str,
        validated: bool,
        helper_argv: tuple[str, ...] = (),
        measurements: dict[str, JSONValue] | None = None,
        reason: str,
    ) -> AdaptationActionResult:
        return AdaptationActionResult(
            step_id=action.step_id,
            action=action.action,
            kind=action.kind,
            target_id=action.target_id,
            condition_id=action.condition_id,
            condition_epoch=self.condition_epoch,
            mode=self.policy.mode,
            status=status,
            validated=validated,
            observed_at=observed_at,
            helper_argv=helper_argv,
            measurements=measurements or {},
            reason=reason,
        )


def disturbance_record(
    result: AdaptationActionResult,
    *,
    run_id: str,
    baseline_id: BaselineId,
    trace_id: str,
    request_id: str,
    scheduled_trigger: str,
) -> DisturbanceEvent:
    """Convert a controller result to the common immutable record."""

    return DisturbanceEvent(
        run_id=run_id,
        baseline_id=baseline_id,
        trace_id=trace_id,
        request_id=request_id,
        event_time=result.observed_at,
        monotonic_timestamp_ns=perf_counter_ns(),
        disturbance_id=f"{result.step_id}:{result.action}",
        disturbance_type=result.kind.value,
        action=result.action,
        target_ids=(result.target_id,),
        condition_epoch=result.condition_epoch,
        scheduled_trigger=scheduled_trigger,
        validated=result.validated,
        metadata={
            "controller_mode": result.mode.value,
            "condition_id": result.condition_id,
            "status": result.status,
            "measurements": result.measurements,
            "reason": result.reason,
        },
    )


class DynamicAdaptationRun:
    """Join semantic scheduling, typed execution, and common record emission."""

    def __init__(
        self,
        *,
        schedule: DisturbanceScheduleController,
        controller: AdaptationController,
        run_id: str,
        baseline_id: BaselineId,
        trace_id: str,
        request_id: str,
        record_sink: Callable[[DisturbanceEvent], None],
    ) -> None:
        self.schedule = schedule
        self.controller = controller
        self.run_id = run_id
        self.baseline_id = baseline_id
        self.trace_id = trace_id
        self.request_id = request_id
        self.record_sink = record_sink
        self._step_triggers = {
            step.step_id: step.trigger for step in schedule.schedule.steps
        }

    def observe(
        self,
        trigger: DisturbanceTrigger,
        *,
        observed_at: datetime,
    ) -> tuple[DisturbanceEvent, ...]:
        self.schedule.observe(trigger, observed_at=observed_at)
        return self.advance(now=observed_at)

    def advance(self, *, now: datetime) -> tuple[DisturbanceEvent, ...]:
        records: list[DisturbanceEvent] = []
        for action in self.schedule.due(now=now):
            result = self.controller.execute(action, observed_at=now)
            record = disturbance_record(
                result,
                run_id=self.run_id,
                baseline_id=self.baseline_id,
                trace_id=self.trace_id,
                request_id=self.request_id,
                scheduled_trigger=self._step_triggers[action.step_id].value,
            )
            self.record_sink(record)
            records.append(record)
        return tuple(records)
