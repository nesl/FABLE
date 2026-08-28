from __future__ import annotations

from time import sleep

from evaluation.calibration_execution import (
    CalibrationAdapterRegistry,
    CalibrationInvocationResult,
    execute_calibration_campaign,
)
from evaluation.experiments.e0_calibration import CalibrationTarget, build


class _MeasuredAdapter:
    def create(self):
        return {"calls": 0}

    def invoke(self, instance, target):
        instance["calls"] += 1
        return CalibrationInvocationResult(
            quality_score=0.9,
            ambiguity_score=0.1,
        )


class _HangingAdapter:
    def create(self):
        return object()

    def invoke(self, instance, target):
        sleep(5)
        return CalibrationInvocationResult(
            quality_score=1,
            ambiguity_score=0,
        )


def _target(name: str) -> CalibrationTarget:
    return CalibrationTarget(
        target_id=f"target-{name}",
        provider_id=name,
        tier="sensor",
        input_class="detection_set.v1",
    )


def test_campaign_records_measurements_and_explicit_missing_adapter(tmp_path) -> None:
    measured = _target("measured")
    missing = _target("missing")
    runs = build(
        (measured, missing),
        warm_repetitions=2,
        cold_repetitions=1,
    )
    registry = CalibrationAdapterRegistry()
    registry.register(measured, _MeasuredAdapter())
    result = execute_calibration_campaign(
        runs,
        registry,
        tmp_path,
        invocation_timeout_seconds=1,
    )
    assert result.planned_runs == 6
    assert result.completed_observations == 3
    assert result.excluded_runs == 3
    assert result.target_coverage == 0.5
    observations = (tmp_path / "observations.jsonl").read_text().splitlines()
    exclusions = (tmp_path / "exclusions.jsonl").read_text().splitlines()
    assert len(observations) == 3
    assert len(exclusions) == 3


def test_hung_adapter_is_terminated_at_deadline(tmp_path) -> None:
    target = _target("hanging")
    runs = build(
        (target,),
        warm_repetitions=1,
        cold_repetitions=1,
    )
    registry = CalibrationAdapterRegistry()
    registry.register(target, _HangingAdapter())
    result = execute_calibration_campaign(
        runs,
        registry,
        tmp_path,
        invocation_timeout_seconds=0.05,
    )
    assert result.completed_observations == 0
    assert result.timed_out_runs == 2
