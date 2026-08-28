from __future__ import annotations

import json

import pytest

from evaluation.experiments.e8_scaling import build
from evaluation.runner import JsonlEventStore
from evaluation.scaling_execution import (
    ScalingExecutionProfile,
    execute_profiled_scaling_run,
)
from evaluation.schemas import BaselineId


def _profile(*, calibrated: bool = True) -> ScalingExecutionProfile:
    return ScalingExecutionProfile(
        profile_id="test-profile",
        calibrated=calibrated,
        calibration_source="unit test",
        base_planning_latency_ms=2,
        latency_per_label_ms=0.08,
        base_cpu_seconds_per_request=0.01,
        cpu_seconds_per_label=0.0005,
        base_memory_bytes=128 * 1024 * 1024,
        memory_bytes_per_live_hypothesis=65536,
        network_bytes_per_provider=4096,
        nominal_timely_recall=0.98,
        overload_label_threshold=4096,
        overload_recall_penalty_per_threshold=0.05,
    )


def _same_point_runs():
    runs = build(repetitions=1, network_profiles=("good_network",))
    grouped = {}
    for run in runs:
        grouped.setdefault(run.point.model_dump_json(), []).append(run)
    return next(
        tuple(items)
        for items in grouped.values()
        if {item.baseline_id for item in items}
        == {
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
            BaselineId.FABLE,
            BaselineId.FABLE_NO_SHARING,
        }
    )


def test_profiled_scaling_is_deterministic_and_emits_common_records(
    tmp_path,
) -> None:
    run = _same_point_runs()[0]
    first = execute_profiled_scaling_run(run, _profile(), tmp_path / "a")
    second = execute_profiled_scaling_run(run, _profile(), tmp_path / "b")
    assert first == second
    store = JsonlEventStore(tmp_path / "a" / run.run_id)
    assert len(store.read("plan_decision")) == 1
    assert len(store.read("resource_sample")) == 1
    document = json.loads(
        (tmp_path / "a" / run.run_id / "scaling_result.json").read_text()
    )
    assert document["execution_mode"] == "PROFILE_DRIVEN"


def test_no_sharing_has_no_savings_for_same_scaling_point(tmp_path) -> None:
    results = {
        run.baseline_id: execute_profiled_scaling_run(
            run,
            _profile(),
            tmp_path,
        )
        for run in _same_point_runs()
    }
    assert results[BaselineId.FABLE].sharing_savings > 0
    assert (
        results[BaselineId.FABLE_NO_SHARING].effective_provider_invocations
        > results[BaselineId.FABLE].effective_provider_invocations
    )
    assert results[BaselineId.FABLE_NO_SHARING].sharing_savings == 0


def test_unmeasured_profile_is_fail_closed(tmp_path) -> None:
    run = _same_point_runs()[0]
    with pytest.raises(ValueError, match="implementation fixtures"):
        execute_profiled_scaling_run(run, _profile(calibrated=False), tmp_path)
    result = execute_profiled_scaling_run(
        run,
        _profile(calibrated=False),
        tmp_path,
        allow_unmeasured_profile=True,
    )
    assert result.calibrated is False
