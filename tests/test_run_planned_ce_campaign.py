import json
from pathlib import Path

from evaluation.experiments.matrix import PlannedRun
from evaluation.schemas import BaselineId
from scripts.run_planned_ce_campaign import (
    _calibration_rank,
    _condition_rank,
    _condition_slug,
    _result_is_valid_for_run,
    _trace_has_network_mutation,
)


def test_empty_condition_id_is_nominal_and_fable_calibrates_before_b1(tmp_path: Path):
    fable = PlannedRun(
        run_id="fable",
        question="RQ1_END_TO_END",
        experiment_id="experiment-1",
        baseline_id="FABLE",
        mode="FULL_STACK",
    )
    b1 = fable.model_copy(
        update={"run_id": "b1", "baseline_id": BaselineId.B1_STATIC_WHOLE_EVENT}
    )
    assert fable.condition_trace_id == ""
    assert _calibration_rank(fable) < _calibration_rank(b1)
    assert _condition_rank("", disturbed_first=False) == 0


def _run(tmp_path: Path) -> PlannedRun:
    trace = tmp_path / "degraded.json"
    trace.write_text(json.dumps({
        "trace_id": "uplink-degraded",
        "transitions": [{
            "transition_id": "degrade",
            "action": "APPLY_NETWORK_PROFILE",
            "offset_s": 20,
        }],
    }))
    return PlannedRun.model_validate({
        "run_id": "run-1",
        "question": "RQ3_OPERATING_ADAPTATION",
        "experiment_id": "experiment-1",
        "baseline_id": "FABLE",
        "mode": "FULL_STACK",
        "condition_trace_id": "uplink-degraded",
        "condition_trace_path": str(trace),
    })


def test_condition_is_part_of_output_identity(tmp_path: Path):
    run = _run(tmp_path)
    assert _condition_slug(run) == "uplink-degraded-offset-0s"
    assert _trace_has_network_mutation(run)


def test_condition_result_requires_trace_and_validated_transition(tmp_path: Path):
    run = _run(tmp_path)
    result = tmp_path / "result.json"
    base = {
        "suite": "pilot",
        "classification": "TRUE_POSITIVE",
        "elapsed_seconds": 30,
        "condition_trace": {"trace_id": "uplink-degraded", "transitions": [{
            "transition_id": "degrade", "offset_s": 20,
        }]},
        "disturbance_results": [],
    }
    result.write_text(json.dumps(base))
    assert not _result_is_valid_for_run(result, run)
    base["disturbance_results"] = [{"transition_id": "degrade", "validated": True}]
    result.write_text(json.dumps(base))
    assert _result_is_valid_for_run(result, run)


def test_netwaggle_epoch_and_profile_validate_transition(tmp_path: Path):
    run = _run(tmp_path)
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "suite": "pilot",
        "classification": "TRUE_POSITIVE",
        "elapsed_seconds": 30,
        "condition_trace": {"trace_id": "uplink-degraded", "transitions": [{
            "transition_id": "degrade", "offset_s": 20,
        }]},
        "disturbance_results": [{
            "transition_id": "degrade",
            "response": {"condition_epoch": 1, "profile": "L1@s_orin13"},
        }],
    }))
    assert _result_is_valid_for_run(result, run)
