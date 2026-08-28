import json
from pathlib import Path

import pytest

from evaluation.condition_trace import ConditionTrace
from scripts.run_replay_accuracy import (
    canonical_resource_kind,
    planner_network_condition,
)
from scripts.run_physical_e4_pilot import (
    effective_hard_cell_timeout,
    evaluate_adaptation_timelines,
    experiment_environment,
    prepare_registry,
    replay_source_configuration,
    result_valid,
)
from scripts.physical_condition_control import parse_compute_clear_output
from scripts.physical_sampling import (
    attach_replay_provenance,
    deterministic_sample_numbers,
)


ROOT = Path(__file__).resolve().parents[1]


def test_physical_watchdog_cannot_preempt_inner_scientific_timeout() -> None:
    assert effective_hard_cell_timeout(360, 300) == 450
    assert effective_hard_cell_timeout(600, 300) == 600


def test_rejected_request_is_not_a_resumable_completed_cell(tmp_path) -> None:
    result = tmp_path / "trace.json"
    result.write_text(
        json.dumps(
            {
                "baseline": "B1_STATIC_WHOLE_EVENT",
                "admitted": False,
                "seed_diagnostics": [{"accepted": False}],
                "condition_trace": None,
                "classification": "FALSE_NEGATIVE",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "report.json").write_text(
        json.dumps({"resources_released": True}), encoding="utf-8"
    )
    assert not result_valid(
        result, baseline="B1_STATIC_WHOLE_EVENT", condition_id="nominal"
    )


def test_physical_e4_pilot_is_matched_nine_cell_matrix() -> None:
    spec = json.loads(
        (ROOT / "evaluation/manifests/adaptation/physical_e4_pilot.json")
        .read_text(encoding="utf-8")
    )
    assert spec["playback_mode"] == "realtime"
    assert spec["playback_speed"] == 1.0
    assert spec["repetitions"] == 1
    assert spec["expected_cells"] == 9
    assert len(spec["baselines"]) * len(spec["conditions"]) == 9
    assert spec["baselines"] == [
        "B1_STATIC_WHOLE_EVENT",
        "B3_TASK_RESOURCE_ADAPTIVE",
        "FABLE",
    ]
    assert spec["required_adaptation_pairs"] == [
        {
            "baseline": "B3_TASK_RESOURCE_ADAPTIVE",
            "condition": "compute_contention",
        },
        {"baseline": "FABLE", "condition": "compute_contention"},
    ]


def test_physical_e4_disturbances_overlap_nominal_active_interval() -> None:
    for filename in (
        "physical_e4_compute_contention.json",
        "physical_e4_network_change.json",
    ):
        trace = ConditionTrace.model_validate_json(
            (ROOT / "evaluation/manifests/adaptation" / filename)
            .read_text(encoding="utf-8")
        )
        offsets = [item.offset_s for item in trace.transitions]
        assert offsets == [10.0, 55.0]
        # The validated nominal physical trace terminated near 34 seconds.
        assert offsets[0] < 34 < offsets[1]


def test_physical_network_condition_uses_protocol_resource_kind() -> None:
    assert canonical_resource_kind("NETWORK_PROFILE") == "NETWORK"
    assert canonical_resource_kind("LINK_STATE") == "LINK_STATE"
    assert planner_network_condition(
        "N0", "physical_link:rpi_to_jetson"
    ) == "P1_JETSON_PATH_DEGRADED"


def test_robbery_physical_pilot_declares_synchronized_hybrid_sources() -> None:
    spec = json.loads(
        (ROOT / "evaluation/manifests/adaptation/physical_e2_robbery32_pilot.json")
        .read_text(encoding="utf-8")
    )
    nodes, physical = replay_source_configuration(spec)
    assert nodes == ["orin11", "orin13", "orin14", "orin15", "orin16"]
    assert physical == "orin15"
    assert len(spec["baselines"]) * len(spec["conditions"]) == spec["expected_cells"] == 4


def test_physical_source_mapping_fails_closed_for_multiple_pi_sources() -> None:
    with pytest.raises(ValueError, match="exactly one physical_pi"):
        replay_source_configuration({
            "replay_sources": [
                {"logical_replay_node": "orin13", "execution": "physical_pi"},
                {"logical_replay_node": "orin15", "execution": "physical_pi"},
            ]
        })


def test_physical_source_mapping_accepts_synchronized_mobile_archives() -> None:
    nodes, physical = replay_source_configuration({
        "replay_sources": [
            {"logical_replay_node": "mobile_archive_1", "execution": "desktop"},
            {"logical_replay_node": "orin1", "execution": "physical_pi"},
        ]
    })
    assert nodes == ["mobile_archive_1", "orin1"]
    assert physical == "orin1"


def test_physical_e4_expanded_matrix_has_four_matched_nine_cell_cases() -> None:
    manifest = json.loads(
        (ROOT / "evaluation/manifests/adaptation/physical_e4_expanded.json")
        .read_text(encoding="utf-8")
    )
    specs = [
        json.loads((ROOT / item).read_text(encoding="utf-8"))
        for item in manifest["case_specs"]
    ]
    assert len(specs) == 4
    assert sum(item["expected_cells"] for item in specs) == 36
    assert all(item["expected_cells"] == 9 for item in specs)
    assert all(item["playback_speed"] == 1.0 for item in specs)
    assert manifest["network_semantics"]["not_a_disconnect"] is True


def test_physical_e4_disconnect_is_a_typed_bounded_link_failure() -> None:
    trace = ConditionTrace.model_validate_json(
        (ROOT / "evaluation/manifests/adaptation/physical_e4_network_disconnect.json")
        .read_text(encoding="utf-8")
    )
    assert [item.action.value for item in trace.transitions] == [
        "FAIL_LINK", "RESTORE_LINK"
    ]
    assert all(
        item.target_id == "physical_link:rpi_to_jetson"
        for item in trace.transitions
    )


def test_non_b1_physical_pilot_does_not_derive_b1_calibration(tmp_path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    registry, placement = prepare_registry(
        tmp_path, calibration, require_b1=False
    )
    assert registry.is_file()
    assert placement["applicable"] is False
    assert placement["fanout_allowed"] is False


def test_joint_active_frontier_pilot_is_gated_and_late() -> None:
    spec = json.loads(
        (
            ROOT
            / "evaluation/manifests/adaptation/physical_e2_robbery32_concurrent_pilot.json"
        ).read_text(encoding="utf-8")
    )
    assert spec["baselines"] == ["B4_GREEDY_FRONTIER", "FABLE"]
    assert spec["expected_cells"] == 2
    assert experiment_environment(spec) == {
        "FABLE_JOINT_RESOURCE_EPOCH_PLANNING": "1"
    }
    trace = ConditionTrace.model_validate_json(
        (
            ROOT
            / "evaluation/manifests/adaptation/physical_e2_compute_contention_late.json"
        ).read_text(encoding="utf-8")
    )
    assert [item.offset_s for item in trace.transitions] == [45.0, 75.0]


def test_physical_pilot_environment_rejects_arbitrary_injection() -> None:
    with pytest.raises(ValueError, match="unsupported experiment environment"):
        experiment_environment({"environment": {"PYTHONPATH": "/tmp/injected"}})


def test_jetson_capacity_live_pilot_uses_two_requests_and_four_cells() -> None:
    spec = json.loads(
        (
            ROOT
            / "evaluation/manifests/adaptation/physical_e2_jetson_capacity_live_pilot.json"
        ).read_text(encoding="utf-8")
    )
    assert spec["concurrent_requests"] == 2
    assert spec["allow_raw_to_trusted_site_edge"] is True
    assert spec["baselines"] == ["B4_GREEDY_FRONTIER", "FABLE"]
    assert len(spec["conditions"]) * len(spec["baselines"]) == 4


def test_physical_sampling_is_media_position_deterministic() -> None:
    expected = (1, 9, 16, 24, 31, 39, 46, 54, 61)
    assert deterministic_sample_numbers(
        frame_count=61, source_fps=15.0, maximum_rate_hz=2.0
    ) == expected
    # Wall-clock inference latency is deliberately not an input.
    assert deterministic_sample_numbers(
        frame_count=61, source_fps=15.0, maximum_rate_hz=2.0
    ) == expected


def test_physical_detection_rows_carry_synchronized_replay_provenance() -> None:
    source = [{"class": "car", "conf": 0.9, "frame_number": 17}]
    stamped = attach_replay_provenance(
        source,
        replay_id="trace-physical-generation-7",
        scenario="trace-physical",
    )
    assert stamped == [{
        "class": "car",
        "conf": 0.9,
        "frame_number": 17,
        "replay_id": "trace-physical-generation-7",
        "scenario": "trace-physical",
    }]
    assert "replay_id" not in source[0]


def test_physical_detection_rows_reject_missing_replay_provenance() -> None:
    with pytest.raises(ValueError, match="requires a replay_id"):
        attach_replay_provenance([{"class": "car"}], replay_id=None)


def test_compute_cleanup_requires_work_and_tegrastats_evidence() -> None:
    output = "\n".join((
        "CONTENTION_RESULT",
        '{"active_seconds": 4.2, "iterations": 17, "terminated": true}',
        "TEGRASTATS_TAIL",
        *("RAM 1000/8000MB GR3D_FREQ 82%" for _ in range(8)),
    ))
    parsed = parse_compute_clear_output(output)
    assert parsed["measurement_validated"] is True
    assert parsed["contention_result"]["iterations"] == 17
    assert parse_compute_clear_output(
        "CONTENTION_RESULT\n{\"active_seconds\": 0, \"iterations\": 0}\n"
        "TEGRASTATS_TAIL\n"
    )["measurement_validated"] is False


def test_e4_discrimination_rejects_reissued_identical_plan() -> None:
    events = [
        {
            "event_kind": "PLAN", "event": "PLAN_SELECTED",
            "relative_seconds": 0, "selected_providers": "yolo@orin7",
        },
        {
            "event_kind": "DISTURBANCE", "event": "APPLY",
            "relative_seconds": 10,
        },
        {
            "event_kind": "PLAN", "event": "PLAN_CHANGED",
            "relative_seconds": 11, "selected_providers": "yolo@orin7",
        },
    ]
    result = evaluate_adaptation_timelines(
        {("compute", "FABLE"): events},
        adaptive_baselines=("FABLE",),
        disturbed_conditions=("compute",),
    )
    assert result["valid"] is False
    events[-1]["selected_providers"] = "yolo@x86server"
    result = evaluate_adaptation_timelines(
        {("compute", "FABLE"): events},
        adaptive_baselines=("FABLE",),
        disturbed_conditions=("compute",),
    )
    assert result["valid"] is True


def test_e4_discrimination_can_scope_required_condition_pairs() -> None:
    changed = [
        {
            "event_kind": "PLAN", "event": "PLAN_SELECTED",
            "relative_seconds": 0, "selected_providers": "yolo@orin7",
        },
        {
            "event_kind": "DISTURBANCE", "event": "APPLY",
            "relative_seconds": 10,
        },
        {
            "event_kind": "PLAN", "event": "PLAN_CHANGED",
            "relative_seconds": 11, "selected_providers": "yolo@x86server",
        },
    ]
    result = evaluate_adaptation_timelines(
        {("compute", "FABLE"): changed},
        adaptive_baselines=("FABLE",),
        disturbed_conditions=("compute", "network"),
        required_pairs=(("FABLE", "compute"),),
    )

    assert result["valid"] is True
    assert [(row["baseline"], row["condition"]) for row in result["rows"]] == [
        ("FABLE", "compute")
    ]
