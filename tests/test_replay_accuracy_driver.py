from evaluation.live_requests import (
    LiveComplexEventDetection,
    LiveComplexEventProgress,
)
import pytest

from scripts.run_replay_accuracy import (
    _available_seed_sources,
    _normalize_relevant_nodes,
    _recording_time,
    _resolve_experiment,
    _scenario_start_datetime,
    _seed_event_time_interval,
    condition_offset,
    evaluation_request_deadline_offset_ms,
    replay_completion_requirements,
    replay_node_readiness_requirements,
    record_condition_notification,
)
from scripts.run_full_ce_suite import (
    candidate_zed_nodes,
    evaluation_nodes_for_variant,
    missing_netwaggle_anchor,
    select_playable_replay_nodes,
    wait_for_orchestrator,
)
import scripts.run_full_ce_suite as full_suite
from fable.common.examples import BASE_TIME


def test_condition_offset_is_unknown_before_condition_start() -> None:
    assert condition_offset(20.0, anchor=None, trace_started=None) is None


def test_evaluation_semantic_deadline_uses_the_bounded_cell_budget() -> None:
    assert evaluation_request_deadline_offset_ms(525.0) == 525_000
    assert evaluation_request_deadline_offset_ms(0.0011) == 2


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_evaluation_semantic_deadline_rejects_invalid_budget(value: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        evaluation_request_deadline_offset_ms(value)


def test_condition_offset_uses_trace_start_until_anchor_is_available() -> None:
    assert condition_offset(20.0, anchor=None, trace_started=12.5) == 7.5
    assert condition_offset(20.0, anchor=18.0, trace_started=12.5) == 2.0


def test_rejected_resource_notification_does_not_interrupt_condition_schedule() -> None:
    row = {}

    def reject() -> None:
        raise RuntimeError("orchestrator rejected resource change")

    record_condition_notification(row, reject)

    assert row == {
        "notification_validated": False,
        "notification_error": "orchestrator rejected resource change",
    }


def test_global_boundary_waits_for_all_active_modalities() -> None:
    assert replay_completion_requirements(
        {"dvpg_gq_orin_11", "mobile_archive_6"},
        {
            "dvpg_gq_orin_11": {"zed", "respeaker"},
            "mobile_archive_6": {"mobile"},
        },
    ) == {
        ("dvpg_gq_orin_11", "zed"),
        ("dvpg_gq_orin_11", "respeaker"),
        ("mobile_archive_6", "mobile"),
    }


def test_analytics_readiness_is_required_only_on_selected_replay_nodes() -> None:
    required = {"zed", "yolo"}
    selected = {"mobile_archive_2"}
    assert replay_node_readiness_requirements(
        required,
        node_id="mobile_archive_2",
        selected_yolo_nodes=selected,
    ) == {"mobile", "yolo"}
    assert replay_node_readiness_requirements(
        required,
        node_id="mobile_archive_1",
        selected_yolo_nodes=selected,
    ) == {"mobile"}
    assert replay_node_readiness_requirements(
        required,
        node_id="dvpg_gq_orin_4",
        selected_yolo_nodes=selected,
    ) == {"zed"}


def test_labeled_experiment_rejects_deployed_wrong_camera() -> None:
    with pytest.raises(ValueError, match="relevant to orin14"):
        _resolve_experiment("20260413-pass-follow-clear-convoy-c1-test-r003")


def test_labeled_experiment_resolves_variant_and_replay_scenario() -> None:
    experiment, scenario = _resolve_experiment(
        "20260413-pass-follow-clear-convoy-c1-test-r003",
        replay_nodes=("orin14",),
    )

    assert experiment.ce_variant == "Pass-follow-clear convoy"
    assert scenario == "20260413_143336"


def test_labeled_experiment_resolves_with_mobile_only_ready_sources() -> None:
    experiment, scenario = _resolve_experiment(
        "20250812-robbery-with-alarm-burglary-a-r010",
        replay_nodes=("mobile_archive_4", "mobile_archive_5", "mobile_archive_6"),
    )

    assert experiment.ce_variant == "Robbery with alarm"
    assert scenario == "20250812_121227"


def test_seed_sources_exclude_ready_node_without_scenario_audio() -> None:
    sources = _available_seed_sources(
        "20260414_152641",
        "AUDIO_EVENT",
        ("orin11", "orin13", "orin14", "orin15", "orin16"),
    )

    assert sources == (
        "orin11_microphone",
        "orin13_microphone",
        "orin15_microphone",
        "orin16_microphone",
    )


def test_labeled_experiment_rejects_nearby_but_nonoverlapping_replay() -> None:
    with pytest.raises(ValueError, match="no replay scenario matches"):
        _resolve_experiment(
            "20241009-two-vehicle-chase-12-r015",
            replay_nodes=("orin1", "orin4"),
        )


def test_relevant_node_ranges_are_normalized() -> None:
    assert _normalize_relevant_nodes(("11-13", "orin15")) == {
        "orin11",
        "orin12",
        "orin13",
        "orin15",
    }


def test_ranked_replay_candidates_are_filtered_after_readiness_probe() -> None:
    selected, fallback = select_playable_replay_nodes(
        ["orin1", "orin4", "orin7"],
        ["orin_5", "orin-6", "orin_7", "orin1"],
        maximum_nodes=2,
    )

    assert selected == ["orin7", "orin1"]
    assert fallback is False


def test_replay_selection_has_bounded_scenario_fallback() -> None:
    selected, fallback = select_playable_replay_nodes(
        ["orin4", "orin7"], ["orin5", "orin6"], maximum_nodes=1
    )

    assert selected == ["orin4"]
    assert fallback is True


def test_rebased_replay_time_is_mapped_to_recording_time() -> None:
    recording_start = datetime(2026, 4, 15, 19, 46, 6, tzinfo=UTC)
    replay_start = datetime(2026, 7, 27, 4, 53, 0, tzinfo=UTC)

    normalized = _recording_time(
        replay_start + timedelta(seconds=49.5),
        replay_started_at=replay_start.timestamp(),
        recording_started_at=recording_start,
    )

    assert normalized == recording_start + timedelta(seconds=49.5)


def test_replay_event_anchor_uses_catalog_recording_time_not_wall_time() -> None:
    scenario_start = _scenario_start_datetime("20260414_152641")

    assert scenario_start.year == 2026
    assert scenario_start.month == 4


def test_seed_event_interval_excludes_post_event_processing_grace() -> None:
    scenario_start = datetime(2024, 10, 8, 16, 11, 52, tzinfo=UTC)
    labeled_end = datetime(2024, 10, 8, 16, 13, 17, tzinfo=UTC)

    interval = _seed_event_time_interval(
        scenario_start=scenario_start,
        replay_start_seconds=0.0,
        # The replay remains alive for the 30-second result deadline.
        replay_end_seconds=115.0,
        labeled_event_end=labeled_end,
    )

    assert interval.start == scenario_start
    assert interval.end == labeled_end


def test_unlabeled_seed_event_interval_uses_explicit_replay_window() -> None:
    scenario_start = datetime(2024, 10, 8, 16, 11, 52, tzinfo=UTC)

    interval = _seed_event_time_interval(
        scenario_start=scenario_start,
        replay_start_seconds=5.0,
        replay_end_seconds=115.0,
        labeled_event_end=None,
    )

    assert interval.start == scenario_start + timedelta(seconds=5)
    assert interval.end == scenario_start + timedelta(seconds=115)


def test_terminal_progress_carries_typed_complex_event_detection() -> None:
    progress = LiveComplexEventProgress(
        request_id="request",
        transition_status="APPLIED",
        terminal=True,
        terminal_lifecycles={"hypothesis": "COMPLETED"},
        detections=(
            LiveComplexEventDetection(
                hypothesis_id="hypothesis",
                event_family="vehicle_convergence",
                event_start_time=BASE_TIME,
                event_end_time=BASE_TIME,
                emitted_at=BASE_TIME,
                bindings={"vehicle": "vehicle-1"},
            ),
        ),
    )

    restored = LiveComplexEventProgress.model_validate_json(
        progress.model_dump_json()
    )
    assert restored.detections[0].bindings == {"vehicle": "vehicle-1"}
from datetime import UTC, datetime, timedelta
def test_mobile_augmentation_includes_all_visual_2025_event_families() -> None:
    nodes = ["orin1", "mobile_archive_4", "mobile_archive_5"]

    assert evaluation_nodes_for_variant(nodes, "Two-visit stalking") == nodes
    assert evaluation_nodes_for_variant(nodes, "Talking/rendezvous") == nodes
    assert evaluation_nodes_for_variant(nodes, "Vehicle rendezvous") == nodes
    assert evaluation_nodes_for_variant(nodes, "Robbery with alarm") == nodes


def test_verified_2024_mobile_nodes_are_retained_for_every_variant() -> None:
    nodes = ["orin1", "mobile_archive_1", "mobile_archive_2"]

    assert evaluation_nodes_for_variant(
        nodes, "Route convoy", campaign_year=2024
    ) == nodes


def test_candidate_zed_nodes_excludes_audio_only_fast_path_nodes() -> None:
    row = {
        "nodes": ["orin1", "orin2", "orin3", "orin4"],
        "zed_nodes": ["orin2", "orin3"],
        "respeaker_nodes": ["orin1", "orin2", "orin3", "orin4"],
    }

    assert candidate_zed_nodes(row, 2025) == ["orin2", "orin3"]


def test_missing_netwaggle_anchor_only_accepts_anchor_container(tmp_path) -> None:
    log = tmp_path / "stack.log"
    log.write_text(
        "Error response from daemon: No such container: netwaggle-node-orin11\n"
    )
    assert missing_netwaggle_anchor(log) == "netwaggle-node-orin11"

    log.write_text("Error response from daemon: No such container: unrelated\n")
    assert missing_netwaggle_anchor(log) is None


def test_orchestrator_readiness_reads_complete_current_container_log(monkeypatch) -> None:
    commands = []

    class Completed:
        def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[1] == "inspect":
            return Completed(stdout="true\n")
        return Completed(
            stdout=("noisy startup line\n" * 50) + "FABLE orchestrator ready id=orchestrator\n"
        )

    monkeypatch.setattr(full_suite, "run", fake_run)
    monkeypatch.setattr(full_suite.time, "sleep", lambda _seconds: None)

    assert wait_for_orchestrator(timeout_seconds=1)
    log_command = next(command for command in commands if command[1] == "logs")
    assert "--tail" not in log_command
