from __future__ import annotations

from evaluation.experiments.e0_calibration import (
    build,
    CalibrationObservation,
    summarize_observations,
    targets_from_inventory,
    write_manifest,
    write_profiles,
)
from evaluation.experiments.e6_spatial import (
    coordination_tracker_for_run,
    coordination_tracker_from_manifest,
)
from evaluation.experiments.matrix import PlannedRun, write_planned_runs
from evaluation.experiments.e8_scaling import build as build_scaling
from evaluation.experiments.specs import ExperimentQuestion
from evaluation.metrics.statistics import (
    LoadSample,
    confidence_interval,
    maximum_sustainable_load,
    paired_comparison,
)
from evaluation.report import generate_scaling_report
from evaluation.schemas import BaselineId, EvaluationMode
from fable.distributed.config import load_deployment_graph
from fable.planning.provider_registry import ProviderRegistry


def test_e0_inventory_covers_feasible_provider_tiers(tmp_path) -> None:
    registry = ProviderRegistry.from_files(
        catalog_path="providers/registry/catalog.yaml",
        data_types_path="providers/registry/data_types.yaml",
    )
    deployment = load_deployment_graph(
        "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    targets = targets_from_inventory(registry, deployment)
    assert targets
    hosted = [
        item
        for item in targets
        if item.provider_id == "hosted_vlm_identity_comparator"
    ]
    assert hosted
    assert {item.tier for item in hosted} == {"server"}
    runs = build(
        targets,
        warm_repetitions=1,
        cold_repetitions=1,
    )
    output = write_manifest(runs, tmp_path / "e0.json")
    assert output.is_file()
    assert len(runs) == 2 * len(targets)


def test_e6_tracker_is_built_from_immutable_run_metadata() -> None:
    run = PlannedRun(
        run_id="run",
        question=ExperimentQuestion.RQ3_SPATIAL_COORDINATION,
        experiment_id="experiment",
        baseline_id=BaselineId.SPATIAL_FABLE,
        mode=EvaluationMode.FULL_STACK,
        spatial_metrics_enabled=True,
        campaign_year=2025,
        replay_supported_sensor_ids=("orin_1", "orin_5"),
        unavailable_mobile_sensor_ids=("mobile_4",),
        topology_deployment_ids=("deployment-a",),
    )
    tracker = coordination_tracker_for_run(
        run,
        upstream_predicate_ids=("PASSES",),
        downstream_predicate_ids=("ENTERS",),
    )
    assert tracker.campaign_year == 2025
    assert tracker.replay_supported_sensor_ids == ("orin_1", "orin_5")
    assert tracker.topology_confidence == "measured"


def test_e6_tracker_loads_exact_run_from_manifest(tmp_path) -> None:
    run = PlannedRun(
        run_id="spatial-run",
        question=ExperimentQuestion.RQ3_SPATIAL_COORDINATION,
        experiment_id="experiment",
        baseline_id=BaselineId.SPATIAL_FABLE,
        mode=EvaluationMode.FULL_STACK,
        spatial_metrics_enabled=True,
        campaign_year=2025,
        replay_supported_sensor_ids=("orin_1",),
        topology_deployment_ids=("deployment-a",),
    )
    manifest = write_planned_runs((run,), tmp_path / "runs.jsonl")
    tracker = coordination_tracker_from_manifest(
        manifest,
        "spatial-run",
        upstream_predicate_ids=("PASSES",),
        downstream_predicate_ids=("ENTERS",),
    )
    assert tracker.replay_supported_sensor_ids == ("orin_1",)


def test_e8_runs_predeclare_service_level_objective() -> None:
    runs = build_scaling(
        repetitions=1,
        network_profiles=("good_network",),
        target_relative_timely_recall=0.97,
        maximum_p95_control_latency_ms=175,
    )
    assert runs
    assert {item.target_relative_timely_recall for item in runs} == {0.97}
    assert {item.maximum_p95_control_latency_ms for item in runs} == {175.0}
    assert {item.baseline_id for item in runs} == {
        BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        BaselineId.FABLE,
        BaselineId.FABLE_NO_SHARING,
    }


def test_statistics_and_sustainable_load_are_deterministic(tmp_path) -> None:
    interval = confidence_interval((1.0, 2.0, 3.0, 4.0))
    assert interval.count == 4
    assert interval.lower_95 < interval.mean < interval.upper_95
    comparison = paired_comparison(
        {"a": 0.9, "b": 0.8},
        {"a": 0.7, "b": 0.75},
    )
    assert comparison.pair_count == 2
    assert comparison.difference.mean > 0
    load = maximum_sustainable_load(
        (
            LoadSample(workload=5, timely_recall=0.95, p95_latency_ms=50),
            LoadSample(workload=10, timely_recall=0.91, p95_latency_ms=80),
            LoadSample(workload=15, timely_recall=0.7, p95_latency_ms=120),
        ),
        target_timely_recall=0.9,
        maximum_p95_latency_ms=100,
    )
    assert load.maximum_sustainable_workload == 10
    assert load.rejected_workloads == (15.0,)
    report = generate_scaling_report(
        (
            LoadSample(workload=5, timely_recall=0.95, p95_latency_ms=50),
            LoadSample(workload=10, timely_recall=0.91, p95_latency_ms=80),
            LoadSample(workload=15, timely_recall=0.7, p95_latency_ms=120),
        ),
        tmp_path,
        target_timely_recall=0.9,
        maximum_p95_latency_ms=100,
    )
    assert report["maximum_sustainable_workload"] == 10


def test_e0_observations_reduce_to_tier_profile(tmp_path) -> None:
    target = targets_from_inventory(
        ProviderRegistry.from_files(
            catalog_path="providers/registry/catalog.yaml",
            data_types_path="providers/registry/data_types.yaml",
        ),
        load_deployment_graph(
            "iobt-minimal-ce-replay/config/fable_deployment.yaml"
        ),
    )[0]
    observations = (
        CalibrationObservation(
            run_id="cold",
            target=target,
            invocation_kind="cold",
            startup_ms=100,
            execution_ms=20,
            quality_score=0.8,
            ambiguity_score=0.2,
        ),
        CalibrationObservation(
            run_id="warm-1",
            target=target,
            invocation_kind="warm",
            startup_ms=0,
            execution_ms=10,
            quality_score=0.9,
            ambiguity_score=0.1,
        ),
        CalibrationObservation(
            run_id="warm-2",
            target=target,
            invocation_kind="warm",
            startup_ms=0,
            execution_ms=12,
            quality_score=1.0,
            ambiguity_score=0.05,
        ),
    )
    (profile,) = summarize_observations(observations)
    assert profile.cold_startup_p95_ms == 100
    assert profile.warm_execution_p95_ms == 12
    assert profile.sample_count == 3
    output = write_profiles((profile,), tmp_path / "profiles.json")
    assert output.is_file()
