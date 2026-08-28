from datetime import datetime, timezone

from evaluation.e2_snapshots import export_checkpoint_snapshot, load_checkpoint_snapshot
from evaluation.runner import EvaluationRunner
from evaluation.schemas import EvaluationMode
from evaluation.planning_cases import compile_evaluation_planning_case
from evaluation.deployment_artifacts import load_deployment_artifacts
from fable.distributed.config import load_deployment_graph
from fable.planning.provider_registry import ProviderRegistry


def test_checkpoint_snapshot_round_trip(tmp_path):
    providers = ProviderRegistry.from_files(
        catalog_path="providers/registry/catalog.yaml",
        data_types_path="providers/registry/data_types.yaml",
        profiles_path="evaluation/manifests/providers/calibrated_desktop_profiles.json",
    )
    deployment = load_deployment_graph("iobt-minimal-ce-replay/config/fable_deployment.yaml")
    artifacts = load_deployment_artifacts(
        "iobt-minimal-ce-replay/config/fable_deployment_artifacts.yaml",
        repository_root=".",
    )
    case = compile_evaluation_planning_case(
        variant="Three-visit stalking",
        run_id="snapshot-run",
        trace_id="snapshot-trace",
        request_id="snapshot-request",
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
        frontier_index=0,
    )
    target = tmp_path / "checkpoint.json"
    exported = export_checkpoint_snapshot(case, target, source_record_paths=("run/predicate_demand.jsonl",))
    loaded = load_checkpoint_snapshot(target)
    assert exported["provenance"]["synthetic_expansion"] is False
    assert loaded.frontier_demands == case.frontier_demands
    assert loaded.frontier_graph == case.frontier_graph


def test_runner_can_capture_full_typed_checkpoint(tmp_path):
    providers = ProviderRegistry.from_files(
        catalog_path="providers/registry/catalog.yaml",
        data_types_path="providers/registry/data_types.yaml",
        profiles_path="evaluation/manifests/providers/calibrated_desktop_profiles.json",
    )
    deployment = load_deployment_graph("iobt-minimal-ce-replay/config/fable_deployment.yaml")
    artifacts = load_deployment_artifacts(
        "iobt-minimal-ce-replay/config/fable_deployment_artifacts.yaml", repository_root="."
    )
    case = compile_evaluation_planning_case(
        variant="Three-visit stalking", run_id="run", trace_id="trace", request_id="request",
        now=datetime(2026, 1, 1, tzinfo=timezone.utc), provider_registry=providers,
        artifact_catalog=artifacts, deployment=deployment, frontier_index=0,
    )
    class Policy:
        baseline_id = __import__("evaluation.schemas", fromlist=["BaselineId"]).BaselineId.FABLE
        def plan(self, value):
            from evaluation.baselines.policies import FablePolicy
            from fable.planning import BoundedLabelPlanner
            return FablePolicy(BoundedLabelPlanner(provider_registry=providers, artifact_catalog=artifacts, deployment=deployment)).plan(value)
    runner = EvaluationRunner(tmp_path, mode=EvaluationMode.PLANNING_REPLAY, capture_e2_snapshots=True)
    runner.run_planning_case(Policy(), case)
    snapshots = list((tmp_path / "e2_checkpoints").glob("*.json"))
    assert len(snapshots) == 1
    assert load_checkpoint_snapshot(snapshots[0]).frontier_demands == case.frontier_demands
