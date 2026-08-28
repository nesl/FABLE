import json
from pathlib import Path

from evaluation.deployment_artifacts import load_deployment_artifacts
from evaluation.planning_cases import compile_evaluation_planning_case
from evaluation.provider_coverage import validate_live_provider_coverage
from evaluation.replay_manifest import ReplayScenario
from fable.common.examples import BASE_TIME
from fable.distributed.config import ProviderRuntimeResolver, load_deployment_graph
from fable.planning.testing import fake_provider_registry


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "iobt-minimal-ce-replay"


def test_sequential_convoy_builds_its_live_pass_realization():
    artifacts = load_deployment_artifacts(
        REPLAY / "config/fable_deployment_artifacts.yaml",
        repository_root=ROOT,
    )
    deployment = load_deployment_graph(REPLAY / "config/fable_deployment.yaml")
    runtimes = ProviderRuntimeResolver.from_yaml(
        REPLAY / "config/fable_provider_runtimes.yaml"
    )
    providers = fake_provider_registry()
    case = compile_evaluation_planning_case(
        variant="Pass-follow-clear convoy",
        run_id="run",
        trace_id="trace",
        request_id="request",
        now=BASE_TIME,
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
        runtime_resolver=runtimes,
    )
    assert "passes_live_vehicle" in {
        item.chain_id for item in case.whole_event_graph.alternatives
    }
    assert not validate_live_provider_coverage(
        case.whole_event_graph,
        registry=providers,
        runtimes=runtimes,
    )


def test_generated_scenario_catalog_rows_ignore_documented_extra_fields():
    document = json.loads(
        (REPLAY / "generated/scenario_catalog.json").read_text(encoding="utf-8")
    )
    scenario = ReplayScenario.from_catalog_row(document["scenarios"][0])
    assert scenario.scenario_id
    assert scenario.nodes
