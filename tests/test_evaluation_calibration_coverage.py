from evaluation.calibration_coverage import (
    classify_calibration_targets,
    readiness_summary,
    worker_coverage_summary,
)
from evaluation.experiments.e0_calibration import targets_from_inventory
from fable.distributed.config import (
    ProviderRuntimeResolver,
    load_deployment_graph,
)
from fable.planning.provider_registry import ProviderRegistry


def test_calibration_readiness_matches_exact_runtime_inventory() -> None:
    registry = ProviderRegistry.from_files(
        catalog_path="providers/registry/catalog.yaml",
        data_types_path="providers/registry/data_types.yaml",
    )
    deployment = load_deployment_graph(
        "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    rows = classify_calibration_targets(
        targets_from_inventory(registry, deployment),
        registry=registry,
        deployment=deployment,
        runtimes=ProviderRuntimeResolver.from_yaml(
            "iobt-minimal-ce-replay/config/fable_provider_runtimes.yaml"
        ),
    )
    summary = readiness_summary(rows)
    assert summary["target_count"] == 77
    hosted = [
        item
        for item in rows
        if item.target.provider_id == "hosted_vlm_identity_comparator"
    ]
    assert len(hosted) == 1
    assert hosted[0].status == "HOSTED_BOUNDED"
    assert hosted[0].runtime_node_ids == ("cloud1",)
    assert any(item.status == "MISSING_RUNTIME" for item in rows)
    assert any(item.status == "REFERENCE_ONLY" for item in rows)
    coverage = worker_coverage_summary(
        rows,
        {
            "pairwise_distance_evaluator": {
                "measurement_status": "MEASURED_PROVIDER",
                "input_classes": ("wrong.v1",),
            }
        },
    )
    assert coverage["measured_worker_target_count"] == 0
