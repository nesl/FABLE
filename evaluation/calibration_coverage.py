"""Pre-execution coverage classification for E0 calibration targets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import Field

from evaluation.experiments.e0_calibration import CalibrationTarget
from fable.common.base import FrozenFableModel
from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.models import RuntimeMode
from fable.planning.deployment import DeploymentGraph
from fable.planning.provider_registry import ProviderRegistry


class CalibrationReadiness(FrozenFableModel):
    schema_version: str = "fable.calibration_readiness.v1"
    target: CalibrationTarget
    status: str = Field(
        pattern=r"^(READY_CONTAINER|HOSTED_BOUNDED|REFERENCE_ONLY|MISSING_RUNTIME)$"
    )
    candidate_node_ids: tuple[str, ...] = ()
    runtime_node_ids: tuple[str, ...] = ()
    requires_replay_fixture: bool
    hosted_external: bool
    reason: str = Field(min_length=1)


def classify_calibration_targets(
    targets: tuple[CalibrationTarget, ...],
    *,
    registry: ProviderRegistry,
    deployment: DeploymentGraph,
    runtimes: ProviderRuntimeResolver,
) -> tuple[CalibrationReadiness, ...]:
    rows = []
    for target in targets:
        provider = registry.provider(target.provider_id)
        candidates = tuple(
            node.node_id
            for node in deployment.nodes.values()
            if node.available
            and node.node_class == target.tier
            and set(provider.required_node_capabilities).issubset(node.capabilities)
            and (
                not provider.eligible_node_classes
                or node.node_class in provider.eligible_node_classes
            )
        )
        configured = tuple(
            runtimes.resolve(node_id=node_id, provider_id=target.provider_id)
            for node_id in sorted(candidates)
            if runtimes.has(node_id, target.provider_id)
        )
        runtime_nodes = tuple(
            item.node_id
            for item in configured
            if item.mode != RuntimeMode.REFERENCE
        )
        reference_nodes = tuple(
            item.node_id
            for item in configured
            if item.mode == RuntimeMode.REFERENCE
        )
        configured_nodes = tuple(
            node_id
            for node_id in sorted(candidates)
            if runtimes.has(node_id, target.provider_id)
        )
        hosted = provider.evaluation_contract.hosted_external
        replay_fixture = target.input_class != "no_external_input"
        if hosted and runtime_nodes:
            status = "HOSTED_BOUNDED"
            reason = (
                "runtime exists; calibration must use the hosted invocation "
                "budget and secret-isolated proxy"
            )
        elif runtime_nodes:
            status = "READY_CONTAINER"
            reason = (
                "at least one exact provider runtime exists on the requested tier; "
                "a typed replay fixture is still required"
            )
        elif reference_nodes:
            status = "REFERENCE_ONLY"
            reason = (
                "only deterministic reference-delay runtimes exist on this tier; "
                "they cannot produce E0 hardware measurements"
            )
        else:
            status = "MISSING_RUNTIME"
            reason = (
                "no configured runtime implements this provider on any eligible "
                f"{target.tier} node"
            )
        rows.append(
            CalibrationReadiness(
                target=target,
                status=status,
                candidate_node_ids=tuple(sorted(candidates)),
                runtime_node_ids=configured_nodes,
                requires_replay_fixture=replay_fixture,
                hosted_external=hosted,
                reason=reason,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                item.status,
                item.target.provider_id,
                item.target.tier,
                item.target.input_class,
            ),
        )
    )


def readiness_summary(
    rows: tuple[CalibrationReadiness, ...],
) -> dict[str, object]:
    statuses = Counter(item.status for item in rows)
    providers = {
        item.target.provider_id for item in rows
    }
    ready_providers = {
        item.target.provider_id
        for item in rows
        if item.status in {"READY_CONTAINER", "HOSTED_BOUNDED"}
    }
    return {
        "schema_version": "fable.calibration_readiness_summary.v1",
        "target_count": len(rows),
        "provider_count": len(providers),
        "ready_provider_count": len(ready_providers),
        "status_counts": dict(sorted(statuses.items())),
    }


def worker_coverage_summary(
    rows: tuple[CalibrationReadiness, ...],
    operations: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Report exact target signatures, not provider-wide approximations."""

    ready = tuple(item for item in rows if item.status == "READY_CONTAINER")
    measured: list[str] = []
    validation_only: list[str] = []
    missing: list[str] = []
    for item in ready:
        operation = operations.get(item.target.provider_id)
        supported = (
            operation is not None
            and item.target.input_class
            in tuple(operation.get("input_classes", ()))
        )
        label = (
            f"{item.target.provider_id}/{item.target.tier}/"
            f"{item.target.input_class}"
        )
        if not supported:
            missing.append(label)
        elif operation.get("measurement_status") == "MEASURED_PROVIDER":
            measured.append(label)
        else:
            validation_only.append(label)
    return {
        "schema_version": "fable.calibration_worker_coverage.v1",
        "ready_container_target_count": len(ready),
        "measured_worker_target_count": len(measured),
        "validation_only_target_count": len(validation_only),
        "missing_worker_target_count": len(missing),
        "measured_targets": measured,
        "validation_only_targets": validation_only,
        "missing_targets": missing,
    }
