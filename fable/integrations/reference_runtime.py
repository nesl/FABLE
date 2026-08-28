"""Synthetic provider implementation used only by development/plumbing profiles."""

from __future__ import annotations

import json
import time

from fable.common.enums import ArtifactAccessMode, ArtifactLocationKind, TruthValue
from fable.common.ids import uuid7
from fable.common.schemas import (
    ArtifactLocation,
    ArtifactProducer,
    ArtifactRef,
    BindingDelta,
    PredicateResult,
    ResultProvenance,
)
from fable.common.time import utc_now
from fable.distributed.output_adapters import (
    ReferenceExecutionContext,
    ReferenceExecutionOutcome,
)
from fable.distributed.models import ActivateProviderCommand


class SyntheticReferenceRuntime:
    """Deterministic oracle-like runtime for control-plane tests, never real evaluation."""

    def execute(
        self,
        command: ActivateProviderCommand,
        context: ReferenceExecutionContext,
    ) -> ReferenceExecutionOutcome | None:
        started = utc_now()
        if command.runtime.reference_delay_ms:
            time.sleep(command.runtime.reference_delay_ms / 1000.0)
        if not context.is_active(command):
            return None
        completed = utc_now()
        artifacts = self._create_artifacts(command, context)
        semantic_outputs = set(command.demand.acceptable_output_types)
        if not semantic_outputs.intersection(command.plan_step.output_data_types):
            return ReferenceExecutionOutcome(artifacts=artifacts)

        source_id = (
            command.demand.eligible_source_ids[0]
            if command.demand.eligible_source_ids
            else context.node_id
        )
        introduced = {
            role: entity
            for role, entity in command.runtime.reference_bindings.items()
            if role in command.demand.unbound_roles
        }
        validated = {
            role: entity
            for role, entity in command.demand.bound_roles.items()
            if command.runtime.reference_bindings.get(role) == entity
        }
        result = PredicateResult(
            occurrence_id=(
                command.occurrence_id_hint or f"reference:{command.demand.demand_id}"
            ),
            demand_id=command.demand.demand_id,
            request_id=command.demand.request_id,
            graph_hash=command.demand.graph_hash,
            hypothesis_id=command.demand.hypothesis_id,
            expected_hypothesis_version=command.issued_hypothesis_version,
            frontier_id=command.demand.frontier_id,
            checkpoint_id=command.demand.checkpoint_id,
            graph_node_id=command.demand.graph_node_id,
            semantic_predicate=command.demand.semantic_predicate,
            truth=TruthValue.TRUE if command.runtime.reference_truth else TruthValue.FALSE,
            confidence=1.0,
            event_time_interval=command.demand.event_time_interval,
            binding_delta=BindingDelta(introduced=introduced, validated=validated),
            artifact_ids=tuple(item.artifact_id for item in artifacts),
            provenance=ResultProvenance(
                provider_id=command.runtime.provider_id,
                provider_contract_version=command.runtime.provider_contract_version,
                node_id=context.node_id,
                source_ids=(source_id,),
            ),
            processing_started_at=started,
            processing_completed_at=completed,
        )
        return ReferenceExecutionOutcome(artifacts=artifacts, result=result)

    @staticmethod
    def _create_artifacts(
        command: ActivateProviderCommand,
        context: ReferenceExecutionContext,
    ) -> tuple[ArtifactRef, ...]:
        artifact_types = command.runtime.reference_artifact_types or tuple(
            data_type
            for data_type in command.plan_step.output_data_types
            if data_type not in {"predicate_match.v1", "predicate_result.v1"}
        )
        artifacts: list[ArtifactRef] = []
        for artifact_type in artifact_types:
            artifact_id = uuid7()
            path = context.artifact_dir / f"{artifact_id}.json"
            path.write_text(
                json.dumps(
                    {
                        "provider_instance_id": command.provider_instance_id,
                        "demand_id": str(command.demand.demand_id),
                        "artifact_type": artifact_type,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            retention_until = max(
                command.demand.deadline.latest_useful_completion,
                command.demand.event_time_interval.end,
            )
            artifacts.append(
                ArtifactRef(
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    artifact_schema_version=artifact_type,
                    producer=ArtifactProducer(
                        provider_id=command.runtime.provider_id,
                        provider_contract_version=command.runtime.provider_contract_version,
                    ),
                    event_time_interval=command.demand.event_time_interval,
                    bindings={
                        **command.demand.bound_roles,
                        **command.runtime.reference_bindings,
                    },
                    location=ArtifactLocation(
                        kind=ArtifactLocationKind.LOCAL_PATH,
                        node_id=context.node_id,
                        uri=str(path),
                    ),
                    access_modes=(
                        ArtifactAccessMode.LOCAL,
                        ArtifactAccessMode.REMOTE_REFERENCE,
                    ),
                    compatible_consumer_families=tuple(
                        family
                        for requirement in command.demand.continuation_requirements
                        if requirement.artifact_type == artifact_type
                        for family in requirement.compatible_consumer_families
                    ),
                    bytes=path.stat().st_size,
                    valid_until=retention_until,
                    expires_at=retention_until,
                )
            )
        return tuple(artifacts)
