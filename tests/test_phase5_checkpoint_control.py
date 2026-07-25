from __future__ import annotations

from datetime import timedelta

from fable.common.enums import (
    ArtifactAccessMode,
    ArtifactLocationKind,
    CheckpointKind,
    ResultKind,
    TruthValue,
)
from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.common.schemas import (
    ArtifactLocation,
    ArtifactProducer,
    ArtifactRef,
    BindingDelta,
    ContinuationRequirement,
    FrontierSnapshot,
    PredicateDemand,
    PredicateResult,
    ResultProvenance,
    SemanticCheckpoint,
)
from fable.common.time import EventTimeInterval
from fable.planning.artifact_catalog import ArtifactCatalog
from fable.planning.testing import fake_deployment, fake_provider_registry
from fable.semantic.models import ApplyStatus, CancellationSet, DerivedFrontier, RuntimeTransition
from fable.scheduling import (
    CapacityLedger,
    CheckpointController,
    MultiTenantScheduler,
    ProviderLifecycleManager,
    TaskSchedulingPolicy,
)
from fable.scheduling.testing import fake_audio_candidate, fake_audio_demand


def _next_demand_with_continuation(base: PredicateDemand, required_until) -> PredicateDemand:
    payload = base.model_dump(mode="python")
    payload.update(
        {
            "demand_id": uuid7(),
            "frontier_id": uuid7(),
            "checkpoint_id": uuid7(),
            "graph_node_id": "departure",
            "continuation_requirements": (
                ContinuationRequirement(
                    artifact_type="audio_event_set.v1",
                    required_until=required_until,
                    required_bindings=("zone",),
                ),
            ),
            "sharing_key": None,
        }
    )
    return PredicateDemand.model_validate(payload)


def test_checkpoint_resolution_completes_winner_cancels_loser_extends_ttl_and_replans() -> None:
    registry = fake_provider_registry()
    lifecycle = ProviderLifecycleManager(
        provider_registry=registry,
        capacity=CapacityLedger(fake_deployment()),
    )
    scheduler = MultiTenantScheduler(lifecycle=lifecycle)
    hypothesis_id = uuid7()
    winner = fake_audio_demand(
        request_id="robbery_task",
        hypothesis_id=hypothesis_id,
        graph_node_id="gunshot",
    )
    loser = fake_audio_demand(
        request_id="robbery_task",
        hypothesis_id=hypothesis_id,
        graph_node_id="threat",
    )
    policy = TaskSchedulingPolicy(request_id="robbery_task")
    scheduler.admit(
        (
            fake_audio_candidate(winner, provider_registry=registry, task_policy=policy),
            fake_audio_candidate(loser, provider_registry=registry, task_policy=policy),
        ),
        now=BASE_TIME,
    )

    artifact = ArtifactRef(
        artifact_type="audio_event_set.v1",
        artifact_schema_version="audio_event_set.v1",
        producer=ArtifactProducer(
            provider_id="audio_event_classifier",
            provider_contract_version=1,
        ),
        event_time_interval=winner.event_time_interval,
        bindings={"zone": "store"},
        location=ArtifactLocation(
            kind=ArtifactLocationKind.LOCAL_PATH,
            node_id="sensor_a",
            uri="file:///tmp/audio-events.json",
        ),
        access_modes=(ArtifactAccessMode.LOCAL,),
        bytes=128,
        created_at=BASE_TIME,
        valid_until=BASE_TIME + timedelta(minutes=5),
        expires_at=BASE_TIME + timedelta(seconds=2),
    )
    artifacts = ArtifactCatalog((artifact,))
    required_until = BASE_TIME + timedelta(minutes=2)
    next_demand = _next_demand_with_continuation(winner, required_until)

    next_checkpoint = SemanticCheckpoint(
        hypothesis_id=hypothesis_id,
        hypothesis_version=2,
        kind=CheckpointKind.PRIMITIVE,
        node_ids=("departure",),
        event_time_interval=winner.event_time_interval,
    )
    frontier = DerivedFrontier(
        snapshot=FrontierSnapshot(
            request_id="robbery_task",
            graph_hash=winner.graph_hash,
            hypothesis_id=hypothesis_id,
            hypothesis_version=2,
            enabled_node_ids=("departure",),
            checkpoint_ids=(next_checkpoint.checkpoint_id,),
            derived_at=BASE_TIME + timedelta(seconds=1),
        ),
        checkpoints=(next_checkpoint,),
    )
    transition = RuntimeTransition(
        status=ApplyStatus.APPLIED,
        hypothesis_ids=(hypothesis_id,),
        frontiers=(frontier,),
        cancellation=CancellationSet(
            node_ids=("threat",),
            branch_ids=("threat",),
            reason="gunshot branch resolved the authored OR",
        ),
    )
    result = PredicateResult(
        demand_id=winner.demand_id,
        request_id=winner.request_id,
        graph_hash=winner.graph_hash,
        hypothesis_id=hypothesis_id,
        expected_hypothesis_version=winner.hypothesis_version,
        frontier_id=winner.frontier_id,
        checkpoint_id=winner.checkpoint_id,
        graph_node_id=winner.graph_node_id,
        semantic_predicate=winner.semantic_predicate,
        occurrence_id="occ_gunshot",
        truth=TruthValue.TRUE,
        event_time_interval=winner.event_time_interval,
        binding_delta=BindingDelta(validated={"zone": "store"}),
        artifact_ids=(artifact.artifact_id,),
        provenance=ResultProvenance(
            provider_id="audio_event_classifier",
            provider_contract_version=1,
            node_id="sensor_a",
            source_ids=("microphone_store",),
        ),
        processing_started_at=BASE_TIME,
        processing_completed_at=BASE_TIME + timedelta(milliseconds=30),
    )

    controller = CheckpointController(
        lifecycle=lifecycle,
        artifact_catalog=artifacts,
    )
    outcome = controller.handle_predicate_result(
        result=result,
        transition=transition,
        request_id="robbery_task",
        hypothesis_id=hypothesis_id,
        next_demands=(next_demand,),
        continuation_artifact_ids=(artifact.artifact_id,),
        now=BASE_TIME + timedelta(seconds=1),
    )

    assert outcome.completed_lease_ids
    assert outcome.cancellation is not None
    assert outcome.cancellation.cancelled_demand_ids == (loser.demand_id,)
    assert outcome.retention_updates[0].new_expires_at == required_until
    assert artifacts.get(artifact.artifact_id).expires_at == required_until
    assert outcome.replan_requests[0].frontier_id == frontier.snapshot.frontier_id
