"""Deterministic fake data and authored graph examples for tests and demos."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .enums import (
    ArtifactAccessMode,
    ArtifactLocationKind,
    CheckpointKind,
    GraphEdgeKind,
    GraphNodeKind,
    ResultKind,
    TemporalGuardKind,
)
from .graph import (
    GraphEdgeDraft,
    GraphNodeDraft,
    SemanticGraphDraft,
    TemporalGuardDraft,
    finalize_semantic_graph,
)
from .ids import occurrence_anchor_id
from .schemas import (
    ArtifactLocation,
    ArtifactProducer,
    ArtifactRef,
    EntityBinding,
    FrontierSnapshot,
    Hypothesis,
    PredicateDemand,
    PredicateRole,
    RoleDefinition,
    SemanticCheckpoint,
    SemanticPredicate,
)
from .time import DeadlineSpec, EventTimeInterval, LatenessPolicy

UTC = timezone.utc
BASE_TIME = datetime(2026, 4, 14, 18, 19, 51, tzinfo=UTC)


def predicate(
    predicate_id: str,
    role_specs: tuple[tuple[str, str, str], ...],
    *,
    result_kind: ResultKind = ResultKind.INSTANT_MATCH,
    parameters: dict | None = None,
) -> SemanticPredicate:
    return SemanticPredicate(
        predicate_id=predicate_id,
        roles=tuple(
            PredicateRole(role_name=name, variable=variable, entity_type=entity_type)
            for name, variable, entity_type in role_specs
        ),
        parameters=parameters or {},
        result_kind=result_kind,
    )


def edge(
    source: str,
    target: str,
    kind: GraphEdgeKind,
    guards: tuple[str, ...] = (),
    branch_label: str | None = None,
) -> GraphEdgeDraft:
    return GraphEdgeDraft(
        source_node_key=source,
        target_node_key=target,
        kind=kind,
        temporal_guard_keys=guards,
        branch_label=branch_label,
    )


def convoy_graph_draft() -> SemanticGraphDraft:
    return SemanticGraphDraft(
        namespace="fable.examples.convoy",
        name="Pass-follow-clear convoy",
        description="Leader passes, follower follows, then the scene becomes clear.",
        root_node_key="convoy_sequence",
        roles=(
            RoleDefinition(role_name="leader", entity_type="vehicle"),
            RoleDefinition(
                role_name="follower",
                entity_type="vehicle",
                distinct_from=("leader",),
            ),
        ),
        nodes=(
            GraphNodeDraft(
                key="convoy_sequence",
                kind=GraphNodeKind.AND,
                name="Convoy progression",
            ),
            GraphNodeDraft(
                key="leader_passes",
                kind=GraphNodeKind.PREDICATE,
                name="Leader passes mobile node",
                predicate=predicate(
                    "PASSES",
                    (
                        ("vehicle", "leader", "vehicle"),
                        ("reference", "mobile_node", "location"),
                    ),
                ),
                checkpoint_boundary=True,
            ),
            GraphNodeDraft(
                key="follower_follows",
                kind=GraphNodeKind.PREDICATE,
                name="Follower follows leader",
                predicate=predicate(
                    "FOLLOWS",
                    (
                        ("leader", "leader", "vehicle"),
                        ("follower", "follower", "vehicle"),
                    ),
                    result_kind=ResultKind.INTERVAL_MATCH,
                    parameters={"max_gap_m": 15.0, "min_duration_ms": 3000},
                ),
                checkpoint_boundary=True,
            ),
            GraphNodeDraft(
                key="scene_clear",
                kind=GraphNodeKind.ABSENT,
                name="No moving vehicles remain",
                checkpoint_boundary=True,
            ),
            GraphNodeDraft(
                key="moving_vehicle",
                kind=GraphNodeKind.PREDICATE,
                name="Vehicle is moving",
                predicate=predicate(
                    "MOVING",
                    (("vehicle", "candidate", "vehicle"),),
                    result_kind=ResultKind.STATE_OBSERVATION,
                ),
            ),
        ),
        edges=(
            edge("convoy_sequence", "leader_passes", GraphEdgeKind.CHILD),
            edge("convoy_sequence", "follower_follows", GraphEdgeKind.CHILD),
            edge("convoy_sequence", "scene_clear", GraphEdgeKind.CHILD),
            edge(
                "leader_passes",
                "follower_follows",
                GraphEdgeKind.SEQUENCE,
                ("follow_window",),
            ),
            edge("follower_follows", "scene_clear", GraphEdgeKind.SEQUENCE),
            edge(
                "scene_clear",
                "moving_vehicle",
                GraphEdgeKind.CHILD,
                ("clear_window",),
            ),
        ),
        temporal_guards=(
            TemporalGuardDraft(
                key="follow_window",
                kind=TemporalGuardKind.WITHIN,
                source_node_keys=("leader_passes",),
                target_node_key="follower_follows",
                maximum_ms=15000,
            ),
            TemporalGuardDraft(
                key="clear_window",
                kind=TemporalGuardKind.ABSENCE_WINDOW,
                source_node_keys=("moving_vehicle",),
                target_node_key="scene_clear",
                minimum_ms=30000,
                required_source_ids=("camera_mobile",),
            ),
        ),
        authored_variant_ids=("pass_follow_clear",),
    )


def robbery_graph_draft() -> SemanticGraphDraft:
    return SemanticGraphDraft(
        namespace="fable.examples.robbery",
        name="Authored robbery family",
        description="Common entry, authored evidence alternatives, and departure.",
        root_node_key="robbery_sequence",
        roles=(
            RoleDefinition(role_name="person", entity_type="person"),
            RoleDefinition(role_name="location", entity_type="zone"),
            RoleDefinition(
                role_name="vehicle",
                entity_type="vehicle",
                cardinality_min=0,
                cardinality_max=1,
            ),
        ),
        nodes=(
            GraphNodeDraft(
                key="robbery_sequence",
                kind=GraphNodeKind.AND,
                name="Robbery progression",
            ),
            GraphNodeDraft(
                key="entry",
                kind=GraphNodeKind.PREDICATE,
                name="Suspicious entry",
                predicate=predicate(
                    "SUSPICIOUS_ENTRY",
                    (
                        ("person", "person", "person"),
                        ("location", "location", "zone"),
                    ),
                ),
                checkpoint_boundary=True,
            ),
            GraphNodeDraft(
                key="evidence_or",
                kind=GraphNodeKind.OR,
                name="Robbery evidence alternatives",
                checkpoint_boundary=True,
            ),
            GraphNodeDraft(
                key="gunshot",
                kind=GraphNodeKind.PREDICATE,
                name="Gunshot or weapon evidence",
                predicate=predicate(
                    "AUDIO_EVENT",
                    (("location", "location", "zone"),),
                    parameters={"label": "gunshot"},
                ),
            ),
            GraphNodeDraft(
                key="threat",
                kind=GraphNodeKind.PREDICATE,
                name="Shouting or threat evidence",
                predicate=predicate(
                    "THREAT_EVENT",
                    (
                        ("person", "person", "person"),
                        ("location", "location", "zone"),
                    ),
                    result_kind=ResultKind.INTERVAL_MATCH,
                ),
            ),
            GraphNodeDraft(
                key="forced_transfer",
                kind=GraphNodeKind.PREDICATE,
                name="Forced transfer",
                predicate=predicate(
                    "FORCED_TRANSFER",
                    (
                        ("person", "person", "person"),
                        ("location", "location", "zone"),
                    ),
                    result_kind=ResultKind.INTERVAL_MATCH,
                ),
            ),
            GraphNodeDraft(
                key="failed_attempt",
                kind=GraphNodeKind.PREDICATE,
                name="Failed attempt and rapid exit",
                predicate=predicate(
                    "FAILED_ATTEMPT_RAPID_EXIT",
                    (
                        ("person", "person", "person"),
                        ("location", "location", "zone"),
                    ),
                ),
            ),
            GraphNodeDraft(
                key="departure",
                kind=GraphNodeKind.PREDICATE,
                name="Departure or escape",
                predicate=predicate(
                    "DEPARTURE_OR_ESCAPE",
                    (
                        ("person", "person", "person"),
                        ("vehicle", "vehicle", "vehicle"),
                    ),
                ),
                checkpoint_boundary=True,
            ),
        ),
        edges=(
            edge("robbery_sequence", "entry", GraphEdgeKind.CHILD),
            edge("robbery_sequence", "evidence_or", GraphEdgeKind.CHILD),
            edge("robbery_sequence", "departure", GraphEdgeKind.CHILD),
            edge("entry", "evidence_or", GraphEdgeKind.SEQUENCE),
            edge(
                "evidence_or",
                "gunshot",
                GraphEdgeKind.ALTERNATIVE,
                branch_label="gunshot",
            ),
            edge(
                "evidence_or",
                "threat",
                GraphEdgeKind.ALTERNATIVE,
                branch_label="threat",
            ),
            edge(
                "evidence_or",
                "forced_transfer",
                GraphEdgeKind.ALTERNATIVE,
                branch_label="forced_transfer",
            ),
            edge(
                "evidence_or",
                "failed_attempt",
                GraphEdgeKind.ALTERNATIVE,
                branch_label="failed_attempt",
            ),
            edge(
                "evidence_or",
                "departure",
                GraphEdgeKind.SEQUENCE,
                ("robbery_window",),
            ),
        ),
        temporal_guards=(
            TemporalGuardDraft(
                key="robbery_window",
                kind=TemporalGuardKind.WITHIN,
                source_node_keys=("entry", "evidence_or"),
                target_node_key="departure",
                maximum_ms=180000,
            ),
        ),
        authored_variant_ids=(
            "gunshot",
            "threat",
            "forced_transfer",
            "failed_attempt",
        ),
    )


def convoy_graph():
    return finalize_semantic_graph(convoy_graph_draft())


def robbery_graph():
    return finalize_semantic_graph(robbery_graph_draft())


def fake_convoy_runtime_records():
    graph = convoy_graph()
    node_by_key = {node.authored_key: node for node in graph.nodes}
    event_interval = EventTimeInterval(
        start=BASE_TIME,
        end=BASE_TIME + timedelta(seconds=3),
    )
    anchor = occurrence_anchor_id(
        "camera_mobile",
        "PASSES",
        BASE_TIME,
        {"leader": "vehicle_17"},
    )
    hypothesis = Hypothesis(
        request_id="request_convoy_demo",
        graph_id=graph.graph_id,
        graph_hash=graph.graph_hash,
        graph_version=graph.graph_version,
        anchor_occurrence_id=anchor,
        role_bindings={
            "leader": EntityBinding(
                role_name="leader",
                entity_type="vehicle",
                canonical_entity_id="vehicle_17",
                local_entity_ids={"camera_mobile": ("track_17",)},
                established_by_occurrence_id=anchor,
            )
        },
        event_time_window=EventTimeInterval(
            start=BASE_TIME,
            end=BASE_TIME + timedelta(seconds=45),
        ),
        deadline=DeadlineSpec(
            latest_start=BASE_TIME + timedelta(seconds=10),
            latest_useful_completion=BASE_TIME + timedelta(seconds=20),
        ),
    )
    checkpoint = SemanticCheckpoint(
        hypothesis_id=hypothesis.hypothesis_id,
        hypothesis_version=hypothesis.version,
        kind=CheckpointKind.PRIMITIVE,
        node_ids=(node_by_key["follower_follows"].node_id,),
        event_time_interval=EventTimeInterval(
            start=BASE_TIME,
            end=BASE_TIME + timedelta(seconds=15),
        ),
        success_activates_node_ids=(node_by_key["scene_clear"].node_id,),
        required_artifact_types_after_resolution=("pair_trajectory.v1",),
    )
    frontier = FrontierSnapshot(
        request_id=hypothesis.request_id,
        graph_hash=graph.graph_hash,
        hypothesis_id=hypothesis.hypothesis_id,
        hypothesis_version=hypothesis.version,
        enabled_node_ids=(node_by_key["follower_follows"].node_id,),
        checkpoint_ids=(checkpoint.checkpoint_id,),
    )
    hypothesis.frontier_id = frontier.frontier_id
    demand = PredicateDemand(
        request_id=hypothesis.request_id,
        graph_hash=graph.graph_hash,
        hypothesis_id=hypothesis.hypothesis_id,
        hypothesis_version=hypothesis.version,
        frontier_id=frontier.frontier_id,
        checkpoint_id=checkpoint.checkpoint_id,
        graph_node_id=node_by_key["follower_follows"].node_id,
        semantic_predicate=node_by_key["follower_follows"].predicate,
        bound_roles={"leader": "vehicle_17"},
        unbound_roles=("follower",),
        event_time_interval=EventTimeInterval(
            start=BASE_TIME,
            end=BASE_TIME + timedelta(seconds=15),
        ),
        deadline=DeadlineSpec(
            latest_start=BASE_TIME + timedelta(seconds=8),
            latest_useful_completion=BASE_TIME + timedelta(seconds=15),
        ),
        lateness_policy=LatenessPolicy(allowed_lateness_ms=1000),
        eligible_source_ids=("camera_mobile", "camera_downstream"),
        required_capabilities=("vehicle_identity", "relative_motion"),
        acceptable_output_types=("predicate_match.v1",),
    )
    artifact = ArtifactRef(
        artifact_type="vehicle_reid_embedding_set.v1",
        artifact_schema_version="vehicle_reid_embedding_set.v1",
        producer=ArtifactProducer(
            provider_id="vehicle_reid_descriptor",
            provider_contract_version=1,
            model_id="vehicle_reid_reference",
            model_version="1.0",
        ),
        event_time_interval=event_interval,
        bindings={"leader": "vehicle_17"},
        location=ArtifactLocation(
            kind=ArtifactLocationKind.LOCAL_PATH,
            node_id="edge_1",
            uri="file:///var/lib/fable/artifacts/leader_17.npy",
        ),
        access_modes=(
            ArtifactAccessMode.LOCAL,
            ArtifactAccessMode.REMOTE_REFERENCE,
        ),
        compatibility_keys={
            "model_id": "vehicle_reid_reference",
            "model_version": "1.0",
            "dimension": 512,
        },
        compatible_consumer_families=("cross_sensor_identity",),
        bytes=2048,
        checksum_sha256="0" * 64,
        created_at=BASE_TIME,
        valid_until=BASE_TIME + timedelta(minutes=5),
        expires_at=BASE_TIME + timedelta(hours=2),
    )
    return graph, hypothesis, frontier, checkpoint, demand, artifact
