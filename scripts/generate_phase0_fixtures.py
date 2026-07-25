#!/usr/bin/env python3
"""Generate example Phase-0 JSON fixtures.

The graph fixtures are fully deterministic. Runtime record identifiers are
valid UUIDv7 values and are generated once when this script is run.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fable.common.enums import (  # noqa: E402
    ArtifactAccessMode,
    NodeAvailability,
    PlanStatus,
    ProviderLeaseStatus,
    TruthValue,
)
from fable.common.examples import BASE_TIME, fake_convoy_runtime_records, robbery_graph  # noqa: E402
from fable.common.ids import occurrence_anchor_id  # noqa: E402
from fable.common.provider_catalog import load_provider_contracts  # noqa: E402
from fable.common.schemas import (  # noqa: E402
    BindingDelta,
    ExecutionPlan,
    NodeCapacity,
    NodeHeartbeat,
    PhysicalPlanLabel,
    PlanCost,
    PlanStep,
    PredicateResult,
    ProviderFamily,
    ProviderLease,
    ResourceReservation,
    ResultProvenance,
    SourceHeartbeat,
)
from fable.common.serialization import write_fixture  # noqa: E402
from fable.common.time import EventTimeInterval  # noqa: E402


def main() -> int:
    fixtures = PROJECT_ROOT / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    graph, hypothesis, frontier, checkpoint, demand, artifact = fake_convoy_runtime_records()
    provider_contracts = load_provider_contracts(
        PROJECT_ROOT / "providers" / "registry" / "catalog.yaml"
    )
    provider_contract = provider_contracts["follows_cross_sensor"]
    family = ProviderFamily(
        family_id="follows",
        description="Physical realizations for the FOLLOWS semantic predicate.",
        predicate_ids=("FOLLOWS",),
        provider_contract_ids=("follows_local_geometry", "follows_cross_sensor"),
        acceptable_input_types=(
            "projected_track_set.v1",
            "canonical_entity_map.v1",
            "route_graph.v1",
        ),
        output_types=("predicate_match.v1", "pair_trajectory.v1"),
    )

    step = PlanStep(
        step_id="step_follows_cross_sensor",
        provider_id="follows_cross_sensor",
        node_id="edge_1",
        input_artifact_ids=(artifact.artifact_id,),
        input_data_types=(
            "canonical_entity_map.v1",
            "projected_track_set.v1",
            "route_graph.v1",
        ),
        output_data_types=("predicate_match.v1", "pair_trajectory.v1"),
        parameters=(("maximum_gap_m", 15.0), ("minimum_duration_s", 3.0)),
        estimated_startup_ms=40,
        estimated_execution_ms=70,
        estimated_transfer_ms=15,
        estimated_transfer_bytes=8192,
    )
    label = PhysicalPlanLabel(
        checkpoint_id=checkpoint.checkpoint_id,
        covered_demand_ids=(demand.demand_id,),
        steps=(step,),
        input_artifact_ids=(artifact.artifact_id,),
        continuation_output_types=("pair_trajectory.v1",),
        cost=PlanCost(
            predicted_completion_ms=125,
            deadline_slack_ms=6875,
            startup_cost_ms=40,
            resource_cost_units=1.25,
            transfer_bytes=8192,
        ),
        hard_constraints_satisfied=True,
        quality_floor_satisfied=True,
        feasibility_reasons=("all required representations are available",),
    )
    plan = ExecutionPlan(
        label_id=label.label_id,
        checkpoint_id=checkpoint.checkpoint_id,
        demand_ids=(demand.demand_id,),
        steps=(step,),
        reservations=(
            ResourceReservation(
                node_id="edge_1",
                cpu_cores=1.0,
                memory_mb=512,
                gpu_memory_mb=0,
                network_bytes=8192,
            ),
        ),
        status=PlanStatus.ADMITTED,
        created_at=BASE_TIME + timedelta(seconds=1),
        expires_at=BASE_TIME + timedelta(seconds=15),
    )
    result = PredicateResult(
        occurrence_id=occurrence_anchor_id(
            "camera_downstream",
            "FOLLOWS",
            BASE_TIME + timedelta(seconds=5),
            {"leader": "vehicle_17", "follower": "vehicle_23"},
        ),
        demand_id=demand.demand_id,
        request_id=demand.request_id,
        graph_hash=demand.graph_hash,
        hypothesis_id=demand.hypothesis_id,
        expected_hypothesis_version=demand.hypothesis_version,
        frontier_id=demand.frontier_id,
        checkpoint_id=demand.checkpoint_id,
        graph_node_id=demand.graph_node_id,
        semantic_predicate=demand.semantic_predicate,
        truth=TruthValue.TRUE,
        confidence=0.93,
        event_time_interval=EventTimeInterval(
            start=BASE_TIME + timedelta(seconds=2),
            end=BASE_TIME + timedelta(seconds=7),
        ),
        binding_delta=BindingDelta(introduced={"follower": "vehicle_23"}),
        artifact_ids=(artifact.artifact_id,),
        provenance=ResultProvenance(
            provider_id="follows_cross_sensor",
            provider_contract_version=1,
            node_id="edge_1",
            source_ids=("camera_mobile", "camera_downstream"),
            source_sequence_ranges={
                "camera_mobile": (101, 140),
                "camera_downstream": (88, 131),
            },
        ),
        processing_started_at=BASE_TIME + timedelta(seconds=7, milliseconds=10),
        processing_completed_at=BASE_TIME + timedelta(seconds=7, milliseconds=125),
    )
    lease = ProviderLease(
        provider_instance_id="provider_instance_edge_1_follows_001",
        provider_id="follows_cross_sensor",
        provider_contract_version=1,
        demand_id=demand.demand_id,
        plan_id=plan.plan_id,
        node_id="edge_1",
        configuration_hash="sha256:" + "1" * 64,
        status=ProviderLeaseStatus.ACTIVE,
        starts_at=BASE_TIME + timedelta(seconds=1),
        expires_at=BASE_TIME + timedelta(seconds=16),
    )
    heartbeat = NodeHeartbeat(
        node_id="edge_1",
        session_id="edge_1_session_demo",
        sequence=42,
        sent_at=BASE_TIME + timedelta(seconds=8),
        availability=NodeAvailability.AVAILABLE,
        sources={
            "camera_downstream": SourceHeartbeat(
                source_id="camera_downstream",
                latest_sequence=131,
                latest_event_time=BASE_TIME + timedelta(seconds=8),
                raw_buffer_interval=EventTimeInterval(
                    start=BASE_TIME - timedelta(minutes=5),
                    end=BASE_TIME + timedelta(seconds=8),
                ),
                operational_coverage=True,
            )
        },
        active_provider_instance_ids=(lease.provider_instance_id,),
        active_demand_ids=(demand.demand_id,),
        capacity=NodeCapacity(
            cpu_free_cores=4.5,
            memory_free_mb=8192,
            gpu_free_mb=4096,
            network_tx_available_mbps=800.0,
            network_rx_available_mbps=850.0,
        ),
    )

    records = {
        "convoy_graph.json": graph,
        "robbery_graph.json": robbery_graph(),
        "hypothesis.json": hypothesis,
        "frontier_snapshot.json": frontier,
        "semantic_checkpoint.json": checkpoint,
        "predicate_demand.json": demand,
        "provider_contract.json": provider_contract,
        "provider_family.json": family,
        "artifact_ref.json": artifact,
        "physical_plan_label.json": label,
        "execution_plan.json": plan,
        "predicate_result.json": result,
        "provider_lease.json": lease,
        "node_heartbeat.json": heartbeat,
    }
    for filename, record in records.items():
        write_fixture(fixtures / filename, record)
        print(f"wrote {fixtures / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
