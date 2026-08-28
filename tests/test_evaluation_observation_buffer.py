from datetime import timedelta

from evaluation.observation_buffer import EarlyObservationBuffer
from fable.common.examples import BASE_TIME
from fable.planning.testing import fake_follow_demand, fake_follow_frontier
from fable.semantic import ScriptedResultSpec, predicate_result_from_spec
from fable.common.time import EventTimeInterval
from fable.common.schemas import BindingDelta, PredicateRole, ResultProvenance
from fable.common.ids import uuid7


def _result(runtime, hypothesis):
    return predicate_result_from_spec(
        runtime,
        hypothesis.hypothesis_id,
        ScriptedResultSpec(
            node_key="follower_follows",
            source_id="camera_mobile",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME,
            ),
            introduced={"follower": "vehicle_18"},
            validated={"leader": "vehicle_17"},
        ),
    )


def test_buffer_rematerializes_only_the_execution_envelope() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    demand = fake_follow_demand()
    observation = _result(runtime, hypothesis)
    buffer = EarlyObservationBuffer()
    buffer.add(observation, now=BASE_TIME)

    (matched,) = buffer.pop_matches(demand, now=BASE_TIME)
    assert matched.result_id != observation.result_id
    assert matched.occurrence_id == observation.occurrence_id
    assert matched.binding_delta == observation.binding_delta
    assert matched.provenance == observation.provenance
    assert matched.demand_id == demand.demand_id
    assert matched.frontier_id == demand.frontier_id
    assert len(buffer) == 0


def test_buffer_does_not_cross_identity_bindings() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    demand = fake_follow_demand()
    observation = _result(runtime, hypothesis).model_copy(
        update={
            "binding_delta": _result(runtime, hypothesis).binding_delta.model_copy(
                update={"validated": {"leader": "different_vehicle"}}
            )
        }
    )
    buffer = EarlyObservationBuffer()
    buffer.add(observation, now=BASE_TIME)
    assert buffer.pop_matches(demand, now=BASE_TIME) == ()
    assert buffer.rejection_counts(demand) == {"BOUND_ROLE": 1}
    assert len(buffer) == 1


def test_buffer_allows_structurally_unbound_role_to_be_grounded() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    demand = fake_follow_demand().model_copy(
        update={"bound_roles": {"leader": "__structural_unbound__:leader"}}
    )
    buffer = EarlyObservationBuffer()
    buffer.add(_result(runtime, hypothesis), now=BASE_TIME)

    assert len(buffer.pop_matches(demand, now=BASE_TIME)) == 1


def test_buffer_resolves_authored_gate_to_concrete_camera_fov() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    demand = fake_follow_demand().model_copy(
        update={"bound_roles": {"leader": "convergence_gate"}}
    )
    observation = _result(runtime, hypothesis).model_copy(
        update={
            "binding_delta": _result(runtime, hypothesis).binding_delta.model_copy(
                update={"validated": {"leader": "camera_fov:dvpg_gq_orin_4"}}
            )
        }
    )
    buffer = EarlyObservationBuffer()
    buffer.add(observation, now=BASE_TIME)

    assert len(buffer.pop_matches(demand, now=BASE_TIME)) == 1


def test_buffer_resolves_configured_runtime_source_alias() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    demand = fake_follow_demand()
    observation = _result(runtime, hypothesis)
    observation = observation.model_copy(
        update={
            "provenance": ResultProvenance(
                provider_id=observation.provenance.provider_id,
                provider_contract_version=(
                    observation.provenance.provider_contract_version
                ),
                node_id="dvpg_gq_orin_13",
                source_ids=("dvpg_gq_orin_13:camera",),
            )
        }
    )
    buffer = EarlyObservationBuffer()
    buffer.add(observation, now=BASE_TIME)

    assert buffer.pop_matches(demand, now=BASE_TIME) == ()
    assert buffer.rejection_counts(demand) == {"ELIGIBLE_SOURCE": 1}
    (matched,) = buffer.pop_matches(
        demand,
        now=BASE_TIME,
        source_aliases={
            "dvpg_gq_orin_13:camera": tuple(demand.eligible_source_ids),
        },
    )
    assert matched.demand_id == demand.demand_id


def test_buffer_projects_equivalent_predicate_across_authored_node_ids() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    demand = fake_follow_demand()
    observation = _result(runtime, hypothesis).model_copy(
        update={"graph_node_id": "planning_projection_node"}
    )
    buffer = EarlyObservationBuffer()
    buffer.add(observation, now=BASE_TIME)

    (matched,) = buffer.match_for_demand(demand, now=BASE_TIME)
    assert matched.graph_node_id == demand.graph_node_id


def test_buffer_rejects_different_typed_predicate_contract() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    demand = fake_follow_demand()
    observation = _result(runtime, hypothesis).model_copy(
        update={
            "graph_node_id": "different_node",
            "semantic_predicate": demand.semantic_predicate.model_copy(
                update={"predicate_id": "DIFFERENT_PREDICATE"}
            ),
        }
    )
    buffer = EarlyObservationBuffer()
    buffer.add(observation, now=BASE_TIME)

    assert buffer.match_for_demand(demand, now=BASE_TIME) == ()
    assert buffer.rejection_counts(demand) == {"SEMANTIC_PREDICATE": 1}


def test_buffer_rebinds_repeated_predicate_role_to_later_stage_variable() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    demand = fake_follow_demand()
    observed_predicate = demand.semantic_predicate.model_copy(
        update={
            "roles": tuple(
                PredicateRole(
                    role_name=role.role_name,
                    variable=("visit_vehicle_2" if role.role_name == "follower" else role.variable),
                    entity_type=role.entity_type,
                )
                for role in demand.semantic_predicate.roles
            )
        }
    )
    target_predicate = demand.semantic_predicate.model_copy(
        update={
            "roles": tuple(
                PredicateRole(
                    role_name=role.role_name,
                    variable=("visit_vehicle_3" if role.role_name == "follower" else role.variable),
                    entity_type=role.entity_type,
                )
                for role in demand.semantic_predicate.roles
            )
        }
    )
    observation = _result(runtime, hypothesis).model_copy(
        update={
            "semantic_predicate": observed_predicate,
            "binding_delta": BindingDelta(
                introduced={"visit_vehicle_2": "camera:session:track-3"},
                validated={"leader": "vehicle_17"},
            ),
        }
    )
    later = demand.model_copy(update={"semantic_predicate": target_predicate})
    buffer = EarlyObservationBuffer()
    buffer.add(observation, now=BASE_TIME)

    (matched,) = buffer.match_for_demand(later, now=BASE_TIME)

    assert matched.binding_delta.introduced == {
        "visit_vehicle_3": "camera:session:track-3"
    }
    assert matched.binding_delta.validated == {"leader": "vehicle_17"}


def test_buffer_is_size_and_time_bounded() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    buffer = EarlyObservationBuffer(max_observations=1, retention_ms=1000)
    first = _result(runtime, hypothesis)
    second = _result(runtime, hypothesis)
    buffer.add(first, now=BASE_TIME)
    buffer.add(second, now=BASE_TIME)
    assert len(buffer) == 1
    assert buffer.expire(now=BASE_TIME + timedelta(seconds=2)) == 1
    assert len(buffer) == 0


def test_retained_observation_is_delivered_once_to_each_hypothesis() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    first = fake_follow_demand()
    second = first.model_copy(update={"hypothesis_id": uuid7()})
    buffer = EarlyObservationBuffer()
    buffer.add(_result(runtime, hypothesis), now=BASE_TIME)

    assert len(buffer.match_for_demand(first, now=BASE_TIME)) == 1
    assert buffer.match_for_demand(first, now=BASE_TIME) == ()
    assert len(buffer.match_for_demand(second, now=BASE_TIME)) == 1
    # Delivery is non-destructive until the bounded retention window expires.
    assert len(buffer) == 1


def test_matching_batch_preserves_all_candidates_for_one_frontier() -> None:
    runtime, hypothesis, _ = fake_follow_frontier()
    demand = fake_follow_demand()
    first = _result(runtime, hypothesis)
    second = _result(runtime, hypothesis).model_copy(
        update={"occurrence_id": "later-compatible-occurrence"}
    )
    buffer = EarlyObservationBuffer()
    buffer.add(first, now=BASE_TIME)
    buffer.add(second, now=BASE_TIME)

    matched = buffer.match_for_demand(demand, now=BASE_TIME)

    assert {item.occurrence_id for item in matched} == {
        first.occurrence_id,
        second.occurrence_id,
    }
    assert buffer.match_for_demand(demand, now=BASE_TIME) == ()
