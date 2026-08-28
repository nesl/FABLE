from __future__ import annotations

from datetime import timedelta

from fable.common.enums import GraphEdgeKind, GraphNodeKind
from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.planning import DemandCompileContext, DemandCompiler, default_predicate_registry
from fable.planning.testing import fake_deployment
from fable.semantic import (
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)
from fable.semantic.definitions import (
    drive_up_shooting_graph,
    package_exchange_graph,
    repeated_visit_graph,
    sequential_vehicle_pass_graph,
    talking_rendezvous_graph,
    uncalibrated_repeated_pass_graph,
    vehicle_convergence_graph,
)
from fable.semantic.request_compiler import EventRequestCompiler, RequestCompileError
from evaluation.live_requests import _partition_multi_seed_acquisition_demand
import pytest


def _node(graph, key):
    return next(item for item in graph.nodes if item.authored_key == key)


def _sequence_keys(graph, root_key):
    root = _node(graph, root_key)
    order = {node.node_id: node.authored_key for node in graph.nodes}
    sequence_edges = [
        edge
        for edge in graph.edges
        if edge.kind == GraphEdgeKind.SEQUENCE
    ]
    successors = {
        order[edge.source_node_id]: order[edge.target_node_id]
        for edge in sequence_edges
    }
    keys = []
    current = next(
        order[edge.target_node_id]
        for edge in graph.edges
        if edge.source_node_id == root.node_id
        and edge.kind == GraphEdgeKind.CHILD
        and order[edge.target_node_id] not in successors.values()
    )
    while current:
        keys.append(current)
        current = successors.get(current)
    return keys


def _assert_rearm_policy(graph):
    root = next(item for item in graph.nodes if item.node_id == graph.root_node_id)
    policy = root.annotations["post_completion_policy"]
    assert policy["mode"] == "scene_clear_rearm"
    assert policy["clear_interval_ms"] == 7_500
    assert policy["reset_interval_label"] == "TRIAL_RESET"
    assert policy["exclude_reset_from_positive_annotations"] is True


def test_vehicle_convergence_requires_convergence_and_every_bound_exit() -> None:
    graph = vehicle_convergence_graph()
    assert _sequence_keys(graph, "convergence") == [
        "seed_passes",
        "vehicles_converge",
        "all_group_members_exit",
    ]
    convergence = _node(graph, "vehicles_converge")
    assert convergence.kind == GraphNodeKind.PREDICATE
    assert convergence.annotations["source_affinity_roles"] == ["seed_vehicle"]
    exits = _node(graph, "all_group_members_exit")
    assert exits.annotations["completion_policy"] == "all_bound_members"
    child_keys = {
        _node(graph, key).authored_key
        for key in ("vehicle_a_exits", "vehicle_b_exits")
    }
    assert child_keys == {"vehicle_a_exits", "vehicle_b_exits"}
    assert _node(graph, "vehicle_a_exits").annotations["source_affinity_roles"] == [
        "vehicle_a"
    ]
    assert _node(graph, "vehicle_b_exits").annotations["source_affinity_roles"] == [
        "vehicle_b"
    ]
    _assert_rearm_policy(graph)


def test_bound_camera_fov_constrains_successor_to_same_source() -> None:
    runtime = SemanticRuntime(
        sequential_vehicle_pass_graph(
            name="Sequential pass affinity",
            namespace_suffix="sequential_pass_affinity",
        ),
        config=SemanticRuntimeConfig(
            request_id="sequential-pass-affinity",
            hypothesis_horizon_ms=120_000,
            deadline_offset_ms=120_000,
        ),
    )
    seeded = runtime.seed(
        seed_result_from_spec(
            runtime,
            ScriptedResultSpec(
                node_key="leader_passes",
                source_id="camera_mobile",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(seconds=1),
                ),
                introduced={
                    "leader": "camera_mobile:session:1",
                    "reference": "camera_fov:sensor_a",
                },
            ),
        )
    )
    hypothesis = runtime.get_hypothesis(seeded.hypothesis_ids[0])
    frontier = runtime.get_frontier(hypothesis.hypothesis_id)
    assert frontier is not None
    follower_id = runtime.graph.nodes_by_key["follower_passes"].node_id
    demands = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=fake_deployment(),
    ).compile_frontier(
        graph=runtime.graph,
        hypothesis=hypothesis,
        frontier=frontier,
        context=DemandCompileContext(
            eligible_source_ids_by_node={
                follower_id: ("camera_mobile", "camera_downstream")
            }
        ),
    )
    assert len(demands) == 1
    assert demands[0].bound_roles["reference"] == "camera_fov:sensor_a"
    assert demands[0].eligible_source_ids == ("camera_mobile",)

    partitioned = _partition_multi_seed_acquisition_demand(
        demands[0].model_copy(
            update={
                "eligible_source_ids": (
                    "camera_mobile",
                    "camera_downstream",
                ),
                "source_preferences": (),
            }
        ),
        eligible_source_ids=("camera_mobile", "camera_downstream"),
        deployment=fake_deployment(),
    )
    assert len(partitioned) == 2
    assert len({item.demand_id for item in partitioned}) == 2
    assert {item.eligible_source_ids for item in partitioned} == {
        ("camera_mobile",),
        ("camera_downstream",),
    }
    assert {
        item.hard_constraints.allowed_node_ids for item in partitioned
    } == {("sensor_a",), ("sensor_b",)}

    transferable = demands[0].model_copy(
        update={
            "eligible_source_ids": ("camera_mobile", "camera_downstream"),
            "source_preferences": (),
            "hard_constraints": demands[0].hard_constraints.model_copy(
                update={
                    "raw_data_must_remain_local": False,
                    "allowed_node_ids": (),
                }
            ),
        }
    )
    transferable_partitions = _partition_multi_seed_acquisition_demand(
        transferable,
        eligible_source_ids=("camera_mobile", "camera_downstream"),
        deployment=fake_deployment(),
    )
    assert {item.eligible_source_ids for item in transferable_partitions} == {
        ("camera_mobile",),
        ("camera_downstream",),
    }
    assert {
        item.hard_constraints.allowed_node_ids
        for item in transferable_partitions
    } == {()}


def test_vehicle_exit_demands_remain_on_identity_originating_source() -> None:
    runtime = SemanticRuntime(
        vehicle_convergence_graph(),
        config=SemanticRuntimeConfig(
            request_id="source-affinity",
            hypothesis_horizon_ms=120_000,
            deadline_offset_ms=120_000,
        ),
    )
    seeded = runtime.seed(
        seed_result_from_spec(
            runtime,
            ScriptedResultSpec(
                node_key="seed_passes",
                source_id="camera_downstream",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(seconds=1),
                ),
                introduced={"seed_vehicle": "camera_downstream:session:seed"},
            ),
        )
    )
    hypothesis_id = seeded.hypothesis_ids[-1]
    convergence = predicate_result_from_spec(
        runtime,
        hypothesis_id,
        ScriptedResultSpec(
            node_key="vehicles_converge",
            source_id="camera_mobile",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME + timedelta(seconds=10),
                end=BASE_TIME + timedelta(seconds=12),
            ),
            introduced={
                "vehicle_a": "camera_mobile:session:1",
                "vehicle_b": "camera_mobile:session:2",
            },
        ),
    )
    progressed = runtime.apply(convergence)
    hypothesis_id = progressed.hypothesis_ids[-1]
    frontier = runtime.get_frontier(hypothesis_id)
    assert frontier is not None
    demands = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=fake_deployment(),
    ).compile_frontier(
        graph=runtime.graph,
        hypothesis=runtime.get_hypothesis(hypothesis_id),
        frontier=frontier,
    )
    assert len(demands) == 2
    assert {d.semantic_predicate.predicate_id for d in demands} == {"EXITS"}
    assert {d.eligible_source_ids for d in demands} == {("camera_mobile",)}


def test_vehicle_convergence_scene_departure_policy_is_explicit() -> None:
    graph = vehicle_convergence_graph(departure_policy="scene_departures")
    exits = _node(graph, "all_group_members_exit")
    assert exits.annotations["completion_policy"] == "two_distinct_scene_departures"
    assert _node(graph, "vehicle_a_exits").predicate.roles[0].variable == (
        "departing_vehicle_a"
    )
    assert _node(graph, "vehicle_b_exits").predicate.roles[0].variable == (
        "departing_vehicle_b"
    )


def test_full_talking_rendezvous_requires_interaction_and_bound_exit() -> None:
    graph = talking_rendezvous_graph()
    assert _sequence_keys(graph, "rendezvous") == [
        "arrival",
        "rendezvous_interaction",
        "arrival_vehicle_exits",
    ]
    departure = _node(graph, "arrival_vehicle_exits")
    assert departure.predicate.predicate_id == "EXITS"
    assert departure.annotations["source_affinity_roles"] == [
        "participant_a",
        "participant_b",
    ]
    _assert_rearm_policy(graph)


def test_package_exchange_requires_transfer_and_receiver_departure() -> None:
    graph = package_exchange_graph()
    assert _sequence_keys(graph, "package_exchange") == [
        "arrivals",
        "transfer",
        "receiver_departs",
    ]
    receiver = _node(graph, "receiver_departs")
    assert receiver.predicate.roles[0].variable == "vehicle_b"
    _assert_rearm_policy(graph)


def test_repeated_visit_definitions_encode_real_occurrence_boundaries() -> None:
    calibrated = repeated_visit_graph()
    assert _sequence_keys(calibrated, "repeated_visit_sequence") == [
        "first_visit",
        "departure",
        "return_within_window",
    ]
    uncalibrated = uncalibrated_repeated_pass_graph()
    returned = _node(uncalibrated, "return_visit")
    assert returned.annotations["requires_prior_track_termination"] is True
    assert returned.annotations["absence_gap_ms"] == 30_000
    assert returned.predicate.roles[1].variable == "visit_reference"
    _assert_rearm_policy(calibrated)
    _assert_rearm_policy(uncalibrated)


def test_drive_up_shooting_has_explicit_bound_vehicle_exit_and_variant() -> None:
    full = drive_up_shooting_graph()
    assert _sequence_keys(full, "drive_up_shooting") == [
        "disembarks",
        "gunshot",
        "boards",
        "departure",
    ]
    assert _node(full, "departure").predicate.predicate_id == "EXITS"
    assert _node(full, "drive_up_shooting").annotations["boarding_required"] is True

    no_boarding = drive_up_shooting_graph(require_boarding=False)
    assert _sequence_keys(no_boarding, "drive_up_shooting") == [
        "disembarks",
        "gunshot",
        "departure",
    ]
    assert (
        _node(no_boarding, "drive_up_shooting").annotations["variant"]
        == "boarding_unobservable"
    )
    _assert_rearm_policy(full)
    _assert_rearm_policy(no_boarding)


def test_request_parameters_expose_gap_and_boarding_variants() -> None:
    repeated = EventRequestCompiler().compile(
        {
            "family_id": "repeated_visit",
            "parameters": {
                "evaluation_profile": "uncalibrated_passes",
                "minimum_return_gap_ms": 45_000,
            },
        }
    ).graph
    assert _node(repeated, "return_visit").annotations["absence_gap_ms"] == 45_000

    shooting = EventRequestCompiler().compile(
        {
            "family_id": "drive_up_shooting",
            "parameters": {"require_boarding": False},
        }
    ).graph
    assert _sequence_keys(shooting, "drive_up_shooting") == [
        "disembarks",
        "gunshot",
        "departure",
    ]

    with pytest.raises(RequestCompileError, match="must be a boolean"):
        EventRequestCompiler().compile(
            {
                "family_id": "drive_up_shooting",
                "parameters": {"require_boarding": "false"},
            }
        )

    talking = EventRequestCompiler().compile(
        {
            "family_id": "rendezvous",
            "parameters": {"evaluation_profile": "full_talking"},
        }
    ).graph
    assert _sequence_keys(talking, "rendezvous") == [
        "arrival",
        "rendezvous_interaction",
        "arrival_vehicle_exits",
    ]
