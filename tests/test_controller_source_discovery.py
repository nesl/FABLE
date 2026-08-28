from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.common.time import EventTimeInterval
from fable.distributed.models import EventRequestSubmission
from fable.orchestration.controller import (
    FableController,
    _DiscoveryCandidate,
    _RequestState,
    _requires_source_discovery_fanout,
)
from fable.planning import DemandCompileContext, RuntimeDeploymentView
from fable.planning.testing import fake_deployment
from fable.semantic import (
    ApplyStatus,
    EventRequestCompiler,
    RuntimeTransition,
    SemanticRuntime,
    SemanticRuntimeConfig,
)


def test_single_retained_seed_still_discovers_every_eligible_source() -> None:
    """Retention cardinality must not become a source-acquisition limit."""

    deployment = fake_deployment()
    controller = object.__new__(FableController)
    controller.deployment_view = RuntimeDeploymentView(deployment)
    compilation = EventRequestCompiler().compile(
        {
            "family_id": "vehicle_convergence",
            "parameters": {"departure_policy": "scene_departures"},
        }
    )
    runtime = SemanticRuntime(
        compilation.graph,
        config=SemanticRuntimeConfig(request_id="source-coverage"),
    )
    interval = EventTimeInterval(
        start=BASE_TIME,
        end=BASE_TIME + timedelta(minutes=5),
    )
    transition = runtime.start(event_time_window=interval, observed_at=BASE_TIME)
    hypothesis_id = transition.hypothesis_ids[0]
    allowed_nodes = ("sensor_a", "sensor_b", "edge_1", "server_1")
    submission = EventRequestSubmission(
        submitter_id="test",
        request_id="source-coverage",
        family_id="vehicle_convergence",
        parameters={"departure_policy": "scene_departures"},
        event_time_window=interval,
        allowed_node_ids=allowed_nodes,
        # One retained hypothesis is intentional. Both cameras must still be
        # searched before the first accepted binding chooses that hypothesis.
        max_seed_hypotheses=1,
    )
    state = _RequestState(
        submission=submission,
        family_id="vehicle_convergence",
        runtime=runtime,
        demand_context=DemandCompileContext(allowed_node_ids=allowed_nodes),
        discovery_hypothesis_id=hypothesis_id,
    )

    demands = controller._compile_frontier_demands(state, hypothesis_id)

    assert {item.eligible_source_ids for item in demands} == {
        ("camera_mobile",),
        ("camera_downstream",),
    }
    assert len({item.demand_id for item in demands}) == 2


def test_source_discovery_partition_preserves_trusted_edge_execution_scope() -> None:
    deployment = fake_deployment()
    controller = object.__new__(FableController)
    controller.deployment_view = RuntimeDeploymentView(deployment)
    compilation = EventRequestCompiler().compile(
        {
            "family_id": "vehicle_convergence",
            "parameters": {"departure_policy": "scene_departures"},
        }
    )
    runtime = SemanticRuntime(
        compilation.graph,
        config=SemanticRuntimeConfig(request_id="offload-source-coverage"),
    )
    interval = EventTimeInterval(
        start=BASE_TIME,
        end=BASE_TIME + timedelta(minutes=5),
    )
    transition = runtime.start(event_time_window=interval, observed_at=BASE_TIME)
    hypothesis_id = transition.hypothesis_ids[0]
    allowed_nodes = ("sensor_a", "sensor_b", "edge_1", "server_1")
    submission = EventRequestSubmission(
        submitter_id="test",
        request_id="offload-source-coverage",
        family_id="vehicle_convergence",
        parameters={"departure_policy": "scene_departures"},
        event_time_window=interval,
        allowed_node_ids=allowed_nodes,
        raw_data_must_remain_local=False,
    )
    state = _RequestState(
        submission=submission,
        family_id="vehicle_convergence",
        runtime=runtime,
        demand_context=DemandCompileContext(
            allowed_node_ids=allowed_nodes,
            raw_data_must_remain_local=False,
        ),
        discovery_hypothesis_id=hypothesis_id,
    )

    demands = controller._compile_frontier_demands(state, hypothesis_id)

    assert len(demands) == 2
    assert all(
        demand.hard_constraints.allowed_node_ids == allowed_nodes
        for demand in demands
    )


def test_unbound_multisource_exit_requires_fable_candidate_fanout() -> None:
    demand = SimpleNamespace(
        eligible_source_ids=("orin11", "orin14", "orin15", "orin16"),
        unbound_roles=("vehicle",),
        bound_roles={},
        semantic_predicate=SimpleNamespace(predicate_id="EXITS"),
    )

    assert _requires_source_discovery_fanout(demand)


def test_bound_or_single_source_exit_does_not_fan_out() -> None:
    bound = SimpleNamespace(
        eligible_source_ids=("orin11", "orin14"),
        unbound_roles=(),
        bound_roles={"vehicle": "orin11:replay:track-1"},
        semantic_predicate=SimpleNamespace(predicate_id="EXITS"),
    )
    single_source = SimpleNamespace(
        eligible_source_ids=("orin11",),
        unbound_roles=("vehicle",),
        bound_roles={},
        semantic_predicate=SimpleNamespace(predicate_id="EXITS"),
    )

    assert not _requires_source_discovery_fanout(bound)
    assert not _requires_source_discovery_fanout(single_source)


def test_transition_persistence_only_writes_named_hypotheses() -> None:
    parent_id, child_id, unrelated_id = uuid7(), uuid7(), uuid7()
    hypotheses = {
        hypothesis_id: SimpleNamespace(hypothesis_id=hypothesis_id)
        for hypothesis_id in (parent_id, child_id, unrelated_id)
    }

    class Runtime:
        def get_hypothesis(self, hypothesis_id):
            return hypotheses[hypothesis_id]

        def get_frontier(self, _hypothesis_id):
            return None

    writes = []
    controller = object.__new__(FableController)
    controller.orchestrator = SimpleNamespace(
        store=SimpleNamespace(put=lambda collection, key, value: writes.append((collection, key)))
    )
    state = SimpleNamespace(runtime=Runtime())
    transition = RuntimeTransition(
        status=ApplyStatus.FORKED,
        parent_hypothesis_id=parent_id,
        hypothesis_ids=(child_id,),
    )

    controller._persist_transition(state, transition)

    assert set(writes) == {
        ("hypotheses", str(parent_id)),
        ("hypotheses", str(child_id)),
    }
    assert ("hypotheses", str(unrelated_id)) not in writes


def test_discovery_eviction_preserves_oldest_equal_progress_causal_seed() -> None:
    oldest_id, newest_id = uuid7(), uuid7()

    class Runtime:
        def get_hypothesis(self, hypothesis_id):
            assert hypothesis_id in {oldest_id, newest_id}
            return SimpleNamespace(
                node_states={
                    "seed": SimpleNamespace(
                        status=SimpleNamespace(value="SATISFIED")
                    ),
                    "future": SimpleNamespace(
                        status=SimpleNamespace(value="ENABLED")
                    ),
                }
            )

    state = SimpleNamespace(runtime=Runtime())
    oldest = _DiscoveryCandidate(
        hypothesis_id=oldest_id,
        partition="camera-a",
        identity="vehicle-1",
        admitted_order=0,
    )
    newest = _DiscoveryCandidate(
        hypothesis_id=newest_id,
        partition="camera-a",
        identity="vehicle-2",
        admitted_order=1,
    )

    victim = min(
        (oldest, newest),
        key=lambda item: FableController._discovery_eviction_key(state, item),
    )

    assert victim.hypothesis_id == newest_id


def test_identity_gated_binding_fork_preserves_candidate_producer() -> None:
    child_id = uuid7()
    demand_id = uuid7()
    parent_id = uuid7()
    runtime = SimpleNamespace(
        get_hypothesis=lambda hypothesis_id: SimpleNamespace(
            hypothesis_id=hypothesis_id,
            role_bindings={
                "vehicle": SimpleNamespace(canonical_entity_id="seed-vehicle")
            },
            anchor_occurrence_id="seed-occurrence",
        )
    )
    state = SimpleNamespace(
        submission=SimpleNamespace(
            allowed_node_ids=("orin11",), max_seed_hypotheses=12
        ),
        runtime=runtime,
        retained_identity_candidate_demands={},
        retained_identity_candidate_times={},
        retained_repeated_visit_candidates={},
    )
    result = SimpleNamespace(
        hypothesis_id=parent_id,
        demand_id=demand_id,
        request_id="identity-candidates",
        event_time_interval=EventTimeInterval(
            start=BASE_TIME,
            end=BASE_TIME + timedelta(seconds=1),
        ),
        binding_delta=SimpleNamespace(introduced={"visit_vehicle_2": "track-2"}),
    )
    transition = SimpleNamespace(
        status=SimpleNamespace(value="FORKED"),
        hypothesis_ids=(child_id,),
    )
    identity_demand = SimpleNamespace(
        semantic_predicate=SimpleNamespace(predicate_id="SAME_ENTITY"),
        unbound_roles=(),
    )

    controller = object.__new__(FableController)
    preserved = controller._preserve_identity_candidate_frontier(
        state,
        result,
        transition,
        [identity_demand],
    )

    assert preserved
    assert state.retained_identity_candidate_demands == {child_id: demand_id}


def test_repeated_visit_candidate_pool_deduplicates_tracker_burst_per_source() -> None:
    first_child, duplicate_child = uuid7(), uuid7()
    demand_id = uuid7()
    invalidated = set()

    class Runtime:
        def invalidate_hypothesis(self, hypothesis_id):
            invalidated.add(hypothesis_id)
            return True

        def get_hypothesis(self, hypothesis_id):
            return SimpleNamespace(
                hypothesis_id=hypothesis_id,
                role_bindings={
                    "vehicle": SimpleNamespace(canonical_entity_id="seed-vehicle")
                },
                anchor_occurrence_id="seed-occurrence",
            )

    controller = object.__new__(FableController)
    controller.orchestrator = SimpleNamespace(
        store=SimpleNamespace(put=lambda *args: None)
    )
    state = SimpleNamespace(
        submission=SimpleNamespace(
            allowed_node_ids=("orin11",), max_seed_hypotheses=12
        ),
        runtime=Runtime(),
        retained_identity_candidate_demands={first_child: demand_id},
        retained_repeated_visit_candidates={
            ("visit_vehicle_2", "orin11"): [
                (BASE_TIME, first_child),
                (BASE_TIME + timedelta(seconds=1), first_child),
                (BASE_TIME + timedelta(seconds=2), first_child),
            ]
        },
    )
    result = SimpleNamespace(
        hypothesis_id=first_child,
        demand_id=demand_id,
        request_id="repeated-visit-pool",
        event_time_interval=EventTimeInterval(
            start=BASE_TIME + timedelta(seconds=2),
            end=BASE_TIME + timedelta(seconds=3),
        ),
        binding_delta=SimpleNamespace(
            introduced={"visit_vehicle_2": "orin11:replay:track-9"}
        ),
    )
    transition = SimpleNamespace(
        status=SimpleNamespace(value="FORKED"),
        hypothesis_ids=(duplicate_child,),
    )
    identity_demand = SimpleNamespace(
        semantic_predicate=SimpleNamespace(predicate_id="SAME_ENTITY"),
        unbound_roles=(),
    )

    assert controller._preserve_identity_candidate_frontier(
        state, result, transition, [identity_demand]
    )
    assert invalidated == {duplicate_child}
    assert duplicate_child not in state.retained_identity_candidate_demands


def test_robbery_departure_identity_candidate_preserves_producer_without_pooling() -> None:
    child_id = uuid7()
    demand_id = uuid7()
    controller = object.__new__(FableController)
    state = SimpleNamespace(
        retained_identity_candidate_demands={},
        retained_repeated_visit_candidates={},
    )
    result = SimpleNamespace(
        demand_id=demand_id,
        binding_delta=SimpleNamespace(
            introduced={"departing_vehicle": "orin16:replay:track-2"}
        ),
    )
    transition = SimpleNamespace(
        status=SimpleNamespace(value="FORKED"),
        hypothesis_ids=(child_id,),
    )
    identity_demand = SimpleNamespace(
        semantic_predicate=SimpleNamespace(predicate_id="SAME_ENTITY"),
        unbound_roles=(),
    )

    assert controller._preserve_identity_candidate_frontier(
        state, result, transition, [identity_demand]
    )
    assert state.retained_identity_candidate_demands == {child_id: demand_id}
    assert state.retained_repeated_visit_candidates == {}


def test_unrelated_non_visit_identity_candidate_keeps_normal_completion() -> None:
    child_id = uuid7()
    demand_id = uuid7()
    controller = object.__new__(FableController)
    state = SimpleNamespace(
        retained_identity_candidate_demands={},
        retained_repeated_visit_candidates={},
    )
    result = SimpleNamespace(
        demand_id=demand_id,
        binding_delta=SimpleNamespace(
            introduced={"returning_vehicle": "orin16:replay:track-2"}
        ),
    )
    transition = SimpleNamespace(
        status=SimpleNamespace(value="FORKED"),
        hypothesis_ids=(child_id,),
    )
    identity_demand = SimpleNamespace(
        semantic_predicate=SimpleNamespace(predicate_id="SAME_ENTITY"),
        unbound_roles=(),
    )

    assert not controller._preserve_identity_candidate_frontier(
        state, result, transition, [identity_demand]
    )
    assert state.retained_identity_candidate_demands == {}
    assert state.retained_repeated_visit_candidates == {}


def test_robbery_candidate_dedup_is_track_and_source_scoped() -> None:
    controller = object.__new__(FableController)
    state = SimpleNamespace(
        family_id="robbery",
        robbery_candidate_observations=set(),
        robbery_candidate_counts={},
    )

    def result(*, source="orin14", entity="orin14:replay:track-7", predicate="EXITS"):
        return SimpleNamespace(
            semantic_predicate=SimpleNamespace(predicate_id=predicate),
            binding_delta=SimpleNamespace(
                introduced={"departing_vehicle": entity}
            ),
            provenance=SimpleNamespace(
                source_ids=(source,), node_id=source
            ),
        )

    assert not controller._duplicate_robbery_candidate(state, result())
    assert controller._duplicate_robbery_candidate(state, result())
    assert controller._duplicate_robbery_candidate(
        state, result(entity="orin14:different-replay:track-7")
    )
    assert not controller._duplicate_robbery_candidate(
        state, result(entity="orin14:replay:track-8")
    )
    assert not controller._duplicate_robbery_candidate(
        state, result(source="orin15")
    )
    assert not controller._duplicate_robbery_candidate(
        state, result(predicate="PASSES")
    )


def test_robbery_candidate_pool_is_bounded_per_source_and_predicate() -> None:
    controller = object.__new__(FableController)
    state = SimpleNamespace(
        family_id="robbery",
        robbery_candidate_observations=set(),
        robbery_candidate_counts={},
    )

    def result(index, source="orin14"):
        return SimpleNamespace(
            semantic_predicate=SimpleNamespace(
                predicate_id="VEHICLE_PRESENT_BEFORE"
            ),
            binding_delta=SimpleNamespace(
                introduced={"vehicle": f"{source}:replay:track-{index}"}
            ),
            provenance=SimpleNamespace(
                source_ids=(source,), node_id=source
            ),
        )

    assert all(
        not controller._duplicate_robbery_candidate(state, result(index))
        for index in range(4)
    )
    assert controller._duplicate_robbery_candidate(state, result(4))
    assert not controller._duplicate_robbery_candidate(
        state, result(4, source="orin15")
    )


def test_repeated_visit_pool_rolls_forward_and_cancels_oldest_candidate() -> None:
    old_children = [uuid7() for _ in range(12)]
    new_child = uuid7()
    old_demand = uuid7()
    old_lease = uuid7()
    invalidated = set()
    cancelled = []
    identity_cancellations = []

    class Runtime:
        def invalidate_hypothesis(self, hypothesis_id):
            invalidated.add(hypothesis_id)
            return True

        def get_hypothesis(self, hypothesis_id):
            return SimpleNamespace(
                hypothesis_id=hypothesis_id,
                role_bindings={
                    "vehicle": SimpleNamespace(canonical_entity_id="seed-vehicle")
                },
                anchor_occurrence_id="seed-occurrence",
            )

    managed = SimpleNamespace(
        hypothesis_id=old_children[0],
        lease=SimpleNamespace(demand_id=old_demand),
    )
    lifecycle = SimpleNamespace(
        leases={old_lease: managed},
        cancel_demand=lambda demand_id: (old_lease,),
    )
    controller = object.__new__(FableController)
    controller.orchestrator = SimpleNamespace(
        store=SimpleNamespace(put=lambda *args: None),
        lifecycle=lifecycle,
        dispatcher=SimpleNamespace(
            send_cancel=lambda lease, reason: cancelled.append((lease, reason)),
            send_identity_demand_cancel=lambda **kwargs: identity_cancellations.append(
                kwargs
            ),
        ),
    )
    state = SimpleNamespace(
        submission=SimpleNamespace(
            allowed_node_ids=("orin11",), max_seed_hypotheses=12
        ),
        runtime=Runtime(),
        retained_identity_candidate_demands={
            child: old_demand for child in old_children
        },
        retained_repeated_visit_candidates={
            ("visit_vehicle_2", "orin11"): [
                (BASE_TIME + timedelta(seconds=11 * index), child)
                for index, child in enumerate(old_children)
            ]
        },
    )
    result = SimpleNamespace(
        hypothesis_id=old_children[0],
        demand_id=uuid7(),
        request_id="rolling-candidates",
        event_time_interval=EventTimeInterval(
            start=BASE_TIME + timedelta(seconds=70),
            end=BASE_TIME + timedelta(seconds=71),
        ),
        binding_delta=SimpleNamespace(
            introduced={"visit_vehicle_2": "orin11:replay:new-track"}
        ),
    )
    transition = SimpleNamespace(
        status=SimpleNamespace(value="FORKED"),
        hypothesis_ids=(new_child,),
    )
    identity_demand = SimpleNamespace(
        semantic_predicate=SimpleNamespace(predicate_id="SAME_ENTITY"),
        unbound_roles=(),
    )

    assert controller._preserve_identity_candidate_frontier(
        state, result, transition, [identity_demand]
    )
    assert old_children[0] in invalidated
    assert old_children[0] not in state.retained_identity_candidate_demands
    assert new_child in state.retained_identity_candidate_demands
    assert len(
        state.retained_repeated_visit_candidates[("visit_vehicle_2", "orin11")]
    ) == 12
    assert cancelled and cancelled[0][1] == "rolling repeated-visit candidate eviction"
    assert identity_cancellations == [
        {
            "request_id": "rolling-candidates",
            "demand_id": old_demand,
            "reason": "rolling repeated-visit candidate eviction",
        }
    ]


def test_non_identity_binding_fork_keeps_normal_checkpoint_completion() -> None:
    state = SimpleNamespace(retained_identity_candidate_demands={})
    result = SimpleNamespace(
        demand_id=uuid7(),
        binding_delta=SimpleNamespace(introduced={"vehicle": "track-2"}),
    )
    transition = SimpleNamespace(
        status=SimpleNamespace(value="FORKED"),
        hypothesis_ids=(uuid7(),),
    )
    ordinary_demand = SimpleNamespace(
        semantic_predicate=SimpleNamespace(predicate_id="EXITS"),
        unbound_roles=(),
    )

    controller = object.__new__(FableController)
    assert not controller._preserve_identity_candidate_frontier(
        state,
        result,
        transition,
        [ordinary_demand],
    )
    assert state.retained_identity_candidate_demands == {}
