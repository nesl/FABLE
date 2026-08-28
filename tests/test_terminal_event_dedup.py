from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.orchestration.controller import FableController
from fable.planning import ArtifactCatalog, RuntimeDeploymentView
from fable.planning.testing import fake_deployment
from fable.semantic import (
    CanonicalBindingManager,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)
from fable.semantic.definitions.vehicle import sequential_vehicle_pass_graph
from fable.semantic.definitions.vehicle import vehicle_convergence_graph

from .fake_phase6_data import make_stack


def _completed_convoys(*, leader_occurrence: str = "leader-occurrence"):
    bindings = CanonicalBindingManager()
    for local_id, canonical_id in (
        ("leader-track", "leader-vehicle"),
        ("follower-a", "follower-vehicle-a"),
        ("follower-b", "follower-vehicle-b"),
    ):
        bindings.register_alias(
            entity_type="vehicle",
            source_id="camera",
            local_entity_id=local_id,
            canonical_entity_id=canonical_id,
        )
    runtime = SemanticRuntime(
        sequential_vehicle_pass_graph(name="Convoy", namespace_suffix="dedup_test"),
        config=SemanticRuntimeConfig(request_id="convoy-request"),
        bindings=bindings,
    )
    seed = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="leader_passes",
            source_id="camera",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(seconds=1),
            ),
            introduced={"leader": "leader-track"},
            occurrence_id=leader_occurrence,
        ),
    )
    parent_id = runtime.seed(seed).hypothesis_ids[0]
    results = []
    for offset, follower in enumerate(("follower-a", "follower-b"), start=2):
        results.append(
            predicate_result_from_spec(
                runtime,
                parent_id,
                ScriptedResultSpec(
                    node_key="follower_passes",
                    source_id="camera",
                    event_time_interval=EventTimeInterval(
                        start=BASE_TIME + timedelta(seconds=offset),
                        end=BASE_TIME + timedelta(seconds=offset + 1),
                    ),
                    introduced={"follower": follower},
                ),
            )
        )
    completed = []
    for result in results:
        completed.append(runtime.get_hypothesis(runtime.apply(result).hypothesis_ids[0]))
    return SimpleNamespace(
        runtime=runtime,
        family_id="convoy",
        terminal_occurrence_times=[],
        submission=SimpleNamespace(planning_policy_id="FABLE"),
    ), completed


def _completed_convergence(*, suffix: str, offset: int = 0):
    runtime = SemanticRuntime(
        vehicle_convergence_graph(departure_policy="scene_departures"),
        config=SemanticRuntimeConfig(request_id="convergence-request"),
    )
    started = runtime.start(
        event_time_window=EventTimeInterval(
            start=BASE_TIME,
            end=BASE_TIME + timedelta(minutes=5),
        ),
        observed_at=BASE_TIME,
    )
    hypothesis_id = started.hypothesis_ids[0]

    def advance(node_key, introduced, seconds):
        nonlocal hypothesis_id
        transition = runtime.apply(
            predicate_result_from_spec(
                runtime,
                hypothesis_id,
                ScriptedResultSpec(
                    node_key=node_key,
                    source_id="camera",
                    event_time_interval=EventTimeInterval(
                        start=BASE_TIME + timedelta(seconds=offset + seconds),
                        end=BASE_TIME + timedelta(seconds=offset + seconds + 1),
                    ),
                    introduced=introduced,
                ),
            )
        )
        hypothesis_id = transition.hypothesis_ids[0]

    advance("seed_passes", {"seed_vehicle": f"seed-{suffix}"}, 1)
    advance(
        "vehicles_converge",
        {"vehicle_a": f"arrival-a-{suffix}", "vehicle_b": f"arrival-b-{suffix}"},
        2,
    )
    advance("vehicle_a_exits", {"departing_vehicle_a": f"exit-a-{suffix}"}, 3)
    advance("vehicle_b_exits", {"departing_vehicle_b": f"exit-b-{suffix}"}, 4)
    completed = runtime.get_hypothesis(hypothesis_id)
    assert completed.lifecycle.value == "COMPLETED"
    return SimpleNamespace(
        runtime=runtime,
        family_id="vehicle_convergence",
        terminal_occurrence_times=[],
        submission=SimpleNamespace(planning_policy_id="FABLE"),
    ), completed


def test_convoy_follower_alternatives_share_one_terminal_occurrence_key() -> None:
    state, completed = _completed_convoys()

    assert completed[0].canonical_key != completed[1].canonical_key
    assert FableController._terminal_event_key(
        state, completed[0]
    ) == FableController._terminal_event_key(state, completed[1])


def test_distinct_leader_occurrences_remain_distinct_terminal_events() -> None:
    first_state, first = _completed_convoys(leader_occurrence="leader-pass-1")
    second_state, second = _completed_convoys(leader_occurrence="leader-pass-2")

    assert FableController._terminal_event_key(
        first_state, first[0]
    ) != FableController._terminal_event_key(second_state, second[0])


def test_convoy_follower_alternatives_emit_one_durable_terminal_event(tmp_path) -> None:
    state, completed = _completed_convoys()
    stack = make_stack(tmp_path)
    try:
        controller = FableController(
            orchestrator=stack.orchestrator,
            provider_registry=stack.registry,
            deployment_view=RuntimeDeploymentView(fake_deployment()),
            artifact_catalog=ArtifactCatalog(),
        )

        controller._emit_completed_event(state, completed[0])
        controller._emit_completed_event(state, completed[1])

        assert len(stack.store.list_raw("terminal_events")) == 1
        assert len(stack.store.list_raw("emitted_events")) == 1
    finally:
        stack.stop()


def test_scene_clear_rearm_suppresses_fragmented_terminal_alternatives(tmp_path) -> None:
    state, first = _completed_convergence(suffix="a")
    _, fragmented = _completed_convergence(suffix="b", offset=2)
    stack = make_stack(tmp_path)
    try:
        controller = FableController(
            orchestrator=stack.orchestrator,
            provider_registry=stack.registry,
            deployment_view=RuntimeDeploymentView(fake_deployment()),
            artifact_catalog=ArtifactCatalog(),
        )
        controller._emit_completed_event(state, first)
        controller._emit_completed_event(state, fragmented)
        assert len(stack.store.list_raw("terminal_events")) == 1
    finally:
        stack.stop()


def test_scene_clear_rearm_allows_later_terminal_occurrence(tmp_path) -> None:
    state, first = _completed_convergence(suffix="a")
    _, later = _completed_convergence(suffix="b", offset=10)
    stack = make_stack(tmp_path)
    try:
        controller = FableController(
            orchestrator=stack.orchestrator,
            provider_registry=stack.registry,
            deployment_view=RuntimeDeploymentView(fake_deployment()),
            artifact_catalog=ArtifactCatalog(),
        )
        controller._emit_completed_event(state, first)
        controller._emit_completed_event(state, later)
        assert len(stack.store.list_raw("terminal_events")) == 2
    finally:
        stack.stop()
