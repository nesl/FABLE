from datetime import timedelta

from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.semantic import (
    ApplyStatus,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
)
from fable.semantic.definitions.vehicle import vehicle_convergence_graph


def _interval(seconds: int) -> EventTimeInterval:
    return EventTimeInterval(
        start=BASE_TIME + timedelta(seconds=seconds),
        end=BASE_TIME + timedelta(seconds=seconds + 1),
    )


def test_concurrent_distinct_departures_cross_binding_forks() -> None:
    runtime = SemanticRuntime(
        vehicle_convergence_graph(departure_policy="scene_departures"),
        config=SemanticRuntimeConfig(request_id="concurrent-departures"),
    )
    started = runtime.start(
        event_time_window=_interval(0).model_copy(
            update={"end": BASE_TIME + timedelta(minutes=5)}
        ),
        observed_at=BASE_TIME,
    )
    hypothesis_id = started.hypothesis_ids[0]

    def advance(node_key: str, introduced: dict[str, str], seconds: int) -> None:
        nonlocal hypothesis_id
        result = predicate_result_from_spec(
            runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key=node_key,
                source_id="camera",
                event_time_interval=_interval(seconds),
                introduced=introduced,
            ),
        )
        transition = runtime.apply(result)
        assert transition.status == ApplyStatus.FORKED
        hypothesis_id = transition.hypothesis_ids[0]

    advance("seed_passes", {"seed_vehicle": "seed"}, 1)
    advance(
        "vehicles_converge",
        {"vehicle_a": "arrival-a", "vehicle_b": "arrival-b"},
        10,
    )

    # Both provider commands were issued against this same parent frontier.
    departure_a = predicate_result_from_spec(
        runtime,
        hypothesis_id,
        ScriptedResultSpec(
            node_key="vehicle_a_exits",
            source_id="camera",
            event_time_interval=_interval(20),
            occurrence_id="exit-a",
            introduced={"departing_vehicle_a": "vehicle-a"},
        ),
    )
    departure_b = predicate_result_from_spec(
        runtime,
        hypothesis_id,
        ScriptedResultSpec(
            node_key="vehicle_b_exits",
            source_id="camera",
            event_time_interval=_interval(21),
            occurrence_id="exit-b",
            introduced={"departing_vehicle_b": "vehicle-b"},
        ),
    )
    assert runtime.apply(departure_a).status == ApplyStatus.FORKED
    completed = runtime.apply(departure_b)

    assert completed.status == ApplyStatus.FORKED
    assert any(
        runtime.get_hypothesis(item).lifecycle.value == "COMPLETED"
        for item in completed.hypothesis_ids
    )
