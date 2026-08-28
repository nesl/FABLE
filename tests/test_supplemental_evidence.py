from datetime import timedelta

from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.semantic import (
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
)
from fable.semantic.definitions import package_exchange_graph
from fable.semantic.phase8_examples import package_exchange_graph as compatibility_graph


def test_package_exchange_compatibility_path_is_canonical() -> None:
    assert compatibility_graph().graph_hash == package_exchange_graph().graph_hash


def test_package_exchange_requires_receiver_departure() -> None:
    runtime = SemanticRuntime(
        package_exchange_graph(),
        config=SemanticRuntimeConfig(request_id="package-regression"),
    )
    transition = runtime.start(
        event_time_window=EventTimeInterval(
            start=BASE_TIME, end=BASE_TIME + timedelta(minutes=5)
        ),
        observed_at=BASE_TIME,
    )
    hypothesis_id = transition.hypothesis_ids[0]
    for index, (key, introduced, validated) in enumerate((
        ("arrive_a", {"vehicle_a": "a"}, {}),
        ("arrive_b", {"vehicle_b": "b"}, {}),
        ("transfer", {"package": "p", "source_holder": "x", "destination_holder": "y"}, {}),
    ), 1):
        transition = runtime.apply(predicate_result_from_spec(
            runtime, hypothesis_id,
            ScriptedResultSpec(
                node_key=key,
                source_id="physical_rpi_camera_replay",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=index),
                    end=BASE_TIME + timedelta(seconds=index, milliseconds=1),
                ),
                introduced=introduced,
                validated=validated,
            ),
        ))
        hypothesis_id = transition.hypothesis_ids[-1]

    hypothesis = runtime.get_hypothesis(hypothesis_id)
    assert hypothesis.lifecycle.value == "ACTIVE"
    active = runtime.get_frontier(hypothesis_id)
    assert {
        runtime.graph.nodes_by_id[node_id].authored_key
        for node_id in active.snapshot.enabled_node_ids
    } == {"receiver_departs"}

    transition = runtime.apply(predicate_result_from_spec(
        runtime, hypothesis_id,
        ScriptedResultSpec(
            node_key="receiver_departs",
            source_id="physical_rpi_camera_replay",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME + timedelta(seconds=4),
                end=BASE_TIME + timedelta(seconds=4, milliseconds=1),
            ),
            validated={"vehicle_b": "b"},
        ),
    ))
    assert runtime.get_hypothesis(transition.hypothesis_ids[-1]).lifecycle.value == "COMPLETED"
