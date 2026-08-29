from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fable.execution import (
    DataflowProviderRuntime,
    DirectCommandTransport,
    DirectResultTransport,
    FableRuntime,
    ManualSourceAdapter,
    NodeAgent,
    ProviderInstanceKey,
    ResultTCPServer,
    TcpResultTransport,
    reconcile_plan,
)
from fable.language import compile_event, parse_event
from fable.planning import ExecutionPlan, NodeState, PlanStep, RuntimeState, SourceState, load_provider_profiles
from fable.providers import BoundingBox, Detection, DetectionFrame, PredicateMatch, VideoFrame
from fable.providers.identity import IdentityAssociation

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeVehicleDetector:
    provider_id = "yolo_vehicle_fast_640"
    provider_version = "test"

    def detect(self, image, *, source_id, event_time, frame_id=""):
        rows = ()
        if image == "car":
            rows = (Detection("car", 0.95, BoundingBox(0, 0, 20, 20), "d1"),)
        return DetectionFrame(source_id, event_time, rows, frame_id)


def _enter_exit_event():
    return compile_event(parse_event({
        "event": "enter_exit_live",
        "roles": {"VEHICLE": {"class": "vehicle"}},
        "pattern": {
            "seq": [
                {"enters": {"object": "VEHICLE"}},
                {"exits": {"object": "VEHICLE"}},
            ]
        },
    }))


def test_live_dataflow_closes_source_provider_result_ce_loop() -> None:
    event = _enter_exit_event()
    source = ManualSourceAdapter("cam", "video_frame")
    holder = {}
    results = DirectResultTransport(
        lambda match: holder["runtime"].handle_predicate_match(match),
        lambda assoc: holder["runtime"].handle_identity_association(assoc, now=T0),
    )
    dataflow = DataflowProviderRuntime(
        result_transport=results,
        provider_factories={"yolo_vehicle_fast_640": FakeVehicleDetector},
        source_adapters={"cam": source},
    )
    agent = NodeAgent(NodeState("local", "local"), dataflow)
    state = RuntimeState(
        nodes={"local": NodeState("local", "local")},
        sources={"cam": SourceState("cam", "local", "video_frame", "site", 250_000)},
        profiles=load_provider_profiles(),
    )
    runtime = FableRuntime(event, state, DirectCommandTransport({"local": agent}))
    holder["runtime"] = runtime

    initial = runtime.start(T0)
    assert "enters_basic" in initial.plan.provider_ids
    assert dataflow.ready(ProviderInstanceKey("enters_basic", "local", ("cam",)))

    # Initialize enters() with an empty view, then observe a vehicle.
    source.emit(VideoFrame("cam", T0, "empty", "0"))
    source.emit(VideoFrame("cam", T0 + timedelta(seconds=1), "car", "1"))
    assert len(runtime.manager.active_instances()) == 1
    assert "exits_basic" in runtime.last_plan.provider_ids

    # exits_basic was activated after the enter match. Let it see the vehicle,
    # then enough empty time to establish the visibility transition.
    source.emit(VideoFrame("cam", T0 + timedelta(seconds=1.1), "car", "2"))
    source.emit(VideoFrame("cam", T0 + timedelta(seconds=2.0), "empty", "3"))
    source.emit(VideoFrame("cam", T0 + timedelta(seconds=2.6), "empty", "4"))

    completed = runtime.manager.completed_instances()
    assert len(completed) == 1
    assert completed[0].bindings["VEHICLE"] == "cam:track_1"
    # Discovery remains live after the continuation finishes.
    assert "enters_basic" in runtime.last_plan.provider_ids
    assert "exits_basic" not in runtime.last_plan.provider_ids


def test_result_tcp_transport_delivers_predicate_and_identity_results() -> None:
    matches = []
    associations = []
    server = ResultTCPServer(matches.append, associations.append, "127.0.0.1", 0)
    server.start_background()
    try:
        host, port = server.address
        client = TcpResultTransport(host, port)
        client.send_predicate_match(PredicateMatch(
            "enters", T0,
            arguments={"object": "cam:track_1"},
            classes={"object": "car"},
            provider_id="enters_basic",
            source_ids=("cam",),
        ))
        client.send_identity_association(IdentityAssociation("cam:a", "cam2:b", 0.9))
        assert matches[0].predicate == "enters"
        assert associations[0].right_object_id == "cam2:b"
    finally:
        server.close()


def test_reconciler_starts_consumers_before_producers() -> None:
    plan = ExecutionPlan(
        steps=(
            PlanStep("yolo_vehicle_fast_640", "edge", ("cam",), "detections"),
            PlanStep("multi_object_tracker", "edge", ("cam",), "tracks"),
            PlanStep("enters_basic", "edge", ("cam",), "predicate_match:enters"),
        ),
        covers={}, predicted_completion_ms=0, transfer_bytes=0,
        new_provider_count=3, peak_resource_fraction=0, quality=1,
    )
    actions = reconcile_plan((), plan)
    assert [spec.key.provider_id for spec in actions.start] == [
        "enters_basic", "multi_object_tracker", "yolo_vehicle_fast_640"
    ]

class FakeAudioEventProvider:
    provider_id = "audio_event_classifier"
    provider_version = "test"

    def classify(self, window):
        return (PredicateMatch(
            "audio_event", window.event_time,
            arguments={"class": "gunshot"},
            provider_id=self.provider_id,
            source_ids=(window.source_id,),
            confidence=0.9,
            provider_version=self.provider_version,
        ),)


def test_continuation_only_source_activates_and_deactivates_with_frontier() -> None:
    from fable.providers import AudioWindow
    import pytest

    event = compile_event(parse_event({
        "event": "vehicle_audio",
        "roles": {"VEHICLE": {"class": "vehicle"}},
        "pattern": {
            "seq": [
                {"enters": {"object": "VEHICLE"}},
                {"audio_event": {"class": "gunshot"}},
                {"exits": {"object": "VEHICLE"}},
            ]
        },
    }))
    mic = ManualSourceAdapter("mic", "audio_window")
    holder = {}
    results = DirectResultTransport(lambda m: holder["runtime"].handle_predicate_match(m))
    dataflow = DataflowProviderRuntime(
        result_transport=results,
        provider_factories={
            "audio_event_classifier": FakeAudioEventProvider,
            "yolo_vehicle_fast_640": FakeVehicleDetector,
        },
        source_adapters={"mic": mic},
    )
    agent = NodeAgent(NodeState("local", "local"), dataflow)
    state = RuntimeState(
        nodes={"local": NodeState("local", "local")},
        sources={
            "cam": SourceState("cam", "local", "video_frame", "site", 250_000),
            "mic": SourceState("mic", "local", "audio_window", "site", 64_000),
        },
        profiles=load_provider_profiles(),
    )
    runtime = FableRuntime(event, state, DirectCommandTransport({"local": agent}))
    holder["runtime"] = runtime
    runtime.start(T0)

    with pytest.raises(RuntimeError, match="not active"):
        mic.emit(AudioWindow("mic", T0, (0.0,) * 10))

    runtime.handle_predicate_match(PredicateMatch(
        "enters", T0 + timedelta(seconds=1),
        arguments={"object": "cam:track_1"}, classes={"object": "car"},
        provider_id="enters_basic", source_ids=("cam",),
    ))
    # The audio continuation made the source/provider live.
    mic.emit(AudioWindow("mic", T0 + timedelta(seconds=2), (0.1,) * 10))
    assert "audio_event_classifier" not in runtime.last_plan.provider_ids

    with pytest.raises(RuntimeError, match="not active"):
        mic.emit(AudioWindow("mic", T0 + timedelta(seconds=3), (0.1,) * 10))
