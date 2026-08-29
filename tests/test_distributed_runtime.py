from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import subprocess

from fable.execution import (
    DirectCommandTransport,
    FableRuntime,
    InProcessProviderRuntime,
    NetworkMonitor,
    NodeAgent,
    NodeAgentTCPServer,
    NodeEndpoint,
    ProviderInstanceKey,
    ProviderInstanceSpec,
    TcpCommandTransport,
    reconcile_plan,
)
from fable.language import compile_event, parse_event
from fable.planning import (
    ExecutionPlan,
    NodeState,
    PlanStep,
    RuntimeState,
    RunningProvider,
    SourceState,
    load_provider_profiles,
)
from fable.providers import PredicateMatch, load_provider_capabilities

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_reconciler_computes_start_keep_stop() -> None:
    running = (
        RunningProvider("detector", "edge", ("cam",)),
        RunningProvider("old", "edge", ("cam",)),
    )
    plan = ExecutionPlan(
        steps=(
            PlanStep("detector", "edge", ("cam",), "detections"),
            PlanStep("tracker", "edge", ("cam",), "tracks"),
        ),
        covers={}, predicted_completion_ms=0, transfer_bytes=0,
        new_provider_count=1, peak_resource_fraction=0, quality=1,
    )
    actions = reconcile_plan(running, plan)
    assert [row.key.provider_id for row in actions.start] == ["tracker"]
    assert [row.key.provider_id for row in actions.keep] == ["detector"]
    assert [row.provider_id for row in actions.stop] == ["old"]


def test_node_agent_tcp_transport_round_trip() -> None:
    runtime = InProcessProviderRuntime({"p": object})
    agent = NodeAgent(NodeState("edge", "edge"), runtime)
    server = NodeAgentTCPServer(agent, "127.0.0.1", 0)
    server.start_background()
    try:
        host, port = server.address
        transport = TcpCommandTransport({"edge": (host, port)})
        spec = ProviderInstanceSpec(ProviderInstanceKey("p", "edge", ("cam",)), "tracks")
        assert transport.start(spec).ok
        assert transport.status("edge").running[0].provider_id == "p"
        assert transport.stop(spec.key).ok
        assert transport.status("edge").running == ()
    finally:
        server.close()


def test_network_monitor_defaults_to_ping_and_can_learn_passive_throughput() -> None:
    calls = []

    def runner(argv, timeout):
        calls.append(tuple(argv))
        if "ping" in argv:
            return subprocess.CompletedProcess(
                argv, 0,
                stdout="rtt min/avg/max/mdev = 4.000/10.000/20.000/1.000 ms\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            argv, 0,
            stdout=json.dumps({"end": {"sum_received": {"bits_per_second": 125_000_000}}}),
            stderr="",
        )

    monitor = NetworkMonitor(command_runner=runner, passive_alpha=1.0)
    a = NodeEndpoint("a", "10.0.0.1")
    b = NodeEndpoint("b", "10.0.0.2")
    link = monitor.measure(a, b)
    assert link.latency_ms == 5.0
    assert link.bandwidth_mbps is None
    assert not any("iperf3" in call for call in calls)

    passive = monitor.record_transfer("a", "b", size_bytes=10_000_000, duration_ms=1000)
    assert passive.bandwidth_mbps == 80.0
    assert passive.bandwidth_source == "passive"

    probed = monitor.measure_bandwidth(a, b)
    assert probed.bandwidth_mbps == 125.0
    assert probed.bandwidth_source == "iperf3"
    assert any("iperf3" in call for call in calls)


def test_fable_runtime_activates_and_deactivates_continuation_work() -> None:
    event = compile_event(parse_event({
        "event": "vehicle_gunshot_departure",
        "roles": {"VEHICLE": {"class": "vehicle"}},
        "pattern": {
            "seq": [
                {"enters": {"object": "VEHICLE"}},
                {"audio_event": {"class": "gunshot"}},
                {"exits": {"object": "VEHICLE"}},
            ]
        },
    }))
    state = RuntimeState(
        nodes={"local": NodeState("local", "local")},
        sources={
            "cam": SourceState("cam", "local", "video_frame", "site", sample_bytes=250_000),
            "mic": SourceState("mic", "local", "audio_window", "site", sample_bytes=64_000),
        },
        profiles=load_provider_profiles(),
    )
    provider_ids = load_provider_capabilities()["providers"]
    agent = NodeAgent(
        NodeState("local", "local"),
        InProcessProviderRuntime({provider_id: object for provider_id in provider_ids}),
    )
    runtime = FableRuntime(event, state, DirectCommandTransport({"local": agent}))

    initial = runtime.start(T0)
    assert "enters_basic" in initial.plan.provider_ids
    assert "audio_event_classifier" not in initial.plan.provider_ids

    after_vehicle = runtime.handle_predicate_match(PredicateMatch(
        "enters", T0 + timedelta(seconds=1),
        arguments={"object": "cam:track_1"}, classes={"object": "car"},
        provider_id="enters_basic", source_ids=("cam",),
    ))
    assert "audio_event_classifier" in after_vehicle.plan.provider_ids
    assert any(spec.key.provider_id == "audio_event_classifier" for spec in after_vehicle.actions.start)

    after_audio = runtime.handle_predicate_match(PredicateMatch(
        "audio_event", T0 + timedelta(seconds=2),
        arguments={"class": "gunshot"}, provider_id="audio_event_classifier",
        source_ids=("mic",),
    ))
    assert "audio_event_classifier" not in after_audio.plan.provider_ids
    assert any(key.provider_id == "audio_event_classifier" for key in after_audio.actions.stop)
    assert "exits_basic" in after_audio.plan.provider_ids


def test_system_resource_probe_returns_node_state() -> None:
    from fable.execution import SystemResourceProbe
    state = SystemResourceProbe("local", "test")()
    assert state.node_id == "local"
    assert state.node_type == "test"
    assert state.cpu_free >= 0
    assert state.memory_mb_free > 0
    assert state.gpu_memory_mb_free >= 0
