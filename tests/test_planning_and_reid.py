from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import subprocess

from fable.language import compile_event, parse_event
from fable.planning import (
    LinkState,
    NodeState,
    PhysicalPlanner,
    ProviderProfile,
    ProviderSearcher,
    RuntimeState,
    SourceState,
    load_provider_profiles,
)
from fable.execution import NetworkMonitor, NodeEndpoint
from fable.providers import (
    EmbeddingVector,
    FastReIDDescriptorBackend,
    TorchreidDescriptorBackend,
    CrossSensorIdentityAssociationProvider,
    load_provider_capabilities,
)
from fable.runtime import CEInstanceManager

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _near_event():
    return compile_event(parse_event({
        "event": "dogs_near",
        "roles": {"DOG_A": {"class": "dog"}, "DOG_B": {"class": "dog"}},
        "pattern": {"near": {"object_a": "DOG_A", "object_b": "DOG_B"}},
    }))


def _person_enter_event():
    return compile_event(parse_event({
        "event": "person_enter",
        "roles": {"PERSON": {"class": "person"}},
        "pattern": {"enters": {"object": "PERSON"}},
    }))


def test_provider_search_discovers_chain_from_declared_io() -> None:
    manager = CEInstanceManager(_near_event())
    item = manager.current_frontier(T0).discovery[0]
    searcher = ProviderSearcher()
    recipes = searcher.recipes_for_frontier_item(item)
    signatures = [recipe.provider_ids() for recipe in recipes]
    assert any(row[-1] == "near_geometry" for row in signatures)
    assert any("multi_object_tracker" in row for row in signatures)
    assert any("yolo_full_context_960" in row for row in signatures)


def test_new_direct_provider_is_discovered_without_authored_chain() -> None:
    catalog = load_provider_capabilities()
    catalog["providers"]["direct_near_model"] = {
        "kind": "predicate_implementation",
        "enabled": True,
        "predicates": {
            "near": {"visual_arguments": {"object_a": "*", "object_b": "*"}, "semantic_literals": {}}
        },
        "inputs": ("video_frame",),
        "outputs": ("predicate_match:near",),
    }
    item = CEInstanceManager(_near_event()).current_frontier(T0).discovery[0]
    recipes = ProviderSearcher(catalog).recipes_for_frontier_item(item)
    assert any(recipe.provider_id == "direct_near_model" for recipe in recipes)


def test_planner_selects_fast_vehicle_discovery_by_default() -> None:
    event = compile_event(parse_event({
        "event": "vehicle_enter",
        "roles": {"VEHICLE": {"class": "vehicle"}},
        "pattern": {"enters": {"object": "VEHICLE"}},
    }))
    frontier = CEInstanceManager(event).current_frontier(T0)
    runtime = RuntimeState(
        nodes={"local": NodeState("local", "local")},
        sources={"cam": SourceState("cam", "local", "video_frame", "site", sample_bytes=250_000)},
        profiles=load_provider_profiles(),
    )
    plan = PhysicalPlanner().plan(frontier, runtime, now=T0)
    assert "yolo_vehicle_fast_640" in plan.provider_ids
    assert "enters_basic" in plan.provider_ids


def test_online_network_conditions_change_placement_choice() -> None:
    event = _person_enter_event()
    frontier = CEInstanceManager(event).current_frontier(T0)
    profiles = load_provider_profiles()
    # Override only the three relevant providers to make sensor vs edge compute
    # distinction explicit and deterministic.
    for provider_id in ("yolo_full_context_960", "multi_object_tracker", "enters_basic"):
        base = profiles[(provider_id, "*")]
        profiles[(provider_id, "sensor")] = ProviderProfile(
            provider_id, "sensor", 0, base.execution_ms * 4, base.cpu, base.memory_mb,
            base.gpu_memory_mb, base.output_bytes, base.quality,
        )
        profiles[(provider_id, "edge")] = ProviderProfile(
            provider_id, "edge", 0, max(1, base.execution_ms / 4), base.cpu, base.memory_mb,
            base.gpu_memory_mb, base.output_bytes, base.quality,
        )
    nodes = {
        "sensor": NodeState("sensor", "sensor", cpu_free=8, memory_mb_free=8192, gpu_memory_mb_free=8192),
        "edge": NodeState("edge", "edge", cpu_free=8, memory_mb_free=8192, gpu_memory_mb_free=8192),
    }
    source = {"cam": SourceState("cam", "sensor", "video_frame", "site", sample_bytes=2_000_000)}
    planner = PhysicalPlanner()

    fast_network = RuntimeState(
        nodes=nodes,
        sources=source,
        links=(LinkState("sensor", "edge", latency_ms=1, bandwidth_mbps=1000),),
        profiles=profiles,
    )
    fast_plan = planner.plan(frontier, fast_network, now=T0)
    assert any(step.provider_id == "yolo_full_context_960" and step.node_id == "edge" for step in fast_plan.steps)

    slow_network = RuntimeState(
        nodes=nodes,
        sources=source,
        links=(LinkState("sensor", "edge", latency_ms=25, bandwidth_mbps=1),),
        profiles=profiles,
    )
    slow_plan = planner.plan(frontier, slow_network, now=T0)
    assert any(step.provider_id == "yolo_full_context_960" and step.node_id == "sensor" for step in slow_plan.steps)


def test_network_monitor_uses_ping_for_latency_and_iperf_for_throughput() -> None:
    calls = []

    def runner(argv, timeout):
        calls.append(tuple(argv))
        if "ping" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="rtt min/avg/max/mdev = 4.000/10.000/20.000/1.000 ms\n", stderr="")
        return subprocess.CompletedProcess(
            argv, 0,
            stdout=json.dumps({"end": {"sum_received": {"bits_per_second": 125_000_000}}}),
            stderr="",
        )

    monitor = NetworkMonitor(command_runner=runner)
    link = monitor.measure(
        NodeEndpoint("a", "10.0.0.1"),
        NodeEndpoint("b", "10.0.0.2"),
        measure_bandwidth=True,
    )
    assert link.available
    assert link.latency_ms == 5.0  # RTT / 2
    assert link.bandwidth_mbps == 125.0
    assert any("iperf3" in call for call in calls)


def test_reid_backends_are_model_adapters_not_placeholder_vectors() -> None:
    fast = FastReIDDescriptorBackend(extractor=lambda images: [[3.0, 4.0] for _ in images])
    person = TorchreidDescriptorBackend(extractor=lambda images: [[0.0, 5.0] for _ in images])
    assert tuple(round(v, 3) for v in fast.embed(object())) == (0.6, 0.8)
    assert tuple(round(v, 3) for v in person.embed(object())) == (0.0, 1.0)


def test_identity_association_requires_same_model_version_and_uses_global_matching() -> None:
    provider = CrossSensorIdentityAssociationProvider(minimum_cosine_similarity=0.8)
    left = [
        EmbeddingVector("a1", "camA", T0, (1.0, 0.0), "reid", "v1"),
        EmbeddingVector("a2", "camA", T0, (0.0, 1.0), "reid", "v1"),
    ]
    right = [
        EmbeddingVector("b1", "camB", T0 + timedelta(seconds=1), (0.99, 0.01), "reid", "v1"),
        EmbeddingVector("b2", "camB", T0 + timedelta(seconds=1), (0.01, 0.99), "reid", "v1"),
        EmbeddingVector("wrong_version", "camB", T0, (1.0, 0.0), "reid", "v2"),
    ]
    assert provider.associate(left, right) == {"a1": "b1", "a2": "b2"}


def test_local_reid_pipeline_applies_identity_without_ce_dedup() -> None:
    from fable.execution import LocalRunner, ReIDPipeline
    from fable.providers import (
        ImageCrop,
        VehicleReIDDescriptorProvider,
        PersonReIDDescriptorProvider,
    )

    vehicle_descriptor = VehicleReIDDescriptorProvider(
        FastReIDDescriptorBackend(extractor=lambda images: [[1.0, 0.0] for _ in images])
    )
    person_descriptor = PersonReIDDescriptorProvider(
        TorchreidDescriptorBackend(extractor=lambda images: [[1.0, 0.0] for _ in images])
    )
    reid = ReIDPipeline(
        vehicle_descriptor=vehicle_descriptor,
        person_descriptor=person_descriptor,
        association_provider=CrossSensorIdentityAssociationProvider(minimum_cosine_similarity=0.9),
    )
    event = compile_event(parse_event({
        "event": "vehicle_enter",
        "roles": {"VEHICLE": {"class": "vehicle"}},
        "pattern": {"enters": {"object": "VEHICLE"}},
    }))
    runner = LocalRunner(event, reid_pipeline=reid)
    left = [ImageCrop("camA:track_1", "camA", T0, object())]
    right = [ImageCrop("camB:track_9", "camB", T0 + timedelta(seconds=1), object())]
    associations = runner.process_reid_crops(left, right, entity_kind="vehicle")
    assert len(associations) == 1
    assert runner.identity_resolver.canonical("camA:track_1") == runner.identity_resolver.canonical("camB:track_9")
