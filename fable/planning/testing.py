"""Deterministic Phase-2/3 fake deployment, artifacts, and semantic demand."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from fable.common.enums import ArtifactAccessMode, ArtifactLocationKind
from fable.common.examples import BASE_TIME, convoy_graph
from fable.common.schemas import ArtifactLocation, ArtifactProducer, ArtifactRef
from fable.common.time import EventTimeInterval
from fable.semantic import (
    CanonicalBindingManager,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    seed_result_from_spec,
)

from .artifact_catalog import ArtifactCatalog
from .demand_compiler import DemandCompileContext, DemandCompiler
from .deployment import DeploymentGraph
from .models import ComputeCapacity, DeploymentNode, NetworkLink, SensorSource
from .predicate_registry import default_predicate_registry
from .provider_registry import ProviderRegistry


def fake_deployment() -> DeploymentGraph:
    return DeploymentGraph(
        nodes=(
            DeploymentNode(
                node_id="sensor_a",
                node_class="sensor",
                region="road_a",
                capacity=ComputeCapacity(cpu_cores=4, memory_mb=8192, gpu_memory_mb=4096),
                capabilities=("vision", "audio", "gpu"),
            ),
            DeploymentNode(
                node_id="sensor_b",
                node_class="sensor",
                region="road_b",
                capacity=ComputeCapacity(cpu_cores=4, memory_mb=8192, gpu_memory_mb=4096),
                capabilities=("vision", "gpu"),
            ),
            DeploymentNode(
                node_id="edge_1",
                node_class="edge",
                region="campus",
                capacity=ComputeCapacity(cpu_cores=16, memory_mb=32768, gpu_memory_mb=8192),
                capabilities=("vision", "audio", "gpu"),
            ),
            DeploymentNode(
                node_id="server_1",
                node_class="server",
                region="cloud",
                capacity=ComputeCapacity(cpu_cores=32, memory_mb=65536, gpu_memory_mb=24576),
                capabilities=("vision", "audio", "gpu"),
            ),
        ),
        sources=(
            SensorSource(
                source_id="camera_mobile",
                node_id="sensor_a",
                region="road_a",
                modalities=("vision",),
                live_data_types=("raw_video_frames.v1",),
                coverage_regions=("road_a", "campus"),
                raw_buffer_interval=EventTimeInterval(
                    start=BASE_TIME - timedelta(minutes=5),
                    end=BASE_TIME + timedelta(minutes=5),
                ),
            ),
            SensorSource(
                source_id="camera_downstream",
                node_id="sensor_b",
                region="road_b",
                modalities=("vision",),
                live_data_types=("raw_video_frames.v1",),
                coverage_regions=("road_b", "campus"),
                raw_buffer_interval=EventTimeInterval(
                    start=BASE_TIME - timedelta(minutes=5),
                    end=BASE_TIME + timedelta(minutes=5),
                ),
            ),
            SensorSource(
                source_id="microphone_store",
                node_id="sensor_a",
                region="store",
                modalities=("audio",),
                live_data_types=("audio_segment.v1",),
                coverage_regions=("store",),
                raw_buffer_interval=EventTimeInterval(
                    start=BASE_TIME - timedelta(minutes=5),
                    end=BASE_TIME + timedelta(minutes=5),
                ),
            ),
        ),
        links=(
            NetworkLink(
                source_node_id="sensor_a",
                target_node_id="edge_1",
                latency_ms=8,
                bandwidth_mbps=200,
            ),
            NetworkLink(
                source_node_id="sensor_b",
                target_node_id="edge_1",
                latency_ms=10,
                bandwidth_mbps=150,
            ),
            NetworkLink(
                source_node_id="edge_1",
                target_node_id="server_1",
                latency_ms=20,
                bandwidth_mbps=500,
            ),
        ),
    )


def _artifact(
    *,
    artifact_type: str,
    node_id: str,
    uri_name: str,
    bindings: dict[str, str],
    bytes_count: int,
    start=BASE_TIME - timedelta(days=1),
    end=BASE_TIME + timedelta(days=1),
    compatibility_keys: dict | None = None,
    consumer_families: tuple[str, ...] = (),
) -> ArtifactRef:
    return ArtifactRef(
        artifact_type=artifact_type,
        artifact_schema_version=artifact_type,
        producer=ArtifactProducer(
            provider_id="fixture_provider",
            provider_contract_version=1,
            model_id="fixture",
            model_version="1",
        ),
        event_time_interval=EventTimeInterval(start=start, end=end),
        bindings=bindings,
        location=ArtifactLocation(
            kind=ArtifactLocationKind.LOCAL_PATH,
            node_id=node_id,
            uri=f"file:///tmp/fable/{uri_name}",
        ),
        access_modes=(
            ArtifactAccessMode.LOCAL,
            ArtifactAccessMode.TRANSFERRED,
            ArtifactAccessMode.REMOTE_REFERENCE,
        ),
        compatibility_keys=compatibility_keys or {},
        compatible_consumer_families=consumer_families,
        bytes=bytes_count,
        created_at=BASE_TIME - timedelta(days=1),
        valid_until=BASE_TIME + timedelta(days=1),
        expires_at=BASE_TIME + timedelta(days=2),
    )


def fake_artifact_catalog() -> ArtifactCatalog:
    interval_start = BASE_TIME
    interval_end = BASE_TIME + timedelta(seconds=30)
    return ArtifactCatalog(
        (
            _artifact(
                artifact_type="camera_calibration.v1",
                node_id="sensor_a",
                uri_name="calibration_a.json",
                bindings={"source_id": "camera_mobile", "coordinate_frame_id": "world"},
                bytes_count=16_000,
            ),
            _artifact(
                artifact_type="camera_calibration.v1",
                node_id="sensor_b",
                uri_name="calibration_b.json",
                bindings={"source_id": "camera_downstream", "coordinate_frame_id": "world"},
                bytes_count=16_000,
            ),
            _artifact(
                artifact_type="route_graph.v1",
                node_id="edge_1",
                uri_name="route.json",
                bindings={"deployment_id": "fake"},
                bytes_count=128_000,
            ),
            _artifact(
                artifact_type="raw_video_frames.v1",
                node_id="sensor_a",
                uri_name="camera_mobile.mkv",
                bindings={"source_id": "camera_mobile"},
                bytes_count=18_000_000,
                start=interval_start,
                end=interval_end,
            ),
            _artifact(
                artifact_type="raw_video_frames.v1",
                node_id="sensor_b",
                uri_name="camera_downstream.mkv",
                bindings={"source_id": "camera_downstream"},
                bytes_count=18_000_000,
                start=interval_start,
                end=interval_end,
            ),
            _artifact(
                artifact_type="detection_set.v1",
                node_id="sensor_a",
                uri_name="retained_detections.jsonl",
                bindings={"source_id": "camera_mobile"},
                bytes_count=512_000,
                start=interval_start,
                end=interval_end,
            ),
        )
    )


def fake_provider_registry() -> ProviderRegistry:
    project_root = Path(__file__).resolve().parents[2]
    return ProviderRegistry.from_files(
        catalog_path=project_root / "providers" / "registry" / "catalog.yaml",
        data_types_path=project_root / "providers" / "registry" / "data_types.yaml",
    )


def fake_follow_frontier():
    bindings = CanonicalBindingManager()
    bindings.register_alias(
        entity_type="vehicle",
        source_id="camera_mobile",
        local_entity_id="track_17",
        canonical_entity_id="vehicle_17",
    )
    runtime = SemanticRuntime(
        convoy_graph(),
        config=SemanticRuntimeConfig(
            request_id="phase23_convoy",
            hypothesis_horizon_ms=60_000,
            deadline_offset_ms=60_000,
            lateness_policy={"allowed_lateness_ms": 1000},
        ),
        bindings=bindings,
    )
    seed = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="leader_passes",
            source_id="camera_mobile",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(seconds=1),
            ),
            introduced={"leader": "track_17"},
        ),
    )
    transition = runtime.seed(seed)
    hypothesis = runtime.get_hypothesis(transition.hypothesis_ids[0])
    frontier = runtime.get_frontier(hypothesis.hypothesis_id)
    assert frontier is not None
    return runtime, hypothesis, frontier


def fake_follow_demand(*, required_continuation: str | None = None):
    deployment = fake_deployment()
    runtime, hypothesis, frontier = fake_follow_frontier()
    checkpoint = frontier.checkpoint_for_node(
        runtime.graph.nodes_by_key["follower_follows"].node_id
    )
    required = (
        {str(checkpoint.checkpoint_id): (required_continuation,)}
        if required_continuation is not None
        else {}
    )
    compiler = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=deployment,
    )
    demands = compiler.compile_frontier(
        graph=runtime.graph,
        hypothesis=hypothesis,
        frontier=frontier,
        context=DemandCompileContext(
            eligible_source_ids_by_node={
                runtime.graph.nodes_by_key["follower_follows"].node_id: (
                    "camera_mobile",
                    "camera_downstream",
                )
            },
            required_continuations_by_checkpoint=required,
        ),
    )
    assert len(demands) == 1
    return demands[0]
