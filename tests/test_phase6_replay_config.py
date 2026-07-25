from __future__ import annotations

from pathlib import Path
import yaml

from fable.distributed.config import ProviderRuntimeResolver, load_deployment_graph
from fable.distributed.models import ReplayOutputAdapter, RuntimeMode


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "iobt-minimal-ce-replay"


def test_replay_runtime_config_adopts_existing_yolo_and_audio_containers():
    resolver = ProviderRuntimeResolver.from_yaml(
        REPLAY / "config/fable_provider_runtimes.yaml"
    )
    audio = resolver.resolve(
        node_id="dvpg_gq_orin_11", provider_id="audio_event_classifier"
    )
    assert audio.mode == RuntimeMode.ADOPT_EXISTING
    # Phase 8 supersedes the Phase-6 loudness adapter with a typed multimodal
    # audio-event provider while preserving adopt-existing semantics.
    assert audio.container_name == "fable-multimodal-orin11"
    assert audio.output_adapter == ReplayOutputAdapter.MULTIMODAL_PREDICATE
    assert audio.readiness.mqtt_topic == "/readiness/dvpg_gq_orin_11/fable_multimodal"

    yolo = resolver.resolve(
        node_id="dvpg_gq_orin_11", provider_id="yolo_vehicle_fast_640"
    )
    assert yolo.container_name == "yolo-detector-orin11"
    assert yolo.output_adapter == ReplayOutputAdapter.YOLO_OBJECT_PRESENT


def test_replay_deployment_and_compose_overlays_reference_expected_services():
    deployment = load_deployment_graph(REPLAY / "config/fable_deployment.yaml")
    assert set(deployment.nodes) == {
        "dvpg_gq_orin_11",
        "x86server",
        "cloud1",
    }
    assert deployment.source("orin11_camera").node_id == "dvpg_gq_orin_11"

    compose = yaml.safe_load((REPLAY / "compose.fable.yaml").read_text())
    assert {
        "mqtt",
        "fable-mongo",
        "fable-orchestrator",
        "fable-agent-orin11",
        "fable-agent-x86server",
    } <= set(compose["services"])
    assert compose["services"]["fable-agent-orin11"]["volumes"][0].startswith(
        "/var/run/docker.sock"
    )


def test_web_ui_subscribes_to_phase6_topics():
    app_text = (REPLAY / "web_ui/app.py").read_text()
    assert '"fable/v1/status/+/heartbeat"' in app_text
    assert '"fable/v1/result/+/+"' in app_text
