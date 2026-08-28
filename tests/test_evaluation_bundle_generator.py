from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = (
    Path(__file__).parents[1]
    / "iobt-minimal-ce-replay"
    / "setup"
    / "generate_evaluation_bundle.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("generate_evaluation_bundle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_nodes_can_target_labeled_cameras(tmp_path: Path) -> None:
    module = _module()
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": "scenario-a",
                        "nodes": ["orin11", "orin14", "orin16"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert module.selected_nodes(
        catalog,
        "scenario-a",
        None,
        requested=["orin14", "orin16"],
    ) == ["orin14", "orin16"]

    with pytest.raises(ValueError, match="does not contain requested nodes"):
        module.selected_nodes(
            catalog,
            "scenario-a",
            None,
            requested=["orin12"],
        )


def test_runtime_generation_preserves_internal_and_emitting_provider_contracts(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "runtimes.yaml"

    module.runtimes(["orin1", "orin10"], output)

    nodes = yaml.safe_load(output.read_text())["nodes"]
    assert set(nodes) == {
        "dvpg_gq_orin_1",
        "dvpg_gq_orin_10",
        "x86server",
        "cloud1",
    }
    providers = nodes["dvpg_gq_orin_1"]["providers"]
    assert providers["multi_object_tracker"]["output_adapter"] == "NONE"
    assert "output_topics" not in providers["multi_object_tracker"]
    assert providers["pass_reference_evaluator"]["output_adapter"] == "VEHICLE_PREDICATE"
    assert providers["pass_reference_evaluator"]["output_topics"] == [
        "/dvpg_gq_orin_1/fable/vehicle/predicates"
    ]
    assert providers["audio_visual_association"]["output_adapter"] == "NONE"
    assert providers["conversation_provider"]["output_adapter"] == "MULTIMODAL_PREDICATE"
    assert providers["conversation_provider"]["container_name"] == "fable-multimodal-orin1"
    cloud_proxy = nodes["cloud1"]["providers"]["hosted_vlm_identity_comparator"]
    assert cloud_proxy["stop_adopted_when_idle"] is False


def test_provider_overlay_is_self_contained_and_uses_generated_config(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "compose.fable.providers.yaml"

    module.provider_overlay(["orin2", "orin3"], output)

    document = yaml.safe_load(output.read_text())
    services = document["services"]
    assert {
        "mqtt",
        "fable-mongo",
        "fable-orchestrator",
        "fable-agent-x86server",
        "fable-agent-orin2",
        "fable-vehicle-orin2",
        "fable-multimodal-orin2",
    } <= set(services)
    assert "fable-agent-orin11" not in services
    orchestrator = services["fable-orchestrator"]
    assert orchestrator["environment"]["FABLE_DEPLOYMENT_CONFIG"] == (
        "/workspace/bundle/fable_deployment.yaml"
    )
    assert orchestrator["environment"]["FABLE_JOINT_RESOURCE_EPOCH_PLANNING"] == (
        "${FABLE_JOINT_RESOURCE_EPOCH_PLANNING:-0}"
    )
    assert f"{tmp_path}:/workspace/bundle:ro" in orchestrator["volumes"]
    assert services["fable-agent-orin2"]["environment"]["FABLE_NODE_ID"] == (
        "dvpg_gq_orin_2"
    )
    aliases = json.loads(
        services["fable-agent-orin2"]["environment"]["FABLE_SOURCE_ALIASES_JSON"]
    )
    assert aliases["camera:dvpg_gq_orin_2"] == "orin2_camera"
    assert aliases["camera:dvpg_gq_orin_3"] == "orin3_camera"
    assert (
        f"{tmp_path / 'fable_vehicle_geometry.json'}:"
        "/workspace/replay/config/fable_vehicle_geometry.json:ro"
    ) in services["fable-vehicle-orin2"]["volumes"]
    site_vehicle = services["fable-vehicle-x86server-orin2"]
    assert site_vehicle["environment"]["FABLE_COMPUTE_TIER"] == "site_local"
    assert site_vehicle["environment"]["FABLE_ASSIGNED_GPU_UUID"]
    assert site_vehicle["gpus"][0]["device_ids"] == [
        site_vehicle["environment"]["FABLE_ASSIGNED_GPU_UUID"]
    ]
    identity = services["fable-identity-x86server"]
    proxy = services["fable-vlm-cloud1"]
    assert identity["environment"]["FABLE_VLM_PROXY_URL"] == (
        "unix:///run/fable-vlm/proxy.sock"
    )
    assert "fable-vlm-runtime:/run/fable-vlm" in identity["volumes"]
    assert "fable-vlm-runtime:/run/fable-vlm" in proxy["volumes"]
    assert proxy["command"][-2:] == [
        "--unix-socket",
        "/run/fable-vlm/proxy.sock",
    ]


def test_mobile_nodes_are_audiovisual_and_do_not_assume_orin_identity(
    tmp_path: Path,
) -> None:
    module = _module()
    overlay = tmp_path / "compose.yaml"
    runtimes = tmp_path / "runtimes.yaml"
    deployment = tmp_path / "deployment.yaml"

    module.provider_overlay(
        ["orin2", "mobile_archive_1"],
        overlay,
    )
    module.runtimes(
        ["orin2", "mobile_archive_1"],
        runtimes,
    )
    module.deployment(
        ["orin2", "mobile_archive_1"],
        deployment,
    )

    services = yaml.safe_load(overlay.read_text())["services"]
    assert "fable-vehicle-mobile_archive_1" in services
    assert "fable-multimodal-mobile_archive_1" in services
    multimodal = services["fable-multimodal-mobile_archive_1"]
    assert multimodal["environment"]["AUDIO_CHANNEL_INDICES"] == "0"
    assert "mobile-mobile_archive_1" in multimodal["depends_on"]
    runtime_providers = yaml.safe_load(runtimes.read_text())["nodes"][
        "mobile_archive_1"
    ]["providers"]
    assert "person_vehicle_relation_provider" in runtime_providers
    sources = yaml.safe_load(deployment.read_text())["sources"]
    assert "mobile_archive_1_camera" in sources
    assert "mobile_archive_1_microphone" in sources


def test_appended_mobile_yolo_requests_gpu_runtime(tmp_path: Path) -> None:
    module = _module()
    compose = tmp_path / "compose.yaml"
    compose.write_text("services: {}\n", encoding="utf-8")
    recording = type(
        "Recording",
        (),
        {
            "logical_id": "mobile_archive_1",
            "segments": (),
            "timeline_start": None,
            "recording_start": module.datetime.fromisoformat(
                "2025-08-12T16:13:05+00:00"
            ),
            "trim_start_seconds": 0.0,
            "trim_end_seconds": 2.0,
            "path": tmp_path / "input.mp4",
        },
    )()

    module.add_mobile_replay_services(
        compose,
        [recording],
        scenario="scenario-a",
    )

    service = yaml.safe_load(compose.read_text())["services"][
        "yolo-mobile_archive_1"
    ]
    assert service["gpus"] == "all"
    assert service["environment"]["YOLO_DEVICE"] == "${YOLO_DEVICE:-auto}"


def test_generated_vehicle_geometry_covers_every_selected_camera_frame(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "fable_vehicle_geometry.json"

    module.vehicle_geometry(["orin1", "orin14"], output)

    zones = json.loads(output.read_text())["zones"]
    assert {zone["coordinate_frame_id"] for zone in zones} == {
        "image:dvpg_gq_orin_1",
        "image:dvpg_gq_orin_14",
    }
    assert json.loads(output.read_text())["references"] == []


def test_generated_mobile_geometry_uses_portrait_frame(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "fable_vehicle_geometry.json"

    module.vehicle_geometry(["mobile_archive_1"], output)

    polygon = json.loads(output.read_text())["zones"][0]["polygon"]
    assert max(point["x"] for point in polygon) == 720.0
    assert max(point["y"] for point in polygon) == 1280.0


def test_deployment_records_replay_event_time_coverage(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "deployment.yaml"

    module.deployment(
        ["orin2"],
        output,
        event_start="2024-10-08T11:08:23Z",
        event_end="2024-10-08T11:09:23Z",
    )

    source = yaml.safe_load(output.read_text())["sources"]["orin2_camera"]
    assert source["raw_buffer_interval"] == {
        "start": "2024-10-08T11:08:23Z",
        "end": "2024-10-08T11:09:23Z",
    }


def test_reid_bundle_uses_pinned_models_and_live_association_runtime(
    tmp_path: Path,
) -> None:
    module = _module()
    overlay = tmp_path / "compose.yaml"
    runtimes = tmp_path / "runtimes.yaml"

    module.provider_overlay(["orin2"], overlay, enable_reid=True)
    module.runtimes(["orin2"], runtimes, enable_reid=True)

    services = yaml.safe_load(overlay.read_text())["services"]
    environment = services["yolo-orin2"]["environment"]
    manifest = yaml.safe_load(module.REID_MANIFEST.read_text())["models"]
    assert environment["PERSON_REID_BACKEND"] == "torchreid"
    assert environment["PERSON_REID_MODEL_VERSION"] == manifest["person"]["version"]
    assert environment["VEHICLE_REID_BACKEND"] == "fastreid"
    assert environment["VEHICLE_REID_MODEL_VERSION"] == manifest["vehicle"]["version"]

    runtime = yaml.safe_load(runtimes.read_text())["nodes"]["x86server"]["providers"]
    assert runtime["cross_sensor_identity_association"]["mode"] == "ADOPT_EXISTING"
    sensor_runtime = yaml.safe_load(runtimes.read_text())["nodes"][
        "dvpg_gq_orin_2"
    ]["providers"]
    assert sensor_runtime["track_crop_extractor"]["mode"] == "ADOPT_EXISTING"
    assert sensor_runtime["track_crop_extractor"]["container_name"] == (
        "fable-vehicle-orin2"
    )
    assert sensor_runtime["track_crop_extractor"]["output_topics"] == [
        "/dvpg_gq_orin_2/fable/identity/bounded-crops"
    ]
    assert "vehicle_reid_descriptor" not in sensor_runtime
    assert runtime["vehicle_reid_descriptor"]["container_name"] == (
        "fable-reid-x86server"
    )
    assert runtime["vehicle_reid_descriptor"]["stop_adopted_when_idle"] is False
    assert runtime["cross_sensor_identity_association"][
        "stop_adopted_when_idle"
    ] is True


def test_site_edge_raw_offload_uses_executable_workers_and_preserves_source(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "runtimes.yaml"

    module.runtimes(["orin2"], output)

    providers = yaml.safe_load(output.read_text())["nodes"]["x86server"]["providers"]
    detector = providers["yolo_vehicle_fast_640"]
    assert detector["mode"] == "ADOPT_EXISTING"
    assert detector["container_name"] == "yolo-edge-orin2"
    assert detector["artifact_topic_outputs"] == {
        "detection_set.v1": "/x86server/from/dvpg_gq_orin_2/analytics/yolo/bbox"
    }

    passes = providers["pass_reference_evaluator"]
    assert passes["mode"] == "ADOPT_EXISTING"
    assert passes["container_name"] == "fable-vehicle-x86server-orin2"
    assert passes["artifact_topic_inputs"] == {
        "detection_set.v1": "/x86server/from/dvpg_gq_orin_2/analytics/yolo/bbox"
    }
    assert passes["output_topics"] == [
        "/x86server/from/dvpg_gq_orin_2/fable/vehicle/predicates"
    ]


def test_generated_deployment_metadata_is_available_to_trusted_edge(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "artifacts.yaml"

    module.artifacts(["orin2"], output)

    artifacts = yaml.safe_load(output.read_text())["artifacts"]
    by_type = {item["artifact_type"]: item for item in artifacts}
    expected = ["LOCAL", "REMOTE_REFERENCE", "TRANSFERRED"]
    assert by_type["camera_calibration.v1"]["access_modes"] == expected
    assert by_type["route_graph.v1"]["access_modes"] == expected
