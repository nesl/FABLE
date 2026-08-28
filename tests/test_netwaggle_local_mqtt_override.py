import json
from pathlib import Path

import yaml

from scripts.run_full_ce_suite import write_netwaggle_override


def test_sensor_services_use_loopback_broker_and_site_services_do_not(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    topology = tmp_path / "topology.json"
    topology.write_text(
        """{
          "logical_nodes": [
            {"name": "orin14", "anchor_container": "netwaggle-node-orin14",
             "containers": ["zed-replay-orin14", "fable-agent-orin14"]},
            {"name": "site-local", "anchor_container": "netwaggle-node-site-local",
             "containers": ["fable-agent-x86server"]}
          ]
        }""",
        encoding="utf-8",
    )
    (bundle / "compose.replay.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "zed-orin14": {"container_name": "zed-replay-orin14"},
                }
            }
        ),
        encoding="utf-8",
    )
    (bundle / "compose.fable.providers.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "agent-orin14": {"container_name": "fable-agent-orin14"},
                    "agent-site": {"container_name": "fable-agent-x86server"},
                    # A name suffix must not misplace this site-level service.
                    "yolo-edge-orin14": {"container_name": "yolo-edge-orin14"},
                }
            }
        ),
        encoding="utf-8",
    )

    bindings = write_netwaggle_override(bundle, topology)
    override = yaml.safe_load(
        (bundle / "compose.netwaggle.override.yaml").read_text(encoding="utf-8")
    )["services"]

    assert override["zed-orin14"]["environment"]["MQTT_HOST"] == "127.0.0.1"
    assert override["agent-orin14"]["environment"]["MQTT_HOST_IP"] == "127.0.0.1"
    assert override["agent-site"]["environment"]["MQTT_HOST"] == "10.255.0.1"
    assert "yolo-edge-orin14" not in override
    assert override["netwaggle-mqtt-orin14"]["network_mode"] == (
        "container:netwaggle-node-orin14"
    )
    assert bindings["netwaggle-mqtt-orin14"] == "netwaggle-node-orin14"
    config = (bundle / "netwaggle-local-mqtt/orin14.conf").read_text(
        encoding="utf-8"
    )
    assert "topic /dvpg_gq_orin_14/fable/vehicle/predicates out 1" in config
    assert "topic /dvpg_gq_orin_14/fable/interactions/predicates out 1" in config
    assert "topic /dvpg_gq_orin_14/fable/vehicle/tracks out 1" in config
    assert "topic /dvpg_gq_orin_14/fable/# out 1" not in config
    assert "topic /replay/config both 1" in config
    assert "topic /fable/v1/retrospective/# both 1" in config
    assert "remote_clientid netwaggle-control-orin14" in config
    assert "remote_clientid netwaggle-evidence-orin14" in config
    assert "topic /dvpg_gq_orin_14/analytics/yolo/bbox out 0" in config
    assert "camera" not in config


def test_sensor_bridge_can_drop_offline_evidence(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    topology = tmp_path / "topology.json"
    topology.write_text(
        json.dumps(
            {
                "logical_nodes": [
                    {
                        "name": "orin14",
                        "anchor_container": "netwaggle-node-orin14",
                        "containers": ["zed-replay-orin14"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (bundle / "compose.replay.yaml").write_text(
        yaml.safe_dump(
            {"services": {"zed-orin14": {"container_name": "zed-replay-orin14"}}}
        ),
        encoding="utf-8",
    )
    (bundle / "compose.fable.providers.yaml").write_text(
        yaml.safe_dump({"services": {}}), encoding="utf-8"
    )

    write_netwaggle_override(bundle, topology, drop_offline_evidence=True)

    config = (bundle / "netwaggle-local-mqtt/orin14.conf").read_text(
        encoding="utf-8"
    )
    assert "queue_qos0_messages false" in config
    assert "max_queued_messages 1" in config
    assert "max_queued_bytes 1" in config
    assert config.count("cleansession true") == 2
    assert config.count("restart_timeout 2 5") == 2
    assert "topic /dvpg_gq_orin_14/fable/vehicle/predicates out 0" in config
    assert "topic /dvpg_gq_orin_14/fable/vehicle/tracks out 0" in config
    assert "topic fable/v1/# both 0" in config
    assert "topic /fable/v1/retrospective/# both 0" in config
    assert "topic /replay/command/# both 0" in config
