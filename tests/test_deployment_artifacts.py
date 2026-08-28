from __future__ import annotations

import json
from datetime import datetime, timezone

import yaml

from evaluation.deployment_artifacts import load_deployment_artifacts
from fable.common.time import EventTimeInterval


def test_loaded_deployment_artifact_keeps_versioned_contract_type(tmp_path):
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({"schema_version": "geometry.v1"}), encoding="utf-8")
    manifest = tmp_path / "artifacts.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "artifacts": [
                    {
                        "artifact_type": "camera_calibration.v1",
                        "path": geometry.name,
                        "node_id": "camera-node",
                        "bindings": {"source_id": "camera"},
                        "access_modes": ["LOCAL"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    catalog = load_deployment_artifacts(manifest, repository_root=tmp_path)
    matches = catalog.query(
        artifact_type="camera_calibration.v1",
        event_time_interval=EventTimeInterval(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
    )

    assert len(matches) == 1
    assert matches[0].artifact_type == "camera_calibration.v1"
    assert matches[0].artifact_schema_version == "camera_calibration.v1"
