"""Fail-closed structural coverage checks for the two primary RQ3a CE families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.models import RuntimeMode


RQ3A_REQUIREMENTS = {
    "pass_follow_clear_convoy": {
        "sensor_local": (
            ("dvpg_gq_orin_11", "yolo_vehicle_fast_640"),
            ("dvpg_gq_orin_11", "multi_object_tracker"),
            ("dvpg_gq_orin_11", "pass_reference_evaluator"),
        ),
        "site_edge": (
            ("x86server", "yolo_vehicle_fast_640"),
            ("x86server", "multi_object_tracker"),
            ("x86server", "pass_reference_evaluator"),
        ),
    },
    "multimodal_robbery": {
        "sensor_trigger": (
            ("dvpg_gq_orin_11", "audio_event_classifier"),
        ),
        "site_identity": (
            ("x86server", "cross_sensor_identity_association"),
        ),
        "cloud_escalation": (
            ("cloud1", "hosted_vlm_identity_comparator"),
        ),
    },
}


def validate_rq3a_provider_coverage(runtime_path: str | Path) -> dict[str, Any]:
    resolver = ProviderRuntimeResolver.from_yaml(runtime_path)
    families = []
    for family_id, alternatives in RQ3A_REQUIREMENTS.items():
        rows = []
        for alternative_id, requirements in alternatives.items():
            missing = []
            non_executable = []
            for node_id, provider_id in requirements:
                if not resolver.has(node_id, provider_id):
                    missing.append(f"{provider_id}@{node_id}")
                    continue
                runtime = resolver.resolve(node_id=node_id, provider_id=provider_id)
                if runtime.mode == RuntimeMode.REFERENCE:
                    non_executable.append(f"{provider_id}@{node_id}")
            rows.append(
                {
                    "alternative_id": alternative_id,
                    "valid": not missing and not non_executable,
                    "missing": missing,
                    "non_executable": non_executable,
                }
            )
        families.append(
            {
                "family_id": family_id,
                "valid": all(item["valid"] for item in rows),
                "alternatives": rows,
            }
        )
    return {
        "schema_version": "fable.rq3a_provider_coverage.v1",
        "valid": all(item["valid"] for item in families),
        "families": families,
    }
