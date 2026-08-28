from pathlib import Path

from evaluation.plan_discrimination import (
    load_plan_discrimination_manifest,
    load_synthetic_profiles,
    validate_expected_plans,
    validate_transition_behavior,
)


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_covers_every_profile_and_profiles_are_non_destructive() -> None:
    manifest = load_plan_discrimination_manifest(
        ROOT / "evaluation/manifests/adaptation/plan_discrimination.yaml"
    )
    source = ROOT / "evaluation/manifests/providers/calibrated_desktop_profiles.json"
    profiles = load_synthetic_profiles(source, manifest.timing_multipliers)
    sensor = next(
        item
        for item in profiles
        if item.provider_id == "yolo_vehicle_fast_640"
        and item.node_class == "sensor"
    )
    server = next(
        item
        for item in profiles
        if item.provider_id == "yolo_vehicle_fast_640"
        and item.node_class == "server"
    )
    assert sensor.execution_ms > server.execution_ms
    assert {item.profile_id for item in manifest.expected_plans} == {
        Path(item).stem for item in manifest.profiles
    }


def test_expected_plan_contract_rejects_insensitive_placements() -> None:
    manifest = load_plan_discrimination_manifest(
        ROOT / "evaluation/manifests/adaptation/plan_discrimination.yaml"
    )
    rows = [
        {
            "profile_id": profile_id,
            "policy_id": "FABLE",
            "detector_node_class": "server",
            "detector_node_id": "x86server",
            "transfer_bytes": 0,
        }
        for profile_id in ("good_network", "constrained_bandwidth")
    ]
    result = validate_expected_plans(rows, manifest)
    assert not result["valid"]
    assert "FABLE" not in result["profile_sensitive_policies"]


def test_transition_contract_distinguishes_fixed_and_adaptive_policies() -> None:
    rows = []
    for policy, placements in {
        "B2_FRONTIER_FIXED_REALIZATION": ("x86server", "x86server"),
        "B3_TASK_RESOURCE_ADAPTIVE": ("x86server", "orin11"),
        "FABLE": ("x86server", "orin11"),
    }.items():
        for epoch, node in enumerate(placements):
            rows.append(
                {
                    "policy_id": policy,
                    "resource_epoch": epoch,
                    "detector_node_id": node,
                    "detector_node_class": "server" if epoch == 0 else (
                        "server" if policy.startswith("B2_") else "sensor"
                    ),
                }
            )
    assert validate_transition_behavior(rows)["valid"]
