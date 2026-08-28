#!/usr/bin/env python3
"""Run the canonical W1+E1 dynamic schedule without privileged host mutation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.adaptation_controller import (
    AdaptationControlPolicy,
    AdaptationController,
    AdaptationControllerMode,
    AllowedDisturbanceTarget,
    CompositeProfileApplier,
    DynamicAdaptationRun,
)
from evaluation.capacity_profiles import (
    ComputeCapacityProfile,
    ProfiledCapacityActionApplier,
)
from evaluation.disturbance_schedule import (
    DisturbanceKind,
    DisturbanceSchedule,
    DisturbanceScheduleController,
    DisturbanceStep,
    DisturbanceTrigger,
)
from evaluation.networking import (
    ProfiledNetworkActionApplier,
    load_netwaggle_profile,
)
from evaluation.runner import JsonlEventStore
from evaluation.schemas import BaselineId
from fable.distributed.config import load_deployment_graph


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    store = JsonlEventStore(args.output_dir)
    deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    identity = {
        "run_id": "milestone3-profiled",
        "baseline_id": BaselineId.FABLE,
        "trace_id": "synthetic-adaptation-validation",
        "request_id": "milestone3-profiled-request",
    }
    network = ProfiledNetworkActionApplier(
        deployment=deployment,
        profiles={
            "N0": load_netwaggle_profile(
                ROOT / "netwaggle/configs/profiles/good_network.json"
            ),
            "W1": load_netwaggle_profile(
                ROOT / "netwaggle/configs/profiles/cloud_degraded.json"
            ),
        },
        record_sink=store.append,
        **identity,
    )
    capacity = ProfiledCapacityActionApplier(
        deployment=deployment,
        profiles={
            "N0": ComputeCapacityProfile(
                profile_id="N0",
                cpu_capacity_fraction=1,
                memory_capacity_fraction=1,
                gpu_capacity_fraction=1,
            ),
            "E1": ComputeCapacityProfile(
                profile_id="E1",
                cpu_capacity_fraction=3 / 8,
                memory_capacity_fraction=0.75,
                gpu_capacity_fraction=0.5,
                execution_time_multiplier=2.25,
                queue_delay_ms=150,
            ),
        },
        record_sink=store.append,
        **identity,
    )
    policy = AdaptationControlPolicy(
        policy_id="milestone3-profiled-w1-e1",
        mode=AdaptationControllerMode.PROFILED,
        targets=(
            AllowedDisturbanceTarget(
                target_id="site_to_cloud",
                kinds=(DisturbanceKind.NETWORK_PROFILE,),
                condition_ids=("N0", "W1"),
            ),
            AllowedDisturbanceTarget(
                target_id="x86server",
                kinds=(DisturbanceKind.CAPACITY_PROFILE,),
                condition_ids=("N0", "E1"),
            ),
        ),
    )
    schedule = DisturbanceScheduleController(
        DisturbanceSchedule(
            schedule_id="milestone3-w1-e1-after-plan",
            steps=(
                DisturbanceStep(
                    step_id="wan",
                    trigger=DisturbanceTrigger.AFTER_PLAN_DISPATCH,
                    kind=DisturbanceKind.NETWORK_PROFILE,
                    target_id="site_to_cloud",
                    condition_id="W1",
                    delay_ms=1_000,
                    duration_ms=20_000,
                    restore_condition_id="N0",
                ),
                DisturbanceStep(
                    step_id="compute",
                    trigger=DisturbanceTrigger.AFTER_PLAN_DISPATCH,
                    kind=DisturbanceKind.CAPACITY_PROFILE,
                    target_id="x86server",
                    condition_id="E1",
                    delay_ms=1_000,
                    duration_ms=20_000,
                    restore_condition_id="N0",
                ),
            ),
            post_restore_observation_ms=10_000,
        )
    )
    run = DynamicAdaptationRun(
        schedule=schedule,
        controller=AdaptationController(
            policy,
            profile_applier=CompositeProfileApplier(
                {
                    DisturbanceKind.NETWORK_PROFILE: network,
                    DisturbanceKind.CAPACITY_PROFILE: capacity,
                }
            ),
        ),
        record_sink=store.append,
        **identity,
    )
    anchor = datetime(2026, 1, 1, tzinfo=UTC)
    run.observe(DisturbanceTrigger.AFTER_PLAN_DISPATCH, observed_at=anchor)
    applied = run.advance(now=anchor + timedelta(seconds=1))
    restored = run.advance(now=anchor + timedelta(seconds=21))
    final_observation_at = anchor + timedelta(seconds=31)

    disturbance_rows = store.read("disturbance_event")
    network_rows = store.read("network_condition")
    resource_rows = store.read("resource_sample")
    summary = {
        "schema_version": "fable.profiled_adaptation_validation.v1",
        "controller_mode": policy.mode.value,
        "host_mutation": False,
        "schedule_complete": schedule.complete,
        "applied_actions": len(applied),
        "restored_actions": len(restored),
        "condition_epochs": [
            row["condition_epoch"] for row in disturbance_rows
        ],
        "disturbance_records": len(disturbance_rows),
        "network_condition_records": len(network_rows),
        "resource_sample_records": len(resource_rows),
        "all_actions_validated": all(
            row["validated"] for row in disturbance_rows
        ),
        "final_observation_at": final_observation_at.isoformat(),
        "network_restored_to": (
            network.latest.profile_id if network.latest is not None else None
        ),
        "capacity_restored_to": (
            capacity.latest.profile_id if capacity.latest is not None else None
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(
        (
            schedule.complete,
            len(applied) == 2,
            len(restored) == 2,
            summary["all_actions_validated"],
            summary["network_restored_to"] == "good_network",
            summary["capacity_restored_to"] == "N0",
        )
    ) else 4


if __name__ == "__main__":
    raise SystemExit(main())
