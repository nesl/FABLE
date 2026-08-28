#!/usr/bin/env python3
"""Run the bounded, resumable E3/RQ2 full-stack matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog
from evaluation.experiments.matrix import build_run_matrix, write_planned_runs
from evaluation.experiments.specs import ExperimentQuestion
from fable.distributed.config import load_deployment_graph


PROFILE_BINDINGS = {
    "good_network": ("N0", "fable_deployment.good_network.yaml"),
    "constrained_bandwidth": ("W1", "fable_deployment.constrained_bandwidth.yaml"),
    "high_latency_cloud": ("W2", "fable_deployment.high_latency_cloud.yaml"),
}
CONFIG_ROOT = ROOT / "iobt-minimal-ce-replay/config"
HELPER = ROOT / "netwaggle/scripts/fable_netwaggle_helper.py"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_profile(profile_id: str, *, epoch: int, action: str = "APPLY") -> dict:
    condition, _ = PROFILE_BINDINGS[profile_id]
    completed = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--kind", "NETWORK_PROFILE",
            "--target", "site_to_cloud",
            "--condition", condition,
            "--action", action,
            "--condition-epoch", str(epoch),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid NetWaggle response: {completed.stdout[-1000:]}") from exc
    if completed.returncode or response.get("validated") is not True:
        raise RuntimeError(f"NetWaggle rejected {profile_id}: {response.get('reason')}")
    measurements = response.get("measurements", {})
    return {
        "condition": response.get("condition"),
        "profile": response.get("profile"),
        "validated": True,
        "reason": response.get("reason"),
        "configured_link_count": measurements.get("configured_link_count"),
        "qdisc_interface_count": measurements.get("qdisc_interface_count"),
        "qdisc_validated": measurements.get("qdisc_validated"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--ready-seconds", type=float, default=30)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv",
        transition_model_path=ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json",
    )
    runs = build_run_matrix(
        catalog,
        ExperimentQuestion.RQ2_PLANNING,
        repetitions=1,
        seed=args.seed,
        playback_mode="realtime",
    )
    manifest = write_planned_runs(runs, output / "run_matrix.jsonl")
    if len(runs) != 81:
        raise RuntimeError(f"bounded RQ2 matrix must contain 81 rows, found {len(runs)}")

    grouped = defaultdict(list)
    for run in runs:
        if run.network_profile_id not in PROFILE_BINDINGS:
            raise RuntimeError(f"unmapped profile: {run.network_profile_id}")
        grouped[(run.network_profile_id, run.baseline_id.value)].append(run)

    validation = {
        "schema_version": "fable.rq2_validation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "planned_runs": len(runs),
        "unique_traces": len({run.experiment_id for run in runs}),
        "groups": len(grouped),
        "planner_profiles": {},
        "netwaggle_profiles": {},
    }
    epoch = 100
    try:
        for profile_id, (condition, config_name) in PROFILE_BINDINGS.items():
            deployment_path = CONFIG_ROOT / config_name
            deployment = load_deployment_graph(deployment_path)
            sensor_path = deployment.shortest_path("dvpg_gq_orin_11", "x86server")
            cloud_path = deployment.shortest_path("x86server", "cloud1")
            validation["planner_profiles"][profile_id] = {
                "condition": condition,
                "path": str(deployment_path),
                "sensor_to_server_latency_ms": sensor_path.latency_ms,
                "sensor_to_server_bandwidth_mbps": sensor_path.bottleneck_bandwidth_mbps,
                "server_to_cloud_latency_ms": cloud_path.latency_ms,
                "server_to_cloud_bandwidth_mbps": cloud_path.bottleneck_bandwidth_mbps,
            }
            validation["netwaggle_profiles"][profile_id] = apply_profile(profile_id, epoch=epoch)
            epoch += 1
    finally:
        validation["restore"] = apply_profile("good_network", epoch=epoch, action="RESTORE")
    write_json(output / "validation.json", validation)
    if args.validation_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    campaign_log = output / "campaign-events.jsonl"
    failures = 0
    try:
        for profile_id in PROFILE_BINDINGS:
            condition, config_name = PROFILE_BINDINGS[profile_id]
            applied = apply_profile(profile_id, epoch=epoch)
            epoch += 1
            for baseline in (
                "B2_FRONTIER_FIXED_REALIZATION",
                "B4_GREEDY_FRONTIER",
                "FABLE",
            ):
                selected = grouped[(profile_id, baseline)]
                group_dir = output / profile_id / baseline / "repetition-01"
                command = [
                    sys.executable,
                    str(ROOT / "scripts/run_full_ce_suite.py"),
                    "--output-dir", str(group_dir),
                    "--baseline", baseline,
                    "--network-profile-id", profile_id,
                    "--deployment-config", str(CONFIG_ROOT / config_name),
                    "--max-seconds", str(args.max_seconds),
                    "--ready-seconds", str(args.ready_seconds),
                    "--playback-mode", "realtime",
                ]
                for run in selected:
                    command.extend(("--experiment-id", run.experiment_id))
                started = datetime.now(timezone.utc).isoformat()
                completed = subprocess.run(command, cwd=ROOT, check=False)
                event = {
                    "started_at": started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "network_profile_id": profile_id,
                    "condition": condition,
                    "netwaggle": applied,
                    "baseline_id": baseline,
                    "planned_run_ids": [run.run_id for run in selected],
                    "returncode": completed.returncode,
                    "output_dir": str(group_dir),
                }
                with campaign_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                failures += int(completed.returncode != 0)
    finally:
        restore = apply_profile("good_network", epoch=epoch, action="RESTORE")
        write_json(output / "final-network-restore.json", restore)

    write_json(
        output / "campaign-report.json",
        {
            "schema_version": "fable.bounded_rq2_campaign.v1",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "planned_runs": len(runs),
            "group_failures": failures,
            "network_restored_to": "N0",
            "manifest": str(manifest),
        },
    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
