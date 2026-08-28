#!/usr/bin/env python3
"""Resumable unattended controller for bounded RQ3a/RQ3b/RQ3c campaigns."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.experiments.matrix import PlannedRun

DISTURBANCES = {
    "good_network": None,
    "constrained_bandwidth": "W1",
    "cloud_degraded": "W2",
    "lossy_edge": "L1",
}
SPATIAL_MODEL_PATH = (
    ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json"
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_manifest(path: Path) -> tuple[PlannedRun, ...]:
    return tuple(
        PlannedRun.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def restore_network(epoch: int) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "netwaggle/scripts/fable_netwaggle_helper.py"),
            "--kind",
            "NETWORK_PROFILE",
            "--target",
            "site_to_cloud",
            "--condition",
            "N0",
            "--action",
            "RESTORE",
            "--condition-epoch",
            str(epoch),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=75,
    )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError:
        response = {"validated": False, "reason": completed.stdout[-1000:]}
    return {"returncode": completed.returncode, "response": response}


def result_complete(path: Path) -> bool:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        # A runner envelope is not a completed experimental observation when
        # setup failed before replay/admission. Keep such cells resumable so a
        # corrected host/runtime preflight retries them instead of silently
        # poisoning the campaign matrix.
        return bool(document.get("suite")) and document.get("classification") != (
            "RUNTIME_FAILURE"
        )
    except (OSError, json.JSONDecodeError):
        return False


def disturbed_cell_preflight(
    root: Path, run: PlannedRun, policy_id: str
) -> dict[str, object]:
    """Validate a disturbed cell against its same-policy nominal control."""

    trace = json.loads(Path(run.condition_trace_path).read_text(encoding="utf-8"))
    transitions = trace.get("transitions", [])
    if not transitions:
        return {"valid": True, "applicable": False, "reason": "nominal control"}
    candidates = sorted(
        root.glob(
            f"rq3a/n0_*-offset-0s/{policy_id}/{run.experiment_id}.json"
        )
    )
    if len(candidates) != 1:
        return {
            "valid": False,
            "applicable": True,
            "reason": f"expected one same-policy nominal control; found {len(candidates)}",
        }
    nominal = json.loads(candidates[0].read_text(encoding="utf-8"))
    binding = nominal.get("suite", {}).get("netwaggle_binding_validation", {})
    target = str(transitions[0].get("target_id", ""))
    evidence = nominal.get("vehicle_predicates_by_source", {})
    if target.startswith("sensor_uplink:s_orin"):
        number = target.rsplit("s_orin", 1)[1]
        target_has_evidence = any(
            f"orin_{number}:" in str(source) and int(count) > 0
            for source, count in evidence.items()
        )
    elif target.startswith("sensor_uplink:s_mobile_archive_"):
        mobile = target.split("sensor_uplink:s_", 1)[1]
        target_has_evidence = any(
            mobile in str(source) and int(count) > 0
            for source, count in evidence.items()
        )
    else:
        observed = nominal.get("observed_messages", {})
        target_has_evidence = sum(
            int(value) for value in observed.values() if isinstance(value, int)
        ) > 0
    profile_changes_path = (
        transitions[0].get("profile_id") in {"L1", "N2"}
        and transitions[-1].get("profile_id") == "N0"
        and [item.get("offset_s") for item in transitions] == [30.0, 75.0]
    )
    valid = (
        nominal.get("classification") == "TRUE_POSITIVE"
        and bool(binding.get("valid"))
        and target_has_evidence
        and profile_changes_path
    )
    return {
        "schema_version": "fable.rq3a_disturbed_cell_preflight.v1",
        "applicable": True,
        "valid": valid,
        "nominal_result": str(candidates[0]),
        "nominal_classification": nominal.get("classification"),
        "target": target,
        "target_has_useful_evidence": target_has_evidence,
        "profile_changes_selected_path": profile_changes_path,
        "traffic_namespace_binding_valid": bool(binding.get("valid")),
        "fallback_available": True,
        "fallback_basis": (
            "all-campaign topology retains alternate sensor, site, and cloud placements"
        ),
    }


def run_group(command: list[str], timeout: float) -> tuple[int, bool]:
    process = subprocess.Popen(command, cwd=ROOT, text=True, start_new_session=True)
    try:
        return process.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        return 124, True
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        raise


def _binding_key(deployment_id: str, bindings: dict[str, object]) -> str | None:
    candidate = deployment_id
    while candidate:
        if candidate in bindings:
            return candidate
        candidate = candidate.rsplit("_", 1)[0] if "_" in candidate else ""
    return None


def _corridor_ids(value: object, known: set[str]) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, str):
        for token in value.split(";"):
            corridor_id = token.strip().split(":", 1)[0]
            if corridor_id in known:
                found.append(corridor_id)
    elif isinstance(value, list):
        for item in value:
            found.extend(_corridor_ids(item, known))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_corridor_ids(item, known))
    return tuple(dict.fromkeys(found))


def topology_shortlist(run: PlannedRun) -> tuple[str, ...]:
    """Return fixed replay nodes covered by the run's authored route model.

    This is deliberately derived from qualitative route bindings, never from
    downstream ground truth or the experiment's ``relevant_nodes`` label.
    """

    document = json.loads(SPATIAL_MODEL_PATH.read_text(encoding="utf-8"))
    corridors = document.get("corridors", {})
    bindings = document.get("scenario_route_bindings", {})
    corridor_ids: list[str] = []
    for deployment_id in run.topology_deployment_ids:
        key = _binding_key(deployment_id, bindings)
        if key is not None:
            corridor_ids.extend(_corridor_ids(bindings[key], set(corridors)))
    sensors: list[str] = []
    for corridor_id in dict.fromkeys(corridor_ids):
        corridor = corridors[corridor_id]
        for field in (
            "fixed_observation_groups_forward",
            "fixed_observation_groups_reverse",
        ):
            for group in corridor.get(field, ()):
                sensors.extend(group)
    available = set(run.replay_supported_sensor_ids)
    selected = tuple(
        sensor.replace("orin_", "orin")
        for sensor in dict.fromkeys(sensors)
        if sensor in available and sensor.startswith("orin_")
    )
    if not selected:
        raise ValueError(
            f"no topology shortlist for {run.experiment_id} "
            f"deployments={run.topology_deployment_ids!r}"
        )
    return selected


def spatial_execution_nodes(run: PlannedRun) -> tuple[str, ...]:
    policy_id = run.baseline_id.value
    if policy_id == "SPATIAL_BROADCAST":
        return ()  # no replay-node restriction means all configured nodes
    if policy_id == "SPATIAL_RESOURCE_ONLY":
        return tuple(
            sensor.replace("orin_", "orin")
            for sensor in sorted(run.replay_supported_sensor_ids)[:2]
        )
    shortlist = topology_shortlist(run)
    if policy_id == "SPATIAL_TOPOLOGY_SHORTLIST":
        return shortlist
    if policy_id == "SPATIAL_FABLE":
        return shortlist[:2]
    raise ValueError(f"unsupported spatial policy: {policy_id}")


def _runtime_sensor_id(sensor: str) -> str | None:
    """Translate authored Orin IDs to the replay runtime's canonical spelling."""

    match = re.fullmatch(r"orin[_-]?(\d+)", sensor.strip(), re.IGNORECASE)
    return f"orin{int(match.group(1))}" if match else None


def spatial_execution_candidates(run: PlannedRun) -> tuple[str, ...]:
    """Return ranked candidates, retaining alternatives for runtime availability.

    Planning operates on catalog capabilities, while the suite discovers which
    recordings are actually decodable only after starting a scenario.  Sending
    only the first two planned nodes made a valid policy look unavailable when
    either recording was absent.  The suite receives the full ranked list and
    applies the policy's activation limit after its readiness probe.
    """

    selected = spatial_execution_nodes(run)
    if run.baseline_id.value not in {"SPATIAL_RESOURCE_ONLY", "SPATIAL_FABLE"}:
        return selected
    available = tuple(
        dict.fromkeys(
            sensor
            for authored in sorted(run.replay_supported_sensor_ids)
            if (sensor := _runtime_sensor_id(authored)) is not None
        )
    )
    return tuple(dict.fromkeys((*selected, *available)))


def group_runs(name: str, runs: tuple[PlannedRun, ...]):
    groups: dict[
        tuple[str, str, tuple[str, ...], str, float], list[PlannedRun]
    ] = defaultdict(list)
    deferred = []
    for run in runs:
        if run.baseline_id.value == "SPATIAL_ORACLE":
            deferred.append(run)
            continue
        profile = run.network_profile_id if name == "rq3a" else "good_network"
        nodes = spatial_execution_candidates(run) if name == "rq3b" else ()
        groups[
            (
                profile,
                run.baseline_id.value,
                nodes,
                run.condition_trace_path if name == "rq3a" else "",
                run.ce_start_offset_seconds if name == "rq3a" else 0.0,
            )
        ].append(run)
    return groups, deferred


def execution_groups(name: str, runs: tuple[PlannedRun, ...]):
    """Run RQ3a network cells first, trace-major within each condition class.

    Keeping every policy adjacent for a trace makes partial campaign results
    directly comparable.  Compute contention is deliberately deferred until
    all network-disturbance cells finish so that its load container cannot
    contaminate the network-only measurements.
    """

    groups, deferred = group_runs(name, runs)
    if name != "rq3a":
        # E3 comparisons are trace-major: execute every online policy for one
        # trace before advancing to the next trace.  Do not batch a policy's
        # entire trace set into one stack invocation; that hid per-cell setup
        # failures and made partial campaign results baseline-major.
        ordered = []
        for run in runs:
            if run.baseline_id.value == "SPATIAL_ORACLE":
                continue
            nodes = spatial_execution_candidates(run) if name == "rq3b" else ()
            key = (
                "good_network",
                run.baseline_id.value,
                nodes,
                "",
                0.0,
            )
            ordered.append((key, [run]))
        return ordered, deferred
    def condition_rank(run: PlannedRun) -> int:
        condition = (
            f"{run.disturbance_profile_id} {run.condition_trace_id} "
            f"{Path(run.condition_trace_path).stem}"
        ).lower()
        return 1 if "compute" in condition else 0

    ordered = []
    # Python's sort is stable, so authored trace/baseline order is retained
    # within the network and compute partitions.
    for run in sorted(runs, key=condition_rank):
        profile = run.network_profile_id
        key = (
            profile,
            run.baseline_id.value,
            (),
            run.condition_trace_path,
            run.ce_start_offset_seconds,
        )
        ordered.append((key, [run]))
    return ordered, deferred


def summarize(
    root: Path,
    *,
    planned_runs: tuple[PlannedRun, ...],
    deferred: int,
) -> dict[str, object]:
    expected_cells = {
        (run.experiment_id, run.baseline_id.value)
        for run in planned_runs
        if run.baseline_id.value != "SPATIAL_ORACLE"
    }
    results = []
    for path in root.glob("rq3*/*/*/*.json"):
        if path.name in {"plan.json", "report.json"}:
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not row.get("experiment_id"):
            continue
        suite = row.get("suite", {})
        policy = str(suite.get("evaluation_policy_id") or "")
        if (str(row["experiment_id"]), policy) in expected_cells:
            results.append(row)
    continuation_counts = Counter()
    spatial_rows = []
    for row in results:
        record_dir = row.get("common_record_dir")
        records = Path(str(record_dir)) if record_dir else None
        if records is not None and records.is_dir():
            for record_type in (
                "artifact_event",
                "retrospective_attempt",
                "hypothesis_transition",
            ):
                path = records / f"{record_type}.jsonl"
                if path.is_file():
                    continuation_counts[record_type] += sum(
                        bool(line.strip())
                        for line in path.read_text(encoding="utf-8").splitlines()
                    )
        suite = row.get("suite", {})
        policy = str(suite.get("evaluation_policy_id") or "")
        if policy.startswith("SPATIAL_"):
            spatial_rows.append(
                {
                    "experiment_id": row.get("experiment_id"),
                    "policy_id": policy,
                    "activated_sensor_count": len(suite.get("replay_nodes", ())),
                    "detected": bool(row.get("detected")),
                    "classification": row.get("classification"),
                }
            )
    write_json(root / "rq3b-spatial-summary.json", spatial_rows)
    return {
        "schema_version": "fable.rq3_unattended_campaign.v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "planned_rows": len(planned_runs),
        "completed_result_files": len(results),
        "deferred_oracle_rows": deferred,
        "classification_counts": Counter(
            str(row.get("classification")) for row in results
        ),
        "continuation_record_counts": continuation_counts,
        "spatial_result_rows": len(spatial_rows),
        "resource_cleanup_confirmed": all(
            row.get("suite", {}).get("runner_returncode") is not None for row in results
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Fresh/resumable result root for this campaign execution.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Directory containing rq3a/b/c.jsonl (defaults to ROOT/manifests).",
    )
    parser.add_argument(
        "--rq3a-manifest",
        type=Path,
        help="Explicit evolving-condition RQ3a JSONL manifest.",
    )
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--ready-seconds", type=float, default=30)
    parser.add_argument("--only", choices=("rq3a", "rq3b", "rq3c"), action="append")
    parser.add_argument("--validation-one-cell", action="store_true")
    parser.add_argument(
        "--e3-pilot",
        action="store_true",
        help=(
            "Run one SPATIAL_FABLE cell and one retrospective trace under "
            "R0/R1/R2; intended only for the bounded E3 readiness pilot."
        ),
    )
    parser.add_argument(
        "--netwaggle-topology",
        type=Path,
        default=ROOT / "netwaggle/configs/site_local_20node.json",
    )
    parser.add_argument(
        "--require-netwaggle-bindings",
        action="store_true",
        help="Fail closed instead of allowing workloads to bypass NetWaggle.",
    )
    parser.add_argument(
        "--require-network-preflight",
        action="store_true",
        help="Skip disturbed cells that lack a valid same-policy nominal control.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate manifests, ordering, paths, and policy scopes without executing cells.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Regenerate aggregate reports from completed cells without executing any cell.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifests = (
        args.manifest_dir.resolve()
        if args.manifest_dir is not None
        else root / "manifests"
    )
    selected = tuple(args.only or ("rq3a", "rq3b", "rq3c"))
    all_runs = {
        name: load_manifest(
            args.rq3a_manifest.resolve()
            if name == "rq3a" and args.rq3a_manifest is not None
            else manifests / f"{name}.jsonl"
        )
        for name in selected
    }
    if args.preflight_only:
        seen_run_ids: set[str] = set()
        campaigns: dict[str, object] = {}
        for name in selected:
            runs = all_runs[name]
            duplicate_ids = sorted(
                run.run_id
                for run in runs
                if run.run_id in seen_run_ids
            )
            seen_run_ids.update(run.run_id for run in runs)
            groups, deferred = execution_groups(name, runs)
            campaigns[name] = {
                "rows": len(runs),
                "online_cells": len(groups),
                "deferred_oracle_rows": len(deferred),
                "duplicate_run_ids": duplicate_ids,
                "policies": sorted({run.baseline_id.value for run in runs}),
                "traces": len({run.experiment_id for run in runs}),
                "trace_major_single_cell_groups": all(len(rows) == 1 for _, rows in groups),
            }
        report = {
            "schema_version": "fable.e3_preflight.v1",
            "valid": all(
                not value["duplicate_run_ids"]
                and value["trace_major_single_cell_groups"]
                for value in campaigns.values()
            ),
            "campaigns": campaigns,
        }
        write_json(root / "preflight.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return int(not report["valid"])
    if args.summarize_only:
        planned_runs = tuple(run for name in selected for run in all_runs[name])
        deferred = sum(
            len(execution_groups(name, all_runs[name])[1]) for name in selected
        )
        report = summarize(root, planned_runs=planned_runs, deferred=deferred)
        report["group_failures"] = 0
        write_json(root / "campaign-report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    events = root / "campaign-events.jsonl"
    failures = 0
    deferred_count = 0
    epoch = 1000
    network_was_mutated = False
    try:
        for name in selected:
            groups, deferred = execution_groups(name, all_runs[name])
            deferred_count += len(deferred)
            if args.validation_one_cell:
                groups = groups[:1]
            if args.e3_pilot:
                if name == "rq3b":
                    groups = [
                        item for item in groups if item[0][1] == "SPATIAL_FABLE"
                    ][:1]
                elif name == "rq3c" and groups:
                    first_experiment = groups[0][1][0].experiment_id
                    groups = [
                        item
                        for item in groups
                        if item[1][0].experiment_id == first_experiment
                    ]
            for (
                profile,
                policy_id,
                replay_nodes,
                condition_trace_path,
                ce_start_offset,
            ), rows in groups:
                condition_cell = (
                    f"{Path(condition_trace_path).stem}-offset-{ce_start_offset:g}s"
                    if condition_trace_path
                    else profile
                )
                group_dir = root / name / condition_cell / policy_id
                pending = [
                    row
                    for row in rows
                    if not result_complete(group_dir / f"{row.experiment_id}.json")
                ]
                if not pending:
                    continue
                if args.require_network_preflight and condition_trace_path:
                    preflight = disturbed_cell_preflight(root, pending[0], policy_id)
                    preflight_path = group_dir / "preflight.json"
                    write_json(preflight_path, preflight)
                    if not preflight["valid"]:
                        event = {
                            "campaign": name,
                            "profile": profile,
                            "evaluation_policy_id": policy_id,
                            "planned_run_ids": [row.run_id for row in pending],
                            "status": "SKIPPED_PREFLIGHT",
                            "preflight": preflight,
                            "output_dir": str(group_dir),
                        }
                        with events.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(event, sort_keys=True) + "\n")
                        failures += 1
                        continue
                execution_baseline = (
                    "FABLE" if policy_id.startswith("SPATIAL_") else policy_id
                )
                if execution_baseline == "B1_HANDWRITTEN_STATIC":
                    execution_baseline = "B1_STATIC_WHOLE_EVENT"
                # E3 retrospective cells isolate evidence-recovery behavior:
                # every cell uses the same FABLE planner/runtime and differs
                # only in the explicitly frozen retrospective policy.
                if name == "rq3c" and pending[0].retrospective_policy_id:
                    execution_baseline = "FABLE"
                command = [
                    sys.executable,
                    str(ROOT / "scripts/run_full_ce_suite.py"),
                    "--output-dir",
                    str(group_dir),
                    "--baseline",
                    execution_baseline,
                    "--evaluation-policy-id",
                    policy_id,
                    "--network-profile-id",
                    profile,
                    "--deployment-config",
                    str(
                        ROOT
                        / "iobt-minimal-ce-replay/config/fable_deployment.good_network.yaml"
                    ),
                    "--playback-mode",
                    "realtime",
                    "--max-seconds",
                    str(args.max_seconds),
                    "--ready-seconds",
                    str(args.ready_seconds),
                ]
                disturbance = (
                    DISTURBANCES.get(profile)
                    if name == "rq3a" and not condition_trace_path
                    else None
                )
                if condition_trace_path:
                    network_was_mutated = True
                    command.extend(
                        (
                            "--condition-trace",
                            condition_trace_path,
                            "--ce-start-offset-seconds",
                            str(ce_start_offset),
                            "--netwaggle-topology",
                            str(args.netwaggle_topology.resolve()),
                        )
                    )
                    if args.require_netwaggle_bindings:
                        command.append("--require-netwaggle-bindings")
                elif disturbance:
                    network_was_mutated = True
                    command.extend(("--network-disturbance", disturbance))
                if name == "rq3b":
                    for node in replay_nodes:
                        command.extend(("--replay-node", node))
                    if policy_id in {"SPATIAL_RESOURCE_ONLY", "SPATIAL_FABLE"}:
                        command.extend(("--maximum-replay-nodes", "2"))
                if name == "rq3c" and pending[0].retrospective_policy_id:
                    command.extend(
                        (
                            "--retrospective-policy-id",
                            pending[0].retrospective_policy_id,
                        )
                    )
                for row in pending:
                    command.extend(("--experiment-id", row.experiment_id))
                started = datetime.now(UTC).isoformat()
                returncode, timed_out = run_group(
                    command,
                    max(900, len(pending) * (args.max_seconds + 240)),
                )
                if condition_trace_path or disturbance:
                    epoch += 1
                    restoration = restore_network(epoch)
                else:
                    restoration = {
                        "returncode": 0,
                        "skipped": True,
                        "reason": "cell did not mutate network state",
                    }
                event = {
                    "campaign": name,
                    "profile": profile,
                    "evaluation_policy_id": policy_id,
                    "execution_baseline_id": execution_baseline,
                    "replay_nodes": replay_nodes,
                    "condition_trace_path": condition_trace_path or None,
                    "ce_start_offset_seconds": ce_start_offset,
                    "planned_run_ids": [row.run_id for row in pending],
                    "started_at": started,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "restoration": restoration,
                    "output_dir": str(group_dir),
                }
                with events.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                failures += int(returncode != 0 or restoration["returncode"] != 0)
    finally:
        if network_was_mutated:
            epoch += 1
            final_restoration = restore_network(epoch)
        else:
            final_restoration = {
                "returncode": 0,
                "skipped": True,
                "reason": "campaign did not mutate network state",
            }
        write_json(root / "final-network-restore.json", final_restoration)
        report = summarize(
            root,
            planned_runs=tuple(
                run for name in selected for run in all_runs[name]
            ),
            deferred=deferred_count,
        )
        report["group_failures"] = failures
        write_json(root / "campaign-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
