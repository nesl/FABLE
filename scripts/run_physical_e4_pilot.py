#!/usr/bin/env python3
"""Prepare or execute the matched nine-cell physical E4 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import re
from datetime import UTC, datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.condition_trace import ConditionTrace
from evaluation.cell_outcome import CellOutcome, OutcomeStatus, ScientificClassification
from scripts.derive_b1_trace_placement import install as install_b1

DEFAULT_SPEC = ROOT / "evaluation/manifests/adaptation/physical_e4_pilot.json"
DEFAULT_OUTPUT = Path(
    "/media/brianw/Extreme SSD2/fable_results/physical_e4_pilot_r013_20260820"
)
BASE_REGISTRY = ROOT / "evaluation/manifests/baselines/static_pipelines.yaml"
IDENTITY = Path("/tmp/fable_deploy_key")


def effective_hard_cell_timeout(
    requested_seconds: float,
    scientific_seconds: float,
    *,
    readiness_seconds: float = 60.0,
    child_cleanup_seconds: float = 30.0,
    outer_margin_seconds: float = 60.0,
) -> float:
    """Ensure the campaign watchdog cannot preempt scientific classification.

    ``run_full_ce_suite`` gives its replay worker a scientific window plus
    readiness and cleanup. The physical campaign adds stack/setup bookkeeping
    around that child. A watchdog shorter than this nested contract kills the
    worker before it can durably write FALSE_NEGATIVE.
    """
    required = (
        scientific_seconds
        + min(readiness_seconds, scientific_seconds)
        + child_cleanup_seconds
        + outer_margin_seconds
    )
    return max(float(requested_seconds), required)


def replay_source_configuration(spec: dict) -> tuple[list[str], str]:
    """Return the synchronized replay set and its one physical Pi source.

    Older manifests declared one ``replay_node``.  New manifests declare each
    synchronized camera explicitly and identify whether it remains on the
    desktop replay stack or is replaced by the physical Pi->Jetson path.
    Current hardware has one Pi stream slot, so accepting zero or multiple
    physical sources would misrepresent the execution topology.
    """

    configured = spec.get("replay_sources")
    if configured is None:
        node = str(spec.get("replay_node") or "").strip()
        if not node:
            raise ValueError("pilot must declare replay_node or replay_sources")
        return [node], node
    if not isinstance(configured, list) or not configured:
        raise ValueError("replay_sources must be a non-empty list")
    nodes: list[str] = []
    physical: list[str] = []
    for index, source in enumerate(configured):
        if not isinstance(source, dict):
            raise ValueError(f"replay_sources[{index}] must be an object")
        node = str(source.get("logical_replay_node") or "").strip()
        execution = str(source.get("execution") or "").strip()
        if not node or not re.fullmatch(r"(?:orin[0-9]+|mobile_archive_[0-9]+)", node):
            raise ValueError(
                "replay_sources["
                f"{index}].logical_replay_node must match orin<N> or mobile_archive_<N>"
            )
        if execution not in {"desktop", "physical_pi"}:
            raise ValueError(
                f"replay_sources[{index}].execution must be desktop or physical_pi"
            )
        if node in nodes:
            raise ValueError(f"duplicate synchronized replay node: {node}")
        nodes.append(node)
        if execution == "physical_pi":
            physical.append(node)
    if len(physical) != 1:
        raise ValueError(
            "current physical runner requires exactly one physical_pi replay source"
        )
    return nodes, physical[0]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_environment(spec: dict) -> dict[str, str]:
    allowed = {"FABLE_JOINT_RESOURCE_EPOCH_PLANNING"}
    configured = spec.get("environment") or {}
    if not isinstance(configured, dict):
        raise ValueError("environment must be an object")
    unknown = set(configured) - allowed
    if unknown:
        raise ValueError("unsupported experiment environment keys: " + ", ".join(sorted(unknown)))
    values = {str(key): str(value) for key, value in configured.items()}
    if any(value not in {"0", "1"} for value in values.values()):
        raise ValueError("experiment feature flags must be 0 or 1")
    return values


def run(argv: list[str], *, timeout: float, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout,
        check=False, env=env,
    )


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def physical_check(action: str, target: str) -> dict:
    completed = run(
        [
            sys.executable, str(ROOT / "scripts/physical_condition_control.py"),
            action, "--identity-file", str(IDENTITY), "--target", target,
            "--execute",
        ],
        timeout=30,
    )
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError:
        document = {"validated": False, "stderr": completed.stderr[-1000:]}
    document["returncode"] = completed.returncode
    return document


def prepare_registry(
    output: Path, calibration: Path | None, *, require_b1: bool
) -> tuple[Path, dict]:
    registry = output / "calibration/static_pipelines.physical-e4.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_REGISTRY, registry)
    placement = (
        install_b1(calibration, registry)  # type: ignore[arg-type]
        if require_b1
        else {
            "applicable": False,
            "reason": "pilot does not include B1_STATIC_WHOLE_EVENT",
            "fanout_allowed": False,
            "adaptation_allowed": False,
        }
    )
    return registry, placement


def result_valid(path: Path, *, baseline: str, condition_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        result = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    trace = result.get("condition_trace") or {}
    has_condition = bool(trace.get("transitions"))
    report_path = path.parent / "report.json"
    try:
        resources_released = load_json(report_path).get("resources_released") is True
    except (OSError, json.JSONDecodeError):
        resources_released = False
    disturbance_results = result.get("disturbance_results") or []
    condition_valid = condition_id == "nominal" or bool(
        len(disturbance_results) >= 2
        and all(
            row.get("response", {}).get("validated") is True
            and row.get("notification_validated") is True
            for row in disturbance_results[:2]
        )
    )
    return bool(
        result.get("baseline") == baseline
        # ``admitted`` means a matching seed reached the semantic runtime; it
        # is legitimately false for a scientifically valid false negative.
        # The request response itself must nevertheless have been accepted.
        # A clean teardown cannot turn a controller-rejected request into a
        # resumable completed cell.
        and any(
            row.get("accepted") is True
            for row in (result.get("seed_diagnostics") or ())
            if isinstance(row, dict)
        )
        and resources_released
        and has_condition == (condition_id != "nominal")
        and condition_valid
    )


def evaluate_adaptation_timelines(
    timelines: dict[tuple[str, str], list[dict]],
    *,
    adaptive_baselines: tuple[str, ...],
    disturbed_conditions: tuple[str, ...],
    required_pairs: tuple[tuple[str, str], ...] | None = None,
) -> dict:
    """Require a disturbance-triggered realization change for headline E4."""

    rows = []
    failures = []
    pairs = required_pairs or tuple(
        (baseline, condition)
        for baseline in adaptive_baselines
        for condition in disturbed_conditions
    )
    for baseline, condition in pairs:
        events = timelines.get((condition, baseline), [])
        initial = next(
            (row for row in events if row.get("event") == "PLAN_SELECTED"),
            None,
        )
        applied = [
            float(row.get("relative_seconds", 0.0))
            for row in events
            if row.get("event_kind") == "DISTURBANCE"
            and row.get("event") == "APPLY"
        ]
        apply_at = min(applied) if applied else None
        initial_signature = (
            str(initial.get("selected_providers") or "") if initial else ""
        )
        changed = [
            row
            for row in events
            if row.get("event_kind") == "PLAN"
            and row.get("event") == "PLAN_CHANGED"
            and apply_at is not None
            and float(row.get("relative_seconds", 0.0)) >= apply_at
            and str(row.get("selected_providers") or "")
            and str(row.get("selected_providers") or "") != initial_signature
        ]
        valid = bool(initial_signature and apply_at is not None and changed)
        rows.append({
            "baseline": baseline,
            "condition": condition,
            "initial_signature": initial_signature,
            "disturbance_applied_at_seconds": apply_at,
            "changed_plan_count": len(changed),
            "changed_signatures": sorted({
                str(row.get("selected_providers") or "") for row in changed
            }),
            "valid": valid,
        })
        if not valid:
            failures.append(
                f"{condition}/{baseline}: no distinct post-disturbance realization"
            )
    return {
        "schema_version": "fable.physical_e4_discrimination.v1",
        "valid": not failures,
        "failures": failures,
        "rows": rows,
    }


def load_execution_timelines(output: Path, spec: dict) -> dict:
    timelines = {}
    experiment_id = str(spec["experiment_id"])
    for condition in spec["conditions"]:
        condition_id = str(condition["condition_id"])
        for baseline in spec["baselines"]:
            path = (
                output / condition_id / baseline / "repetition-01"
                / f"{experiment_id}.records" / "execution_changes.jsonl"
            )
            rows = []
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            timelines[(condition_id, baseline)] = rows
    return timelines


def restore_physical_conditions() -> list[dict]:
    return [
        physical_check("clear-compute", "physical_jetson"),
        physical_check("restore-network", "rpi_to_jetson"),
    ]


def stop_physical_streams() -> list[dict]:
    """Close the two fixed remote workers so proxy byte records are durable."""

    commands = (
        ("rpi", "/home/rpi/project/FABLE/replay-cache/ffmpeg.pid"),
        ("jetson", "/home/nesl/FABLE/state/physical-yolo.pid"),
    )
    results = []
    for host, pidfile in commands:
        completed = run(
            [
                "ssh", "-i", str(IDENTITY), "-o", "BatchMode=yes", host,
                f"p=$(cat {pidfile} 2>/dev/null); "
                "test -n \"$p\" && kill \"$p\" 2>/dev/null || true",
            ],
            timeout=20,
        )
        results.append({
            "host": host,
            "pidfile": pidfile,
            "returncode": completed.returncode,
            "stderr": completed.stderr,
            "validated": completed.returncode == 0,
        })
    # CLOSED records are emitted asynchronously by both proxy pumps.
    time.sleep(2.0)
    return results


def main() -> int:
    global IDENTITY
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--identity-file", type=Path, default=IDENTITY)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hard-cell-timeout", type=float, default=390)
    args = parser.parse_args()

    IDENTITY = args.identity_file.resolve()
    spec_path = args.spec.resolve(strict=True)
    spec = load_json(spec_path)
    replay_nodes, physical_replay_node = replay_source_configuration(spec)
    feature_environment = experiment_environment(spec)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require_b1 = "B1_STATIC_WHOLE_EVENT" in spec["baselines"]
    calibration_value = spec.get("b1_calibration_result")
    calibration = (
        Path(calibration_value).resolve(strict=True)
        if calibration_value
        else None
    )
    if require_b1 and calibration is None:
        raise ValueError("B1 execution requires b1_calibration_result")

    for condition in spec["conditions"]:
        trace_path = condition.get("condition_trace")
        if trace_path:
            ConditionTrace.model_validate(load_json((ROOT / trace_path).resolve(strict=True)))
    registry, placement = prepare_registry(
        output, calibration, require_b1=require_b1
    )

    checks = {
        "identity_readable": IDENTITY.is_file() and os.access(IDENTITY, os.R_OK),
        "control_socket": Path("/run/netwaggle/fable-physical-control.sock").is_socket(),
        # rpi-video-egress listens inside the physical-RPi namespace at
        # 10.255.21.2:18091; the two required host-visible ingress sockets are
        # sufficient here. The completed-cell proxy validator proves all
        # three legs transferred bytes.
        "proxy_ports": {str(port): port_open(port) for port in (21883, 28091)},
        "jetson_compute": physical_check("check", "physical_jetson"),
        "rpi_network": physical_check("check", "rpi_to_jetson"),
        "b1_placement": placement,
        "calibration_sha256": (
            hashlib.sha256(calibration.read_bytes()).hexdigest()
            if calibration is not None else None
        ),
        "synchronized_replay_nodes": replay_nodes,
        "physical_replay_node": physical_replay_node,
    }
    checks["validated"] = bool(
        checks["identity_readable"]
        and checks["control_socket"]
        and all(checks["proxy_ports"].values())
        and checks["jetson_compute"].get("validated")
        and checks["rpi_network"].get("validated")
        and placement.get("fanout_allowed") is False
        and placement.get("adaptation_allowed") is False
    )
    (output / "preflight.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not checks["validated"]:
        print(json.dumps(checks, indent=2, sort_keys=True))
        return 2
    if not args.execute:
        print(json.dumps({
            "prepared": True,
            "execute": False,
            "expected_cells": spec["expected_cells"],
            "output": str(output),
            "registry": str(registry),
        }, indent=2))
        return 0
    if os.environ.get("FABLE_CONFIRM_EXPERIMENT") != "YES":
        parser.error("set FABLE_CONFIRM_EXPERIMENT=YES for physical mutation/execution")

    events = output / "campaign-events.jsonl"
    invalid_cells = 0
    infrastructure_failures = 0
    for condition in spec["conditions"]:
        condition_id = str(condition["condition_id"])
        for baseline in spec["baselines"]:
            stage = output / condition_id / baseline / "repetition-01"
            stage.mkdir(parents=True, exist_ok=True)
            result = stage / f"{spec['experiment_id']}.json"
            if result_valid(result, baseline=baseline, condition_id=condition_id):
                continue
            cleanup_before = restore_physical_conditions()
            proxy_since = time.time()
            command = [
                sys.executable, str(ROOT / "scripts/run_full_ce_suite.py"),
                "--experiment-id", spec["experiment_id"],
                "--output-dir", str(stage),
                "--baseline", baseline,
                "--playback-mode", "realtime",
                "--max-seconds", str(spec.get("max_seconds", 240)),
                "--ready-seconds", "60",
                "--required-ready-services",
                str(spec.get("required_ready_services", "zed,yolo")),
                "--maximum-replay-nodes", str(len(replay_nodes)),
                "--stage-physical-rpi", "--execute-physical-rpi",
                "--physical-rpi-replay-node", physical_replay_node,
                "--physical-rpi-host", "rpi", "--physical-jetson-host", "jetson",
                "--physical-rpi-identity-file", str(IDENTITY),
                "--physical-netwaggle-proxies",
                "--physical-compute-planner-node-id", spec["logical_physical_node_id"],
                "--physical-network-planner-node-id", spec["logical_physical_node_id"],
            ]
            concurrent_requests = int(spec.get("concurrent_requests", 1))
            if concurrent_requests != 1:
                command.extend(("--concurrent-requests", str(concurrent_requests)))
            if spec.get("allow_raw_to_trusted_site_edge") is True:
                command.append("--allow-raw-to-trusted-site-edge")
            for replay_node in replay_nodes:
                command.extend(("--replay-node", replay_node))
            trace_path = condition.get("condition_trace")
            if trace_path:
                command.extend(("--condition-trace", str((ROOT / trace_path).resolve())))
            env = os.environ.copy()
            env["FABLE_STATIC_PIPELINE_REGISTRY"] = str(registry)
            # The physical detector is started and readiness-checked before
            # admission. Preserve its resource claim but do not charge a
            # second synthetic cold start in the planner.
            env["FABLE_PROVIDER_STARTUP_OVERRIDES_JSON"] = json.dumps(
                {
                    "yolo_vehicle_fast_640@"
                    + str(spec["logical_physical_node_id"]): 0
                },
                separators=(",", ":"),
            )
            env.update(feature_environment)
            started = datetime.now(UTC).isoformat()
            timed_out = False
            cell_timeout = effective_hard_cell_timeout(
                args.hard_cell_timeout,
                float(spec.get("max_seconds", 240)),
            )
            try:
                completed = run(command, timeout=cell_timeout, env=env)
            except subprocess.TimeoutExpired:
                completed = None
                timed_out = True
            stream_cleanup = stop_physical_streams()
            cleanup_after = restore_physical_conditions()
            proxy_report = stage / "physical_proxy_validation.json"
            proxy_validation = run(
                [
                    sys.executable,
                    str(ROOT / "scripts/validate_netwaggle_external_proxy_path.py"),
                    "--topology", str(ROOT / "netwaggle/configs/physical_external_3node.json"),
                    "--metrics", str(ROOT / "runs/netwaggle/physical_proxy.jsonl"),
                    "--since-wall-time", str(proxy_since),
                    "--output", str(proxy_report),
                ],
                timeout=30,
            )
            valid = bool(
                result_valid(result, baseline=baseline, condition_id=condition_id)
                and proxy_validation.returncode == 0
            )
            try:
                result_document = load_json(result)
            except (OSError, json.JSONDecodeError):
                result_document = {}
            disturbance_rows = result_document.get("disturbance_results") or []
            protocol_ok = condition_id == "nominal" or all(
                item.get("notification_validated") is True
                for item in disturbance_rows[:2]
            )
            mutation_ok = condition_id == "nominal" or all(
                item.get("response", {}).get("validated") is True
                for item in disturbance_rows[:2]
            )
            adaptation_values = [
                str(item.get("notification_ack", {}).get("adaptation_status") or "")
                for item in disturbance_rows[:2]
            ]
            adaptation_status = (
                OutcomeStatus.NOT_APPLICABLE
                if condition_id == "nominal"
                else OutcomeStatus.INFEASIBLE
                if any(value == "INFEASIBLE" for value in adaptation_values)
                else OutcomeStatus.FAILED
                if any(value == "ERROR" for value in adaptation_values)
                else OutcomeStatus.SUCCEEDED
            )
            try:
                scientific = ScientificClassification(
                    str(result_document.get("classification") or "UNKNOWN")
                )
            except ValueError:
                scientific = ScientificClassification.UNKNOWN
            outcome = CellOutcome(
                infrastructure_status=(
                    OutcomeStatus.FAILED
                    if timed_out or completed is None
                    or (completed is not None and completed.returncode != 0)
                    or proxy_validation.returncode != 0
                    else OutcomeStatus.SUCCEEDED
                ),
                protocol_status=OutcomeStatus.VALID if protocol_ok else OutcomeStatus.INVALID,
                mutation_status=OutcomeStatus.SUCCEEDED if mutation_ok else OutcomeStatus.FAILED,
                adaptation_status=adaptation_status,
                measurement_status=OutcomeStatus.VALID if valid else OutcomeStatus.INVALID,
                scientific_classification=scientific,
                cleanup_status=(
                    OutcomeStatus.SUCCEEDED
                    if (result.parent / "report.json").is_file()
                    and load_json(result.parent / "report.json").get("resources_released") is True
                    else OutcomeStatus.UNKNOWN
                ),
            )
            invalid_cells += int(not valid)
            infrastructure_failures += int(
                timed_out
                or completed is None
                or (completed is not None and completed.returncode != 0)
                or proxy_validation.returncode != 0
            )
            row = {
                "baseline": baseline,
                "condition": condition_id,
                "started_at": started,
                "finished_at": datetime.now(UTC).isoformat(),
                "result": str(result),
                "valid": valid,
                "timed_out": timed_out,
                "effective_hard_cell_timeout_seconds": cell_timeout,
                "returncode": completed.returncode if completed else None,
                "cleanup_before": cleanup_before,
                "cleanup_after": cleanup_after,
                "stream_cleanup": stream_cleanup,
                "proxy_validation": str(proxy_report),
                "proxy_validation_returncode": proxy_validation.returncode,
                "outcome": outcome.model_dump(mode="json"),
            }
            with events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if completed:
                (stage / "campaign.stdout.log").write_text(completed.stdout, encoding="utf-8")
                (stage / "campaign.stderr.log").write_text(completed.stderr, encoding="utf-8")

    adaptive_baselines = tuple(spec.get("adaptive_baselines") or ())
    disturbed_conditions = tuple(
        str(item["condition_id"])
        for item in spec["conditions"]
        if item.get("condition_trace")
    )
    discrimination = evaluate_adaptation_timelines(
        load_execution_timelines(output, spec),
        adaptive_baselines=adaptive_baselines,
        disturbed_conditions=disturbed_conditions,
        required_pairs=tuple(
            (str(item["baseline"]), str(item["condition"]))
            for item in (spec.get("required_adaptation_pairs") or ())
        ) or None,
    )
    (output / "adaptation-discrimination.json").write_text(
        json.dumps(discrimination, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    require_discrimination = bool(spec.get("require_adaptation_discrimination"))
    report = {
        "schema_version": "fable.physical_e4_campaign_report.v1",
        "spec": str(spec_path),
        "expected_cells": spec["expected_cells"],
        "failed_cells": invalid_cells,
        "infrastructure_failures": infrastructure_failures,
        "adaptation_discrimination": discrimination,
        "headline_ready": bool(
            invalid_cells == 0
            and (discrimination["valid"] or not require_discrimination)
        ),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (output / "campaign-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return int(
        infrastructure_failures > 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
