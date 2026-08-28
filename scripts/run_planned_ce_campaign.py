#!/usr/bin/env python3
"""Execute an immutable PlannedRun JSONL as resumable baseline suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.experiments.matrix import PlannedRun  # noqa: E402
from evaluation.baselines.static_registry import StaticPipelineRegistry  # noqa: E402
from scripts.derive_b1_trace_placement import install as install_b1_placement  # noqa: E402


STATIC_PIPELINES = ROOT / "evaluation/manifests/baselines/static_pipelines.yaml"

_EXECUTABLE_SOURCE_ROOTS = (
    ROOT / "fable",
    ROOT / "evaluation",
    ROOT / "scripts",
    ROOT / "providers",
    ROOT / "iobt-minimal-ce-replay/services",
    ROOT / "iobt-minimal-ce-replay/setup",
)


def executable_source_digest() -> str:
    """Hash code/config inputs so one matrix cannot mix implementations."""

    digest = hashlib.sha256()
    for root in _EXECUTABLE_SOURCE_ROOTS:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _condition_slug(run: PlannedRun) -> str:
    label = run.condition_trace_id or run.disturbance_profile_id or "nominal"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or "nominal"
    offset = f"{run.ce_start_offset_seconds:g}".replace(".", "p")
    return f"{safe}-offset-{offset}s"


def _condition_rank(
    condition_trace_id: str | None,
    *,
    disturbed_first: bool,
) -> int:
    """Return a stable within-trace rank without changing legacy defaults."""

    disturbed = bool(condition_trace_id)
    return int(not disturbed) if disturbed_first else int(disturbed)


def _calibration_rank(run: PlannedRun) -> tuple[int, str]:
    """Put nominal FABLE calibration before any B1 cell for a trace."""

    baseline = run.baseline_id.value
    nominal = not run.condition_trace_id
    if nominal and baseline == "FABLE":
        return (0, baseline)
    if baseline == "B1_STATIC_WHOLE_EVENT":
        # A disturbed B1 cell is only meaningful after this exact frozen
        # placement has passed its nominal in-campaign preflight.
        return (1 if nominal else 2, baseline)
    return (3 if not nominal else 4, baseline)


def _install_successful_nominal_fable(
    path: Path, registry_path: Path
) -> dict[str, object] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        result.get("baseline") != "FABLE"
        or result.get("classification") != "TRUE_POSITIVE"
        or (result.get("condition_trace") or {}).get("transitions")
        or not bool((result.get("execution_conformance") or {}).get("valid"))
    ):
        return None
    try:
        return install_b1_placement(path, registry_path)
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
        # A trace whose successful FABLE record cannot yield a coherent exact
        # placement makes only its paired B1 cells unavailable. It must not
        # terminate the remaining campaign.
        return None


def _has_exact_b1_trace_placement(
    experiment_id: str, registry_path: Path
) -> bool:
    """Require B1 to use a placement calibrated for this exact trace.

    CE-level placement templates are useful for small manual pilots, but are
    not a valid substitute for the nominal-FABLE calibration contract used by
    the paired disconnect campaign.
    """

    registry = StaticPipelineRegistry.load(registry_path)
    # PlannedRun identifies the catalog experiment, whereas the registry is
    # keyed by the recording trace timestamp (for example 20260414_152233).
    # Match the immutable experiment ID stored in the exact placement rather
    # than incorrectly treating the experiment ID as that dictionary key.
    return any(
        placement.experiment_id == experiment_id
        for placement in registry.trace_placements.values()
    )


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _binding_sensor_nodes(result: dict[str, object]) -> set[str]:
    nodes: set[str] = set()
    for prediction in result.get("predictions") or ():
        if prediction.get("accepted") is not True:
            continue
        for value in (prediction.get("bindings") or {}).values():
            text = str(value)
            if text.startswith("dvpg_gq_orin_") and ":" in text:
                nodes.add(text.split(":", 1)[0])
            elif text.startswith("mobile_archive_") and ":" in text:
                nodes.add(text.split(":", 1)[0])
    return nodes


def _sensor_switch(node_id: str) -> str:
    if node_id.startswith("dvpg_gq_orin_"):
        return "s_orin" + node_id.removeprefix("dvpg_gq_orin_")
    if node_id.startswith("mobile_archive_"):
        return "s_mob" + node_id.removeprefix("mobile_archive_")
    raise ValueError(f"accepted binding is not a sensor node: {node_id}")


def _materialize_trace_conditions(
    result_path: Path,
    *,
    runs: tuple[PlannedRun, ...],
    output: Path,
    topology_path: Path,
) -> dict[str, Path]:
    """Freeze one causal cut from nominal FABLE for every paired policy."""

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("classification") != "TRUE_POSITIVE":
        raise ValueError("causal calibration must be a nominal true positive")
    predictions = [
        item for item in result.get("predictions") or ()
        if item.get("accepted") is True
    ]
    if not predictions:
        raise ValueError("causal calibration has no accepted prediction")
    prediction = predictions[0]
    nodes = _binding_sensor_nodes(result)
    if len(nodes) != 1:
        raise ValueError(
            "causal calibration must identify exactly one accepted sensor; "
            f"found {sorted(nodes)}"
        )
    sensor_node = next(iter(nodes))
    switch_id = _sensor_switch(sensor_node)
    timing = result.get("timing") or {}
    observed_start = timing.get("observed_event_time_start")
    event_start = prediction.get("event_start_time") or prediction.get("event_time")
    if not observed_start or not event_start:
        raise ValueError("causal calibration lacks event-time alignment")
    offset = max(0.1, (_parse_time(str(event_start)) - _parse_time(str(observed_start))).total_seconds())

    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    links = {
        frozenset((str(item["from"]), str(item["to"])))
        for item in topology.get("links") or ()
    }
    if frozenset((switch_id, "s_edge")) not in links:
        raise ValueError(f"topology lacks calibrated link {switch_id}<->s_edge")

    condition_root = output / "calibrated-condition-traces" / str(result["experiment_id"])
    condition_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for run in runs:
        if run.experiment_id != result["experiment_id"] or not run.condition_trace_path:
            continue
        trace = json.loads(Path(run.condition_trace_path).read_text(encoding="utf-8"))
        for transition in trace.get("transitions") or ():
            if transition.get("action") in {"FAIL_LINK", "RESTORE_LINK"}:
                transition["target_id"] = f"link:{switch_id}:s_edge"
            if transition.get("action") == "FAIL_LINK":
                transition["offset_s"] = round(offset, 3)
        path = condition_root / f"{run.baseline_id.value}.json"
        path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        paths[run.run_id] = path
    provenance = {
        "schema_version": "fable.calibrated_causal_cut.v1",
        "experiment_id": result["experiment_id"],
        "source_result": str(result_path.resolve()),
        "accepted_sensor_node": sensor_node,
        "disconnect_link": f"link:{switch_id}:s_edge",
        "disconnect_offset_seconds": round(offset, 3),
        "accepted_event_start_time": event_start,
        "observed_event_time_start": observed_start,
        "same_cut_for_all_baselines": True,
    }
    (condition_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def _trace_has_network_mutation(run: PlannedRun) -> bool:
    if not run.condition_trace_path:
        return False
    trace = json.loads(Path(run.condition_trace_path).read_text(encoding="utf-8"))
    return any(
        str(item.get("action", "")).endswith("NETWORK_PROFILE")
        or str(item.get("action", "")) in {"FAIL_LINK", "RESTORE_LINK"}
        for item in trace.get("transitions", ())
    )


def _result_is_valid_for_run(path: Path, run: PlannedRun) -> bool:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not document.get("suite") or document.get("classification") == "RUNTIME_FAILURE":
        return False
    if run.condition_trace_path:
        trace = document.get("condition_trace") or {}
        if trace.get("trace_id") != run.condition_trace_id:
            return False
        expected = {
            item["transition_id"]
            for item in trace.get("transitions", ())
            if float(item.get("offset_s", 0)) <= float(document.get("elapsed_seconds", 0))
        }
        validated = set()
        for item in document.get("disturbance_results", ()):
            response = item.get("response") or {}
            # Provider/compute helpers expose an explicit boolean. NetWaggle's
            # typed response instead proves application with the requested
            # profile plus a monotonically assigned condition epoch.
            successful = item.get("validated") is True or (
                response.get("condition_epoch") is not None
                and bool(response.get("profile") or response.get("condition"))
            )
            if successful:
                validated.add(item.get("transition_id"))
        if not expected.issubset(validated):
            return False
    return True


def _b1_no_fanout_conformance(
    path: Path, registry_path: Path
) -> dict[str, object]:
    """Fail closed if B1 expands beyond its frozen authored placement."""

    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"applicable": True, "valid": False, "reason": str(exc)}
    record_dir = Path(str(result.get("common_record_dir") or ""))
    plans_path = record_dir / "plan_decision.jsonl"
    commands_path = record_dir / "provider_command.jsonl"
    try:
        plans = [
            json.loads(line)
            for line in plans_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        commands = [
            json.loads(line)
            for line in commands_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return {"applicable": True, "valid": False, "reason": str(exc)}
    registry = StaticPipelineRegistry.load(registry_path)
    placement = registry.get_placement(
        str(result.get("variant") or ""), trace_id=str(result.get("scenario") or "")
    )
    if placement is None:
        return {
            "applicable": True,
            "valid": False,
            "reason": "B1 has no authored placement for this trace/variant",
        }
    allowed = set(placement.allowed_node_ids)
    plan_nodes = {
        str(node)
        for plan in plans
        for node in plan.get("selected_node_ids", ())
    }
    command_nodes = {str(item.get("node_id")) for item in commands if item.get("node_id")}
    expansion_reasons = [
        str(plan.get("reason") or "")
        for plan in plans
        if "coverage expansion" in str(plan.get("reason") or "").lower()
    ]
    invalid_scopes = sorted(
        {
            str(plan.get("planning_scope"))
            for plan in plans
            if plan.get("planning_scope") not in {
                "HANDWRITTEN_STATIC_WHOLE_EVENT",
                "STATIC_LATE_BOUND_INSTANTIATION",
            }
        }
    )
    unfrozen = sum(plan.get("frozen") is not True for plan in plans)
    unexpected = sorted((plan_nodes | command_nodes) - allowed)
    valid = bool(plans) and not (expansion_reasons or invalid_scopes or unfrozen or unexpected)
    return {
        "schema_version": "fable.b1_no_fanout_conformance.v1",
        "applicable": True,
        "valid": valid,
        "authored_placement": placement.experiment_id,
        "allowed_node_ids": sorted(allowed),
        "selected_node_ids": sorted(plan_nodes),
        "command_node_ids": sorted(command_nodes),
        "unexpected_node_ids": unexpected,
        "coverage_expansion_reason_count": len(expansion_reasons),
        "invalid_planning_scopes": invalid_scopes,
        "unfrozen_plan_count": unfrozen,
        "plan_count": len(plans),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--ready-seconds", type=float, default=30)
    parser.add_argument("--mobile-root", type=Path, default=Path("/media/brianw/Extreme SSD3"))
    parser.add_argument("--netwaggle-topology", type=Path)
    parser.add_argument("--require-netwaggle-bindings", action="store_true")
    parser.add_argument(
        "--drop-offline-evidence",
        action="store_true",
        help=(
            "Pilot-only: drop disconnected MQTT evidence and require explicit "
            "bounded raw catch-up after source recovery."
        ),
    )
    parser.add_argument(
        "--close-live-evidence-at-replay-end",
        action="store_true",
        help="Reject stale ordinary provider results after natural replay EOF.",
    )
    parser.add_argument(
        "--allow-raw-to-trusted-site-edge",
        action="store_true",
        help=(
            "Apply one explicit raw-placement policy to every paired campaign "
            "cell; never infer it from condition-trace presence."
        ),
    )
    parser.add_argument(
        "--execution-order",
        choices=("baseline-major", "trace-major", "ce-round-robin"),
        default="baseline-major",
        help=(
            "Order independent cells; trace-major keeps paired policies "
            "adjacent, while ce-round-robin also alternates CE families."
        ),
    )
    parser.add_argument(
        "--condition-order",
        choices=("nominal-first", "disturbed-first"),
        default="nominal-first",
        help=(
            "Order paired conditions within a trace. The default preserves "
            "existing manifests; robustness campaigns may put the disturbed "
            "block first."
        ),
    )
    parser.add_argument(
        "--seed-static-pipeline-registry",
        type=Path,
        help=(
            "Seed the campaign-local frozen B1 registry from prior validated "
            "executions instead of the repository default."
        ),
    )
    parser.add_argument(
        "--use-precalibrated-disconnects",
        action="store_true",
        help=(
            "Run a disturbed-only manifest using its immutable condition "
            "traces and exact B1 placements already present in the seeded "
            "registry; do not require an in-campaign nominal replay."
        ),
    )
    parser.add_argument(
        "--use-precalibrated-pairing",
        action="store_true",
        help=(
            "Run nominal/disturbed pairs from immutable condition traces and "
            "an already validated exact B1 registry. This avoids an in-campaign "
            "FABLE calibration and permits the requested condition-first order."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Execute only this baseline ID from the manifest (repeatable).",
    )
    parser.add_argument(
        "--experiment-id",
        action="append",
        default=[],
        help="Execute only this experiment ID from the manifest (repeatable).",
    )
    args = parser.parse_args()
    manifest = args.manifest.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Never mutate the repository-wide authored baseline registry while a
    # campaign is running. Every child receives this campaign-local snapshot.
    registry_path = output / "calibration" / "static_pipelines.frozen.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if not registry_path.exists():
        source_registry = args.seed_static_pipeline_registry or STATIC_PIPELINES
        shutil.copy2(source_registry, registry_path)
    runs = tuple(
        PlannedRun.model_validate_json(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if args.baseline:
        selected = set(args.baseline)
        runs = tuple(run for run in runs if run.baseline_id.value in selected)
        if not runs:
            parser.error(
                "none of the requested baselines are present in the manifest: "
                + ", ".join(sorted(selected))
            )
    if args.experiment_id:
        selected_experiments = set(args.experiment_id)
        runs = tuple(
            run for run in runs if run.experiment_id in selected_experiments
        )
        if not runs:
            parser.error(
                "none of the requested experiments are present in the manifest: "
                + ", ".join(sorted(selected_experiments))
            )
    if any(_trace_has_network_mutation(run) for run in runs):
        if args.netwaggle_topology is None or not args.require_netwaggle_bindings:
            parser.error(
                "network condition traces require --netwaggle-topology and "
                "--require-netwaggle-bindings"
            )
    if args.use_precalibrated_disconnects and any(
        not run.condition_trace_path for run in runs
    ):
        parser.error(
            "--use-precalibrated-disconnects requires a disturbed-only manifest"
        )
    if args.use_precalibrated_disconnects and args.use_precalibrated_pairing:
        parser.error("choose only one precalibrated campaign mode")
    cells: list[PlannedRun] = []
    if args.execution_order in {"trace-major", "ce-round-robin"}:
        experiment_ids = sorted({run.experiment_id for run in runs})
        if args.execution_order == "ce-round-robin":
            from evaluation.catalog import ExperimentCatalog

            catalog = ExperimentCatalog.from_csv(
                ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
            )
            variant_by_experiment = {
                item.experiment_id: item.ce_variant for item in catalog.experiments
            }
            by_variant: dict[str, list[str]] = defaultdict(list)
            for experiment_id in experiment_ids:
                by_variant[variant_by_experiment[experiment_id]].append(experiment_id)
            for variant_experiments in by_variant.values():
                variant_experiments.sort()
            round_robin_experiments = []
            for trace_index in range(max(map(len, by_variant.values()))):
                for variant in sorted(by_variant):
                    variant_experiments = by_variant[variant]
                    if trace_index < len(variant_experiments):
                        round_robin_experiments.append(
                            variant_experiments[trace_index]
                        )
            experiment_ids = round_robin_experiments
        experiment_rank = {value: index for index, value in enumerate(experiment_ids)}
        if args.use_precalibrated_pairing:
            baseline_rank = {
                "B1_STATIC_WHOLE_EVENT": 0,
                "FABLE": 1,
            }
            cells = sorted(
                runs,
                key=lambda run: (
                    experiment_rank[run.experiment_id],
                    _condition_rank(
                        run.condition_trace_id,
                        disturbed_first=args.condition_order == "disturbed-first",
                    ),
                    baseline_rank.get(run.baseline_id.value, 2),
                    run.repetition,
                    run.run_id,
                ),
            )
        else:
            cells = sorted(
                runs,
                key=lambda run: (
                    experiment_rank[run.experiment_id],
                    _calibration_rank(run),
                    run.repetition,
                    run.run_id,
                ),
            )
    else:
        cells = sorted(
            runs,
            key=lambda run: (
                run.baseline_id.value,
                run.repetition,
                run.experiment_id,
                _condition_rank(
                    run.condition_trace_id,
                    disturbed_first=args.condition_order == "disturbed-first",
                ),
                run.run_id,
            ),
        )
    events = output / "campaign-events.jsonl"
    failures = 0
    source_digest = executable_source_digest()
    aborted_reason = ""
    consecutive_startup_failures = 0
    condition_overrides: dict[str, Path] = (
        {
            run.run_id: Path(str(run.condition_trace_path)).resolve()
            for run in runs
            if run.condition_trace_path
        }
        if args.use_precalibrated_disconnects or args.use_precalibrated_pairing
        else {}
    )
    calibration_errors: dict[str, str] = {}
    nominal_b1_validated: set[str] = (
        {
            run.experiment_id
            for run in runs
            if _has_exact_b1_trace_placement(run.experiment_id, registry_path)
        }
        if args.use_precalibrated_disconnects or args.use_precalibrated_pairing
        else set()
    )

    def calibrate(result_path: Path, experiment_id: str) -> dict[str, object] | None:
        placement = _install_successful_nominal_fable(result_path, registry_path)
        if placement is None:
            calibration_errors[experiment_id] = "nominal FABLE did not yield a B1 placement"
            return None
        # Exact B1 placement calibration is also required by nominal E1. A
        # causal network transition is a separate artifact and must only be
        # derived for a matrix that actually contains disturbed cells.
        if any(run.condition_trace_path for run in runs):
            if args.netwaggle_topology is None:
                calibration_errors[experiment_id] = (
                    "condition-trace calibration requires a NetWaggle topology"
                )
                return None
            try:
                condition_overrides.update(_materialize_trace_conditions(
                    result_path,
                    runs=runs,
                    output=output,
                    topology_path=args.netwaggle_topology.resolve(),
                ))
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                calibration_errors[experiment_id] = str(exc)
                return None
        calibration_errors.pop(experiment_id, None)
        return placement

    for run in cells:
        current_digest = executable_source_digest()
        if current_digest != source_digest:
            aborted_reason = (
                "executable source changed during campaign; refusing to mix "
                f"implementations ({source_digest} -> {current_digest})"
            )
            failures += 1
            break
        baseline = run.baseline_id.value
        repetition = run.repetition
        experiment_id = run.experiment_id
        stage_dir = (
            output / "rq3a" / _condition_slug(run) / baseline
            / f"repetition-{repetition:02d}"
        )
        result = stage_dir / f"{experiment_id}.json"
        if _result_is_valid_for_run(result, run):
            if (
                baseline == "FABLE"
                and not run.condition_trace_id
                and not args.use_precalibrated_pairing
                and not args.use_precalibrated_disconnects
            ):
                calibrate(result, experiment_id)
            if baseline == "B1_STATIC_WHOLE_EVENT" and not run.condition_trace_id:
                conformance = _b1_no_fanout_conformance(result, registry_path)
                if conformance.get("valid") and json.loads(
                    result.read_text(encoding="utf-8")
                ).get("classification") == "TRUE_POSITIVE":
                    nominal_b1_validated.add(experiment_id)
            continue
        if (
            baseline == "B1_STATIC_WHOLE_EVENT"
            and (
                not _has_exact_b1_trace_placement(experiment_id, registry_path)
                or experiment_id in calibration_errors
            )
        ):
            failures += 1
            with events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "baseline_id": baseline,
                    "repetition": repetition,
                    "experiment_ids": [experiment_id],
                    "run_id": run.run_id,
                    "condition_trace_id": run.condition_trace_id,
                    "condition_trace_path": run.condition_trace_path,
                    "result_path": str(result),
                    "result_valid": False,
                    "b1_no_fanout_conformance": {
                        "applicable": True,
                        "valid": False,
                        "reason": (
                            calibration_errors.get(experiment_id)
                            or "missing exact B1 trace placement; nominal FABLE "
                            "calibration did not produce a successful placement"
                        ),
                    },
                    "calibration_placement": None,
                    "started_at": datetime.now(UTC).isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "returncode": None,
                }, sort_keys=True) + "\n")
            continue
        if run.condition_trace_path and run.run_id not in condition_overrides:
            failures += 1
            with events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "baseline_id": baseline,
                    "repetition": repetition,
                    "experiment_ids": [experiment_id],
                    "run_id": run.run_id,
                    "condition_trace_id": run.condition_trace_id,
                    "condition_trace_path": run.condition_trace_path,
                    "result_path": str(result),
                    "result_valid": False,
                    "calibration_placement": None,
                    "calibration_error": calibration_errors.get(
                        experiment_id, "missing calibrated condition trace"
                    ),
                    "started_at": datetime.now(UTC).isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "returncode": None,
                }, sort_keys=True) + "\n")
            continue
        if (
            baseline == "B1_STATIC_WHOLE_EVENT"
            and bool(run.condition_trace_id)
            and experiment_id not in nominal_b1_validated
        ):
            failures += 1
            calibration_errors[experiment_id] = (
                "disturbed B1 blocked: exact frozen placement has not passed "
                "the nominal in-campaign preflight"
            )
            continue
        command = [
            sys.executable,
            str(ROOT / "scripts/run_full_ce_suite.py"),
            "--output-dir", str(stage_dir),
            "--baseline", baseline,
            "--max-seconds", str(args.max_seconds),
            "--ready-seconds", str(args.ready_seconds),
            "--playback-mode", "realtime",
            "--mobile-root", str(args.mobile_root),
            "--experiment-id", experiment_id,
        ]
        if run.condition_trace_path:
            command.extend((
                "--condition-trace", str(condition_overrides[run.run_id].resolve()),
                "--ce-start-offset-seconds", str(run.ce_start_offset_seconds),
            ))
        if args.netwaggle_topology is not None:
            command.extend(("--netwaggle-topology", str(args.netwaggle_topology.resolve())))
        if args.require_netwaggle_bindings:
            command.append("--require-netwaggle-bindings")
        if args.drop_offline_evidence:
            command.append("--drop-offline-evidence")
        if args.close_live_evidence_at_replay_end:
            command.append("--close-live-evidence-at-replay-end")
        if args.allow_raw_to_trusted_site_edge:
            command.append("--allow-raw-to-trusted-site-edge")
        started = datetime.now(UTC).isoformat()
        child_env = os.environ.copy()
        child_env["FABLE_STATIC_PIPELINE_REGISTRY"] = str(registry_path)
        completed = subprocess.run(command, cwd=ROOT, check=False, env=child_env)
        valid_result = _result_is_valid_for_run(result, run)
        startup_failure = completed.returncode != 0 and not result.is_file()
        consecutive_startup_failures = (
            consecutive_startup_failures + 1 if startup_failure else 0
        )
        calibration_placement = None
        if (
            valid_result
            and baseline == "FABLE"
            and not run.condition_trace_id
            and not args.use_precalibrated_pairing
            and not args.use_precalibrated_disconnects
        ):
            calibration_placement = calibrate(result, experiment_id)
        b1_conformance = None
        if baseline == "B1_STATIC_WHOLE_EVENT" and result.is_file():
            b1_conformance = _b1_no_fanout_conformance(result, registry_path)
            (stage_dir / f"{experiment_id}.b1-no-fanout.json").write_text(
                json.dumps(b1_conformance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            valid_result = valid_result and bool(b1_conformance["valid"])
            classification = json.loads(
                result.read_text(encoding="utf-8")
            ).get("classification")
            if not run.condition_trace_id:
                valid_result = valid_result and classification == "TRUE_POSITIVE"
            if not run.condition_trace_id and valid_result:
                nominal_b1_validated.add(experiment_id)
        failures += int(completed.returncode != 0 or not valid_result)
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "baseline_id": baseline,
                "repetition": repetition,
                "experiment_ids": [experiment_id],
                "run_id": run.run_id,
                "condition_trace_id": run.condition_trace_id,
                "condition_trace_path": run.condition_trace_path,
                "result_path": str(result),
                "result_valid": valid_result,
                "b1_no_fanout_conformance": b1_conformance,
                "calibration_placement": calibration_placement,
                "started_at": started,
                "finished_at": datetime.now(UTC).isoformat(),
                "returncode": completed.returncode,
            }, sort_keys=True) + "\n")
        if consecutive_startup_failures >= 3:
            aborted_reason = (
                "three consecutive child processes exited before writing a "
                "result; campaign stopped to prevent an invalid failure cascade"
            )
            break
    report = {
        "schema_version": "fable.planned_ce_campaign.v1",
        "manifest": str(manifest),
        "planned_runs": len(runs),
        "execution_order": args.execution_order,
        "condition_order": args.condition_order,
        "failed_suites": failures,
        "aborted": bool(aborted_reason),
        "aborted_reason": aborted_reason,
        "executable_source_sha256": source_digest,
        "frozen_static_pipeline_registry": str(registry_path),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (output / "campaign-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
