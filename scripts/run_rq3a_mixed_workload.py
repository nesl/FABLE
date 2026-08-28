#!/usr/bin/env python3
"""Execute the canonical mixed RQ3a schedule on one persistent monotonic clock.

This is the repeatable profile/replayed-output execution requested by section 8
of update_rq3a.txt.  It never substitutes independent per-episode stack runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.condition_trace import ConditionTrace  # noqa: E402
from evaluation.mixed_workload import MixedRequestWorkload  # noqa: E402


ADAPTIVE = {"B3_TASK_RESOURCE_ADAPTIVE", "FABLE"}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute_run(*, workload: MixedRequestWorkload, trace: ConditionTrace,
                policy: str, run_id: str, output: Path, clock_scale: float) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    try:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = {}
    if existing.get("status") == "complete":
        return existing

    scheduled: list[tuple[float, int, str, object]] = []
    for episode in workload.episodes:
        scheduled.append((episode.request_offset_s, 1, "REQUEST_ADMITTED", episode))
        scheduled.append((episode.end_offset_s, 3, "REQUEST_COMPLETED", episode))
    for transition in trace.transitions:
        scheduled.append((transition.offset_s, 2, "CONDITION_TRANSITION", transition))
        if transition.duration_s is not None:
            scheduled.append((transition.offset_s + transition.duration_s, 0,
                              "CONDITION_AUTO_RESTORE", transition))
    scheduled.sort(key=lambda item: (item[0], item[1]))
    active: dict[str, object] = {}
    rows = []
    adaptation_count = 0
    start = time.monotonic()
    for offset, _priority, kind, payload in scheduled:
        target = start + offset * clock_scale
        while clock_scale > 0:
            remaining = target - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.25))
        row: dict[str, object] = {
            "run_id": run_id, "policy_id": policy, "offset_s": offset,
            "event": kind, "wall_time": datetime.now(UTC).isoformat(),
            "active_request_ids": sorted(active),
        }
        if kind == "REQUEST_ADMITTED":
            episode = payload
            active[episode.episode_id] = episode
            row.update({"episode_id": episode.episode_id,
                        "experiment_id": episode.experiment_id,
                        "execution_mode": episode.execution_mode,
                        "action": "ADMISSION_PLAN"})
        elif kind == "REQUEST_COMPLETED":
            episode = payload
            active.pop(episode.episode_id, None)
            row.update({"episode_id": episode.episode_id,
                        "experiment_id": episode.experiment_id,
                        "action": "REPLAYED_OUTPUT_COMPLETE"})
        else:
            transition = payload
            action = transition.action.value if kind == "CONDITION_TRANSITION" else "AUTO_RESTORE"
            replanned = sorted(active) if policy in ADAPTIVE else []
            adaptation_count += len(replanned)
            row.update({"transition_id": transition.transition_id,
                        "condition_action": action,
                        "target_id": transition.target_id,
                        "profile_id": transition.profile_id,
                        "replanned_request_ids": replanned,
                        "fixed_request_ids": sorted(active) if policy not in ADAPTIVE else []})
        row["active_request_ids_after"] = sorted(active)
        rows.append(row)
        with (output / "timeline.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # Retain the authored recovery/observation tail even when the last request
    # or transition ends before the declared workload duration.
    final_offset = scheduled[-1][0] if scheduled else 0.0
    tail_seconds = max(0.0, workload.duration_s - final_offset)
    if clock_scale > 0 and tail_seconds:
        time.sleep(tail_seconds * clock_scale)

    result = {
        "schema_version": "fable.rq3a_mixed_execution.v1",
        "run_id": run_id, "policy_id": policy, "status": "complete",
        "workload_id": workload.workload_id, "condition_trace_id": trace.trace_id,
        "duration_s": workload.duration_s, "clock_scale": clock_scale,
        "recovery_tail_seconds": tail_seconds,
        "execution_semantics": "profile_driven_control_schedule",
        "raw_sensor_recordings_replayed": False,
        "provider_outputs_replayed": False,
        "publishable_ce_accuracy": False,
        "valid_for": [
            "request_overlap_schedule",
            "condition_overlap_schedule",
            "fixed_vs_adaptive_control_actions",
        ],
        "episodes_admitted": sum(row["event"] == "REQUEST_ADMITTED" for row in rows),
        "episodes_completed": sum(row["event"] == "REQUEST_COMPLETED" for row in rows),
        "condition_events": sum(row["event"].startswith("CONDITION_") for row in rows),
        "request_replans": adaptation_count,
        "maximum_concurrent_requests": max(len(row["active_request_ids_after"]) for row in rows),
        "wall_seconds": round(time.monotonic() - start, 3),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clock-scale", type=float, default=1.0,
                        help="wall seconds per workload second; 0 is deterministic validation")
    args = parser.parse_args()
    if args.clock_scale < 0 or args.clock_scale > 1:
        parser.error("--clock-scale must be between 0 and 1")
    matrix = [json.loads(line) for line in args.matrix.read_text().splitlines() if line.strip()]
    if not matrix:
        parser.error("matrix is empty")
    results = []
    for row in matrix:
        workload = MixedRequestWorkload.model_validate_json(
            (ROOT / row["workload_path"]).read_text(encoding="utf-8")
        )
        trace = ConditionTrace.model_validate_json(
            (ROOT / row["condition_trace_path"]).read_text(encoding="utf-8")
        )
        results.append(execute_run(
            workload=workload, trace=trace, policy=row["baseline_id"],
            run_id=row["run_id"], output=args.output_dir / row["baseline_id"],
            clock_scale=args.clock_scale,
        ))
    report = {"schema_version": "fable.rq3a_mixed_campaign.v1", "status": "complete",
              "runs": results, "completed_runs": len(results)}
    atomic_json(args.output_dir / "campaign-report.json", report)
    print(json.dumps({"status": "complete", "completed_runs": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
