#!/usr/bin/env python3
"""Create a durable pause/resume snapshot for an RQ3a campaign."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve(strict=True)
    campaign = args.campaign_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    planned = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    outcomes = []
    remaining = []
    counts: dict[str, Counter] = defaultdict(Counter)
    for run in planned:
        condition = Path(str(run["condition_trace_path"])).stem
        condition_cell = f"{condition}-offset-{run['ce_start_offset_seconds']:g}s"
        result_path = (
            campaign
            / "rq3a"
            / condition_cell
            / str(run["baseline_id"])
            / f"{run['experiment_id']}.json"
        )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = None
        complete = bool(result and result.get("suite")) and result.get(
            "classification"
        ) != "RUNTIME_FAILURE"
        classification = (
            str(result.get("classification") or "UNKNOWN")
            if complete
            else "NOT_COMPLETED"
        )
        if not complete:
            remaining.append(run)
        counts[str(run["baseline_id"])][classification] += 1
        outcomes.append(
            {
                "experiment_id": run["experiment_id"],
                "condition_trace_id": run["condition_trace_id"],
                "baseline_id": run["baseline_id"],
                "classification": classification,
                "result_path": str(result_path) if result_path.exists() else "",
            }
        )
    remaining_path = output / "remaining_runs.jsonl"
    remaining_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in remaining),
        encoding="utf-8",
    )
    with (output / "outcomes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    summary_rows = []
    for baseline in sorted(counts):
        summary_rows.append(
            {
                "baseline_id": baseline,
                "planned": sum(counts[baseline].values()),
                "true_positive": counts[baseline]["TRUE_POSITIVE"],
                "false_negative": counts[baseline]["FALSE_NEGATIVE"],
                "runtime_failure": counts[baseline]["RUNTIME_FAILURE"],
                "not_completed": counts[baseline]["NOT_COMPLETED"],
            }
        )
    with (output / "baseline_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    resume = (
        f"cd {shlex.quote(str(ROOT))} && "
        "tmux new-session -d -s rq3a-9ce-five-policy-pilot-resume \""
        ".venv/bin/python scripts/run_rq3_campaigns.py "
        f"--root '{campaign}' --rq3a-manifest '{remaining_path}' --only rq3a "
        "--max-seconds 300 --ready-seconds 30 "
        f"2>&1 | tee -a '{campaign / 'campaign-resume.log'}'\""
    )
    state = {
        "schema_version": "fable.rq3a_pause_snapshot.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "original_manifest": str(manifest),
        "campaign_dir": str(campaign),
        "planned_cells": len(planned),
        "complete_cells": len(planned) - len(remaining),
        "remaining_cells": len(remaining),
        "network_restored_to": "N0",
        "resume_command": resume,
        "baseline_summary": summary_rows,
    }
    (output / "resume_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "RESUME_COMMAND.txt").write_text(resume + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "RQ3a five-policy pilot paused\n"
        "===============================\n\n"
        f"Completed cells: {state['complete_cells']} / {state['planned_cells']}\n"
        f"Remaining cells: {state['remaining_cells']}\n"
        "NetWaggle was restored to N0 and the evaluation Compose stack was stopped.\n"
        "The experiment design is under reconsideration because historical plans rarely selected cloud providers, while W1 changes only the site-cloud link.\n"
        "Use RESUME_COMMAND.txt only if retaining this WAN treatment remains desirable.\n",
        encoding="utf-8",
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
