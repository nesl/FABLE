#!/usr/bin/env python3
"""Create a compact, auditable summary of a paired B1/FABLE campaign."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


def _split(value: str | None) -> set[str]:
    return {item for item in (value or "").split(";") if item}


def _classification(document: dict, valid: bool, returncode: int) -> str:
    value = str(document.get("classification") or "")
    if value:
        return value
    if not valid or returncode:
        return "INFRASTRUCTURE_FAILURE"
    return "MISSING_RESULT"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    events = []
    for line in (campaign / "campaign-events.jsonl").read_text().splitlines():
        event = json.loads(line)
        result_path = Path(event["result_path"])
        document = json.loads(result_path.read_text()) if result_path.is_file() else {}
        condition = "DISCONNECT" if event.get("condition_trace_id") else "NOMINAL"
        events.append((event, document, condition))

    outcome_rows = []
    grouped: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for event, document, condition in events:
        outcome = _classification(
            document,
            bool(event.get("result_valid")),
            int(event.get("returncode") or 0),
        )
        baseline = str(event["baseline_id"])
        grouped[(baseline, condition)][outcome] += 1
        outcome_rows.append(
            {
                "experiment_id": event["experiment_ids"][0],
                "baseline": baseline,
                "condition": condition,
                "classification": outcome,
                "admitted": document.get("admitted", ""),
                "terminal": document.get("terminal", ""),
                "elapsed_seconds": document.get("elapsed_seconds", ""),
                "error": document.get("error", ""),
                "result_valid": event.get("result_valid", False),
                "result_path": event["result_path"],
            }
        )

    with (output / "run_outcomes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outcome_rows[0]))
        writer.writeheader()
        writer.writerows(outcome_rows)

    summary_rows = []
    for (baseline, condition), counts in sorted(grouped.items()):
        evaluated = sum(counts.values())
        tp = counts["TRUE_POSITIVE"]
        summary_rows.append(
            {
                "baseline": baseline,
                "condition": condition,
                "runs": evaluated,
                "true_positive": tp,
                "false_negative": counts["FALSE_NEGATIVE"],
                "infrastructure_failure": counts["INFRASTRUCTURE_FAILURE"],
                "missing_result": counts["MISSING_RESULT"],
                "observed_true_positive_rate": round(tp / evaluated, 6) if evaluated else "",
            }
        )
    with (output / "accuracy_by_baseline_condition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    adaptation_rows = []
    timeline_index_rows = []
    timeline_output = output / "fable_disconnect_timelines"
    timeline_output.mkdir(exist_ok=True)
    for event, document, condition in events:
        if event["baseline_id"] != "FABLE" or condition != "DISCONNECT":
            continue
        disturbances = [
            item for item in document.get("disturbance_results", [])
            if item.get("action") == "FAIL"
        ]
        fail_at = (
            float(disturbances[0].get("applied_offset_s", 0.0))
            if disturbances else None
        )
        timeline_path = Path(
            (document.get("execution_change_timeline") or {}).get("csv", "")
        )
        timeline_jsonl_path = Path(
            (document.get("execution_change_timeline") or {}).get("jsonl", "")
        )
        rows = []
        if timeline_path.is_file():
            with timeline_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        plans = [item for item in rows if item.get("event_kind") == "PLAN"]
        pre = [item for item in plans if fail_at is not None and float(item["relative_seconds"]) < fail_at]
        post = [item for item in plans if fail_at is not None and float(item["relative_seconds"]) >= fail_at]
        before = pre[-1] if pre else {}
        after = post[0] if post else {}
        before_nodes = _split(before.get("selected_nodes"))
        after_nodes = _split(after.get("selected_nodes"))
        before_providers = _split(before.get("selected_providers"))
        after_providers = _split(after.get("selected_providers"))
        instrumentation = (document.get("adaptation_instrumentation") or [{}])[0]
        disconnect_target = (
            disturbances[0].get("response", {})
            .get("measurements", {})
            .get("target", "")
            if disturbances else ""
        )
        target_token = disconnect_target.split(":")[1] if ":" in disconnect_target else ""
        if target_token.startswith("s_orin"):
            target_node = "dvpg_gq_orin_" + target_token.removeprefix("s_orin")
        elif target_token.startswith("s_mobile_archive_"):
            target_node = target_token.removeprefix("s_")
        else:
            target_node = target_token.removeprefix("s_")
        experiment_id = event["experiment_ids"][0]
        copied_csv = timeline_output / f"{experiment_id}.execution_changes.csv"
        copied_jsonl = timeline_output / f"{experiment_id}.execution_changes.jsonl"
        if timeline_path.is_file():
            shutil.copy2(timeline_path, copied_csv)
        if timeline_jsonl_path.is_file():
            shutil.copy2(timeline_jsonl_path, copied_jsonl)
        timeline_index_rows.append(
            {
                "experiment_id": experiment_id,
                "classification": document.get("classification", ""),
                "disconnect_target": disconnect_target,
                "timeline_csv": str(copied_csv.relative_to(output)) if copied_csv.is_file() else "",
                "timeline_jsonl": str(copied_jsonl.relative_to(output)) if copied_jsonl.is_file() else "",
                "source_timeline_csv": str(timeline_path),
                "source_timeline_jsonl": str(timeline_jsonl_path),
            }
        )
        adaptation_rows.append(
            {
                "experiment_id": experiment_id,
                "classification": document.get("classification", ""),
                "disconnect_target": disconnect_target,
                "disconnect_target_node": target_node,
                "disconnect_applied_seconds": fail_at if fail_at is not None else "",
                "first_post_disconnect_plan_seconds": after.get("relative_seconds", ""),
                "condition_to_first_plan_seconds": instrumentation.get("condition_to_first_plan_seconds", ""),
                "condition_to_first_predicate_output_seconds": instrumentation.get("condition_to_first_predicate_output_seconds", ""),
                "plan_changed_after_disconnect": bool(after),
                "target_in_plan_before": target_node in before_nodes,
                "target_in_first_plan_after": target_node in after_nodes,
                "target_removed_from_first_plan": (
                    target_node in before_nodes and target_node not in after_nodes
                ),
                "nodes_before": ";".join(sorted(before_nodes)),
                "nodes_after": ";".join(sorted(after_nodes)),
                "nodes_added": ";".join(sorted(after_nodes - before_nodes)),
                "nodes_removed": ";".join(sorted(before_nodes - after_nodes)),
                "providers_before": ";".join(sorted(before_providers)),
                "providers_after": ";".join(sorted(after_providers)),
                "providers_added": ";".join(sorted(after_providers - before_providers)),
                "providers_removed": ";".join(sorted(before_providers - after_providers)),
                "post_disconnect_plan_trigger": after.get("trigger", ""),
                "timeline_csv": str(copied_csv.relative_to(output)) if copied_csv.is_file() else "",
                "timeline_jsonl": str(copied_jsonl.relative_to(output)) if copied_jsonl.is_file() else "",
            }
        )
    with (output / "fable_disconnect_adaptation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adaptation_rows[0]))
        writer.writeheader()
        writer.writerows(adaptation_rows)
    with (output / "fable_disconnect_timeline_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timeline_index_rows[0]))
        writer.writeheader()
        writer.writerows(timeline_index_rows)

    changed = [row for row in adaptation_rows if row["plan_changed_after_disconnect"]]
    tp = [row for row in adaptation_rows if row["classification"] == "TRUE_POSITIVE"]
    target_removed = [
        row for row in adaptation_rows if row["target_removed_from_first_plan"]
    ]
    target_avoided = [
        row for row in adaptation_rows
        if not row["target_in_first_plan_after"]
    ]
    added_nodes = Counter(
        node for row in adaptation_rows for node in _split(str(row["nodes_added"]))
    )
    removed_nodes = Counter(
        node for row in adaptation_rows for node in _split(str(row["nodes_removed"]))
    )
    analysis = [
        "FABLE DISCONNECT ADAPTATION ANALYSIS",
        "",
        f"Campaign: {campaign}",
        f"FABLE disconnect runs recorded: {len(adaptation_rows)}",
        f"Runs with a post-disconnect plan decision: {len(changed)}",
        f"FABLE disconnect true positives: {len(tp)}",
        f"First post-disconnect plans that avoided the failed sensor: {len(target_avoided)}",
        f"First post-disconnect plans that explicitly removed a previously selected failed sensor: {len(target_removed)}",
        "",
        "Interpretation",
        "",
        "A post-disconnect plan is counted only when PLAN_SELECTED or PLAN_CHANGED appears at or after the validated FAIL_LINK application. Provider lifecycle noise alone is not counted as adaptation.",
        "",
        "The before/after columns report the last selected plan before failure and the first selected plan after failure. Added/removed nodes and providers therefore describe the first observable replanning response, not every later semantic-frontier transition.",
        "",
        "All first post-disconnect plan records in this campaign carry the trigger label INITIAL_ADMISSION. The timing and node/provider deltas prove that execution changed after the validated outage, but that trigger label does not by itself prove that a resource monitor directly caused each change; some changes coincide with semantic progression. Treat this as an instrumentation limitation.",
        "",
        "Most frequently added nodes after disconnect:",
    ]
    analysis.extend(f"  {node}: {count}" for node, count in added_nodes.most_common())
    analysis.append("Most frequently removed nodes after disconnect:")
    analysis.extend(f"  {node}: {count}" for node, count in removed_nodes.most_common())
    analysis.extend(["", "Per-run details are in fable_disconnect_adaptation.csv."])
    (output / "FABLE_ADAPTATION_ANALYSIS.txt").write_text("\n".join(analysis) + "\n")

    shutil.copy2(campaign / "campaign-report.json", output / "source_campaign_report.json")
    readme = (
        "B1/FABLE NOMINAL AND CAUSAL-DISCONNECT CAMPAIGN SUMMARY\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
        f"Source campaign: {campaign}\n\n"
        "Files:\n"
        "  accuracy_by_baseline_condition.csv - outcome counts partitioned by baseline and condition.\n"
        "  run_outcomes.csv - one row per attempted campaign cell, including invalid infrastructure cells.\n"
        "  fable_disconnect_adaptation.csv - FABLE plan/provider changes around validated disconnects.\n"
        "  fable_disconnect_timeline_index.csv - index of copied per-trace timelines.\n"
        "  fable_disconnect_timelines/ - local CSV and JSONL timeline copies for every FABLE disconnect run.\n"
        "  FABLE_ADAPTATION_ANALYSIS.txt - concise interpretation and aggregate change counts.\n"
        "  source_campaign_report.json - immutable copy of the campaign-level source report.\n\n"
        "Caution: infrastructure failures and missing results are not counted as false negatives; they are reported separately. The campaign used legacy B1 placements for some traces, so B1 accuracy should not be treated as final until those calibrations are repaired.\n"
    )
    (output / "README.txt").write_text(readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
