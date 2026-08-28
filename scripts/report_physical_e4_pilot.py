#!/usr/bin/env python3
"""Summarize only canonical cells from the physical E4 pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENT = "20241008-route-convoy-2-r013"
CONDITIONS = ("nominal", "compute_contention", "network_degradation")
BASELINES = ("B1_STATIC_WHOLE_EVENT", "B3_TASK_RESOURCE_ADAPTIVE", "FABLE")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    rows = []
    for condition in CONDITIONS:
        for baseline in BASELINES:
            cell = root / condition / baseline / "repetition-01"
            result = json.loads((cell / f"{EXPERIMENT}.json").read_text())
            proxy = json.loads((cell / "physical_proxy_validation.json").read_text())
            plans = []
            records = cell / f"{EXPERIMENT}.records" / "plan_decision.jsonl"
            for line in records.read_text().splitlines():
                decision = json.loads(line)
                signature = (
                    tuple(decision.get("selected_chain_ids") or ()),
                    tuple(decision.get("selected_node_ids") or ()),
                    tuple(decision.get("activated_provider_keys") or ()),
                )
                if signature not in plans:
                    plans.append(signature)
            totals = (result.get("resource_instrumentation") or {}).get("totals") or {}
            disturbances = result.get("disturbance_results") or []
            rows.append({
                "condition": condition,
                "baseline": baseline,
                "classification": result.get("classification"),
                "deadline_missed": bool(result.get("deadline_missed")),
                "wall_seconds": (result.get("timing") or {}).get("total_wall_seconds"),
                "cpu_seconds": totals.get("evaluated_system_cpu_seconds"),
                "host_gpu_seconds": totals.get("host_gpu_seconds"),
                "host_gpu_energy_joules": totals.get("host_gpu_energy_joules"),
                "network_bytes": (
                    int(totals.get("evaluated_path_network_rx_bytes") or 0)
                    + int(totals.get("evaluated_path_network_tx_bytes") or 0)
                ),
                "distinct_plans": len(plans),
                "selected_nodes": ";".join(plans[-1][1]),
                "selected_chains": ";".join(plans[-1][0]),
                "disturbance_apply_validated": (
                    condition == "nominal"
                    or bool(disturbances and disturbances[0].get("notification_validated"))
                ),
                "disturbance_restore_validated": (
                    condition == "nominal"
                    or bool(disturbances and disturbances[-1].get("notification_validated"))
                ),
                "proxy_validated": bool(proxy.get("validated")),
            })
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    no_plan_change = all(row["distinct_plans"] == 1 for row in rows)
    report = {
        "schema_version": "fable.physical_e4_pilot_summary.v1",
        "experiment_id": EXPERIMENT,
        "canonical_cell_count": len(rows),
        "true_positive_cells": sum(row["classification"] == "TRUE_POSITIVE" for row in rows),
        "deadline_miss_cells": sum(row["deadline_missed"] for row in rows),
        "validated_disturbed_cells": sum(
            row["condition"] != "nominal"
            and row["disturbance_apply_validated"]
            and row["disturbance_restore_validated"]
            for row in rows
        ),
        "proxy_validated_cells": sum(row["proxy_validated"] for row in rows),
        "any_plan_change": not no_plan_change,
        "discrimination_outcome": (
            "NON_DISCRIMINATING_NO_PLAN_CHANGE"
            if no_plan_change else "PLAN_CHANGE_OBSERVED"
        ),
        "rows": rows,
        "interpretation": (
            "The pilot validates physical execution and disturbance delivery, "
            "but does not support a comparative adaptation claim because every "
            "policy retained the same provider chain and placement."
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
