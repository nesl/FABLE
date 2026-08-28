#!/usr/bin/env python3
"""Report fail-closed N0/WAN attribution for the paired RQ3a pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_documents(root: Path) -> list[dict]:
    documents = []
    for path in root.rglob("*.json"):
        if path.name in {"plan.json", "report.json", "campaign-report.json"}:
            continue
        try:
            document = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if document.get("experiment_id") and document.get("baseline"):
            document["_path"] = str(path.resolve())
            documents.append(document)
    return documents


def disturbance_valid(document: dict) -> bool:
    transitions = document.get("disturbance_results") or []
    applied = [row for row in transitions if row.get("action") == "APPLY"]
    restored = [row for row in transitions if row.get("action") == "RESTORE"]
    return bool(applied and restored) and all(
        bool(row.get("validated") or (row.get("response") or {}).get("validated"))
        for row in (*applied, *restored)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trace_ids = {}
    for row in manifest:
        trace = load_json(Path(row["condition_trace_path"]))
        trace_ids[row["condition_trace_id"]] = trace["trace_id"]

    results = {}
    for document in result_documents(args.results_root):
        trace_id = (document.get("condition_trace") or {}).get("trace_id")
        key = (document["experiment_id"], document["baseline"], trace_id)
        results[key] = document

    pairs = []
    identities = sorted(
        {(row["experiment_id"], row["baseline_id"]) for row in manifest}
    )
    for experiment_id, baseline_id in identities:
        nominal = results.get(
            (experiment_id, baseline_id, trace_ids["n0-control"])
        )
        wan = results.get((experiment_id, baseline_id, trace_ids["wan"]))
        nominal_class = nominal.get("classification") if nominal else "MISSING"
        wan_class = wan.get("classification") if wan else "MISSING"
        if nominal is None or wan is None:
            attribution = "INCOMPLETE_PAIR"
        elif nominal_class != "TRUE_POSITIVE":
            attribution = "NOMINAL_RUNTIME_INVALID"
        elif not disturbance_valid(wan):
            attribution = "DISTURBANCE_APPLICATION_INVALID"
        elif wan_class != "TRUE_POSITIVE":
            attribution = "NETWORK_ATTRIBUTABLE_REGRESSION"
        else:
            attribution = "QUALITY_PRESERVED_UNDER_WAN"
        pairs.append(
            {
                "experiment_id": experiment_id,
                "baseline_id": baseline_id,
                "nominal_classification": nominal_class,
                "wan_classification": wan_class,
                "attribution": attribution,
                "nominal_result": nominal.get("_path") if nominal else None,
                "wan_result": wan.get("_path") if wan else None,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_attribution.json").write_text(
        json.dumps(
            {
                "schema_version": "fable.rq3a_paired_attribution.v1",
                "pairs": pairs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "paired_attribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps({"pairs": len(pairs), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
