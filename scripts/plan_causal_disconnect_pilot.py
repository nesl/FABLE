#!/usr/bin/env python3
"""Derive a sensor-link outage from a successful nominal CE execution."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def derive_causal_cut(result_path: Path, base_trace_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    predictions = result.get("predictions") or []
    if result.get("classification") != "TRUE_POSITIVE" or not predictions:
        raise ValueError("causal-cut source must be a successful nominal result")
    prediction = predictions[0]
    hypothesis_id = prediction["hypothesis_id"]
    records = result_path.with_suffix(".records") / "predicate_observation.jsonl"
    observations = [
        json.loads(line)
        for line in records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terminal = [
        row for row in observations
        if row.get("hypothesis_id") == hypothesis_id
        and row.get("sensor_id")
    ]
    if not terminal:
        raise ValueError("accepted hypothesis has no sensor-provenanced observation")
    terminal.sort(key=lambda row: (_time(row["event_end_time"]), row["sensor_id"]))
    decisive = terminal[-1]
    seeds = [row for row in observations if row.get("predicate_id") == "AUDIO_EVENT"]
    if not seeds:
        raise ValueError("robbery causal cut requires an AUDIO_EVENT seed")
    seed_time = min(_time(row["event_time"]) for row in seeds)
    decisive_time = _time(decisive["event_time"])
    if decisive_time <= seed_time:
        raise ValueError("decisive observation does not follow the seed")
    # Place the cut halfway between the first seed and the decisive predicate.
    # This is fixed from the nominal execution before either policy is tested.
    disconnect_at = round((decisive_time - seed_time).total_seconds() / 2.0, 3)
    match = re.fullmatch(r"orin(\d+)_camera", str(decisive["sensor_id"]))
    if match is None:
        raise ValueError(f"unsupported decisive sensor: {decisive['sensor_id']}")
    switch_id = f"s_orin{match.group(1)}"
    trace = json.loads(base_trace_path.read_text(encoding="utf-8"))
    trace["trace_id"] = f"causal-cut-{result['experiment_id']}"
    trace["anchor"] = "TRACE_START"
    trace["transitions"][0].update({
        "transition_id": f"{result['experiment_id']}:causal-cut-disconnect",
        "offset_s": disconnect_at,
        "target_id": f"link:{switch_id}:s_edge",
    })
    trace["transitions"][1]["transition_id"] = (
        f"{result['experiment_id']}:causal-cut-post-eof-restore"
    )
    trace["causal_cut"] = {
        "source_result": str(result_path),
        "accepted_hypothesis_id": hypothesis_id,
        "seed_predicate": "AUDIO_EVENT",
        "seed_event_time": seed_time.isoformat(),
        "decisive_predicate": decisive["predicate_id"],
        "decisive_sensor_id": decisive["sensor_id"],
        "decisive_event_time": decisive_time.isoformat(),
        "disconnect_offset_seconds": disconnect_at,
        "selection_policy": "midpoint_between_seed_and_decisive_observation",
    }
    return trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--base-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = derive_causal_cut(args.result.resolve(), args.base_trace.resolve())
    provenance = trace.pop("causal_cut")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "condition_trace": trace,
        "causal_provenance": provenance,
        "provenance_path": str(provenance_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
