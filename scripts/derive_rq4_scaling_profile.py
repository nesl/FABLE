#!/usr/bin/env python3
"""Derive the RQ4 desktop profile from measured E0 and evaluation records."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import statistics
from pathlib import Path

from evaluation.scaling_execution import ScalingExecutionProfile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0-observations", type=Path, required=True)
    parser.add_argument("--planner-record-root", type=Path, required=True)
    parser.add_argument("--provider-profiles", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--rq1-report", type=Path, required=True)
    parser.add_argument(
        "--network-profile",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observations = [
        json.loads(line)
        for line in args.e0_observations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    warm_ms = [
        float(row["execution_ms"])
        for row in observations
        if row["invocation_kind"] == "warm" and row["successful"]
    ]
    cold_ms = [
        float(row["startup_ms"])
        for row in observations
        if row["invocation_kind"] == "cold" and row["successful"]
    ]

    plan_paths = sorted(
        Path(path)
        for path in glob.glob(
            str(args.planner_record_root / "records/*/plan_decision.jsonl")
        )
    )
    plan_rows = [
        json.loads(line)
        for path in plan_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latency_points: dict[int, list[float]] = {}
    for row in plan_rows:
        labels = int(row.get("labels_generated") or 0)
        latency = float(row.get("planning_latency_ms") or 0)
        if labels > 0 and latency > 0:
            latency_points.setdefault(labels, []).append(latency)
    medians = sorted(
        (labels, statistics.median(values))
        for labels, values in latency_points.items()
    )
    slopes = [
        (right_latency - left_latency) / (right_labels - left_labels)
        for index, (left_labels, left_latency) in enumerate(medians)
        for right_labels, right_latency in medians[index + 1 :]
        if right_labels > left_labels
    ]
    latency_per_label = max(1e-6, statistics.median(slopes))
    base_latency = max(
        1e-6,
        statistics.median(
            latency - latency_per_label * labels for labels, latency in medians
        ),
    )

    provider_document = json.loads(args.provider_profiles.read_text(encoding="utf-8"))
    memory_claims = [
        int(row["memory_mb"])
        for row in provider_document["profiles"]
        if row.get("metadata", {}).get("measurement_status")
        != "UNCALIBRATED_FALLBACK"
    ]
    memory_per_hypothesis = round(statistics.median(memory_claims)) * 1024 * 1024

    artifact_paths = sorted(args.artifact_root.glob("**/*.records/artifact_event.jsonl"))
    artifact_bytes: list[int] = []
    artifact_count = 0
    for path in artifact_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            artifact_count += 1
            size = int(json.loads(line).get("bytes") or 0)
            if size > 0:
                artifact_bytes.append(size)

    rq1 = json.loads(args.rq1_report.read_text(encoding="utf-8"))
    fable_counts: dict[str, int] = {}
    for key, counts in rq1["by_variant"].items():
        if key.startswith("FABLE|") and key != "FABLE|":
            for classification, count in counts.items():
                fable_counts[classification] = (
                    fable_counts.get(classification, 0) + int(count)
                )
    positives = fable_counts.get("TRUE_POSITIVE", 0)
    false_negatives = fable_counts.get("FALSE_NEGATIVE", 0)
    nominal_recall = positives / (positives + false_negatives)

    # The overload knee is a resource-capacity estimate: the median free-memory
    # level observed during live desktop runs divided by the median measured
    # provider memory claim.  The fixed 5% penalty is a predeclared conservative
    # response beyond that measured knee, not fitted to RQ4 outcomes.
    free_memory_samples: list[int] = []
    for path in args.artifact_root.glob("**/*.records/resource_sample.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = int(json.loads(line).get("memory_bytes") or 0)
                if value > 0:
                    free_memory_samples.append(value)
    median_free_memory = round(statistics.median(free_memory_samples))
    overload_threshold = max(1, median_free_memory // memory_per_hypothesis)

    source_paths = [
        args.e0_observations,
        args.provider_profiles,
        args.rq1_report,
        *args.network_profile,
        *plan_paths,
    ]
    digest_payload = "\n".join(
        f"{path.resolve()}:{_sha256(path)}" for path in source_paths
    )
    profile_id = "desktop_x86_rq4_" + hashlib.sha256(
        digest_payload.encode("utf-8")
    ).hexdigest()[:12]
    network_profiles: dict[str, dict[str, float]] = {}
    for path in args.network_profile:
        document = json.loads(path.read_text(encoding="utf-8"))
        links = document["links"]
        sensor_delays = [
            float(str(link["delay"]).removesuffix("ms"))
            for link in links
            if str(link["from"]).startswith("s_orin")
        ]
        backbone_delays = [
            float(str(link["delay"]).removesuffix("ms"))
            for link in links
            if not str(link["from"]).startswith("s_orin")
        ]
        one_way_ms = max(sensor_delays, default=0.0) + sum(backbone_delays)
        network_profiles[document["name"]] = {
            "round_trip_latency_ms": 2.0 * one_way_ms,
            "bandwidth_mbps": min(float(link["bw"]) for link in links),
            "loss_fraction": max(float(link.get("loss", 0)) for link in links) / 100.0,
        }

    profile = ScalingExecutionProfile(
        profile_id=profile_id,
        calibrated=True,
        calibration_source=(
            "Measured E0 provider execution/startup, RQ2 planner latency, "
            "RQ1 FABLE recall, and RQ3 live artifact/resource records"
        ),
        calibration_metadata={
            "source_digests": digest_payload.splitlines(),
            "e0_observation_count": len(observations),
            "e0_warm_sample_count": len(warm_ms),
            "e0_cold_sample_count": len(cold_ms),
            "planner_sample_count": sum(len(values) for values in latency_points.values()),
            "planner_label_points": [labels for labels, _ in medians],
            "artifact_sample_count": artifact_count,
            "resource_sample_count": len(free_memory_samples),
            "rq1_fable_true_positives": positives,
            "rq1_fable_false_negatives": false_negatives,
            "overload_penalty_source": "predeclared conservative 0.05 per capacity multiple",
        },
        base_planning_latency_ms=base_latency,
        latency_per_label_ms=latency_per_label,
        base_cpu_seconds_per_request=statistics.median(cold_ms) / 1000.0,
        cpu_seconds_per_label=statistics.median(warm_ms) / 1000.0,
        base_memory_bytes=memory_per_hypothesis,
        memory_bytes_per_live_hypothesis=memory_per_hypothesis,
        network_bytes_per_provider=round(statistics.median(artifact_bytes)),
        nominal_timely_recall=nominal_recall,
        overload_label_threshold=overload_threshold,
        overload_recall_penalty_per_threshold=0.05,
        network_profiles=network_profiles,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(profile.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
