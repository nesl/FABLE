#!/usr/bin/env python3
"""Convert promoted desktop E0 timing observations into planner profiles."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from evaluation.experiments.e0_calibration import CalibrationObservation
from fable.planning.provider_registry import default_provider_profiles
from providers.vehicle.profiling import ProviderProfileRecord, save_profile_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--node-class",
        action="append",
        default=[],
        help="logical placement class on the same calibrated desktop",
    )
    args = parser.parse_args()
    rows = tuple(
        CalibrationObservation.model_validate_json(line)
        for line in args.observations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    grouped: dict[str, list[CalibrationObservation]] = defaultdict(list)
    for row in rows:
        grouped[row.target.provider_id].append(row)
    defaults = {row.provider_id: row for row in default_provider_profiles()}
    node_classes = tuple(args.node_class or ("sensor", "server"))
    digest = hashlib.sha256(args.observations.read_bytes()).hexdigest()
    profiles = []
    for provider_id, samples in sorted(grouped.items()):
        default = defaults[provider_id]
        cold = tuple(
            row.startup_ms
            for row in samples
            if row.invocation_kind == "cold" and row.successful
        )
        warm = tuple(
            row.execution_ms
            for row in samples
            if row.invocation_kind == "warm" and row.successful
        )
        if not cold or not warm:
            raise ValueError(f"incomplete E0 samples for {provider_id}")
        quality = sum(row.quality_score for row in samples) / len(samples)
        input_classes = sorted({row.target.input_class for row in samples})
        for node_class in node_classes:
            profiles.append(
                ProviderProfileRecord(
                    provider_id=provider_id,
                    node_class=node_class,
                    cold_start_samples_ms=cold,
                    warm_execution_samples_ms=warm,
                    cpu_cores=default.cpu_cores,
                    memory_mb=default.memory_mb,
                    gpu_memory_mb=default.gpu_memory_mb,
                    quality_score=quality,
                    metadata={
                        "timing_source": str(args.observations.resolve()),
                        "timing_source_sha256": digest,
                        "physical_host_class": "desktop_x86",
                        "logical_placement_class": node_class,
                        "input_classes": input_classes,
                        "resource_claim_source": (
                            "checked-in provider requirement; E0 measured "
                            "startup/execution/quality, not peak CPU/RAM"
                        ),
                    },
                )
            )
    # Loading a profile file replaces the registry defaults. Preserve
    # explicitly labeled wildcard requirements for providers that E0 could
    # not measure, while exact sensor/server records above override these for
    # every measured desktop provider.
    for provider_id, default in sorted(defaults.items()):
        if provider_id in grouped:
            continue
        profiles.append(
            ProviderProfileRecord(
                provider_id=provider_id,
                node_class="*",
                cold_start_samples_ms=(float(default.startup_ms),),
                warm_execution_samples_ms=(float(default.execution_ms),),
                cpu_cores=default.cpu_cores,
                memory_mb=default.memory_mb,
                gpu_memory_mb=default.gpu_memory_mb,
                quality_score=default.quality_score,
                metadata={
                    "measurement_status": "UNCALIBRATED_FALLBACK",
                    "reason": (
                        "no production-measured E0 worker target; retained "
                        "only to keep mixed-profile planning executable"
                    ),
                },
            )
        )
    save_profile_records(args.output, profiles)
    print(
        json.dumps(
            {
                "provider_count": len(grouped),
                "profile_count": len(profiles),
                "node_classes": node_classes,
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
