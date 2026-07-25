#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FABLE_ROOT = Path(__file__).resolve().parents[2]
if str(FABLE_ROOT) not in sys.path:
    sys.path.insert(0, str(FABLE_ROOT))

from evaluation.catalog import ExperimentCatalog


def main() -> None:
    parser = argparse.ArgumentParser(description="List FABLE ground-truth experiments and replay/spatial eligibility.")
    parser.add_argument("--year", type=int)
    parser.add_argument("--variant")
    parser.add_argument("--recommended-only", action="store_true")
    parser.add_argument("--spatial-only", action="store_true")
    args = parser.parse_args()
    catalog = ExperimentCatalog.from_csv(
        FABLE_ROOT / "evaluation/labels/filtered_complex_event_experiments.csv",
        transition_model_path=FABLE_ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json",
    )
    rows = list(catalog.experiments)
    if args.year is not None:
        rows = [item for item in rows if item.campaign_year == args.year]
    if args.variant:
        query = args.variant.lower()
        rows = [item for item in rows if query in item.ce_variant.lower()]
    if args.recommended_only:
        rows = [item for item in rows if item.recommended_for_use]
    if args.spatial_only:
        rows = [item for item in rows if item.spatial_coordination_eligible]
    for item in rows:
        print(json.dumps({
            "experiment_id": item.experiment_id,
            "start": item.recording_start.isoformat(),
            "end": item.recording_end.isoformat(),
            "variant": item.ce_variant,
            "quality": item.quality_status,
            "recommended": item.recommended_for_use,
            "spatial_eligible": item.spatial_coordination_eligible,
            "topology_deployments": item.topology_deployment_ids,
            "replay_scope": item.replay_sensor_scope,
            "mobile_replay_deferred": item.unavailable_mobile_sensor_ids,
        }))


if __name__ == "__main__":
    main()
