#!/usr/bin/env python3
"""Replay-free validation of the camera-local three-visit stalking contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

from fable.common.enums import HypothesisLifecycle
from fable.common.schemas import EventTimeInterval
from fable.semantic.definitions.vehicle import uncalibrated_repeated_pass_graph
from fable.semantic.models import ScriptedResultSpec, SemanticRuntimeConfig
from fable.semantic.runtime import SemanticRuntime
from fable.semantic.testing import predicate_result_from_spec, seed_result_from_spec


VISITS = (
    datetime(2026, 4, 14, 20, 10, 20, 599803, tzinfo=UTC),
    datetime(2026, 4, 14, 20, 11, 33, 949649, tzinfo=UTC),
    datetime(2026, 4, 14, 20, 12, 52, 122299, tzinfo=UTC),
)
REFERENCE = "camera_fov:dvpg_gq_orin_14"
VEHICLE = "dvpg_gq_orin_14:synthetic-session:9"


def interval(start: datetime) -> EventTimeInterval:
    return EventTimeInterval(start=start, end=start + timedelta(milliseconds=500))


def main() -> int:
    runtime = SemanticRuntime(
        uncalibrated_repeated_pass_graph(
            visit_count=3,
            minimum_return_gap_ms=30_000,
            identity_confirmation=True,
        ),
        config=SemanticRuntimeConfig(
            request_id="synthetic-orin14-track9",
            hypothesis_horizon_ms=600_000,
            deadline_offset_ms=600_000,
        ),
    )
    seeded = runtime.seed(
        seed_result_from_spec(
            runtime,
            ScriptedResultSpec(
                node_key="first_visit",
                source_id="orin14_camera",
                event_time_interval=interval(VISITS[0]),
                introduced={"vehicle": VEHICLE, "visit_reference": REFERENCE},
            ),
        )
    )
    hypothesis_id = seeded.hypothesis_ids[0]
    transitions = [{"stage": "first_visit", "status": seeded.status.value}]

    stages = (
        ("return_visit", VISITS[1], "visit_vehicle_2"),
        ("same_vehicle", VISITS[1] + timedelta(seconds=1), None),
        ("return_visit_2", VISITS[2], "visit_vehicle_3"),
        ("same_vehicle_2", VISITS[2] + timedelta(seconds=1), None),
    )
    for node_key, event_time, introduced_variable in stages:
        validated = (
            {"visit_reference": REFERENCE}
            if introduced_variable
            else {
                "vehicle": VEHICLE,
                (
                    "visit_vehicle_2"
                    if node_key == "same_vehicle"
                    else "visit_vehicle_3"
                ): VEHICLE,
            }
        )
        transition = runtime.apply(
            predicate_result_from_spec(
                runtime,
                hypothesis_id,
                ScriptedResultSpec(
                    node_key=node_key,
                    source_id="orin14_camera",
                    event_time_interval=interval(event_time),
                    introduced=(
                        {introduced_variable: VEHICLE}
                        if introduced_variable
                        else {}
                    ),
                    validated=validated,
                ),
            )
        )
        if transition.hypothesis_ids:
            hypothesis_id = transition.hypothesis_ids[-1]
        transitions.append({"stage": node_key, "status": transition.status.value})

    lifecycle = runtime.get_hypothesis(hypothesis_id).lifecycle
    passed = lifecycle == HypothesisLifecycle.COMPLETED
    print(
        json.dumps(
            {
                "schema_version": "fable.synthetic_stalking_validation.v1",
                "passed": passed,
                "camera_reference": REFERENCE,
                "vehicle_identity": VEHICLE,
                "visit_times": [item.isoformat() for item in VISITS],
                "visit_gaps_seconds": [
                    (VISITS[index] - VISITS[index - 1]).total_seconds()
                    for index in range(1, len(VISITS))
                ],
                "transitions": transitions,
                "terminal_lifecycle": lifecycle.value,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
