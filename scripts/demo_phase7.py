#!/usr/bin/env python3
"""Run deterministic Phase-7 vehicle-provider and planner demonstrations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from types import SimpleNamespace

from fable.common.examples import BASE_TIME
from fable.planning import PhysicalAlternativeGraphBuilder
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_demand,
    fake_provider_registry,
)
from providers.vehicle.detector import LegacyReplayYoloAdapter
from providers.vehicle.geometry import PassReferenceEvaluator, ReferenceLine, RouteMapMatcher, RoutePolyline
from providers.vehicle.models import Point2D
from providers.vehicle.tracker import RoboflowTrackerAdapter


class FakeTracker:
    def __init__(self) -> None:
        self.index = 0

    def update(self, detections, frame=None, timestamp=None):
        self.index += 1
        return SimpleNamespace(
            xyxy=[[float(self.index), 0.0, float(self.index + 4), 2.0]],
            tracker_id=[17],
            confidence=[0.9],
            class_id=[0],
        )

    def reset(self):
        return None


def main() -> int:
    rows = [
        json.loads(line)
        for line in (ROOT / "providers/tests/phase7_fixtures/legacy_yolo_frames.jsonl")
        .read_text()
        .splitlines()
    ]
    detector = LegacyReplayYoloAdapter()
    tracker = RoboflowTrackerAdapter(
        tracker=FakeTracker(), detections_factory=lambda _: object(), session_id="demo"
    )
    route = RoutePolyline(
        route_id="eastbound",
        coordinate_frame_id="replay_world",
        points=(
            Point2D(x=-20, y=0, coordinate_frame_id="replay_world"),
            Point2D(x=20, y=0, coordinate_frame_id="replay_world"),
        ),
    )
    gate = ReferenceLine(
        reference_id="camera_a_gate",
        start=Point2D(x=0, y=-10, coordinate_frame_id="replay_world"),
        end=Point2D(x=0, y=10, coordinate_frame_id="replay_world"),
        direction=-1,
    )
    matcher = RouteMapMatcher()
    passes = PassReferenceEvaluator()
    observations = []
    for index, payload in enumerate(rows):
        frame = detector.parse(payload, source_id="dvpg_gq_orin_11", frame_id=f"frame_{index}")
        track_set = tracker.update(frame)
        projected = track_set.model_copy(
            update={"tracks": tuple(matcher.match(track, (route,)) for track in track_set.tracks)}
        )
        for track in projected.tracks:
            result = passes.update(track, gate)
            if result is not None:
                observations.append(result)

    registry = fake_provider_registry()
    demand = fake_follow_demand()
    graph = PhysicalAlternativeGraphBuilder(
        provider_registry=registry,
        artifact_catalog=fake_artifact_catalog(),
        deployment=fake_deployment(),
    ).build((demand,), now=BASE_TIME)
    summary = {
        "tracks_scoped_as": "source_id:tracker_session_id:local_track_id",
        "passes": [item.model_dump(mode="json") for item in observations],
        "follow_chains": sorted({item.chain_id for item in graph.alternatives}),
        "retrospective_chain": "follows_local_from_retained_detections",
        "tracker_snapshot_claimed": "tracker_state_snapshot.v1" in registry.data_types,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
