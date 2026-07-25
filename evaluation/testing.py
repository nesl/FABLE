"""Deterministic fake evaluation records for regression tests and examples."""

from __future__ import annotations

from datetime import timedelta

from fable.common.examples import BASE_TIME

from evaluation.metrics.event_matching import GroundTruthEvent
from evaluation.schemas import BaselineId, ComplexEventResult, CoordinationEpisode


def fake_ground_truth_event() -> GroundTruthEvent:
    return GroundTruthEvent(
        event_id="gt_convoy_1",
        event_family="route_convoy",
        start_time=BASE_TIME,
        end_time=BASE_TIME + timedelta(seconds=10),
        deadline=BASE_TIME + timedelta(seconds=15),
        bindings={"leader": "vehicle_1", "follower": "vehicle_2"},
    )


def fake_complex_event_result() -> ComplexEventResult:
    return ComplexEventResult(
        run_id="fake_run",
        baseline_id=BaselineId.FABLE,
        trace_id="fake_trace",
        request_id="fake_request",
        event_time=BASE_TIME,
        monotonic_timestamp_ns=1,
        result_id="ce_result_1",
        event_family="route_convoy",
        event_start_time=BASE_TIME,
        event_end_time=BASE_TIME + timedelta(seconds=10),
        emitted_at=BASE_TIME + timedelta(seconds=12),
        bindings={"leader": "vehicle_1", "follower": "vehicle_2"},
    )


def fake_coordination_episodes() -> tuple[CoordinationEpisode, ...]:
    common = dict(
        run_id="fake_run",
        baseline_id=BaselineId.FABLE,
        trace_id="fake_trace",
        request_id="fake_request",
        event_time=BASE_TIME,
        monotonic_timestamp_ns=1,
        upstream_sensor_id="orin_6",
        prediction_time=BASE_TIME,
        replay_supported_sensor_ids=("orin_1", "orin_5", "orin_6"),
        predicted_sensor_ids=("orin_5", "d3"),
        unavailable_mobile_sensor_ids=("d3",),
    )
    return (
        CoordinationEpisode(
            **common,
            episode_id="eligible_fixed_target",
            campaign_year=2025,
            spatial_evaluation_eligible=True,
            activated_sensor_ids=("orin_5",),
            actual_downstream_sensor_id="orin_5",
            downstream_observation_time=BASE_TIME + timedelta(seconds=3),
            deadline=BASE_TIME + timedelta(seconds=5),
        ),
        CoordinationEpisode(
            **common,
            episode_id="mobile_target_deferred",
            campaign_year=2025,
            spatial_evaluation_eligible=True,
            activated_sensor_ids=(),
            actual_downstream_sensor_id="d3",
            downstream_observation_time=BASE_TIME + timedelta(seconds=3),
            deadline=BASE_TIME + timedelta(seconds=5),
        ),
        CoordinationEpisode(
            **common,
            episode_id="unknown_2026_topology",
            campaign_year=2026,
            spatial_evaluation_eligible=False,
            activated_sensor_ids=(),
            actual_downstream_sensor_id=None,
        ),
    )
