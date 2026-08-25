from datetime import timedelta

from fable.common.examples import BASE_TIME

from evaluation.metrics.event_matching import GroundTruthEvent, evaluate_event_results
from evaluation.metrics.spatial_coordination import evaluate_spatial_coordination
from evaluation.schemas import BaselineId, ComplexEventResult, CoordinationEpisode


def test_event_matching_reports_timely_binding_accuracy() -> None:
    truth = GroundTruthEvent(
        event_id="gt-1",
        event_family="route_convoy",
        start_time=BASE_TIME,
        end_time=BASE_TIME + timedelta(seconds=10),
        deadline=BASE_TIME + timedelta(seconds=15),
        bindings={"leader": "v1", "follower": "v2"},
    )
    result = ComplexEventResult(
        run_id="run",
        baseline_id=BaselineId.FABLE,
        trace_id="trace",
        request_id="request",
        event_time=BASE_TIME,
        monotonic_timestamp_ns=1,
        result_id="result-1",
        event_family="route_convoy",
        event_start_time=BASE_TIME,
        event_end_time=BASE_TIME + timedelta(seconds=10),
        emitted_at=BASE_TIME + timedelta(seconds=12),
        bindings={"leader": "v1", "follower": "v2"},
    )
    metrics = evaluate_event_results((truth,), (result,))
    assert metrics.f1 == 1.0
    assert metrics.timely_recall == 1.0
    assert metrics.role_binding_accuracy == 1.0


def test_event_matching_accepts_configured_boundary_tolerance() -> None:
    truth = GroundTruthEvent(
        event_id="gt-1",
        event_family="robbery",
        start_time=BASE_TIME,
        end_time=BASE_TIME + timedelta(seconds=10),
    )
    result = ComplexEventResult(
        run_id="run",
        baseline_id=BaselineId.FABLE,
        trace_id="trace",
        request_id="request",
        event_time=BASE_TIME + timedelta(seconds=11),
        monotonic_timestamp_ns=1,
        result_id="result-1",
        event_family="robbery",
        event_start_time=BASE_TIME + timedelta(seconds=11),
        event_end_time=BASE_TIME + timedelta(seconds=12),
        emitted_at=BASE_TIME + timedelta(seconds=12),
    )
    strict = evaluate_event_results((truth,), (result,))
    tolerant = evaluate_event_results(
        (truth,), (result,), temporal_boundary_tolerance_seconds=1.0
    )
    assert strict.true_positives == 0
    assert tolerant.true_positives == 1


def test_event_matching_rejects_negative_boundary_tolerance() -> None:
    try:
        evaluate_event_results((), (), temporal_boundary_tolerance_seconds=-0.1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative tolerance must be rejected")


def _episode(**updates):
    values = dict(
        run_id="run",
        baseline_id=BaselineId.FABLE,
        trace_id="trace",
        request_id="request",
        event_time=BASE_TIME,
        monotonic_timestamp_ns=1,
        episode_id="episode",
        campaign_year=2025,
        spatial_evaluation_eligible=True,
        upstream_sensor_id="orin_6",
        predicted_sensor_ids=("orin_5", "d3"),
        activated_sensor_ids=("orin_5",),
        replay_supported_sensor_ids=("orin_1", "orin_5", "orin_6"),
        unavailable_mobile_sensor_ids=("d3",),
        actual_downstream_sensor_id="orin_5",
        prediction_time=BASE_TIME,
        downstream_observation_time=BASE_TIME + timedelta(seconds=3),
        deadline=BASE_TIME + timedelta(seconds=5),
    )
    values.update(updates)
    return CoordinationEpisode(**values)


def test_spatial_metrics_exclude_unknown_topology_and_mobile_targets() -> None:
    valid = _episode()
    unknown = _episode(
        episode_id="2026",
        campaign_year=2026,
        spatial_evaluation_eligible=False,
    )
    mobile_target = _episode(
        episode_id="mobile",
        actual_downstream_sensor_id="d3",
        activated_sensor_ids=(),
    )
    metrics = evaluate_spatial_coordination((valid, unknown, mobile_target))
    assert metrics.total_episodes == 3
    assert metrics.evaluated_episodes == 1
    assert metrics.excluded_unknown_topology == 1
    assert metrics.excluded_unavailable_mobile_target == 1
    assert metrics.timely_handoff_recall == 1.0
    assert metrics.weighted_activation_fanout_reduction > 0
