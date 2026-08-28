from evaluation.condition_trace import (
    ConditionAnchor,
    ConditionTrace,
    MonotonicConditionTraceController,
    classify_disturbance_exposure,
)


def _trace() -> ConditionTrace:
    return ConditionTrace.model_validate_json(
        open(
            "evaluation/manifests/adaptation/rq3a_short_wan.json",
            encoding="utf-8",
        ).read()
    )


def test_condition_trace_is_released_by_elapsed_time_only():
    controller = MonotonicConditionTraceController(_trace())
    assert controller.due(elapsed_s=29.9) == ()
    assert [
        item.transition.transition_id
        for item in controller.due(elapsed_s=30.0)
    ] == ["wan-degrade"]
    assert controller.due(elapsed_s=74.9) == ()
    assert [
        item.transition.transition_id
        for item in controller.due(elapsed_s=75.0)
    ] == ["wan-recover"]
    assert controller.complete


def test_condition_trace_can_anchor_transitions_to_live_admission():
    trace = ConditionTrace.model_validate_json(
        open(
            "evaluation/manifests/adaptation/rq3a_short_sensor_uplink.json",
            encoding="utf-8",
        ).read()
    )

    assert trace.anchor == ConditionAnchor.ADMISSION
    assert [item.offset_s for item in trace.transitions] == [30.0, 75.0]
    assert trace.duration_s == 360.0


def test_exposure_classification_covers_fixed_offset_cases():
    assert classify_disturbance_exposure(
        demand_start_s=20,
        demand_end_s=50,
        disturbance_start_s=30,
        disturbance_end_s=75,
    ) == "ACTIVE_DEMAND_CROSSES_DISTURBANCE_ONSET"
    assert classify_disturbance_exposure(
        demand_start_s=45,
        demand_end_s=90,
        disturbance_start_s=30,
        disturbance_end_s=75,
    ) == "ACTIVE_DEMAND_CROSSES_RECOVERY"
    assert classify_disturbance_exposure(
        demand_start_s=40,
        demand_end_s=60,
        disturbance_start_s=30,
        disturbance_end_s=75,
    ) == "DEMAND_BEGINS_UNDER_DISTURBANCE"


def test_updated_rq3a_uses_per_replay_midpoint_schedule():
    for name, target, expected_offsets in (
        ("n1_pass_follow_orin13.json", "sensor_uplink:s_orin13", [60.0, 114.0]),
        ("n1_convergence_mobile1.json", "sensor_uplink:s_mobile_archive_1", [36.0, 68.4]),
        ("n2_robbery.json", "site_backbone", [90.0, 171.0]),
    ):
        trace = ConditionTrace.model_validate_json(
            open(
                f"evaluation/manifests/adaptation/rq3a_updated/{name}",
                encoding="utf-8",
            ).read()
        )
        assert trace.anchor == ConditionAnchor.TRACE_START
        assert [item.offset_s for item in trace.transitions] == expected_offsets
        assert [item.target_id for item in trace.transitions] == [target, target]


def test_compute_contention_midpoint_schedule_preserves_recovery_tail():
    from evaluation.condition_trace import midpoint_disturbance_schedule

    assert midpoint_disturbance_schedule(120) == (60.0, 114.0)
    assert midpoint_disturbance_schedule(60) == (30.0, 57.0)
