from evaluation.condition_trace import ConditionTrace
from evaluation.schemas import BaselineId
from scripts.plan_rq3_network_post_eof_full import (
    B1_RECOVERY_GRACE_SECONDS,
    RECOVERY_ALLOWANCE_SECONDS,
    post_eof_trace,
    recovery_allowance_seconds,
)
from scripts.run_planned_ce_campaign import _condition_rank


def test_post_eof_restore_is_strictly_after_requested_replay_window() -> None:
    trace = ConditionTrace.model_validate(
        post_eof_trace("example", "s_orin13", 85.0, 50.0)
    )

    assert trace.anchor.value == "TRACE_START"
    assert trace.transitions[0].action.value == "FAIL_LINK"
    assert trace.transitions[0].offset_s == 12.5
    assert trace.transitions[1].action.value == "RESTORE_LINK"
    assert trace.transitions[1].offset_s == 90.0
    assert trace.duration_s > trace.transitions[1].offset_s


def test_post_eof_campaign_orders_disturbance_before_nominal() -> None:
    assert _condition_rank("post-eof-trace", disturbed_first=True) == 0
    assert _condition_rank(None, disturbed_first=True) == 1


def test_legacy_campaign_order_remains_nominal_first() -> None:
    assert _condition_rank(None, disturbed_first=False) == 0
    assert _condition_rank("condition-trace", disturbed_first=False) == 1


def test_recovery_budget_is_policy_specific() -> None:
    assert recovery_allowance_seconds(BaselineId.B1_STATIC_WHOLE_EVENT) == 15.0
    assert B1_RECOVERY_GRACE_SECONDS == 15.0
    for baseline in (
        BaselineId.B2_FRONTIER_FIXED_REALIZATION,
        BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        BaselineId.B4_GREEDY_FRONTIER,
        BaselineId.FABLE,
    ):
        assert recovery_allowance_seconds(baseline) == 225.0
    assert RECOVERY_ALLOWANCE_SECONDS == 225.0


def test_trace_duration_uses_supplied_recovery_budget() -> None:
    b1 = post_eof_trace("example", "s_orin13", 85.0, 50.0, 15.0)
    fable = post_eof_trace("example", "s_orin13", 85.0, 50.0, 225.0)

    assert b1["transitions"][1]["offset_s"] == 90.0
    assert b1["duration_s"] == 105.0
    assert fable["duration_s"] == 315.0
