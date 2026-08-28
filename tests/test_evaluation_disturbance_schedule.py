from datetime import timedelta

import pytest

from evaluation.disturbance_schedule import (
    DisturbanceKind,
    DisturbanceSchedule,
    DisturbanceScheduleController,
    DisturbanceStep,
    DisturbanceTrigger,
)
from fable.common.examples import BASE_TIME


def test_semantic_trigger_emits_bounded_apply_and_restore_once() -> None:
    controller = DisturbanceScheduleController(
        DisturbanceSchedule(
            schedule_id="w1-after-dispatch",
            steps=(
                DisturbanceStep(
                    step_id="wan",
                    trigger=DisturbanceTrigger.AFTER_PLAN_DISPATCH,
                    kind=DisturbanceKind.NETWORK_PROFILE,
                    target_id="site_to_cloud",
                    condition_id="W1",
                ),
            ),
        )
    )
    controller.observe(
        DisturbanceTrigger.AFTER_PLAN_DISPATCH,
        observed_at=BASE_TIME,
    )
    controller.observe(
        DisturbanceTrigger.AFTER_PLAN_DISPATCH,
        observed_at=BASE_TIME,
    )
    assert controller.due(now=BASE_TIME) == ()
    (apply,) = controller.due(now=BASE_TIME + timedelta(seconds=1))
    assert (apply.action, apply.condition_id) == ("APPLY", "W1")
    (restore,) = controller.due(now=BASE_TIME + timedelta(seconds=21))
    assert (restore.action, restore.condition_id) == ("RESTORE", "N0")
    assert controller.complete
    assert controller.due(now=BASE_TIME + timedelta(minutes=1)) == ()


def test_disturbance_duration_is_hard_bounded() -> None:
    with pytest.raises(ValueError):
        DisturbanceStep(
            step_id="unsafe",
            trigger=DisturbanceTrigger.DURING_PROVIDER_EXECUTION,
            kind=DisturbanceKind.PROVIDER_UNAVAILABLE,
            target_id="provider",
            condition_id="F1",
            duration_ms=30_001,
        )
