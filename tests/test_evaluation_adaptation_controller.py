from datetime import timedelta

import pytest

from evaluation.adaptation_controller import (
    AdaptationControlPolicy,
    AdaptationController,
    AdaptationControllerMode,
    AllowedDisturbanceTarget,
    CompositeProfileApplier,
    DynamicAdaptationRun,
)
from evaluation.disturbance_schedule import (
    DisturbanceKind,
    DisturbanceSchedule,
    DisturbanceScheduleController,
    DisturbanceStep,
    DisturbanceTrigger,
    ScheduledDisturbanceAction,
)
from evaluation.schemas import BaselineId
from fable.common.examples import BASE_TIME


def _policy(mode=AdaptationControllerMode.DRY_RUN):
    return AdaptationControlPolicy(
        policy_id="adaptation-test",
        mode=mode,
        targets=(
            AllowedDisturbanceTarget(
                target_id="site_to_cloud",
                kinds=(DisturbanceKind.NETWORK_PROFILE,),
                condition_ids=("N0", "W1"),
            ),
        ),
    )


def _action(condition_id="W1"):
    return ScheduledDisturbanceAction(
        step_id="wan",
        action="APPLY",
        kind=DisturbanceKind.NETWORK_PROFILE,
        target_id="site_to_cloud",
        condition_id=condition_id,
        due_at=BASE_TIME,
    )


def test_dry_run_validates_without_claiming_host_application() -> None:
    result = AdaptationController(_policy()).execute(
        _action(),
        observed_at=BASE_TIME,
    )

    assert result.status == "DRY_RUN"
    assert not result.validated
    assert result.condition_epoch == 1
    assert result.helper_argv == ()


def test_profiled_controller_applies_only_allowlisted_typed_action() -> None:
    calls = []
    controller = AdaptationController(
        _policy(AdaptationControllerMode.PROFILED),
        profile_applier=lambda action, epoch: calls.append((action, epoch))
        or {"latency_ms": 75},
    )

    result = controller.execute(_action(), observed_at=BASE_TIME)

    assert result.status == "APPLIED"
    assert result.validated
    assert result.measurements == {"latency_ms": 75}
    assert calls[0][1] == 1
    with pytest.raises(ValueError, match="not allowlisted"):
        controller.execute(
            _action("W2"),
            observed_at=BASE_TIME,
        )


def test_dynamic_run_emits_apply_and_restore_common_records() -> None:
    records = []
    schedule = DisturbanceScheduleController(
        DisturbanceSchedule(
            schedule_id="wan-after-plan",
            steps=(
                DisturbanceStep(
                    step_id="wan",
                    trigger=DisturbanceTrigger.AFTER_PLAN_DISPATCH,
                    kind=DisturbanceKind.NETWORK_PROFILE,
                    target_id="site_to_cloud",
                    condition_id="W1",
                    delay_ms=1000,
                    duration_ms=20_000,
                    restore_condition_id="N0",
                ),
            ),
        )
    )
    run = DynamicAdaptationRun(
        schedule=schedule,
        controller=AdaptationController(
            _policy(AdaptationControllerMode.PROFILED),
            profile_applier=lambda action, epoch: {
                "profile": action.condition_id,
                "epoch": epoch,
            },
        ),
        run_id="run",
        baseline_id=BaselineId.FABLE,
        trace_id="trace",
        request_id="request",
        record_sink=records.append,
    )

    assert run.observe(
        DisturbanceTrigger.AFTER_PLAN_DISPATCH,
        observed_at=BASE_TIME,
    ) == ()
    (applied,) = run.advance(now=BASE_TIME + timedelta(seconds=1))
    (restored,) = run.advance(now=BASE_TIME + timedelta(seconds=21))

    assert applied.action == "APPLY"
    assert applied.condition_epoch == 1
    assert applied.validated
    assert restored.action == "RESTORE"
    assert restored.condition_epoch == 2
    assert [item.disturbance_id for item in records] == [
        "wan:APPLY",
        "wan:RESTORE",
    ]


def test_host_helper_policy_rejects_relative_executable() -> None:
    with pytest.raises(ValueError, match="absolute helper"):
        AdaptationControlPolicy(
            policy_id="unsafe",
            mode=AdaptationControllerMode.HOST_HELPER,
            helper_path="scripts/arbitrary",
            targets=_policy().targets,
        )


def test_composite_profile_applier_routes_only_by_typed_kind() -> None:
    calls = []
    router = CompositeProfileApplier(
        {
            DisturbanceKind.NETWORK_PROFILE: (
                lambda action, epoch: calls.append((action.kind, epoch))
                or {"routed": True}
            )
        }
    )

    assert router(_action(), 3) == {"routed": True}
    assert calls == [(DisturbanceKind.NETWORK_PROFILE, 3)]
