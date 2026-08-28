from datetime import datetime, timezone

from evaluation.adaptation_controller import (
    AdaptationControlPolicy,
    AdaptationController,
    AdaptationControllerMode,
    AllowedDisturbanceTarget,
)
from evaluation.disturbance_schedule import (
    DisturbanceKind,
    ScheduledDisturbanceAction,
)
from evaluation.live_validation import (
    validate_apply_restore,
    validate_concurrent_logging,
)
from evaluation.schemas import (
    ArtifactEvent,
    BaselineId,
    ProviderLeaseEvent,
    ResourceSample,
    RetrospectiveAttempt,
)


NOW = datetime.now(timezone.utc)


def _action(action: str, condition: str) -> ScheduledDisturbanceAction:
    return ScheduledDisturbanceAction(
        step_id="network",
        action=action,
        kind=DisturbanceKind.NETWORK_PROFILE,
        target_id="site",
        condition_id=condition,
        due_at=NOW,
    )


def test_live_condition_validation_always_restores_after_probe() -> None:
    calls = []
    controller = AdaptationController(
        AdaptationControlPolicy(
            policy_id="profiled",
            mode=AdaptationControllerMode.PROFILED,
            targets=(
                AllowedDisturbanceTarget(
                    target_id="site",
                    kinds=(DisturbanceKind.NETWORK_PROFILE,),
                    condition_ids=("W1", "N0"),
                ),
            ),
        ),
        profile_applier=lambda action, epoch: calls.append(action.action) or {
            "epoch": epoch
        },
    )
    result = validate_apply_restore(
        controller,
        apply_action=_action("APPLY", "W1"),
        restore_action=_action("RESTORE", "N0"),
        condition_probe=lambda _: {"path_validated": True},
        restored_probe=lambda _: {"path_validated": True},
        observed_at=NOW,
    )
    assert result.successful
    assert calls == ["APPLY", "RESTORE"]


def _record(request_id: str) -> dict:
    return {
        "run_id": "run",
        "baseline_id": BaselineId.FABLE,
        "trace_id": "trace",
        "request_id": request_id,
        "event_time": NOW,
        "monotonic_timestamp_ns": 1,
    }


def test_concurrent_validation_joins_sharing_capacity_and_artifact_failure() -> None:
    leases = tuple(
        ProviderLeaseEvent(
            **_record(request),
            lease_id=f"lease-{request}",
            provider_instance_id="shared-provider",
            demand_id=f"demand-{request}",
            lease_event="ATTACHED",
            attached_at=NOW,
        )
        for request in ("a", "b")
    )
    resources = tuple(
        ResourceSample(
            **_record(request),
            node_id="node",
            metadata={
                "attribution": "shared_active_demand_allocation",
                "session_id": "session",
                "heartbeat_sequence": 1,
                "allocation_fraction": 0.5,
            },
        )
        for request in ("a", "b")
    )
    attempt = RetrospectiveAttempt(
        **_record("a"),
        attempt_id="attempt",
        checkpoint_id="checkpoint",
        predicate_id="PASSES",
        replay_policy="retained",
        retained_interval_start=NOW,
        retained_interval_end=NOW,
        outcome="BUFFER_EXPIRED",
    )
    artifact = ArtifactEvent(
        **_record("a"),
        artifact_id="artifact",
        artifact_type="raw_video",
        action="BUFFER_EXPIRED",
        node_id="node",
        metadata={"derived_from_attempt_id": "attempt"},
    )
    report = validate_concurrent_logging(
        leases=leases,
        resources=resources,
        retrospective_attempts=(attempt,),
        artifacts=(artifact,),
    )
    assert report.successful


def test_resource_sample_accepts_interval_gpu_time() -> None:
    sample = ResourceSample(
        **_record("gpu"),
        node_id="gpu:0",
        gpu_utilization=0.5,
        gpu_time_seconds=0.25,
    )
    assert sample.gpu_time_seconds == 0.25
