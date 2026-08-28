from datetime import datetime, timezone

from evaluation.calibration_promotion import (
    CalibrationPromotionPolicy,
    promote_desktop_observations,
)
from evaluation.experiments.e0_calibration import (
    CalibrationObservation,
    CalibrationTarget,
)


def _target(provider: str = "pairwise_distance_evaluator") -> CalibrationTarget:
    return CalibrationTarget(
        target_id="target",
        provider_id=provider,
        tier="sensor",
        input_class="projected_track_set.v1",
    )


def _observation(kind: str, repetition: int) -> CalibrationObservation:
    return CalibrationObservation(
        run_id=f"{kind}-{repetition}",
        target=_target(),
        invocation_kind=kind,
        startup_ms=10,
        execution_ms=1,
        quality_score=1,
        ambiguity_score=0,
    )


def test_promotion_requires_complete_warm_and_cold_measurements() -> None:
    operations = {
        "pairwise_distance_evaluator": {
            "measurement_status": "MEASURED_PROVIDER",
            "input_classes": ("projected_track_set.v1",),
        }
    }
    incomplete = promote_desktop_observations(
        (_observation("warm", 1),),
        (_target(),),
        worker_operations=operations,
        policy=CalibrationPromotionPolicy(
            host_id="desktop",
            minimum_warm_samples=2,
            minimum_cold_samples=1,
        ),
        generated_at=datetime.now(timezone.utc),
    )
    assert not incomplete.promotable
    assert incomplete.insufficient_targets

    complete = promote_desktop_observations(
        (
            _observation("warm", 1),
            _observation("warm", 2),
            _observation("cold", 1),
        ),
        (_target(),),
        worker_operations=operations,
        policy=CalibrationPromotionPolicy(
            host_id="desktop",
            minimum_warm_samples=2,
            minimum_cold_samples=1,
        ),
        generated_at=datetime.now(timezone.utc),
    )
    assert complete.promotable
    assert complete.promoted_profile_count == 1


def test_validation_only_backend_cannot_be_promoted() -> None:
    report = promote_desktop_observations(
        (_observation("warm", 1), _observation("cold", 1)),
        (_target(),),
        worker_operations={
            "pairwise_distance_evaluator": {
                "measurement_status": "IMPLEMENTATION_VALIDATION_ONLY",
                "input_classes": ("projected_track_set.v1",),
            }
        },
        policy=CalibrationPromotionPolicy(
            host_id="desktop",
            minimum_warm_samples=1,
            minimum_cold_samples=1,
        ),
        generated_at=datetime.now(timezone.utc),
    )
    assert not report.promotable
    assert report.non_measured_providers
