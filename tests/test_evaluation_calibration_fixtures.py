from __future__ import annotations

import sys

import pytest

from evaluation.calibration_execution import execute_calibration_campaign
from evaluation.calibration_fixtures import (
    CalibrationFixture,
    CalibrationFixtureManifest,
)
from evaluation.experiments.e0_calibration import CalibrationTarget, build


def _target() -> CalibrationTarget:
    return CalibrationTarget(
        target_id="target",
        provider_id="provider",
        tier="sensor",
        input_class="track_set.v1",
    )


def test_json_worker_fixture_executes_without_a_shell(tmp_path) -> None:
    target = _target()
    manifest = CalibrationFixtureManifest(
        fixtures=(
            CalibrationFixture(
                fixture_id="fixture",
                target=target,
                argv=(
                    sys.executable,
                    "tests/fixtures/calibration_echo_worker.py",
                ),
                payload={"quality_score": 0.85, "ambiguity_score": 0.15},
            ),
        )
    )
    result = execute_calibration_campaign(
        build(
            (target,),
            warm_repetitions=1,
            cold_repetitions=1,
        ),
        manifest.adapter_registry(),
        tmp_path,
        invocation_timeout_seconds=2,
    )
    assert result.completed_observations == 1
    assert result.excluded_runs == 1


def test_fixture_manifest_rejects_shell_and_duplicate_target() -> None:
    target = _target()
    with pytest.raises(ValueError, match="cannot invoke a shell"):
        CalibrationFixture(
            fixture_id="bad",
            target=target,
            argv=("bash", "-c", "anything"),
            payload={},
        )
    fixture = CalibrationFixture(
        fixture_id="one",
        target=target,
        argv=("worker",),
        payload={},
    )
    with pytest.raises(ValueError, match="unique"):
        CalibrationFixtureManifest(fixtures=(fixture, fixture))
