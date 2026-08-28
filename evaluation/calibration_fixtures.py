"""Typed fixture manifests for provider-level E0 worker adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from evaluation.calibration_execution import (
    CalibrationAdapterRegistry,
    JsonCommandCalibrationAdapter,
)
from evaluation.experiments.e0_calibration import CalibrationTarget
from fable.common.base import FrozenFableModel


class CalibrationFixture(FrozenFableModel):
    fixture_id: str = Field(min_length=1)
    target: CalibrationTarget
    argv: tuple[str, ...]
    payload: dict[str, Any]
    subprocess_timeout_seconds: float = Field(default=60, gt=0, le=300)
    supported_invocation_kinds: tuple[str, ...] = ("warm",)

    @model_validator(mode="after")
    def _argv_is_shell_free(self):
        if not self.argv:
            raise ValueError("fixture requires a worker argv")
        if self.argv[0] in {"sh", "bash", "/bin/sh", "/bin/bash"}:
            raise ValueError("calibration fixtures cannot invoke a shell")
        return self


class CalibrationFixtureManifest(FrozenFableModel):
    schema_version: str = "fable.calibration_fixture_manifest.v1"
    fixtures: tuple[CalibrationFixture, ...]

    @model_validator(mode="after")
    def _unique_targets(self):
        keys = [
            (
                item.target.provider_id,
                item.target.tier,
                item.target.input_class,
            )
            for item in self.fixtures
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("calibration fixture targets must be unique")
        return self

    def adapter_registry(self) -> CalibrationAdapterRegistry:
        registry = CalibrationAdapterRegistry()
        for fixture in self.fixtures:
            registry.register(
                fixture.target,
                JsonCommandCalibrationAdapter(
                    argv=fixture.argv,
                    fixture=fixture.payload,
                    subprocess_timeout_seconds=(
                        fixture.subprocess_timeout_seconds
                    ),
                    supported_invocation_kinds=(
                        fixture.supported_invocation_kinds
                    ),
                ),
            )
        return registry


def load_calibration_fixture_manifest(
    path: str | Path,
) -> CalibrationFixtureManifest:
    return CalibrationFixtureManifest.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
