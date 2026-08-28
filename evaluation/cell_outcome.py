"""Orthogonal status contract for one evaluation cell."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from fable.common.base import FrozenFableModel


class OutcomeStatus(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    VALID = "VALID"
    INVALID = "INVALID"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"


class ScientificClassification(StrEnum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    TRUE_NEGATIVE = "TRUE_NEGATIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    UNKNOWN = "UNKNOWN"


class CellOutcome(FrozenFableModel):
    schema_version: Literal["fable.cell_outcome.v1"] = "fable.cell_outcome.v1"
    infrastructure_status: OutcomeStatus
    protocol_status: OutcomeStatus
    mutation_status: OutcomeStatus
    adaptation_status: OutcomeStatus
    measurement_status: OutcomeStatus
    scientific_classification: ScientificClassification
    cleanup_status: OutcomeStatus
    reasons: tuple[str, ...] = ()
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @property
    def safe_to_continue(self) -> bool:
        return self.infrastructure_status != OutcomeStatus.FAILED
