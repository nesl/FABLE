"""Bounded generation of retrospective predicate demands."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from uuid import UUID

from pydantic import Field, field_validator

from fable.common.base import FableModel
from fable.common.ids import uuid7
from fable.common.schemas import PredicateDemand
from fable.common.time import DeadlineSpec, EventTimeInterval, ensure_utc, utc_now

from .models import (
    HistoricalDemand,
    HistoricalDemandRejection,
    HistoricalDemandStatus,
    HistoricalGenerationResult,
)


class RetrospectiveConfig(FableModel):
    maximum_interval_ms: int = Field(default=120_000, ge=1)
    maximum_lookback_ms: int = Field(default=1_800_000, ge=1)
    maximum_outstanding_per_hypothesis: int = Field(default=4, ge=1)
    completion_safety_margin_ms: int = Field(default=50, ge=0)


class HistoricalDemandSpec(FableModel):
    original_demand: PredicateDemand
    historical_interval: EventTimeInterval
    source_id: str = Field(min_length=1)
    retained_input_type: str = Field(min_length=1)
    raw_buffer_interval: EventTimeInterval
    buffer_expires_at: datetime
    reason: str = Field(min_length=1)
    latest_useful_completion: datetime | None = None

    @field_validator("buffer_expires_at", "latest_useful_completion")
    @classmethod
    def _normalize_times(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)


class RetrospectiveDemandGenerator:
    """Creates historical work only while evidence and semantic state are useful."""

    def __init__(self, config: RetrospectiveConfig | None = None) -> None:
        self.config = config or RetrospectiveConfig()
        self._outstanding_by_hypothesis: dict[UUID, set[str]] = {}
        self._records: dict[str, HistoricalDemand] = {}

    @property
    def outstanding(self) -> tuple[HistoricalDemand, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self._records.values()
                    if item.status in (
                        HistoricalDemandStatus.CREATED,
                        HistoricalDemandStatus.ADMITTED,
                    )
                ),
                key=lambda item: item.historical_id or "",
            )
        )

    def generate(
        self,
        specs: Iterable[HistoricalDemandSpec],
        *,
        now: datetime | None = None,
    ) -> HistoricalGenerationResult:
        observed_now = ensure_utc(now or utc_now())
        generated: list[HistoricalDemand] = []
        rejections: list[HistoricalDemandRejection] = []
        for spec in sorted(
            tuple(specs),
            key=lambda item: (
                item.original_demand.deadline.latest_useful_completion,
                item.historical_interval.start,
                str(item.original_demand.demand_id),
            ),
        ):
            demand = spec.original_demand
            code, reason = self._reject_reason(spec, observed_now)
            if code is not None:
                rejections.append(
                    HistoricalDemandRejection(
                        original_demand_id=demand.demand_id,
                        code=code,
                        reason=reason,
                    )
                )
                continue

            completion_deadline = min(
                demand.deadline.latest_useful_completion,
                spec.latest_useful_completion
                or demand.deadline.latest_useful_completion,
                spec.buffer_expires_at
                - timedelta(milliseconds=self.config.completion_safety_margin_ms),
            )
            required_inputs = tuple(
                sorted(
                    set(demand.required_input_artifact_types)
                    | {spec.retained_input_type}
                )
            )
            payload = demand.model_dump(mode="python")
            payload.update(
                {
                    "demand_id": uuid7(),
                    "event_time_interval": spec.historical_interval,
                    "deadline": DeadlineSpec(
                        latest_useful_completion=completion_deadline,
                    ),
                    "eligible_source_ids": (spec.source_id,),
                    "required_input_artifact_types": required_inputs,
                    "sharing_key": None,
                }
            )
            retrospective = PredicateDemand.model_validate(payload)
            historical = HistoricalDemand(
                original_demand_id=demand.demand_id,
                demand=retrospective,
                source_id=spec.source_id,
                retained_input_type=spec.retained_input_type,
                historical_interval=spec.historical_interval,
                buffer_expires_at=spec.buffer_expires_at,
                reason=spec.reason,
                created_at=observed_now,
            )
            assert historical.historical_id is not None
            self._records[historical.historical_id] = historical
            self._outstanding_by_hypothesis.setdefault(demand.hypothesis_id, set()).add(
                historical.historical_id
            )
            generated.append(historical)
        return HistoricalGenerationResult(
            demands=tuple(generated),
            rejections=tuple(rejections),
        )

    def mark_admitted(self, historical_id: str) -> None:
        record = self._record(historical_id)
        record.status = HistoricalDemandStatus.ADMITTED

    def mark_finished(
        self,
        historical_id: str,
        *,
        status: HistoricalDemandStatus = HistoricalDemandStatus.COMPLETED,
    ) -> None:
        if status in (HistoricalDemandStatus.CREATED, HistoricalDemandStatus.ADMITTED):
            raise ValueError("mark_finished requires a terminal status")
        record = self._record(historical_id)
        record.status = status
        self._outstanding_by_hypothesis.get(record.demand.hypothesis_id, set()).discard(
            historical_id
        )

    def _reject_reason(
        self,
        spec: HistoricalDemandSpec,
        now: datetime,
    ) -> tuple[str | None, str]:
        demand = spec.original_demand
        if not spec.raw_buffer_interval.contains_interval(spec.historical_interval):
            return "BUFFER_MISS", "raw buffer does not contain the requested interval"
        if spec.buffer_expires_at <= now:
            return "BUFFER_EXPIRED", "retained input has already expired"
        duration_ms = int(spec.historical_interval.duration.total_seconds() * 1000)
        if duration_ms > self.config.maximum_interval_ms:
            return "INTERVAL_TOO_LARGE", (
                f"historical interval is {duration_ms} ms; maximum is "
                f"{self.config.maximum_interval_ms} ms"
            )
        lookback_ms = int((now - spec.historical_interval.start).total_seconds() * 1000)
        if lookback_ms > self.config.maximum_lookback_ms:
            return "LOOKBACK_TOO_OLD", (
                f"historical start is {lookback_ms} ms old; maximum is "
                f"{self.config.maximum_lookback_ms} ms"
            )
        latest = min(
            demand.deadline.latest_useful_completion,
            spec.latest_useful_completion or demand.deadline.latest_useful_completion,
            spec.buffer_expires_at
            - timedelta(milliseconds=self.config.completion_safety_margin_ms),
        )
        if latest <= now:
            return "DEADLINE_EXPIRED", "historical work cannot finish before its useful boundary"
        outstanding = self._outstanding_by_hypothesis.get(demand.hypothesis_id, set())
        if len(outstanding) >= self.config.maximum_outstanding_per_hypothesis:
            return "HYPOTHESIS_LIMIT", "historical-demand bound reached for this hypothesis"
        return None, ""

    def _record(self, historical_id: str) -> HistoricalDemand:
        try:
            return self._records[historical_id]
        except KeyError as exc:
            raise KeyError(f"unknown historical demand {historical_id}") from exc
