"""Validated manifests for the canonical RQ3a mixed-request workload."""

from __future__ import annotations

from pydantic import Field, model_validator

from fable.common.base import FrozenFableModel


class WorkloadEpisode(FrozenFableModel):
    episode_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    ce_family: str = Field(min_length=1)
    request_offset_s: float = Field(ge=0)
    source_offset_s: float = Field(default=0, ge=0)
    duration_s: float = Field(gt=0)
    execution_mode: str = Field(pattern="^(profile_driven|replayed_provider_output)$")
    expected_condition_overlap: tuple[str, ...] = ()

    @property
    def end_offset_s(self) -> float:
        return self.request_offset_s + self.duration_s


class MixedRequestWorkload(FrozenFableModel):
    schema_version: str = "fable.mixed_request_workload.v1"
    workload_id: str = Field(min_length=1)
    duration_s: float = Field(gt=0)
    random_seed: int
    condition_trace_path: str = Field(min_length=1)
    vlm_mode: str = Field(default="replayed_response", pattern="^replayed_response$")
    episodes: tuple[WorkloadEpisode, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _validate_schedule(self) -> "MixedRequestWorkload":
        ids = [episode.episode_id for episode in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("episode_id values must be unique")
        if any(episode.end_offset_s > self.duration_s for episode in self.episodes):
            raise ValueError("episodes must finish within workload duration")
        starts = [episode.request_offset_s for episode in self.episodes]
        if starts != sorted(starts):
            raise ValueError("episodes must be ordered by request_offset_s")
        if not any(
            left.request_offset_s < right.request_offset_s < left.end_offset_s
            for left in self.episodes
            for right in self.episodes
            if left is not right
        ):
            raise ValueError("mixed workload must contain at least one overlap")
        return self
