from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field

from fable.common.base import FableModel


class StaticPipelineSpec(FableModel):
    event_family: str = Field(min_length=1)
    preferred_chain_ids: tuple[str, ...]
    fixed_sensor_policy: str = "all_replay_supported_orin"
    fixed_representation_policy: str = ""


class StaticPipelineRegistry:
    def __init__(self, specs: dict[str, StaticPipelineSpec]) -> None:
        self.specs = specs

    @classmethod
    def load(cls, path: str | Path) -> "StaticPipelineRegistry":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        specs: dict[str, StaticPipelineSpec] = {}
        for family, raw in (payload.get("pipelines") or {}).items():
            specs[_normalize(family)] = StaticPipelineSpec(
                event_family=_normalize(family),
                **raw,
            )
        return cls(specs)

    def get(self, event_family: str) -> StaticPipelineSpec | None:
        return self.specs.get(_normalize(event_family))


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
