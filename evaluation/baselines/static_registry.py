from __future__ import annotations

from pathlib import Path
import os

import yaml
from pydantic import Field

from fable.common.base import FableModel


# Durable B1 placement artifacts predate the redesigned provider catalog.
# Resolve only explicitly versioned aliases; unknown chain names continue to
# fail closed instead of silently selecting a different physical pipeline.
STATIC_CHAIN_ALIASES = {
    "recover_vehicle_from_local_segments": "recover_vehicle_before_audio_event",
    "same_entity_cross_camera_reid": "follows_cross_camera_reid",
}


def resolve_static_chain_id(chain_id: str) -> str:
    return STATIC_CHAIN_ALIASES.get(chain_id, chain_id)


class StaticPipelineSpec(FableModel):
    event_family: str = Field(min_length=1)
    preferred_chain_ids: tuple[str, ...]
    fixed_sensor_policy: str = "all_replay_supported_orin"
    fixed_representation_policy: str = ""


class StaticPlacementSpec(FableModel):
    """Frozen authored placement used by B0/B1 evaluation execution."""

    experiment_id: str = ""
    exemplar_trace_id: str = ""
    exemplar_experiment_id: str = ""
    allowed_chain_ids: tuple[str, ...] = ()
    allowed_provider_ids: tuple[str, ...] = ()
    allowed_node_ids: tuple[str, ...] = ()
    allowed_source_ids: tuple[str, ...] = ()
    allowed_branch_ids: tuple[str, ...] = ()
    allowed_chain_node_ids: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    fanout_allowed: bool = False
    adaptation_allowed: bool = False


class StaticPipelineRegistry:
    def __init__(
        self,
        specs: dict[str, StaticPipelineSpec],
        *,
        placement_templates: dict[str, StaticPlacementSpec] | None = None,
        trace_placements: dict[str, StaticPlacementSpec] | None = None,
    ) -> None:
        self.specs = specs
        self.placement_templates = placement_templates or {}
        self.trace_placements = trace_placements or {}

    @classmethod
    def load(cls, path: str | Path) -> "StaticPipelineRegistry":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        specs: dict[str, StaticPipelineSpec] = {}
        for family, raw in (payload.get("pipelines") or {}).items():
            specs[_normalize(family)] = StaticPipelineSpec(
                event_family=_normalize(family),
                **raw,
            )
        def placement(raw: dict) -> StaticPlacementSpec:
            # Provenance/hash fields are audit metadata, not runtime contract
            # fields. Select the allowlisted placement keys explicitly.
            fields = StaticPlacementSpec.model_fields
            return StaticPlacementSpec.model_validate(
                {key: value for key, value in raw.items() if key in fields}
            )

        templates = {
            _normalize(str(key)): placement(raw)
            for key, raw in (payload.get("placement_templates") or {}).items()
        }
        traces = {
            str(key): placement(raw)
            for key, raw in (payload.get("trace_placements") or {}).items()
        }
        return cls(specs, placement_templates=templates, trace_placements=traces)

    def get(self, event_family: str) -> StaticPipelineSpec | None:
        return self.specs.get(_normalize(event_family))

    def get_placement(
        self, event_family: str, *, trace_id: str = ""
    ) -> StaticPlacementSpec | None:
        if trace_id and trace_id in self.trace_placements:
            return self.trace_placements[trace_id]
        return self.placement_templates.get(_normalize(event_family))


def static_pipeline_registry_path() -> Path:
    """Resolve the experiment-scoped registry without hidden global state."""

    configured = os.environ.get("FABLE_STATIC_PIPELINE_REGISTRY")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "manifests/baselines/static_pipelines.yaml"


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
