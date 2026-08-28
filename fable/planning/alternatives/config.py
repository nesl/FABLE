"""Configuration for physical-alternative enumeration."""

from pydantic import Field

from fable.common.base import FableModel


class AlternativeBuildConfig(FableModel):
    max_external_assignments_per_chain: int = Field(default=32, ge=1)
    max_placement_variants_per_assignment: int = Field(default=24, ge=1)
    max_total_alternatives: int = Field(default=128, ge=1)
    max_alternatives_per_chain: int = Field(default=32, ge=1)
    max_candidate_nodes_per_step: int = Field(default=2, ge=1)
    allow_remote_reference: bool = True
    allow_transfer: bool = True
    # Real deployments without an artifact router may still use explicitly
    # declared same-node broker topics, but must not enumerate cross-node
    # intermediate placements that the executor cannot realize.
    require_internal_step_colocation: bool = False
    default_queue_ms: int = Field(default=0, ge=0)
    node_execution_time_multipliers: dict[str, float] = Field(default_factory=dict)
    node_queue_delay_ms: dict[str, int] = Field(default_factory=dict)


__all__ = ["AlternativeBuildConfig"]
