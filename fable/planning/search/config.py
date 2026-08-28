"""Configuration for bounded physical-plan search."""

from pydantic import Field

from fable.common.base import FableModel


class BeamSearchConfig(FableModel):
    beam_width: int = Field(default=8, ge=1)
    fallback_count: int = Field(default=2, ge=0)
    minimum_quality_score: float = Field(default=0.0, ge=0, le=1)
    minimum_quality_by_predicate: dict[str, float] = Field(default_factory=dict)
    near_expiry_horizon_ms: int = Field(default=5_000, ge=0)
    require_declared_binding_capabilities: bool = False
    run_oracle: bool = True
    oracle_max_combinations: int = Field(default=50_000, ge=1)
