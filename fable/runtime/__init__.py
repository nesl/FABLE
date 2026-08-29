"""Candidate CE instances and semantic frontiers."""

from .ce_instance import CEInstance, PatternPath
from .frontier import (
    ActiveFrontier,
    FrontierItem,
    advance_instance,
    derive_continuation_frontier,
    derive_discovery_frontier,
    is_complete,
    is_failed,
    seed_instance_from_match,
)
from .instance_manager import CEInstanceManager

__all__ = [
    "ActiveFrontier",
    "CEInstance",
    "CEInstanceManager",
    "FrontierItem",
    "PatternPath",
    "advance_instance",
    "derive_continuation_frontier",
    "derive_discovery_frontier",
    "is_complete",
    "is_failed",
    "seed_instance_from_match",
]
