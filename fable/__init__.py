"""Minimal FABLE rebuild."""

from .providers.predicate_result import PredicateMatch
from .language import (
    Event,
    EventCompilationError,
    Expr,
    compile_event,
    load_and_compile_event,
    load_event,
    load_predicates,
    parse_event,
    walk_pattern,
)
from .execution import IdentityResolver, LocalRunner
from .runtime import (
    ActiveFrontier,
    CEInstance,
    CEInstanceManager,
    FrontierItem,
    PatternPath,
    derive_continuation_frontier,
    derive_discovery_frontier,
    is_complete,
    is_failed,
)

__all__ = [
    "ActiveFrontier",
    "CEInstance",
    "CEInstanceManager",
    "Event",
    "EventCompilationError",
    "Expr",
    "FrontierItem",
    "IdentityResolver",
    "LocalRunner",
    "PatternPath",
    "PredicateMatch",
    "compile_event",
    "derive_continuation_frontier",
    "derive_discovery_frontier",
    "is_complete",
    "is_failed",
    "load_and_compile_event",
    "load_event",
    "load_predicates",
    "parse_event",
    "walk_pattern",
]

__version__ = "0.10.0"
