"""FABLE semantic-to-physical planning and bounded Phase-4 search API."""

from .alternative_graph import (
    AlternativeBuildConfig,
    AlternativeGraphError,
    PhysicalAlternativeGraphBuilder,
)
from .artifact_catalog import ArtifactCatalog, ArtifactCatalogError
from .demand_compiler import DemandCompileContext, DemandCompileError, DemandCompiler
from .deployment import DeploymentGraph, DeploymentGraphError
from .models import *  # noqa: F401,F403
from .predicate_registry import (
    PredicateSchemaError,
    PredicateSchemaRegistry,
    default_predicate_registry,
)
from .provider_registry import (
    ProviderRegistry,
    ProviderRegistryError,
    default_provider_profiles,
)

__all__ = [
    "AlternativeBuildConfig",
    "AlternativeGraphError",
    "ArtifactCatalog",
    "ArtifactCatalogError",
    "DemandCompileContext",
    "DemandCompileError",
    "DemandCompiler",
    "DeploymentGraph",
    "DeploymentGraphError",
    "PhysicalAlternativeGraphBuilder",
    "PredicateSchemaError",
    "PredicateSchemaRegistry",
    "ProviderRegistry",
    "ProviderRegistryError",
    "default_predicate_registry",
    "default_provider_profiles",
]

# Phase 4: bounded representation-aware plan search.
from .beam_search import BeamSearchConfig, BoundedLabelPlanner, PlanSearchError
from .representation import RepresentationCompatibility
from .search_models import (
    BeamBoundaryTrace,
    FeasibilityFailure,
    LabelSearchState,
    NodeResourceFootprint,
    OracleComparison,
    OracleStatus,
    PlanSearchResult,
    PlanSearchTrace,
    PruneCode,
    PruningRecord,
)

__all__ += [
    "BeamSearchConfig",
    "BoundedLabelPlanner",
    "PlanSearchError",
    "RepresentationCompatibility",
    "BeamBoundaryTrace",
    "FeasibilityFailure",
    "LabelSearchState",
    "NodeResourceFootprint",
    "OracleComparison",
    "OracleStatus",
    "PlanSearchResult",
    "PlanSearchTrace",
    "PruneCode",
    "PruningRecord",
]

from .runtime_deployment import RuntimeDeploymentView
from fable.contracts.telemetry import RuntimeLinkUpdate, RuntimeNodeUpdate
__all__ += ["RuntimeDeploymentView", "RuntimeLinkUpdate", "RuntimeNodeUpdate"]
