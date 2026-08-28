"""Provider-output adaptation boundary for distributed node agents.

The distributed runtime owns transport and lifecycle.  Provider-specific replay
payloads are interpreted by adapters injected through this interface, keeping
``NodeAgent`` independent of vehicle/audio/vision model classes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fable.common.schemas import ArtifactRef, PredicateResult

from .models import ActivateProviderCommand, ReplayOutputAdapter


@dataclass(frozen=True)
class AdaptedProviderEvidence:
    """Canonical evidence produced by a provider-output adapter."""

    occurrence_id: str
    event_interval: Any
    introduced_bindings: dict[str, str]
    confidence: float
    # Concrete provider payloads may retain the originating sensor even when
    # they are consumed through a replay/archive worker on another node.  Do
    # not rewrite that provenance to the command's eligible source: identity
    # affinity and successor placement depend on the actual observation
    # source.
    source_ids: tuple[str, ...] = ()


class ProviderOutputAdapter(Protocol):
    """Translate one provider-native payload for one active demand."""

    def __call__(
        self,
        command: ActivateProviderCommand,
        document: Any,
    ) -> AdaptedProviderEvidence | None: ...


class ProviderOutputAdapterRegistry:
    """Registry of provider/testbed adapters keyed by transport adapter kind."""

    def __init__(self) -> None:
        self._adapters: dict[ReplayOutputAdapter, ProviderOutputAdapter] = {}

    def register(
        self,
        key: ReplayOutputAdapter,
        adapter: ProviderOutputAdapter,
    ) -> None:
        if key == ReplayOutputAdapter.NONE:
            raise ValueError("NONE is not an executable output adapter")
        self._adapters[key] = adapter

    def adapt(
        self,
        key: ReplayOutputAdapter,
        command: ActivateProviderCommand,
        document: Any,
    ) -> AdaptedProviderEvidence | None:
        adapter = self._adapters.get(key)
        if adapter is None:
            return None
        return adapter(command, document)

    def supports(self, key: ReplayOutputAdapter) -> bool:
        return key in self._adapters



@dataclass(frozen=True)
class ReferenceExecutionContext:
    """Host services exposed to a synthetic/reference runtime adapter."""

    node_id: str
    artifact_dir: Path
    is_active: Callable[[ActivateProviderCommand], bool]


@dataclass(frozen=True)
class ReferenceExecutionOutcome:
    artifacts: tuple[ArtifactRef, ...] = ()
    result: PredicateResult | None = None


class ReferenceRuntimeAdapter(Protocol):
    """Evaluation-only implementation of a REFERENCE provider invocation."""

    def execute(
        self,
        command: ActivateProviderCommand,
        context: ReferenceExecutionContext,
    ) -> ReferenceExecutionOutcome | None: ...

__all__ = [
    "AdaptedProviderEvidence",
    "ProviderOutputAdapter",
    "ProviderOutputAdapterRegistry",
    "ReferenceExecutionContext",
    "ReferenceExecutionOutcome",
    "ReferenceRuntimeAdapter",
]
