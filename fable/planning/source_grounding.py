"""Deployment-aware grounding of semantic evidence needs to concrete sources.

This compatibility component isolates *where* evidence may come from. Demand
Compiler still calls it today, but the semantic binding/time logic no longer
owns camera enumeration or sensor-local identity namespace parsing. It can move
behind PhysicalPlanner later without changing these rules.
"""

from __future__ import annotations

from .deployment import DeploymentGraph


class SourceGrounder:
    """Resolve candidate sources and sensor-local identity namespaces."""

    def __init__(self, deployment: DeploymentGraph) -> None:
        self.deployment = deployment

    def infer_sources(self, capabilities: tuple[str, ...]) -> tuple[str, ...]:
        wants_audio = "audio" in capabilities
        selected = []
        for source in self.deployment.sources.values():
            if not source.available:
                continue
            if wants_audio and "audio" not in source.modalities:
                continue
            if not wants_audio and "vision" not in source.modalities:
                continue
            selected.append(source.source_id)
        return tuple(sorted(selected))

    def identity_source_id(
        self,
        canonical_entity_id: str,
        *,
        eligible_sources: tuple[str, ...],
    ) -> str | None:
        """Resolve the source namespace of a sensor-local canonical identity."""

        if canonical_entity_id.startswith("camera_fov:"):
            camera_owner = canonical_entity_id.removeprefix("camera_fov:")
            if camera_owner in self.deployment.sources:
                return camera_owner
            node_sources = tuple(
                source.source_id
                for source in self.deployment.sources.values()
                if source.node_id == camera_owner and "vision" in source.modalities
            )
            if len(node_sources) == 1:
                return node_sources[0]
            return None
        namespace, separator, _ = canonical_entity_id.partition(":")
        if not separator:
            return None
        if namespace in self.deployment.sources or namespace in set(eligible_sources):
            return namespace
        node_sources = tuple(
            source.source_id
            for source in self.deployment.sources.values()
            if source.node_id == namespace and "vision" in source.modalities
        )
        if len(node_sources) == 1:
            return node_sources[0]
        return None


__all__ = ["SourceGrounder"]
