"""External-input enumeration for physical alternatives."""

from __future__ import annotations

from datetime import datetime
from itertools import product

from fable.common.enums import ArtifactAccessMode
from fable.common.ids import deterministic_id
from fable.common.schemas import PredicateDemand
from fable.planning.artifact_catalog import ArtifactCatalog
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import ExternalInputKind, ExternalInputRealization, PrunedAlternative
from fable.planning.provider_registry import ProviderRegistry
from fable.planning.alternatives.config import AlternativeBuildConfig
from fable.planning.alternatives.internal import estimated_size


class AlternativeInputResolver:
    """Enumerate live/retained input assignments without choosing placement."""

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        artifact_catalog: ArtifactCatalog,
        deployment: DeploymentGraph,
        config: AlternativeBuildConfig,
    ) -> None:
        self.providers = provider_registry
        self.artifacts = artifact_catalog
        self.deployment = deployment
        self.config = config

    def _external_assignments(
            self,
            demand: PredicateDemand,
            chain_id: str,
            *,
            now: datetime,
        ) -> tuple[tuple[tuple[ExternalInputRealization, ...], ...], tuple[PrunedAlternative, ...]]:
            chain = self.providers.chain(chain_id)
            if (
                "same_source_pair_only" in chain.capability_tags
                and not self._bound_identity_roles_share_source(demand)
            ):
                return (), (
                    PrunedAlternative(
                        candidate_id=deterministic_id(
                            "candidate",
                            {"demand": demand.demand_id, "chain": chain_id, "same_source": False},
                        ),
                        demand_id=demand.demand_id,
                        chain_id=chain_id,
                        code="INCOMPATIBLE_EXTERNAL_INPUTS",
                        reason="same-source identity chain requires both bound identities to share one camera",
                    ),
                )
            choices: list[tuple[ExternalInputRealization, ...]] = []
            pruned: list[PrunedAlternative] = []
            for external in chain.external_inputs:
                candidates = self._input_candidates(
                    demand, chain_id, external.name, external.data_type, now=now
                )
                if external.optional:
                    candidates = (
                        *candidates,
                        ExternalInputRealization(
                            input_name=external.name,
                            data_type=external.data_type,
                            kind=ExternalInputKind.OMITTED_OPTIONAL,
                        ),
                    )
                if not candidates:
                    pruned.append(
                        PrunedAlternative(
                            candidate_id=deterministic_id(
                                "candidate",
                                {"demand": demand.demand_id, "chain": chain_id, "input": external.name},
                            ),
                            demand_id=demand.demand_id,
                            chain_id=chain_id,
                            code="MISSING_EXTERNAL_INPUT",
                            reason=f"no compatible source or artifact for {external.name}:{external.data_type}",
                        )
                    )
                    return (), tuple(pruned)
                choices.append(candidates)

            assignments: list[tuple[ExternalInputRealization, ...]] = []
            for combination in product(*choices):
                ordered = tuple(sorted(combination, key=lambda item: item.input_name))
                if not self._assignment_compatible(
                    ordered,
                    allow_same_source_pair=(
                        "allow_same_source_pair" in chain.capability_tags
                    ),
                ):
                    continue
                assignments.append(ordered)
                if len(assignments) >= self.config.max_external_assignments_per_chain:
                    break
            if assignments:
                retrospective = [
                    item
                    for item in assignments
                    if not any(part.kind == ExternalInputKind.LIVE_SOURCE for part in item)
                ]
                live = [
                    item
                    for item in assignments
                    if any(part.kind == ExternalInputKind.LIVE_SOURCE for part in item)
                ]
                balanced: list[tuple[ExternalInputRealization, ...]] = []
                for index in range(max(len(retrospective), len(live))):
                    if index < len(retrospective):
                        balanced.append(retrospective[index])
                    if index < len(live):
                        balanced.append(live[index])
                assignments = balanced
            if not assignments:
                pruned.append(
                    PrunedAlternative(
                        candidate_id=deterministic_id(
                            "candidate", {"demand": demand.demand_id, "chain": chain_id, "assignment": "none"}
                        ),
                        demand_id=demand.demand_id,
                        chain_id=chain_id,
                        code="INCOMPATIBLE_EXTERNAL_INPUTS",
                        reason="external inputs exist individually but cannot form one compatible assignment",
                    )
                )
            return tuple(assignments), tuple(pruned)

    def _input_candidates(
            self,
            demand: PredicateDemand,
            chain_id: str,
            input_name: str,
            data_type: str,
            *,
            now: datetime,
        ) -> tuple[ExternalInputRealization, ...]:
            definition = self.providers.data_type(data_type)
            eligible_source_ids = self._eligible_sources_for_input(
                demand, input_name
            )
            candidates: list[ExternalInputRealization] = []
            is_raw = definition.kind == "raw_sensor"
            if is_raw:
                for source in self.deployment.candidate_sources(
                    data_type=data_type,
                    interval=demand.event_time_interval,
                    eligible_source_ids=eligible_source_ids,
                    eligible_regions=demand.eligible_regions,
                    require_live=True,
                ):
                    candidates.append(
                        ExternalInputRealization(
                            input_name=input_name,
                            data_type=data_type,
                            kind=ExternalInputKind.LIVE_SOURCE,
                            node_id=source.node_id,
                            source_id=source.source_id,
                            bytes=estimated_size(data_type),
                            access_modes=(ArtifactAccessMode.LOCAL,),
                        )
                    )

            artifacts = self.artifacts.query(
                artifact_type=data_type,
                event_time_interval=demand.event_time_interval,
                required_access_modes=demand.hard_constraints.required_access_modes,
                now=now,
                require_interval_containment=True,
            )
            for artifact in artifacts:
                source_id = artifact.bindings.get("source_id")
                if eligible_source_ids and source_id:
                    if source_id not in eligible_source_ids:
                        continue
                kind = (
                    ExternalInputKind.RETAINED_ARTIFACT
                    if definition.kind in {"raw_sensor", "derived_evidence", "feature", "provider_state"}
                    else ExternalInputKind.DEPLOYMENT_ARTIFACT
                )
                candidates.append(
                    ExternalInputRealization(
                        input_name=input_name,
                        data_type=data_type,
                        kind=kind,
                        node_id=artifact.location.node_id,
                        source_id=source_id,
                        artifact_id=artifact.artifact_id,
                        bytes=artifact.bytes or estimated_size(data_type),
                        access_modes=artifact.access_modes,
                        expires_at=artifact.expires_at,
                    )
                )
            return tuple(
                sorted(
                    candidates,
                    key=lambda item: (
                        item.kind,
                        item.source_id or "",
                        str(item.artifact_id or ""),
                    ),
                )
            )

    def _eligible_sources_for_input(
        self,
        demand: PredicateDemand,
        input_name: str,
    ) -> tuple[str, ...]:
        """Specialize exact identity-pair inputs to their owning cameras."""

        if demand.semantic_predicate.predicate_id != "SAME_ENTITY":
            return demand.eligible_source_ids
        role = None
        if input_name.endswith("_a"):
            role = "left"
        elif input_name.endswith("_b"):
            role = "right"
        if role is None or role not in demand.bound_roles:
            return demand.eligible_source_ids
        entity_id = demand.bound_roles[role]
        namespace, separator, _ = entity_id.partition(":")
        if not separator:
            return demand.eligible_source_ids
        if namespace in self.deployment.sources:
            return (namespace,)
        matching = tuple(
            sorted(
                source.source_id
                for source in self.deployment.sources.values()
                if source.node_id == namespace and "vision" in source.modalities
            )
        )
        return matching if len(matching) == 1 else demand.eligible_source_ids

    def _bound_identity_roles_share_source(self, demand: PredicateDemand) -> bool:
        sources: set[str] = set()
        for role in ("left", "right"):
            entity_id = demand.bound_roles.get(role)
            if not entity_id:
                return False
            namespace, separator, _ = entity_id.partition(":")
            if not separator:
                return False
            if namespace in self.deployment.sources:
                sources.add(namespace)
                continue
            matches = {
                source.source_id
                for source in self.deployment.sources.values()
                if source.node_id == namespace and "vision" in source.modalities
            }
            if len(matches) != 1:
                return False
            sources.update(matches)
        return len(sources) == 1

    @staticmethod
    def _assignment_compatible(
        assignment: tuple[ExternalInputRealization, ...],
        *,
        allow_same_source_pair: bool = False,
    ) -> bool:
            by_name = {item.input_name: item for item in assignment}
            for suffix in ("a", "b", "left", "right"):
                matched = [
                    item
                    for name, item in by_name.items()
                    if name.endswith(f"_{suffix}") and item.kind != ExternalInputKind.OMITTED_OPTIONAL
                ]
                source_ids = {item.source_id for item in matched if item.source_id is not None}
                if len(source_ids) > 1:
                    return False
            video = by_name.get("video")
            for companion_name in (
                "calibration",
                "tracker_checkpoint",
                "reference",
                "zone",
            ):
                companion = by_name.get(companion_name)
                if video and companion and video.source_id and companion.source_id:
                    if video.source_id != companion.source_id:
                        return False
            video_a = by_name.get("video_a")
            video_b = by_name.get("video_b")
            if video_a and video_b and video_a.source_id and video_b.source_id:
                if (
                    video_a.source_id == video_b.source_id
                    and not allow_same_source_pair
                ):
                    return False
            return True


__all__ = ["AlternativeInputResolver"]
