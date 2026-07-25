"""Compile active semantic-frontier leaves into provider-independent demands."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from uuid import UUID

from pydantic import Field

from fable.common.base import FableModel
from fable.common.enums import (
    ArtifactAccessMode,
    BindingCapability,
    CancellationScope,
    CheckpointKind,
)
from fable.common.schemas import (
    ContinuationRequirement,
    DataMovementConstraints,
    DemandBindingPolicy,
    Hypothesis,
    PredicateDemand,
    SourcePreference,
)
from fable.semantic.compiled import CompiledSemanticGraph
from fable.semantic.models import DerivedFrontier
from fable.spatial.models import SpatialFilterMode, SpatialObservation, SpatialSensorBindings
from fable.spatial.transition_model import SiteSensorTransitionModel

from .deployment import DeploymentGraph
from .predicate_registry import PredicateSchemaError, PredicateSchemaRegistry


class DemandCompileError(ValueError):
    """Raised when an active semantic leaf cannot be specialized safely."""


class DemandCompileContext(FableModel):
    """Deployment- or request-specific grounding not encoded in the semantic graph."""

    eligible_source_ids_by_node: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    eligible_regions_by_node: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    required_continuations_by_checkpoint: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    required_access_modes_by_node: dict[str, tuple[ArtifactAccessMode, ...]] = Field(default_factory=dict)
    raw_data_must_remain_local: bool = True
    allowed_node_ids: tuple[str, ...] = ()
    allowed_regions: tuple[str, ...] = ()
    maximum_transfer_bytes: int | None = Field(default=None, ge=0)
    cancellation_scope_by_node: dict[str, CancellationScope] = Field(default_factory=dict)
    spatial_observation_by_node: dict[str, SpatialObservation] = Field(default_factory=dict)


class DemandCompiler:
    def __init__(
        self,
        *,
        predicate_registry: PredicateSchemaRegistry,
        deployment: DeploymentGraph,
        spatial_model: SiteSensorTransitionModel | None = None,
        spatial_bindings: SpatialSensorBindings | None = None,
    ) -> None:
        self.predicate_registry = predicate_registry
        self.deployment = deployment
        self.spatial_model = spatial_model
        self.spatial_bindings = spatial_bindings or SpatialSensorBindings()

    def compile_frontier(
        self,
        *,
        graph: CompiledSemanticGraph,
        hypothesis: Hypothesis,
        frontier: DerivedFrontier,
        context: DemandCompileContext | None = None,
    ) -> tuple[PredicateDemand, ...]:
        context = context or DemandCompileContext()
        if frontier.snapshot.hypothesis_id != hypothesis.hypothesis_id:
            raise DemandCompileError("frontier and hypothesis identifiers do not match")
        if frontier.snapshot.hypothesis_version != hypothesis.version:
            raise DemandCompileError("frontier is stale for the supplied hypothesis version")
        if frontier.snapshot.graph_hash != graph.graph.graph_hash:
            raise DemandCompileError("frontier graph hash does not match compiled graph")

        demands = [
            self.compile_node(
                graph=graph,
                hypothesis=hypothesis,
                frontier=frontier,
                graph_node_id=node_id,
                context=context,
            )
            for node_id in frontier.snapshot.enabled_node_ids
        ]
        return tuple(sorted(demands, key=lambda demand: (str(demand.checkpoint_id), demand.graph_node_id)))

    def compile_node(
        self,
        *,
        graph: CompiledSemanticGraph,
        hypothesis: Hypothesis,
        frontier: DerivedFrontier,
        graph_node_id: str,
        context: DemandCompileContext,
    ) -> PredicateDemand:
        if graph_node_id not in frontier.snapshot.enabled_node_ids:
            raise DemandCompileError(f"graph node is not active in this frontier: {graph_node_id}")
        node = graph.nodes_by_id[graph_node_id]
        if node.predicate is None:
            raise DemandCompileError(f"active graph node is not an executable predicate: {graph_node_id}")
        try:
            predicate_schema = self.predicate_registry.validate(node.predicate)
        except PredicateSchemaError as exc:
            raise DemandCompileError(str(exc)) from exc
        checkpoint = frontier.checkpoint_for_node(graph_node_id)

        graph_role_names = {role.role_name for role in graph.graph.roles}
        bound_roles: dict[str, str] = {}
        unbound_roles: list[str] = []
        role_modes: dict[str, BindingCapability] = {}

        schema_roles = {role.role_name: role for role in predicate_schema.roles}
        for predicate_role in node.predicate.roles:
            role_schema = schema_roles[predicate_role.role_name]
            variable = predicate_role.variable
            binding = hypothesis.role_bindings.get(variable)
            if binding is not None:
                bound_roles[predicate_role.role_name] = binding.canonical_entity_id
                role_modes[predicate_role.role_name] = _bound_mode(role_schema.binding_capabilities)
                continue
            if variable not in graph_role_names:
                # Constants such as ``mobile_node`` are grounded authored values,
                # not unbound entity variables.
                bound_roles[predicate_role.role_name] = variable
                role_modes[predicate_role.role_name] = _bound_mode(role_schema.binding_capabilities)
                continue
            unbound_roles.append(predicate_role.role_name)
            role_modes[predicate_role.role_name] = _unbound_mode(role_schema.binding_capabilities)

        forkable = tuple(
            role_name
            for role_name in predicate_schema.forkable_roles
            if role_name in set(unbound_roles)
        )
        eligible_sources = context.eligible_source_ids_by_node.get(graph_node_id)
        if eligible_sources is None:
            eligible_sources = self._infer_sources(predicate_schema.required_capabilities)
        eligible_regions = context.eligible_regions_by_node.get(graph_node_id, ())

        source_preferences: tuple[SourcePreference, ...] = ()
        spatial_prediction_id: str | None = None
        spatial_observation = context.spatial_observation_by_node.get(graph_node_id)
        if spatial_observation is not None:
            if self.spatial_model is None:
                raise DemandCompileError(
                    "spatial observation was supplied but no site transition model is configured"
                )
            prediction = self.spatial_model.predict(
                spatial_observation, bindings=self.spatial_bindings
            )
            spatial_prediction_id = prediction.prediction_id
            predicted_sources = prediction.recommended_source_ids
            if spatial_observation.filter_mode == SpatialFilterMode.LIMIT_TO_PREDICTED:
                if not predicted_sources:
                    raise DemandCompileError(
                        "hard spatial filtering requested but no predicted runtime sources were mapped"
                    )
                if eligible_sources:
                    eligible_sources = tuple(
                        source_id
                        for source_id in eligible_sources
                        if source_id in set(predicted_sources)
                    )
                else:
                    eligible_sources = predicted_sources
                if not eligible_sources:
                    raise DemandCompileError(
                        "spatial prediction and explicit source eligibility have no overlap"
                    )

            eligible_set = set(eligible_sources)
            preferences: list[SourcePreference] = []
            for group in prediction.groups:
                for sensor_id in group.sensor_ids:
                    for source_id in self.spatial_bindings.sources(
                        sensor_id, spatial_observation.active_deployment_id
                    ):
                        if eligible_set and source_id not in eligible_set:
                            continue
                        preferences.append(
                            SourcePreference(
                                source_id=source_id,
                                priority_rank=group.group_rank,
                                confidence=group.confidence_score,
                                reason=group.reason,
                                sensor_id=sensor_id,
                                observation_group=group.group_rank,
                                corridor_id=prediction.corridor_id,
                                prediction_id=prediction.prediction_id,
                            )
                        )
            # A source can occur in more than one qualitative rule/group. Keep
            # its strongest (lowest-rank, highest-confidence) preference.
            by_source: dict[str, SourcePreference] = {}
            for item in preferences:
                previous = by_source.get(item.source_id)
                if previous is None or (
                    item.priority_rank, -item.confidence, item.source_id
                ) < (
                    previous.priority_rank, -previous.confidence, previous.source_id
                ):
                    by_source[item.source_id] = item
            source_preferences = tuple(
                sorted(
                    by_source.values(),
                    key=lambda item: (item.priority_rank, -item.confidence, item.source_id),
                )
            )

        explicit_continuations = list(checkpoint.required_artifact_types_after_resolution)
        explicit_continuations.extend(
            context.required_continuations_by_checkpoint.get(str(checkpoint.checkpoint_id), ())
        )
        continuation_requirements = tuple(
            ContinuationRequirement(
                artifact_type=artifact_type,
                required_until=hypothesis.deadline.latest_useful_completion,
                required_bindings=tuple(sorted(hypothesis.role_bindings)),
            )
            for artifact_type in sorted(set(explicit_continuations))
        )
        required_access = context.required_access_modes_by_node.get(graph_node_id, ())
        hard_constraints = DataMovementConstraints(
            raw_data_must_remain_local=context.raw_data_must_remain_local,
            allowed_node_ids=context.allowed_node_ids,
            allowed_regions=context.allowed_regions,
            maximum_transfer_bytes=context.maximum_transfer_bytes,
            required_access_modes=required_access,
        )

        cancellation_scope = context.cancellation_scope_by_node.get(
            graph_node_id,
            CancellationScope.BRANCH
            if checkpoint.kind == CheckpointKind.OR_RESOLUTION
            else CancellationScope.HYPOTHESIS,
        )
        return PredicateDemand(
            request_id=hypothesis.request_id,
            graph_hash=hypothesis.graph_hash,
            hypothesis_id=hypothesis.hypothesis_id,
            hypothesis_version=hypothesis.version,
            frontier_id=frontier.snapshot.frontier_id,
            checkpoint_id=checkpoint.checkpoint_id,
            graph_node_id=graph_node_id,
            semantic_predicate=node.predicate,
            bound_roles=bound_roles,
            unbound_roles=tuple(unbound_roles),
            event_time_interval=checkpoint.event_time_interval,
            deadline=hypothesis.deadline,
            eligible_source_ids=tuple(eligible_sources),
            eligible_regions=tuple(eligible_regions),
            source_preferences=source_preferences,
            spatial_prediction_id=spatial_prediction_id,
            required_capabilities=predicate_schema.required_capabilities,
            required_input_artifact_types=predicate_schema.required_input_artifact_types,
            acceptable_output_types=predicate_schema.acceptable_output_types,
            hard_constraints=hard_constraints,
            continuation_requirements=continuation_requirements,
            desired_continuation_types=predicate_schema.default_continuation_types,
            binding_policy=DemandBindingPolicy(
                role_modes=role_modes,
                forkable_roles=forkable,
                fork_unit=predicate_schema.fork_unit,
            ),
            cancellation_scope=cancellation_scope,
        )

    def _infer_sources(self, capabilities: tuple[str, ...]) -> tuple[str, ...]:
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


def _bound_mode(capabilities: tuple[BindingCapability, ...]) -> BindingCapability:
    for candidate in (
        BindingCapability.VALIDATE,
        BindingCapability.CONSUME,
        BindingCapability.INTRODUCE_OR_VALIDATE,
        BindingCapability.OBSERVE_ONLY,
    ):
        if candidate in capabilities:
            return candidate
    raise DemandCompileError(
        f"bound role has no VALIDATE, CONSUME, INTRODUCE_OR_VALIDATE, or OBSERVE_ONLY capability: {capabilities}"
    )


def _unbound_mode(capabilities: tuple[BindingCapability, ...]) -> BindingCapability:
    for candidate in (
        BindingCapability.INTRODUCE,
        BindingCapability.INTRODUCE_OR_VALIDATE,
        BindingCapability.OBSERVE_ONLY,
        BindingCapability.AGGREGATE,
    ):
        if candidate in capabilities:
            return candidate
    raise DemandCompileError(f"unbound role has no introduction/observation capability: {capabilities}")
