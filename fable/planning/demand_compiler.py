"""Compile active semantic-frontier leaves into provider-independent demands."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
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
from fable.common.time import EventTimeInterval
from fable.semantic.compiled import CompiledSemanticGraph
from fable.semantic.models import DerivedFrontier
from fable.spatial.models import SpatialFilterMode, SpatialObservation, SpatialSensorBindings
from fable.spatial.transition_model import SiteSensorTransitionModel

from .deployment import DeploymentGraph
from .predicate_registry import PredicateSchemaError, PredicateSchemaRegistry
from .source_grounding import SourceGrounder


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
    """Translate an Active Frontier into provider-independent evidence needs."""
    def __init__(
        self,
        *,
        predicate_registry: PredicateSchemaRegistry,
        deployment: DeploymentGraph | None = None,
        source_grounder: SourceGrounder | None = None,
        spatial_model: SiteSensorTransitionModel | None = None,
        spatial_bindings: SpatialSensorBindings | None = None,
    ) -> None:
        self.predicate_registry = predicate_registry
        # A caller may inject a custom grounder for simulation/testing, but it
        # must not also provide a deployment whose source rules could disagree.
        if deployment is not None and source_grounder is not None:
            raise ValueError("provide deployment or source_grounder, not both")
        self.deployment = deployment
        self.source_grounder = source_grounder or (
            SourceGrounder(deployment) if deployment is not None else None
        )
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
        """Compile every active predicate in deterministic order.

        Graph, hypothesis, and frontier identity/version are validated first.
        Each active node delegates to :meth:`compile_node`; for a FOLLOWS node,
        an already-bound leader is copied from the hypothesis while an unbound
        follower remains an introducible role in the resulting demand.
        """
        context = context or DemandCompileContext()
        # These checks are the semantic/physical handoff fence. Planning from a
        # stale frontier could otherwise launch work for a retired branch.
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
        # Stable ordering makes plan traces and deterministic IDs reproducible
        # regardless of dictionary or graph traversal order.
        return tuple(sorted(demands, key=lambda demand: (str(demand.checkpoint_id), demand.graph_node_id)))

    def compile_node(
        self,
        *,
        graph: CompiledSemanticGraph,
        hypothesis: Hypothesis,
        frontier: DerivedFrontier,
        graph_node_id: str,
        context: DemandCompileContext,
        structural_universe: bool = False,
    ) -> PredicateDemand:
        """Compile one active predicate node into one ``PredicateDemand``.

        The method resolves current role bindings, derives introduce/validate
        behavior, attaches checkpoint time/deadline/cancellation information,
        and records required capabilities and acceptable result types. Source
        grounding is currently isolated in the latter half of this method and
        remains deployment-aware for serialized compatibility.

        ``structural_universe`` marks compilation performed before a concrete
        branch/identity is selected (static whole-event baselines). Consumer-
        only roles such as both sides of SAME_ENTITY cannot introduce a live
        binding.  For structural planning only, ground those roles to a typed
        symbolic value so the complete task graph can be costed. Runtime
        frontier compilation remains fail-closed and requires real bindings.
        """
        if graph_node_id not in frontier.snapshot.enabled_node_ids:
            raise DemandCompileError(f"graph node is not active in this frontier: {graph_node_id}")
        node = graph.nodes_by_id[graph_node_id]
        if node.predicate is None:
            raise DemandCompileError(f"active graph node is not an executable predicate: {graph_node_id}")
        try:
            # Schema validation is intentionally provider-independent. It
            # validates the logical vocabulary before any chain is considered.
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
                # Bound semantic variables become validation requirements for
                # providers; they are never silently reintroduced.
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
            try:
                role_modes[predicate_role.role_name] = _unbound_mode(
                    role_schema.binding_capabilities
                )
            except DemandCompileError as exc:
                if structural_universe:
                    unbound_roles.pop()
                    bound_roles[predicate_role.role_name] = (
                        f"__structural_unbound__:{variable}"
                    )
                    role_modes[predicate_role.role_name] = _bound_mode(
                        role_schema.binding_capabilities
                    )
                    continue
                raise DemandCompileError(
                    f"predicate {node.predicate.predicate_id} role "
                    f"{predicate_role.role_name}: {exc}"
                ) from exc

        forkable = tuple(
            role_name
            for role_name in predicate_schema.forkable_roles
            if role_name in set(unbound_roles)
        )
        eligible_sources = context.eligible_source_ids_by_node.get(graph_node_id)
        if eligible_sources is None:
            # Automatic source grounding consults deployment capabilities, not
            # provider implementations. Chain selection remains a later phase.
            eligible_sources = (
                self.source_grounder.infer_sources(
                    predicate_schema.required_capabilities
                )
                if self.source_grounder is not None
                else ()
            )
        # Some predicates consume identities whose namespace is local to the
        # sensor/tracker which introduced them.  Authored source-affinity roles
        # make that locality explicit.  A spatial prediction may rank where to
        # search next, but it must never move an exact tracker-local identity to
        # another camera where that identifier cannot exist.
        affinity_sources = {
            source_id
            for role_name in node.annotations.get("source_affinity_roles", ())
            if (binding := hypothesis.role_bindings.get(str(role_name))) is not None
            if (
                source_id := self._identity_source_id(
                    binding.canonical_entity_id,
                    eligible_sources=tuple(eligible_sources),
                )
            )
            is not None
        }
        # A concrete camera field-of-view is itself a sensor-local binding.
        # Once an earlier predicate grounds it, executing a successor on a
        # different camera cannot validate that role. Apply this invariant at
        # the demand boundary rather than relying on policy-specific fan-out.
        affinity_sources.update(
            source_id
            for entity_id in bound_roles.values()
            if entity_id.startswith("camera_fov:")
            if (
                source_id := self._identity_source_id(
                    entity_id,
                    eligible_sources=tuple(eligible_sources),
                )
            )
            is not None
        )
        if node.predicate.predicate_id == "SAME_ENTITY":
            # Both roles are already concrete identities. Their namespaces
            # identify the only cameras that can supply faithful comparison
            # evidence. Do not let an exact-pair demand fan out over unrelated
            # sensors and consume the bounded alternative budget.
            affinity_sources.update(
                source_id
                for entity_id in bound_roles.values()
                if (
                    source_id := self._identity_source_id(
                        entity_id,
                        eligible_sources=tuple(eligible_sources),
                    )
                )
                is not None
            )
        if len(affinity_sources) > 1:
            if node.predicate.predicate_id != "SAME_ENTITY":
                raise DemandCompileError(
                    f"predicate {node.predicate.predicate_id} has incompatible "
                    f"source-affinity bindings: {sorted(affinity_sources)}"
                )
            if eligible_sources and not affinity_sources.issubset(set(eligible_sources)):
                raise DemandCompileError(
                    "SAME_ENTITY identity sources are outside explicit source "
                    f"eligibility: {sorted(affinity_sources)}"
                )
            eligible_sources = tuple(sorted(affinity_sources))
        if affinity_sources:
            if len(affinity_sources) == 1:
                affinity_source = next(iter(affinity_sources))
                if eligible_sources and affinity_source not in set(eligible_sources):
                    raise DemandCompileError(
                        f"source-affinity binding requires {affinity_source}, which is "
                        "outside explicit source eligibility"
                    )
                eligible_sources = (affinity_source,)
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
        event_time_interval = checkpoint.event_time_interval
        if node.annotations.get("execution_mode") == "retrospective":
            anchor_spec = node.annotations.get("retrospective_anchor") or {}
            authored_keys = anchor_spec.get("trigger_authored_keys") or ()
            if anchor_spec.get("trigger_authored_key"):
                authored_keys = (*authored_keys, anchor_spec["trigger_authored_key"])
            anchor_intervals = [
                interval
                for authored_key in authored_keys
                if authored_key in graph.nodes_by_key
                for interval in hypothesis.node_states[
                    graph.nodes_by_key[authored_key].node_id
                ].event_time_intervals
            ]
            if anchor_intervals:
                anchor_time = max(interval.end for interval in anchor_intervals)
                start = anchor_time - timedelta(
                    milliseconds=int(node.annotations.get("lookback_ms", 0))
                )
                if anchor_spec.get("clamp_to_hypothesis_window", True):
                    start = max(start, hypothesis.event_time_window.start)
                event_time_interval = EventTimeInterval(
                    start=start,
                    end=(
                        max(anchor_time, checkpoint.event_time_interval.end)
                        if node.annotations.get("catch_up_and_follow", False)
                        else anchor_time
                    ),
                )

        retrospective_context = None
        if node.annotations.get("execution_mode") == "retrospective":
            anchor_spec = node.annotations.get("retrospective_anchor") or {}
            authored_keys = tuple(anchor_spec.get("trigger_authored_keys") or ())
            if anchor_spec.get("trigger_authored_key"):
                authored_keys = (*authored_keys, anchor_spec["trigger_authored_key"])
            resolved_key = next(
                (
                    key
                    for key in authored_keys
                    if key in graph.nodes_by_key
                    and hypothesis.node_states[
                        graph.nodes_by_key[key].node_id
                    ].event_time_intervals
                ),
                authored_keys[0] if authored_keys else None,
            )
            resolved_intervals = (
                hypothesis.node_states[
                    graph.nodes_by_key[resolved_key].node_id
                ].event_time_intervals
                if resolved_key is not None and resolved_key in graph.nodes_by_key
                else ()
            )
            anchor_time = (
                max(interval.end for interval in resolved_intervals)
                if resolved_intervals
                else event_time_interval.end
            )
            retrospective_context = {
                "anchor_node_id": (
                    graph.nodes_by_key[resolved_key].node_id
                    if resolved_key is not None
                    else graph_node_id
                ),
                "anchor_authored_key": resolved_key or node.authored_key,
                "anchor_event_time": anchor_time.isoformat(),
                "anchor_kind": anchor_spec.get("kind", "trigger_node_end"),
                "lookback_ms": int(node.annotations.get("lookback_ms", 0)),
                "catch_up_and_follow": bool(
                    node.annotations.get("catch_up_and_follow", False)
                ),
            }
            if structural_universe:
                retrospective_context["structural_template"] = anchor_spec

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
            event_time_interval=event_time_interval,
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
            retrospective_context=retrospective_context,
            binding_policy=DemandBindingPolicy(
                role_modes=role_modes,
                forkable_roles=forkable,
                fork_unit=predicate_schema.fork_unit,
            ),
            cancellation_scope=cancellation_scope,
        )

    def _identity_source_id(
        self,
        canonical_entity_id: str,
        *,
        eligible_sources: tuple[str, ...],
    ) -> str | None:
        if canonical_entity_id.startswith("camera_fov:"):
            camera = canonical_entity_id.partition(":")[2]
            matches = tuple(
                source_id
                for source_id in eligible_sources
                if (source := self.deployment.sources.get(source_id)) is not None
                and "vision" in source.modalities
                and camera
                in {
                    source.node_id,
                    source.node_id.removeprefix("dvpg_gq_"),
                    "orin"
                    + source.node_id.removeprefix("dvpg_gq_orin_"),
                }
            )
            return matches[0] if len(matches) == 1 else None
        if self.source_grounder is None:
            namespace, separator, _ = canonical_entity_id.partition(":")
            return namespace if separator and namespace in set(eligible_sources) else None
        return self.source_grounder.identity_source_id(
            canonical_entity_id, eligible_sources=eligible_sources
        )

    def compile(
        self,
        *,
        graph: CompiledSemanticGraph,
        hypothesis: Hypothesis,
        frontier: DerivedFrontier,
        context: DemandCompileContext | None = None,
    ) -> tuple[PredicateDemand, ...]:
        """Architecture-level alias for :meth:`compile_frontier`."""

        return self.compile_frontier(
            graph=graph,
            hypothesis=hypothesis,
            frontier=frontier,
            context=context,
        )

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
