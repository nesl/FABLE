"""Low-level direct-plan builders used by replay/debug integration tools.

These bypass semantic request compilation intentionally and exercise the normal
scheduler/distributed execution contracts.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fable.common.enums import (
    BindingCapability,
    ExecutionInputKind,
    ExecutionMode,
    PlanStatus,
    ResultKind,
)
from fable.common.ids import deterministic_id, uuid7
from fable.common.schemas import (
    DataMovementConstraints,
    DemandBindingPolicy,
    ExecutionInput,
    ExecutionPlan,
    PhysicalPlanLabel,
    PlanCost,
    PlanStep,
    PredicateDemand,
    PredicateRole,
    ResourceReservation,
    SemanticPredicate,
)
from fable.common.time import DeadlineSpec, EventTimeInterval, ensure_utc, utc_now
from fable.planning.provider_registry import ProviderRegistry
from fable.scheduling.models import PlanCandidate, TaskSchedulingPolicy


def build_replay_audio_candidate(
    *,
    provider_registry: ProviderRegistry,
    node_id: str,
    source_id: str,
    event_interval: EventTimeInterval,
    label: str = "loud_audio",
    request_id: str = "replay_audio_demo",
    node_class: str = "sensor",
    deadline_seconds: float = 300.0,
    now: datetime | None = None,
) -> PlanCandidate:
    """Create one live AUDIO_EVENT demand for the existing replay audio topic."""

    observed_now = ensure_utc(now or utc_now())
    deadline = observed_now + timedelta(seconds=deadline_seconds)
    demand = PredicateDemand(
        request_id=request_id,
        graph_hash="sha256:" + "6" * 64,
        hypothesis_id=uuid7(),
        hypothesis_version=1,
        frontier_id=uuid7(),
        checkpoint_id=uuid7(),
        graph_node_id="replay_audio_event",
        semantic_predicate=SemanticPredicate(
            predicate_id="AUDIO_EVENT",
            roles=(
                PredicateRole(role_name="zone", variable="replay_zone", entity_type="zone"),
            ),
            parameters={"label": label},
            result_kind=ResultKind.INSTANT_MATCH,
        ),
        bound_roles={"zone": "replay_zone"},
        event_time_interval=event_interval,
        deadline=DeadlineSpec(latest_useful_completion=deadline),
        eligible_source_ids=(source_id,),
        acceptable_output_types=("predicate_match.v1",),
        hard_constraints=DataMovementConstraints(
            raw_data_must_remain_local=True,
            allowed_node_ids=(node_id,),
        ),
    )

    profile = provider_registry.profile("audio_event_classifier", node_class)
    alternative_id = deterministic_id(
        "alt",
        {
            "demand_id": demand.demand_id,
            "node": node_id,
            "source": source_id,
            "label": label,
        },
        length=32,
    )
    external = ExecutionInput(
        name="audio",
        data_type="audio_segment.v1",
        kind=ExecutionInputKind.LIVE_SOURCE,
        node_id=node_id,
        source_id=source_id,
        bytes=0,
    )
    step = PlanStep(
        step_id=f"{alternative_id}:classify",
        provider_id="audio_event_classifier",
        node_id=node_id,
        demand_id=demand.demand_id,
        alternative_id=alternative_id,
        chain_id="detect_audio_event",
        execution_mode=ExecutionMode.LIVE,
        inputs=(external,),
        input_data_types=("audio_segment.v1",),
        output_data_types=("audio_event_set.v1", "predicate_match.v1"),
        parameters=(("label", label),),
        cpu_cores=profile.cpu_cores,
        memory_mb=profile.memory_mb,
        gpu_memory_mb=profile.gpu_memory_mb,
        quality_score=profile.quality_score,
        estimated_startup_ms=profile.startup_ms,
        estimated_execution_ms=profile.execution_ms,
    )
    completion_ms = profile.startup_ms + profile.execution_ms
    slack_ms = int((deadline - observed_now).total_seconds() * 1000) - completion_ms
    cost = PlanCost(
        predicted_completion_ms=completion_ms,
        deadline_slack_ms=slack_ms,
        startup_cost_ms=profile.startup_ms,
        resource_cost_units=(
            profile.cpu_cores
            + profile.memory_mb / 1024.0
            + 2.0 * profile.gpu_memory_mb / 1024.0
        ),
        transfer_bytes=0,
    )
    label_contract = PhysicalPlanLabel(
        checkpoint_id=demand.checkpoint_id,
        covered_demand_ids=(demand.demand_id,),
        steps=(step,),
        continuation_output_types=("audio_event_set.v1",),
        cost=cost,
        hard_constraints_satisfied=True,
        quality_floor_satisfied=True,
        feasibility_reasons=("replay audio provider is local to its source",),
    )
    plan = ExecutionPlan(
        label_id=label_contract.label_id or "",
        checkpoint_id=demand.checkpoint_id,
        demand_ids=(demand.demand_id,),
        steps=(step,),
        reservations=(
            ResourceReservation(
                node_id=node_id,
                cpu_cores=profile.cpu_cores,
                memory_mb=profile.memory_mb,
                gpu_memory_mb=profile.gpu_memory_mb,
            ),
        ),
        status=PlanStatus.CANDIDATE,
        expires_at=deadline,
    )
    return PlanCandidate(
        plan=plan,
        demands=(demand,),
        task_policy=TaskSchedulingPolicy(request_id=request_id),
        predicted_completion_ms=completion_ms,
        startup_cost_ms=profile.startup_ms,
        incremental_resource_cost_units=cost.resource_cost_units,
        transfer_bytes=0,
        created_at=observed_now,
    )


def build_replay_vehicle_candidate(
    *,
    provider_registry: ProviderRegistry,
    node_id: str,
    source_id: str,
    event_interval: EventTimeInterval,
    predicate_id: str = "PASSES",
    request_id: str = "replay_vehicle_demo",
    node_class: str = "sensor",
    leader_id: str | None = None,
    reference_id: str = "camera_a_gate",
    deadline_seconds: float = 300.0,
    now: datetime | None = None,
) -> PlanCandidate:
    """Build one direct vehicle-predicate candidate for replay integration.

    This helper intentionally bypasses request compilation so the Phase-7 MQTT
    adapter can be exercised in isolation. Normal FABLE operation still derives
    the same demand from a graph frontier and uses the physical planner.
    """

    observed_now = ensure_utc(now or utc_now())
    deadline = observed_now + timedelta(seconds=deadline_seconds)
    predicate_id = predicate_id.upper()
    if predicate_id == "PASSES":
        provider_id = "pass_reference_evaluator"
        roles = (
            PredicateRole(role_name="vehicle", variable="vehicle", entity_type="vehicle"),
            PredicateRole(role_name="reference", variable="reference", entity_type="location"),
        )
        bound_roles = {"reference": reference_id}
        unbound_roles = ("vehicle",)
        binding_policy = DemandBindingPolicy(
            role_modes={
                "vehicle": BindingCapability.INTRODUCE,
                "reference": BindingCapability.VALIDATE,
            },
            forkable_roles=("vehicle",),
        )
        result_kind = ResultKind.INSTANT_MATCH
    elif predicate_id in {"MOVING", "STOPPED"}:
        provider_id = "motion_state_evaluator"
        roles = (PredicateRole(role_name="vehicle", variable="vehicle", entity_type="vehicle"),)
        bound_roles = {}
        unbound_roles = ("vehicle",)
        binding_policy = DemandBindingPolicy(
            role_modes={"vehicle": BindingCapability.OBSERVE_ONLY}
        )
        result_kind = ResultKind.STATE_OBSERVATION
    elif predicate_id == "FOLLOWS":
        if not leader_id:
            raise ValueError("FOLLOWS replay candidate requires leader_id")
        provider_id = "follows_local_geometry"
        roles = (
            PredicateRole(role_name="leader", variable="leader", entity_type="vehicle"),
            PredicateRole(role_name="follower", variable="follower", entity_type="vehicle"),
        )
        bound_roles = {"leader": leader_id}
        unbound_roles = ("follower",)
        binding_policy = DemandBindingPolicy(
            role_modes={
                "leader": BindingCapability.VALIDATE,
                "follower": BindingCapability.INTRODUCE,
            },
            forkable_roles=("follower",),
        )
        result_kind = ResultKind.INTERVAL_MATCH
    else:
        raise ValueError(f"unsupported replay vehicle predicate: {predicate_id}")

    demand = PredicateDemand(
        request_id=request_id,
        graph_hash="sha256:" + "7" * 64,
        hypothesis_id=uuid7(),
        hypothesis_version=1,
        frontier_id=uuid7(),
        checkpoint_id=uuid7(),
        graph_node_id=f"replay_{predicate_id.lower()}",
        semantic_predicate=SemanticPredicate(
            predicate_id=predicate_id,
            roles=roles,
            result_kind=result_kind,
        ),
        bound_roles=bound_roles,
        unbound_roles=unbound_roles,
        binding_policy=binding_policy,
        event_time_interval=event_interval,
        deadline=DeadlineSpec(latest_useful_completion=deadline),
        eligible_source_ids=(source_id,),
        acceptable_output_types=("predicate_match.v1",),
        hard_constraints=DataMovementConstraints(
            raw_data_must_remain_local=True,
            allowed_node_ids=(node_id,),
        ),
    )
    profile = provider_registry.profile(provider_id, node_class)
    alternative_id = deterministic_id(
        "alt",
        {
            "demand_id": demand.demand_id,
            "provider": provider_id,
            "node": node_id,
            "source": source_id,
        },
        length=32,
    )
    external = ExecutionInput(
        name="tracks",
        data_type="projected_track_set.v1",
        kind=ExecutionInputKind.LIVE_SOURCE,
        node_id=node_id,
        source_id=source_id,
        bytes=0,
    )
    step = PlanStep(
        step_id=f"{alternative_id}:evaluate",
        provider_id=provider_id,
        node_id=node_id,
        demand_id=demand.demand_id,
        alternative_id=alternative_id,
        chain_id=f"replay_{predicate_id.lower()}_direct",
        execution_mode=ExecutionMode.LIVE,
        inputs=(external,),
        input_data_types=("projected_track_set.v1",),
        output_data_types=("predicate_match.v1", "track_summary.v1"),
        cpu_cores=profile.cpu_cores,
        memory_mb=profile.memory_mb,
        gpu_memory_mb=profile.gpu_memory_mb,
        quality_score=profile.quality_score,
        estimated_startup_ms=profile.startup_ms,
        estimated_execution_ms=profile.execution_ms,
    )
    completion_ms = profile.startup_ms + profile.execution_ms
    cost = PlanCost(
        predicted_completion_ms=completion_ms,
        deadline_slack_ms=int((deadline - observed_now).total_seconds() * 1000) - completion_ms,
        startup_cost_ms=profile.startup_ms,
        resource_cost_units=profile.cpu_cores + profile.memory_mb / 1024.0,
        transfer_bytes=0,
    )
    label_contract = PhysicalPlanLabel(
        checkpoint_id=demand.checkpoint_id,
        covered_demand_ids=(demand.demand_id,),
        steps=(step,),
        continuation_output_types=("track_summary.v1",),
        cost=cost,
        hard_constraints_satisfied=True,
        quality_floor_satisfied=True,
        feasibility_reasons=("vehicle predicate provider is local to replay source",),
    )
    plan = ExecutionPlan(
        label_id=label_contract.label_id or "",
        checkpoint_id=demand.checkpoint_id,
        demand_ids=(demand.demand_id,),
        steps=(step,),
        reservations=(
            ResourceReservation(
                node_id=node_id,
                cpu_cores=profile.cpu_cores,
                memory_mb=profile.memory_mb,
                gpu_memory_mb=profile.gpu_memory_mb,
            ),
        ),
        status=PlanStatus.CANDIDATE,
        expires_at=deadline,
    )
    return PlanCandidate(
        plan=plan,
        demands=(demand,),
        task_policy=TaskSchedulingPolicy(request_id=request_id),
        predicted_completion_ms=completion_ms,
        startup_cost_ms=profile.startup_ms,
        incremental_resource_cost_units=cost.resource_cost_units,
        transfer_bytes=0,
        created_at=observed_now,
    )


def build_replay_multimodal_candidate(
    *,
    provider_registry: ProviderRegistry,
    node_id: str,
    source_id: str,
    event_interval: EventTimeInterval,
    predicate_id: str,
    request_id: str = "replay_multimodal_demo",
    node_class: str = "sensor",
    label: str | None = None,
    bound_roles: dict[str, str] | None = None,
    deadline_seconds: float = 300.0,
    now: datetime | None = None,
) -> PlanCandidate:
    """Build a direct Phase-8 predicate candidate for replay integration.

    This is a diagnostic entry point for the distributed adapter. Normal FABLE
    execution derives equivalent demands from the active semantic frontier and
    selects among the full registered chains.
    """

    observed_now = ensure_utc(now or utc_now())
    deadline = observed_now + timedelta(seconds=deadline_seconds)
    predicate_id = predicate_id.upper()
    supplied = dict(bound_roles or {})

    if predicate_id == "AUDIO_EVENT":
        provider_id = "audio_event_classifier"
        roles = (
            PredicateRole(role_name="location", variable="location", entity_type="zone"),
        )
        semantic_parameters = {"label": label or "gunshot"}
        bound = {"location": supplied["location"]} if "location" in supplied else {}
        unbound = () if bound else ("location",)
        binding_policy = DemandBindingPolicy(
            role_modes={
                "location": (
                    BindingCapability.VALIDATE
                    if bound
                    else BindingCapability.INTRODUCE
                )
            },
            forkable_roles=unbound,
        )
        result_kind = ResultKind.INSTANT_MATCH
        input_types = ("audio_segment.v1",)
        continuation = ("audio_event_observation.v1", "audio_localization.v1")
        chain_id = "detect_audio_event"
    elif predicate_id in {"DISEMBARKS", "BOARDS"}:
        provider_id = "person_vehicle_relation_provider"
        roles = (
            PredicateRole(role_name="person", variable="person", entity_type="person"),
            PredicateRole(role_name="vehicle", variable="vehicle", entity_type="vehicle"),
        )
        semantic_parameters = {}
        bound = {key: value for key, value in supplied.items() if key in {"person", "vehicle"}}
        unbound = tuple(role for role in ("person", "vehicle") if role not in bound)
        binding_policy = DemandBindingPolicy(
            role_modes={
                role: (
                    BindingCapability.VALIDATE
                    if role in bound
                    else BindingCapability.INTRODUCE
                )
                for role in ("person", "vehicle")
            },
            forkable_roles=unbound,
        )
        result_kind = ResultKind.INSTANT_MATCH
        input_types = ("projected_track_set.v1",)
        continuation = ("track_summary.v1",)
        chain_id = "person_vehicle_transition"
    elif predicate_id == "CONVERSATION":
        provider_id = "conversation_provider"
        roles = (
            PredicateRole(role_name="participant_a", variable="participant_a", entity_type="person"),
            PredicateRole(role_name="participant_b", variable="participant_b", entity_type="person"),
        )
        semantic_parameters = {}
        bound = {
            key: value
            for key, value in supplied.items()
            if key in {"participant_a", "participant_b"}
        }
        unbound = tuple(
            role for role in ("participant_a", "participant_b") if role not in bound
        )
        binding_policy = DemandBindingPolicy(
            role_modes={
                role: (
                    BindingCapability.VALIDATE
                    if role in bound
                    else BindingCapability.INTRODUCE
                )
                for role in ("participant_a", "participant_b")
            },
            forkable_roles=unbound,
        )
        result_kind = ResultKind.INTERVAL_MATCH
        input_types = ("projected_track_set.v1", "speaker_turn_set.v1")
        continuation = ("speaker_turn_set.v1", "track_summary.v1")
        chain_id = "conversation_proximity_diarization"
    elif predicate_id == "TRANSFER":
        provider_id = "object_transfer_reasoner"
        roles = (
            PredicateRole(role_name="object", variable="object", entity_type="package"),
            PredicateRole(role_name="source", variable="source", entity_type="entity"),
            PredicateRole(role_name="destination", variable="destination", entity_type="entity"),
        )
        semantic_parameters = {}
        bound = {
            key: value
            for key, value in supplied.items()
            if key in {"object", "source", "destination"}
        }
        unbound = tuple(
            role for role in ("object", "source", "destination") if role not in bound
        )
        binding_policy = DemandBindingPolicy(
            role_modes={
                role: (
                    BindingCapability.VALIDATE
                    if role in bound
                    else BindingCapability.INTRODUCE
                )
                for role in ("object", "source", "destination")
            },
            forkable_roles=unbound,
        )
        result_kind = ResultKind.INTERVAL_MATCH
        input_types = ("projected_track_set.v1", "interaction_evidence_set.v1")
        continuation = ("custody_state.v1",)
        chain_id = "package_transfer_high_resolution"
    else:
        raise ValueError(f"unsupported replay multimodal predicate: {predicate_id}")

    demand = PredicateDemand(
        request_id=request_id,
        graph_hash="sha256:" + "8" * 64,
        hypothesis_id=uuid7(),
        hypothesis_version=1,
        frontier_id=uuid7(),
        checkpoint_id=uuid7(),
        graph_node_id=f"replay_{predicate_id.lower()}",
        semantic_predicate=SemanticPredicate(
            predicate_id=predicate_id,
            roles=roles,
            parameters=semantic_parameters,
            result_kind=result_kind,
        ),
        bound_roles=bound,
        unbound_roles=unbound,
        binding_policy=binding_policy,
        event_time_interval=event_interval,
        deadline=DeadlineSpec(latest_useful_completion=deadline),
        eligible_source_ids=(source_id,),
        acceptable_output_types=("predicate_match.v1",),
        hard_constraints=DataMovementConstraints(
            raw_data_must_remain_local=True,
            allowed_node_ids=(node_id,),
        ),
    )
    profile = provider_registry.profile(provider_id, node_class)
    alternative_id = deterministic_id(
        "alt",
        {
            "demand_id": demand.demand_id,
            "provider": provider_id,
            "node": node_id,
            "source": source_id,
            "predicate": predicate_id,
        },
        length=32,
    )
    external_inputs = tuple(
        ExecutionInput(
            name=f"input_{index}",
            data_type=data_type,
            kind=ExecutionInputKind.LIVE_SOURCE,
            node_id=node_id,
            source_id=source_id,
            bytes=0,
        )
        for index, data_type in enumerate(input_types)
    )
    step = PlanStep(
        step_id=f"{alternative_id}:evaluate",
        provider_id=provider_id,
        node_id=node_id,
        demand_id=demand.demand_id,
        alternative_id=alternative_id,
        chain_id=chain_id,
        execution_mode=ExecutionMode.LIVE,
        inputs=external_inputs,
        input_data_types=input_types,
        output_data_types=(*continuation, "predicate_match.v1"),
        parameters=tuple(sorted((str(key), value) for key, value in semantic_parameters.items())),
        cpu_cores=profile.cpu_cores,
        memory_mb=profile.memory_mb,
        gpu_memory_mb=profile.gpu_memory_mb,
        quality_score=profile.quality_score,
        estimated_startup_ms=profile.startup_ms,
        estimated_execution_ms=profile.execution_ms,
    )
    completion_ms = profile.startup_ms + profile.execution_ms
    cost = PlanCost(
        predicted_completion_ms=completion_ms,
        deadline_slack_ms=int((deadline - observed_now).total_seconds() * 1000) - completion_ms,
        startup_cost_ms=profile.startup_ms,
        resource_cost_units=(
            profile.cpu_cores
            + profile.memory_mb / 1024.0
            + 2.0 * profile.gpu_memory_mb / 1024.0
        ),
        transfer_bytes=0,
    )
    label_contract = PhysicalPlanLabel(
        checkpoint_id=demand.checkpoint_id,
        covered_demand_ids=(demand.demand_id,),
        steps=(step,),
        continuation_output_types=continuation,
        cost=cost,
        hard_constraints_satisfied=True,
        quality_floor_satisfied=True,
        feasibility_reasons=("Phase-8 replay provider is local to the evidence source",),
    )
    plan = ExecutionPlan(
        label_id=label_contract.label_id or "",
        checkpoint_id=demand.checkpoint_id,
        demand_ids=(demand.demand_id,),
        steps=(step,),
        reservations=(
            ResourceReservation(
                node_id=node_id,
                cpu_cores=profile.cpu_cores,
                memory_mb=profile.memory_mb,
                gpu_memory_mb=profile.gpu_memory_mb,
            ),
        ),
        status=PlanStatus.CANDIDATE,
        expires_at=deadline,
    )
    return PlanCandidate(
        plan=plan,
        demands=(demand,),
        task_policy=TaskSchedulingPolicy(request_id=request_id),
        predicted_completion_ms=completion_ms,
        startup_cost_ms=profile.startup_ms,
        incremental_resource_cost_units=cost.resource_cost_units,
        transfer_bytes=0,
        created_at=observed_now,
    )
