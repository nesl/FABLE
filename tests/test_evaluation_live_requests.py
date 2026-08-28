from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest

from evaluation.capacity_profiles import (
    ComputeCapacityProfile,
    apply_compute_capacity_profile,
)
from evaluation.deployment_artifacts import load_deployment_artifacts
from evaluation.live_requests import (
    LiveComplexEventCancelRequest,
    LiveComplexEventRequest,
    LiveRequestManager,
    ObservedSeed,
    PendingSeedRegistry,
    SENSOR_LOCAL_FANOUT_PREDICATES,
)
from evaluation.schemas import BaselineId
from fable.common.examples import BASE_TIME
from fable.common.enums import ExecutionMode
from fable.common.time import EventTimeInterval
from fable.distributed.config import ProviderRuntimeResolver, load_deployment_graph
from fable.planning import (
    BoundedLabelPlanner,
    DemandCompiler,
    PhysicalAlternativeGraphBuilder,
    default_predicate_registry,
)
from fable.planning.provider_registry import ProviderRegistry
from fable.scheduling.capacity import CapacityLedger
from fable.scheduling.control import CheckpointController
from fable.scheduling.lifecycle import ProviderLifecycleManager
from fable.semantic import ScriptedResultSpec, predicate_result_from_spec

from tests.test_evaluation_live_orchestration import RecordingDispatcher


def test_sparse_sensor_local_predicates_use_selected_node_fanout() -> None:
    assert {
        "AUDIO_EVENT",
        "VEHICLE_PRESENT_BEFORE",
        "DEPARTURE_OR_ESCAPE",
        "PERSON_PROXIMITY",
        "CONVERSATION",
        "TRANSFER",
        "FOLLOWS",
        "PASSES",
    }.issubset(SENSOR_LOCAL_FANOUT_PREDICATES)


def test_uncalibrated_convergence_uses_sensor_local_fanout() -> None:
    assert "DISTANCE_LT" in SENSOR_LOCAL_FANOUT_PREDICATES


def test_unlocalized_audio_seed_binds_conservative_sensor_scene() -> None:
    observation = PendingSeedRegistry._normalize_observation(
        "/dvpg_gq_orin_13/fable/audio/events",
        {
            "schema_version": "audio_event_observation.v1",
            "source_id": "dvpg_gq_orin_13",
            "label": "gunshot",
        },
    )

    assert observation is not None
    assert observation["bindings"] == {
        "location": "sensor_scene:dvpg_gq_orin_13"
    }


def test_durable_audio_result_preserves_label_for_pending_seed_match() -> None:
    observation = PendingSeedRegistry._normalize_observation(
        "fable/v1/durable/seed-result",
        {
            "schema_version": "fable.predicate_result.v1",
            "request_id": "robbery-request-1",
            "occurrence_id": "gunshot-1",
            "truth": "TRUE",
            "confidence": 0.91,
            "event_time_interval": {
                "start": BASE_TIME.isoformat(),
                "end": BASE_TIME.isoformat(),
            },
            "semantic_predicate": {
                "predicate_id": "AUDIO_EVENT",
                "parameters": {"label": "gunshot"},
            },
            "binding_delta": {
                "introduced": {"location": "front"},
                "validated": {},
            },
            "provenance": {
                "source_ids": ["orin16_microphone"],
                "node_id": "dvpg_gq_orin_16",
            },
        },
    )

    assert observation is not None
    assert observation["predicate_id"] == "AUDIO_EVENT"
    assert observation["label"] == "gunshot"
    assert observation["_trusted_request_id"] == "robbery-request-1"
    assert PendingSeedRegistry._parameters_match(
        {"label": "gunshot", "minimum_watch_age_ms": 0}, observation
    )


def _manager(*, evaluation_record_sink=None) -> tuple[LiveRequestManager, RecordingDispatcher]:
    deployment = load_deployment_graph(
        "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    providers = ProviderRegistry.from_files(
        catalog_path="providers/registry/catalog.yaml",
        data_types_path="providers/registry/data_types.yaml",
    )
    artifacts = load_deployment_artifacts(
        "iobt-minimal-ce-replay/config/fable_deployment_artifacts.yaml",
        repository_root=".",
    )
    runtimes = ProviderRuntimeResolver.from_yaml(
        "iobt-minimal-ce-replay/config/fable_provider_runtimes.yaml"
    )
    lifecycle = ProviderLifecycleManager(
        provider_registry=providers,
        capacity=CapacityLedger(deployment),
    )
    dispatcher = RecordingDispatcher()
    return (
        LiveRequestManager(
            provider_registry=providers,
            deployment=deployment,
            runtime_resolver=runtimes,
            graph_builder=PhysicalAlternativeGraphBuilder(
                provider_registry=providers,
                artifact_catalog=artifacts,
                deployment=deployment,
            ),
            demand_compiler=DemandCompiler(
                predicate_registry=default_predicate_registry(),
                deployment=deployment,
            ),
            planner=BoundedLabelPlanner(
                provider_registry=providers,
                artifact_catalog=artifacts,
                deployment=deployment,
            ),
            dispatcher=dispatcher,
            lifecycle=lifecycle,
            checkpoint_controller=CheckpointController(
                lifecycle=lifecycle,
                artifact_catalog=artifacts,
            ),
            evaluation_record_sink=evaluation_record_sink,
        ),
        dispatcher,
    )


def _request(**updates) -> LiveComplexEventRequest:
    values = {
        "request_id": "typed-convoy-1",
        "submitter_id": "test-submit",
        "run_id": "run-1",
        "trace_id": "trace-1",
        "family_id": "convoy",
        "baseline_id": BaselineId.B4_GREEDY_FRONTIER,
        "seed": ObservedSeed(
            graph_node_key="leader_passes",
            occurrence_id="leader-pass-1",
            source_id="orin11_camera",
            node_id="dvpg_gq_orin_11",
            event_time_interval=EventTimeInterval(start=BASE_TIME, end=BASE_TIME),
            introduced_bindings={"leader": "vehicle-1"},
        ),
    }
    values.update(updates)
    return LiveComplexEventRequest(**values)


def test_raw_site_edge_authorization_accepts_mobile_sensor_nodes() -> None:
    request = _request(
        allow_raw_to_trusted_site_edge=True,
        allowed_execution_node_ids=("mobile_archive_6", "x86server"),
    )
    assert request.allow_raw_to_trusted_site_edge is True


def test_raw_site_edge_authorization_rejects_cloud_nodes() -> None:
    with pytest.raises(ValueError, match="limited to sensors and x86server"):
        _request(
            allow_raw_to_trusted_site_edge=True,
            allowed_execution_node_ids=("mobile_archive_6", "cloud1"),
        )


def test_typed_observed_seed_dispatches_first_real_frontier() -> None:
    manager, dispatcher = _manager()

    admission = manager.admit(_request())

    assert admission.response.accepted
    assert admission.response.hypothesis_ids
    assert admission.response.candidate_ids
    assert admission.response.command_count > 0
    assert dispatcher.submissions


def test_controlled_baseline_seed_accepts_admitted_reused_provider() -> None:
    manager, _ = _manager()

    class ReusedProviderDispatcher(RecordingDispatcher):
        def submit_candidates(
            self,
            candidates,
            *,
            runtime_overrides=None,
            now=None,
            allow_capacity_overcommit=False,
        ):
            self.submissions.append(tuple(candidates))
            return SimpleNamespace(admitted_plan_ids=("reused-plan",)), ()

    manager.dispatcher = ReusedProviderDispatcher()
    planning = manager.acquire_seed(
        _request(
            request_id="b0-reused-seed-provider",
            baseline_id=BaselineId.B0_PRODUCE_ALL,
            seed=None,
            seed_graph_node_key="leader_passes",
        )
    )

    assert planning.candidates
    assert not planning.commands


def test_b1_seed_watch_activates_authored_whole_event_pipeline() -> None:
    manager, _ = _manager()
    planning = manager.acquire_seed(
        _request(
            request_id="b1-always-on-from-watch",
            baseline_id=BaselineId.B1_STATIC_WHOLE_EVENT,
            seed=None,
            seed_graph_node_key="leader_passes",
        )
    )
    predicates = {
        demand.semantic_predicate.predicate_id
        for candidate in planning.candidates
        for demand in candidate.demands
    }
    assert "PASSES" in predicates
    # The current convoy definition is sequential PASSES rather than the
    # retired FOLLOWS formulation; both authored stages share this predicate.
    assert sum(
        demand.semantic_predicate.predicate_id == "PASSES"
        for candidate in planning.candidates
        for demand in candidate.demands
    ) >= 2


def test_e4_b1_identifier_uses_authored_whole_event_seed_pipeline() -> None:
    manager, _ = _manager()
    planning = manager.acquire_seed(
        _request(
            request_id="b1-handwritten-always-on-from-watch",
            baseline_id=BaselineId.B1_HANDWRITTEN_STATIC,
            seed=None,
            seed_graph_node_key="leader_passes",
        )
    )

    predicates = {
        demand.semantic_predicate.predicate_id
        for candidate in planning.candidates
        for demand in candidate.demands
    }
    assert "PASSES" in predicates
    assert planning.decision.baseline_id == BaselineId.B1_HANDWRITTEN_STATIC


def test_e4_b1_robbery_watch_starts_fixed_exit_provider_at_admission() -> None:
    manager, _ = _manager()
    planning = manager.acquire_seed(
        _request(
            request_id="b1-handwritten-robbery-whole-event-watch",
            family_id="robbery",
            baseline_id=BaselineId.B1_HANDWRITTEN_STATIC,
            seed=None,
            seed_graph_node_key="alarm_branch",
        )
    )

    predicates = {
        demand.semantic_predicate.predicate_id
        for candidate in planning.candidates
        for demand in candidate.demands
    }
    assert {"AUDIO_EVENT", "EXITS"} <= predicates
    assert "SAME_ENTITY" not in predicates


def test_b0_seed_watch_activates_same_ce_pipeline_as_b1() -> None:
    manager, _ = _manager()
    planning = manager.acquire_seed(
        _request(
            request_id="b0-ce-authored-all-node-watch",
            baseline_id=BaselineId.B0_PRODUCE_ALL,
            seed=None,
            seed_graph_node_key="leader_passes",
        )
    )
    predicates = {
        demand.semantic_predicate.predicate_id
        for candidate in planning.candidates
        for demand in candidate.demands
    }
    assert {"PASSES"} <= predicates


def test_b0_seed_watch_preserves_authored_seed_cardinality() -> None:
    manager, _ = _manager()
    registry = PendingSeedRegistry(manager)
    request = _request(
        request_id="b0-reference-diverse-seeds",
        baseline_id=BaselineId.B0_PRODUCE_ALL,
        seed=None,
        seed_graph_node_key="leader_passes",
        allowed_seed_node_ids=(
            "dvpg_gq_orin_11",
            "dvpg_gq_orin_12",
            "dvpg_gq_orin_13",
        ),
    )

    response = registry.register(request)

    assert response.accepted
    pending = registry._pending[request.request_id]
    assert pending.request.seed_admission_strategy == request.seed_admission_strategy
    assert pending.request.max_seed_hypotheses == request.max_seed_hypotheses


def test_pending_fable_seed_watch_replans_on_resource_epoch() -> None:
    records = []
    manager, dispatcher = _manager(evaluation_record_sink=records.append)
    request = _request(
        request_id="pending-seed-resource-change",
        baseline_id=BaselineId.FABLE,
        seed=None,
        seed_graph_node_key="leader_passes",
    )
    manager.acquire_seed(request)
    before = len(dispatcher.submissions)

    replans = manager.handle_seed_resource_change(
        observed_at=BASE_TIME + timedelta(seconds=2),
        reason="validated compute contention",
    )

    assert len(replans) == 1
    assert len(dispatcher.submissions) > before
    resource_plans = [
        item
        for item in records
        if item.record_type == "plan_decision" and item.resource_epoch == 1
    ]
    assert resource_plans
    assert resource_plans[-1].replan_trigger.endswith(":PENDING_SEED")


def test_raw_authorized_seed_graph_retains_site_edge_yolo_alternative() -> None:
    manager, _ = _manager()
    request = _request(
        request_id="pending-seed-site-edge-alternative",
        baseline_id=BaselineId.FABLE,
        seed=None,
        seed_graph_node_key="leader_passes",
        allowed_seed_source_ids=("orin11_camera",),
        allowed_seed_node_ids=("dvpg_gq_orin_11",),
        allowed_execution_node_ids=("dvpg_gq_orin_11", "x86server"),
        allow_raw_to_trusted_site_edge=True,
    )

    manager.acquire_seed(request)
    alternatives = manager._seed_acquisitions[request.request_id].case.frontier_graph.alternatives

    assert any(
        any(
            step.provider_id == "yolo_vehicle_fast_640"
            and step.node_id == "x86server"
            for step in alternative.step_placements
        )
        for alternative in alternatives
    )


def test_pending_seed_moves_yolo_to_site_when_device_gpu_slot_disappears() -> None:
    manager, _ = _manager()
    device = "dvpg_gq_orin_11"
    request = _request(
        request_id="pending-seed-device-gpu-loss",
        baseline_id=BaselineId.FABLE,
        seed=None,
        seed_graph_node_key="leader_passes",
        allowed_seed_source_ids=("orin11_camera",),
        allowed_seed_node_ids=(device,),
        allowed_execution_node_ids=(device, "x86server"),
        allow_raw_to_trusted_site_edge=True,
    )
    nominal = manager.acquire_seed(request)
    assert device in nominal.decision.selected_node_ids

    applied = apply_compute_capacity_profile(
        manager.deployment,
        target_node_id=device,
        profile=ComputeCapacityProfile(
            profile_id="E1",
            cpu_capacity_fraction=0.25,
            memory_capacity_fraction=0.50,
            gpu_capacity_fraction=0.10,
            execution_time_multiplier=2.25,
            queue_delay_ms=125,
        ),
        resource_epoch=1,
    )
    manager.update_network_deployment(
        applied.deployment,
        execution_time_multiplier=applied.execution_time_multiplier,
        queue_delay_ms=applied.queue_delay_ms,
        compute_target_node_id=device,
    )
    replans = manager.handle_seed_resource_change(
        observed_at=BASE_TIME + timedelta(seconds=2),
        reason="CAPACITY_PROFILE:E1:APPLY",
    )

    assert len(replans) == 1
    decision = replans[0][1].decision
    assert "x86server" in decision.selected_node_ids
    selected = {
        item.alternative_id: item
        for item in manager._seed_acquisitions[
            request.request_id
        ].case.frontier_graph.alternatives
    }[decision.selected_alternative_ids[0]]
    assert any(
        step.provider_id == "yolo_vehicle_fast_640"
        and step.node_id == "x86server"
        for step in selected.step_placements
    )


def test_multi_hypothesis_seed_uses_partitioned_demands_for_search_contract() -> None:
    manager, _ = _manager()
    device = "dvpg_gq_orin_11"
    request = _request(
        request_id="multi-pending-seed-device-gpu-loss",
        baseline_id=BaselineId.FABLE,
        seed=None,
        seed_graph_node_key="leader_passes",
        allowed_seed_source_ids=("orin11_camera",),
        allowed_seed_node_ids=(device,),
        allowed_execution_node_ids=(device, "x86server"),
        allow_raw_to_trusted_site_edge=True,
        max_seed_hypotheses=2,
    )
    manager.acquire_seed(request)
    case = manager._seed_acquisitions[request.request_id].case
    assert set(case.frontier_graph.demand_ids) == {
        demand.demand_id for demand in case.frontier_demands
    }


def test_dynamic_frontier_request_retains_racing_observations() -> None:
    manager, _ = _manager()

    admission = manager.admit(_request(request_id="buffered-dynamic-request"))

    assert admission.response.accepted
    state = manager._executions[admission.response.request_id].request_state(
        admission.response.request_id
    )
    assert state.early_observations is not None


def test_live_admission_emits_common_plan_and_demand_records() -> None:
    records = []
    manager, _ = _manager(evaluation_record_sink=records.append)
    admission = manager.admit(_request(request_id="recorded-live-request"))
    assert admission.response.accepted
    record_types = {item.record_type for item in records}
    assert {
        "predicate_demand",
        "plan_decision",
    } <= record_types
    assert all(item.request_id == "recorded-live-request" for item in records)


def test_live_resource_change_replans_only_adaptive_request() -> None:
    records = []
    manager, dispatcher = _manager(evaluation_record_sink=records.append)
    request = _request(request_id="resource-adaptive-live-request")
    admission = manager.admit(request)
    assert admission.response.accepted
    state = manager._executions[request.request_id].request_state(request.request_id)
    before = len(dispatcher.submissions)
    previous_builder = manager._executions[request.request_id].graph_builder
    manager.update_network_deployment(manager.deployment)
    assert manager._executions[request.request_id].graph_builder is not previous_builder

    replans = manager.handle_resource_change(
        demand_ids=(state.whole_event_demands[0].demand_id,),
        reason="validated W1 condition epoch",
        observed_at=BASE_TIME + timedelta(seconds=2),
    )

    assert len(replans) == 1
    assert len(dispatcher.submissions) > before
    resource_plans = [
        item
        for item in records
        if item.record_type == "plan_decision" and item.resource_epoch == 1
    ]
    assert resource_plans
    assert resource_plans[-1].replan_trigger.startswith("RESOURCE_EPOCH:")


def test_fable_resource_change_batches_multiple_live_requests(
    monkeypatch,
) -> None:
    records = []
    manager, dispatcher = _manager(evaluation_record_sink=records.append)
    first = _request(
        request_id="joint-resource-request-a", baseline_id=BaselineId.FABLE
    )
    second = _request(
        request_id="joint-resource-request-b", baseline_id=BaselineId.FABLE
    )
    assert manager.admit(first).response.accepted
    assert manager.admit(second).response.accepted
    monkeypatch.setenv("FABLE_JOINT_RESOURCE_EPOCH_PLANNING", "1")
    before = len(dispatcher.submissions)

    replans = manager.handle_resource_change(
        demand_ids=(),
        reason="CAPACITY_PROFILE:E1:APPLY",
        observed_at=BASE_TIME + timedelta(seconds=2),
    )

    assert {request_id for request_id, _ in replans} == {
        first.request_id,
        second.request_id,
    }
    assert len(dispatcher.submissions) > before
    joint_records = [
        item
        for item in records
        if item.record_type == "plan_decision"
        and item.resource_epoch == 1
        and "JOINT_REQUESTS" in item.replan_trigger
    ]
    assert {item.request_id for item in joint_records} == {
        first.request_id,
        second.request_id,
    }


def test_fable_resource_change_is_request_local_without_joint_gate(
    monkeypatch,
) -> None:
    records = []
    manager, _ = _manager(evaluation_record_sink=records.append)
    first = _request(
        request_id="local-resource-request-a", baseline_id=BaselineId.FABLE
    )
    second = _request(
        request_id="local-resource-request-b", baseline_id=BaselineId.FABLE
    )
    assert manager.admit(first).response.accepted
    assert manager.admit(second).response.accepted
    monkeypatch.delenv("FABLE_JOINT_RESOURCE_EPOCH_PLANNING", raising=False)

    manager.handle_resource_change(
        demand_ids=(),
        reason="CAPACITY_PROFILE:E1:APPLY",
        observed_at=BASE_TIME + timedelta(seconds=2),
    )

    resource_records = [
        item
        for item in records
        if item.record_type == "plan_decision" and item.resource_epoch == 1
    ]
    assert {item.request_id for item in resource_records} == {
        first.request_id,
        second.request_id,
    }
    assert all("JOINT_REQUESTS" not in item.replan_trigger for item in resource_records)


def test_failed_node_snapshot_precedes_resource_replan() -> None:
    records = []
    manager, _ = _manager(evaluation_record_sink=records.append)
    request = _request(request_id="failed-source-resource-replan")
    admission = manager.admit(request)
    assert admission.response.accepted
    state = manager._executions[request.request_id].request_state(request.request_id)
    demand_ids = tuple(item.demand_id for item in state.whole_event_demands)

    manager.update_network_deployment(
        manager.deployment.with_node_availability(
            "dvpg_gq_orin_11", available=False
        )
    )
    manager.handle_resource_change(
        demand_ids=demand_ids,
        reason="VALIDATED_LINK_STATE:link:s_orin11:s_edge:FAIL",
        observed_at=BASE_TIME + timedelta(seconds=2),
    )

    resource_plans = [
        item
        for item in records
        if item.record_type == "plan_decision"
        and item.replan_trigger.startswith("RESOURCE_EPOCH:")
    ]
    assert resource_plans
    assert all(
        "dvpg_gq_orin_11" not in item.selected_node_ids
        and "orin11_camera" not in item.selected_source_ids
        for item in resource_plans
    )
    assert any(
        "WAIT_FOR_SOURCE_RECOVERY" in item.reason
        for item in resource_plans
    )


def test_source_recovery_with_no_affected_demands_is_not_a_wildcard() -> None:
    manager, dispatcher = _manager()
    request = _request(request_id="unrelated-source-recovery")
    admission = manager.admit(request)
    assert admission.response.accepted
    before = len(dispatcher.submissions)

    replans = manager.handle_source_recovery(
        node_id="dvpg_gq_orin_16",
        recovery_intervals={
            "orin16_camera": EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(seconds=10),
            )
        },
        demand_ids=(),
        reason="node session recovery",
        observed_at=BASE_TIME + timedelta(seconds=11),
    )

    assert replans == ()
    assert len(dispatcher.submissions) == before


def test_pre_watermark_source_recovery_uses_bounded_active_demand_interval() -> None:
    manager, dispatcher = _manager()
    request = _request(request_id="pre-watermark-recovery")
    admission = manager.admit(request)
    assert admission.response.accepted
    state = manager._executions[request.request_id].request_state(request.request_id)
    demand = next(
        item
        for item in state.whole_event_demands
        if "orin11_camera" in item.eligible_source_ids
    )
    before = len(dispatcher.submissions)

    replans = manager.handle_source_recovery(
        node_id="dvpg_gq_orin_11",
        recovery_intervals={},
        demand_ids=(demand.demand_id,),
        reason="restored before first source watermark",
        observed_at=BASE_TIME + timedelta(seconds=11),
    )

    assert len(replans) == 1
    assert len(dispatcher.submissions) > before
    recovered_demands = [
        item
        for batch in dispatcher.submissions[before:]
        for candidate in batch
        for item in candidate.demands
    ]
    assert recovered_demands
    assert all(item.event_time_interval == demand.event_time_interval for item in recovered_demands)
    assert all(item.eligible_source_ids == ("orin11_camera",) for item in recovered_demands)


def test_source_recovery_does_not_replace_canonical_frontier_case() -> None:
    manager, _ = _manager()
    request = _request(request_id="recovery-preserves-frontier")
    admission = manager.admit(request)
    assert admission.response.accepted
    execution = manager._executions[request.request_id]
    state = execution.request_state(request.request_id)
    canonical = dict(state.planning_cases)
    demand = next(
        item
        for item in state.whole_event_demands
        if "orin11_camera" in item.eligible_source_ids
    )

    replans = manager.handle_source_recovery(
        node_id="dvpg_gq_orin_11",
        recovery_intervals={"orin11_camera": demand.event_time_interval},
        demand_ids=(demand.demand_id,),
        reason="validated restore",
        observed_at=BASE_TIME + timedelta(seconds=11),
    )

    assert replans
    assert state.planning_cases == canonical


def test_exact_source_recovery_redelivery_is_idempotent() -> None:
    manager, dispatcher = _manager()
    request = _request(request_id="idempotent-source-recovery")
    admission = manager.admit(request)
    assert admission.response.accepted
    state = manager._executions[request.request_id].request_state(request.request_id)
    demand = next(
        item
        for item in state.whole_event_demands
        if "orin11_camera" in item.eligible_source_ids
    )
    arguments = {
        "node_id": "dvpg_gq_orin_11",
        "recovery_intervals": {"orin11_camera": demand.event_time_interval},
        "demand_ids": (demand.demand_id,),
        "reason": "validated restore",
        "observed_at": BASE_TIME + timedelta(seconds=11),
    }

    first = manager.handle_source_recovery(**arguments)
    after_first = len(dispatcher.submissions)
    second = manager.handle_source_recovery(**arguments)

    assert first
    assert second == ()
    assert len(dispatcher.submissions) == after_first


def test_pending_seed_recovery_dispatches_node_scoped_raw_catchup() -> None:
    manager, dispatcher = _manager()
    registry = PendingSeedRegistry(manager)
    request = _request(
        request_id="pending-seed-recovery",
        baseline_id=BaselineId.FABLE,
        seed=None,
        seed_graph_node_key="leader_passes",
    )
    response = registry.register(request)
    assert response.accepted
    pending = registry._pending[request.request_id]
    before = len(dispatcher.submissions)

    recovered = registry.recover_node(
        node_id="dvpg_gq_orin_11",
        demand_ids=tuple(pending.acquisition_demand_ids or ()),
    )

    assert len(recovered) == 1
    assert len(dispatcher.submissions) > before
    demands = [
        demand
        for candidate in recovered[0].candidates
        for demand in candidate.demands
    ]
    assert demands
    assert all("orin11_camera" in demand.eligible_source_ids for demand in demands)
    assert all(
        all(source_id.startswith("orin11_") for source_id in demand.eligible_source_ids)
        for demand in demands
    )
    assert all(demand.retrospective_context["outage_recovery"] for demand in demands)
    assert all(demand.retrospective_context["recovery_policy_stage"] == "RAW_FALLBACK" for demand in demands)


def test_strong_robbery_seed_preserves_person_vehicle_identity() -> None:
    manager, _ = _manager()
    request = _request(
        request_id="typed-identity-robbery-1",
        family_id="drive_up_shooting",
        seed=ObservedSeed(
            graph_node_key="disembarks",
            occurrence_id="disembarks-1",
            source_id="orin11_camera",
            node_id="dvpg_gq_orin_11",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME,
            ),
            introduced_bindings={
                "person": "person-7",
                "vehicle": "vehicle-3",
            },
        ),
    )

    admission = manager.admit(request)

    assert admission.response.accepted
    live = manager._executions[request.request_id]
    state = live._requests[request.request_id]
    assert {
        demand.semantic_predicate.predicate_id
        for demand in state.whole_event_demands
    } == {"AUDIO_EVENT", "BOARDS", "EXITS"}
    hypothesis = state.runtime.active_hypotheses[0]
    assert hypothesis.role_bindings["person"].canonical_entity_id == "person-7"
    assert hypothesis.role_bindings["vehicle"].canonical_entity_id == "vehicle-3"


def test_alarm_seed_dispatches_retrospective_vehicle_recovery() -> None:
    manager, dispatcher = _manager()
    request = _request(
        request_id="typed-retrospective-robbery-1",
        family_id="robbery",
        seed=ObservedSeed(
            graph_node_key="alarm_branch",
            occurrence_id="alarm-1",
            source_id="orin11_microphone",
            node_id="dvpg_gq_orin_11",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME,
            ),
            introduced_bindings={"location": "scene"},
        ),
    )

    admission = manager.admit(request)

    assert admission.response.accepted
    assert dispatcher.submissions
    live = manager._executions[request.request_id]
    state = live._requests[request.request_id]
    recovery = tuple(
        demand
        for demand in state.whole_event_demands
        if demand.semantic_predicate.predicate_id == "VEHICLE_PRESENT_BEFORE"
    )
    assert len(recovery) == 1
    assert recovery[0].event_time_interval == EventTimeInterval(
        start=BASE_TIME - timedelta(seconds=120),
        end=BASE_TIME,
    )
    recovery_alternatives = tuple(
        alternative
        for batch in dispatcher.submissions
        for candidate in batch
        for alternative in candidate.alternatives
        if any(
            demand.demand_id == alternative.demand_id
            and demand.semantic_predicate.predicate_id
            == "VEHICLE_PRESENT_BEFORE"
            for demand in candidate.demands
        )
    )
    assert recovery_alternatives
    assert all(
        alternative.execution_mode == ExecutionMode.RETROSPECTIVE
        for alternative in recovery_alternatives
    )
    assert all(
        tuple(step.provider_id for step in alternative.step_placements)
        == (
            "yolo_vehicle_fast_640",
            "multi_object_tracker",
            "historical_vehicle_interval_matcher",
        )
        for alternative in recovery_alternatives
    )


def test_recovered_vehicle_advances_to_executable_exit_frontier() -> None:
    records = []
    manager, dispatcher = _manager(evaluation_record_sink=records.append)
    request = _request(
        request_id="typed-robbery-exit-handoff",
        family_id="robbery",
        seed=ObservedSeed(
            graph_node_key="alarm_branch",
            occurrence_id="alarm-exit-handoff",
            source_id="orin11_microphone",
            node_id="dvpg_gq_orin_11",
            event_time_interval=EventTimeInterval(start=BASE_TIME, end=BASE_TIME),
            introduced_bindings={"location": "scene"},
        ),
    )
    admission = manager.admit(request)
    assert admission.response.accepted
    live = manager._executions[request.request_id]
    state = live.request_state(request.request_id)
    hypothesis = state.runtime.active_hypotheses[0]
    recovery = next(
        demand
        for demand in state.whole_event_demands
        if demand.semantic_predicate.predicate_id == "VEHICLE_PRESENT_BEFORE"
    )
    result = predicate_result_from_spec(
        state.runtime,
        hypothesis.hypothesis_id,
        ScriptedResultSpec(
            node_key="prior_entry",
            source_id="orin11_camera",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME - timedelta(seconds=20),
                end=BASE_TIME,
            ),
            introduced={"vehicle": "vehicle-7"},
            validated={"location": "scene"},
        ),
    ).model_copy(update={"demand_id": recovery.demand_id})

    progression = live.handle_result(result)

    exit_plans = [
        item
        for item in records
        if item.record_type == "plan_decision"
        and item.semantic_epoch > 0
        and "track_lifecycle_exit_live_vehicle" in item.selected_chain_ids
    ]
    assert progression.planning
    assert exit_plans
    assert exit_plans[-1].selected_alternative_ids
    exit_demand = next(
        demand
        for case in state.planning_cases.values()
        for demand in case.frontier_demands
        if demand.semantic_predicate.predicate_id == "EXITS"
    )
    assert exit_demand.event_time_interval.start == BASE_TIME
    assert exit_demand.event_time_interval.end == (
        BASE_TIME
        + timedelta(milliseconds=state.runtime.config.hypothesis_horizon_ms)
    )
    assert exit_demand.retrospective_context == {
        "anchor_node_id": state.runtime.graph.nodes_by_key["prior_entry"].node_id,
        "anchor_authored_key": "prior_entry",
        "anchor_event_time": BASE_TIME.isoformat(),
        "anchor_kind": "trigger_node_end",
        "lookback_ms": 0,
        "catch_up_and_follow": True,
    }


def test_request_rejects_unknown_seed_key_without_dispatch() -> None:
    manager, dispatcher = _manager()

    admission = manager.admit(
        _request(
            request_id="bad-seed",
            seed=_request().seed.model_copy(
                update={"graph_node_key": "client_invented_predicate"}
            ),
        )
    )

    assert not admission.response.accepted
    assert "authored graph" in admission.response.reason
    assert not dispatcher.submissions


def test_legacy_whole_event_baseline_remains_rejected() -> None:
    manager, dispatcher = _manager()

    admission = manager.admit(
        _request(request_id="b0-rejected", baseline_id=BaselineId.B0_ALWAYS_ON)
    )

    assert not admission.response.accepted
    assert "not supported" in admission.response.reason
    assert not dispatcher.submissions


@pytest.mark.parametrize(
    "baseline_id",
    (
        BaselineId.B0_PRODUCE_ALL,
        BaselineId.B1_STATIC_WHOLE_EVENT,
    ),
)
def test_canonical_whole_event_baseline_dispatches_structural_universe(
    baseline_id,
) -> None:
    manager, dispatcher = _manager()
    admission = manager.admit(
        _request(
            request_id=f"canonical-{baseline_id.value.lower()}",
            baseline_id=baseline_id,
        )
    )
    assert admission.response.accepted
    state = manager._executions[admission.response.request_id].request_state(
        admission.response.request_id
    )
    assert state.early_observations is not None
    assert {
        demand.semantic_predicate.predicate_id
        for demand in state.whole_event_demands
    } >= {"PASSES"}
    assert dispatcher.submissions


def test_pending_request_is_automatically_seeded_by_typed_provider_observation() -> None:
    manager, dispatcher = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="watched-convoy",
        submitter_id="watch-client",
        run_id="run-watch",
        trace_id="trace-watch",
        family_id="convoy",
        baseline_id=BaselineId.B4_GREEDY_FRONTIER,
        seed_graph_node_key="leader_passes",
        allowed_seed_source_ids=("orin11_camera",),
    )

    watching = pending.register(request)
    handoff_order = []
    release_seed_acquisition = manager.release_seed_acquisition
    admit = manager.admit

    def recording_release(*args, **kwargs):
        handoff_order.append("release")
        return release_seed_acquisition(*args, **kwargs)

    def recording_admit(*args, **kwargs):
        handoff_order.append("admit")
        return admit(*args, **kwargs)

    manager.release_seed_acquisition = recording_release
    manager.admit = recording_admit
    responses = pending.observe(
        "/dvpg_gq_orin_11/fable/vehicle/predicates",
        {
            "schema_version": "vehicle_predicate_observation.v1",
            "occurrence_id": "observed-pass-1",
            "predicate_id": "PASSES",
            "truth": True,
            "confidence": 0.92,
            "event_time_interval": {
                "start": BASE_TIME.isoformat(),
                "end": BASE_TIME.isoformat(),
            },
            "bindings": {"leader": "vehicle-17", "reference": "gate"},
            # Runtime providers declare their node-scoped identity; the seed
            # gateway must resolve it to the one allowed deployment source.
            "source_ids": ["dvpg_gq_orin_11"],
            "provider_id": "pass_reference_evaluator",
            "provider_version": "1",
        },
    )

    assert watching.status == "WATCHING"
    assert len(responses) == 1
    assert responses[0].status == "ADMITTED"
    assert responses[0].command_count > 0
    assert handoff_order[:2] == ["release", "admit"]
    assert dispatcher.submissions


def test_joint_fable_watches_share_trusted_seed_within_exact_replay(
    monkeypatch,
) -> None:
    manager, _ = _manager()
    pending = PendingSeedRegistry(manager)
    monkeypatch.setenv("FABLE_JOINT_RESOURCE_EPOCH_PLANNING", "1")
    requests = [
        LiveComplexEventRequest(
            request_id=request_id,
            submitter_id="watch-client",
            run_id="run-shared-seed",
            trace_id="trace-shared-seed",
            replay_id="shared-replay-1",
            family_id="convoy",
            baseline_id=BaselineId.FABLE,
            seed_graph_node_key="leader_passes",
            allowed_seed_source_ids=("orin11_camera",),
            max_seed_hypotheses=1,
        )
        for request_id in ("shared-seed-a", "shared-seed-b")
    ]
    assert all(pending.register(request).status == "WATCHING" for request in requests)
    payload = {
        "schema_version": "fable.predicate_result.v1",
        "request_id": requests[0].request_id,
        "occurrence_id": "shared-observed-pass-1",
        "truth": "TRUE",
        "confidence": 0.92,
        "semantic_predicate": {
            "predicate_id": "PASSES",
            "parameters": {},
        },
        "event_time_interval": {
            "start": BASE_TIME.isoformat(),
            "end": BASE_TIME.isoformat(),
        },
        "binding_delta": {
            "introduced": {"leader": "vehicle-17", "reference": "gate"},
            "validated": {},
        },
        "provenance": {
            "provider_id": "pass_reference_evaluator",
            "provider_contract_version": 1,
            "node_id": "dvpg_gq_orin_11",
            "source_ids": ["orin11_camera"],
        },
    }

    responses = pending.observe("fable/v1/durable/seed-result", payload)

    assert {response.request_id for response in responses} == {
        "shared-seed-a",
        "shared-seed-b",
    }
    assert all(response.status == "ADMITTED" for response in responses)


def test_b1_releases_provisional_capacity_before_authoritative_admission() -> None:
    manager, _ = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="watched-b1-convoy",
        submitter_id="watch-client",
        run_id="run-watch-b1",
        trace_id="trace-watch-b1",
        family_id="convoy",
        baseline_id=BaselineId.B1_STATIC_WHOLE_EVENT,
        seed_graph_node_key="leader_passes",
        allowed_seed_source_ids=("orin11_camera",),
        max_seed_hypotheses=1,
    )
    assert pending.register(request).status == "WATCHING"
    registered = pending._pending[request.request_id]
    assert registered.acquisition_demand_ids == {
        str(demand.demand_id)
        for batch in manager.dispatcher.submissions
        for candidate in batch
        for demand in candidate.demands
    }
    handoff_order = []
    release_seed_acquisition = manager.release_seed_acquisition
    admit = manager.admit

    def recording_release(*args, **kwargs):
        handoff_order.append("release")
        return release_seed_acquisition(*args, **kwargs)

    def recording_admit(*args, **kwargs):
        handoff_order.append("admit")
        return admit(*args, **kwargs)

    manager.release_seed_acquisition = recording_release
    manager.admit = recording_admit
    responses = pending.observe(
        "/dvpg_gq_orin_11/fable/vehicle/predicates",
        {
            "schema_version": "vehicle_predicate_observation.v1",
            "occurrence_id": "observed-b1-pass-1",
            "predicate_id": "PASSES",
            "truth": True,
            "confidence": 0.92,
            "event_time_interval": {
                "start": BASE_TIME.isoformat(),
                "end": BASE_TIME.isoformat(),
            },
            "bindings": {"leader": "vehicle-17", "reference": "gate"},
            "source_ids": ["dvpg_gq_orin_11"],
            "provider_id": "pass_reference_evaluator",
            "provider_version": "1",
        },
    )

    assert responses and responses[0].status == "ADMITTED"
    assert handoff_order[:2] == ["release", "admit"]


def test_pending_request_can_fan_out_to_bounded_distinct_seed_hypotheses() -> None:
    manager, dispatcher = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="multi-seed-convoy",
        submitter_id="watch-client",
        run_id="run-watch",
        trace_id="trace-watch",
        family_id="convoy",
        parameters={"evaluation_profile": "sequential_passes"},
        baseline_id=BaselineId.B4_GREEDY_FRONTIER,
        seed_graph_node_key="leader_passes",
        allowed_seed_source_ids=("orin11_camera",),
        max_seed_hypotheses=2,
    )
    pending.register(request)

    def observation(occurrence_id: str, vehicle: str):
        return {
            "schema_version": "vehicle_predicate_observation.v1",
            "occurrence_id": occurrence_id,
            "predicate_id": "PASSES",
            "truth": True,
            "confidence": 0.92,
            "event_time_interval": {
                "start": BASE_TIME.isoformat(),
                "end": BASE_TIME.isoformat(),
            },
            "bindings": {
                "vehicle": vehicle,
                "reference": "camera_fov:orin11",
            },
            "source_ids": ["orin11_camera"],
            "provider_id": "pass_reference_evaluator",
            "provider_version": "1",
        }

    initial = pending.observe(
        "/dvpg_gq_orin_11/fable/vehicle/predicates",
        observation("observed-pass-1", "vehicle-17"),
    )
    additional = pending.observe(
        "/dvpg_gq_orin_11/fable/vehicle/predicates",
        observation("observed-pass-2", "vehicle-23"),
    )

    assert initial[0].seed_action == "INITIAL_ADMISSION"
    state = (
        manager._executions[request.request_id]
        .request_state(request.request_id)
    )
    assert {
        demand.bound_roles.get("reference")
        for demand in state.whole_event_demands
    } == {"camera_fov:orin11"}
    follower_demands = tuple(
        demand
        for demand in state.whole_event_demands
        if demand.semantic_predicate.predicate_id == "PASSES"
        and "follower" in {
            role.variable for role in demand.semantic_predicate.roles
        }
    )
    assert follower_demands
    assert all(
        demand.eligible_source_ids == ("orin11_camera",)
        for demand in follower_demands
    )
    assert additional[0].seed_action == "ADDITIONAL_HYPOTHESIS"
    assert additional[0].active_seed_hypothesis_count == 2
    assert len(
        state.runtime.active_hypotheses
    ) == 2
    assert len(dispatcher.submissions) >= 2


def test_reference_diverse_seed_admission_does_not_let_track_fragments_fill_slots() -> None:
    manager, dispatcher = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="reference-diverse-stalking",
        submitter_id="watch-client",
        run_id="run-watch",
        trace_id="trace-watch",
        family_id="repeated_visit",
        parameters={
            "visit_count": 2,
            "evaluation_profile": "uncalibrated_passes",
        },
        baseline_id=BaselineId.B4_GREEDY_FRONTIER,
        seed_graph_node_key="first_visit",
        max_seed_hypotheses=2,
        seed_admission_strategy="reference_diverse",
    )
    pending.register(request)

    def observation(
        occurrence_id: str,
        vehicle: str,
        source: str,
        reference: str,
    ):
        return {
            "schema_version": "vehicle_predicate_observation.v1",
            "occurrence_id": occurrence_id,
            "predicate_id": "PASSES",
            "truth": True,
            "confidence": 0.92,
            "event_time_interval": {
                "start": BASE_TIME.isoformat(),
                "end": BASE_TIME.isoformat(),
            },
            "bindings": {"vehicle": vehicle, "reference": reference},
            "source_ids": [source],
            "provider_id": "pass_reference_evaluator",
            "provider_version": "1",
        }

    initial = pending.observe(
        "/orin11/predicates",
        observation(
            "pass-1",
            "orin11:track-1",
            "orin11_camera",
            "camera_fov:orin11",
        ),
    )
    fragmented = pending.observe(
        "/orin11/predicates",
        observation(
            "pass-2",
            "orin11:track-2",
            "orin11_camera",
            "camera_fov:orin11",
        ),
    )
    another_fragment = pending.observe(
        "/orin11/predicates",
        observation(
            "pass-2b",
            "orin11:track-3",
            "orin11_camera",
            "camera_fov:orin11",
        ),
    )
    other_view = pending.observe(
        "/orin12/predicates",
        observation(
            "pass-3",
            "orin12:track-1",
            "orin12_camera",
            "camera_fov:orin12",
        ),
    )

    assert initial[0].seed_action == "INITIAL_ADMISSION"
    assert fragmented[0].seed_action == "DUPLICATE_IGNORED"
    assert fragmented[0].active_seed_hypothesis_count == 1
    assert another_fragment == ()
    assert other_view[0].seed_action == "ADDITIONAL_HYPOTHESIS"
    state = manager._executions[request.request_id].request_state(
        request.request_id
    )
    assert len(state.runtime.active_hypotheses) == 2
    assert len(dispatcher.submissions) >= 2


def test_reference_bounded_seed_admission_reserves_capacity_for_later_cameras() -> None:
    manager, _ = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="reference-bounded-stalking",
        submitter_id="watch-client",
        run_id="run-watch",
        trace_id="trace-watch",
        family_id="repeated_visit",
        parameters={"visit_count": 2, "evaluation_profile": "uncalibrated_passes"},
        baseline_id=BaselineId.B4_GREEDY_FRONTIER,
        seed_graph_node_key="first_visit",
        allowed_seed_node_ids=("orin11", "orin14"),
        max_seed_hypotheses=4,
        seed_admission_strategy="reference_bounded",
    )
    pending.register(request)

    def observed(index: int, node: str):
        return {
            "schema_version": "vehicle_predicate_observation.v1",
            "occurrence_id": f"pass-{node}-{index}",
            "predicate_id": "PASSES",
            "truth": True,
            "confidence": 0.9,
            "event_time_interval": {
                "start": BASE_TIME.isoformat(),
                "end": BASE_TIME.isoformat(),
            },
            "bindings": {
                "vehicle": f"{node}:track-{index}",
                "reference": f"camera_fov:{node}",
            },
            "source_ids": [node],
            "node_id": node,
            "provider_id": "pass_reference_evaluator",
            "provider_version": "1",
        }

    first = pending.observe("/orin11/predicates", observed(1, "orin11"))
    first_hypothesis_id = first[0].hypothesis_ids[0]
    assert pending.observe("/orin11/predicates", observed(2, "orin11"))
    replacement = pending.observe(
        "/orin11/predicates", observed(3, "orin11")
    )
    assert replacement[0].seed_action == "ADDITIONAL_HYPOTHESIS"
    state = manager._executions[request.request_id].request_state(
        request.request_id
    )
    assert (
        state.runtime.get_hypothesis(UUID(first_hypothesis_id)).lifecycle.value
        == "INVALIDATED"
    )
    assert pending.observe("/orin14/predicates", observed(1, "orin14"))
    final = pending.observe("/orin14/predicates", observed(2, "orin14"))
    assert final
    assert final[0].active_seed_hypothesis_count == 4


def test_reference_bounded_seed_deduplicates_same_camera_local_identity() -> None:
    manager, _ = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="reference-bounded-dedup",
        submitter_id="watch-client",
        run_id="run-watch",
        trace_id="trace-watch",
        family_id="repeated_visit",
        parameters={"visit_count": 3, "evaluation_profile": "uncalibrated_passes"},
        baseline_id=BaselineId.B4_GREEDY_FRONTIER,
        seed_graph_node_key="first_visit",
        allowed_seed_node_ids=("orin11", "orin14"),
        max_seed_hypotheses=4,
        seed_admission_strategy="reference_bounded",
    )
    pending.register(request)

    def observed(occurrence: str):
        return {
            "schema_version": "vehicle_predicate_observation.v1",
            "occurrence_id": occurrence,
            "predicate_id": "PASSES",
            "truth": True,
            "confidence": 0.9,
            "event_time_interval": {
                "start": BASE_TIME.isoformat(),
                "end": BASE_TIME.isoformat(),
            },
            "bindings": {
                "vehicle": "orin11:stable-track",
                "reference": "camera_fov:orin11",
            },
            "source_ids": ["orin11"],
            "node_id": "orin11",
            "provider_id": "pass_reference_evaluator",
            "provider_version": "1",
        }

    first = pending.observe("/orin11/predicates", observed("pass-1"))
    duplicate = pending.observe("/orin11/predicates", observed("pass-1-extended"))

    assert first[0].seed_action == "INITIAL_ADMISSION"
    assert duplicate[0].seed_action == "DUPLICATE_IGNORED"
    assert duplicate[0].active_seed_hypothesis_count == 1


def test_reference_bounded_reconciles_invalidated_candidate_slots() -> None:
    manager, _ = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="reference-bounded-reconcile",
        submitter_id="watch-client",
        run_id="run-watch",
        trace_id="trace-watch",
        family_id="repeated_visit",
        parameters={"visit_count": 3, "evaluation_profile": "uncalibrated_passes"},
        baseline_id=BaselineId.B4_GREEDY_FRONTIER,
        seed_graph_node_key="first_visit",
        allowed_seed_node_ids=("orin11",),
        max_seed_hypotheses=1,
        seed_admission_strategy="reference_bounded",
    )
    pending.register(request)

    def observed(index: int):
        return {
            "schema_version": "vehicle_predicate_observation.v1",
            "occurrence_id": f"pass-{index}",
            "predicate_id": "PASSES",
            "truth": True,
            "confidence": 0.9,
            "event_time_interval": {
                "start": BASE_TIME.isoformat(),
                "end": BASE_TIME.isoformat(),
            },
            "bindings": {
                "vehicle": f"orin11:track-{index}",
                "reference": "camera_fov:orin11",
            },
            "source_ids": ["orin11"],
            "node_id": "orin11",
            "provider_id": "pass_reference_evaluator",
            "provider_version": "1",
        }

    first = pending.observe("/orin11/predicates", observed(1))
    first_id = UUID(first[0].hypothesis_ids[0])
    state = manager._executions[request.request_id].request_state(request.request_id)
    assert state.runtime.invalidate_unprogressed_hypothesis(
        first_id,
        seed_event_time=state.runtime.get_hypothesis(first_id).event_time_window,
        minimum_progress_gap_ms=30_000,
        force=True,
    )

    replacement = pending.observe("/orin11/predicates", observed(2))

    assert replacement[0].accepted
    assert replacement[0].seed_action == "ADDITIONAL_HYPOTHESIS"
    assert replacement[0].active_seed_hypothesis_count == 1
    assert len(pending._pending[request.request_id].admitted_seed_candidates or []) == 1


def test_pending_request_ignores_wrong_predicate_and_disallowed_source() -> None:
    manager, dispatcher = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="filtered-watch",
        submitter_id="watch-client",
        run_id="run-watch",
        trace_id="trace-watch",
        family_id="convoy",
        baseline_id=BaselineId.B4_GREEDY_FRONTIER,
        seed_graph_node_key="leader_passes",
        allowed_seed_source_ids=("orin11_camera",),
    )
    pending.register(request)

    assert pending.observe(
        "/dvpg_gq_orin_11/fable/vehicle/predicates",
        {
            "schema_version": "vehicle_predicate_observation.v1",
            "occurrence_id": "wrong",
            "predicate_id": "MOVING",
            "truth": True,
            "confidence": 1.0,
            "event_time_interval": {
                "start": BASE_TIME.isoformat(),
                "end": BASE_TIME.isoformat(),
            },
            "bindings": {},
            "source_ids": ["other_camera"],
            "provider_id": "motion_state_evaluator",
            "provider_version": "1",
        },
    ) == ()
    # Registration itself dispatches the leased seed-acquisition frontier;
    # an unrelated observation must not create an additional plan.
    assert len(dispatcher.submissions) == 1


def test_pending_request_ignores_seed_outside_labeled_event_time_window() -> None:
    manager, dispatcher = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="time-scoped-watch",
        submitter_id="watch-client",
        run_id="run-watch",
        trace_id="trace-watch",
        family_id="convoy",
        parameters={"evaluation_profile": "sequential_passes"},
        baseline_id=BaselineId.B4_GREEDY_FRONTIER,
        seed_graph_node_key="leader_passes",
        allowed_seed_source_ids=("orin11_camera",),
        allowed_seed_event_time_interval=EventTimeInterval(
            start=BASE_TIME + timedelta(minutes=2),
            end=BASE_TIME + timedelta(minutes=3),
        ),
    )
    pending.register(request)

    assert pending.observe(
        "/dvpg_gq_orin_11/fable/vehicle/predicates",
        {
            "schema_version": "vehicle_predicate_observation.v1",
            "occurrence_id": "too-early",
            "predicate_id": "PASSES",
            "truth": True,
            "confidence": 1.0,
            "event_time_interval": {
                "start": BASE_TIME.isoformat(),
                "end": (BASE_TIME + timedelta(seconds=2)).isoformat(),
            },
            "bindings": {
                "vehicle": "vehicle-1",
                "reference": "camera_fov:orin11",
            },
            "source_ids": ["orin11_camera"],
            "provider_id": "pass_reference_evaluator",
            "provider_version": "1",
        },
    ) == ()
    # The out-of-window observation is ignored while the explicit seed
    # acquisition plan remains the only dispatch.
    assert len(dispatcher.submissions) == 1


def test_pending_seed_watch_can_be_cancelled_idempotently() -> None:
    manager, _ = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="cancel-pending",
        submitter_id="watch-client",
        run_id="run-watch",
        trace_id="trace-watch",
        family_id="convoy",
        baseline_id=BaselineId.B4_GREEDY_FRONTIER,
        seed_graph_node_key="leader_passes",
    )
    pending.register(request)
    cancellation = LiveComplexEventCancelRequest(
        request_id=request.request_id,
        submitter_id=request.submitter_id,
    )

    first = pending.cancel(cancellation)
    second = pending.cancel(cancellation)

    assert first.status == "CANCELLED"
    assert first.cancelled_pending_watch
    assert second.status == "CANCELLED"
    assert second.cancel_command_count > 0


def test_active_live_request_cancellation_removes_execution() -> None:
    manager, dispatcher = _manager()
    request = _request(request_id="cancel-active")
    assert manager.admit(request).response.status == "ADMITTED"

    response = manager.cancel(
        LiveComplexEventCancelRequest(
            request_id=request.request_id,
            submitter_id=request.submitter_id,
            reason="test cleanup",
        )
    )

    assert response.status == "CANCELLED"
    assert response.cancelled_active_execution
    assert not manager.has_request(request.request_id)
    assert dispatcher.cancelled_leases == []


def test_b1_buffers_only_current_replay_track_history_before_seed() -> None:
    manager, _ = _manager()
    pending = PendingSeedRegistry(manager)
    request = LiveComplexEventRequest(
        request_id="watched-b1-track-history",
        submitter_id="watch-client",
        run_id="run-watch-b1-tracks",
        trace_id="trace-watch-b1-tracks",
        replay_id="replay-current",
        family_id="robbery",
        baseline_id=BaselineId.B1_STATIC_WHOLE_EVENT,
        seed_graph_node_key="alarm_branch",
        allowed_seed_source_ids=("orin16_microphone",),
        max_seed_hypotheses=1,
    )
    assert pending.register(request).status == "WATCHING"

    payload = {
        "schema_version": "track_set.v1",
        "source_id": "orin15_camera",
        "replay_id": "replay-current",
        "event_time": BASE_TIME.isoformat(),
        "tracks": [{
            "scoped_track_id": "orin15:vehicle-7",
            "class_name": "car",
            "confidence": 0.88,
            "event_time": BASE_TIME.isoformat(),
        }],
    }
    assert pending.observe("/orin15_camera/fable/vehicle/tracks", payload) == ()
    history = pending._pending[request.request_id].static_vehicle_history
    assert len(history) == 1
    assert history[0]["bindings"] == {"vehicle": "orin15:vehicle-7"}

    assert pending.observe(
        "/orin15_camera/fable/vehicle/tracks",
        dict(payload, replay_id="replay-stale"),
    ) == ()
    assert len(history) == 1
