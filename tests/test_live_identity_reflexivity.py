from __future__ import annotations

from evaluation.live_execution import AuthoritativeLiveExecution
from datetime import timedelta
import json
import threading
from types import SimpleNamespace

from fable.common.enums import HypothesisLifecycle, TruthValue
from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.common.schemas import PredicateRole
from fable.common.time import EventTimeInterval
from fable.planning import DemandCompiler, default_predicate_registry
from fable.planning.testing import fake_follow_demand
from fable.planning.testing import fake_deployment
from fable.semantic import (
    ApplyStatus,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)
from fable.semantic.definitions import uncalibrated_repeated_pass_graph
from fable.distributed.models import AgentProviderStatus, ReplayOutputAdapter, RuntimeMode
from fable.distributed.node_agent import NodeAgent, _identity_comparison_demand_payload


def test_equal_scoped_identities_resolve_without_provider_execution() -> None:
    original = fake_follow_demand()
    demand = original.model_copy(
        update={
            "semantic_predicate": original.semantic_predicate.model_copy(
                update={
                    "predicate_id": "SAME_ENTITY",
                    "roles": (
                        PredicateRole(role_name="left", variable="vehicle", entity_type="vehicle"),
                        PredicateRole(
                            role_name="right",
                            variable="visit_vehicle_2",
                            entity_type="vehicle",
                        ),
                    ),
                }
            ),
            "bound_roles": {"left": "camera-a:session:7", "right": "camera-a:session:7"},
            "unbound_roles": (),
        }
    )
    result = AuthoritativeLiveExecution._reflexive_identity_result(
        demand, now=BASE_TIME
    )

    assert result is not None
    assert result.truth == TruthValue.TRUE
    assert result.confidence == 1.0
    assert result.provenance.provider_id == "identity_reflexivity"
    assert result.provenance.source_ids == ("orchestrator:identity_reflexivity",)
    # Results validate graph variables, never provider-facing role labels.
    assert result.binding_delta.validated == {
        "vehicle": "camera-a:session:7",
        "visit_vehicle_2": "camera-a:session:7",
    }


def test_distinct_scoped_identities_still_require_identity_provider() -> None:
    original = fake_follow_demand()
    demand = original.model_copy(
        update={
            "semantic_predicate": original.semantic_predicate.model_copy(
                update={
                    "predicate_id": "SAME_ENTITY",
                    "roles": (
                        PredicateRole(role_name="left", variable="vehicle", entity_type="vehicle"),
                        PredicateRole(
                            role_name="right",
                            variable="visit_vehicle_2",
                            entity_type="vehicle",
                        ),
                    ),
                }
            ),
            "bound_roles": {"left": "camera-a:session:7", "right": "camera-b:session:7"},
            "unbound_roles": (),
        }
    )
    assert (
        AuthoritativeLiveExecution._reflexive_identity_result(
            demand, now=BASE_TIME
        )
        is None
    )


def test_identity_provider_activation_publishes_exact_bound_pair() -> None:
    original = fake_follow_demand()
    predicate = original.semantic_predicate.model_copy(
        update={
            "predicate_id": "SAME_ENTITY",
            "roles": (
                PredicateRole(role_name="left", variable="vehicle", entity_type="vehicle"),
                PredicateRole(
                    role_name="right",
                    variable="visit_vehicle_2",
                    entity_type="vehicle",
                ),
            ),
        }
    )
    demand = original.model_copy(
        update={
            "semantic_predicate": predicate,
            "bound_roles": {
                "left": "orin11:session:first",
                "right": "orin11:session:second",
            },
            "unbound_roles": (),
        }
    )
    command = SimpleNamespace(
        runtime=SimpleNamespace(provider_id="cross_sensor_identity_association"),
        demand=demand,
    )

    payload = _identity_comparison_demand_payload(command)

    assert payload is not None
    document = json.loads(payload)
    assert document["left_local_entity_id"] == "orin11:session:first"
    assert document["right_local_entity_id"] == "orin11:session:second"
    assert document["entity_kind"] == "vehicle"


def test_identity_demand_is_published_after_worker_readiness() -> None:
    original = fake_follow_demand()
    demand = original.model_copy(
        update={
            "semantic_predicate": original.semantic_predicate.model_copy(
                update={
                    "predicate_id": "SAME_ENTITY",
                    "roles": (
                        PredicateRole(
                            role_name="left",
                            variable="vehicle",
                            entity_type="vehicle",
                        ),
                        PredicateRole(
                            role_name="right",
                            variable="visit_vehicle_2",
                            entity_type="vehicle",
                        ),
                    ),
                }
            ),
            "bound_roles": {
                "left": "orin11:session:first",
                "right": "orin11:session:second",
            },
            "unbound_roles": (),
        }
    )
    runtime = SimpleNamespace(
        provider_id="cross_sensor_identity_association",
        readiness=SimpleNamespace(ready_field="ready", ready_value=True),
        mode=RuntimeMode.ADOPT_EXISTING,
    )
    command = SimpleNamespace(runtime=runtime, demand=demand)
    published = []
    agent = NodeAgent.__new__(NodeAgent)
    agent._lock = threading.RLock()
    agent.providers = {
        "identity-instance": SimpleNamespace(
            runtime=runtime,
            status=AgentProviderStatus.STARTING,
            updated_at=BASE_TIME,
            worker_id="identity-worker",
            active_leases={"lease": command},
        )
    }
    agent.workers = {}
    agent.transport = SimpleNamespace(
        publish=lambda topic, payload, **kwargs: published.append(
            (topic, json.loads(payload), kwargs)
        )
    )
    agent._publish_provider_status_record = lambda _record: None

    agent._on_readiness(
        "identity-instance",
        "/readiness/x86server/fable_identity",
        json.dumps({"ready": True}).encode(),
    )

    assert agent.providers["identity-instance"].status == AgentProviderStatus.READY
    assert len(published) == 1
    assert published[0][0] == "/fable/identity/demands"
    assert published[0][1]["left_local_entity_id"] == "orin11:session:first"
    assert published[0][1]["right_local_entity_id"] == "orin11:session:second"
    assert published[0][2] == {"qos": 1, "retain": False}


def test_shared_identity_worker_readiness_publishes_all_sibling_demands() -> None:
    """One worker readiness transition must release every attached pair."""

    original = fake_follow_demand()
    predicate = original.semantic_predicate.model_copy(
        update={
            "predicate_id": "SAME_ENTITY",
            "roles": (
                PredicateRole(
                    role_name="left", variable="vehicle", entity_type="vehicle"
                ),
                PredicateRole(
                    role_name="right",
                    variable="visit_vehicle_2",
                    entity_type="vehicle",
                ),
            ),
        }
    )
    first_demand = original.model_copy(
        update={
            "semantic_predicate": predicate,
            "bound_roles": {
                "left": "orin11:session:first",
                "right": "orin11:session:second",
            },
            "unbound_roles": (),
        }
    )
    second_demand = first_demand.model_copy(
        update={
            "demand_id": uuid7(),
            "bound_roles": {
                "left": "orin11:session:first",
                "right": "orin11:session:third",
            },
        }
    )
    runtime = SimpleNamespace(
        provider_id="cross_sensor_identity_association",
        readiness=SimpleNamespace(ready_field="ready", ready_value=True),
        mode=RuntimeMode.ADOPT_EXISTING,
    )
    first_command = SimpleNamespace(runtime=runtime, demand=first_demand)
    second_command = SimpleNamespace(runtime=runtime, demand=second_demand)
    published = []
    agent = NodeAgent.__new__(NodeAgent)
    agent._lock = threading.RLock()
    agent.providers = {
        "identity-first": SimpleNamespace(
            runtime=runtime,
            status=AgentProviderStatus.STARTING,
            updated_at=BASE_TIME,
            worker_id="identity-worker",
            active_leases={"first": first_command},
        ),
        "identity-second": SimpleNamespace(
            runtime=runtime,
            status=AgentProviderStatus.STARTING,
            updated_at=BASE_TIME,
            worker_id="identity-worker",
            active_leases={"second": second_command},
        ),
    }
    agent.workers = {
        "identity-worker": SimpleNamespace(status=AgentProviderStatus.STARTING)
    }
    agent.transport = SimpleNamespace(
        publish=lambda topic, payload, **kwargs: published.append(
            (topic, json.loads(payload), kwargs)
        )
    )
    agent._publish_provider_status_record = lambda _record: None

    agent._on_readiness(
        "identity-first",
        "/readiness/x86server/fable_identity",
        json.dumps({"ready": True}).encode(),
    )

    assert all(
        record.status == AgentProviderStatus.READY
        for record in agent.providers.values()
    )
    assert len(published) == 2
    assert {
        item[1]["right_local_entity_id"] for item in published
    } == {"orin11:session:second", "orin11:session:third"}

    # The same retained readiness document can be delivered once per logical
    # subscription. It must not replay prior demands or generate status churn.
    agent._on_readiness(
        "identity-second",
        "/readiness/x86server/fable_identity",
        json.dumps({"ready": True}).encode(),
    )
    assert len(published) == 2


def test_shared_identity_worker_output_fans_out_to_all_logical_leases() -> None:
    """A single MQTT callback must evaluate every sibling exact-pair demand."""

    runtime = SimpleNamespace(
        output_adapter=ReplayOutputAdapter.IDENTITY_ASSOCIATION,
    )
    first_command = SimpleNamespace(demand=SimpleNamespace(demand_id=uuid7()))
    second_command = SimpleNamespace(demand=SimpleNamespace(demand_id=uuid7()))
    agent = NodeAgent.__new__(NodeAgent)
    agent._lock = threading.RLock()
    agent.providers = {
        "identity-first": SimpleNamespace(
            provider_instance_id="identity-first",
            provider_id="cross_sensor_identity_association",
            worker_id="identity-worker",
            runtime=runtime,
            status=AgentProviderStatus.READY,
            active_leases={"first": first_command},
        ),
        "identity-second": SimpleNamespace(
            provider_instance_id="identity-second",
            provider_id="cross_sensor_identity_association",
            worker_id="identity-worker",
            runtime=runtime,
            status=AgentProviderStatus.READY,
            active_leases={"second": second_command},
        ),
    }
    agent.workers = {
        "identity-worker": SimpleNamespace(
            handle=SimpleNamespace(running=True),
            status=AgentProviderStatus.READY,
        )
    }
    buffered = []
    forwarded = []
    agent._buffer_provider_output = lambda *args: buffered.append(args)
    agent._forward_provider_document = lambda **kwargs: forwarded.append(kwargs)

    agent._on_provider_output(
        "identity-second",
        "/fable/identity/associations",
        json.dumps({"associations": []}).encode(),
    )

    assert len(buffered) == 1
    assert {
        (item["provider_instance_id"], item["command"].demand.demand_id)
        for item in forwarded
    } == {
        ("identity-first", first_command.demand.demand_id),
        ("identity-second", second_command.demand.demand_id),
    }

def test_demand_generated_reflexive_results_complete_three_visit_graph() -> None:
    vehicle = "orin11:session:track0"
    reference = "camera_fov:orin11"
    runtime = SemanticRuntime(
        uncalibrated_repeated_pass_graph(
            visit_count=3,
            minimum_return_gap_ms=30_000,
            identity_confirmation=True,
        ),
        config=SemanticRuntimeConfig(
            request_id="live-reflexive-three-visit",
            hypothesis_horizon_ms=600_000,
            deadline_offset_ms=600_000,
        ),
    )
    compiler = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=fake_deployment(),
    )
    seed = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="first_visit",
            source_id="orin11_camera",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(seconds=1),
            ),
            introduced={"vehicle": vehicle, "visit_reference": reference},
        ),
    )
    hypothesis_id = runtime.seed(seed).hypothesis_ids[-1]

    for index, (node_key, variable, seconds) in enumerate(
        (
            ("return_visit", "visit_vehicle_2", 78),
            ("return_visit_2", "visit_vehicle_3", 152),
        ),
        start=1,
    ):
        visit = predicate_result_from_spec(
            runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key=node_key,
                source_id="orin11_camera",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=seconds),
                    end=BASE_TIME + timedelta(seconds=seconds + 1),
                ),
                introduced={variable: vehicle},
                validated={"visit_reference": reference},
            ),
        )
        visit_transition = runtime.apply(visit)
        assert visit_transition.status == ApplyStatus.FORKED, visit_transition.reason
        hypothesis_id = visit_transition.hypothesis_ids[-1]

        hypothesis = runtime.get_hypothesis(hypothesis_id)
        frontier = runtime.get_frontier(hypothesis_id)
        assert frontier is not None
        demands = compiler.compile_frontier(
            graph=runtime.graph,
            hypothesis=hypothesis,
            frontier=frontier,
        )
        assert len(demands) == 1
        identity = AuthoritativeLiveExecution._reflexive_identity_result(
            demands[0], now=BASE_TIME + timedelta(seconds=seconds + 1)
        )
        assert identity is not None
        identity_transition = runtime.apply(identity)
        assert identity_transition.status == ApplyStatus.APPLIED
        hypothesis_id = identity_transition.hypothesis_ids[-1]

    assert runtime.get_hypothesis(hypothesis_id).lifecycle == HypothesisLifecycle.COMPLETED


def test_three_visit_non_reflexive_frontier_compiles_same_entity_demand() -> None:
    """A later PASSES must advance into an executable identity frontier.

    This is the exact graph-progression boundary that failed in the E4 trace:
    the semantic result was accepted, but demand compilation raised because
    SAME_ENTITY was absent from the packaged logical predicate catalog.
    """

    runtime = SemanticRuntime(
        uncalibrated_repeated_pass_graph(
            visit_count=3,
            minimum_return_gap_ms=10_000,
            identity_confirmation=True,
        ),
        config=SemanticRuntimeConfig(
            request_id="three-visit-non-reflexive-frontier",
            hypothesis_horizon_ms=600_000,
            deadline_offset_ms=600_000,
        ),
    )
    seed = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="first_visit",
            source_id="orin11_camera",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(seconds=1),
            ),
            introduced={
                "vehicle": "orin11:first-track",
                "visit_reference": "camera_fov:orin11",
            },
        ),
    )
    hypothesis_id = runtime.seed(seed).hypothesis_ids[-1]
    second = predicate_result_from_spec(
        runtime,
        hypothesis_id,
        ScriptedResultSpec(
            node_key="return_visit",
            source_id="orin11_camera",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME + timedelta(seconds=20),
                end=BASE_TIME + timedelta(seconds=21),
            ),
            introduced={"visit_vehicle_2": "orin11:second-track"},
            validated={"visit_reference": "camera_fov:orin11"},
        ),
    )
    transition = runtime.apply(second)
    assert transition.status == ApplyStatus.FORKED
    progressed_id = transition.hypothesis_ids[-1]
    frontier = runtime.get_frontier(progressed_id)
    assert frontier is not None

    demands = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=fake_deployment(),
    ).compile_frontier(
        graph=runtime.graph,
        hypothesis=runtime.get_hypothesis(progressed_id),
        frontier=frontier,
    )

    assert len(demands) == 1
    assert demands[0].semantic_predicate.predicate_id == "SAME_ENTITY"
    assert demands[0].required_capabilities == (
        "vehicle_identity",
        "cross_sensor_identity",
    )


def test_same_entity_demand_is_anchored_to_bound_identity_camera() -> None:
    runtime = SemanticRuntime(
        uncalibrated_repeated_pass_graph(
            visit_count=3,
            minimum_return_gap_ms=10_000,
            identity_confirmation=True,
        ),
        config=SemanticRuntimeConfig(
            request_id="camera-anchored-identity-frontier",
            hypothesis_horizon_ms=600_000,
            deadline_offset_ms=600_000,
        ),
    )
    seed = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="first_visit",
            source_id="camera_mobile",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(seconds=1),
            ),
            introduced={
                "vehicle": "camera_mobile:first-track",
                "visit_reference": "camera_fov:camera_mobile",
            },
        ),
    )
    hypothesis_id = runtime.seed(seed).hypothesis_ids[-1]
    second = predicate_result_from_spec(
        runtime,
        hypothesis_id,
        ScriptedResultSpec(
            node_key="return_visit",
            source_id="camera_mobile",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME + timedelta(seconds=20),
                end=BASE_TIME + timedelta(seconds=21),
            ),
            introduced={"visit_vehicle_2": "camera_mobile:second-track"},
            validated={"visit_reference": "camera_fov:camera_mobile"},
        ),
    )
    progressed_id = runtime.apply(second).hypothesis_ids[-1]
    frontier = runtime.get_frontier(progressed_id)
    assert frontier is not None
    demands = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=fake_deployment(),
    ).compile_frontier(
        graph=runtime.graph,
        hypothesis=runtime.get_hypothesis(progressed_id),
        frontier=frontier,
    )

    assert len(demands) == 1
    assert demands[0].semantic_predicate.predicate_id == "SAME_ENTITY"
    assert demands[0].eligible_source_ids == ("camera_mobile",)


def test_progressed_visit_chain_has_priority_over_new_rolling_seed() -> None:
    runtime = SemanticRuntime(
        uncalibrated_repeated_pass_graph(
            visit_count=3,
            minimum_return_gap_ms=30_000,
            identity_confirmation=True,
        ),
        config=SemanticRuntimeConfig(
            request_id="rolling-seed-priority",
            hypothesis_horizon_ms=600_000,
            deadline_offset_ms=600_000,
        ),
    )
    first = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="first_visit",
            source_id="orin11_camera",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(seconds=1),
            ),
            introduced={
                "vehicle": "orin11:track0",
                "visit_reference": "camera_fov:orin11",
            },
        ),
    )
    first_id = runtime.seed(first).hypothesis_ids[-1]
    second = predicate_result_from_spec(
        runtime,
        first_id,
        ScriptedResultSpec(
            node_key="return_visit",
            source_id="orin11_camera",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME + timedelta(seconds=78),
                end=BASE_TIME + timedelta(seconds=79),
            ),
            introduced={"visit_vehicle_2": "orin11:track0"},
            validated={"visit_reference": "camera_fov:orin11"},
        ),
    )
    progressed_id = runtime.apply(second).hypothesis_ids[-1]
    fresh = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="first_visit",
            source_id="orin11_camera",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME + timedelta(seconds=152),
                end=BASE_TIME + timedelta(seconds=153),
            ),
            introduced={
                "vehicle": "orin11:track-new",
                "visit_reference": "camera_fov:orin11",
            },
        ),
    )
    fresh_id = runtime.seed(fresh).hypothesis_ids[-1]

    priority = AuthoritativeLiveExecution._hypothesis_observation_priority
    assert priority(runtime.get_hypothesis(progressed_id)) > priority(
        runtime.get_hypothesis(fresh_id)
    )
