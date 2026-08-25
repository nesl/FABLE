from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import yaml

from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.demo import build_replay_multimodal_candidate
from fable.distributed.models import ReplayOutputAdapter, RuntimeMode
from fable.distributed.node_agent import NodeAgent
from fable.planning.provider_registry import ProviderRegistry
from fable.semantic import (
    ApplyStatus,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)
from fable.integrations.replay import build_replay_output_adapter_registry
from fable.semantic.phase8_examples import (
    drive_up_shooting_graph,
    multimodal_robbery_graph,
    package_exchange_graph,
)
from fable.semantic.definitions.multimodal import (
    multimodal_robbery_graph as authored_multimodal_robbery_graph,
)
from providers.multimodal.models import (
    AudioEventObservation,
    InteractionPredicateObservation,
)


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "iobt-minimal-ce-replay"


def registry() -> ProviderRegistry:
    return ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )


def candidate(predicate_id: str, *, label: str | None = None):
    return build_replay_multimodal_candidate(
        provider_registry=registry(),
        node_id="dvpg_gq_orin_11",
        source_id="dvpg_gq_orin_11",
        event_interval=EventTimeInterval(
            start=BASE_TIME,
            end=BASE_TIME + timedelta(minutes=5),
        ),
        predicate_id=predicate_id,
        label=label,
        now=BASE_TIME,
    )


def test_catalog_exposes_phase8_physical_alternatives() -> None:
    expected = {
        "AUDIO_EVENT": {
            "detect_audio_event",
            "audio_event_with_localization",
            "audio_event_visual_association",
        },
        "DISEMBARKS": {"person_vehicle_transition"},
        "BOARDS": {"person_vehicle_transition"},
        "CONVERSATION": {
            "conversation_proximity_diarization",
            "conversation_with_required_content",
        },
        "TRANSFER": {"package_transfer_high_resolution"},
    }
    provider_registry = registry()
    for predicate_id, chain_ids in expected.items():
        demand = candidate(predicate_id, label="gunshot").demands[0]
        observed = {item.chain_id for item in provider_registry.candidate_chains(demand)}
        assert chain_ids.issubset(observed)
    assert "recover_vehicle_before_audio_event" in provider_registry.chains
    assert "custody_state.v1" in provider_registry.data_types


def test_drive_up_gunshot_activates_history_and_live_tracking() -> None:
    runtime = SemanticRuntime(
        drive_up_shooting_graph(),
        config=SemanticRuntimeConfig(request_id="drive_up_phase8"),
    )
    result = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="gunshot",
            source_id="store_mic",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(milliseconds=100),
            ),
            introduced={"location": "store_front"},
        ),
    )
    transition = runtime.seed(result)
    assert transition.status == ApplyStatus.CREATED
    frontier = runtime.get_frontier(transition.hypothesis_ids[0])
    keys = {
        runtime.graph.nodes_by_id[node_id].authored_key
        for node_id in frontier.snapshot.enabled_node_ids
    }
    assert keys == {"recover_vehicle_history", "live_escape_tracking"}
    historical = runtime.graph.nodes_by_key["recover_vehicle_history"]
    live = runtime.graph.nodes_by_key["live_escape_tracking"]
    assert historical.annotations["execution_mode"] == "retrospective"
    assert live.annotations["execution_mode"] == "live"


def test_robbery_or_resolution_retires_losing_multimodal_work() -> None:
    runtime = SemanticRuntime(
        multimodal_robbery_graph(),
        config=SemanticRuntimeConfig(request_id="robbery_phase8"),
    )
    seeded = runtime.seed(
        seed_result_from_spec(
            runtime,
            ScriptedResultSpec(
                node_key="entry",
                source_id="store_camera",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(seconds=1),
                ),
                introduced={"person": "person_7", "location": "store"},
            ),
        )
    )
    hypothesis_id = seeded.hypothesis_ids[0]
    transition = runtime.apply(
        predicate_result_from_spec(
            runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key="gunshot_branch",
                source_id="store_mic",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=2),
                    end=BASE_TIME + timedelta(seconds=2),
                ),
                validated={"location": "store"},
            ),
        )
    )
    assert transition.status == ApplyStatus.APPLIED
    assert set(transition.cancellation.branch_ids) == {"alarm", "threat"}
    next_frontier = runtime.get_frontier(hypothesis_id)
    keys = {
        runtime.graph.nodes_by_id[node_id].authored_key
        for node_id in next_frontier.snapshot.enabled_node_ids
    }
    assert keys == {"departure"}


def test_robbery_departure_is_independent_then_identity_is_explicit() -> None:
    graph = SemanticRuntime(
        authored_multimodal_robbery_graph(),
        config=SemanticRuntimeConfig(request_id="robbery-contract"),
    ).graph
    departure = graph.nodes_by_key["departure"]
    same_vehicle = graph.nodes_by_key["same_vehicle"]

    assert departure.predicate.predicate_id == "EXITS"
    assert departure.predicate.roles[0].variable == "departing_vehicle"
    assert same_vehicle.predicate.predicate_id == "SAME_ENTITY"
    assert {
        role.role_name: role.variable for role in same_vehicle.predicate.roles
    } == {"left": "vehicle", "right": "departing_vehicle"}
    assert same_vehicle.predicate.parameters["minimum_confidence"] == 0.40


def test_package_graph_marks_high_resolution_checkpoint_and_compact_continuation() -> None:
    graph = package_exchange_graph()
    transfer = next(node for node in graph.nodes if node.authored_key == "transfer")
    assert transfer.checkpoint_boundary is True
    assert transfer.annotations["analysis_mode"] == "high_resolution_interaction"
    assert transfer.annotations["continuation_artifact_types"] == ["custody_state.v1"] or transfer.annotations["continuation_artifact_types"] == ("custody_state.v1",)


def test_node_agent_adapts_typed_audio_and_interaction_outputs() -> None:
    agent = object.__new__(NodeAgent)
    agent.node_id = "dvpg_gq_orin_11"
    agent.output_adapters = build_replay_output_adapter_registry()
    audio_demand = candidate("AUDIO_EVENT", label="gunshot").demands[0]
    interval = EventTimeInterval(start=BASE_TIME, end=BASE_TIME + timedelta(seconds=1))
    audio = AudioEventObservation(
        occurrence_id="gunshot-1",
        label="gunshot",
        confidence=0.9,
        event_time_interval=interval,
        source_id="dvpg_gq_orin_11",
        provider_id="audio_event_classifier",
        provider_version="1",
        localized_zone_id="store_front",
    )
    adapted = agent._adapt_provider_output(
        SimpleNamespace(demand=audio_demand, runtime=SimpleNamespace(output_label_aliases={})),
        ReplayOutputAdapter.MULTIMODAL_PREDICATE,
        audio.model_dump(mode="json"),
    )
    assert adapted is not None and adapted[2] == {"location": "store_front"}

    aliased_role = audio_demand.semantic_predicate.roles[0].model_copy(
        update={"variable": "trigger_location"}
    )
    aliased_demand = audio_demand.model_copy(
        update={
            "semantic_predicate": audio_demand.semantic_predicate.model_copy(
                update={"roles": (aliased_role,)}
            )
        }
    )
    adapted_alias = agent._adapt_provider_output(
        SimpleNamespace(demand=aliased_demand, runtime=SimpleNamespace(output_label_aliases={})),
        ReplayOutputAdapter.MULTIMODAL_PREDICATE,
        audio.model_dump(mode="json"),
    )
    assert adapted_alias is not None
    assert adapted_alias[2] == {"trigger_location": "store_front"}

    transfer_demand = candidate("TRANSFER").demands[0]
    interaction = InteractionPredicateObservation(
        occurrence_id="transfer-1",
        predicate_id="TRANSFER",
        truth=True,
        confidence=0.85,
        event_time_interval=interval,
        bindings={"object": "bag_1", "source": "person_1", "destination": "person_2"},
        source_ids=("camera_a",),
        provider_id="object_transfer_reasoner",
        provider_version="1",
    )
    adapted = agent._adapt_provider_output(
        SimpleNamespace(demand=transfer_demand, runtime=SimpleNamespace(output_label_aliases={})),
        ReplayOutputAdapter.MULTIMODAL_PREDICATE,
        interaction.model_dump(mode="json"),
    )
    assert adapted is not None
    assert adapted[2] == interaction.bindings


def test_phase8_replay_overlay_uses_local_audio_ipc_and_typed_topics() -> None:
    compose = yaml.safe_load((REPLAY / "compose.fable.phase8.yaml").read_text())
    service = compose["services"]["fable-multimodal-orin11"]
    assert "/tmp/iobt-orin11:/tmp" in service["volumes"]
    assert service["depends_on"]["respeaker-orin11"]["condition"] == "service_started"
    assert service["environment"]["FABLE_AUDIO_BACKEND"].startswith("${FABLE_AUDIO_BACKEND")

    resolver = ProviderRuntimeResolver.from_yaml(REPLAY / "config/fable_provider_runtimes.yaml")
    audio = resolver.resolve(
        node_id="dvpg_gq_orin_11", provider_id="audio_event_classifier"
    )
    assert audio.mode == RuntimeMode.ADOPT_EXISTING
    assert audio.container_name == "fable-multimodal-orin11"
    assert audio.output_adapter == ReplayOutputAdapter.MULTIMODAL_PREDICATE
    transfer = resolver.resolve(
        node_id="dvpg_gq_orin_11", provider_id="object_transfer_reasoner"
    )
    assert transfer.output_adapter == ReplayOutputAdapter.MULTIMODAL_PREDICATE
