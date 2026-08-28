from __future__ import annotations

from dataclasses import replace

import yaml

from evaluation.baselines.policies import HandwrittenStaticPolicy
from evaluation.live_planning import LiveBaselinePlanningPolicy
from evaluation.schemas import BaselineId
from evaluation.baselines.static_registry import (
    StaticPipelineRegistry,
    resolve_static_chain_id,
    resolve_static_provider_id,
)
from evaluation.planning_cases import compile_evaluation_planning_case
from fable.common.examples import BASE_TIME
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_provider_registry,
)


def test_static_registry_loads_trace_placement_and_ignores_audit_metadata(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
pipelines: {}
placement_templates:
  Route convoy:
    allowed_chain_ids: [passes_live_vehicle]
    allowed_node_ids: [camera-a]
trace_placements:
  trace-1:
    experiment_id: experiment-1
    allowed_chain_ids: [passes_live_vehicle]
    allowed_provider_ids: [multi_object_tracker]
    allowed_node_ids: [camera-b]
    allowed_source_ids: [camera-b-source]
    allowed_chain_node_ids: {passes_live_vehicle: [camera-b]}
    placement_sha256: audit-only
    fanout_allowed: false
    adaptation_allowed: false
""",
        encoding="utf-8",
    )

    registry = StaticPipelineRegistry.load(path)
    trace = registry.get_placement("Route convoy", trace_id="trace-1")
    assert trace is not None
    assert trace.allowed_node_ids == ("camera-b",)
    assert trace.allowed_chain_node_ids == {"passes_live_vehicle": ("camera-b",)}
    template = registry.get_placement("Route convoy")
    assert template is not None
    assert template.allowed_node_ids == ("camera-a",)


def test_b1_policy_enforces_trace_chain_node_provider_and_source(tmp_path):
    case = compile_evaluation_planning_case(
        variant="Pass-follow-clear convoy",
        run_id="run",
        trace_id="trace-1",
        request_id="request",
        now=BASE_TIME,
        provider_registry=fake_provider_registry(),
        artifact_catalog=fake_artifact_catalog(),
        deployment=fake_deployment(),
    )
    target = case.whole_event_graph.alternatives[0]
    target_nodes = sorted({step.node_id for step in target.step_placements})
    target_providers = sorted({step.provider_id for step in target.step_placements})
    target_sources = sorted(
        item.source_id
        for item in target.external_inputs
        if item.source_id is not None
    )
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "pipelines": {},
                "trace_placements": {
                    "trace-1": {
                        "allowed_chain_ids": [target.chain_id],
                        "allowed_provider_ids": target_providers,
                        "allowed_node_ids": target_nodes,
                        "allowed_source_ids": target_sources,
                        "allowed_chain_node_ids": {
                            target.chain_id: target_nodes,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    decision = HandwrittenStaticPolicy(
        StaticPipelineRegistry.load(registry_path)
    ).plan(replace(case, placement_id="Pass-follow-clear convoy"))
    selected = {
        item.alternative_id: item for item in case.whole_event_graph.alternatives
    }

    assert decision.selected_alternative_ids
    assert all(
        {
            step.node_id
            for step in selected[alternative_id].step_placements
        }.issubset(set(target_nodes))
        for alternative_id in decision.selected_alternative_ids
    )
    assert all(
        selected[alternative_id].chain_id == target.chain_id
        for alternative_id in decision.selected_alternative_ids
    )


def test_b1_resolves_only_versioned_static_chain_aliases() -> None:
    assert resolve_static_chain_id("recover_vehicle_from_local_segments") == (
        "recover_vehicle_before_audio_event"
    )
    assert resolve_static_chain_id("unknown_recovery_chain") == (
        "unknown_recovery_chain"
    )


def test_b1_resolves_only_versioned_static_provider_aliases() -> None:
    assert resolve_static_provider_id("yolo_vehicle_fast_640") == (
        "yolo_vehicle_balanced_960"
    )
    assert resolve_static_provider_id("audio_event_classifier") == (
        "audio_event_classifier"
    )


def test_live_b1_constrains_sources_and_nodes_before_enumeration(tmp_path) -> None:
    case = compile_evaluation_planning_case(
        variant="Pass-follow-clear convoy",
        run_id="run",
        trace_id="trace-1",
        request_id="request",
        now=BASE_TIME,
        provider_registry=fake_provider_registry(),
        artifact_catalog=fake_artifact_catalog(),
        deployment=fake_deployment(),
    )
    demand = case.frontier_demands[0]
    source = demand.eligible_source_ids[-1]
    node = fake_deployment().sources[source].node_id
    path = tmp_path / "registry.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "pipelines": {},
                "trace_placements": {
                    "trace-1": {
                        "allowed_node_ids": [node],
                        "allowed_source_ids": [source],
                        "allowed_chain_ids": ["passes_live_vehicle"],
                        "allowed_chain_node_ids": {
                            "passes_live_vehicle": [node]
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    policy = LiveBaselinePlanningPolicy(
        BaselineId.B1_HANDWRITTEN_STATIC,
        static_registry_path=str(path),
    )

    constrained = policy.constrain_frontier_demands(
        trace_id="trace-1",
        placement_id="Pass-follow-clear convoy",
        demands=(demand,),
    )[0]

    assert constrained.eligible_source_ids == (source,)
    assert constrained.hard_constraints.allowed_node_ids == (node,)
    assert constrained.sharing_key is not None
    assert constrained.sharing_key != demand.sharing_key
