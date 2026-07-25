"""Deterministic fake-result builders for Phase-1 tests and demos."""

from __future__ import annotations

from datetime import timedelta

from fable.common.enums import TruthValue
from fable.common.ids import occurrence_anchor_id, uuid7
from fable.common.schemas import BindingDelta, PredicateResult, ResultProvenance

from .models import ScriptedResultSpec, SeedPredicateResult
from .runtime import SemanticRuntime


def seed_result_from_spec(runtime: SemanticRuntime, spec: ScriptedResultSpec) -> SeedPredicateResult:
    node = runtime.graph.nodes_by_key[spec.node_key]
    occurrence_id = spec.occurrence_id or occurrence_anchor_id(
        spec.source_id,
        node.predicate.predicate_id if node.predicate else spec.node_key,
        spec.event_time_interval.start,
        {**spec.introduced, **spec.validated},
    )
    return SeedPredicateResult(
        occurrence_id=occurrence_id,
        request_id=runtime.config.request_id,
        graph_hash=runtime.graph.graph.graph_hash,
        graph_node_id=node.node_id,
        semantic_predicate=node.predicate,
        truth=spec.truth,
        confidence=spec.confidence,
        event_time_interval=spec.event_time_interval,
        binding_delta=BindingDelta(
            introduced=spec.introduced,
            validated=spec.validated,
        ),
        provenance=ResultProvenance(
            provider_id=spec.provider_id,
            provider_contract_version=1,
            node_id="scripted_node",
            source_ids=(spec.source_id,),
        ),
        observed_at=spec.event_time_interval.end,
    )


def predicate_result_from_spec(
    runtime: SemanticRuntime,
    hypothesis_id,
    spec: ScriptedResultSpec,
) -> PredicateResult:
    hypothesis = runtime.get_hypothesis(hypothesis_id)
    frontier = runtime.get_frontier(hypothesis_id)
    if frontier is None:
        raise ValueError("hypothesis has no active frontier")
    node = runtime.graph.nodes_by_key[spec.node_key]
    checkpoint = frontier.checkpoint_for_node(node.node_id)
    occurrence_id = spec.occurrence_id or occurrence_anchor_id(
        spec.source_id,
        node.predicate.predicate_id if node.predicate else spec.node_key,
        spec.event_time_interval.start,
        {**spec.introduced, **spec.validated},
    )
    return PredicateResult(
        result_id=uuid7(),
        occurrence_id=occurrence_id,
        demand_id=uuid7(),
        request_id=runtime.config.request_id,
        graph_hash=runtime.graph.graph.graph_hash,
        hypothesis_id=hypothesis_id,
        expected_hypothesis_version=hypothesis.version,
        frontier_id=frontier.snapshot.frontier_id,
        checkpoint_id=checkpoint.checkpoint_id,
        graph_node_id=node.node_id,
        semantic_predicate=node.predicate,
        truth=spec.truth,
        confidence=spec.confidence,
        event_time_interval=spec.event_time_interval,
        binding_delta=BindingDelta(
            introduced=spec.introduced,
            validated=spec.validated,
        ),
        provenance=ResultProvenance(
            provider_id=spec.provider_id,
            provider_contract_version=1,
            node_id="scripted_node",
            source_ids=(spec.source_id,),
        ),
        processing_started_at=spec.event_time_interval.end,
        processing_completed_at=spec.event_time_interval.end + timedelta(milliseconds=1),
    )
