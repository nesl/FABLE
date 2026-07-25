"""Common evaluation runner and append-only JSONL event store."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter_ns
from typing import Iterable

from fable.common.ids import deterministic_id

from evaluation.baselines.models import BaselineDecision, BaselinePlanningCase
from evaluation.baselines.policies import BaselinePolicy
from evaluation.schemas import EvaluationMode, PlanDecision, PredicateObservation
from evaluation.schemas.records import EvaluationRecord


class JsonlEventStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, record: EvaluationRecord) -> Path:
        path = self.root / f"{record.record_type}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json(exclude_none=True) + "\n")
        return path

    def read(self, record_type: str) -> tuple[dict[str, object], ...]:
        path = self.root / f"{record_type}.jsonl"
        if not path.exists():
            return ()
        return tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )


class EvaluationRunner:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        mode: EvaluationMode,
    ) -> None:
        self.mode = mode
        self.store = JsonlEventStore(output_dir)

    def run_planning_case(
        self,
        policy: BaselinePolicy,
        case: BaselinePlanningCase,
    ) -> BaselineDecision:
        started = perf_counter_ns()
        decision = policy.plan(case)
        elapsed_ms = (perf_counter_ns() - started) / 1_000_000
        if decision.planning_latency_ms == 0:
            decision = decision.model_copy(update={"planning_latency_ms": elapsed_ms})
        plan_record = PlanDecision(
            run_id=case.run_id,
            baseline_id=decision.baseline_id,
            trace_id=case.trace_id,
            request_id=case.request_id,
            event_time=case.now,
            monotonic_timestamp_ns=perf_counter_ns(),
            decision_id=deterministic_id(
                "evaluation_plan_decision",
                {
                    "run_id": case.run_id,
                    "baseline": decision.baseline_id,
                    "request": case.request_id,
                    "checkpoint": decision.checkpoint_id,
                    "resource_epoch": case.resource_epoch,
                    "semantic_epoch": case.semantic_epoch,
                },
                length=32,
            ),
            checkpoint_id=str(decision.checkpoint_id),
            planning_scope=decision.planning_scope,
            selected_alternative_ids=decision.selected_alternative_ids,
            selected_chain_ids=decision.selected_chain_ids,
            selected_node_ids=decision.selected_node_ids,
            selected_source_ids=decision.selected_source_ids,
            activated_provider_keys=decision.activated_provider_keys,
            continuation_types=decision.continuation_types,
            predicted_completion_ms=decision.predicted_completion_ms,
            predicted_transfer_bytes=decision.predicted_transfer_bytes,
            planning_latency_ms=decision.planning_latency_ms,
            labels_generated=decision.labels_generated,
            labels_pruned=decision.labels_pruned,
            labels_retained=decision.labels_retained,
            oracle_gap_ms=decision.oracle_gap_ms,
            reason=decision.reason,
            frozen=decision.frozen,
            resource_epoch=decision.resource_epoch,
            semantic_epoch=decision.semantic_epoch,
            metadata={
                "evaluation_mode": self.mode.value,
                "excluded_mobile_or_unavailable_sources": list(
                    decision.excluded_mobile_or_unavailable_sources
                ),
            },
        )
        self.store.append(plan_record)
        return decision

    def replay_common_perception(
        self,
        observations: Iterable[PredicateObservation],
    ) -> tuple[PredicateObservation, ...]:
        ordered = tuple(
            sorted(
                observations,
                key=lambda item: (item.event_time, item.source_sequence or -1, item.observation_id),
            )
        )
        for observation in ordered:
            self.store.append(observation)
        return ordered
