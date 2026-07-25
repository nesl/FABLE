from datetime import timedelta
from pathlib import Path

import yaml

from evaluation.runtime_logging import EvaluationMessageNormalizer, RuntimeLoggingContext
from evaluation.schemas import BaselineId, PredicateObservation
from fable.common.enums import TruthValue
from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.common.schemas import BindingDelta, PredicateResult, ResultProvenance
from fable.distributed.models import ReliablePredicateResult
from fable.planning.testing import fake_follow_demand

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_normalizer_converts_reliable_predicate_result() -> None:
    demand = fake_follow_demand()
    result = PredicateResult(
        demand_id=demand.demand_id,
        request_id=demand.request_id,
        graph_hash=demand.graph_hash,
        hypothesis_id=demand.hypothesis_id,
        expected_hypothesis_version=demand.hypothesis_version,
        frontier_id=demand.frontier_id,
        checkpoint_id=demand.checkpoint_id,
        graph_node_id=demand.graph_node_id,
        semantic_predicate=demand.semantic_predicate,
        occurrence_id="occ-eval",
        truth=TruthValue.TRUE,
        confidence=0.9,
        event_time_interval=demand.event_time_interval,
        binding_delta=BindingDelta(introduced={"follower": "vehicle_2"}),
        provenance=ResultProvenance(
            provider_id="follows_local_geometry",
            provider_contract_version=1,
            node_id="sensor_a",
            source_ids=("camera_mobile",),
            source_sequence_ranges={"camera_mobile": (10, 20)},
        ),
        processing_started_at=BASE_TIME,
        processing_completed_at=BASE_TIME + timedelta(milliseconds=10),
    )
    envelope = ReliablePredicateResult(
        node_id="sensor_a",
        session_id="session-a",
        provider_instance_id="provider-instance-a",
        attempt_id=uuid7(),
        result=result,
        emitted_at=BASE_TIME + timedelta(milliseconds=11),
    )
    normalizer = EvaluationMessageNormalizer(
        RuntimeLoggingContext(
            run_id="run",
            baseline_id=BaselineId.FABLE,
            trace_id="trace",
            default_request_id=demand.request_id,
        )
    )
    record = normalizer.normalize("fable/v1/result/request/FOLLOWS", envelope.model_dump_json())
    assert isinstance(record, PredicateObservation)
    assert record.predicate_id == "FOLLOWS"
    assert record.bindings["follower"] == "vehicle_2"
    assert record.source_sequence == 20


def test_evaluation_compose_overlay_is_additive_and_valid_yaml() -> None:
    payload = yaml.safe_load(
        (ROOT / "iobt-minimal-ce-replay/compose.fable.evaluation.yaml").read_text()
    )
    service = payload["services"]["fable-evaluation-logger"]
    assert service["depends_on"]["mqtt"]["condition"] == "service_started"
    assert "evaluation-runs" in service["volumes"][0]
