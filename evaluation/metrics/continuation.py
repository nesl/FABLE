from __future__ import annotations

from fable.common.base import FableModel
from evaluation.schemas import ArtifactEvent, HypothesisTransition


class ContinuationMetrics(FableModel):
    continuation_artifacts_created: int = 0
    continuation_artifacts_consumed: int = 0
    continuation_success_rate: float = 0.0
    state_transfer_bytes: int = 0
    retrospective_recoveries: int = 0
    buffer_expirations: int = 0
    compatibility_failures: int = 0


def summarize_continuation(
    artifacts: tuple[ArtifactEvent, ...],
    transitions: tuple[HypothesisTransition, ...] = (),
) -> ContinuationMetrics:
    created = sum(item.action.upper() in {"CREATE", "RETAIN", "WRITE"} for item in artifacts)
    consumed = sum(item.action.upper() in {"CONSUME", "READ", "IMPORT"} for item in artifacts)
    transfers = sum(item.bytes for item in artifacts if item.action.upper() == "TRANSFER")
    retrospective = sum(
        item.transition_kind.upper() == "RETROSPECTIVE_RECOVERY" for item in transitions
    )
    expirations = sum(item.action.upper() == "BUFFER_EXPIRED" for item in artifacts)
    failures = sum(item.action.upper() == "COMPATIBILITY_FAILURE" for item in artifacts)
    return ContinuationMetrics(
        continuation_artifacts_created=created,
        continuation_artifacts_consumed=consumed,
        continuation_success_rate=(consumed / created if created else 0.0),
        state_transfer_bytes=transfers,
        retrospective_recoveries=retrospective,
        buffer_expirations=expirations,
        compatibility_failures=failures,
    )
