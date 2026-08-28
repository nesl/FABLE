from __future__ import annotations

from statistics import median

from fable.common.base import FableModel

from evaluation.schemas import ArtifactEvent, RetrospectiveAttempt


class ReplayMetrics(FableModel):
    attempts: int = 0
    successful_attempts: int = 0
    success_rate: float = 0.0
    buffer_expirations: int = 0
    median_replay_seconds: float | None = None
    p95_replay_seconds: float | None = None
    raw_bytes_read: int = 0
    transferred_bytes: int = 0
    artifact_reuse_rate: float = 0.0
    compatibility_failures: int = 0


def summarize_replay(
    attempts: tuple[RetrospectiveAttempt, ...],
    artifacts: tuple[ArtifactEvent, ...] = (),
) -> ReplayMetrics:
    successes = sum(
        item.outcome.upper() in {"SUCCESS", "RECOVERED", "MATCHED"}
        for item in attempts
    )
    expirations = sum(
        bool(item.buffer_expiration_reason)
        or item.outcome.upper() in {"EXPIRED", "BUFFER_EXPIRED"}
        for item in attempts
    )
    latencies = sorted(item.processing_seconds for item in attempts)
    creates = sum(
        item.action.upper() in {"CREATE", "WRITE", "RETAIN"} for item in artifacts
    )
    reuse = sum(
        item.action.upper() in {"READ", "CONSUME", "IMPORT"} for item in artifacts
    )
    compatibility = sum(
        item.action.upper() == "COMPATIBILITY_FAILURE" for item in artifacts
    )
    return ReplayMetrics(
        attempts=len(attempts),
        successful_attempts=successes,
        success_rate=successes / len(attempts) if attempts else 0.0,
        buffer_expirations=expirations,
        median_replay_seconds=median(latencies) if latencies else None,
        p95_replay_seconds=_percentile(latencies, 0.95),
        raw_bytes_read=sum(item.raw_bytes_read for item in attempts),
        transferred_bytes=sum(item.transferred_bytes for item in attempts),
        artifact_reuse_rate=reuse / (creates + reuse) if creates + reuse else 0.0,
        compatibility_failures=compatibility,
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return values[int(round((len(values) - 1) * fraction))]
