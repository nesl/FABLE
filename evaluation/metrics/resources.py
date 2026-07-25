from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from fable.common.base import FableModel

from evaluation.schemas import ArtifactEvent, ProviderLifecycleEvent, ResourceSample


class ResourceMetrics(FableModel):
    provider_active_seconds: float = 0.0
    provider_starts: int = 0
    cpu_seconds: float = 0.0
    gpu_energy_joules: float = 0.0
    network_tx_bytes: int = 0
    network_rx_bytes: int = 0
    artifact_bytes_written: int = 0
    raw_media_bytes: int = 0


def summarize_resources(
    lifecycle: tuple[ProviderLifecycleEvent, ...],
    samples: tuple[ResourceSample, ...],
    artifacts: tuple[ArtifactEvent, ...],
) -> ResourceMetrics:
    starts: dict[str, datetime] = {}
    active = 0.0
    provider_starts = 0
    for item in sorted(lifecycle, key=lambda value: value.wall_timestamp):
        event = item.lifecycle_event.upper()
        if event in {"READY", "ACTIVE", "STARTED"} and item.provider_instance_id not in starts:
            starts[item.provider_instance_id] = item.wall_timestamp
            provider_starts += 1
        if event in {"STOPPED", "FAILED", "DRAINED"}:
            start = starts.pop(item.provider_instance_id, None)
            if start is not None:
                active += max(0.0, (item.wall_timestamp - start).total_seconds())
    return ResourceMetrics(
        provider_active_seconds=active,
        provider_starts=provider_starts,
        cpu_seconds=sum(item.cpu_time_seconds for item in samples),
        gpu_energy_joules=sum(item.gpu_energy_joules or 0.0 for item in samples),
        network_tx_bytes=sum(item.network_tx_bytes for item in samples),
        network_rx_bytes=sum(item.network_rx_bytes for item in samples),
        artifact_bytes_written=sum(item.bytes for item in artifacts if item.action.upper() in {"WRITE", "CREATE", "RETAIN"}),
        raw_media_bytes=sum(
            item.bytes
            for item in artifacts
            if item.action.upper() in {"WRITE", "CREATE", "RETAIN", "TRANSFER"}
            and item.artifact_type.startswith(("raw_", "audio_segment", "video_segment"))
        ),
    )
