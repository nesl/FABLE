"""Node heartbeats, replay source progress, and failure-state tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
import shutil
import subprocess
import threading
from typing import Any

from fable.common.enums import NodeAvailability
from fable.common.schemas import NodeCapacity, NodeHeartbeat, SourceHeartbeat
from fable.common.time import EventTimeInterval, ensure_utc, utc_now

from .segment_store import SegmentStore

UTC = timezone.utc


@dataclass(frozen=True)
class NodeTransition:
    node_id: str
    previous: NodeAvailability | None
    current: NodeAvailability
    session_id: str
    reason: str
    occurred_at: datetime


@dataclass
class _ObservedNode:
    heartbeat: NodeHeartbeat
    received_at: datetime
    availability: NodeAvailability
    recovery_confirmations: int = 0


class HeartbeatMonitor:
    """Applies the 1 s / 3-miss / 5-miss / 2-recovery policy."""

    def __init__(
        self,
        *,
        interval: timedelta = timedelta(seconds=1),
        suspect_misses: int = 3,
        unavailable_misses: int = 5,
        recovery_confirmations: int = 2,
    ) -> None:
        if suspect_misses < 1 or unavailable_misses <= suspect_misses:
            raise ValueError("unavailable_misses must be greater than suspect_misses")
        self.interval = interval
        self.suspect_misses = suspect_misses
        self.unavailable_misses = unavailable_misses
        self.required_recovery_confirmations = recovery_confirmations
        self._nodes: dict[str, _ObservedNode] = {}
        self._lock = threading.RLock()

    def record(
        self,
        heartbeat: NodeHeartbeat,
        *,
        received_at: datetime | None = None,
    ) -> tuple[NodeTransition, ...]:
        observed_at = ensure_utc(received_at or utc_now())
        with self._lock:
            prior = self._nodes.get(heartbeat.node_id)
            if prior is None:
                availability = heartbeat.availability
                if availability in (NodeAvailability.UNAVAILABLE, NodeAvailability.RECOVERING):
                    availability = NodeAvailability.RECOVERING
                    confirmations = 1
                else:
                    availability = NodeAvailability.AVAILABLE
                    confirmations = self.required_recovery_confirmations
                self._nodes[heartbeat.node_id] = _ObservedNode(
                    heartbeat=heartbeat,
                    received_at=observed_at,
                    availability=availability,
                    recovery_confirmations=confirmations,
                )
                return (
                    NodeTransition(
                        node_id=heartbeat.node_id,
                        previous=None,
                        current=availability,
                        session_id=heartbeat.session_id,
                        reason="first heartbeat",
                        occurred_at=observed_at,
                    ),
                )

            previous = prior.availability
            session_changed = prior.heartbeat.session_id != heartbeat.session_id
            if session_changed or previous in (
                NodeAvailability.UNAVAILABLE,
                NodeAvailability.RECOVERING,
            ):
                if session_changed:
                    prior.recovery_confirmations = 1
                else:
                    prior.recovery_confirmations += 1
                availability = (
                    NodeAvailability.AVAILABLE
                    if prior.recovery_confirmations >= self.required_recovery_confirmations
                    else NodeAvailability.RECOVERING
                )
            else:
                prior.recovery_confirmations = self.required_recovery_confirmations
                availability = NodeAvailability.AVAILABLE

            prior.heartbeat = heartbeat
            prior.received_at = observed_at
            prior.availability = availability
            if availability == previous:
                return ()
            return (
                NodeTransition(
                    node_id=heartbeat.node_id,
                    previous=previous,
                    current=availability,
                    session_id=heartbeat.session_id,
                    reason=(
                        "new node session confirmed"
                        if availability == NodeAvailability.AVAILABLE and session_changed
                        else "node session recovery"
                    ),
                    occurred_at=observed_at,
                ),
            )

    def tick(self, *, now: datetime | None = None) -> tuple[NodeTransition, ...]:
        observed_now = ensure_utc(now or utc_now())
        transitions: list[NodeTransition] = []
        with self._lock:
            for node_id, observed in sorted(self._nodes.items()):
                elapsed = observed_now - observed.received_at
                misses = int(elapsed.total_seconds() // self.interval.total_seconds())
                previous = observed.availability
                if misses >= self.unavailable_misses:
                    current = NodeAvailability.UNAVAILABLE
                elif misses >= self.suspect_misses:
                    current = NodeAvailability.SUSPECT
                else:
                    continue
                if current == previous:
                    continue
                observed.availability = current
                observed.recovery_confirmations = 0
                transitions.append(
                    NodeTransition(
                        node_id=node_id,
                        previous=previous,
                        current=current,
                        session_id=observed.heartbeat.session_id,
                        reason=f"missed approximately {misses} heartbeats",
                        occurred_at=observed_now,
                    )
                )
        return tuple(transitions)

    def heartbeat(self, node_id: str) -> NodeHeartbeat | None:
        with self._lock:
            observed = self._nodes.get(node_id)
            return None if observed is None else observed.heartbeat

    def availability(self, node_id: str) -> NodeAvailability | None:
        with self._lock:
            observed = self._nodes.get(node_id)
            return None if observed is None else observed.availability

    @property
    def nodes(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._nodes))


class ReplaySourceProgressTracker:
    """Normalizes the existing ``iobt-minimal-ce-replay`` status topics.

    The replay services publish their original event-time start/end plus current
    progress.  This adapter converts those messages, detector frame probes, and
    ordinary timestamped events into ``SourceHeartbeat`` records used by FABLE's
    event-time and retrospective-execution logic.
    """

    def __init__(
        self,
        *,
        node_id: str,
        segment_store: SegmentStore | None = None,
        source_aliases: dict[str, str] | None = None,
    ) -> None:
        self.node_id = node_id
        self.segment_store = segment_store
        self.source_aliases = source_aliases or {}
        self._sources: dict[str, SourceHeartbeat] = {}
        self._lock = threading.RLock()

    def update(self, topic: str, payload: bytes | str | dict[str, Any] | list[Any]) -> SourceHeartbeat | None:
        parsed = _parse_payload(payload)
        if parsed is None:
            return None
        if topic.startswith("/replay/status/") and isinstance(parsed, dict):
            return self._update_replay_status(topic, parsed)
        if "/analytics/yolo/frame" in topic and isinstance(parsed, dict):
            source_id = self._source_id(topic, suffix="camera")
            timestamp = _event_datetime(parsed.get("data_t") or parsed.get("t"))
            sequence = int(parsed.get("input_frames_total") or 0)
            return self._record(source_id, timestamp, sequence, operational=True)
        if topic.endswith("/analytics/yolo/bbox"):
            events = parsed if isinstance(parsed, list) else [parsed]
            event_dicts = [item for item in events if isinstance(item, dict)]
            if not event_dicts:
                return None
            latest = max(_event_datetime(item.get("t")) for item in event_dicts)
            source_id = self._source_id(topic, suffix="camera")
            current = self._sources.get(source_id)
            sequence = (current.latest_sequence + len(event_dicts)) if current else len(event_dicts)
            return self._record(source_id, latest, sequence, operational=True)
        if topic.endswith("/audio_detector/detections") and isinstance(parsed, dict):
            source_id = self._source_id(topic, suffix="audio")
            timestamp = _event_datetime(parsed.get("t"))
            current = self._sources.get(source_id)
            return self._record(
                source_id,
                timestamp,
                (current.latest_sequence + 1) if current else 1,
                operational=True,
            )
        return None

    def _update_replay_status(self, topic: str, payload: dict[str, Any]) -> SourceHeartbeat:
        parts = topic.strip("/").split("/")
        service = parts[2] if len(parts) > 2 else str(payload.get("service", "source"))
        node = parts[3] if len(parts) > 3 else self.node_id
        default_suffix = "camera" if service == "zed" else "audio" if service == "respeaker" else service
        source_id = self.source_aliases.get(f"{service}:{node}", f"{node}:{default_suffix}")
        start = _event_datetime(payload.get("start_time"))
        end = _event_datetime(payload.get("end_time"))
        current_value = float(payload.get("current") or 0.0)
        latest = min(end, start + timedelta(seconds=max(0.0, current_value)))
        current = self._sources.get(source_id)
        sequence = (current.latest_sequence + 1) if current else 1
        operational = payload.get("event") != "error"
        explicit_buffer = EventTimeInterval(start=start, end=end)
        return self._record(
            source_id,
            latest,
            sequence,
            operational=operational,
            raw_buffer_interval=explicit_buffer,
        )

    def _record(
        self,
        source_id: str,
        latest_event_time: datetime,
        sequence: int,
        *,
        operational: bool,
        raw_buffer_interval: EventTimeInterval | None = None,
    ) -> SourceHeartbeat:
        if raw_buffer_interval is None and self.segment_store is not None:
            raw_buffer_interval = self.segment_store.source_buffer_interval(
                source_id, require_existing_file=False
            )
        heartbeat = SourceHeartbeat(
            source_id=source_id,
            latest_sequence=max(0, sequence),
            latest_event_time=latest_event_time,
            raw_buffer_interval=raw_buffer_interval,
            operational_coverage=operational,
        )
        with self._lock:
            prior = self._sources.get(source_id)
            if prior is not None and heartbeat.latest_event_time < prior.latest_event_time:
                # Keep event-time progress monotonic within a replay session.
                heartbeat.latest_event_time = prior.latest_event_time
                heartbeat.latest_sequence = max(prior.latest_sequence, heartbeat.latest_sequence)
            self._sources[source_id] = heartbeat
        return heartbeat

    def _source_id(self, topic: str, *, suffix: str) -> str:
        parts = topic.strip("/").split("/")
        node = parts[0] if parts else self.node_id
        return self.source_aliases.get(f"{suffix}:{node}", f"{node}:{suffix}")

    @property
    def sources(self) -> dict[str, SourceHeartbeat]:
        with self._lock:
            return dict(self._sources)


class CapacitySampler:
    def __init__(self, *, gpu_free_mb_override: int | None = None) -> None:
        self.gpu_free_mb_override = gpu_free_mb_override

    def sample(self) -> NodeCapacity:
        try:
            import psutil

            cpu_count = psutil.cpu_count(logical=True) or 1
            cpu_free = max(0.0, cpu_count * (1.0 - psutil.cpu_percent(interval=None) / 100.0))
            memory_free_mb = int(psutil.virtual_memory().available / (1024 * 1024))
        except Exception:
            cpu_free = float(os.cpu_count() or 1)
            memory_free_mb = 0
        gpu_free = (
            self.gpu_free_mb_override
            if self.gpu_free_mb_override is not None
            else _sample_nvidia_free_mb()
        )
        return NodeCapacity(
            cpu_free_cores=round(cpu_free, 3),
            memory_free_mb=max(0, memory_free_mb),
            gpu_free_mb=max(0, gpu_free),
        )


def build_node_heartbeat(
    *,
    node_id: str,
    session_id: str,
    sequence: int,
    sources: dict[str, SourceHeartbeat],
    active_provider_instance_ids: tuple[str, ...],
    active_demand_ids,
    capacity: NodeCapacity,
    availability: NodeAvailability = NodeAvailability.AVAILABLE,
    sent_at: datetime | None = None,
) -> NodeHeartbeat:
    """Create the typed heartbeat that NodeAgent publishes once per interval."""
    return NodeHeartbeat(
        node_id=node_id,
        session_id=session_id,
        sequence=sequence,
        sent_at=sent_at or utc_now(),
        availability=availability,
        sources=sources,
        active_provider_instance_ids=active_provider_instance_ids,
        active_demand_ids=tuple(active_demand_ids),
        capacity=capacity,
    )


def _sample_nvidia_free_mb() -> int:
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        return sum(values)
    except Exception:
        return 0


def _parse_payload(payload: bytes | str | dict[str, Any] | list[Any]) -> Any | None:
    if isinstance(payload, (dict, list)):
        return payload
    try:
        raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return json.loads(raw)
    except Exception:
        return None


def _event_datetime(value: Any) -> datetime:
    if value is None:
        return utc_now()
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, (float, int)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    try:
        return datetime.fromtimestamp(float(text), tz=UTC)
    except ValueError:
        pass
    for candidate in (text, text.replace("/", "-")):
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            continue
    # Existing replay services use yyyy/mm/dd HH:MM:SS.ffffff.
    try:
        return datetime.strptime(text, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)
    except ValueError:
        return utc_now()
