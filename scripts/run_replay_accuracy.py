#!/usr/bin/env python3
"""Run one bounded positive-case replay from seed watch to terminal CE state."""

from __future__ import annotations

import argparse
import math
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable
from uuid import uuid4

import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog
from evaluation.baselines.static_registry import (
    StaticPipelineRegistry,
    static_pipeline_registry_path,
)
from evaluation.defaults import DEFAULT_PLAYBACK_MODE, DEFAULT_PLAYBACK_SPEED
from evaluation.condition_trace import (
    ConditionAnchor,
    ConditionAction,
    ConditionTrace,
    MonotonicConditionTraceController,
    classify_disturbance_exposure,
)
from evaluation.e4_identity_judging import IdentityEvidenceCapture
from evaluation.live_requests import (
    LiveComplexEventCancelRequest,
    LiveComplexEventCancelResponse,
    LiveComplexEventProgress,
    LiveComplexEventRequest,
    LiveComplexEventResponse,
)
from evaluation.metrics.event_matching import GroundTruthEvent, evaluate_event_results
from evaluation.networking import canonical_sensor_link_target
from evaluation.planning_cases import VARIANT_TEMPLATES
from evaluation.provenance import build_run_provenance
from evaluation.replay_manifest import ReplayScenario, match_replay_scenario
from evaluation.resource_monitor import RunResourceMonitor
from evaluation.runner import JsonlEventStore
from evaluation.runtime_logging import (
    EvaluationMessageNormalizer,
    MqttEvaluationLogger,
    RuntimeLoggingContext,
)
from evaluation.schemas import BaselineId, ComplexEventResult, DisturbanceEvent
from fable.distributed.codec import decode_model, encode_model
from fable.distributed.models import (
    FaultCommand,
    FaultKind,
    ReliablePredicateResult,
    ReplayReadiness,
    ResourceChange,
    ResourceChangeAck,
    ResourceKind,
)
from fable.distributed.topics import (
    live_request_cancel_response_topic,
    live_request_cancel_topic,
    live_request_progress_topic,
    live_request_response_topic,
    live_request_topic,
    fault_topic,
    heartbeat_filter,
    resource_change_ack_filter,
    resource_change_topic,
)
from fable.common.schemas import NodeHeartbeat
from fable.common.time import EventTimeInterval


def evaluation_request_deadline_offset_ms(max_seconds: float) -> int:
    """Keep the semantic request alive for the driver's bounded run.

    ``LiveComplexEventRequest`` has a production-oriented five-minute default,
    but evaluation cells may deliberately replay a longer event interval or
    process real-time media below 1.0 effective throughput.  The driver already
    owns a strict wall-clock bound, so propagate that same bound instead of
    allowing an unrelated schema default to expire later semantic frontiers.
    """

    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be a positive finite value")
    return max(1, math.ceil(max_seconds * 1000.0))


def record_condition_notification(
    result_row: dict[str, object],
    notify: Callable[[], object],
) -> None:
    """Record planner notification outcome without interrupting a condition trace."""

    try:
        acknowledgement = notify()
    except Exception as exc:
        result_row["notification_validated"] = False
        result_row["notification_error"] = str(exc)
    else:
        result_row["notification_validated"] = True
        if isinstance(acknowledgement, ResourceChangeAck):
            result_row["notification_ack"] = acknowledgement.model_dump(mode="json")


def canonical_resource_kind(kind: str) -> str:
    """Map evaluator-internal condition labels onto the wire protocol enum."""

    return "NETWORK" if kind == "NETWORK_PROFILE" else kind


def decode_replay_readiness(topic: str, payload: bytes) -> ReplayReadiness:
    """Decode the typed protocol, with one explicit legacy boundary adapter."""

    document = json.loads(payload.decode("utf-8"))
    if document.get("schema_version") == ReplayReadiness.SCHEMA_VERSION:
        return ReplayReadiness.model_validate(document)
    parts = topic.strip("/").split("/")
    service = str(document.get("service") or (parts[2] if len(parts) > 2 else ""))
    node_id = str(
        document.get("source_id")
        or document.get("node_id")
        or document.get("node")
        or (parts[1] if len(parts) > 2 else "")
    )
    timestamp = float(document.get("t") or 0)
    observed_at = datetime.fromtimestamp(timestamp, tz=UTC)
    reserved = {
        "kind", "node", "node_id", "source_id", "service", "ready",
        "reason", "state", "pid", "t", "replay_id", "scenario",
    }
    return ReplayReadiness(
        message_id=uuid4(),
        node_id=node_id,
        service_id=service,
        process_instance_id=f"legacy:{document.get('pid') or 'unknown'}",
        generation=max(0, int(timestamp * 1000)),
        ready=bool(document.get("ready")),
        reason=str(document.get("reason") or ""),
        state=str(document.get("state") or ""),
        replay_id=(str(document["replay_id"]) if document.get("replay_id") else None),
        scenario=(str(document["scenario"]) if document.get("scenario") else None),
        observed_at=observed_at,
        metadata={key: value for key, value in document.items() if key not in reserved},
    )


def condition_offset(
    timestamp: float,
    *,
    anchor: float | None,
    trace_started: float | None,
) -> float | None:
    """Return a condition-relative offset when execution reached an anchor.

    Fail-safe restoration also runs after setup/readiness failures.  In that
    path neither condition anchor exists, so cleanup telemetry must report an
    unknown offset rather than attempting arithmetic with ``None``.
    """

    effective_anchor = anchor if anchor is not None else trace_started
    if effective_anchor is None:
        return None
    return round(timestamp - effective_anchor, 6)


def planner_network_condition(condition: str, target: str) -> str:
    """Retain the physical profile identity across apply and restoration."""

    if target == "physical_link:rpi_to_jetson":
        return "P1_JETSON_PATH_DEGRADED"
    return condition


@dataclass
class RunState:
    mqtt_connected: threading.Event = field(default_factory=threading.Event)
    watching: threading.Event = field(default_factory=threading.Event)
    admitted: threading.Event = field(default_factory=threading.Event)
    terminal: threading.Event = field(default_factory=threading.Event)
    readiness: set[str] = field(default_factory=set)
    readiness_by_node: dict[str, set[str]] = field(default_factory=dict)
    readiness_documents: dict[str, dict[str, ReplayReadiness]] = field(
        default_factory=dict
    )
    provider_heartbeats: dict[str, NodeHeartbeat] = field(default_factory=dict)
    response: LiveComplexEventResponse | None = None
    seed_responses: list[LiveComplexEventResponse] = field(default_factory=list)
    progress: list[LiveComplexEventProgress] = field(default_factory=list)
    progress_status_counts: dict[str, int] = field(default_factory=dict)
    yolo_messages: int = 0
    yolo_class_counts: dict[str, int] = field(default_factory=dict)
    context_track_messages: int = 0
    context_track_class_counts: dict[str, int] = field(default_factory=dict)
    vehicle_predicate_messages: int = 0
    vehicle_predicate_counts: dict[str, int] = field(default_factory=dict)
    vehicle_predicate_source_counts: dict[str, int] = field(default_factory=dict)
    vehicle_predicate_samples: list[dict[str, object]] = field(default_factory=list)
    audio_event_messages: int = 0
    audio_event_counts: dict[str, int] = field(default_factory=dict)
    audio_event_time_ranges: dict[str, tuple[str, str]] = field(default_factory=dict)
    audio_events_by_node: dict[str, dict[str, int]] = field(default_factory=dict)
    audio_recording_samples: list[dict[str, object]] = field(default_factory=list)
    recovered_person_samples: list[dict[str, object]] = field(default_factory=list)
    recovered_vehicle_samples: list[dict[str, object]] = field(default_factory=list)
    interaction_predicate_messages: int = 0
    interaction_predicate_counts: dict[str, int] = field(default_factory=dict)
    interaction_predicate_samples: list[dict[str, object]] = field(default_factory=list)
    identity_association_messages: int = 0
    identity_associations_by_kind: dict[str, int] = field(default_factory=dict)
    identity_association_samples: list[dict[str, object]] = field(default_factory=list)
    retrospective_replay_statuses: list[dict[str, object]] = field(default_factory=list)
    first_retrospective_started_monotonic: float | None = None
    detector_totals_latest: dict[str, dict[str, int | float]] = field(default_factory=dict)
    detector_totals_baseline: dict[str, dict[str, int | float]] = field(default_factory=dict)
    detector_sampling_diagnostics: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    cancellation: LiveComplexEventCancelResponse | None = None
    cancellation_received: threading.Event = field(default_factory=threading.Event)
    error: str = ""
    watch_registered_at: float | None = None
    admitted_at: float | None = None
    terminal_at: float | None = None
    observed_event_time_start: datetime | None = None
    observed_event_time_end: datetime | None = None


def replay_completion_requirements(
    ready_nodes: set[str],
    readiness_by_node: dict[str, set[str]],
) -> set[tuple[str, str]]:
    """Return replay status keys required before the global close boundary."""

    required: set[tuple[str, str]] = set()
    for node_id in ready_nodes:
        services = readiness_by_node.get(node_id, set())
        if "mobile" in services:
            required.add((node_id, "mobile"))
            continue
        if "zed" in services:
            required.add((node_id, "zed"))
        if "respeaker" in services:
            required.add((node_id, "respeaker"))
    return required


def replay_node_readiness_requirements(
    required_services: set[str],
    *,
    node_id: str,
    selected_yolo_nodes: set[str],
) -> set[str]:
    """Separate raw-producer readiness from plan-selected analytics.

    Every replay node must have its raw producer ready. YOLO is lease and
    placement controlled, so only a replay node selected by an emitted
    provider command may be required to advertise YOLO readiness.
    """

    mobile_node = node_id.startswith("mobile_archive_")
    required = set(required_services)
    require_yolo = "yolo" in required and node_id in selected_yolo_nodes
    required.discard("yolo")
    if mobile_node and "zed" in required:
        required.remove("zed")
        required.add("mobile")
    if require_yolo:
        required.add("yolo")
    return required


def _scenario_exists(scenario: str) -> bool:
    path = ROOT / "iobt-minimal-ce-replay/generated/scenario_catalog.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    return any(row.get("scenario_id") == scenario for row in document["scenarios"])


def _scenario_start_datetime(scenario: str) -> datetime:
    path = ROOT / "iobt-minimal-ce-replay/generated/scenario_catalog.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    row = next(
        (item for item in document["scenarios"] if item.get("scenario_id") == scenario),
        None,
    )
    if row is None:
        raise ValueError(f"scenario is not in generated scenario catalog: {scenario}")
    return ReplayScenario.from_catalog_row(row).start_datetime


def _scenario_duration_seconds(scenario: str) -> float:
    path = ROOT / "iobt-minimal-ce-replay/generated/scenario_catalog.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    row = next(
        (item for item in document["scenarios"] if item.get("scenario_id") == scenario),
        None,
    )
    if row is None:
        raise ValueError(f"scenario is not in generated scenario catalog: {scenario}")
    return float(row["duration_seconds"])


def _seed_event_time_interval(
    *,
    scenario_start: datetime,
    replay_start_seconds: float,
    replay_end_seconds: float,
    labeled_event_end: datetime | None,
) -> EventTimeInterval:
    """Return the event-time coverage required by the semantic seed watch.

    ``replay_end_seconds`` may include post-event deadline/grace time so the
    runtime can receive late results.  That processing allowance is not raw
    sensor evidence and must not make the planner require media beyond a
    labeled event's end.  Unlabeled/ad-hoc runs retain their explicit replay
    window as the only available semantic bound.
    """

    replay_end = scenario_start + timedelta(seconds=replay_end_seconds)
    return EventTimeInterval(
        start=scenario_start + timedelta(seconds=replay_start_seconds),
        end=min(replay_end, labeled_event_end) if labeled_event_end else replay_end,
    )


def _recording_time(
    value: datetime,
    *,
    replay_started_at: float,
    recording_started_at: datetime,
) -> datetime:
    """Map a rebased replay timestamp back into the recording time domain."""

    replay_anchor = datetime.fromtimestamp(replay_started_at, tz=UTC)
    return recording_started_at + (value - replay_anchor)


def _event_recording_time(
    value: datetime,
    *,
    replay_started_at: float,
    recording_started_at: datetime,
) -> datetime:
    """Preserve provider event time when it is already recording-scoped.

    Live terminal detections now carry the original provider event clock. The
    older driver assumed every timestamp had first been rebased onto wall time
    and consequently shifted valid April 2026 events into December 2025.
    Retain timestamps already near the recording; keep the legacy conversion
    for wall-clock/rebased terminal payloads.
    """

    if abs((value - recording_started_at).total_seconds()) <= 24 * 60 * 60:
        return value
    return _recording_time(
        value,
        replay_started_at=replay_started_at,
        recording_started_at=recording_started_at,
    )


def _normalize_relevant_nodes(nodes: tuple[str, ...]) -> set[str]:
    """Translate label node notation (for example ``14`` or ``11-16``) to replay IDs."""
    normalized: set[str] = set()
    for raw_node in nodes:
        node = raw_node.strip().lower()
        if not node:
            continue
        if node.startswith("orin"):
            normalized.add(node)
            continue
        if "-" in node:
            start_text, end_text = node.split("-", 1)
            if start_text.isdigit() and end_text.isdigit():
                start, end = int(start_text), int(end_text)
                if start <= end:
                    normalized.update(
                        f"orin{number}" for number in range(start, end + 1)
                    )
                    continue
        if node.isdigit():
            normalized.add(f"orin{int(node)}")
            continue
        raise ValueError(f"unsupported relevant-node label: {raw_node!r}")
    return normalized


def _resolve_experiment(
    experiment_id: str,
    *,
    replay_nodes: tuple[str, ...] = ("orin11",),
) -> tuple[object, str]:
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv",
        transition_model_path=(
            ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json"
        ),
    )
    try:
        experiment = catalog.by_id[experiment_id]
    except KeyError as exc:
        raise ValueError(f"unknown labeled experiment: {experiment_id}") from exc
    relevant_nodes = _normalize_relevant_nodes(experiment.relevant_nodes)
    # Mobile archive IDs are supplemental replay sources and intentionally do
    # not appear in the fixed-node scenario catalog.  Filtering that catalog by
    # a mobile-only readiness result erases an otherwise valid scenario match.
    catalog_replay_nodes = tuple(
        node for node in replay_nodes if not node.startswith("mobile_archive_")
    )
    deployed_nodes = set(catalog_replay_nodes)
    if relevant_nodes and deployed_nodes and not relevant_nodes & deployed_nodes:
        raise ValueError(
            f"labeled experiment {experiment_id} is relevant to "
            f"{', '.join(sorted(relevant_nodes))}, but deployed replay nodes are "
            f"{', '.join(sorted(deployed_nodes))}"
        )
    scenario_document = json.loads(
        (ROOT / "iobt-minimal-ce-replay/generated/scenario_catalog.json").read_text(
            encoding="utf-8"
        )
    )
    scenarios = tuple(
        ReplayScenario.from_catalog_row(row)
        for row in scenario_document["scenarios"]
        if row.get("valid", True)
        and (
            not catalog_replay_nodes
            or set(catalog_replay_nodes) & set(row.get("nodes", ()))
        )
        and (not relevant_nodes or relevant_nodes & set(row.get("nodes", ())))
    )
    replay = match_replay_scenario(experiment, scenarios)
    if replay.replay_scenario_id is None:
        raise ValueError(
            f"no replay scenario matches labeled experiment {experiment_id}"
        )
    if experiment.ce_variant not in VARIANT_TEMPLATES:
        raise ValueError(f"unsupported complex-event variant: {experiment.ce_variant}")
    return experiment, replay.replay_scenario_id


def _available_seed_sources(
    scenario_id: str,
    seed_node_key: str,
    replay_nodes: tuple[str, ...],
) -> tuple[str, ...]:
    """Return only scenario-backed sources for the authored seed modality.

    Node readiness proves that a replay service is healthy; it does not prove
    that the selected scenario contains that modality on the node.  In
    particular, a ZED-only node can have a healthy (idle) audio container.
    """

    document = json.loads(
        (ROOT / "iobt-minimal-ce-replay/generated/scenario_catalog.json").read_text(
            encoding="utf-8"
        )
    )
    row = next(
        (
            item
            for item in document["scenarios"]
            if item.get("scenario_id") == scenario_id
        ),
        None,
    )
    if row is None:
        return ()
    seed_key = seed_node_key.upper()
    if "AUDIO" in seed_key or "GUNSHOT" in seed_key or "ALARM" in seed_key:
        available = set(row.get("respeaker_nodes", ()))
        suffix = "microphone"
    elif "GPS" in seed_key or "LOCATION" in seed_key:
        available = set(row.get("gps_objects", ()))
        suffix = "gps"
    else:
        available = set(row.get("zed_nodes", ()))
        suffix = "camera"
    selected = []
    for node in replay_nodes:
        if node.startswith("mobile_archive_"):
            # The mobile resolver admits only recordings with the requested
            # synchronized media bundle; its logical source is multimodal.
            selected.append(f"{node}_{suffix}")
        elif node in available:
            selected.append(f"{node}_{suffix}")
    return tuple(selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id")
    parser.add_argument("--scenario")
    parser.add_argument("--variant", choices=tuple(VARIANT_TEMPLATES))
    parser.add_argument(
        "--baseline",
        choices=tuple(
            item.value
            for item in (
                BaselineId.B0_PRODUCE_ALL,
                BaselineId.B1_STATIC_WHOLE_EVENT,
                BaselineId.B2_FRONTIER_FIXED_REALIZATION,
                BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
                BaselineId.B4_GREEDY_FRONTIER,
                BaselineId.FABLE,
            )
        ),
        default=BaselineId.FABLE.value,
    )
    parser.add_argument(
        "--model-id",
        default="unspecified",
        help="declared runtime detector/model configuration for provenance",
    )
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--orchestrator-id", default="orchestrator")
    parser.add_argument(
        "--replay-id",
        help=(
            "explicit replay correlation ID; concurrent validation drivers "
            "must share this value"
        ),
    )
    parser.add_argument(
        "--shared-replay-role",
        choices=("owner", "follower"),
        default="owner",
        help=(
            "Owner configures and stops the replay; a follower only registers "
            "an independent live request against the owner's replay ID."
        ),
    )
    parser.add_argument(
        "--shared-replay-owner-start-delay-seconds",
        type=float,
        default=0.0,
        help="Bounded delay after owner watch registration so followers can register.",
    )
    parser.add_argument(
        "--shared-replay-owner-stop-grace-seconds",
        type=float,
        default=0.0,
        help="Bounded grace after owner completion before it stops shared replay.",
    )
    parser.add_argument(
        "--shared-replay-joint-admission-barrier",
        action="store_true",
        help=(
            "After all concurrent watches register but before replay starts, "
            "issue a typed nominal resource epoch so FABLE plans the active "
            "requests as one capacity-constrained batch."
        ),
    )
    parser.add_argument(
        "--resource-ack-timeout-seconds",
        type=float,
        default=60.0,
        help=(
            "Bounded wait for the orchestrator to finish resource-epoch "
            "replanning and acknowledge the transition."
        ),
    )
    parser.add_argument(
        "--required-ready-services",
        default="zed",
        help=(
            "comma-separated raw replay services required on every replay node; "
            "lease-controlled analytics providers such as yolo must not be "
            "listed because the selected plan may place them on only a subset"
        ),
    )
    parser.add_argument(
        "--replay-nodes",
        default="orin11",
        help="comma-separated replay node folders actually deployed",
    )
    parser.add_argument("--ready-seconds", type=float, default=30)
    parser.add_argument(
        "--vehicle-recovery-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Fail early when raw retrospective replay has started but no "
            "VEHICLE_PRESENT_BEFORE result arrives within this wall time; "
            "zero disables the watchdog."
        ),
    )
    parser.add_argument(
        "--replay-drain-seconds",
        type=float,
        default=5.0,
        help="Quiet period after an explicit replay stop before returning.",
    )
    parser.add_argument("--replay-start", type=float, default=0.0)
    parser.add_argument("--replay-end", type=float, default=-1.0)
    parser.add_argument(
        "--playback-mode",
        choices=("max", "realtime", "scaled"),
        default=DEFAULT_PLAYBACK_MODE,
    )
    parser.add_argument(
        "--playback-speed", type=float, default=DEFAULT_PLAYBACK_SPEED
    )
    parser.add_argument("--deadline-seconds", type=float, default=30.0)
    parser.add_argument(
        "--network-disturbance",
        choices=("W1", "W2", "L1"),
        help="Apply one allowlisted NetWaggle condition after replay sync.",
    )
    parser.add_argument("--disturbance-delay-seconds", type=float, default=0.1)
    parser.add_argument("--disturbance-duration-seconds", type=float, default=20.0)
    parser.add_argument(
        "--condition-trace",
        type=Path,
        help="Versioned exogenous RQ3a condition trace (monotonic wall-clock schedule).",
    )
    parser.add_argument(
        "--physical-condition-identity-file",
        type=Path,
        help="SSH identity used only for allowlisted physical condition targets.",
    )
    parser.add_argument(
        "--physical-compute-planner-node-id",
        help=(
            "Planner node representing the physical Jetson. Required when a "
            "physical compute condition is used with a logical replay facade."
        ),
    )
    parser.add_argument(
        "--physical-network-planner-node-id",
        help=(
            "Planner sensor node whose uplink represents the physical "
            "Pi-to-Jetson path."
        ),
    )
    parser.add_argument(
        "--allow-raw-to-trusted-site-edge",
        action="store_true",
        help=(
            "Permit raw-evidence placement at the trusted site edge. This is "
            "an explicit planning policy and is intentionally independent of "
            "whether a condition trace is present."
        ),
    )
    parser.add_argument(
        "--ce-start-offset-seconds",
        type=float,
        default=0.0,
        help="Start replay this many seconds after the independent condition trace.",
    )
    parser.add_argument("--minimum-temporal-iou", type=float, default=0.1)
    parser.add_argument(
        "--event-match-tolerance-seconds",
        type=float,
        default=30.0,
        help=(
            "Allow a detected CE interval to miss either ground-truth boundary "
            "by at most this many seconds; raw timestamps remain unchanged."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--identity-judge-evidence-dir",
        type=Path,
        help="Capture boxed visual evidence for offline, blinded E4 judging.",
    )
    parser.add_argument("--maximum-identity-judge-predictions", type=int, default=50)
    parser.add_argument(
        "--identity-judge-policy-id",
        default="",
        help="Policy label recorded in E4 evidence independently of orchestration baseline.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write --output without echoing the potentially large JSON document",
    )
    parser.add_argument(
        "--common-record-dir",
        type=Path,
        help=(
            "JSONL directory for common live records; defaults to "
            "<output-stem>.records when --output is set"
        ),
    )
    parser.add_argument(
        "--resource-sample-interval-seconds",
        type=float,
        default=1.0,
        help="Host/container resource sampling interval; set to 0 to disable.",
    )
    args = parser.parse_args()
    if args.shared_replay_owner_start_delay_seconds < 0:
        parser.error("--shared-replay-owner-start-delay-seconds must be nonnegative")
    if args.shared_replay_owner_stop_grace_seconds < 0:
        parser.error("--shared-replay-owner-stop-grace-seconds must be nonnegative")
    if not 1 <= args.resource_ack_timeout_seconds <= 120:
        parser.error("--resource-ack-timeout-seconds must be between 1 and 120")
    if args.max_seconds <= 0 or args.max_seconds > 900:
        parser.error("--max-seconds must be greater than zero and at most 900")
    if args.ready_seconds < 0 or args.ready_seconds > args.max_seconds:
        parser.error("--ready-seconds must be between zero and --max-seconds")
    if not 0 <= args.replay_drain_seconds <= 30:
        parser.error("--replay-drain-seconds must be between zero and 30")
    if args.deadline_seconds < 0:
        parser.error("--deadline-seconds cannot be negative")
    if args.playback_speed <= 0:
        parser.error("--playback-speed must be positive")
    if args.ce_start_offset_seconds < 0:
        parser.error("--ce-start-offset-seconds cannot be negative")
    if args.condition_trace is not None and args.network_disturbance is not None:
        parser.error("--condition-trace cannot be combined with --network-disturbance")
    condition_trace = None
    if args.condition_trace is not None:
        try:
            condition_trace = ConditionTrace.model_validate_json(
                args.condition_trace.read_text(encoding="utf-8")
            )
        except Exception as exc:
            parser.error(f"invalid condition trace: {exc}")
        if args.ce_start_offset_seconds > condition_trace.duration_s:
            parser.error("--ce-start-offset-seconds exceeds condition trace duration")
    if not 0 <= args.minimum_temporal_iou <= 1:
        parser.error("--minimum-temporal-iou must be between zero and one")
    if not 0 <= args.event_match_tolerance_seconds <= 300:
        parser.error("--event-match-tolerance-seconds must be between zero and 300")
    if not 1 <= args.maximum_identity_judge_predictions <= 100:
        parser.error("--maximum-identity-judge-predictions must be between 1 and 100")
    if not 0 <= args.resource_sample_interval_seconds <= 30:
        parser.error("--resource-sample-interval-seconds must be between 0 and 30")
    experiment = None
    replay_nodes = tuple(
        item.strip() for item in args.replay_nodes.split(",") if item.strip()
    )
    if args.experiment_id:
        if args.scenario or args.variant:
            parser.error(
                "--experiment-id cannot be combined with --scenario or --variant"
            )
        try:
            experiment, args.scenario = _resolve_experiment(
                args.experiment_id, replay_nodes=replay_nodes
            )
        except ValueError as exc:
            parser.error(str(exc))
        args.variant = experiment.ce_variant
    elif not args.scenario or not args.variant:
        parser.error("provide --experiment-id, or both --scenario and --variant")
    if not _scenario_exists(args.scenario):
        parser.error(f"scenario is not in generated scenario catalog: {args.scenario}")
    identity_capture = (
        IdentityEvidenceCapture(
            args.identity_judge_evidence_dir.resolve(),
            experiment_id=args.experiment_id or args.scenario,
            baseline_id=args.identity_judge_policy_id or args.baseline,
            maximum_predictions=args.maximum_identity_judge_predictions,
        )
        if args.identity_judge_evidence_dir is not None
        else None
    )
    if experiment is not None and args.replay_start == 0.0 and args.replay_end == -1.0:
        scenario_start = _scenario_start_datetime(args.scenario)
        args.replay_start = max(
            0.0,
            (experiment.recording_start - scenario_start).total_seconds() - 5.0,
        )
        args.replay_end = (
            experiment.recording_end - scenario_start
        ).total_seconds() + args.deadline_seconds

    template = VARIANT_TEMPLATES[args.variant]
    allowed_seed_nodes = tuple(
        (
            f"dvpg_gq_orin_{node[4:]}"
            if node.startswith("orin") and node[4:].isdigit()
            else node
        )
        for node in replay_nodes
    )
    semantic_seed_source_ids: tuple[str, ...] = ()
    semantic_seed_node_ids: tuple[str, ...] = ()
    static_placement = None
    if args.baseline in {
        BaselineId.B0_PRODUCE_ALL.value,
        BaselineId.B1_STATIC_WHOLE_EVENT.value,
    }:
        static_placement = StaticPipelineRegistry.load(
            static_pipeline_registry_path()
        ).get_placement(args.variant, trace_id=args.scenario)
    if args.baseline == BaselineId.B0_PRODUCE_ALL.value:
        placement = static_placement
        if placement is not None:
            available_sources = set(
                _available_seed_sources(
                    args.scenario, template.seed_node_key, replay_nodes
                )
            )
            semantic_seed_source_ids = tuple(
                source_id
                for source_id in placement.allowed_source_ids
                if source_id in available_sources
            )
            replay_node_set = set(allowed_seed_nodes)
            semantic_seed_node_ids = tuple(
                node_id
                for node_id in placement.allowed_node_ids
                if node_id in replay_node_set
            )
    request_id = f"accuracy-{args.scenario}-{uuid4().hex[:8]}"
    submitter_id = f"accuracy-driver-{uuid4().hex[:8]}"
    replay_id = args.replay_id or f"{args.scenario}-{uuid4().hex[:8]}"
    allowed_execution_nodes = set(allowed_seed_nodes)
    allowed_execution_nodes.add("x86server")
    if (
        args.baseline == BaselineId.B1_STATIC_WHOLE_EVENT.value
        and static_placement is not None
    ):
        # B1's fixed chain may place compute on a node which does not own a
        # selected replay source. Include and await every authored execution
        # node; otherwise a freshly restarted orchestrator rejects an exact,
        # valid placement before that node's heartbeat is visible.
        allowed_execution_nodes.update(static_placement.allowed_node_ids)
    request = LiveComplexEventRequest(
        request_id=request_id,
        submitter_id=submitter_id,
        run_id=request_id,
        trace_id=args.scenario,
        replay_id=replay_id,
        family_id=template.family_id,
        baseline_placement_id=args.variant,
        parameters=template.request_parameters or {},
        baseline_id=BaselineId(args.baseline),
        seed_graph_node_key=template.seed_node_key,
        allowed_seed_source_ids=_available_seed_sources(
            args.scenario, template.seed_node_key, replay_nodes
        ),
        allowed_seed_node_ids=allowed_seed_nodes,
        semantic_seed_source_ids=semantic_seed_source_ids,
        semantic_seed_node_ids=semantic_seed_node_ids,
        # Live provider services consume node-local replay topics. Keep every
        # sensor-side step on a selected replay node while allowing the
        # central identity/orchestration providers on x86server.
        allowed_execution_node_ids=tuple(sorted(allowed_execution_nodes)),
        allow_raw_to_trusted_site_edge=args.allow_raw_to_trusted_site_edge,
        reject_seed_before_registration=True,
        allowed_seed_event_time_interval=_seed_event_time_interval(
            scenario_start=_scenario_start_datetime(args.scenario),
            replay_start_seconds=args.replay_start,
            replay_end_seconds=args.replay_end,
            labeled_event_end=(experiment.recording_end if experiment else None),
        ),
        seed_timeout_seconds=max(1, int(args.max_seconds)),
        # The evaluation driver owns the cell's wall-clock bound. Propagate it
        # to semantic planning so a later frontier cannot expire at the model
        # schema's unrelated five-minute default while the cell is still
        # legitimately replaying and processing evidence.
        deadline_offset_ms=evaluation_request_deadline_offset_ms(args.max_seconds),
        # B0 and B1 use the same CE-specific authored evidence semantics. B0
        # differs only by broadcasting those providers across eligible nodes,
        # so it must retain the same hypothesis capacity as B1/FABLE.
        max_seed_hypotheses=template.max_seed_hypotheses,
        seed_admission_strategy=template.seed_admission_strategy,
    )
    common_record_dir = args.common_record_dir
    if common_record_dir is None and args.output is not None:
        common_record_dir = args.output.parent / f"{args.output.stem}.records"
    record_store = JsonlEventStore(common_record_dir) if common_record_dir is not None else None
    evaluation_logger = (
        MqttEvaluationLogger(
            host=args.mqtt_host,
            port=args.mqtt_port,
            client_id=f"evaluation-logger-{uuid4().hex[:8]}",
            store=record_store,
            normalizer=EvaluationMessageNormalizer(
                RuntimeLoggingContext(
                    run_id=request.run_id,
                    baseline_id=request.baseline_id,
                    trace_id=request.trace_id,
                    default_request_id=request.request_id,
                )
            ),
        )
        if common_record_dir is not None
        else None
    )
    resource_monitor = (
        RunResourceMonitor(
            store=record_store,
            run_id=request.run_id,
            baseline_id=request.baseline_id,
            trace_id=request.trace_id,
            request_id=request.request_id,
            interval_seconds=args.resource_sample_interval_seconds,
        )
        if record_store is not None and args.resource_sample_interval_seconds > 0
        else None
    )
    replay_started_at: float | None = None
    replay_configured_at: float | None = None
    ready_replay_nodes: set[str] = set()
    expected_replay_completions: set[tuple[str, str]] = set()
    observed_replay_completions: set[tuple[str, str]] = set()
    generation_boundary_published = False
    state = RunState()
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=submitter_id,
        protocol=mqtt.MQTTv311,
    )
    telemetry_connected = threading.Event()
    telemetry_client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"{submitter_id}-telemetry",
        protocol=mqtt.MQTTv311,
    )
    resource_ack_lock = threading.Lock()
    resource_ack_events: dict[str, threading.Event] = {}
    resource_ack_payloads: dict[str, ResourceChangeAck] = {}
    shared_admission_members: set[str] = set()
    shared_admission_lock = threading.Lock()
    shared_admission_ready = threading.Event()
    shared_admission_prefix = f"fable/v1/evaluation/shared_admission/{replay_id}/"
    shared_admission_announced = False

    def observe_event_interval(document: dict[str, object]) -> None:
        interval = document.get("event_time_interval")
        if isinstance(interval, dict):
            values = (interval.get("start"), interval.get("end"))
        else:
            values = (document.get("event_time"), document.get("t"))
        for value in values:
            if not isinstance(value, str):
                continue
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
            if (
                state.observed_event_time_start is None
                or parsed < state.observed_event_time_start
            ):
                state.observed_event_time_start = parsed
            if (
                state.observed_event_time_end is None
                or parsed > state.observed_event_time_end
            ):
                state.observed_event_time_end = parsed

    def on_connect(client, _userdata, _flags, reason_code, _properties=None):
        if int(getattr(reason_code, "value", reason_code)) != 0:
            state.error = f"MQTT connect failed: {reason_code}"
            state.terminal.set()
            return
        client.subscribe(live_request_response_topic(submitter_id), qos=1)
        client.subscribe(live_request_cancel_response_topic(submitter_id), qos=1)
        client.subscribe(live_request_progress_topic(request_id), qos=1)
        client.subscribe(f"fable/v1/result/{request_id}/+", qos=1)
        client.subscribe("/readiness/#", qos=0)
        client.subscribe("/replay/status/#", qos=1)
        client.subscribe("/fable/v1/retrospective/+/raw-video/status", qos=1)
        client.subscribe(heartbeat_filter(), qos=0)
        # Install one stable subscription before declaring the MQTT channel
        # ready. Dynamic subscribe-then-publish races lost fast resource ACKs
        # when the broker delivered the response before the SUBACK completed.
        client.subscribe(resource_change_ack_filter(request.run_id), qos=1)
        client.subscribe(f"{shared_admission_prefix}+", qos=1)
        state.mqtt_connected.set()

    def on_telemetry_connect(
        telemetry, _userdata, _flags, reason_code, _properties=None
    ):
        if int(getattr(reason_code, "value", reason_code)) != 0:
            state.error = f"MQTT telemetry connect failed: {reason_code}"
            state.terminal.set()
            return
        telemetry.subscribe("/+/analytics/yolo/bbox", qos=0)
        telemetry.subscribe("/+/fable/context/tracks", qos=0)
        telemetry.subscribe("/+/fable/vehicle/predicates", qos=0)
        telemetry.subscribe("/+/fable/audio/events", qos=0)
        telemetry.subscribe("/+/fable/interactions/predicates", qos=0)
        telemetry.subscribe("/fable/identity/associations", qos=1)
        telemetry.subscribe("/debug/+/analytics/yolo/status", qos=0)
        telemetry.subscribe("/debug/+/audio_detector/status", qos=0)
        if identity_capture is not None:
            telemetry.subscribe("/+/fable/identity/descriptors", qos=0)
        telemetry_connected.set()

    def on_message(_client, _userdata, message):
        nonlocal generation_boundary_published, shared_admission_announced
        try:
            if message.topic.startswith(shared_admission_prefix):
                member = message.topic.removeprefix(shared_admission_prefix)
                if member:
                    with shared_admission_lock:
                        shared_admission_members.add(member)
                        if len(shared_admission_members) >= 2:
                            shared_admission_ready.set()
            elif message.topic.startswith(
                "fable/v1/evaluation/resource_change_ack/"
            ):
                acknowledgement = decode_model(message.payload, ResourceChangeAck)
                message_id = str(acknowledgement.request_message_id)
                if acknowledgement.run_id != request.run_id:
                    return
                with resource_ack_lock:
                    resource_ack_payloads[message_id] = acknowledgement
                    event = resource_ack_events.get(message_id)
                if event is not None:
                    event.set()
            elif message.topic == live_request_response_topic(submitter_id):
                response = decode_model(message.payload, LiveComplexEventResponse)
                if response.request_message_id != request.message_id:
                    return
                state.response = response
                state.seed_responses.append(response)
                if response.status == "WATCHING":
                    state.watch_registered_at = time.monotonic()
                    state.watching.set()
                elif response.status == "ADMITTED":
                    state.admitted_at = time.monotonic()
                    state.admitted.set()
                    if not shared_admission_announced:
                        shared_admission_announced = True
                        _client.publish(
                            f"{shared_admission_prefix}{request_id}",
                            json.dumps({"request_id": request_id}),
                            qos=1,
                            retain=False,
                        )
                else:
                    # A bounded additional seed may be rejected while an
                    # earlier hypothesis remains active. Preserve that
                    # diagnostic without terminating the admitted request.
                    if not state.admitted.is_set():
                        state.error = response.reason
                        state.terminal.set()
            elif message.topic == live_request_cancel_response_topic(submitter_id):
                response = decode_model(message.payload, LiveComplexEventCancelResponse)
                if response.request_id == request_id:
                    state.cancellation = response
                    state.cancellation_received.set()
            elif message.topic == live_request_progress_topic(request_id):
                progress = decode_model(message.payload, LiveComplexEventProgress)
                if not state.admitted.is_set():
                    # Request-scoped semantic progress is authoritative proof
                    # of admission even if a congested MQTT client missed the
                    # earlier acknowledgement.
                    state.admitted_at = time.monotonic()
                    state.admitted.set()
                    if not shared_admission_announced:
                        shared_admission_announced = True
                        _client.publish(
                            f"{shared_admission_prefix}{request_id}",
                            json.dumps({"request_id": request_id}),
                            qos=1,
                            retain=False,
                        )
                state.progress.append(progress)
                state.progress_status_counts[progress.transition_status] = (
                    state.progress_status_counts.get(progress.transition_status, 0) + 1
                )
                if progress.terminal:
                    state.terminal_at = time.monotonic()
                    state.terminal.set()
            elif message.topic.startswith(f"fable/v1/result/{request_id}/"):
                wrapper = decode_model(message.payload, ReliablePredicateResult)
                if not state.admitted.is_set():
                    state.admitted_at = time.monotonic()
                    state.admitted.set()
                if (
                    wrapper.result.semantic_predicate.predicate_id
                    == "PERSON_PRESENT_BEFORE"
                ):
                    state.recovered_person_samples.append(
                        {
                            "node_id": wrapper.node_id,
                            "event_time_interval": (
                                wrapper.result.event_time_interval.model_dump(
                                    mode="json"
                                )
                            ),
                            "person": (
                                wrapper.result.binding_delta.introduced.get("person")
                            ),
                        }
                    )
                elif (
                    wrapper.result.semantic_predicate.predicate_id
                    == "VEHICLE_PRESENT_BEFORE"
                ):
                    state.recovered_vehicle_samples.append(
                        {
                            "node_id": wrapper.node_id,
                            "event_time_interval": (
                                wrapper.result.event_time_interval.model_dump(
                                    mode="json"
                                )
                            ),
                            "vehicle": (
                                wrapper.result.binding_delta.introduced.get("vehicle")
                            ),
                        }
                    )
            elif message.topic.startswith("/replay/status/"):
                document = json.loads(message.payload.decode("utf-8"))
                if (
                    document.get("event") == "complete"
                    and str(document.get("replay_id") or "") == replay_id
                ):
                    parts = message.topic.strip("/").split("/")
                    service = parts[2] if len(parts) >= 4 else ""
                    node_id = str(
                        document.get("node")
                        or document.get("node_id")
                        or (parts[-1] if parts else "")
                    )
                    observed_replay_completions.add((node_id, service))
                    if (
                        expected_replay_completions
                        and not generation_boundary_published
                        and expected_replay_completions.issubset(
                            observed_replay_completions
                        )
                    ):
                        boundary = {
                            "schema_version": "fable.replay_generation_boundary.v1",
                            "event": "CLOSE",
                            "replay_id": replay_id,
                            "target_nodes": sorted(ready_replay_nodes),
                            "completed_sources": sorted(
                                f"{node}:{service}"
                                for node, service in observed_replay_completions
                            ),
                            "t": datetime.now(UTC).isoformat(),
                        }
                        _client.publish(
                            "/fable/v1/replay/generation-boundary",
                            json.dumps(boundary),
                            qos=1,
                            retain=True,
                        )
                        generation_boundary_published = True
            elif message.topic.startswith("/readiness/"):
                readiness = decode_replay_readiness(message.topic, message.payload)
                service = readiness.service_id
                node_id = readiness.node_id
                if service and node_id:
                    previous = state.readiness_documents.setdefault(node_id, {}).get(service)
                    if previous is not None and (
                        previous.process_instance_id == readiness.process_instance_id
                        and previous.generation >= readiness.generation
                    ):
                        return
                    state.readiness_documents[node_id][service] = readiness
                    node_services = state.readiness_by_node.setdefault(node_id, set())
                    if readiness.ready:
                        node_services.add(service)
                        state.readiness.add(service)
                    else:
                        node_services.discard(service)
            elif message.topic.startswith("fable/v1/status/") and message.topic.endswith(
                "/heartbeat"
            ):
                heartbeat = decode_model(message.payload, NodeHeartbeat)
                state.provider_heartbeats[heartbeat.node_id] = heartbeat
            elif (
                message.topic.startswith("/fable/v1/retrospective/")
                and message.topic.endswith("/raw-video/status")
            ):
                document = json.loads(message.payload.decode("utf-8"))
                state.retrospective_replay_statuses.append(document)
                if (
                    document.get("status") == "STARTED"
                    and state.first_retrospective_started_monotonic is None
                ):
                    state.first_retrospective_started_monotonic = time.monotonic()
            elif message.topic.endswith("/analytics/yolo/bbox"):
                state.yolo_messages += 1
                document = json.loads(message.payload.decode("utf-8"))
                rows = document if isinstance(document, list) else [document]
                for row in rows:
                    if isinstance(row, dict):
                        class_name = str(
                            row.get("class") or row.get("label") or "UNKNOWN"
                        )
                        state.yolo_class_counts[class_name] = (
                            state.yolo_class_counts.get(class_name, 0) + 1
                        )
            elif message.topic.endswith("/analytics/yolo/status"):
                document = json.loads(message.payload.decode("utf-8"))
                node = str(
                    document.get("hostname")
                    or document.get("node")
                    or document.get("source_host")
                    or message.topic
                )
                key = f"yolo:{node}"
                if document.get("sampling_policy") is not None:
                    state.detector_sampling_diagnostics[key] = {
                        "sampling_policy": document.get("sampling_policy"),
                        "source_fps": float(document.get("source_fps") or 0.0),
                        "sample_period_frames": float(
                            document.get("sample_period_frames") or 0.0
                        ),
                        "sampled_frame_count": int(
                            document.get("sampled_frame_count") or 0
                        ),
                        "last_sampled_frame_number": document.get(
                            "last_sampled_frame_number"
                        ),
                        "sampled_frame_numbers_tail": list(
                            document.get("sampled_frame_numbers_tail") or ()
                        ),
                    }
                observed = {
                    "frames": int(document.get("input_frames_total") or 0),
                    "detections": int(document.get("detections_total") or 0),
                    "dropped_frames": int(document.get("dropped_superseded_frames") or 0),
                    "inference_count": int(document.get("inference_count") or 0),
                    "inference_wall_seconds": float(
                        document.get("inference_wall_seconds") or 0.0
                    ),
                    "gpu_inference_seconds": float(
                        document.get("gpu_inference_seconds") or 0.0
                    ),
                    "reid_inference_count": int(
                        document.get("reid_inference_count") or 0
                    ),
                    "reid_inference_wall_seconds": float(
                        document.get("reid_inference_wall_seconds") or 0.0
                    ),
                    "reid_gpu_inference_seconds": float(
                        document.get("reid_gpu_inference_seconds") or 0.0
                    ),
                    "model_resident_seconds": float(
                        document.get("model_resident_seconds") or 0.0
                    ),
                }
                # Reassignment can briefly leave two implementations reporting
                # the same logical node. Preserve proof of work from either one
                # instead of letting a later idle status erase nonzero totals.
                previous = state.detector_totals_latest.get(key, {})
                state.detector_totals_latest[key] = {
                    metric: max(previous.get(metric, 0), value)
                    for metric, value in observed.items()
                }
            elif message.topic.endswith("/audio_detector/status"):
                document = json.loads(message.payload.decode("utf-8"))
                node = str(document.get("node") or message.topic)
                state.detector_totals_latest[f"audio:{node}"] = {
                    "windows": int(document.get("frames_total") or 0),
                    "detections": int(document.get("detections_total") or 0),
                }
            elif message.topic.endswith("/fable/context/tracks"):
                state.context_track_messages += 1
                document = json.loads(message.payload.decode("utf-8"))
                observe_event_interval(document)
                for track in document.get("tracks", ()):
                    class_name = str(track.get("class_name") or "UNKNOWN")
                    state.context_track_class_counts[class_name] = (
                        state.context_track_class_counts.get(class_name, 0) + 1
                    )
            elif message.topic.endswith("/fable/vehicle/predicates"):
                state.vehicle_predicate_messages += 1
                document = json.loads(message.payload.decode("utf-8"))
                observe_event_interval(document)
                predicate_id = str(document.get("predicate_id") or "UNKNOWN")
                state.vehicle_predicate_counts[predicate_id] = (
                    state.vehicle_predicate_counts.get(predicate_id, 0) + 1
                )
                source_ids = tuple(
                    str(item) for item in document.get("source_ids", ()) if str(item)
                )
                for source_id in source_ids:
                    key = f"{source_id}:{predicate_id}"
                    state.vehicle_predicate_source_counts[key] = (
                        state.vehicle_predicate_source_counts.get(key, 0) + 1
                    )
                predicate_sample_count = sum(
                    1
                    for sample in state.vehicle_predicate_samples
                    if sample.get("predicate_id") == predicate_id
                )
                # Retain enough of a normal-speed visit trace to diagnose
                # identity fragmentation across later semantic frontiers.
                if predicate_sample_count < 250:
                    state.vehicle_predicate_samples.append(
                        {
                            "topic": message.topic,
                            "predicate_id": predicate_id,
                            "replay_id": document.get("replay_id"),
                            "source_ids": source_ids,
                            "bindings": document.get("bindings") or {},
                            "event_time_interval": (
                                document.get("event_time_interval") or {}
                            ),
                            "measurements": document.get("measurements") or {},
                        }
                    )
            elif message.topic.endswith("/fable/audio/events"):
                document = json.loads(message.payload.decode("utf-8"))
                observed_replay_id = (
                    document.get("replay_id")
                    or (document.get("attributes") or {}).get("replay_id")
                )
                if observed_replay_id != replay_id:
                    return
                observe_event_interval(document)
                state.audio_event_messages += 1
                label = str(
                    document.get("label")
                    or document.get("event_type")
                    or document.get("class_name")
                    or "UNKNOWN"
                )
                state.audio_event_counts[label] = (
                    state.audio_event_counts.get(label, 0) + 1
                )
                source_id = str(
                    document.get("source_id")
                    or message.topic.strip("/").split("/", 1)[0]
                )
                node_counts = state.audio_events_by_node.setdefault(source_id, {})
                node_counts[label] = node_counts.get(label, 0) + 1
                interval = document.get("event_time_interval") or {}
                event_time = str(interval.get("start") or document.get("t") or "")
                if event_time:
                    prior = state.audio_event_time_ranges.get(label)
                    state.audio_event_time_ranges[label] = (
                        min(prior[0], event_time) if prior else event_time,
                        max(prior[1], event_time) if prior else event_time,
                    )
                recording_interval = document.get("recording_time_interval") or {}
                if recording_interval.get("start"):
                    state.audio_recording_samples.append(
                        {
                            "node_id": source_id,
                            "label": label,
                            "event_time_interval": interval,
                            "recording_time_interval": recording_interval,
                            "replay_id": document.get("replay_id"),
                        }
                    )
            elif message.topic.endswith("/fable/interactions/predicates"):
                state.interaction_predicate_messages += 1
                document = json.loads(message.payload.decode("utf-8"))
                observe_event_interval(document)
                predicate_id = str(document.get("predicate_id") or "UNKNOWN")
                state.interaction_predicate_counts[predicate_id] = (
                    state.interaction_predicate_counts.get(predicate_id, 0) + 1
                )
                # Keep a representative sample of every predicate. A frequent
                # predicate such as PERSON_PROXIMITY must not hide sparse
                # DISEMBARKS or BOARDS transitions later in the replay.
                predicate_sample_count = sum(
                    1
                    for sample in state.interaction_predicate_samples
                    if sample.get("predicate_id") == predicate_id
                )
                if predicate_sample_count < 100:
                    state.interaction_predicate_samples.append(
                        {
                            "topic": message.topic,
                            "predicate_id": predicate_id,
                            "occurrence_id": document.get("occurrence_id"),
                            "event_time_interval": document.get("event_time_interval"),
                            "bindings": document.get("bindings") or {},
                            "source_ids": document.get("source_ids") or (),
                            "replay_id": document.get("replay_id"),
                        }
                    )
            elif message.topic.endswith("/fable/identity/descriptors"):
                if identity_capture is not None:
                    identity_capture.observe_descriptor(
                        json.loads(message.payload.decode("utf-8"))
                    )
            elif message.topic == "/fable/identity/associations":
                document = json.loads(message.payload.decode("utf-8"))
                if document.get("schema_version") != "canonical_entity_map.v1":
                    return
                associations = document.get("associations") or ()
                entity_kind = str(document.get("entity_kind") or "unknown")
                state.identity_association_messages += 1
                state.identity_associations_by_kind[entity_kind] = (
                    state.identity_associations_by_kind.get(entity_kind, 0)
                    + len(associations)
                )
                if len(state.identity_association_samples) < 100:
                    state.identity_association_samples.append(
                        {
                            "entity_kind": entity_kind,
                            "left_source_id": document.get("left_source_id"),
                            "right_source_id": document.get("right_source_id"),
                            "associations": associations,
                        }
                    )
                if identity_capture is not None:
                    identity_capture.observe_associations(document)
        except Exception as exc:
            state.error = f"message handling failed: {exc}"
            state.terminal.set()

    client.on_connect = on_connect
    client.on_message = on_message
    telemetry_client.on_connect = on_telemetry_connect
    telemetry_client.on_message = on_message
    started = time.monotonic()
    started_at = datetime.now(UTC)
    # Registration and replay-service readiness are setup phases.  They must
    # not consume the bounded experiment/replay budget.
    setup_deadline = started + args.ready_seconds
    deadline = setup_deadline
    replay_configured_monotonic: float | None = None
    replay_sync_monotonic: float | None = None
    cleanup_started_monotonic: float | None = None
    cleanup_completed_monotonic: float | None = None
    replay_stop_started_monotonic: float | None = None
    replay_stop_completed_monotonic: float | None = None
    disturbance_results: list[dict[str, object]] = []
    disturbance_stop = threading.Event()
    condition_initial_transitions_applied = threading.Event()
    disturbance_thread: threading.Thread | None = None
    condition_trace_started_monotonic: float | None = None
    condition_trace_started_at: datetime | None = None
    condition_trace_anchor_monotonic: float | None = None
    condition_trace_anchor_at: datetime | None = None
    actual_ce_start_offset_s: float | None = None

    def notify_resource_change(
        condition: str,
        action: str,
        epoch: int,
        *,
        target_id: str | None = None,
        resource_kind: str = "NETWORK",
    ) -> ResourceChangeAck:
        change = ResourceChange(
            run_id=request.run_id,
            condition=condition,
            action=action,
            condition_epoch=epoch,
            target_id=target_id,
            resource_kind=ResourceKind(resource_kind),
            observed_at=datetime.now(UTC),
        )
        correlation_id = str(change.message_id)
        ack_received = threading.Event()
        with resource_ack_lock:
            resource_ack_payloads.pop(correlation_id, None)
            resource_ack_events[correlation_id] = ack_received
        notification = client.publish(
            resource_change_topic(),
            encode_model(change),
            qos=1,
            retain=False,
        )
        notification.wait_for_publish(timeout=5.0)
        try:
            if not ack_received.wait(args.resource_ack_timeout_seconds):
                raise RuntimeError(
                    f"orchestrator did not acknowledge resource epoch {epoch}"
                )
            with resource_ack_lock:
                acknowledgement = resource_ack_payloads.get(correlation_id)
            if acknowledgement is None or not acknowledgement.accepted:
                raise RuntimeError(
                    "orchestrator rejected resource change: "
                    f"{acknowledgement.reason if acknowledgement else 'unknown reason'}"
                )
            return acknowledgement
        finally:
            with resource_ack_lock:
                resource_ack_events.pop(correlation_id, None)
                resource_ack_payloads.pop(correlation_id, None)

    def apply_network_condition(
        condition: str,
        action: str,
        epoch: int,
        *,
        notify: bool = True,
        transition_id: str | None = None,
        requested_offset_s: float | None = None,
        target: str = "site_to_cloud",
        kind: str = "NETWORK_PROFILE",
    ) -> bool:
        requested_at = datetime.now(UTC)
        apply_started_at = datetime.now(UTC)
        apply_started = time.monotonic()
        if target == "physical_link:rpi_to_jetson":
            if args.physical_condition_identity_file is None:
                raise RuntimeError("physical condition target requires --physical-condition-identity-file")
            physical_action = (
                "disconnect-network"
                if kind == "LINK_STATE" and action == "FAIL"
                else "apply-network"
                if action == "APPLY"
                else "restore-network"
            )
            command = [
                sys.executable, str(ROOT / "scripts/physical_condition_control.py"),
                physical_action, "--identity-file",
                str(args.physical_condition_identity_file.resolve()),
                "--target", "rpi_to_jetson", "--profile",
                "P1_JETSON_PATH_DEGRADED" if physical_action == "apply-network" else "N0",
                "--execute",
            ]
        else:
            helper = ROOT / "netwaggle/scripts/fable_netwaggle_helper.py"
            command = [
                sys.executable, str(helper), "--kind", kind, "--target", target,
                "--condition", condition, "--action", action,
                "--condition-epoch", str(epoch),
            ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=75,
        )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError:
            response = {"validated": False, "reason": completed.stdout[-1000:]}
        result_row = {
                "transition_id": transition_id,
                "condition": condition,
                "action": action,
                "returncode": completed.returncode,
                "requested_offset_s": requested_offset_s,
                "requested_at": requested_at.isoformat(),
                "apply_started_at": apply_started_at.isoformat(),
                "applied_at": datetime.now(UTC).isoformat(),
                "observed_at": datetime.now(UTC).isoformat(),
                "helper_wall_seconds": round(time.monotonic() - apply_started, 6),
                "applied_offset_s": (
                    round(time.monotonic() - condition_trace_started_monotonic, 6)
                    if condition_trace_started_monotonic is not None
                    else None
                ),
                "response": response,
            }
        disturbance_results.append(result_row)
        validated = response.get("validated") is True
        if evaluation_logger is not None:
            evaluation_logger.store.append(
                DisturbanceEvent(
                    run_id=request.run_id,
                    baseline_id=request.baseline_id,
                    trace_id=request.trace_id,
                    request_id=request.request_id,
                    event_time=requested_at,
                    wall_timestamp=datetime.now(UTC),
                    monotonic_timestamp_ns=time.perf_counter_ns(),
                    disturbance_id=(
                        f"{transition_id or condition}:{action}:{epoch}"
                    ),
                    disturbance_type=kind,
                    action=action,
                    target_ids=(target,),
                    condition_epoch=epoch,
                    scheduled_trigger=(
                        f"MONOTONIC_OFFSET:{requested_offset_s}"
                        if requested_offset_s is not None
                        else "LEGACY_REPLAY_RELATIVE"
                    ),
                    validated=validated,
                    metadata={
                        "condition_id": condition,
                        "requested_at": requested_at.isoformat(),
                        "apply_started_at": apply_started_at.isoformat(),
                        "applied_at": result_row["applied_at"],
                        "requested_offset_s": requested_offset_s,
                        "applied_offset_s": result_row["applied_offset_s"],
                        "helper_wall_seconds": result_row["helper_wall_seconds"],
                        "helper_returncode": completed.returncode,
                    },
                )
            )
        if validated and notify:
            planner_condition = condition
            planner_target = target
            if target == "physical_link:rpi_to_jetson":
                # Preserve the physical profile identity on RESTORE so the
                # orchestrator restores the saved base deployment rather than
                # routing N0 through the unrelated Mininet profile loader.
                planner_node = args.physical_network_planner_node_id
                if not planner_node:
                    raise RuntimeError(
                        "physical network condition requires "
                        "--physical-network-planner-node-id"
                    )
                if kind == "LINK_STATE":
                    planner_condition = condition
                    planner_target = canonical_sensor_link_target(planner_node)
                else:
                    planner_condition = planner_network_condition(condition, target)
                    planner_target = planner_node
            # The host mutation schedule is exogenous to every evaluated
            # policy.  A baseline may reject (or be unable to route) the
            # resource-change notification, but that must not terminate the
            # condition worker and suppress later immutable transitions such
            # as RESTORE_LINK.  Preserve the rejection as adaptation evidence
            # while continuing the schedule.
            record_condition_notification(
                result_row,
                lambda: notify_resource_change(
                    planner_condition,
                    action,
                    epoch,
                    target_id=planner_target,
                    resource_kind=canonical_resource_kind(kind),
                ),
            )
        return validated

    def apply_compute_condition(
        condition: str,
        action: str,
        epoch: int,
        transition_id: str,
        requested_offset_s: float,
        target: str = "x86server",
    ) -> bool:
        """Invoke only the fixed YOLO contention controller."""

        started_at = datetime.now(UTC)
        started_mono = time.monotonic()
        anchor_mono = condition_trace_anchor_monotonic
        if target == "physical_jetson":
            if args.physical_condition_identity_file is None:
                raise RuntimeError("physical compute target requires --physical-condition-identity-file")
            command = [
                sys.executable, str(ROOT / "scripts/physical_condition_control.py"),
                "apply-compute" if action == "APPLY" else "clear-compute",
                "--identity-file", str(args.physical_condition_identity_file.resolve()),
                "--target", "physical_jetson", "--profile", condition,
                "--execute",
            ]
        else:
            command = [sys.executable, str(ROOT / "scripts/fable_compute_contention.py"),
                       "--condition", condition, "--action", action]
        completed = subprocess.run(
            command,
            cwd=ROOT, text=True, capture_output=True, check=False, timeout=45,
        )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError:
            response = {
                "validated": False,
                "reason": (completed.stderr or completed.stdout)[-1000:],
            }
        validated = response.get("validated") is True
        completed_mono = time.monotonic()
        result_row = {
            "transition_id": transition_id,
            "condition": condition,
            "action": action,
            "requested_offset_s": requested_offset_s,
            "requested_at": started_at.isoformat(),
            "applied_at": datetime.now(UTC).isoformat(),
            "condition_epoch": epoch,
            "resource_kind": "COMPUTE",
            "application_started_offset_s": (
                round(started_mono - anchor_mono, 6)
                if anchor_mono is not None else None
            ),
            "applied_offset_s": condition_offset(
                completed_mono,
                anchor=anchor_mono,
                trace_started=condition_trace_started_monotonic,
            ),
            "helper_wall_seconds": round(completed_mono - started_mono, 6),
            "returncode": completed.returncode,
            "response": response,
        }
        disturbance_results.append(result_row)
        if validated:
            planner_condition = condition
            planner_target = target
            if target == "physical_jetson":
                planner_condition = "E1" if action == "APPLY" else "N0"
                planner_target = args.physical_compute_planner_node_id
                if not planner_target:
                    raise RuntimeError(
                        "physical compute condition requires "
                        "--physical-compute-planner-node-id"
                    )
            record_condition_notification(
                result_row,
                lambda: notify_resource_change(
                    planner_condition,
                    action,
                    epoch,
                    target_id=planner_target,
                    resource_kind="COMPUTE",
                ),
            )
        return validated

    def apply_provider_fault(
        target: str,
        action: str,
        epoch: int,
        transition_id: str,
        requested_offset_s: float,
    ) -> bool:
        """Publish a typed provider-family fault; target is node:provider."""

        try:
            node_id, provider_id = target.split(":", 1)
        except ValueError:
            validated = False
            reason = "provider target must be <node_id>:<provider_id>"
        else:
            kind = FaultKind.FAIL_PROVIDER if action == "FAIL" else FaultKind.RESTORE_PROVIDER
            message = FaultCommand(
                target_id=node_id,
                kind=kind,
                provider_id=provider_id,
                reason=f"RQ3a transition {transition_id}",
            )
            publish = client.publish(
                fault_topic(node_id), encode_model(message), qos=1, retain=False
            )
            publish.wait_for_publish(timeout=5.0)
            validated = publish.is_published()
            reason = "typed fault command published" if validated else "MQTT publish failed"
        disturbance_results.append({
            "transition_id": transition_id,
            "action": action,
            "target": target,
            "condition_epoch": epoch,
            "requested_offset_s": requested_offset_s,
            "applied_offset_s": round(
                time.monotonic() - condition_trace_started_monotonic, 6
            ),
            "validated": validated,
            "reason": reason,
        })
        return validated

    def disturbance_worker() -> None:
        # Stage the host condition as soon as replay is synchronized.  The helper can
        # take longer than short CE traces remain active, so notification is deferred
        # until admission: this preserves causal ordering without losing the only
        # opportunity for the live planner to adapt.
        validated = apply_network_condition(
            args.network_disturbance,
            "APPLY",
            1,
            notify=False,
        )
        if not validated:
            return
        while not state.admitted.is_set():
            if disturbance_stop.wait(0.1):
                return
        if disturbance_stop.wait(args.disturbance_delay_seconds):
            return
        notify_resource_change(args.network_disturbance, "APPLY", 1)
        if not disturbance_stop.wait(args.disturbance_duration_seconds):
            apply_network_condition("N0", "RESTORE", 2)

    def condition_trace_worker() -> None:
        nonlocal condition_trace_anchor_monotonic, condition_trace_anchor_at
        assert condition_trace is not None
        assert condition_trace_started_monotonic is not None
        if condition_trace.anchor == ConditionAnchor.ADMISSION:
            while not state.admitted.is_set():
                if disturbance_stop.wait(0.02):
                    return
            condition_trace_anchor_monotonic = state.admitted_at or time.monotonic()
            condition_trace_anchor_at = datetime.now(UTC)
        else:
            condition_trace_anchor_monotonic = condition_trace_started_monotonic
            condition_trace_anchor_at = condition_trace_started_at
        controller = MonotonicConditionTraceController(condition_trace)
        epoch = 0
        while not disturbance_stop.is_set():
            elapsed = time.monotonic() - condition_trace_anchor_monotonic
            for due in controller.due(elapsed_s=elapsed):
                transition = due.transition
                if transition.action == ConditionAction.APPLY_NETWORK_PROFILE:
                    helper_action = "APPLY"
                elif transition.action == ConditionAction.RESTORE_NETWORK_PROFILE:
                    helper_action = "RESTORE"
                elif transition.action in {
                    ConditionAction.FAIL_LINK, ConditionAction.RESTORE_LINK
                }:
                    epoch += 1
                    apply_network_condition(
                        "L1" if transition.action == ConditionAction.FAIL_LINK else "N0",
                        "FAIL" if transition.action == ConditionAction.FAIL_LINK else "RESTORE",
                        epoch,
                        transition_id=transition.transition_id,
                        requested_offset_s=transition.offset_s,
                        target=transition.target_id or "",
                        kind="LINK_STATE",
                        notify=True,
                    )
                    continue
                elif transition.action in {
                    ConditionAction.APPLY_COMPUTE_CONTENTION,
                    ConditionAction.CLEAR_COMPUTE_CONTENTION,
                }:
                    epoch += 1
                    apply_compute_condition(
                        transition.profile_id or condition_trace.initial_compute_profile,
                        "APPLY" if transition.action == ConditionAction.APPLY_COMPUTE_CONTENTION else "RESTORE",
                        epoch,
                        transition.transition_id,
                        transition.offset_s,
                        transition.target_id or "x86server",
                    )
                    continue
                elif transition.action in {
                    ConditionAction.FAIL_PROVIDER, ConditionAction.RESTORE_PROVIDER
                }:
                    epoch += 1
                    apply_provider_fault(
                        transition.target_id or "",
                        "FAIL" if transition.action == ConditionAction.FAIL_PROVIDER else "RESTORE",
                        epoch,
                        transition.transition_id,
                        transition.offset_s,
                    )
                    continue
                else:
                    disturbance_results.append(
                        {
                            "transition_id": transition.transition_id,
                            "action": transition.action.value,
                            "requested_offset_s": transition.offset_s,
                            "requested_at": datetime.now(UTC).isoformat(),
                            "validated": False,
                            "reason": (
                                "live runner has no allowlisted host adapter for "
                                f"{transition.action.value}"
                            ),
                        }
                    )
                    continue
                epoch += 1
                apply_network_condition(
                    transition.profile_id or condition_trace.initial_network_profile,
                    helper_action,
                    epoch,
                    transition_id=transition.transition_id,
                    requested_offset_s=transition.offset_s,
                    target=transition.target_id or "site_to_cloud",
                )
            # TRACE_START transitions at offset zero must complete before the
            # replay synchronization message is published. Otherwise the
            # condition worker races the first sensor frames and a nominal
            # prefix leaks into a link-loss experiment.
            condition_initial_transitions_applied.set()
            if controller.complete or elapsed >= condition_trace.duration_s:
                return
            disturbance_stop.wait(0.02)

    if evaluation_logger is not None:
        evaluation_logger.start()
    if resource_monitor is not None:
        resource_monitor.start()
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
    client.loop_start()
    telemetry_client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
    telemetry_client.loop_start()
    try:
        if not state.mqtt_connected.wait(min(args.ready_seconds, 15.0)):
            state.error = "timed out connecting to MQTT"
            state.terminal.set()
        if not telemetry_connected.wait(min(args.ready_seconds, 15.0)):
            state.error = "timed out connecting MQTT telemetry channel"
            state.terminal.set()
        expected_provider_nodes = set(request.allowed_execution_node_ids)
        provider_ready_deadline = time.monotonic() + args.ready_seconds
        while (
            not state.terminal.is_set()
            and not expected_provider_nodes.issubset(state.provider_heartbeats)
            and time.monotonic() < provider_ready_deadline
        ):
            state.terminal.wait(0.1)
        missing_provider_nodes = sorted(
            expected_provider_nodes - set(state.provider_heartbeats)
        )
        unavailable_provider_nodes = sorted(
            node_id
            for node_id in expected_provider_nodes
            if node_id in state.provider_heartbeats
            and state.provider_heartbeats[node_id].availability.value != "AVAILABLE"
        )
        if missing_provider_nodes or unavailable_provider_nodes:
            state.error = (
                "provider-catalog readiness barrier failed: "
                f"missing={missing_provider_nodes}, unavailable={unavailable_provider_nodes}"
            )
            state.terminal.set()
        if not state.terminal.is_set():
            # Register the watch only after every planner-visible execution
            # node has advertised its current session and capacity.
            client.publish("/replay/sync", payload=b"", qos=1, retain=True)
            client.publish(
                live_request_topic(args.orchestrator_id),
                encode_model(request),
                qos=1,
            )
        while (
            not state.watching.is_set()
            and not state.terminal.is_set()
            and time.monotonic() < deadline
        ):
            state.watching.wait(0.1)
        if not state.watching.is_set():
            state.error = state.error or "timed out registering seed watch"
        elif args.shared_replay_role == "follower":
            # The owner publishes the one generation boundary, replay config,
            # and synchronization command. This request observes the same
            # replay-ID-scoped evidence without restarting or duplicating any
            # sensor producer.
            deadline = time.monotonic() + args.max_seconds + args.ready_seconds
            while not state.terminal.is_set() and time.monotonic() < deadline:
                state.terminal.wait(0.1)
            if not state.terminal.is_set():
                state.error = "shared replay follower timed out awaiting terminal result"
        else:
            if args.shared_replay_owner_start_delay_seconds:
                state.terminal.wait(args.shared_replay_owner_start_delay_seconds)
            config = {
                "scenario": args.scenario,
                "start_time": args.replay_start,
                "end_time": args.replay_end,
                "playback_mode": args.playback_mode,
                "speed": args.playback_speed,
                "replay_id": replay_id,
                "target_nodes": sorted(replay_nodes),
            }
            replay_configured_at = time.time()
            replay_configured_monotonic = time.monotonic()
            # A boundary is replay-ID scoped, but clear the retained prior
            # generation before announcing a new replay so restarted agents
            # cannot observe stale control state before /replay/sync.
            client.publish(
                "/fable/v1/replay/generation-boundary",
                payload=None,
                qos=1,
                retain=True,
            ).wait_for_publish(timeout=2.0)
            state.detector_totals_baseline = {
                key: dict(value) for key, value in state.detector_totals_latest.items()
            }
            config_publish = client.publish(
                "/replay/config", json.dumps(config), qos=1, retain=True
            )
            config_publish.wait_for_publish(timeout=2.0)
            required = {
                item.strip()
                for item in args.required_ready_services.split(",")
                if item.strip()
            }
            ready_deadline = time.monotonic() + args.ready_seconds
            readiness_refresh_at = time.monotonic() + min(10.0, args.ready_seconds / 3.0)
            expected_nodes = {
                (
                    f"dvpg_gq_orin_{node[4:]}"
                    if node.startswith("orin") and node[4:].isdigit()
                    else node
                )
                for node in replay_nodes
            }

            def current_selected_yolo_nodes() -> set[str]:
                if record_store is None:
                    return set()
                return {
                    str(row.get("node_id"))
                    for row in record_store.read("provider_command")
                    if str(row.get("provider_id", "")).startswith("yolo_")
                    and row.get("node_id")
                }

            def current_ready_nodes() -> set[str]:
                ready_nodes = set()
                selected_yolo = current_selected_yolo_nodes()
                for node_id in expected_nodes:
                    mobile_node = node_id.startswith("mobile_archive_")
                    node_required = replay_node_readiness_requirements(
                        required,
                        node_id=node_id,
                        selected_yolo_nodes=selected_yolo,
                    )
                    if not node_required.issubset(
                        state.readiness_by_node.get(node_id, set())
                    ):
                        continue
                    scene_service = "mobile" if mobile_node else "zed"
                    readiness = state.readiness_documents.get(node_id, {}).get(
                        scene_service
                    )
                    if (
                        scene_service in node_required
                        and (
                            readiness is None
                            or readiness.scenario != args.scenario
                            or readiness.replay_id != replay_id
                        )
                    ):
                        continue
                    readiness_time = (
                        readiness.observed_at.timestamp() if readiness is not None else 0
                    )
                    if (
                        replay_configured_at is not None
                        and readiness_time < replay_configured_at
                    ):
                        continue
                    ready_nodes.add(node_id)
                return ready_nodes

            while not state.terminal.is_set() and time.monotonic() < ready_deadline:
                selected_yolo = current_selected_yolo_nodes()
                analytics_selected = "yolo" not in required or bool(selected_yolo)
                if analytics_selected and expected_nodes.issubset(current_ready_nodes()):
                    break
                # Selected analytics may start after the first retained replay
                # configuration was published. Re-publishing the identical
                # generation is idempotent and asks late-starting providers to
                # refresh readiness; replay cannot begin until /replay/sync.
                # This closes the rotating one-node YOLO readiness race without
                # weakening the all-selected-node synchronization contract.
                if time.monotonic() >= readiness_refresh_at:
                    client.publish(
                        "/replay/config", json.dumps(config), qos=1, retain=True
                    ).wait_for_publish(timeout=2.0)
                    readiness_refresh_at = time.monotonic() + min(
                        10.0, max(2.0, args.ready_seconds / 3.0)
                    )
                state.terminal.wait(0.1)
            ready_replay_nodes = current_ready_nodes()
            if expected_nodes.issubset(ready_replay_nodes):
                # Close globally only when every replay modality that passed
                # the readiness barrier reports natural EOF. This avoids both
                # placement-dependent leakage and premature closure when ZED
                # finishes slightly before a co-located microphone replay.
                expected_replay_completions.clear()
                observed_replay_completions.clear()
                expected_replay_completions.update(
                    replay_completion_requirements(
                        ready_replay_nodes,
                        state.readiness_by_node,
                    )
                )
                if condition_trace is not None:
                    condition_trace_started_monotonic = time.monotonic()
                    condition_trace_started_at = datetime.now(UTC)
                    disturbance_thread = threading.Thread(
                        target=condition_trace_worker,
                        name="rq3a-condition-trace",
                        daemon=True,
                    )
                    disturbance_thread.start()
                    if (
                        condition_trace.anchor == ConditionAnchor.TRACE_START
                        and any(
                            transition.offset_s == 0
                            for transition in condition_trace.transitions
                        )
                        and not condition_initial_transitions_applied.wait(30.0)
                    ):
                        state.error = (
                            "timed out applying zero-offset condition transitions "
                            "before replay"
                        )
                        state.terminal.set()
                    target = (
                        condition_trace_started_monotonic
                        + args.ce_start_offset_seconds
                    )
                    while time.monotonic() < target and not state.terminal.is_set():
                        state.terminal.wait(min(0.05, target - time.monotonic()))
                replay_started_at = time.time() + 0.5
                replay_sync_monotonic = time.monotonic()
                # The advertised max-seconds budget starts at the replay sync,
                # not while containers and decoders are becoming ready.
                deadline = replay_sync_monotonic + args.max_seconds
                if condition_trace_started_monotonic is not None:
                    actual_ce_start_offset_s = (
                        replay_sync_monotonic - condition_trace_started_monotonic
                    )
                client.publish(
                    "/replay/sync",
                    json.dumps(
                        {
                            "scenario": args.scenario,
                            "start_at": replay_started_at,
                            # Keep the synchronization barrier in wall-clock
                            # time, but stamp evidence in the source recording
                            # time domain.  Conflating these made retained April
                            # media appear unavailable to August evaluations.
                            "event_start_at": (
                                _scenario_start_datetime(args.scenario)
                                + timedelta(seconds=args.replay_start)
                            ).timestamp(),
                            "playback_mode": args.playback_mode,
                            "speed": args.playback_speed,
                            "replay_id": replay_id,
                            "target_nodes": sorted(replay_nodes),
                        }
                    ),
                    qos=1,
                    retain=False,
                )
                if args.shared_replay_joint_admission_barrier:
                    if not state.admitted.wait(min(60.0, args.max_seconds)):
                        state.error = (
                            "joint admission barrier timed out awaiting owner admission"
                        )
                        state.terminal.set()
                    else:
                        # PendingSeedRegistry admits every matching watch in
                        # one synchronous observe() pass before publishing any
                        # ADMITTED response. Receiving the owner's response is
                        # therefore the authoritative server-side barrier;
                        # peer MQTT announcements are diagnostic only.
                        notify_resource_change(
                            condition="N0",
                            action="RESTORE",
                            epoch=900_000,
                            target_id=(
                                args.physical_compute_planner_node_id or "x86server"
                            ),
                            resource_kind="COMPUTE",
                        )
                if args.network_disturbance is not None:
                    disturbance_thread = threading.Thread(
                        target=disturbance_worker,
                        name="bounded-network-disturbance",
                        daemon=True,
                    )
                    disturbance_thread.start()
                if condition_trace is not None:
                    assert condition_trace_started_monotonic is not None
                    trace_end = (
                        condition_trace_started_monotonic + condition_trace.duration_s
                    )
                    # A condition trace is a maximum observation horizon, not
                    # a mandatory sleep.  Stop as soon as the request reaches
                    # a terminal state; cleanup still restores any active
                    # network/compute mutation in the normal finally path.
                    while (
                        not state.terminal.is_set()
                        and time.monotonic() < min(deadline, trace_end)
                    ):
                        recovery_started = state.first_retrospective_started_monotonic
                        if (
                            args.vehicle_recovery_timeout_seconds > 0
                            and recovery_started is not None
                            and not state.recovered_vehicle_samples
                            and time.monotonic() - recovery_started
                            >= args.vehicle_recovery_timeout_seconds
                        ):
                            state.error = (
                                "vehicle recovery produced no candidate within "
                                f"{args.vehicle_recovery_timeout_seconds:g}s of "
                                "raw retrospective replay start"
                            )
                        if state.error:
                            break
                        time.sleep(0.05)
                else:
                    while not state.terminal.is_set() and time.monotonic() < deadline:
                        recovery_started = state.first_retrospective_started_monotonic
                        if (
                            args.vehicle_recovery_timeout_seconds > 0
                            and recovery_started is not None
                            and not state.recovered_vehicle_samples
                            and time.monotonic() - recovery_started
                            >= args.vehicle_recovery_timeout_seconds
                        ):
                            state.error = (
                                "vehicle recovery produced no candidate within "
                                f"{args.vehicle_recovery_timeout_seconds:g}s of "
                                "raw retrospective replay start"
                            )
                            break
                        state.terminal.wait(0.1)
            else:
                missing_nodes = sorted(expected_nodes - ready_replay_nodes)
                readiness_failures = []
                for node_id in missing_nodes:
                    mobile_node = node_id.startswith("mobile_archive_")
                    scene_service = "mobile" if mobile_node else "zed"
                    node_required = (
                        (required - {"zed"}) | {"mobile"}
                        if mobile_node and "zed" in required
                        else required
                    )
                    observed_services = state.readiness_by_node.get(node_id, set())
                    absent = sorted(node_required - observed_services)
                    readiness = state.readiness_documents.get(node_id, {}).get(
                        scene_service
                    )
                    if absent:
                        reason = "missing services " + "+".join(absent)
                    elif readiness is None:
                        reason = f"missing {scene_service} readiness document"
                    elif readiness.scenario != args.scenario:
                        reason = (
                            f"{scene_service} scenario mismatch "
                            f"observed={readiness.scenario!r} expected={args.scenario!r}"
                        )
                    elif readiness.replay_id != replay_id:
                        reason = (
                            f"{scene_service} replay generation mismatch "
                            f"observed={readiness.replay_id!r} expected={replay_id!r}"
                        )
                    elif (
                        replay_configured_at is not None
                        and readiness.observed_at.timestamp() < replay_configured_at
                    ):
                        reason = f"stale {scene_service} readiness"
                    else:
                        reason = "readiness requirements not satisfied"
                    readiness_failures.append(f"{node_id} ({reason})")
                state.error = (
                    "readiness timeout; selected replay nodes are not ready: "
                    + "; ".join(readiness_failures)
                )
    finally:
        disturbance_stop.set()
        if disturbance_thread is not None:
            disturbance_thread.join(timeout=25)
        if args.network_disturbance is not None and not any(
            item.get("action") == "RESTORE" for item in disturbance_results
        ):
            apply_network_condition("N0", "RESTORE", 3)
        # Fail-safe cleanup is per disturbance family.  A WAN restore must not
        # accidentally suppress cleanup of compute load, a failed provider, or
        # a down sensor link from the same trace.
        if condition_trace is not None:
            cleanup_epoch = 10_000
            transitions = condition_trace.transitions
            if any(
                item.action == ConditionAction.APPLY_COMPUTE_CONTENTION
                for item in transitions
            ):
                apply_compute_condition(
                    "N0", "RESTORE", cleanup_epoch,
                    "cleanup:compute", condition_trace.duration_s,
                    next((item.target_id for item in transitions if item.action == ConditionAction.APPLY_COMPUTE_CONTENTION and item.target_id), "x86server"),
                )
                cleanup_epoch += 1
            for target in sorted({
                item.target_id
                for item in transitions
                if item.action == ConditionAction.FAIL_PROVIDER and item.target_id
            }):
                apply_provider_fault(
                    target, "RESTORE", cleanup_epoch,
                    "cleanup:provider", condition_trace.duration_s,
                )
                cleanup_epoch += 1
            for target in sorted({
                item.target_id
                for item in transitions
                if item.action == ConditionAction.FAIL_LINK and item.target_id
            }):
                apply_network_condition(
                    "N0", "RESTORE", cleanup_epoch,
                    transition_id="cleanup:link",
                    requested_offset_s=condition_trace.duration_s,
                    target=target,
                    kind="LINK_STATE",
                    notify=False,
                )
                cleanup_epoch += 1
            for target in sorted({
                item.target_id or "site_to_cloud"
                for item in transitions
                if item.action == ConditionAction.APPLY_NETWORK_PROFILE
            }):
                apply_network_condition(
                    "N0", "RESTORE", cleanup_epoch,
                    transition_id="cleanup:network",
                    requested_offset_s=condition_trace.duration_s,
                    target=target,
                )
                cleanup_epoch += 1
        cleanup_started_monotonic = time.monotonic()
        cancellation = LiveComplexEventCancelRequest(
            request_id=request_id,
            submitter_id=submitter_id,
            reason="bounded replay accuracy run finished",
        )
        cancellation_publish = client.publish(
            live_request_cancel_topic(args.orchestrator_id),
            encode_model(cancellation),
            qos=1,
        )
        cancellation_publish.wait_for_publish(timeout=2.0)
        state.cancellation_received.wait(timeout=5.0)
        if (
            args.shared_replay_role == "owner"
            and args.shared_replay_owner_stop_grace_seconds
        ):
            time.sleep(args.shared_replay_owner_stop_grace_seconds)
        # Stop replay production before the next matrix cell can publish a new
        # replay ID. Without this barrier the previous ZED/mobile playback and
        # queued YOLO frames can overlap the next run.
        replay_stop_started_monotonic = time.monotonic()
        if args.shared_replay_role == "owner":
            stop_payload = {
                "action": "STOP",
                "replay_id": replay_id,
                "target_nodes": sorted(replay_nodes),
            }
            stop_config = client.publish(
                "/replay/config",
                json.dumps(stop_payload),
                qos=1,
                retain=True,
            )
            stop_sync = client.publish(
                "/replay/sync",
                json.dumps(stop_payload),
                qos=1,
                retain=True,
            )
            stop_config.wait_for_publish(timeout=2.0)
            stop_sync.wait_for_publish(timeout=2.0)
            time.sleep(args.replay_drain_seconds)
            # A restarted supervisor must wait for the next typed START rather
            # than replaying either the old start or the cleanup command.
            client.publish("/replay/config", payload=None, qos=1, retain=True).wait_for_publish(timeout=2.0)
            client.publish("/replay/sync", payload=None, qos=1, retain=True).wait_for_publish(timeout=2.0)
        replay_stop_completed_monotonic = time.monotonic()
        client.disconnect()
        client.loop_stop()
        telemetry_client.disconnect()
        telemetry_client.loop_stop()
        if evaluation_logger is not None:
            evaluation_logger.stop()
        if resource_monitor is not None:
            resource_monitor.stop()
        cleanup_completed_monotonic = time.monotonic()

    terminal = (
        state.progress[-1] if state.progress and state.progress[-1].terminal else None
    )
    detections = tuple(
        detection for progress in state.progress for detection in progress.detections
    )
    detected = bool(detections)
    metrics = None
    predictions: tuple[ComplexEventResult, ...] = ()
    ground_truth: tuple[GroundTruthEvent, ...] = ()
    if experiment is not None:
        recording_started_at = _scenario_start_datetime(args.scenario) + timedelta(
            seconds=args.replay_start
        )
        ground_truth = (
            GroundTruthEvent(
                event_id=experiment.experiment_id,
                event_family=template.family_id,
                start_time=experiment.recording_start,
                end_time=experiment.recording_end,
                deadline=experiment.recording_end
                + timedelta(seconds=args.deadline_seconds),
            ),
        )
        predictions = tuple(
            ComplexEventResult(
                run_id=request_id,
                baseline_id=BaselineId(args.baseline),
                trace_id=experiment.experiment_id,
                request_id=request_id,
                hypothesis_id=detection.hypothesis_id,
                event_time=_event_recording_time(
                    detection.event_end_time,
                    replay_started_at=replay_started_at,
                    recording_started_at=recording_started_at,
                ),
                monotonic_timestamp_ns=0,
                result_id=f"{request_id}:{detection.hypothesis_id}",
                event_family=detection.event_family,
                event_start_time=_event_recording_time(
                    detection.event_start_time,
                    replay_started_at=replay_started_at,
                    recording_started_at=recording_started_at,
                ),
                event_end_time=_event_recording_time(
                    detection.event_end_time,
                    replay_started_at=replay_started_at,
                    recording_started_at=recording_started_at,
                ),
                emitted_at=_recording_time(
                    detection.emitted_at,
                    replay_started_at=replay_started_at,
                    recording_started_at=recording_started_at,
                ),
                bindings=detection.bindings,
            )
            for detection in detections
            if replay_started_at is not None
        )
        metrics = evaluate_event_results(
            ground_truth,
            predictions,
            minimum_temporal_iou=args.minimum_temporal_iou,
            temporal_boundary_tolerance_seconds=args.event_match_tolerance_seconds,
        )
    if state.error:
        outcome_error = state.error
    elif terminal is not None or detected:
        # A completed detection carried by a progress response is sufficient
        # evidence of success even if the redundant terminal-lifecycle message
        # is delayed or lost on a shaped network.
        outcome_error = ""
    elif not state.admitted.is_set():
        outcome_error = "matching seed observation not received before deadline"
    else:
        outcome_error = (
            "terminal CE result not observed after admission before deadline"
        )
    person_gunshot_alignment = []
    if state.audio_recording_samples:
        anchor = state.audio_recording_samples[0]
        anchor_event = datetime.fromisoformat(
            str(anchor["event_time_interval"]["start"]).replace("Z", "+00:00")
        )
        anchor_recording = datetime.fromisoformat(
            str(anchor["recording_time_interval"]["start"]).replace("Z", "+00:00")
        )
        gunshots = [
            (
                sample,
                datetime.fromisoformat(
                    str(sample["recording_time_interval"]["start"]).replace(
                        "Z", "+00:00"
                    )
                ),
            )
            for sample in state.audio_recording_samples
            if sample["label"] == "gunshot"
        ]
        for person in state.recovered_person_samples:
            person_event = datetime.fromisoformat(
                str(person["event_time_interval"]["start"]).replace("Z", "+00:00")
            )
            person_recording = anchor_recording + (person_event - anchor_event)
            nearest = (
                min(
                    gunshots,
                    key=lambda item: abs((person_recording - item[1]).total_seconds()),
                )
                if gunshots
                else None
            )
            person_gunshot_alignment.append(
                {
                    **person,
                    "recording_time": person_recording.isoformat(),
                    "nearest_gunshot_node": (
                        nearest[0]["node_id"] if nearest else None
                    ),
                    "nearest_gunshot_recording_time": (
                        nearest[1].isoformat() if nearest else None
                    ),
                    "signed_person_minus_gunshot_seconds": (
                        round(
                            (person_recording - nearest[1]).total_seconds(),
                            6,
                        )
                        if nearest
                        else None
                    ),
                    "absolute_time_difference_seconds": (
                        round(
                            abs((person_recording - nearest[1]).total_seconds()),
                            6,
                        )
                        if nearest
                        else None
                    ),
                }
            )
    catalog_cross_sensor_span_seconds = _scenario_duration_seconds(args.scenario)
    requested_recording_duration_seconds = (
        max(0.0, args.replay_end - args.replay_start) if args.replay_end >= 0 else None
    )
    labeled_experiment_duration_seconds = (
        round(
            (experiment.recording_end - experiment.recording_start).total_seconds(),
            3,
        )
        if experiment is not None
        else None
    )

    def duration_between(
        start: float | None,
        end: float | None,
    ) -> float | None:
        if start is None or end is None:
            return None
        return round(max(0.0, end - start), 3)

    finished_monotonic = time.monotonic()
    execution_end = state.terminal_at or cleanup_started_monotonic
    replay_execution_seconds = duration_between(
        replay_sync_monotonic,
        execution_end,
    )
    observed_event_time_span_seconds = (
        round(
            (
                state.observed_event_time_end - state.observed_event_time_start
            ).total_seconds(),
            3,
        )
        if state.observed_event_time_start is not None
        and state.observed_event_time_end is not None
        else None
    )
    disturbance_exposure = []
    # A registered seed watch is already an active CE demand.  Requiring
    # admission here erased exposure metadata precisely when a disturbance
    # prevented the seed observation from reaching the orchestrator.
    # The seed watch is the beginning of active demand.  Admission can be
    # reported after a fast terminal transition (especially when a shaped
    # bridge drains queued responses), so using ``admitted_at`` can construct
    # an impossible reversed interval and crash result serialization.
    demand_started_monotonic = replay_sync_monotonic
    demand_phase = (
        "ADMITTED_EXECUTION" if state.admitted_at is not None else "SEED_WATCH"
    )
    if (
        condition_trace_started_monotonic is not None
        and demand_started_monotonic is not None
    ):
        exposure_anchor = (
            condition_trace_anchor_monotonic
            if condition_trace_anchor_monotonic is not None
            else condition_trace_started_monotonic
        )
        demand_start_s = (
            demand_started_monotonic - exposure_anchor
        )
        demand_end_s = execution_end - exposure_anchor
        active_apply = None
        for row in disturbance_results:
            offset = row.get("application_started_offset_s")
            if offset is None:
                offset = row.get("applied_offset_s")
            if row.get("action") in {"APPLY", "FAIL"} and offset is not None:
                active_apply = row
            elif (
                row.get("action") == "RESTORE"
                and offset is not None
                and active_apply is not None
            ):
                disturbance_exposure.append(
                    {
                        "apply_transition_id": active_apply.get("transition_id"),
                        "restore_transition_id": row.get("transition_id"),
                        "disturbance_start_s": (
                            active_apply.get("application_started_offset_s")
                            if active_apply.get("application_started_offset_s") is not None
                            else active_apply["applied_offset_s"]
                        ),
                        "disturbance_end_s": offset,
                        "demand_start_s": round(demand_start_s, 6),
                        "demand_end_s": round(demand_end_s, 6),
                        "demand_phase": demand_phase,
                        "classification": classify_disturbance_exposure(
                            demand_start_s=demand_start_s,
                            demand_end_s=demand_end_s,
                            disturbance_start_s=float(
                                active_apply.get("application_started_offset_s")
                                if active_apply.get("application_started_offset_s") is not None
                                else active_apply["applied_offset_s"]
                            ),
                            disturbance_end_s=float(offset),
                        ),
                    }
                )
                active_apply = None
    adaptation_instrumentation = []
    recorded_plans: list[dict[str, object]] = []
    recorded_provider_commands: list[dict[str, object]] = []
    if record_store is not None:
        recorded_plans = record_store.read("plan_decision")
        recorded_provider_commands = record_store.read("provider_command")
        recorded_observations = record_store.read("predicate_observation")

        def parsed_wall(row: dict[str, object]) -> datetime | None:
            value = row.get("wall_timestamp") or row.get("event_time")
            if not value:
                return None
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return None

        for disturbance in disturbance_results:
            if disturbance.get("action") not in {"APPLY", "FAIL"}:
                continue
            applied_value = disturbance.get("applied_at") or disturbance.get("requested_at")
            try:
                applied_at = datetime.fromisoformat(str(applied_value).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
            condition_epoch = int(disturbance.get("condition_epoch") or 0)
            later_plans = [
                timestamp
                for row in recorded_plans
                if int(row.get("resource_epoch") or 0) >= condition_epoch
                and (timestamp := parsed_wall(row)) is not None
                and timestamp >= applied_at
            ]
            later_outputs = [
                timestamp
                for row in recorded_observations
                if (timestamp := parsed_wall(row)) is not None
                and timestamp >= applied_at
            ]
            exposure = next(
                (
                    row.get("classification")
                    for row in disturbance_exposure
                    if row.get("apply_transition_id") == disturbance.get("transition_id")
                ),
                None,
            )
            adaptation_instrumentation.append(
                {
                    "transition_id": disturbance.get("transition_id"),
                    "condition_epoch": condition_epoch,
                    "action": disturbance.get("action"),
                    "condition": disturbance.get("condition"),
                    "target": disturbance.get("target"),
                    "validated": bool(
                        disturbance.get("validated")
                        or (disturbance.get("response") or {}).get("validated")
                    ),
                    "condition_application_overhead_seconds": disturbance.get("helper_wall_seconds"),
                    "condition_to_first_plan_seconds": (
                        round((min(later_plans) - applied_at).total_seconds(), 6)
                        if later_plans else None
                    ),
                    "condition_to_first_predicate_output_seconds": (
                        round((min(later_outputs) - applied_at).total_seconds(), 6)
                        if later_outputs else None
                    ),
                    "demand_exposure": exposure,
                }
            )
    timing = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "total_wall_seconds": round(finished_monotonic - started, 3),
        "watch_registration_seconds": duration_between(
            started,
            state.watch_registered_at,
        ),
        "replay_configuration_to_sync_seconds": duration_between(
            replay_configured_monotonic,
            replay_sync_monotonic,
        ),
        "sync_to_admission_seconds": duration_between(
            replay_sync_monotonic,
            state.admitted_at,
        ),
        "sync_to_terminal_or_cutoff_seconds": replay_execution_seconds,
        "cleanup_seconds": duration_between(
            cleanup_started_monotonic,
            cleanup_completed_monotonic,
        ),
        "replay_stop_and_drain_seconds": duration_between(
            replay_stop_started_monotonic,
            replay_stop_completed_monotonic,
        ),
        "catalog_cross_sensor_timestamp_span_seconds": round(
            catalog_cross_sensor_span_seconds,
            3,
        ),
        "catalog_cross_sensor_timestamp_span_note": (
            "This is an aggregate timestamp envelope, not media duration. "
            "It may include known sensor clock offsets."
        ),
        "labeled_experiment_duration_seconds": (labeled_experiment_duration_seconds),
        "requested_recording_duration_seconds": (
            round(requested_recording_duration_seconds, 3)
            if requested_recording_duration_seconds is not None
            else None
        ),
        "requested_recording_duration_note": (
            "Null means replay continued to source EOF or the wall-time "
            "cutoff; it must not be inferred from the cross-sensor envelope."
        ),
        "observed_event_time_span_seconds": observed_event_time_span_seconds,
        "observed_event_time_start": (
            state.observed_event_time_start.isoformat()
            if state.observed_event_time_start is not None
            else None
        ),
        "observed_event_time_end": (
            state.observed_event_time_end.isoformat()
            if state.observed_event_time_end is not None
            else None
        ),
        "observed_event_time_to_wall_ratio": (
            round(
                observed_event_time_span_seconds / replay_execution_seconds,
                3,
            )
            if replay_execution_seconds and observed_event_time_span_seconds
            else None
        ),
        "playback_mode": args.playback_mode,
        "configured_playback_speed": args.playback_speed,
        "container_startup_seconds": None,
        "container_startup_note": (
            "The single-run driver does not start containers; stack startup "
            "must be supplied by the enclosing pilot/stack runner."
        ),
    }
    if identity_capture is not None:
        identity_capture.finalize()
    processed_workload = {
        key: {
            field: max(
                0,
                value - state.detector_totals_baseline.get(key, {}).get(field, 0),
            )
            for field, value in totals.items()
        }
        for key, totals in sorted(state.detector_totals_latest.items())
    }
    # Provider commands are the authoritative execution contract.  A plan's
    # activated_provider_keys summarizes the selected alternatives, but the
    # dispatcher may expand those alternatives into required upstream steps
    # (for example both camera YOLO providers needed by a multi-camera robbery
    # frontier).  Comparing workload only with the compact plan summary
    # incorrectly labels those explicitly dispatched dependencies as ambient
    # execution.
    selected_yolo_nodes = sorted(
        {
            str(row.get("node_id"))
            for row in recorded_provider_commands
            if str(row.get("provider_id", "")).startswith("yolo_")
            and row.get("node_id")
        }
    )
    observed_yolo_nodes = sorted(
        {
            key.removeprefix("yolo:")
            for key in processed_workload
            if key.startswith("yolo:")
        }
    )
    zero_input_yolo_nodes = sorted(
        key.removeprefix("yolo:")
        for key, counters in processed_workload.items()
        if key.startswith("yolo:") and int(counters.get("frames", 0)) == 0
    )
    selected_yolo_processing = {
        node: {
            "status_observed": f"yolo:{node}" in processed_workload,
            "frames": int(
                processed_workload.get(f"yolo:{node}", {}).get("frames", 0)
            ),
            "detections": int(
                processed_workload.get(f"yolo:{node}", {}).get("detections", 0)
            ),
            "dropped_frames": int(
                processed_workload.get(f"yolo:{node}", {}).get("dropped_frames", 0)
            ),
        }
        for node in selected_yolo_nodes
    }
    execution_conformance = {
        "schema_version": "fable.execution_conformance.v2",
        # A selected watch provider cannot process frames until replay has
        # acknowledged the synchronization edge. Do not replace the real
        # replay/seed failure with a secondary placement mismatch.
        "applicable": bool(
            selected_yolo_nodes and replay_sync_monotonic is not None
        ),
        "replay_synchronized": replay_sync_monotonic is not None,
        "selection_source": "provider_command",
        "selected_yolo_nodes": selected_yolo_nodes,
        "observed_yolo_nodes": observed_yolo_nodes,
        "selected_yolo_processing": selected_yolo_processing,
        "selected_yolo_nodes_with_zero_input_frames": sorted(
            set(selected_yolo_nodes) & set(zero_input_yolo_nodes)
        ),
        "missing_selected_yolo_nodes": sorted(
            set(selected_yolo_nodes) - set(observed_yolo_nodes)
        ),
        "unexpected_active_yolo_nodes": sorted(
            set(observed_yolo_nodes) - set(selected_yolo_nodes)
        ),
    }
    execution_conformance["valid"] = bool(
        not execution_conformance["applicable"]
        or (
            not execution_conformance["missing_selected_yolo_nodes"]
            and not execution_conformance["selected_yolo_nodes_with_zero_input_frames"]
            and not execution_conformance["unexpected_active_yolo_nodes"]
        )
    )
    summary = {
        "schema_version": "fable.replay_accuracy_run.v2",
        "provenance": build_run_provenance(
            ROOT,
            runner_arguments=vars(args),
        ),
        "request_id": request_id,
        "experiment_id": args.experiment_id,
        "campaign_year": (experiment.campaign_year if experiment is not None else None),
        "experiment_label": (
            f"{experiment.campaign_year} | {experiment.ce_variant} | "
            f"{experiment.experiment_id}"
            if experiment is not None
            else f"unlabeled | {args.variant} | {args.scenario}"
        ),
        "scenario": args.scenario,
        "variant": args.variant,
        "family_id": template.family_id,
        "baseline": args.baseline,
        "network_disturbance": args.network_disturbance,
        "disturbance_results": disturbance_results,
        "condition_trace": (
            condition_trace.model_dump(mode="json")
            if condition_trace is not None
            else None
        ),
        "condition_trace_started_at": (
            condition_trace_started_at.isoformat()
            if condition_trace_started_at is not None
            else None
        ),
        "condition_trace_anchor": (
            condition_trace.anchor.value if condition_trace is not None else None
        ),
        "condition_trace_anchor_at": (
            condition_trace_anchor_at.isoformat()
            if condition_trace_anchor_at is not None
            else None
        ),
        "requested_ce_start_offset_seconds": args.ce_start_offset_seconds,
        "actual_ce_start_offset_seconds": actual_ce_start_offset_s,
        "disturbance_exposure": disturbance_exposure,
        "adaptation_instrumentation": adaptation_instrumentation,
        "execution_conformance": execution_conformance,
        "playback_mode": args.playback_mode,
        "playback_speed": args.playback_speed,
        "ground_truth_positive": experiment is not None,
        "event_matching_policy": {
            "minimum_temporal_iou": args.minimum_temporal_iou,
            "allow_contained_predictions": True,
            "temporal_boundary_tolerance_seconds": args.event_match_tolerance_seconds,
            "raw_timestamps_preserved": True,
        },
        "detected": detected,
        "classification": (
            (
                "TRUE_POSITIVE"
                if metrics is not None and metrics.true_positives
                else "FALSE_NEGATIVE"
            )
            if experiment is not None
            else ("DETECTED" if detected else "NOT_DETECTED")
        ),
        "metrics": metrics.model_dump(mode="json") if metrics is not None else None,
        "deadline_missed": bool(metrics is not None and metrics.timely_recall < 1.0),
        "ground_truth": [item.model_dump(mode="json") for item in ground_truth],
        "predictions": [item.model_dump(mode="json") for item in predictions],
        "watch_registered": state.watching.is_set(),
        "admitted": state.admitted.is_set(),
        "seed_diagnostics": [
            response.model_dump(mode="json") for response in state.seed_responses
        ],
        "terminal": terminal is not None,
        "terminal_lifecycles": (
            terminal.terminal_lifecycles if terminal is not None else {}
        ),
        "progress_messages": len(state.progress),
        "progress_statuses": dict(sorted(state.progress_status_counts.items())),
        "progress_diagnostics": [
            progress.model_dump(mode="json") for progress in state.progress
        ],
        "cleanup": (
            state.cancellation.model_dump(mode="json")
            if state.cancellation is not None
            else {
                "status": "NO_RESPONSE",
                "reason": "cancellation response not received within 5 seconds",
            }
        ),
        "observed_messages": {
            "audio_events": state.audio_event_messages,
            "interaction_predicates": state.interaction_predicate_messages,
            "identity_association_maps": state.identity_association_messages,
            "vehicle_predicates": state.vehicle_predicate_messages,
            "yolo": state.yolo_messages,
            "context_tracks": state.context_track_messages,
        },
        "processed_workload": processed_workload,
        "detector_sampling_diagnostics": dict(
            sorted(state.detector_sampling_diagnostics.items())
        ),
        "yolo_classes": dict(sorted(state.yolo_class_counts.items())),
        "context_track_classes": dict(sorted(state.context_track_class_counts.items())),
        "vehicle_predicates_by_id": dict(
            sorted(state.vehicle_predicate_counts.items())
        ),
        "vehicle_predicates_by_source": dict(
            sorted(state.vehicle_predicate_source_counts.items())
        ),
        "vehicle_predicate_samples": state.vehicle_predicate_samples,
        "audio_events_by_label": dict(sorted(state.audio_event_counts.items())),
        "audio_events_by_node": {
            node_id: dict(sorted(counts.items()))
            for node_id, counts in sorted(state.audio_events_by_node.items())
        },
        "audio_event_time_ranges": dict(sorted(state.audio_event_time_ranges.items())),
        "audio_recording_samples": state.audio_recording_samples,
        "recovered_person_samples": state.recovered_person_samples,
        "recovered_vehicle_samples": state.recovered_vehicle_samples,
        "person_gunshot_recording_time_alignment": person_gunshot_alignment,
        "interaction_predicates_by_id": dict(
            sorted(state.interaction_predicate_counts.items())
        ),
        "interaction_predicate_samples": state.interaction_predicate_samples,
        "identity_associations_by_kind": dict(
            sorted(state.identity_associations_by_kind.items())
        ),
        "identity_association_samples": state.identity_association_samples,
        "retrospective_replay_statuses": state.retrospective_replay_statuses,
        "readiness_services": sorted(state.readiness),
        "ready_replay_nodes": sorted(ready_replay_nodes),
        "readiness_by_node": {
            node_id: sorted(services)
            for node_id, services in sorted(state.readiness_by_node.items())
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "common_record_dir": (
            str(common_record_dir) if common_record_dir is not None else None
        ),
        "resource_instrumentation": (
            resource_monitor.summary() if resource_monitor is not None else None
        ),
        "timing": timing,
        "error": outcome_error,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not args.quiet:
        print(rendered, end="")
    return (
        0
        if (
            (metrics is not None and metrics.true_positives > 0)
            or (metrics is None and detected)
        )
        else 4
    )


if __name__ == "__main__":
    raise SystemExit(main())
