"""Adapters for provider-native replay outputs used by the IoBT testbed.

These functions deliberately live outside ``fable.distributed``: they know
about concrete vehicle/multimodal provider schemas, while the node agent only
knows how to route canonical predicate evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import Any

from fable.common.ids import occurrence_anchor_id
from fable.common.time import EventTimeInterval, ensure_utc, utc_now
from fable.distributed.models import ActivateProviderCommand, ReplayOutputAdapter
from fable.distributed.output_adapters import (
    AdaptedProviderEvidence,
    ProviderOutputAdapterRegistry,
)

LOGGER = logging.getLogger(__name__)

_SYMBOLIC_CAMERA_REFERENCES = frozenset(
    {
        "chase_gate",
        "convoy_gate",
        "convergence_gate",
        "rendezvous_gate",
        "route_gate",
        "visit_reference",
    }
)


def _compatible_bound_value(expected: str, actual: str) -> bool:
    """Match authored camera roles to concrete uncalibrated camera FOVs."""

    return expected == actual or (
        expected in _SYMBOLIC_CAMERA_REFERENCES
        and actual.startswith("camera_fov:")
    )


def build_replay_output_adapter_registry() -> ProviderOutputAdapterRegistry:
    registry = ProviderOutputAdapterRegistry()
    registry.register(ReplayOutputAdapter.AUDIO_DETECTION, _adapt_audio_detection)
    registry.register(ReplayOutputAdapter.VEHICLE_PREDICATE, _adapt_vehicle_predicate)
    registry.register(ReplayOutputAdapter.MULTIMODAL_PREDICATE, _adapt_multimodal_predicate)
    registry.register(ReplayOutputAdapter.YOLO_OBJECT_PRESENT, _adapt_yolo_object_present)
    registry.register(ReplayOutputAdapter.IDENTITY_ASSOCIATION, _adapt_identity_association)
    return registry


def _adapt_audio_detection(
    command: ActivateProviderCommand,
    document: Any,
) -> AdaptedProviderEvidence | None:
    if not isinstance(document, dict):
        return None
    demand = command.demand
    source_id = demand.eligible_source_ids[0] if demand.eligible_source_ids else command.node_id
    aliases = command.runtime.output_label_aliases
    event_name = str(document.get("event") or document.get("label") or "")
    requested = str(demand.semantic_predicate.parameters.get("label") or "any")
    accepted = {event_name, *aliases.get(event_name, ())}
    if requested not in ("any", "*") and requested not in accepted:
        return None
    timestamp = payload_event_time(document.get("t"))
    interval = instant_interval(timestamp)
    if not demand.event_time_interval.overlaps(interval):
        return None
    confidence = _clamp_confidence(document.get("confidence", 1.0))
    if confidence < float(demand.semantic_predicate.parameters.get("minimum_confidence", 0.0)):
        return None
    occurrence = occurrence_anchor_id(
        source_id,
        demand.semantic_predicate.predicate_id,
        timestamp,
        {**demand.bound_roles, "event": event_name},
    )
    introduced: dict[str, str] = {}
    if "location" in demand.unbound_roles:
        # Audio classifiers do not perform geometric localization.  Their
        # source is nevertheless the concrete observation location and is the
        # correct binding for source-scoped trigger graphs.
        introduced[_binding_variable(demand, "location")] = source_id
    return AdaptedProviderEvidence(occurrence, interval, introduced, confidence)


def _adapt_vehicle_predicate(
    command: ActivateProviderCommand,
    document: Any,
) -> AdaptedProviderEvidence | None:
    if not isinstance(document, dict):
        return None
    demand = command.demand
    try:
        from providers.vehicle.models import PredicateObservation

        observation = PredicateObservation.model_validate(document)
    except Exception:
        LOGGER.debug("ignored invalid vehicle predicate payload")
        return None
    if observation.predicate_id != demand.semantic_predicate.predicate_id:
        return None
    if not demand.event_time_interval.overlaps(observation.event_time_interval):
        return None
    for role, entity_id in demand.bound_roles.items():
        if role in observation.bindings and not _compatible_bound_value(
            entity_id, observation.bindings[role]
        ):
            return None
    introduced = {
        _binding_variable(demand, role): entity_id
        for role, entity_id in observation.bindings.items()
        if role in demand.unbound_roles
    }
    if demand.unbound_roles and not introduced:
        return None
    return AdaptedProviderEvidence(
        observation.occurrence_id,
        observation.event_time_interval,
        introduced,
        observation.confidence,
        tuple(observation.source_ids),
    )


def _adapt_multimodal_predicate(
    command: ActivateProviderCommand,
    document: Any,
) -> AdaptedProviderEvidence | None:
    if not isinstance(document, dict):
        return None
    demand = command.demand
    schema_version = str(document.get("schema_version") or "")
    if schema_version == "audio_event_observation.v1":
        try:
            from providers.multimodal.models import AudioEventObservation

            observation = AudioEventObservation.model_validate(document)
        except Exception:
            LOGGER.debug("ignored invalid typed audio-event payload")
            return None
        if demand.semantic_predicate.predicate_id != "AUDIO_EVENT":
            return None
        requested = str(demand.semantic_predicate.parameters.get("label") or "any")
        if requested not in ("any", "*") and requested != observation.label:
            return None
        if not demand.event_time_interval.overlaps(observation.event_time_interval):
            return None
        if observation.confidence < float(
            demand.semantic_predicate.parameters.get("minimum_confidence", 0.0)
        ):
            return None
        observed_location = observation.localized_zone_id or observation.source_id
        bound_location = demand.bound_roles.get("location")
        if bound_location is not None and bound_location != observed_location:
            return None
        introduced: dict[str, str] = {}
        if "location" in demand.unbound_roles:
            introduced[_binding_variable(demand, "location")] = observed_location
        return AdaptedProviderEvidence(
            observation.occurrence_id,
            observation.event_time_interval,
            introduced,
            observation.confidence,
        )
    if schema_version == "interaction_predicate_observation.v1":
        try:
            from providers.multimodal.models import InteractionPredicateObservation

            observation = InteractionPredicateObservation.model_validate(document)
        except Exception:
            LOGGER.debug("ignored invalid interaction predicate payload")
            return None
        if observation.predicate_id != demand.semantic_predicate.predicate_id:
            return None
        if not observation.truth:
            return None
        if not demand.event_time_interval.overlaps(observation.event_time_interval):
            return None
        for role, entity_id in demand.bound_roles.items():
            if role in observation.bindings and observation.bindings[role] != entity_id:
                return None
        introduced = {
            _binding_variable(demand, role): entity_id
            for role, entity_id in observation.bindings.items()
            if role in demand.unbound_roles
        }
        if demand.unbound_roles and not introduced:
            return None
        return AdaptedProviderEvidence(
            observation.occurrence_id,
            observation.event_time_interval,
            introduced,
            observation.confidence,
        )
    return None


def _adapt_yolo_object_present(
    command: ActivateProviderCommand,
    document: Any,
) -> AdaptedProviderEvidence | None:
    demand = command.demand
    # The catalog contract for YOLO is detection_set.v1 / OBJECT_PRESENT.
    # YOLO is only an intermediate stage in behavioral and identity chains;
    # a raw box must never be relabelled as their terminal predicate.
    if demand.semantic_predicate.predicate_id != "OBJECT_PRESENT":
        return None
    source_id = demand.eligible_source_ids[0] if demand.eligible_source_ids else command.node_id
    rows = document if isinstance(document, list) else [document]
    rows = [row for row in rows if isinstance(row, dict)]
    requested_raw = demand.semantic_predicate.parameters.get(
        "class_allowlist",
        demand.semantic_predicate.parameters.get("class", ()),
    )
    if isinstance(requested_raw, str):
        requested = {requested_raw}
    else:
        requested = {str(item) for item in requested_raw or ()}
    matching = [
        row
        for row in rows
        if not requested or str(row.get("class") or row.get("label")) in requested
    ]
    if not matching:
        return None
    row = max(matching, key=lambda item: float(item.get("conf", 0.0)))
    timestamp = payload_event_time(row.get("t"))
    interval = instant_interval(timestamp)
    if not demand.event_time_interval.overlaps(interval):
        return None
    object_label = str(row.get("class") or row.get("label") or "object")
    object_id = str(
        row.get("track_id")
        or row.get("id")
        or occurrence_anchor_id(
            source_id,
            f"object:{object_label}",
            timestamp,
            {"box": row.get("box", [])},
        )
    )
    introduced: dict[str, str] = {}
    if demand.unbound_roles:
        role = demand.unbound_roles[0]
        introduced[_binding_variable(demand, role)] = object_id
    occurrence = occurrence_anchor_id(
        source_id,
        demand.semantic_predicate.predicate_id,
        timestamp,
        {**demand.bound_roles, **introduced, "class": object_label},
    )
    return AdaptedProviderEvidence(
        occurrence,
        interval,
        introduced,
        _clamp_confidence(row.get("conf", 1.0)),
    )


def _adapt_identity_association(
    command: ActivateProviderCommand,
    document: Any,
) -> AdaptedProviderEvidence | None:
    demand = command.demand
    if demand.semantic_predicate.predicate_id != "SAME_ENTITY" or not isinstance(document, dict):
        return None
    rows = document.get("associations")
    candidates = [row for row in rows or () if isinstance(row, dict)]
    bound_left = demand.bound_roles.get("left")
    bound_right = demand.bound_roles.get("right")
    if bound_left and bound_right:
        candidates = [
            row
            for row in candidates
            if {
                str(row.get("left_local_entity_id") or ""),
                str(row.get("right_local_entity_id") or ""),
            }
            == {bound_left, bound_right}
        ]
    if not candidates:
        return None
    row = max(candidates, key=lambda item: float(item.get("confidence", 0.0)))
    confidence = _clamp_confidence(row.get("confidence", 0.0))
    if confidence < float(demand.semantic_predicate.parameters.get("minimum_confidence", 0.0)):
        return None
    canonical = str(row.get("canonical_entity_id") or "")
    if not canonical:
        return None
    try:
        interval = EventTimeInterval.model_validate(document.get("event_time_interval") or {})
    except Exception:
        return None
    # A fully bound SAME_ENTITY demand is commonly created only after a later
    # graph checkpoint exposes the identity question. Its bounded descriptor
    # crops are therefore retrospective and legitimately predate the demand's
    # live frontier interval. The exact endpoint filter above is the scope and
    # safety boundary in that case. Keep the ordinary overlap requirement for
    # open-ended association demands.
    exact_pair_demand = bool(bound_left and bound_right)
    if not exact_pair_demand and not demand.event_time_interval.overlaps(interval):
        return None
    introduced = {
        _binding_variable(demand, role): canonical
        for role in demand.unbound_roles
    }
    occurrence = occurrence_anchor_id(
        str(document.get("right_source_id") or document.get("left_source_id") or command.node_id),
        "SAME_ENTITY",
        interval.end,
        {
            **demand.bound_roles,
            **introduced,
            "left": row.get("left_local_entity_id"),
            "right": row.get("right_local_entity_id"),
            "canonical": canonical,
        },
    )
    return AdaptedProviderEvidence(occurrence, interval, introduced, confidence)


def payload_event_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1e12:
            numeric /= 1e6
        return datetime.fromtimestamp(numeric, tz=UTC)
    if value is None:
        return utc_now()
    text = str(value).strip()
    try:
        numeric = float(text)
        if numeric > 1e12:
            numeric /= 1e6
        return datetime.fromtimestamp(numeric, tz=UTC)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        pass
    try:
        return datetime.strptime(text, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)
    except ValueError:
        return utc_now()


def instant_interval(timestamp: datetime) -> EventTimeInterval:
    return EventTimeInterval(start=timestamp, end=timestamp)


def _clamp_confidence(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def _binding_variable(demand: Any, role_name: str) -> str:
    """Translate a provider-facing predicate role into its graph variable."""

    for role in demand.semantic_predicate.roles:
        if role.role_name == role_name:
            return role.variable
    return role_name
