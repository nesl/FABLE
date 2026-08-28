"""Live MQTT aggregation for person and vehicle cross-sensor identity.

Sensor-local ReID producers publish ``DescriptorSet`` payloads.  This service
matches only compatible, calibrated feature spaces and emits canonical maps.
It deliberately fails closed when checkpoints, entity kinds, or event-time
windows differ.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None  # type: ignore[assignment]

from fable.common.ids import deterministic_id
from fable.common.time import EventTimeInterval

from .association import CrossSensorIdentityAssociator, cosine_distance
from .errors import ArtifactCompatibilityError, InvalidProviderInput
from .models import (
    DescriptorSet,
    EntityAssociation,
    EntityAssociationSet,
    IdentityComparisonCancellation,
    IdentityComparisonDemand,
)
from .vlm_reid import OpenAIVisionIdentityComparator, RemoteVisionIdentityComparator

LOGGER = logging.getLogger(__name__)

LIVE_ESCALATION_POLICIES = {
    "C0_CHEAP_ONLY",
    "C1_STRONG_ONLY",
    "C2_FIXED_CASCADE",
    "C3_FABLE_ESCALATION",
    "C4_FABLE_NO_ESCALATION",
}


@dataclass(frozen=True)
class IdentityServiceConfig:
    input_topic: str = "/+/fable/identity/descriptors"
    output_topic: str = "/fable/identity/associations"
    readiness_topic: str = "/readiness/x86server/fable_identity"
    maximum_event_time_gap_s: float = 30.0
    maximum_cosine_distance: float = 0.25
    same_camera_maximum_event_time_gap_s: float | None = None
    same_camera_maximum_cosine_distance: float | None = None
    same_camera_history_snapshots: int = 512
    vlm_fallback_enabled: bool = False
    vlm_maximum_calls_per_replay: int = 10
    vlm_minimum_confidence: float = 0.5
    vlm_candidate_maximum_cosine_distance: float = 0.8
    escalation_policy_id: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.vlm_maximum_calls_per_replay <= 10:
            raise ValueError("VLM ReID budget must be between zero and ten calls")
        if not 0 <= self.vlm_minimum_confidence <= 1:
            raise ValueError("VLM ReID confidence threshold must be in [0, 1]")
        if not 0 <= self.vlm_candidate_maximum_cosine_distance <= 2:
            raise ValueError("VLM ReID candidate distance must be in [0, 2]")
        if self.escalation_policy_id and self.escalation_policy_id not in LIVE_ESCALATION_POLICIES:
            raise ValueError("unsupported live identity escalation policy")


class IdentityAssociationProcessor:
    def __init__(
        self,
        config: IdentityServiceConfig,
        *,
        vlm_comparator: Any | None = None,
    ) -> None:
        self.config = config
        self.associator = CrossSensorIdentityAssociator(
            maximum_cosine_distance=config.maximum_cosine_distance,
            require_identity_calibration=True,
        )
        self.same_camera_associator = CrossSensorIdentityAssociator(
            maximum_cosine_distance=(
                config.same_camera_maximum_cosine_distance
                if config.same_camera_maximum_cosine_distance is not None
                else config.maximum_cosine_distance
            ),
            require_identity_calibration=True,
        )
        self._latest: dict[tuple[str, str, tuple[object, ...]], DescriptorSet] = {}
        self._history: dict[
            tuple[str, str, tuple[object, ...]], deque[DescriptorSet]
        ] = defaultdict(
            lambda: deque(maxlen=config.same_camera_history_snapshots)
        )
        # Tracker-local identifiers are only unique within an entity kind.
        # Person and vehicle trackers can legitimately emit the same scoped
        # suffix, so every identity registry must include the kind.
        self._canonical_by_local: dict[tuple[str, str, str], str] = {}
        self._published_pairs: set[
            tuple[str, tuple[str, str], tuple[str, str]]
        ] = set()
        self._vlm_attempted_pairs: set[
            tuple[str, tuple[str, str], tuple[str, str]]
        ] = set()
        self._vlm_calls = 0
        self._preferred_pairs: set[
            tuple[str, tuple[str, str], tuple[str, str]]
        ] = set()
        self._pair_by_demand_id: dict[
            str, tuple[str, tuple[str, str], tuple[str, str]]
        ] = {}
        self._vlm_comparator = vlm_comparator
        self._replay_id: str | None = None

    def reset(self, replay_id: str | None = None) -> bool:
        if replay_id and replay_id == self._replay_id:
            return False
        self._latest.clear()
        self._history.clear()
        self._canonical_by_local.clear()
        self._published_pairs.clear()
        self._vlm_attempted_pairs.clear()
        self._vlm_calls = 0
        self._preferred_pairs.clear()
        self._pair_by_demand_id.clear()
        self._replay_id = replay_id
        if self._vlm_comparator is not None and hasattr(
            self._vlm_comparator,
            "set_run_id",
        ):
            self._vlm_comparator.set_run_id(replay_id)
        return True

    def register_demand(self, demand: IdentityComparisonDemand) -> bool:
        pair = _identity_pair(
            demand.entity_kind,
            (_source_from_entity_id(demand.left_local_entity_id), demand.left_local_entity_id),
            (_source_from_entity_id(demand.right_local_entity_id), demand.right_local_entity_id),
        )
        before = len(self._preferred_pairs)
        self._preferred_pairs.add(pair)
        self._pair_by_demand_id[demand.demand_id] = pair
        return len(self._preferred_pairs) != before

    def cancel_demand(self, demand_id: str) -> bool:
        """Forget an exact pair when its last semantic demand is cancelled."""

        pair = self._pair_by_demand_id.pop(demand_id, None)
        if pair is None:
            return False
        if pair not in self._pair_by_demand_id.values():
            self._preferred_pairs.discard(pair)
        return True

    def resolve_demand(
        self, demand: IdentityComparisonDemand
    ) -> tuple[EntityAssociationSet, ...]:
        """Reconsider retained descriptors for an exact semantic identity pair.

        Identity demands are commonly emitted only after a bound predicate sees
        a replacement tracker ID.  Both descriptors can therefore predate the
        demand.  Merely recording the preferred pair leaves that request stuck
        until another frame happens to arrive; replay the newest retained set
        containing either endpoint so the existing bounded matcher/VLM cascade
        can answer immediately.
        """

        self.register_demand(demand)
        wanted = {
            demand.left_local_entity_id,
            demand.right_local_entity_id,
        }
        retained = [
            snapshot
            for snapshots in self._history.values()
            for snapshot in snapshots
        ] + list(self._latest.values())
        matching = [
            snapshot
            for snapshot in retained
            if any(row.local_entity_id in wanted for row in snapshot.records)
        ]
        endpoint_records = {
            entity_id: [
                (snapshot, row)
                for snapshot in matching
                for row in snapshot.records
                if row.local_entity_id == entity_id
            ]
            for entity_id in wanted
        }
        left_rows = endpoint_records.get(demand.left_local_entity_id, [])
        right_rows = endpoint_records.get(demand.right_local_entity_id, [])
        comparisons: list[dict[str, Any]] = []
        for left_snapshot, left_record in left_rows:
            for right_snapshot, right_record in right_rows:
                compatible = (
                    left_snapshot.compatibility_key
                    == right_snapshot.compatibility_key
                )
                distance: float | None = None
                if compatible:
                    try:
                        distance = cosine_distance(
                            left_record.vector,
                            right_record.vector,
                        )
                    except (ArtifactCompatibilityError, InvalidProviderInput):
                        distance = None
                comparisons.append(
                    {
                        "compatible": compatible,
                        "cosine_distance": distance,
                        "event_gap_s": _interval_gap(
                            left_snapshot, right_snapshot
                        ).total_seconds(),
                    }
                )
        LOGGER.info(
            "identity demand diagnostics demand_id=%s left_sets=%s "
            "right_sets=%s comparisons=%s",
            demand.demand_id,
            len(left_rows),
            len(right_rows),
            json.dumps(comparisons, separators=(",", ":"), sort_keys=True),
        )
        if len(
            {
                row.local_entity_id
                for snapshot in matching
                for row in snapshot.records
                if row.local_entity_id in wanted
            }
        ) < 2:
            return ()
        newest = max(matching, key=lambda row: row.event_time_interval.end)
        return self.update(newest)

    def update(self, current: DescriptorSet) -> tuple[EntityAssociationSet, ...]:
        key = (current.source_id, current.entity_kind, current.compatibility_key)
        outputs: list[EntityAssociationSet] = []
        candidates: list[
            tuple[str, str, tuple[object, ...], DescriptorSet]
        ] = []
        for historical_key, snapshots in tuple(self._history.items()):
            candidates.extend((*historical_key, snapshot) for snapshot in snapshots)
        candidates.extend((*historical_key, snapshot) for historical_key, snapshot in self._latest.items())
        for source_id, entity_kind, compatibility, previous in candidates:
            if entity_kind != current.entity_kind:
                continue
            same_camera = source_id == current.source_id
            # Two different tracks visible at the same camera at the same
            # event time are negative identity evidence. Older code compared
            # overlapping snapshots and could collapse an entire vehicle
            # group into one identity merely because their descriptors were
            # visually similar.
            if (
                same_camera
                and previous.event_time_interval.overlaps(
                    current.event_time_interval
                )
            ):
                continue
            maximum_gap = timedelta(
                seconds=(
                    (
                        self.config.same_camera_maximum_event_time_gap_s
                        if self.config.same_camera_maximum_event_time_gap_s
                        is not None
                        else self.config.maximum_event_time_gap_s
                    )
                    if same_camera
                    else self.config.maximum_event_time_gap_s
                )
            )
            gap = _interval_gap(previous, current)
            demanded_pair_present = self._contains_preferred_pair(
                previous, current
            )
            # The generic gap bounds opportunistic streaming association. An
            # exact semantic demand is already bounded by its graph/event-time
            # window and may deliberately compare retrospective evidence with
            # a later observation (for example, a pre-alarm vehicle with a
            # post-alarm exit). Do not discard that exact pair before the
            # calibrated ReID/VLM cascade can evaluate it.
            if gap > maximum_gap and not demanded_pair_present:
                continue
            compatible_features = compatibility == current.compatibility_key
            if compatible_features:
                associated = (
                    self.same_camera_associator
                    if same_camera
                    else self.associator
                ).associate(previous, current)
            elif same_camera:
                # Local continuity is handled by the tracking processor; do
                # not compare incompatible embedding spaces numerically.
                continue
            else:
                # A bounded VLM comparison is meaningful across cameras even
                # when those cameras run different ReID models. Construct an
                # empty typed association result so the existing fail-closed,
                # budgeted VLM fallback can consider the full annotated frame.
                associated = EntityAssociationSet(
                    left_source_id=previous.source_id,
                    right_source_id=current.source_id,
                    event_time_interval=EventTimeInterval(
                        start=min(
                            previous.event_time_interval.start,
                            current.event_time_interval.start,
                        ),
                        end=max(
                            previous.event_time_interval.end,
                            current.event_time_interval.end,
                        ),
                    ),
                    entity_kind=current.entity_kind,
                    feature_space_key=current.compatibility_key,
                    associations=(),
                    unmatched_left=tuple(
                        row.local_entity_id for row in previous.records
                    ),
                    unmatched_right=tuple(
                        row.local_entity_id for row in current.records
                    ),
                )
            if same_camera:
                associated = associated.model_copy(
                    update={
                        "associations": tuple(
                            row
                            for row in associated.associations
                            if row.left_local_entity_id != row.right_local_entity_id
                        )
                    }
                )
            # A local ReID row can satisfy the model's permissive candidate
            # distance while still falling below the semantic graph's accepted
            # confidence floor. For an exact graph-requested pair, treating
            # that weak row as terminal suppresses the configured VLM cascade
            # and guarantees that the output adapter will reject it. Mark only
            # that bounded ambiguous pair unmatched so the normal, budgeted VLM
            # path can decide it. Opportunistic associations remain unchanged.
            if (
                demanded_pair_present
                and associated.associations
                and self._vlm_allowed_for_policy()
                and max(row.confidence for row in associated.associations)
                < self.config.vlm_minimum_confidence
            ):
                associated = associated.model_copy(
                    update={
                        "associations": (),
                        "unmatched_left": tuple(
                            row.local_entity_id for row in previous.records
                        ),
                        "unmatched_right": tuple(
                            row.local_entity_id for row in current.records
                        ),
                    }
                )
            if not same_camera:
                associated = self._apply_escalation_policy(previous, current, associated)
                if (
                    not associated.associations
                    and self._vlm_allowed_for_policy()
                ):
                    associated = self._vlm_fallback(
                        previous, current, associated
                    )
            elif (
                not associated.associations
                and self._vlm_allowed_for_policy()
                and self._preferred_pairs
            ):
                # Same-camera tracker fragmentation is precisely the case that
                # creates a bound historical ID followed by a new EXITS ID.
                # Keep broad same-camera VLM comparisons disabled, but permit
                # the bounded hosted fallback for an exact graph-requested pair.
                associated = self._vlm_fallback(previous, current, associated)
            associated = associated.model_copy(
                update={
                    "associations": tuple(
                        row
                        for row in associated.associations
                        if self._association_pair(associated, row)
                        not in self._published_pairs
                    )
                }
            )
            if not associated.associations:
                continue
            self._published_pairs.update(
                self._association_pair(associated, row)
                for row in associated.associations
            )
            outputs.append(self._canonicalize(associated))
        previous = self._latest.get(key)
        if previous is not None and {
            row.local_entity_id for row in previous.records
        } != {row.local_entity_id for row in current.records}:
            self._history[key].append(previous)
        self._latest[key] = current
        return tuple(outputs)

    def _vlm_allowed_for_policy(self) -> bool:
        return self.config.escalation_policy_id not in {
            "C0_CHEAP_ONLY",
            "C4_FABLE_NO_ESCALATION",
        }

    def _contains_preferred_pair(
        self,
        left: DescriptorSet,
        right: DescriptorSet,
    ) -> bool:
        return any(
            _identity_pair(
                left.entity_kind,
                (left.source_id, left_record.local_entity_id),
                (right.source_id, right_record.local_entity_id),
            )
            in self._preferred_pairs
            for left_record in left.records
            for right_record in right.records
        )

    def _apply_escalation_policy(
        self,
        left: DescriptorSet,
        right: DescriptorSet,
        associated: EntityAssociationSet,
    ) -> EntityAssociationSet:
        """Map controlled E4 policies onto the live local-ReID/VLM cascade."""

        policy = self.config.escalation_policy_id
        if not policy or policy in {
            "C0_CHEAP_ONLY",
            "C2_FIXED_CASCADE",
            "C4_FABLE_NO_ESCALATION",
        }:
            return associated
        if policy == "C3_FABLE_ESCALATION" and associated.associations:
            # Treat weak local bindings as ambiguous rather than terminal. The
            # hosted tier is then invoked under the existing per-replay budget.
            if max(row.confidence for row in associated.associations) >= self.config.vlm_minimum_confidence:
                return associated
        # Strong-only always bypasses local decisions; FABLE reaches this path
        # only for missing/ambiguous local evidence.
        return associated.model_copy(
            update={
                "associations": (),
                "unmatched_left": tuple(row.local_entity_id for row in left.records),
                "unmatched_right": tuple(row.local_entity_id for row in right.records),
            }
        )

    @property
    def vlm_calls(self) -> int:
        return self._vlm_calls

    @property
    def vlm_available(self) -> bool:
        return self.config.vlm_fallback_enabled and self._vlm_comparator is not None

    def _vlm_fallback(
        self,
        left: DescriptorSet,
        right: DescriptorSet,
        empty: EntityAssociationSet,
    ) -> EntityAssociationSet:
        if (
            not self.config.vlm_fallback_enabled
            or self._vlm_comparator is None
            or self._vlm_calls >= self.config.vlm_maximum_calls_per_replay
        ):
            return empty
        candidates = []
        for left_record in left.records:
            left_images = (
                left_record.source_context_image_data_urls
                or left_record.source_crop_data_urls
            )
            if not left_images:
                continue
            for right_record in right.records:
                # Repeated descriptor snapshots commonly contain the same
                # sensor-local track.  That is already one identity by
                # construction, so asking the VLM to compare it with itself
                # cannot create a useful association and wastes the bounded
                # external-call budget.
                if (
                    left.source_id == right.source_id
                    and left_record.local_entity_id
                    == right_record.local_entity_id
                ):
                    continue
                right_images = (
                    right_record.source_context_image_data_urls
                    or right_record.source_crop_data_urls
                )
                if not right_images:
                    continue
                pair = _identity_pair(
                    left.entity_kind,
                    (left.source_id, left_record.local_entity_id),
                    (right.source_id, right_record.local_entity_id),
                )
                if pair in self._published_pairs or pair in self._vlm_attempted_pairs:
                    continue
                # Once the semantic graph has named exact identities, the
                # bounded hosted budget belongs to those questions. Do not
                # consume it on unrelated descriptors that merely arrived
                # first during fan-out or raw replay.
                if self._preferred_pairs and pair not in self._preferred_pairs:
                    continue
                try:
                    distance = cosine_distance(
                        left_record.vector,
                        right_record.vector,
                    )
                except (ArtifactCompatibilityError, InvalidProviderInput):
                    # VLM consumes the annotated full images, not these
                    # incompatible vectors. Keep such candidates eligible but
                    # rank calibrated-vector candidates ahead of them.
                    distance = self.config.vlm_candidate_maximum_cosine_distance
                if distance > self.config.vlm_candidate_maximum_cosine_distance:
                    continue
                # Prefer a visually usable pair before using embedding distance.
                # This matters especially at image edges, where a tiny clipped
                # detection can have a deceptively favorable embedding but gives
                # the hosted comparator very little identity evidence.
                pair_quality = min(left_record.quality, right_record.quality)
                event_gap_s = _interval_gap(left, right).total_seconds()
                candidates.append(
                    (
                        -pair_quality,
                        event_gap_s,
                        distance,
                        pair,
                        left_record,
                        right_record,
                        left_images[0],
                        right_images[0],
                    )
                )
        if not candidates:
            return empty
        (
            _,
            _,
            _,
            pair,
            left_record,
            right_record,
            left_image,
            right_image,
        ) = min(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[4].local_entity_id,
                item[5].local_entity_id,
            ),
        )
        self._vlm_attempted_pairs.add(pair)
        self._vlm_calls += 1
        LOGGER.info(
            "bounded VLM ReID attempt call=%s/%s replay_id=%s "
            "entity_kind=%s left_source=%s right_source=%s",
            self._vlm_calls,
            self.config.vlm_maximum_calls_per_replay,
            self._replay_id or "unknown",
            left.entity_kind,
            left.source_id,
            right.source_id,
        )
        try:
            decision = self._vlm_comparator.compare(
                entity_kind=left.entity_kind,
                left_image_url=left_image,
                right_image_url=right_image,
                left_local_entity_id=left_record.local_entity_id,
                right_local_entity_id=right_record.local_entity_id,
            )
        except Exception:
            LOGGER.exception(
                "bounded VLM ReID fallback failed call=%s/%s",
                self._vlm_calls,
                self.config.vlm_maximum_calls_per_replay,
            )
            return empty
        LOGGER.info(
            "bounded VLM ReID result call=%s/%s replay_id=%s "
            "entity_kind=%s accepted=%s confidence=%.3f reason=%s",
            self._vlm_calls,
            self.config.vlm_maximum_calls_per_replay,
            self._replay_id or "unknown",
            left.entity_kind,
            (
                decision.same_identity
                and decision.confidence >= self.config.vlm_minimum_confidence
            ),
            decision.confidence,
            decision.reason.replace("\n", " ")[:240],
        )
        if (
            not decision.same_identity
            or decision.confidence < self.config.vlm_minimum_confidence
        ):
            return empty
        canonical_id = deterministic_id(
            f"canonical_{left.entity_kind}",
            {
                "basis": "vlm_fallback",
                "left": [left.source_id, left_record.local_entity_id],
                "right": [right.source_id, right_record.local_entity_id],
            },
            length=32,
        )
        association = EntityAssociation(
            left_local_entity_id=left_record.local_entity_id,
            right_local_entity_id=right_record.local_entity_id,
            canonical_entity_id=canonical_id,
            distance=max(0.0, 1.0 - decision.confidence),
            confidence=decision.confidence,
            route_time_compatible=True,
            association_basis="vlm_fallback",
            association_model_id=str(
                getattr(self._vlm_comparator, "model", "openai-vlm")
            ),
        )
        return empty.model_copy(
            update={
                "associations": (association,),
                "unmatched_left": tuple(
                    item
                    for item in empty.unmatched_left
                    if item != left_record.local_entity_id
                ),
                "unmatched_right": tuple(
                    item
                    for item in empty.unmatched_right
                    if item != right_record.local_entity_id
                ),
            }
        )

    @staticmethod
    def _association_pair(value: EntityAssociationSet, row: Any) -> tuple[
        str, tuple[str, str], tuple[str, str]
    ]:
        left = (value.left_source_id, row.left_local_entity_id)
        right = (value.right_source_id, row.right_local_entity_id)
        return _identity_pair(value.entity_kind, left, right)

    def _canonicalize(self, value: EntityAssociationSet) -> EntityAssociationSet:
        rows = []
        for row in value.associations:
            left_key = (
                value.entity_kind,
                value.left_source_id,
                row.left_local_entity_id,
            )
            right_key = (
                value.entity_kind,
                value.right_source_id,
                row.right_local_entity_id,
            )
            canonical = (
                self._canonical_by_local.get(left_key)
                or self._canonical_by_local.get(right_key)
                or deterministic_id(
                    f"canonical_{value.entity_kind}",
                    {"anchor": min(left_key, right_key)},
                    length=32,
                )
            )
            self._canonical_by_local[left_key] = canonical
            self._canonical_by_local[right_key] = canonical
            rows.append(row.model_copy(update={"canonical_entity_id": canonical}))
        return value.model_copy(update={"associations": tuple(rows)})


def _interval_gap(left: DescriptorSet, right: DescriptorSet) -> timedelta:
    if left.event_time_interval.overlaps(right.event_time_interval):
        return timedelta()
    if left.event_time_interval.end < right.event_time_interval.start:
        return right.event_time_interval.start - left.event_time_interval.end
    return left.event_time_interval.start - right.event_time_interval.end


def _source_from_entity_id(entity_id: str) -> str:
    source, separator, _ = entity_id.partition(":")
    if not separator or not source:
        raise ValueError("identity comparison demand contains an unscoped entity ID")
    return source


def _identity_pair(
    entity_kind: str,
    left: tuple[str, str],
    right: tuple[str, str],
) -> tuple[str, tuple[str, str], tuple[str, str]]:
    first, second = sorted((left, right))
    return entity_kind, first, second


class IdentityMqttService:
    def __init__(
        self,
        *,
        config: IdentityServiceConfig,
        processor: IdentityAssociationProcessor,
        host: str,
        port: int,
        client: Any | None = None,
    ) -> None:
        if client is None and mqtt is None:
            raise RuntimeError("paho-mqtt is required for the identity service")
        self.config = config
        self.processor = processor
        self.host = host
        self.port = port
        self.client = client or mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="fable-identity-x86server",
            clean_session=False,
        )
        self._stop = threading.Event()
        self._descriptor_ready = False
        self._descriptor_generation: str | None = None
        self._pending_demands: dict[str, IdentityComparisonDemand] = {}
        self._crop_request_attempts: dict[str, int] = {}
        self._crop_request_timers: dict[str, threading.Timer] = {}
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        if int(getattr(reason_code, "value", reason_code)) != 0:
            LOGGER.error("MQTT connection failed: %s", reason_code)
            return
        client.subscribe(self.config.input_topic, qos=0)
        client.subscribe("/fable/identity/demands", qos=1)
        client.subscribe("/fable/identity/cancellations", qos=1)
        client.subscribe("/readiness/x86server/fable_reid_descriptor", qos=1)
        client.subscribe("/replay/sync", qos=1)
        client.publish(
            self.config.readiness_topic,
            json.dumps(
                {
                    "ready": True,
                    "vlm_fallback_enabled": self.config.vlm_fallback_enabled,
                    "vlm_available": self.processor.vlm_available,
                    "vlm_maximum_calls_per_replay": self.config.vlm_maximum_calls_per_replay,
                    "escalation_policy_id": self.config.escalation_policy_id or "default",
                }
            ),
            qos=1,
            retain=True,
        )

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        try:
            if message.topic == "/readiness/x86server/fable_reid_descriptor":
                document = json.loads(message.payload.decode("utf-8"))
                self._descriptor_ready = bool(document.get("ready"))
                if self._descriptor_ready:
                    generation = document.get("worker_generation")
                    generation_changed = bool(
                        generation and generation != self._descriptor_generation
                    )
                    if generation_changed:
                        self._descriptor_generation = str(generation)
                    for demand in tuple(self._pending_demands.values()):
                        # A retained readiness message can belong to a worker
                        # that was replaced while its successor warms the ReID
                        # model.  A new worker generation gets one fresh bounded
                        # retry budget; repeated readiness from the same worker
                        # remains idempotent.
                        if generation_changed:
                            self._crop_request_attempts.pop(demand.demand_id, None)
                            timer = self._crop_request_timers.pop(demand.demand_id, None)
                            if timer is not None:
                                timer.cancel()
                        if self._crop_request_attempts.get(demand.demand_id, 0) == 0:
                            self._request_historical_crops_with_retry(client, demand)
                return
            if message.topic == "/replay/sync":
                if not message.payload:
                    return
                document = json.loads(message.payload.decode("utf-8"))
                replay_id = document.get("replay_id") or document.get("command_id")
                changed = self.processor.reset(str(replay_id) if replay_id else None)
                if changed:
                    self._cancel_crop_request_timers()
                    self._pending_demands.clear()
                    self._crop_request_attempts.clear()
                    LOGGER.info("reset identity state for replay_id=%s", replay_id)
                return
            if message.topic == "/fable/identity/demands":
                demand = IdentityComparisonDemand.model_validate_json(
                    message.payload
                )
                resolved = self.processor.resolve_demand(demand)
                LOGGER.info(
                    "identity demand demand_id=%s pair=(%s,%s) resolved_sets=%d "
                    "vlm_available=%s vlm_calls=%d",
                    demand.demand_id,
                    demand.left_local_entity_id,
                    demand.right_local_entity_id,
                    len(resolved),
                    self.processor.vlm_available,
                    self.processor.vlm_calls,
                )
                for association in resolved:
                    client.publish(
                        self.config.output_topic,
                        association.model_dump_json(),
                        qos=1,
                    )
                if not resolved:
                    was_pending = demand.demand_id in self._pending_demands
                    self._pending_demands[demand.demand_id] = demand
                    if self._descriptor_ready and not was_pending:
                        self._request_historical_crops_with_retry(client, demand)
                return
            if message.topic == "/fable/identity/cancellations":
                cancellation = IdentityComparisonCancellation.model_validate_json(
                    message.payload
                )
                was_pending = self._pending_demands.pop(
                    cancellation.demand_id, None
                ) is not None
                self._crop_request_attempts.pop(cancellation.demand_id, None)
                timer = self._crop_request_timers.pop(
                    cancellation.demand_id, None
                )
                if timer is not None:
                    timer.cancel()
                was_registered = self.processor.cancel_demand(
                    cancellation.demand_id
                )
                LOGGER.info(
                    "cancelled identity demand demand_id=%s pending=%s registered=%s reason=%s",
                    cancellation.demand_id,
                    was_pending,
                    was_registered,
                    cancellation.reason,
                )
                return
            document = json.loads(message.payload.decode("utf-8"))
            descriptors = DescriptorSet.model_validate(document)
            for association in self.processor.update(descriptors):
                client.publish(
                    self.config.output_topic,
                    association.model_dump_json(),
                    qos=1,
                )
                self._complete_satisfied_demands(association)
        except Exception:
            LOGGER.exception("identity provider rejected topic=%s", message.topic)

    def _request_historical_crops_with_retry(
        self,
        client: Any,
        demand: IdentityComparisonDemand,
    ) -> None:
        """Request retained crops with a bounded startup-race retry.

        MQTT readiness is retained.  A newly started identity service can see
        the previous descriptor worker's ready message just before the new
        worker has installed its crop subscription.  Three QoS-1 attempts,
        spaced one second apart, bridge that narrow handoff without turning
        identity extraction into an always-on provider.
        """

        if demand.demand_id not in self._pending_demands:
            return
        previous_attempts = self._crop_request_attempts.get(demand.demand_id, 0)
        # A generation-aware worker has acknowledged the exact subscription
        # that consumes this QoS-1 request. One publication is therefore the
        # bounded contract. Retrying three times was only needed for legacy
        # readiness messages and can multiply an expensive descriptor FIFO by
        # three for every candidate pair.
        maximum_attempts = 1 if self._descriptor_generation is not None else 3
        if previous_attempts >= maximum_attempts:
            return
        attempt = previous_attempts + 1
        self._crop_request_attempts[demand.demand_id] = attempt
        self._publish_historical_crop_request(client, demand)
        LOGGER.info(
            "bounded identity crop request demand_id=%s attempt=%d/%d",
            demand.demand_id,
            attempt,
            maximum_attempts,
        )
        if attempt >= maximum_attempts:
            self._crop_request_timers.pop(demand.demand_id, None)
            return
        timer = threading.Timer(
            1.0,
            self._request_historical_crops_with_retry,
            args=(client, demand),
        )
        timer.daemon = True
        previous = self._crop_request_timers.get(demand.demand_id)
        if previous is not None:
            previous.cancel()
        self._crop_request_timers[demand.demand_id] = timer
        timer.start()

    @staticmethod
    def _publish_historical_crop_request(client: Any, demand: IdentityComparisonDemand) -> None:
        client.publish(
            "/fable/identity/crop-demands",
            json.dumps(
                {
                    "schema_version": "fable.bounded_identity_crop_request.v1",
                    "request_id": demand.request_id,
                    "demand_id": demand.demand_id,
                    "entity_kind": demand.entity_kind,
                    "local_entity_ids": [
                        demand.left_local_entity_id,
                        demand.right_local_entity_id,
                    ],
                    "event_time_interval": demand.event_time_interval.model_dump(
                        mode="json"
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            qos=1,
            retain=False,
        )

    def _complete_satisfied_demands(
        self,
        association: EntityAssociationSet,
    ) -> None:
        associated = {
            frozenset(
                (row.left_local_entity_id, row.right_local_entity_id)
            )
            for row in association.associations
        }
        for demand_id, demand in tuple(self._pending_demands.items()):
            wanted = frozenset(
                (demand.left_local_entity_id, demand.right_local_entity_id)
            )
            if wanted not in associated:
                continue
            self._pending_demands.pop(demand_id, None)
            self._crop_request_attempts.pop(demand_id, None)
            timer = self._crop_request_timers.pop(demand_id, None)
            if timer is not None:
                timer.cancel()

    def _cancel_crop_request_timers(self) -> None:
        for timer in self._crop_request_timers.values():
            timer.cancel()
        self._crop_request_timers.clear()

    def run(self) -> None:
        self.client.connect(self.host, self.port, keepalive=60)
        self.client.loop_start()
        try:
            self._stop.wait()
        finally:
            self.client.publish(
                self.config.readiness_topic,
                json.dumps({"ready": False}),
                qos=1,
                retain=True,
            )
            self.client.loop_stop()
            self.client.disconnect()

    def stop(self) -> None:
        self._cancel_crop_request_timers()
        self._stop.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mqtt-host", default=os.getenv("MQTT_HOST", "mqtt"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument(
        "--maximum-event-time-gap",
        type=float,
        default=float(os.getenv("FABLE_REID_MAXIMUM_EVENT_TIME_GAP_SEC", "30")),
    )
    parser.add_argument(
        "--vlm-proxy-url",
        default=os.getenv("FABLE_VLM_PROXY_URL", ""),
    )
    parser.add_argument(
        "--maximum-cosine-distance",
        type=float,
        default=float(os.getenv("FABLE_REID_MAXIMUM_COSINE_DISTANCE", "0.25")),
    )
    parser.add_argument(
        "--same-camera-maximum-event-time-gap",
        type=float,
        default=float(
            os.getenv("FABLE_REID_SAME_CAMERA_MAXIMUM_EVENT_TIME_GAP_SEC", "90")
        ),
    )
    parser.add_argument(
        "--same-camera-maximum-cosine-distance",
        type=float,
        default=float(
            os.getenv("FABLE_REID_SAME_CAMERA_MAXIMUM_COSINE_DISTANCE", "0.30")
        ),
    )
    parser.add_argument(
        "--vlm-fallback-enabled",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("FABLE_VLM_REID_ENABLED", "false").lower()
        in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--vlm-model",
        default=os.getenv(
            "FABLE_VLM_REID_MODEL",
            "gpt-4o-mini-2024-07-18",
        ),
    )
    parser.add_argument(
        "--vlm-maximum-calls-per-replay",
        type=int,
        default=int(os.getenv("FABLE_VLM_REID_MAX_CALLS", "10")),
    )
    parser.add_argument(
        "--vlm-minimum-confidence",
        type=float,
        default=float(os.getenv("FABLE_VLM_REID_MIN_CONFIDENCE", "0.60")),
    )
    parser.add_argument(
        "--escalation-policy-id",
        choices=tuple(sorted(LIVE_ESCALATION_POLICIES)),
        default=os.getenv("FABLE_IDENTITY_ESCALATION_POLICY", "") or None,
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = IdentityServiceConfig(
        maximum_event_time_gap_s=args.maximum_event_time_gap,
        maximum_cosine_distance=args.maximum_cosine_distance,
        same_camera_maximum_event_time_gap_s=(
            args.same_camera_maximum_event_time_gap
        ),
        same_camera_maximum_cosine_distance=(
            args.same_camera_maximum_cosine_distance
        ),
        vlm_fallback_enabled=args.vlm_fallback_enabled,
        vlm_maximum_calls_per_replay=args.vlm_maximum_calls_per_replay,
        vlm_minimum_confidence=args.vlm_minimum_confidence,
        escalation_policy_id=args.escalation_policy_id or "",
    )
    api_key = os.getenv("OPENAI_API_KEY", "")
    comparator = None
    if config.vlm_fallback_enabled:
        if args.vlm_proxy_url:
            comparator = RemoteVisionIdentityComparator(
                endpoint=args.vlm_proxy_url,
            )
        elif api_key:
            comparator = OpenAIVisionIdentityComparator(
                api_key=api_key,
                model=args.vlm_model,
            )
        else:
            LOGGER.warning(
                "FABLE VLM ReID fallback requested but OPENAI_API_KEY is unset; "
                "fallback is disabled"
            )
    service = IdentityMqttService(
        config=config,
        processor=IdentityAssociationProcessor(
            config,
            vlm_comparator=comparator,
        ),
        host=args.mqtt_host,
        port=args.mqtt_port,
    )
    signal.signal(signal.SIGTERM, lambda *_: service.stop())
    signal.signal(signal.SIGINT, lambda *_: service.stop())
    service.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
