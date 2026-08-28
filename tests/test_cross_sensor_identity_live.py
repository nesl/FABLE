from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from fable.common.time import EventTimeInterval
from providers.vehicle import (
    CrossSensorIdentityAssociator,
    DeterministicDescriptorProvider,
    FastReidEntityDescriptor,
)
from providers.vehicle.errors import ArtifactCompatibilityError
from providers.vehicle.identity_service import (
    IdentityAssociationProcessor,
    IdentityMqttService,
    IdentityServiceConfig,
)
from providers.vehicle.models import (
    IdentityComparisonCancellation,
    IdentityComparisonDemand,
)
from providers.vehicle.vlm_reid import (
    OpenAIVisionIdentityComparator,
    VlmIdentityDecision,
)


def test_duplicate_replay_sync_does_not_clear_identity_history() -> None:
    processor = IdentityAssociationProcessor(IdentityServiceConfig())
    processor._latest[("camera", "vehicle", ("model",))] = object()  # type: ignore[assignment]

    assert processor.reset("replay-1") is True
    processor._latest[("camera", "vehicle", ("model",))] = object()  # type: ignore[assignment]
    assert processor.reset("replay-1") is False
    assert processor._latest
    assert processor.reset("replay-2") is True
    assert not processor._latest


def test_canonical_identity_cache_is_namespaced_by_entity_kind() -> None:
    processor = IdentityAssociationProcessor(IdentityServiceConfig())

    processor.update(_descriptors("camera-a", "shared-id", kind="person"))
    person = processor.update(
        _descriptors("camera-b", "shared-id", kind="person")
    )[0].associations[0]
    processor.update(_descriptors("camera-a", "shared-id", kind="vehicle"))
    vehicle = processor.update(
        _descriptors("camera-b", "shared-id", kind="vehicle")
    )[0].associations[0]

    assert person.canonical_entity_id.startswith("canonical_person_")
    assert vehicle.canonical_entity_id.startswith("canonical_vehicle_")
    assert person.canonical_entity_id != vehicle.canonical_entity_id


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _descriptors(source: str, entity: str, *, kind: str = "vehicle", at=NOW):
    return DeterministicDescriptorProvider(entity_kind=kind).encode_ids(
        (entity,),
        source_id=source,
        event_time=at,
    )


def _with_crop(value, *, vector: tuple[float, ...] | None = None):
    record = value.records[0]
    return value.model_copy(
        update={
            "records": (
                record.model_copy(
                    update={
                        "vector": vector or record.vector,
                        "source_crop_data_urls": (
                            "data:image/jpeg;base64,dGVzdA==",
                        ),
                        "source_context_image_data_urls": (
                            "data:image/jpeg;base64,ZnVsbC1mcmFtZQ==",
                        ),
                    }
                ),
            )
        }
    )


class AcceptingVlm:
    model = "gpt-4o-mini-test"

    def __init__(self) -> None:
        self.calls = 0
        self.last_arguments = {}

    def compare(self, **kwargs):
        self.calls += 1
        self.last_arguments = kwargs
        return VlmIdentityDecision(
            same_identity=True,
            confidence=0.91,
            reason="same visible identity",
        )


class _IdentityClient:
    def __init__(self) -> None:
        self.published = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))


def test_missing_exact_identity_evidence_is_replayed_after_descriptor_ready() -> None:
    client = _IdentityClient()
    processor = IdentityAssociationProcessor(IdentityServiceConfig())
    service = IdentityMqttService(
        config=IdentityServiceConfig(),
        processor=processor,
        host="mqtt",
        port=1883,
        client=client,
    )
    demand = IdentityComparisonDemand(
        request_id="request-1",
        demand_id="demand-1",
        left_local_entity_id="camera:session:0",
        right_local_entity_id="camera:session:1",
        event_time_interval=EventTimeInterval(start=NOW, end=NOW),
    )

    service._on_message(
        client,
        None,
        SimpleNamespace(
            topic="/fable/identity/demands",
            payload=demand.model_dump_json().encode(),
        ),
    )
    assert not [row for row in client.published if row[0].endswith("crop-demands")]

    service._on_message(
        client,
        None,
        SimpleNamespace(
            topic="/readiness/x86server/fable_reid_descriptor",
            payload=json.dumps({"ready": True}).encode(),
        ),
    )

    requests = [row for row in client.published if row[0].endswith("crop-demands")]
    assert len(requests) == 1
    payload = json.loads(requests[0][1])
    assert payload["local_entity_ids"] == [
        "camera:session:0",
        "camera:session:1",
    ]
    assert requests[0][2:] == (1, False)


def test_duplicate_demand_and_readiness_do_not_exceed_crop_retry_budget() -> None:
    client = _IdentityClient()
    service = IdentityMqttService(
        config=IdentityServiceConfig(),
        processor=IdentityAssociationProcessor(IdentityServiceConfig()),
        host="mqtt",
        port=1883,
        client=client,
    )
    demand = IdentityComparisonDemand(
        request_id="request-1",
        demand_id="demand-1",
        left_local_entity_id="camera:session:0",
        right_local_entity_id="camera:session:1",
        event_time_interval=EventTimeInterval(start=NOW, end=NOW),
    )
    service._descriptor_ready = True

    demand_message = SimpleNamespace(
        topic="/fable/identity/demands",
        payload=demand.model_dump_json().encode(),
    )
    service._on_message(client, None, demand_message)
    service._on_message(client, None, demand_message)
    service._crop_request_attempts[demand.demand_id] = 3
    service._request_historical_crops_with_retry(client, demand)
    service._on_message(
        client,
        None,
        SimpleNamespace(
            topic="/readiness/x86server/fable_reid_descriptor",
            payload=json.dumps({"ready": True}).encode(),
        ),
    )

    requests = [row for row in client.published if row[0].endswith("crop-demands")]
    assert len(requests) == 1


def test_new_descriptor_worker_generation_reopens_bounded_crop_retry() -> None:
    client = _IdentityClient()
    service = IdentityMqttService(
        config=IdentityServiceConfig(),
        processor=IdentityAssociationProcessor(IdentityServiceConfig()),
        host="mqtt",
        port=1883,
        client=client,
    )
    demand = IdentityComparisonDemand(
        request_id="request-1",
        demand_id="demand-1",
        left_local_entity_id="camera:session:0",
        right_local_entity_id="camera:session:1",
        event_time_interval=EventTimeInterval(start=NOW, end=NOW),
    )
    service._descriptor_ready = True
    service._descriptor_generation = "old-worker"
    service._on_message(
        client,
        None,
        SimpleNamespace(
            topic="/fable/identity/demands",
            payload=demand.model_dump_json().encode(),
        ),
    )
    service._crop_request_attempts[demand.demand_id] = 3

    service._on_message(
        client,
        None,
        SimpleNamespace(
            topic="/readiness/x86server/fable_reid_descriptor",
            payload=json.dumps(
                {"ready": True, "worker_generation": "new-worker"}
            ).encode(),
        ),
    )

    requests = [row for row in client.published if row[0].endswith("crop-demands")]
    assert len(requests) == 2
    assert service._crop_request_attempts[demand.demand_id] == 1


def test_generation_aware_descriptor_uses_single_qos1_crop_request() -> None:
    client = _IdentityClient()
    service = IdentityMqttService(
        config=IdentityServiceConfig(),
        processor=IdentityAssociationProcessor(IdentityServiceConfig()),
        host="mqtt",
        port=1883,
        client=client,
    )
    service._descriptor_ready = True
    service._descriptor_generation = "worker-1"
    demand = IdentityComparisonDemand(
        request_id="request-1",
        demand_id="demand-qos1",
        left_local_entity_id="camera:session:0",
        right_local_entity_id="camera:session:1",
        event_time_interval=EventTimeInterval(start=NOW, end=NOW),
    )

    service._on_message(
        client,
        None,
        SimpleNamespace(
            topic="/fable/identity/demands",
            payload=demand.model_dump_json().encode(),
        ),
    )

    timer = service._crop_request_timers.get(demand.demand_id)
    assert timer is None
    requests = [row for row in client.published if row[0].endswith("crop-demands")]
    assert len(requests) == 1


def test_cancelled_identity_demand_stops_retries_and_forgets_exact_pair() -> None:
    client = _IdentityClient()
    processor = IdentityAssociationProcessor(IdentityServiceConfig())
    service = IdentityMqttService(
        config=IdentityServiceConfig(),
        processor=processor,
        host="mqtt",
        port=1883,
        client=client,
    )
    demand = IdentityComparisonDemand(
        request_id="request-1",
        demand_id="demand-1",
        left_local_entity_id="camera:session:0",
        right_local_entity_id="camera:session:1",
        event_time_interval=EventTimeInterval(start=NOW, end=NOW),
    )
    service._descriptor_ready = True
    service._on_message(
        client,
        None,
        SimpleNamespace(
            topic="/fable/identity/demands",
            payload=demand.model_dump_json().encode(),
        ),
    )
    timer = service._crop_request_timers[demand.demand_id]

    cancellation = IdentityComparisonCancellation(
        request_id=demand.request_id,
        demand_id=demand.demand_id,
        reason="hypothesis evicted",
    )
    service._on_message(
        client,
        None,
        SimpleNamespace(
            topic="/fable/identity/cancellations",
            payload=cancellation.model_dump_json().encode(),
        ),
    )

    assert demand.demand_id not in service._pending_demands
    assert demand.demand_id not in service._crop_request_attempts
    assert demand.demand_id not in service._crop_request_timers
    assert timer.finished.is_set()
    assert not processor._preferred_pairs


def test_cancelling_one_duplicate_pair_keeps_other_live_demand_registered() -> None:
    processor = IdentityAssociationProcessor(IdentityServiceConfig())
    first = IdentityComparisonDemand(
        request_id="request-1",
        demand_id="demand-1",
        left_local_entity_id="camera:session:0",
        right_local_entity_id="camera:session:1",
        event_time_interval=EventTimeInterval(start=NOW, end=NOW),
    )
    second = first.model_copy(update={"demand_id": "demand-2"})

    processor.register_demand(first)
    processor.register_demand(second)
    assert processor.cancel_demand(first.demand_id)
    assert processor._preferred_pairs
    assert processor.cancel_demand(second.demand_id)
    assert not processor._preferred_pairs


def test_active_identity_demand_reserves_vlm_budget_for_exact_pair():
    vlm = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_cosine_distance=0.01,
            vlm_fallback_enabled=True,
            vlm_candidate_maximum_cosine_distance=2.0,
            escalation_policy_id="C3_FABLE_ESCALATION",
        ),
        vlm_comparator=vlm,
    )
    processor.register_demand(
        IdentityComparisonDemand(
            request_id="request-1",
            demand_id="demand-1",
            left_local_entity_id="camera-a:wanted-left",
            right_local_entity_id="camera-b:wanted-right",
            event_time_interval=EventTimeInterval(start=NOW, end=NOW),
        )
    )
    assert processor.update(
        _with_crop(_descriptors("camera-a", "unrelated-left"))
    ) == ()
    assert processor.update(
        _with_crop(_descriptors("camera-b", "unrelated-right"))
    ) == ()
    assert vlm.calls == 0
    assert processor.update(
        _with_crop(_descriptors("camera-a", "camera-a:wanted-left"))
    ) == ()
    result = processor.update(
        _with_crop(_descriptors("camera-b", "camera-b:wanted-right"))
    )
    assert vlm.calls == 1
    assert result[0].associations[0].left_local_entity_id == (
        "camera-a:wanted-left"
    )


def test_default_exact_pair_escalates_local_match_below_semantic_floor() -> None:
    vlm = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            same_camera_maximum_cosine_distance=0.5,
            vlm_fallback_enabled=True,
            vlm_minimum_confidence=0.5,
        ),
        vlm_comparator=vlm,
    )
    left_id = "camera-a:session:1"
    right_id = "camera-a:session:2"
    processor.register_demand(
        IdentityComparisonDemand(
            request_id="request-weak",
            demand_id="demand-weak",
            left_local_entity_id=left_id,
            right_local_entity_id=right_id,
            event_time_interval=EventTimeInterval(
                start=NOW, end=NOW + timedelta(seconds=5)
            ),
        )
    )
    left = _with_crop(
        _descriptors("camera-a", left_id), vector=(1.0, 0.0)
    ).model_copy(update={"dimension": 2})
    right = _with_crop(
        _descriptors(
            "camera-a", right_id, at=NOW + timedelta(seconds=5)
        ),
        vector=(0.7, 0.714142842854285),
    ).model_copy(update={"dimension": 2})

    assert processor.update(left) == ()
    result = processor.update(right)

    assert vlm.calls == 1
    assert result[0].associations[0].association_basis == "vlm_fallback"


def test_exact_retrospective_identity_demand_bypasses_streaming_gap() -> None:
    vlm = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_event_time_gap_s=30,
            maximum_cosine_distance=0.01,
            vlm_fallback_enabled=True,
            vlm_candidate_maximum_cosine_distance=2.0,
            escalation_policy_id="C3_FABLE_ESCALATION",
        ),
        vlm_comparator=vlm,
    )
    processor.register_demand(
        IdentityComparisonDemand(
            request_id="request-retrospective",
            demand_id="demand-retrospective",
            left_local_entity_id="camera-a:before",
            right_local_entity_id="camera-b:after",
            event_time_interval=EventTimeInterval(
                start=NOW, end=NOW + timedelta(seconds=90)
            ),
        )
    )
    left = _with_crop(_descriptors("camera-a", "camera-a:before", at=NOW))
    right = _with_crop(
        _descriptors(
            "camera-b",
            "camera-b:after",
            at=NOW + timedelta(seconds=45),
        ),
        vector=tuple(-item for item in left.records[0].vector),
    )

    assert processor.update(left) == ()
    result = processor.update(right)

    assert vlm.calls == 1
    assert result[0].associations[0].association_basis == "vlm_fallback"


def test_late_same_camera_demand_reconsiders_retained_track_fragments() -> None:
    vlm = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_cosine_distance=0.01,
            same_camera_maximum_cosine_distance=0.01,
            vlm_fallback_enabled=True,
            vlm_candidate_maximum_cosine_distance=2.0,
            escalation_policy_id="C3_FABLE_ESCALATION",
        ),
        vlm_comparator=vlm,
    )
    before = _with_crop(_descriptors("camera-a", "camera-a:session:1"))
    after = _with_crop(
        _descriptors(
            "camera-a",
            "camera-a:session:2",
            at=NOW + timedelta(seconds=5),
        ),
        vector=tuple(reversed(before.records[0].vector)),
    )
    assert processor.update(before) == ()
    assert processor.update(after) == ()

    result = processor.resolve_demand(
        IdentityComparisonDemand(
            request_id="request-late",
            demand_id="exit-bound-pair",
            left_local_entity_id="camera-a:session:1",
            right_local_entity_id="camera-a:session:2",
            event_time_interval=EventTimeInterval(
                start=NOW,
                end=NOW + timedelta(seconds=10),
            ),
        )
    )

    assert vlm.calls == 1
    assert len(result) == 1
    assert result[0].associations[0].association_basis == "vlm_fallback"


def test_late_demand_does_not_merge_simultaneous_same_camera_tracks() -> None:
    vlm = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            vlm_fallback_enabled=True,
            vlm_candidate_maximum_cosine_distance=2.0,
        ),
        vlm_comparator=vlm,
    )
    left = _with_crop(_descriptors("camera-a", "camera-a:session:1"))
    right = _with_crop(_descriptors("camera-a", "camera-a:session:2"))
    processor.update(left)
    processor.update(right)

    result = processor.resolve_demand(
        IdentityComparisonDemand(
            request_id="request-negative",
            demand_id="simultaneous-pair",
            left_local_entity_id="camera-a:session:1",
            right_local_entity_id="camera-a:session:2",
            event_time_interval=EventTimeInterval(start=NOW, end=NOW),
        )
    )

    assert result == ()
    assert vlm.calls == 0


def test_live_strong_only_bypasses_a_valid_local_match():
    vlm = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_cosine_distance=0.01,
            vlm_fallback_enabled=True,
            escalation_policy_id="C1_STRONG_ONLY",
        ),
        vlm_comparator=vlm,
    )
    assert processor.update(_with_crop(_descriptors("camera-a", "same"))) == ()
    result = processor.update(_with_crop(_descriptors("camera-b", "same")))
    assert vlm.calls == 1
    assert result[0].associations[0].association_basis == "vlm_fallback"


def test_live_no_escalation_policy_never_invokes_vlm():
    vlm = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_cosine_distance=0.01,
            vlm_fallback_enabled=True,
            escalation_policy_id="C4_FABLE_NO_ESCALATION",
        ),
        vlm_comparator=vlm,
    )
    assert processor.update(_with_crop(_descriptors("camera-a", "left"))) == ()
    assert processor.update(_with_crop(_descriptors("camera-b", "right"))) == ()
    assert vlm.calls == 0


def test_live_fable_escalates_ambiguous_local_match():
    vlm = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_cosine_distance=0.5,
            vlm_fallback_enabled=True,
            vlm_minimum_confidence=0.8,
            escalation_policy_id="C3_FABLE_ESCALATION",
        ),
        vlm_comparator=vlm,
    )
    left = _with_crop(_descriptors("camera-a", "same"))
    right = _with_crop(_descriptors("camera-b", "same"))
    assert processor.update(left) == ()
    result = processor.update(right)
    # Deterministic identical descriptors normally form a high-confidence local
    # match, so force the policy boundary directly with a low-confidence copy.
    local = result[0]
    weak = local.model_copy(
        update={
            "associations": (
                local.associations[0].model_copy(update={"confidence": 0.2}),
            )
        }
    )
    escalated = processor._apply_escalation_policy(left, right, weak)
    assert escalated.associations == ()


def test_associator_keeps_person_and_vehicle_feature_spaces_separate() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="person and vehicle"):
        CrossSensorIdentityAssociator().associate(
            _descriptors("camera-a", "entity-1", kind="person"),
            _descriptors("camera-b", "entity-1", kind="vehicle"),
        )


def test_live_processor_emits_stable_canonical_identity_across_three_sensors() -> None:
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(maximum_cosine_distance=0.01)
    )
    assert processor.update(_descriptors("camera-a", "same")) == ()
    ab = processor.update(_descriptors("camera-b", "same"))
    bc_and_ac = processor.update(_descriptors("camera-c", "same"))

    canonical = ab[0].associations[0].canonical_entity_id
    assert canonical.startswith("canonical_vehicle_")
    assert {
        row.canonical_entity_id
        for result in bc_and_ac
        for row in result.associations
    } == {canonical}


def test_live_processor_stitches_same_camera_track_id_change() -> None:
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(maximum_cosine_distance=0.01)
    )
    assert processor.update(_descriptors("camera-a", "track-before")) == ()
    # The deterministic fixture encodes the entity text, so use an identical
    # vector with a different local track ID to model an occlusion reset.
    after = _descriptors("camera-a", "track-before").model_copy(
        update={
            "records": (
                _descriptors("camera-a", "track-before").records[0].model_copy(
                    update={"local_entity_id": "track-after"}
                ),
            ),
            "event_time_interval": EventTimeInterval(
                start=NOW + timedelta(seconds=1),
                end=NOW + timedelta(seconds=1),
            ),
        }
    )
    result = processor.update(after)
    assert len(result) == 1
    assert result[0].associations[0].left_local_entity_id == "track-before"
    assert result[0].associations[0].right_local_entity_id == "track-after"


def test_live_processor_does_not_merge_simultaneous_same_camera_tracks() -> None:
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(maximum_cosine_distance=0.01)
    )
    first = _descriptors("camera-a", "track-a")
    second = first.model_copy(
        update={
            "records": (
                first.records[0].model_copy(
                    update={"local_entity_id": "track-b"}
                ),
            )
        }
    )

    processor.update(first)

    assert processor.update(second) == ()


def test_live_processor_stitches_track_that_reappears_after_an_intervening_track() -> None:
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(maximum_cosine_distance=0.01)
    )
    before = _descriptors("camera-a", "track-before")
    processor.update(before)
    processor.update(
        _descriptors("camera-a", "unrelated", at=NOW + timedelta(seconds=1))
    )
    after = before.model_copy(
        update={
            "records": (
                before.records[0].model_copy(
                    update={"local_entity_id": "track-after"}
                ),
            ),
            "event_time_interval": EventTimeInterval(
                start=NOW + timedelta(seconds=2),
                end=NOW + timedelta(seconds=2),
            ),
        }
    )

    result = processor.update(after)

    assert len(result) == 1
    assert result[0].associations[0].left_local_entity_id == "track-before"
    assert result[0].associations[0].right_local_entity_id == "track-after"


def test_live_processor_rejects_out_of_window_descriptors() -> None:
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_event_time_gap_s=2,
            maximum_cosine_distance=0.01,
        )
    )
    processor.update(_descriptors("camera-a", "same"))
    assert processor.update(
        _descriptors("camera-b", "same", at=NOW + timedelta(seconds=3))
    ) == ()


def test_live_processor_reconnects_same_camera_tracklet_beyond_cross_camera_window() -> None:
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_event_time_gap_s=2,
            same_camera_maximum_event_time_gap_s=60,
            maximum_cosine_distance=0.01,
            same_camera_maximum_cosine_distance=0.01,
        )
    )
    before = _descriptors("camera-a", "same", kind="person")
    processor.update(before)
    after = before.model_copy(
        update={
            "records": (
                before.records[0].model_copy(
                    update={"local_entity_id": "track-after"}
                ),
            ),
            "event_time_interval": EventTimeInterval(
                start=NOW + timedelta(seconds=40),
                end=NOW + timedelta(seconds=40),
            ),
        }
    )

    result = processor.update(after)

    assert len(result) == 1
    assert result[0].associations[0].left_local_entity_id == "same"
    assert result[0].associations[0].right_local_entity_id == "track-after"


def test_live_processor_does_not_extend_cross_camera_time_window() -> None:
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_event_time_gap_s=2,
            same_camera_maximum_event_time_gap_s=60,
            maximum_cosine_distance=0.01,
            same_camera_maximum_cosine_distance=0.01,
        )
    )
    processor.update(_descriptors("camera-a", "same", kind="person"))

    assert processor.update(
        _descriptors(
            "camera-b",
            "same",
            kind="person",
            at=NOW + timedelta(seconds=40),
        )
    ) == ()


def test_replay_reset_removes_stale_identity_state() -> None:
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(maximum_cosine_distance=0.01)
    )
    processor.update(_descriptors("camera-a", "same"))
    processor.reset("new-replay")
    assert processor.update(_descriptors("camera-b", "same")) == ()


def test_vlm_fallback_runs_only_after_calibrated_reid_fails() -> None:
    comparator = AcceptingVlm()
    config = IdentityServiceConfig(
        maximum_cosine_distance=0.01,
        vlm_fallback_enabled=True,
        vlm_candidate_maximum_cosine_distance=2.0,
    )
    processor = IdentityAssociationProcessor(
        config,
        vlm_comparator=comparator,
    )
    left = _with_crop(_descriptors("camera-a", "left"))
    right = _with_crop(
        _descriptors("camera-b", "right"),
        vector=tuple(-item for item in left.records[0].vector),
    )

    assert processor.update(left) == ()
    result = processor.update(right)

    assert comparator.calls == 1
    assert comparator.last_arguments["left_image_url"].endswith(
        "ZnVsbC1mcmFtZQ=="
    )
    assert result[0].associations[0].association_basis == "vlm_fallback"
    assert result[0].associations[0].association_model_id == "gpt-4o-mini-test"


def test_vlm_fallback_supports_failed_person_reid_without_vehicle_substitution() -> None:
    comparator = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_cosine_distance=0.01,
            vlm_fallback_enabled=True,
            vlm_candidate_maximum_cosine_distance=2.0,
        ),
        vlm_comparator=comparator,
    )
    left = _with_crop(_descriptors("camera-a", "person-left", kind="person"))
    right = _with_crop(
        _descriptors("camera-b", "person-right", kind="person"),
        vector=tuple(-item for item in left.records[0].vector),
    )

    processor.update(left)
    result = processor.update(right)

    association = result[0].associations[0]
    assert result[0].entity_kind == "person"
    assert association.association_basis == "vlm_fallback"
    assert association.canonical_entity_id.startswith("canonical_person_")


def test_vlm_fallback_can_bridge_incompatible_cross_camera_reid_models() -> None:
    comparator = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            vlm_fallback_enabled=True,
            vlm_candidate_maximum_cosine_distance=2.0,
        ),
        vlm_comparator=comparator,
    )
    left = _with_crop(_descriptors("orin-camera", "vehicle-left"))
    right = _with_crop(
        _descriptors(
            "mobile-camera",
            "vehicle-right",
            at=NOW + timedelta(seconds=1),
        ).model_copy(update={"preprocessing_id": "mobile-preprocessing"})
    )

    processor.update(left)
    result = processor.update(right)

    assert comparator.calls == 1
    assert result[0].associations[0].association_basis == "vlm_fallback"


def test_vlm_fallback_does_not_compare_same_local_track_with_itself() -> None:
    comparator = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_cosine_distance=0.0,
            vlm_fallback_enabled=True,
            vlm_candidate_maximum_cosine_distance=2.0,
        ),
        vlm_comparator=comparator,
    )
    first = _with_crop(_descriptors("camera-a", "same-track"))
    second = _with_crop(
        _descriptors(
            "camera-a",
            "same-track",
            at=NOW + timedelta(seconds=1),
        ),
        vector=tuple(-item for item in first.records[0].vector),
    )

    processor.update(first)

    assert processor.update(second) == ()
    assert comparator.calls == 0
    assert processor.vlm_calls == 0


def test_vlm_fallback_never_compares_distinct_same_camera_tracklets() -> None:
    comparator = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_cosine_distance=0.0,
            vlm_fallback_enabled=True,
            vlm_candidate_maximum_cosine_distance=2.0,
        ),
        vlm_comparator=comparator,
    )
    first = _with_crop(_descriptors("camera-a", "track-one"))
    second = _with_crop(
        _descriptors(
            "camera-a",
            "track-two",
            at=NOW + timedelta(seconds=1),
        ),
        vector=tuple(-item for item in first.records[0].vector),
    )

    processor.update(first)

    assert processor.update(second) == ()
    assert comparator.calls == 0
    assert processor.vlm_calls == 0


def test_successful_reid_does_not_spend_vlm_budget() -> None:
    comparator = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_cosine_distance=0.01,
            vlm_fallback_enabled=True,
        ),
        vlm_comparator=comparator,
    )
    processor.update(_with_crop(_descriptors("camera-a", "same")))
    result = processor.update(_with_crop(_descriptors("camera-b", "same")))

    assert result[0].associations[0].association_basis == "reid"
    assert comparator.calls == 0
    assert processor.vlm_calls == 0


def test_vlm_budget_is_bounded_at_ten_and_resets_per_replay() -> None:
    comparator = AcceptingVlm()
    processor = IdentityAssociationProcessor(
        IdentityServiceConfig(
            maximum_cosine_distance=0.0,
            vlm_fallback_enabled=True,
            vlm_maximum_calls_per_replay=10,
            vlm_candidate_maximum_cosine_distance=2.0,
        ),
        vlm_comparator=comparator,
    )
    for index in range(14):
        processor.update(
            _with_crop(_descriptors(f"camera-{index}", f"entity-{index}"))
        )

    assert comparator.calls == 10
    assert processor.vlm_calls == 10
    assert processor.reset("next-replay") is True
    assert processor.vlm_calls == 0


def test_openai_vlm_comparator_parses_responses_output() -> None:
    requests = []

    def transport(payload):
        requests.append(payload)
        return {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"same_identity": true, "confidence": 0.88, '
                                '"reason": "matching vehicle"}'
                            ),
                        }
                    ]
                }
            ]
        }

    comparator = OpenAIVisionIdentityComparator(
        api_key="test-key",
        transport=transport,
    )

    decision = comparator.compare(
        entity_kind="vehicle",
        left_image_url="data:image/jpeg;base64,bGVmdA==",
        right_image_url="data:image/jpeg;base64,cmlnaHQ=",
    )

    assert decision.same_identity is True
    assert decision.confidence == pytest.approx(0.88)
    content = requests[0]["input"][0]["content"]
    assert "pointers, not hard object boundaries" in content[0]["text"]
    assert content[1]["detail"] == "high"
    assert content[2]["detail"] == "high"


def test_fastreid_vehicle_adapter_emits_typed_normalized_descriptors(
    tmp_path,
) -> None:
    extractor = lambda images: [[3.0, 4.0] for _ in images]
    provider = FastReidEntityDescriptor(
        entity_kind="vehicle",
        config_path=tmp_path / "unused.yaml",
        model_path=tmp_path / "unused.pth",
        model_id="fastreid:sbs_R50_ibn:vehicle",
        model_version="veri-v0.1.1",
        preprocessing_id="fastreid-veri-256x256-rgb",
        extractor=extractor,
    )
    result = provider.encode(
        (("track-1", object()),),
        source_id="camera-a",
        event_time_interval=EventTimeInterval(start=NOW, end=NOW),
    )
    assert result.entity_kind == "vehicle"
    assert result.records[0].vector == pytest.approx((0.6, 0.8))


def test_fastreid_warmup_validates_and_retains_injected_extractor(tmp_path) -> None:
    extractor = lambda images: [[1.0, 0.0] for _ in images]
    provider = FastReidEntityDescriptor(
        entity_kind="vehicle",
        config_path=tmp_path / "unused.yaml",
        model_path=tmp_path / "unused.pth",
        model_id="test-model",
        model_version="test-version",
        preprocessing_id="test-preprocessing",
        extractor=extractor,
    )

    provider.warmup()

    assert provider._extractor is extractor
