from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fable.language import compile_event, load_and_compile_event, parse_event
from fable.providers import PredicateMatch
from fable.runtime import (
    CEInstanceManager,
    derive_continuation_frontier,
    derive_discovery_frontier,
)


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT / "ce_definitions"
T0 = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def match(
    predicate: str,
    at: datetime,
    arguments: dict[str, object],
    *,
    source: str = "camera_1",
    confidence: float = 1.0,
    classes: dict[str, str] | None = None,
) -> PredicateMatch:
    return PredicateMatch(
        predicate=predicate,
        event_time=at,
        arguments=arguments,
        provider_id=f"test_{predicate}",
        source_ids=(source,),
        confidence=confidence,
        classes={} if classes is None else classes,
    )


def vehicle_person_departure_event():
    return compile_event(parse_event({
        "event": "vehicle_person_departure",
        "roles": {
            "VEHICLE": {"class": "vehicle"},
            "PERSON": {"class": "person"},
        },
        "pattern": {
            "seq": [
                {"enters": {"object": "VEHICLE"}},
                {
                    "within": {
                        "max": "10s",
                        "pattern": {"enters": {"object": "PERSON"}},
                    }
                },
                {"exits": {"object": "VEHICLE"}},
            ]
        },
    }))


def test_discovery_frontier_is_resolved_provider_independent_requirement() -> None:
    event = load_and_compile_event(DEFINITIONS / "two_vehicle_chase.yaml")
    frontier = derive_discovery_frontier(event)
    assert len(frontier) == 1
    item = frontier[0]
    assert item.predicate == "enters"
    assert item.arguments == {"object": None}
    assert item.classes == {"object": "vehicle"}
    assert item.role_refs == {"object": "LEADER"}
    assert item.expires_at is None


def test_vehicle_enters_at_t1_and_t5_create_two_candidates() -> None:
    event = vehicle_person_departure_event()
    manager = CEInstanceManager(event)

    manager.handle_match(match("enters", T0 + timedelta(seconds=1), {"object": "V1"}, classes={"object": "vehicle"}))
    manager.handle_match(match("enters", T0 + timedelta(seconds=5), {"object": "V2"}, classes={"object": "vehicle"}))

    active = manager.active_instances()
    assert len(active) == 2
    assert {(i.bindings["VEHICLE"], i.matched_at) for i in active} == {
        ("V1", T0 + timedelta(seconds=1)),
        ("V2", T0 + timedelta(seconds=5)),
    }


def test_person_at_t12_expires_first_candidate_but_advances_second() -> None:
    event = vehicle_person_departure_event()
    manager = CEInstanceManager(event)

    manager.handle_match(match("enters", T0 + timedelta(seconds=1), {"object": "V1"}, classes={"object": "vehicle"}))
    manager.handle_match(match("enters", T0 + timedelta(seconds=5), {"object": "V2"}, classes={"object": "vehicle"}))
    produced = manager.handle_match(
        match("enters", T0 + timedelta(seconds=12), {"object": "P1"}, classes={"object": "person"})
    )

    # V1's 10-second continuation window ended at t=11.
    assert any(i.bindings.get("VEHICLE") == "V1" for i in manager.expired_instances())

    # At least one new branch advances V2 with the person binding.
    advanced = [
        i for i in produced
        if i.bindings.get("VEHICLE") == "V2" and i.bindings.get("PERSON") == "P1"
    ]
    assert len(advanced) == 1
    frontier = derive_continuation_frontier(event, advanced[0], T0 + timedelta(seconds=12))
    assert len(frontier) == 1
    assert frontier[0].predicate == "exits"
    assert frontier[0].arguments == {"object": "V2"}

    # The unadvanced V2 prefix is intentionally retained until its t=15 expiry
    # so a different person can still form another valid candidate.
    assert any(
        i.bindings == {"VEHICLE": "V2"}
        for i in manager.active_instances()
    )


def test_two_people_can_fork_same_vehicle_prefix() -> None:
    event = vehicle_person_departure_event()
    manager = CEInstanceManager(event)
    manager.handle_match(match("enters", T0 + timedelta(seconds=1), {"object": "V1"}, classes={"object": "vehicle"}))

    manager.handle_match(match("enters", T0 + timedelta(seconds=2), {"object": "P1"}, classes={"object": "person"}))
    manager.handle_match(match("enters", T0 + timedelta(seconds=3), {"object": "P2"}, classes={"object": "person"}))

    bindings = [i.bindings for i in manager.active_instances()]
    assert {"VEHICLE": "V1", "PERSON": "P1"} in bindings
    assert {"VEHICLE": "V1", "PERSON": "P2"} in bindings
    assert {"VEHICLE": "V1"} in bindings


def test_same_seed_message_processed_twice_creates_one_candidate() -> None:
    event = vehicle_person_departure_event()
    manager = CEInstanceManager(event)
    observation = match("enters", T0 + timedelta(seconds=1), {"object": "V1"}, classes={"object": "vehicle"})

    manager.handle_match(observation)
    assert manager.handle_match(observation) == ()
    assert len(manager.active_instances()) == 1


def test_same_vehicle_can_seed_again_at_later_time() -> None:
    event = vehicle_person_departure_event()
    manager = CEInstanceManager(event)

    manager.handle_match(match("enters", T0 + timedelta(seconds=1), {"object": "V1"}, classes={"object": "vehicle"}))
    manager.handle_match(match("enters", T0 + timedelta(minutes=30), {"object": "V1"}, classes={"object": "vehicle"}))

    # The first one has long since expired waiting for the person, but the new
    # occurrence is a distinct candidate even though it has the same binding.
    assert any(i.matched_at == T0 + timedelta(seconds=1) for i in manager.expired_instances())
    assert any(
        i.matched_at == T0 + timedelta(minutes=30)
        and i.bindings == {"VEHICLE": "V1"}
        for i in manager.active_instances()
    )


def test_discovery_frontier_remains_active_after_candidate_progress() -> None:
    event = vehicle_person_departure_event()
    manager = CEInstanceManager(event)
    manager.handle_match(match("enters", T0 + timedelta(seconds=1), {"object": "V1"}, classes={"object": "vehicle"}))

    active = manager.current_frontier(T0 + timedelta(seconds=2))
    assert [item.predicate for item in active.discovery] == ["enters"]
    assert active.discovery[0].arguments == {"object": None}

    continuation = list(active.continuation.values())
    assert len(continuation) == 1
    assert continuation[0][0].predicate == "enters"
    assert continuation[0][0].classes == {"object": "person"}
    assert continuation[0][0].expires_at == T0 + timedelta(seconds=11)


def test_one_person_match_can_advance_multiple_candidates() -> None:
    event = vehicle_person_departure_event()
    manager = CEInstanceManager(event)

    manager.handle_match(match("enters", T0 + timedelta(seconds=1), {"object": "V1"}, classes={"object": "vehicle"}))
    manager.handle_match(match("enters", T0 + timedelta(seconds=2), {"object": "V2"}, classes={"object": "vehicle"}))
    produced = manager.handle_match(
        match("enters", T0 + timedelta(seconds=3), {"object": "P1"}, classes={"object": "person"})
    )

    advanced_vehicle_ids = {
        i.bindings.get("VEHICLE")
        for i in produced
        if i.bindings.get("PERSON") == "P1"
    }
    assert advanced_vehicle_ids == {"V1", "V2"}


def test_all_uses_five_minute_semantic_expiry() -> None:
    event = load_and_compile_event(DEFINITIONS / "package_exchange.yaml")
    manager = CEInstanceManager(event)

    # The root all has two discovery leaves, so a vehicle observation may seed
    # either authored vehicle role.  Inspect one of the branches.
    produced = manager.handle_match(
        match("enters", T0 + timedelta(seconds=10), {"object": "V1"})
    )
    assert len(produced) == 2
    candidate = produced[0]
    frontier = derive_continuation_frontier(event, candidate, T0 + timedelta(seconds=10))
    assert len(frontier) == 1
    assert frontier[0].predicate == "enters"
    assert frontier[0].expires_at == T0 + timedelta(minutes=5, seconds=10)

    manager.expire(T0 + timedelta(minutes=5, seconds=11))
    assert not any(i.instance_id == candidate.instance_id for i in manager.active_instances())


def test_k_of_n_discovery_exposes_all_children_and_continuation_expires() -> None:
    event = compile_event(parse_event({
        "event": "two_of_three_sounds",
        "roles": {},
        "pattern": {
            "k_of_n": {
                "k": 2,
                "patterns": [
                    {"audio_event": {"class": "gunshot"}},
                    {"audio_event": {"class": "alarm"}},
                    {"audio_event": {"class": "gunshot"}},
                ],
            }
        },
    }))
    manager = CEInstanceManager(event)
    assert len(manager.discovery_frontier()) == 3

    produced = manager.handle_match(
        match("audio_event", T0 + timedelta(seconds=3), {"class": "alarm"}, source="mic_1")
    )
    assert len(produced) == 1
    frontier = derive_continuation_frontier(event, produced[0], T0 + timedelta(seconds=3))
    assert len(frontier) == 2
    assert all(
        item.expires_at == T0 + timedelta(minutes=5, seconds=3)
        for item in frontier
    )


def test_for_sustained_predicate_can_complete_candidate() -> None:
    event = load_and_compile_event(DEFINITIONS / "two_vehicle_chase.yaml")
    manager = CEInstanceManager(event)

    manager.handle_match(match("enters", T0 + timedelta(seconds=1), {"object": "leader"}))
    manager.handle_match(match(
        "follows",
        T0 + timedelta(seconds=2),
        {"leader": "leader", "follower": "follower"},
    ))
    manager.handle_match(match(
        "follows",
        T0 + timedelta(seconds=4),
        {"leader": "leader", "follower": "follower"},
    ))
    manager.handle_match(match(
        "follows",
        T0 + timedelta(seconds=5),
        {"leader": "leader", "follower": "follower"},
    ))

    assert any(
        i.bindings == {"LEADER": "leader", "FOLLOWER": "follower"}
        for i in manager.completed_instances()
    )


def test_different_roles_cannot_bind_same_object_identity() -> None:
    event = load_and_compile_event(DEFINITIONS / "two_vehicle_chase.yaml")
    manager = CEInstanceManager(event)
    manager.handle_match(match("enters", T0 + timedelta(seconds=1), {"object": "same"}))

    produced = manager.handle_match(match(
        "follows",
        T0 + timedelta(seconds=2),
        {"leader": "same", "follower": "same"},
    ))
    assert not any("FOLLOWER" in i.bindings for i in produced)


def test_same_match_can_advance_instance_and_seed_new_instance() -> None:
    # all(enters(A), enters(B)) is useful because the same enters observation is
    # simultaneously a continuation for one partial candidate and a valid new
    # seed for another candidate.
    event = compile_event(parse_event({
        "event": "two_arrivals",
        "roles": {
            "A": {"class": "vehicle"},
            "B": {"class": "vehicle"},
        },
        "pattern": {
            "all": [
                {"enters": {"object": "A"}},
                {"enters": {"object": "B"}},
            ]
        },
    }))
    manager = CEInstanceManager(event)
    manager.handle_match(match("enters", T0 + timedelta(seconds=1), {"object": "V1"}, classes={"object": "vehicle"}))
    produced = manager.handle_match(
        match("enters", T0 + timedelta(seconds=2), {"object": "V2"}, classes={"object": "vehicle"})
    )

    assert manager.completed_instances()  # V1/V2 pair completed in at least one ordering.
    assert any(i.matched_at == T0 + timedelta(seconds=2) for i in produced)
