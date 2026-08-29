from __future__ import annotations

import json
from pathlib import Path

import pytest

from fable.language import load_event, load_predicates, parse_event, walk_pattern


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT / "ce_definitions"


def _role(class_name: str) -> dict[str, str]:
    return {"class": class_name}


def test_all_ce_definitions_load() -> None:
    paths = sorted(DEFINITIONS.glob("*.yaml"))
    assert len(paths) == 8
    events = [load_event(path) for path in paths]
    assert {event.name for event in events} == {
        "convoy",
        "drive_up_shooting",
        "package_exchange",
        "repeated_visit",
        "rendezvous",
        "robbery",
        "two_vehicle_chase",
        "vehicle_convergence",
    }


def test_roles_are_object_roles_with_semantic_classes() -> None:
    event = load_event(DEFINITIONS / "package_exchange.yaml")
    assert event.roles == {
        "VEHICLE_A": "vehicle",
        "VEHICLE_B": "vehicle",
        "PERSON_A": "person",
        "PERSON_B": "person",
        "PACKAGE": "package",
    }


def test_old_scalar_role_syntax_is_rejected() -> None:
    bad = {
        "event": "old_roles",
        "roles": {"VEHICLE": "vehicle"},
        "pattern": {"present": {"object": "VEHICLE"}},
    }
    with pytest.raises(ValueError, match="expected an object such as"):
        parse_event(bad, source="old.yaml")


def test_role_class_vocabulary_is_open() -> None:
    raw = {
        "event": "dog_motion",
        "roles": {"DOG": _role("dog")},
        "pattern": {"moving": {"object": "DOG"}},
    }
    event = parse_event(raw, source="dog.yaml")
    assert event.roles["DOG"] == "dog"
    assert event.pattern.args == {"object": "DOG"}


def test_role_class_must_be_lowercase_identifier() -> None:
    bad = {
        "event": "bad_class",
        "roles": {"DOG": _role("Dog")},
        "pattern": {"present": {"object": "DOG"}},
    }
    with pytest.raises(ValueError, match="lowercase semantic class identifier"):
        parse_event(bad, source="bad.yaml")


def test_visual_object_arguments_accept_any_declared_object_class() -> None:
    raw = {
        "event": "mixed_near",
        "roles": {
            "PERSON": _role("person"),
            "BICYCLE": _role("bicycle"),
        },
        "pattern": {
            "near": {
                "object_a": "PERSON",
                "object_b": "BICYCLE",
            }
        },
    }
    event = parse_event(raw, source="mixed.yaml")
    assert event.pattern.args == {
        "object_a": "PERSON",
        "object_b": "BICYCLE",
    }


def test_unknown_uppercase_visual_object_role_is_reported() -> None:
    bad = {
        "event": "bad_role",
        "roles": {"LEADER": _role("vehicle")},
        "pattern": {"follows": {"leader": "LEADER", "follower": "FOLLOWER"}},
    }
    with pytest.raises(ValueError, match="unknown role 'FOLLOWER'"):
        parse_event(bad, source="bad.yaml")


def test_lowercase_literal_is_rejected_for_visual_object_argument() -> None:
    bad = {
        "event": "bad_role",
        "roles": {"LEADER": _role("vehicle")},
        "pattern": {"moving": {"object": "vehicle"}},
    }
    with pytest.raises(ValueError, match="expected an UPPERCASE visual-object role name"):
        parse_event(bad, source="bad.yaml")


def test_present_is_public_predicate() -> None:
    raw = {
        "event": "already_visible",
        "roles": {"VEHICLE": _role("vehicle")},
        "pattern": {"present": {"object": "VEHICLE"}},
    }
    event = parse_event(raw, source="present.yaml")
    assert event.pattern.op == "present"
    assert event.pattern.args == {"object": "VEHICLE"}


def test_chase_has_compact_seq_for_follows_tree() -> None:
    event = load_event(DEFINITIONS / "two_vehicle_chase.yaml")
    assert event.pattern.op == "seq"
    assert len(event.pattern.children) == 2

    seed, sustained_follow = event.pattern.children
    assert seed.op == "enters"
    assert seed.args == {"object": "LEADER"}

    assert sustained_follow.op == "for"
    assert sustained_follow.duration_ms == 3_000
    follows = sustained_follow.children[0]
    assert follows.op == "follows"
    assert follows.args == {
        "leader": "LEADER",
        "follower": "FOLLOWER",
        "max_gap_m": 30,
    }


def test_transfer_uses_unambiguous_argument_names() -> None:
    event = load_event(DEFINITIONS / "package_exchange.yaml")
    transfer = next(node for node in walk_pattern(event.pattern) if node.op == "transfer")
    assert transfer.args == {
        "item": "PACKAGE",
        "giver": "PERSON_A",
        "receiver": "PERSON_B",
    }


def test_old_near_argument_names_are_rejected() -> None:
    bad = {
        "event": "old_near",
        "roles": {"A": _role("person"), "B": _role("person")},
        "pattern": {"near": {"left": "A", "right": "B"}},
    }
    with pytest.raises(ValueError, match="unknown fields"):
        parse_event(bad, source="old_near.yaml")


def test_audio_event_uses_audio_class_literal() -> None:
    event = load_event(DEFINITIONS / "drive_up_shooting.yaml")
    gunshot = next(node for node in walk_pattern(event.pattern) if node.op == "audio_event")
    assert gunshot.args == {"class": "gunshot"}


def test_audio_class_must_be_lowercase_identifier() -> None:
    bad = {
        "event": "bad_audio",
        "roles": {},
        "pattern": {"audio_event": {"class": "Gunshot"}},
    }
    with pytest.raises(ValueError, match="lowercase audio class identifier"):
        parse_event(bad, source="bad_audio.yaml")


def test_predicate_catalog_uses_one_arguments_mapping() -> None:
    catalog = load_predicates()
    assert "arguments" in catalog["near"]
    assert "roles" not in catalog["near"]
    assert "parameters" not in catalog["near"]
    assert catalog["near"]["arguments"]["object_a"]["type"] == "visual_object"
    assert catalog["audio_event"]["arguments"]["class"]["type"] == "audio_class"


def test_within_parses_min_and_max_durations() -> None:
    event = load_event(DEFINITIONS / "repeated_visit.yaml")
    within = event.pattern.children[2]
    assert within.op == "within"
    assert within.min_ms == 30_000
    assert within.max_ms == 300_000


def test_all_has_default_five_minute_join_window() -> None:
    event = load_event(DEFINITIONS / "package_exchange.yaml")
    arrival_all = event.pattern.children[0]
    assert arrival_all.op == "all"
    assert arrival_all.window_ms == 5 * 60 * 1_000


def test_k_of_n_parses_with_default_join_window() -> None:
    raw = {
        "event": "two_of_three_motion",
        "roles": {
            "OBJECT_A": _role("car"),
            "OBJECT_B": _role("truck"),
            "OBJECT_C": _role("bicycle"),
        },
        "pattern": {
            "k_of_n": {
                "k": 2,
                "patterns": [
                    {"moving": {"object": "OBJECT_A"}},
                    {"moving": {"object": "OBJECT_B"}},
                    {"moving": {"object": "OBJECT_C"}},
                ],
            }
        },
    }
    event = parse_event(raw, source="k_of_n.yaml")
    assert event.pattern.op == "k_of_n"
    assert event.pattern.k == 2
    assert len(event.pattern.children) == 3
    assert event.pattern.window_ms == 5 * 60 * 1_000


@pytest.mark.parametrize("k", [0, 4])
def test_k_of_n_rejects_invalid_k(k: int) -> None:
    raw = {
        "event": "bad_k",
        "roles": {
            "OBJECT_A": _role("car"),
            "OBJECT_B": _role("truck"),
            "OBJECT_C": _role("bicycle"),
        },
        "pattern": {
            "k_of_n": {
                "k": k,
                "patterns": [
                    {"moving": {"object": "OBJECT_A"}},
                    {"moving": {"object": "OBJECT_B"}},
                    {"moving": {"object": "OBJECT_C"}},
                ],
            }
        },
    }
    with pytest.raises(ValueError, match="must be between 1 and the number of patterns"):
        parse_event(raw, source="bad_k.yaml")


def test_passes_is_not_a_public_predicate() -> None:
    bad = {
        "event": "ambiguous_pass",
        "roles": {"VEHICLE": _role("vehicle")},
        "pattern": {"passes": {"object": "VEHICLE"}},
    }
    with pytest.raises(ValueError, match="unknown predicate 'passes'"):
        parse_event(bad, source="passes.yaml")


def test_legacy_unimplemented_predicate_is_not_public() -> None:
    bad = {
        "event": "bad_event",
        "roles": {"PERSON": _role("person")},
        "pattern": {"suspicious_entry": {"object": "PERSON"}},
    }
    with pytest.raises(ValueError, match="unknown predicate 'suspicious_entry'"):
        parse_event(bad, source="bad.yaml")


def test_group_requires_at_least_two_children() -> None:
    bad = {
        "event": "bad_group",
        "roles": {"VEHICLE": _role("vehicle")},
        "pattern": {"seq": [{"enters": {"object": "VEHICLE"}}]},
    }
    with pytest.raises(ValueError, match="requires at least two child patterns"):
        parse_event(bad, source="bad.yaml")


def test_json_is_accepted_through_same_loader(tmp_path: Path) -> None:
    raw = {
        "version": 1,
        "event": "json_example",
        "roles": {"VEHICLE": _role("vehicle")},
        "pattern": {"moving": {"object": "VEHICLE"}},
    }
    path = tmp_path / "json_example.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    event = load_event(path)
    assert event.name == "json_example"
    assert event.pattern.op == "moving"


def test_language_definitions_do_not_name_physical_resources() -> None:
    banned = {
        "camera_id",
        "node14",
        "node15",
        "provider",
        "artifact_type",
        "source_id",
        "zone",
        "reference_line",
    }
    for path in DEFINITIONS.glob("*.yaml"):
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, f"{path.name} contains banned physical term {word!r}"


def test_pattern_parser_module_is_the_parser_home() -> None:
    from fable.language import pattern_parser

    assert pattern_parser.parse_pattern is not None
    assert pattern_parser.DEFAULT_JOIN_WINDOW_MS == 5 * 60 * 1_000
