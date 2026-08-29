from __future__ import annotations

from pathlib import Path

import pytest

from fable.language import (
    EventCompilationError,
    compile_event,
    load_and_compile_event,
    load_event,
    parse_event,
)
from fable.providers import (
    CURRENT_PUBLIC_PREDICATES,
    load_provider_capabilities,
    native_labels_for_visual_class,
    predicate_providers,
    semantic_literal_values,
    supported_visual_classes,
    visual_providers_for_class,
)


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT / "ce_definitions"


def _role(class_name: str) -> dict[str, str]:
    return {"class": class_name}


def test_all_current_ce_definitions_compile() -> None:
    paths = sorted(DEFINITIONS.glob("*.yaml"))
    assert len(paths) == 8
    for path in paths:
        event = load_and_compile_event(path)
        assert event.name == path.stem


def test_compile_returns_same_small_event_ast() -> None:
    event = load_event(DEFINITIONS / "two_vehicle_chase.yaml")
    compiled = compile_event(event)
    assert compiled is event


def test_every_public_predicate_has_enabled_implementation() -> None:
    capabilities = load_provider_capabilities()
    missing = {
        predicate
        for predicate in CURRENT_PUBLIC_PREDICATES
        if not predicate_providers(capabilities, predicate)
    }
    assert missing == set()


def test_coco_native_class_dog_is_supported() -> None:
    capabilities = load_provider_capabilities()
    providers = visual_providers_for_class(capabilities, "dog")
    assert "yolo_full_context_960" in providers
    assert native_labels_for_visual_class(capabilities, "dog")["yolo_full_context_960"] == ("dog",)


def test_composite_vehicle_class_maps_to_native_labels() -> None:
    capabilities = load_provider_capabilities()
    mappings = native_labels_for_visual_class(capabilities, "vehicle")
    assert mappings["yolo_vehicle_fast_640"] == ("car", "motorcycle", "bus", "truck")
    assert mappings["yolo_full_context_960"] == ("car", "motorcycle", "bus", "truck")


def test_package_semantic_class_is_supported() -> None:
    capabilities = load_provider_capabilities()
    mappings = native_labels_for_visual_class(capabilities, "package")
    assert mappings["package_detector"] == ("backpack", "handbag", "suitcase")


def test_native_labels_with_spaces_are_exposed_as_snake_case_semantic_classes() -> None:
    capabilities = load_provider_capabilities()
    assert "traffic_light" in supported_visual_classes(capabilities)
    mapping = native_labels_for_visual_class(capabilities, "traffic_light")
    assert mapping["yolo_full_context_960"] == ("traffic light",)


def test_parser_allows_open_visual_class_but_compiler_rejects_dragon() -> None:
    raw = {
        "event": "dragon_motion",
        "roles": {"DRAGON": _role("dragon")},
        "pattern": {"moving": {"object": "DRAGON"}},
    }
    event = parse_event(raw, source="dragon.yaml")
    assert event.roles["DRAGON"] == "dragon"
    with pytest.raises(EventCompilationError, match="unsupported visual class 'dragon'"):
        compile_event(event)


def test_supported_coco_class_compiles_without_language_change() -> None:
    raw = {
        "event": "dog_motion",
        "roles": {"DOG": _role("dog")},
        "pattern": {"moving": {"object": "DOG"}},
    }
    event = parse_event(raw, source="dog.yaml")
    assert compile_event(event) is event


def test_audio_class_support_is_declared_by_predicate_provider() -> None:
    capabilities = load_provider_capabilities()
    assert semantic_literal_values(capabilities, "audio_event", "class") == (
        "alarm",
        "gunshot",
    )


def test_compiler_rejects_unsupported_audio_class() -> None:
    raw = {
        "event": "dragon_roar",
        "roles": {},
        "pattern": {"audio_event": {"class": "dragon_roar"}},
    }
    event = parse_event(raw, source="dragon_roar.yaml")
    with pytest.raises(EventCompilationError, match="unsupported semantic value 'dragon_roar'"):
        compile_event(event)


def test_specialized_predicate_class_constraints_are_checked() -> None:
    raw = {
        "event": "dog_boards_car",
        "roles": {
            "DOG": _role("dog"),
            "CAR": _role("car"),
        },
        "pattern": {"boards": {"person": "DOG", "vehicle": "CAR"}},
    }
    event = parse_event(raw, source="dog_boards.yaml")
    with pytest.raises(EventCompilationError, match="none accepts the authored visual classes"):
        compile_event(event)


def test_near_is_generic_over_supported_visual_classes() -> None:
    raw = {
        "event": "dogs_near",
        "roles": {"DOG_A": _role("dog"), "DOG_B": _role("dog")},
        "pattern": {"near": {"object_a": "DOG_A", "object_b": "DOG_B"}},
    }
    event = parse_event(raw, source="dogs_near.yaml")
    assert compile_event(event) is event


def test_compiler_rejects_predicate_when_all_implementations_are_disabled() -> None:
    capabilities = load_provider_capabilities()
    capabilities["providers"]["near_geometry"]["enabled"] = False
    raw = {
        "event": "people_near",
        "roles": {"A": _role("person"), "B": _role("person")},
        "pattern": {"near": {"object_a": "A", "object_b": "B"}},
    }
    event = parse_event(raw, source="near.yaml")
    with pytest.raises(EventCompilationError, match="predicate 'near' has no enabled implementation provider"):
        compile_event(event, provider_capabilities=capabilities)


def test_compiler_rejects_visual_class_when_required_detectors_are_disabled() -> None:
    capabilities = load_provider_capabilities()
    capabilities["providers"]["yolo_full_context_960"]["enabled"] = False
    raw = {
        "event": "dog_motion",
        "roles": {"DOG": _role("dog")},
        "pattern": {"moving": {"object": "DOG"}},
    }
    event = parse_event(raw, source="dog.yaml")
    with pytest.raises(EventCompilationError, match="unsupported visual class 'dog'"):
        compile_event(event, provider_capabilities=capabilities)


def test_capability_loader_returns_independent_copies() -> None:
    first = load_provider_capabilities()
    second = load_provider_capabilities()
    first["providers"]["yolo_full_context_960"]["enabled"] = False
    assert second["providers"]["yolo_full_context_960"]["enabled"] is True
