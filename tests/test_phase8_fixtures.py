from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase8_fixture_summary_contains_required_paths() -> None:
    document = json.loads(
        (ROOT / "tests/phase8_fixtures/multimodal_planning_summary.json").read_text()
    )
    # The historical recovery chain realizes VEHICLE_PRESENT_BEFORE and is
    # checked separately in the integration test.
    assert "audio_event_with_localization" in document["predicates"]["AUDIO_EVENT"]["candidate_chain_ids"]
    assert document["predicates"]["TRANSFER"]["continuation_output_types"] == ["custody_state.v1"]


def test_typed_audio_fixture_is_not_loud_audio_relabeling() -> None:
    document = json.loads(
        (ROOT / "providers/tests/phase8_fixtures/typed_audio_event.json").read_text()
    )
    assert document["label"] == "gunshot"
    assert document["attributes"]["backend_id"] == "deterministic_audio_backend"
