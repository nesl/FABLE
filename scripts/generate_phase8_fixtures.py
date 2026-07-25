#!/usr/bin/env python3
"""Regenerate deterministic Phase-8 multimodal contract/planning fixtures."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.distributed.demo import build_replay_multimodal_candidate
from fable.planning.provider_registry import ProviderRegistry
from fable.semantic.phase8_examples import (
    drive_up_shooting_graph,
    multimodal_robbery_graph,
    package_exchange_graph,
)
from providers.multimodal.audio import AudioEventClassifier, DeterministicAudioEventBackend
from providers.multimodal.models import AudioWindow

PROVIDER_OUTPUT = ROOT / "providers/tests/phase8_fixtures"
TEST_OUTPUT = ROOT / "tests/phase8_fixtures"


def main() -> int:
    PROVIDER_OUTPUT.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    interval = EventTimeInterval(start=BASE_TIME, end=BASE_TIME + timedelta(milliseconds=100))
    audio = AudioWindow(
        source_id="mic_a",
        event_time_interval=interval,
        sample_rate_hz=16_000,
        channel_ids=("ch1", "ch2"),
        waveform=(tuple([0.1] * 1600), tuple([0.1] * 1600)),
        source_sequence=1,
    )
    events = AudioEventClassifier(
        DeterministicAudioEventBackend(
            {"Gunshot, gunfire": 0.91, "Fire alarm": 0.12}
        )
    ).classify(audio)
    (PROVIDER_OUTPUT / "typed_audio_event.json").write_text(
        json.dumps(events[0].model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    registry = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )
    candidates = {}
    for predicate, label in (
        ("AUDIO_EVENT", "gunshot"),
        ("DISEMBARKS", None),
        ("BOARDS", None),
        ("CONVERSATION", None),
        ("TRANSFER", None),
    ):
        plan = build_replay_multimodal_candidate(
            provider_registry=registry,
            node_id="dvpg_gq_orin_11",
            source_id="dvpg_gq_orin_11",
            event_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(minutes=5),
            ),
            predicate_id=predicate,
            label=label,
            now=BASE_TIME,
        )
        demand = plan.demands[0]
        candidates[predicate] = {
            "candidate_chain_ids": [
                item.chain_id for item in registry.candidate_chains(demand)
            ],
            "direct_provider_id": plan.plan.steps[0].provider_id,
            "continuation_output_types": list(
                plan.alternatives[0].continuation_output_types
            ),
        }
    document = {
        "predicates": candidates,
        "graphs": {
            "drive_up_shooting_hash": drive_up_shooting_graph().graph_hash,
            "robbery_hash": multimodal_robbery_graph().graph_hash,
            "package_exchange_hash": package_exchange_graph().graph_hash,
        },
        "audio_baseline_note": (
            "Deterministic fixture only; the replay spectral-rule backend is a deployment "
            "smoke-test baseline and not an evaluation classifier."
        ),
    }
    (TEST_OUTPUT / "multimodal_planning_summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
