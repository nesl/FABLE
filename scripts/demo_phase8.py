#!/usr/bin/env python3
"""Print a deterministic summary of Phase-8 semantic and physical choices."""

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
from fable.semantic import ScriptedResultSpec, SemanticRuntime, SemanticRuntimeConfig, seed_result_from_spec
from fable.semantic.phase8_examples import drive_up_shooting_graph, package_exchange_graph


def main() -> int:
    registry = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )
    graph = drive_up_shooting_graph()
    runtime = SemanticRuntime(
        graph,
        config=SemanticRuntimeConfig(request_id="phase8_demo"),
    )
    transition = runtime.seed(
        seed_result_from_spec(
            runtime,
            ScriptedResultSpec(
                node_key="gunshot",
                source_id="store_mic",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(milliseconds=100),
                ),
                introduced={"location": "store_front"},
            ),
        )
    )
    frontier = runtime.get_frontier(transition.hypothesis_ids[0])
    active = sorted(
        runtime.graph.nodes_by_id[item].authored_key
        for item in frontier.snapshot.enabled_node_ids
    )
    demands = {}
    for predicate in ("AUDIO_EVENT", "DISEMBARKS", "CONVERSATION", "TRANSFER"):
        candidate = build_replay_multimodal_candidate(
            provider_registry=registry,
            node_id="dvpg_gq_orin_11",
            source_id="dvpg_gq_orin_11",
            event_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(minutes=5),
            ),
            predicate_id=predicate,
            label="gunshot" if predicate == "AUDIO_EVENT" else None,
            now=BASE_TIME,
        )
        demand = candidate.demands[0]
        demands[predicate] = {
            "registered_alternatives": [
                chain.chain_id for chain in registry.candidate_chains(demand)
            ],
            "direct_replay_provider": candidate.plan.steps[0].provider_id,
            "continuation": list(candidate.alternatives[0].continuation_output_types),
        }
    package_graph = package_exchange_graph()
    transfer = next(item for item in package_graph.nodes if item.authored_key == "transfer")
    print(
        json.dumps(
            {
                "gunshot_checkpoint_next_frontier": active,
                "predicate_realizations": demands,
                "package_transfer_analysis_mode": transfer.annotations["analysis_mode"],
                "package_continuation": transfer.annotations[
                    "continuation_artifact_types"
                ],
                "raw_replay_audio_transport": "local IPC /tmp/respeaker.ipc",
                "default_audio_backend": "spectral-rule deployment smoke-test baseline",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
