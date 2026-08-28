#!/usr/bin/env python3
"""Freeze a trace-specific B1 placement from a successful nominal execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "evaluation/manifests/baselines/static_pipelines.yaml"

# FABLE's adaptive realization names may differ from the equivalent authored
# B1 contract. Calibration freezes semantics, not an implementation-specific
# chain identifier.
FABLE_TO_B1_CHAIN = {
    "recover_vehicle_before_audio_event": "recover_vehicle_from_local_segments",
}

# A terminal result can bind roles from several cameras.  Those cameras are
# not interchangeable for a fixed pipeline: the seed camera is not
# necessarily where the pair converged, and the convergence camera is not
# necessarily where both departures were observed.  Preserve the causal role
# projection for chains whose semantics make that distinction explicit.
CHAIN_CAUSAL_ROLES = {
    "passes_live_vehicle": ("seed_vehicle", "leader", "reference"),
    "pairwise_distance_live_vehicle": ("vehicle_a", "vehicle_b"),
    "track_lifecycle_exit_live_vehicle": (
        "departing_vehicle_a",
        "departing_vehicle_b",
    ),
    "follows_local_tracks": ("leader", "follower"),
}

# Historical FABLE plans could record raw-camera inference at x86server.  The
# current raw-local alternative graph deliberately does not recreate those
# tuples.  B1 is an authored fixed pipeline, so freeze these source-local
# stages on the one causal camera instead of installing an impossible
# provider/node contract.  Site-only identity/audio models are intentionally
# absent from this set.
SOURCE_LOCAL_PROVIDERS = {
    "camera_projection",
    "multi_object_tracker",
    "pairwise_distance_evaluator",
    "pass_reference_evaluator",
    "track_lifecycle_exit_evaluator",
    "yolo_vehicle_fast_640",
    "yolo_vehicle_balanced_960",
}


def _records(path: Path, name: str) -> list[dict[str, object]]:
    record_path = path.with_suffix(".records") / name
    return [
        json.loads(line)
        for line in record_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _binding_nodes(predictions: list[dict[str, object]]) -> set[str]:
    nodes: set[str] = set()
    for prediction in predictions:
        if prediction.get("accepted") is not True:
            continue
        for value in (prediction.get("bindings") or {}).values():
            text = str(value)
            if text.startswith("dvpg_gq_orin_") and ":" in text:
                nodes.add(text.split(":", 1)[0])
    return nodes


def _binding_nodes_for_roles(
    predictions: list[dict[str, object]], roles: tuple[str, ...]
) -> set[str]:
    nodes: set[str] = set()
    for prediction in predictions:
        if prediction.get("accepted") is not True:
            continue
        bindings = prediction.get("bindings") or {}
        for role in roles:
            text = str(bindings.get(role) or "")
            if text.startswith("dvpg_gq_orin_") and ":" in text:
                nodes.add(text.split(":", 1)[0])
            elif text.startswith("mobile_archive_") and ":" in text:
                nodes.add(text.split(":", 1)[0])
    return nodes


def _source_node(source_id: str) -> str:
    local = source_id.removesuffix("_camera").removesuffix("_microphone")
    return f"dvpg_gq_orin_{local[4:]}" if local.startswith("orin") else local


def _camera_source(node_id: str) -> str:
    prefix = "dvpg_gq_orin_"
    if node_id.startswith(prefix):
        return f"orin{node_id[len(prefix):]}_camera"
    if node_id.startswith("mobile_archive_"):
        return f"{node_id}_camera"
    return ""


def derive(result_path: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    source_baseline = str(result.get("baseline") or "")
    if source_baseline not in {"FABLE", "B1_STATIC_WHOLE_EVENT"}:
        raise ValueError(
            "calibration result must use FABLE or B1_STATIC_WHOLE_EVENT"
        )
    if result.get("classification") != "TRUE_POSITIVE":
        raise ValueError("calibration result must be a true positive")
    if (result.get("condition_trace") or {}).get("transitions"):
        raise ValueError("calibration result must be nominal")

    observations = _records(result_path, "predicate_observation.jsonl")
    audio = [item for item in observations if item.get("predicate_id") == "AUDIO_EVENT"]
    accepted_predictions = [
        item
        for item in (result.get("predictions") or [])
        if item.get("accepted") is True
    ]
    terminal_nodes = _binding_nodes(accepted_predictions)
    terminal_hypothesis_ids = {
        str(item.get("hypothesis_id") or "") for item in accepted_predictions
    } - {""}
    required_nodes = set(terminal_nodes)
    # Preserve chain-specific causal roles. A fixed pipeline may use one
    # microphone and one camera, but that does not authorize every chain on the
    # union of those nodes (which would be hidden fan-out).
    audio_nodes = {
        str((item.get("metadata") or {}).get("node_id") or "") for item in audio
    }
    if audio and not audio_nodes:
        raise ValueError(
            "accepted result has audio evidence without an execution node; "
            "cannot freeze a coherent minimal B1 pipeline"
        )
    # Cross-sensor CEs intentionally bind evidence produced at different
    # nodes. Requiring the terminal visual binding to also be the audio node
    # rejects exactly the authored pipeline B1 is meant to represent. The
    # audio seed below freezes its own execution node, while visual/identity
    # chains are independently restricted to terminal-producing camera nodes.
    seed_node = ""
    seed_source = ""
    allowed_branch_ids: list[str] = []
    if audio:
        seed = max(audio, key=lambda item: float(item.get("confidence") or 0.0))
        seed_node = str((seed.get("metadata") or {}).get("node_id") or "")
        seed_source = str(seed.get("sensor_id") or "")
        if seed_node:
            required_nodes.add(seed_node)
        seed_time = datetime.fromisoformat(
            str(seed.get("event_time")).replace("Z", "+00:00")
        )
        for label, interval in (result.get("audio_event_time_ranges") or {}).items():
            if len(interval) != 2:
                continue
            start = datetime.fromisoformat(str(interval[0]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(interval[1]).replace("Z", "+00:00"))
            if start <= seed_time <= end:
                allowed_branch_ids.append(str(label))
    if not required_nodes:
        raise ValueError("could not recover causal nodes from the accepted result")

    decisions = _records(result_path, "plan_decision.jsonl")
    # Non-audio CEs do not have robbery's seed/retrospective relationship.
    # Select the latest complete plan for each chain whose source produced an
    # accepted terminal binding. Retain every provider placement in that plan:
    # a source-local predicate may legitimately use tracking on a peer and
    # inference at site-local compute.
    selected_decisions = decisions
    preserve_complete_plans = not audio
    if preserve_complete_plans:
        latest_by_chain_node: dict[tuple[str, str], dict[str, object]] = {}
        for decision in decisions:
            source_nodes = {
                _source_node(str(source))
                for source in decision.get("selected_source_ids") or ()
            }
            selected_nodes = {
                str(node) for node in decision.get("selected_node_ids") or ()
            }
            if terminal_nodes and not (
                source_nodes.intersection(terminal_nodes)
                or (not source_nodes and selected_nodes.intersection(terminal_nodes))
            ):
                continue
            for raw_chain in decision.get("selected_chain_ids") or ():
                chain = FABLE_TO_B1_CHAIN.get(str(raw_chain), str(raw_chain))
                role_nodes = _binding_nodes_for_roles(
                    accepted_predictions, CHAIN_CAUSAL_ROLES.get(chain, ())
                )
                causal_nodes = (
                    (source_nodes | selected_nodes)
                    & (role_nodes or terminal_nodes)
                )
                for node in causal_nodes:
                    latest_by_chain_node[(chain, node)] = decision
        selected_decisions = list(
            {id(item): item for item in latest_by_chain_node.values()}.values()
        )
        if not selected_decisions:
            raise ValueError(
                "could not recover a terminal-producing provider chain from "
                "the accepted result"
            )

    chains: list[str] = []
    providers: list[str] = []
    sources: list[str] = []
    chain_node_ids: dict[str, list[str]] = {}
    chain_provider_node_ids: dict[str, dict[str, list[str]]] = {}
    for decision in selected_decisions:
        selected_nodes = {str(item) for item in decision.get("selected_node_ids") or ()}
        decision_source_nodes = {
            _source_node(str(source))
            for source in decision.get("selected_source_ids") or ()
        }
        if preserve_complete_plans:
            # FABLE records an atomically admitted fan-out as one decision.
            # A terminal binding on one camera does not make every sibling
            # camera causal. Retain terminal-producing sensor nodes plus any
            # explicit site/cloud execution stages required by that chain.
            causal_sensor_nodes = (
                (decision_source_nodes & terminal_nodes)
                or (selected_nodes & terminal_nodes)
            )
            selected_nodes = causal_sensor_nodes | {
                node for node in selected_nodes if node in {"x86server", "cloud1"}
            }
        if not preserve_complete_plans and not selected_nodes.intersection(required_nodes):
            continue
        decision_chains = [
            FABLE_TO_B1_CHAIN.get(str(item), str(item))
            for item in decision.get("selected_chain_ids") or ()
        ]
        # Identity calibration follows the hypothesis that actually completed
        # the CE. Other FABLE identity decisions are bounded candidates, not
        # part of B1's single fixed authored pipeline.
        if (
            any(chain.startswith("same_entity_") for chain in decision_chains)
            and terminal_hypothesis_ids
            and str(decision.get("hypothesis_id") or "")
            not in terminal_hypothesis_ids
        ):
            continue
        for chain in decision_chains:
            if str(chain) not in chains:
                chains.append(str(chain))
            if preserve_complete_plans:
                role_nodes = _binding_nodes_for_roles(
                    accepted_predictions, CHAIN_CAUSAL_ROLES.get(chain, ())
                )
                selected = sorted(
                    (selected_nodes & role_nodes) if role_nodes else selected_nodes
                )
            elif chain == "detect_audio_event":
                selected = sorted({seed_node} if seed_node else set())
            elif chain.startswith("same_entity_"):
                # A fixed identity chain can span the causal camera and one
                # authored site matcher. Preserve that calibrated placement;
                # it is not fan-out and it cannot adapt at runtime.
                selected = sorted(selected_nodes)
            else:
                # The audio seed node is not a fallback placement for visual
                # successor chains. Freeze recovery/exit on the camera whose
                # concrete vehicle identity completed the accepted CE.
                selected = sorted(selected_nodes.intersection(terminal_nodes))
            if selected:
                chain_nodes = chain_node_ids.setdefault(chain, [])
                for node in selected:
                    if node not in chain_nodes:
                        chain_nodes.append(node)
                if preserve_complete_plans:
                    required_nodes.update(selected)
                elif chain.startswith("same_entity_"):
                    required_nodes.update(selected)
            exact = chain_provider_node_ids.setdefault(chain, {})
            for provider_key in decision.get("activated_provider_keys") or ():
                provider, separator, node = str(provider_key).partition("@")
                if not separator:
                    continue
                provider_nodes = set(selected)
                if (
                    provider in SOURCE_LOCAL_PROVIDERS
                    and node in {"x86server", "cloud1"}
                    and len(role_nodes if preserve_complete_plans else ()) == 1
                ):
                    # Normalize an obsolete raw offload onto its sole causal
                    # source camera. This is one fixed placement, not fan-out.
                    node = next(iter(role_nodes))
                    provider_nodes.add(node)
                if node in provider_nodes and node not in exact.setdefault(provider, []):
                    exact[provider].append(node)
        for source in decision.get("selected_source_ids") or ():
            source = str(source)
            node = _source_node(source)
            source_is_causal = (preserve_complete_plans and node in terminal_nodes) or (
                source.endswith("_microphone")
                and "detect_audio_event" in decision_chains
                and node == seed_node
            ) or (
                source.endswith("_camera")
                and any(chain != "detect_audio_event" for chain in decision_chains)
                and node in terminal_nodes
            )
            if source_is_causal and source not in sources:
                sources.append(source)
                if preserve_complete_plans:
                    required_nodes.add(node)
        if preserve_complete_plans:
            # Artifact-reuse plans can omit the original live source from the
            # decision record. Reconstruct the fixed camera source from the
            # accepted terminal binding instead of leaving B1 unrestricted.
            for node in sorted(causal_sensor_nodes):
                source = _camera_source(node)
                if source and source not in sources:
                    sources.append(source)
        for provider_key in decision.get("activated_provider_keys") or ():
            provider, _, node = str(provider_key).partition("@")
            if (
                (node in selected_nodes if preserve_complete_plans else node in required_nodes)
                and provider not in providers
            ):
                providers.append(provider)
                if preserve_complete_plans and node:
                    required_nodes.add(node)

    # Redesigned planning records can omit external source IDs. B1 still needs
    # exact endpoints rather than treating an empty source list as a wildcard.
    if seed_source:
        if seed_source not in sources:
            sources.append(seed_source)
    elif seed_node:
        prefix = "dvpg_gq_orin_"
        if seed_node.startswith(prefix):
            microphone = f"orin{seed_node[len(prefix):]}_microphone"
            if microphone not in sources:
                sources.append(microphone)
    for node in sorted(terminal_nodes):
        camera = _camera_source(node)
        if camera and camera not in sources:
            sources.append(camera)

    canonical = {
        "experiment_id": str(result["experiment_id"]),
        "trace_id": str(result["scenario"]),
        "allowed_chain_ids": chains,
        "allowed_provider_ids": providers,
        "allowed_node_ids": sorted(required_nodes),
        "allowed_source_ids": sources,
        "allowed_branch_ids": allowed_branch_ids,
        "allowed_chain_node_ids": chain_node_ids,
        "allowed_chain_provider_node_ids": chain_provider_node_ids,
    }
    placement_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **canonical,
        "source_fable_result": str(result_path.resolve()),
        "source_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "placement_sha256": placement_hash,
        "derived_at": datetime.now(UTC).isoformat(),
        "calibration_outcome": "TRUE_POSITIVE",
        "fanout_allowed": False,
        "adaptation_allowed": False,
    }


def install(result_path: Path, registry_path: Path) -> dict[str, object]:
    placement = derive(result_path)
    document = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    trace_id = str(placement["trace_id"])
    stored = dict(placement)
    stored.pop("trace_id")
    # placement_sha256 is retained in YAML for audit, while the runtime model
    # ignores unknown audit metadata under the repository's FableModel policy.
    document.setdefault("trace_placements", {})[trace_id] = stored
    registry_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return placement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    placement = derive(args.result) if args.check_only else install(args.result, args.registry)
    print(json.dumps(placement, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
