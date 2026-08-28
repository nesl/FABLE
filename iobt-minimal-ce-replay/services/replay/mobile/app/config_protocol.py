"""Pure control-plane validation for the fixed-media mobile replay service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ConfigDecision:
    action: str
    accepted: bool
    reason: str
    signature: tuple[Any, ...] | None = None


def _target_aliases(node_id: str) -> set[str]:
    aliases = {node_id}
    prefix = "dvpg_gq_orin_"
    if node_id.startswith(prefix):
        aliases.add(f"orin{node_id.removeprefix(prefix)}")
    return aliases


def evaluate_config(
    payload: Mapping[str, Any],
    *,
    node_id: str,
    loaded_scenario: str,
    prior_signature: tuple[Any, ...] | None,
) -> ConfigDecision:
    action = str(payload.get("action") or "START").strip().upper()
    if action not in {"START", "PROBE", "STOP"}:
        return ConfigDecision(action, False, "unsupported_action")
    targets = payload.get("target_nodes")
    if isinstance(targets, (list, tuple)) and targets:
        if not _target_aliases(node_id).intersection(map(str, targets)):
            return ConfigDecision(action, False, "not_targeted")
    scenario = str(payload.get("scenario") or "").strip()
    if action != "STOP" and scenario and scenario != loaded_scenario:
        return ConfigDecision(action, False, "scenario_mismatch")
    if action == "STOP":
        return ConfigDecision(action, True, "stopped", None)
    signature = (
        action,
        scenario or loaded_scenario,
        str(payload.get("replay_id") or ""),
        payload.get("start_time"),
        payload.get("end_time"),
        str(payload.get("playback_mode") or ""),
        payload.get("speed"),
        tuple(sorted(map(str, targets or ()))),
    )
    reason = "duplicate_config" if signature == prior_signature else "accepted"
    return ConfigDecision(action, True, reason, signature)

