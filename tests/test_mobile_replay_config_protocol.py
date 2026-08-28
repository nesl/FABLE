from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "iobt-minimal-ce-replay/services/replay/mobile/app/config_protocol.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("mobile_config_protocol_test", PROTOCOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_for_mobile_node_is_accepted_without_a_playback_command() -> None:
    decision = _module().evaluate_config(
        {
            "action": "PROBE",
            "scenario": "trace",
            "target_nodes": ["mobile_archive_1"],
            "replay_id": "probe-1",
        },
        node_id="mobile_archive_1",
        loaded_scenario="trace",
        prior_signature=None,
    )
    assert decision.accepted
    assert decision.action == "PROBE"


def test_config_for_another_mobile_node_or_scenario_fails_closed() -> None:
    protocol = _module()
    wrong_node = protocol.evaluate_config(
        {"action": "PROBE", "scenario": "trace", "target_nodes": ["mobile_archive_2"]},
        node_id="mobile_archive_1",
        loaded_scenario="trace",
        prior_signature=None,
    )
    wrong_scenario = protocol.evaluate_config(
        {"action": "START", "scenario": "other", "target_nodes": ["mobile_archive_1"]},
        node_id="mobile_archive_1",
        loaded_scenario="trace",
        prior_signature=None,
    )
    assert (wrong_node.accepted, wrong_node.reason) == (False, "not_targeted")
    assert (wrong_scenario.accepted, wrong_scenario.reason) == (
        False,
        "scenario_mismatch",
    )


def test_duplicate_start_configuration_is_idempotently_identified() -> None:
    protocol = _module()
    payload = {
        "action": "START",
        "scenario": "trace",
        "replay_id": "run-1",
        "target_nodes": ["mobile_archive_1"],
        "playback_mode": "realtime",
        "speed": 1.0,
    }
    first = protocol.evaluate_config(
        payload,
        node_id="mobile_archive_1",
        loaded_scenario="trace",
        prior_signature=None,
    )
    second = protocol.evaluate_config(
        payload,
        node_id="mobile_archive_1",
        loaded_scenario="trace",
        prior_signature=first.signature,
    )
    assert first.reason == "accepted"
    assert second.reason == "duplicate_config"
    assert first.signature == second.signature


def test_unknown_configuration_action_is_rejected() -> None:
    decision = _module().evaluate_config(
        {"action": "SHELL", "target_nodes": ["mobile_archive_1"]},
        node_id="mobile_archive_1",
        loaded_scenario="trace",
        prior_signature=None,
    )
    assert not decision.accepted
    assert decision.reason == "unsupported_action"
