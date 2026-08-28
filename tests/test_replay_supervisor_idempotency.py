from __future__ import annotations

import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = (
    ROOT
    / "iobt-minimal-ce-replay/services/replay/common/replay_entrypoint.py"
)


def _supervisor_class():
    spec = importlib.util.spec_from_file_location(
        "replay_entrypoint_idempotency_test", ENTRYPOINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReplaySupervisor


def test_duplicate_retained_start_does_not_restart_running_child() -> None:
    supervisor_type = _supervisor_class()
    supervisor = supervisor_type.__new__(supervisor_type)
    supervisor.service = "zed"
    supervisor.node_name = "dvpg_gq_orin_13"
    supervisor.lock = threading.RLock()
    supervisor.last_config_payload = None
    supervisor.last_config_signature = None
    supervisor.last_sync_payload = {"scenario": "trace", "replay_id": "r1"}
    supervisor.child = None
    events = []
    supervisor._is_targeted = lambda payload: True
    supervisor._normalize_playback_timing = lambda payload: ("realtime", 1.0)
    supervisor.status = lambda **payload: events.append(
        ("status", payload.get("state"))
    )
    supervisor.error = lambda *args, **kwargs: events.append(("error", args))
    supervisor.stop_child = lambda: events.append(("stop", None))
    supervisor.prepare_symlinks = lambda scenario: True

    class RunningChild:
        @staticmethod
        def poll():
            return None

    def start_child(scenario, start, end, mode, speed):
        events.append(("start", (scenario, start, end, mode, speed)))
        supervisor.child = RunningChild()

    supervisor.start_child = start_child
    payload = {
        "action": "START",
        "scenario": "trace",
        "replay_id": "r1",
        "start_time": 0,
        "end_time": 85,
        "playback_mode": "realtime",
        "speed": 1.0,
    }

    supervisor.handle_config(payload)
    supervisor.handle_config(dict(payload))

    assert sum(event[0] == "stop" for event in events) == 1
    assert sum(event[0] == "start" for event in events) == 1
    assert ("status", "duplicate_config_ignored") in events


def test_explicit_stop_terminates_probe_child_without_starting_another() -> None:
    supervisor_type = _supervisor_class()
    supervisor = supervisor_type.__new__(supervisor_type)
    supervisor.service = "zed"
    supervisor.node_name = "dvpg_gq_orin_7"
    supervisor.lock = threading.RLock()
    supervisor.last_config_payload = {"scenario": "trace"}
    supervisor.last_config_signature = ("trace",)
    supervisor.last_sync_payload = {"scenario": "trace", "replay_id": "probe-1"}
    supervisor.child = object()
    events = []
    supervisor.status = lambda **payload: events.append(("status", payload.get("state")))
    supervisor.error = lambda *args, **kwargs: events.append(("error", args))
    supervisor.stop_child = lambda: events.append(("stop", None))
    supervisor.start_child = lambda *args: events.append(("start", args))

    supervisor.handle_config(
        {
            "action": "STOP",
            "replay_id": "probe-1",
            "target_nodes": ["orin7"],
        }
    )

    assert events == [("stop", None), ("status", "stopped")]
    assert supervisor.last_config_payload is None
    assert supervisor.last_config_signature is None
    assert supervisor.last_sync_payload is None


def test_config_for_another_target_does_not_touch_running_child() -> None:
    supervisor_type = _supervisor_class()
    supervisor = supervisor_type.__new__(supervisor_type)
    supervisor.service = "zed"
    supervisor.node_name = "dvpg_gq_orin_7"
    supervisor.lock = threading.RLock()
    supervisor.last_config_payload = None
    supervisor.last_config_signature = None
    supervisor.last_sync_payload = None
    supervisor.child = object()
    events = []
    supervisor.status = lambda **payload: events.append(("status", payload))
    supervisor.error = lambda *args, **kwargs: events.append(("error", args))
    supervisor.stop_child = lambda: events.append(("stop", None))
    supervisor.start_child = lambda *args: events.append(("start", args))

    supervisor.handle_config(
        {
            "action": "START",
            "scenario": "trace",
            "target_nodes": ["orin11"],
        }
    )

    assert events == []


def test_empty_retained_sync_cannot_seed_or_rebroadcast_probe_playback() -> None:
    supervisor_type = _supervisor_class()
    supervisor = supervisor_type.__new__(supervisor_type)

    assert supervisor._valid_sync_payload({"raw": ""}) is False
    assert supervisor._valid_sync_payload({"action": "STOP"}) is False
    assert supervisor._sync_matches_scenario("trace", {"raw": ""}) is False
    assert supervisor._sync_matches_scenario(
        "trace",
        {"scenario": "trace", "start_at": 123.5, "replay_id": "run-1"},
    ) is True


def test_one_sync_command_schedules_only_one_supervisor_rebroadcast() -> None:
    supervisor_type = _supervisor_class()
    supervisor = supervisor_type.__new__(supervisor_type)
    supervisor.service = "zed"
    supervisor.node_name = "dvpg_gq_orin_11"
    supervisor.lock = threading.RLock()
    supervisor.last_sync_payload = None
    supervisor.args = SimpleNamespace(sync_rebroadcast_delay=0.25)
    scheduled = []
    supervisor.status = lambda **_payload: None
    supervisor.rebroadcast_sync_after_child_ready = (
        lambda delay: scheduled.append(delay)
    )

    message = SimpleNamespace(
        topic="/replay/sync",
        payload=b'{"scenario":"trace","start_at":123.5,"replay_id":"run-1"}',
    )
    supervisor.on_message(None, None, message)

    assert scheduled == [0.25]
    assert supervisor.last_sync_payload["replay_id"] == "run-1"


def test_probe_stages_media_and_reports_ready_without_starting_child() -> None:
    supervisor_type = _supervisor_class()
    supervisor = supervisor_type.__new__(supervisor_type)
    supervisor.service = "zed"
    supervisor.node_name = "dvpg_gq_orin_7"
    supervisor.lock = threading.RLock()
    supervisor.last_config_payload = None
    supervisor.last_config_signature = None
    supervisor.last_sync_payload = None
    supervisor.child = None
    events = []
    supervisor.client = object()
    supervisor.prepare_symlinks = lambda scenario: events.append(
        ("prepare", scenario)
    ) or True
    supervisor.start_child = lambda *args: events.append(("start", args))
    supervisor.stop_child = lambda: events.append(("stop", None))
    supervisor.status = lambda **payload: events.append(
        ("status", payload.get("state"))
    )
    supervisor.error = lambda *args, **kwargs: events.append(("error", args))

    module_globals = supervisor_type.handle_config.__globals__
    original_publish = module_globals["publish_json"]
    module_globals["publish_json"] = lambda *args, **kwargs: events.append(
        ("publish", args[1])
    )
    try:
        supervisor.handle_config(
            {
                "action": "PROBE",
                "scenario": "trace",
                "target_nodes": ["orin7"],
            }
        )
    finally:
        module_globals["publish_json"] = original_publish

    assert ("prepare", "trace") in events
    assert ("publish", "/readiness/dvpg_gq_orin_7/zed") in events
    assert ("status", "probe_ready") in events
    assert not any(event[0] in {"start", "stop", "error"} for event in events)
