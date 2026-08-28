from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "netwaggle/scripts/make_netwaggle_compose.py"
SPEC = importlib.util.spec_from_file_location("make_netwaggle_compose", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_anchor_has_bounded_probe_dependencies_and_server() -> None:
    service = MODULE.anchor_service("netwaggle-node-orin11")
    assert service["image"] == "fable/netwaggle-anchor:alpine3.20"
    assert service["build"] == {
        "context": "../netwaggle",
        "dockerfile": "Dockerfile.anchor",
    }
    assert service["network_mode"] == "none"
    assert "iperf3 --server --daemon" in service["command"][2]
    health_command = service["healthcheck"]["test"][1]
    assert "command -v ping" in health_command
    assert "command -v iperf3" in health_command
