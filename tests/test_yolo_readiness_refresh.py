from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
YOLO_APP = (
    ROOT
    / "iobt-minimal-ce-replay/services/analytics/yolo_detector/app/app.py"
)


def test_replay_config_callback_refreshes_yolo_readiness_without_starting_replay() -> None:
    """Keep the replay barrier's refresh protocol wired into YOLO.

    Parse the callback rather than importing the detector image's heavyweight
    CUDA/Ultralytics dependencies into the host unit-test environment.
    """

    tree = ast.parse(YOLO_APP.read_text(encoding="utf-8"))
    callback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "get_replay_config"
    )
    calls = [
        node
        for node in ast.walk(callback)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "publish_ready_state"
    ]
    assert len(calls) == 1
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "replay_requested"
            for target in node.targets
        )
        for node in ast.walk(callback)
    )
