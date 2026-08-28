from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE_BASE = ROOT / "iobt-minimal-ce-replay/lib/iobt_max_service.py"
RESPEAKER_REPLAY = (
    ROOT / "iobt-minimal-ce-replay/services/replay/respeaker/app/app.py"
)


def test_replay_service_timestamp_serialization_is_explicit_utc() -> None:
    """Guard against host-local timestamps being consumed as replay UTC."""

    tree = ast.parse(SERVICE_BASE.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "ts_to_string"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="datetime",
                names=[
                    ast.alias(name="datetime"),
                    ast.alias(name="timezone"),
                ],
                level=0,
            ),
            ast.FunctionDef(
                name="serialize",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="self"), ast.arg(arg="ts")],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=method.body,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(SERVICE_BASE), "exec"), namespace)
    serialize = namespace["serialize"]

    rendered = serialize(None, 0.0)  # type: ignore[operator]

    assert rendered == "1970-01-01T00:00:00.000000Z"
    parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    assert parsed.tzinfo == timezone.utc


def test_respeaker_replay_uses_source_elapsed_event_time() -> None:
    """Max-speed audio must remain aligned with camera source time."""

    tree = ast.parse(RESPEAKER_REPLAY.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "timestamp"
            for target in node.targets
        )
    ]

    assert any(
        isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Add)
        and isinstance(node.value.left, ast.Attribute)
        and node.value.left.attr == "event_start_at"
        and isinstance(node.value.right, ast.Name)
        and node.value.right.id == "frame_elapsed_original"
        for node in assignments
    )
