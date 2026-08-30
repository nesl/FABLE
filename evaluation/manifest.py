"""Small, versioned evaluation-manifest contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_POLICIES = frozenset({"FABLE", "B1_STATIC", "B3_RESOURCE", "B4_GREEDY"})


@dataclass(frozen=True)
class EvaluationCell:
    cell_id: str
    event: Path
    deployment: Path
    policy: str = "FABLE"
    repetition: int = 1
    replay: Path | None = None
    condition: Path | None = None
    static_placements: Path | None = None


@dataclass(frozen=True)
class EvaluationManifest:
    version: int
    name: str
    cells: tuple[EvaluationCell, ...]


def _resolve(base: Path, value: object | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: str | Path) -> EvaluationManifest:
    source = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if int(raw.get("version", 0)) != 1:
        raise ValueError("evaluation manifest version must be 1")
    cells: list[EvaluationCell] = []
    seen: set[str] = set()
    for row in raw.get("cells", ()):
        cell_id = str(row["id"])
        if cell_id in seen:
            raise ValueError(f"duplicate evaluation cell id {cell_id!r}")
        seen.add(cell_id)
        policy = str(row.get("policy", "FABLE"))
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported evaluation policy {policy!r}")
        event = _resolve(source.parent, row["event"])
        deployment = _resolve(source.parent, row["deployment"])
        assert event is not None and deployment is not None
        if not event.is_file() or not deployment.is_file():
            raise FileNotFoundError(f"cell {cell_id!r} references a missing input")
        cells.append(
            EvaluationCell(
                cell_id=cell_id,
                event=event,
                deployment=deployment,
                policy=policy,
                repetition=int(row.get("repetition", 1)),
                replay=_resolve(source.parent, row.get("replay")),
                condition=_resolve(source.parent, row.get("condition")),
                static_placements=_resolve(source.parent, row.get("static_placements")),
            )
        )
    if not cells:
        raise ValueError("evaluation manifest must contain at least one cell")
    return EvaluationManifest(1, str(raw.get("name") or source.stem), tuple(cells))
