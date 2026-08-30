"""Durable, versioned completion artifacts and binary CE scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CompletionArtifact:
    event: str
    completed_at: datetime
    matched_at: datetime
    matched_source: str | None
    bindings: dict[str, str]
    cell_id: str | None = None
    schema_version: str = "fable.completion.v1"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["completed_at"] = self.completed_at.isoformat()
        value["matched_at"] = self.matched_at.isoformat()
        return value

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "CompletionArtifact":
        if raw.get("schema_version") != "fable.completion.v1":
            raise ValueError("unsupported completion artifact schema")
        return cls(
            event=str(raw["event"]),
            completed_at=datetime.fromisoformat(str(raw["completed_at"])),
            matched_at=datetime.fromisoformat(str(raw["matched_at"])),
            matched_source=None if raw.get("matched_source") is None else str(raw["matched_source"]),
            bindings={str(k): str(v) for k, v in dict(raw.get("bindings") or {}).items()},
            cell_id=None if raw.get("cell_id") is None else str(raw["cell_id"]),
        )


class CompletionWriter:
    """Append completions durably; one JSON object is one recoverable record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, artifact: CompletionArtifact) -> None:
        encoded = (json.dumps(artifact.to_dict(), sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def load_completions(path: str | Path) -> tuple[CompletionArtifact, ...]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(CompletionArtifact.from_dict(json.loads(line)))
    return tuple(rows)


@dataclass(frozen=True)
class BinaryScore:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0


def score_presence(expected_positive: bool, completions: Iterable[CompletionArtifact]) -> BinaryScore:
    predicted_positive = any(True for _ in completions)
    return BinaryScore(
        true_positive=int(expected_positive and predicted_positive),
        false_positive=int(not expected_positive and predicted_positive),
        false_negative=int(expected_positive and not predicted_positive),
        true_negative=int(not expected_positive and not predicted_positive),
    )
