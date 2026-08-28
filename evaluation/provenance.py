"""Reproducible, Git-independent provenance for evaluation results."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from pathlib import Path
import platform
import sys
from typing import Iterable, Mapping


PROVENANCE_SCHEMA_VERSION = "fable.evaluation_provenance.v1"

# These inputs define evaluation semantics, labels, replay selection, and the
# live request/result protocol.  The combined digest changes when any of them
# changes, without requiring the repository to be committed to Git.
DEFAULT_INPUTS = (
    "evaluation/catalog.py",
    "evaluation/live_requests.py",
    "evaluation/metrics/event_matching.py",
    "evaluation/planning_cases.py",
    "evaluation/replay_manifest.py",
    "evaluation/labels/filtered_complex_event_experiments.csv",
    "evaluation/labels/site_sensor_transition_model_2024_2025.json",
    "iobt-minimal-ce-replay/generated/scenario_catalog.json",
    "scripts/run_replay_accuracy.py",
)

INPUT_TREES = (
    "evaluation",
    "fable",
    "providers",
)

INPUT_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json", ".csv"})

DEFAULT_MODEL_CANDIDATES = (
    "iobt-minimal-replay/services/analytics/yolo_detector/app/yolov8s.pt",
    "iobt-minimal-replay/services/analytics/yolo_detector/app/yolov8n.pt",
    "iobt-minimal-ce-replay/services/analytics/yolo_detector/app/yolov8s.pt",
    "iobt-minimal-ce-replay/services/analytics/yolo_detector/app/yolov8n.pt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(
    root: Path,
    paths: Iterable[str],
) -> tuple[list[dict[str, object]], str]:
    """Return per-file metadata and a stable digest of the complete input set."""

    records: list[dict[str, object]] = []
    combined = hashlib.sha256()
    for relative in sorted(set(paths)):
        path = root / relative
        if not path.is_file():
            records.append({"path": relative, "status": "missing"})
            combined.update(f"{relative}\0missing\n".encode())
            continue
        digest = _sha256(path)
        size = path.stat().st_size
        records.append(
            {
                "path": relative,
                "status": "present",
                "sha256": digest,
                "size_bytes": size,
            }
        )
        combined.update(f"{relative}\0{digest}\0{size}\n".encode())
    return records, combined.hexdigest()


def discover_input_paths(root: Path) -> tuple[str, ...]:
    """Find result-affecting source/config files while excluding generated output."""

    discovered = set(DEFAULT_INPUTS)
    for tree in INPUT_TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if (
                path.is_file()
                and path.suffix in INPUT_SUFFIXES
                and "__pycache__" not in path.parts
                and "results" not in path.parts
                and "runs" not in path.parts
                and "schemas/json" not in path.as_posix()
            ):
                discovered.add(path.relative_to(root).as_posix())
    replay_root = root / "iobt-minimal-ce-replay"
    if replay_root.is_dir():
        for pattern in (
            "compose*.yaml",
            "config/*.yaml",
            "config/*.yml",
            "config/*.json",
            "services/**/Dockerfile",
            "services/**/app/*.py",
        ):
            discovered.update(
                path.relative_to(root).as_posix()
                for path in replay_root.glob(pattern)
                if path.is_file()
            )
    return tuple(sorted(discovered))


def build_run_provenance(
    root: Path,
    *,
    runner_arguments: Mapping[str, object],
    input_paths: Iterable[str] | None = None,
    model_candidates: Iterable[str] = DEFAULT_MODEL_CANDIDATES,
) -> dict[str, object]:
    """Build a self-contained provenance record without dumping environment data."""

    inputs, configuration_digest = fingerprint_files(
        root,
        discover_input_paths(root) if input_paths is None else input_paths,
    )
    models, model_digest = fingerprint_files(
        root,
        (
            candidate
            for candidate in model_candidates
            if (root / candidate).is_file()
        ),
    )
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "configuration_digest": configuration_digest,
        "model_digest": model_digest,
        "runner_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in sorted(runner_arguments.items())
        },
        "inputs": inputs,
        "models": models,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "changelog": "evaluation/CHANGELOG.md",
        "notes": (
            "No Git commit is required. Compare configuration_digest, model_digest, "
            "runner_arguments, and the changelog when comparing runs."
        ),
    }
