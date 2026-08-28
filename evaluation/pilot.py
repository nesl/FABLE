"""Manifest and reporting support for bounded live replay pilots."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
import yaml

from fable.common.base import FableModel


class PilotSettings(FableModel):
    baseline: str = "FABLE"
    model_id: str = "unspecified"
    repetitions: int = Field(default=2, ge=1, le=10)
    max_seconds: float = Field(default=180, gt=0, le=300)
    ready_seconds: float = Field(default=30, ge=0)
    playback_mode: Literal["max", "realtime", "scaled"] = "realtime"
    playback_speed: float = Field(default=1.0, gt=0)
    deadline_seconds: float = Field(default=30, ge=0)
    minimum_temporal_iou: float = Field(default=0.1, ge=0, le=1)
    # Raw replay readiness is independent of plan-selected analytics leases.
    required_ready_services: str = "zed"
    replay_nodes: tuple[str, ...] = ()
    inter_run_seconds: float = Field(default=5, ge=0, le=60)
    process_grace_seconds: float = Field(default=20, ge=1, le=60)


class PilotCase(FableModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    family: str
    experiment_id: str | None = None
    scenario: str | None = None
    variant: str | None = None
    expected_positive: bool = True
    repetitions: int | None = Field(default=None, ge=1, le=10)
    replay_nodes: tuple[str, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def _validate_selector(self) -> "PilotCase":
        if self.experiment_id:
            if self.scenario or self.variant:
                raise ValueError(
                    "experiment_id cannot be combined with scenario or variant"
                )
            if not self.expected_positive:
                raise ValueError("experiment cases must be expected positives")
        elif not self.scenario or not self.variant:
            raise ValueError(
                "controls require scenario and variant when experiment_id is absent"
            )
        return self


class PilotManifest(FableModel):
    schema_version: Literal["fable.bounded_pilot.v1"]
    pilot_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    settings: PilotSettings
    cases: tuple[PilotCase, ...]

    @model_validator(mode="after")
    def _unique_cases(self) -> "PilotManifest":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("pilot case IDs must be unique")
        return self


def load_pilot_manifest(path: Path) -> PilotManifest:
    return PilotManifest.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def classify_failure(result: dict[str, object]) -> str:
    classification = str(result.get("classification") or "")
    if classification in {"TRUE_POSITIVE", "NOT_DETECTED"}:
        return "NONE"
    error = str(result.get("error") or "").lower()
    if (
        not result.get("watch_registered")
        or not result.get("admitted")
        or any(
            marker in error
            for marker in (
                "readiness",
                "connect",
                "timed out registering",
                "cancel",
                "infrastructure",
            )
        )
    ):
        return "RUNTIME"
    if result.get("detected"):
        return "METRIC_MATCH"
    progress = result.get("progress_statuses") or {}
    if isinstance(progress, dict) and int(progress.get("APPLIED", 0) or 0) > 0:
        return "SEMANTIC_GRAPH"
    variant = str(result.get("variant") or "").lower()
    vehicle = result.get("vehicle_predicates_by_id") or {}
    interaction = result.get("interaction_predicates_by_id") or {}
    audio = result.get("audio_events_by_label") or {}
    if not isinstance(vehicle, dict):
        vehicle = {}
    if not isinstance(interaction, dict):
        interaction = {}
    if not isinstance(audio, dict):
        audio = {}
    # An admitted seed plus all major predicate families means the failure is
    # temporal/identity graph assembly, not absence of low-level evidence.
    semantic_evidence = False
    if "stalking" in variant:
        semantic_evidence = all(int(vehicle.get(key, 0) or 0) > 0 for key in ("ENTERS", "EXITS"))
    elif "pass-follow-clear" in variant:
        semantic_evidence = all(int(vehicle.get(key, 0) or 0) > 0 for key in ("PASSES", "FOLLOWS"))
    elif "robbery" in variant:
        semantic_evidence = (
            any(int(value or 0) > 0 for value in audio.values())
            and any(int(value or 0) > 0 for value in interaction.values())
        )
    elif "vehicle rendezvous" in variant:
        semantic_evidence = int(vehicle.get("DISTANCE_LT", 0) or 0) > 0
    elif "rendezvous" in variant or "talking" in variant:
        semantic_evidence = int(interaction.get("PERSON_PROXIMITY", 0) or 0) > 0
    if result.get("admitted") and semantic_evidence:
        return "SEMANTIC_GRAPH"
    observed = result.get("observed_messages") or {}
    if isinstance(observed, dict) and any(
        int(observed.get(key, 0) or 0) > 0
        for key in ("audio_events", "context_tracks", "vehicle_predicates", "yolo")
    ):
        return "PREDICATE"
    return "SENSOR_EVIDENCE"


def generate_pilot_report(
    result_dir: Path,
    manifest: PilotManifest,
) -> dict[str, object]:
    cases = {case.case_id: case for case in manifest.cases}
    rows: list[dict[str, object]] = []
    for path in sorted(result_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = document.get("pilot") or {}
        if not isinstance(metadata, dict):
            continue
        case_id = str(metadata.get("case_id") or "")
        if case_id not in cases:
            continue
        classification = str(document.get("classification") or "UNKNOWN")
        case = cases[case_id]
        passed = (
            classification == "TRUE_POSITIVE"
            if case.expected_positive
            else classification == "NOT_DETECTED"
        )
        provenance = document.get("provenance") or {}
        timing = document.get("timing") or {}
        rows.append(
            {
                "case_id": case_id,
                "family": case.family,
                "repetition": metadata.get("repetition"),
                "expected_positive": case.expected_positive,
                "classification": classification,
                "passed": passed,
                "failure_layer": "NONE" if passed else classify_failure(document),
                "elapsed_seconds": document.get("elapsed_seconds"),
                "timing": timing if isinstance(timing, dict) else {},
                "pilot_process_wall_seconds": metadata.get(
                    "process_wall_seconds"
                ),
                "configuration_digest": (
                    provenance.get("configuration_digest")
                    if isinstance(provenance, dict)
                    else None
                ),
                "model_digest": (
                    provenance.get("model_digest")
                    if isinstance(provenance, dict)
                    else None
                ),
                "model_id": (
                    (provenance.get("runner_arguments") or {}).get("model_id")
                    if isinstance(provenance, dict)
                    and isinstance(provenance.get("runner_arguments"), dict)
                    else None
                ),
                "result_path": str(path),
            }
        )

    by_family: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    for family, family_rows in sorted(grouped.items()):
        passes = sum(bool(row["passed"]) for row in family_rows)
        positive_rows = [
            row for row in family_rows if bool(row["expected_positive"])
        ]
        control_rows = [
            row for row in family_rows if not bool(row["expected_positive"])
        ]
        positive_passes = sum(bool(row["passed"]) for row in positive_rows)
        control_passes = sum(bool(row["passed"]) for row in control_rows)
        by_family[family] = {
            "runs": len(family_rows),
            "passes": passes,
            "pass_rate": round(passes / len(family_rows), 4),
            "positive_runs": len(positive_rows),
            "positive_passes": positive_passes,
            "positive_recall": (
                round(positive_passes / len(positive_rows), 4)
                if positive_rows
                else None
            ),
            "control_runs": len(control_rows),
            "control_passes": control_passes,
            "control_specificity": (
                round(control_passes / len(control_rows), 4)
                if control_rows
                else None
            ),
            "classifications": dict(
                sorted(Counter(str(row["classification"]) for row in family_rows).items())
            ),
            "failure_layers": dict(
                sorted(
                    Counter(
                        str(row["failure_layer"])
                        for row in family_rows
                        if row["failure_layer"] != "NONE"
                    ).items()
                )
            ),
        }
    configuration_digests = sorted(
        {str(row["configuration_digest"]) for row in rows if row["configuration_digest"]}
    )
    model_digests = sorted(
        {str(row["model_digest"]) for row in rows if row["model_digest"]}
    )
    model_ids = sorted({str(row["model_id"]) for row in rows if row["model_id"]})
    expected_model_id = manifest.settings.model_id
    model_declaration_complete = all(
        row["model_id"] == expected_model_id for row in rows
    )
    return {
        "schema_version": "fable.bounded_pilot_report.v1",
        "pilot_id": manifest.pilot_id,
        "planned_runs": sum(
            case.repetitions or manifest.settings.repetitions
            for case in manifest.cases
        ),
        "completed_runs": len(rows),
        "configuration_consistent": len(configuration_digests) <= 1,
        "model_consistent": (
            len(model_digests) <= 1
            and len(model_ids) <= 1
            and model_declaration_complete
        ),
        "expected_model_id": expected_model_id,
        "model_declaration_complete": model_declaration_complete,
        "configuration_digests": configuration_digests,
        "model_digests": model_digests,
        "model_ids": model_ids,
        "by_family": by_family,
        "runs": rows,
    }
