#!/usr/bin/env python3
"""Scenario catalog scanner for IoBT replay data roots.

Scans parent directories that contain date folders like 20260413/orin11/... and
produces a compact catalog of replayable scenario IDs plus available sensors,
devices, and best-effort start/end datetimes.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCENARIO_RE = re.compile(r"(?P<sid>\d{8}_\d{6})")
DATE_DIR_RE = re.compile(r"\d{8}")
TIMESTAMP_COLUMNS = ["Timestamp", "timestamp", "DateTime", "datetime", "Datetime", "time", "Time"]
DEFAULT_HOST_DATA_ROOTS = [
    Path("/media/brianw/Extreme SSD/West Point Experimentation"),
    Path("/media/brianw/Extreme SSD/GQ Data"),
]
DEFAULT_CONTAINER_DATA_ROOTS = [
    Path("/data_roots/west_point"),
    Path("/data_roots/gq"),
]


def parse_roots(raw: str | None = None, *, container_default: bool = False) -> list[Path]:
    if raw is None:
        raw = os.environ.get("IOBT_DATA_ROOTS") or os.environ.get("IOBT_HOST_DATA_ROOTS")
    if raw:
        roots = [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]
    else:
        roots = DEFAULT_CONTAINER_DATA_ROOTS if container_default else DEFAULT_HOST_DATA_ROOTS
    out: list[Path] = []
    seen: set[str] = set()
    for p in roots:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def root_label(root: Path) -> str:
    name = root.name.strip() or str(root)
    lowered = str(root).lower()
    if "west" in lowered:
        return "west_point"
    if "gq" in lowered:
        return "gq"
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "data_root"


def scenario_start_from_id(scenario_id: str) -> str | None:
    try:
        return datetime.strptime(scenario_id, "%Y%m%d_%H%M%S").isoformat(timespec="seconds")
    except ValueError:
        return None


def parse_datetime_value(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # pandas-style strings often include timezone Z or fractional seconds.
    text = text.replace("Z", "+00:00")
    candidates = [text]
    if "/" in text:
        candidates.append(text.replace("/", "-"))
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    # Common fallback formats seen in logs/CSV exports.
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def csv_first_last_timestamp(path: Path, *, max_rows: int | None = None) -> tuple[datetime | None, datetime | None]:
    """Best-effort first/last timestamp from a CSV without loading it into memory."""
    try:
        with path.open("r", newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None, None
            column = next((c for c in TIMESTAMP_COLUMNS if c in reader.fieldnames), None)
            if column is None:
                return None, None
            first: datetime | None = None
            last: datetime | None = None
            for i, row in enumerate(reader):
                ts = parse_datetime_value(row.get(column))
                if ts is not None:
                    if first is None:
                        first = ts
                    last = ts
                if max_rows is not None and i >= max_rows:
                    break
            return first, last
    except Exception:
        return None, None


def detect_sensor(path: Path) -> str | None:
    name = path.name.lower()
    if "zed" in name and (name.endswith(".svo") or name.endswith(".svo2") or name.endswith(".csv") or name.endswith(".mp4")):
        return "zed"
    if "respeaker" in name and (name.endswith(".flac") or name.endswith(".csv") or name.endswith(".wav")):
        return "respeaker"
    if "gps" in name and name.endswith(".csv"):
        return "gps"
    return None


@dataclass
class ScenarioRecord:
    scenario_id: str
    date: str
    source_root: str
    source_label: str
    date_dir: str
    start_datetime: str | None = None
    observed_start_datetime: str | None = None
    observed_end_datetime: str | None = None
    duration_seconds: float | None = None
    zed_nodes: set[str] = field(default_factory=set)
    respeaker_nodes: set[str] = field(default_factory=set)
    gps_objects: set[str] = field(default_factory=set)
    file_count: int = 0
    zed_file_count: int = 0
    respeaker_file_count: int = 0
    gps_file_count: int = 0
    sample_files: list[str] = field(default_factory=list)

    def add_file(self, path: Path, sensor: str, date_dir: Path, node_or_object: str | None) -> None:
        self.file_count += 1
        rel = str(path.relative_to(date_dir)) if path.is_relative_to(date_dir) else str(path)
        if len(self.sample_files) < 12:
            self.sample_files.append(rel)
        if sensor == "zed":
            self.zed_file_count += 1
            if node_or_object:
                self.zed_nodes.add(node_or_object)
        elif sensor == "respeaker":
            self.respeaker_file_count += 1
            if node_or_object:
                self.respeaker_nodes.add(node_or_object)
        elif sensor == "gps":
            self.gps_file_count += 1
            if node_or_object:
                self.gps_objects.add(node_or_object)
        if path.suffix.lower() == ".csv":
            first, last = csv_first_last_timestamp(path)
            self._merge_observed_times(first, last)

    def _merge_observed_times(self, first: datetime | None, last: datetime | None) -> None:
        cur_first = parse_datetime_value(self.observed_start_datetime)
        cur_last = parse_datetime_value(self.observed_end_datetime)
        if first is not None and (cur_first is None or first < cur_first):
            self.observed_start_datetime = first.isoformat()
        if last is not None and (cur_last is None or last > cur_last):
            self.observed_end_datetime = last.isoformat()
        a = parse_datetime_value(self.observed_start_datetime)
        b = parse_datetime_value(self.observed_end_datetime)
        if a is not None and b is not None and b >= a:
            self.duration_seconds = round((b - a).total_seconds(), 3)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["zed_nodes"] = sorted(self.zed_nodes)
        d["respeaker_nodes"] = sorted(self.respeaker_nodes)
        d["gps_objects"] = sorted(self.gps_objects)
        d["nodes"] = sorted(set(d["zed_nodes"]) | set(d["respeaker_nodes"]))
        d["modalities"] = [m for m, n in (("zed", self.zed_file_count), ("respeaker", self.respeaker_file_count), ("gps", self.gps_file_count)) if n]
        d["valid"] = bool(self.file_count)
        return d


def iter_candidate_files(date_dir: Path) -> Iterable[tuple[Path, str, str | None]]:
    """Yield (path, sensor, node_or_object) for files that look replay-relevant."""
    try:
        children = list(date_dir.iterdir())
    except OSError:
        return
    for child in children:
        if not child.is_dir():
            continue
        if child.name == "GPS":
            for obj_dir in child.iterdir() if child.exists() else []:
                if not obj_dir.is_dir():
                    continue
                for path in obj_dir.iterdir():
                    if path.is_file():
                        sensor = detect_sensor(path)
                        if sensor == "gps":
                            yield path, sensor, obj_dir.name
        else:
            for path in child.iterdir():
                if not path.is_file():
                    continue
                sensor = detect_sensor(path)
                if sensor is not None:
                    yield path, sensor, child.name


def scan_scenarios(roots: Iterable[Path]) -> list[dict[str, Any]]:
    records: dict[tuple[str, str], ScenarioRecord] = {}
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        label = root_label(root)
        try:
            date_dirs = sorted(p for p in root.iterdir() if p.is_dir() and DATE_DIR_RE.fullmatch(p.name))
        except OSError:
            continue
        for date_dir in date_dirs:
            for path, sensor, node_or_object in iter_candidate_files(date_dir):
                match = SCENARIO_RE.search(path.name)
                if not match:
                    continue
                scenario_id = match.group("sid")
                key = (str(root), scenario_id)
                rec = records.get(key)
                if rec is None:
                    rec = ScenarioRecord(
                        scenario_id=scenario_id,
                        date=scenario_id.split("_")[0],
                        source_root=str(root),
                        source_label=label,
                        date_dir=str(date_dir),
                        start_datetime=scenario_start_from_id(scenario_id),
                    )
                    records[key] = rec
                rec.add_file(path, sensor, date_dir, node_or_object)
    out = [rec.to_dict() for rec in records.values() if rec.file_count]
    out.sort(key=lambda r: (r.get("start_datetime") or r["scenario_id"], r["source_label"], r["scenario_id"]))
    return out


def write_catalog(records: list[dict[str, Any]], output_dir: Path, *, roots: Iterable[Path]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "roots": [str(Path(r)) for r in roots],
        "count": len(records),
    }
    json_path = output_dir / "scenario_catalog.json"
    csv_path = output_dir / "scenario_catalog.csv"
    payload = {"metadata": metadata, "scenarios": records}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "scenario_id", "source_label", "date", "start_datetime", "observed_start_datetime",
        "observed_end_datetime", "duration_seconds", "nodes", "zed_nodes",
        "respeaker_nodes", "gps_objects", "modalities", "file_count", "zed_file_count",
        "respeaker_file_count", "gps_file_count", "date_dir", "source_root",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k) for k in fieldnames}
            for k in ("nodes", "zed_nodes", "respeaker_nodes", "gps_objects", "modalities"):
                row[k] = ";".join(row[k] or [])
            writer.writerow(row)
    return {"json": str(json_path), "csv": str(csv_path)}


def build_and_write_catalog(roots: Iterable[Path], output_dir: Path) -> dict[str, Any]:
    roots = [Path(r).expanduser() for r in roots]
    records = scan_scenarios(roots)
    paths = write_catalog(records, Path(output_dir), roots=roots)
    return {"metadata": {"roots": [str(r) for r in roots], "count": len(records)}, "paths": paths, "scenarios": records}
