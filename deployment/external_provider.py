"""Typed JSONL bridge for providers running in a different Python runtime."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Mapping, Sequence

from fable.execution import DataflowProviderRuntime
from fable.execution.plan_reconciler import ProviderInstanceKey, ProviderInstanceSpec
from fable.providers.data_models import BoundingBox, Detection, DetectionFrame
from fable.execution.input_gate import InputGateConfig


class ExternalProviderBridgeRuntime:
    """Route allowlisted providers to child runtimes and all others locally."""

    def __init__(
        self,
        local: DataflowProviderRuntime,
        commands: Mapping[str, Sequence[str]],
        *,
        cwd: str | Path | None = None,
        ready_timeout_seconds: float = 30.0,
        input_gates: Mapping[str, InputGateConfig] | None = None,
    ) -> None:
        self.local = local
        self.commands = {name: tuple(command) for name, command in commands.items()}
        self.cwd = None if cwd is None else str(cwd)
        self.ready_timeout_seconds = float(ready_timeout_seconds)
        self.input_gates = dict(input_gates or {})
        self.processes: dict[ProviderInstanceKey, subprocess.Popen[str]] = {}
        self._ready: dict[ProviderInstanceKey, threading.Event] = {}
        self._errors: dict[ProviderInstanceKey, str] = {}

    def start(self, spec: ProviderInstanceSpec) -> None:
        command = self.commands.get(spec.key.provider_id)
        if command is None:
            self.local.start(spec)
            return
        existing = self.processes.get(spec.key)
        if existing is not None and existing.poll() is None:
            return
        env = os.environ.copy()
        env.update({
            "FABLE_PROVIDER_ID": spec.key.provider_id,
            "FABLE_NODE_ID": spec.key.node_id,
            "FABLE_SOURCE_IDS": ",".join(spec.key.source_ids),
            "FABLE_OUTPUT_TYPE": spec.output_type,
        })
        source_gates = {
            source_id: self.input_gates[source_id].to_dict()
            for source_id in spec.key.source_ids if source_id in self.input_gates
        }
        if source_gates:
            env["FABLE_INPUT_GATES"] = json.dumps(source_gates, sort_keys=True)
        process = subprocess.Popen(
            command,
            cwd=self.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            shell=False,
        )
        self.processes[spec.key] = process
        ready = self._ready[spec.key] = threading.Event()
        threading.Thread(
            target=self._read_stdout, args=(spec.key, process, ready), daemon=True
        ).start()
        threading.Thread(
            target=self._read_stderr, args=(spec.key, process), daemon=True
        ).start()
        if not ready.wait(self.ready_timeout_seconds):
            self.stop(spec.key)
            raise RuntimeError(f"external provider {spec.key.provider_id} did not become ready")
        if spec.key in self._errors:
            message = self._errors[spec.key]
            self.stop(spec.key)
            raise RuntimeError(message)

    def stop(self, key: ProviderInstanceKey) -> None:
        if key not in self.processes:
            self.local.stop(key)
            return
        process = self.processes.pop(key)
        self._ready.pop(key, None)
        self._errors.pop(key, None)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def running(self) -> tuple[ProviderInstanceKey, ...]:
        dead = [key for key, process in self.processes.items() if process.poll() is not None]
        for key in dead:
            self.processes.pop(key, None)
            self._ready.pop(key, None)
        return tuple(sorted((*self.local.running(), *self.processes)))

    def ready(self, key: ProviderInstanceKey) -> bool:
        if key not in self.processes:
            return self.local.ready(key)
        process = self.processes[key]
        return process.poll() is None and self._ready[key].is_set() and key not in self._errors

    def _read_stdout(self, key, process, ready) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if not line.strip():
                    continue
                document = json.loads(line)
                kind = document.get("type")
                if kind == "ready":
                    ready.set()
                elif kind == "detection_frame":
                    frame = _detection_frame(document)
                    if frame.source_id not in key.source_ids:
                        raise ValueError("external result source is outside provider placement")
                    self.local.publish_source(frame.source_id, "detections", frame)
                elif kind == "error":
                    raise RuntimeError(str(document.get("message") or "external provider error"))
                else:
                    raise ValueError(f"unknown external provider message {kind!r}")
        except Exception as exc:
            self._errors[key] = f"{type(exc).__name__}: {exc}"
            ready.set()

    def _read_stderr(self, key, process) -> None:
        assert process.stderr is not None
        tail = []
        for line in process.stderr:
            tail.append(line.rstrip())
            tail = tail[-20:]
        if process.poll() not in (None, 0) and key not in self._errors:
            self._errors[key] = "external provider exited: " + " | ".join(tail)
            event = self._ready.get(key)
            if event is not None:
                event.set()


def _detection_frame(raw: dict) -> DetectionFrame:
    allowed = {"type", "source_id", "event_time", "frame_id", "image_width", "image_height", "detections"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown detection-frame fields: {sorted(unknown)}")
    detections = []
    for index, row in enumerate(raw.get("detections", ())):
        if set(row) - {"class_name", "confidence", "bbox", "detection_id"}:
            raise ValueError("unknown detection fields")
        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox must contain four coordinates")
        detections.append(Detection(
            str(row["class_name"]), float(row["confidence"]),
            BoundingBox(*map(float, bbox)),
            str(row.get("detection_id") or f"external:{index}"),
        ))
    return DetectionFrame(
        str(raw["source_id"]),
        datetime.fromisoformat(str(raw["event_time"]).replace("Z", "+00:00")),
        tuple(detections),
        str(raw.get("frame_id") or ""),
        None if raw.get("image_width") is None else int(raw["image_width"]),
        None if raw.get("image_height") is None else int(raw["image_height"]),
    )
