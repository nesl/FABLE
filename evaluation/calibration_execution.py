"""Bounded process-isolated execution of E0 calibration runs."""

from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
from collections import Counter
from pathlib import Path
from queue import Empty
from time import perf_counter_ns
from typing import Any, Protocol

from pydantic import Field

from evaluation.experiments.e0_calibration import (
    CalibrationObservation,
    CalibrationTarget,
    PlannedCalibrationRun,
)
from fable.common.base import FrozenFableModel


class CalibrationInvocationResult(FrozenFableModel):
    successful: bool = True
    quality_score: float = Field(ge=0, le=1)
    ambiguity_score: float = Field(ge=0, le=2)
    provider_execution_ms: float | None = Field(default=None, ge=0)


class CalibrationAdapter(Protocol):
    """One typed provider fixture; implementations must not invoke a shell."""

    def create(self) -> Any: ...

    def invoke(
        self,
        instance: Any,
        target: CalibrationTarget,
    ) -> CalibrationInvocationResult: ...


class CalibrationExclusion(FrozenFableModel):
    schema_version: str = "fable.calibration_exclusion.v1"
    run_id: str
    target: CalibrationTarget
    reason_code: str = Field(
        pattern=r"^(NO_TYPED_ADAPTER|UNSUPPORTED_INVOCATION_KIND|TIMEOUT|INVOCATION_ERROR)$"
    )
    reason: str = Field(min_length=1)


class CalibrationCampaignResult(FrozenFableModel):
    schema_version: str = "fable.calibration_campaign_result.v1"
    planned_runs: int = Field(ge=0)
    completed_observations: int = Field(ge=0)
    excluded_runs: int = Field(ge=0)
    timed_out_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    target_coverage: float = Field(ge=0, le=1)


class CalibrationAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[tuple[str, str, str], CalibrationAdapter] = {}

    def register(
        self,
        target: CalibrationTarget,
        adapter: CalibrationAdapter,
    ) -> None:
        key = _target_key(target)
        if key in self._adapters:
            raise ValueError(f"duplicate calibration adapter: {key}")
        self._adapters[key] = adapter

    def resolve(self, target: CalibrationTarget) -> CalibrationAdapter | None:
        return self._adapters.get(_target_key(target))


class JsonCommandCalibrationAdapter:
    """Invoke one audited JSON worker with an argument array and ``shell=False``."""

    def __init__(
        self,
        *,
        argv: tuple[str, ...],
        fixture: dict[str, Any],
        subprocess_timeout_seconds: float = 60,
        supported_invocation_kinds: tuple[str, ...] = ("warm",),
    ) -> None:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError("calibration worker argv must be a non-empty string array")
        if subprocess_timeout_seconds <= 0:
            raise ValueError("subprocess timeout must be positive")
        self.argv = argv
        self.fixture = dict(fixture)
        self.subprocess_timeout_seconds = subprocess_timeout_seconds
        if not supported_invocation_kinds or not set(
            supported_invocation_kinds
        ).issubset({"warm", "cold"}):
            raise ValueError("supported invocation kinds must be warm/cold")
        self.supported_invocation_kinds = supported_invocation_kinds

    def create(self) -> dict[str, int]:
        return {"invocations": 0}

    def invoke(
        self,
        instance: dict[str, int],
        target: CalibrationTarget,
    ) -> CalibrationInvocationResult:
        instance["invocations"] += 1
        payload = {
            "schema_version": "fable.calibration_worker_request.v1",
            "target": target.model_dump(mode="json"),
            "invocation_number": instance["invocations"],
            "fixture": self.fixture,
        }
        completed = subprocess.run(
            self.argv,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            shell=False,
            timeout=self.subprocess_timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                f"calibration worker exited {completed.returncode}: "
                f"{stderr[:500]}"
            )
        try:
            document = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("calibration worker returned invalid JSON") from exc
        if document.get("schema_version") != (
            "fable.calibration_worker_response.v1"
        ):
            raise RuntimeError("calibration worker response schema mismatch")
        return CalibrationInvocationResult.model_validate(
            {
                "successful": document.get("successful", True),
                "quality_score": document["quality_score"],
                "ambiguity_score": document["ambiguity_score"],
                "provider_execution_ms": document.get(
                    "provider_execution_ms"
                ),
            }
        )


def execute_calibration_campaign(
    runs: tuple[PlannedCalibrationRun, ...],
    registry: CalibrationAdapterRegistry,
    output_dir: str | Path,
    *,
    invocation_timeout_seconds: float = 120,
) -> CalibrationCampaignResult:
    if invocation_timeout_seconds <= 0:
        raise ValueError("invocation timeout must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    observations: list[CalibrationObservation] = []
    exclusions: list[CalibrationExclusion] = []
    workers: dict[tuple[str, str, str], _AdapterWorker] = {}
    try:
        for run in sorted(runs, key=lambda item: item.run_id):
            adapter = registry.resolve(run.target)
            if adapter is None:
                exclusions.append(
                    CalibrationExclusion(
                        run_id=run.run_id,
                        target=run.target,
                        reason_code="NO_TYPED_ADAPTER",
                        reason=(
                            "no exact provider/tier/input fixture adapter is "
                            "registered; no measurement was fabricated"
                        ),
                    )
                )
                continue
            supported = getattr(
                adapter,
                "supported_invocation_kinds",
                ("warm", "cold"),
            )
            if run.invocation_kind not in supported:
                exclusions.append(
                    CalibrationExclusion(
                        run_id=run.run_id,
                        target=run.target,
                        reason_code="UNSUPPORTED_INVOCATION_KIND",
                        reason=(
                            f"adapter does not implement {run.invocation_kind} "
                            "lifecycle measurement"
                        ),
                    )
                )
                continue
            key = _target_key(run.target)
            worker = workers.get(key)
            if worker is None or not worker.alive:
                worker = _AdapterWorker(adapter)
                worker.start()
                workers[key] = worker
            try:
                measured = worker.invoke(
                    run.target,
                    invocation_kind=run.invocation_kind,
                    timeout_seconds=invocation_timeout_seconds,
                )
            except TimeoutError as exc:
                exclusions.append(
                    CalibrationExclusion(
                        run_id=run.run_id,
                        target=run.target,
                        reason_code="TIMEOUT",
                        reason=str(exc),
                    )
                )
                worker.stop()
                workers.pop(key, None)
                continue
            except RuntimeError as exc:
                exclusions.append(
                    CalibrationExclusion(
                        run_id=run.run_id,
                        target=run.target,
                        reason_code="INVOCATION_ERROR",
                        reason=str(exc),
                    )
                )
                continue
            observations.append(
                CalibrationObservation(
                    run_id=run.run_id,
                    target=run.target,
                    invocation_kind=run.invocation_kind,
                    startup_ms=measured["startup_ms"],
                    execution_ms=measured["execution_ms"],
                    quality_score=measured["result"].quality_score,
                    ambiguity_score=measured["result"].ambiguity_score,
                    successful=measured["result"].successful,
                )
            )
    finally:
        for worker in workers.values():
            worker.stop()

    _write_jsonl(output / "observations.jsonl", observations)
    _write_jsonl(output / "exclusions.jsonl", exclusions)
    covered = { _target_key(item.target) for item in observations }
    planned_targets = {_target_key(item.target) for item in runs}
    counts = Counter(item.reason_code for item in exclusions)
    result = CalibrationCampaignResult(
        planned_runs=len(runs),
        completed_observations=len(observations),
        excluded_runs=len(exclusions),
        timed_out_runs=counts["TIMEOUT"],
        failed_runs=counts["INVOCATION_ERROR"],
        target_coverage=(
            len(covered) / len(planned_targets) if planned_targets else 0
        ),
    )
    (output / "campaign_summary.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return result


class _AdapterWorker:
    def __init__(self, adapter: CalibrationAdapter) -> None:
        self._adapter = adapter
        self._context = mp.get_context("fork")
        self._commands = self._context.Queue()
        self._results = self._context.Queue()
        self._process: mp.Process | None = None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        self._process = self._context.Process(
            target=_worker_main,
            args=(self._adapter, self._commands, self._results),
            daemon=True,
        )
        self._process.start()

    def invoke(
        self,
        target: CalibrationTarget,
        *,
        invocation_kind: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self._commands.put((invocation_kind, target))
        try:
            payload = self._results.get(timeout=timeout_seconds)
        except Empty as exc:
            self.stop()
            raise TimeoutError(
                f"calibration invocation exceeded {timeout_seconds} seconds"
            ) from exc
        if payload["error"] is not None:
            raise RuntimeError(payload["error"])
        return {
            "startup_ms": payload["startup_ms"],
            "execution_ms": payload["execution_ms"],
            "result": CalibrationInvocationResult.model_validate(
                payload["result"]
            ),
        }

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            self._commands.put(None)
            process.join(timeout=0.25)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        self._process = None


def _worker_main(adapter, commands, results) -> None:
    warm_instance = None
    while True:
        command = commands.get()
        if command is None:
            return
        invocation_kind, target = command
        try:
            startup_ms = 0.0
            if invocation_kind == "cold" or warm_instance is None:
                started = perf_counter_ns()
                instance = adapter.create()
                created_ms = (perf_counter_ns() - started) / 1_000_000
                if invocation_kind == "warm":
                    warm_instance = instance
                else:
                    startup_ms = created_ms
            else:
                instance = warm_instance
            started = perf_counter_ns()
            result = adapter.invoke(instance, target)
            wall_execution_ms = (perf_counter_ns() - started) / 1_000_000
            execution_ms = (
                result.provider_execution_ms
                if result.provider_execution_ms is not None
                else wall_execution_ms
            )
            results.put(
                {
                    "error": None,
                    "startup_ms": startup_ms,
                    "execution_ms": execution_ms,
                    "result": result.model_dump(mode="json"),
                }
            )
        except Exception as exc:
            results.put(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "startup_ms": 0,
                    "execution_ms": 0,
                    "result": None,
                }
            )


def _target_key(target: CalibrationTarget) -> tuple[str, str, str]:
    return (target.provider_id, target.tier, target.input_class)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in rows
        ),
        encoding="utf-8",
    )
