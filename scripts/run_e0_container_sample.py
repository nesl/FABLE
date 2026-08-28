#!/usr/bin/env python3
"""Run one exact worker target with persistent warm and fresh cold containers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import subprocess
from time import perf_counter_ns

from evaluation.experiments.e0_calibration import (
    CalibrationObservation,
    CalibrationTarget,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--cold", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--volume",
        action="append",
        default=[],
        help="audited Docker bind mount in absolute-host:absolute-container:ro form",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = CalibrationTarget.model_validate(
        json.loads(args.target.read_text(encoding="utf-8"))
    )
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    observations: list[CalibrationObservation] = []

    mount_args: list[str] = []
    for specification in args.volume:
        parts = specification.split(":")
        if (
            len(parts) != 3
            or parts[2] != "ro"
            or not Path(parts[0]).is_absolute()
            or not Path(parts[1]).is_absolute()
        ):
            parser.error(
                "--volume must be absolute-host:absolute-container:ro"
            )
        mount_args.extend(("--volume", specification))
    argv = (
        "docker",
        "run",
        "--rm",
        "-i",
        *mount_args,
        "--entrypoint",
        "python",
        args.image,
        "-m",
        "providers.calibration_worker",
    )
    warm = subprocess.Popen(
        (*argv, "--serve"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert warm.stdin is not None and warm.stdout is not None
    try:
        for repetition in range(1, args.warm + 1):
            request = _request(target, fixture, repetition)
            started = perf_counter_ns()
            warm.stdin.write(json.dumps(request) + "\n")
            warm.stdin.flush()
            readable, _, _ = select.select(
                (warm.stdout,),
                (),
                (),
                args.timeout,
            )
            if not readable:
                raise TimeoutError(
                    f"warm calibration response exceeded {args.timeout}s"
                )
            response = json.loads(warm.stdout.readline())
            wall_ms = (perf_counter_ns() - started) / 1_000_000
            observations.append(
                _observation(
                    target,
                    "warm",
                    repetition,
                    response,
                    startup_ms=0,
                    wall_ms=wall_ms,
                )
            )
    finally:
        warm.terminate()
        try:
            warm.wait(timeout=5)
        except subprocess.TimeoutExpired:
            warm.kill()
            warm.wait(timeout=5)

    for repetition in range(1, args.cold + 1):
        request = _request(target, fixture, repetition)
        started = perf_counter_ns()
        completed = subprocess.run(
            argv,
            input=json.dumps(request),
            text=True,
            capture_output=True,
            shell=False,
            check=True,
            timeout=args.timeout,
        )
        wall_ms = (perf_counter_ns() - started) / 1_000_000
        response = json.loads(completed.stdout)
        provider_ms = float(response.get("provider_execution_ms", 0))
        observations.append(
            _observation(
                target,
                "cold",
                repetition,
                response,
                startup_ms=max(0, wall_ms - provider_ms),
                wall_ms=wall_ms,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(item.model_dump_json() + "\n" for item in observations),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "target": target.model_dump(mode="json"),
                "warm": args.warm,
                "cold": args.cold,
                "successful": sum(item.successful for item in observations),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def _request(target: CalibrationTarget, fixture: dict, repetition: int) -> dict:
    return {
        "schema_version": "fable.calibration_worker_request.v1",
        "target": target.model_dump(mode="json"),
        "invocation_number": repetition,
        "fixture": fixture,
    }


def _observation(
    target: CalibrationTarget,
    kind: str,
    repetition: int,
    response: dict,
    *,
    startup_ms: float,
    wall_ms: float,
) -> CalibrationObservation:
    provider_ms = float(response.get("provider_execution_ms", wall_ms))
    return CalibrationObservation(
        run_id=f"validation-{target.target_id}-{kind}-{repetition}",
        target=target,
        invocation_kind=kind,
        startup_ms=startup_ms,
        execution_ms=provider_ms,
        quality_score=float(response["quality_score"]),
        ambiguity_score=float(response["ambiguity_score"]),
        successful=bool(response["successful"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
