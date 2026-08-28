#!/usr/bin/env python3
"""Validate two or more live common-record directories as one concurrency run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.live_validation import validate_concurrent_logging
from evaluation.schemas import (
    ArtifactEvent,
    ProviderLeaseEvent,
    ResourceSample,
    RetrospectiveAttempt,
)


def _load(path: Path, model):
    if not path.is_file():
        return ()
    return tuple(
        model.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    leases = ()
    resources = ()
    attempts = ()
    artifacts = ()
    for directory in args.record_dirs:
        leases += _load(directory / "provider_lease.jsonl", ProviderLeaseEvent)
        resources += _load(directory / "resource_sample.jsonl", ResourceSample)
        attempts += _load(
            directory / "retrospective_attempt.jsonl",
            RetrospectiveAttempt,
        )
        artifacts += _load(directory / "artifact_event.jsonl", ArtifactEvent)
    report = validate_concurrent_logging(
        leases=leases,
        resources=resources,
        retrospective_attempts=attempts,
        artifacts=artifacts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(report.model_dump_json())
    return 0 if report.successful else 2


if __name__ == "__main__":
    raise SystemExit(main())
