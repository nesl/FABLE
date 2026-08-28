#!/usr/bin/env python3
"""Generate compact execution-change timelines for existing result trees."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.execution_timeline import write_execution_timeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args()
    generated = []
    for path in sorted(args.results_root.rglob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict):
            continue
        record_dir = result.get("common_record_dir")
        if result.get("schema_version") != "fable.replay_accuracy_run.v2" or not record_dir:
            continue
        directory = Path(str(record_dir))
        if directory.is_dir():
            generated.append({"result": str(path), **write_execution_timeline(result, directory)})
    print(json.dumps({"generated": len(generated), "timelines": generated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
