#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.rq3a_provider_coverage import validate_rq3a_provider_coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT / "iobt-minimal-ce-replay/config/fable_provider_runtimes.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_rq3a_provider_coverage(args.runtime_config)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
