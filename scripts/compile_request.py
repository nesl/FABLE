#!/usr/bin/env python3
"""Compile an authored event-family request and print the semantic graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fable.semantic.request_compiler import EventRequestCompiler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", help='e.g. "detect a convoy"')
    args = parser.parse_args()
    result = EventRequestCompiler().compile(args.request)
    print(
        json.dumps(
            result.model_dump(mode="json", exclude_none=True),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
