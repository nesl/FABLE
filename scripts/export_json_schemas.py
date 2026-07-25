#!/usr/bin/env python3
"""Export JSON Schema documents for every registered Phase-0 record."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fable.common.serialization import SCHEMA_REGISTRY  # noqa: E402


def main() -> int:
    output_dir = PROJECT_ROOT / "schema_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    for version, model in sorted(SCHEMA_REGISTRY.items()):
        filename = version.replace(".", "_") + ".schema.json"
        path = output_dir / filename
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
