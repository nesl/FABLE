#!/usr/bin/env python3
"""Build a replay scenario catalog from local IoBT data roots."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.scenario_catalog import build_and_write_catalog, parse_roots, scan_scenarios


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan IoBT replay data roots and write scenario_catalog.{json,csv}")
    ap.add_argument("--data-root", action="append", default=None,
                    help="Parent folder containing <YYYYMMDD>/orin*/... data. Can be repeated. Defaults to your two SSD roots or IOBT_HOST_DATA_ROOTS/IOBT_DATA_ROOTS.")
    ap.add_argument("--output-dir", default="generated", help="Directory for scenario_catalog.json and scenario_catalog.csv")
    ap.add_argument("--json", action="store_true", help="Print the catalog JSON payload to stdout as well as writing files")
    args = ap.parse_args()

    if args.data_root:
        raw = os.pathsep.join(args.data_root)
        roots = parse_roots(raw)
    else:
        roots = parse_roots(container_default=False)

    result = build_and_write_catalog(roots, Path(args.output_dir))
    print(f"Wrote {result['metadata']['count']} scenarios")
    print(f"JSON: {result['paths']['json']}")
    print(f"CSV : {result['paths']['csv']}")
    if args.json:
        print(json.dumps({"metadata": result["metadata"], "scenarios": result["scenarios"]}, indent=2))


if __name__ == "__main__":
    main()
