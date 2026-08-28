#!/usr/bin/env python3
"""Run all exact desktop E0 fixtures with per-target hard deadlines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image", default="fable/vehicle-stack:phase7")
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--cold", type=int, default=2)
    parser.add_argument("--target-timeout", type=float, default=90)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="run only the named provider (repeatable)",
    )
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [
        row
        for row in document["fixtures"]
        if not args.provider or row["provider_id"] in set(args.provider)
    ]
    observations = []
    results = []
    for index, row in enumerate(selected, 1):
        output = args.output / (
            f"{row['provider_id']}__"
            f"{row['input_class'].replace('+', '__')}.jsonl"
        )
        command = (
            sys.executable,
            str(ROOT / "scripts/run_e0_container_sample.py"),
            "--image",
            args.image,
            "--target",
            str(ROOT / row["target"]),
            "--fixture",
            str(ROOT / row["fixture"]),
            "--warm",
            str(args.warm),
            "--cold",
            str(args.cold),
            "--timeout",
            str(args.target_timeout),
            "--volume",
            f"{args.media_dir.resolve()}:/calibration:ro",
            "--output",
            str(output),
        )
        print(
            f"[{index}/{len(selected)}] "
            f"{row['provider_id']}/{row['input_class']}",
            flush=True,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.target_timeout * (args.cold + 1) + 15,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            completed = None
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        else:
            stdout = completed.stdout
            stderr = completed.stderr
        success = (
            not timed_out
            and completed is not None
            and completed.returncode == 0
            and output.is_file()
        )
        if success:
            observations.extend(
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        results.append(
            {
                **row,
                "successful": success,
                "timed_out": timed_out,
                "returncode": (
                    completed.returncode if completed is not None else 124
                ),
                "stdout": stdout[-2000:],
                "stderr": stderr[-2000:],
            }
        )
    combined = args.output / "observations.jsonl"
    combined.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in observations),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "fable.e0_desktop_campaign.v1",
        "target_count": len(results),
        "successful_target_count": sum(row["successful"] for row in results),
        "observation_count": len(observations),
        "warm_repetitions": args.warm,
        "cold_repetitions": args.cold,
        "successful": all(row["successful"] for row in results),
        "results": results,
    }
    (args.output / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in (
        "target_count",
        "successful_target_count",
        "observation_count",
        "successful",
    )}))
    return 0 if summary["successful"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
