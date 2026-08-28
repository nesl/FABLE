#!/usr/bin/env python3
"""Provision pinned person and vehicle ReID checkpoints with SHA-256 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


REPLAY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPLAY_ROOT / "models/reid/models.json"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    destination = args.manifest.parent
    destination.mkdir(parents=True, exist_ok=True)
    for entity_kind, model in manifest["models"].items():
        target = destination / model["filename"]
        expected = model["sha256"]
        if target.is_file() and digest(target) == expected:
            print(f"{entity_kind}: verified {target}")
            continue
        if args.verify_only:
            raise SystemExit(f"{entity_kind}: missing or checksum mismatch: {target}")
        partial = target.with_suffix(target.suffix + ".part")
        subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "3",
                "--output",
                os.fspath(partial),
                model["url"],
            ],
            check=True,
        )
        actual = digest(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise SystemExit(
                f"{entity_kind}: checksum mismatch expected={expected} actual={actual}"
            )
        partial.replace(target)
        print(f"{entity_kind}: installed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
