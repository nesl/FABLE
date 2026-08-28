#!/usr/bin/env python3
"""Inspect and initialize FABLE generated-data storage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fable.common.storage import load_storage_config


def _mount_for(path: Path) -> Path:
    current = path.resolve()
    while current.parent != current and not os.path.ismount(current):
        current = current.parent
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("show", "initialize"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_storage_config(args.config, repository_root=ROOT)
    mount = _mount_for(config.storage_root)
    if config.require_external_mount and mount == Path("/"):
        raise SystemExit(
            f"configured storage root is not on a mounted external filesystem: {config.storage_root}"
        )
    if args.command == "initialize":
        config.storage_root.mkdir(parents=True, exist_ok=True)
        for relative in config.paths.values():
            (config.storage_root / relative).mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": config.schema_version,
        "storage_root": str(config.storage_root),
        "mount": str(mount),
        "paths": {name: str(config.path(name)) for name in sorted(config.paths)},
        "repository_links": {
            str(local.relative_to(ROOT)): str(target)
            for local, target in config.link_targets(ROOT).items()
        },
        "initialized": args.command == "initialize",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
