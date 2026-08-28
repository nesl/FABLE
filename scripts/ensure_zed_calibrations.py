#!/usr/bin/env python3
"""Install the allowlisted ZED calibration files required by FABLE replay.

The ZED SDK cannot open an SVO when its serial-specific calibration file is
missing.  Evaluation mounts the settings directory read-only, so relying on
an implicit SDK download inside a container is both unreliable and unable to
repair the host cache.  This helper fetches only the known West Point camera
serials, validates the response shape, and installs each file atomically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_DIR = ROOT / "iobt-minimal-ce-replay/setup/zed_settings"
WEST_POINT_SERIALS = (
    "31366375",
    "35309867",
    "36577075",
    "37711387",
    "39164952",
)


def valid_calibration(payload: bytes) -> bool:
    return (
        payload.startswith(b"[LEFT_CAM_2K]")
        and b"[RIGHT_CAM_2K]" in payload
        and b"[STEREO]" in payload
        and len(payload) >= 1000
    )


def install(settings_dir: Path, *, force: bool = False) -> list[Path]:
    settings_dir.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    for serial in WEST_POINT_SERIALS:
        target = settings_dir / f"SN{serial}.conf"
        if target.is_file() and valid_calibration(target.read_bytes()) and not force:
            continue
        with urlopen(
            f"https://calib.stereolabs.com/?SN={serial}", timeout=15
        ) as response:
            payload = response.read()
        if not valid_calibration(payload):
            raise RuntimeError(f"invalid calibration response for ZED serial {serial}")
        with tempfile.NamedTemporaryFile(dir=settings_dir, delete=False) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.chmod(0o664)
        temporary.replace(target)
        installed.append(target)
    return installed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-dir", type=Path, default=DEFAULT_SETTINGS_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    installed = install(args.settings_dir.resolve(), force=args.force)
    print(f"ZED calibration inventory ready: {len(WEST_POINT_SERIALS)} required")
    for path in installed:
        print(f"installed {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
