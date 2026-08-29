"""Download and verify the pinned person/vehicle ReID checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "fable" / "providers" / "reid" / "models.json"
DEST = ROOT / "fable" / "providers" / "reid" / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    DEST.mkdir(parents=True, exist_ok=True)
    for kind, spec in manifest["models"].items():
        output = DEST / spec["filename"]
        expected = spec["sha256"]
        if output.is_file() and sha256(output) == expected:
            print(f"{kind}: already provisioned: {output}")
            continue
        print(f"{kind}: downloading {spec['url']}")
        with urlopen(spec["url"], timeout=60) as response, output.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        observed = sha256(output)
        if observed != expected:
            output.unlink(missing_ok=True)
            raise RuntimeError(
                f"{kind}: checkpoint hash mismatch: expected {expected}, got {observed}"
            )
        print(f"{kind}: ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
