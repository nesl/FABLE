#!/usr/bin/env python3
"""Stage exactly one experiment video in the Raspberry Pi replay cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import os
from uuid import uuid4


REMOTE_ROOT = Path("/home/rpi/project/FABLE/replay-cache")
REMOTE_MANIFEST = REMOTE_ROOT / "current.json"
DEFAULT_DATA_ROOTS = (
    Path("/media/brianw/Extreme SSD/West Point Experimentation"),
    Path("/media/brianw/Extreme SSD/GQ Data"),
)
DEFAULT_CONVERSION_CACHE = Path(
    "/media/brianw/Extreme SSD2/fable_results/physical_replay_cache"
)
ZED_REPLAY_IMAGE = "iobt-minimal/zed-replay:latest"


def run(*args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        text=True,
        capture_output=capture,
    )


def ssh(host: str, command: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return run("ssh", "-o", "BatchMode=yes", host, command, capture=capture)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def validate_mp4(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate:format=duration",
            "-of", "json", str(path),
        ],
        text=True, capture_output=True, check=False, timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(f"ffprobe rejected {path}: {completed.stderr.strip()}")
    document = json.loads(completed.stdout)
    streams = document.get("streams") or []
    duration = float((document.get("format") or {}).get("duration") or 0)
    if not streams or duration <= 0 or path.stat().st_size <= 0:
        raise RuntimeError(f"converted MP4 is empty or has no video stream: {path}")
    return {"duration_seconds": duration, **streams[0]}


def ensure_left_mp4(
    source: Path,
    *,
    cache_root: Path = DEFAULT_CONVERSION_CACHE,
    image: str = ZED_REPLAY_IMAGE,
) -> tuple[Path, dict[str, object]]:
    """Return a validated cached left-eye MP4, converting an SVO if needed."""

    source = source.resolve(strict=True)
    if source.suffix.lower() in {".mp4", ".mkv", ".avi"}:
        return source, {"converted": False, "validation": validate_mp4(source)}
    if source.suffix.lower() not in {".svo", ".svo2"}:
        raise ValueError(f"unsupported physical replay source: {source}")
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    source_sha = digest(source)
    output = cache_root / f"{source.stem}-{source_sha[:16]}_zed_left.mp4"
    provenance = output.with_suffix(".provenance.json")
    if output.is_file() and provenance.is_file():
        record = json.loads(provenance.read_text(encoding="utf-8"))
        if record.get("source_sha256") == source_sha:
            record["validation"] = validate_mp4(output)
            record["cache_hit"] = True
            return output, record

    temporary = cache_root / f".{output.name}.{uuid4().hex}.partial.mp4"
    converter = Path(__file__).with_name("export_svo_left_mp4.py").resolve(strict=True)
    command = [
        # The ZED SDK wheel in the vendor image contains executable extension
        # modules with root-only read permissions.  Match the normal replay
        # service (container root); the exFAT cache mount maps output ownership
        # to the desktop user.
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{source.parent}:/input:ro",
        "-v", f"{cache_root}:/output",
        "-v", f"{converter}:/export_svo_left_mp4.py:ro",
        image, "python3", "/export_svo_left_mp4.py",
        f"/input/{source.name}", f"/output/{temporary.name}",
    ]
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False, timeout=1800
        )
        if completed.returncode:
            raise RuntimeError(
                "ZED conversion failed: " + (completed.stderr or completed.stdout)[-4000:]
            )
        validation = validate_mp4(temporary)
        os.replace(temporary, output)
        record = {
            "schema_version": "fable.physical_replay_conversion.v1",
            "source": str(source),
            "source_sha256": source_sha,
            "output": str(output),
            "output_sha256": digest(output),
            "image": image,
            "converter": str(converter),
            "validation": validation,
            "cache_hit": False,
            "converter_stdout": completed.stdout.strip(),
        }
        provenance.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return output, record
    finally:
        temporary.unlink(missing_ok=True)


def resolve_scenario_video(
    *,
    scenario_id: str,
    replay_node: str,
    data_roots: tuple[Path, ...],
    asset_kind: str = "zed",
) -> Path:
    date = scenario_id.split("_", 1)[0]
    searched = []
    for root in data_roots:
        node_dir = root.expanduser() / date / replay_node
        searched.append(str(node_dir))
        patterns = (
            (f"{scenario_id}_*zed_left.mp4",)
            if asset_kind == "left-mp4"
            else (f"{scenario_id}_*zed*.svo2", f"{scenario_id}_*zed*.svo")
        )
        for pattern in patterns:
            matches = sorted(node_dir.glob(pattern))
            if matches:
                return matches[0].resolve(strict=True)
    raise FileNotFoundError(
        f"no ZED video for scenario={scenario_id} node={replay_node}; "
        f"searched {searched}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace the Pi's managed replay slot with one video."
    )
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--source-node", default="physical_rpi")
    parser.add_argument(
        "--replay-node",
        help="Resolve this logical replay node's ZED video when VIDEO is omitted.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        type=Path,
        help="Host recording root used for scenario/node resolution (repeatable).",
    )
    parser.add_argument(
        "--asset-kind", choices=("zed", "left-mp4"), default="zed"
    )
    parser.add_argument("--host", default="rpi")
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.video is not None and args.replay_node:
        parser.error("provide VIDEO or --replay-node, not both")
    if args.video is None and not args.replay_node:
        parser.error("VIDEO or --replay-node is required")
    video = (
        args.video.expanduser().resolve(strict=True)
        if args.video is not None
        else resolve_scenario_video(
            scenario_id=args.scenario_id,
            replay_node=args.replay_node,
            data_roots=tuple(args.data_root) or DEFAULT_DATA_ROOTS,
            asset_kind=args.asset_kind,
        )
    )
    if not video.is_file():
        parser.error(f"video is not a regular file: {video}")
    if video.suffix.lower() not in {".svo", ".svo2", ".mp4", ".mkv", ".avi"}:
        parser.error("video must be .svo, .svo2, .mp4, .mkv, or .avi")
    if args.reserve_gib < 0:
        parser.error("--reserve-gib cannot be negative")

    ssh_base = ["ssh", "-o", "BatchMode=yes"]
    scp_base = ["scp", "-o", "BatchMode=yes"]
    if args.identity_file:
        identity = str(args.identity_file.expanduser().resolve(strict=True))
        ssh_base[1:1] = ["-i", identity]
        scp_base[1:1] = ["-i", identity]

    size = video.stat().st_size
    reserve = int(args.reserve_gib * 1024**3)
    safe_suffix = video.suffix.lower()
    incoming = REMOTE_ROOT / f".incoming-{uuid4().hex}{safe_suffix}"
    current = REMOTE_ROOT / f"current{safe_suffix}"
    print(f"source={video}")
    print(f"size_bytes={size}")
    print(f"destination={args.host}:{current}")
    if args.dry_run:
        return 0

    def remote(command: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
        return run(*ssh_base, args.host, command, capture=capture)

    remote(f"mkdir -p {shlex.quote(str(REMOTE_ROOT))}")
    available = int(
        remote(
            f"df -B1 --output=avail {shlex.quote(str(REMOTE_ROOT))} | tail -1",
            capture=True,
        ).stdout.strip()
    )

    # Preserve the current slot when the filesystem can hold both assets.
    # Otherwise, free only files owned by this managed cache before uploading.
    if available < size + reserve:
        cleanup = (
            "from pathlib import Path; "
            f"r=Path({str(REMOTE_ROOT)!r}); "
            "[(p.unlink() if p.is_file() or p.is_symlink() else "
            "__import__('shutil').rmtree(p)) for p in r.iterdir()]"
        )
        remote(f"python3 -c {shlex.quote(cleanup)}")
        available = int(
            remote(
                f"df -B1 --output=avail {shlex.quote(str(REMOTE_ROOT))} | tail -1",
                capture=True,
            ).stdout.strip()
        )
    if available < size + reserve:
        raise RuntimeError(
            f"Pi has {available} bytes available after managed-cache cleanup; "
            f"needs {size + reserve}"
        )

    try:
        run(*scp_base, str(video), f"{args.host}:{incoming}")
        local_sha = digest(video)
        remote_sha = remote(
            f"sha256sum {shlex.quote(str(incoming))}", capture=True
        ).stdout.split()[0]
        if remote_sha != local_sha:
            raise RuntimeError(
                f"checksum mismatch: local={local_sha} remote={remote_sha}"
            )

        manifest = {
            "schema_version": "fable.physical_replay_slot.v1",
            "experiment_id": args.experiment_id,
            "scenario_id": args.scenario_id,
            "source_node": args.source_node,
            "source_file": str(video),
            "remote_video": str(current),
            "size_bytes": size,
            "sha256": local_sha,
            "staged_at": datetime.now(timezone.utc).isoformat(),
        }
        finalize = (
            "from pathlib import Path; import json, os, shutil; "
            f"r=Path({str(REMOTE_ROOT)!r}); "
            f"incoming=Path({str(incoming)!r}); current=Path({str(current)!r}); "
            "[p.unlink() if p.is_file() or p.is_symlink() else shutil.rmtree(p) "
            "for p in list(r.iterdir()) if p != incoming]; "
            "os.replace(incoming,current); "
            f"Path({str(REMOTE_MANIFEST)!r}).write_text(" 
            f"json.dumps({manifest!r},sort_keys=True,indent=2)+'\\n')"
        )
        remote(f"python3 -c {shlex.quote(finalize)}")
    except Exception:
        discard = (
            "from pathlib import Path; "
            f"p=Path({str(incoming)!r}); "
            "p.unlink(missing_ok=True)"
        )
        remote(f"python3 -c {shlex.quote(discard)}")
        raise

    print(f"sha256={local_sha}")
    print(f"manifest={args.host}:{REMOTE_MANIFEST}")
    print("staged=1")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
