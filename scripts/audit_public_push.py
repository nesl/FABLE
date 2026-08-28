#!/usr/bin/env python3
"""Fail closed when files selected for a public push look private or unsafe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024

PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "OpenAI-style secret": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Slack token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    # Device service accounts (for example /home/rpi) are deployment defaults,
    # not human identity. Personal workstation usernames remain prohibited.
    "personal home path": re.compile(rb"/(?:home|Users)/brianw(?:/|\b)"),
    "physical camera serial": re.compile(rb"\bSN[0-9]{8}\b"),
    "known lab hostname": re.compile(rb"\bnesl-orin-2\b"),
    "known lab subnet": re.compile(rb"\b172\.17\.(?:15|50)\.[0-9]{1,3}\b"),
}

RISKY_NAMES = (
    re.compile(r"(^|/)\.env(?:\.|$)"),
    re.compile(r"\.(?:pem|key|p12|pfx)$", re.IGNORECASE),
    re.compile(r"(^|/)id_(?:rsa|ed25519)(?:\.|$)"),
    re.compile(r"(^|/)SN[0-9]{8}\.conf$"),
)


def git_paths(mode: str) -> tuple[Path, ...]:
    if mode == "staged":
        commands = [["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]]
    else:
        # Excludes should filter only untracked files. A tracked file remains part
        # of a future clone even when a later .gitignore rule happens to match it.
        commands = [
            ["git", "ls-files", "-c", "-z"],
            ["git", "ls-files", "-o", "--exclude-standard", "-z"],
        ]
    output = b"".join(
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True).stdout
        for command in commands
    )
    return tuple(
        ROOT / os.fsdecode(item)
        for item in output.split(b"\0")
        if item
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-visible",
        action="store_true",
        help="scan tracked plus non-ignored untracked files instead of only staged files",
    )
    args = parser.parse_args()
    mode = "all-visible" if args.all_visible else "staged"
    findings: list[str] = []
    paths = git_paths(mode)
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        # A tracked path scheduled for deletion is intentionally absent from the
        # next commit and therefore has no content or filename to publish.
        if not path.is_file():
            continue
        if any(pattern.search(relative) for pattern in RISKY_NAMES):
            findings.append(f"{relative}: risky credential filename")
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            findings.append(f"{relative}: file is {size} bytes (limit {MAX_FILE_BYTES})")
            continue
        data = path.read_bytes()
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                findings.append(f"{relative}: {label}")

    if findings:
        print(f"public-push audit FAILED ({mode}, {len(paths)} files)")
        for finding in sorted(set(findings)):
            print(f"- {finding}")
        return 1
    print(f"public-push audit passed ({mode}, {len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
