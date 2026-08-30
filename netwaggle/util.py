from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable


class NetWaggleError(RuntimeError):
    """Raised for recoverable NetWaggle setup/configuration errors."""


def run(cmd: Iterable[str] | str, *, check: bool = True, capture: bool = False, shell: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a command with consistent error reporting."""
    if isinstance(cmd, str):
        printable = cmd
        args = cmd if shell else shlex.split(cmd)
    else:
        args = list(cmd)
        printable = " ".join(shlex.quote(str(x)) for x in args)
    proc = subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        shell=shell,
    )
    if check and proc.returncode != 0:
        detail = ""
        if capture:
            detail = f"\nstdout:\n{proc.stdout or ''}\nstderr:\n{proc.stderr or ''}"
        raise NetWaggleError(f"Command failed ({proc.returncode}): {printable}{detail}")
    return proc


def sudo_check() -> None:
    if os.geteuid() != 0:
        raise NetWaggleError("NetWaggle must run as root because it creates veth pairs, modifies namespaces, and controls OVS/TC.")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def safe_ifname(prefix: str, name: str, max_len: int = 15) -> str:
    """Linux interface names are limited to IFNAMSIZ-1 bytes, usually 15 chars."""
    cleaned = "".join(c if c.isalnum() else "" for c in name.lower())
    if not cleaned:
        cleaned = "x"
    base = f"{prefix}{cleaned}"
    return base[:max_len]

