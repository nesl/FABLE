"""Local process/container backends used by a node agent.

A ProviderRuntime owns *how* a provider instance is materialized on one node.
The planner and reconciler never import subprocess or Docker logic.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Callable, Mapping, Protocol, Sequence

from .plan_reconciler import ProviderInstanceKey, ProviderInstanceSpec


class ProviderRuntime(Protocol):
    def start(self, spec: ProviderInstanceSpec) -> None: ...
    def stop(self, key: ProviderInstanceKey) -> None: ...
    def running(self) -> tuple[ProviderInstanceKey, ...]: ...


class InProcessProviderRuntime:
    """Instantiate provider objects in the node-agent process.

    This is useful for tests and lightweight deployments.  A factory may return
    any object; if it exposes ``start()``, ``stop()``, or ``close()`` those are
    called when appropriate.
    """

    def __init__(self, factories: Mapping[str, Callable[[], object]]) -> None:
        self.factories = dict(factories)
        self.instances: dict[ProviderInstanceKey, object] = {}

    def start(self, spec: ProviderInstanceSpec) -> None:
        if spec.key in self.instances:
            return
        try:
            factory = self.factories[spec.key.provider_id]
        except KeyError as exc:
            raise RuntimeError(f"no in-process factory for provider {spec.key.provider_id!r}") from exc
        instance = factory()
        start = getattr(instance, "start", None)
        if callable(start):
            start()
        self.instances[spec.key] = instance

    def stop(self, key: ProviderInstanceKey) -> None:
        instance = self.instances.pop(key, None)
        if instance is None:
            return
        stop = getattr(instance, "stop", None)
        close = getattr(instance, "close", None)
        if callable(stop):
            stop()
        elif callable(close):
            close()

    def running(self) -> tuple[ProviderInstanceKey, ...]:
        return tuple(sorted(self.instances))


class SubprocessProviderRuntime:
    """Launch configured provider commands as OS processes.

    Commands are supplied by deployment configuration rather than the semantic
    provider catalog.  The child receives provider/source metadata through
    environment variables so the command itself can remain reusable.
    """

    def __init__(
        self,
        commands: Mapping[str, Sequence[str]],
        *,
        cwd: str | Path | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        self.commands = {key: tuple(value) for key, value in commands.items()}
        self.cwd = None if cwd is None else str(cwd)
        self.extra_env = dict(extra_env or {})
        self.processes: dict[ProviderInstanceKey, subprocess.Popen] = {}

    def start(self, spec: ProviderInstanceSpec) -> None:
        existing = self.processes.get(spec.key)
        if existing is not None and existing.poll() is None:
            return
        try:
            command = self.commands[spec.key.provider_id]
        except KeyError as exc:
            raise RuntimeError(f"no process command for provider {spec.key.provider_id!r}") from exc
        env = os.environ.copy()
        env.update(self.extra_env)
        env.update({
            "FABLE_PROVIDER_ID": spec.key.provider_id,
            "FABLE_NODE_ID": spec.key.node_id,
            "FABLE_SOURCE_IDS": ",".join(spec.key.source_ids),
            "FABLE_OUTPUT_TYPE": spec.output_type,
        })
        process = subprocess.Popen(command, cwd=self.cwd, env=env)
        self.processes[spec.key] = process

    def stop(self, key: ProviderInstanceKey) -> None:
        process = self.processes.pop(key, None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def running(self) -> tuple[ProviderInstanceKey, ...]:
        dead = [key for key, process in self.processes.items() if process.poll() is not None]
        for key in dead:
            self.processes.pop(key, None)
        return tuple(sorted(self.processes))


class DockerProviderRuntime:
    """Launch provider containers through the Docker CLI.

    Using the CLI keeps Docker an optional deployment dependency rather than a
    mandatory Python package.  Provider images are deployment configuration.
    """

    def __init__(
        self,
        images: Mapping[str, str],
        *,
        docker_binary: str = "docker",
        extra_args: Sequence[str] = (),
    ) -> None:
        self.images = dict(images)
        self.docker_binary = docker_binary
        self.extra_args = tuple(extra_args)
        self.containers: dict[ProviderInstanceKey, str] = {}

    def start(self, spec: ProviderInstanceSpec) -> None:
        if spec.key in self.containers:
            return
        try:
            image = self.images[spec.key.provider_id]
        except KeyError as exc:
            raise RuntimeError(f"no Docker image for provider {spec.key.provider_id!r}") from exc
        name = _container_name(spec.key)
        argv = [
            self.docker_binary, "run", "--detach", "--rm", "--name", name,
            "--env", f"FABLE_PROVIDER_ID={spec.key.provider_id}",
            "--env", f"FABLE_NODE_ID={spec.key.node_id}",
            "--env", f"FABLE_SOURCE_IDS={','.join(spec.key.source_ids)}",
            "--env", f"FABLE_OUTPUT_TYPE={spec.output_type}",
            *self.extra_args,
            image,
        ]
        completed = subprocess.run(argv, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"docker start failed for {spec.key.provider_id}: {completed.stderr.strip()}")
        self.containers[spec.key] = name

    def stop(self, key: ProviderInstanceKey) -> None:
        name = self.containers.pop(key, None)
        if name is None:
            return
        subprocess.run(
            (self.docker_binary, "stop", name),
            check=False,
            capture_output=True,
            text=True,
        )

    def running(self) -> tuple[ProviderInstanceKey, ...]:
        return tuple(sorted(self.containers))


def _container_name(key: ProviderInstanceKey) -> str:
    base = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"fable-{key.provider_id}-{key.node_id}")
    digest = hashlib.sha1("|".join(key.source_ids).encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"[:120]
