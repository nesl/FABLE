"""Provider container lifecycle adapters for node agents."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Protocol

from fable.common.time import utc_now

from .models import ProviderRuntimeSpec, ResourceLimits, RuntimeMode


@dataclass(frozen=True)
class ContainerHandle:
    container_id: str
    name: str
    provider_instance_id: str
    managed: bool
    adopted: bool
    running: bool
    healthy: bool | None = None


class ContainerRuntime(Protocol):
    def start(
        self,
        *,
        provider_instance_id: str,
        spec: ProviderRuntimeSpec,
        limits: ResourceLimits,
    ) -> ContainerHandle: ...

    def adopt(
        self,
        *,
        provider_instance_id: str,
        spec: ProviderRuntimeSpec,
    ) -> ContainerHandle: ...

    def inspect(self, provider_instance_id: str) -> ContainerHandle | None: ...

    def stop(self, provider_instance_id: str, *, timeout_seconds: int = 5, force: bool = False) -> bool: ...

    def crash(self, provider_instance_id: str) -> bool: ...


class DockerSDKRuntime:
    """Docker Engine implementation used by real node-agent containers.

    The node agent normally mounts ``/var/run/docker.sock``.  For providers in
    ``ADOPT_EXISTING`` mode, no container is created; the agent attaches FABLE
    leases to the replay stack's already-running detector container.
    """

    def __init__(self, *, docker_client=None) -> None:
        if docker_client is None:
            try:
                import docker
            except ImportError as exc:  # pragma: no cover - optional runtime dependency
                raise RuntimeError("docker Python SDK is required for DockerSDKRuntime") from exc
            docker_client = docker.from_env()
        self.client = docker_client
        self._handles: dict[str, ContainerHandle] = {}
        self._lock = threading.RLock()

    def start(
        self,
        *,
        provider_instance_id: str,
        spec: ProviderRuntimeSpec,
        limits: ResourceLimits,
    ) -> ContainerHandle:
        if spec.mode != RuntimeMode.MANAGED_CONTAINER:
            raise ValueError("Docker start requires MANAGED_CONTAINER runtime mode")
        with self._lock:
            existing = self.inspect(provider_instance_id)
            if existing is not None and existing.running:
                return existing
            name = spec.container_name or f"fable-{provider_instance_id[:40]}"
            labels = {
                "fable.provider_instance_id": provider_instance_id,
                "fable.provider_id": spec.provider_id,
                "fable.node_id": spec.node_id,
                **spec.labels,
            }
            kwargs = {
                "image": spec.image,
                "name": name,
                "detach": True,
                "environment": spec.environment,
                "labels": labels,
                "auto_remove": False,
            }
            if spec.command:
                kwargs["command"] = list(spec.command)
            if spec.entrypoint:
                kwargs["entrypoint"] = list(spec.entrypoint)
            if spec.working_dir:
                kwargs["working_dir"] = spec.working_dir
            if spec.network_mode:
                kwargs["network_mode"] = spec.network_mode
            if spec.volumes:
                kwargs["volumes"] = _docker_volumes(spec.volumes)
            if limits.cpu_cores > 0:
                kwargs["nano_cpus"] = int(limits.cpu_cores * 1_000_000_000)
            if limits.memory_mb > 0:
                kwargs["mem_limit"] = f"{limits.memory_mb}m"
            if limits.pids_limit is not None:
                kwargs["pids_limit"] = limits.pids_limit
            if limits.gpu_count > 0 or limits.gpu_memory_mb > 0:
                try:
                    import docker

                    kwargs["device_requests"] = [
                        docker.types.DeviceRequest(count=max(1, limits.gpu_count), capabilities=[["gpu"]])
                    ]
                except Exception:
                    # Older Docker SDKs may not expose DeviceRequest.  The
                    # container still starts; readiness will reveal if GPU access
                    # was actually required and unavailable.
                    pass
            container = self.client.containers.run(**kwargs)
            container.reload()
            handle = _handle_from_container(
                container,
                provider_instance_id=provider_instance_id,
                managed=True,
                adopted=False,
            )
            self._handles[provider_instance_id] = handle
            return handle

    def adopt(
        self,
        *,
        provider_instance_id: str,
        spec: ProviderRuntimeSpec,
    ) -> ContainerHandle:
        if spec.mode != RuntimeMode.ADOPT_EXISTING:
            raise ValueError("Docker adopt requires ADOPT_EXISTING runtime mode")
        assert spec.container_name is not None
        with self._lock:
            container = self.client.containers.get(spec.container_name)
            container.reload()
            handle = _handle_from_container(
                container,
                provider_instance_id=provider_instance_id,
                managed=False,
                adopted=True,
            )
            self._handles[provider_instance_id] = handle
            return handle

    def inspect(self, provider_instance_id: str) -> ContainerHandle | None:
        with self._lock:
            known = self._handles.get(provider_instance_id)
        try:
            if known is not None:
                container = self.client.containers.get(known.container_id)
            else:
                matches = self.client.containers.list(
                    all=True,
                    filters={"label": f"fable.provider_instance_id={provider_instance_id}"},
                )
                if not matches:
                    return None
                container = matches[0]
            container.reload()
            handle = _handle_from_container(
                container,
                provider_instance_id=provider_instance_id,
                managed=bool(known.managed) if known else True,
                adopted=bool(known.adopted) if known else False,
            )
            with self._lock:
                self._handles[provider_instance_id] = handle
            return handle
        except Exception:
            return None

    def stop(self, provider_instance_id: str, *, timeout_seconds: int = 5, force: bool = False) -> bool:
        handle = self.inspect(provider_instance_id)
        if handle is None:
            return False
        if handle.adopted:
            return False
        try:
            container = self.client.containers.get(handle.container_id)
            if force:
                container.kill()
            else:
                container.stop(timeout=timeout_seconds)
            with self._lock:
                self._handles[provider_instance_id] = ContainerHandle(
                    **{**handle.__dict__, "running": False}
                )
            return True
        except Exception:
            if not force:
                try:
                    container.kill()
                    return True
                except Exception:
                    pass
            return False

    def crash(self, provider_instance_id: str) -> bool:
        handle = self.inspect(provider_instance_id)
        if handle is None:
            return False
        try:
            self.client.containers.get(handle.container_id).kill()
            return True
        except Exception:
            return False


class FakeContainerRuntime:
    """Deterministic in-memory container runtime for tests."""

    def __init__(self) -> None:
        self.handles: dict[str, ContainerHandle] = {}
        self.start_count: dict[str, int] = {}
        self.stop_count: dict[str, int] = {}
        self.fail_start_for: set[str] = set()
        self.available_adopted_names: set[str] = set()
        self._lock = threading.RLock()

    def start(
        self,
        *,
        provider_instance_id: str,
        spec: ProviderRuntimeSpec,
        limits: ResourceLimits,
    ) -> ContainerHandle:
        with self._lock:
            if provider_instance_id in self.fail_start_for:
                raise RuntimeError(f"injected start failure for {provider_instance_id}")
            existing = self.handles.get(provider_instance_id)
            if existing and existing.running:
                return existing
            self.start_count[provider_instance_id] = self.start_count.get(provider_instance_id, 0) + 1
            handle = ContainerHandle(
                container_id=f"fake-{provider_instance_id}",
                name=spec.container_name or f"fable-{provider_instance_id}",
                provider_instance_id=provider_instance_id,
                managed=True,
                adopted=False,
                running=True,
                healthy=True,
            )
            self.handles[provider_instance_id] = handle
            return handle

    def adopt(
        self,
        *,
        provider_instance_id: str,
        spec: ProviderRuntimeSpec,
    ) -> ContainerHandle:
        assert spec.container_name is not None
        with self._lock:
            if self.available_adopted_names and spec.container_name not in self.available_adopted_names:
                raise RuntimeError(f"container {spec.container_name} is unavailable")
            existing = self.handles.get(provider_instance_id)
            if existing and existing.running:
                return existing
            handle = ContainerHandle(
                container_id=f"adopted-{spec.container_name}",
                name=spec.container_name,
                provider_instance_id=provider_instance_id,
                managed=False,
                adopted=True,
                running=True,
                healthy=True,
            )
            self.handles[provider_instance_id] = handle
            return handle

    def inspect(self, provider_instance_id: str) -> ContainerHandle | None:
        with self._lock:
            return self.handles.get(provider_instance_id)

    def stop(self, provider_instance_id: str, *, timeout_seconds: int = 5, force: bool = False) -> bool:
        with self._lock:
            handle = self.handles.get(provider_instance_id)
            if handle is None or not handle.running or handle.adopted:
                return False
            self.stop_count[provider_instance_id] = self.stop_count.get(provider_instance_id, 0) + 1
            self.handles[provider_instance_id] = ContainerHandle(
                **{**handle.__dict__, "running": False}
            )
            return True

    def crash(self, provider_instance_id: str) -> bool:
        with self._lock:
            handle = self.handles.get(provider_instance_id)
            if handle is None:
                return False
            self.handles[provider_instance_id] = ContainerHandle(
                **{**handle.__dict__, "running": False, "healthy": False}
            )
            return True


def _docker_volumes(volumes: dict[str, str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for host_path, container_spec in volumes.items():
        if ":" in container_spec:
            bind, mode = container_spec.rsplit(":", 1)
        else:
            bind, mode = container_spec, "rw"
        result[host_path] = {"bind": bind, "mode": mode}
    return result


def _handle_from_container(
    container,
    *,
    provider_instance_id: str,
    managed: bool,
    adopted: bool,
) -> ContainerHandle:
    attrs = getattr(container, "attrs", {}) or {}
    state = attrs.get("State", {})
    health = state.get("Health", {}).get("Status")
    return ContainerHandle(
        container_id=str(container.id),
        name=str(container.name),
        provider_instance_id=provider_instance_id,
        managed=managed,
        adopted=adopted,
        running=state.get("Status") == "running" or getattr(container, "status", None) == "running",
        healthy=None if health is None else health == "healthy",
    )
