from __future__ import annotations

from types import SimpleNamespace

from fable.distributed.docker_runtime import DockerSDKRuntime
from fable.distributed.models import ProviderRuntimeSpec, RuntimeMode


class _Container:
    def __init__(self, status: str) -> None:
        self.id = "container-1"
        self.name = "lease-worker"
        self.status = status
        self.started = 0
        self.attrs = {"State": {"Status": status}}

    def reload(self) -> None:
        self.attrs = {"State": {"Status": self.status}}

    def start(self) -> None:
        self.started += 1
        self.status = "running"


def _runtime(container: _Container) -> DockerSDKRuntime:
    containers = SimpleNamespace(get=lambda _name: container)
    return DockerSDKRuntime(docker_client=SimpleNamespace(containers=containers))


def _spec() -> ProviderRuntimeSpec:
    return ProviderRuntimeSpec(
        provider_id="audio_event_classifier",
        provider_contract_version=1,
        node_id="sensor-1",
        mode=RuntimeMode.ADOPT_EXISTING,
        container_name="lease-worker",
    )


def test_adopting_created_worker_starts_it_before_returning_handle() -> None:
    container = _Container("created")
    handle = _runtime(container).adopt(provider_instance_id="provider-1", spec=_spec())
    assert container.started == 1
    assert handle.running is True


def test_adopting_running_worker_is_idempotent() -> None:
    container = _Container("running")
    handle = _runtime(container).adopt(provider_instance_id="provider-1", spec=_spec())
    assert container.started == 0
    assert handle.running is True
