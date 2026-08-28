from __future__ import annotations

from pathlib import Path

from fable.distributed.docker_runtime import ManagedProcessRuntime
from fable.distributed.models import ProviderRuntimeSpec, ResourceLimits, RuntimeMode


def test_managed_process_is_started_and_stopped_with_lease_runtime(tmp_path: Path) -> None:
    runtime = ManagedProcessRuntime(state_dir=tmp_path)
    spec = ProviderRuntimeSpec(
        provider_id="test_provider",
        provider_contract_version=1,
        node_id="physical_test",
        mode=RuntimeMode.MANAGED_PROCESS,
        command=("/bin/sleep", "30"),
    )

    handle = runtime.start(
        provider_instance_id="test-instance",
        spec=spec,
        limits=ResourceLimits(),
    )

    assert handle.running
    assert runtime.inspect("test-instance").running
    assert runtime.stop("test-instance", timeout_seconds=1)
    assert not runtime.inspect("test-instance").running


def test_managed_process_rejects_relative_executable(tmp_path: Path) -> None:
    runtime = ManagedProcessRuntime(state_dir=tmp_path)
    spec = ProviderRuntimeSpec(
        provider_id="test_provider",
        provider_contract_version=1,
        node_id="physical_test",
        mode=RuntimeMode.MANAGED_PROCESS,
        command=("sleep", "30"),
    )

    try:
        runtime.start(
            provider_instance_id="bad-instance",
            spec=spec,
            limits=ResourceLimits(),
        )
    except ValueError as exc:
        assert "absolute path" in str(exc)
    else:
        raise AssertionError("relative managed-process command was accepted")
