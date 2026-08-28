"""Evaluation-mode and invocation-budget gates for hosted providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os

from fable.common.schemas import ProviderContract


@dataclass(frozen=True)
class HostedInvocationPermit:
    provider_id: str
    run_id: str
    mode: str
    invocation_number: int
    maximum_invocations: int


class HostedProviderRunGate:
    """Fail-closed per-run gate; it never stores or returns secret values."""

    def __init__(
        self,
        *,
        run_id: str,
        mode: str,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if mode not in {"LIVE", "PROFILED"}:
            raise ValueError("hosted provider mode must be LIVE or PROFILED")
        self.run_id = run_id
        self.mode = mode
        self._environment = environment if environment is not None else os.environ
        self._counts: dict[str, int] = {}

    def acquire(self, provider: ProviderContract) -> HostedInvocationPermit:
        contract = provider.evaluation_contract
        if not contract.hosted_external:
            raise ValueError(f"provider is not hosted external: {provider.provider_id}")
        if self.mode not in contract.supported_modes:
            raise ValueError(
                f"provider {provider.provider_id} does not support {self.mode}"
            )
        maximum = contract.maximum_invocations_per_run
        if maximum is None:
            raise ValueError("hosted provider has no invocation bound")
        if self.mode == "LIVE":
            missing = [
                name
                for name in contract.required_secret_names
                if not self._environment.get(name)
            ]
            if missing:
                raise RuntimeError(
                    "hosted provider credentials are unavailable: "
                    + ", ".join(missing)
                )
        count = self._counts.get(provider.provider_id, 0)
        if count >= maximum:
            raise RuntimeError(
                f"hosted provider invocation budget exhausted: {provider.provider_id}"
            )
        count += 1
        self._counts[provider.provider_id] = count
        return HostedInvocationPermit(
            provider_id=provider.provider_id,
            run_id=self.run_id,
            mode=self.mode,
            invocation_number=count,
            maximum_invocations=maximum,
        )

    def fallback_eligible(
        self,
        provider: ProviderContract,
        *,
        ambiguity_score: float,
    ) -> bool:
        threshold = provider.evaluation_contract.ambiguity_trigger_maximum
        if threshold is None:
            return False
        return 0 <= ambiguity_score <= threshold

    def invocation_count(self, provider_id: str) -> int:
        return self._counts.get(provider_id, 0)

