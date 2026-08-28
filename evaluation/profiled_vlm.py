"""Deterministic PROFILED replay for hosted VLM comparison decisions."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
import yaml

from evaluation.provider_execution import HostedInvocationPermit, HostedProviderRunGate
from fable.common.base import FrozenFableModel
from fable.common.schemas import ProviderContract


class ProfiledVlmDecision(FrozenFableModel):
    invocation_id: str = Field(min_length=1)
    same_identity: bool
    confidence: float = Field(ge=0, le=1)
    latency_ms: int = Field(ge=0)
    reason: str = ""


class ProfiledVlmManifest(FrozenFableModel):
    schema_version: str = "fable.profiled_vlm_manifest.v1"
    profile_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    decisions: tuple[ProfiledVlmDecision, ...]

    @model_validator(mode="after")
    def _unique_invocations(self):
        ids = [item.invocation_id for item in self.decisions]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("profiled VLM invocation IDs must be non-empty and unique")
        return self


class ProfiledHostedVlmReplay:
    def __init__(
        self,
        *,
        manifest: ProfiledVlmManifest,
        provider: ProviderContract,
        gate: HostedProviderRunGate,
    ) -> None:
        if gate.mode != "PROFILED":
            raise ValueError("profiled VLM replay requires PROFILED mode")
        if not provider.evaluation_contract.hosted_external:
            raise ValueError("profiled VLM replay requires a hosted provider")
        self.manifest = manifest
        self.provider = provider
        self.gate = gate
        self._decisions = {
            item.invocation_id: item for item in manifest.decisions
        }
        self._consumed: set[str] = set()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        provider: ProviderContract,
        gate: HostedProviderRunGate,
    ) -> "ProfiledHostedVlmReplay":
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            manifest=ProfiledVlmManifest.model_validate(document),
            provider=provider,
            gate=gate,
        )

    def invoke(
        self,
        invocation_id: str,
    ) -> tuple[HostedInvocationPermit, ProfiledVlmDecision]:
        if invocation_id in self._consumed:
            raise RuntimeError(
                f"profiled VLM invocation already consumed: {invocation_id}"
            )
        try:
            decision = self._decisions[invocation_id]
        except KeyError as exc:
            raise RuntimeError(
                f"profiled VLM decision is missing: {invocation_id}"
            ) from exc
        permit = self.gate.acquire(self.provider)
        self._consumed.add(invocation_id)
        return permit, decision

