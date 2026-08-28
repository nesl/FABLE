"""Scheduling data contracts for FABLE."""

from ._shared import *  # noqa: F401,F403

class ProviderLease(VersionedModel):
    """Demand-scoped permission to use one provider instance on one node.

    The lifecycle manager creates leases; NodeAgent execution consumes them.
    ``provider_id`` names the logical implementation, while
    ``provider_contract_version`` versions its declared ports/semantics (not
    its ML weights). ``provider_instance_id`` names the currently running or
    warm instance. ``demand_id`` and ``plan_id`` identify why it is used;
    ``attempt_id`` identifies this concrete activation attempt. The
    configuration hash protects safe instance sharing/reuse. Model identity,
    when relevant, belongs in result provenance as ``model_id/model_version``.
    """
    SCHEMA_VERSION: ClassVar[str] = "fable.provider_lease.v1"
    schema_version: Literal["fable.provider_lease.v1"] = SCHEMA_VERSION
    lease_id: UUID7 = Field(default_factory=uuid7)
    provider_instance_id: NonEmptyStr
    provider_id: NonEmptyStr
    provider_contract_version: int = Field(ge=1)
    demand_id: UUID7
    plan_id: UUID7
    node_id: NonEmptyStr
    configuration_hash: NonEmptyStr
    status: ProviderLeaseStatus = ProviderLeaseStatus.REQUESTED
    starts_at: datetime
    expires_at: datetime
    attempt_id: UUID7 = Field(default_factory=uuid7)

    @field_validator("starts_at", "expires_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_lease_time(self) -> "ProviderLease":
        if self.expires_at <= self.starts_at:
            raise ValueError("provider lease expires_at must be after starts_at")
        return self
