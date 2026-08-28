"""Artifact data contracts for FABLE."""

from ._shared import *  # noqa: F401,F403

class ArtifactLocation(FableModel):
    """Resolvable storage/transport location for one evidence artifact."""
    kind: ArtifactLocationKind
    node_id: str | None = None
    uri: str | None = None
    inline_data_base64: str | None = None

    @model_validator(mode="after")
    def _validate_location(self) -> "ArtifactLocation":
        if self.kind == ArtifactLocationKind.INLINE:
            if not self.inline_data_base64:
                raise ValueError("INLINE artifact location requires inline_data_base64")
        elif not self.uri:
            raise ValueError("non-inline artifact location requires uri")
        return self

class ArtifactProducer(FableModel):
    """Provider and optional ML-model identity that created an artifact."""
    provider_id: NonEmptyStr
    provider_contract_version: int = Field(ge=1)
    model_id: str | None = None
    model_version: str | None = None

class ArtifactRef(VersionedModel):
    """Typed retained evidence announced by node execution and read by planning."""
    SCHEMA_VERSION: ClassVar[str] = "fable.artifact_ref.v1"
    schema_version: Literal["fable.artifact_ref.v1"] = SCHEMA_VERSION
    artifact_id: UUID7 = Field(default_factory=uuid7)
    artifact_type: NonEmptyStr
    artifact_schema_version: NonEmptyStr
    producer: ArtifactProducer
    event_time_interval: EventTimeInterval
    bindings: dict[str, str] = Field(default_factory=dict)
    location: ArtifactLocation
    access_modes: tuple[ArtifactAccessMode, ...]
    compatibility_keys: dict[str, JSONValue] = Field(default_factory=dict)
    compatible_consumer_families: tuple[str, ...] = ()
    bytes: int | None = Field(default=None, ge=0)
    checksum_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    created_at: datetime = Field(default_factory=utc_now)
    valid_until: datetime | None = None
    expires_at: datetime | None = None
    policy_tags: tuple[str, ...] = ()

    @field_validator("created_at", "valid_until", "expires_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @model_validator(mode="after")
    def _validate_lifetime(self) -> "ArtifactRef":
        if not self.access_modes:
            raise ValueError("artifact must expose at least one access mode")
        if self.valid_until and self.valid_until < self.event_time_interval.end:
            raise ValueError("valid_until cannot precede the artifact event interval")
        if self.expires_at and self.expires_at < self.created_at:
            raise ValueError("expires_at cannot precede created_at")
        return self
