"""Provider data contracts for FABLE."""

from ._shared import *  # noqa: F401,F403

class ProviderRoleCapability(FableModel):
    """Ways a provider result can introduce, consume, or validate one role."""
    role_name: NonEmptyStr
    capabilities: tuple[BindingCapability, ...]

class ProviderSemanticCapabilities(FableModel):
    """Semantic predicates and role behavior implemented by a provider."""
    predicate_ids: tuple[str, ...]
    role_capabilities: tuple[ProviderRoleCapability, ...] = ()
    result_kinds: tuple[ResultKind, ...] = ()

    @model_validator(mode="after")
    def _require_predicate(self) -> "ProviderSemanticCapabilities":
        if not self.predicate_ids:
            raise ValueError("provider must implement at least one semantic predicate")
        return self

class ProviderPort(FableModel):
    """Named, typed input/output used to wire providers into chain DAGs."""
    name: NonEmptyStr
    kind: ProviderPortKind
    data_type: NonEmptyStr
    required: bool = True
    purpose: str | None = None

class ParameterSpec(FableModel):
    """Schema for one provider setting, not the setting's runtime value.

    Catalog loading validates this declaration. Concrete values originate in
    semantic demands/plans, are validated during planning, and are copied into
    the activation command consumed by the provider implementation.
    """
    type: NonEmptyStr
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    enum: tuple[JSONValue, ...] = ()
    default: JSONValue = None

    @model_validator(mode="after")
    def _validate_range(self) -> "ParameterSpec":
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("parameter maximum must be >= minimum")
        return self

class ProviderExecutionCapabilities(FableModel):
    """Supported live/retrospective modes, sharing, and evidence access modes.

    ``LOCAL`` means computation and input reside on the same logical node.
    ``TRANSFERRED`` authorizes copying the artifact bytes to the execution
    node. ``REMOTE_REFERENCE`` passes a URI/reference and leaves the artifact
    at its source; the provider runtime must implement dereferencing it.
    ``INLINE`` embeds a small value in the command/result envelope.
    """
    modes: tuple[ExecutionMode, ...]
    supports_shared_execution: bool = False
    accepted_input_access: tuple[ArtifactAccessMode, ...] = ()
    state_operations: tuple[str, ...] = ()

class CompatibilityGroup(FableModel):
    """Input ports whose runtime keys must match for safe provider reuse."""
    ports: tuple[str, ...]
    require_same_runtime_keys: tuple[str, ...] = ()

class ProviderEvaluationContract(FableModel):
    """Bounded evaluation policy for external or profiled provider calls."""
    supported_modes: tuple[str, ...] = ()
    hosted_external: bool = False
    maximum_invocations_per_run: int | None = Field(default=None, ge=1)
    required_secret_names: tuple[str, ...] = ()
    ambiguity_trigger_maximum: float | None = Field(default=None, ge=0, le=1)

class ProviderContract(VersionedModel):
    """One executable implementation contract, never a hypothesis controller."""

    SCHEMA_VERSION: ClassVar[str] = "fable.provider_contract.v1"
    schema_version: Literal["fable.provider_contract.v1"] = SCHEMA_VERSION
    provider_id: NonEmptyStr
    contract_version: int = Field(default=1, ge=1)
    description: NonEmptyStr
    semantic_capabilities: ProviderSemanticCapabilities
    ports: tuple[ProviderPort, ...]
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    execution_capabilities: ProviderExecutionCapabilities
    compatibility_groups: tuple[CompatibilityGroup, ...] = ()
    eligible_node_classes: tuple[str, ...] = ()
    required_node_capabilities: tuple[str, ...] = ()
    evaluation_contract: ProviderEvaluationContract = Field(
        default_factory=ProviderEvaluationContract
    )
    immutable_image_digest: str | None = None

    @model_validator(mode="after")
    def _validate_ports(self) -> "ProviderContract":
        names = [port.name for port in self.ports]
        if len(names) != len(set(names)):
            raise ValueError("provider port names must be unique")
        input_names = {
            port.name
            for port in self.ports
            if port.kind in (ProviderPortKind.INPUT, ProviderPortKind.STATE_INPUT)
        }
        for group in self.compatibility_groups:
            unknown = set(group.ports) - input_names
            if unknown:
                raise ValueError(f"compatibility group references unknown input ports: {unknown}")
        return self

class ProviderFamily(VersionedModel):
    """Discoverable group of implementations serving related predicates."""
    SCHEMA_VERSION: ClassVar[str] = "fable.provider_family.v1"
    schema_version: Literal["fable.provider_family.v1"] = SCHEMA_VERSION
    family_id: NonEmptyStr
    description: NonEmptyStr
    predicate_ids: tuple[str, ...]
    provider_contract_ids: tuple[str, ...]
    acceptable_input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _require_members(self) -> "ProviderFamily":
        if not self.predicate_ids:
            raise ValueError("provider family requires at least one predicate")
        if not self.provider_contract_ids:
            raise ValueError("provider family requires at least one contract")
        return self
