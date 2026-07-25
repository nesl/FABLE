"""Typed records for FABLE Phase 2 and Phase 3 planning.

The semantic graph and hypothesis remain authoritative.  These records describe
how a provider-independent predicate demand may be realized physically through
one semantic checkpoint.  They intentionally do not advance hypotheses.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from fable.common.base import FableModel, FrozenFableModel, JSONValue
from fable.common.enums import (
    ArtifactAccessMode,
    BindingCapability,
    ExecutionMode,
    ResultKind,
)
from fable.common.time import EventTimeInterval, ensure_utc, utc_now


class PredicateExpressionKind(StrEnum):
    STATE = "STATE"
    TRANSITION = "TRANSITION"
    DERIVED_BEHAVIOR = "DERIVED_BEHAVIOR"
    SENSOR_EVENT = "SENSOR_EVENT"
    AGGREGATE = "AGGREGATE"


class PredicateRoleSchema(FableModel):
    role_name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    binding_capabilities: tuple[BindingCapability, ...]
    identity_required: bool = False

    @model_validator(mode="after")
    def _require_capabilities(self) -> "PredicateRoleSchema":
        if not self.binding_capabilities:
            raise ValueError("predicate role requires at least one binding capability")
        return self


class PredicateParameterSchema(FableModel):
    type: Literal["string", "integer", "number", "boolean"]
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    enum: tuple[JSONValue, ...] = ()
    default: JSONValue = None

    @model_validator(mode="after")
    def _validate_range(self) -> "PredicateParameterSchema":
        if self.minimum is not None and self.maximum is not None:
            if self.maximum < self.minimum:
                raise ValueError("parameter maximum must be >= minimum")
        return self


class LogicalPredicateSchema(FableModel):
    predicate_id: str = Field(min_length=1)
    expression_kind: PredicateExpressionKind
    result_kind: ResultKind
    roles: tuple[PredicateRoleSchema, ...] = ()
    parameters: dict[str, PredicateParameterSchema] = Field(default_factory=dict)
    provider_family_ids: tuple[str, ...]
    acceptable_output_types: tuple[str, ...] = ("predicate_match.v1",)
    required_capabilities: tuple[str, ...] = ()
    required_input_artifact_types: tuple[str, ...] = ()
    default_continuation_types: tuple[str, ...] = ()
    forkable_roles: tuple[str, ...] = ()
    fork_unit: str = "CANONICAL_BINDING_TUPLE"

    @model_validator(mode="after")
    def _validate_schema(self) -> "LogicalPredicateSchema":
        role_names = [role.role_name for role in self.roles]
        if len(role_names) != len(set(role_names)):
            raise ValueError("logical predicate role names must be unique")
        if not self.provider_family_ids:
            raise ValueError("logical predicate requires at least one provider family")
        if not self.acceptable_output_types:
            raise ValueError("logical predicate requires at least one output type")
        unknown_fork_roles = set(self.forkable_roles) - set(role_names)
        if unknown_fork_roles:
            raise ValueError(f"forkable roles are not declared: {sorted(unknown_fork_roles)}")
        return self


class DataTypeDefinition(FableModel):
    data_type: str = Field(min_length=1)
    description: str = ""
    kind: str = Field(min_length=1)
    modality: str | None = None
    compatibility_keys: tuple[str, ...] = ()
    transferable: bool | Literal["policy_dependent"] = True
    remote_reference_allowed: bool = True
    inline_allowed: bool = False
    continuation_eligible: bool = False
    continuation_category: str | None = None


class ChainExternalInput(FableModel):
    name: str = Field(min_length=1)
    data_type: str = Field(min_length=1)
    optional: bool = False


class ChainStep(FableModel):
    step_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    bindings: dict[str, str] = Field(default_factory=dict)


class ProviderChainContract(FableModel):
    chain_id: str = Field(min_length=1)
    description: str = ""
    predicate_ids: tuple[str, ...]
    external_inputs: tuple[ChainExternalInput, ...]
    steps: tuple[ChainStep, ...]
    outputs: dict[str, str]
    output_types: dict[str, str]
    continuation_output_types: tuple[str, ...] = ()
    capability_tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_chain(self) -> "ProviderChainContract":
        if not self.steps:
            raise ValueError("provider chain requires at least one step")
        if not self.predicate_ids:
            raise ValueError("provider chain must implement at least one predicate")
        if "result" not in self.outputs:
            raise ValueError("provider chain must expose a result output")
        return self


class ActiveProviderInstance(FableModel):
    provider_instance_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    configuration_hash: str = "default"
    source_ids: tuple[str, ...] = ()
    output_data_types: tuple[str, ...] = ()
    available: bool = True


class ProviderResourceProfile(FableModel):
    provider_id: str = Field(min_length=1)
    node_class: str = Field(min_length=1)
    startup_ms: int = Field(default=0, ge=0)
    execution_ms: int = Field(default=0, ge=0)
    cpu_cores: float = Field(default=0.1, ge=0)
    memory_mb: int = Field(default=64, ge=0)
    gpu_memory_mb: int = Field(default=0, ge=0)
    quality_score: float = Field(default=1.0, ge=0, le=1)


class ArtifactQueryRejection(FrozenFableModel):
    artifact_id: UUID
    code: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ArtifactQueryResult(FableModel):
    matches: tuple[UUID, ...] = ()
    rejections: tuple[ArtifactQueryRejection, ...] = ()


class ComputeCapacity(FableModel):
    cpu_cores: float = Field(ge=0)
    memory_mb: int = Field(ge=0)
    gpu_memory_mb: int = Field(default=0, ge=0)


class DeploymentNode(FableModel):
    node_id: str = Field(min_length=1)
    node_class: str = Field(min_length=1)
    region: str = Field(min_length=1)
    capacity: ComputeCapacity
    capabilities: tuple[str, ...] = ()
    policy_tags: tuple[str, ...] = ()
    available: bool = True


class SensorSource(FableModel):
    source_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    modalities: tuple[str, ...]
    live_data_types: tuple[str, ...]
    coverage_regions: tuple[str, ...] = ()
    raw_buffer_interval: EventTimeInterval | None = None
    policy_tags: tuple[str, ...] = ()
    available: bool = True


class NetworkLink(FableModel):
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    latency_ms: int = Field(default=0, ge=0)
    bandwidth_mbps: float = Field(gt=0)
    policy_tags: tuple[str, ...] = ()
    available: bool = True
    bidirectional: bool = True


class NetworkPath(FableModel):
    node_ids: tuple[str, ...]
    latency_ms: int = Field(ge=0)
    bottleneck_bandwidth_mbps: float = Field(gt=0)

    @model_validator(mode="after")
    def _require_path(self) -> "NetworkPath":
        if len(self.node_ids) < 2:
            raise ValueError("network path requires at least two nodes")
        return self


class ExternalInputKind(StrEnum):
    LIVE_SOURCE = "LIVE_SOURCE"
    RETAINED_ARTIFACT = "RETAINED_ARTIFACT"
    DEPLOYMENT_ARTIFACT = "DEPLOYMENT_ARTIFACT"
    OMITTED_OPTIONAL = "OMITTED_OPTIONAL"


class ExternalInputRealization(FrozenFableModel):
    input_name: str = Field(min_length=1)
    data_type: str = Field(min_length=1)
    kind: ExternalInputKind
    node_id: str | None = None
    source_id: str | None = None
    artifact_id: UUID | None = None
    bytes: int = Field(default=0, ge=0)
    access_modes: tuple[ArtifactAccessMode, ...] = ()
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def _normalize_expires_at(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)


class StepPlacement(FrozenFableModel):
    step_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    node_class: str = Field(min_length=1)
    startup_ms: int = Field(ge=0)
    execution_ms: int = Field(ge=0)
    cpu_cores: float = Field(ge=0)
    memory_mb: int = Field(ge=0)
    gpu_memory_mb: int = Field(ge=0)
    quality_score: float = Field(ge=0, le=1)
    reused_provider_instance_id: str | None = None


class TransferMode(StrEnum):
    LOCAL = "LOCAL"
    TRANSFER = "TRANSFER"
    REMOTE_REFERENCE = "REMOTE_REFERENCE"


class DataTransfer(FrozenFableModel):
    source_ref: str = Field(min_length=1)
    target_step_id: str = Field(min_length=1)
    target_port: str = Field(min_length=1)
    data_type: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    mode: TransferMode
    bytes: int = Field(default=0, ge=0)
    estimated_ms: int = Field(default=0, ge=0)
    path_node_ids: tuple[str, ...] = ()


class AlternativeNodeKind(StrEnum):
    LIVE_SOURCE = "LIVE_SOURCE"
    RETAINED_ARTIFACT = "RETAINED_ARTIFACT"
    PROVIDER_OPERATION = "PROVIDER_OPERATION"
    TRANSFER = "TRANSFER"
    CONTINUATION_SINK = "CONTINUATION_SINK"
    CHECKPOINT_RESULT_SINK = "CHECKPOINT_RESULT_SINK"


class AlternativeEdgeKind(StrEnum):
    DATA = "DATA"
    PRODUCES = "PRODUCES"
    SATISFIES = "SATISFIES"


class AlternativeGraphNode(FableModel):
    node_id: str = Field(min_length=1)
    kind: AlternativeNodeKind
    label: str = Field(min_length=1)
    demand_id: UUID
    chain_id: str | None = None
    step_id: str | None = None
    provider_id: str | None = None
    execution_node_id: str | None = None
    data_type: str | None = None
    source_id: str | None = None
    artifact_id: UUID | None = None
    annotations: dict[str, JSONValue] = Field(default_factory=dict)


class AlternativeGraphEdge(FableModel):
    edge_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    kind: AlternativeEdgeKind
    data_type: str | None = None
    annotations: dict[str, JSONValue] = Field(default_factory=dict)


class PhysicalAlternative(FableModel):
    alternative_id: str = Field(min_length=1)
    demand_id: UUID
    checkpoint_id: UUID
    chain_id: str = Field(min_length=1)
    execution_mode: ExecutionMode
    external_inputs: tuple[ExternalInputRealization, ...]
    step_placements: tuple[StepPlacement, ...]
    transfers: tuple[DataTransfer, ...]
    result_output_type: str = Field(min_length=1)
    continuation_output_types: tuple[str, ...] = ()
    estimated_completion_ms: int = Field(ge=0)
    estimated_transfer_bytes: int = Field(ge=0)
    minimum_quality_score: float = Field(ge=0, le=1)
    graph_node_ids: tuple[str, ...]
    graph_edge_ids: tuple[str, ...]
    spatial_preference_penalty: int = Field(default=0, ge=0)
    spatial_preference_reason: str = ""


class PrunedAlternative(FableModel):
    candidate_id: str = Field(min_length=1)
    demand_id: UUID
    chain_id: str = Field(min_length=1)
    code: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class PhysicalAlternativeGraph(FableModel):
    graph_id: str = Field(min_length=1)
    checkpoint_ids: tuple[UUID, ...]
    demand_ids: tuple[UUID, ...]
    nodes: tuple[AlternativeGraphNode, ...]
    edges: tuple[AlternativeGraphEdge, ...]
    alternatives: tuple[PhysicalAlternative, ...]
    pruned: tuple[PrunedAlternative, ...] = ()
    built_at: datetime = Field(default_factory=utc_now)

    @field_validator("built_at")
    @classmethod
    def _normalize_built_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_graph(self) -> "PhysicalAlternativeGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("alternative graph node IDs must be unique")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("alternative graph edge IDs must be unique")
        node_set = set(node_ids)
        for edge in self.edges:
            if edge.source_node_id not in node_set or edge.target_node_id not in node_set:
                raise ValueError("alternative graph edge references an unknown node")
        known_demands = set(self.demand_ids)
        if any(alt.demand_id not in known_demands for alt in self.alternatives):
            raise ValueError("physical alternative references demand outside graph horizon")
        return self
