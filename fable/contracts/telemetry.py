"""Telemetry and runtime-deployment data contracts for FABLE."""

from ._shared import *  # noqa: F401,F403

class SourceHeartbeat(FableModel):
    """Latest event-time progress and retained interval for one sensor source."""
    source_id: NonEmptyStr
    latest_sequence: int = Field(ge=0)
    latest_event_time: datetime
    raw_buffer_interval: EventTimeInterval | None = None
    operational_coverage: bool = True
    replay_complete: bool = False

    @field_validator("latest_event_time")
    @classmethod
    def _normalize_latest_event_time(cls, value: datetime) -> datetime:
        return ensure_utc(value)

class NodeCapacity(FableModel):
    """Current free compute/network capacity sampled by a node agent."""
    cpu_free_cores: float = Field(ge=0)
    memory_free_mb: int = Field(ge=0)
    gpu_free_mb: int = Field(default=0, ge=0)
    network_tx_available_mbps: float | None = Field(default=None, ge=0)
    network_rx_available_mbps: float | None = Field(default=None, ge=0)
    # Runtime telemetry used by physical/E4 agents. These are planning signals,
    # not resource reservations, and therefore remain optional for older nodes.
    gpu_utilization: float | None = Field(default=None, ge=0, le=1)
    logical_accelerator_slots_available: int | None = Field(default=None, ge=0)
    provider_queue_delay_ms: float | None = Field(default=None, ge=0)

class NodeHeartbeat(VersionedModel):
    """Periodic node/resource/source observation sent by a ``NodeAgent``.

    Node agents build and publish it; the heartbeat monitor and
    ``RuntimeDeploymentView`` consume it to advance effective deployment and
    resource epochs. Session and sequence reject stale/out-of-order telemetry.
    """
    SCHEMA_VERSION: ClassVar[str] = "fable.node_heartbeat.v1"
    schema_version: Literal["fable.node_heartbeat.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    node_id: NonEmptyStr
    session_id: NonEmptyStr
    sequence: int = Field(ge=0)
    sent_at: datetime = Field(default_factory=utc_now)
    availability: NodeAvailability = NodeAvailability.AVAILABLE
    sources: dict[str, SourceHeartbeat] = Field(default_factory=dict)
    active_provider_instance_ids: tuple[str, ...] = ()
    active_demand_ids: tuple[UUID7, ...] = ()
    capacity: NodeCapacity

    @field_validator("sent_at")
    @classmethod
    def _normalize_sent_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _keys_match_sources(self) -> "NodeHeartbeat":
        mismatches = [key for key, value in self.sources.items() if key != value.source_id]
        if mismatches:
            raise ValueError(f"heartbeat source keys must match source_id: {mismatches}")
        return self

class RuntimeNodeUpdate(FrozenFableModel):
    """Typed override of the current resources/availability for one planning node."""

    node_id: NonEmptyStr
    available: bool | None = None
    cpu_available_cores: float | None = Field(default=None, ge=0)
    memory_available_mb: int | None = Field(default=None, ge=0)
    gpu_memory_available_mb: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _require_change(self) -> "RuntimeNodeUpdate":
        if (
            self.available is None
            and self.cpu_available_cores is None
            and self.memory_available_mb is None
            and self.gpu_memory_available_mb is None
        ):
            raise ValueError("runtime node update must change availability or capacity")
        return self

class RuntimeLinkUpdate(FrozenFableModel):
    """Typed override of one logical network link used by physical planning."""

    source_node_id: NonEmptyStr
    target_node_id: NonEmptyStr
    latency_ms: float | None = Field(default=None, ge=0)
    bandwidth_mbps: float | None = Field(default=None, gt=0)
    available: bool = True

    @model_validator(mode="after")
    def _different_endpoints(self) -> "RuntimeLinkUpdate":
        if self.source_node_id == self.target_node_id:
            raise ValueError("runtime link endpoints must differ")
        return self
