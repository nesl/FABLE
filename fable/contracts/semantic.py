"""Semantic data contracts for FABLE."""

from ._shared import *  # noqa: F401,F403

class RoleDefinition(FableModel):
    """Named event-level entity variable and its cardinality/type constraints."""
    role_name: NonEmptyStr
    entity_type: NonEmptyStr
    cardinality_min: int = Field(default=1, ge=0)
    cardinality_max: int | None = Field(default=1, ge=1)
    distinct_from: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_cardinality(self) -> "RoleDefinition":
        if self.cardinality_max is not None and self.cardinality_max < self.cardinality_min:
            raise ValueError("cardinality_max must be >= cardinality_min")
        return self

class PredicateRole(FableModel):
    """Binding from a predicate-local argument to an event-level variable."""
    role_name: NonEmptyStr
    variable: NonEmptyStr
    entity_type: NonEmptyStr

class SemanticPredicate(FableModel):
    """Provider-independent semantic condition."""

    predicate_id: NonEmptyStr
    roles: tuple[PredicateRole, ...] = ()
    parameters: dict[str, JSONValue] = Field(default_factory=dict)
    result_kind: ResultKind

    @model_validator(mode="after")
    def _unique_role_names(self) -> "SemanticPredicate":
        names = [role.role_name for role in self.roles]
        if len(names) != len(set(names)):
            raise ValueError("semantic predicate role names must be unique")
        return self

class TemporalGuard(FableModel):
    """Static event-time constraint attached to graph structure."""
    guard_id: NonEmptyStr
    kind: TemporalGuardKind
    source_node_ids: tuple[str, ...]
    target_node_id: str | None = None
    minimum_ms: int | None = Field(default=None, ge=0)
    maximum_ms: int | None = Field(default=None, ge=0)
    count: int | None = Field(default=None, ge=1)
    required_source_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_bounds(self) -> "TemporalGuard":
        if self.minimum_ms is not None and self.maximum_ms is not None:
            if self.maximum_ms < self.minimum_ms:
                raise ValueError("maximum_ms must be >= minimum_ms")
        if not self.source_node_ids:
            raise ValueError("temporal guard must reference at least one source node")
        if self.kind == TemporalGuardKind.REPEAT_WITHIN and self.count is None:
            raise ValueError("REPEAT_WITHIN requires count")
        return self

class GraphNode(FableModel):
    """One static predicate or structural operator in an Event Graph."""
    node_id: NonEmptyStr
    authored_key: NonEmptyStr
    kind: GraphNodeKind
    name: NonEmptyStr
    predicate: SemanticPredicate | None = None
    k: int | None = Field(default=None, ge=1)
    checkpoint_boundary: bool = False
    annotations: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> "GraphNode":
        if self.kind == GraphNodeKind.PREDICATE and self.predicate is None:
            raise ValueError("PREDICATE node requires predicate")
        if self.kind != GraphNodeKind.PREDICATE and self.predicate is not None:
            raise ValueError("only PREDICATE nodes may carry a semantic predicate")
        if self.kind == GraphNodeKind.K_OF_N and self.k is None:
            raise ValueError("K_OF_N node requires k")
        if self.kind != GraphNodeKind.K_OF_N and self.k is not None:
            raise ValueError("k is valid only for K_OF_N nodes")
        return self

class GraphEdge(FableModel):
    """Static hierarchy, alternative, sequence, or dependency relationship."""
    edge_id: NonEmptyStr
    source_node_id: NonEmptyStr
    target_node_id: NonEmptyStr
    kind: GraphEdgeKind
    temporal_guard_ids: tuple[str, ...] = ()
    branch_label: str | None = None

class SemanticGraph(VersionedModel):
    """Immutable, provider-independent definition of one complex-event family.

    A CE factory/compiler creates it; ``SemanticRuntime`` compiles indexes over
    and evaluates it. ``graph_hash`` is its deterministic content identity.
    """
    SCHEMA_VERSION: ClassVar[str] = "fable.semantic_graph.v1"
    schema_version: Literal["fable.semantic_graph.v1"] = SCHEMA_VERSION
    graph_id: NonEmptyStr
    graph_hash: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    graph_version: int = Field(default=1, ge=1)
    name: NonEmptyStr
    description: str | None = None
    root_node_id: NonEmptyStr
    roles: tuple[RoleDefinition, ...] = ()
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...] = ()
    temporal_guards: tuple[TemporalGuard, ...] = ()
    authored_variant_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_references(self) -> "SemanticGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node identifiers must be unique")
        authored_keys = [node.authored_key for node in self.nodes]
        if len(authored_keys) != len(set(authored_keys)):
            raise ValueError("graph authored keys must be unique")
        if self.root_node_id not in set(node_ids):
            raise ValueError("root_node_id does not reference a graph node")
        guard_ids = [guard.guard_id for guard in self.temporal_guards]
        if len(guard_ids) != len(set(guard_ids)):
            raise ValueError("temporal guard identifiers must be unique")
        node_set = set(node_ids)
        guard_set = set(guard_ids)
        for edge in self.edges:
            if edge.source_node_id not in node_set or edge.target_node_id not in node_set:
                raise ValueError(f"edge {edge.edge_id} references an unknown node")
            unknown_guards = set(edge.temporal_guard_ids) - guard_set
            if unknown_guards:
                raise ValueError(f"edge {edge.edge_id} references unknown guards {unknown_guards}")
        for guard in self.temporal_guards:
            if set(guard.source_node_ids) - node_set:
                raise ValueError(f"guard {guard.guard_id} references an unknown source node")
            if guard.target_node_id and guard.target_node_id not in node_set:
                raise ValueError(f"guard {guard.guard_id} references an unknown target node")
        return self
