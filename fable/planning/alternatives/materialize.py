"""Materialize one physical alternative into graph nodes and edges."""

from __future__ import annotations

from fable.common.ids import deterministic_id
from fable.common.schemas import PredicateDemand
from fable.planning.models import (
    AlternativeEdgeKind,
    AlternativeGraphEdge,
    AlternativeGraphNode,
    AlternativeNodeKind,
    ExternalInputKind,
    ExternalInputRealization,
    TransferMode,
)
from fable.planning.provider_registry import ProviderRegistry
from fable.planning.alternatives.internal import PlacementState, add_edge


class AlternativeGraphMaterializer:
    """Create explanatory graph nodes/edges for an enumerated realization."""

    def __init__(self, *, provider_registry: ProviderRegistry) -> None:
        self.providers = provider_registry

    def _materialize_graph(
            self,
            *,
            alternative_id: str,
            demand: PredicateDemand,
            chain_id: str,
            assignment: tuple[ExternalInputRealization, ...],
            state: PlacementState,
        ) -> tuple[tuple[AlternativeGraphNode, ...], tuple[AlternativeGraphEdge, ...]]:
            chain = self.providers.chain(chain_id)
            nodes: dict[str, AlternativeGraphNode] = {}
            edges: dict[str, AlternativeGraphEdge] = {}
            source_graph_nodes: dict[str, str] = {}

            for item in assignment:
                if item.kind == ExternalInputKind.OMITTED_OPTIONAL:
                    continue
                node_id = deterministic_id(
                    "pan",
                    {"alternative": alternative_id, "external": item.input_name, "realization": item},
                )
                source_graph_nodes[f"external.{item.input_name}"] = node_id
                nodes[node_id] = AlternativeGraphNode(
                    node_id=node_id,
                    kind=(
                        AlternativeNodeKind.LIVE_SOURCE
                        if item.kind == ExternalInputKind.LIVE_SOURCE
                        else AlternativeNodeKind.RETAINED_ARTIFACT
                    ),
                    label=f"{item.input_name}: {item.data_type}",
                    demand_id=demand.demand_id,
                    chain_id=chain_id,
                    execution_node_id=item.node_id,
                    data_type=item.data_type,
                    source_id=item.source_id,
                    artifact_id=item.artifact_id,
                    annotations={
                        "input_kind": item.kind.value,
                        "bytes": item.bytes,
                        "expires_at": (
                            item.expires_at.isoformat().replace("+00:00", "Z")
                            if item.expires_at is not None
                            else None
                        ),
                    },
                )

            placements = {placement.step_id: placement for placement in state.steps}
            step_graph_nodes: dict[str, str] = {}
            for step in chain.steps:
                placement = placements[step.step_id]
                node_id = deterministic_id(
                    "pan",
                    {"alternative": alternative_id, "step": step.step_id, "placement": placement},
                )
                step_graph_nodes[step.step_id] = node_id
                nodes[node_id] = AlternativeGraphNode(
                    node_id=node_id,
                    kind=AlternativeNodeKind.PROVIDER_OPERATION,
                    label=f"{step.provider_id} @ {placement.node_id}",
                    demand_id=demand.demand_id,
                    chain_id=chain_id,
                    step_id=step.step_id,
                    provider_id=step.provider_id,
                    execution_node_id=placement.node_id,
                    annotations={
                        "startup_ms": placement.startup_ms,
                        "execution_ms": placement.execution_ms,
                        "quality_score": placement.quality_score,
                        "sharing_key": demand.sharing_key,
                        "reused_provider_instance_id": placement.reused_provider_instance_id,
                    },
                )

            transfer_lookup = {
                (transfer.target_step_id, transfer.target_port, transfer.source_ref): transfer
                for transfer in state.transfers
            }
            for step in chain.steps:
                target_node = step_graph_nodes[step.step_id]
                for port_name, source_ref in sorted(step.bindings.items()):
                    source_node = source_graph_nodes.get(source_ref)
                    if source_node is None and "." in source_ref:
                        producer_step = source_ref.split(".", 1)[0]
                        source_node = step_graph_nodes.get(producer_step)
                    if source_node is None:
                        continue
                    transfer = transfer_lookup.get((step.step_id, port_name, source_ref))
                    if transfer is not None and transfer.mode != TransferMode.LOCAL:
                        transfer_node_id = deterministic_id(
                            "pan",
                            {
                                "alternative": alternative_id,
                                "transfer": transfer,
                            },
                        )
                        nodes[transfer_node_id] = AlternativeGraphNode(
                            node_id=transfer_node_id,
                            kind=AlternativeNodeKind.TRANSFER,
                            label=f"{transfer.mode.value}: {transfer.data_type}",
                            demand_id=demand.demand_id,
                            chain_id=chain_id,
                            execution_node_id=transfer.target_node_id,
                            data_type=transfer.data_type,
                            annotations={
                                "source_node_id": transfer.source_node_id,
                                "target_node_id": transfer.target_node_id,
                                "bytes": transfer.bytes,
                                "estimated_ms": transfer.estimated_ms,
                                "path": list(transfer.path_node_ids),
                            },
                        )
                        add_edge(
                            edges,
                            alternative_id,
                            source_node,
                            transfer_node_id,
                            AlternativeEdgeKind.DATA,
                            transfer.data_type,
                        )
                        add_edge(
                            edges,
                            alternative_id,
                            transfer_node_id,
                            target_node,
                            AlternativeEdgeKind.DATA,
                            transfer.data_type,
                        )
                    else:
                        data_type = transfer.data_type if transfer is not None else None
                        add_edge(
                            edges,
                            alternative_id,
                            source_node,
                            target_node,
                            AlternativeEdgeKind.DATA,
                            data_type,
                        )

            result_ref = chain.outputs["result"]
            result_step = result_ref.split(".", 1)[0]
            result_sink_id = deterministic_id(
                "pan",
                {"alternative": alternative_id, "checkpoint_sink": str(demand.checkpoint_id)},
            )
            nodes[result_sink_id] = AlternativeGraphNode(
                node_id=result_sink_id,
                kind=AlternativeNodeKind.CHECKPOINT_RESULT_SINK,
                label=f"checkpoint {demand.checkpoint_id}",
                demand_id=demand.demand_id,
                chain_id=chain_id,
                data_type=chain.output_types["result"],
                annotations={"graph_node_id": demand.graph_node_id},
            )
            add_edge(
                edges,
                alternative_id,
                step_graph_nodes[result_step],
                result_sink_id,
                AlternativeEdgeKind.SATISFIES,
                chain.output_types["result"],
            )

            for output_name, output_ref in sorted(chain.outputs.items()):
                if output_name == "result":
                    continue
                data_type = chain.output_types[output_name]
                if data_type not in chain.continuation_output_types:
                    continue
                producer_step = output_ref.split(".", 1)[0]
                sink_id = deterministic_id(
                    "pan",
                    {"alternative": alternative_id, "continuation": output_name, "type": data_type},
                )
                nodes[sink_id] = AlternativeGraphNode(
                    node_id=sink_id,
                    kind=AlternativeNodeKind.CONTINUATION_SINK,
                    label=f"retain {data_type}",
                    demand_id=demand.demand_id,
                    chain_id=chain_id,
                    data_type=data_type,
                    annotations={
                        "required": data_type
                        in {req.artifact_type for req in demand.continuation_requirements},
                        "desired": data_type in set(demand.desired_continuation_types),
                    },
                )
                add_edge(
                    edges,
                    alternative_id,
                    step_graph_nodes[producer_step],
                    sink_id,
                    AlternativeEdgeKind.PRODUCES,
                    data_type,
                )
            return (
                tuple(sorted(nodes.values(), key=lambda node: node.node_id)),
                tuple(sorted(edges.values(), key=lambda edge: edge.edge_id)),
            )


__all__ = ["AlternativeGraphMaterializer"]
