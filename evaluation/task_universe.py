"""Structural complete-task demand compilation for B0, B1, and B3."""

from __future__ import annotations

from collections import defaultdict

from fable.common.enums import HypothesisNodeStatus
from fable.common.schemas import FrontierSnapshot, Hypothesis, PredicateDemand
from fable.planning import DemandCompileContext, DemandCompiler
from fable.planning.demand_compiler import DemandCompileError
from fable.semantic.compiled import CompiledSemanticGraph
from fable.semantic.frontier import FrontierDeriver
from fable.semantic.models import DerivedFrontier


_TERMINAL = {
    HypothesisNodeStatus.SATISFIED,
    HypothesisNodeStatus.FAILED,
    HypothesisNodeStatus.INVALIDATED,
    HypothesisNodeStatus.EXPIRED,
}


class TaskDemandUniverseBuilder:
    """Compile every remaining primitive structurally, including all branches."""

    def __init__(self, compiler: DemandCompiler) -> None:
        self.compiler = compiler

    def build(
        self,
        *,
        graph: CompiledSemanticGraph,
        hypothesis: Hypothesis,
        context: DemandCompileContext | None = None,
        skip_uncompilable: bool = False,
    ) -> tuple[PredicateDemand, ...]:
        context = context or DemandCompileContext()
        deriver = FrontierDeriver(graph)
        deriver.initialize_node_states(hypothesis)
        node_ids = tuple(
            node_id
            for node_id in graph.executable_predicate_nodes()
            if hypothesis.node_states[node_id].status not in _TERMINAL
        )
        if not node_ids:
            return ()
        grouped: dict[str, list[str]] = defaultdict(list)
        for node_id in node_ids:
            grouped[graph.nearest_checkpoint_boundary(node_id)].append(node_id)
        checkpoints = tuple(
            deriver._checkpoint_for_group(  # noqa: SLF001 - shared semantic rule
                hypothesis,
                boundary_id,
                tuple(sorted(group)),
            )
            for boundary_id, group in sorted(grouped.items())
        )
        snapshot = FrontierSnapshot(
            request_id=hypothesis.request_id,
            graph_hash=hypothesis.graph_hash,
            hypothesis_id=hypothesis.hypothesis_id,
            hypothesis_version=hypothesis.version,
            enabled_node_ids=tuple(sorted(node_ids)),
            checkpoint_ids=tuple(item.checkpoint_id for item in checkpoints),
        )
        frontier = DerivedFrontier(snapshot=snapshot, checkpoints=checkpoints)
        demands = []
        for node_id in node_ids:
            try:
                demands.append(
                    self.compiler.compile_node(
                        graph=graph,
                        hypothesis=hypothesis,
                        frontier=frontier,
                        graph_node_id=node_id,
                        context=context,
                        structural_universe=True,
                    )
                )
            except DemandCompileError:
                if not skip_uncompilable:
                    raise
        return tuple(
            sorted(
                demands,
                key=lambda item: (
                    graph.nodes_by_id[item.graph_node_id].authored_key,
                    str(item.demand_id),
                ),
            )
        )
