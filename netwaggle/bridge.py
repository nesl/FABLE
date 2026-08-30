"""Translate measured NetWaggle links into the refactored planner state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from fable.planning import LinkState, RuntimeState


@dataclass(frozen=True)
class NetwaggleLinkObservation:
    source_node: str
    destination_node: str
    latency_ms: float
    bandwidth_mbps: float | None
    available: bool = True

    def to_link_state(self) -> LinkState:
        return LinkState(
            self.source_node,
            self.destination_node,
            self.latency_ms,
            self.bandwidth_mbps,
            self.available,
        )


def apply_link_observations(
    state: RuntimeState,
    observations: Iterable[NetwaggleLinkObservation],
) -> RuntimeState:
    """Return a state snapshot with only the observed directed links replaced."""

    replacements = {
        (row.source_node, row.destination_node): row.to_link_state()
        for row in observations
    }
    links = [
        replacements.pop((link.source_node, link.destination_node), link)
        for link in state.links
    ]
    links.extend(replacements.values())
    return replace(state, links=tuple(links))
