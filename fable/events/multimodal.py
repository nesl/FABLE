"""Compatibility exports for the canonical multimodal semantic definitions.

New code should import these builders from :mod:`fable.semantic.definitions`.
Keeping this module as a re-export preserves the historical public path without
maintaining a second, potentially divergent copy of each authored graph.
"""

from fable.semantic.definitions import (
    drive_up_shooting_graph,
    multimodal_robbery_graph,
    package_exchange_graph,
)

__all__ = [
    "drive_up_shooting_graph",
    "multimodal_robbery_graph",
    "package_exchange_graph",
]
