"""Compatibility re-exports for historical vehicle-grouped imports.

Production definitions live in per-complex-event modules; the all-constructs
graph is explicitly a teaching/test example.
"""

from .arrival_with_person_presence import arrival_with_person_presence_graph
from .example_event import all_constructs_graph
from .repeated_visit import repeated_visit_graph, uncalibrated_repeated_pass_graph
from .route_convoy import sequential_vehicle_pass_graph
from .talking_rendezvous import talking_rendezvous_graph
from .two_vehicle_chase import two_vehicle_chase_graph
from .vehicle_convergence import vehicle_convergence_graph

__all__ = [
    "all_constructs_graph",
    "arrival_with_person_presence_graph",
    "repeated_visit_graph",
    "sequential_vehicle_pass_graph",
    "talking_rendezvous_graph",
    "two_vehicle_chase_graph",
    "uncalibrated_repeated_pass_graph",
    "vehicle_convergence_graph",
]
