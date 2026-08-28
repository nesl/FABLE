"""Canonical home for FABLE complex-event semantic graph definitions.

Evaluation code, request compilation, demos, and new callers should import
graphs from this package. Compatibility modules at the historical import
locations remain available for older integrations.
"""

from fable.common.examples import convoy_graph

from .drive_up_shooting import drive_up_shooting_graph
from .exchange_rendezvous import exchange_rendezvous_graph
from .example_event import all_constructs_graph
from .arrival_with_person_presence import arrival_with_person_presence_graph
from .package_exchange import package_exchange_graph
from .repeated_visit import repeated_visit_graph, uncalibrated_repeated_pass_graph
from .robbery_with_alarm import (
    alarm_departure_graph,
    multimodal_robbery_graph,
    robbery_with_alarm_graph,
)
from .route_convoy import route_convoy_graph, sequential_vehicle_pass_graph
from .talking_rendezvous import talking_rendezvous_graph
from .two_vehicle_chase import two_vehicle_chase_graph
from .vehicle_convergence import vehicle_convergence_graph
from .registry import PRODUCTION_DEFINITIONS, get_definition

__all__ = [
    "alarm_departure_graph",
    "all_constructs_graph",
    "arrival_with_person_presence_graph",
    "convoy_graph",
    "drive_up_shooting_graph",
    "exchange_rendezvous_graph",
    "multimodal_robbery_graph",
    "package_exchange_graph",
    "repeated_visit_graph",
    "robbery_with_alarm_graph",
    "route_convoy_graph",
    "sequential_vehicle_pass_graph",
    "talking_rendezvous_graph",
    "two_vehicle_chase_graph",
    "uncalibrated_repeated_pass_graph",
    "vehicle_convergence_graph",
    "PRODUCTION_DEFINITIONS",
    "get_definition",
]
