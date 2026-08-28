"""NetWaggle profiles as first-class FABLE evaluation inputs."""

from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter_ns
from pathlib import Path

from pydantic import Field, model_validator

from fable.common.base import FableModel
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import NetworkLink
from fable.planning.models import PhysicalAlternativeGraph
from evaluation.baselines.models import BaselinePlanningCase
from evaluation.disturbance_schedule import DisturbanceKind, ScheduledDisturbanceAction
from evaluation.schemas import BaselineId, NetworkCondition


class EmulatedLink(FableModel):
    source_switch: str = Field(min_length=1)
    target_switch: str = Field(min_length=1)
    latency_ms: int = Field(ge=0)
    bandwidth_mbps: float = Field(gt=0)
    packet_loss_fraction: float = Field(default=0, ge=0, le=1)


class EvaluationNetworkProfile(FableModel):
    profile_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_path: Path
    links: tuple[EmulatedLink, ...]
    description: str = ""

    @model_validator(mode="after")
    def _require_links(self):
        if not self.links:
            raise ValueError("network profile requires at least one link")
        return self


@dataclass(frozen=True)
class AppliedNetworkProfile:
    profile_id: str
    resource_epoch: int
    deployment: DeploymentGraph
    packet_loss_by_link: dict[str, float]


DEFAULT_NETWAGGLE_NODE_SWITCHES = {
    "dvpg_gq_orin_11": "s_orin11",
    "x86server": "s_edge",
    "cloud1": "s_cloud",
}


def load_netwaggle_profile(path: str | Path) -> EvaluationNetworkProfile:
    source = Path(path).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    links = tuple(
        EmulatedLink(
            source_switch=item["from"],
            target_switch=item["to"],
            latency_ms=_parse_delay_ms(item.get("delay", "0ms")),
            bandwidth_mbps=float(item.get("bw", 1_000)),
            packet_loss_fraction=float(item.get("loss", 0)) / 100.0,
        )
        for item in document.get("links", ())
    )
    return EvaluationNetworkProfile(
        profile_id=str(document.get("name") or source.stem).lower(),
        source_path=source,
        links=links,
        description=str(document.get("description", "")),
    )


def retarget_sensor_uplink_profile(
    profile: EvaluationNetworkProfile,
    *,
    target_switch: str,
    template_switch: str = "s_orin11",
) -> EvaluationNetworkProfile:
    """Apply a sensor-uplink condition to the sensor named by the trace.

    The canonical L1 profile uses ``s_orin11`` as its representative impaired
    uplink. Live RQ3a traces can select another replay sensor, so the planner
    must see the same target that the host controller shapes.
    """

    if re.fullmatch(r"s_orin\d+", target_switch) is None:
        raise ValueError(f"invalid sensor switch target: {target_switch}")
    template = next(
        (
            item
            for item in profile.links
            if {item.source_switch, item.target_switch}
            == {template_switch, "s_edge"}
        ),
        None,
    )
    if template is None:
        raise ValueError(
            f"profile {profile.profile_id} has no {template_switch}<->s_edge template"
        )
    links = tuple(
        item
        for item in profile.links
        if {item.source_switch, item.target_switch} != {target_switch, "s_edge"}
    ) + (
        template.model_copy(
            update={"source_switch": target_switch, "target_switch": "s_edge"}
        ),
    )
    return profile.model_copy(update={"links": links})


def apply_netwaggle_profile(
    deployment: DeploymentGraph,
    profile: EvaluationNetworkProfile,
    *,
    resource_epoch: int,
    node_switches: dict[str, str] | None = None,
) -> AppliedNetworkProfile:
    """Return a deployment whose costs match the actual Mininet/TC profile."""

    mapping = dict(DEFAULT_NETWAGGLE_NODE_SWITCHES)
    if node_switches:
        mapping.update(node_switches)
    for node_id in deployment.nodes:
        match = re.fullmatch(r"dvpg_gq_orin_?(\d+)", node_id)
        if match and node_id not in mapping:
            mapping[node_id] = f"s_orin{int(match.group(1))}"
    missing = sorted(set(deployment.nodes) - set(mapping))
    if missing:
        raise ValueError(
            "no NetWaggle switch mapping for deployment nodes: " + ", ".join(missing)
        )

    links: list[NetworkLink] = []
    loss_by_link: dict[str, float] = {}
    node_ids = sorted(deployment.nodes)
    for index, source in enumerate(node_ids):
        for target in node_ids[index + 1 :]:
            path = _shortest_switch_path(
                profile.links, mapping[source], mapping[target]
            )
            if path is None:
                continue
            latency, bandwidth, loss = path
            key = f"{source}->{target}"
            loss_by_link[key] = loss
            links.append(
                NetworkLink(
                    source_node_id=source,
                    target_node_id=target,
                    latency_ms=latency,
                    # The planner has no packet-loss dimension. Applying the
                    # expected successful-throughput factor avoids silently
                    # treating a lossy link as its raw TC bandwidth.
                    bandwidth_mbps=max(0.001, bandwidth * (1.0 - loss)),
                    available=loss < 1.0,
                    bidirectional=True,
                    policy_tags=(
                        f"netwaggle:{profile.profile_id}",
                        f"packet_loss:{loss:.6f}",
                    ),
                )
            )
    profiled = DeploymentGraph(
        nodes=deployment.nodes.values(),
        sources=deployment.sources.values(),
        links=links,
        resource_pools=deployment.resource_pools,
    )
    return AppliedNetworkProfile(
        profile_id=profile.profile_id,
        resource_epoch=resource_epoch,
        deployment=profiled,
        packet_loss_by_link=loss_by_link,
    )


def discover_netwaggle_profiles(root: str | Path) -> tuple[EvaluationNetworkProfile, ...]:
    return tuple(
        load_netwaggle_profile(path)
        for path in sorted(Path(root).glob("*.json"))
    )


class NetworkExperimentState:
    """Apply profiles and advance the resource epoch only for real changes."""

    def __init__(
        self,
        deployment: DeploymentGraph,
        *,
        node_switches: dict[str, str] | None = None,
    ) -> None:
        self.base_deployment = deployment
        self.node_switches = node_switches
        self.resource_epoch = 0
        self._profile_id: str | None = None

    def activate(
        self, profile: EvaluationNetworkProfile
    ) -> AppliedNetworkProfile:
        if self._profile_id is not None and profile.profile_id != self._profile_id:
            self.resource_epoch += 1
        self._profile_id = profile.profile_id
        return apply_netwaggle_profile(
            self.base_deployment,
            profile,
            resource_epoch=self.resource_epoch,
            node_switches=self.node_switches,
        )


class ProfiledNetworkActionApplier:
    """Apply typed disturbance conditions to planner-visible network state."""

    def __init__(
        self,
        *,
        deployment: DeploymentGraph,
        profiles: dict[str, EvaluationNetworkProfile],
        run_id: str,
        baseline_id: BaselineId,
        trace_id: str,
        request_id: str,
        record_sink,
        node_switches: dict[str, str] | None = None,
    ) -> None:
        if not profiles:
            raise ValueError("profiled network controller requires condition profiles")
        self.deployment = deployment
        self.profiles = dict(profiles)
        self.run_id = run_id
        self.baseline_id = baseline_id
        self.trace_id = trace_id
        self.request_id = request_id
        self.record_sink = record_sink
        self.node_switches = node_switches
        self.latest: AppliedNetworkProfile | None = None

    def __call__(
        self,
        action: ScheduledDisturbanceAction,
        condition_epoch: int,
    ) -> dict[str, int | float | str | bool]:
        if action.kind != DisturbanceKind.NETWORK_PROFILE:
            raise ValueError("network applier accepts only NETWORK_PROFILE actions")
        try:
            profile = self.profiles[action.condition_id]
        except KeyError as exc:
            raise ValueError(
                f"no planner-visible network profile for {action.condition_id}"
            ) from exc
        applied = apply_netwaggle_profile(
            self.deployment,
            profile,
            resource_epoch=condition_epoch,
            node_switches=self.node_switches,
        )
        records = network_condition_records(
            applied,
            run_id=self.run_id,
            baseline_id=self.baseline_id,
            trace_id=self.trace_id,
            request_id=self.request_id,
            event_time=action.due_at,
        )
        for record in records:
            self.record_sink(record)
        self.latest = applied
        links = tuple(applied.deployment.links)
        return {
            "profile_id": applied.profile_id,
            "network_links": len(links),
            "maximum_latency_ms": max(
                (item.latency_ms for item in links),
                default=0,
            ),
            "minimum_bandwidth_mbps": min(
                (item.bandwidth_mbps for item in links),
                default=0,
            ),
            "unavailable_links": sum(not item.available for item in links),
        }


def bind_network_to_planning_case(
    case: BaselinePlanningCase,
    applied: AppliedNetworkProfile,
    *,
    frontier_graph: PhysicalAlternativeGraph,
    whole_event_graph: PhysicalAlternativeGraph,
) -> BaselinePlanningCase:
    """Bind graphs recompiled with the profile and its resource epoch."""

    return BaselinePlanningCase(
        **{
            **case.__dict__,
            "frontier_graph": frontier_graph,
            "whole_event_graph": whole_event_graph,
            "resource_epoch": applied.resource_epoch,
        }
    )


def network_condition_records(
    applied: AppliedNetworkProfile,
    *,
    run_id: str,
    baseline_id: BaselineId,
    trace_id: str,
    request_id: str,
    event_time: datetime,
) -> tuple[NetworkCondition, ...]:
    """Create common evaluation records from the planner-visible profile."""

    return tuple(
        NetworkCondition(
            run_id=run_id,
            baseline_id=baseline_id,
            trace_id=trace_id,
            request_id=request_id,
            event_time=event_time,
            monotonic_timestamp_ns=perf_counter_ns(),
            source_node_id=link.source_node_id,
            target_node_id=link.target_node_id,
            latency_ms=link.latency_ms,
            bandwidth_mbps=link.bandwidth_mbps,
            packet_loss_fraction=applied.packet_loss_by_link.get(
                f"{link.source_node_id}->{link.target_node_id}", 0.0
            ),
            available=link.available,
            condition_epoch=applied.resource_epoch,
            metadata={"netwaggle_profile": applied.profile_id},
        )
        for link in applied.deployment.links
    )


def _parse_delay_ms(value: object) -> int:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)?\s*", str(value))
    if not match:
        raise ValueError(f"unsupported NetWaggle delay: {value!r}")
    amount = float(match.group(1))
    return round(amount * 1_000) if match.group(2) == "s" else round(amount)


def _shortest_switch_path(
    links: tuple[EmulatedLink, ...], source: str, target: str
) -> tuple[int, float, float] | None:
    if source == target:
        return (0, float("inf"), 0.0)
    adjacency: dict[str, list[tuple[str, EmulatedLink]]] = {}
    for link in links:
        adjacency.setdefault(link.source_switch, []).append((link.target_switch, link))
        adjacency.setdefault(link.target_switch, []).append((link.source_switch, link))
    queue = [(0, source, float("inf"), 1.0)]
    best: dict[str, int] = {}
    while queue:
        latency, current, bandwidth, success = heapq.heappop(queue)
        if current in best and best[current] <= latency:
            continue
        best[current] = latency
        if current == target:
            return latency, bandwidth, 1.0 - success
        for neighbor, link in adjacency.get(current, ()):
            heapq.heappush(
                queue,
                (
                    latency + link.latency_ms,
                    neighbor,
                    min(bandwidth, link.bandwidth_mbps),
                    success * (1.0 - link.packet_loss_fraction),
                ),
            )
    return None
import re
from fable.common.identities import node_for_sensor_link, sensor_link_id
def validated_link_target_node_id(target_id: str) -> str:
    """Map an allowlisted NetWaggle sensor link to its deployment node."""

    try:
        node = node_for_sensor_link(target_id)
    except ValueError as exc:
        raise ValueError("link-state target is not an allowlisted sensor link")
    number = int(node.value.rsplit("_", 1)[1])
    if not 1 <= number <= 30:
        raise ValueError("link-state target is not an allowlisted sensor link")
    return node.value


def canonical_sensor_link_target(node_id: str) -> str:
    """Convert a deployment/replay node identity to its canonical sensor link."""

    target = sensor_link_id(node_id).value
    number = int(re.search(r"[0-9]+", target).group())
    if not 1 <= number <= 30:
        raise ValueError(f"node is not an allowlisted sensor identity: {node_id}")
    return target
