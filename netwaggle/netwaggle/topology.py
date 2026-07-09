from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .docker_attach import DockerAttachment


@dataclass(frozen=True)
class Gateway:
    ip: str
    switch: str
    host_ifname: str = "nwg-host"
    switch_ifname: str = "nwg-sw"
    mtu: int = 1500

    @property
    def ip_without_prefix(self) -> str:
        return self.ip.split("/", 1)[0]


@dataclass(frozen=True)
class Link:
    src: str
    dst: str
    bw: float | None = None
    delay: str | None = None
    jitter: str | None = None
    loss: float | None = None
    max_queue_size: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Link":
        return cls(
            src=data["from"],
            dst=data["to"],
            bw=data.get("bw"),
            delay=data.get("delay"),
            jitter=data.get("jitter"),
            loss=data.get("loss"),
            max_queue_size=data.get("max_queue_size"),
        )

    def tc_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.bw is not None:
            params["bw"] = self.bw
        if self.delay is not None:
            params["delay"] = self.delay
        if self.jitter is not None:
            params["jitter"] = self.jitter
        if self.loss is not None:
            params["loss"] = self.loss
        if self.max_queue_size is not None:
            params["max_queue_size"] = self.max_queue_size
        return params


@dataclass(frozen=True)
class NetWaggleTopology:
    name: str
    switches: list[str]
    links: list[Link]
    gateway: Gateway
    attachments: list[DockerAttachment]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetWaggleTopology":
        gw = Gateway(**data.get("gateway", {}))
        nodes = data.get("logical_nodes", [])
        attachments: list[DockerAttachment] = []
        for node in nodes:
            attachments.append(
                DockerAttachment(
                    name=node["name"],
                    anchor_container=node.get("anchor_container", f"netwaggle-node-{node['name']}"),
                    switch=node["switch"],
                    ip=node["ip"],
                    gateway=node.get("gateway", gw.ip_without_prefix),
                    mtu=int(node.get("mtu", 1500)),
                    container_ifname=node.get("container_ifname", "netwaggle0"),
                    switch_ifname=node.get("switch_ifname"),
                )
            )
        switches = list(data.get("switches", []))
        for required in [gw.switch] + [a.switch for a in attachments]:
            if required not in switches:
                switches.append(required)
        return cls(
            name=data.get("name", "netwaggle"),
            switches=switches,
            links=[Link.from_dict(x) for x in data.get("links", [])],
            gateway=gw,
            attachments=attachments,
        )

    def with_profile(self, profile: dict[str, Any] | None) -> "NetWaggleTopology":
        if not profile:
            return self
        if "links" not in profile:
            return self
        profile_links = [Link.from_dict(x) for x in profile["links"]]
        return NetWaggleTopology(
            name=f"{self.name}:{profile.get('name', 'profile')}",
            switches=self.switches,
            links=profile_links,
            gateway=self.gateway,
            attachments=self.attachments,
        )
