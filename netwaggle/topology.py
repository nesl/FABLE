from __future__ import annotations

from dataclasses import dataclass, field
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
    reverse: dict[str, Any] | None = None

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
            reverse=data.get("reverse"),
        )

    def tc_params(self, *, reverse: bool = False) -> dict[str, Any]:
        if reverse:
            return dict(self.reverse or self.tc_params())
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
class ExternalProxy:
    """Allowlisted TCP bridge between a physical peer and a NetWaggle namespace."""

    name: str
    listen_host: str
    listen_port: int
    target_host: str
    target_port: int
    protocol: str = "tcp"
    listen_namespace_anchor: str | None = None
    outbound_namespace_anchor: str | None = None
    expected_source_ip: str | None = None
    allowed_peer_ips: tuple[str, ...] = ()
    required: bool = False
    connect_timeout_seconds: float = 5.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, allowed_anchors: set[str]) -> "ExternalProxy":
        fields = {
            "name", "listen_host", "listen_port", "target_host", "target_port",
            "protocol", "listen_namespace_anchor", "outbound_namespace_anchor",
            "expected_source_ip", "allowed_peer_ips", "required",
            "connect_timeout_seconds",
        }
        unknown = set(data) - fields
        if unknown:
            raise ValueError("unknown external proxy fields: " + ", ".join(sorted(unknown)))
        values = dict(data)
        for key in ("listen_namespace_anchor", "outbound_namespace_anchor"):
            anchor = values.get(key)
            if anchor and anchor not in allowed_anchors:
                raise ValueError(f"unknown anchor {anchor!r} for external proxy")
        values["allowed_peer_ips"] = tuple(values.get("allowed_peer_ips", ()))
        proxy = cls(**values)
        if proxy.protocol != "tcp":
            raise ValueError("external proxy protocol must be tcp")
        if not (1 <= proxy.listen_port <= 65535 and 1 <= proxy.target_port <= 65535):
            raise ValueError("external proxy ports must be between 1 and 65535")
        return proxy


@dataclass(frozen=True)
class NetWaggleTopology:
    name: str
    switches: list[str]
    links: list[Link]
    gateway: Gateway
    attachments: list[DockerAttachment]
    external_proxies: list[ExternalProxy] = field(default_factory=list)

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
        allowed_anchors = {item.anchor_container for item in attachments}
        external_proxies = [
            ExternalProxy.from_dict(item, allowed_anchors=allowed_anchors)
            for item in data.get("external_proxies", [])
        ]
        return cls(
            name=data.get("name", "netwaggle"),
            switches=switches,
            links=[Link.from_dict(x) for x in data.get("links", [])],
            gateway=gw,
            attachments=attachments,
            external_proxies=external_proxies,
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
            external_proxies=self.external_proxies,
        )

