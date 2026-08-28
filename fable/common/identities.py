"""Canonical identities and the only supported legacy-alias translation boundary."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class NodeId:
    value: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"(?:dvpg_gq_orin_[1-9][0-9]*|mobile_archive_[1-9][0-9]*|x86server)", self.value) is None:
            raise ValueError(f"invalid canonical node ID: {self.value}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class LinkId:
    value: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"link:s_(?:orin|mob)[1-9][0-9]*:s_edge", self.value) is None:
            raise ValueError(f"invalid canonical link ID: {self.value}")

    def __str__(self) -> str:
        return self.value


def canonical_node_id(value: str) -> NodeId:
    raw = value.strip()
    if raw == "x86server":
        return NodeId(raw)
    match = re.fullmatch(r"(?:dvpg_gq_)?orin_?([1-9][0-9]*)", raw)
    if match:
        return NodeId(f"dvpg_gq_orin_{match.group(1)}")
    match = re.fullmatch(r"(?:mobile_archive_|mobile_?|mob)([1-9][0-9]*)", raw)
    if match:
        return NodeId(f"mobile_archive_{match.group(1)}")
    raise ValueError(f"unknown node alias: {value}")


def sensor_link_id(value: str | NodeId) -> LinkId:
    node = value if isinstance(value, NodeId) else canonical_node_id(value)
    orin = re.fullmatch(r"dvpg_gq_orin_([1-9][0-9]*)", node.value)
    if orin:
        return LinkId(f"link:s_orin{orin.group(1)}:s_edge")
    mobile = re.fullmatch(r"mobile_archive_([1-9][0-9]*)", node.value)
    if mobile:
        return LinkId(f"link:s_mob{mobile.group(1)}:s_edge")
    raise ValueError(f"node has no sensor uplink: {node}")


def node_for_sensor_link(value: str | LinkId) -> NodeId:
    link = value if isinstance(value, LinkId) else LinkId(value)
    match = re.fullmatch(r"link:s_(orin|mob)([1-9][0-9]*):s_edge", link.value)
    assert match is not None
    return NodeId(
        f"dvpg_gq_orin_{match.group(2)}"
        if match.group(1) == "orin"
        else f"mobile_archive_{match.group(2)}"
    )
