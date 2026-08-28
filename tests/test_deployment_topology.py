from __future__ import annotations

from evaluation.deployment_topology import (
    build_network_profile,
    build_site_local_deployment,
    validate_unique_network_identities,
)
from netwaggle.netwaggle.topology import NetWaggleTopology


def test_canonical_deployment_scales_from_five_to_twenty_devices() -> None:
    for count in (5, 12, 20):
        deployment = build_site_local_deployment(count)
        validate_unique_network_identities(deployment)
        assert deployment["device_count"] == count
        assert len(deployment["logical_nodes"]) == count + 2
        assert len(deployment["links"]) == count + 2
        parsed = NetWaggleTopology.from_dict(deployment)
        assert len(parsed.attachments) == count + 2


def test_profiles_cover_exactly_the_generated_topology() -> None:
    deployment = build_site_local_deployment(20)
    expected = {
        frozenset((link["from"], link["to"])) for link in deployment["links"]
    }
    for condition in ("N0", "W1", "W2", "L1"):
        profile = build_network_profile(deployment, condition)
        actual = {
            frozenset((link["from"], link["to"])) for link in profile["links"]
        }
        assert actual == expected
    w1 = build_network_profile(deployment, "W1")
    wan = next(
        link
        for link in w1["links"]
        if {link["from"], link["to"]} == {"s_site", "s_cloud"}
    )
    assert wan["bw"] == 10
    assert wan["max_queue_size"] == 100
    l1 = build_network_profile(deployment, "L1")
    assert l1["links"][0]["loss"] == 5
    assert all(link["loss"] != 5 for link in l1["links"][1:])


def test_device_network_identities_do_not_collide() -> None:
    deployment = build_site_local_deployment(20)
    devices = [
        node for node in deployment["logical_nodes"] if node["tier"] == "embedded"
    ]
    assert len({node["ip"] for node in devices}) == 20
    assert len({node["anchor_container"] for node in devices}) == 20
