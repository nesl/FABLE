import pytest

from fable.common.identities import (
    canonical_node_id,
    node_for_sensor_link,
    sensor_link_id,
)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("orin11", "dvpg_gq_orin_11"),
        ("dvpg_gq_orin_11", "dvpg_gq_orin_11"),
        ("mobile6", "mobile_archive_6"),
        ("mob6", "mobile_archive_6"),
    ],
)
def test_node_aliases_normalize_at_one_boundary(alias: str, canonical: str) -> None:
    assert canonical_node_id(alias).value == canonical


def test_node_link_round_trip() -> None:
    link = sensor_link_id("mobile_archive_6")
    assert link.value == "link:s_mob6:s_edge"
    assert node_for_sensor_link(link).value == "mobile_archive_6"


def test_unknown_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown node alias"):
        canonical_node_id("orin-eleven")
