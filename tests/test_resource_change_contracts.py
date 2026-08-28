from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluation.networking import (
    canonical_sensor_link_target,
    validated_link_target_node_id,
)
from fable.common.ids import uuid7
from fable.distributed.models import ResourceChange, ResourceChangeAck
from fable.distributed.topics import (
    resource_change_ack_filter,
    resource_change_ack_topic,
)


def test_physical_node_round_trips_through_canonical_link_identity() -> None:
    target = canonical_sensor_link_target("dvpg_gq_orin_1")
    assert target == "link:s_orin1:s_edge"
    assert validated_link_target_node_id(target) == "dvpg_gq_orin_1"


@pytest.mark.parametrize("target", ["dvpg_gq_orin_1", "physical_link:rpi_to_jetson"])
def test_link_state_rejects_noncanonical_target(target: str) -> None:
    with pytest.raises(ValidationError, match="canonical sensor link"):
        ResourceChange(
            run_id="run-1",
            condition="L1",
            action="FAIL",
            condition_epoch=1,
            target_id=target,
            resource_kind="LINK_STATE",
        )


def test_ack_is_correlated_by_run_and_message_not_epoch() -> None:
    first = ResourceChange(
        run_id="run-1",
        condition="E1",
        action="APPLY",
        condition_epoch=3,
        target_id="x86server",
        resource_kind="COMPUTE",
    )
    second = first.model_copy(update={"message_id": uuid7()})
    first_ack = ResourceChangeAck(
        request_message_id=first.message_id,
        run_id=first.run_id,
        condition_epoch=first.condition_epoch,
        accepted=True,
        adaptation_status="APPLIED",
    )
    second_ack = ResourceChangeAck(
        request_message_id=second.message_id,
        run_id=second.run_id,
        condition_epoch=second.condition_epoch,
        accepted=True,
        adaptation_status="UNCHANGED",
    )
    assert first_ack.request_message_id != second_ack.request_message_id
    assert resource_change_ack_topic(first.run_id, str(first.message_id)) != resource_change_ack_topic(
        second.run_id, str(second.message_id)
    )
    assert resource_change_ack_filter("run-1").endswith("/run-1/+")
