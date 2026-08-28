import base64
from datetime import datetime, timezone

import pytest

from fable.common.time import EventTimeInterval
from providers.vehicle.descriptors import DeterministicDescriptorProvider
from providers.vehicle.descriptor_service import CropDescriptorProcessor, SiteDescriptorConfig


class _Descriptor:
    def encode(self, crops, *, source_id, event_time_interval):
        rows = tuple(crops)
        return DeterministicDescriptorProvider(
            calibrated_for_identity=True, entity_kind="vehicle"
        ).encode_ids(
            [item[0] for item in rows],
            source_id=source_id,
            event_time=event_time_interval.end,
        )


def _payload():
    encoded = b"bounded-jpeg-fixture"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {
        "schema_version": "bounded_reid_crop_set.v1",
        "source_id": "orin11_camera",
        "event_time_interval": {
            "start": start.isoformat(),
            "end": start.isoformat(),
        },
        "records": [{
            "local_entity_id": "orin11:track-1",
            "image_data_url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
        }],
    }


def test_site_descriptor_accepts_only_bounded_typed_jpeg_crops(monkeypatch):
    monkeypatch.setattr(
        "providers.vehicle.descriptor_service._decode_jpeg",
        lambda _raw: [[[]]],
    )
    result = CropDescriptorProcessor(_Descriptor()).process(_payload())
    assert result.source_id == "orin11_camera"
    assert result.records[0].local_entity_id == "orin11:track-1"
    assert result.records[0].source_crop_data_urls == (
        _payload()["records"][0]["image_data_url"],
    )


def test_site_descriptor_rejects_raw_or_untyped_input():
    payload = _payload()
    payload["schema_version"] = "raw_video_frames.v1"
    with pytest.raises(ValueError, match="only bounded_reid_crop_set"):
        CropDescriptorProcessor(_Descriptor()).process(payload)


def test_site_descriptor_enforces_crop_count_bound():
    payload = _payload()
    payload["records"] *= 2
    processor = CropDescriptorProcessor(
        _Descriptor(), SiteDescriptorConfig(maximum_crops_per_message=1)
    )
    with pytest.raises(ValueError, match="bounded record count"):
        processor.process(payload)
