"""End-to-end ReID execution: crops -> embeddings -> cross-sensor associations."""
from __future__ import annotations

from typing import Sequence

from fable.providers.data_models import ImageCrop
from fable.providers.identity import CrossSensorIdentityAssociationProvider, IdentityAssociation
from fable.providers.visual_features import (
    PersonReIDDescriptorProvider,
    VehicleReIDDescriptorProvider,
)


class ReIDPipeline:
    """Run model-backed person/vehicle ReID over two crop sets."""

    def __init__(
        self,
        *,
        vehicle_descriptor: VehicleReIDDescriptorProvider | None = None,
        person_descriptor: PersonReIDDescriptorProvider | None = None,
        association_provider: CrossSensorIdentityAssociationProvider | None = None,
    ) -> None:
        self.vehicle_descriptor = vehicle_descriptor or VehicleReIDDescriptorProvider()
        self.person_descriptor = person_descriptor or PersonReIDDescriptorProvider()
        self.association_provider = association_provider or CrossSensorIdentityAssociationProvider()

    def associate(
        self,
        left_crops: Sequence[ImageCrop],
        right_crops: Sequence[ImageCrop],
        *,
        entity_kind: str,
    ) -> tuple[IdentityAssociation, ...]:
        if entity_kind == "vehicle":
            descriptor = self.vehicle_descriptor
        elif entity_kind == "person":
            descriptor = self.person_descriptor
        else:
            raise ValueError("entity_kind must be 'vehicle' or 'person'")
        left = descriptor.describe(left_crops)
        right = descriptor.describe(right_crops)
        return self.association_provider.associate_records(left, right)
