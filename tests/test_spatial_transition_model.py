from __future__ import annotations

import unittest
from pathlib import Path

from fable.spatial import (
    SiteSensorTransitionModel,
    SpatialMatchKind,
    SpatialObservation,
    SpatialSensorBindings,
    heading_from_vector,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "evaluation" / "labels" / "site_sensor_transition_model_2024_2025.json"


class SpatialTransitionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = SiteSensorTransitionModel.from_json(MODEL_PATH)

    def test_directional_rule_preserves_overlapping_next_sensor_group(self) -> None:
        bindings = SpatialSensorBindings(
            source_ids_by_sensor={
                "orin_5": ("camera_5",),
            },
            node_ids_by_sensor={
                "orin_5": ("node_5",),
            },
            source_ids_by_deployment={
                "2025_package_exchange": {"d3": ("package_d3_camera",)},
            },
            node_ids_by_deployment={
                "2025_package_exchange": {"d3": ("package_d3",)},
            },
        )
        prediction = self.model.predict(
            SpatialObservation(
                current_sensor_id="orin_6",
                observed_heading="southwest",
                active_deployment_id="2025_package_exchange",
            ),
            bindings=bindings,
        )
        self.assertEqual(prediction.match_kind, SpatialMatchKind.DIRECTIONAL_RULE)
        self.assertEqual(prediction.corridor_id, "northeast_to_central_east_arc")
        self.assertEqual(prediction.groups[0].sensor_ids, ("orin_5", "d3"))
        self.assertEqual(
            prediction.groups[0].source_ids,
            ("camera_5", "package_d3_camera"),
        )
        self.assertEqual(prediction.wake_node_ids, ("node_5", "package_d3"))

    def test_corridor_fallback_returns_next_observation_groups(self) -> None:
        bindings = SpatialSensorBindings(
            source_ids_by_sensor={
                "orin_8": ("camera_8",),
                "orin_10": ("camera_10",),
                "orin_9": ("camera_9",),
            },
            node_ids_by_sensor={
                "orin_8": ("node_8",),
                "orin_10": ("node_10",),
                "orin_9": ("node_9",),
            },
        )
        prediction = self.model.predict(
            SpatialObservation(
                current_sensor_id="orin_1",
                corridor_id="central_to_west_lower_corridor",
                branch_unresolved=True,
                maximum_observation_groups=1,
            ),
            bindings=bindings,
        )
        self.assertEqual(prediction.match_kind, SpatialMatchKind.CORRIDOR)
        self.assertEqual(prediction.groups[0].sensor_ids, ("orin_8", "orin_10"))
        self.assertEqual(prediction.groups[1].sensor_ids, ("orin_9",))

    def test_mobile_aliases_are_deployment_scoped(self) -> None:
        bindings = SpatialSensorBindings(
            source_ids_by_deployment={
                "2025_package_exchange": {"d3": ("package_d3_camera",)},
                "2025_flee_police": {"d3": ("flee_d3_camera",)},
            }
        )
        self.assertEqual(
            bindings.sources("d3", "2025_package_exchange"),
            ("package_d3_camera",),
        )
        self.assertEqual(
            bindings.sources("d3", "2025_flee_police"),
            ("flee_d3_camera",),
        )
        self.assertIsNone(bindings.sensor_for_source("package_d3_camera", "2025_flee_police"))

    def test_heading_quantization_uses_model_coordinate_frame(self) -> None:
        self.assertEqual(heading_from_vector(1, 0), "E")
        self.assertEqual(heading_from_vector(-1, 1), "NW")
        self.assertEqual(heading_from_vector(0, -2), "S")


if __name__ == "__main__":
    unittest.main()
