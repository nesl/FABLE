from __future__ import annotations

import unittest
from datetime import timedelta

from fable.common.enums import ArtifactAccessMode, ArtifactLocationKind
from fable.common.examples import BASE_TIME
from fable.common.schemas import ArtifactLocation, ArtifactProducer, ArtifactRef
from fable.common.time import EventTimeInterval
from fable.planning import ArtifactCatalog, ProviderRegistryError
from fable.planning.testing import fake_artifact_catalog, fake_deployment, fake_provider_registry


class Phase2ArtifactAndDeploymentTests(unittest.TestCase):
    def test_artifact_catalog_rejects_schema_version_mismatch(self) -> None:
        invalid = ArtifactRef(
            artifact_type="vehicle_reid_embedding_set.v1",
            artifact_schema_version="vehicle_reid_embedding_set.v0",
            producer=ArtifactProducer(
                provider_id="vehicle_reid_descriptor",
                provider_contract_version=1,
            ),
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(seconds=10),
            ),
            bindings={"leader": "vehicle_17"},
            location=ArtifactLocation(
                kind=ArtifactLocationKind.LOCAL_PATH,
                node_id="edge_1",
                uri="file:///tmp/invalid.npy",
            ),
            access_modes=(ArtifactAccessMode.LOCAL,),
            created_at=BASE_TIME,
            expires_at=BASE_TIME + timedelta(hours=1),
        )
        catalog = ArtifactCatalog((invalid,))
        result = catalog.query_with_rejections(
            artifact_type="vehicle_reid_embedding_set.v1",
            event_time_interval=invalid.event_time_interval,
            now=BASE_TIME,
        )
        self.assertEqual(result.matches, ())
        self.assertEqual(result.rejections[0].code, "SCHEMA_VERSION_MISMATCH")

    def test_artifact_catalog_matches_source_specific_calibration(self) -> None:
        catalog = fake_artifact_catalog()
        matches = catalog.query(
            artifact_type="camera_calibration.v1",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(seconds=15),
            ),
            required_bindings={"source_id": "camera_mobile"},
            now=BASE_TIME,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].location.node_id, "sensor_a")


    def test_provider_rejects_mismatched_feature_versions(self) -> None:
        def embedding(port_suffix: str, model_version: str) -> ArtifactRef:
            return ArtifactRef(
                artifact_type="vehicle_reid_embedding_set.v1",
                artifact_schema_version="vehicle_reid_embedding_set.v1",
                producer=ArtifactProducer(
                    provider_id="vehicle_reid_descriptor",
                    provider_contract_version=1,
                    model_id="vehicle_reid",
                    model_version=model_version,
                ),
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(seconds=10),
                ),
                bindings={"source_id": f"camera_{port_suffix}"},
                location=ArtifactLocation(
                    kind=ArtifactLocationKind.LOCAL_PATH,
                    node_id="edge_1",
                    uri=f"file:///tmp/{port_suffix}.npy",
                ),
                access_modes=(ArtifactAccessMode.LOCAL,),
                compatibility_keys={
                    "model_id": "vehicle_reid",
                    "model_version": model_version,
                    "preprocessing_id": "default",
                    "dimension": 512,
                    "normalization": "l2",
                    "distance_metric": "cosine",
                },
                created_at=BASE_TIME,
                expires_at=BASE_TIME + timedelta(hours=1),
            )

        registry = fake_provider_registry()
        with self.assertRaises(ProviderRegistryError):
            registry.validate_runtime_input_compatibility(
                "cross_sensor_identity_association",
                {
                    "left_embeddings": embedding("left", "1"),
                    "right_embeddings": embedding("right", "2"),
                },
            )

    def test_deployment_graph_finds_multi_hop_path(self) -> None:
        deployment = fake_deployment()
        path = deployment.shortest_path("sensor_a", "server_1")
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.node_ids, ("sensor_a", "edge_1", "server_1"))
        self.assertEqual(path.latency_ms, 28)
        self.assertGreater(deployment.estimate_transfer_ms(path, 1_000_000), 28)


if __name__ == "__main__":
    unittest.main()
