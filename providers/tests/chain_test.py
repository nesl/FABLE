#!/usr/bin/env python3
"""Tests and examples for the FABLE chain validator.

Run directly:
    python providers/tests/chain_test.py

Or with unittest discovery:
    python -m unittest discover -s providers/tests -p "*_test.py" -v
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from providers.tools.chain_validator import ChainValidator, load_yaml  # noqa: E402


DATA_TYPES_PATH = PROJECT_ROOT / "providers" / "registry" / "data_types.yaml"
CATALOG_PATH = PROJECT_ROOT / "providers" / "registry" / "catalog.yaml"


class ChainValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_types_doc = load_yaml(DATA_TYPES_PATH)
        cls.catalog_doc = load_yaml(CATALOG_PATH)

    def make_validator(self) -> ChainValidator:
        return ChainValidator(self.data_types_doc, self.catalog_doc)

    def assert_has_code(self, report, code: str) -> None:
        codes = [issue.code for issue in report.issues]
        self.assertIn(code, codes, msg=f"Expected {code}; got {codes}")

    def test_catalog_and_all_named_chains_are_valid(self) -> None:
        report = self.make_validator().validate_all()
        self.assertTrue(report.ok, msg=report.format())

    def test_valid_local_follows_chain(self) -> None:
        report = self.make_validator().validate_chain("follows_local_tracks")
        self.assertTrue(report.ok, msg=report.format())

    def test_valid_cross_camera_reid_chain(self) -> None:
        report = self.make_validator().validate_chain("follows_cross_camera_reid")
        self.assertTrue(report.ok, msg=report.format())

    def test_valid_continuation_state_chain(self) -> None:
        report = self.make_validator().validate_chain(
            "follows_local_from_retained_detections"
        )
        self.assertTrue(report.ok, msg=report.format())

    def test_invalid_type_mismatch(self) -> None:
        chain = copy.deepcopy(
            self.catalog_doc["chains"]["follows_local_tracks"]
        )
        # camera_projection.tracks requires track_set.v1, but detector output is
        # detection_set.v1.
        project = next(step for step in chain["steps"] if step["id"] == "project")
        project["bind"]["tracks"] = "detect.detections"

        report = self.make_validator().validate_chain(
            "invalid_type_mismatch",
            chain_override=chain,
        )
        self.assertFalse(report.ok)
        self.assert_has_code(report, "TYPE_MISMATCH")

    def test_invalid_missing_required_input(self) -> None:
        chain = copy.deepcopy(
            self.catalog_doc["chains"]["follows_local_tracks"]
        )
        project = next(step for step in chain["steps"] if step["id"] == "project")
        del project["bind"]["calibration"]

        report = self.make_validator().validate_chain(
            "invalid_missing_calibration",
            chain_override=chain,
        )
        self.assertFalse(report.ok)
        self.assert_has_code(report, "MISSING_REQUIRED_INPUT")

    def test_invalid_forward_step_reference(self) -> None:
        chain = copy.deepcopy(
            self.catalog_doc["chains"]["follows_local_tracks"]
        )
        # The first step cannot consume an output from a later step.
        chain["steps"][0]["bind"]["frames"] = "track.tracks"

        report = self.make_validator().validate_chain(
            "invalid_forward_reference",
            chain_override=chain,
        )
        self.assertFalse(report.ok)
        self.assert_has_code(report, "FORWARD_STEP_REFERENCE")

    def test_invalid_unknown_provider(self) -> None:
        chain = copy.deepcopy(
            self.catalog_doc["chains"]["detect_audio_event"]
        )
        chain["steps"][0]["provider"] = "not_registered"

        report = self.make_validator().validate_chain(
            "invalid_unknown_provider",
            chain_override=chain,
        )
        self.assertFalse(report.ok)
        self.assert_has_code(report, "UNKNOWN_PROVIDER")

    def test_invalid_continuation_type(self) -> None:
        chain = copy.deepcopy(
            self.catalog_doc["chains"]["follows_local_from_retained_detections"]
        )
        # detections requires detection_set.v1, while calibration is
        # camera_calibration.v1.
        track = next(step for step in chain["steps"] if step["id"] == "track")
        track["bind"]["detections"] = "external.calibration"

        report = self.make_validator().validate_chain(
            "invalid_continuation_type",
            chain_override=chain,
        )
        self.assertFalse(report.ok)
        self.assert_has_code(report, "TYPE_MISMATCH")


if __name__ == "__main__":
    unittest.main(verbosity=2)
