"""Build reproducible workload manifests from the experiment ground-truth CSV."""

from __future__ import annotations

import json
from pathlib import Path

from .catalog import ExperimentCatalog


class ManifestBuilder:
    def __init__(self, catalog: ExperimentCatalog) -> None:
        self.catalog = catalog

    def write(self, output_dir: str | Path) -> tuple[Path, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        jsonl_path = root / "ground_truth.jsonl"
        summary_path = root / "catalog_summary.json"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for item in self.catalog.experiments:
                handle.write(item.model_dump_json(exclude_none=True) + "\n")
        summary_path.write_text(
            self.catalog.summary().model_dump_json(indent=2),
            encoding="utf-8",
        )
        return jsonl_path, summary_path
