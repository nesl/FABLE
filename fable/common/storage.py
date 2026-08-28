"""Repository-wide generated-data path configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from fable.common.base import FrozenFableModel


class StorageConfig(FrozenFableModel):
    schema_version: str = "fable.storage.v1"
    storage_root: Path
    require_external_mount: bool = True
    expected_mount_uuid: str | None = None
    paths: dict[str, Path] = Field(default_factory=dict)
    repository_links: dict[str, Path] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_standard_paths(self) -> "StorageConfig":
        missing = {"results", "runs", "debug"} - set(self.paths)
        if missing:
            raise ValueError(
                "storage config is missing standard paths: "
                + ", ".join(sorted(missing))
            )
        for name, value in (*self.paths.items(), *self.repository_links.items()):
            if value.is_absolute() or ".." in value.parts:
                raise ValueError(f"storage path {name} must be relative and contained")
        return self

    def path(self, name: str) -> Path:
        try:
            relative = self.paths[name]
        except KeyError as exc:
            raise KeyError(f"unknown configured storage path: {name}") from exc
        return self.storage_root / relative

    def link_targets(self, repository_root: Path) -> dict[Path, Path]:
        return {
            repository_root / local: self.storage_root / remote
            for local, remote in self.repository_links.items()
        }


def load_storage_config(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> StorageConfig:
    root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    source = Path(path) if path is not None else root / "config/storage.yaml"
    document: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    override = os.environ.get("FABLE_STORAGE_ROOT")
    if override:
        document["storage_root"] = override
    return StorageConfig.model_validate(document)
