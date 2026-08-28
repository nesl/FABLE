"""Load immutable deployment geometry into the typed planning artifact catalog."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml

from fable.common.enums import ArtifactAccessMode, ArtifactLocationKind
from fable.common.schemas import ArtifactLocation, ArtifactProducer, ArtifactRef
from fable.common.time import EventTimeInterval
from fable.planning.artifact_catalog import ArtifactCatalog


def load_deployment_artifacts(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> ArtifactCatalog:
    """Load and verify root-owned/static geometry declared for live planning."""

    manifest_path = Path(path).resolve()
    root = Path(repository_root).resolve()
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    artifacts = []
    for raw in document.get("artifacts", ()):
        source_path = (root / raw["path"]).resolve()
        if not source_path.is_relative_to(root):
            raise ValueError(f"deployment artifact escapes repository root: {source_path}")
        if not source_path.is_file():
            raise FileNotFoundError(f"deployment artifact does not exist: {source_path}")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        declared = raw.get("sha256")
        if declared and declared != digest:
            raise ValueError(f"deployment artifact checksum mismatch: {source_path}")
        artifact_type = str(raw["artifact_type"])
        artifacts.append(
            ArtifactRef(
                # ArtifactCatalog treats the versioned type as the contract
                # identity (matching normal provider-produced ArtifactRefs).
                # Stripping ``.v1`` here makes every deployment artifact
                # unqueryable even though its schema version is correct.
                artifact_type=artifact_type,
                artifact_schema_version=artifact_type,
                producer=ArtifactProducer(
                    provider_id="deployment_configuration",
                    provider_contract_version=1,
                ),
                event_time_interval=EventTimeInterval(
                    start=datetime(2000, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2100, 1, 1, tzinfo=timezone.utc),
                ),
                bindings={str(k): str(v) for k, v in raw.get("bindings", {}).items()},
                location=ArtifactLocation(
                    kind=ArtifactLocationKind.LOCAL_PATH,
                    node_id=str(raw["node_id"]),
                    uri=str(raw.get("runtime_uri") or source_path.as_uri()),
                ),
                access_modes=tuple(
                    ArtifactAccessMode(item)
                    for item in raw.get("access_modes", ("LOCAL",))
                ),
                compatibility_keys=raw.get("compatibility_keys", {}),
                bytes=source_path.stat().st_size,
                checksum_sha256=digest,
                created_at=datetime.fromtimestamp(
                    source_path.stat().st_mtime, tz=timezone.utc
                ),
                valid_until=datetime(2100, 1, 1, tzinfo=timezone.utc),
                policy_tags=("deployment-static",),
            )
        )
    return ArtifactCatalog(artifacts)
