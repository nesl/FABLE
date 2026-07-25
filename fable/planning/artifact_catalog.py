"""In-memory typed artifact catalog used by Phase-2/3 planning tests.

The persistence backend can later be MongoDB.  The compatibility rules live in
this module so storage choice does not affect physical-plan validity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from uuid import UUID

from fable.common.enums import ArtifactAccessMode
from fable.common.schemas import ArtifactRef
from fable.common.time import EventTimeInterval, ensure_utc, utc_now

from .models import ArtifactQueryRejection, ArtifactQueryResult


class ArtifactCatalogError(ValueError):
    """Raised for duplicate artifacts or impossible compatibility requests."""


class ArtifactCatalog:
    def __init__(self, artifacts: Iterable[ArtifactRef] = ()) -> None:
        self._artifacts: dict[UUID, ArtifactRef] = {}
        for artifact in artifacts:
            self.register(artifact)

    def register(self, artifact: ArtifactRef, *, replace: bool = False) -> None:
        if artifact.artifact_id in self._artifacts and not replace:
            raise ArtifactCatalogError(f"artifact already registered: {artifact.artifact_id}")
        self._artifacts[artifact.artifact_id] = artifact

    def remove(self, artifact_id: UUID) -> None:
        self._artifacts.pop(artifact_id, None)

    def get(self, artifact_id: UUID) -> ArtifactRef:
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise ArtifactCatalogError(f"unknown artifact: {artifact_id}") from exc


    def extend_retention(
        self,
        artifact_id: UUID,
        *,
        required_until: datetime,
    ) -> ArtifactRef:
        """Extend an artifact TTL without shortening an existing retention horizon."""

        required_until = ensure_utc(required_until)
        artifact = self.get(artifact_id)
        expires_at = artifact.expires_at
        if expires_at is not None and expires_at >= required_until:
            return artifact
        updated = artifact.model_copy(update={"expires_at": required_until})
        self.register(updated, replace=True)
        return updated

    @property
    def artifacts(self) -> tuple[ArtifactRef, ...]:
        return tuple(sorted(self._artifacts.values(), key=lambda item: str(item.artifact_id)))

    def query(
        self,
        *,
        artifact_type: str,
        event_time_interval: EventTimeInterval | None = None,
        required_bindings: Mapping[str, str] | None = None,
        required_compatibility_keys: Mapping[str, object] | None = None,
        consumer_family: str | None = None,
        required_access_modes: Iterable[ArtifactAccessMode] = (),
        location_node_id: str | None = None,
        now: datetime | None = None,
        require_interval_containment: bool = True,
    ) -> tuple[ArtifactRef, ...]:
        result = self.query_with_rejections(
            artifact_type=artifact_type,
            event_time_interval=event_time_interval,
            required_bindings=required_bindings,
            required_compatibility_keys=required_compatibility_keys,
            consumer_family=consumer_family,
            required_access_modes=required_access_modes,
            location_node_id=location_node_id,
            now=now,
            require_interval_containment=require_interval_containment,
        )
        return tuple(self._artifacts[artifact_id] for artifact_id in result.matches)

    def query_with_rejections(
        self,
        *,
        artifact_type: str,
        event_time_interval: EventTimeInterval | None = None,
        required_bindings: Mapping[str, str] | None = None,
        required_compatibility_keys: Mapping[str, object] | None = None,
        consumer_family: str | None = None,
        required_access_modes: Iterable[ArtifactAccessMode] = (),
        location_node_id: str | None = None,
        now: datetime | None = None,
        require_interval_containment: bool = True,
    ) -> ArtifactQueryResult:
        observed_now = ensure_utc(now or utc_now())
        required_bindings = dict(required_bindings or {})
        required_compatibility_keys = dict(required_compatibility_keys or {})
        required_access = set(required_access_modes)
        matches: list[UUID] = []
        rejections: list[ArtifactQueryRejection] = []

        for artifact in self.artifacts:
            code, reason = self._incompatibility(
                artifact,
                artifact_type=artifact_type,
                event_time_interval=event_time_interval,
                required_bindings=required_bindings,
                required_compatibility_keys=required_compatibility_keys,
                consumer_family=consumer_family,
                required_access=required_access,
                location_node_id=location_node_id,
                now=observed_now,
                require_interval_containment=require_interval_containment,
            )
            if code is None:
                matches.append(artifact.artifact_id)
            else:
                rejections.append(
                    ArtifactQueryRejection(
                        artifact_id=artifact.artifact_id,
                        code=code,
                        reason=reason,
                    )
                )
        return ArtifactQueryResult(matches=tuple(matches), rejections=tuple(rejections))

    @staticmethod
    def _incompatibility(
        artifact: ArtifactRef,
        *,
        artifact_type: str,
        event_time_interval: EventTimeInterval | None,
        required_bindings: Mapping[str, str],
        required_compatibility_keys: Mapping[str, object],
        consumer_family: str | None,
        required_access: set[ArtifactAccessMode],
        location_node_id: str | None,
        now: datetime,
        require_interval_containment: bool,
    ) -> tuple[str | None, str]:
        if artifact.artifact_type != artifact_type:
            return "TYPE_MISMATCH", f"expected {artifact_type}, got {artifact.artifact_type}"
        if artifact.artifact_schema_version != artifact_type:
            return (
                "SCHEMA_VERSION_MISMATCH",
                f"artifact schema {artifact.artifact_schema_version} is not {artifact_type}",
            )
        if artifact.expires_at is not None and artifact.expires_at <= now:
            return "EXPIRED", "artifact retention has expired"
        if artifact.valid_until is not None and artifact.valid_until < now:
            return "INVALID", "artifact validity interval has ended"
        if event_time_interval is not None:
            if artifact.valid_until is not None and artifact.valid_until < event_time_interval.end:
                return "EVENT_TIME_INVALID", "artifact becomes invalid before required event interval ends"
            if require_interval_containment:
                compatible_interval = artifact.event_time_interval.contains_interval(event_time_interval)
            else:
                compatible_interval = artifact.event_time_interval.overlaps(event_time_interval)
            if not compatible_interval:
                return "EVENT_TIME_MISMATCH", "artifact does not cover the requested event interval"
        for key, expected in required_bindings.items():
            if artifact.bindings.get(key) != expected:
                return "BINDING_MISMATCH", f"artifact binding {key} does not equal {expected}"
        for key, expected in required_compatibility_keys.items():
            if artifact.compatibility_keys.get(key) != expected:
                return "COMPATIBILITY_KEY_MISMATCH", f"compatibility key {key} does not equal {expected!r}"
        if consumer_family and artifact.compatible_consumer_families:
            if consumer_family not in artifact.compatible_consumer_families:
                return "CONSUMER_INCOMPATIBLE", f"artifact does not permit consumer family {consumer_family}"
        if required_access and not required_access.issubset(set(artifact.access_modes)):
            return "ACCESS_MODE_MISMATCH", "artifact does not expose all required access modes"
        if location_node_id is not None and artifact.location.node_id != location_node_id:
            return "LOCATION_MISMATCH", f"artifact is not located on node {location_node_id}"
        return None, ""
