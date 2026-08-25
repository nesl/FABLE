"""Durable state and append-only control-event persistence.

The orchestrator remains authoritative.  MongoDB stores compact typed records
and supports compare-and-set updates; large media/artifact payloads remain in
node-local stores and are represented by ``ArtifactRef`` documents.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
import threading
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from fable.common.base import to_jsonable
from fable.common.time import ensure_utc, utc_now

from .models import ControlEvent

T = TypeVar("T", bound=BaseModel)


class StateConflictError(RuntimeError):
    """Raised when a compare-and-set version no longer matches."""


class StateStore(Protocol):
    def put(
        self,
        collection: str,
        key: str,
        value: BaseModel | dict[str, Any],
        *,
        expected_version: int | None = None,
    ) -> None: ...

    def get(self, collection: str, key: str, model_type: type[T]) -> T | None: ...

    def list(self, collection: str, model_type: type[T]) -> tuple[T, ...]: ...

    def delete(self, collection: str, key: str) -> bool: ...

    def contains(self, collection: str, key: str) -> bool: ...

    def append_event(self, event: ControlEvent) -> None: ...

    def list_events(self, *, after: datetime | None = None) -> tuple[ControlEvent, ...]: ...


class InMemoryStateStore:
    def __init__(self) -> None:
        self._collections: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._events: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def put(
        self,
        collection: str,
        key: str,
        value: BaseModel | dict[str, Any],
        *,
        expected_version: int | None = None,
    ) -> None:
        payload = _payload(value)
        with self._lock:
            existing = self._collections[collection].get(key)
            if expected_version is not None:
                if existing is None:
                    if expected_version != -1:
                        raise StateConflictError(
                            f"{collection}/{key} does not exist for expected version {expected_version}"
                        )
                elif int(existing.get("version", -1)) != expected_version:
                    raise StateConflictError(
                        f"{collection}/{key} expected version {expected_version}; "
                        f"found {existing.get('version')}"
                    )
            self._collections[collection][key] = deepcopy(payload)

    def get(self, collection: str, key: str, model_type: type[T]) -> T | None:
        with self._lock:
            payload = self._collections[collection].get(key)
            return None if payload is None else model_type.model_validate(deepcopy(payload))

    def get_raw(self, collection: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._collections[collection].get(key)
            return None if payload is None else deepcopy(payload)

    def list(self, collection: str, model_type: type[T]) -> tuple[T, ...]:
        with self._lock:
            rows = [deepcopy(item) for _, item in sorted(self._collections[collection].items())]
        return tuple(model_type.model_validate(item) for item in rows)

    def list_raw(self, collection: str) -> tuple[dict[str, Any], ...]:
        with self._lock:
            rows = [deepcopy(item) for _, item in sorted(self._collections[collection].items())]
        return tuple(rows)

    def delete(self, collection: str, key: str) -> bool:
        with self._lock:
            return self._collections[collection].pop(key, None) is not None

    def contains(self, collection: str, key: str) -> bool:
        with self._lock:
            return key in self._collections[collection]

    def append_event(self, event: ControlEvent) -> None:
        key = str(event.event_id)
        with self._lock:
            self._events.setdefault(key, _payload(event))

    def list_events(self, *, after: datetime | None = None) -> tuple[ControlEvent, ...]:
        after_value = None if after is None else ensure_utc(after)
        with self._lock:
            events = [ControlEvent.model_validate(deepcopy(item)) for item in self._events.values()]
        events.sort(key=lambda item: (item.created_at, str(item.event_id)))
        if after_value is not None:
            events = [item for item in events if item.created_at > after_value]
        return tuple(events)


class MongoStateStore:
    """MongoDB-backed compact record store.

    Records are serialized in JSON mode so UUIDs and timezone-aware datetimes
    remain portable across PyMongo versions.  Every collection uses a string
    ``_id`` and carries ``updated_at``.  Versioned records may be updated with a
    compare-and-set filter on their ``version`` field.
    """

    DEFAULT_COLLECTIONS = (
        "tasks",
        "graphs",
        "hypotheses",
        "frontiers",
        "demands",
        "plans",
        "plan_execution",
        "provider_instances",
        "leases",
        "results",
        "artifacts",
        "nodes",
        "processed_messages",
        "emitted_events",
        "control_events",
        "event_request_responses",
        "runtime_disturbance_acks",
        "terminal_events",
    )

    def __init__(
        self,
        uri: str,
        *,
        database: str = "fable",
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from pymongo import MongoClient
            except ImportError as exc:  # pragma: no cover - optional runtime dependency
                raise RuntimeError("pymongo is required for MongoStateStore") from exc
            client = MongoClient(uri, tz_aware=True, connect=False)
        self.client = client
        self.db = client[database]
        self.ensure_indexes()

    def ensure_indexes(self) -> None:
        try:
            self.db["hypotheses"].create_index([("request_id", 1), ("lifecycle", 1)])
            self.db["frontiers"].create_index([("hypothesis_id", 1), ("created_at", -1)])
            self.db["demands"].create_index([("hypothesis_id", 1), ("checkpoint_id", 1)])
            self.db["plans"].create_index([("status", 1), ("created_at", -1)])
            self.db["provider_instances"].create_index([("share_key.node_id", 1), ("lifecycle", 1)])
            self.db["leases"].create_index([("lease.node_id", 1), ("lease.status", 1)])
            self.db["results"].create_index([("result.request_id", 1), ("result.hypothesis_id", 1)])
            self.db["artifacts"].create_index([("location.node_id", 1), ("expires_at", 1)])
            self.db["nodes"].create_index([("node_id", 1), ("sent_at", -1)])
            self.db["control_events"].create_index([("created_at", 1), ("event_id", 1)])
            self.db["control_events"].create_index([("entity_type", 1), ("entity_id", 1)])
            self.db["terminal_events"].create_index([("request_id", 1), ("emitted_at", 1)])
            self.db["runtime_disturbance_acks"].create_index([("disturbance_id", 1), ("resource_epoch", 1)])
        except Exception:
            # Index creation may be intentionally deferred while Mongo is down;
            # ordinary operations will surface connection errors later.
            pass

    def put(
        self,
        collection: str,
        key: str,
        value: BaseModel | dict[str, Any],
        *,
        expected_version: int | None = None,
    ) -> None:
        payload = _payload(value)
        document = {"_id": str(key), **payload, "_stored_at": utc_now().isoformat()}
        coll = self.db[collection]
        if expected_version is None:
            coll.replace_one({"_id": str(key)}, document, upsert=True)
            return
        if expected_version == -1:
            try:
                coll.insert_one(document)
            except Exception as exc:
                raise StateConflictError(f"{collection}/{key} already exists") from exc
            return
        result = coll.replace_one(
            {"_id": str(key), "version": expected_version},
            document,
            upsert=False,
        )
        if result.matched_count != 1:
            raise StateConflictError(
                f"{collection}/{key} compare-and-set failed for version {expected_version}"
            )

    def get(self, collection: str, key: str, model_type: type[T]) -> T | None:
        document = self.db[collection].find_one({"_id": str(key)})
        if document is None:
            return None
        return model_type.model_validate(_strip_mongo(document))

    def get_raw(self, collection: str, key: str) -> dict[str, Any] | None:
        document = self.db[collection].find_one({"_id": str(key)})
        return None if document is None else _strip_mongo(document)

    def list(self, collection: str, model_type: type[T]) -> tuple[T, ...]:
        documents = self.db[collection].find({}).sort("_id", 1)
        return tuple(model_type.model_validate(_strip_mongo(item)) for item in documents)

    def list_raw(self, collection: str) -> tuple[dict[str, Any], ...]:
        documents = self.db[collection].find({}).sort("_id", 1)
        return tuple(_strip_mongo(item) for item in documents)

    def delete(self, collection: str, key: str) -> bool:
        return self.db[collection].delete_one({"_id": str(key)}).deleted_count == 1

    def contains(self, collection: str, key: str) -> bool:
        return self.db[collection].count_documents({"_id": str(key)}, limit=1) > 0

    def append_event(self, event: ControlEvent) -> None:
        document = {"_id": str(event.event_id), **_payload(event)}
        try:
            self.db["control_events"].insert_one(document)
        except Exception as exc:
            # Duplicate control-event IDs are idempotent.  Re-raise other failures.
            existing = self.db["control_events"].find_one({"_id": str(event.event_id)})
            if existing is None:
                raise exc

    def list_events(self, *, after: datetime | None = None) -> tuple[ControlEvent, ...]:
        query: dict[str, Any] = {}
        if after is not None:
            query["created_at"] = {"$gt": ensure_utc(after).isoformat()}
        documents = self.db["control_events"].find(query).sort([("created_at", 1), ("_id", 1)])
        return tuple(ControlEvent.model_validate(_strip_mongo(item)) for item in documents)

    def close(self) -> None:
        self.client.close()


def _payload(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        data = value.model_dump(mode="json", exclude_none=False)
    else:
        data = to_jsonable(value)
    if not isinstance(data, dict):
        raise TypeError("persisted record must serialize to a mapping")
    return data


def _strip_mongo(document: dict[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("_id", None)
    result.pop("_stored_at", None)
    return result
