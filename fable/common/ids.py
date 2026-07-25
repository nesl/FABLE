"""UUIDv7 and deterministic identifier helpers."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import uuid
from typing import Any, Iterable

from .base import to_jsonable


_UUID7_LOCK = threading.Lock()
_UUID7_LAST_MS = -1
_UUID7_LAST_RANDOM = 0
_UUID7_RANDOM_MASK = (1 << 74) - 1


def uuid7(now_ms: int | None = None) -> uuid.UUID:
    """Return an RFC 9562 UUIDv7 without requiring a third-party package.

    Values generated in the same millisecond are monotonic within this process.
    The optional ``now_ms`` argument exists only for deterministic tests.
    """

    global _UUID7_LAST_MS, _UUID7_LAST_RANDOM

    observed_ms = int(time.time_ns() // 1_000_000) if now_ms is None else int(now_ms)
    if observed_ms < 0 or observed_ms >= (1 << 48):
        raise ValueError("UUIDv7 timestamp must fit in 48 unsigned bits")

    with _UUID7_LOCK:
        effective_ms = max(observed_ms, _UUID7_LAST_MS)
        if effective_ms == _UUID7_LAST_MS:
            random_74 = (_UUID7_LAST_RANDOM + 1) & _UUID7_RANDOM_MASK
            if random_74 == 0:
                effective_ms += 1
                random_74 = secrets.randbits(74)
        else:
            random_74 = secrets.randbits(74)
        _UUID7_LAST_MS = effective_ms
        _UUID7_LAST_RANDOM = random_74

    rand_a = (random_74 >> 62) & 0xFFF
    rand_b = random_74 & ((1 << 62) - 1)
    value = (
        (effective_ms << 80)
        | (0x7 << 76)
        | (rand_a << 64)
        | (0b10 << 62)
        | rand_b
    )
    return uuid.UUID(int=value)


def uuid7_str(now_ms: int | None = None) -> str:
    return str(uuid7(now_ms=now_ms))


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value to canonical UTF-8 JSON suitable for hashing."""

    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def deterministic_id(prefix: str, value: Any, *, length: int = 24) -> str:
    """Create a human-readable deterministic ID from canonical content."""

    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("prefix must contain only alphanumerics and underscores")
    if length < 12 or length > 64:
        raise ValueError("length must be between 12 and 64")
    return f"{prefix}_{sha256_hex(value)[:length]}"


def graph_node_id(graph_namespace: str, authored_key: str, node_payload: Any) -> str:
    return deterministic_id(
        "gn",
        {"graph_namespace": graph_namespace, "authored_key": authored_key, "node": node_payload},
    )


def temporal_guard_id(graph_namespace: str, authored_key: str, guard_payload: Any) -> str:
    return deterministic_id(
        "tg",
        {"graph_namespace": graph_namespace, "authored_key": authored_key, "guard": guard_payload},
    )


def graph_edge_id(graph_namespace: str, edge_payload: Any) -> str:
    return deterministic_id("ge", {"graph_namespace": graph_namespace, "edge": edge_payload})


def occurrence_anchor_id(
    source_id: str,
    predicate_id: str,
    event_time_anchor: Any,
    canonical_bindings: Any,
) -> str:
    return deterministic_id(
        "occ",
        {
            "source_id": source_id,
            "predicate_id": predicate_id,
            "event_time_anchor": event_time_anchor,
            "canonical_bindings": canonical_bindings,
        },
        length=32,
    )


def canonical_hypothesis_key(
    request_id: str,
    graph_hash: str,
    anchor_occurrence_id: str,
    canonical_bindings: Any,
    structural_branch_ids: Iterable[str] = (),
) -> str:
    return deterministic_id(
        "hypkey",
        {
            "request_id": request_id,
            "graph_hash": graph_hash,
            "anchor_occurrence_id": anchor_occurrence_id,
            "canonical_bindings": canonical_bindings,
            "structural_branch_ids": sorted(structural_branch_ids),
        },
        length=32,
    )


def demand_sharing_key(
    semantic_predicate: Any,
    event_time_interval: Any,
    acceptable_output_types: Iterable[str],
    hard_constraints: Any,
) -> str:
    return deterministic_id(
        "share",
        {
            "semantic_predicate": semantic_predicate,
            "event_time_interval": event_time_interval,
            "acceptable_output_types": sorted(acceptable_output_types),
            "hard_constraints": hard_constraints,
        },
        length=32,
    )


def physical_plan_label_id(label_payload: Any) -> str:
    return deterministic_id("label", label_payload, length=32)
