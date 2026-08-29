"""Small same-node typed stream bus used by live provider workers.

This is deliberately not a distributed streaming system.  A node agent owns one
bus, providers on that node publish ordinary FABLE provider data models to it,
and downstream workers subscribe by data type/source.  Cross-node intermediate
transport remains out of scope for the simplified runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable


@dataclass(frozen=True, slots=True, order=True)
class StreamKey:
    data_type: str
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.data_type:
            raise ValueError("data_type must be non-empty")
        object.__setattr__(self, "source_ids", tuple(sorted(self.source_ids)))


StreamCallback = Callable[[StreamKey, object], None]


@dataclass(frozen=True, slots=True)
class Subscription:
    token: int


class StreamBus:
    """Synchronous, thread-safe publish/subscribe bus for one compute node."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_token = 1
        self._subscriptions: dict[int, tuple[str, frozenset[str] | None, StreamCallback]] = {}

    def subscribe(
        self,
        data_type: str,
        callback: StreamCallback,
        *,
        source_ids: tuple[str, ...] | None = None,
    ) -> Subscription:
        if not data_type:
            raise ValueError("data_type must be non-empty")
        allowed = None if source_ids is None else frozenset(source_ids)
        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._subscriptions[token] = (data_type, allowed, callback)
        return Subscription(token)

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            self._subscriptions.pop(subscription.token, None)

    def publish(self, key: StreamKey, value: object) -> int:
        """Publish one value and return the number of subscribers invoked."""
        with self._lock:
            callbacks = []
            actual_sources = frozenset(key.source_ids)
            for data_type, allowed_sources, callback in self._subscriptions.values():
                if data_type != key.data_type:
                    continue
                if allowed_sources is not None and not actual_sources.issubset(allowed_sources):
                    continue
                callbacks.append(callback)
        for callback in callbacks:
            callback(key, value)
        return len(callbacks)
