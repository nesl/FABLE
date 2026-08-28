"""Generic runtime-telemetry integration contracts for orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from fable.distributed.transport import Transport
from fable.common.schemas import RuntimeLinkUpdate

RuntimeLinkCallback = Callable[[tuple[RuntimeLinkUpdate, ...], str], None]


class NetworkTelemetrySource(Protocol):
    """Adapter that turns external network telemetry into generic link updates."""

    def bind(self, transport: Transport, callback: RuntimeLinkCallback) -> None: ...
