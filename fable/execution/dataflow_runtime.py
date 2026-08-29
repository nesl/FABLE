"""In-process live provider dataflow backend for one node agent."""
from __future__ import annotations

from collections import Counter
from typing import Callable, Mapping

from fable.providers.provider_capabilities import load_provider_capabilities

from .plan_reconciler import ProviderInstanceKey, ProviderInstanceSpec
from .provider_worker import DefaultProviderFactory, ProviderFactory, ProviderWorker
from .result_transport import ResultTransport
from .source_adapters import SourceAdapter
from .stream_bus import StreamBus, StreamKey


class NullResultTransport:
    def send_predicate_match(self, match) -> None:  # pragma: no cover - defensive default
        pass
    def send_identity_association(self, association) -> None:  # pragma: no cover
        pass


class DataflowProviderRuntime:
    """Materialize planned providers as connected workers on one node.

    This is the canonical live backend for the simplified runtime: providers run
    in the node-agent process and exchange typed Python data models over one
    same-node ``StreamBus``.  It avoids introducing a distributed intermediate
    artifact protocol while still making dynamic START/KEEP/STOP actions affect
    real provider computation.
    """

    def __init__(
        self,
        *,
        result_transport: ResultTransport | None = None,
        provider_factories: Mapping[str, Callable[[], object]] | None = None,
        provider_factory: ProviderFactory | None = None,
        source_adapters: Mapping[str, SourceAdapter] | None = None,
        provider_catalog: Mapping[str, object] | None = None,
        bus: StreamBus | None = None,
    ) -> None:
        self.results = result_transport or NullResultTransport()
        self.provider_factories = dict(provider_factories or {})
        self.provider_factory = provider_factory or DefaultProviderFactory()
        self.source_adapters = dict(source_adapters or {})
        self.catalog = provider_catalog if provider_catalog is not None else load_provider_capabilities()
        self.bus = bus or StreamBus()
        self.workers: dict[ProviderInstanceKey, ProviderWorker] = {}
        self._source_refcounts: Counter[str] = Counter()

    def start(self, spec: ProviderInstanceSpec) -> None:
        if spec.key in self.workers:
            return
        provider = self._make_provider(spec.key.provider_id)
        worker = ProviderWorker(
            spec,
            provider,
            self.bus,
            self.results,
            provider_catalog=self.catalog,
        )
        worker.start()
        self.workers[spec.key] = worker
        try:
            self._retain_sources(worker)
        except Exception:
            self.workers.pop(spec.key, None)
            worker.stop()
            raise

    def stop(self, key: ProviderInstanceKey) -> None:
        worker = self.workers.pop(key, None)
        if worker is None:
            return
        self._release_sources(worker)
        worker.stop()

    def running(self) -> tuple[ProviderInstanceKey, ...]:
        return tuple(sorted(self.workers))

    def ready(self, key: ProviderInstanceKey) -> bool:
        worker = self.workers.get(key)
        return bool(worker is not None and worker.ready)

    def worker_status(self, key: ProviderInstanceKey):
        worker = self.workers.get(key)
        return None if worker is None else worker.status()

    def publish_source(self, source_id: str, data_type: str, value: object) -> int:
        """Inject a source value directly (useful for replay/tests)."""
        return self.bus.publish(StreamKey(data_type, (source_id,)), value)

    def _make_provider(self, provider_id: str) -> object:
        factory = self.provider_factories.get(provider_id)
        return factory() if factory is not None else self.provider_factory(provider_id)

    def _retain_sources(self, worker: ProviderWorker) -> None:
        raw_types = {"video_frame", "audio_window", "multichannel_audio"}
        if not raw_types.intersection(worker.input_types):
            return
        for source_id in worker.spec.key.source_ids:
            adapter = self.source_adapters.get(source_id)
            if adapter is None or adapter.data_type not in worker.input_types:
                continue
            self._source_refcounts[source_id] += 1
            if self._source_refcounts[source_id] == 1:
                adapter.start(self.bus)

    def _release_sources(self, worker: ProviderWorker) -> None:
        raw_types = {"video_frame", "audio_window", "multichannel_audio"}
        if not raw_types.intersection(worker.input_types):
            return
        for source_id in worker.spec.key.source_ids:
            adapter = self.source_adapters.get(source_id)
            if adapter is None or adapter.data_type not in worker.input_types:
                continue
            if self._source_refcounts[source_id] <= 1:
                self._source_refcounts.pop(source_id, None)
                adapter.stop()
            else:
                self._source_refcounts[source_id] -= 1
