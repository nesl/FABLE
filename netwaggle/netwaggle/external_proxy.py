"""Bounded TCP forwarding for external physical NetWaggle endpoints."""

from __future__ import annotations

import ctypes
import json
import os
import socket
import threading
import time
from pathlib import Path

from .docker_attach import docker_pid
from .topology import ExternalProxy
from .util import NetWaggleError


CLONE_NEWNET = 0x40000000


def _enter_network_namespace_fd(fd: int, label: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.setns(fd, CLONE_NEWNET) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), label)


def _enter_network_namespace(pid: int) -> None:
    path = f"/proc/{pid}/ns/net"
    fd = os.open(path, os.O_RDONLY)
    try:
        _enter_network_namespace_fd(fd, path)
    finally:
        os.close(fd)


class ProxyMetrics:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def emit(self, event: dict[str, object]) -> None:
        if self.path is None:
            return
        row = {"schema_version": "netwaggle.external_proxy_event.v1", **event}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


class TcpExternalProxy:
    def __init__(self, spec: ExternalProxy, metrics: ProxyMetrics) -> None:
        self.spec = spec
        self.metrics = metrics
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connection_number = 0
        self._counter_lock = threading.Lock()
        self._ready = threading.Event()
        self._start_error: OSError | None = None
        self._host_namespace_fd = os.open("/proc/self/ns/net", os.O_RDONLY)

    @property
    def bound_address(self) -> tuple[str, int] | None:
        return self._listener.getsockname()[:2] if self._listener else None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._setup_and_accept,
            name=f"netwaggle-proxy-{self.spec.name}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise OSError(f"proxy {self.spec.name} listener did not become ready")
        if self._start_error is not None:
            raise self._start_error

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _setup_and_accept(self) -> None:
        listener: socket.socket | None = None
        try:
            if self.spec.listen_namespace_anchor:
                _enter_network_namespace(
                    docker_pid(self.spec.listen_namespace_anchor)
                )
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.spec.listen_host, self.spec.listen_port))
            listener.listen(32)
            listener.settimeout(0.5)
            self._listener = listener
            self.metrics.emit({
                "event": "LISTENING", "proxy": self.spec.name,
                "listen_host": self.spec.listen_host,
                "listen_port": self.bound_address[1],
                "target_host": self.spec.target_host,
                "target_port": self.spec.target_port,
                "listen_namespace_anchor": self.spec.listen_namespace_anchor,
                "outbound_namespace_anchor": self.spec.outbound_namespace_anchor,
                "wall_time": time.time(),
            })
        except OSError as exc:
            self._start_error = exc
            if listener is not None:
                listener.close()
            self._ready.set()
            return
        self._ready.set()
        self._accept_loop()

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                client, peer = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if self.spec.allowed_peer_ips and peer[0] not in self.spec.allowed_peer_ips:
                self.metrics.emit({
                    "event": "REJECTED", "proxy": self.spec.name,
                    "peer": str(peer[0]), "reason": "peer IP is not allowlisted",
                    "wall_time": time.time(),
                })
                client.close()
                continue
            with self._counter_lock:
                self._connection_number += 1
                connection_id = f"{self.spec.name}:{self._connection_number}"
            threading.Thread(
                target=self._serve,
                args=(client, peer, connection_id),
                name=f"netwaggle-flow-{connection_id}",
                daemon=True,
            ).start()

    def _serve(self, client: socket.socket, peer, connection_id: str) -> None:
        started = time.monotonic()
        upstream: socket.socket | None = None
        counters = {"client_to_target_bytes": 0, "target_to_client_bytes": 0}
        error: str | None = None
        source_ip: str | None = None
        try:
            if self.spec.outbound_namespace_anchor:
                _enter_network_namespace(docker_pid(self.spec.outbound_namespace_anchor))
            elif self.spec.listen_namespace_anchor:
                _enter_network_namespace_fd(
                    self._host_namespace_fd, "/proc/self/ns/net",
                )
            upstream = socket.create_connection(
                (self.spec.target_host, self.spec.target_port),
                timeout=self.spec.connect_timeout_seconds,
            )
            upstream.settimeout(None)
            source_ip = str(upstream.getsockname()[0])
            if self.spec.expected_source_ip and source_ip != self.spec.expected_source_ip:
                raise NetWaggleError(
                    f"proxy {self.spec.name} bypassed its logical node: "
                    f"source {source_ip}, expected {self.spec.expected_source_ip}"
                )
            self.metrics.emit({
                "event": "CONNECTED", "proxy": self.spec.name,
                "connection_id": connection_id, "peer": str(peer[0]),
                "outbound_source_ip": source_ip, "wall_time": time.time(),
            })
            threads = (
                threading.Thread(target=self._pump, args=(client, upstream, counters, "client_to_target_bytes")),
                threading.Thread(target=self._pump, args=(upstream, client, counters, "target_to_client_bytes")),
            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        except Exception as exc:  # recorded per connection; listener remains bounded
            error = f"{type(exc).__name__}: {exc}"
        finally:
            for endpoint in (client, upstream):
                if endpoint is not None:
                    try:
                        endpoint.close()
                    except OSError:
                        pass
            self.metrics.emit({
                "event": "CLOSED", "proxy": self.spec.name,
                "connection_id": connection_id, **counters,
                "duration_seconds": time.monotonic() - started,
                "outbound_source_ip": source_ip, "error": error,
                "wall_time": time.time(),
            })

    @staticmethod
    def _pump(source: socket.socket, destination: socket.socket, counters: dict[str, int], key: str) -> None:
        try:
            while True:
                payload = source.recv(65536)
                if not payload:
                    break
                destination.sendall(payload)
                counters[key] += len(payload)
        except OSError:
            pass
        finally:
            try:
                destination.shutdown(socket.SHUT_WR)
            except OSError:
                pass


class ExternalProxyManager:
    def __init__(self, specs: list[ExternalProxy], metrics_path: Path | None = None) -> None:
        metrics = ProxyMetrics(metrics_path)
        self.proxies = [TcpExternalProxy(spec, metrics) for spec in specs]

    def start(self) -> None:
        started: list[TcpExternalProxy] = []
        try:
            for proxy in self.proxies:
                try:
                    proxy.start()
                    started.append(proxy)
                except OSError as exc:
                    if proxy.spec.required:
                        raise NetWaggleError(
                            f"required external proxy {proxy.spec.name} could not start: {exc}"
                        ) from exc
        except Exception:
            for proxy in reversed(started):
                proxy.stop()
            raise

    def stop(self) -> None:
        for proxy in reversed(self.proxies):
            proxy.stop()
