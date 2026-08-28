#!/usr/bin/env python3
"""Run every recommended complex-event example with bounded stack lifecycles."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import paho.mqtt.client as mqtt
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPLAY_ROOT = ROOT / "iobt-minimal-ce-replay"
CATALOG_PATH = ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
SCENARIO_PATH = REPLAY_ROOT / "generated/scenario_catalog.json"
PROVIDER_CATALOG_PATH = ROOT / "providers/registry/catalog.yaml"
PLAYABLE_ZED_NODES_BY_CAMPAIGN = {
    2024: {"orin1", "orin4", "orin7"},
    2025: {"orin1", "orin4"},
}
MOBILE_AUGMENTED_VARIANTS = {
    "Robbery with alarm",
    "Talking/rendezvous",
    # Retained for reproducibility of historical manifests. Current corrected
    # rows 26–28 use Vehicle rendezvous.
    "Two-visit stalking",
    "Vehicle rendezvous",
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog
from evaluation.baselines.static_registry import (
    StaticPipelineRegistry,
    static_pipeline_registry_path,
)
from evaluation.defaults import DEFAULT_PLAYBACK_MODE, DEFAULT_PLAYBACK_SPEED
from evaluation.execution_timeline import write_execution_timeline
from evaluation.schemas import BaselineId
from scripts.run_replay_accuracy import _resolve_experiment
from scripts.stage_rpi_replay import (
    DEFAULT_CONVERSION_CACHE,
    DEFAULT_DATA_ROOTS,
    ensure_left_mp4,
    resolve_scenario_video,
)

STATIC_BASELINES = {
    BaselineId.B0_PRODUCE_ALL.value,
    BaselineId.B1_STATIC_WHOLE_EVENT.value,
}

# Static placements are durable evaluation artifacts and may refer to chain
# names from the pre-redesign catalog.  Resolve only the two explicitly
# versioned aliases; an arbitrary unknown chain must still fail closed.
STATIC_CHAIN_ALIASES = {
    "same_entity_cross_camera_reid": "follows_cross_camera_reid",
    "recover_vehicle_from_local_segments": "recover_vehicle_before_audio_event",
}


def resolve_static_chain_id(chain_id: str) -> str:
    return STATIC_CHAIN_ALIASES.get(chain_id, chain_id)
def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, check=False, **kwargs)


def command_without_options(
    command: list[str], option_names: set[str]
) -> list[str]:
    """Remove allowlisted value-taking options from a child argv."""

    result: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item in option_names:
            index += 2
            continue
        result.append(item)
        index += 1
    return result


def start_physical_rpi_execution(args, *, logical_node: str) -> dict[str, object]:
    """Start one controlled Pi->PC->Jetson replay path for a suite cell."""
    jetson_root = args.physical_jetson_root.rstrip("/")
    rpi_root = args.physical_rpi_root.rstrip("/")
    identity_args = (
        ["-i", str(args.physical_rpi_identity_file.resolve(strict=True))]
        if args.physical_rpi_identity_file is not None
        else []
    )
    subprocess.run(
        [
            "scp", *identity_args, "-o", "BatchMode=yes",
            str(ROOT / "scripts/physical_jetson_yolo.py"),
            f"{args.physical_jetson_host}:{jetson_root}/scripts/physical_jetson_yolo.py",
        ], check=True, text=True,
    )
    subprocess.run(
        [
            "scp", *identity_args, "-o", "BatchMode=yes",
            str(ROOT / "scripts/physical_sampling.py"),
            f"{args.physical_jetson_host}:{jetson_root}/scripts/physical_sampling.py",
        ], check=True, text=True,
    )
    subprocess.run(
        [
            "scp", *identity_args, "-o", "BatchMode=yes",
            str(ROOT / "scripts/physical_jetson_load.py"),
            f"{args.physical_jetson_host}:{jetson_root}/scripts/physical_jetson_load.py",
        ], check=True, text=True,
    )
    if args.physical_netwaggle_proxies:
        for port in (21883, 28091):
            listener = subprocess.run(
                ["ss", "-ltn", f"sport = :{port}"],
                check=False, capture_output=True, text=True,
            )
            if f":{port}" not in listener.stdout:
                raise RuntimeError(
                    f"required NetWaggle physical proxy is not listening on port {port}"
                )
        stream_port = 28091
        mqtt_port = 21883
        relay_description = (
            "Jetson:28091 -> s_jetson -> s_edge -> s_rpi -> Pi:8090"
        )
    else:
        listener = subprocess.run(
            ["sh", "-c", "ss -ltn | grep -q ':8091 '"], check=False
        )
        if listener.returncode:
            subprocess.Popen(
                [
                    "socat",
                    "TCP-LISTEN:8091,reuseaddr,fork",
                    f"TCP:{args.physical_rpi_address}:8090",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        stream_port = 8091
        mqtt_port = 1883
        relay_description = (
            f"{args.physical_host_address}:8091 -> "
            f"{args.physical_rpi_address}:8090"
        )
    pi_command = (
        f"cd {shlex.quote(rpi_root)} && "
        "if test -f replay-cache/ffmpeg.pid; then "
        "kill $(cat replay-cache/ffmpeg.pid) 2>/dev/null || true; fi; "
        "nohup ffmpeg -re -i replay-cache/current.mp4 -an -c:v copy "
        "-f mpegts -listen 1 http://0.0.0.0:8090/stream.ts "
        "> replay-cache/ffmpeg.log 2>&1 & echo $! > replay-cache/ffmpeg.pid"
    )
    subprocess.run(
        ["ssh", *identity_args, "-o", "BatchMode=yes", args.physical_rpi_host, pi_command],
        check=True, text=True,
    )
    source_id = (
        f"dvpg_gq_orin_{logical_node[4:]}"
        if logical_node.startswith("orin") and logical_node[4:].isdigit()
        else logical_node
    )
    jetson_command = (
        f"cd {shlex.quote(jetson_root)} && "
        "if test -f state/physical-yolo.pid; then "
        "kill $(cat state/physical-yolo.pid) 2>/dev/null || true; fi; "
        "nohup /usr/bin/python3 scripts/physical_jetson_yolo.py "
        f"--stream-url http://{args.physical_host_address}:{stream_port}/stream.ts "
        f"--model models/yolov8n.pt --broker {args.physical_host_address} --port {mqtt_port} "
        f"--source-id {source_id} --maximum-rate-hz {args.physical_yolo_rate_hz:g} "
        "--controlled-replay "
        "--publish-raw-frames "
        "> logs/physical-yolo.log 2>&1 & echo $! > state/physical-yolo.pid"
    )
    subprocess.run(
        [
            "ssh", *identity_args, "-o", "BatchMode=yes",
            args.physical_jetson_host, jetson_command,
        ], check=True, text=True,
    )
    return {
        "logical_replay_node": logical_node,
        "source_id": source_id,
        "pi_stream": f"http://{args.physical_rpi_address}:8090/stream.ts",
        "pc_relay": relay_description,
        "netwaggle_external_proxies": bool(args.physical_netwaggle_proxies),
        "jetson_worker": args.physical_jetson_host,
        "provider": "yolov8n-jetson-cuda",
    }


def condition_recovery_budget_seconds(
    condition_trace_path: str | Path | None,
    *,
    recovery_allowance_seconds: float = 90.0,
) -> float | None:
    """Return the minimum wall budget for a late condition transition.

    The ordinary scenario bound ends at replay EOF plus the evaluation
    deadline. A robustness trace may deliberately restore connectivity after
    EOF, so it also needs a bounded interval in which explicit catch-up can
    run. Track-forming retrospective providers sample at 3 FPS and can require
    roughly one wall second per three seconds of an outage interval; the
    additional allowance also covers the detector queue and semantic result
    propagation. This changes only the evaluator wall budget; it does not
    extend the raw replay interval or ordinary evidence boundary.
    """

    if condition_trace_path is None:
        return None
    document = json.loads(Path(condition_trace_path).read_text(encoding="utf-8"))
    offsets = [
        float(item["offset_s"])
        for item in document.get("transitions", ())
        if item.get("action") in {"RESTORE_LINK", "RESTORE_PROVIDER"}
    ]
    if not offsets:
        return None
    # ``ADMISSION`` anchors occur after replay synchronization/readiness.
    # The trace duration is expressed in that same anchored domain and is the
    # authoritative bounded wall allowance when it is larger.
    return max(
        max(offsets) + recovery_allowance_seconds,
        float(document.get("duration_s") or 0.0),
    )


def compose_command(bundle: Path, *arguments: str) -> list[str]:
    command = [
        "docker",
        "compose",
        "-f",
        str(REPLAY_ROOT / "compose.server.yaml"),
        "-f",
        str(bundle / "compose.replay.yaml"),
        "-f",
        str(bundle / "compose.fable.providers.yaml"),
    ]
    netwaggle_override = bundle / "compose.netwaggle.override.yaml"
    if netwaggle_override.is_file():
        command.extend(("-f", str(netwaggle_override)))
    static_registry_override = bundle / "compose.static-registry.override.yaml"
    if static_registry_override.is_file():
        command.extend(("-f", str(static_registry_override)))
    command.extend(arguments)
    return command


def write_netwaggle_override(
    bundle: Path,
    topology_path: Path,
    *,
    drop_offline_evidence: bool = False,
) -> dict[str, str]:
    """Attach services to NetWaggle with node-local MQTT data planes.

    A workload that shares a logical sensor namespace must not send traffic to
    another workload in that same namespace through the shaped sensor uplink.
    Each sensor therefore receives a loopback Mosquitto broker.  Its bridge
    exports control, readiness, compact semantic evidence, and detector
    metadata to the host broker; raw replay payloads remain node-local.
    """

    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    anchor_by_container = {
        str(container): str(node.get("anchor_container", f"netwaggle-node-{node['name']}"))
        for node in topology.get("logical_nodes", topology.get("nodes", ()))
        for container in node.get("containers", ())
    }
    anchor_by_node = {
        str(node["name"]).replace("_", "-"): str(
            node.get("anchor_container", f"netwaggle-node-{node['name']}")
        )
        for node in topology.get("logical_nodes", topology.get("nodes", ()))
    }
    services: dict[str, object] = {}
    bindings: dict[str, str] = {}
    local_broker_by_anchor: dict[str, str] = {}
    local_broker_config_dir = bundle / "netwaggle-local-mqtt"

    def local_broker(anchor: str) -> str | None:
        node_name = next(
            (
                name
                for name, candidate in anchor_by_node.items()
                if candidate == anchor
            ),
            None,
        )
        if node_name is None or not re.fullmatch(
            r"(?:orin(?:[1-9]|[12][0-9]|30)|mobile-archive-[1-6])", node_name
        ):
            return None
        existing = local_broker_by_anchor.get(anchor)
        if existing is not None:
            return existing
        service_name = f"netwaggle-mqtt-{node_name}"
        container_name = service_name
        node_topic = node_name.replace("orin", "dvpg_gq_orin_").replace(
            "mobile-archive-", "mobile_archive_"
        )
        config_path = local_broker_config_dir / f"{node_name}.conf"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_qos = 0 if drop_offline_evidence else 1
        control_qos = 0 if drop_offline_evidence else 1
        max_queued_messages = 1 if drop_offline_evidence else 20000
        max_queued_bytes = 1 if drop_offline_evidence else 0
        clean_session = "true" if drop_offline_evidence else "false"
        reconnect_timeout = "restart_timeout 2 5" if drop_offline_evidence else "restart_timeout 5 30"
        config_path.write_text(
            "\n".join(
                (
                    "listener 1883 127.0.0.1",
                    "allow_anonymous true",
                    "persistence false",
                    f"max_queued_messages {max_queued_messages}",
                    f"max_queued_bytes {max_queued_bytes}",
                    "max_inflight_messages 200",
                    "queue_qos0_messages false",
                    "log_dest stdout",
                    "log_type warning",
                    "connection host-control",
                    "address 10.255.0.1:1883",
                    "bridge_protocol_version mqttv311",
                    reconnect_timeout,
                    f"cleansession {clean_session}",
                    "try_private true",
                    f"remote_clientid netwaggle-control-{node_name}",
                    # Typed orchestration traffic has an independent TCP
                    # session so dense evidence cannot create head-of-line
                    # blocking for results, commands, or acknowledgements.
                    f"topic fable/v1/# both {control_qos}",
                    # Raw retrospective recovery uses the historical leading-
                    # slash namespace. MQTT topic names are byte-exact, so the
                    # canonical FABLE filter above does not include it.
                    f"topic /fable/v1/retrospective/# both {control_qos}",
                    f"topic /replay/sync both {control_qos}",
                    f"topic /replay/config both {control_qos}",
                    f"topic /replay/command/# both {control_qos}",
                    f"topic /readiness/# out {control_qos}",
                    # Seed discovery is control-plane traffic.  These compact
                    # typed observations must not sit behind the per-frame
                    # evidence backlog when a sensor uplink is degraded.
                    f"topic /{node_topic}/fable/vehicle/predicates out {evidence_qos}",
                    f"topic /{node_topic}/fable/interactions/predicates out {evidence_qos}",
                    f"topic /{node_topic}/fable/audio/events out {evidence_qos}",
                    "connection host-evidence",
                    "address 10.255.0.1:1883",
                    "bridge_protocol_version mqttv311",
                    reconnect_timeout,
                    f"cleansession {clean_session}",
                    "try_private true",
                    f"remote_clientid netwaggle-evidence-{node_name}",
                    # Raw replay frame topics remain absent. Compact evidence
                    # and measurement metadata may queue without delaying the
                    # control/result session above.
                    "topic /replay/status/# out 0",
                    f"topic /{node_topic}/fable/vehicle/tracks out {evidence_qos}",
                    f"topic /{node_topic}/fable/identity/descriptors out {evidence_qos}",
                    f"topic /{node_topic}/analytics/yolo/bbox out 0",
                    f"topic /debug/{node_topic}/# out 0",
                    f"topic /{node_topic}/audio_detector/detections out 0",
                    "",
                )
            ),
            encoding="utf-8",
        )
        services[service_name] = {
            "image": "eclipse-mosquitto:2",
            "container_name": container_name,
            "network_mode": f"container:{anchor}",
            "volumes": [f"{config_path}:/mosquitto/config/mosquitto.conf:ro"],
            "restart": "unless-stopped",
        }
        bindings[container_name] = anchor
        local_broker_by_anchor[anchor] = service_name
        return service_name

    for source in (bundle / "compose.replay.yaml", bundle / "compose.fable.providers.yaml"):
        document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        source_changed = False
        for service_name, service in document.get("services", {}).items():
            if not isinstance(service, dict):
                continue
            container_name = str(service.get("container_name", ""))
            anchor = anchor_by_container.get(container_name)
            if (
                anchor is None
                and "x86server" not in container_name
                and not container_name.startswith("yolo-edge-")
            ):
                match = re.search(r"-(orin(?:[1-9]|[12][0-9]|30))$", container_name)
                if match is not None:
                    anchor = anchor_by_node.get(match.group(1))
            if anchor is None:
                continue
            broker_service = local_broker(anchor)
            # Compose cannot combine a shared/external network namespace with
            # per-container network configuration inherited from the bundle.
            for incompatible in (
                "extra_hosts", "networks", "ports", "hostname", "dns", "dns_search"
            ):
                source_changed = bool(service.pop(incompatible, None)) or source_changed
            override_service: dict[str, object] = {
                "network_mode": f"container:{anchor}",
                "environment": {
                    "MQTT_HOST_IP": "127.0.0.1" if broker_service else "10.255.0.1",
                    "MQTT_HOST": "127.0.0.1" if broker_service else "10.255.0.1",
                },
            }
            if broker_service:
                override_service["depends_on"] = {
                    broker_service: {"condition": "service_started"}
                }
            services[str(service_name)] = override_service
            bindings[container_name] = anchor
        if source_changed:
            source.write_text(
                yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
            )
    if not bindings:
        raise ValueError(
            f"no generated services in {bundle} matched topology {topology_path}"
        )
    override = bundle / "compose.netwaggle.override.yaml"
    override.write_text(
        yaml.safe_dump({"services": services}, sort_keys=False), encoding="utf-8"
    )
    return bindings


def missing_netwaggle_anchor(stack_log: Path) -> str | None:
    """Return the missing external anchor named by a Compose startup failure.

    This is intentionally narrower than accepting any ``No such container``
    error: only generated NetWaggle anchors are optional when the host topology
    is implemented by Mininet/OVS rather than Docker namespace containers.
    """

    try:
        output = stack_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(
        r"No such container:\s+(netwaggle-node-[A-Za-z0-9_.-]+)", output
    )
    return match.group(1) if match is not None else None


def validate_netwaggle_bindings(bindings: dict[str, str]) -> dict[str, object]:
    """Prove that each workload container shares its declared anchor netns."""

    rows: list[dict[str, object]] = []
    for container, anchor in sorted(bindings.items()):
        inspected = run(["docker", "inspect", container, anchor], capture_output=True)
        row: dict[str, object] = {
            "container": container,
            "anchor": anchor,
            "valid": False,
            "inspection_error": inspected.stderr.strip(),
        }
        if inspected.returncode == 0:
            try:
                workload, anchor_doc = json.loads(inspected.stdout)
                workload_running = bool(workload["State"]["Running"])
                anchor_running = bool(anchor_doc["State"]["Running"])
                workload_netns = str(
                    workload.get("NetworkSettings", {}).get("SandboxKey") or ""
                )
                anchor_netns = str(
                    anchor_doc.get("NetworkSettings", {}).get("SandboxKey") or ""
                )
                network_mode = str(workload.get("HostConfig", {}).get("NetworkMode") or "")
                anchor_id = str(anchor_doc.get("Id") or "")
                mode_matches_anchor = (
                    network_mode == f"container:{anchor_id}"
                    or (
                        network_mode.startswith("container:")
                        and anchor_id.startswith(network_mode.split(":", 1)[1])
                    )
                )
                row.update(
                    {
                        "workload_running": workload_running,
                        "anchor_running": anchor_running,
                        "workload_netns": workload_netns,
                        "anchor_netns": anchor_netns,
                        "network_mode": network_mode,
                        "network_mode_matches_anchor": mode_matches_anchor,
                        "validation_method": "docker_hostconfig_network_mode",
                        "valid": (
                            workload_running
                            and anchor_running
                            and mode_matches_anchor
                        ),
                    }
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                row["inspection_error"] = str(exc)
        rows.append(row)
    return {
        "schema_version": "fable.netwaggle_binding_validation.v1",
        "binding_count": len(rows),
        "valid": bool(rows) and all(bool(row["valid"]) for row in rows),
        "bindings": rows,
    }
def stop_stack(bundle: Path | None) -> None:
    if bundle is not None:
        down_arguments = ["down", "--volumes", "--timeout", "20"]
        # Never ask a scenario bundle to remove project orphans. External
        # NetWaggle anchors share this Compose project but are intentionally
        # absent from ordinary replay bundles. Explicit cleanup below removes
        # stale scenario containers while preserving netwaggle-node-*.
        run(
            compose_command(bundle, *down_arguments),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    # Compose can leave adopted/orphaned services after switching override files.
    ids = run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.docker.compose.project=iobt-minimal-ce-replay",
        ],
        capture_output=True,
    ).stdout.split()
    if ids:
        names = run(
            ["docker", "inspect", "--format", "{{.Name}}", *ids],
            capture_output=True,
        ).stdout.splitlines()
        removable = [
            container_id
            for container_id, name in zip(ids, names)
            if not name.lstrip("/").startswith("netwaggle-node-")
        ]
        if removable:
            run(["docker", "rm", "-f", *removable], stdout=subprocess.DEVNULL)

    # ``docker compose down --volumes`` can return early when the shared
    # NetWaggle network still has physical proxy endpoints.  In that case its
    # experiment Mongo volume survives and the next isolated cell restores old
    # leases, plans, and capacity reservations.  Remove only the explicitly
    # named, project-scoped FABLE state volume after all of its containers are
    # gone; never enumerate or prune unrelated Docker volumes.
    mongo_volume = "iobt-minimal-ce-replay_fable-mongo-data"
    mongo_users = run(
        ["docker", "ps", "-aq", "--filter", f"volume={mongo_volume}"],
        capture_output=True,
    ).stdout.split()
    if not mongo_users:
        run(["docker", "volume", "rm", mongo_volume], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def quiesce_lease_controlled_providers(
    bundle: Path, *, preserve: set[str] | None = None
) -> list[str]:
    """Stop generated provider containers until a FABLE lease selects them.

    The replay sources, broker, database, orchestrator, and node agents are not
    present in the runtime catalog and are deliberately left running.
    """

    document = yaml.safe_load(
        (bundle / "fable_provider_runtimes.yaml").read_text(encoding="utf-8")
    ) or {}
    preserved = preserve or set()
    names = sorted(
        {
            str(runtime["container_name"])
            for node in document.get("nodes", {}).values()
            for runtime in node.get("providers", {}).values()
            if runtime.get("mode") == "ADOPT_EXISTING"
            and runtime.get("stop_adopted_when_idle") is True
            and runtime.get("container_name")
            and str(runtime["container_name"]) not in preserved
        }
    )
    if not names:
        return []
    existing = run(
        ["docker", "inspect", "--format", "{{.Name}}", *names],
        capture_output=True,
    )
    if existing.returncode:
        raise RuntimeError(
            "lease-controlled provider inventory is incomplete: "
            + existing.stderr.strip()
        )
    completed = run(["docker", "stop", "--time", "10", *names], capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "could not quiesce lease-controlled providers: "
            + completed.stderr.strip()
        )
    return names


def lease_controlled_provider_containers(bundle: Path) -> set[str]:
    """Return containers which must remain stopped until a provider lease.

    Starting the entire generated Compose bundle and stopping these containers
    later creates a correctness race: node agents can adopt the briefly-running
    providers and reserve their shared GPU pool before the first request is
    planned.  The planner then rejects an otherwise feasible device-local
    realization.  Compose now creates every container, but startup excludes
    this inventory so ADOPT_EXISTING still has a concrete stopped container.
    """

    document = yaml.safe_load(
        (bundle / "fable_provider_runtimes.yaml").read_text(encoding="utf-8")
    ) or {}
    return {
        str(runtime["container_name"])
        for node in document.get("nodes", {}).values()
        for runtime in node.get("providers", {}).values()
        if runtime.get("mode") == "ADOPT_EXISTING"
        and runtime.get("stop_adopted_when_idle") is True
        and runtime.get("container_name")
    }


WARM_RESIDENT_PROVIDER_CONTAINERS = frozenset(
    {"fable-reid-x86server", "fable-vlm-cloud1"}
)


def create_stack_with_quiescent_providers(
    bundle: Path, *, log_handle
) -> subprocess.CompletedProcess[str]:
    """Create the bundle while keeping inference providers quiescent.

    The site ReID descriptor worker is deliberately model-warm but work-idle.
    Loading FastReID only after a late SAME_ENTITY demand consumed most of the
    graph deadline and prevented even the bounded VLM fallback from running.
    Starting this one worker does not activate a provider chain: it receives no
    crop messages until the identity service issues a typed, bounded request.
    Its model-residency interval remains visible in resource instrumentation.
    """

    created = run(
        compose_command(bundle, "create", "--no-build"),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    if created.returncode:
        return created
    configured = run(
        compose_command(bundle, "config", "--format", "json"),
        capture_output=True,
        timeout=60,
    )
    if configured.returncode:
        log_handle.write(configured.stderr)
        return configured
    compose_model = json.loads(configured.stdout)
    controlled = lease_controlled_provider_containers(bundle)
    start_services = sorted(
        service_name
        for service_name, service in (compose_model.get("services") or {}).items()
        if (
            str(service.get("container_name") or service_name) not in controlled
            or str(service.get("container_name") or service_name)
            in WARM_RESIDENT_PROVIDER_CONTAINERS
        )
    )
    if not start_services:
        raise RuntimeError("generated bundle contains no non-provider services")
    return run(
        compose_command(bundle, "start", *start_services),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        timeout=180,
    )


def pin_authored_static_provider_containers(
    bundle: Path,
    *,
    baseline_id: str,
    placement_id: str,
    trace_id: str = "",
) -> list[str]:
    """Keep only the authored B0/B1 containers resident for the whole run.

    Compose initially starts generated providers. Ordinary FABLE evaluation
    then quiesces lease-controlled containers and lets the node agents start
    and stop them. Static baselines instead pin their CE-specific containers:
    B1 on exemplar nodes and B0 on every generated sensor node. All runtime
    entries sharing a physical container are marked non-stoppable, because a
    released logical lease must not tear down another authored static stage.
    """

    if baseline_id not in STATIC_BASELINES:
        return []
    placement = StaticPipelineRegistry.load(static_pipeline_registry_path()).get_placement(
        placement_id, trace_id=trace_id
    )
    if placement is None or not placement.allowed_provider_ids:
        raise RuntimeError(
            f"static baseline {baseline_id} has no provider inventory for {placement_id}"
        )
    path = bundle / "fable_provider_runtimes.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    nodes = document.get("nodes") or {}
    authored_nodes = set(placement.allowed_node_ids)
    if baseline_id == BaselineId.B0_PRODUCE_ALL.value:
        authored_nodes = {
            str(node_id)
            for node_id in nodes
            if str(node_id) not in {"x86server", "cloud1"}
        }

    # B0 deliberately broadcasts the CE-specific provider inventory. B1 is
    # different: its authored contract is a set of exact chain/node pairs.
    # Combining all allowed providers with all allowed nodes would turn B1
    # into a partial fan-out baseline (for example pinning vehicle YOLO on an
    # audio-only node). Resolve each chain's providers independently instead.
    provider_ids = set(placement.allowed_provider_ids)
    providers_by_node: dict[str, set[str]] = {}
    if baseline_id == BaselineId.B1_STATIC_WHOLE_EVENT.value:
        if not placement.allowed_chain_node_ids:
            raise RuntimeError(
                f"B1 placement {placement_id} has no exact chain/node mapping"
            )
        exact_provider_nodes = placement.allowed_chain_provider_node_ids
        catalog = yaml.safe_load(PROVIDER_CATALOG_PATH.read_text(encoding="utf-8")) or {}
        chains = catalog.get("chains") or {}
        for chain_id, chain_nodes in placement.allowed_chain_node_ids.items():
            resolved_chain_id = resolve_static_chain_id(chain_id)
            chain = chains.get(resolved_chain_id)
            if not isinstance(chain, dict):
                raise RuntimeError(
                    f"B1 placement {placement_id} references unknown chain "
                    f"{chain_id} (resolved={resolved_chain_id})"
                )
            exact_chain = exact_provider_nodes.get(chain_id) or {}
            if exact_chain:
                for provider_id, node_ids in exact_chain.items():
                    for node_id in node_ids:
                        providers_by_node.setdefault(str(node_id), set()).add(
                            str(provider_id)
                        )
                continue
            chain_provider_ids = {
                str(step["provider"])
                for step in (chain.get("steps") or ())
                if isinstance(step, dict) and step.get("provider")
            }
            if provider_ids:
                chain_provider_ids &= provider_ids
            for node_id in chain_nodes:
                providers_by_node.setdefault(str(node_id), set()).update(
                    chain_provider_ids
                )
    else:
        providers_by_node = {node_id: set(provider_ids) for node_id in authored_nodes}

    pinned: set[str] = set()
    for node_id, node in nodes.items():
        node_provider_ids = providers_by_node.get(str(node_id), set())
        if not node_provider_ids:
            continue
        for provider_id, runtime in (node.get("providers") or {}).items():
            if str(provider_id) not in node_provider_ids:
                continue
            container = str(runtime.get("container_name") or "")
            if container:
                pinned.add(container)
    if not pinned:
        raise RuntimeError(
            f"static provider inventory for {placement_id} matched no generated containers"
        )
    for node in nodes.values():
        for runtime in (node.get("providers") or {}).values():
            if str(runtime.get("container_name") or "") in pinned:
                runtime["stop_adopted_when_idle"] = False
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return sorted(pinned)


def inspect_static_provider_lifecycle(
    container_names: list[str], *, runner_started_at: datetime,
    physical_execution: dict[str, object] | None = None,
) -> dict[str, object]:
    if not container_names:
        return {"applicable": False}
    inspected = run(
        [
            "docker", "inspect", "--format",
            "{{.Name}}|{{.State.Running}}|{{.State.StartedAt}}",
            *container_names,
        ],
        capture_output=True,
    )
    records = []
    for line in inspected.stdout.splitlines():
        name, running, started_at = line.split("|", 2)
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        physical_node = str((physical_execution or {}).get("logical_replay_node") or "")
        physical_detector = bool(
            physical_node
            and (physical_execution or {}).get("provider")
            and name.lstrip("/") == f"yolo-detector-{physical_node}"
        )
        records.append(
            {
                "container_name": name.lstrip("/"),
                "running_after_terminal": running == "true",
                "started_at": started.isoformat(),
                "started_before_runner": started <= runner_started_at,
                "satisfied_by_physical_replacement": physical_detector,
            }
        )
    valid = (
        inspected.returncode == 0
        and len(records) == len(container_names)
        and all(
            (
                row["running_after_terminal"] and row["started_before_runner"]
            )
            or row["satisfied_by_physical_replacement"]
            for row in records
        )
    )
    return {
        "applicable": True,
        "schema_version": "fable.static_provider_lifecycle.v1",
        "expected_container_count": len(container_names),
        "containers": records,
        "valid": valid,
        "inspection_error": inspected.stderr.strip(),
    }


def evaluation_nodes_for_variant(
    nodes: list[str],
    variant: str,
    campaign_year: int | None = None,
) -> list[str]:
    """Keep verified 2024 mobiles and explicitly visual 2025 profiles."""

    return [
        node
        for node in nodes
        if (
            not node.startswith("mobile_archive_")
            or campaign_year == 2024
            or variant in MOBILE_AUGMENTED_VARIANTS
        )
    ]


def candidate_zed_nodes(row: dict[str, object], campaign_year: int) -> list[str]:
    """Choose recorded camera nodes, falling back beyond the fast-path set.

    ``row['nodes']`` is the union of camera and microphone sources.  Treating an
    audio-only Orin as a camera produced a stack that could never become ZED
    ready.  The campaign fast-path is a preference, not a claim that every
    other recorded ZED stream is unplayable.
    """

    recorded = {
        str(node)
        for node in row.get("zed_nodes", ())  # type: ignore[union-attr]
    }
    preferred = recorded & PLAYABLE_ZED_NODES_BY_CAMPAIGN.get(campaign_year, recorded)
    if preferred:
        return sorted(preferred)
    # Keep the bounded suite at two cameras when it must leave the calibrated
    # fast path; the readiness probe will still fail closed on bad media.
    return sorted(recorded)[:2]


def canonical_replay_node(value: str) -> str:
    """Normalize equivalent Orin spellings without changing archive node IDs."""

    match = re.fullmatch(r"orin[_-]?(\d+)", value.strip(), re.IGNORECASE)
    return f"orin{int(match.group(1))}" if match else value.strip()


def select_playable_replay_nodes(
    playable_nodes: list[str],
    requested_nodes: list[str],
    maximum_nodes: int | None = None,
) -> tuple[list[str], bool]:
    """Reconcile ranked planned candidates with scenario-observed availability."""

    playable_by_id = {
        canonical_replay_node(node): node for node in playable_nodes
    }
    ranked = list(
        dict.fromkeys(canonical_replay_node(node) for node in requested_nodes)
    )
    selected = [playable_by_id[node] for node in ranked if node in playable_by_id]
    fallback_used = bool(ranked and not selected)
    if not ranked or fallback_used:
        selected = list(playable_nodes)
    if maximum_nodes is not None:
        selected = selected[:maximum_nodes]
    return selected, fallback_used


def wait_for_orchestrator(timeout_seconds: float = 45) -> bool:
    # The ready marker can be displaced from a short log tail almost
    # immediately on larger deployments.  Each suite creates/recreates this
    # named container, so reading its complete current-container log is both
    # bounded and substantially more reliable than ``--tail 20``.
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        inspected = run(
            [
                "docker",
                "inspect",
                "fable-orchestrator",
                "--format",
                "{{.State.Running}}",
            ],
            capture_output=True,
        )
        logs = run(
            ["docker", "logs", "fable-orchestrator"],
            capture_output=True,
        )
        if (
            inspected.returncode == 0
            and inspected.stdout.strip() == "true"
            and "FABLE orchestrator ready" in (logs.stdout + logs.stderr)
        ):
            # Ensure MQTT subscriptions established by agents immediately
            # after the orchestrator are also live before a one-shot request.
            time.sleep(3)
            return True
        time.sleep(0.5)
    return False


def probe_playable_nodes(
    scenario: str,
    nodes: list[str],
    *,
    timeout_seconds: float = 30,
) -> list[str]:
    ready: set[str] = set()
    lock = threading.Lock()
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"full-suite-probe-{uuid4().hex[:8]}",
        protocol=mqtt.MQTTv311,
    )

    def on_connect(client, _userdata, _flags, reason_code, _properties=None):
        if int(getattr(reason_code, "value", reason_code)) != 0:
            return
        client.subscribe("/readiness/#", qos=0)
        client.publish(
            "/replay/config",
            json.dumps(
                {
                    "action": "PROBE",
                    "scenario": scenario,
                    "start_time": 0,
                    "end_time": -1,
                    "playback_mode": "max",
                    "speed": 1,
                    "replay_id": f"probe-{scenario}-{uuid4().hex[:8]}",
                    "target_nodes": nodes,
                }
            ),
            qos=1,
            retain=True,
        )

    def on_message(_client, _userdata, message):
        try:
            document = json.loads(message.payload)
        except (TypeError, json.JSONDecodeError):
            return
        if (
            document.get("service") not in {"zed", "mobile"}
            or document.get("scenario") != scenario
            or not document.get("ready")
        ):
            return
        node = str(document.get("node") or document.get("node_id") or "")
        if node.startswith("dvpg_gq_orin_"):
            node = f"orin{node.removeprefix('dvpg_gq_orin_')}"
        if node in nodes:
            with lock:
                ready.add(node)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("127.0.0.1", 1883, keepalive=60)
    client.loop_start()
    deadline = time.monotonic() + timeout_seconds
    try:
        while time.monotonic() < deadline:
            with lock:
                if ready == set(nodes):
                    break
            time.sleep(0.1)
    finally:
        stop = json.dumps(
            {
                "action": "STOP",
                "replay_id": f"probe-{scenario}",
                "target_nodes": nodes,
            }
        )
        client.publish("/replay/config", stop, qos=1, retain=True).wait_for_publish(timeout=2.0)
        client.publish("/replay/sync", stop, qos=1, retain=True).wait_for_publish(timeout=2.0)
        time.sleep(2.0)
        client.publish("/replay/config", payload=None, qos=1, retain=True).wait_for_publish(timeout=2.0)
        client.publish("/replay/sync", payload=None, qos=1, retain=True).wait_for_publish(timeout=2.0)
        client.disconnect()
        client.loop_stop()
    with lock:
        return sorted(ready)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation/results/full_ce_all_examples_20260728",
    )
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--ready-seconds", type=float, default=45)
    parser.add_argument(
        "--required-ready-services",
        default="zed",
        help="Comma-separated replay/analytics services required before sync.",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help=(
            "Submit this many independent live requests against one shared "
            "replay generation (bounded physical E2 validation)."
        ),
    )
    parser.add_argument(
        "--baseline",
        choices=tuple(
            item.value
            for item in (
                BaselineId.B0_PRODUCE_ALL,
                BaselineId.B1_STATIC_WHOLE_EVENT,
                BaselineId.B2_FRONTIER_FIXED_REALIZATION,
                BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
                BaselineId.B4_GREEDY_FRONTIER,
                BaselineId.FABLE,
            )
        ),
        default=BaselineId.FABLE.value,
    )
    parser.add_argument(
        "--playback-mode",
        choices=("max", "realtime", "scaled"),
        default=DEFAULT_PLAYBACK_MODE,
    )
    parser.add_argument("--keep-stack", action="store_true")
    parser.add_argument(
        "--maximum-false-negatives",
        type=int,
        help=(
            "Stop the campaign as soon as this many false negatives have been "
            "exceeded (for example, 2 stops after the third false negative)."
        ),
    )
    parser.add_argument("--network-profile-id", default="good_network")
    parser.add_argument(
        "--network-disturbance",
        choices=("W1", "W2", "L1"),
        help="Apply an allowlisted in-replay network disturbance.",
    )
    parser.add_argument("--condition-trace", type=Path)
    parser.add_argument(
        "--allow-raw-to-trusted-site-edge",
        action="store_true",
        help=(
            "Use the same explicit trusted-site raw placement policy for "
            "nominal and disturbed cells."
        ),
    )
    parser.add_argument("--ce-start-offset-seconds", type=float, default=0.0)
    parser.add_argument(
        "--netwaggle-topology",
        type=Path,
        help=(
            "Attach matching generated sensor services to already-running "
            "NetWaggle anchors described by this topology JSON."
        ),
    )
    parser.add_argument(
        "--require-netwaggle-bindings",
        action="store_true",
        help=(
            "Fail closed unless every generated workload container is proven "
            "to share its declared NetWaggle anchor network namespace."
        ),
    )
    parser.add_argument(
        "--drop-offline-evidence",
        action="store_true",
        help=(
            "Pilot-only robustness mode: export sensor evidence at QoS 0 and "
            "disable QoS-0 bridge queuing so outage evidence must be recovered "
            "through explicit bounded raw replay; control stays at QoS 1."
        ),
    )
    parser.add_argument(
        "--close-live-evidence-at-replay-end",
        action="store_true",
        help=(
            "Close the ordinary live-result generation at natural replay EOF; "
            "only explicitly correlated outage-recovery demands may emit later."
        ),
    )
    parser.add_argument(
        "--evaluation-policy-id",
        default="",
        help="Experiment policy label when it differs from the executable planning baseline.",
    )
    parser.add_argument(
        "--replay-node",
        action="append",
        default=[],
        help="Restrict replay to these node IDs after readiness probing (repeatable).",
    )
    parser.add_argument(
        "--maximum-replay-nodes",
        type=int,
        help="Bound activated nodes after ranked candidates meet readiness results.",
    )
    parser.add_argument(
        "--deployment-config",
        type=Path,
        default=REPLAY_ROOT / "config/fable_deployment.yaml",
        help="Planner-visible deployment YAML mounted through replay/config.",
    )
    parser.add_argument(
        "--experiment-id",
        action="append",
        default=[],
        help="Run only the named experiment (repeatable).",
    )
    parser.add_argument(
        "--identity-judge-evidence-root",
        type=Path,
        help="Capture per-experiment identity evidence for offline E4 VLM judging.",
    )
    parser.add_argument(
        "--identity-escalation-policy-id",
        choices=(
            "C0_CHEAP_ONLY",
            "C1_STRONG_ONLY",
            "C2_FIXED_CASCADE",
            "C3_FABLE_ESCALATION",
            "C4_FABLE_NO_ESCALATION",
        ),
        help="Apply a controlled E4 policy to the live identity cascade.",
    )
    parser.add_argument(
        "--retrospective-policy-id",
        choices=("R0_NO_REPLAY", "R1_RAW_REPLAY", "R2_FABLE_TYPED_REPLAY"),
        help="Control historical evidence recovery for bounded E7 runs.",
    )
    parser.add_argument(
        "--mobile-root",
        type=Path,
        default=Path(os.environ.get("FABLE_MOBILE_ROOT", "/mnt/fable/mobile")),
        help="Android archive root; 2025 runs automatically use timestamp-only files.",
    )
    parser.add_argument(
        "--stage-physical-rpi",
        action="store_true",
        help="Stage one selected ZED recording in the physical Pi's managed replay slot.",
    )
    parser.add_argument("--physical-rpi-host", default="rpi")
    parser.add_argument("--physical-jetson-host", default="jetson")
    parser.add_argument(
        "--physical-rpi-address",
        default=os.environ.get("FABLE_PHYSICAL_RPI_ADDRESS", "rpi.example.invalid"),
        help="Pi data-plane address used by the host relay.",
    )
    parser.add_argument(
        "--physical-host-address",
        default=os.environ.get("FABLE_PHYSICAL_HOST_ADDRESS", "host.example.invalid"),
        help="Host address reachable from the physical Jetson.",
    )
    parser.add_argument(
        "--physical-rpi-root",
        default=os.environ.get("FABLE_PHYSICAL_RPI_ROOT", "/opt/fable"),
    )
    parser.add_argument(
        "--physical-jetson-root",
        default=os.environ.get("FABLE_PHYSICAL_JETSON_ROOT", "/opt/fable"),
    )
    parser.add_argument("--physical-rpi-identity-file", type=Path)
    parser.add_argument(
        "--physical-yolo-rate-hz",
        type=float,
        default=5.0,
        help=(
            "Physical Jetson inference cadence. Five Hz preserves ByteTrack "
            "continuity on the recorded vehicle motion while remaining "
            "sustainable in real time."
        ),
    )
    parser.add_argument("--physical-compute-planner-node-id")
    parser.add_argument("--physical-network-planner-node-id")
    parser.add_argument(
        "--physical-rpi-replay-node",
        help="Logical replay node assigned to the Pi; defaults to the first selected ZED node.",
    )
    parser.add_argument(
        "--physical-replay-data-root",
        action="append",
        default=[],
        type=Path,
        help="Host recording root for physical replay staging (repeatable).",
    )
    parser.add_argument(
        "--physical-replay-conversion-cache",
        type=Path,
        default=DEFAULT_CONVERSION_CACHE,
        help="SSD-backed cache for validated SVO/SVO2 left-eye MP4 exports.",
    )
    parser.add_argument(
        "--execute-physical-rpi",
        action="store_true",
        help=(
            "Replace the selected node's desktop ZED/YOLO services with the "
            "Pi->PC relay->Jetson CUDA execution path."
        ),
    )
    parser.add_argument(
        "--physical-netwaggle-proxies",
        action="store_true",
        help=(
            "Require the external Pi/Jetson NetWaggle proxies and route physical "
            "video/MQTT through them; never fall back to the direct socat relay."
        ),
    )
    args = parser.parse_args()
    if args.maximum_replay_nodes is not None and args.maximum_replay_nodes < 1:
        parser.error("--maximum-replay-nodes must be at least 1")
    if args.concurrent_requests not in {1, 2}:
        parser.error("--concurrent-requests currently supports only 1 or 2")
    if args.physical_yolo_rate_hz <= 0:
        parser.error("--physical-yolo-rate-hz must be positive")
    if args.maximum_false_negatives is not None and args.maximum_false_negatives < 0:
        parser.error("--maximum-false-negatives must be non-negative")
    if args.execute_physical_rpi and not args.stage_physical_rpi:
        parser.error("--execute-physical-rpi requires --stage-physical-rpi")
    if args.physical_netwaggle_proxies and not args.execute_physical_rpi:
        parser.error("--physical-netwaggle-proxies requires --execute-physical-rpi")
    deployment_config = args.deployment_config.resolve(strict=True)
    replay_config_root = (REPLAY_ROOT / "config").resolve(strict=True)
    try:
        deployment_relative = deployment_config.relative_to(replay_config_root)
    except ValueError:
        parser.error("--deployment-config must be inside iobt-minimal-ce-replay/config")
    os.environ["FABLE_DEPLOYMENT_CONFIG"] = (
        f"/workspace/replay/config/{deployment_relative.as_posix()}"
    )
    if args.identity_escalation_policy_id:
        os.environ["FABLE_IDENTITY_ESCALATION_POLICY"] = args.identity_escalation_policy_id
        os.environ["FABLE_VLM_REID_ENABLED"] = str(
            args.identity_escalation_policy_id
            in {"C1_STRONG_ONLY", "C2_FIXED_CASCADE", "C3_FABLE_ESCALATION"}
        ).lower()
    if args.retrospective_policy_id:
        os.environ["FABLE_RETROSPECTIVE_POLICY"] = args.retrospective_policy_id
    if args.close_live_evidence_at_replay_end:
        os.environ["FABLE_CLOSE_LIVE_EVIDENCE_AT_REPLAY_END"] = "true"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = ExperimentCatalog.from_csv(CATALOG_PATH).recommended()
    if args.experiment_id:
        selected_ids = set(args.experiment_id)
        catalog = tuple(
            experiment
            for experiment in catalog
            if experiment.experiment_id in selected_ids
        )
        missing_ids = selected_ids - {
            experiment.experiment_id for experiment in catalog
        }
        if missing_ids:
            parser.error(
                "unknown or non-recommended experiment IDs: "
                + ", ".join(sorted(missing_ids))
            )
    scenarios = {
        row["scenario_id"]: row
        for row in json.loads(SCENARIO_PATH.read_text())["scenarios"]
    }
    grouped: dict[str, list[object]] = {}
    gaps: list[dict[str, object]] = []
    for experiment in catalog:
        try:
            _, scenario = _resolve_experiment(experiment.experiment_id, replay_nodes=())
        except ValueError as exc:
            gaps.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "variant": experiment.ce_variant,
                    "classification": "COVERAGE_GAP",
                    "reason": str(exc),
                }
            )
            continue
        grouped.setdefault(scenario, []).append(experiment)

    plan = {
        "schema_version": "fable.full_ce_suite_plan.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": args.baseline,
        "network_profile_id": args.network_profile_id,
        "deployment_config": str(deployment_config),
        "total_recommended": len(catalog),
        "runnable": sum(map(len, grouped.values())),
        "coverage_gaps": gaps,
        "scenarios": {
            scenario: [item.experiment_id for item in experiments]
            for scenario, experiments in grouped.items()
        },
    }
    write_json(output_dir / "plan.json", plan)

    current_bundle: Path | None = None
    started = time.monotonic()
    false_negative_count = 0
    early_stop_reason: str | None = None
    try:
        for scenario, experiments in grouped.items():
            if early_stop_reason is not None:
                break
            scenario_complete = True
            for experiment in experiments:
                prior_path = output_dir / f"{experiment.experiment_id}.json"
                try:
                    prior = json.loads(prior_path.read_text())
                except (OSError, json.JSONDecodeError):
                    prior = {}
                if (
                    not prior.get("suite")
                    or prior.get("classification") == "RUNTIME_FAILURE"
                ):
                    scenario_complete = False
                    break
            if scenario_complete:
                continue
            row = scenarios[scenario]
            campaign_year = experiments[0].campaign_year
            nodes = candidate_zed_nodes(row, campaign_year)
            if args.replay_node:
                requested_nodes = {
                    item.strip().lower().replace("orin_", "orin").replace("orin-", "orin")
                    for item in args.replay_node
                }
                nodes = [node for node in nodes if node.lower() in requested_nodes]
            if not nodes:
                for experiment in experiments:
                    gaps.append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "variant": experiment.ce_variant,
                            "classification": "COVERAGE_GAP",
                            "reason": "no ZED-decodable replay node for scenario",
                        }
                    )
                continue

            bundle = ROOT / "runs/evaluation_bundles" / f"full-suite-{scenario}"
            generate = [
                sys.executable,
                str(REPLAY_ROOT / "setup/generate_evaluation_bundle.py"),
                "--scenario",
                scenario,
                "--output-dir",
                str(bundle),
                "--enable-reid",
            ]
            for node in nodes:
                generate.extend(["--node", node])
            if campaign_year in {2024, 2025}:
                generate.extend(
                    [
                        (
                            "--mobile-prefixed-any"
                            if campaign_year == 2024
                            else "--mobile-timestamp-only"
                        ),
                        "--mobile-optional",
                        "--mobile-root",
                        str(args.mobile_root),
                    ]
                )
            generated = run(generate, capture_output=True)
            if generated.returncode:
                for experiment in experiments:
                    gaps.append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "variant": experiment.ce_variant,
                            "classification": "INFRASTRUCTURE_FAILURE",
                            "reason": generated.stderr[-2000:],
                        }
                    )
                continue

            selected_registry = static_pipeline_registry_path().resolve()
            registry_override = bundle / "compose.static-registry.override.yaml"
            if os.environ.get("FABLE_STATIC_PIPELINE_REGISTRY"):
                registry_override.write_text(
                    yaml.safe_dump(
                        {
                            "services": {
                                "fable-orchestrator": {
                                    "environment": {
                                        "FABLE_STATIC_PIPELINE_REGISTRY": (
                                            "/workspace/fable-static-pipelines.yaml"
                                        )
                                    },
                                    "volumes": [
                                        f"{selected_registry}:/workspace/fable-static-pipelines.yaml:ro"
                                    ],
                                }
                            }
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
            else:
                registry_override.unlink(missing_ok=True)

            try:
                pinned_static_containers = pin_authored_static_provider_containers(
                    bundle,
                    baseline_id=args.baseline,
                    placement_id=experiments[0].ce_variant,
                    trace_id=scenario,
                )
            except RuntimeError as exc:
                for experiment in experiments:
                    gaps.append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "variant": experiment.ce_variant,
                            "classification": "INFRASTRUCTURE_FAILURE",
                            "reason": str(exc),
                        }
                    )
                continue

            netwaggle_bindings: dict[str, str] = {}
            netwaggle_override = bundle / "compose.netwaggle.override.yaml"
            if args.netwaggle_topology is not None:
                try:
                    netwaggle_bindings = write_netwaggle_override(
                        bundle,
                        args.netwaggle_topology.resolve(strict=True),
                        drop_offline_evidence=args.drop_offline_evidence,
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    for experiment in experiments:
                        gaps.append(
                            {
                                "experiment_id": experiment.experiment_id,
                                "variant": experiment.ce_variant,
                                "classification": "INFRASTRUCTURE_FAILURE",
                                "reason": f"NetWaggle binding failed: {exc}",
                            }
                        )
                    continue
            else:
                netwaggle_override.unlink(missing_ok=True)

            stop_stack(current_bundle)
            stack_started = time.monotonic()
            stack_log = output_dir / f"{scenario}.stack.log"
            with stack_log.open("w") as handle:
                up = create_stack_with_quiescent_providers(
                    bundle,
                    log_handle=handle,
                )
            missing_anchor = missing_netwaggle_anchor(stack_log) if up.returncode else None
            if missing_anchor is not None and netwaggle_override.is_file():
                # The live NetWaggle topology can be a host Mininet/OVS graph
                # without Docker namespace anchors.  Nominal/replay validation
                # must remain executable in that mode.  Do not mask unrelated
                # Compose errors, and preserve an explicit audit marker so an
                # RQ3 network-effect analysis cannot mistake this for shaped
                # container traffic.
                if args.require_netwaggle_bindings:
                    with stack_log.open("a") as handle:
                        handle.write(
                            "\nFABLE_NETWAGGLE_BINDING_REQUIRED: missing "
                            f"{missing_anchor}; refusing unshaped fallback\n"
                        )
                else:
                    netwaggle_override.unlink()
                    netwaggle_bindings = {}
                    with stack_log.open("a") as handle:
                        handle.write(
                            "\nFABLE_NETWAGGLE_ANCHOR_FALLBACK: missing "
                            f"{missing_anchor}; retrying without Docker namespace bindings\n"
                        )
                        up = create_stack_with_quiescent_providers(
                            bundle,
                            log_handle=handle,
                        )
            if up.returncode:
                for experiment in experiments:
                    gaps.append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "variant": experiment.ce_variant,
                            "classification": "INFRASTRUCTURE_FAILURE",
                            "reason": f"Compose up failed; see {stack_log}",
                        }
                    )
                current_bundle = bundle
                continue
            current_bundle = bundle
            netwaggle_binding_validation = validate_netwaggle_bindings(
                netwaggle_bindings
            ) if netwaggle_bindings else {
                "schema_version": "fable.netwaggle_binding_validation.v1",
                "binding_count": 0,
                "valid": False,
                "bindings": [],
            }
            if (
                args.require_netwaggle_bindings
                and not netwaggle_binding_validation["valid"]
            ):
                for experiment in experiments:
                    gaps.append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "variant": experiment.ce_variant,
                            "classification": "INFRASTRUCTURE_FAILURE",
                            "reason": "NetWaggle namespace binding validation failed",
                            "binding_validation": netwaggle_binding_validation,
                        }
                    )
                continue
            bundle_document = json.loads(
                (bundle / "bundle.json").read_text(encoding="utf-8")
            )
            mobile_nodes = [
                str(item) for item in bundle_document.get("mobile_nodes", ())
            ]
            if not wait_for_orchestrator():
                readiness_log = output_dir / f"{scenario}.orchestrator-readiness.log"
                inspected = run(
                    ["docker", "inspect", "fable-orchestrator", "--format", "{{json .State}}"],
                    capture_output=True,
                )
                orchestrator_logs = run(
                    ["docker", "logs", "fable-orchestrator"], capture_output=True
                )
                readiness_log.write_text(
                    "INSPECT\n"
                    + inspected.stdout
                    + inspected.stderr
                    + "\nLOGS\n"
                    + orchestrator_logs.stdout
                    + orchestrator_logs.stderr,
                    encoding="utf-8",
                )
                for experiment in experiments:
                    # Persist a retryable per-cell envelope.  Previously this
                    # diagnostic lived only in report.json, which a later
                    # trace-major invocation overwrote and left the campaign
                    # with an unexplained missing result.
                    write_json(
                        output_dir / f"{experiment.experiment_id}.json",
                        {
                            "schema_version": "fable.replay_accuracy_run.v2",
                            "experiment_id": experiment.experiment_id,
                            "variant": experiment.ce_variant,
                            "campaign_year": experiment.campaign_year,
                            "classification": "RUNTIME_FAILURE",
                            "error": "orchestrator did not become ready after Compose up",
                            "suite": {
                                "scenario": scenario,
                                "runner_returncode": 1,
                                "stack_log": str(stack_log),
                                "orchestrator_readiness_log": str(readiness_log),
                                "evaluation_policy_id": args.evaluation_policy_id
                                or args.baseline,
                                "execution_baseline_id": args.baseline,
                            },
                        },
                    )
                    gaps.append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "variant": experiment.ce_variant,
                            "classification": "INFRASTRUCTURE_FAILURE",
                            "reason": "orchestrator did not become ready after Compose up",
                        }
                    )
                continue
            playable_nodes = probe_playable_nodes(scenario, [*nodes, *mobile_nodes])
            if not playable_nodes:
                for experiment in experiments:
                    gaps.append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "variant": experiment.ce_variant,
                            "classification": "COVERAGE_GAP",
                            "reason": (
                                "no selected ZED stream became decodable during "
                                "the per-scenario readiness probe"
                            ),
                            "candidate_nodes": nodes,
                        }
                    )
                continue
            nodes, replay_fallback_used = select_playable_replay_nodes(
                playable_nodes,
                args.replay_node,
                args.maximum_replay_nodes,
            )
            physical_rpi_stage = None
            physical_rpi_node = None
            if args.stage_physical_rpi:
                assigned_node = args.physical_rpi_replay_node or (nodes[0] if nodes else None)
                if assigned_node is None or assigned_node not in nodes:
                    reason = (
                        "physical Pi replay node is not among the selected playable ZED nodes: "
                        f"assigned={assigned_node!r} selected={nodes!r}"
                    )
                    for experiment in experiments:
                        gaps.append(
                            {
                                "experiment_id": experiment.experiment_id,
                                "variant": experiment.ce_variant,
                                "classification": "INFRASTRUCTURE_FAILURE",
                                "reason": reason,
                            }
                        )
                    continue
                try:
                    source_video = resolve_scenario_video(
                        scenario_id=scenario,
                        replay_node=assigned_node,
                        data_roots=tuple(args.physical_replay_data_root)
                        or DEFAULT_DATA_ROOTS,
                        asset_kind="zed",
                    )
                    video, conversion = ensure_left_mp4(
                        source_video,
                        cache_root=args.physical_replay_conversion_cache,
                    )
                    stage_command = [
                        sys.executable,
                        str(ROOT / "scripts/stage_rpi_replay.py"),
                        str(video),
                        "--experiment-id",
                        experiments[0].experiment_id,
                        "--scenario-id",
                        scenario,
                        "--host",
                        args.physical_rpi_host,
                    ]
                    if args.physical_rpi_identity_file is not None:
                        stage_command.extend(
                            (
                                "--identity-file",
                                str(args.physical_rpi_identity_file.resolve(strict=True)),
                            )
                        )
                    staged = subprocess.run(
                        stage_command,
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    manifest_command = ["ssh", "-o", "BatchMode=yes"]
                    if args.physical_rpi_identity_file is not None:
                        manifest_command.extend(
                            ("-i", str(args.physical_rpi_identity_file.resolve(strict=True)))
                        )
                    manifest_command.extend(
                        (
                            args.physical_rpi_host,
                            f"cat {shlex.quote(args.physical_rpi_root.rstrip('/') + '/replay-cache/current.json')}",
                        )
                    )
                    manifest_result = subprocess.run(
                        manifest_command,
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    physical_rpi_stage = {
                        **json.loads(manifest_result.stdout),
                        "logical_replay_node": assigned_node,
                        "conversion": conversion,
                        "staging_stdout": staged.stdout.strip(),
                    }
                    physical_rpi_node = assigned_node
                except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                    reason = f"physical Pi replay staging failed: {exc}"
                    for experiment in experiments:
                        gaps.append(
                            {
                                "experiment_id": experiment.experiment_id,
                                "variant": experiment.ce_variant,
                                "classification": "INFRASTRUCTURE_FAILURE",
                                "reason": reason,
                            }
                        )
                    continue
            if args.replay_node:
                if not nodes:
                    for experiment in experiments:
                        gaps.append(
                            {
                                "experiment_id": experiment.experiment_id,
                                "variant": experiment.ce_variant,
                                "classification": "COVERAGE_GAP",
                                "reason": "spatial policy selected no playable replay node",
                                "requested_nodes": args.replay_node,
                                "playable_nodes": playable_nodes,
                            }
                        )
                    continue
            try:
                quiesced_provider_containers = quiesce_lease_controlled_providers(
                    bundle,
                    preserve=(
                        set(pinned_static_containers)
                        # This is an idle, secret-owning infrastructure proxy,
                        # not a per-demand model worker. The request budget and
                        # provider lease still govern actual VLM inference.
                        | {"fable-vlm-cloud1"}
                    ),
                )
            except RuntimeError as exc:
                for experiment in experiments:
                    gaps.append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "variant": experiment.ce_variant,
                            "classification": "INFRASTRUCTURE_FAILURE",
                            "reason": str(exc),
                        }
                    )
                continue
            startup_seconds = round(time.monotonic() - stack_started, 3)

            if args.execute_physical_rpi and physical_rpi_node is not None:
                stopped = run(
                    compose_command(
                        bundle,
                        "stop",
                        f"zed-{physical_rpi_node}",
                        f"yolo-{physical_rpi_node}",
                    ),
                    capture_output=True,
                )
                if stopped.returncode:
                    for experiment in experiments:
                        gaps.append(
                            {
                                "experiment_id": experiment.experiment_id,
                                "variant": experiment.ce_variant,
                                "classification": "INFRASTRUCTURE_FAILURE",
                                "reason": "could not suppress duplicate desktop replay services",
                                "stderr": stopped.stderr[-2000:],
                            }
                        )
                    continue

            for experiment in experiments:
                experiment_nodes = evaluation_nodes_for_variant(
                    nodes, experiment.ce_variant, experiment.campaign_year
                )
                node_text = ",".join(experiment_nodes)
                output = output_dir / f"{experiment.experiment_id}.json"
                if output.is_file():
                    try:
                        existing = json.loads(output.read_text())
                    except json.JSONDecodeError:
                        existing = {}
                    if (
                        existing.get("suite")
                        and existing.get("classification") != "RUNTIME_FAILURE"
                    ):
                        continue
                # A negative result cannot change after the selected recording
                # reaches EOF plus its evaluation deadline.  Keep enough
                # post-EOF time for a frontier exposed by the final observation
                # to cold-start and complete one bounded retrospective chain.
                # The previous 45-second allowance cut off the second identity
                # check in three-visit graphs only seconds after activation.
                scenario_duration = float(experiment.duration_seconds)
                run_max_seconds = min(
                    args.max_seconds,
                    max(
                        90.0,
                        scenario_duration
                        + 30.0
                        + 45.0,
                    ),
                )
                if args.retrospective_policy_id:
                    # E3 deliberately exposes historical work only after a
                    # trigger. Preserve a bounded 150-second completion window
                    # so a valid final frontier is not killed by the generic
                    # 75-second post-recording allowance.
                    run_max_seconds = min(
                        args.max_seconds,
                        max(run_max_seconds, scenario_duration + 150.0),
                    )
                late_recovery_budget = condition_recovery_budget_seconds(
                    args.condition_trace
                )
                if late_recovery_budget is not None:
                    run_max_seconds = min(
                        args.max_seconds,
                        max(run_max_seconds, late_recovery_budget),
                    )
                if args.playback_mode == "max":
                    run_max_seconds = min(run_max_seconds, 120.0)
                run_ready_seconds = min(
                    args.ready_seconds,
                    60.0,
                    run_max_seconds,
                )
                stdout_log = output.with_suffix(".stdout.log")
                stderr_log = output.with_suffix(".stderr.log")
                physical_execution = None
                if args.execute_physical_rpi and physical_rpi_node is not None:
                    try:
                        physical_execution = start_physical_rpi_execution(
                            args, logical_node=physical_rpi_node
                        )
                    except (OSError, subprocess.CalledProcessError) as exc:
                        write_json(
                            output,
                            {
                                "schema_version": "fable.replay_accuracy_run.v2",
                                "experiment_id": experiment.experiment_id,
                                "variant": experiment.ce_variant,
                                "classification": "RUNTIME_FAILURE",
                                "error": f"physical replay startup failed: {exc}",
                            },
                        )
                        continue
                command = [
                    sys.executable,
                    str(ROOT / "scripts/run_replay_accuracy.py"),
                    "--experiment-id",
                    experiment.experiment_id,
                    "--baseline",
                    args.baseline,
                    "--model-id",
                    "yolov8s.pt",
                    "--max-seconds",
                    str(run_max_seconds),
                    "--ready-seconds",
                    str(run_ready_seconds),
                    "--playback-mode",
                    args.playback_mode,
                    "--playback-speed",
                    str(DEFAULT_PLAYBACK_SPEED),
                    "--deadline-seconds",
                    "30",
                    "--replay-drain-seconds",
                    "5",
                    "--required-ready-services",
                    args.required_ready_services,
                    "--replay-nodes",
                    node_text,
                    "--output",
                    str(output),
                ]
                if "robbery" in experiment.ce_variant.lower():
                    command.extend(("--vehicle-recovery-timeout-seconds", "90"))
                if args.identity_judge_evidence_root is not None:
                    command.extend(
                        (
                            "--identity-judge-evidence-dir",
                            str(
                                args.identity_judge_evidence_root.resolve()
                                / experiment.experiment_id
                            ),
                        )
                    )
                    command.extend(
                        (
                            "--identity-judge-policy-id",
                            args.identity_escalation_policy_id
                            or args.evaluation_policy_id
                            or args.baseline,
                        )
                    )
                if args.network_disturbance:
                    command.extend(("--network-disturbance", args.network_disturbance))
                if args.condition_trace is not None:
                    command.extend(
                        (
                            "--condition-trace",
                            str(args.condition_trace.resolve()),
                            "--ce-start-offset-seconds",
                            str(args.ce_start_offset_seconds),
                        )
                    )
                    if args.physical_rpi_identity_file is not None:
                        command.extend((
                            "--physical-condition-identity-file",
                            str(args.physical_rpi_identity_file.resolve(strict=True)),
                        ))
                    if args.physical_compute_planner_node_id:
                        command.extend((
                            "--physical-compute-planner-node-id",
                            args.physical_compute_planner_node_id,
                        ))
                    if args.physical_network_planner_node_id:
                        command.extend((
                            "--physical-network-planner-node-id",
                            args.physical_network_planner_node_id,
                        ))
                if args.allow_raw_to_trusted_site_edge:
                    command.append("--allow-raw-to-trusted-site-edge")
                follower_command: list[str] | None = None
                follower_output = output.with_name(
                    f"{output.stem}.request-02{output.suffix}"
                )
                shared_replay_id = f"{scenario}-shared-{uuid4().hex[:8]}"
                if args.concurrent_requests == 2:
                    follower_command = command_without_options(
                        list(command),
                        {
                            "--condition-trace",
                            "--ce-start-offset-seconds",
                            "--physical-condition-identity-file",
                            "--physical-compute-planner-node-id",
                            "--physical-network-planner-node-id",
                            "--network-disturbance",
                        },
                    )
                    output_index = follower_command.index("--output") + 1
                    follower_command[output_index] = str(follower_output)
                    follower_command.extend(
                        (
                            "--replay-id",
                            shared_replay_id,
                            "--shared-replay-role",
                            "follower",
                        )
                    )
                    command.extend(
                        (
                            "--replay-id",
                            shared_replay_id,
                            "--shared-replay-role",
                            "owner",
                            "--shared-replay-owner-start-delay-seconds",
                            "8",
                            "--shared-replay-owner-stop-grace-seconds",
                            "8",
                        )
                    )
                    if args.baseline == BaselineId.FABLE.value:
                        command.append("--shared-replay-joint-admission-barrier")
                run_started = time.monotonic()
                runner_started_at = datetime.now(UTC)
                with stdout_log.open("w") as stdout, stderr_log.open("w") as stderr:
                    follower_process = None
                    follower_stdout = None
                    follower_stderr = None
                    try:
                        if follower_command is not None:
                            follower_stdout = follower_output.with_suffix(
                                ".stdout.log"
                            ).open("w")
                            follower_stderr = follower_output.with_suffix(
                                ".stderr.log"
                            ).open("w")
                            follower_process = subprocess.Popen(
                                follower_command,
                                cwd=ROOT,
                                text=True,
                                stdout=follower_stdout,
                                stderr=follower_stderr,
                            )
                            time.sleep(2.0)
                        completed = run(
                            command,
                            stdout=stdout,
                            stderr=stderr,
                            # Child setup/readiness has an independent budget;
                            # do not let the parent kill a valid experiment
                            # before its replay-time budget has elapsed.
                            timeout=run_max_seconds + run_ready_seconds + 30,
                        )
                        returncode = completed.returncode
                        hard_timeout = False
                        if follower_process is not None:
                            try:
                                follower_returncode = follower_process.wait(
                                    timeout=run_max_seconds + run_ready_seconds + 30
                                )
                            except subprocess.TimeoutExpired:
                                follower_process.terminate()
                                follower_returncode = 124
                            if follower_returncode != 0 and returncode == 0:
                                returncode = follower_returncode
                    except subprocess.TimeoutExpired:
                        returncode = 124
                        hard_timeout = True
                        if follower_process is not None:
                            follower_process.terminate()
                    finally:
                        if follower_stdout is not None:
                            follower_stdout.close()
                        if follower_stderr is not None:
                            follower_stderr.close()
                wall_seconds = round(time.monotonic() - run_started, 3)
                # Preserve the controller's planner/admission diagnostics before
                # the per-scenario stack is removed.  A zero-command admission
                # is otherwise indistinguishable in the result envelope from a
                # provider that simply emitted no matching observation.
                orchestrator_log = output.with_suffix(".orchestrator.log")
                orchestrator_logs = run(
                    compose_command(
                        bundle, "logs", "--no-color", "fable-orchestrator"
                    ),
                    capture_output=True,
                )
                orchestrator_log.write_text(
                    (orchestrator_logs.stdout or "")
                    + (orchestrator_logs.stderr or ""),
                    encoding="utf-8",
                )
                service_log = output.with_suffix(".services.log")
                service_logs = run(
                    compose_command(bundle, "logs", "--no-color"),
                    capture_output=True,
                )
                service_log.write_text(
                    (service_logs.stdout or "") + (service_logs.stderr or ""),
                    encoding="utf-8",
                )
                if output.is_file():
                    document = json.loads(output.read_text())
                else:
                    document = {
                        "schema_version": "fable.replay_accuracy_run.v2",
                        "experiment_id": experiment.experiment_id,
                        "variant": experiment.ce_variant,
                        "classification": "RUNTIME_FAILURE",
                        "error": "suite hard timeout"
                        if hard_timeout
                        else ("runner exited without a result"),
                    }
                follower_document = None
                if args.concurrent_requests == 2 and follower_output.is_file():
                    follower_document = json.loads(follower_output.read_text())
                conformance = document.get("execution_conformance") or {}
                if conformance.get("applicable") and not conformance.get("valid"):
                    # Conformance is an execution diagnostic, not the CE
                    # classifier.  Preserve the semantic outcome so missing
                    # processing cannot silently turn a false negative (or a
                    # true positive) into a different outcome class.
                    document["execution_conformance_warning"] = (
                        "selected provider placement did not match observed processing"
                    )
                static_lifecycle = inspect_static_provider_lifecycle(
                    pinned_static_containers,
                    runner_started_at=runner_started_at,
                    physical_execution=physical_execution,
                )
                document["static_provider_lifecycle"] = static_lifecycle
                if static_lifecycle.get("applicable") and not static_lifecycle.get("valid"):
                    document["static_provider_lifecycle_warning"] = (
                        "authored static providers were not resident for the full runner interval"
                    )
                document["suite"] = {
                    "scenario": scenario,
                    "replay_nodes": experiment_nodes,
                    "requested_replay_nodes": sorted(set(args.replay_node)),
                    "replay_availability_fallback_used": replay_fallback_used,
                    "maximum_replay_nodes": args.maximum_replay_nodes,
                    "physical_rpi_stage": physical_rpi_stage,
                    "physical_execution": physical_execution,
                    "identity_judge_evidence_dir": (
                        str(
                            args.identity_judge_evidence_root.resolve()
                            / experiment.experiment_id
                        )
                        if args.identity_judge_evidence_root is not None
                        else None
                    ),
                    "identity_escalation_policy_id": args.identity_escalation_policy_id,
                    "stack_startup_seconds": startup_seconds,
                    "lease_controlled_provider_containers": quiesced_provider_containers,
                    "pinned_static_provider_containers": pinned_static_containers,
                    "process_wall_seconds": wall_seconds,
                    "runner_returncode": returncode,
                    "hard_timeout": hard_timeout,
                    "network_profile_id": args.network_profile_id,
                    "netwaggle_topology": (
                        str(args.netwaggle_topology.resolve())
                        if args.netwaggle_topology is not None
                        else None
                    ),
                    "netwaggle_bindings": netwaggle_bindings,
                    "netwaggle_binding_validation": netwaggle_binding_validation,
                    "deployment_config": str(deployment_config),
                    "evaluation_policy_id": args.evaluation_policy_id or args.baseline,
                    "execution_baseline_id": args.baseline,
                    "concurrent_requests": args.concurrent_requests,
                    "shared_replay_id": (
                        shared_replay_id if args.concurrent_requests == 2 else None
                    ),
                    "secondary_request_result": (
                        str(follower_output) if follower_document is not None else None
                    ),
                    "secondary_request_classification": (
                        follower_document.get("classification")
                        if follower_document is not None else None
                    ),
                }
                common_record_dir = document.get("common_record_dir")
                if common_record_dir and Path(str(common_record_dir)).is_dir():
                    document["execution_change_timeline"] = write_execution_timeline(
                        document, Path(str(common_record_dir))
                    )
                write_json(output, document)
                print(
                    f"{experiment.campaign_year} | {experiment.ce_variant} | "
                    f"{experiment.experiment_id}: "
                    f"{document.get('classification')} ({wall_seconds}s)",
                    flush=True,
                )
                if document.get("classification") == "FALSE_NEGATIVE":
                    false_negative_count += 1
                    if (
                        args.maximum_false_negatives is not None
                        and false_negative_count > args.maximum_false_negatives
                    ):
                        early_stop_reason = (
                            f"false-negative safety limit exceeded: "
                            f"{false_negative_count} > "
                            f"{args.maximum_false_negatives}"
                        )
                        print(f"EARLY STOP: {early_stop_reason}", flush=True)
                        break
    finally:
        if not args.keep_stack:
            stop_stack(current_bundle)

    # ``output_dir`` is intentionally reusable across resumable/trace-major
    # invocations.  Summarize only the experiments selected for this
    # invocation; otherwise earlier results make ``len(results) !=
    # len(catalog)`` and falsely turn every later successful cell into a
    # failed suite.
    selected_experiment_ids = {item.experiment_id for item in catalog}
    results = []
    for path in output_dir.glob("*.json"):
        if path.name in {"plan.json", "report.json"}:
            continue
        if "-playable.json" in path.name or ".invalid-readiness.json" in path.name:
            continue
        try:
            result = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if result.get("experiment_id") in selected_experiment_ids:
            results.append(result)
    counts = Counter(str(item.get("classification")) for item in results)
    by_variant: dict[str, Counter[str]] = {}
    by_year: dict[int, Counter[str]] = {}
    for item in results:
        by_variant.setdefault(str(item.get("variant")), Counter()).update(
            [str(item.get("classification"))]
        )
        year = item.get("campaign_year")
        if isinstance(year, int):
            by_year.setdefault(year, Counter()).update(
                [str(item.get("classification"))]
            )
    report = {
        "schema_version": "fable.full_ce_suite_report.v1",
        "finished_at": datetime.now(UTC).isoformat(),
        "baseline": args.baseline,
        "evaluation_policy_id": args.evaluation_policy_id or args.baseline,
        "network_profile_id": args.network_profile_id,
        "deployment_config": str(deployment_config),
        "resume_invocation_wall_seconds": round(time.monotonic() - started, 3),
        "total_recommended": len(catalog),
        "completed_results": len(results),
        "classification_counts": counts,
        "by_variant": by_variant,
        "by_year": by_year,
        "coverage_gaps": gaps,
        "false_negative_count": false_negative_count,
        "maximum_false_negatives": args.maximum_false_negatives,
        "early_stopped": early_stop_reason is not None,
        "early_stop_reason": early_stop_reason,
        "resources_released": not args.keep_stack,
    }
    write_json(output_dir / "report.json", report)
    infrastructure_gaps = sum(
        gap.get("classification") == "INFRASTRUCTURE_FAILURE" for gap in gaps
    )
    incomplete = len(results) != len(catalog)
    return int(bool(infrastructure_gaps or incomplete))


if __name__ == "__main__":
    raise SystemExit(main())
