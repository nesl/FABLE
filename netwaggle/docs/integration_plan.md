# FABLE / NetWaggle integration plan

## Implemented now

1. Logical-node bundle abstraction for replay containers.
2. NetWaggle-aware compose generation using anchor containers with `network_mode: none`.
3. Docker namespace attachment to Mininet via veth pairs.
4. Host-side MQTT gateway at `10.255.0.1/16`.
5. Wired Mininet topology with TC-shaped switch-to-switch links.
6. Initial network-condition profiles.
7. Simple qdisc/interface metrics and MQTT payload-size tracing.

## Architecture

Local traffic remains local:

```text
ZED replay -> /tmp/zed.ipc -> YOLO detector
ReSpeaker replay -> /tmp/respeaker.ipc -> audio detector
```

Network-scoped traffic goes through MQTT and is shaped by Mininet:

```text
logical node -> Mininet/TC -> host gateway 10.255.0.1 -> host MQTT broker
```

## Missing next steps

1. Validate on the real desktop with Mininet/OVS/Docker privileges.
2. Confirm Docker Compose accepts `network_mode: service:<anchor>` with the GPU and privileged service options used by ZED/YOLO.
3. Add a small smoke-test service pair inside NetWaggle before running the full replay.
4. Decide whether `/replay/config` and `/replay/sync` should remain on the same shaped broker or move to a stable control broker.
5. Add explicit CE payload categories for artifacts, predicate results, and continuation payloads.
6. Add runtime profile changes without restarting Mininet, using `tc qdisc change`.
7. Add raw-stream-over-network as a separate baseline mode.
8. Re-enable wireless/mobility only after wired single-host evaluation is stable.
