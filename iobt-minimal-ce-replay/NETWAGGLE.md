# Running this replay stack with NetWaggle

NetWaggle lives as a sibling directory:

```text
FABLE/
├── iobt-minimal-ce-replay/
└── netwaggle/
```

The integration keeps raw local replay streams local and routes network-scoped
MQTT traffic through Mininet:

```text
ZED replay -> /tmp/zed.ipc -> YOLO detector       # local
ReSpeaker replay -> /tmp/respeaker.ipc -> audio   # local
replay/detection/CE MQTT topics -> Mininet -> host MQTT broker
```

## Typical run sequence

```bash
# From FABLE/iobt-minimal-ce-replay
python3 setup/generate_replay_compose.py --device orin11 --compose-out compose.replay.yaml
./tools/make_netwaggle_compose.sh

docker compose -f compose.server.yaml up --build
# second terminal
docker compose -f compose.netwaggle.yaml up --build
# third terminal
cd ../netwaggle
sudo PYTHONPATH=. python3 -m netwaggle.runner \
  --topology configs/fable_single_host.json \
  --profile configs/profiles/good_network.json \
  --no-cli --hold
```

The generated `compose.netwaggle.yaml` changes replay service `MQTT_HOST_IP` to
`10.255.0.1` by default. NetWaggle creates this host-side gateway interface and
connects it to the Mininet topology.

## Notes

- `compose.server.yaml` is unchanged. The MQTT broker and web UI remain stable
  host-side services for the first integration.
- `compose.replay.yaml` remains the normal non-NetWaggle compose file.
- `compose.netwaggle.yaml` is generated and can be deleted/recreated.
- If you include multiple devices, make sure `../netwaggle/configs/fable_single_host.json`
  includes matching logical-node mappings.

## Evaluation wrappers

The easiest way to bring the integration up is from the repository root:

```bash
cd FABLE
./evaluation/netwaggle/start_stack.sh --profile smoke_fixed_latency --anchors-only
./evaluation/netwaggle/smoke_fixed_latency.sh
```

This starts MQTT/UI, NetWaggle anchor containers, and Mininet, then publishes
synthetic MQTT messages through the `orin11` logical node. The messages appear
in the existing web UI at `http://localhost:8080` under debug YOLO/audio status,
CE output, and replay status topics.

For a full replay run:

```bash
./evaluation/netwaggle/start_stack.sh --profile good_network --regenerate-compose
./evaluation/netwaggle/list_scenarios.sh
./evaluation/netwaggle/run_replay_window.sh --scenario <SCENARIO_ID> --start 0 --end 30
```

Stop everything with:

```bash
./evaluation/netwaggle/stop_stack.sh
```
