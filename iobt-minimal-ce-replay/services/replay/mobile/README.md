# Mobile MP4 replay

Mobile recordings are selected by exact recording prefix and filename epoch.
Directory names identify physical archives only; they are not assumed to equal
the `n1`/`n2`/`n3` aliases drawn on an experiment map.

Generate a bundle with stable archive identities:

```bash
python setup/generate_evaluation_bundle.py \
  --scenario 20241008_101228 \
  --mobile-recording-prefix spatial_ce1_1 \
  --mobile-root "/media/brianw/Extreme SSD3"
```

If a replay scenario contains a much wider multi-node envelope than the labeled
event, pass the ground-truth site-local interval with `--mobile-event-start` and
`--mobile-event-end`. This prevents valid five-minute mobile chunks from being
rejected against an unrelated longer capture envelope.

For a run whose handset placement has been verified, copy
`config/mobile_alias_map.example.json`, correct its mapping, and add:

```bash
--mobile-alias-map /path/to/verified-run-map.json
```

The adapter publishes portrait MP4 frames through the existing local ZED frame
ABI. Existing YOLO, tracking, predicates, and ReID services therefore consume
mobile and Jetson camera data through the same validated interfaces.
