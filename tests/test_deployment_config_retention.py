from __future__ import annotations

from fable.distributed.config import load_deployment_graph


def test_load_deployment_graph_preserves_source_raw_buffer(tmp_path):
    config = tmp_path / "deployment.yaml"
    config.write_text(
        """
nodes:
  camera-node:
    node_class: sensor
    region: site
    capacity: {cpu_cores: 1, memory_mb: 512, gpu_memory_mb: 0}
sources:
  camera:
    node_id: camera-node
    region: site
    modalities: [vision]
    live_data_types: [raw_video_frames.v1]
    raw_buffer_interval:
      start: 2026-01-01T00:00:00Z
      end: 2026-01-01T00:01:00Z
links: []
""",
        encoding="utf-8",
    )

    source = load_deployment_graph(config).source("camera")
    assert source.raw_buffer_interval is not None
    assert (
        source.raw_buffer_interval.end - source.raw_buffer_interval.start
    ).total_seconds() == 60.0
