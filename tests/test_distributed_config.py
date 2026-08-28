from __future__ import annotations

from fable.distributed.config import load_deployment_graph


def test_deployment_loader_preserves_source_raw_buffer_interval(tmp_path) -> None:
    path = tmp_path / "deployment.yaml"
    path.write_text(
        """
nodes:
  sensor:
    node_class: sensor
    region: test
    capacity: {cpu_cores: 1, memory_mb: 128, gpu_memory_mb: 0}
sources:
  camera:
    node_id: sensor
    region: test
    modalities: [vision]
    live_data_types: [raw_video_frames.v1]
    raw_buffer_interval:
      start: 2024-10-08T11:08:23Z
      end: 2024-10-08T11:09:23Z
""".strip()
    )

    deployment = load_deployment_graph(path)

    interval = deployment.sources["camera"].raw_buffer_interval
    assert interval is not None
    assert interval.duration.total_seconds() == 60
