from __future__ import annotations

import importlib.util
from pathlib import Path


def _worker_module():
    path = Path(__file__).parents[1] / "iobt-minimal-ce-replay/tools/fable_e4_worker.py"
    spec = importlib.util.spec_from_file_location("fable_e4_worker_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e4_worker_normalizes_replay_orin_node_names():
    normalize = _worker_module()._deployment_node_id
    assert normalize("orin11") == "dvpg_gq_orin_11"
    assert normalize("dvpg_gq_orin_11") == "dvpg_gq_orin_11"
    assert normalize("x86server") == "x86server"
