"""任务 4.5：backend-aware run 离线报告。"""
from __future__ import annotations

import os

from dl_helper.training.artifacts import RunLayout, read_json
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.reporting import generate_run_report


def _torch_run(tmp_path, run_id, max_epochs=1):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    cfg = parse_config(schema)
    layout = RunLayout(str(tmp_path / "runs" / run_id))
    layout.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    return layout


def test_torch_multiclass_report(tmp_path):
    layout = _torch_run(tmp_path, "report-torch")
    index = generate_run_report(layout.run_dir)
    assert os.path.exists(index)
    content = open(index, encoding="utf-8").read()
    assert "Run Report: report-torch" in content
    assert "backend" in content
    # 指标表
    assert "val/accuracy" in content or "accuracy" in content
    # 有相对图片（混淆矩阵）或至少非空
    assets = layout.path("report", "assets")
    if os.path.exists(assets):
        assert any(os.listdir(assets))


def test_report_idempotent(tmp_path):
    layout = _torch_run(tmp_path, "report-idem")
    index1 = generate_run_report(layout.run_dir)
    content1 = open(index1, encoding="utf-8").read()
    index2 = generate_run_report(layout.run_dir)
    content2 = open(index2, encoding="utf-8").read()
    assert content1 == content2
    assert index1 == index2


def test_report_does_not_import_experiment(tmp_path):
    """报告在独立子进程中只读 Artifact，不导入 experiments。"""
    import json
    import subprocess
    import sys

    layout = _torch_run(tmp_path, "report-noimport")
    code = (
        "import sys, json\n"
        "from dl_helper.training.reporting import generate_run_report\n"
        "generate_run_report(%r)\n"
        "loaded = [m for m in sys.modules if m.startswith('experiments.')]\n"
        "assert not loaded, f'报告导入了实验: {loaded}'\n"
        "print('OK')\n"
    ) % layout.run_dir
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, encoding="utf-8", check=False)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
