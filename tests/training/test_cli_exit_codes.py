"""任务 3.7：CLI 退出码 0/75/其他非零传播。"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import yaml

from dl_helper.training.config import default_schema
from dl_helper.training.launcher import launch_torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable


def _write_base_cfg(path, run_id, experiment="experiments.toy_multiclass:build_experiment", output_root=None):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["run"]["output_root"] = output_root
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["checkpoint"]["every_epochs"] = None
    path.write_text(yaml.safe_dump(schema, allow_unicode=True), encoding="utf-8")
    return schema


def _run_cli(args, cwd=REPO):
    return subprocess.run(
        [PY, "-m", "dl_helper.training.cli", *args],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def test_train_success_exit_zero(tmp_path):
    import uuid
    run_id = f"exit-0-{uuid.uuid4().hex[:8]}"
    _write_base_cfg(tmp_path / "base.yaml", run_id, output_root=str(tmp_path))
    proc = _run_cli(["train", "--config", str(tmp_path / "base.yaml"),
                     "--experiment", "experiments.toy_multiclass:build_experiment"])
    assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


def test_train_failure_exit_nonzero(tmp_path):
    _write_base_cfg(tmp_path / "base.yaml", "exit-1", output_root=str(tmp_path))
    # 使用不存在的实验引用
    proc = _run_cli(["train", "--config", str(tmp_path / "base.yaml"),
                     "--experiment", "nonexistent_module:build_experiment"])
    assert proc.returncode != 0
    assert proc.returncode != 75  # 不是预占


def test_bad_config_exit_nonzero(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("schema_version: 1\nbogus: 1\n", encoding="utf-8")
    proc = _run_cli(["train", "--config", str(cfg_path),
                     "--experiment", "experiments.toy_multiclass:build_experiment"])
    assert proc.returncode != 0


def test_launcher_preempted_exit_75(tmp_path):
    """worker 返回 preempted → launcher 返回 75。"""
    from dl_helper.training.backends.base import BackendResult

    def fake_worker(ref, config, layout, rank, world, resume, publish_terminal=True,
                    budget_monotonic=None, execution_policy=None):
        return BackendResult(status="preempted", epoch=1, global_step=5)

    from dl_helper.training.artifacts import RunLayout
    from dl_helper.training.config import parse_config

    schema = default_schema()
    schema["run"]["id"] = "exit-75"
    cfg = parse_config(schema)
    layout = RunLayout(str(tmp_path / "runs" / "exit-75"))
    layout.ensure()
    code = launch_torch("ref", cfg, layout.run_dir, 1, "none", worker_fn=fake_worker)
    assert code == 75
