"""任务 3.2：launcher 单/多进程启动测试（CPU gloo）。"""
from __future__ import annotations

import os

import pytest
import torch

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.launcher import launch_torch


def _cfg(num_processes, max_epochs=1):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["distributed"]["num_processes"] = num_processes
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = f"launcher-{num_processes}p"
    schema["checkpoint"]["every_epochs"] = None
    schema["checkpoint"]["keep_last"] = 1
    schema["backend"]["torch"]["mixed_precision"] = "no"
    return parse_config(schema)


def test_single_process_run(tmp_path):
    cfg = _cfg(1)
    layout = RunLayout(str(tmp_path / "runs" / "launcher-1p"))
    layout.ensure()
    code = launch_torch("experiments.toy_multiclass:build_experiment", cfg, layout.run_dir, 1, "none")
    assert code == 0
    assert os.path.exists(layout.summary_json)
    assert os.path.exists(layout.path("models", "last", "model.safetensors"))


@pytest.mark.skipif(os.name != "posix" and not os.environ.get("DLH_ALLOW_MP"), reason="spawn 多进程在部分平台不稳定")
def test_two_process_run(tmp_path, monkeypatch):
    """两进程 CPU gloo 训练（OSR-011：自屏蔽 CUDA，宿主是否多卡不改变行为）。"""
    # 屏蔽 CUDA：无论宿主 GPU 多少，本测试始终按 CPU gloo 两进程执行
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    cfg = _cfg(2, max_epochs=1)
    layout = RunLayout(str(tmp_path / "runs" / "launcher-2p"))
    layout.ensure()
    code = launch_torch("experiments.toy_multiclass:build_experiment", cfg, layout.run_dir, 2, "none")
    assert code == 0
    assert os.path.exists(layout.summary_json)
