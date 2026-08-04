"""任务 3.3：Torch 端到端训练（模型/指标/Artifact）。"""
from __future__ import annotations

import json
import os

import pytest
import torch

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config


def _torch_cfg(run_id, max_epochs=2, mid_resume_ok=False):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 20, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val", "test"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = 1
    schema["checkpoint"]["keep_last"] = 2
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


def test_torch_multiclass_training(tmp_path):
    cfg = _torch_cfg("it-torch-mc")
    layout = RunLayout(str(tmp_path / "runs" / "it-torch-mc"))
    layout.ensure()
    result = run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    assert result.status == "succeeded"
    assert result.epoch == cfg.training.max_epochs
    # 模型 Artifact
    assert os.path.exists(layout.path("models", "best", "model.safetensors"))
    assert os.path.exists(layout.path("models", "last", "model.safetensors"))
    # summary
    summary = json.load(open(layout.summary_json, encoding="utf-8"))
    assert summary["backend"] == "torch"
    assert summary["status"] == "succeeded"
    assert summary["selection"]["best_value"] is not None
    # metrics jsonl 有 train/val 记录
    lines = [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]
    stages = {l["stage"] for l in lines}
    assert {"train", "val"} <= stages
    train_metrics = next(l for l in lines if l["stage"] == "train")
    assert "train/loss" in train_metrics["metrics"]
    # 预测分片
    pred_dir = layout.path("predictions", "val")
    assert os.path.exists(os.path.join(pred_dir, "prediction-manifest.json"))
    # 检查点
    ckpts = [d for d in os.listdir(layout.path("checkpoints")) if d.startswith("epoch-")]
    assert len(ckpts) >= 1


def test_torch_resume_matches(tmp_path):
    """已完成 run 拒绝重跑改写（OSR-005 幂等性）。"""
    run_dir = str(tmp_path / "runs" / "it-torch-resume")
    cfg1 = _torch_cfg("it-torch-resume", max_epochs=2)
    layout1 = RunLayout(run_dir)
    layout1.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg1, layout1, 0, 1, "none")
    # 已完成 → 二次运行被拒绝
    cfg2 = _torch_cfg("it-torch-resume", max_epochs=4)
    layout2 = RunLayout(run_dir)
    layout2.ensure()
    with pytest.raises(Exception):
        run_worker("experiments.toy_multiclass:build_experiment", cfg2, layout2, 0, 1, "auto")
