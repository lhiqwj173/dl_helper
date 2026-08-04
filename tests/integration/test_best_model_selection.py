"""任务 OSR-007：best 模型只在 selection 改善时快照，最终 best 非最后模型。"""
from __future__ import annotations

import hashlib
import json
import os

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config


def _cfg(run_id, max_epochs=5, patience=20, lr=0.5):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["experiment"] = {"lr": lr}
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": patience, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_best_not_equal_last_when_overfits(tmp_path):
    """高学习率使 val loss 先降后升；best 应停在最佳 epoch，best != last。"""
    cfg = _cfg("best-sel", max_epochs=6, patience=30, lr=1.5)
    layout = RunLayout(str(tmp_path / "runs" / "best-sel"))
    layout.ensure()
    result = run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    assert result.status == "succeeded"
    best_path = layout.path("models", "best", "model.safetensors")
    last_path = layout.path("models", "last", "model.safetensors")
    # 过拟合场景：best 在中间 epoch，best != last（若 best=last 则说明 bug 复发）
    assert _sha(best_path) != _sha(last_path)
    summary = json.load(open(layout.summary_json, encoding="utf-8"))
    assert summary["selection"]["best_epoch"] < result.epoch  # best 早于最终 epoch


def test_best_improvement_snapshotted(tmp_path):
    """selection 改善时快照；best_value 来自选择指标。"""
    cfg = _cfg("best-imp", max_epochs=3, patience=20, lr=0.1)
    layout = RunLayout(str(tmp_path / "runs" / "best-imp"))
    layout.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", cfg, layout, 0, 1, "none")
    summary = json.load(open(layout.summary_json, encoding="utf-8"))
    assert summary["selection"]["best_value"] is not None
    assert summary["selection"]["best_epoch"] is not None
    assert summary["selection"]["best_step"] is not None
    # best 快照存在
    assert os.path.exists(layout.path("models", "best", "model.safetensors"))


class _Clock:
    def __init__(self, mult):
        self.calls = 0
        self._mult = mult

    def __call__(self):
        self.calls += 1
        return self.calls * self._mult


def test_best_survives_pause_resume_without_improvement(tmp_path):
    """OSR-007：暂停检查点保存 best 权重；恢复后不再改善时导出的 best 与恢复前一致。"""
    import torch
    import safetensors.torch

    # 高学习率过拟合：best 在早期 epoch；预算在中途预占
    from dataclasses import replace

    run_dir = str(tmp_path / "runs" / "best-resume")
    cfg1 = _cfg("best-resume", max_epochs=6, patience=30, lr=1.5)
    cfg1 = replace(cfg1, checkpoint=replace(cfg1.checkpoint, every_optimizer_steps=4),
                   runtime=replace(cfg1.runtime, max_minutes=10, shutdown_grace_minutes=2))
    layout = RunLayout(run_dir)
    layout.ensure()
    r1 = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg1, layout, 0, 1, "auto",
                    budget_monotonic=_Clock(40))
    assert r1.status == "preempted"

    # 预算检查点保存了非空 best 权重
    from dl_helper.training.checkpoint import read_latest
    latest = read_latest(layout.path("checkpoints"))
    ckpt_dir = os.path.join(layout.path("checkpoints"), latest["path"])
    best_ckpt_path = os.path.join(ckpt_dir, "best-model-state.pt")
    assert os.path.exists(best_ckpt_path), "检查点必须保存 best 权重"
    best_ckpt = torch.load(best_ckpt_path, weights_only=True, map_location="cpu")

    # 恢复并训练至完成（过拟合持续，不再改善）
    cfg2 = _cfg("best-resume", max_epochs=6, patience=30, lr=1.5)
    cfg2 = replace(cfg2, checkpoint=replace(cfg2.checkpoint, every_optimizer_steps=4))
    r2 = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg2, layout, 0, 1, "auto")
    assert r2.status == "succeeded"

    # 导出的 best == 检查点中的 best 权重（恢复后未改善不覆盖）
    exported = safetensors.torch.load_file(layout.path("models", "best", "model.safetensors"))
    assert set(exported) == set(best_ckpt)
    for k in exported:
        assert exported[k].shape == best_ckpt[k].shape
        assert bool((exported[k] == best_ckpt[k]).all()), f"恢复后 best 被覆盖: {k}"
