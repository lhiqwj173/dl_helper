"""任务 OSR-004：Torch auto 恢复遇到损坏/漂移 latest 必须失败。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.checkpoint import CheckpointError, read_latest
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import ExecutionPolicy

_BUDGET = ExecutionPolicy(platform="local", max_minutes=10.0, shutdown_grace_minutes=2.0)


def _cfg(run_id, max_epochs=1):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 30, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = 1
    schema["checkpoint"]["every_optimizer_steps"] = None
    schema["checkpoint"]["keep_last"] = 5
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["backend"]["torch"]["deterministic"] = "off"
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


class _AdvancingClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls * 100.0


def _make_checkpoint(tmp_path, run_id):
    cfg = _cfg(run_id)
    layout = RunLayout(str(tmp_path / "runs" / run_id))
    layout.ensure()
    # 预算预占：产生 checkpoint + pause 终态，无 success 终态
    run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg, layout, 0, 1, "none",
               budget_monotonic=_AdvancingClock(), execution_policy=_BUDGET)
    return layout


def test_corrupt_latest_engine_state_fails(tmp_path):
    layout = _make_checkpoint(tmp_path, "corrupt-eng")
    latest = read_latest(layout.path("checkpoints"))
    ckpt_dir = os.path.join(layout.path("checkpoints"), latest["path"])
    with open(os.path.join(ckpt_dir, "engine-state.json"), "w", encoding="utf-8") as f:
        f.write("{corrupt")
    with pytest.raises(CheckpointError):
        run_worker("experiments.toy_multiclass_resumable:build_experiment", _cfg("corrupt-eng"),
                   layout, 0, 1, "auto")


def test_missing_checkpoint_file_fails(tmp_path):
    layout = _make_checkpoint(tmp_path, "corrupt-missing")
    latest = read_latest(layout.path("checkpoints"))
    ckpt_dir = os.path.join(layout.path("checkpoints"), latest["path"])
    os.remove(os.path.join(ckpt_dir, "datamodule-state.json"))
    with pytest.raises(CheckpointError):
        run_worker("experiments.toy_multiclass_resumable:build_experiment", _cfg("corrupt-missing"),
                   layout, 0, 1, "auto")


def test_fingerprint_drift_fails(tmp_path):
    layout = _make_checkpoint(tmp_path, "drift-fp")
    # 改变训练参数使 resume fingerprint 变化（log_every_steps 不在允许变化列表）
    cfg = _cfg("drift-fp")
    from dataclasses import replace
    cfg = replace(cfg, training=replace(cfg.training, log_every_steps=99))
    with pytest.raises(CheckpointError):
        run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg, layout, 0, 1, "auto")


def test_required_with_no_checkpoint_fails(tmp_path):
    layout = RunLayout(str(tmp_path / "runs" / "required-none"))
    layout.ensure()
    with pytest.raises(CheckpointError):
        run_worker("experiments.toy_multiclass_resumable:build_experiment", _cfg("required-none"),
                   layout, 0, 1, "required")
