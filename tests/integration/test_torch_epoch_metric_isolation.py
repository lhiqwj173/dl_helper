"""任务 OSR-010：epoch 指标隔离 —— 每 epoch 独立清零，不累积历史轮次。

旧实现 train/val MetricState 在全部 epoch 间复用且从不 reset，第二轮的
sample_count/loss 会包含上一轮数据；DDP 下已归约的历史状态还会被再次按 rank 求和。
"""
from __future__ import annotations

import json

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config


def _cfg(run_id, max_epochs):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 30, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    return parse_config(schema)


def _records(layout):
    return [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]


def test_epoch_metrics_isolated_sample_counts(tmp_path):
    """每 epoch 的 train/val sample_count 只反映本轮，不累积上一轮。"""
    run_dir = str(tmp_path / "runs" / "iso")
    layout = RunLayout(run_dir)
    layout.ensure()
    r = run_worker("experiments.toy_multiclass:build_experiment", _cfg("iso", 2), layout, 0, 1, "none")
    assert r.status == "succeeded"

    train_records = {m["epoch"]: m for m in _records(layout) if m["stage"] == "train"}
    val_records = {m["epoch"]: m for m in _records(layout) if m["stage"] == "val"}
    assert set(train_records) == {0, 1}
    assert set(val_records) == {0, 1}

    for ep in (0, 1):
        assert train_records[ep]["extended"]["train/sample_count"] == 128, f"train epoch {ep} 应只含一轮"
        assert val_records[ep]["extended"]["val/sample_count"] == 32, f"val epoch {ep} 应只含一轮"


def test_epoch_metrics_deterministic_first_epoch(tmp_path):
    """1 epoch 与 2 epoch 的首轮记录一致（确定性）；第二轮不再包含首轮。"""
    layout1 = RunLayout(str(tmp_path / "runs" / "iso-1"))
    layout1.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", _cfg("iso-1", 1), layout1, 0, 1, "none")

    layout2 = RunLayout(str(tmp_path / "runs" / "iso-2"))
    layout2.ensure()
    run_worker("experiments.toy_multiclass:build_experiment", _cfg("iso-2", 2), layout2, 0, 1, "none")

    rec1 = [m for m in _records(layout1) if m["stage"] == "train"][0]
    rec2_ep0 = [m for m in _records(layout2) if m["stage"] == "train" and m["epoch"] == 0][0]
    assert rec1["metrics"]["train/loss"] == rec2_ep0["metrics"]["train/loss"]
    assert rec1["extended"]["train/sample_count"] == rec2_ep0["extended"]["train/sample_count"]
    assert rec2_ep0["metrics"]["train/loss"] == rec2_ep0["metrics"]["train/loss"]
