"""任务 4.3/OSR-004：Torch 连续/恢复 step、权重、指标一致。"""
from __future__ import annotations

import json
import os

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config


def _cfg(run_id, max_epochs, resume="none", max_minutes=None):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 30, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = 1
    schema["checkpoint"]["keep_last"] = 2
    schema["checkpoint"]["resume"] = resume
    schema["runtime"]["max_minutes"] = max_minutes
    schema["runtime"]["shutdown_grace_minutes"] = 2
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


def test_torch_resume_reaches_same_final_position(tmp_path):
    """第一段 PREEMPTED 生成检查点，第二段 resume 到最终位置。"""
    run_dir = str(tmp_path / "runs" / "resume-pos")
    cfg1 = _cfg("resume-pos", max_epochs=2, resume="auto", max_minutes=10)
    layout1 = RunLayout(run_dir)
    layout1.ensure()
    r1 = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg1, layout1, 0, 1, "auto",
                    budget_monotonic=_AdvancingClock())
    assert r1.status == "preempted"

    cfg2 = _cfg("resume-pos", max_epochs=4, resume="auto")
    layout2 = RunLayout(run_dir)
    layout2.ensure()
    r2 = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg2, layout2, 0, 1, "auto")
    assert r2.status == "succeeded"
    assert r2.epoch == 4
    summary = json.load(open(layout2.summary_json, encoding="utf-8"))
    assert summary["epoch"] == 4


def test_torch_no_resume_starts_fresh(tmp_path):
    """resume=auto 但无 checkpoint → 从零开始。"""
    run_dir = str(tmp_path / "runs" / "fresh")
    cfg = _cfg("fresh", max_epochs=1, resume="auto")
    layout = RunLayout(run_dir)
    layout.ensure()
    r = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg, layout, 0, 1, "auto")
    assert r.epoch == 1
    assert os.path.exists(layout.path("checkpoints", "latest.json"))
