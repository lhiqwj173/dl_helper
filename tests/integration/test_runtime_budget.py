"""任务 7.3：monotonic 预算下两 backend 暂停恢复、终态互斥。"""
from __future__ import annotations

import json
import os

from dl_helper.training.artifacts import RunLayout, existing_terminal
from dl_helper.training.backends.sklearn_backend import (
    build_sklearn_experiment,
    run_sklearn_worker_experiment,
)
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import Platform


class _AdvancingClock:
    """每次调用返回递增时间，使预算在首个 step 检查即命中。"""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls * 100.0


def _torch_cfg(run_id, resume="none", max_epochs=5):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 20, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = 1
    schema["checkpoint"]["keep_last"] = 5
    schema["checkpoint"]["resume"] = resume
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["runtime"]["max_minutes"] = 10
    schema["runtime"]["shutdown_grace_minutes"] = 2
    return parse_config(schema)


def test_torch_budget_preempts_and_resumes(tmp_path):
    run_dir = str(tmp_path / "runs" / "budget-torch")
    cfg1 = _torch_cfg("budget-torch", resume="auto", max_epochs=5)
    layout1 = RunLayout(run_dir)
    layout1.ensure()
    r1 = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg1, layout1, 0, 1, "auto",
                    budget_monotonic=_AdvancingClock())
    assert r1.status == "preempted"
    # pause manifest 存在且互斥
    assert existing_terminal(run_dir) == "pause-manifest.json"
    pause = json.load(open(os.path.join(run_dir, "pause-manifest.json"), encoding="utf-8"))
    assert pause["status"] == "preempted"
    assert pause["resume_checkpoint"]
    # OSR-005：preempted 时 summary 状态一致（非 succeeded）
    summary = json.load(open(layout1.summary_json, encoding="utf-8"))
    assert summary["status"] == "preempted"

    # resume：max_epochs 允许增大；用新时钟让预算不再命中
    cfg2 = _torch_cfg("budget-torch", resume="auto", max_epochs=8)
    layout2 = RunLayout(run_dir)
    layout2.ensure()
    r2 = run_worker("experiments.toy_multiclass_resumable:build_experiment", cfg2, layout2, 0, 1, "auto")
    assert r2.status == "succeeded"
    assert existing_terminal(run_dir) == "run-manifest.json"


def _skl_cfg(run_id, resume="none", max_epochs=5):
    schema = default_schema()
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": "incremental", "evaluation_batch_size": 4096,
                                     "n_jobs": None, "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": max_epochs, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 20, "min_delta": 0.0}
    schema["checkpoint"]["every_epochs"] = 1
    schema["checkpoint"]["keep_last"] = 5
    schema["checkpoint"]["resume"] = resume
    schema["runtime"]["max_minutes"] = 10
    schema["runtime"]["shutdown_grace_minutes"] = 2
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    return parse_config(schema)


def test_sklearn_incremental_budget_preempts(tmp_path):
    run_dir = str(tmp_path / "runs" / "budget-skl")
    cfg = _skl_cfg("budget-skl", resume="auto", max_epochs=5)
    layout = RunLayout(run_dir)
    layout.ensure()
    exp = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg.experiment)
    r = run_sklearn_worker_experiment(exp, cfg, Platform(), layout, budget_monotonic=_AdvancingClock())
    assert r.status == "preempted"
    assert existing_terminal(run_dir) == "pause-manifest.json"
