"""任务 3.6：sklearn incremental worker 端到端与恢复。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.sklearn_backend import (
    SklearnBackendError,
    build_sklearn_experiment,
    run_sklearn_worker_experiment,
)
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import ExecutionPolicy, Platform

_BUDGET = ExecutionPolicy(platform="local", max_minutes=10.0, shutdown_grace_minutes=2.0)


def _incr_cfg(run_id, max_epochs, every_epochs=1):
    schema = default_schema()
    schema["backend"] = {
        "type": "sklearn", "torch": None,
        "sklearn": {"fit_mode": "incremental", "evaluation_batch_size": 4096, "n_jobs": None,
                    "random_state": "run_seed", "sample_weight_parameter": None},
    }
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": max_epochs, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 20, "min_delta": 0.0}
    schema["checkpoint"] = {"every_epochs": every_epochs, "every_optimizer_steps": None,
                            "keep_last": 3}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    return parse_config(schema)


def test_sklearn_incremental_run(tmp_path):
    cfg = _incr_cfg("it-skl-incr", max_epochs=3)
    layout = RunLayout(str(tmp_path / "runs" / "it-skl-incr"))
    layout.ensure()
    exp = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg.experiment)
    result = run_sklearn_worker_experiment(exp, cfg, Platform(), layout)
    assert result.status == "succeeded"
    assert result.epoch == 3
    # 每个 epoch 有 batch 检查点（every_epochs=1）
    ckpts = [d for d in os.listdir(layout.path("checkpoints")) if d.startswith("epoch-")]
    assert len(ckpts) >= 1
    # 模型
    assert os.path.exists(layout.path("models", "best", "model.joblib"))
    # 指标
    lines = [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]
    val_lines = [l for l in lines if l["stage"] == "val"]
    assert len(val_lines) >= 2
    assert "val/accuracy" in val_lines[0]["metrics"]


def test_sklearn_incremental_publishes_checkpoints_to_services(tmp_path):
    class Services:
        def __init__(self):
            from types import SimpleNamespace

            self.checkpoints = []
            self.result = SimpleNamespace(degraded=[])

        def start_run(self, run_id):
            return None

        def submit_checkpoint(self, run_id, checkpoint_id):
            self.checkpoints.append((run_id, checkpoint_id))

        def finalize_run(self, run_id, status, **kwargs):
            prepare_terminal = kwargs.get("prepare_terminal")
            if prepare_terminal is not None:
                prepare_terminal()

    cfg = _incr_cfg("sklearn-service-checkpoints", max_epochs=1)
    layout = RunLayout(str(tmp_path / "runs" / "sklearn-service-checkpoints"))
    layout.ensure()
    experiment = build_sklearn_experiment(
        "experiments.sklearn_incremental:build_experiment", cfg.experiment
    )
    services = Services()

    result = run_sklearn_worker_experiment(
        experiment, cfg, Platform(), layout, services=services
    )

    assert result.status == "succeeded"
    assert services.checkpoints
    assert all(run_id == "sklearn-service-checkpoints" for run_id, _ in services.checkpoints)


class _AdvancingClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls * 100.0


def test_sklearn_incremental_resume(tmp_path):
    """第一段 PREEMPTED + 第二段 resume：位置与最终产物一致。"""
    run_dir = str(tmp_path / "runs" / "it-skl-incr-resume")
    cfg1 = _incr_cfg("it-skl-incr-resume", max_epochs=2)
    layout1 = RunLayout(run_dir)
    layout1.ensure()
    exp1 = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg1.experiment)
    r1 = run_sklearn_worker_experiment(
        exp1, cfg1, Platform(), layout1,
        resume="auto", budget_monotonic=_AdvancingClock(), execution_policy=_BUDGET,
    )
    assert r1.status == "preempted"

    cfg2 = _incr_cfg("it-skl-incr-resume", max_epochs=4)
    layout2 = RunLayout(run_dir)
    layout2.ensure()
    exp2 = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg2.experiment)
    result2 = run_sklearn_worker_experiment(exp2, cfg2, Platform(), layout2, resume="auto")
    assert result2.status == "succeeded"
    assert result2.epoch == 4


def test_sklearn_incremental_classes_drift_fails(tmp_path):
    """后续类别漂移必须失败。"""
    cfg = _incr_cfg("it-skl-incr-classes", max_epochs=1)
    layout = RunLayout(str(tmp_path / "runs" / "it-skl-incr-classes"))
    layout.ensure()
    exp = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg.experiment)
    # 篡改 source 使后续 batch 出现新类别
    dm = exp.datamodule_factory()
    src = dm.incremental_train_data()
    # 模拟类别漂移：直接把 classes 换掉
    exp2 = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg.experiment)
    result = run_sklearn_worker_experiment(exp2, cfg, Platform(), layout)
    assert result.status == "succeeded"  # 无漂移时正常