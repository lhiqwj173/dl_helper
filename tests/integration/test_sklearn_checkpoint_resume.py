"""任务 4.4：可信 sklearn checkpoint 恢复与外部来源拒绝。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout, read_json
from dl_helper.training.backends.sklearn_backend import (
    build_sklearn_experiment,
    run_sklearn_worker_experiment,
)
from dl_helper.training.checkpoint import CheckpointError, read_latest
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import Platform


def _incr_cfg(run_id, max_epochs, resume="none", max_minutes=None):
    schema = default_schema()
    schema["backend"] = {
        "type": "sklearn", "torch": None,
        "sklearn": {"fit_mode": "incremental", "evaluation_batch_size": 4096, "n_jobs": None,
                    "random_state": "run_seed", "sample_weight_parameter": None},
    }
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": max_epochs, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 20, "min_delta": 0.0}
    schema["runtime"] = {"max_minutes": max_minutes, "shutdown_grace_minutes": 5}
    schema["checkpoint"] = {"every_epochs": 1, "every_optimizer_steps": None,
                            "keep_last": 3, "resume": resume}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    return parse_config(schema)


class _AdvancingClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls * 100.0


def test_sklearn_checkpoint_resume(tmp_path):
    run_dir = str(tmp_path / "runs" / "skl-resume")
    cfg1 = _incr_cfg("skl-resume", max_epochs=2, resume="auto", max_minutes=10)
    layout1 = RunLayout(run_dir)
    layout1.ensure()
    exp1 = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg1.experiment)
    r1 = run_sklearn_worker_experiment(exp1, cfg1, Platform(), layout1, budget_monotonic=_AdvancingClock())
    assert r1.status == "preempted"
    latest = read_latest(layout1.path("checkpoints"))
    assert latest is not None

    cfg2 = _incr_cfg("skl-resume", max_epochs=4, resume="auto")
    layout2 = RunLayout(run_dir)
    layout2.ensure()
    exp2 = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg2.experiment)
    r2 = run_sklearn_worker_experiment(exp2, cfg2, Platform(), layout2)
    assert r2.epoch == 4


def test_sklearn_tampered_checkpoint_rejected(tmp_path):
    run_dir = str(tmp_path / "runs" / "skl-tamper")
    cfg = _incr_cfg("skl-tamper", max_epochs=2, resume="auto", max_minutes=10)
    layout = RunLayout(run_dir)
    layout.ensure()
    exp = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg.experiment)
    run_sklearn_worker_experiment(exp, cfg, Platform(), layout, budget_monotonic=_AdvancingClock())
    # 篡改 latest 指向的 estimator.joblib
    latest = read_latest(layout.path("checkpoints"))
    ckpt_dir = os.path.join(layout.path("checkpoints"), latest["path"])
    with open(os.path.join(ckpt_dir, "estimator.joblib"), "wb") as f:
        f.write(b"tampered")
    # 第二次 resume 必须拒绝
    exp2 = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg.experiment)
    with pytest.raises(CheckpointError):
        run_sklearn_worker_experiment(exp2, cfg, Platform(), layout)
