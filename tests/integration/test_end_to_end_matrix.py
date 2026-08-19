"""任务 10.2：双后端端到端恢复矩阵 —— 断言模型/step/指标/checksum/报告。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.sklearn_backend import (
    build_sklearn_experiment,
    run_sklearn_worker_experiment,
)
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import ExecutionPolicy, Platform


def _torch_cfg(run_id, experiment_cfg=None, selection="default", max_epochs=1):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    if experiment_cfg:
        schema["experiment"] = experiment_cfg
    if selection == "default":
        schema["selection"] = {"metric": "val/loss", "mode": "min",
                               "patience": 20, "min_delta": 0.0}
    elif selection == "none":
        schema["selection"] = None
    else:
        schema["selection"] = selection
    return parse_config(schema)


def _skl_cfg(run_id, fit_mode="batch", max_epochs=1):
    schema = default_schema()
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": fit_mode, "evaluation_batch_size": 4096,
                                     "n_jobs": None, "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": max_epochs, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 20, "min_delta": 0.0}
    schema["checkpoint"] = {"every_epochs": 1, "every_optimizer_steps": None,
                            "keep_last": 1}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    return parse_config(schema)


@pytest.mark.parametrize("ref,run_id,exp_cfg,sel", [
    ("experiments.toy_multiclass:build_experiment", "e2e-mc", None, "default"),
    ("experiments.toy_multilabel:build_experiment", "e2e-ml", None, "none"),
    ("experiments.toy_regression:build_experiment", "e2e-reg", None, "none"),
    ("experiments.toy_multi_input:build_experiment", "e2e-mi", None, "none"),
])
def test_torch_end_to_end(tmp_path, ref, run_id, exp_cfg, sel):
    cfg = _torch_cfg(run_id, experiment_cfg=exp_cfg, selection=sel)
    layout = RunLayout(str(tmp_path / "runs" / run_id))
    layout.ensure()
    result = run_worker(ref, cfg, layout, 0, 1, "none")
    assert result.status == "succeeded"
    assert result.global_step > 0
    # 模型 + manifest + checksum
    assert os.path.exists(layout.path("models", "last", "model.safetensors"))
    assert os.path.exists(layout.path("models", "last", "model-manifest.json"))
    from dl_helper.training.artifacts import sha256_file
    manifest = json.load(open(layout.path("models", "last", "model-manifest.json"), encoding="utf-8"))
    assert manifest["files"]["model.safetensors"]["sha256"] == sha256_file(
        layout.path("models", "last", "model.safetensors"))
    # 指标定义存在
    lines = [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]
    train = next(l for l in lines if l["stage"] == "train")
    assert train["metrics"]
    # summary
    summary = json.load(open(layout.summary_json, encoding="utf-8"))
    assert summary["backend"] == "torch"


def test_sklearn_batch_and_pipeline_matrix(tmp_path):
    for ref, run_id in (("experiments.sklearn_batch:build_experiment", "e2e-skl-b"),
                        ("experiments.sklearn_pipeline:build_experiment", "e2e-skl-p")):
        cfg = _skl_cfg(run_id)
        layout = RunLayout(str(tmp_path / "runs" / run_id))
        layout.ensure()
        exp = build_sklearn_experiment(ref, cfg.experiment)
        result = run_sklearn_worker_experiment(exp, cfg, Platform(), layout)
        assert result.status == "succeeded"
        manifest = json.load(open(layout.path("models", "best", "model-manifest.json"), encoding="utf-8"))
        assert manifest["format"] == "joblib"
        from dl_helper.training.artifacts import sha256_file
        assert manifest["files"]["model.joblib"]["sha256"] == sha256_file(
            layout.path("models", "best", "model.joblib"))


class _AdvancingClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls * 100.0


def test_sklearn_incremental_and_resume(tmp_path):
    run_dir = str(tmp_path / "runs" / "e2e-skl-incr")
    cfg1 = _skl_cfg("e2e-skl-incr", fit_mode="incremental", max_epochs=2)
    layout1 = RunLayout(run_dir)
    layout1.ensure()
    exp1 = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg1.experiment)
    r1 = run_sklearn_worker_experiment(
        exp1, cfg1, Platform(), layout1, resume="auto",
        budget_monotonic=_AdvancingClock(),
        execution_policy=ExecutionPolicy(platform="local", max_minutes=10, shutdown_grace_minutes=5),
    )
    assert r1.status == "preempted"

    cfg2 = _skl_cfg("e2e-skl-incr", fit_mode="incremental", max_epochs=3)
    layout2 = RunLayout(run_dir)
    layout2.ensure()
    exp2 = build_sklearn_experiment("experiments.sklearn_incremental:build_experiment", cfg2.experiment)
    r2 = run_sklearn_worker_experiment(exp2, cfg2, Platform(), layout2, resume="auto")
    assert r2.status == "succeeded"
    assert r2.epoch == 3
