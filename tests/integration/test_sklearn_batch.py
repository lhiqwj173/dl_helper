"""任务 3.5：sklearn batch worker 端到端。"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.sklearn_backend import (
    SklearnBackendError,
    build_sklearn_experiment,
    run_sklearn_worker_experiment,
)
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import Platform


def _skl_cfg(run_id, fit_mode="batch", max_epochs=1):
    schema = default_schema()
    schema["backend"] = {
        "type": "sklearn", "torch": None,
        "sklearn": {"fit_mode": fit_mode, "evaluation_batch_size": 4096, "n_jobs": None,
                    "random_state": "run_seed", "sample_weight_parameter": None},
    }
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": max_epochs, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 5, "min_delta": 0.0}
    schema["checkpoint"] = {"every_epochs": None, "every_optimizer_steps": None,
                            "keep_last": 1}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    return parse_config(schema)


def test_sklearn_batch_run(tmp_path):
    cfg = _skl_cfg("it-skl-batch")
    layout = RunLayout(str(tmp_path / "runs" / "it-skl-batch"))
    layout.ensure()
    exp = build_sklearn_experiment("experiments.sklearn_batch:build_experiment", cfg.experiment)
    result = run_sklearn_worker_experiment(exp, cfg, Platform(), layout)
    assert result.status == "succeeded"
    # best=last joblib
    assert os.path.exists(layout.path("models", "best", "model.joblib"))
    assert os.path.exists(layout.path("models", "last", "model.joblib"))
    summary = json.load(open(layout.summary_json, encoding="utf-8"))
    assert summary["backend"] == "sklearn"
    assert summary["model_artifact"]["format"] == "joblib"
    # 指标
    lines = [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]
    stages = {l["stage"] for l in lines}
    assert {"train", "val"} <= stages
    val = next(l for l in lines if l["stage"] == "val")
    assert "val/accuracy" in val["metrics"]


def test_sklearn_batch_sample_weight(tmp_path):
    # sample_weight_parameter 配置但数据无权重 → fit 前失败
    cfg = _skl_cfg("it-skl-sw")
    schema_backend = cfg.backend  # noqa
    # 重新构造带 sample_weight_parameter 的配置
    schema = default_schema()
    schema["backend"] = {
        "type": "sklearn", "torch": None,
        "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": None,
                    "random_state": "run_seed", "sample_weight_parameter": "sample_weight"},
    }
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 5, "min_delta": 0.0}
    schema["run"]["id"] = "it-skl-sw"
    cfg = parse_config(schema)
    layout = RunLayout(str(tmp_path / "runs" / "it-skl-sw"))
    layout.ensure()
    exp = build_sklearn_experiment("experiments.sklearn_batch:build_experiment", cfg.experiment)
    with pytest.raises(SklearnBackendError):
        run_sklearn_worker_experiment(exp, cfg, Platform(), layout)


def test_sklearn_batch_never_reads_checkpoint(tmp_path):
    # batch 从不读 checkpoint：配置可解析，直接 fit 成功且不产生任何检查点
    cfg = _skl_cfg("it-skl-budget")
    layout = RunLayout(str(tmp_path / "runs" / "it-skl-budget"))
    layout.ensure()
    exp = build_sklearn_experiment("experiments.sklearn_batch:build_experiment", cfg.experiment)
    result = run_sklearn_worker_experiment(exp, cfg, Platform(), layout)
    assert result.status == "succeeded"
    assert not os.path.exists(layout.path("checkpoints", "latest.json"))
    assert not os.path.isdir(layout.path("checkpoints"))