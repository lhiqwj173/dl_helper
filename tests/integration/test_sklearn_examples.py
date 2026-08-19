"""任务 8.2：sklearn batch/Pipeline/incremental 示例端到端。"""
from __future__ import annotations

import json
import os

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.sklearn_backend import (
    build_sklearn_experiment,
    run_sklearn_worker_experiment,
)
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import Platform


def _skl_cfg(run_id, fit_mode="batch", max_epochs=1):
    schema = default_schema()
    schema["backend"] = {"type": "sklearn", "torch": None,
                         "sklearn": {"fit_mode": fit_mode, "evaluation_batch_size": 4096,
                                     "n_jobs": None, "random_state": "run_seed",
                                     "sample_weight_parameter": None}}
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": max_epochs, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 5, "min_delta": 0.0}
    schema["checkpoint"] = {"every_epochs": 1, "every_optimizer_steps": None,
                            "keep_last": 1}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    return parse_config(schema)


def _run(tmp_path, ref, run_id, fit_mode="batch", max_epochs=1):
    cfg = _skl_cfg(run_id, fit_mode=fit_mode, max_epochs=max_epochs)
    layout = RunLayout(str(tmp_path / "runs" / run_id))
    layout.ensure()
    exp = build_sklearn_experiment(ref, cfg.experiment)
    result = run_sklearn_worker_experiment(exp, cfg, Platform(), layout)
    assert result.status == "succeeded"
    return layout


def test_sklearn_batch_example(tmp_path):
    layout = _run(tmp_path, "experiments.sklearn_batch:build_experiment", "ex-skl-batch")
    assert os.path.exists(layout.path("models", "best", "model.joblib"))
    manifest = json.load(open(layout.path("models", "best", "model-manifest.json"), encoding="utf-8"))
    assert manifest["format"] == "joblib"


def test_sklearn_pipeline_example(tmp_path):
    layout = _run(tmp_path, "experiments.sklearn_pipeline:build_experiment", "ex-skl-pipe")
    lines = [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]
    val = next(l for l in lines if l["stage"] == "val")
    assert "val/accuracy" in val["metrics"]


def test_sklearn_incremental_example(tmp_path):
    layout = _run(tmp_path, "experiments.sklearn_incremental:build_experiment", "ex-skl-incr",
                  fit_mode="incremental", max_epochs=2)
    assert os.path.exists(layout.path("models", "best", "model.joblib"))
    assert os.path.exists(layout.path("checkpoints", "latest.json"))