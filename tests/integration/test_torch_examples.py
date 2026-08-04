"""任务 8.1：无需网络的 PyTorch 通用示例端到端。"""
from __future__ import annotations

import json
import os

import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config


def _cfg(run_id, max_epochs=2, selection=None):
    schema = default_schema()
    schema["training"]["max_epochs"] = max_epochs
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    if selection is not None:
        schema["selection"] = selection
    else:
        schema["selection"] = None
    return parse_config(schema)


def _run(tmp_path, ref, run_id, max_epochs=2, selection=None):
    cfg = _cfg(run_id, max_epochs=max_epochs, selection=selection)
    layout = RunLayout(str(tmp_path / "runs" / run_id))
    layout.ensure()
    result = run_worker(ref, cfg, layout, 0, 1, "none")
    assert result.status == "succeeded"
    assert os.path.exists(layout.summary_json)
    assert os.path.exists(layout.path("models", "last", "model.safetensors"))
    return layout


def test_toy_multi_input(tmp_path):
    layout = _run(tmp_path, "experiments.toy_multi_input:build_experiment", "it-mi")
    summary = json.load(open(layout.summary_json, encoding="utf-8"))
    assert summary["backend"] == "torch"


def test_toy_multilabel(tmp_path):
    layout = _run(tmp_path, "experiments.toy_multilabel:build_experiment", "it-ml")
    lines = [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]
    train = next(l for l in lines if l["stage"] == "train")
    assert "train/subset_accuracy" in train["metrics"]


def test_toy_regression(tmp_path):
    layout = _run(tmp_path, "experiments.toy_regression:build_experiment", "it-reg")
    lines = [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]
    train = next(l for l in lines if l["stage"] == "train")
    assert "train/r2" in train["metrics"]


def test_toy_custom_task(tmp_path):
    layout = _run(tmp_path, "experiments.toy_custom_task:build_experiment", "it-custom",
                  selection={"metric": "val/r2", "mode": "max", "patience": 5, "min_delta": 0.0})
    lines = [json.loads(l) for l in open(layout.metrics_jsonl, encoding="utf-8")]
    train = next(l for l in lines if l["stage"] == "train")
    assert "train/mae" in train["metrics"]
    val = next(l for l in lines if l["stage"] == "val")
    assert "val/r2" in val["metrics"]
