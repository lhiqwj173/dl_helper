"""任务 4.4：sklearn joblib 持久化安全（manifest、batch 不创建 latest、外部拒绝）。"""
from __future__ import annotations

import os

import numpy as np
import pytest

from dl_helper.training.artifacts import RunLayout, read_json, sha256_file
from dl_helper.training.backends.sklearn_backend import (
    SklearnBackendError,
    build_sklearn_experiment,
    run_sklearn_worker_experiment,
)
from dl_helper.training.checkpoint import read_latest, write_model_manifest
from dl_helper.training.config import default_schema, parse_config
from dl_helper.training.platform import Platform


def _skl_batch_cfg(run_id):
    schema = default_schema()
    schema["backend"] = {
        "type": "sklearn", "torch": None,
        "sklearn": {"fit_mode": "batch", "evaluation_batch_size": 4096, "n_jobs": None,
                    "random_state": "run_seed", "sample_weight_parameter": None},
    }
    schema["distributed"] = {"num_processes": 1}
    schema["training"] = {"max_epochs": 1, "log_every_steps": 1}
    schema["selection"] = {"metric": "val/accuracy", "mode": "max", "patience": 5, "min_delta": 0.0}
    schema["runtime"] = {"max_minutes": None, "shutdown_grace_minutes": 5}
    schema["checkpoint"] = {"every_epochs": None, "every_optimizer_steps": None,
                            "keep_last": 1, "resume": "none"}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    return parse_config(schema)


def test_batch_does_not_create_latest(tmp_path):
    """batch fit 不创建 mid-fit/latest checkpoint。"""
    cfg = _skl_batch_cfg("skl-batch-no-latest")
    layout = RunLayout(str(tmp_path / "runs" / "skl-batch-no-latest"))
    layout.ensure()
    exp = build_sklearn_experiment("experiments.sklearn_batch:build_experiment", cfg.experiment)
    run_sklearn_worker_experiment(exp, cfg, Platform(), layout)
    assert read_latest(layout.path("checkpoints")) is None
    assert not os.path.exists(layout.path("checkpoints", "latest.json"))


def test_model_manifest_recorded(tmp_path):
    """joblib model manifest 记录 size/SHA256/origin/version。"""
    cfg = _skl_batch_cfg("skl-model-manifest")
    layout = RunLayout(str(tmp_path / "runs" / "skl-model-manifest"))
    layout.ensure()
    exp = build_sklearn_experiment("experiments.sklearn_batch:build_experiment", cfg.experiment)
    run_sklearn_worker_experiment(exp, cfg, Platform(), layout)
    manifest = read_json(layout.path("models", "best", "model-manifest.json"))
    assert manifest["backend"] == "sklearn"
    assert manifest["format"] == "joblib"
    assert manifest["origin_run_id"] == "skl-model-manifest"
    assert "model.joblib" in manifest["files"]
    assert manifest["runtime_versions"]["sklearn"]
    # checksum 与实际文件一致
    path = layout.path("models", "best", "model.joblib")
    assert manifest["files"]["model.joblib"]["sha256"] == sha256_file(path)


def test_write_model_manifest_direct(tmp_path):
    target = str(tmp_path / "m")
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, "model.joblib")
    with open(path, "wb") as f:
        f.write(b"x")
    manifest = write_model_manifest(
        target, "sklearn", {"class": "x"}, "run-1",
        {"model.joblib": {"size": 1, "sha256": sha256_file(path)}},
    )
    assert manifest["format"] == "joblib"
    assert read_json(os.path.join(target, "model-manifest.json"))["origin_run_id"] == "run-1"


def test_batch_rejects_external_joblib(tmp_path):
    """外部用户路径 joblib 在反序列化前拒绝。"""
    from dl_helper.training.backends.sklearn_backend import (
        validate_sklearn_checkpoint_source,
    )
    from dl_helper.training.checkpoint import CheckpointError

    external = tmp_path / "external.joblib"
    external.write_bytes(b"not-a-valid-checkpoint")
    # 无 manifest 的路径
    with pytest.raises(CheckpointError):
        validate_sklearn_checkpoint_source(
            str(tmp_path), "external.joblib", "run-1", "fp", "data", {"class": "x"})
