"""任务 8.3：MNIST 示例 —— 临时 fixture 完成训练，缺路径明确失败。"""
from __future__ import annotations

import numpy as np
import pytest

from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config


def _mnist_npz(tmp_path, n=64):
    rng = np.random.default_rng(0)
    path = tmp_path / "mnist.npz"
    np.savez(path, images=rng.random((n, 28, 28)).astype(np.float32),
             labels=rng.integers(0, 10, n).astype(np.int64))
    return str(path)


def _cfg(run_id, data_path):
    schema = default_schema()
    schema["training"]["max_epochs"] = 1
    schema["selection"] = {"metric": "val/loss", "mode": "min", "patience": 5, "min_delta": 0.0}
    schema["report"]["prediction_splits"] = ["val"]
    schema["run"]["id"] = run_id
    schema["checkpoint"]["every_epochs"] = None
    schema["backend"]["torch"]["mixed_precision"] = "no"
    schema["distributed"]["num_processes"] = 1
    schema["experiment"]["data_path"] = data_path
    return parse_config(schema)


def test_mnist_trains_with_fixture(tmp_path):
    data_path = _mnist_npz(tmp_path)
    cfg = _cfg("mnist-fixture", data_path)
    layout = RunLayout(str(tmp_path / "runs" / "mnist-fixture"))
    layout.ensure()
    result = run_worker("experiments.mnist:build_experiment", cfg, layout, 0, 1, "none")
    assert result.status == "succeeded"
    assert layout.path("models", "last", "model.safetensors")


def test_mnist_missing_path_fails(tmp_path):
    cfg = _cfg("mnist-missing", str(tmp_path / "nonexistent.npz"))
    layout = RunLayout(str(tmp_path / "runs" / "mnist-missing"))
    layout.ensure()
    with pytest.raises(FileNotFoundError):
        run_worker("experiments.mnist:build_experiment", cfg, layout, 0, 1, "none")


def test_mnist_requires_explicit_path(tmp_path):
    cfg = _cfg("mnist-nopath", "")
    layout = RunLayout(str(tmp_path / "runs" / "mnist-nopath"))
    layout.ensure()
    with pytest.raises(ValueError):
        run_worker("experiments.mnist:build_experiment", cfg, layout, 0, 1, "none")
