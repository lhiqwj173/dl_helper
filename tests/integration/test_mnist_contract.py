"""任务 8.3：MNIST 示例 —— 临时 fixture 完成训练，缺路径明确失败。"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.mnist import _load_npz
from dl_helper.training.artifacts import RunLayout
from dl_helper.training.backends.torch_backend import run_worker
from dl_helper.training.config import default_schema, parse_config


def _mnist_npz(tmp_path, n=64):
    rng = np.random.default_rng(0)
    path = tmp_path / "mnist.npz"
    np.savez(path, images=rng.random((n, 28, 28)).astype(np.float32),
             labels=rng.integers(0, 10, n).astype(np.int64))
    return str(path)


def _kaggle_mnist_npz(tmp_path):
    path = tmp_path / "kaggle-mnist.npz"
    np.savez(
        path,
        x_train=np.zeros((3, 28, 28), dtype=np.uint8),
        x_test=np.ones((2, 28, 28), dtype=np.uint8),
        y_train=np.array([1, 2, 3], dtype=np.uint8),
        y_test=np.array([4, 5], dtype=np.uint8),
    )
    return path


def test_mnist_loads_kaggle_train_test_keys(tmp_path):
    images, labels = _load_npz(str(_kaggle_mnist_npz(tmp_path)))
    assert images.shape == (5, 28, 28)
    assert labels.tolist() == [1, 2, 3, 4, 5]
    assert images[0, 0, 0] == 0.0
    assert images[-1, 0, 0] == 1.0 / 255.0


def test_mnist_rejects_incomplete_npz_keys(tmp_path):
    path = tmp_path / "invalid-mnist.npz"
    np.savez(path, x_train=np.zeros((2, 28, 28), dtype=np.uint8), y_train=np.zeros(2, dtype=np.uint8))
    with pytest.raises(ValueError, match="必须包含"):
        _load_npz(str(path))


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
