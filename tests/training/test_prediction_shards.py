"""任务 4.2：无 pickle 预测分片与确定性优先级抽样。"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from dl_helper.training.artifacts import (
    ArtifactError,
    merge_topk_candidates,
    priority_sample,
    write_prediction_manifest,
    write_prediction_shard,
)


def _arrays(n=10):
    return {
        "targets": np.random.randint(0, 3, n).astype(np.int64),
        "predictions": np.random.randint(0, 3, n).astype(np.int64),
        "scores": np.random.rand(n, 3).astype(np.float64),
        "names": np.array([f"row{i}" for i in range(n)]),
    }


def test_write_read_shard(tmp_path):
    dir_path = str(tmp_path)
    arrays = _arrays()
    entry = write_prediction_shard(dir_path, 0, 0, arrays, 10)
    assert entry["sample_count"] == 10
    assert entry["file"].startswith("part-rank00000-000000")
    assert entry["sha256"]
    data = np.load(os.path.join(dir_path, entry["file"]), allow_pickle=False)
    assert data["targets"].shape == (10,)
    assert data["scores"].shape == (10, 3)
    assert data["names"].dtype.kind == "U"
    # manifest
    manifest = write_prediction_manifest(dir_path, "val", [entry], 10, sampled=False)
    assert manifest["sample_count"] == 10
    assert os.path.exists(os.path.join(dir_path, "prediction-manifest.json"))


def test_object_dtype_rejected(tmp_path):
    arr = np.empty(3, dtype=object)
    arr[:] = ["a", "b", "c"]
    with pytest.raises(ArtifactError):
        write_prediction_shard(str(tmp_path), 0, 0, {"x": arr}, 3)


def test_invalid_field_name_rejected(tmp_path):
    with pytest.raises(ArtifactError):
        write_prediction_shard(str(tmp_path), 0, 0, {"1bad": np.arange(3)}, 3)
    with pytest.raises(ArtifactError):
        write_prediction_shard(str(tmp_path), 0, 0, {"has space": np.arange(3)}, 3)


def test_shape_mismatch_rejected(tmp_path):
    with pytest.raises(ArtifactError):
        write_prediction_shard(str(tmp_path), 0, 0, {"x": np.arange(3)}, 5)


def test_non_finite_rejected(tmp_path):
    with pytest.raises(ArtifactError):
        write_prediction_shard(str(tmp_path), 0, 0, {"x": np.array([1.0, np.nan])}, 2)


def test_priority_sample_deterministic():
    rng = np.random.default_rng(0)
    scores = rng.random(100)
    idx1 = priority_sample(100, 42, "val", 10, sample_ids=None)
    idx2 = priority_sample(100, 42, "val", 10, sample_ids=None)
    assert np.array_equal(idx1, idx2)
    # 不同 seed 不同结果
    idx3 = priority_sample(100, 43, "val", 10, sample_ids=None)
    assert not np.array_equal(idx1, idx3)
    assert len(idx1) == 10


def test_priority_sample_with_ids():
    sample_ids = np.arange(100)
    idx = priority_sample(100, 1, "test", 20, sample_ids=sample_ids)
    assert len(idx) == 20
    assert len(set(idx.tolist())) == 20  # 无重复
    # 相同 ID 序列确定性
    idx2 = priority_sample(100, 1, "test", 20, sample_ids=sample_ids.copy())
    assert np.array_equal(idx, idx2)


def test_priority_sample_position_when_no_ids():
    idx = priority_sample(50, 7, "train", 5, sample_ids=None)
    assert idx.shape == (5,)


def test_merge_topk_candidates():
    a = np.array([3, 1, 4])
    b = np.array([0, 2, 5])
    merged = merge_topk_candidates([a, b], 3)
    assert len(merged) == 3
    assert len(set(merged.tolist())) == 3
